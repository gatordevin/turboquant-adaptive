"""
Clean Scaling Curves: PPL Delta vs Context Length
==================================================
GPT-2 XL (best model), multiple text domains, fine-grained
sequence lengths, averaged across texts for stable curves.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from turboquant_gpt2 import TurboQuantizer
from experiments.exp12_full_stack import FP16Layer, FullStackLayer

_qcache = {}
def get_q(hd, bits, device, seed):
    k = (hd, bits, seed)
    if k not in _qcache:
        _qcache[k] = TurboQuantizer(hd, bits, device=device, seed=seed)
    return _qcache[k]

def make_cache(nl, layer_fn):
    cache = DynamicCache()
    cache.layers = [layer_fn(li) for li in range(nl)]
    cache.layer_class_to_replicate = None
    return cache

def compute_ppl(model, ids, cache=None):
    with torch.no_grad():
        out = model(ids, past_key_values=cache)
        logits = out.logits[:, :-1, :]
        targets = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return torch.exp(loss).item()


# ── Diverse evaluation texts ──
TEXTS = {
    'history': """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that commercial developers and researchers had grossly underestimated the difficulty of the project. In 1974, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an AI winter, a period when obtaining funding for AI projects was difficult. In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic research. However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into disrepute, and a second, longer lasting winter began. AI revived in the late 1990s and early 21st century with its focus on solving specific subproblems. The narrow focus allowed researchers to produce verifiable results, exploit more mathematical methods, and collaborate with other fields. By 2000, solutions developed by AI researchers were being widely used in technology.""",

    'science': """The universe began with the Big Bang approximately 13.8 billion years ago. In the first moments, temperatures were so extreme that matter as we know it could not exist. As the universe expanded and cooled, quarks combined to form protons and neutrons. Within minutes, these particles fused into the nuclei of the lightest elements, primarily hydrogen and helium. For hundreds of thousands of years, the universe was a hot, dense plasma. Then, about 380,000 years after the Big Bang, the universe had cooled enough for electrons to combine with nuclei to form neutral atoms in an event known as recombination. This released photons that had been trapped in the plasma, creating the cosmic microwave background radiation that we can still observe today. Over hundreds of millions of years, gravity pulled matter together into the first stars and galaxies. These early stars were massive and short-lived, burning through their fuel quickly and ending in spectacular supernovae that spread heavier elements throughout the cosmos. Subsequent generations of stars formed from this enriched material, eventually leading to stars like our Sun, which formed about 4.6 billion years ago from a cloud of gas and dust. The leftover material formed a disk around the young Sun, from which the planets of our solar system coalesced. Earth formed about 4.5 billion years ago and was initially a molten world bombarded by asteroids and comets. Gradually, the surface cooled and solidified, and water began to accumulate on the surface, forming the first oceans. Within a billion years of Earth's formation, the first simple life forms appeared. These were single-celled organisms that lived in the oceans. For billions of years, life remained microscopic, but these early organisms transformed the planet by producing oxygen through photosynthesis, gradually changing the atmosphere from one dominated by carbon dioxide and methane to one rich in oxygen.""",

    'narrative': """The old lighthouse keeper climbed the spiral staircase for what felt like the thousandth time. Each step creaked under his weight, a familiar symphony of aging wood and rusty nails. At the top, he paused to catch his breath and looked out through the salt-crusted windows at the gray expanse of ocean stretching to the horizon. The storm was coming. He could feel it in his bones before he saw the dark clouds gathering in the west. He had lived through forty years of storms on this rocky point, and this one had the feel of something different. The barometric pressure had been dropping steadily all day, and the seabirds had fled inland hours ago. He checked the lamp mechanism one final time, running his calloused fingers over the brass gears and the pristine glass of the Fresnel lens. Everything was in order. The light would burn through the night, a beacon for any ship foolish enough to be caught in what was coming. He descended the stairs slowly and made his way to the small cottage attached to the lighthouse base. His dog, a weathered border collie named Salt, was already curled up beneath the heavy oak table, sensing the approaching tempest. He put the kettle on and sat in his worn armchair by the window, watching the first heavy drops of rain begin to fall. The wind picked up gradually, then suddenly, howling around the corners of the building with increasing fury. By midnight, waves were crashing against the rocks sixty feet below with enough force to send tremors through the foundations. The lighthouse stood firm, as it had for over a century, its beam cutting through sheets of rain and spray. He stayed awake through the whole night, as he always did during storms, listening to the building groan and flex against the wind. Toward dawn, the fury began to subside.""",
}


def main():
    device = 'cuda'
    print("=" * 75)
    print("SCALING CURVES: GPT-2 XL — All Strategies vs Context Length")
    print("=" * 75)

    tok = GPT2Tokenizer.from_pretrained('gpt2-xl')
    model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device).half().eval()
    hd, nl, nh = 64, 48, 25

    # Precompute codebooks
    print("Precomputing codebooks...")
    for bits in [2, 3, 4]:
        for li in range(nl):
            get_q(hd, bits, device, 42 + li * 2)
            get_q(hd, bits, device, 42 + li * 2 + 1)
        for rank in [24, 48]:
            get_q(rank, bits, device, 500 + rank * 10 + bits)

    # ── Define strategies ──
    strategies = {}

    strategies['Vanilla TQ\nK4/V2 w32'] = lambda li: FullStackLayer(
        rank_k=hd, rank_v=hd,
        kq=get_q(hd, 4, device, 42+li*2), vq=get_q(hd, 2, device, 42+li*2+1),
        num_sinks=0, residual_window=32, min_chunk_for_svd=9999)

    strategies['L0fp16 + TQ\nK4/V2 w32 s4'] = lambda li: (
        FP16Layer() if li == 0 else FullStackLayer(
            rank_k=hd, rank_v=hd,
            kq=get_q(hd, 4, device, 42+li*2), vq=get_q(hd, 2, device, 42+li*2+1),
            num_sinks=4, residual_window=32, min_chunk_for_svd=9999))

    strategies['L0fp16 + TQ\nK4/V2 w8 s4'] = lambda li: (
        FP16Layer() if li == 0 else FullStackLayer(
            rank_k=hd, rank_v=hd,
            kq=get_q(hd, 4, device, 42+li*2), vq=get_q(hd, 2, device, 42+li*2+1),
            num_sinks=4, residual_window=8, min_chunk_for_svd=9999))

    strategies['SVD(48/24)\nonly w8 s4'] = lambda li: (
        FP16Layer() if li == 0 else FullStackLayer(
            rank_k=48, rank_v=24, kq=None, vq=None,
            num_sinks=4, residual_window=8))

    strategies['SVD(48/24) +\nTQ K4/V4 w8 s4'] = lambda li: (
        FP16Layer() if li == 0 else FullStackLayer(
            rank_k=48, rank_v=24,
            kq=get_q(48, 4, device, 500+48*10+4), vq=get_q(24, 4, device, 500+24*10+4),
            num_sinks=4, residual_window=8))

    strategies['SVD(48/24) +\nTQ K4/V3 w8 s4'] = lambda li: (
        FP16Layer() if li == 0 else FullStackLayer(
            rank_k=48, rank_v=24,
            kq=get_q(48, 4, device, 500+48*10+4), vq=get_q(24, 3, device, 500+24*10+3),
            num_sinks=4, residual_window=8))

    strategies['L0fp16 + TQ\nK3/V2 w8 s4'] = lambda li: (
        FP16Layer() if li == 0 else FullStackLayer(
            rank_k=hd, rank_v=hd,
            kq=get_q(hd, 3, device, 42+li*2), vq=get_q(hd, 2, device, 42+li*2+1),
            num_sinks=4, residual_window=8, min_chunk_for_svd=9999))

    # ── Sequence lengths to test ──
    lengths = [64, 96, 128, 192, 256, 384, 512, 640, 768, 896, 1024]

    # ── Run all ──
    # results[strategy_name][text_name] = {lengths: [...], deltas: [...]}
    results = {s: {} for s in strategies}
    baselines = {}  # baselines[text_name] = {length: ppl}

    for text_name, raw_text in TEXTS.items():
        print(f"\n  Text: {text_name}")
        # Encode full text, repeat if needed to reach max length
        full_ids = tok.encode(raw_text)
        while len(full_ids) < max(lengths) + 10:
            full_ids = full_ids + tok.encode(raw_text)

        baselines[text_name] = {}

        for sname in strategies:
            results[sname][text_name] = {'lengths': [], 'deltas': [], 'ppls': []}

        for target_len in lengths:
            ids = torch.tensor([full_ids[:target_len]], device=device)
            actual_len = ids.shape[1]

            ppl_base = compute_ppl(model, ids)
            baselines[text_name][actual_len] = ppl_base

            row = f"    {actual_len:>5d}t  base={ppl_base:>6.2f}"

            for sname, layer_fn in strategies.items():
                cache = make_cache(nl, layer_fn)
                ppl = compute_ppl(model, ids, cache=cache)
                delta = ((ppl - ppl_base) / ppl_base) * 100

                results[sname][text_name]['lengths'].append(actual_len)
                results[sname][text_name]['deltas'].append(delta)
                results[sname][text_name]['ppls'].append(ppl)

                short = sname.replace('\n', ' ')[:15]
                row += f"  {short}={delta:>+5.1f}%"

            print(row)

    # ── Average across texts ──
    avg_results = {}
    for sname in strategies:
        avg_results[sname] = {'lengths': lengths, 'mean_delta': [], 'std_delta': [],
                               'min_delta': [], 'max_delta': []}
        for i, l in enumerate(lengths):
            deltas_at_l = []
            for text_name in TEXTS:
                if i < len(results[sname][text_name]['deltas']):
                    deltas_at_l.append(results[sname][text_name]['deltas'][i])
            if deltas_at_l:
                avg_results[sname]['mean_delta'].append(np.mean(deltas_at_l))
                avg_results[sname]['std_delta'].append(np.std(deltas_at_l))
                avg_results[sname]['min_delta'].append(min(deltas_at_l))
                avg_results[sname]['max_delta'].append(max(deltas_at_l))
            else:
                avg_results[sname]['mean_delta'].append(float('nan'))
                avg_results[sname]['std_delta'].append(0)
                avg_results[sname]['min_delta'].append(float('nan'))
                avg_results[sname]['max_delta'].append(float('nan'))

    # ══════════════════════════════════════════════════════════════
    # PLOT: Main scaling curves (averaged across texts)
    # ══════════════════════════════════════════════════════════════

    style = {
        'Vanilla TQ\nK4/V2 w32':       {'color': '#888888', 'ls': '-',  'marker': 'o', 'lw': 2.5, 'ms': 7},
        'L0fp16 + TQ\nK4/V2 w32 s4':   {'color': '#555555', 'ls': '-',  'marker': 's', 'lw': 2,   'ms': 7},
        'L0fp16 + TQ\nK4/V2 w8 s4':    {'color': '#333333', 'ls': '-',  'marker': 'D', 'lw': 2,   'ms': 6},
        'SVD(48/24)\nonly w8 s4':       {'color': '#3498db', 'ls': '--', 'marker': '^', 'lw': 2.5, 'ms': 8},
        'SVD(48/24) +\nTQ K4/V4 w8 s4':{'color': '#e74c3c', 'ls': '-',  'marker': '*', 'lw': 3,   'ms': 11},
        'SVD(48/24) +\nTQ K4/V3 w8 s4':{'color': '#e67e22', 'ls': '-',  'marker': 'P', 'lw': 2.5, 'ms': 9},
        'L0fp16 + TQ\nK3/V2 w8 s4':    {'color': '#27ae60', 'ls': ':',  'marker': 'v', 'lw': 2,   'ms': 7},
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    for sname, avg in avg_results.items():
        s = style[sname]
        x = avg['lengths']
        y = avg['mean_delta']
        y_min = avg['min_delta']
        y_max = avg['max_delta']

        # Skip NaN
        valid = [(xi, yi, lo, hi) for xi, yi, lo, hi in zip(x, y, y_min, y_max)
                 if not np.isnan(yi) and abs(yi) < 50]
        if not valid:
            continue
        vx, vy, vlo, vhi = zip(*valid)

        label = sname.replace('\n', ' ')
        ax.plot(vx, vy, color=s['color'], ls=s['ls'], marker=s['marker'],
                lw=s['lw'], markersize=s['ms'], label=label, alpha=0.9, zorder=3)

        # Shaded min-max range
        ax.fill_between(vx, vlo, vhi, color=s['color'], alpha=0.08)

    ax.axhline(0, color='black', lw=1.2)
    ax.axhspan(-2, 2, alpha=0.06, color='green')
    ax.axhline(2, color='#27ae60', ls='--', alpha=0.4, lw=1)
    ax.axhline(-2, color='#27ae60', ls='--', alpha=0.4, lw=1)
    ax.axhline(5, color='#e67e22', ls=':', alpha=0.3, lw=1)

    ax.set_xlabel('Sequence Length (tokens)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Perplexity Change vs Baseline (%)', fontsize=13, fontweight='bold')
    ax.set_title('GPT-2 XL (1.5B) — KV Cache Compression Quality vs Context Length\n'
                 '(averaged over 3 text domains, shading = min/max range)',
                 fontsize=14, fontweight='bold')

    ax.legend(fontsize=9, loc='upper left', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(50, 1050)

    # Annotations
    ax.annotate('SVD denoising\nzone (quality boost)',
               xy=(150, -3), fontsize=9, color='#3498db', style='italic',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#3498db', alpha=0.7))

    ax.annotate('SVD crossover\n(~400-600 tokens)',
               xy=(500, 1), fontsize=9, color='#888', style='italic',
               ha='center', va='center',
               arrowprops=dict(arrowstyle='->', color='#888'),
               xytext=(500, -2.5))

    plt.tight_layout()
    plt.savefig('results/scaling_curves_main.png', dpi=180, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: results/scaling_curves_main.png")

    # ══════════════════════════════════════════════════════════════
    # PLOT 2: Per-text-domain breakdown
    # ══════════════════════════════════════════════════════════════

    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6.5))

    for ax2, text_name in zip(axes2, TEXTS.keys()):
        for sname in strategies:
            s = style[sname]
            r = results[sname][text_name]
            valid = [(l, d) for l, d in zip(r['lengths'], r['deltas'])
                     if not np.isnan(d) and abs(d) < 50]
            if valid:
                vx, vy = zip(*valid)
                label = sname.replace('\n', ' ')
                ax2.plot(vx, vy, color=s['color'], ls=s['ls'], marker=s['marker'],
                        lw=s['lw']*0.7, markersize=s['ms']*0.8, label=label, alpha=0.85)

        ax2.axhline(0, color='black', lw=0.8)
        ax2.axhspan(-2, 2, alpha=0.05, color='green')
        ax2.set_xlabel('Sequence Length', fontsize=10)
        ax2.set_ylabel('PPL Change (%)', fontsize=10)
        ax2.set_title(f'Text: {text_name}', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.15)
        ax2.set_xlim(50, 1050)

    axes2[0].legend(fontsize=6.5, loc='upper left', ncol=1)
    fig2.suptitle('GPT-2 XL — Per-Domain Breakdown\n(consistency check across text types)',
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/scaling_curves_per_domain.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: results/scaling_curves_per_domain.png")

    # ── Print summary table ──
    print(f"\n{'='*95}")
    print("SUMMARY TABLE: Mean PPL Delta (%) Across 3 Texts")
    print(f"{'='*95}")
    header = f"{'Strategy':35s}"
    for l in lengths:
        header += f" {l:>6d}t"
    print(header)
    print("-" * 95)

    for sname in strategies:
        avg = avg_results[sname]
        row = f"{sname.replace(chr(10), ' '):35s}"
        for d in avg['mean_delta']:
            if np.isnan(d) or abs(d) > 99:
                row += f"  {'X':>5s}"
            else:
                row += f" {d:>+5.1f}%"
        print(row)

    # Best at each length
    print(f"\n{'Best strategy per length':35s}", end="")
    for i, l in enumerate(lengths):
        best_name = None
        best_delta = float('inf')
        for sname, avg in avg_results.items():
            d = avg['mean_delta'][i]
            if not np.isnan(d) and abs(d) < best_delta:
                best_delta = abs(d)
                best_name = sname.replace('\n', ' ')[:18]
        print(f" {best_name:>6s}" if best_name else "   ???", end="")
    print()


if __name__ == "__main__":
    main()
