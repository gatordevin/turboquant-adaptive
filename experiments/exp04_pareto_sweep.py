"""
TurboQuant Pareto Frontier Benchmark
=====================================
Sweeps (key_bits, value_bits, residual_window) configs on GPT-2,
measures perplexity, memory, and throughput, then plots Pareto frontiers.
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
from matplotlib.lines import Line2D
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from turboquant_gpt2 import (
    TurboQuantizer, TurboQuantLayer, compute_perplexity
)
from transformers.cache_utils import DynamicCache


# ── Codebook cache to avoid recomputing ──
_quantizer_cache = {}

def get_quantizer(head_dim, bits, device, seed):
    key = (head_dim, bits, seed)
    if key not in _quantizer_cache:
        _quantizer_cache[key] = TurboQuantizer(head_dim, bits, device=device, seed=seed,
                                                use_exact_pdf=True)
    return _quantizer_cache[key]


def make_cache_fast(head_dim, num_layers, key_bits, value_bits, residual_window, device):
    """Make TurboQuant cache reusing cached quantizers."""
    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None

    for layer_idx in range(num_layers):
        kq = get_quantizer(head_dim, key_bits, device, seed=42 + layer_idx * 2)
        vq = get_quantizer(head_dim, value_bits, device, seed=42 + layer_idx * 2 + 1)
        cache.layers.append(TurboQuantLayer(kq, vq, residual_window=residual_window))

    return cache


def measure_throughput(model, tokenizer, cache_factory, prompt, num_tokens=150, warmup=2):
    """Measure tokens/sec during generation."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    for _ in range(warmup):
        cache = cache_factory() if cache_factory else None
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=10, past_key_values=cache,
                          do_sample=False, pad_token_id=tokenizer.eos_token_id)

    torch.cuda.synchronize()
    cache = cache_factory() if cache_factory else None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=num_tokens, past_key_values=cache,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (outputs.shape[1] - input_ids.shape[1]) / (t1 - t0)


def compute_memory_ratio(seq_len, head_dim, key_bits, value_bits, residual_window):
    compressed_tokens = max(0, seq_len - residual_window)
    recent_tokens = min(seq_len, residual_window)
    fp16_per_token = head_dim * 2 * 2
    compressed_per_token = head_dim * key_bits / 8 + 2 + head_dim * value_bits / 8 + 2
    total_compressed = compressed_tokens * compressed_per_token + recent_tokens * fp16_per_token
    total_fp16 = seq_len * fp16_per_token
    return total_fp16 / total_compressed


def main():
    device = 'cuda'
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).half()
    model.eval()

    head_dim, num_layers, num_heads = 64, 12, 12
    seq_len = 512

    eval_text = (
        "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
        "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of "
        "modern AI were planted by philosophers who attempted to describe the process of human thinking "
        "as the mechanical manipulation of symbols. This work culminated in the invention of the "
        "programmable digital computer in the 1940s, a machine based on the abstract essence of "
        "mathematical reasoning. This device and the ideas behind it inspired a handful of scientists "
        "to begin seriously discussing the possibility of building an electronic brain. The field of AI "
        "research was founded at a workshop held on the campus of Dartmouth College during the summer "
        "of 1956. Those who attended would become the leaders of AI research for decades. Many of them "
        "predicted that a machine as intelligent as a human being would exist in no more than a "
        "generation, and they were given millions of dollars to make this vision come true. Eventually, "
        "it became obvious that commercial developers and researchers had grossly underestimated the "
        "difficulty of the project."
    )
    prompt = "The future of artificial intelligence is"

    # ── Pre-compute all codebooks ──
    print("Pre-computing Lloyd-Max codebooks...")
    t0 = time.time()
    for bits in [1, 2, 3, 4]:
        for layer_idx in range(num_layers):
            get_quantizer(head_dim, bits, device, seed=42 + layer_idx * 2)
            get_quantizer(head_dim, bits, device, seed=42 + layer_idx * 2 + 1)
    print(f"  All codebooks ready in {time.time()-t0:.1f}s")

    # ── Define sweep ──
    configs = []
    for kb in [1, 2, 3, 4]:
        for vb in [1, 2, 3, 4]:
            if vb > kb:
                continue
            for win in [16, 32, 64, 128]:
                configs.append((f"K{kb}/V{vb} w{win}", kb, vb, win))

    print(f"\nSweeping {len(configs)} TurboQuant configs + baseline...")
    print(f"{'Config':25s} {'Compress':>10s} {'PPL':>8s} {'ΔPPL':>8s} {'tok/s':>8s}")
    print("-" * 65)

    # ── Baseline ──
    ppl_base = compute_perplexity(model, tokenizer, eval_text)
    tps_base = measure_throughput(model, tokenizer, None, prompt)
    print(f"  {'Baseline fp16':23s} {'1.00x':>10s} {ppl_base:>8.2f} {'—':>8s} {tps_base:>8.1f}")

    results = [{
        'name': 'Baseline fp16', 'key_bits': 16, 'value_bits': 16,
        'window': 9999, 'compression': 1.0, 'ppl': ppl_base,
        'throughput': tps_base, 'is_baseline': True
    }]

    # ── Sweep ──
    for i, (name, kb, vb, win) in enumerate(configs):
        compression = compute_memory_ratio(seq_len, head_dim, kb, vb, win)

        cache = make_cache_fast(head_dim, num_layers, kb, vb, win, device)
        ppl = compute_perplexity(model, tokenizer, eval_text, cache=cache)

        factory = lambda kb=kb, vb=vb, win=win: make_cache_fast(
            head_dim, num_layers, kb, vb, win, device)
        tps = measure_throughput(model, tokenizer, factory, prompt)

        delta = ((ppl - ppl_base) / ppl_base) * 100
        results.append({
            'name': name, 'key_bits': kb, 'value_bits': vb,
            'window': win, 'compression': compression, 'ppl': ppl,
            'throughput': tps, 'is_baseline': False
        })
        print(f"  {name:23s} {compression:>9.2f}x {ppl:>8.2f} {delta:>+7.1f}% {tps:>8.1f}"
              f"  [{i+1}/{len(configs)}]")

    # ── Pareto frontiers ──
    def pareto_front(points, x_key, y_key, minimize_y=True):
        sorted_pts = sorted(points, key=lambda p: p[x_key])
        front = []
        best_y = float('inf') if minimize_y else float('-inf')
        for p in sorted_pts:
            if minimize_y and p[y_key] <= best_y:
                best_y = p[y_key]
                front.append(p)
            elif not minimize_y and p[y_key] >= best_y:
                best_y = p[y_key]
                front.append(p)
        return front

    pareto_ppl = pareto_front(results, 'compression', 'ppl', minimize_y=True)

    # ── Plot ──
    kb_colors = {1: '#e74c3c', 2: '#e67e22', 3: '#2ecc71', 4: '#3498db', 16: '#2c3e50'}
    win_markers = {16: 'v', 32: 's', 64: 'D', 128: '^', 9999: '*'}

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('TurboQuant KV-Cache Compression — GPT-2 Pareto Frontiers',
                 fontsize=16, fontweight='bold', y=1.02)

    # ── Panel 1: Compression vs Perplexity ──
    ax = axes[0]
    for r in results:
        c = kb_colors.get(r['key_bits'], '#999')
        m = win_markers.get(r['window'], 'o')
        sz = 150 if r['is_baseline'] else 70
        edge = 'black' if r in pareto_ppl else 'none'
        lw = 2.5 if r in pareto_ppl else 0
        ax.scatter(r['compression'], r['ppl'], c=c, marker=m, s=sz,
                   edgecolors=edge, linewidths=lw, zorder=3, alpha=0.85)

    px = [p['compression'] for p in pareto_ppl]
    py = [p['ppl'] for p in pareto_ppl]
    ax.plot(px, py, 'k--', alpha=0.6, linewidth=2, label='Pareto frontier')
    ax.axhline(ppl_base, color='gray', ls=':', alpha=0.5, label=f'Baseline PPL={ppl_base:.1f}')
    ax.axhline(ppl_base * 1.05, color='orange', ls=':', alpha=0.5, label='5% degradation')
    ax.fill_between([0.5, 20], ppl_base, ppl_base * 1.05, alpha=0.06, color='green')
    ax.set_xlabel('Compression Ratio (×)', fontsize=12)
    ax.set_ylabel('Perplexity (↓ better)', fontsize=12)
    ax.set_title('Quality vs Compression', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.8, None)

    # ── Panel 2: Compression vs Throughput ──
    ax = axes[1]
    for r in results:
        c = kb_colors.get(r['key_bits'], '#999')
        m = win_markers.get(r['window'], 'o')
        sz = 150 if r['is_baseline'] else 70
        ax.scatter(r['compression'], r['throughput'], c=c, marker=m, s=sz,
                   edgecolors='none', linewidths=0, zorder=3, alpha=0.85)

    ax.axhline(tps_base, color='gray', ls=':', alpha=0.5, label=f'Baseline={tps_base:.0f} tok/s')
    ax.set_xlabel('Compression Ratio (×)', fontsize=12)
    ax.set_ylabel('Throughput (tok/s, ↑ better)', fontsize=12)
    ax.set_title('Speed vs Compression', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.8, None)

    # ── Panel 3: Throughput vs Perplexity (the real tradeoff) ──
    ax = axes[2]
    compressions = [r['compression'] for r in results]
    norm = plt.Normalize(min(compressions), max(compressions))
    cmap = plt.cm.viridis

    for r in results:
        c = cmap(norm(r['compression']))
        m = win_markers.get(r['window'], 'o')
        sz = 150 if r['is_baseline'] else 70
        ax.scatter(r['ppl'], r['throughput'], c=[c], marker=m, s=sz,
                   edgecolors='black' if r['is_baseline'] else 'none',
                   linewidths=2.5 if r['is_baseline'] else 0, zorder=3, alpha=0.85)
        if r['is_baseline']:
            ax.annotate('Baseline', (r['ppl'], r['throughput']),
                       fontsize=9, fontweight='bold', ha='right', va='bottom',
                       xytext=(-10, 5), textcoords='offset points')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Compression Ratio (×)', fontsize=10)

    ax.axvline(ppl_base * 1.05, color='orange', ls=':', alpha=0.5, label='5% PPL threshold')
    ax.axhline(tps_base, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Perplexity (← better)', fontsize=12)
    ax.set_ylabel('Throughput (tok/s, ↑ better)', fontsize=12)
    ax.set_title('Speed vs Quality', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Shared legend ──
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=kb_colors[1], markersize=9, label='K=1 bit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=kb_colors[2], markersize=9, label='K=2 bit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=kb_colors[3], markersize=9, label='K=3 bit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=kb_colors[4], markersize=9, label='K=4 bit'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=kb_colors[16], markersize=12, label='Baseline'),
        Line2D([], [], color='none', label=''),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=8, label='win=16'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='win=32'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=8, label='win=64'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='win=128'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=10, fontsize=8.5,
               frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    plt.savefig('/home/farmspace/aitest/turboquant_pareto.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"\nSaved: turboquant_pareto.png")

    # ── Summary table ──
    print("\n" + "=" * 75)
    print("PARETO-OPTIMAL CONFIGS (Quality frontier)")
    print("=" * 75)
    print(f"{'Config':25s} {'Compress':>10s} {'PPL':>8s} {'ΔPPL':>8s} {'tok/s':>8s} {'Speedup':>8s}")
    print("-" * 75)
    for p in pareto_ppl:
        delta = ((p['ppl'] - ppl_base) / ppl_base) * 100
        speedup = p['throughput'] / tps_base
        print(f"  {p['name']:23s} {p['compression']:>9.2f}x {p['ppl']:>8.2f} {delta:>+7.1f}% "
              f"{p['throughput']:>8.1f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
