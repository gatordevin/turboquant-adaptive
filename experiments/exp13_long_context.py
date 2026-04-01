"""
Long Context & Multi-Model Stress Test
========================================
Test the full compression stack at sequence lengths up to model max,
across GPT-2 Large, GPT-2 XL, and Qwen2.5-0.5B.

Key questions:
1. Does SVD(48/24) denoising hold at 512-1024 tokens?
2. Does the SVD+TurboQuant combo stay stable at long context?
3. Does it work on Qwen (GQA, different tokenizer)?
4. How does compression ratio scale with sequence length?
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, CacheLayerMixin

from turboquant_gpt2 import TurboQuantizer

_qcache = {}
def get_q(hd, bits, device, seed):
    k = (hd, bits, seed)
    if k not in _qcache:
        _qcache[k] = TurboQuantizer(hd, bits, device=device, seed=seed)
    return _qcache[k]


# ── Reuse layers from exp12 ──
from experiments.exp12_full_stack import FP16Layer, FullStackLayer


def compute_ppl(model, tokenizer, text, cache=None):
    ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(ids, past_key_values=cache)
        logits = out.logits[:, :-1, :]
        targets = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return torch.exp(loss).item()


def make_cache(nl, layer_fn):
    cache = DynamicCache()
    cache.layers = [layer_fn(li) for li in range(nl)]
    cache.layer_class_to_replicate = None
    return cache


def compute_compression(seq_len, hd, nh, nl, config_name,
                         rank_k=48, rank_v=24, kb=4, vb=3, sinks=4, win=8):
    """Estimate theoretical compression ratio."""
    fp16_per_tok = hd * 2 * 2  # K+V fp16
    total_fp16 = seq_len * fp16_per_tok * nh * nl

    total_comp = 0
    for li in range(nl):
        if li == 0:
            # fp16 layer
            total_comp += seq_len * fp16_per_tok * nh
        else:
            s = min(sinks, seq_len)
            w = min(win, max(0, seq_len - s))
            c = max(0, seq_len - s - w)

            if 'svd' in config_name.lower() or 'SVD' in config_name:
                # SVD coefficients: c tokens × rank × bits/8 + rank × hd × 2 (basis)
                # Basis is per-chunk, amortized over c tokens
                comp_k = c * rank_k * kb / 8 + c * 2 + rank_k * hd * 2  # coeffs + norms + basis
                comp_v = c * rank_v * vb / 8 + c * 2 + rank_v * hd * 2
            else:
                comp_k = c * (hd * kb / 8 + 2)
                comp_v = c * (hd * vb / 8 + 2)

            total_comp += ((s + w) * fp16_per_tok + comp_k + comp_v) * nh

    return total_fp16 / total_comp if total_comp > 0 else 1.0


# ============================================================================
# Long text generation (fill context window with real text)
# ============================================================================

LONG_TEXT = """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain. The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true. Eventually, it became obvious that commercial developers and researchers had grossly underestimated the difficulty of the project. In 1974, in response to the criticism from James Lighthill and ongoing pressure from congress, the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an AI winter, a period when obtaining funding for AI projects was difficult. In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic research. However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into disrepute, and a second, longer lasting winter began. AI revived in the late 1990s and early 21st century with its focus on solving specific subproblems. The narrow focus allowed researchers to produce verifiable results, exploit more mathematical methods, and collaborate with other fields. By 2000, solutions developed by AI researchers were being widely used in technology, although they were rarely described as artificial intelligence. In the early 21st century, AI began to be used for logistics, data mining, medical diagnosis and other areas throughout the technology industry. The success was due to several factors: the increasing computational power of computers, a greater emphasis on solving specific problems, new ties between AI and other fields working on similar problems, and a new commitment by researchers to mathematical methods and scientific standards. Deep Blue became the first computer chess playing system to beat a reigning world chess champion, Garry Kasparov, on May 11, 1997. In 2011, in a Jeopardy quiz show exhibition match, IBM's question answering system, Watson, defeated the two greatest champions, Brad Rutter and Ken Jennings, by a significant margin. Faster computers, algorithmic improvements, and access to large amounts of data enabled advances in machine learning and perception. By the 2010s, machine learning applications were used throughout the world. In a 2017 survey, one in five companies reported having incorporated AI in some offerings or processes. The amount of research into AI increased dramatically throughout the 2010s. According to one estimate, the number of scientific papers related to AI published each year increased by 50 percent between 2015 and 2019. Around 2016, China significantly accelerated its government funding with its national plan for the development of AI technologies. It was in this environment that large language models emerged, marking a new chapter in the history of artificial intelligence. These models, trained on vast corpora of text from the internet, demonstrated an unexpected ability to generate coherent and contextually relevant text across a wide range of topics."""


def get_text_at_length(tokenizer, target_tokens):
    """Get text that's approximately target_tokens long."""
    text = LONG_TEXT
    while len(tokenizer.encode(text)) < target_tokens:
        text = text + " " + LONG_TEXT
    # Trim to exact length
    ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(ids)


# ============================================================================
# Main
# ============================================================================

def main():
    device = 'cuda'
    print("=" * 85)
    print("LONG CONTEXT & MULTI-MODEL STRESS TEST")
    print("=" * 85)

    models_config = [
        {
            'name': 'gpt2-large',
            'max_len': 1024,
            'lengths': [128, 256, 512, 768, 1024],
            'is_gpt2': True,
        },
        {
            'name': 'gpt2-xl',
            'max_len': 1024,
            'lengths': [128, 256, 512, 768, 1024],
            'is_gpt2': True,
        },
        {
            'name': 'Qwen/Qwen2.5-0.5B',
            'max_len': 4096,  # Qwen supports much longer
            'lengths': [128, 256, 512, 1024, 2048],
            'is_gpt2': False,
        },
    ]

    all_model_data = {}

    for mcfg in models_config:
        model_name = mcfg['name']
        print(f"\n{'='*85}")
        print(f"MODEL: {model_name}")
        print(f"{'='*85}")

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).half()
        model.eval()

        config = model.config
        hd = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        nl = config.num_hidden_layers
        nh = config.num_attention_heads
        kv_heads = getattr(config, 'num_key_value_heads', nh)
        has_gqa = kv_heads < nh

        print(f"  layers={nl}, hd={hd}, heads={nh}, kv_heads={kv_heads}"
              f"{' GQA!' if has_gqa else ''}")
        print(f"  GPU mem: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

        # Precompute codebooks for this head_dim
        print("  Precomputing codebooks...")
        t0 = time.time()
        for bits in [2, 3, 4]:
            for li in range(nl):
                get_q(hd, bits, device, 42 + li * 2)
                get_q(hd, bits, device, 42 + li * 2 + 1)
            # SVD coefficient codebooks
            for rank in [16, 24, 32, 48]:
                if rank <= hd:
                    get_q(rank, bits, device, 500 + rank * 10 + bits)
        print(f"  Done in {time.time()-t0:.1f}s")

        # Determine SVD ranks based on head_dim
        # Rule: rank_k = 3/4 * hd, rank_v = 3/8 * hd
        rank_k = min(48, int(hd * 3 / 4))
        rank_v = min(24, int(hd * 3 / 8))
        if hd == 128:
            rank_k, rank_v = 96, 48  # scale up for larger hd
        print(f"  SVD ranks: K={rank_k}, V={rank_v} (hd={hd})")

        # For Qwen with GQA: the cache has kv_heads, not nh
        # Our FullStackLayer handles this transparently since it
        # operates on whatever shape (B, H, T, D) comes in

        # Define configs
        def make_vanilla_tq(li, hd=hd):
            kq = get_q(hd, 4, device, 42 + li * 2)
            vq = get_q(hd, 2, device, 42 + li * 2 + 1)
            return FullStackLayer(rank_k=hd, rank_v=hd, kq=kq, vq=vq,
                                   num_sinks=0, residual_window=32, min_chunk_for_svd=9999)

        def make_our_tq(li, hd=hd):
            if li == 0: return FP16Layer()
            kq = get_q(hd, 4, device, 42 + li * 2)
            vq = get_q(hd, 2, device, 42 + li * 2 + 1)
            return FullStackLayer(rank_k=hd, rank_v=hd, kq=kq, vq=vq,
                                   num_sinks=4, residual_window=8, min_chunk_for_svd=9999)

        def make_svd_only(li, rk=rank_k, rv=rank_v):
            if li == 0: return FP16Layer()
            return FullStackLayer(rank_k=rk, rank_v=rv, kq=None, vq=None,
                                   num_sinks=4, residual_window=8)

        def make_full_stack(li, rk=rank_k, rv=rank_v, hd=hd):
            if li == 0: return FP16Layer()
            kq_r = min(rk, hd)
            vq_r = min(rv, hd)
            kq = get_q(kq_r, 4, device, 500 + kq_r * 10 + 4)
            vq = get_q(vq_r, 3, device, 500 + vq_r * 10 + 3)
            return FullStackLayer(rank_k=rk, rank_v=rv, kq=kq, vq=vq,
                                   num_sinks=4, residual_window=8)

        def make_full_stack_k4v4(li, rk=rank_k, rv=rank_v, hd=hd):
            if li == 0: return FP16Layer()
            kq_r = min(rk, hd)
            vq_r = min(rv, hd)
            kq = get_q(kq_r, 4, device, 500 + kq_r * 10 + 4)
            vq = get_q(vq_r, 4, device, 500 + vq_r * 10 + 4)
            return FullStackLayer(rank_k=rk, rank_v=rv, kq=kq, vq=vq,
                                   num_sinks=4, residual_window=8)

        named_configs = [
            ("Vanilla TQ K4/V2 w32", make_vanilla_tq, 'vanilla'),
            ("L0fp16+TQ K4/V2 w8 s4", make_our_tq, 'our_tq'),
            (f"SVD({rank_k}/{rank_v}) only w8 s4", make_svd_only, 'svd_only'),
            (f"SVD({rank_k}/{rank_v})+TQ K4/V3 w8 s4", make_full_stack, 'full_k4v3'),
            (f"SVD({rank_k}/{rank_v})+TQ K4/V4 w8 s4", make_full_stack_k4v4, 'full_k4v4'),
        ]

        model_data = {'lengths': [], 'configs': {}}
        for cfg_name, _, cfg_key in named_configs:
            model_data['configs'][cfg_key] = {'name': cfg_name, 'ppl': [], 'delta': [], 'comp': []}

        print(f"\n  {'Len':>5s} {'Baseline':>9s}", end="")
        for cfg_name, _, _ in named_configs:
            short = cfg_name[:22]
            print(f" {short:>24s}", end="")
        print()
        print(f"  {'-'*135}")

        for target_len in mcfg['lengths']:
            text = get_text_at_length(tok, target_len)
            actual_len = len(tok.encode(text))
            model_data['lengths'].append(actual_len)

            ppl_base = compute_ppl(model, tok, text)

            print(f"  {actual_len:>5d} {ppl_base:>9.2f}", end="")

            for cfg_name, layer_fn, cfg_key in named_configs:
                try:
                    cache = make_cache(nl, layer_fn)
                    ppl = compute_ppl(model, tok, text, cache=cache)
                    delta = ((ppl - ppl_base) / ppl_base) * 100

                    # Compression ratio
                    comp = compute_compression(
                        actual_len, hd, kv_heads if has_gqa else nh, nl,
                        cfg_name, rank_k=rank_k, rank_v=rank_v
                    )

                    model_data['configs'][cfg_key]['ppl'].append(ppl)
                    model_data['configs'][cfg_key]['delta'].append(delta)
                    model_data['configs'][cfg_key]['comp'].append(comp)

                    indicator = "!!" if delta < -1 else ("*" if delta < 2 else ("" if delta < 10 else "X"))
                    print(f" {delta:>+6.1f}% {comp:>4.1f}x {indicator:>2s}", end="")
                except Exception as e:
                    print(f" {'ERROR':>14s}", end="")
                    model_data['configs'][cfg_key]['ppl'].append(float('nan'))
                    model_data['configs'][cfg_key]['delta'].append(float('nan'))
                    model_data['configs'][cfg_key]['comp'].append(float('nan'))

            print()

        all_model_data[model_name] = model_data

        del model
        torch.cuda.empty_cache()

    # ── Plot: PPL delta vs sequence length ──
    fig, axes = plt.subplots(1, len(all_model_data), figsize=(8 * len(all_model_data), 7))
    if len(all_model_data) == 1:
        axes = [axes]

    style = {
        'vanilla':  {'color': '#aaaaaa', 'marker': 'o', 'ls': '-',  'label': 'Vanilla TQ K4/V2'},
        'our_tq':   {'color': '#666666', 'marker': 's', 'ls': '-',  'label': 'L0fp16+TQ K4/V2'},
        'svd_only': {'color': '#3498db', 'marker': 'D', 'ls': '--', 'label': 'SVD only'},
        'full_k4v3':{'color': '#e74c3c', 'marker': '*', 'ls': '-',  'label': 'SVD+TQ K4/V3'},
        'full_k4v4':{'color': '#e67e22', 'marker': '^', 'ls': '-',  'label': 'SVD+TQ K4/V4'},
    }

    for ax, (model_name, mdata) in zip(axes, all_model_data.items()):
        lengths = mdata['lengths']

        for cfg_key, cfg_data in mdata['configs'].items():
            deltas = cfg_data['delta']
            s = style[cfg_key]
            valid = [(l, d) for l, d in zip(lengths, deltas) if not np.isnan(d) and abs(d) < 50]
            if valid:
                ls, ds = zip(*valid)
                ax.plot(ls, ds, color=s['color'], marker=s['marker'], ls=s['ls'],
                       lw=2, markersize=8, label=s['label'], alpha=0.85)

        ax.axhline(0, color='black', lw=1)
        ax.axhline(2, color='orange', ls='--', alpha=0.3, lw=1)
        ax.axhline(-2, color='green', ls='--', alpha=0.3, lw=1)
        ax.fill_between([0, max(lengths)*1.1], -2, 2, alpha=0.04, color='green')

        ax.set_xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
        ax.set_ylabel('PPL Degradation (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.15)
        ax.set_xlim(0, max(lengths) * 1.05)

    fig.suptitle('Compression Quality vs Sequence Length\n'
                 '(green band = within 2% of baseline)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/long_context_scaling.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: results/long_context_scaling.png")

    # ── Plot: Compression ratio vs sequence length ──
    fig2, axes2 = plt.subplots(1, len(all_model_data), figsize=(8 * len(all_model_data), 6))
    if len(all_model_data) == 1:
        axes2 = [axes2]

    for ax, (model_name, mdata) in zip(axes2, all_model_data.items()):
        lengths = mdata['lengths']
        for cfg_key, cfg_data in mdata['configs'].items():
            comps = cfg_data['comp']
            s = style[cfg_key]
            valid = [(l, c) for l, c in zip(lengths, comps) if not np.isnan(c)]
            if valid:
                ls, cs = zip(*valid)
                ax.plot(ls, cs, color=s['color'], marker=s['marker'], ls=s['ls'],
                       lw=2, markersize=8, label=s['label'], alpha=0.85)

        ax.set_xlabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    fig2.suptitle('Compression Ratio vs Sequence Length\n(higher = more compression)',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/compression_vs_length.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: results/compression_vs_length.png")

    # ── Final summary table ──
    print(f"\n{'='*85}")
    print("FINAL SUMMARY")
    print(f"{'='*85}")

    for model_name, mdata in all_model_data.items():
        print(f"\n  {model_name}:")
        lengths = mdata['lengths']

        # Find the config that's best (lowest avg |delta|) while staying under 5%
        best_cfg = None
        best_score = float('inf')
        for cfg_key, cfg_data in mdata['configs'].items():
            deltas = [d for d in cfg_data['delta'] if not np.isnan(d)]
            if not deltas:
                continue
            max_delta = max(deltas)
            avg_delta = np.mean(deltas)
            if max_delta < 10:  # don't consider broken configs
                score = avg_delta
                if score < best_score:
                    best_score = score
                    best_cfg = cfg_key

        if best_cfg:
            cfg = mdata['configs'][best_cfg]
            print(f"    Best overall: {cfg['name']}")
            for i, l in enumerate(lengths):
                if i < len(cfg['delta']):
                    d = cfg['delta'][i]
                    c = cfg['comp'][i]
                    print(f"      {l:>5d} tokens: {d:>+6.1f}% PPL, {c:.1f}x compression")


if __name__ == "__main__":
    main()
