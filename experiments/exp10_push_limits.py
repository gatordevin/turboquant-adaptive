"""
Pushing the Limits: More compression at equal or better quality
================================================================
Based on everything we've learned, test strategies that trade off
different axes to maximize compression while minimizing quality loss.

Key insights driving these experiments:
1. Keys matter 10x more than values (softmax amplification)
2. Recent tokens matter most (residual window)
3. Layer 0 (and sometimes 1-2) are load-bearing
4. Sink tokens cost almost nothing to protect
5. The 2-bit key cliff is absolute — but 1-bit VALUES might be okay

Strategies:
A. ASYMMETRIC EXTREME: K4/V1 — push values to minimum
B. WINDOW OPTIMIZATION: trade bits for window size
C. PER-LAYER WINDOWS: sensitive layers get big windows
D. MULTI-LAYER FP16: protect first N layers entirely
E. HYBRID: combine the best of A-D
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
from transformers.cache_utils import DynamicCache, CacheLayerMixin

from turboquant_gpt2 import TurboQuantizer

# ── Quantizer cache ──
_qcache = {}
def get_q(head_dim, bits, device, seed):
    key = (head_dim, bits, seed)
    if key not in _qcache:
        _qcache[key] = TurboQuantizer(head_dim, bits, device=device, seed=seed)
    return _qcache[key]


# ============================================================================
# Flexible cache layers
# ============================================================================

class FP16Layer(CacheLayerMixin):
    is_sliding = False
    def lazy_initialization(self, k, v):
        self.dtype, self.device = k.dtype, k.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True
    def update(self, k, v, cache_kwargs=None):
        if not self.is_initialized: self.lazy_initialization(k, v)
        self.keys = k if self.keys.numel() == 0 else torch.cat([self.keys, k], dim=-2)
        self.values = v if self.values.numel() == 0 else torch.cat([self.values, v], dim=-2)
        return self.keys, self.values
    def get_seq_length(self):
        return 0 if not self.is_initialized or self.keys.numel() == 0 else self.keys.shape[-2]
    def get_max_cache_shape(self): return -1
    def crop(self, m): pass
    def batch_repeat_interleave(self, r):
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(r, dim=0)
            self.values = self.values.repeat_interleave(r, dim=0)
    def batch_select_indices(self, idx):
        if self.get_seq_length() > 0:
            self.keys = self.keys[idx, ...]; self.values = self.values[idx, ...]
    def get_mask_sizes(self, cp): return self.get_seq_length() + cp.shape[0], 0


class FlexLayer(CacheLayerMixin):
    """Flexible quantized layer with sink protection and configurable bits/window."""
    is_sliding = False

    def __init__(self, kq, vq, num_sinks=0, residual_window=32):
        super().__init__()
        self.kq, self.vq = kq, vq
        self.num_sinks = num_sinks
        self.residual_window = residual_window
        self.sink_keys = self.sink_values = None
        self.comp_k, self.comp_v, self.comp_shapes = [], [], []
        self.recent_keys = self.recent_values = None
        self.total_seen = 0

    def lazy_initialization(self, k, v):
        self.dtype, self.device = k.dtype, k.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        B, H, T, D = key_states.shape

        if self.total_seen == 0:
            if self.num_sinks > 0 and T >= self.num_sinks:
                self.sink_keys = key_states[:, :, :self.num_sinks, :].clone()
                self.sink_values = value_states[:, :, :self.num_sinks, :].clone()
                rem_k = key_states[:, :, self.num_sinks:, :]
                rem_v = value_states[:, :, self.num_sinks:, :]
            else:
                rem_k, rem_v = key_states, value_states
            self.recent_keys = rem_k if rem_k.shape[-2] > 0 else key_states[:, :, :0, :]
            self.recent_values = rem_v if rem_v.shape[-2] > 0 else value_states[:, :, :0, :]
            self.total_seen = T
        else:
            if self.recent_keys.shape[-2] == 0:
                self.recent_keys, self.recent_values = key_states, value_states
            else:
                self.recent_keys = torch.cat([self.recent_keys, key_states], dim=-2)
                self.recent_values = torch.cat([self.recent_values, value_states], dim=-2)
            self.total_seen += T

        # Compress overflow
        if self.recent_keys.shape[-2] > self.residual_window:
            overflow = self.recent_keys.shape[-2] - self.residual_window
            k_o = self.recent_keys[:, :, :overflow, :]
            v_o = self.recent_values[:, :, :overflow, :]
            self.recent_keys = self.recent_keys[:, :, overflow:, :].contiguous()
            self.recent_values = self.recent_values[:, :, overflow:, :].contiguous()

            B2, H2, T2, D2 = k_o.shape
            kf, vf = k_o.reshape(B2*H2*T2, D2), v_o.reshape(B2*H2*T2, D2)
            ki, kn = self.kq.quantize(kf)
            vi, vn = self.vq.quantize(vf)
            self.comp_k.append((ki, kn))
            self.comp_v.append((vi, vn))
            self.comp_shapes.append((B2, H2, T2))

        # Reconstruct
        pk, pv = [], []
        if self.sink_keys is not None:
            pk.append(self.sink_keys); pv.append(self.sink_values)
        for i, (B2, H2, T2) in enumerate(self.comp_shapes):
            ki, kn = self.comp_k[i]
            vi, vn = self.comp_v[i]
            pk.append(self.kq.dequantize(ki, kn).reshape(B2, H2, T2, -1))
            pv.append(self.vq.dequantize(vi, vn).reshape(B2, H2, T2, -1))
        pk.append(self.recent_keys); pv.append(self.recent_values)
        self.keys = torch.cat(pk, dim=-2)
        self.values = torch.cat(pv, dim=-2)
        return self.keys, self.values

    def get_seq_length(self):
        s = self.sink_keys.shape[-2] if self.sink_keys is not None else 0
        c = sum(sh[2] for sh in self.comp_shapes)
        r = self.recent_keys.shape[-2] if self.recent_keys is not None and self.recent_keys.numel() > 0 else 0
        return s + c + r
    def get_max_cache_shape(self): return -1
    def crop(self, m): pass
    def batch_repeat_interleave(self, r):
        for t in [self.sink_keys, self.sink_values, self.recent_keys, self.recent_values]:
            if t is not None and t.numel() > 0: t.data = t.repeat_interleave(r, dim=0)
    def batch_select_indices(self, idx):
        for t in [self.sink_keys, self.sink_values, self.recent_keys, self.recent_values]:
            if t is not None and t.numel() > 0: t.data = t[idx, ...]
    def get_mask_sizes(self, cp): return self.get_seq_length() + cp.shape[0], 0


# ============================================================================
# Cache builders
# ============================================================================

def build_cache(head_dim, num_layers, device, layer_configs):
    """Build cache from per-layer config list.

    layer_configs: list of dicts with keys:
        'mode': 'fp16' | 'quant'
        'kb', 'vb': key/value bits (for quant mode)
        'sinks': num sink tokens
        'window': residual window size
    """
    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None
    for li, cfg in enumerate(layer_configs):
        if cfg['mode'] == 'fp16':
            cache.layers.append(FP16Layer())
        else:
            kq = get_q(head_dim, cfg['kb'], device, 42 + li * 2)
            vq = get_q(head_dim, cfg['vb'], device, 42 + li * 2 + 1)
            cache.layers.append(FlexLayer(kq, vq, num_sinks=cfg.get('sinks', 0),
                                           residual_window=cfg.get('window', 32)))
    return cache


def compute_theoretical_compression(num_layers, head_dim, num_heads, seq_len, layer_configs):
    """Compute theoretical compression ratio."""
    fp16_per_tok = head_dim * 2 * 2  # K+V in fp16
    total_fp16 = seq_len * fp16_per_tok * num_heads * num_layers

    total_compressed = 0
    for cfg in layer_configs:
        if cfg['mode'] == 'fp16':
            total_compressed += seq_len * fp16_per_tok * num_heads
        else:
            sinks = cfg.get('sinks', 0)
            win = cfg.get('window', 32)
            kb, vb = cfg['kb'], cfg['vb']
            sink_tok = min(sinks, seq_len)
            recent_tok = min(win, max(0, seq_len - sink_tok))
            comp_tok = max(0, seq_len - sink_tok - recent_tok)
            comp_per = head_dim * kb / 8 + 2 + head_dim * vb / 8 + 2
            total_compressed += (sink_tok * fp16_per_tok + recent_tok * fp16_per_tok +
                                  comp_tok * comp_per) * num_heads
    return total_fp16 / total_compressed if total_compressed > 0 else 1.0


def compute_ppl(model, tokenizer, text, cache=None):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(input_ids, past_key_values=cache)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return torch.exp(loss).item()


# ============================================================================
# Strategy definitions
# ============================================================================

def strategies_for_model(num_layers, head_dim):
    """Generate strategy configs for a given model size."""
    S = []

    # ── References ──
    S.append(("Vanilla K4/V2 w64", [
        {'mode': 'quant', 'kb': 4, 'vb': 2, 'sinks': 0, 'window': 64}
    ] * num_layers))

    S.append(("Vanilla K4/V2 w32", [
        {'mode': 'quant', 'kb': 4, 'vb': 2, 'sinks': 0, 'window': 32}
    ] * num_layers))

    S.append(("Sink(4)+K4/V2 w32", [
        {'mode': 'quant', 'kb': 4, 'vb': 2, 'sinks': 4, 'window': 32}
    ] * num_layers))

    # ── Strategy A: Asymmetric extreme — push values to V1 ──
    S.append(("A: K4/V1 w32 sink(4)", [
        {'mode': 'quant', 'kb': 4, 'vb': 1, 'sinks': 4, 'window': 32}
    ] * num_layers))

    S.append(("A: K4/V1 w64 sink(4)", [
        {'mode': 'quant', 'kb': 4, 'vb': 1, 'sinks': 4, 'window': 64}
    ] * num_layers))

    S.append(("A: K3/V1 w64 sink(4)", [
        {'mode': 'quant', 'kb': 3, 'vb': 1, 'sinks': 4, 'window': 64}
    ] * num_layers))

    # ── Strategy B: Window optimization — trade bits for window ──
    # Same memory budget as K4/V2 w32 but different allocation
    S.append(("B: K3/V2 w128 sink(4)", [
        {'mode': 'quant', 'kb': 3, 'vb': 2, 'sinks': 4, 'window': 128}
    ] * num_layers))

    S.append(("B: K4/V1 w128 sink(4)", [
        {'mode': 'quant', 'kb': 4, 'vb': 1, 'sinks': 4, 'window': 128}
    ] * num_layers))

    S.append(("B: K3/V1 w128 sink(4)", [
        {'mode': 'quant', 'kb': 3, 'vb': 1, 'sinks': 4, 'window': 128}
    ] * num_layers))

    # ── Strategy C: Per-layer windows ──
    # Layer 0: fp16, layers 1-3: big window, rest: small window
    cfgs_c1 = []
    for li in range(num_layers):
        if li == 0:
            cfgs_c1.append({'mode': 'fp16'})
        elif li <= 3:
            cfgs_c1.append({'mode': 'quant', 'kb': 4, 'vb': 2, 'sinks': 4, 'window': 64})
        else:
            cfgs_c1.append({'mode': 'quant', 'kb': 4, 'vb': 1, 'sinks': 4, 'window': 16})
    S.append(("C: L0=fp16, L1-3=K4V2w64, rest=K4V1w16", cfgs_c1))

    cfgs_c2 = []
    for li in range(num_layers):
        if li == 0:
            cfgs_c2.append({'mode': 'fp16'})
        elif li <= 2:
            cfgs_c2.append({'mode': 'quant', 'kb': 4, 'vb': 2, 'sinks': 4, 'window': 64})
        else:
            cfgs_c2.append({'mode': 'quant', 'kb': 3, 'vb': 1, 'sinks': 4, 'window': 32})
    S.append(("C: L0=fp16, L1-2=K4V2w64, rest=K3V1w32", cfgs_c2))

    # ── Strategy D: Multi-layer fp16 ──
    for n_fp16 in [1, 2, 3]:
        cfgs_d = []
        for li in range(num_layers):
            if li < n_fp16:
                cfgs_d.append({'mode': 'fp16'})
            else:
                cfgs_d.append({'mode': 'quant', 'kb': 4, 'vb': 1, 'sinks': 4, 'window': 32})
        fp16_pct = n_fp16 / num_layers * 100
        S.append((f"D: L0-{n_fp16-1}=fp16, rest=K4V1w32 ({fp16_pct:.0f}%fp16)", cfgs_d))

    # ── Strategy E: Hybrid best-of-all ──
    # L0 fp16, L1-2 K4/V2 big window, rest K4/V1 small window, sinks everywhere
    cfgs_e1 = []
    for li in range(num_layers):
        if li == 0:
            cfgs_e1.append({'mode': 'fp16'})
        elif li <= 2:
            cfgs_e1.append({'mode': 'quant', 'kb': 4, 'vb': 2, 'sinks': 4, 'window': 64})
        else:
            cfgs_e1.append({'mode': 'quant', 'kb': 4, 'vb': 1, 'sinks': 4, 'window': 24})
    S.append(("E1: L0=fp16 L1-2=K4V2w64 rest=K4V1w24 s4", cfgs_e1))

    # Aggressive: fewer fp16 layers, more compression
    cfgs_e2 = []
    for li in range(num_layers):
        if li == 0:
            cfgs_e2.append({'mode': 'fp16'})
        else:
            cfgs_e2.append({'mode': 'quant', 'kb': 4, 'vb': 1, 'sinks': 4, 'window': 32})
    S.append(("E2: L0=fp16 rest=K4V1w32 s4", cfgs_e2))

    # Ultra-aggressive: K3/V1 everywhere except L0
    cfgs_e3 = []
    for li in range(num_layers):
        if li == 0:
            cfgs_e3.append({'mode': 'fp16'})
        elif li <= 2:
            cfgs_e3.append({'mode': 'quant', 'kb': 4, 'vb': 2, 'sinks': 4, 'window': 48})
        else:
            cfgs_e3.append({'mode': 'quant', 'kb': 3, 'vb': 1, 'sinks': 4, 'window': 24})
    S.append(("E3: L0=fp16 L1-2=K4V2w48 rest=K3V1w24 s4", cfgs_e3))

    return S


# ============================================================================
# Main
# ============================================================================

def run_model(model_name, device='cuda'):
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation='eager').to(device).half()
    model.eval()

    head_dim = model.config.n_embd // model.config.n_head
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    print(f"  {num_layers} layers, hd={head_dim}, heads={num_heads}")

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
        "generation, and they were given millions of dollars to make this vision come true."
    )
    seq_len = len(tokenizer.encode(eval_text))
    print(f"  Eval: {seq_len} tokens")

    # Precompute codebooks
    print(f"  Precomputing codebooks...")
    for bits in [1, 2, 3, 4]:
        for li in range(num_layers):
            get_q(head_dim, bits, device, 42 + li * 2)
            get_q(head_dim, bits, device, 42 + li * 2 + 1)

    ppl_base = compute_ppl(model, tokenizer, eval_text)
    print(f"  Baseline fp16: PPL = {ppl_base:.2f}\n")

    strategies = strategies_for_model(num_layers, head_dim)
    results = []

    print(f"  {'Strategy':55s} {'PPL':>7s} {'ΔPPL':>7s} {'Comp':>6s} {'Cat':>5s}")
    print(f"  {'-'*85}")

    for name, layer_configs in strategies:
        cache = build_cache(head_dim, num_layers, device, layer_configs)
        ppl = compute_ppl(model, tokenizer, eval_text, cache=cache)
        delta = ((ppl - ppl_base) / ppl_base) * 100
        comp = compute_theoretical_compression(num_layers, head_dim, num_heads, seq_len, layer_configs)

        cat = name[0] if name[0] in 'ABCDE' else 'ref'
        results.append((name, ppl, delta, comp, cat))
        print(f"  {name:55s} {ppl:>7.2f} {delta:>+6.1f}% {comp:>5.1f}x  {cat}")

    # ── Pareto frontier ──
    print(f"\n  PARETO FRONTIER (best PPL at each compression level):")
    print(f"  {'Strategy':55s} {'PPL':>7s} {'ΔPPL':>7s} {'Comp':>6s}")
    print(f"  {'-'*75}")

    sorted_results = sorted(results, key=lambda r: r[3])  # sort by compression
    pareto = []
    best_ppl = float('inf')
    for r in sorted_results:
        if r[1] <= best_ppl:
            best_ppl = r[1]
            pareto.append(r)

    for r in pareto:
        star = " ★" if r[2] < 3.0 and r[3] > 2.0 else ""
        print(f"  {r[0]:55s} {r[1]:>7.2f} {r[2]:>+6.1f}% {r[3]:>5.1f}x{star}")

    del model
    torch.cuda.empty_cache()
    return results, ppl_base, num_layers, head_dim, num_heads, seq_len


def plot_results(all_model_results):
    """Generate Pareto frontier plot for all models."""
    fig, axes = plt.subplots(1, len(all_model_results), figsize=(8 * len(all_model_results), 7))
    if len(all_model_results) == 1:
        axes = [axes]

    cat_colors = {'r': '#888888', 'A': '#e74c3c', 'B': '#2ecc71', 'C': '#9b59b6',
                  'D': '#3498db', 'E': '#e67e22'}
    cat_markers = {'r': 'o', 'A': 's', 'B': 'D', 'C': '^', 'D': 'v', 'E': '*'}
    cat_labels = {'r': 'Reference', 'A': 'A: Asymmetric (V1)', 'B': 'B: Window tradeoff',
                  'C': 'C: Per-layer window', 'D': 'D: Multi-layer fp16', 'E': 'E: Hybrid'}

    for ax, (model_name, results, ppl_base, nl, hd, nh, sl) in zip(axes, all_model_results):
        for name, ppl, delta, comp, cat in results:
            c = cat_colors.get(cat, '#999')
            m = cat_markers.get(cat, 'o')
            sz = 150 if cat == 'E' else 70
            edge = 'black' if cat == 'E' else 'none'
            lw = 2 if cat == 'E' else 0
            ax.scatter(comp, delta, c=c, marker=m, s=sz, alpha=0.8,
                      edgecolors=edge, linewidths=lw, zorder=3)

        # Pareto line
        sorted_r = sorted(results, key=lambda r: r[3])
        pareto = []
        best = float('inf')
        for r in sorted_r:
            if r[2] <= best:
                best = r[2]
                pareto.append(r)
        px = [p[3] for p in pareto]
        py = [p[2] for p in pareto]
        ax.plot(px, py, 'k--', alpha=0.4, lw=1.5)

        # Label Pareto points
        for p in pareto:
            if p[2] < 5 and p[3] > 2:
                ax.annotate(p[0][:25], (p[3], p[2]), fontsize=6, ha='left',
                           xytext=(4, 3), textcoords='offset points', alpha=0.7)

        ax.axhline(0, color='black', lw=0.8)
        ax.axhline(2, color='orange', ls='--', alpha=0.3, label='2% threshold')
        ax.axhline(5, color='red', ls='--', alpha=0.2, label='5% threshold')
        ax.fill_between([0, 10], 0, 2, alpha=0.04, color='green')

        ax.set_xlabel('Compression Ratio (×)', fontsize=11, fontweight='bold')
        ax.set_ylabel('PPL Degradation (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}\n(baseline PPL={ppl_base:.1f}, {nl}L)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.15)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker=cat_markers[c], color='w', markerfacecolor=cat_colors[c],
               markersize=9, label=cat_labels[c])
        for c in ['r', 'A', 'B', 'C', 'D', 'E']
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Pushing the Limits: Compression vs Quality Pareto Frontiers',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/push_limits_pareto.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: results/push_limits_pareto.png")


def main():
    device = 'cuda'
    print("=" * 80)
    print("PUSHING THE LIMITS — More Compression, Less Quality Loss")
    print("=" * 80)

    all_results = []

    for model_name in ['gpt2-large', 'gpt2-xl']:
        results, ppl_base, nl, hd, nh, sl = run_model(model_name, device)
        all_results.append((model_name, results, ppl_base, nl, hd, nh, sl))

    plot_results(all_results)

    # ── Final summary ──
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: BEST STRATEGIES PER MODEL")
    print("=" * 80)

    for model_name, results, ppl_base, nl, hd, nh, sl in all_results:
        print(f"\n  {model_name} (baseline PPL={ppl_base:.2f}):")

        # Best at <2% PPL
        under2 = [r for r in results if r[2] < 2.0 and r[3] > 1.0]
        if under2:
            best = max(under2, key=lambda r: r[3])
            print(f"    Best <2% PPL:  {best[0]:50s} = {best[2]:+.1f}% at {best[3]:.1f}x")

        # Best at <5% PPL
        under5 = [r for r in results if r[2] < 5.0 and r[3] > 1.0]
        if under5:
            best = max(under5, key=lambda r: r[3])
            print(f"    Best <5% PPL:  {best[0]:50s} = {best[2]:+.1f}% at {best[3]:.1f}x")

        # Most compressed
        most = max(results, key=lambda r: r[3])
        print(f"    Most compressed: {most[0]:48s} = {most[2]:+.1f}% at {most[3]:.1f}x")


if __name__ == "__main__":
    main()
