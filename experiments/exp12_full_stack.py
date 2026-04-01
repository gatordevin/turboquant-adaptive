"""
Full Stack: SVD + TurboQuant + Sinks + Adaptive Tiers
======================================================
Test whether ALL our findings compose:
  1. Low-rank SVD (cross-token compression)
  2. TurboQuant (per-vector quantization)
  3. Sink protection (fp16 on first 4 tokens)
  4. Layer 0 in fp16 (sensitivity)
  5. Adaptive tiers (K4/V2 vs K3/V2 per layer)

Test across: GPT-2 Large, GPT-2 XL
Test at: short (200 tok), medium (400 tok), long (800 tok)
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

_qcache = {}
def get_q(hd, bits, device, seed):
    k = (hd, bits, seed)
    if k not in _qcache:
        _qcache[k] = TurboQuantizer(hd, bits, device=device, seed=seed)
    return _qcache[k]


# ============================================================================
# Full-stack cache layer: SVD + TurboQuant + Sinks
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


class FullStackLayer(CacheLayerMixin):
    """The full compression pipeline:

    On write (overflow from residual window):
      1. Separate sink tokens (fp16, permanent)
      2. For remaining tokens in a chunk:
         a. Per-head SVD: K_chunk ≈ U_k @ diag(S_k) @ Vt_k (keep top rank_k)
         b. Coefficients C_k = U_k * S_k  (shape: T × rank_k)
         c. TurboQuant the coefficients at kb bits
         d. Store: quantized coefficients + fp16 basis Vt_k
         e. Same for values with rank_v and vb bits

    On read:
      1. Dequantize coefficients
      2. Multiply by basis: K_hat = C_hat @ Vt
      3. Concatenate: [sinks_fp16 | compressed_chunks | recent_fp16]

    Compression ratio:
      Original: T × D × 2 bytes (fp16)
      Compressed: T × rank × bits/8 + rank × D × 2 + T × 2 (norms)
      For rank=32, D=64, bits=4: T×32×0.5 + 32×64×2 + T×2 = 18T + 4096
      vs original T×64×2 = 128T
      At T=200: 7696 vs 25600 = 3.3x (just from SVD+quant on compressed part)
    """

    is_sliding = False

    def __init__(self, rank_k=32, rank_v=16, kq=None, vq=None,
                 num_sinks=4, residual_window=8, min_chunk_for_svd=8):
        super().__init__()
        self.rank_k = rank_k
        self.rank_v = rank_v
        self.kq = kq
        self.vq = vq
        self.num_sinks = num_sinks
        self.residual_window = residual_window
        self.min_chunk_for_svd = min_chunk_for_svd

        self.sink_keys = self.sink_values = None
        self.chunks = []  # list of compressed chunk data
        self.recent_keys = self.recent_values = None
        self.total_seen = 0

    def lazy_initialization(self, k, v):
        self.dtype, self.device = k.dtype, k.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def _compress_chunk(self, X, rank, quantizer):
        """SVD + TurboQuant on a chunk. X: (BH, T, D)."""
        BH, T, D = X.shape

        if T < self.min_chunk_for_svd or rank >= min(T, D):
            # Too small for SVD — just quantize directly
            if quantizer is not None:
                flat = X.reshape(BH * T, D)
                idx, norms = quantizer.quantize(flat)
                return {'type': 'quant', 'idx': idx, 'norms': norms, 'BH': BH, 'T': T, 'D': D}
            else:
                return {'type': 'raw', 'data': X}

        # SVD
        X_f = X.float()
        U, S, Vt = torch.linalg.svd(X_f, full_matrices=False)

        # Keep top rank
        r = min(rank, T, D)
        U_r = U[:, :, :r]           # (BH, T, r)
        S_r = S[:, :r]               # (BH, r)
        Vt_r = Vt[:, :r, :].half()   # (BH, r, D) — basis, stored fp16

        # Coefficients
        coeffs = (U_r * S_r.unsqueeze(1)).half()  # (BH, T, r)

        # Quantize coefficients
        if quantizer is not None:
            c_flat = coeffs.reshape(BH * T, r)
            idx, norms = quantizer.quantize(c_flat)
            return {'type': 'svd+quant', 'idx': idx, 'norms': norms,
                    'basis': Vt_r, 'BH': BH, 'T': T, 'r': r, 'D': D}
        else:
            return {'type': 'svd', 'coeffs': coeffs, 'basis': Vt_r,
                    'BH': BH, 'T': T, 'r': r, 'D': D}

    def _decompress_chunk(self, chunk, quantizer):
        """Reconstruct from compressed chunk."""
        if chunk['type'] == 'raw':
            return chunk['data']
        elif chunk['type'] == 'quant':
            hat = quantizer.dequantize(chunk['idx'], chunk['norms'])
            return hat.reshape(chunk['BH'], chunk['T'], chunk['D'])
        elif chunk['type'] == 'svd':
            return torch.bmm(chunk['coeffs'], chunk['basis'])
        elif chunk['type'] == 'svd+quant':
            c_hat = quantizer.dequantize(chunk['idx'], chunk['norms'])
            c_hat = c_hat.reshape(chunk['BH'], chunk['T'], chunk['r'])
            return torch.bmm(c_hat, chunk['basis'])

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        B, H, T, D = key_states.shape

        # First call: capture sinks
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
            k_flat = k_o.reshape(B2 * H2, T2, D2)
            v_flat = v_o.reshape(B2 * H2, T2, D2)

            k_comp = self._compress_chunk(k_flat, self.rank_k, self.kq)
            v_comp = self._compress_chunk(v_flat, self.rank_v, self.vq)
            self.chunks.append((k_comp, v_comp, B2, H2, T2, D2))

        # Reconstruct
        pk, pv = [], []
        if self.sink_keys is not None:
            pk.append(self.sink_keys); pv.append(self.sink_values)

        for k_comp, v_comp, B2, H2, T2, D2 in self.chunks:
            k_hat = self._decompress_chunk(k_comp, self.kq).reshape(B2, H2, T2, D2)
            v_hat = self._decompress_chunk(v_comp, self.vq).reshape(B2, H2, T2, D2)
            pk.append(k_hat); pv.append(v_hat)

        pk.append(self.recent_keys); pv.append(self.recent_values)
        self.keys = torch.cat(pk, dim=-2)
        self.values = torch.cat(pv, dim=-2)
        return self.keys, self.values

    def get_seq_length(self):
        s = self.sink_keys.shape[-2] if self.sink_keys is not None else 0
        c = sum(ch[4] for ch in self.chunks)
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

def make_cache(nl, layer_fn):
    cache = DynamicCache()
    cache.layers = [layer_fn(li) for li in range(nl)]
    cache.layer_class_to_replicate = None
    return cache


def compute_ppl(model, tokenizer, text, cache=None):
    ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(ids, past_key_values=cache)
        logits = out.logits[:, :-1, :]
        targets = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return torch.exp(loss).item()


# ============================================================================
# Main
# ============================================================================

def main():
    device = 'cuda'
    print("=" * 85)
    print("FULL STACK: SVD + TurboQuant + Sinks + Adaptive — Cross-Model Validation")
    print("=" * 85)

    # Eval texts of increasing length
    base_text = (
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
        "difficulty of the project. In 1974, in response to the criticism from James Lighthill and "
        "ongoing pressure from congress, the U.S. and British governments cut off exploratory research "
        "in AI. The next few years would later be called an AI winter, a period when funding for AI "
        "projects was extremely difficult. In the early 1980s, AI research was revived by the commercial "
        "success of expert systems, a form of AI program that simulated the knowledge and analytical "
        "skills of human experts."
    )

    models_to_test = ['gpt2-large', 'gpt2-xl']
    all_results = {}

    for model_name in models_to_test:
        print(f"\n{'='*85}")
        print(f"MODEL: {model_name}")
        print(f"{'='*85}")

        tok = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation='eager').to(device).half()
        model.eval()

        hd = model.config.n_embd // model.config.n_head
        nl = model.config.n_layer
        nh = model.config.n_head
        print(f"  {nl} layers, hd={hd}, heads={nh}")

        # Precompute codebooks
        print("  Precomputing codebooks...")
        for bits in [2, 3, 4]:
            for li in range(nl):
                get_q(hd, bits, device, 42 + li * 2)
                get_q(hd, bits, device, 42 + li * 2 + 1)
            for rank in [16, 24, 32, 48]:
                get_q(rank, bits, device, 500 + rank * 10 + bits)

        # Test at multiple sequence lengths
        texts = {
            'short': base_text[:500],
            'medium': base_text,
            'long': base_text * 2,
        }

        model_results = {}

        for length_name, text in texts.items():
            seq = len(tok.encode(text))
            print(f"\n  --- {length_name}: {seq} tokens ---")

            ppl_base = compute_ppl(model, tok, text)
            print(f"  Baseline fp16: PPL = {ppl_base:.2f}")

            configs = []

            # ── Reference: vanilla TurboQuant ──
            def ref_vanilla(li):
                kq = get_q(hd, 4, device, 42 + li * 2)
                vq = get_q(hd, 2, device, 42 + li * 2 + 1)
                return FullStackLayer(rank_k=hd, rank_v=hd, kq=kq, vq=vq,
                                       num_sinks=0, residual_window=32, min_chunk_for_svd=9999)
            configs.append(("Vanilla TQ K4/V2 w32", ref_vanilla))

            # ── Our best from exp10: L0fp16 + K4V2 w8 + sinks ──
            def ref_best(li):
                if li == 0: return FP16Layer()
                kq = get_q(hd, 4, device, 42 + li * 2)
                vq = get_q(hd, 2, device, 42 + li * 2 + 1)
                return FullStackLayer(rank_k=hd, rank_v=hd, kq=kq, vq=vq,
                                       num_sinks=4, residual_window=8, min_chunk_for_svd=9999)
            configs.append(("L0fp16+TQ K4/V2 w8 s4", ref_best))

            # ── SVD only (no quantization) ──
            def svd_only_48_24(li):
                if li == 0: return FP16Layer()
                return FullStackLayer(rank_k=48, rank_v=24, kq=None, vq=None,
                                       num_sinks=4, residual_window=8)
            configs.append(("L0fp16+SVD(48/24) w8 s4", svd_only_48_24))

            def svd_only_32_16(li):
                if li == 0: return FP16Layer()
                return FullStackLayer(rank_k=32, rank_v=16, kq=None, vq=None,
                                       num_sinks=4, residual_window=8)
            configs.append(("L0fp16+SVD(32/16) w8 s4", svd_only_32_16))

            # ── FULL STACK: SVD + TurboQuant ──
            def full_48_24_k4v3(li):
                if li == 0: return FP16Layer()
                kq = get_q(48, 4, device, 500 + 48 * 10 + 4)
                vq = get_q(24, 3, device, 500 + 24 * 10 + 3)
                return FullStackLayer(rank_k=48, rank_v=24, kq=kq, vq=vq,
                                       num_sinks=4, residual_window=8)
            configs.append(("FULL: L0fp16+SVD(48/24)+TQ(K4/V3) w8 s4", full_48_24_k4v3))

            def full_48_24_k4v4(li):
                if li == 0: return FP16Layer()
                kq = get_q(48, 4, device, 500 + 48 * 10 + 4)
                vq = get_q(24, 4, device, 500 + 24 * 10 + 4)
                return FullStackLayer(rank_k=48, rank_v=24, kq=kq, vq=vq,
                                       num_sinks=4, residual_window=8)
            configs.append(("FULL: L0fp16+SVD(48/24)+TQ(K4/V4) w8 s4", full_48_24_k4v4))

            def full_32_16_k4v4(li):
                if li == 0: return FP16Layer()
                kq = get_q(32, 4, device, 500 + 32 * 10 + 4)
                vq = get_q(16, 4, device, 500 + 16 * 10 + 4)
                return FullStackLayer(rank_k=32, rank_v=16, kq=kq, vq=vq,
                                       num_sinks=4, residual_window=8)
            configs.append(("FULL: L0fp16+SVD(32/16)+TQ(K4/V4) w8 s4", full_32_16_k4v4))

            def full_32_16_k4v3(li):
                if li == 0: return FP16Layer()
                kq = get_q(32, 4, device, 500 + 32 * 10 + 4)
                vq = get_q(16, 3, device, 500 + 16 * 10 + 3)
                return FullStackLayer(rank_k=32, rank_v=16, kq=kq, vq=vq,
                                       num_sinks=4, residual_window=8)
            configs.append(("FULL: L0fp16+SVD(32/16)+TQ(K4/V3) w8 s4", full_32_16_k4v3))

            # ── FULL STACK with adaptive tiers ──
            def full_adaptive(li):
                if li == 0: return FP16Layer()
                # High-sens layers: SVD 48/24 + K4/V4
                # Low-sens layers: SVD 32/16 + K4/V3
                if li <= max(3, nl // 5):
                    kq = get_q(48, 4, device, 500 + 48 * 10 + 4)
                    vq = get_q(24, 4, device, 500 + 24 * 10 + 4)
                    return FullStackLayer(rank_k=48, rank_v=24, kq=kq, vq=vq,
                                           num_sinks=4, residual_window=8)
                else:
                    kq = get_q(32, 4, device, 500 + 32 * 10 + 4)
                    vq = get_q(16, 3, device, 500 + 16 * 10 + 3)
                    return FullStackLayer(rank_k=32, rank_v=16, kq=kq, vq=vq,
                                           num_sinks=4, residual_window=8)
            configs.append(("FULL+ADAPT: hi=SVD48+K4V4, lo=SVD32+K4V3", full_adaptive))

            # ── Maximum compression: SVD 32/16 + K3/V3 ──
            def full_max_compress(li):
                if li == 0: return FP16Layer()
                kq = get_q(32, 3, device, 500 + 32 * 10 + 3)
                vq = get_q(16, 3, device, 500 + 16 * 10 + 3)
                return FullStackLayer(rank_k=32, rank_v=16, kq=kq, vq=vq,
                                       num_sinks=4, residual_window=8)
            configs.append(("MAX COMPRESS: SVD(32/16)+TQ(K3/V3) w8 s4", full_max_compress))

            # Run all
            print(f"\n  {'Config':60s} {'PPL':>7s} {'ΔPPL':>7s}")
            print(f"  {'-'*77}")

            length_results = []
            for name, layer_fn in configs:
                cache = make_cache(nl, layer_fn)
                ppl = compute_ppl(model, tok, text, cache=cache)
                delta = ((ppl - ppl_base) / ppl_base) * 100
                length_results.append((name, ppl, delta))
                star = " ***" if delta < 0 else (" **" if delta < 2 else (" *" if delta < 5 else ""))
                print(f"  {name:60s} {ppl:>7.2f} {delta:>+6.1f}%{star}")

            model_results[length_name] = (seq, ppl_base, length_results)

        all_results[model_name] = model_results

        del model
        torch.cuda.empty_cache()

    # ── Cross-model, cross-length summary ──
    print(f"\n\n{'='*85}")
    print("CROSS-MODEL CROSS-LENGTH SUMMARY")
    print(f"{'='*85}")

    # Focus on key configs
    key_configs = [
        "Vanilla TQ K4/V2 w32",
        "L0fp16+TQ K4/V2 w8 s4",
        "FULL: L0fp16+SVD(48/24)+TQ(K4/V3) w8 s4",
        "FULL: L0fp16+SVD(32/16)+TQ(K4/V4) w8 s4",
        "FULL+ADAPT: hi=SVD48+K4V4, lo=SVD32+K4V3",
        "MAX COMPRESS: SVD(32/16)+TQ(K3/V3) w8 s4",
    ]

    for model_name, model_results in all_results.items():
        print(f"\n  {model_name}:")
        print(f"  {'Config':60s}", end="")
        for ln in ['short', 'medium', 'long']:
            if ln in model_results:
                seq, _, _ = model_results[ln]
                print(f" {ln}({seq}t)", end="")
        print()
        print(f"  {'-'*85}")

        for cfg_name in key_configs:
            print(f"  {cfg_name:60s}", end="")
            for ln in ['short', 'medium', 'long']:
                if ln in model_results:
                    _, _, results = model_results[ln]
                    match = [r for r in results if r[0] == cfg_name]
                    if match:
                        print(f" {match[0][2]:>+6.1f}%", end="")
                    else:
                        print(f"    N/A", end="")
            print()

    # ── Plot ──
    fig, axes = plt.subplots(1, len(all_results), figsize=(9 * len(all_results), 7))
    if len(all_results) == 1:
        axes = [axes]

    colors = {
        'Vanilla TQ K4/V2 w32': '#aaaaaa',
        'L0fp16+TQ K4/V2 w8 s4': '#888888',
        'L0fp16+SVD(48/24) w8 s4': '#3498db',
        'L0fp16+SVD(32/16) w8 s4': '#2980b9',
        'FULL: L0fp16+SVD(48/24)+TQ(K4/V3) w8 s4': '#e74c3c',
        'FULL: L0fp16+SVD(48/24)+TQ(K4/V4) w8 s4': '#c0392b',
        'FULL: L0fp16+SVD(32/16)+TQ(K4/V4) w8 s4': '#e67e22',
        'FULL: L0fp16+SVD(32/16)+TQ(K4/V3) w8 s4': '#d35400',
        'FULL+ADAPT: hi=SVD48+K4V4, lo=SVD32+K4V3': '#9b59b6',
        'MAX COMPRESS: SVD(32/16)+TQ(K3/V3) w8 s4': '#27ae60',
    }

    for ax, (model_name, model_results) in zip(axes, all_results.items()):
        lengths = ['short', 'medium', 'long']
        x_labels = []
        x_positions = []

        for i, ln in enumerate(lengths):
            if ln not in model_results:
                continue
            seq, ppl_base, results = model_results[ln]
            x_labels.append(f"{ln}\n({seq}t)")

            for j, (name, ppl, delta) in enumerate(results):
                color = colors.get(name, '#999')
                x = i + (j - len(results)/2) * 0.06
                marker = '*' if 'FULL' in name or 'ADAPT' in name else ('D' if 'SVD' in name else 'o')
                sz = 120 if 'FULL' in name else 60
                ax.scatter(x, delta, c=color, s=sz, marker=marker, alpha=0.8, zorder=3)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.axhline(0, color='black', lw=1)
        ax.axhline(2, color='orange', ls='--', alpha=0.3)
        ax.axhline(-2, color='green', ls='--', alpha=0.3)
        ax.set_ylabel('PPL Degradation (%)', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.15, axis='y')

    # Legend
    legend_elements = []
    for name, color in list(colors.items())[:8]:
        short_name = name[:40]
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=color, markersize=8, label=short_name))

    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle('Full Stack Compression: SVD + TurboQuant + Sinks + Adaptive\nAcross Models and Sequence Lengths',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('results/full_stack_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: results/full_stack_results.png")


if __name__ == "__main__":
    main()
