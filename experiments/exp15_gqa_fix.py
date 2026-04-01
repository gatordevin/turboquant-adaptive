"""
Fixing TurboQuant for GQA Models
==================================
Two problems to solve:
1. Extreme key norms (779 at layer 0 on Qwen) -> MSE = norm² × (1-cos_sim)
2. GQA amplification (each KV head error affects N query heads)

Key insight: GQA models have FEWER KV heads, so the KV cache is already
small. We can afford MORE bits per element. A 7:1 GQA model with K8/V8
uses LESS total KV memory than an MHA model with K4/V2.

Also: int4 model weights don't affect KV cache — attention still
computes in fp16. Our compression works identically on int4 models.

Fixes to try:
A. HIGHER BITS: K6, K8 — affordable because GQA = fewer KV heads
B. SCALE NORMALIZATION: Divide by per-layer scale before rotation
C. CHANNEL CLIPPING: Clip outlier channels before quantizing
D. MIXED: fp16 for high-norm layers, quantize the rest
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, CacheLayerMixin

from turboquant_gpt2 import TurboQuantizer, compute_lloyd_max_codebook
from experiments.exp12_full_stack import FP16Layer, FullStackLayer

_qcache = {}
def get_q(hd, bits, device, seed):
    k = (hd, bits, seed)
    if k not in _qcache:
        _qcache[k] = TurboQuantizer(hd, bits, device=device, seed=seed)
    return _qcache[k]


# ============================================================================
# Fix B: Scale-Normalized Quantizer
# ============================================================================

class ScaleNormalizedQuantizer:
    """TurboQuant with per-vector scale normalization.

    Standard TurboQuant saves the L2 norm and quantizes the unit vector.
    This works when norms are ~1 but fails when norms are 100-1000
    because denormalized error = norm² × quantization_error.

    Fix: Instead of normalizing to unit vectors, normalize to a
    FIXED scale (e.g., std=1 per dimension). This bounds the
    absolute error regardless of the original norm.

    Specifically:
      1. Compute mean and std per-vector
      2. Standardize: x_std = (x - mean) / std
      3. Quantize x_std (which has unit variance, matching the codebook)
      4. Store mean (fp16 scalar) and std (fp16 scalar) alongside indices
      5. Reconstruct: x_hat = x_std_hat * std + mean
    """

    def __init__(self, head_dim, bits, device='cuda', seed=42):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        # Codebook designed for N(0, 1/d) distribution (standard TurboQuant)
        centroids_np, boundaries_np = compute_lloyd_max_codebook(head_dim, bits)
        self.centroids = torch.from_numpy(centroids_np).half().to(device)
        self.boundaries = torch.from_numpy(boundaries_np).half().to(device)

        # Rotation matrix
        rng = torch.Generator().manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=rng)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        self.Pi = (Q * diag_sign.unsqueeze(0)).half().to(device)

        # Scale the codebook to match standardized data
        # After standardization, each coord of rotated unit vector ~ N(0, 1/d)
        # But standardized data has std=1, not std=1/sqrt(d)
        # So we need codebook scaled by sqrt(d)
        scale = np.sqrt(head_dim)
        self.centroids_scaled = self.centroids * scale
        self.boundaries_scaled = self.boundaries * scale

    def quantize(self, x):
        """Quantize with mean/std normalization."""
        x = x.half()
        # Per-vector statistics
        means = x.mean(dim=-1, keepdim=True)   # (..., 1)
        stds = x.std(dim=-1, keepdim=True).clamp(min=1e-6)  # (..., 1)

        # Standardize
        x_std = (x - means) / stds  # now ~N(0,1) per dimension

        # Normalize to unit vector for rotation
        norms = x_std.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x_std / norms

        # Rotate
        y = x_unit @ self.Pi.T

        # Quantize
        indices = torch.searchsorted(self.boundaries, y.contiguous())

        return indices, norms.squeeze(-1), means.squeeze(-1), stds.squeeze(-1)

    def dequantize(self, indices, norms, means, stds):
        """Dequantize and de-standardize."""
        y_hat = self.centroids[indices]
        x_unit_hat = y_hat @ self.Pi
        x_std_hat = x_unit_hat * norms.unsqueeze(-1)
        x_hat = x_std_hat * stds.unsqueeze(-1) + means.unsqueeze(-1)
        return x_hat


class ScaleNormLayer(CacheLayerMixin):
    """Cache layer using scale-normalized quantizer."""
    is_sliding = False

    def __init__(self, kq, vq, num_sinks=4, residual_window=16):
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

    def _compress(self, X, quantizer):
        """X: (BH, T, D)"""
        BH, T, D = X.shape
        flat = X.reshape(BH * T, D)
        idx, norms, means, stds = quantizer.quantize(flat)
        return (idx, norms, means, stds, BH, T, D)

    def _decompress(self, data, quantizer):
        idx, norms, means, stds, BH, T, D = data
        flat = quantizer.dequantize(idx, norms, means, stds)
        return flat.reshape(BH, T, D)

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

        if self.recent_keys.shape[-2] > self.residual_window:
            overflow = self.recent_keys.shape[-2] - self.residual_window
            k_o = self.recent_keys[:, :, :overflow, :].reshape(B*H, overflow, D)
            v_o = self.recent_values[:, :, :overflow, :].reshape(B*H, overflow, D)
            self.recent_keys = self.recent_keys[:, :, overflow:, :].contiguous()
            self.recent_values = self.recent_values[:, :, overflow:, :].contiguous()

            self.comp_k.append(self._compress(k_o, self.kq))
            self.comp_v.append(self._compress(v_o, self.vq))
            self.comp_shapes.append((B, H, overflow, D))

        # Reconstruct
        pk, pv = [], []
        if self.sink_keys is not None:
            pk.append(self.sink_keys); pv.append(self.sink_values)

        for i, (B2, H2, T2, D2) in enumerate(self.comp_shapes):
            k_hat = self._decompress(self.comp_k[i], self.kq).reshape(B2, H2, T2, D2)
            v_hat = self._decompress(self.comp_v[i], self.vq).reshape(B2, H2, T2, D2)
            pk.append(k_hat); pv.append(v_hat)

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
# Helpers
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

_snq_cache = {}
def get_snq(hd, bits, device, seed):
    k = (hd, bits, seed, 'snq')
    if k not in _snq_cache:
        _snq_cache[k] = ScaleNormalizedQuantizer(hd, bits, device=device, seed=seed)
    return _snq_cache[k]


# ============================================================================
# Main
# ============================================================================

def main():
    device = 'cuda'
    print("=" * 80)
    print("FIXING TURBOQUANT FOR GQA MODELS")
    print("=" * 80)

    # ── Test on Qwen2.5-0.5B first (small, fast, GQA) ──
    model_name = 'Qwen/Qwen2.5-0.5B'
    print(f"\nLoading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).half().eval()

    config = model.config
    hd = config.hidden_size // config.num_attention_heads
    nl = config.num_hidden_layers
    nh = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    gqa_ratio = nh // kv_heads

    print(f"  layers={nl}, hd={hd}, heads={nh}, kv_heads={kv_heads}, GQA={gqa_ratio}:1")
    print(f"  KV cache is {gqa_ratio}x smaller than MHA equivalent")

    eval_text = (
        "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
        "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of "
        "modern AI were planted by philosophers who attempted to describe the process of human thinking "
        "as the mechanical manipulation of symbols."
    )
    seq = len(tok.encode(eval_text))
    print(f"  Eval: {seq} tokens")

    # Precompute codebooks
    print("\n  Precomputing codebooks...")
    t0 = time.time()
    for bits in [2, 3, 4]:
        for li in range(nl):
            get_q(hd, bits, device, 42 + li * 2)
            get_q(hd, bits, device, 42 + li * 2 + 1)
            get_snq(hd, bits, device, 42 + li * 2)
            get_snq(hd, bits, device, 42 + li * 2 + 1)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Analyze key norms per layer ──
    print("\n  Key/Value norm analysis:")
    with torch.no_grad():
        out = model(tok.encode(eval_text, return_tensors='pt').to(device), use_cache=True)
    for li in [0, 1, 2, nl//2, nl-1]:
        layer = out.past_key_values.layers[li]
        k_norm = layer.keys.float().norm(dim=-1).mean().item()
        v_norm = layer.values.float().norm(dim=-1).mean().item()
        print(f"    Layer {li:2d}: K_norm={k_norm:>8.1f}  V_norm={v_norm:>8.1f}")

    # ── Baseline ──
    ppl_base = compute_ppl(model, tok, eval_text)
    print(f"\n  Baseline fp16: PPL = {ppl_base:.2f}")

    # ── Test strategies ──
    print(f"\n  {'Strategy':55s} {'PPL':>7s} {'ΔPPL':>7s}")
    print(f"  {'-'*72}")

    results = []
    def test(name, cache, cat=''):
        ppl = compute_ppl(model, tok, eval_text, cache=cache)
        delta = ((ppl - ppl_base) / ppl_base) * 100
        indicator = " ***" if delta < 0 else (" **" if delta < 3 else (" *" if delta < 10 else ("" if delta < 50 else " X")))
        print(f"  {name:55s} {ppl:>7.2f} {delta:>+6.1f}%{indicator}")
        results.append((name, ppl, delta, cat))
        return ppl, delta

    # ── A: Vanilla TurboQuant at various bits ──
    print("\n  --- A: Vanilla TurboQuant (original, norm-based) ---")
    for kb, vb in [(4, 2), (4, 4)]:
        for win in [32, 64]:
            test(f"Vanilla K{kb}/V{vb} w{win}",
                 make_cache(nl, lambda li, kb=kb, vb=vb, win=win:
                     FullStackLayer(rank_k=hd, rank_v=hd,
                                     kq=get_q(hd, kb, device, 42+li*2),
                                     vq=get_q(hd, vb, device, 42+li*2+1),
                                     num_sinks=0, residual_window=win, min_chunk_for_svd=9999)),
                 'vanilla')

    # ── B: Scale-Normalized TurboQuant ──
    print("\n  --- B: Scale-Normalized TurboQuant (mean/std normalization) ---")
    for kb, vb in [(4, 2), (4, 4), (3, 3), (4, 3)]:
        for win in [16, 32]:
            test(f"ScaleNorm K{kb}/V{vb} w{win} s4",
                 make_cache(nl, lambda li, kb=kb, vb=vb, win=win:
                     ScaleNormLayer(get_snq(hd, kb, device, 42+li*2),
                                    get_snq(hd, vb, device, 42+li*2+1),
                                    num_sinks=4, residual_window=win)),
                 'scale_norm')

    # ── C: L0fp16 + Scale-Normalized ──
    print("\n  --- C: L0fp16 + Scale-Normalized ---")
    for kb, vb in [(4, 2), (4, 4), (4, 3), (3, 3)]:
        test(f"L0fp16 + ScaleNorm K{kb}/V{vb} w16 s4",
             make_cache(nl, lambda li, kb=kb, vb=vb:
                 FP16Layer() if li == 0 else
                 ScaleNormLayer(get_snq(hd, kb, device, 42+li*2),
                                get_snq(hd, vb, device, 42+li*2+1),
                                num_sinks=4, residual_window=16)),
             'l0_scale')

    # ── D: L0-2 fp16 + Scale-Normalized (protect high-norm layers) ──
    print("\n  --- D: Multi-layer fp16 + Scale-Normalized ---")
    for n_fp16 in [1, 2, 3, 4]:
        for kb, vb in [(4, 3), (4, 4)]:
            test(f"L0-{n_fp16-1}fp16 + ScaleNorm K{kb}/V{vb} w16 s4",
                 make_cache(nl, lambda li, n=n_fp16, kb=kb, vb=vb:
                     FP16Layer() if li < n else
                     ScaleNormLayer(get_snq(hd, kb, device, 42+li*2),
                                    get_snq(hd, vb, device, 42+li*2+1),
                                    num_sinks=4, residual_window=16)),
                 'multi_fp16')

    # ── E: Adaptive — fp16 for high-norm layers, scale-norm for rest ──
    print("\n  --- E: Norm-Adaptive (fp16 where norm > threshold) ---")

    # Measure norms
    layer_k_norms = []
    with torch.no_grad():
        out = model(tok.encode(eval_text, return_tensors='pt').to(device), use_cache=True)
        for layer in out.past_key_values.layers:
            layer_k_norms.append(layer.keys.float().norm(dim=-1).mean().item())

    # Threshold: fp16 for layers with norm > median
    norm_median = np.median(layer_k_norms)
    norm_p75 = np.percentile(layer_k_norms, 75)

    for threshold, label in [(norm_p75, 'p75'), (norm_median, 'median'), (50, 'norm>50')]:
        fp16_count = sum(1 for n in layer_k_norms if n > threshold)
        test(f"NormAdapt({label}, {fp16_count}xfp16) + ScaleNorm K4/V3 w16 s4",
             make_cache(nl, lambda li, thr=threshold:
                 FP16Layer() if layer_k_norms[li] > thr else
                 ScaleNormLayer(get_snq(hd, 4, device, 42+li*2),
                                get_snq(hd, 3, device, 42+li*2+1),
                                num_sinks=4, residual_window=16)),
             'norm_adapt')

    # ── F: Compare memory usage ──
    print(f"\n  --- Memory comparison (GQA = {gqa_ratio}:1) ---")
    fp16_kv_per_token = kv_heads * hd * 2 * 2  # K+V fp16 bytes
    mha_equiv_per_token = nh * hd * 2 * 2  # what MHA would cost

    print(f"    MHA equivalent KV per token: {mha_equiv_per_token} bytes")
    print(f"    GQA actual KV per token:     {fp16_kv_per_token} bytes ({gqa_ratio}x smaller)")
    print(f"    GQA + K4/V3 quantized:       ~{int(kv_heads * hd * (4+3) / 8 * 2 + kv_heads * 4)} bytes/token")
    print(f"    Effective compression vs MHA fp16: "
          f"{mha_equiv_per_token / (kv_heads * hd * (4+3) / 8 * 2 + kv_heads * 4):.1f}x")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("SUMMARY: Best GQA-Compatible Strategies")
    print(f"{'='*80}")
    print(f"\n  Baseline PPL: {ppl_base:.2f}")

    for cat in ['vanilla', 'scale_norm', 'l0_scale', 'multi_fp16', 'norm_adapt']:
        cat_results = [r for r in results if r[3] == cat]
        if cat_results:
            best = min(cat_results, key=lambda r: abs(r[2]))
            status = "WORKS" if abs(best[2]) < 10 else "BROKEN"
            print(f"\n  {cat:15s}: best = {best[0]:45s} = {best[2]:+.1f}% [{status}]")

    del model
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════
    # Now test on the actual target: Qwen2.5-Coder-7B
    # ══════════════════════════════════════════════════════════════

    print(f"\n\n{'='*80}")
    print("TESTING ON TARGET: Qwen2.5-Coder-7B")
    print(f"{'='*80}")

    model_name = 'Qwen/Qwen2.5-Coder-7B'
    try:
        tok2 = AutoTokenizer.from_pretrained(model_name)
        model2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()

        config2 = model2.config
        hd2 = config2.hidden_size // config2.num_attention_heads
        nl2 = config2.num_hidden_layers
        nh2 = config2.num_attention_heads
        kv2 = config2.num_key_value_heads
        gqa2 = nh2 // kv2

        print(f"  layers={nl2}, hd={hd2}, heads={nh2}, kv_heads={kv2}, GQA={gqa2}:1")
        print(f"  Model GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

        code_text = '''def fibonacci(n):
    """Return the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def merge_sort(arr):
    """Sort an array using merge sort algorithm."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result'''

        seq2 = len(tok2.encode(code_text))
        print(f"  Eval: {seq2} tokens (Python code)")

        # Precompute codebooks for hd2
        print("  Precomputing codebooks...")
        for bits in [3, 4]:
            for li in range(nl2):
                get_q(hd2, bits, device, 42+li*2)
                get_q(hd2, bits, device, 42+li*2+1)
                get_snq(hd2, bits, device, 42+li*2)
                get_snq(hd2, bits, device, 42+li*2+1)

        ppl_base2 = compute_ppl(model2, tok2, code_text)
        print(f"\n  Baseline fp16: PPL = {ppl_base2:.2f}")

        # Analyze norms
        print("\n  Key norms per layer:")
        with torch.no_grad():
            out2 = model2(tok2.encode(code_text, return_tensors='pt').to(device), use_cache=True)
            norms2 = []
            for li, layer in enumerate(out2.past_key_values.layers):
                n = layer.keys.float().norm(dim=-1).mean().item()
                norms2.append(n)
                if li < 4 or li >= nl2-2:
                    print(f"    Layer {li:2d}: K_norm={n:.1f}")
            print(f"    ... max={max(norms2):.1f} (layer {np.argmax(norms2)}), "
                  f"median={np.median(norms2):.1f}")

        print(f"\n  {'Strategy':55s} {'PPL':>7s} {'ΔPPL':>7s}")
        print(f"  {'-'*72}")

        # Test best strategies from 0.5B
        for name, fn in [
            ("Vanilla K4/V4 w32", lambda li:
                FullStackLayer(hd2, hd2, get_q(hd2,4,device,42+li*2), get_q(hd2,4,device,42+li*2+1),
                               0, 32, 9999)),
            ("L0fp16 + ScaleNorm K4/V4 w16 s4", lambda li:
                FP16Layer() if li == 0 else
                ScaleNormLayer(get_snq(hd2,4,device,42+li*2), get_snq(hd2,4,device,42+li*2+1), 4, 16)),
            ("L0fp16 + ScaleNorm K4/V3 w16 s4", lambda li:
                FP16Layer() if li == 0 else
                ScaleNormLayer(get_snq(hd2,4,device,42+li*2), get_snq(hd2,3,device,42+li*2+1), 4, 16)),
            ("L0-2fp16 + ScaleNorm K4/V3 w16 s4", lambda li:
                FP16Layer() if li < 3 else
                ScaleNormLayer(get_snq(hd2,4,device,42+li*2), get_snq(hd2,3,device,42+li*2+1), 4, 16)),
            ("L0-3fp16 + ScaleNorm K4/V4 w16 s4", lambda li:
                FP16Layer() if li < 4 else
                ScaleNormLayer(get_snq(hd2,4,device,42+li*2), get_snq(hd2,4,device,42+li*2+1), 4, 16)),
        ]:
            cache = make_cache(nl2, fn)
            ppl = compute_ppl(model2, tok2, code_text, cache=cache)
            delta = ((ppl - ppl_base2) / ppl_base2) * 100
            ind = " ***" if delta < 0 else (" **" if delta < 3 else (" *" if delta < 10 else ""))
            print(f"  {name:55s} {ppl:>7.2f} {delta:>+6.1f}%{ind}")

        del model2
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}")
    print("Done!")


if __name__ == "__main__":
    main()
