"""
New Directions: Exploiting Temporal Redundancy in KV Cache
============================================================
Everything so far quantizes each token independently. But KV vectors
across adjacent tokens are highly correlated — the residual stream
changes smoothly. We can exploit this with:

1. DELTA CODING: Store first token at full precision, then quantize
   the difference (delta) between consecutive tokens. If the KV
   stream is smooth, deltas are small and quantize better.

2. CROSS-TOKEN LOW-RANK: Group N tokens, compute low-rank
   approximation (SVD), store only top-k components. Exploits the
   fact that KV vectors across a context window often live in a
   low-dimensional subspace.

3. MEAN-RESIDUAL: Compute per-group mean, store it at high precision,
   quantize only the residual (deviation from mean). If tokens within
   a group are similar, residuals are small.

These are ORTHOGONAL to TurboQuant and can stack on top.
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
# Analysis: How correlated ARE adjacent KV vectors?
# ============================================================================

def analyze_kv_correlation(model, tokenizer, text, device='cuda'):
    """Measure temporal correlation in KV cache."""
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    print(f"\n  KV Temporal Correlation Analysis ({input_ids.shape[1]} tokens)")
    print(f"  {'Layer':>7s} {'K cos(t,t+1)':>14s} {'V cos(t,t+1)':>14s} "
          f"{'K delta/norm':>13s} {'V delta/norm':>13s} {'K rank90%':>10s} {'V rank90%':>10s}")
    print(f"  {'-'*85}")

    all_k_cos, all_v_cos = [], []
    all_k_delta, all_v_delta = [], []
    all_k_rank, all_v_rank = [], []

    for li, layer in enumerate(outputs.past_key_values.layers):
        K = layer.keys.float()     # (1, H, T, D)
        V = layer.values.float()

        T = K.shape[2]
        if T < 2:
            continue

        # Adjacent cosine similarity
        k_cos = F.cosine_similarity(K[:, :, :-1, :], K[:, :, 1:, :], dim=-1).mean().item()
        v_cos = F.cosine_similarity(V[:, :, :-1, :], V[:, :, 1:, :], dim=-1).mean().item()

        # Delta magnitude relative to vector norm
        k_delta = (K[:, :, 1:, :] - K[:, :, :-1, :]).norm(dim=-1).mean().item()
        k_norm = K.norm(dim=-1).mean().item()
        v_delta = (V[:, :, 1:, :] - V[:, :, :-1, :]).norm(dim=-1).mean().item()
        v_norm = V.norm(dim=-1).mean().item()

        k_delta_ratio = k_delta / (k_norm + 1e-8)
        v_delta_ratio = v_delta / (v_norm + 1e-8)

        # Effective rank: how many singular values capture 90% of variance
        # Reshape to (H, T, D) and compute SVD per head
        k_reshaped = K[0]  # (H, T, D)
        v_reshaped = V[0]
        k_ranks, v_ranks = [], []
        for h in range(k_reshaped.shape[0]):
            _, sk, _ = torch.svd(k_reshaped[h])
            cumvar = (sk ** 2).cumsum(0) / (sk ** 2).sum()
            k_ranks.append((cumvar < 0.9).sum().item() + 1)

            _, sv, _ = torch.svd(v_reshaped[h])
            cumvar = (sv ** 2).cumsum(0) / (sv ** 2).sum()
            v_ranks.append((cumvar < 0.9).sum().item() + 1)

        k_rank = np.mean(k_ranks)
        v_rank = np.mean(v_ranks)

        all_k_cos.append(k_cos)
        all_v_cos.append(v_cos)
        all_k_delta.append(k_delta_ratio)
        all_v_delta.append(v_delta_ratio)
        all_k_rank.append(k_rank)
        all_v_rank.append(v_rank)

        if li < 5 or li >= K.shape[2] - 3 or li % 8 == 0:
            print(f"  {li:>5d}   {k_cos:>12.4f}   {v_cos:>12.4f}   "
                  f"{k_delta_ratio:>11.4f}   {v_delta_ratio:>11.4f}   "
                  f"{k_rank:>8.1f}   {v_rank:>8.1f}")

    print(f"  {'avg':>7s} {np.mean(all_k_cos):>14.4f} {np.mean(all_v_cos):>14.4f} "
          f"{np.mean(all_k_delta):>13.4f} {np.mean(all_v_delta):>13.4f} "
          f"{np.mean(all_k_rank):>10.1f} {np.mean(all_v_rank):>10.1f}")

    return {
        'k_cos': all_k_cos, 'v_cos': all_v_cos,
        'k_delta': all_k_delta, 'v_delta': all_v_delta,
        'k_rank': all_k_rank, 'v_rank': all_v_rank,
    }


# ============================================================================
# Strategy 1: Delta Coding Cache Layer
# ============================================================================

class DeltaCodingLayer(CacheLayerMixin):
    """Stores KV as: anchor (fp16) + quantized deltas.

    Instead of quantizing K[t] directly, stores:
      anchor = K[0] at fp16
      delta[t] = K[t] - K[t-1]  quantized via TurboQuant

    On read: cumulative sum to reconstruct K[0], K[0]+d[1], K[0]+d[1]+d[2], ...

    If adjacent tokens are highly correlated (cos ~0.9+), deltas are
    small and quantize with less absolute error.
    """

    is_sliding = False

    def __init__(self, kq, vq, num_sinks=0, residual_window=16):
        super().__init__()
        self.kq, self.vq = kq, vq
        self.num_sinks = num_sinks
        self.residual_window = residual_window

        self.sink_keys = self.sink_values = None
        self.anchors_k = []  # fp16 anchor per chunk
        self.anchors_v = []
        self.delta_k = []    # quantized deltas
        self.delta_v = []
        self.delta_shapes = []
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

        # Compress overflow as delta-coded
        if self.recent_keys.shape[-2] > self.residual_window:
            overflow = self.recent_keys.shape[-2] - self.residual_window
            k_o = self.recent_keys[:, :, :overflow, :]
            v_o = self.recent_values[:, :, :overflow, :]
            self.recent_keys = self.recent_keys[:, :, overflow:, :].contiguous()
            self.recent_values = self.recent_values[:, :, overflow:, :].contiguous()

            B2, H2, T2, D2 = k_o.shape

            # Anchor = first token of chunk (fp16)
            self.anchors_k.append(k_o[:, :, 0:1, :].clone())
            self.anchors_v.append(v_o[:, :, 0:1, :].clone())

            if T2 > 1:
                # Deltas = differences between consecutive tokens
                k_deltas = k_o[:, :, 1:, :] - k_o[:, :, :-1, :]  # (B, H, T2-1, D)
                v_deltas = v_o[:, :, 1:, :] - v_o[:, :, :-1, :]

                # Quantize deltas
                kd_flat = k_deltas.reshape(B2 * H2 * (T2 - 1), D2)
                vd_flat = v_deltas.reshape(B2 * H2 * (T2 - 1), D2)

                ki, kn = self.kq.quantize(kd_flat)
                vi, vn = self.vq.quantize(vd_flat)

                self.delta_k.append((ki, kn))
                self.delta_v.append((vi, vn))
            else:
                self.delta_k.append(None)
                self.delta_v.append(None)

            self.delta_shapes.append((B2, H2, T2))

        # Reconstruct
        pk, pv = [], []
        if self.sink_keys is not None:
            pk.append(self.sink_keys)
            pv.append(self.sink_values)

        for i, (B2, H2, T2) in enumerate(self.delta_shapes):
            # Start with anchor
            k_recon = [self.anchors_k[i]]  # (B, H, 1, D)
            v_recon = [self.anchors_v[i]]

            if T2 > 1 and self.delta_k[i] is not None:
                ki, kn = self.delta_k[i]
                vi, vn = self.delta_v[i]

                # Dequantize deltas
                kd_hat = self.kq.dequantize(ki, kn).reshape(B2, H2, T2 - 1, -1)
                vd_hat = self.vq.dequantize(vi, vn).reshape(B2, H2, T2 - 1, -1)

                # Cumulative sum to reconstruct
                current_k = self.anchors_k[i]
                current_v = self.anchors_v[i]
                for t in range(T2 - 1):
                    current_k = current_k + kd_hat[:, :, t:t+1, :]
                    current_v = current_v + vd_hat[:, :, t:t+1, :]
                    k_recon.append(current_k)
                    v_recon.append(current_v)

            pk.append(torch.cat(k_recon, dim=-2))
            pv.append(torch.cat(v_recon, dim=-2))

        pk.append(self.recent_keys)
        pv.append(self.recent_values)
        self.keys = torch.cat(pk, dim=-2)
        self.values = torch.cat(pv, dim=-2)
        return self.keys, self.values

    def get_seq_length(self):
        s = self.sink_keys.shape[-2] if self.sink_keys is not None else 0
        c = sum(sh[2] for sh in self.delta_shapes)
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
# Strategy 2: Low-Rank Cross-Token Compression
# ============================================================================

class LowRankLayer(CacheLayerMixin):
    """Compress groups of N tokens using low-rank approximation.

    For a group of N tokens with KV in R^{N x D}:
      U, S, V = SVD(K_group)
      Keep top-k singular values: K_approx = U[:,:k] @ diag(S[:k]) @ V[:k,:]

    Compression: store k*(N+D+1) instead of N*D values.
    For k << min(N,D), this is significant.
    """

    is_sliding = False

    def __init__(self, rank_k=16, rank_v=8, num_sinks=0, residual_window=16):
        super().__init__()
        self.rank_k = rank_k
        self.rank_v = rank_v
        self.num_sinks = num_sinks
        self.residual_window = residual_window

        self.sink_keys = self.sink_values = None
        self.compressed_k = []  # list of (U, S, Vt) tuples
        self.compressed_v = []
        self.comp_shapes = []
        self.recent_keys = self.recent_values = None
        self.total_seen = 0

    def lazy_initialization(self, k, v):
        self.dtype, self.device = k.dtype, k.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def _compress_svd(self, X, rank):
        """Low-rank approximation of X (B*H, T, D) -> (U, S, Vt)."""
        # Per-head SVD
        B_H, T, D = X.shape
        if T <= rank or rank >= D:
            return X, None, None, False  # don't compress

        X_float = X.float()
        U, S, Vt = torch.linalg.svd(X_float, full_matrices=False)
        # Keep top rank components
        U_k = U[:, :, :rank].half()      # (B*H, T, rank)
        S_k = S[:, :rank].half()           # (B*H, rank)
        Vt_k = Vt[:, :rank, :].half()     # (B*H, rank, D)
        return U_k, S_k, Vt_k, True

    def _decompress_svd(self, U, S, Vt, compressed):
        if not compressed:
            return U  # U is actually the original tensor
        # Reconstruct: U @ diag(S) @ Vt
        return torch.bmm(U * S.unsqueeze(1), Vt)  # (B*H, T, D)

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
            k_o = self.recent_keys[:, :, :overflow, :]
            v_o = self.recent_values[:, :, :overflow, :]
            self.recent_keys = self.recent_keys[:, :, overflow:, :].contiguous()
            self.recent_values = self.recent_values[:, :, overflow:, :].contiguous()

            B2, H2, T2, D2 = k_o.shape
            k_flat = k_o.reshape(B2 * H2, T2, D2)
            v_flat = v_o.reshape(B2 * H2, T2, D2)

            k_comp = self._compress_svd(k_flat, self.rank_k)
            v_comp = self._compress_svd(v_flat, self.rank_v)

            self.compressed_k.append(k_comp)
            self.compressed_v.append(v_comp)
            self.comp_shapes.append((B2, H2, T2, D2))

        # Reconstruct
        pk, pv = [], []
        if self.sink_keys is not None:
            pk.append(self.sink_keys)
            pv.append(self.sink_values)

        for i, (B2, H2, T2, D2) in enumerate(self.comp_shapes):
            U, S, Vt, comp = self.compressed_k[i]
            k_hat = self._decompress_svd(U, S, Vt, comp).reshape(B2, H2, T2, D2)
            U, S, Vt, comp = self.compressed_v[i]
            v_hat = self._decompress_svd(U, S, Vt, comp).reshape(B2, H2, T2, D2)
            pk.append(k_hat)
            pv.append(v_hat)

        pk.append(self.recent_keys)
        pv.append(self.recent_values)
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
# Strategy 3: Low-Rank + TurboQuant hybrid
# ============================================================================

class LowRankQuantLayer(CacheLayerMixin):
    """SVD to reduce rank, then TurboQuant the reduced representation.

    For K in R^{T x D}:
      1. SVD: K ≈ U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
      2. Store Vt (basis) in fp16 (small: k x D)
      3. Quantize coefficients C = U @ diag(S) via TurboQuant (T x k, smaller than T x D)
      4. Reconstruct: C_hat @ Vt

    Double compression: rank reduction + quantization.
    """

    is_sliding = False

    def __init__(self, rank_k=32, rank_v=16, kq=None, vq=None,
                 num_sinks=0, residual_window=16):
        super().__init__()
        self.rank_k = rank_k
        self.rank_v = rank_v
        self.kq = kq  # TurboQuantizer for the rank_k-dim coefficients
        self.vq = vq  # TurboQuantizer for the rank_v-dim coefficients
        self.num_sinks = num_sinks
        self.residual_window = residual_window

        self.sink_keys = self.sink_values = None
        self.comp_k = []  # (coeff_indices, coeff_norms, Vt_basis)
        self.comp_v = []
        self.comp_shapes = []
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

        if self.recent_keys.shape[-2] > self.residual_window:
            overflow = self.recent_keys.shape[-2] - self.residual_window
            k_o = self.recent_keys[:, :, :overflow, :]
            v_o = self.recent_values[:, :, :overflow, :]
            self.recent_keys = self.recent_keys[:, :, overflow:, :].contiguous()
            self.recent_values = self.recent_values[:, :, overflow:, :].contiguous()

            B2, H2, T2, D2 = k_o.shape

            # Process keys
            if T2 > self.rank_k and self.kq is not None:
                k_flat = k_o.reshape(B2 * H2, T2, D2).float()
                U, S, Vt = torch.linalg.svd(k_flat, full_matrices=False)
                coeffs = U[:, :, :self.rank_k] * S[:, :self.rank_k].unsqueeze(1)  # (BH, T, rank_k)
                basis = Vt[:, :self.rank_k, :].half()  # (BH, rank_k, D)

                # Quantize coefficients
                c_flat = coeffs.reshape(B2 * H2 * T2, self.rank_k).half()
                ci, cn = self.kq.quantize(c_flat)
                self.comp_k.append((ci, cn, basis, B2, H2, T2))
            else:
                # Fall back to storing raw
                self.comp_k.append(('raw', k_o.clone(), None, B2, H2, T2))

            # Process values
            if T2 > self.rank_v and self.vq is not None:
                v_flat = v_o.reshape(B2 * H2, T2, D2).float()
                U, S, Vt = torch.linalg.svd(v_flat, full_matrices=False)
                coeffs = U[:, :, :self.rank_v] * S[:, :self.rank_v].unsqueeze(1)
                basis = Vt[:, :self.rank_v, :].half()

                c_flat = coeffs.reshape(B2 * H2 * T2, self.rank_v).half()
                ci, cn = self.vq.quantize(c_flat)
                self.comp_v.append((ci, cn, basis, B2, H2, T2))
            else:
                self.comp_v.append(('raw', v_o.clone(), None, B2, H2, T2))

            self.comp_shapes.append((B2, H2, T2, D2))

        # Reconstruct
        pk, pv = [], []
        if self.sink_keys is not None:
            pk.append(self.sink_keys); pv.append(self.sink_values)

        for i, (B2, H2, T2, D2) in enumerate(self.comp_shapes):
            # Keys
            kc = self.comp_k[i]
            if kc[0] == 'raw':
                pk.append(kc[1])
            else:
                ci, cn, basis, _, _, _ = kc
                c_hat = self.kq.dequantize(ci, cn).reshape(B2 * H2, T2, self.rank_k)
                k_hat = torch.bmm(c_hat, basis).reshape(B2, H2, T2, D2)
                pk.append(k_hat)

            # Values
            vc = self.comp_v[i]
            if vc[0] == 'raw':
                pv.append(vc[1])
            else:
                ci, cn, basis, _, _, _ = vc
                c_hat = self.vq.dequantize(ci, cn).reshape(B2 * H2, T2, self.rank_v)
                v_hat = torch.bmm(c_hat, basis).reshape(B2, H2, T2, D2)
                pv.append(v_hat)

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
# Evaluation
# ============================================================================

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


# ============================================================================
# Main
# ============================================================================

def main():
    device = 'cuda'
    print("=" * 80)
    print("NEW DIRECTIONS: Temporal Redundancy in KV Cache")
    print("=" * 80)

    tok = GPT2Tokenizer.from_pretrained('gpt2-xl')
    model = GPT2LMHeadModel.from_pretrained('gpt2-xl', attn_implementation='eager').to(device).half()
    model.eval()

    hd, nl, nh = 64, 48, 25

    text = (
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
        "difficulty of the project. In 1974 the U.S. and British governments cut off research in AI. "
        "The next few years would later be called an AI winter."
    )
    seq = len(tok.encode(text))
    print(f"  GPT-2 XL, {seq} tokens\n")

    # Precompute codebooks for various dimensions
    print("  Precomputing codebooks...")
    for bits in [2, 3, 4]:
        for li in range(nl):
            get_q(hd, bits, device, 42 + li * 2)
            get_q(hd, bits, device, 42 + li * 2 + 1)
        for rank in [8, 16, 32, 48]:
            get_q(rank, bits, device, 500 + rank)
            get_q(rank, bits, device, 600 + rank)

    # ── Analysis: How correlated are KV vectors? ──
    corr = analyze_kv_correlation(model, tok, text, device)

    # ── Baseline ──
    ppl_base = compute_ppl(model, tok, text)
    print(f"\n  Baseline fp16: PPL = {ppl_base:.2f}")

    # ── Reference configs ──
    print(f"\n  {'Config':55s} {'PPL':>7s} {'ΔPPL':>7s}")
    print(f"  {'-'*72}")

    def test(name, cache):
        ppl = compute_ppl(model, tok, text, cache=cache)
        delta = ((ppl - ppl_base) / ppl_base) * 100
        print(f"  {name:55s} {ppl:>7.2f} {delta:>+6.1f}%")
        return ppl, delta

    # References
    from experiments.exp10_push_limits import FlexLayer, FP16Layer
    test("Ref: Vanilla K4/V2 w32",
         make_cache(nl, lambda li: FlexLayer(get_q(hd,4,device,42+li*2), get_q(hd,2,device,42+li*2+1),
                                              num_sinks=0, residual_window=32)))

    test("Ref: L0fp16+K4V2w8+s4 (our best)",
         make_cache(nl, lambda li: FP16Layer() if li==0 else
                    FlexLayer(get_q(hd,4,device,42+li*2), get_q(hd,2,device,42+li*2+1),
                              num_sinks=4, residual_window=8)))

    # ── Delta coding ──
    print(f"\n  --- Delta Coding ---")
    for kb, vb in [(4, 2), (4, 3), (3, 2)]:
        for win in [8, 16, 32]:
            name = f"Delta K{kb}/V{vb} w{win} s4"
            cache = make_cache(nl, lambda li, kb=kb, vb=vb, win=win:
                FP16Layer() if li == 0 else
                DeltaCodingLayer(get_q(hd, kb, device, 42+li*2),
                                  get_q(hd, vb, device, 42+li*2+1),
                                  num_sinks=4, residual_window=win))
            test(name, cache)

    # ── Low-rank SVD ──
    print(f"\n  --- Low-Rank SVD ---")
    for rk, rv in [(32, 16), (24, 12), (16, 8), (48, 24)]:
        for win in [8, 16]:
            name = f"SVD K-rank={rk} V-rank={rv} w{win} s4"
            cache = make_cache(nl, lambda li, rk=rk, rv=rv, win=win:
                FP16Layer() if li == 0 else
                LowRankLayer(rank_k=rk, rank_v=rv, num_sinks=4, residual_window=win))
            test(name, cache)

    # ── SVD + TurboQuant hybrid ──
    print(f"\n  --- SVD + TurboQuant Hybrid ---")
    for rk, rv, kb, vb in [(32, 16, 4, 4), (32, 16, 3, 3), (48, 24, 4, 3)]:
        for win in [8, 16]:
            name = f"SVD(k{rk}/v{rv})+TQ(K{kb}/V{vb}) w{win} s4"
            cache = make_cache(nl, lambda li, rk=rk, rv=rv, kb=kb, vb=vb, win=win:
                FP16Layer() if li == 0 else
                LowRankQuantLayer(rank_k=rk, rank_v=rv,
                                   kq=get_q(rk, kb, device, 500+rk),
                                   vq=get_q(rv, vb, device, 600+rv),
                                   num_sinks=4, residual_window=win))
            test(name, cache)

    print(f"\n{'='*80}")
    print("Done!")


if __name__ == "__main__":
    main()
