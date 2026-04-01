"""
TurboQuant Dynamic — Runtime-Adaptive KV-Cache Quantization
=============================================================
Core idea: monitor attention patterns during generation and dynamically
adjust per-layer/per-head compression. Layers that aren't doing useful
work get compressed harder; layers that are actively discriminating get
promoted to higher precision.

Key innovations over vanilla TurboQuant:

1. ATTENTION ENTROPY MONITORING: Track attention entropy per layer.
   High entropy = layer is broadly attending = doing real work = needs bits.
   Low entropy = layer is peaked on 1-2 tokens = routine = compress hard.

2. DUAL-RESOLUTION STORAGE: At insertion time, store BOTH a high-res
   (K4) and low-res (K2) copy. Promote/demote by switching which copy
   gets used — no re-quantization noise. This fixes the progressive
   aging failure from v3.

3. DYNAMIC BIT BUDGET: Given a total memory budget, reallocate bits
   across layers every N tokens based on observed attention patterns.
   Active layers steal bits from dormant ones.

4. ATTENTION-SINK DETECTION: Identify "sink" tokens (first few tokens
   that always get high attention regardless of context) and keep them
   in fp16 permanently — they're load-bearing infrastructure.
"""

import math
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache, CacheLayerMixin

from turboquant_gpt2 import (
    TurboQuantizer, TurboQuantLayer, compute_perplexity, make_turboquant_cache
)


# ============================================================================
# Dual-Resolution Cache Layer
# ============================================================================

class DualResolutionLayer(CacheLayerMixin):
    """Stores tokens at TWO precisions simultaneously.

    At insertion: quantize at both hi-res (e.g. K4/V2) and lo-res (e.g. K2/V1).
    At read time: use hi-res or lo-res per token based on importance signal.

    This avoids the catastrophic re-quantization noise from progressive aging.
    Tradeoff: uses ~1.5x the memory of single-resolution, but can dynamically
    switch between quality levels without any quality loss from re-compression.
    """

    is_sliding = False

    def __init__(self, hi_key_q, hi_val_q, lo_key_q, lo_val_q,
                 fp16_window=32, hi_res_budget=64):
        super().__init__()
        self.hi_kq = hi_key_q
        self.hi_vq = hi_val_q
        self.lo_kq = lo_key_q
        self.lo_vq = lo_val_q
        self.fp16_window = fp16_window
        self.hi_res_budget = hi_res_budget  # how many old tokens get hi-res

        # Storage
        self.hi_k_chunks = []   # (indices, norms, B, H, T)
        self.lo_k_chunks = []
        self.hi_v_chunks = []
        self.lo_v_chunks = []
        self.recent_keys = None
        self.recent_values = None

        # Which resolution to use per chunk (True = hi-res)
        self.use_hi_res = []

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Append to fp16 buffer
        if self.recent_keys is None:
            self.recent_keys = key_states
            self.recent_values = value_states
        else:
            self.recent_keys = torch.cat([self.recent_keys, key_states], dim=-2)
            self.recent_values = torch.cat([self.recent_values, value_states], dim=-2)

        # Overflow -> quantize at BOTH resolutions
        recent_len = self.recent_keys.shape[-2]
        if recent_len > self.fp16_window:
            overflow = recent_len - self.fp16_window
            k_over = self.recent_keys[:, :, :overflow, :]
            v_over = self.recent_values[:, :, :overflow, :]
            self.recent_keys = self.recent_keys[:, :, overflow:, :].contiguous()
            self.recent_values = self.recent_values[:, :, overflow:, :].contiguous()

            B, H, T, D = k_over.shape
            k_flat = k_over.reshape(B * H * T, D)
            v_flat = v_over.reshape(B * H * T, D)

            # Hi-res
            hi_k_idx, hi_k_n = self.hi_kq.quantize(k_flat)
            hi_v_idx, hi_v_n = self.hi_vq.quantize(v_flat)
            self.hi_k_chunks.append((hi_k_idx, hi_k_n, B, H, T))
            self.hi_v_chunks.append((hi_v_idx, hi_v_n, B, H, T))

            # Lo-res
            lo_k_idx, lo_k_n = self.lo_kq.quantize(k_flat)
            lo_v_idx, lo_v_n = self.lo_vq.quantize(v_flat)
            self.lo_k_chunks.append((lo_k_idx, lo_k_n, B, H, T))
            self.lo_v_chunks.append((lo_v_idx, lo_v_n, B, H, T))

            # Default: newest compressed chunks get hi-res
            self.use_hi_res.append(True)

            # Enforce budget: oldest chunks get demoted to lo-res
            hi_count = sum(c[4] for c, u in zip(self.hi_k_chunks, self.use_hi_res) if u)
            while hi_count > self.hi_res_budget and len(self.use_hi_res) > 0:
                # Find oldest hi-res chunk and demote
                for i in range(len(self.use_hi_res)):
                    if self.use_hi_res[i]:
                        self.use_hi_res[i] = False
                        hi_count -= self.hi_k_chunks[i][4]
                        break

        # Reconstruct
        all_keys = []
        all_values = []

        for i in range(len(self.hi_k_chunks)):
            if self.use_hi_res[i]:
                k_idx, k_n, B, H, T = self.hi_k_chunks[i]
                v_idx, v_n, _, _, _ = self.hi_v_chunks[i]
                k_hat = self.hi_kq.dequantize(k_idx, k_n).reshape(B, H, T, -1)
                v_hat = self.hi_vq.dequantize(v_idx, v_n).reshape(B, H, T, -1)
            else:
                k_idx, k_n, B, H, T = self.lo_k_chunks[i]
                v_idx, v_n, _, _, _ = self.lo_v_chunks[i]
                k_hat = self.lo_kq.dequantize(k_idx, k_n).reshape(B, H, T, -1)
                v_hat = self.lo_vq.dequantize(v_idx, v_n).reshape(B, H, T, -1)

            all_keys.append(k_hat)
            all_values.append(v_hat)

        all_keys.append(self.recent_keys)
        all_values.append(self.recent_values)

        full_keys = torch.cat(all_keys, dim=-2)
        full_values = torch.cat(all_values, dim=-2)
        self.keys = full_keys
        self.values = full_values
        return full_keys, full_values

    def get_seq_length(self):
        c = sum(ch[4] for ch in self.hi_k_chunks)
        r = self.recent_keys.shape[-2] if self.recent_keys is not None else 0
        return c + r

    def get_max_cache_shape(self):
        return -1

    def crop(self, max_length):
        pass

    def batch_repeat_interleave(self, repeats):
        if self.recent_keys is not None:
            self.recent_keys = self.recent_keys.repeat_interleave(repeats, dim=0)
            self.recent_values = self.recent_values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        if self.recent_keys is not None:
            self.recent_keys = self.recent_keys[indices, ...]
            self.recent_values = self.recent_values[indices, ...]

    def get_mask_sizes(self, cache_position):
        return self.get_seq_length() + cache_position.shape[0], 0


# ============================================================================
# Attention-Sink-Aware Cache Layer
# ============================================================================

class SinkAwareLayer(CacheLayerMixin):
    """Keeps attention-sink tokens in fp16 permanently.

    Attention sinks are the first few tokens that get disproportionately
    high attention regardless of context (observed in most transformer LMs).
    Quantizing them hurts quality disproportionately.

    Structure:
        [sink tokens fp16] [compressed old tokens] [recent tokens fp16]

    The sink tokens are identified by position (first N tokens).
    """

    is_sliding = False

    def __init__(self, key_quantizer, value_quantizer,
                 num_sinks=4, residual_window=32):
        super().__init__()
        self.kq = key_quantizer
        self.vq = value_quantizer
        self.num_sinks = num_sinks
        self.residual_window = residual_window

        self.sink_keys = None       # fp16, first N tokens
        self.sink_values = None
        self.compressed_k = []      # quantized middle tokens
        self.compressed_v = []
        self.compressed_shapes = []
        self.recent_keys = None     # fp16, last M tokens
        self.recent_values = None
        self.total_seen = 0

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        B, H, T, D = key_states.shape

        # First call: capture sink tokens
        if self.total_seen == 0 and T >= self.num_sinks:
            self.sink_keys = key_states[:, :, :self.num_sinks, :].clone()
            self.sink_values = value_states[:, :, :self.num_sinks, :].clone()
            # Rest goes to recent buffer
            if T > self.num_sinks:
                remaining_k = key_states[:, :, self.num_sinks:, :]
                remaining_v = value_states[:, :, self.num_sinks:, :]
                self.recent_keys = remaining_k
                self.recent_values = remaining_v
            else:
                self.recent_keys = key_states[:, :, :0, :]  # empty
                self.recent_values = value_states[:, :, :0, :]
            self.total_seen = T
        elif self.total_seen == 0:
            # Not enough tokens for sinks yet, just buffer
            self.sink_keys = key_states.clone()
            self.sink_values = value_states.clone()
            self.recent_keys = key_states[:, :, :0, :]
            self.recent_values = value_states[:, :, :0, :]
            self.total_seen = T
        else:
            # Append to recent
            if self.recent_keys.shape[-2] == 0:
                self.recent_keys = key_states
                self.recent_values = value_states
            else:
                self.recent_keys = torch.cat([self.recent_keys, key_states], dim=-2)
                self.recent_values = torch.cat([self.recent_values, value_states], dim=-2)
            self.total_seen += T

        # Compress overflow from recent buffer
        recent_len = self.recent_keys.shape[-2]
        if recent_len > self.residual_window:
            overflow = recent_len - self.residual_window
            k_over = self.recent_keys[:, :, :overflow, :]
            v_over = self.recent_values[:, :, :overflow, :]
            self.recent_keys = self.recent_keys[:, :, overflow:, :].contiguous()
            self.recent_values = self.recent_values[:, :, overflow:, :].contiguous()

            B2, H2, T2, D2 = k_over.shape
            k_flat = k_over.reshape(B2 * H2 * T2, D2)
            v_flat = v_over.reshape(B2 * H2 * T2, D2)
            k_idx, k_n = self.kq.quantize(k_flat)
            v_idx, v_n = self.vq.quantize(v_flat)
            self.compressed_k.append((k_idx, k_n))
            self.compressed_v.append((v_idx, v_n))
            self.compressed_shapes.append((B2, H2, T2))

        # Reconstruct: [sinks fp16] [compressed] [recent fp16]
        parts_k = [self.sink_keys]
        parts_v = [self.sink_values]

        for i, (B2, H2, T2) in enumerate(self.compressed_shapes):
            k_idx, k_n = self.compressed_k[i]
            v_idx, v_n = self.compressed_v[i]
            k_hat = self.kq.dequantize(k_idx, k_n).reshape(B2, H2, T2, -1)
            v_hat = self.vq.dequantize(v_idx, v_n).reshape(B2, H2, T2, -1)
            parts_k.append(k_hat)
            parts_v.append(v_hat)

        parts_k.append(self.recent_keys)
        parts_v.append(self.recent_values)

        full_keys = torch.cat(parts_k, dim=-2)
        full_values = torch.cat(parts_v, dim=-2)
        self.keys = full_keys
        self.values = full_values
        return full_keys, full_values

    def get_seq_length(self):
        s = self.sink_keys.shape[-2] if self.sink_keys is not None else 0
        c = sum(sh[2] for sh in self.compressed_shapes)
        r = self.recent_keys.shape[-2] if self.recent_keys is not None else 0
        return s + c + r

    def get_max_cache_shape(self):
        return -1

    def crop(self, max_length):
        pass

    def batch_repeat_interleave(self, repeats):
        if self.sink_keys is not None:
            self.sink_keys = self.sink_keys.repeat_interleave(repeats, dim=0)
            self.sink_values = self.sink_values.repeat_interleave(repeats, dim=0)
        if self.recent_keys is not None:
            self.recent_keys = self.recent_keys.repeat_interleave(repeats, dim=0)
            self.recent_values = self.recent_values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        if self.sink_keys is not None:
            self.sink_keys = self.sink_keys[indices, ...]
            self.sink_values = self.sink_values[indices, ...]
        if self.recent_keys is not None:
            self.recent_keys = self.recent_keys[indices, ...]
            self.recent_values = self.recent_values[indices, ...]

    def get_mask_sizes(self, cache_position):
        return self.get_seq_length() + cache_position.shape[0], 0


# ============================================================================
# Entropy-Monitored Dynamic Cache
# ============================================================================

class EntropyMonitoredLayer(CacheLayerMixin):
    """Cache layer that monitors its own attention entropy and adjusts.

    We hook into the attention mechanism to observe the entropy of
    attention weights. When entropy is low (peaked attention), we
    compress more aggressively. When entropy is high (broad attention),
    we use higher precision.

    Implementation: store at TWO precisions (dual-res). A controller
    periodically adjusts how many tokens get hi-res based on entropy.
    """

    is_sliding = False

    def __init__(self, hi_kq, hi_vq, lo_kq, lo_vq,
                 fp16_window=32, base_hi_budget=48):
        super().__init__()
        self.hi_kq = hi_kq
        self.hi_vq = hi_vq
        self.lo_kq = lo_kq
        self.lo_vq = lo_vq
        self.fp16_window = fp16_window
        self.base_hi_budget = base_hi_budget

        # Entropy tracking
        self.entropy_history = deque(maxlen=50)
        self.current_hi_budget = base_hi_budget

        # Storage (same as DualResolutionLayer)
        self.hi_k_chunks = []
        self.hi_v_chunks = []
        self.lo_k_chunks = []
        self.lo_v_chunks = []
        self.use_hi_res = []
        self.recent_keys = None
        self.recent_values = None

    def set_entropy(self, entropy):
        """Called externally by the attention monitor hook."""
        self.entropy_history.append(entropy)
        # Adjust budget: high entropy -> more hi-res tokens
        if len(self.entropy_history) >= 5:
            avg_entropy = np.mean(list(self.entropy_history)[-5:])
            # Scale: entropy 0->1 maps to 0.5x->1.5x base budget
            scale = 0.5 + avg_entropy
            self.current_hi_budget = max(16, int(self.base_hi_budget * scale))

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if self.recent_keys is None:
            self.recent_keys = key_states
            self.recent_values = value_states
        else:
            self.recent_keys = torch.cat([self.recent_keys, key_states], dim=-2)
            self.recent_values = torch.cat([self.recent_values, value_states], dim=-2)

        recent_len = self.recent_keys.shape[-2]
        if recent_len > self.fp16_window:
            overflow = recent_len - self.fp16_window
            k_over = self.recent_keys[:, :, :overflow, :]
            v_over = self.recent_values[:, :, :overflow, :]
            self.recent_keys = self.recent_keys[:, :, overflow:, :].contiguous()
            self.recent_values = self.recent_values[:, :, overflow:, :].contiguous()

            B, H, T, D = k_over.shape
            k_flat = k_over.reshape(B * H * T, D)
            v_flat = v_over.reshape(B * H * T, D)

            self.hi_k_chunks.append((self.hi_kq.quantize(k_flat)[0], self.hi_kq.quantize(k_flat)[1], B, H, T))
            self.hi_v_chunks.append((self.hi_vq.quantize(v_flat)[0], self.hi_vq.quantize(v_flat)[1], B, H, T))
            self.lo_k_chunks.append((self.lo_kq.quantize(k_flat)[0], self.lo_kq.quantize(k_flat)[1], B, H, T))
            self.lo_v_chunks.append((self.lo_vq.quantize(v_flat)[0], self.lo_vq.quantize(v_flat)[1], B, H, T))
            self.use_hi_res.append(True)

            # Enforce dynamic budget
            hi_count = sum(c[4] for c, u in zip(self.hi_k_chunks, self.use_hi_res) if u)
            while hi_count > self.current_hi_budget:
                for i in range(len(self.use_hi_res)):
                    if self.use_hi_res[i]:
                        self.use_hi_res[i] = False
                        hi_count -= self.hi_k_chunks[i][4]
                        break
                else:
                    break

        # Reconstruct
        all_keys = []
        all_values = []
        for i in range(len(self.hi_k_chunks)):
            if self.use_hi_res[i]:
                k_idx, k_n, B, H, T = self.hi_k_chunks[i]
                v_idx, v_n, _, _, _ = self.hi_v_chunks[i]
                kq, vq = self.hi_kq, self.hi_vq
            else:
                k_idx, k_n, B, H, T = self.lo_k_chunks[i]
                v_idx, v_n, _, _, _ = self.lo_v_chunks[i]
                kq, vq = self.lo_kq, self.lo_vq
            all_keys.append(kq.dequantize(k_idx, k_n).reshape(B, H, T, -1))
            all_values.append(vq.dequantize(v_idx, v_n).reshape(B, H, T, -1))

        all_keys.append(self.recent_keys)
        all_values.append(self.recent_values)
        full_keys = torch.cat(all_keys, dim=-2)
        full_values = torch.cat(all_values, dim=-2)
        self.keys = full_keys
        self.values = full_values
        return full_keys, full_values

    def get_seq_length(self):
        c = sum(ch[4] for ch in self.hi_k_chunks)
        r = self.recent_keys.shape[-2] if self.recent_keys is not None else 0
        return c + r

    def get_max_cache_shape(self):
        return -1

    def crop(self, max_length):
        pass

    def batch_repeat_interleave(self, repeats):
        if self.recent_keys is not None:
            self.recent_keys = self.recent_keys.repeat_interleave(repeats, dim=0)
            self.recent_values = self.recent_values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        if self.recent_keys is not None:
            self.recent_keys = self.recent_keys[indices, ...]
            self.recent_values = self.recent_values[indices, ...]

    def get_mask_sizes(self, cache_position):
        return self.get_seq_length() + cache_position.shape[0], 0


# ============================================================================
# Cache Factories
# ============================================================================

# Shared quantizer cache
_qcache = {}

def _get_q(head_dim, bits, device, seed):
    key = (head_dim, bits, seed)
    if key not in _qcache:
        _qcache[key] = TurboQuantizer(head_dim, bits, device=device, seed=seed)
    return _qcache[key]


class FP16Layer(CacheLayerMixin):
    """Simple fp16 cache layer."""
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


def make_sink_cache(head_dim, num_layers, key_bits, value_bits,
                    num_sinks, residual_window, device,
                    sensitivity_tiers=None):
    """Sink-aware cache with optional per-layer sensitivity."""
    if sensitivity_tiers is None:
        sensitivity_tiers = {i: (key_bits, value_bits) for i in range(num_layers)}

    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None

    for layer_idx in range(num_layers):
        kb, vb = sensitivity_tiers.get(layer_idx, (key_bits, value_bits))
        kq = _get_q(head_dim, kb, device, 42 + layer_idx * 2)
        vq = _get_q(head_dim, vb, device, 42 + layer_idx * 2 + 1)
        cache.layers.append(SinkAwareLayer(kq, vq, num_sinks=num_sinks,
                                            residual_window=residual_window))
    return cache


def make_dual_res_cache(head_dim, num_layers, device,
                         fp16_window=32, hi_budget=48,
                         hi_kb=4, hi_vb=2, lo_kb=2, lo_vb=1):
    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None
    for layer_idx in range(num_layers):
        hi_kq = _get_q(head_dim, hi_kb, device, 42 + layer_idx * 2)
        hi_vq = _get_q(head_dim, hi_vb, device, 42 + layer_idx * 2 + 1)
        lo_kq = _get_q(head_dim, lo_kb, device, 300 + layer_idx * 2)
        lo_vq = _get_q(head_dim, lo_vb, device, 300 + layer_idx * 2 + 1)
        cache.layers.append(DualResolutionLayer(
            hi_kq, hi_vq, lo_kq, lo_vq,
            fp16_window=fp16_window, hi_res_budget=hi_budget
        ))
    return cache


def make_full_dynamic_cache(head_dim, num_layers, device):
    """The full dynamic system:
    - Layer 0: fp16 (5-25x more sensitive)
    - Layers 1,2,4,5: sink-aware + K4/V2 (high sensitivity)
    - Layers 3,6-11: sink-aware + K3/V2 (low sensitivity)
    - All layers: protect first 4 tokens as sinks
    """
    critical = {0}
    high_sens = {1, 2, 4, 5}

    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None

    for layer_idx in range(num_layers):
        if layer_idx in critical:
            cache.layers.append(FP16Layer())
        elif layer_idx in high_sens:
            kq = _get_q(head_dim, 4, device, 42 + layer_idx * 2)
            vq = _get_q(head_dim, 2, device, 42 + layer_idx * 2 + 1)
            cache.layers.append(SinkAwareLayer(kq, vq, num_sinks=4, residual_window=32))
        else:
            kq = _get_q(head_dim, 3, device, 42 + layer_idx * 2)
            vq = _get_q(head_dim, 2, device, 42 + layer_idx * 2 + 1)
            cache.layers.append(SinkAwareLayer(kq, vq, num_sinks=4, residual_window=32))

    return cache


# ============================================================================
# Attention Sink Analysis
# ============================================================================

def analyze_attention_sinks(model, tokenizer, text, device='cuda'):
    """Measure which token positions get the most attention across layers."""
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    # outputs.attentions is tuple of (B, H, T, T) per layer
    attention_maps = outputs.attentions

    print(f"\n  Attention Sink Analysis ({seq_len} tokens)")
    print(f"  Average attention received by token position (across all heads):\n")

    per_layer_sinks = []

    for layer_idx, attn in enumerate(attention_maps):
        # attn: (1, H, T, T) — attention FROM each query TO each key
        # Sum attention received by each key position (sum over queries and heads)
        attn_received = attn.float().mean(dim=(0, 1, 2))  # (T,) — avg attention on each key
        top_positions = attn_received.topk(min(8, seq_len)).indices.cpu().numpy()
        top_values = attn_received.topk(min(8, seq_len)).values.cpu().numpy()

        per_layer_sinks.append(top_positions)

        # Show first 10 positions' attention share
        first_n = min(10, seq_len)
        shares = attn_received[:first_n].cpu().numpy()
        bar = " ".join(f"{s:.3f}" for s in shares)
        print(f"    Layer {layer_idx:2d}: [{bar}]  top={top_positions[:4]}")

    # Find universal sinks (positions that are top-4 in most layers)
    from collections import Counter
    all_top = []
    for tops in per_layer_sinks:
        all_top.extend(tops[:4])
    sink_counts = Counter(all_top)
    universal_sinks = [pos for pos, count in sink_counts.most_common(8) if count >= 6]
    print(f"\n    Universal sinks (top-4 in ≥50% of layers): positions {universal_sinks}")

    # Compute attention entropy per layer
    entropies = []
    for layer_idx, attn in enumerate(attention_maps):
        # Per-query entropy, averaged
        attn_f = attn.float().clamp(min=1e-10)
        entropy = -(attn_f * attn_f.log()).sum(dim=-1).mean().item()
        entropy_normalized = entropy / math.log(seq_len)  # normalize to [0, 1]
        entropies.append(entropy_normalized)

    print(f"\n  Attention entropy per layer (0=peaked, 1=uniform):")
    for i, e in enumerate(entropies):
        bar = "█" * int(e * 40)
        print(f"    Layer {i:2d}: {e:.3f} {bar}")

    return universal_sinks, entropies


# ============================================================================
# Benchmark
# ============================================================================

def main():
    device = 'cuda'
    print("=" * 75)
    print("TurboQuant Dynamic — Runtime-Adaptive Experiments")
    print("=" * 75)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation='eager').to(device).half()
    model.eval()

    head_dim, num_layers, num_heads = 64, 12, 12

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
        "difficulty of the project. In 1974, in response to the criticism from James Lighthill and "
        "ongoing pressure from congress, the U.S. and British governments cut off exploratory research "
        "in AI. The next few years would later be called an AI winter."
    )
    eval_long = eval_text * 2
    seq_len = len(tokenizer.encode(eval_long))
    print(f"\n  Eval: {seq_len} tokens")

    # ── Pre-compute codebooks ──
    print("\n  Pre-computing codebooks...")
    t0 = time.time()
    for bits in [1, 2, 3, 4]:
        for layer_idx in range(num_layers):
            _get_q(head_dim, bits, device, 42 + layer_idx * 2)
            _get_q(head_dim, bits, device, 42 + layer_idx * 2 + 1)
            _get_q(head_dim, bits, device, 300 + layer_idx * 2)
            _get_q(head_dim, bits, device, 300 + layer_idx * 2 + 1)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Attention Sink Analysis ──
    print("\n" + "=" * 75)
    print("Analysis: Attention Sinks & Entropy")
    print("=" * 75)
    sinks, entropies = analyze_attention_sinks(model, tokenizer, eval_text, device)

    # ── Baseline ──
    ppl_base = compute_perplexity(model, tokenizer, eval_long)
    print(f"\n  Baseline fp16: PPL = {ppl_base:.2f}")

    # ── Experiments ──
    results = []

    def test(name, cache, category):
        ppl = compute_perplexity(model, tokenizer, eval_long, cache=cache)
        delta = ((ppl - ppl_base) / ppl_base) * 100
        results.append((name, ppl, delta, category))
        print(f"    {name:45s}: PPL={ppl:.2f} ({delta:+.1f}%)")
        return ppl

    # Reference configs
    print("\n  --- Reference (vanilla TurboQuant) ---")
    test("Vanilla K4/V2 w64", make_turboquant_cache(head_dim, num_layers, 4, 2, 64, device), 'ref')
    test("Vanilla K4/V2 w32", make_turboquant_cache(head_dim, num_layers, 4, 2, 32, device), 'ref')
    test("Vanilla K3/V2 w32", make_turboquant_cache(head_dim, num_layers, 3, 2, 32, device), 'ref')

    # Sink-aware (same bits, but protect first 4 tokens in fp16)
    print("\n  --- Sink-Aware (protect first N tokens in fp16) ---")
    for n_sinks in [2, 4, 8, 16]:
        cache = make_sink_cache(head_dim, num_layers, 4, 2, n_sinks, 32, device)
        test(f"Sink({n_sinks}) + K4/V2 w32", cache, 'sink')

    for n_sinks in [4, 8]:
        cache = make_sink_cache(head_dim, num_layers, 3, 2, n_sinks, 32, device)
        test(f"Sink({n_sinks}) + K3/V2 w32", cache, 'sink')

    # Sink-aware + sensitivity tiers
    print("\n  --- Sink-Aware + Sensitivity Tiers ---")
    tiers = {}
    for i in range(num_layers):
        if i == 0: tiers[i] = (4, 4)       # critical
        elif i in {1,2,4,5}: tiers[i] = (4, 2)  # high
        else: tiers[i] = (3, 2)            # low

    for n_sinks in [4, 8]:
        cache = make_sink_cache(head_dim, num_layers, 4, 2, n_sinks, 32, device,
                                sensitivity_tiers=tiers)
        test(f"Sink({n_sinks}) + Adaptive tiers w32", cache, 'sink+adaptive')

    # Dual-resolution
    print("\n  --- Dual-Resolution (hi=K4/V2, lo=K2/V1) ---")
    for fp16_win, hi_bud in [(16, 32), (32, 32), (32, 64)]:
        cache = make_dual_res_cache(head_dim, num_layers, device,
                                     fp16_window=fp16_win, hi_budget=hi_bud)
        test(f"DualRes fp16={fp16_win} hi={hi_bud}", cache, 'dual')

    # Full dynamic system
    print("\n  --- Full Dynamic (L0=fp16 + sink-aware + adaptive) ---")
    cache = make_full_dynamic_cache(head_dim, num_layers, device)
    test("Full Dynamic", cache, 'dynamic')

    # Also test: L0 fp16 + sink + aggressive compression
    print("\n  --- Full Dynamic + more aggressive low-sens layers ---")

    def make_aggressive_dynamic(head_dim, num_layers, device):
        critical = {0}
        high_sens = {1, 2, 4, 5}
        cache = DynamicCache()
        cache.layers = []
        cache.layer_class_to_replicate = None
        for li in range(num_layers):
            if li in critical:
                cache.layers.append(FP16Layer())
            elif li in high_sens:
                kq = _get_q(head_dim, 4, device, 42 + li * 2)
                vq = _get_q(head_dim, 2, device, 42 + li * 2 + 1)
                cache.layers.append(SinkAwareLayer(kq, vq, num_sinks=4, residual_window=48))
            else:
                kq = _get_q(head_dim, 3, device, 42 + li * 2)
                vq = _get_q(head_dim, 1, device, 42 + li * 2 + 1)
                cache.layers.append(SinkAwareLayer(kq, vq, num_sinks=4, residual_window=16))
        return cache

    test("Aggressive Dynamic (low=K3/V1 w16)", make_aggressive_dynamic(head_dim, num_layers, device), 'dynamic')

    # ── Summary ──
    print("\n" + "=" * 75)
    print("FINAL RANKING (sorted by PPL)")
    print("=" * 75)
    print(f"\n  Baseline: PPL = {ppl_base:.2f}\n")
    print(f"  {'Strategy':47s} {'PPL':>7s} {'ΔPPL':>8s} {'Type':>15s}")
    print("  " + "-" * 80)
    for name, ppl, delta, cat in sorted(results, key=lambda x: x[1]):
        star = " ★" if delta < 5.0 else ""
        print(f"  {name:47s} {ppl:>7.2f} {delta:>+7.1f}% {cat:>15s}{star}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(14, 8))

    cat_colors = {
        'ref': '#888888', 'sink': '#e74c3c', 'sink+adaptive': '#e67e22',
        'dual': '#3498db', 'dynamic': '#9b59b6'
    }
    cat_markers = {
        'ref': 'o', 'sink': 's', 'sink+adaptive': 'D',
        'dual': '^', 'dynamic': '*'
    }

    for i, (name, ppl, delta, cat) in enumerate(results):
        c = cat_colors.get(cat, '#999')
        m = cat_markers.get(cat, 'o')
        sz = 200 if cat == 'dynamic' else 100
        ax.scatter(i, ppl, c=c, marker=m, s=sz, zorder=3, alpha=0.85,
                   edgecolors='black' if cat == 'dynamic' else 'none',
                   linewidths=2 if cat == 'dynamic' else 0)

    ax.axhline(ppl_base, color='gray', ls=':', alpha=0.5, label=f'Baseline = {ppl_base:.2f}')
    ax.axhline(ppl_base * 1.02, color='green', ls='--', alpha=0.4, lw=1.5, label='2% threshold')
    ax.axhline(ppl_base * 1.05, color='orange', ls='--', alpha=0.5, lw=1.5, label='5% threshold')

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r[0] for r in results], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Perplexity (↓ better)', fontsize=12, fontweight='bold')
    ax.set_title('TurboQuant Dynamic — All Strategies Compared', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_colors['ref'], markersize=10, label='Vanilla TurboQuant'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=cat_colors['sink'], markersize=10, label='Sink-Aware'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=cat_colors['sink+adaptive'], markersize=10, label='Sink + Adaptive'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=cat_colors['dual'], markersize=10, label='Dual-Resolution'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=cat_colors['dynamic'], markersize=14, label='Full Dynamic'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig('/home/farmspace/aitest/turboquant_dynamic_results.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"\n  Saved: turboquant_dynamic_results.png")


if __name__ == "__main__":
    main()
