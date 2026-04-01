"""
TurboQuant v3 — Improvements that actually work
=================================================
Based on v2 experiment results, we focus on three ideas with real signal:

1. PROTECT LAYER 0: It's 5-25x more sensitive. Give it fp16 or K4/V4
   while compressing deeper layers at K3/V2.

2. SMART ADAPTIVE: Fix the naive allocation — set K3 floor (not K2)
   to avoid the 2-bit cliff, concentrate extra bits on layers 0,1,2,4,5.

3. PROGRESSIVE AGING: Fresh tokens at full precision, gradually
   compress as they age. Instead of a hard fp16/quantized boundary,
   use K4 -> K3 -> K2 tiers as tokens get older.

4. MIXED ROTATION: Use Hadamard for speed but keep random orthogonal
   for the most sensitive layers (0, 1, 2).
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

from turboquant_gpt2 import (
    TurboQuantizer, TurboQuantLayer, compute_perplexity, make_turboquant_cache
)
from experiments.exp01_rotation_and_residual import HadamardQuantizer


# ============================================================================
# Strategy 1: Layer-0-Protected Cache
# ============================================================================

class FP16Layer(CacheLayerMixin):
    """A cache layer that stores everything in fp16 (no quantization)."""
    is_sliding = False

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if self.keys.numel() == 0:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_seq_length(self):
        return 0 if not self.is_initialized or self.keys.numel() == 0 else self.keys.shape[-2]

    def get_max_cache_shape(self):
        return -1

    def crop(self, max_length):
        if self.get_seq_length() > max_length:
            self.keys = self.keys[..., :max_length, :]
            self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats):
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]

    def get_mask_sizes(self, cache_position):
        return self.get_seq_length() + cache_position.shape[0], 0


def make_layer0_protected_cache(head_dim, num_layers, key_bits, value_bits,
                                 residual_window, device, protect_layers=None):
    """Layer 0 (and optionally others) stay fp16, rest get quantized."""
    if protect_layers is None:
        protect_layers = {0}

    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None

    for layer_idx in range(num_layers):
        if layer_idx in protect_layers:
            cache.layers.append(FP16Layer())
        else:
            kq = TurboQuantizer(head_dim, key_bits, device=device, seed=42 + layer_idx * 2)
            vq = TurboQuantizer(head_dim, value_bits, device=device, seed=42 + layer_idx * 2 + 1)
            cache.layers.append(TurboQuantLayer(kq, vq, residual_window=residual_window))

    return cache


# ============================================================================
# Strategy 2: Smart Adaptive (K3 floor, sensitivity-aware)
# ============================================================================

def make_smart_adaptive_cache(head_dim, num_layers, residual_window, device,
                               avg_key_bits=3.0, avg_value_bits=2.0):
    """Allocate bits using known GPT-2 layer sensitivities.

    Layer 0 is ~5-25x more sensitive than others.
    Layers 1, 2, 4, 5 are medium sensitivity.
    Layers 3, 6-11 are low sensitivity.

    KEY RULE: Never go below K3 (2-bit cliff is catastrophic).
    """
    # Hard-coded from our sensitivity analysis
    # Layer sensitivities: [1.0, 0.19, 0.18, 0.04, 0.20, 0.20, 0.03, 0.03, 0.06, 0.05, 0.03, 0.01]
    sensitivity_tiers = {
        'critical': [0],        # Layer 0
        'high':     [1, 2, 4, 5],  # Medium sensitivity
        'low':      [3, 6, 7, 8, 9, 10, 11],  # Low sensitivity
    }

    # Bit allocation per tier
    tier_bits = {
        'critical': (4, 4),  # K4/V4
        'high':     (4, 2),  # K4/V2
        'low':      (3, 2),  # K3/V2 (floor at K3!)
    }

    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None

    key_bits_used = []
    val_bits_used = []

    for layer_idx in range(num_layers):
        if layer_idx in sensitivity_tiers['critical']:
            kb, vb = tier_bits['critical']
        elif layer_idx in sensitivity_tiers['high']:
            kb, vb = tier_bits['high']
        else:
            kb, vb = tier_bits['low']

        key_bits_used.append(kb)
        val_bits_used.append(vb)

        kq = TurboQuantizer(head_dim, kb, device=device, seed=42 + layer_idx * 2)
        vq = TurboQuantizer(head_dim, vb, device=device, seed=42 + layer_idx * 2 + 1)
        cache.layers.append(TurboQuantLayer(kq, vq, residual_window=residual_window))

    return cache, key_bits_used, val_bits_used


# ============================================================================
# Strategy 3: Progressive Aging Cache
# ============================================================================

class ProgressiveAgingLayer(CacheLayerMixin):
    """Multi-tier compression: tokens get more compressed as they age.

    Tier 0 (newest):  fp16          — last `tier0_size` tokens
    Tier 1 (middle):  K4/V2 quant  — next `tier1_size` tokens
    Tier 2 (oldest):  K3/V1 quant  — everything older

    This is like PM-KVQ's progressive mixed-precision approach.
    """

    is_sliding = False

    def __init__(self, head_dim, device, tier0_size=32, tier1_size=64,
                 tier1_key_bits=4, tier1_val_bits=2,
                 tier2_key_bits=3, tier2_val_bits=1):
        super().__init__()
        self.head_dim = head_dim
        self.device = device
        self.tier0_size = tier0_size
        self.tier1_size = tier1_size

        # Tier 1 quantizers (medium compression)
        self.tier1_kq = TurboQuantizer(head_dim, tier1_key_bits, device=device, seed=100)
        self.tier1_vq = TurboQuantizer(head_dim, tier1_val_bits, device=device, seed=101)

        # Tier 2 quantizers (heavy compression)
        self.tier2_kq = TurboQuantizer(head_dim, tier2_key_bits, device=device, seed=200)
        self.tier2_vq = TurboQuantizer(head_dim, tier2_val_bits, device=device, seed=201)

        # Storage
        self.tier2_chunks = []  # list of (k_idx, k_norms, v_idx, v_norms, B, H, T)
        self.tier1_chunks = []
        self.recent_keys = None  # fp16
        self.recent_values = None

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True
        self.recent_keys = None
        self.recent_values = None

    def _compress(self, keys, values, kq, vq):
        """Quantize a batch of KV vectors."""
        B, H, T, D = keys.shape
        k_flat = keys.reshape(B * H * T, D)
        v_flat = values.reshape(B * H * T, D)
        k_idx, k_norms = kq.quantize(k_flat)
        v_idx, v_norms = vq.quantize(v_flat)
        return (k_idx, k_norms, v_idx, v_norms, B, H, T)

    def _decompress(self, chunk, kq, vq):
        """Dequantize a compressed chunk."""
        k_idx, k_norms, v_idx, v_norms, B, H, T = chunk
        k_hat = kq.dequantize(k_idx, k_norms).reshape(B, H, T, -1)
        v_hat = vq.dequantize(v_idx, v_norms).reshape(B, H, T, -1)
        return k_hat, v_hat

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Append to recent (tier 0)
        if self.recent_keys is None:
            self.recent_keys = key_states
            self.recent_values = value_states
        else:
            self.recent_keys = torch.cat([self.recent_keys, key_states], dim=-2)
            self.recent_values = torch.cat([self.recent_values, value_states], dim=-2)

        # Cascade: if tier 0 overflows, push to tier 1
        recent_len = self.recent_keys.shape[-2]
        if recent_len > self.tier0_size:
            overflow = recent_len - self.tier0_size
            overflow_k = self.recent_keys[:, :, :overflow, :]
            overflow_v = self.recent_values[:, :, :overflow, :]
            self.recent_keys = self.recent_keys[:, :, overflow:, :].contiguous()
            self.recent_values = self.recent_values[:, :, overflow:, :].contiguous()

            # Compress at tier 1 level
            self.tier1_chunks.append(
                self._compress(overflow_k, overflow_v, self.tier1_kq, self.tier1_vq)
            )

        # Count tier 1 tokens
        tier1_total = sum(c[6] for c in self.tier1_chunks)

        # If tier 1 overflows, re-compress oldest tier1 chunks to tier 2
        while tier1_total > self.tier1_size and len(self.tier1_chunks) > 0:
            oldest = self.tier1_chunks.pop(0)
            # Decompress from tier 1
            k_hat, v_hat = self._decompress(oldest, self.tier1_kq, self.tier1_vq)
            # Re-compress at tier 2 level
            self.tier2_chunks.append(
                self._compress(k_hat, v_hat, self.tier2_kq, self.tier2_vq)
            )
            tier1_total -= oldest[6]

        # Reconstruct full KV for attention
        all_keys = []
        all_values = []

        for chunk in self.tier2_chunks:
            k, v = self._decompress(chunk, self.tier2_kq, self.tier2_vq)
            all_keys.append(k)
            all_values.append(v)

        for chunk in self.tier1_chunks:
            k, v = self._decompress(chunk, self.tier1_kq, self.tier1_vq)
            all_keys.append(k)
            all_values.append(v)

        all_keys.append(self.recent_keys)
        all_values.append(self.recent_values)

        full_keys = torch.cat(all_keys, dim=-2)
        full_values = torch.cat(all_values, dim=-2)

        self.keys = full_keys
        self.values = full_values
        return full_keys, full_values

    def get_seq_length(self):
        t2 = sum(c[6] for c in self.tier2_chunks)
        t1 = sum(c[6] for c in self.tier1_chunks)
        t0 = self.recent_keys.shape[-2] if self.recent_keys is not None else 0
        return t2 + t1 + t0

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


def make_progressive_cache(head_dim, num_layers, device,
                            tier0=32, tier1=64,
                            t1_kb=4, t1_vb=2, t2_kb=3, t2_vb=1):
    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None
    for _ in range(num_layers):
        cache.layers.append(ProgressiveAgingLayer(
            head_dim, device, tier0_size=tier0, tier1_size=tier1,
            tier1_key_bits=t1_kb, tier1_val_bits=t1_vb,
            tier2_key_bits=t2_kb, tier2_val_bits=t2_vb
        ))
    return cache


# ============================================================================
# Strategy 4: Combined — best of everything
# ============================================================================

def make_combined_cache(head_dim, num_layers, device):
    """The ultimate cache:
    - Layer 0: fp16 (it's 5-25x more sensitive)
    - Layers 1,2,4,5 (high sensitivity): progressive aging K4/V2 -> K3/V2
    - Layers 3,6-11 (low sensitivity): aggressive K3/V2 with small window
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
            cache.layers.append(ProgressiveAgingLayer(
                head_dim, device, tier0_size=32, tier1_size=64,
                tier1_key_bits=4, tier1_val_bits=2,
                tier2_key_bits=3, tier2_val_bits=2
            ))
        else:
            kq = TurboQuantizer(head_dim, 3, device=device, seed=42 + layer_idx * 2)
            vq = TurboQuantizer(head_dim, 2, device=device, seed=42 + layer_idx * 2 + 1)
            cache.layers.append(TurboQuantLayer(kq, vq, residual_window=32))

    return cache


# ============================================================================
# Benchmark
# ============================================================================

def compute_cache_memory(config_name, seq_len, head_dim, num_heads, num_layers):
    """Estimate memory for different strategies."""
    fp16_per_token = head_dim * 2 * 2  # K+V, fp16

    if config_name == 'baseline':
        return seq_len * fp16_per_token * num_heads * num_layers

    elif config_name == 'layer0_protected_K3V2_w32':
        total = 0
        for layer in range(num_layers):
            if layer == 0:
                total += seq_len * fp16_per_token * num_heads
            else:
                win = min(32, seq_len)
                compressed = max(0, seq_len - win)
                comp_per = head_dim * 3 / 8 + 2 + head_dim * 2 / 8 + 2
                total += (compressed * comp_per + win * fp16_per_token) * num_heads
        return total

    elif config_name == 'smart_adaptive_w64':
        bits_map = {0: (4,4), 1: (4,2), 2: (4,2), 3: (3,2), 4: (4,2), 5: (4,2),
                    6: (3,2), 7: (3,2), 8: (3,2), 9: (3,2), 10: (3,2), 11: (3,2)}
        total = 0
        win = 64
        for layer in range(num_layers):
            kb, vb = bits_map[layer]
            compressed = max(0, seq_len - win)
            recent = min(seq_len, win)
            comp_per = head_dim * kb / 8 + 2 + head_dim * vb / 8 + 2
            total += (compressed * comp_per + recent * fp16_per_token) * num_heads
        return total

    elif config_name == 'progressive':
        # Rough: tier0 fp16, tier1 K4/V2, tier2 K3/V1
        total = 0
        t0, t1 = 32, 64
        for layer in range(num_layers):
            t0_tokens = min(seq_len, t0)
            t1_tokens = min(max(0, seq_len - t0), t1)
            t2_tokens = max(0, seq_len - t0 - t1)
            t1_per = head_dim * 4 / 8 + 2 + head_dim * 2 / 8 + 2
            t2_per = head_dim * 3 / 8 + 2 + head_dim * 1 / 8 + 2
            total += (t0_tokens * fp16_per_token + t1_tokens * t1_per + t2_tokens * t2_per) * num_heads
        return total

    elif config_name == 'combined':
        total = 0
        critical = {0}
        high_sens = {1, 2, 4, 5}
        for layer in range(num_layers):
            if layer in critical:
                total += seq_len * fp16_per_token * num_heads
            elif layer in high_sens:
                t0, t1 = 32, 64
                t0_tok = min(seq_len, t0)
                t1_tok = min(max(0, seq_len - t0), t1)
                t2_tok = max(0, seq_len - t0 - t1)
                t1_per = head_dim * 4 / 8 + 2 + head_dim * 2 / 8 + 2
                t2_per = head_dim * 3 / 8 + 2 + head_dim * 2 / 8 + 2
                total += (t0_tok * fp16_per_token + t1_tok * t1_per + t2_tok * t2_per) * num_heads
            else:
                win = 32
                compressed = max(0, seq_len - win)
                comp_per = head_dim * 3 / 8 + 2 + head_dim * 2 / 8 + 2
                total += (compressed * comp_per + win * fp16_per_token) * num_heads
        return total

    return seq_len * fp16_per_token * num_heads * num_layers


def main():
    device = 'cuda'
    print("=" * 75)
    print("TurboQuant v3 — Strategies That Work")
    print("=" * 75)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).half()
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

    # Longer text for more interesting compression behavior
    eval_long = eval_text * 2  # ~400 tokens

    seq_len = len(tokenizer.encode(eval_long))
    print(f"\n  Eval text: {seq_len} tokens")

    # ── Baseline ──
    ppl_base = compute_perplexity(model, tokenizer, eval_long)
    print(f"\n  Baseline fp16: PPL = {ppl_base:.2f}")

    # ── All strategies ──
    strategies = []

    # Reference: vanilla TurboQuant configs
    print("\n  --- Reference configs ---")

    for name, kb, vb, win in [("K4/V2 w64", 4, 2, 64), ("K3/V2 w64", 3, 2, 64),
                                ("K4/V2 w32", 4, 2, 32), ("K3/V2 w32", 3, 2, 32)]:
        cache = make_turboquant_cache(head_dim, num_layers, kb, vb, win, device)
        ppl = compute_perplexity(model, tokenizer, eval_long, cache=cache)
        comp_tokens = max(0, seq_len - win)
        rec_tokens = min(seq_len, win)
        fp16_per = head_dim * 2 * 2
        comp_per = head_dim * kb / 8 + 2 + head_dim * vb / 8 + 2
        mem = (comp_tokens * comp_per + rec_tokens * fp16_per) * num_heads * num_layers
        mem_base = seq_len * fp16_per * num_heads * num_layers
        ratio = mem_base / mem
        delta = ((ppl - ppl_base) / ppl_base) * 100
        strategies.append((name, ppl, delta, ratio, 'reference'))
        print(f"    {name:30s}: PPL={ppl:.2f} ({delta:+.1f}%)  {ratio:.2f}x")

    # Strategy 1: Layer-0 protected
    print("\n  --- Strategy 1: Protect Layer 0 ---")

    for protect, kb, vb, win, label in [
        ({0}, 3, 2, 32, "L0=fp16, rest=K3/V2 w32"),
        ({0}, 4, 2, 32, "L0=fp16, rest=K4/V2 w32"),
        ({0, 1, 2}, 3, 2, 32, "L0-2=fp16, rest=K3/V2 w32"),
    ]:
        cache = make_layer0_protected_cache(head_dim, num_layers, kb, vb, win, device, protect)
        ppl = compute_perplexity(model, tokenizer, eval_long, cache=cache)

        # Memory estimate
        fp16_per = head_dim * 2 * 2
        comp_per = head_dim * kb / 8 + 2 + head_dim * vb / 8 + 2
        mem = 0
        for l in range(num_layers):
            if l in protect:
                mem += seq_len * fp16_per * num_heads
            else:
                c = max(0, seq_len - win)
                r = min(seq_len, win)
                mem += (c * comp_per + r * fp16_per) * num_heads
        mem_base = seq_len * fp16_per * num_heads * num_layers
        ratio = mem_base / mem
        delta = ((ppl - ppl_base) / ppl_base) * 100
        strategies.append((label, ppl, delta, ratio, 'layer0'))
        print(f"    {label:30s}: PPL={ppl:.2f} ({delta:+.1f}%)  {ratio:.2f}x")

    # Strategy 2: Smart adaptive
    print("\n  --- Strategy 2: Smart Adaptive (sensitivity-tiered) ---")

    for win in [32, 64]:
        cache, kb_list, vb_list = make_smart_adaptive_cache(head_dim, num_layers, win, device)
        ppl = compute_perplexity(model, tokenizer, eval_long, cache=cache)

        avg_kb = np.mean(kb_list)
        avg_vb = np.mean(vb_list)
        label = f"Adaptive K{avg_kb:.1f}/V{avg_vb:.1f} w{win}"

        # Memory
        fp16_per = head_dim * 2 * 2
        mem = 0
        for l in range(num_layers):
            kb, vb = kb_list[l], vb_list[l]
            comp_per = head_dim * kb / 8 + 2 + head_dim * vb / 8 + 2
            c = max(0, seq_len - win)
            r = min(seq_len, win)
            mem += (c * comp_per + r * fp16_per) * num_heads
        mem_base = seq_len * fp16_per * num_heads * num_layers
        ratio = mem_base / mem
        delta = ((ppl - ppl_base) / ppl_base) * 100
        strategies.append((label, ppl, delta, ratio, 'adaptive'))
        print(f"    {label:30s}: PPL={ppl:.2f} ({delta:+.1f}%)  {ratio:.2f}x")
        for l in range(num_layers):
            print(f"      L{l:2d}: K{kb_list[l]}/V{vb_list[l]}")

    # Strategy 3: Progressive aging
    print("\n  --- Strategy 3: Progressive Aging ---")

    for t0, t1, t1kb, t1vb, t2kb, t2vb in [
        (32, 64, 4, 2, 3, 1, ),
        (32, 64, 4, 2, 3, 2, ),
        (16, 48, 4, 2, 3, 1, ),
        (32, 96, 4, 3, 3, 2, ),
    ]:
        label = f"fp16({t0})->K{t1kb}V{t1vb}({t1})->K{t2kb}V{t2vb}"
        cache = make_progressive_cache(head_dim, num_layers, device,
                                        tier0=t0, tier1=t1,
                                        t1_kb=t1kb, t1_vb=t1vb,
                                        t2_kb=t2kb, t2_vb=t2vb)
        ppl = compute_perplexity(model, tokenizer, eval_long, cache=cache)

        # Memory
        fp16_per = head_dim * 2 * 2
        t0_tok = min(seq_len, t0)
        t1_tok = min(max(0, seq_len - t0), t1)
        t2_tok = max(0, seq_len - t0 - t1)
        t1_per = head_dim * t1kb / 8 + 2 + head_dim * t1vb / 8 + 2
        t2_per = head_dim * t2kb / 8 + 2 + head_dim * t2vb / 8 + 2
        mem = (t0_tok * fp16_per + t1_tok * t1_per + t2_tok * t2_per) * num_heads * num_layers
        mem_base = seq_len * fp16_per * num_heads * num_layers
        ratio = mem_base / mem
        delta = ((ppl - ppl_base) / ppl_base) * 100
        strategies.append((label, ppl, delta, ratio, 'progressive'))
        print(f"    {label:30s}: PPL={ppl:.2f} ({delta:+.1f}%)  {ratio:.2f}x")

    # Strategy 4: Combined
    print("\n  --- Strategy 4: Combined (L0=fp16 + progressive high + aggressive low) ---")
    cache = make_combined_cache(head_dim, num_layers, device)
    ppl = compute_perplexity(model, tokenizer, eval_long, cache=cache)
    mem = compute_cache_memory('combined', seq_len, head_dim, num_heads, num_layers)
    mem_base = compute_cache_memory('baseline', seq_len, head_dim, num_heads, num_layers)
    ratio = mem_base / mem
    delta = ((ppl - ppl_base) / ppl_base) * 100
    strategies.append(("Combined best", ppl, delta, ratio, 'combined'))
    print(f"    {'Combined best':30s}: PPL={ppl:.2f} ({delta:+.1f}%)  {ratio:.2f}x")

    # ── Summary & Plot ──
    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)
    print(f"\n  Baseline PPL: {ppl_base:.2f}")
    print(f"\n  {'Strategy':40s} {'PPL':>8s} {'ΔPPL':>8s} {'Compress':>10s} {'Type':>12s}")
    print("  " + "-" * 82)

    for name, ppl, delta, ratio, stype in sorted(strategies, key=lambda x: x[2]):
        marker = " ★" if delta < 3.0 and ratio > 2.5 else ""
        print(f"  {name:40s} {ppl:>8.2f} {delta:>+7.1f}% {ratio:>9.2f}x {stype:>12s}{marker}")

    # ── Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    type_colors = {
        'reference': '#888888',
        'layer0': '#e74c3c',
        'adaptive': '#2ecc71',
        'progressive': '#3498db',
        'combined': '#9b59b6',
    }
    type_markers = {
        'reference': 'o',
        'layer0': 's',
        'adaptive': 'D',
        'progressive': '^',
        'combined': '*',
    }

    for name, ppl, delta, ratio, stype in strategies:
        c = type_colors[stype]
        m = type_markers[stype]
        sz = 200 if stype == 'combined' else 100
        ax.scatter(ratio, ppl, c=c, marker=m, s=sz, zorder=3, alpha=0.85,
                   edgecolors='black' if stype == 'combined' else 'none',
                   linewidths=2 if stype == 'combined' else 0)

        # Label sweet-spot configs
        if delta < 5.0 and ratio > 2.0:
            ax.annotate(name, (ratio, ppl), fontsize=7, ha='left', va='bottom',
                       xytext=(5, 4), textcoords='offset points', color=c,
                       fontweight='bold' if stype == 'combined' else 'normal')

    # Baseline
    ax.scatter(1.0, ppl_base, c='black', marker='*', s=300, zorder=5, label='Baseline fp16')
    ax.annotate('Baseline fp16', (1.0, ppl_base), fontsize=9, fontweight='bold',
               xytext=(10, -10), textcoords='offset points')

    # Reference lines
    ax.axhline(ppl_base, color='gray', ls=':', alpha=0.5)
    ax.axhline(ppl_base * 1.02, color='green', ls='--', alpha=0.4, lw=1.5, label='2% degradation')
    ax.axhline(ppl_base * 1.05, color='orange', ls='--', alpha=0.5, lw=1.5, label='5% degradation')
    ax.fill_between([0.5, 6], ppl_base, ppl_base * 1.02, alpha=0.06, color='green')
    ax.fill_between([0.5, 6], ppl_base * 1.02, ppl_base * 1.05, alpha=0.04, color='orange')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=type_colors['reference'],
               markersize=10, label='Vanilla TurboQuant'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=type_colors['layer0'],
               markersize=10, label='Layer-0 Protected'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=type_colors['adaptive'],
               markersize=10, label='Smart Adaptive'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=type_colors['progressive'],
               markersize=10, label='Progressive Aging'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=type_colors['combined'],
               markersize=14, label='Combined Best'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left')

    ax.set_xlabel('Compression Ratio (×)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Perplexity (↓ better)', fontsize=13, fontweight='bold')
    ax.set_title('TurboQuant v3 — Improved Strategies vs Vanilla\nGPT-2, ~400 tokens',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.8, 5.5)
    ax.set_ylim(ppl_base - 1, ppl_base * 1.25)

    plt.tight_layout()
    plt.savefig('/home/farmspace/aitest/turboquant_v3_results.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"\nSaved: turboquant_v3_results.png")


if __name__ == "__main__":
    main()
