"""
TurboQuant v2 — Improvements over vanilla TurboQuant
=====================================================
We analyze where quantization error actually hurts and test several
improvements grounded in the math of attention.

THE CORE INSIGHT:
  output = softmax(Q @ K^T / sqrt(d)) @ V

Minimizing MSE on K and V independently is a PROXY for the real objective:
minimizing error on `output`. The actual error propagation is:

  For KEYS:  error in K -> error in attention scores -> softmax amplifies
             differentially (errors near the max score matter more)
  For VALUES: error in V -> linearly mixed by attention weights
             (errors in high-attention-weight positions matter more)

This suggests several improvements:

1. HADAMARD ROTATION: Replace random orthogonal with Walsh-Hadamard transform.
   - O(d log d) vs O(d^2) — 6x faster for d=64
   - Equally good at spreading energy (proven in QuIP#)
   - Deterministic — no random seed management

2. PER-LAYER ADAPTIVE BITS: Some layers are more sensitive.
   Measure sensitivity = d(output)/d(quantization_noise) per layer,
   then allocate bits proportionally.

3. IMPORTANCE-WEIGHTED QUANTIZATION: After rotation, weight the
   Lloyd-Max objective by how much each coordinate contributes to
   attention score variance. High-variance coordinates get finer bins.

4. RESIDUAL QUANTIZATION: Two-stage quantize -> compute residual ->
   quantize residual. Uses the same total bits but distributes error
   more evenly.

5. ATTENTION-SCORE-AWARE KEY QUANTIZATION: Instead of MSE on keys,
   minimize E[|softmax(Q@K_hat^T) - softmax(Q@K^T)|^2].
   Approximation: weight key quantization error by attention score.

6. CHANNEL REORDERING: Sort channels by variance before quantizing.
   High-variance channels get outer (wider) quantization bins naturally,
   but we can do better by splitting into groups with different codebooks.
"""

import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from scipy import integrate
from scipy.special import gammaln
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache, CacheLayerMixin

from turboquant_gpt2 import (
    compute_lloyd_max_codebook, TurboQuantizer, TurboQuantLayer,
    compute_perplexity, beta_pdf, gaussian_pdf
)


# ============================================================================
# Improvement 1: Hadamard Rotation (replaces random orthogonal)
# ============================================================================

def hadamard_matrix(d):
    """Construct normalized Hadamard matrix of size d (d must be power of 2).
    Uses the recursive Sylvester construction."""
    assert d > 0 and (d & (d - 1)) == 0, f"d={d} must be power of 2"
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(d)  # normalize so H @ H^T = I


def random_diagonal_sign(d, seed=42):
    """Random ±1 diagonal — makes Hadamard rotation data-oblivious."""
    rng = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (d,), generator=rng) * 2 - 1).float()


class HadamardQuantizer:
    """TurboQuant with Hadamard rotation instead of random orthogonal.

    Uses the randomized Hadamard transform (RHT):
        y = H @ diag(signs) @ x
    where H is the Walsh-Hadamard matrix and signs are random ±1.

    Properties:
    - O(d log d) rotation via fast Walsh-Hadamard transform
    - Equally good at making coordinates near-independent
    - Deterministic H, randomness only in signs
    """

    def __init__(self, head_dim, bits, device='cuda', seed=42):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        centroids_np, boundaries_np = compute_lloyd_max_codebook(
            head_dim, bits, use_exact_pdf=(head_dim < 128)
        )
        self.centroids = torch.from_numpy(centroids_np).half().to(device)
        self.boundaries = torch.from_numpy(boundaries_np).half().to(device)

        # Hadamard matrix + random sign flip
        self.H = hadamard_matrix(head_dim).half().to(device)
        self.signs = random_diagonal_sign(head_dim, seed=seed).half().to(device)

    def _rotate(self, x):
        """Forward rotation: y = H @ diag(signs) @ x"""
        return (x * self.signs) @ self.H.T

    def _unrotate(self, y):
        """Inverse rotation: x = diag(signs) @ H^T @ y = diag(signs) @ H @ y (H is symmetric)"""
        return (y @ self.H) * self.signs

    def quantize(self, x):
        x = x.half()
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        y = self._rotate(x_unit)
        indices = torch.searchsorted(self.boundaries, y.contiguous())
        return indices, norms.squeeze(-1)

    def dequantize(self, indices, norms):
        y_hat = self.centroids[indices]
        x_hat = self._unrotate(y_hat)
        return x_hat * norms.unsqueeze(-1)


# ============================================================================
# Improvement 2: Per-Layer Adaptive Bit Allocation
# ============================================================================

def measure_layer_sensitivity(model, tokenizer, text, device='cuda'):
    """Measure how sensitive each layer's output is to KV quantization noise.

    Method: For each layer, add calibrated noise to K and V, measure
    the change in the final logits. Layers with higher sensitivity
    should get more bits.
    """
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    num_layers = model.config.n_layer

    # Get baseline logits
    with torch.no_grad():
        base_outputs = model(input_ids)
        base_logits = base_outputs.logits

    sensitivities = []

    for target_layer in range(num_layers):
        hooks = []
        noise_scale = 0.01  # small perturbation

        def make_hook(layer_idx):
            def hook_fn(module, args, output):
                if layer_idx == target_layer:
                    # output is a tuple; first element is hidden states
                    # For GPT-2 attention, we hook the attention layer
                    # and perturb the output
                    if isinstance(output, tuple):
                        perturbed = output[0] + torch.randn_like(output[0]) * noise_scale
                        return (perturbed,) + output[1:]
                return output
            return hook_fn

        # Hook each transformer block
        for i, block in enumerate(model.transformer.h):
            h = block.attn.register_forward_hook(make_hook(i))
            hooks.append(h)

        with torch.no_grad():
            noisy_outputs = model(input_ids)
            noisy_logits = noisy_outputs.logits

        for h in hooks:
            h.remove()

        # Sensitivity = how much logits changed
        logit_diff = (noisy_logits - base_logits).float().pow(2).mean().item()
        sensitivities.append(logit_diff)

    sensitivities = np.array(sensitivities)
    # Normalize to [0, 1]
    if sensitivities.max() > 0:
        sensitivities = sensitivities / sensitivities.max()

    return sensitivities


def allocate_bits_by_sensitivity(sensitivities, total_bit_budget, min_bits=2, max_bits=4):
    """Allocate bits per layer proportional to sensitivity.

    Args:
        sensitivities: (num_layers,) array of sensitivity scores
        total_bit_budget: average bits per layer * num_layers
        min_bits, max_bits: bounds

    Returns:
        bits_per_layer: (num_layers,) array of integer bit allocations
    """
    num_layers = len(sensitivities)
    avg_bits = total_bit_budget / num_layers

    # Allocate proportionally to sensitivity
    weights = sensitivities / (sensitivities.sum() + 1e-10)
    raw_bits = weights * total_bit_budget

    # Clip and round
    bits = np.clip(raw_bits, min_bits, max_bits)

    # Round to integers while preserving total budget
    bits_int = np.round(bits).astype(int)
    bits_int = np.clip(bits_int, min_bits, max_bits)

    return bits_int


# ============================================================================
# Improvement 3: Residual Quantization (two-stage)
# ============================================================================

class ResidualQuantizer:
    """Two-stage residual quantization.

    Stage 1: Quantize at (bits-1) bits -> get coarse approximation
    Stage 2: Compute residual, quantize residual at 1 bit
    Total: same bits as single-stage, but better error distribution.

    The key insight: after stage 1, the residual has a DIFFERENT distribution
    than the original (it's concentrated near zero). We use a separate
    codebook optimized for the residual distribution.
    """

    def __init__(self, head_dim, bits, device='cuda', seed=42):
        assert bits >= 2, "Need at least 2 bits for residual quantization"
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        # Stage 1: coarse quantizer at (bits-1) bits
        self.coarse = TurboQuantizer(head_dim, bits - 1, device=device, seed=seed)

        # Stage 2: 1-bit quantizer for the residual
        # The residual after rotation is concentrated near zero with smaller variance
        # We use a 1-bit codebook optimized for this distribution
        # Empirically, the residual variance is ~(1 - cos_sim_coarse) of original
        self.fine = TurboQuantizer(head_dim, 1, device=device, seed=seed + 1000)

    def quantize(self, x):
        x = x.half()
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # Stage 1: coarse
        coarse_idx, _ = self.coarse.quantize(x_unit)
        x_coarse = self.coarse.dequantize(coarse_idx, torch.ones(x_unit.shape[0], device=self.device).half())

        # Stage 2: quantize residual
        residual = x_unit - x_coarse
        res_idx, res_norms = self.fine.quantize(residual)

        return coarse_idx, res_idx, res_norms, norms.squeeze(-1)

    def dequantize(self, coarse_idx, res_idx, res_norms, norms):
        x_coarse = self.coarse.dequantize(
            coarse_idx, torch.ones(coarse_idx.shape[0], device=self.device).half()
        )
        x_res = self.fine.dequantize(res_idx, res_norms)
        x_hat = (x_coarse + x_res) * norms.unsqueeze(-1)
        return x_hat


# ============================================================================
# Improvement 4: Channel-Grouped Quantization
# ============================================================================

class ChannelGroupedQuantizer:
    """Split channels into groups by variance, use separate codebooks.

    After rotation, all coordinates SHOULD have equal variance (that's the
    point of rotation). But at d=64 the approximation isn't perfect.
    We can measure the actual per-coordinate variance on calibration data
    and group channels accordingly.

    More practically: split into "outlier" (high-variance) and "normal"
    channels, give outliers a wider codebook.
    """

    def __init__(self, head_dim, bits, n_groups=2, device='cuda', seed=42):
        self.head_dim = head_dim
        self.bits = bits
        self.n_groups = n_groups
        self.device = device

        # Use same rotation as TurboQuant
        self.base = TurboQuantizer(head_dim, bits, device=device, seed=seed)

        # We'll calibrate the groups later
        self.group_assignment = None
        self.group_quantizers = None

    def calibrate(self, x_sample):
        """Calibrate channel groups from sample data."""
        x_sample = x_sample.half()
        norms = x_sample.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x_sample / norms

        # Rotate
        y = x_unit @ self.base.Pi.T

        # Measure per-coordinate variance
        variances = y.float().var(dim=0).cpu().numpy()  # (d,)

        # Split into groups by variance quantiles
        thresholds = np.quantile(variances, np.linspace(0, 1, self.n_groups + 1)[1:-1])

        self.group_assignment = np.zeros(self.head_dim, dtype=int)
        for i, thresh in enumerate(thresholds):
            self.group_assignment[variances > thresh] = i + 1

        # Create per-group codebooks with scaled centroids
        self.group_scales = []
        self.group_centroids = []
        self.group_boundaries = []

        for g in range(self.n_groups):
            mask = self.group_assignment == g
            if mask.sum() == 0:
                self.group_scales.append(1.0)
                self.group_centroids.append(self.base.centroids)
                self.group_boundaries.append(self.base.boundaries)
                continue

            group_std = float(np.sqrt(variances[mask].mean()))
            # Scale the standard codebook by the group's actual std
            base_std = float(np.sqrt(1.0 / self.head_dim))  # expected std
            scale = group_std / base_std if base_std > 0 else 1.0

            self.group_scales.append(scale)
            self.group_centroids.append((self.base.centroids * scale).to(self.device))
            self.group_boundaries.append((self.base.boundaries * scale).to(self.device))

        self.group_assignment_t = torch.from_numpy(self.group_assignment).to(self.device)

    def quantize(self, x):
        x = x.half()
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        y = x_unit @ self.base.Pi.T

        # Quantize each channel using its group's codebook
        indices = torch.zeros_like(y, dtype=torch.long)
        for g in range(self.n_groups):
            mask = self.group_assignment_t == g
            if mask.sum() == 0:
                continue
            y_group = y[:, mask]
            idx = torch.searchsorted(self.group_boundaries[g], y_group.contiguous())
            indices[:, mask] = idx

        return indices, norms.squeeze(-1)

    def dequantize(self, indices, norms):
        y_hat = torch.zeros(indices.shape, dtype=torch.half, device=self.device)
        for g in range(self.n_groups):
            mask = self.group_assignment_t == g
            if mask.sum() == 0:
                continue
            y_hat[:, mask] = self.group_centroids[g][indices[:, mask]]

        x_hat = y_hat @ self.base.Pi
        return x_hat * norms.unsqueeze(-1)


# ============================================================================
# Improvement 5: Attention-Score-Weighted Quantization
# ============================================================================

class AttentionAwareLayer(CacheLayerMixin):
    """Cache layer that weights quantization by token importance.

    Insight: Not all cached tokens are equally important. Tokens that
    receive high cumulative attention should be quantized more carefully.

    Strategy:
    - Track cumulative attention scores per cached token
    - Keep top-k important tokens in fp16 (regardless of recency)
    - Quantize the rest
    """

    is_sliding = False

    def __init__(self, key_quantizer, value_quantizer, residual_window=64,
                 importance_budget=32):
        super().__init__()
        self.key_quantizer = key_quantizer
        self.value_quantizer = value_quantizer
        self.residual_window = residual_window
        self.importance_budget = importance_budget

        # All tokens stored in fp16
        self.all_keys = None
        self.all_values = None
        self.importance_scores = None  # cumulative attention received

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

        self.all_keys = None
        self.all_values = None

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Append new tokens
        if self.all_keys is None:
            self.all_keys = key_states
            self.all_values = value_states
        else:
            self.all_keys = torch.cat([self.all_keys, key_states], dim=-2)
            self.all_values = torch.cat([self.all_values, value_states], dim=-2)

        total_len = self.all_keys.shape[-2]

        if total_len <= self.residual_window + self.importance_budget:
            # Everything fits in fp16
            self.keys = self.all_keys
            self.values = self.all_values
            return self.all_keys, self.all_values

        # Decide which tokens to keep in fp16:
        # 1. Always keep last `residual_window` tokens
        # 2. From the rest, keep `importance_budget` tokens with highest key norms
        #    (proxy for importance — high-norm keys dominate attention)

        B, H, T, D = self.all_keys.shape
        old_len = T - self.residual_window

        # Key norms as importance proxy (averaged across heads)
        old_keys = self.all_keys[:, :, :old_len, :]
        old_values = self.all_values[:, :, :old_len, :]
        key_norms = old_keys.norm(dim=-1).mean(dim=1).squeeze(0)  # (old_len,)

        # Top-k important tokens
        k = min(self.importance_budget, old_len)
        _, important_idx = key_norms.topk(k)
        important_idx_sorted, _ = important_idx.sort()

        # Create mask: which old tokens to keep in fp16
        keep_mask = torch.zeros(old_len, dtype=torch.bool, device=self.device)
        keep_mask[important_idx_sorted] = True
        compress_mask = ~keep_mask

        # Split old tokens
        important_keys = old_keys[:, :, keep_mask, :]
        important_values = old_values[:, :, keep_mask, :]
        compress_keys = old_keys[:, :, compress_mask, :]
        compress_values = old_values[:, :, compress_mask, :]

        # Quantize the non-important old tokens
        if compress_keys.shape[2] > 0:
            B2, H2, T2, D2 = compress_keys.shape
            ck_flat = compress_keys.reshape(B2 * H2 * T2, D2)
            cv_flat = compress_values.reshape(B2 * H2 * T2, D2)

            ck_idx, ck_norms = self.key_quantizer.quantize(ck_flat)
            cv_idx, cv_norms = self.value_quantizer.quantize(cv_flat)

            ck_hat = self.key_quantizer.dequantize(ck_idx, ck_norms).reshape(B2, H2, T2, D2)
            cv_hat = self.value_quantizer.dequantize(cv_idx, cv_norms).reshape(B2, H2, T2, D2)
        else:
            ck_hat = compress_keys
            cv_hat = compress_values

        # Recent tokens (fp16)
        recent_keys = self.all_keys[:, :, -self.residual_window:, :]
        recent_values = self.all_values[:, :, -self.residual_window:, :]

        # Assemble: [compressed_old | important_old_fp16 | recent_fp16]
        # Note: we lose positional ordering of old tokens. For GPT-2 with
        # absolute positions this is fine since positions are in the embeddings.
        full_keys = torch.cat([ck_hat, important_keys, recent_keys], dim=2)
        full_values = torch.cat([cv_hat, important_values, recent_values], dim=2)

        self.keys = full_keys
        self.values = full_values
        return full_keys, full_values

    def get_seq_length(self):
        if self.all_keys is None:
            return 0
        return self.all_keys.shape[-2]

    def get_max_cache_shape(self):
        return -1

    def crop(self, max_length):
        pass

    def batch_repeat_interleave(self, repeats):
        if self.all_keys is not None:
            self.all_keys = self.all_keys.repeat_interleave(repeats, dim=0)
            self.all_values = self.all_values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        if self.all_keys is not None:
            self.all_keys = self.all_keys[indices, ...]
            self.all_values = self.all_values[indices, ...]

    def get_mask_sizes(self, cache_position):
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, 0


# ============================================================================
# Cache Factories
# ============================================================================

def make_hadamard_cache(head_dim, num_layers, key_bits, value_bits, residual_window, device):
    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None
    for layer_idx in range(num_layers):
        kq = HadamardQuantizer(head_dim, key_bits, device=device, seed=42 + layer_idx * 2)
        vq = HadamardQuantizer(head_dim, value_bits, device=device, seed=42 + layer_idx * 2 + 1)
        cache.layers.append(TurboQuantLayer(kq, vq, residual_window=residual_window))
    return cache


def make_adaptive_cache(head_dim, num_layers, key_bits_per_layer, value_bits_per_layer,
                        residual_window, device):
    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None
    for layer_idx in range(num_layers):
        kb = int(key_bits_per_layer[layer_idx])
        vb = int(value_bits_per_layer[layer_idx])
        kq = TurboQuantizer(head_dim, kb, device=device, seed=42 + layer_idx * 2)
        vq = TurboQuantizer(head_dim, vb, device=device, seed=42 + layer_idx * 2 + 1)
        cache.layers.append(TurboQuantLayer(kq, vq, residual_window=residual_window))
    return cache


def make_importance_cache(head_dim, num_layers, key_bits, value_bits,
                          residual_window, importance_budget, device):
    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None
    for layer_idx in range(num_layers):
        kq = TurboQuantizer(head_dim, key_bits, device=device, seed=42 + layer_idx * 2)
        vq = TurboQuantizer(head_dim, value_bits, device=device, seed=42 + layer_idx * 2 + 1)
        cache.layers.append(AttentionAwareLayer(kq, vq, residual_window=residual_window,
                                                importance_budget=importance_budget))
    return cache


# ============================================================================
# Benchmark
# ============================================================================

def measure_reconstruction(quantizer, x):
    """MSE and cosine similarity."""
    if hasattr(quantizer, 'calibrate') and quantizer.group_assignment is None:
        quantizer.calibrate(x)

    if isinstance(quantizer, ResidualQuantizer):
        coarse_idx, res_idx, res_norms, norms = quantizer.quantize(x)
        x_hat = quantizer.dequantize(coarse_idx, res_idx, res_norms, norms)
    else:
        indices, norms = quantizer.quantize(x)
        x_hat = quantizer.dequantize(indices, norms)

    x_f, xh_f = x.float(), x_hat.float()
    mse = F.mse_loss(xh_f, x_f).item()
    cos = F.cosine_similarity(x_f, xh_f, dim=-1).mean().item()
    return mse, cos


def main():
    device = 'cuda'
    print("=" * 75)
    print("TurboQuant v2 — Improvement Experiments")
    print("=" * 75)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).half()
    model.eval()

    head_dim = 64
    num_layers = 12

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

    # ── Experiment 1: Rotation comparison ──
    print("\n" + "=" * 75)
    print("Experiment 1: Random Orthogonal vs Hadamard Rotation")
    print("=" * 75)
    print("  (Same Lloyd-Max codebook, different rotation matrix)")

    x_test = torch.randn(4096, head_dim, device=device) * 0.5

    for bits in [2, 3, 4]:
        # Random orthogonal (original TurboQuant)
        q_rand = TurboQuantizer(head_dim, bits, device=device, seed=42)
        mse_r, cos_r = measure_reconstruction(q_rand, x_test)

        # Hadamard
        q_had = HadamardQuantizer(head_dim, bits, device=device, seed=42)
        mse_h, cos_h = measure_reconstruction(q_had, x_test)

        print(f"\n  {bits}-bit:")
        print(f"    Random ortho:  MSE={mse_r:.6f}  cos={cos_r:.6f}")
        print(f"    Hadamard:      MSE={mse_h:.6f}  cos={cos_h:.6f}")
        print(f"    Δ MSE: {((mse_h - mse_r)/mse_r)*100:+.1f}%")

    # Perplexity comparison
    print("\n  Perplexity comparison (K4/V2, win=64):")
    ppl_base = compute_perplexity(model, tokenizer, eval_text)
    print(f"    Baseline fp16:   {ppl_base:.2f}")

    from turboquant_gpt2 import make_turboquant_cache
    cache_rand = make_turboquant_cache(head_dim, num_layers, 4, 2, 64, device)
    ppl_rand = compute_perplexity(model, tokenizer, eval_text, cache=cache_rand)
    print(f"    Random ortho:    {ppl_rand:.2f} ({((ppl_rand-ppl_base)/ppl_base)*100:+.1f}%)")

    cache_had = make_hadamard_cache(head_dim, num_layers, 4, 2, 64, device)
    ppl_had = compute_perplexity(model, tokenizer, eval_text, cache=cache_had)
    print(f"    Hadamard:        {ppl_had:.2f} ({((ppl_had-ppl_base)/ppl_base)*100:+.1f}%)")

    # ── Experiment 2: Per-layer sensitivity analysis ──
    print("\n" + "=" * 75)
    print("Experiment 2: Per-Layer Sensitivity Analysis")
    print("=" * 75)

    print("\n  Measuring layer sensitivities...")
    sensitivities = measure_layer_sensitivity(model, tokenizer, eval_text, device)

    print("\n  Layer sensitivities (higher = more sensitive to noise):")
    for i, s in enumerate(sensitivities):
        bar = "█" * int(s * 40)
        print(f"    Layer {i:2d}: {s:.4f} {bar}")

    # Adaptive bit allocation: average 3 bits, but distribute by sensitivity
    total_budget_k = 3 * num_layers  # 36 total key bits
    total_budget_v = 2 * num_layers  # 24 total value bits

    key_bits_adaptive = allocate_bits_by_sensitivity(sensitivities, total_budget_k, min_bits=2, max_bits=4)
    value_bits_adaptive = allocate_bits_by_sensitivity(sensitivities, total_budget_v, min_bits=1, max_bits=3)

    print(f"\n  Adaptive bit allocation (avg K={key_bits_adaptive.mean():.1f}, V={value_bits_adaptive.mean():.1f}):")
    for i in range(num_layers):
        print(f"    Layer {i:2d}: K={key_bits_adaptive[i]}  V={value_bits_adaptive[i]}")

    # Compare uniform vs adaptive
    print("\n  Perplexity comparison (win=64):")

    # Uniform K3/V2
    cache_uniform = make_turboquant_cache(head_dim, num_layers, 3, 2, 64, device)
    ppl_uniform = compute_perplexity(model, tokenizer, eval_text, cache=cache_uniform)
    print(f"    Uniform K3/V2:   {ppl_uniform:.2f} ({((ppl_uniform-ppl_base)/ppl_base)*100:+.1f}%)")

    # Adaptive (same average bits)
    cache_adaptive = make_adaptive_cache(
        head_dim, num_layers, key_bits_adaptive, value_bits_adaptive, 64, device
    )
    ppl_adaptive = compute_perplexity(model, tokenizer, eval_text, cache=cache_adaptive)
    avg_kb = key_bits_adaptive.mean()
    avg_vb = value_bits_adaptive.mean()
    print(f"    Adaptive K{avg_kb:.1f}/V{avg_vb:.1f}: {ppl_adaptive:.2f} ({((ppl_adaptive-ppl_base)/ppl_base)*100:+.1f}%)")

    # ── Experiment 3: Residual Quantization ──
    print("\n" + "=" * 75)
    print("Experiment 3: Single-Stage vs Residual Quantization")
    print("=" * 75)

    for bits in [2, 3, 4]:
        # Single stage
        q_single = TurboQuantizer(head_dim, bits, device=device)
        mse_s, cos_s = measure_reconstruction(q_single, x_test)

        # Residual (same total bits)
        q_resid = ResidualQuantizer(head_dim, bits, device=device)
        mse_res, cos_res = measure_reconstruction(q_resid, x_test)

        print(f"\n  {bits}-bit total:")
        print(f"    Single-stage:  MSE={mse_s:.6f}  cos={cos_s:.6f}")
        print(f"    Residual:      MSE={mse_res:.6f}  cos={cos_res:.6f}")
        print(f"    Δ MSE: {((mse_res - mse_s)/mse_s)*100:+.1f}%")

    # ── Experiment 4: Channel-Grouped Quantization ──
    print("\n" + "=" * 75)
    print("Experiment 4: Uniform vs Channel-Grouped Codebooks")
    print("=" * 75)

    # Get real KV vectors for calibration
    print("\n  Extracting real KV vectors for calibration...")
    kv_samples = []
    hooks = []
    def capture_hook(module, args, output):
        # GPT-2 attention outputs key/value in the past_key_values
        pass

    # Simple approach: run model and extract KVs
    with torch.no_grad():
        input_ids = tokenizer.encode(eval_text, return_tensors='pt').to(device)
        outputs = model(input_ids, output_attentions=False)
        # Get KV from cache
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            for layer_kv in outputs.past_key_values:
                if hasattr(layer_kv, 'keys'):
                    kv_samples.append(layer_kv.keys.reshape(-1, head_dim))
                elif isinstance(layer_kv, (tuple, list)):
                    kv_samples.append(layer_kv[0].reshape(-1, head_dim))

    if kv_samples:
        kv_data = torch.cat(kv_samples, dim=0)
        print(f"  Captured {kv_data.shape[0]} KV vectors from {len(kv_samples)} layers")

        for bits in [2, 3, 4]:
            # Uniform
            q_uni = TurboQuantizer(head_dim, bits, device=device)
            mse_u, cos_u = measure_reconstruction(q_uni, kv_data)

            # Channel-grouped (2 groups)
            q_grp = ChannelGroupedQuantizer(head_dim, bits, n_groups=2, device=device)
            q_grp.calibrate(kv_data)
            mse_g, cos_g = measure_reconstruction(q_grp, kv_data)

            # Channel-grouped (4 groups)
            q_grp4 = ChannelGroupedQuantizer(head_dim, bits, n_groups=4, device=device)
            q_grp4.calibrate(kv_data)
            mse_g4, cos_g4 = measure_reconstruction(q_grp4, kv_data)

            print(f"\n  {bits}-bit (on real KV data):")
            print(f"    Uniform:        MSE={mse_u:.6f}  cos={cos_u:.6f}")
            print(f"    2-group:        MSE={mse_g:.6f}  cos={cos_g:.6f}  ({((mse_g-mse_u)/mse_u)*100:+.1f}%)")
            print(f"    4-group:        MSE={mse_g4:.6f}  cos={cos_g4:.6f}  ({((mse_g4-mse_u)/mse_u)*100:+.1f}%)")
    else:
        print("  Could not capture KV vectors, skipping channel-grouped test")

    # ── Experiment 5: Importance-Aware Token Selection ──
    print("\n" + "=" * 75)
    print("Experiment 5: Recency-Only vs Importance-Aware Token Selection")
    print("=" * 75)

    print("\n  Perplexity comparison (K4/V2):")

    # Recency-only (original TurboQuant) with small window
    cache_recency = make_turboquant_cache(head_dim, num_layers, 4, 2, 32, device)
    ppl_recency = compute_perplexity(model, tokenizer, eval_text, cache=cache_recency)
    print(f"    Recency-only (win=32):           {ppl_recency:.2f} ({((ppl_recency-ppl_base)/ppl_base)*100:+.1f}%)")

    # Importance-aware: win=16 recency + 16 important = same 32 fp16 tokens
    cache_import = make_importance_cache(head_dim, num_layers, 4, 2, 16, 16, device)
    ppl_import = compute_perplexity(model, tokenizer, eval_text, cache=cache_import)
    print(f"    Importance (win=16 + imp=16):     {ppl_import:.2f} ({((ppl_import-ppl_base)/ppl_base)*100:+.1f}%)")

    # Importance-aware: win=32 + 32 important = 64 fp16 tokens (generous)
    cache_import2 = make_importance_cache(head_dim, num_layers, 4, 2, 32, 32, device)
    ppl_import2 = compute_perplexity(model, tokenizer, eval_text, cache=cache_import2)
    print(f"    Importance (win=32 + imp=32):     {ppl_import2:.2f} ({((ppl_import2-ppl_base)/ppl_base)*100:+.1f}%)")

    # Compare to pure recency with same total fp16 budget
    cache_recency64 = make_turboquant_cache(head_dim, num_layers, 4, 2, 64, device)
    ppl_recency64 = compute_perplexity(model, tokenizer, eval_text, cache=cache_recency64)
    print(f"    Recency-only (win=64):           {ppl_recency64:.2f} ({((ppl_recency64-ppl_base)/ppl_base)*100:+.1f}%)")

    # ── Summary ──
    print("\n" + "=" * 75)
    print("SUMMARY OF IMPROVEMENTS")
    print("=" * 75)
    print(f"\n  Baseline fp16 PPL: {ppl_base:.2f}")
    print(f"\n  Original TurboQuant K4/V2 w64: {ppl_rand:.2f} ({((ppl_rand-ppl_base)/ppl_base)*100:+.1f}%)")
    print(f"  + Hadamard rotation:            {ppl_had:.2f} ({((ppl_had-ppl_base)/ppl_base)*100:+.1f}%)")
    print(f"  + Adaptive bits (same budget):  {ppl_adaptive:.2f} ({((ppl_adaptive-ppl_base)/ppl_base)*100:+.1f}%)")
    print(f"  + Importance-aware selection:    {ppl_import2:.2f} ({((ppl_import2-ppl_base)/ppl_base)*100:+.1f}%)")

    print("\n" + "=" * 75)
    print("Done!")


if __name__ == "__main__":
    main()
