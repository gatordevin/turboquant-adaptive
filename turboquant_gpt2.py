"""
TurboQuant KV-Cache Quantization for GPT-2
===========================================
Implementation of Google's TurboQuant (arXiv:2504.19874) for compressing
the key-value cache during LLM inference.

Algorithm:
1. Save L2 norm of each KV vector
2. Rotate unit vector by random orthogonal matrix (Haar-distributed)
3. Per-coordinate Lloyd-Max scalar quantization (optimal for the resulting Beta distribution)
4. On read: lookup centroids, inverse-rotate, rescale

We skip QJL (residual correction for unbiased inner products) because
community experiments show MSE-only reconstruction gives better generation
quality — softmax exponentially amplifies QJL's variance.
"""

import math
import time
import torch
import torch.nn.functional as F
from scipy import integrate
from scipy.special import gammaln
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer, CacheLayerMixin


# ============================================================================
# Lloyd-Max Codebook
# ============================================================================

def beta_pdf(x, d):
    """PDF of each coordinate after Haar rotation of a unit vector in R^d."""
    if abs(x) >= 1.0:
        return 0.0
    log_norm = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    log_val = log_norm + ((d - 3) / 2) * np.log(1 - x * x)
    return np.exp(log_val)


def gaussian_pdf(x, d):
    """Gaussian approximation N(0, 1/d) — valid for d >= 64."""
    sigma2 = 1.0 / d
    return np.exp(-x * x / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)


def compute_lloyd_max_codebook(d, bits, use_exact_pdf=False, max_iter=300, tol=1e-10):
    """Compute optimal Lloyd-Max scalar quantizer for the post-rotation distribution."""
    n_levels = 2 ** bits
    pdf = lambda x: beta_pdf(x, d) if use_exact_pdf else gaussian_pdf(x, d)

    sigma = 1.0 / np.sqrt(d)
    lo, hi = -4 * sigma, 4 * sigma

    # Initialize centroids at quantile midpoints via numerical CDF
    x_grid = np.linspace(lo, hi, 10000)
    cdf_vals = np.array([integrate.quad(pdf, lo, x)[0] for x in x_grid])
    cdf_vals = cdf_vals / cdf_vals[-1]

    quantiles = np.linspace(0, 1, n_levels + 1)
    quantile_midpoints = (quantiles[:-1] + quantiles[1:]) / 2
    centroids = np.interp(quantile_midpoints, cdf_vals, x_grid)

    for _ in range(max_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        b_ext = np.concatenate([[lo], boundaries, [hi]])

        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            a, b = b_ext[i], b_ext[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids[i] = num / den if den > 1e-15 else (a + b) / 2

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return centroids.astype(np.float32), boundaries.astype(np.float32)


# ============================================================================
# TurboQuant Core
# ============================================================================

class TurboQuantizer:
    """TurboQuant MSE quantizer for a single bit-width and dimension."""

    def __init__(self, head_dim, bits, device='cuda', seed=42, use_exact_pdf=False):
        self.head_dim = head_dim
        self.bits = bits
        self.n_levels = 2 ** bits
        self.device = device

        centroids_np, boundaries_np = compute_lloyd_max_codebook(
            head_dim, bits, use_exact_pdf=use_exact_pdf
        )
        self.centroids = torch.from_numpy(centroids_np).half().to(device)
        self.boundaries = torch.from_numpy(boundaries_np).half().to(device)

        # Haar-distributed random orthogonal matrix
        rng = torch.Generator().manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=rng)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        self.Pi = (Q * diag_sign.unsqueeze(0)).half().to(device)

    def quantize(self, x):
        """Quantize vectors. x: (..., head_dim) -> indices, norms"""
        x = x.half()
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        y = x_unit @ self.Pi.T
        indices = torch.searchsorted(self.boundaries, y.contiguous())
        return indices, norms.squeeze(-1)

    def dequantize(self, indices, norms):
        """Dequantize back. indices: (..., head_dim), norms: (...) -> x_hat"""
        y_hat = self.centroids[indices]
        x_hat = y_hat @ self.Pi
        x_hat = x_hat * norms.unsqueeze(-1)
        return x_hat


# ============================================================================
# TurboQuant Cache Layer (plugs into transformers DynamicCache)
# ============================================================================

class TurboQuantLayer(CacheLayerMixin):
    """A cache layer that compresses KV using TurboQuant.

    Keeps the most recent `residual_window` tokens in fp16 and compresses
    older tokens using rotation + Lloyd-Max quantization.
    """

    is_sliding = False

    def __init__(self, key_quantizer, value_quantizer, residual_window=64):
        super().__init__()
        self.key_quantizer = key_quantizer
        self.value_quantizer = value_quantizer
        self.residual_window = residual_window

        # Compressed storage
        self.compressed_k_indices = []
        self.compressed_k_norms = []
        self.compressed_v_indices = []
        self.compressed_v_norms = []
        self.compressed_shapes = []  # (B, H, T) per chunk

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        """Append new KV, compress overflow beyond residual window."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Append new tokens to recent buffer
        if self.keys.numel() == 0:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)

        recent_len = self.keys.shape[-2]
        if recent_len > self.residual_window:
            overflow = recent_len - self.residual_window

            # Extract overflow tokens: (B, H, overflow, D)
            k_overflow = self.keys[:, :, :overflow, :]
            v_overflow = self.values[:, :, :overflow, :]

            # Keep only residual window
            self.keys = self.keys[:, :, overflow:, :].contiguous()
            self.values = self.values[:, :, overflow:, :].contiguous()

            # Compress
            B, H, T, D = k_overflow.shape
            k_flat = k_overflow.reshape(B * H * T, D)
            v_flat = v_overflow.reshape(B * H * T, D)

            k_idx, k_norms = self.key_quantizer.quantize(k_flat)
            v_idx, v_norms = self.value_quantizer.quantize(v_flat)

            self.compressed_k_indices.append(k_idx)
            self.compressed_k_norms.append(k_norms)
            self.compressed_v_indices.append(v_idx)
            self.compressed_v_norms.append(v_norms)
            self.compressed_shapes.append((B, H, T))

        # Reconstruct full KV for attention
        all_keys = []
        all_values = []

        for i, (B, H, T) in enumerate(self.compressed_shapes):
            k_hat = self.key_quantizer.dequantize(
                self.compressed_k_indices[i], self.compressed_k_norms[i]
            ).reshape(B, H, T, -1)
            v_hat = self.value_quantizer.dequantize(
                self.compressed_v_indices[i], self.compressed_v_norms[i]
            ).reshape(B, H, T, -1)
            all_keys.append(k_hat)
            all_values.append(v_hat)

        all_keys.append(self.keys)
        all_values.append(self.values)

        full_keys = torch.cat(all_keys, dim=-2)
        full_values = torch.cat(all_values, dim=-2)

        return full_keys, full_values

    def get_seq_length(self):
        if not self.is_initialized or self.keys.numel() == 0:
            compressed_len = sum(s[2] for s in self.compressed_shapes)
            return compressed_len
        compressed_len = sum(s[2] for s in self.compressed_shapes)
        return compressed_len + self.keys.shape[-2]

    def get_max_cache_shape(self):
        return -1

    def crop(self, max_length):
        pass  # Not needed for this demo

    def batch_repeat_interleave(self, repeats):
        if self.keys.numel() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        if self.keys.numel() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]

    def get_mask_sizes(self, cache_position):
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset


def make_turboquant_cache(head_dim=64, num_layers=12, key_bits=4, value_bits=2,
                          residual_window=64, device='cuda'):
    """Create a DynamicCache pre-filled with TurboQuant layers."""
    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None  # prevent auto-creation of DynamicLayer

    for layer_idx in range(num_layers):
        kq = TurboQuantizer(head_dim, key_bits, device=device, seed=42 + layer_idx * 2)
        vq = TurboQuantizer(head_dim, value_bits, device=device, seed=42 + layer_idx * 2 + 1)
        cache.layers.append(TurboQuantLayer(kq, vq, residual_window=residual_window))

    return cache


# ============================================================================
# Evaluation
# ============================================================================

def measure_reconstruction_error(quantizer, x):
    """Measure MSE and cosine similarity of quantize-dequantize roundtrip."""
    indices, norms = quantizer.quantize(x)
    x_hat = quantizer.dequantize(indices, norms)
    x_f, x_hat_f = x.float(), x_hat.float()
    mse = F.mse_loss(x_hat_f, x_f).item()
    cos_sim = F.cosine_similarity(x_f, x_hat_f, dim=-1).mean().item()
    return mse, cos_sim


def compute_perplexity(model, tokenizer, text, cache=None):
    """Compute perplexity on a text string."""
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=cache)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return torch.exp(loss).item()


def generate_text(model, tokenizer, prompt, max_new_tokens=100, cache=None):
    """Generate text with optional TurboQuant cache."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            past_key_values=cache,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print("=" * 70)
    print("TurboQuant KV-Cache Quantization — GPT-2 Test")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # ── Load model ──
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).half()
    model.eval()

    head_dim = model.config.n_embd // model.config.n_head  # 64
    num_layers = model.config.n_layer  # 12
    num_heads = model.config.n_head    # 12
    print(f"  head_dim={head_dim}, num_heads={num_heads}, num_layers={num_layers}")

    # ── Test 1: Reconstruction quality ──
    print("\n" + "=" * 70)
    print("Test 1: Reconstruction Quality (1024 random vectors, d=64)")
    print("=" * 70)

    x_test = torch.randn(1024, head_dim, device=device) * 0.5

    for bits in [1, 2, 3, 4]:
        t0 = time.time()
        q = TurboQuantizer(head_dim, bits, device=device, use_exact_pdf=True)
        t1 = time.time()
        mse, cos_sim = measure_reconstruction_error(q, x_test)
        compression = 16.0 / bits  # vs fp16
        print(f"  {bits}-bit: MSE={mse:.6f}  cos_sim={cos_sim:.6f}  "
              f"compression={compression:.1f}x  codebook={t1-t0:.2f}s")

    # ── Test 2: Perplexity ──
    print("\n" + "=" * 70)
    print("Test 2: Perplexity Comparison")
    print("=" * 70)

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

    print("\n  Baseline (fp16 cache)...")
    t0 = time.time()
    ppl_base = compute_perplexity(model, tokenizer, eval_text)
    print(f"  Baseline:        PPL = {ppl_base:.2f}  ({time.time()-t0:.3f}s)")

    configs = [
        ("K4/V4 win=64",  4, 4, 64),
        ("K4/V2 win=64",  4, 2, 64),
        ("K4/V2 win=32",  4, 2, 32),
        ("K2/V2 win=64",  2, 2, 64),
        ("K2/V2 win=32",  2, 2, 32),
    ]

    for name, kb, vb, win in configs:
        print(f"\n  Precomputing codebooks for {name}...")
        t0 = time.time()
        cache = make_turboquant_cache(
            head_dim=head_dim, num_layers=num_layers,
            key_bits=kb, value_bits=vb,
            residual_window=win, device=device
        )
        t_codebook = time.time() - t0

        t0 = time.time()
        ppl = compute_perplexity(model, tokenizer, eval_text, cache=cache)
        t_inf = time.time() - t0
        delta = ((ppl - ppl_base) / ppl_base) * 100
        print(f"  {name:20s}: PPL = {ppl:.2f} ({delta:+.1f}%)  "
              f"codebook={t_codebook:.1f}s  inference={t_inf:.3f}s")

    # ── Test 3: Memory savings ──
    print("\n" + "=" * 70)
    print("Test 3: Memory Savings (512 tokens)")
    print("=" * 70)

    seq_len = 512
    for name, kb, vb, win in configs:
        compressed_tokens = max(0, seq_len - win)
        recent_tokens = min(seq_len, win)

        fp16_per_token = head_dim * 2 * 2  # K+V, 2 bytes each
        compressed_k = head_dim * kb / 8 + 2  # indices + norm(fp16)
        compressed_v = head_dim * vb / 8 + 2
        compressed_per_token = compressed_k + compressed_v

        total_compressed = (compressed_tokens * compressed_per_token +
                           recent_tokens * fp16_per_token)
        total_fp16 = seq_len * fp16_per_token
        total_compressed *= num_heads * num_layers
        total_fp16 *= num_heads * num_layers

        ratio = total_fp16 / total_compressed
        print(f"  {name:20s}: {total_fp16/1024:.0f} KB -> {total_compressed/1024:.0f} KB "
              f"({ratio:.2f}x compression)")

    # ── Test 4: Generation ──
    print("\n" + "=" * 70)
    print("Test 4: Text Generation")
    print("=" * 70)

    prompt = "The future of artificial intelligence is"
    print(f'\n  Prompt: "{prompt}"\n')

    torch.manual_seed(42)
    print("  [Baseline fp16]:")
    text = generate_text(model, tokenizer, prompt, max_new_tokens=80)
    print(f"  {text}\n")

    torch.manual_seed(42)
    print("  [TurboQuant K4/V2 win=64]:")
    cache = make_turboquant_cache(head_dim=head_dim, num_layers=num_layers,
                                  key_bits=4, value_bits=2, residual_window=64, device=device)
    text = generate_text(model, tokenizer, prompt, max_new_tokens=80, cache=cache)
    print(f"  {text}\n")

    torch.manual_seed(42)
    print("  [TurboQuant K2/V2 win=32]:")
    cache = make_turboquant_cache(head_dim=head_dim, num_layers=num_layers,
                                  key_bits=2, value_bits=2, residual_window=32, device=device)
    text = generate_text(model, tokenizer, prompt, max_new_tokens=80, cache=cache)
    print(f"  {text}\n")

    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
