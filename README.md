# TurboQuant-Adaptive: Sink-Aware & Sensitivity-Tiered KV-Cache Quantization

**TL;DR**: We implement Google's [TurboQuant](https://arxiv.org/abs/2504.19874) from scratch in PyTorch, then systematically test improvements. By combining attention-sink protection with per-layer sensitivity-tiered bit allocation, we cut the quality degradation from **+2.4% to +1.1% perplexity** on GPT-2 at comparable compression ratios.

## What is TurboQuant?

TurboQuant (Google, ICLR 2026) compresses the key-value cache during LLM inference by:

1. Saving the L2 norm of each KV vector
2. Rotating it by a random orthogonal matrix (Haar-distributed)
3. Applying per-coordinate Lloyd-Max scalar quantization (optimal for the resulting Beta distribution)
4. On read: lookup centroids, inverse-rotate, rescale

It's training-free, data-free, and achieves ~3-4x KV cache compression with minimal quality loss.

## What We Did

We implemented TurboQuant from scratch and ran a systematic experimental campaign to understand where quantization error actually hurts. We tested **40+ configurations** across 5 improvement strategies.

### The Attention Sink Discovery

By analyzing GPT-2's attention patterns, we found that **token 0 absorbs 35-66% of all attention mass** in layers 3-11. This is the well-documented [attention sink phenomenon](https://arxiv.org/abs/2309.17453) — the model uses early tokens as "attention parking lots" for unused attention mass.

```
Layer  0: token 0 gets  2.8% attention (uniform-ish)
Layer  5: token 0 gets 59.2% attention (MASSIVE sink)
Layer  7: token 0 gets 65.9% attention
Layer 11: token 0 gets 48.1% attention
```

Quantizing these tokens — even at 4-bit — introduces error that propagates through every attention computation. Keeping just 4-8 sink tokens in fp16 costs ~0.3% of cache memory but removes the biggest error source.

### Per-Layer Sensitivity Analysis

We measured each layer's sensitivity to quantization noise by injecting calibrated perturbations and measuring logit changes:

```
Layer  0: 1.0000 ████████████████████████████████████████  <- 5-25x more sensitive!
Layer  1: 0.1907 ███████
Layer  2: 0.1828 ███████
Layer  4: 0.2026 ████████
Layer  5: 0.2006 ████████
Layer  3: 0.0406 █
Layer  6: 0.0315 █
...
Layer 11: 0.0073
```

Layer 0 is dramatically more sensitive than all others. This motivates giving it more bits.

### The Winning Recipe

| Layer | Config | Why |
|-------|--------|-----|
| Layer 0 | K4/V4 | 5-25x more sensitive to noise |
| Layers 1,2,4,5 | K4/V2 | Medium sensitivity |
| Layers 3,6-11 | K3/V2 | Low sensitivity (K3 floor!) |
| All layers | First 8 tokens in fp16 | Attention sinks |
| All layers | Last 32 tokens in fp16 | Residual window |

## Results

### Final Comparison

| Strategy | PPL | Delta | Notes |
|----------|-----|-------|-------|
| Baseline fp16 | 5.26 | — | No compression |
| **Sink(8) + Adaptive tiers** | **5.32** | **+1.1%** | Our best |
| Sink(4) + Adaptive tiers | 5.33 | +1.3% | Close second |
| Vanilla TurboQuant K4/V2 | 5.39 | +2.4% | Published algorithm |
| Vanilla K3/V2 | 5.91 | +12.4% | Too aggressive |

### What We Tested That Didn't Work

| Idea | Result | Why |
|------|--------|-----|
| Hadamard rotation | +7.1% PPL (worse) | Structure interacts poorly with learned representations |
| Residual quantization | +11-31% MSE (worse) | Splitting 3-4 bits into stages loses more than it gains |
| Channel grouping | -0.3% MSE (negligible) | Rotation already equalizes variance |
| Importance by key norm | +117% PPL (catastrophic) | Reordering breaks positional encoding |
| Dual-resolution storage | +671% PPL (catastrophic) | 2-bit fallback poisons attention scores |
| Progressive aging | +497% PPL (catastrophic) | Re-quantizing compounds noise multiplicatively |
| Naive adaptive (K2 floor) | +27.6% PPL (worse) | 2-bit keys hit a quality cliff |

### Key Insight: The 2-Bit Cliff

Going below 3-bit keys causes a discontinuous quality collapse. At head_dim=64:

| Key Bits | Cosine Similarity | PPL Impact |
|----------|------------------|------------|
| K4 | 0.996 | +2-3% |
| K3 | 0.984 | +5-15% |
| **K2** | **0.942** | **+24-75%** |
| K1 | 0.801 | +130-530% |

The jump from K3 to K2 is catastrophic because softmax exponentially amplifies small attention score errors.

## Reconstruction Quality

The Lloyd-Max codebook (computed with the exact Beta PDF for d=64):

| Bits | MSE | Cosine Sim | Compression vs fp16 |
|------|-----|-----------|---------------------|
| 1-bit | 0.090 | 0.801 | 16x |
| 2-bit | 0.029 | 0.942 | 8x |
| 3-bit | 0.008 | 0.984 | 5.3x |
| 4-bit | 0.002 | 0.996 | 4x |

## Usage

### Quick Start

```bash
pip install torch transformers scipy

# Run the full benchmark
python turboquant_gpt2.py

# Run improvement experiments
python turboquant_dynamic.py

# Generate Pareto frontier plots
python turboquant_plot.py
```

### Using the Quantizer Directly

```python
from turboquant_gpt2 import TurboQuantizer

# Create a 4-bit quantizer for head_dim=64
q = TurboQuantizer(head_dim=64, bits=4, device='cuda')

# Quantize
indices, norms = q.quantize(key_vectors)  # key_vectors: (..., 64)

# Dequantize
key_hat = q.dequantize(indices, norms)
```

### Using the Adaptive Cache with HuggingFace

```python
from turboquant_dynamic import make_sink_cache

# Create sink-aware + sensitivity-tiered cache
tiers = {
    0: (4, 4),                          # Layer 0: K4/V4
    **{i: (4, 2) for i in [1,2,4,5]},   # High-sens: K4/V2
    **{i: (3, 2) for i in [3,6,7,8,9,10,11]},  # Low-sens: K3/V2
}
cache = make_sink_cache(
    head_dim=64, num_layers=12,
    key_bits=4, value_bits=2,
    num_sinks=8, residual_window=32,
    device='cuda', sensitivity_tiers=tiers
)

# Use with any HuggingFace model
outputs = model(input_ids, past_key_values=cache)
```

## Files

| File | Description |
|------|-------------|
| `turboquant_gpt2.py` | Core TurboQuant implementation + GPT-2 evaluation |
| `turboquant_benchmark.py` | Full 40-config Pareto frontier sweep |
| `turboquant_plot.py` | Pareto frontier visualization from benchmark data |
| `turboquant_v2.py` | Failed improvement experiments (Hadamard, residual, etc.) |
| `turboquant_v3.py` | Layer-0 protection + smart adaptive + progressive aging |
| `turboquant_dynamic.py` | Sink-aware + sensitivity-tiered + dual-resolution experiments |

## Prior Work & Credit

The individual techniques we combine are established in the literature:

- **TurboQuant** (Zandieh et al., ICLR 2026) — the base rotation + Lloyd-Max quantizer
- **Attention sinks** (Xiao et al., StreamingLLM 2023) — discovered the sink phenomenon
- **KVQuant** (Hooper et al., NeurIPS 2024) — first to protect sink tokens from quantization
- **KVTuner** (ICML 2025) — per-layer sensitivity-aware mixed-precision KV cache
- **RotateKV** (IJCAI 2025) — combines rotation with sink-aware quantization
- **KIVI** (ICML 2024) — asymmetric key/value quantization

Our contribution is the **systematic empirical study** of what works and what doesn't when building on TurboQuant, and a concrete recipe that composes these ideas effectively. We make no novelty claims on the individual techniques.

## Hardware

All experiments ran on a single NVIDIA RTX 3090 with GPT-2 (124M parameters).

## License

MIT
