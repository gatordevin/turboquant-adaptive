"""
Debug: Why does vanilla TurboQuant fail catastrophically on Qwen2.5-1.5B?

Hypotheses:
1. GQA amplification: 2 KV heads serve 12 query heads, 6x error amplification
2. Bug: our rotation matrix is wrong size for head_dim=128
3. Bug: KV shape mismatch with GQA (num_kv_heads != num_attention_heads)
4. RoPE interaction: rotation applied post-RoPE might conflict
5. Larger head_dim=128 should actually quantize BETTER (Gaussian approximation more accurate)

Let's test each systematically.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from turboquant_gpt2 import TurboQuantizer


def main():
    device = 'cuda'
    print("=" * 70)
    print("QWEN2.5-1.5B DEBUG INVESTIGATION")
    print("=" * 70)

    # ── Load model ──
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B').to(device).half()
    model.eval()

    config = model.config
    hidden_size = config.hidden_size     # 1536
    num_heads = config.num_attention_heads  # 12
    kv_heads = config.num_key_value_heads   # 2
    head_dim = hidden_size // num_heads     # 128

    print(f"\n  hidden_size={hidden_size}, num_heads={num_heads}, kv_heads={kv_heads}, head_dim={head_dim}")
    print(f"  GQA ratio: {num_heads // kv_heads}:1 (each KV head serves {num_heads // kv_heads} query heads)")

    eval_text = "The history of artificial intelligence began in antiquity, with myths and stories."
    input_ids = tokenizer.encode(eval_text, return_tensors='pt').to(device)
    seq_len = input_ids.shape[1]
    print(f"  Test sequence: {seq_len} tokens")

    # ── Test 1: Verify KV cache shapes ──
    print("\n" + "=" * 70)
    print("Test 1: KV Cache Shape Verification")
    print("=" * 70)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        kv = outputs.past_key_values

    # Check what the cache looks like
    print(f"  Cache type: {type(kv)}")
    if hasattr(kv, 'layers'):
        layer0 = kv.layers[0]
        print(f"  Layer 0 type: {type(layer0)}")
        if hasattr(layer0, 'keys'):
            k = layer0.keys
            v = layer0.values
            print(f"  Key shape:   {k.shape}")  # expect (B, kv_heads, T, head_dim)
            print(f"  Value shape: {v.shape}")
            print(f"  Key dtype:   {k.dtype}")

            actual_kv_heads = k.shape[1]
            actual_head_dim = k.shape[3]
            print(f"\n  ACTUAL: kv_heads={actual_kv_heads}, head_dim={actual_head_dim}")
            print(f"  CONFIG: kv_heads={kv_heads}, head_dim={head_dim}")

            if actual_kv_heads != kv_heads:
                print(f"  *** MISMATCH: KV heads in cache ({actual_kv_heads}) != config ({kv_heads})!")
            if actual_head_dim != head_dim:
                print(f"  *** MISMATCH: head_dim in cache ({actual_head_dim}) != computed ({head_dim})!")

    # ── Test 2: Quantize actual KV vectors and measure error ──
    print("\n" + "=" * 70)
    print("Test 2: Reconstruction Quality on ACTUAL Qwen KV Vectors")
    print("=" * 70)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        kv = outputs.past_key_values

    layer0 = kv.layers[0]
    actual_keys = layer0.keys     # (1, kv_heads, T, head_dim)
    actual_values = layer0.values

    print(f"  Key stats:   mean={actual_keys.float().mean():.4f}, std={actual_keys.float().std():.4f}, "
          f"max={actual_keys.float().abs().max():.4f}")
    print(f"  Value stats: mean={actual_values.float().mean():.4f}, std={actual_values.float().std():.4f}, "
          f"max={actual_values.float().abs().max():.4f}")

    # Flatten and quantize
    B, H, T, D = actual_keys.shape
    k_flat = actual_keys.reshape(B * H * T, D)
    v_flat = actual_values.reshape(B * H * T, D)

    for bits in [2, 3, 4]:
        q = TurboQuantizer(D, bits, device=device, seed=42)
        k_idx, k_norms = q.quantize(k_flat)
        k_hat = q.dequantize(k_idx, k_norms)

        mse = F.mse_loss(k_hat.float(), k_flat.float()).item()
        cos = F.cosine_similarity(k_flat.float(), k_hat.float(), dim=-1).mean().item()
        print(f"  {bits}-bit keys:   MSE={mse:.6f}  cos_sim={cos:.6f}")

    for bits in [2, 3, 4]:
        q = TurboQuantizer(D, bits, device=device, seed=43)
        v_idx, v_norms = q.quantize(v_flat)
        v_hat = q.dequantize(v_idx, v_norms)

        mse = F.mse_loss(v_hat.float(), v_flat.float()).item()
        cos = F.cosine_similarity(v_flat.float(), v_hat.float(), dim=-1).mean().item()
        print(f"  {bits}-bit values: MSE={mse:.6f}  cos_sim={cos:.6f}")

    # ── Test 3: Does our cache produce correct shapes? ──
    print("\n" + "=" * 70)
    print("Test 3: Does Our QuantizedLayer Produce Correct Shapes?")
    print("=" * 70)

    from experiments.exp07_larger_models import QuantizedLayer, get_quantizer

    kq = get_quantizer(D, 4, device, seed=42)
    vq = get_quantizer(D, 2, device, seed=43)
    layer = QuantizedLayer(kq, vq, num_sinks=0, residual_window=8)

    # Simulate feeding KV through our cache layer
    test_k = actual_keys.clone()
    test_v = actual_values.clone()
    print(f"  Input K shape: {test_k.shape}")

    out_k, out_v = layer.update(test_k, test_v)
    print(f"  Output K shape: {out_k.shape}")
    print(f"  Output V shape: {out_v.shape}")

    if out_k.shape != test_k.shape:
        print(f"  *** SHAPE MISMATCH! Input {test_k.shape} → Output {out_k.shape}")
    else:
        print(f"  Shapes match: OK")

    # Check reconstruction error through the cache
    k_error = F.mse_loss(out_k.float(), test_k.float()).item()
    print(f"  Cache roundtrip K MSE: {k_error:.6f}")
    print(f"  (Should be 0 since all {seq_len} tokens fit in residual_window=8? "
          f"residual_window={8}, seq_len={seq_len})")

    # Now with a small window to force compression
    layer2 = QuantizedLayer(kq, vq, num_sinks=0, residual_window=4)
    out_k2, out_v2 = layer2.update(test_k, test_v)
    k_error2 = F.mse_loss(out_k2.float(), test_k.float()).item()
    cos2 = F.cosine_similarity(out_k2.reshape(-1, D).float(), test_k.reshape(-1, D).float(), dim=-1).mean().item()
    print(f"  Cache with win=4: K MSE={k_error2:.6f}, cos_sim={cos2:.6f}")

    # ── Test 4: Compare attention scores with/without quantization ──
    print("\n" + "=" * 70)
    print("Test 4: Attention Score Error From Quantization")
    print("=" * 70)

    # Simulate one attention head
    # For GQA: queries have 12 heads, KV has 2 heads
    # Each KV head is shared across 6 query heads
    queries = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=torch.half) * 0.1

    # Original attention scores (with GQA expansion)
    # Expand KV from (1, 2, T, D) to (1, 12, T, D) by repeating
    gqa_ratio = num_heads // kv_heads
    keys_expanded = actual_keys.repeat_interleave(gqa_ratio, dim=1)  # (1, 12, T, D)
    scores_orig = torch.matmul(queries, keys_expanded.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_orig = F.softmax(scores_orig, dim=-1)

    # Quantized attention scores
    k_flat_all = actual_keys.reshape(-1, D)
    k_idx, k_norms = kq.quantize(k_flat_all)
    k_hat = kq.dequantize(k_idx, k_norms).reshape(actual_keys.shape)
    keys_hat_expanded = k_hat.repeat_interleave(gqa_ratio, dim=1)
    scores_quant = torch.matmul(queries, keys_hat_expanded.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_quant = F.softmax(scores_quant, dim=-1)

    score_mse = F.mse_loss(scores_quant.float(), scores_orig.float()).item()
    attn_mse = F.mse_loss(attn_quant.float(), attn_orig.float()).item()
    attn_kl = F.kl_div(attn_quant.float().log().clamp(min=-100), attn_orig.float(),
                        reduction='batchmean').item()

    print(f"  Attention score MSE:  {score_mse:.6f}")
    print(f"  Attention weight MSE: {attn_mse:.6f}")
    print(f"  Attention KL div:     {attn_kl:.6f}")

    # Compare to GPT-2 scale errors
    print(f"\n  For reference, if these numbers are similar to GPT-2's,")
    print(f"  then the failure is NOT in the quantizer but in how the")
    print(f"  cache integrates with the model.")

    # ── Test 5: Direct forward pass with manually quantized KV ──
    print("\n" + "=" * 70)
    print("Test 5: Perplexity With Manual Quantization (bypass our cache)")
    print("=" * 70)

    # Run model normally, extract KV, quantize, put back, re-run
    with torch.no_grad():
        outputs_orig = model(input_ids, use_cache=True)
        logits_orig = outputs_orig.logits
        loss_orig = F.cross_entropy(logits_orig[:, :-1].reshape(-1, logits_orig.size(-1)),
                                     input_ids[:, 1:].reshape(-1))
        ppl_orig = torch.exp(loss_orig).item()
        print(f"  Original PPL: {ppl_orig:.2f}")

    # Quantize all KV in the cache and re-run
    cache_orig = outputs_orig.past_key_values
    for li, layer in enumerate(cache_orig.layers):
        k = layer.keys   # (1, kv_heads, T, D)
        v = layer.values

        B, H, T, D_actual = k.shape
        kq_layer = TurboQuantizer(D_actual, 4, device=device, seed=42 + li * 2)
        vq_layer = TurboQuantizer(D_actual, 2, device=device, seed=42 + li * 2 + 1)

        k_flat = k.reshape(B * H * T, D_actual)
        v_flat = v.reshape(B * H * T, D_actual)

        k_idx, k_n = kq_layer.quantize(k_flat)
        k_hat = kq_layer.dequantize(k_idx, k_n).reshape(B, H, T, D_actual)

        v_idx, v_n = vq_layer.quantize(v_flat)
        v_hat = vq_layer.dequantize(v_idx, v_n).reshape(B, H, T, D_actual)

        layer.keys = k_hat
        layer.values = v_hat

    # Generate one more token using the quantized cache
    next_token = input_ids[:, -1:]
    with torch.no_grad():
        outputs_quant = model(input_ids, past_key_values=None)  # fresh forward pass

        # Actually, to test the cache we need to split: prefill + decode
        # Simpler: just measure logits on the same input with quantized cache as context

        # Re-encode with the quantized cache
        # The cache already has all tokens, so we can't easily re-use it for the same input
        # Instead, let's measure: how different are the logits?
        pass

    # Simpler test: quantize KV at every layer and check logit difference
    print("\n  Checking logit difference with quantized KV...")
    with torch.no_grad():
        # Approach: hook into each layer, intercept KV, quantize in-place
        quantized_outputs = {'logits': None}

        hooks = []
        def make_quant_hook(layer_idx):
            def hook_fn(module, args, kwargs, output):
                # For Qwen, the attention module outputs (attn_output, attn_weights, past_kv)
                # We need to intercept the KV before attention, not after
                pass
            return hook_fn

        # Simpler approach: just run the full model twice and compare
        # 1st run: normal
        out1 = model(input_ids)
        logits1 = out1.logits

    # Compute per-layer quantization error
    print("\n  Per-layer KV quantization error (on actual model activations):")
    total_k_cos = []
    total_v_cos = []
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        for li, layer in enumerate(out.past_key_values.layers):
            k = layer.keys
            v = layer.values
            B, H, T, D_actual = k.shape

            kq_l = TurboQuantizer(D_actual, 4, device=device, seed=42 + li * 2)
            vq_l = TurboQuantizer(D_actual, 2, device=device, seed=42 + li * 2 + 1)

            k_flat = k.reshape(-1, D_actual)
            v_flat = v.reshape(-1, D_actual)

            ki, kn = kq_l.quantize(k_flat)
            kh = kq_l.dequantize(ki, kn)
            k_cos = F.cosine_similarity(k_flat.float(), kh.float(), dim=-1).mean().item()

            vi, vn = vq_l.quantize(v_flat)
            vh = vq_l.dequantize(vi, vn)
            v_cos = F.cosine_similarity(v_flat.float(), vh.float(), dim=-1).mean().item()

            total_k_cos.append(k_cos)
            total_v_cos.append(v_cos)

            if li < 5 or li >= 25:
                print(f"    Layer {li:2d}: K4 cos={k_cos:.6f}  V2 cos={v_cos:.6f}  "
                      f"K_norm_mean={k_flat.float().norm(dim=-1).mean():.3f}  "
                      f"V_norm_mean={v_flat.float().norm(dim=-1).mean():.3f}")
        print(f"    ...")
        print(f"    Avg K4 cos: {np.mean(total_k_cos):.6f}")
        print(f"    Avg V2 cos: {np.mean(total_v_cos):.6f}")
        print(f"    Min K4 cos: {min(total_k_cos):.6f} (layer {np.argmin(total_k_cos)})")
        print(f"    Min V2 cos: {min(total_v_cos):.6f} (layer {np.argmin(total_v_cos)})")

    # ── Test 6: Is the problem GQA or just our cache implementation? ──
    print("\n" + "=" * 70)
    print("Test 6: FP16 Cache Layer Sanity Check")
    print("=" * 70)

    from experiments.exp07_larger_models import FP16Layer

    # Build all-fp16 cache
    cache_fp16 = DynamicCache()
    cache_fp16.layers = [FP16Layer() for _ in range(config.num_hidden_layers)]
    cache_fp16.layer_class_to_replicate = None

    with torch.no_grad():
        out_fp16 = model(input_ids, past_key_values=cache_fp16)
        logits_fp16 = out_fp16.logits
        loss_fp16 = F.cross_entropy(logits_fp16[:, :-1].reshape(-1, logits_fp16.size(-1)),
                                     input_ids[:, 1:].reshape(-1))
        ppl_fp16 = torch.exp(loss_fp16).item()
        print(f"  PPL with our FP16Layer cache: {ppl_fp16:.2f}")
        print(f"  PPL with default cache:       {ppl_orig:.2f}")
        print(f"  Difference: {abs(ppl_fp16 - ppl_orig):.4f}")
        if abs(ppl_fp16 - ppl_orig) > 0.1:
            print(f"  *** WARNING: Our FP16 cache gives different PPL! Cache integration bug likely.")
        else:
            print(f"  OK: Our cache layer integrates correctly at fp16.")

    # ── Test 7: Quantized cache with LARGE window (almost no compression) ──
    print("\n" + "=" * 70)
    print("Test 7: Quantized Cache With Window = seq_len (no compression)")
    print("=" * 70)

    from experiments.exp07_larger_models import QuantizedLayer, get_quantizer

    D_actual = head_dim
    cache_big_win = DynamicCache()
    cache_big_win.layers = []
    cache_big_win.layer_class_to_replicate = None
    for li in range(config.num_hidden_layers):
        kq = get_quantizer(D_actual, 4, device, 42 + li * 2)
        vq = get_quantizer(D_actual, 2, device, 42 + li * 2 + 1)
        cache_big_win.layers.append(QuantizedLayer(kq, vq, num_sinks=0, residual_window=512))

    with torch.no_grad():
        out_bw = model(input_ids, past_key_values=cache_big_win)
        logits_bw = out_bw.logits
        loss_bw = F.cross_entropy(logits_bw[:, :-1].reshape(-1, logits_bw.size(-1)),
                                   input_ids[:, 1:].reshape(-1))
        ppl_bw = torch.exp(loss_bw).item()
        print(f"  PPL with window=512 (no compression): {ppl_bw:.2f}")
        print(f"  PPL baseline:                         {ppl_orig:.2f}")
        print(f"  If these match, compression itself is the issue, not our cache plumbing.")

    # ── Test 8: Quantized cache with small window ──
    print("\n" + "=" * 70)
    print("Test 8: Quantized Cache With Window = 4 (heavy compression)")
    print("=" * 70)

    cache_small = DynamicCache()
    cache_small.layers = []
    cache_small.layer_class_to_replicate = None
    for li in range(config.num_hidden_layers):
        kq = get_quantizer(D_actual, 4, device, 42 + li * 2)
        vq = get_quantizer(D_actual, 2, device, 42 + li * 2 + 1)
        cache_small.layers.append(QuantizedLayer(kq, vq, num_sinks=0, residual_window=4))

    with torch.no_grad():
        out_sm = model(input_ids, past_key_values=cache_small)
        logits_sm = out_sm.logits
        loss_sm = F.cross_entropy(logits_sm[:, :-1].reshape(-1, logits_sm.size(-1)),
                                   input_ids[:, 1:].reshape(-1))
        ppl_sm = torch.exp(loss_sm).item()
        print(f"  PPL with window=4 (heavy compression): {ppl_sm:.2f}")
        print(f"  PPL baseline:                          {ppl_orig:.2f}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(f"  FP16 cache PPL:     {ppl_fp16:.2f} (our cache plumbing)")
    print(f"  Big window PPL:     {ppl_bw:.2f} (no compression)")
    print(f"  Small window PPL:   {ppl_sm:.2f} (heavy compression)")
    print(f"  Original PPL:       {ppl_orig:.2f} (baseline)")

    if abs(ppl_fp16 - ppl_orig) > 0.5:
        print(f"\n  VERDICT: Cache plumbing bug — our FP16Layer doesn't match baseline.")
    elif abs(ppl_bw - ppl_orig) > 0.5:
        print(f"\n  VERDICT: Window logic bug — even with no compression, results differ.")
    elif ppl_sm > 100:
        print(f"\n  VERDICT: Quantization actually breaks Qwen — GQA amplifies errors severely.")
    else:
        print(f"\n  VERDICT: Everything works — need to investigate further.")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
