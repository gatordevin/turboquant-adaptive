"""
System benchmark — measures actual GPU memory, wall-clock time, and throughput
for all key configurations. Generates data for the README.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from turboquant_gpt2 import TurboQuantizer, TurboQuantLayer, make_turboquant_cache
from experiments.exp03_sink_aware_dynamic import make_sink_cache, _get_q, FP16Layer, SinkAwareLayer
from transformers.cache_utils import DynamicCache


def measure_gpu_memory(func, *args, **kwargs):
    """Run func and measure peak GPU memory delta."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_after = torch.cuda.memory_allocated()
    return result, (mem_peak - mem_before) / 1024**2, (mem_after - mem_before) / 1024**2


def make_best_cache(head_dim, num_layers, device):
    """Our best config: Sink(8) + Adaptive tiers w32."""
    tiers = {0: (4, 4)}
    for i in [1, 2, 4, 5]:
        tiers[i] = (4, 2)
    for i in [3, 6, 7, 8, 9, 10, 11]:
        tiers[i] = (3, 2)
    return make_sink_cache(head_dim, num_layers, 4, 2, 8, 32, device, sensitivity_tiers=tiers)


def compute_perplexity(model, tokenizer, text, cache=None):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=cache)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return torch.exp(loss).item()


def measure_throughput(model, tokenizer, cache_factory, prompt, num_tokens=200, warmup=3):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    for _ in range(warmup):
        cache = cache_factory() if cache_factory else None
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=10, past_key_values=cache,
                          do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    cache = cache_factory() if cache_factory else None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=num_tokens, past_key_values=cache,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (out.shape[1] - input_ids.shape[1]) / (t1 - t0), t1 - t0


def main():
    import platform, os, psutil

    device = 'cuda'
    print("=" * 75)
    print("SYSTEM BENCHMARK")
    print("=" * 75)

    # System info
    print(f"\n  Hardware:")
    print(f"    GPU:            {torch.cuda.get_device_name(0)}")
    print(f"    GPU Memory:     {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"    CPU Cores:      {os.cpu_count()}")
    print(f"    RAM:            {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"\n  Software:")
    print(f"    Python:         {platform.python_version()}")
    print(f"    PyTorch:        {torch.__version__}")
    print(f"    CUDA:           {torch.version.cuda}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).half()
    model.eval()

    model_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"\n  Model memory:     {model_mem:.1f} MB (GPT-2 124M, fp16)")

    head_dim, num_layers = 64, 12

    eval_text = (
        "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
        "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of "
        "modern AI were planted by philosophers who attempted to describe the process of human thinking "
        "as the mechanical manipulation of symbols. This work culminated in the invention of the "
        "programmable digital computer in the 1940s, a machine based on the abstract essence of "
        "mathematical reasoning. This device and the ideas behind it inspired a handful of scientists "
        "to begin seriously discussing the possibility of building an electronic brain. The field of AI "
        "research was founded at a workshop held on the campus of Dartmouth College during the summer "
        "of 1956. Those who attended would become the leaders of AI research for decades."
    ) * 2
    prompt = "The future of artificial intelligence is"
    seq_len = len(tokenizer.encode(eval_text))
    print(f"  Eval text:        {seq_len} tokens")

    # Pre-compute codebooks
    print("\n  Pre-computing Lloyd-Max codebooks...")
    t0 = time.time()
    for bits in [1, 2, 3, 4]:
        for li in range(num_layers):
            _get_q(head_dim, bits, device, 42 + li * 2)
            _get_q(head_dim, bits, device, 42 + li * 2 + 1)
    codebook_time = time.time() - t0
    print(f"  Codebook computation: {codebook_time:.1f}s (one-time cost, {num_layers}×4 codebooks)")

    # ── Benchmark all configs ──
    print("\n" + "=" * 75)
    print("PERPLEXITY & MEMORY")
    print("=" * 75)

    configs = [
        ("Baseline fp16",         lambda: None),
        ("TurboQuant K4/V2 w64",  lambda: make_turboquant_cache(head_dim, num_layers, 4, 2, 64, device)),
        ("TurboQuant K4/V2 w32",  lambda: make_turboquant_cache(head_dim, num_layers, 4, 2, 32, device)),
        ("TurboQuant K3/V2 w32",  lambda: make_turboquant_cache(head_dim, num_layers, 3, 2, 32, device)),
        ("Sink(8)+Adaptive w32",  lambda: make_best_cache(head_dim, num_layers, device)),
    ]

    results = []
    print(f"\n  {'Config':30s} {'PPL':>7s} {'ΔPPL':>7s} {'Peak MB':>9s} {'Cache MB':>9s} {'tok/s':>7s} {'Time':>7s}")
    print("  " + "-" * 82)

    for name, factory in configs:
        cache = factory()

        # Perplexity + memory
        _, peak_mb, cache_mb = measure_gpu_memory(
            compute_perplexity, model, tokenizer, eval_text, cache
        )

        ppl = compute_perplexity(model, tokenizer, eval_text, cache=factory())

        # Throughput
        tps, wall = measure_throughput(model, tokenizer,
                                        factory if name != "Baseline fp16" else None,
                                        prompt, num_tokens=200, warmup=3)

        results.append((name, ppl, peak_mb, cache_mb, tps, wall))

    ppl_base = results[0][1]
    for name, ppl, peak_mb, cache_mb, tps, wall in results:
        delta = ((ppl - ppl_base) / ppl_base) * 100
        delta_str = f"{delta:+.1f}%" if name != "Baseline fp16" else "—"
        print(f"  {name:30s} {ppl:>7.2f} {delta_str:>7s} {peak_mb:>8.1f} {cache_mb:>8.1f} {tps:>7.1f} {wall:>6.2f}s")

    # ── Codebook sizes ──
    print(f"\n" + "=" * 75)
    print("CODEBOOK & ROTATION MATRIX SIZES")
    print("=" * 75)
    for bits in [1, 2, 3, 4]:
        n_levels = 2 ** bits
        centroid_bytes = n_levels * 2  # fp16
        boundary_bytes = (n_levels - 1) * 2
        rotation_bytes = head_dim * head_dim * 2  # fp16 matrix
        total_per_layer = centroid_bytes + boundary_bytes + rotation_bytes
        print(f"  {bits}-bit: centroids={centroid_bytes}B  boundaries={boundary_bytes}B  "
              f"rotation={rotation_bytes}B  total={total_per_layer}B/layer  "
              f"({total_per_layer * num_layers / 1024:.1f} KB for {num_layers} layers)")

    # ── KV cache size breakdown ──
    print(f"\n" + "=" * 75)
    print(f"KV CACHE SIZE BREAKDOWN ({seq_len} tokens)")
    print("=" * 75)
    num_heads = 12
    fp16_per_token_per_head = head_dim * 2  # bytes
    fp16_total = seq_len * fp16_per_token_per_head * 2 * num_heads * num_layers  # K+V
    print(f"  Full fp16 cache:  {fp16_total / 1024:.1f} KB = {fp16_total / 1024**2:.2f} MB")
    print(f"    Per layer:      {seq_len * fp16_per_token_per_head * 2 * num_heads / 1024:.1f} KB")
    print(f"    Per head:       {seq_len * fp16_per_token_per_head * 2 / 1024:.1f} KB")

    for name, kb, vb, win, sinks in [
        ("K4/V2 w64", 4, 2, 64, 0),
        ("K4/V2 w32", 4, 2, 32, 0),
        ("Sink(8)+Adaptive w32", 3.4, 2.2, 32, 8),
    ]:
        compressed = max(0, seq_len - win - sinks)
        recent = min(seq_len, win)
        sink_tokens = min(sinks, seq_len)
        comp_per = head_dim * kb / 8 + 2 + head_dim * vb / 8 + 2  # indices + norms
        total = (compressed * comp_per + (recent + sink_tokens) * fp16_per_token_per_head * 2) * num_heads * num_layers
        ratio = fp16_total / total
        print(f"\n  {name}:")
        print(f"    Sink fp16:      {sink_tokens} tokens × {fp16_per_token_per_head * 2}B = "
              f"{sink_tokens * fp16_per_token_per_head * 2 * num_heads * num_layers / 1024:.1f} KB")
        print(f"    Recent fp16:    {recent} tokens = "
              f"{recent * fp16_per_token_per_head * 2 * num_heads * num_layers / 1024:.1f} KB")
        print(f"    Compressed:     {compressed} tokens × {comp_per:.0f}B = "
              f"{compressed * comp_per * num_heads * num_layers / 1024:.1f} KB")
        print(f"    Total:          {total / 1024:.1f} KB ({ratio:.2f}x compression)")

    # ── Wall clock breakdown ──
    print(f"\n" + "=" * 75)
    print("WALL-CLOCK BREAKDOWN")
    print("=" * 75)
    print(f"  Codebook computation (one-time): {codebook_time:.1f}s")
    print(f"    ({codebook_time / (4 * num_layers * 2):.2f}s per codebook, "
          f"{4 * num_layers * 2} codebooks total)")
    print(f"  Model loading:                   ~1s")
    print(f"  Generation (200 tokens):")
    for name, _, _, _, tps, wall in results:
        print(f"    {name:30s}: {wall:.2f}s ({tps:.0f} tok/s)")


if __name__ == "__main__":
    main()
