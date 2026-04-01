"""
TurboQuant on Larger Models
============================
Test whether sink-aware + sensitivity-tiered quantization scales
to larger models with more layers and different architectures.

Models tested:
  - GPT-2 Small  (124M, 12 layers, hd=64)  — baseline reference
  - GPT-2 Medium (355M, 24 layers, hd=64)  — 2x layers
  - GPT-2 Large  (774M, 36 layers, hd=64)  — 3x layers
  - GPT-2 XL    (1.5B, 48 layers, hd=64)   — 4x layers
  - Qwen2.5-1.5B (1.5B, 28 layers, hd=128) — modern arch, GQA, RoPE, larger head_dim

Key questions:
  1. Does the layer-0 sensitivity finding hold for deeper models?
  2. Does sink protection help more or less at scale?
  3. Does head_dim=128 (Qwen) quantize better than head_dim=64 (GPT-2)?
  4. How does GQA (grouped query attention) interact with KV quantization?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache, CacheLayerMixin

from turboquant_gpt2 import TurboQuantizer, compute_lloyd_max_codebook


# ============================================================================
# Quantizer cache (avoid recomputing codebooks)
# ============================================================================

_qcache = {}

def get_quantizer(head_dim, bits, device, seed):
    key = (head_dim, bits, seed)
    if key not in _qcache:
        _qcache[key] = TurboQuantizer(head_dim, bits, device=device, seed=seed,
                                       use_exact_pdf=(head_dim < 128))
    return _qcache[key]


# ============================================================================
# Cache layers (streamlined from previous experiments)
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


class QuantizedLayer(CacheLayerMixin):
    """TurboQuant cache layer with optional sink protection."""
    is_sliding = False

    def __init__(self, kq, vq, num_sinks=0, residual_window=32):
        super().__init__()
        self.kq, self.vq = kq, vq
        self.num_sinks = num_sinks
        self.residual_window = residual_window
        self.sink_keys = self.sink_values = None
        self.compressed_k = []
        self.compressed_v = []
        self.compressed_shapes = []
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

        # First call: capture sinks
        if self.total_seen == 0:
            if self.num_sinks > 0 and T >= self.num_sinks:
                self.sink_keys = key_states[:, :, :self.num_sinks, :].clone()
                self.sink_values = value_states[:, :, :self.num_sinks, :].clone()
                self.recent_keys = key_states[:, :, self.num_sinks:, :] if T > self.num_sinks else key_states[:, :, :0, :]
                self.recent_values = value_states[:, :, self.num_sinks:, :] if T > self.num_sinks else value_states[:, :, :0, :]
            else:
                if self.num_sinks > 0:
                    self.sink_keys = key_states.clone()
                    self.sink_values = value_states.clone()
                    self.recent_keys = key_states[:, :, :0, :]
                    self.recent_values = value_states[:, :, :0, :]
                else:
                    self.recent_keys = key_states
                    self.recent_values = value_states
            self.total_seen = T
        else:
            if self.recent_keys is None or self.recent_keys.shape[-2] == 0:
                self.recent_keys = key_states
                self.recent_values = value_states
            else:
                self.recent_keys = torch.cat([self.recent_keys, key_states], dim=-2)
                self.recent_values = torch.cat([self.recent_values, value_states], dim=-2)
            self.total_seen += T

        # Compress overflow
        if self.recent_keys is not None and self.recent_keys.shape[-2] > self.residual_window:
            overflow = self.recent_keys.shape[-2] - self.residual_window
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

        # Reconstruct
        parts_k, parts_v = [], []
        if self.sink_keys is not None:
            parts_k.append(self.sink_keys)
            parts_v.append(self.sink_values)
        for i, (B2, H2, T2) in enumerate(self.compressed_shapes):
            k_idx, k_n = self.compressed_k[i]
            v_idx, v_n = self.compressed_v[i]
            parts_k.append(self.kq.dequantize(k_idx, k_n).reshape(B2, H2, T2, -1))
            parts_v.append(self.vq.dequantize(v_idx, v_n).reshape(B2, H2, T2, -1))
        if self.recent_keys is not None:
            parts_k.append(self.recent_keys)
            parts_v.append(self.recent_values)

        self.keys = torch.cat(parts_k, dim=-2)
        self.values = torch.cat(parts_v, dim=-2)
        return self.keys, self.values

    def get_seq_length(self):
        s = self.sink_keys.shape[-2] if self.sink_keys is not None else 0
        c = sum(sh[2] for sh in self.compressed_shapes)
        r = self.recent_keys.shape[-2] if self.recent_keys is not None and self.recent_keys.numel() > 0 else 0
        return s + c + r

    def get_max_cache_shape(self): return -1
    def crop(self, m): pass
    def batch_repeat_interleave(self, r):
        for t in [self.sink_keys, self.sink_values, self.recent_keys, self.recent_values]:
            if t is not None and t.numel() > 0:
                t.data = t.repeat_interleave(r, dim=0)
    def batch_select_indices(self, idx):
        for t in [self.sink_keys, self.sink_values, self.recent_keys, self.recent_values]:
            if t is not None and t.numel() > 0:
                t.data = t[idx, ...]
    def get_mask_sizes(self, cp):
        return self.get_seq_length() + cp.shape[0], 0


# ============================================================================
# Cache factories
# ============================================================================

def make_vanilla_cache(head_dim, num_layers, kb, vb, win, device):
    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None
    for li in range(num_layers):
        kq = get_quantizer(head_dim, kb, device, 42 + li * 2)
        vq = get_quantizer(head_dim, vb, device, 42 + li * 2 + 1)
        cache.layers.append(QuantizedLayer(kq, vq, num_sinks=0, residual_window=win))
    return cache


def make_sink_adaptive_cache(head_dim, num_layers, num_sinks, win, device,
                              layer_sensitivities=None):
    """Sink-aware + sensitivity-tiered cache.

    If layer_sensitivities is provided, uses it to assign tiers.
    Otherwise, uses a heuristic: layer 0 = critical, first/last 20% = high, rest = low.
    """
    cache = DynamicCache()
    cache.layers = []
    cache.layer_class_to_replicate = None

    if layer_sensitivities is not None:
        # Assign tiers based on measured sensitivity
        max_s = max(layer_sensitivities)
        tiers = []
        for s in layer_sensitivities:
            ratio = s / max_s if max_s > 0 else 0
            if ratio > 0.5:
                tiers.append('critical')
            elif ratio > 0.1:
                tiers.append('high')
            else:
                tiers.append('low')
    else:
        # Heuristic: first layer critical, early layers high, rest low
        tiers = []
        for li in range(num_layers):
            if li == 0:
                tiers.append('critical')
            elif li <= max(2, num_layers // 5):
                tiers.append('high')
            else:
                tiers.append('low')

    tier_bits = {
        'critical': (4, 4),
        'high': (4, 2),
        'low': (3, 2),
    }

    for li in range(num_layers):
        tier = tiers[li]
        kb, vb = tier_bits[tier]

        if tier == 'critical' and kb == 4 and vb == 4:
            # Critical layers: fp16
            cache.layers.append(FP16Layer())
        else:
            kq = get_quantizer(head_dim, kb, device, 42 + li * 2)
            vq = get_quantizer(head_dim, vb, device, 42 + li * 2 + 1)
            cache.layers.append(QuantizedLayer(kq, vq, num_sinks=num_sinks,
                                                residual_window=win))

    return cache, tiers


# ============================================================================
# Sensitivity measurement
# ============================================================================

def measure_layer_sensitivity(model, tokenizer, text, device='cuda', model_type='gpt2'):
    """Measure per-layer sensitivity to noise."""
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    with torch.no_grad():
        base_outputs = model(input_ids)
        base_logits = base_outputs.logits

    num_layers = model.config.num_hidden_layers
    sensitivities = []

    for target_layer in range(num_layers):
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, args, output):
                if layer_idx == target_layer:
                    if isinstance(output, tuple):
                        perturbed = output[0] + torch.randn_like(output[0]) * 0.01
                        return (perturbed,) + output[1:]
                return output
            return hook_fn

        # Hook attention modules
        if model_type == 'gpt2':
            for i, block in enumerate(model.transformer.h):
                h = block.attn.register_forward_hook(make_hook(i))
                hooks.append(h)
        else:
            # Generic: hook into model.model.layers[i].self_attn
            for i, layer in enumerate(model.model.layers):
                h = layer.self_attn.register_forward_hook(make_hook(i))
                hooks.append(h)

        with torch.no_grad():
            noisy_outputs = model(input_ids)
            noisy_logits = noisy_outputs.logits

        for h in hooks:
            h.remove()

        diff = (noisy_logits - base_logits).float().pow(2).mean().item()
        sensitivities.append(diff)

    s = np.array(sensitivities)
    if s.max() > 0:
        s = s / s.max()
    return s


# ============================================================================
# Attention sink analysis
# ============================================================================

def analyze_sinks(model, tokenizer, text, device='cuda', model_type='gpt2'):
    """Measure attention concentration on first few tokens."""
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    if not outputs.attentions or len(outputs.attentions) == 0:
        return None

    sink_attention = []
    for layer_idx, attn in enumerate(outputs.attentions):
        # attn: (B, H, T, T)
        # Average attention received by token 0
        attn_on_first = attn[:, :, :, 0].float().mean().item()
        sink_attention.append(attn_on_first)

    return sink_attention


# ============================================================================
# Evaluation
# ============================================================================

def compute_perplexity(model, tokenizer, text, cache=None):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=cache)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return torch.exp(loss).item()


# ============================================================================
# Per-model experiment
# ============================================================================

def run_model_experiment(model_name, device='cuda'):
    """Run full experiment suite on a single model."""
    print(f"\n{'='*75}")
    print(f"MODEL: {model_name}")
    print(f"{'='*75}")

    # Determine model type
    is_gpt2 = 'gpt2' in model_name.lower()

    # Load
    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if is_gpt2:
        load_kwargs['attn_implementation'] = 'eager'  # needed for output_attentions

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device).half()
    model.eval()

    config = model.config
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    has_gqa = kv_heads < num_heads

    model_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"  Params:     {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"  GPU memory: {model_mem:.0f} MB")
    print(f"  head_dim:   {head_dim}")
    print(f"  layers:     {num_layers}")
    print(f"  attn heads: {num_heads} (KV heads: {kv_heads}{'  GQA!' if has_gqa else ''})")

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
        "generation, and they were given millions of dollars to make this vision come true."
    )
    seq_len = len(tokenizer.encode(eval_text))
    print(f"  Eval tokens: {seq_len}")

    # ── Pre-compute codebooks ──
    print(f"\n  Pre-computing codebooks for head_dim={head_dim}...")
    t0 = time.time()
    for bits in [2, 3, 4]:
        for li in range(num_layers):
            get_quantizer(head_dim, bits, device, 42 + li * 2)
            get_quantizer(head_dim, bits, device, 42 + li * 2 + 1)
    print(f"  Codebooks ready in {time.time()-t0:.1f}s")

    # ── Reconstruction quality ──
    print(f"\n  Reconstruction quality (head_dim={head_dim}):")
    x_test = torch.randn(2048, head_dim, device=device) * 0.5
    for bits in [2, 3, 4]:
        q = get_quantizer(head_dim, bits, device, seed=9999)
        idx, norms = q.quantize(x_test)
        x_hat = q.dequantize(idx, norms)
        mse = F.mse_loss(x_hat.float(), x_test.float()).item()
        cos = F.cosine_similarity(x_test.float(), x_hat.float(), dim=-1).mean().item()
        print(f"    {bits}-bit: MSE={mse:.6f}  cos_sim={cos:.6f}")

    # ── Sensitivity analysis ──
    print(f"\n  Layer sensitivity analysis...")
    model_type = 'gpt2' if is_gpt2 else 'generic'
    sensitivities = measure_layer_sensitivity(model, tokenizer, eval_text, device, model_type)

    print(f"\n  Layer sensitivities:")
    for i, s in enumerate(sensitivities):
        bar = "█" * int(s * 30)
        tier = "CRIT" if s > 0.5 else ("HIGH" if s > 0.1 else "low")
        print(f"    Layer {i:3d}: {s:.4f} {bar:30s} [{tier}]")

    # ── Attention sink analysis ──
    print(f"\n  Attention sink analysis...")
    try:
        sink_attn = analyze_sinks(model, tokenizer, eval_text, device, model_type)
        if sink_attn:
            print(f"  Attention on token 0 per layer:")
            for i, a in enumerate(sink_attn):
                bar = "█" * int(a * 50)
                print(f"    Layer {i:3d}: {a:.3f} {bar}")
            avg_sink = np.mean(sink_attn)
            max_sink = max(sink_attn)
            print(f"  Avg attention on token 0: {avg_sink:.3f}, max: {max_sink:.3f}")
        else:
            print(f"  Could not extract attention weights")
            sink_attn = None
    except Exception as e:
        print(f"  Attention analysis failed: {e}")
        sink_attn = None

    # ── Perplexity experiments ──
    print(f"\n  Perplexity experiments:")
    ppl_base = compute_perplexity(model, tokenizer, eval_text)
    print(f"    Baseline fp16:           PPL = {ppl_base:.2f}")

    results = [("Baseline fp16", ppl_base, 0.0, 'baseline')]

    # Vanilla TurboQuant
    for kb, vb, win in [(4, 2, 64), (4, 2, 32), (3, 2, 32)]:
        name = f"Vanilla K{kb}/V{vb} w{win}"
        cache = make_vanilla_cache(head_dim, num_layers, kb, vb, win, device)
        ppl = compute_perplexity(model, tokenizer, eval_text, cache=cache)
        delta = ((ppl - ppl_base) / ppl_base) * 100
        results.append((name, ppl, delta, 'vanilla'))
        print(f"    {name:35s}: PPL = {ppl:.2f} ({delta:+.1f}%)")

    # Sink-aware + adaptive (heuristic tiers)
    for n_sinks in [4, 8]:
        name = f"Sink({n_sinks})+Heuristic w32"
        cache, tiers = make_sink_adaptive_cache(
            head_dim, num_layers, n_sinks, 32, device
        )
        ppl = compute_perplexity(model, tokenizer, eval_text, cache=cache)
        delta = ((ppl - ppl_base) / ppl_base) * 100
        results.append((name, ppl, delta, 'heuristic'))
        print(f"    {name:35s}: PPL = {ppl:.2f} ({delta:+.1f}%)")

    # Sink-aware + adaptive (measured sensitivity)
    for n_sinks in [4, 8]:
        name = f"Sink({n_sinks})+Measured w32"
        cache, tiers = make_sink_adaptive_cache(
            head_dim, num_layers, n_sinks, 32, device,
            layer_sensitivities=sensitivities
        )
        ppl = compute_perplexity(model, tokenizer, eval_text, cache=cache)
        delta = ((ppl - ppl_base) / ppl_base) * 100
        results.append((name, ppl, delta, 'measured'))
        tier_str = "".join(t[0].upper() for t in tiers)
        print(f"    {name:35s}: PPL = {ppl:.2f} ({delta:+.1f}%)  tiers=[{tier_str}]")

    # Just sinks, no adaptive (to isolate sink contribution)
    for n_sinks in [4, 8]:
        name = f"Sink({n_sinks})+K4/V2 w32"
        cache = DynamicCache()
        cache.layers = []
        cache.layer_class_to_replicate = None
        for li in range(num_layers):
            kq = get_quantizer(head_dim, 4, device, 42 + li * 2)
            vq = get_quantizer(head_dim, 2, device, 42 + li * 2 + 1)
            cache.layers.append(QuantizedLayer(kq, vq, num_sinks=n_sinks, residual_window=32))
        ppl = compute_perplexity(model, tokenizer, eval_text, cache=cache)
        delta = ((ppl - ppl_base) / ppl_base) * 100
        results.append((name, ppl, delta, 'sink_only'))
        print(f"    {name:35s}: PPL = {ppl:.2f} ({delta:+.1f}%)")

    # ── Summary ──
    print(f"\n  {'─'*65}")
    print(f"  SUMMARY for {model_name}:")
    print(f"  {'─'*65}")
    best_vanilla = min([r for r in results if r[3] == 'vanilla'], key=lambda x: x[1])
    best_ours = min([r for r in results if r[3] in ('heuristic', 'measured', 'sink_only')], key=lambda x: x[1])
    improvement = best_vanilla[2] - best_ours[2]
    print(f"  Best vanilla:  {best_vanilla[0]:35s} = {best_vanilla[1]:.2f} ({best_vanilla[2]:+.1f}%)")
    print(f"  Best ours:     {best_ours[0]:35s} = {best_ours[1]:.2f} ({best_ours[2]:+.1f}%)")
    print(f"  Improvement:   {improvement:.1f} percentage points less degradation")

    # Clean up GPU
    del model
    torch.cuda.empty_cache()

    return {
        'model': model_name,
        'head_dim': head_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'kv_heads': kv_heads,
        'sensitivities': sensitivities.tolist(),
        'sink_attention': sink_attn,
        'results': results,
        'ppl_base': ppl_base,
        'best_vanilla_delta': best_vanilla[2],
        'best_ours_delta': best_ours[2],
    }


# ============================================================================
# Main
# ============================================================================

def main():
    device = 'cuda'
    print("=" * 75)
    print("TurboQuant — Larger Model Experiments")
    print("=" * 75)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Free VRAM: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

    models = [
        'gpt2',
        'gpt2-medium',
        'gpt2-large',
        'gpt2-xl',
    ]

    # Try Qwen if available
    try:
        AutoConfig.from_pretrained('Qwen/Qwen2.5-1.5B')
        models.append('Qwen/Qwen2.5-1.5B')
    except Exception:
        print("Qwen2.5-1.5B not available, skipping")

    all_results = []
    for model_name in models:
        try:
            result = run_model_experiment(model_name, device)
            all_results.append(result)
        except Exception as e:
            print(f"\n  FAILED: {model_name}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()

    # ── Cross-model comparison ──
    print("\n" + "=" * 75)
    print("CROSS-MODEL COMPARISON")
    print("=" * 75)
    print(f"\n  {'Model':25s} {'Params':>8s} {'Layers':>7s} {'hd':>4s} {'Base PPL':>10s} "
          f"{'Vanilla':>10s} {'Ours':>10s} {'Improv':>8s}")
    print("  " + "-" * 90)

    for r in all_results:
        params = {'gpt2': '124M', 'gpt2-medium': '355M', 'gpt2-large': '774M',
                  'gpt2-xl': '1.5B', 'Qwen/Qwen2.5-1.5B': '1.5B'}.get(r['model'], '?')
        print(f"  {r['model']:25s} {params:>8s} {r['num_layers']:>7d} {r['head_dim']:>4d} "
              f"{r['ppl_base']:>10.2f} {r['best_vanilla_delta']:>+9.1f}% "
              f"{r['best_ours_delta']:>+9.1f}% {r['best_vanilla_delta'] - r['best_ours_delta']:>7.1f}pp")

    # ── Sensitivity patterns across models ──
    print(f"\n  Sensitivity patterns:")
    for r in all_results:
        s = r['sensitivities']
        top3 = sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:3]
        top3_str = ", ".join(f"L{i}({s[i]:.2f})" for i in top3)
        print(f"    {r['model']:25s}: most sensitive = {top3_str}")

    # ── Sink patterns across models ──
    print(f"\n  Attention sink (token 0) patterns:")
    for r in all_results:
        if r['sink_attention']:
            avg = np.mean(r['sink_attention'])
            mx = max(r['sink_attention'])
            mx_layer = r['sink_attention'].index(mx)
            print(f"    {r['model']:25s}: avg={avg:.3f}, max={mx:.3f} (layer {mx_layer})")

    print(f"\n{'='*75}")
    print("Done!")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
