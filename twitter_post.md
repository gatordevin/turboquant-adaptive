# Twitter Thread Draft

---

**Tweet 1 (main):**

We implemented Google's TurboQuant KV-cache quantization from scratch and found two simple tricks that cut its quality loss by more than half.

+2.4% PPL -> +1.1% PPL at similar compression.

Here's what worked, what didn't, and why most "obvious" improvements fail.

Code: github.com/gatordevin/turboquant-adaptive

🧵

---

**Tweet 2 (TurboQuant explainer):**

TurboQuant (ICLR 2026) compresses the KV cache during LLM inference:

1. Rotate each vector by a random orthogonal matrix
2. Now every coordinate follows a known Beta distribution
3. Apply an optimal scalar quantizer per coordinate
4. 3-4x memory savings, training-free, works on any model

---

**Tweet 3 (the discovery):**

We swept 40+ configs on GPT-2 and found two things matter:

1. Token 0 absorbs 35-66% of ALL attention in deeper layers. It's the model's "attention parking lot." Quantizing it even at 4-bit hurts everything downstream.

2. Layer 0 is 5-25x more sensitive to noise than other layers.

---

**Tweet 4 (attention sink visual):**

Attention received by token position 0:

Layer 0:   2.8%  (normal)
Layer 3:  35.8%
Layer 5:  59.2%  ← >half of all attention!
Layer 7:  65.9%
Layer 11: 48.1%

These "attention sinks" (Xiao et al., 2023) are load-bearing infrastructure. Protecting just 8 tokens in fp16 costs ~0.3% memory.

---

**Tweet 5 (what works):**

The winning recipe:
- Layer 0: K4/V4 (it's 5-25x more sensitive)
- Layers 1,2,4,5: K4/V2 (medium sensitivity)
- Layers 3,6-11: K3/V2 (low sensitivity)
- First 8 tokens: fp16 always (attention sinks)
- Last 32 tokens: fp16 (residual window)

Result: +1.1% PPL vs +2.4% for uniform K4/V2.

---

**Tweet 6 (what DOESN'T work):**

Most "improvements" we tried made things WORSE:

- Hadamard rotation: +7.1% PPL (worse than random orthogonal!)
- Residual quantization: +11-31% MSE
- Progressive aging: +497% PPL (re-quantizing compounds noise)
- Importance sampling: +117% PPL (breaks position encoding)
- Dual-resolution storage: +671% PPL (2-bit fallback is toxic)

---

**Tweet 7 (the 2-bit cliff):**

The most important finding: there's a HARD CLIFF at 2-bit keys.

K4: cos_sim=0.996, PPL +2-3%
K3: cos_sim=0.984, PPL +5-15%
K2: cos_sim=0.942, PPL +24-75%  ← catastrophic
K1: cos_sim=0.801, PPL +130%+

Softmax exponentially amplifies small attention score errors. Never go below K3.

---

**Tweet 8 (takeaway):**

The big insight: the bottleneck in KV cache quantization is NOT the quantizer math.

TurboQuant's rotation + Lloyd-Max is already near the information-theoretic limit. The gains come from:

→ Don't quantize what the model depends on (sinks)
→ Don't waste bits where it doesn't care (low-sensitivity layers)
→ Never cross the 2-bit cliff

---

**Tweet 9 (credit):**

Credit where due — sink protection from quantization is established (KVQuant, NeurIPS 2024; KVSink, COLM 2025) and per-layer adaptive bits exist (KVTuner, ICML 2025).

Our contribution: systematic ablation showing what composes well with TurboQuant and a concrete recipe that works.

Full code + all experiments: github.com/gatordevin/turboquant-adaptive
