"""Block C's ceiling, printed before the card: Llama-3.3-70B FP8 decode on one MI300X.

Every figure comes from bench/roofline.py's own functions. The model is not added to
predictions.MODELS, because a model there reaches the page by itself
(bench/tests/test_export_site_data.py). Run from the repository root:
    python3 docs/benchmarks/raw/mi300x-2026-10-17/run5/ceiling.py
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "bench"))

from roofline import ACCELERATORS, Model, kv_cache_tokens, tpot_floor  # noqa: E402

# Memory side: the safetensors index's total_size at one byte per unit, so the BF16
# embedding and lm_head count at their real size. Compute side: 70.55e9 less both,
# 1.05e9 each, as Qwen3-8B's 6.95e9 leaves out both.
LLAMA_70B_FP8 = Model(
    name="Llama-3.3-70B-Instruct-FP8-KV",
    params_total=72_656_376_512,
    params_non_embedding=68.45e9,
    num_layers=80,
    num_kv_heads=8,
    head_dim=128,
    weight_dtype_bytes=1,
    kv_dtype_bytes=1,
)

GMU = 0.94                         # InferenceX's --gpu-memory-utilization
LEVELS = (4, 16, 64)               # the concurrencies block C visits
CONTEXT = 7_373 + 922 // 2         # mean input + half the mean output (runsheet §5)

for key in ("mi300x-run2", "mi300x-run1", "mi300x"):
    accel = ACCELERATORS[key]
    pool = kv_cache_tokens(LLAMA_70B_FP8, accel, GMU)
    print(f"{key}: eff_mem {accel.achieved_bandwidth}, pool <= {pool:,.0f} tokens "
          f"at {GMU} ({accel.provenance or 'no provenance'})")
    for c in LEVELS:
        step = tpot_floor(LLAMA_70B_FP8, accel, c, CONTEXT)
        print(f"  c = {c:>2}: TPOT >= {step.seconds * 1e3:5.1f} ms ({step.bound_by}), "
              f"<= {1 / step.seconds:5.1f} tok/s per user, "
              f"<= {c / step.seconds:6.0f} tok/s per GPU, decode only")
# Weights alone, batch 1: the fastest any single stream can decode on this card.
for key in ("mi300x-run2", "mi300x"):
    step = tpot_floor(LLAMA_70B_FP8, ACCELERATORS[key], 1, 0)
    print(f"{key}: weights only, batch 1: {1 / step.seconds:.1f} tok/s per user at most")

# §8's fallback, if the engine refuses the FP8 KV cache: the same levels with BF16 KV.
bf16_kv = replace(LLAMA_70B_FP8, kv_dtype_bytes=2)
for c in LEVELS:
    step = tpot_floor(bf16_kv, ACCELERATORS["mi300x-run2"], c, CONTEXT)
    print(f"mi300x-run2, BF16 KV: c = {c:>2}: <= {1 / step.seconds:5.1f} tok/s per user, "
          f"<= {c / step.seconds:6.0f} tok/s per GPU")

# The claim's point at TP = 1: the largest batch whose step stays inside 1/50 s, on a
# card reading at 100 % of its peak bandwidth, which no run has measured.
peak = replace(ACCELERATORS["mi300x"], achieved_bandwidth=1.0)
batch = max(c for c in range(1, 257)
            if tpot_floor(LLAMA_70B_FP8, peak, c, CONTEXT).seconds <= 1 / 50)
print(f"at 100 % of peak bandwidth, 50 tok/s per user holds up to batch {batch}: "
      f"<= {batch * 50:,} tok/s per GPU against the claim's 1,482")
