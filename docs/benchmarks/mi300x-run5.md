---
card: mi300x
run: 5
date: 2026-10-17
asked: What makes two engines on one card equal, and does a 70B obey run 2's coefficient?
found: one hardware queue per process ends the asymmetry, and the 70B's decode step held to run 2's eff_mem at TP = 1
predicted_with: mi300x-run2
---

# MI300X run 5 — 2026-10-17

Two engines of one model on one card, first with the card's compute units split
between them, then with each process held to one hardware queue. After that,
InferenceX's Llama-3.3-70B FP8 server line at TP = 1, faced with a ceiling
printed before the card. Every prediction the run faced was written in
`docs/benchmarks/runsheets/mi300x-run-5.md` before the droplet existed, and the
scripts that ran were committed beside it.

Raw evidence, verbatim: `docs/benchmarks/raw/mi300x-2026-10-17/run5/`. The
arithmetic behind every figure below: `python3 bench/measured_mi300x_run5.py`.
Where a number here disagrees with a derivation elsewhere in the repo, **this
file wins for this droplet and only for it** (`docs/SLO.md` §9).

**The run in five sentences.** The runsheet's definition of done was met, and
B2 and c = 16 ran as well:

- **Run 4's asymmetry came back on a new droplet**: the engine launched first
  served 51.19 ms per token against 26.69, a ratio of 1.92 (§3).
- **With one hardware queue per process it was gone.** Both engines read
  22.21 and 22.19 ms, both calls lasted 43.4 s, and the pair produced the most
  tokens of the three steps. In no call did the kernel report an oversubscribed
  runlist, so the sheet's candidate for why queues would matter was not the
  mechanism (§3).
- **Splitting the CUs only partly helped.** The MQDs show the mask on the
  hardware, 19 CUs per XCD for each engine and disjoint. The ratio by median ITL
  fell to 1.38, but by TPOT it hardly moved, 1.29 → 1.25, and the pair lost 19 %
  of its throughput (§3).
- **The 70B's decode step obeyed run 2's coefficient.** The floor fell short of
  the median ITL by 0.3 % at c = 16 and by 6.3 % at c = 4, from a coefficient
  fitted on Qwen3-8B BF16. No level crossed the ceiling, and TP = 1 never came
  near InferenceX's 50 tok/s per user (§5).
- **What one user sees fell far below that step, through prefill rather than
  bandwidth.** At c = 16 the median TPOT read 87.2 ms against a 31.0 ms step.
  Gaps over three steps long were 1.3 % of the gaps and 64 % of the decode
  time. The 131 072-token budget let each step carry whole 7 400-token prompts.
  It also cost 28.4 GiB of activations, and the pool came out 38 % under the
  ceiling (§4, §5).

---

## 1. What was run

| | |
|---|---|
| Accelerator | 1 × AMD Instinct MI300X VF, gfx942, 205 822 885 888 B reported by `rocm-smi` |
| Provider | AMD Developer Cloud, on-demand, **$1.99/h** on the create page, paid from credits |
| Host | *Quick Start → ROCm Software* 7.14 on Ubuntu 24.04. Kernel `6.8.0-138-generic`, amdgpu `6.19.14`, Intel Xeon Platinum 8568Y+, 20 vCPUs: the same record as runs 2–4. `sched_policy` 0 (HWS), `hws_max_conc_proc` −1 |
| Image | `vllm/vllm-openai-rocm:v0.27.1`, digest `sha256:bb44b39a…f0e7`, `vllm --version` 0.27.1+rocm723 |
| Block B model and server line | `Qwen/Qwen3-8B` BF16, `--gpu-memory-utilization 0.45 --max-model-len 9000 --max-num-batched-tokens 2048` on ports 8000 and 8001, prefix caching on, `VLLM_SERVER_DEV_MODE=1`; B1 adds `HSA_CU_MASK`, B2 `GPU_MAX_HW_QUEUES=1` |
| Block B load | `fleet-bill-half` on both engines at once: 96 unique 4 000-token prompts at c = 32 per engine, 200 out, after a cache reset |
| Block C model and server line | `amd/Llama-3.3-70B-Instruct-FP8-KV`, TP = 1, InferenceX's line with the sheet's three changes (§4 of the sheet): `--dtype float16 --kv-cache-dtype fp8 --gpu-memory-utilization 0.94 --max-model-len 9216 --max-num-seqs 64 --max-num-batched-tokens 131072 --no-enable-prefix-caching`, three `custom_ops` off, AITER off |
| Block C load | `vllm bench serve --ignore-eos`, 7 373 ± 11.1 % in and 922 ± 11.1 % out, 5 × c prompts at c ∈ {4, 16, 64}, `--seed 0` |
| Failures | zero failed requests in every call; zero preemptions in every call except c = 64, which had 31 |
| Clock | droplet created ≈ 14:10 UTC, load 14:17–14:55, harvest 14:55–14:57, destroyed after it; ≈ 0.8 h, ≈ $1.6 against ≈ $3.2 expected. The invoice outranks this figure |

The droplet check read **5.02 ms** at c = 1, against 5.04 on runs 3 and 4, so
every predicted time uses `FIT = mi300x-run2`. Its TTFT, **154.8 ms**, is B1's
baseline.

---

## 2. Checkpoint A — seven startup logs

| Launch | 8000 | 8001 | Graph |
|---|---|---|---|
| warm step, one engine alone | 492 074 tokens, 67.64 GiB | — | compiled |
| plain pair (B0) | 506 525, 69.62 GiB | 506 525, 69.62 GiB | both loaded |
| masked pair (B1) | 506 525, 69.62 GiB | 506 525, 69.62 GiB | both loaded |
| one queue each (B2) | 506 525, 69.62 GiB | 506 525, 69.62 GiB | both loaded |

The warm step did what the sheet built it for. The engine that compiled the
graph logged run 4's smaller cold pool to the token, and every engine after it
loaded the graph and logged runs 3–4's pool. **The mask changed no
allocation**, as predicted. Dev mode, the `/reset_prefix_cache` route and an
idle `{"success":true}` were present on all six engines of the three pairs.

---

## 3. Block B — what removes the asymmetry

| Step | 8000, median ITL | 8001, median ITL | Ratio by ITL | Ratio by TPOT | Pair, tok/s | Seconds, 8000 / 8001 |
|---|---|---|---|---|---|---|
| B0, plain | **51.19 ms** | 26.69 ms | **1.92** | 1.29 | 863 | 48.9 / 40.8 |
| B1, CUs split | **25.33 ms** | 18.31 ms | **1.38** | 1.25 | 697 | 58.9 / 51.8 |
| B2, one queue each | 22.21 ms | 22.19 ms | **1.00** | 1.00 | 884 | 43.4 / 43.4 |

In every step 8000 was launched first.

**B0 is run 4's pair.** Run 4 read 1.90 and 1.96, and this droplet read 1.92
with the first-launched engine slow. The sheet's first table therefore sends B0
to B1, read against run 4.

**The mask reached the hardware, and the gate saw it.** At c = 1 the masked
8001 read a TTFT of 260.3 ms against 154.8 plain, a ratio of 1.68 against the
gate's 1.5: `MASK_TAKEN`. `env-masked.txt` shows the variable in all six
processes. The host's MQD dump confirms what the sheet's driver reading
predicted. Each engine's compute queues carry 19 CUs on each of the eight XCDs,
the same mask on all five queues, and the two engines' masks are disjoint. Their
union is all 38 CUs of every die. The plain and one-queue dumps carry no mask.

**B1 lands in the sheet's "between" row, and the two medians disagree about
how far.** By median ITL the ratio fell from 1.92 to 1.38. By median TPOT it
was 1.29 plain and 1.25 masked. Both masked engines decoded faster per step
than either plain engine. But each now prefilled on half the CUs: the c = 1
TTFT rose by 68 %. Each engine's own prompts at budget 2 048 interrupt its
decode for longer, so TPOT rose on both ports, and the pair's throughput fell
from 863 to 697 tok/s. Splitting the CUs removes about half of the asymmetry in
the decode step and costs more than it saves.

**B2 removed it.** Held to `GPU_MAX_HW_QUEUES=1`, each process opened two
compute queues instead of five. The two engines then read 22.21 and 22.19 ms
median ITL, 64.30 and 64.31 ms median TPOT, and 442.1 tok/s each. Both calls
ended at 43.4 s. Both engines were faster than the plain pair's fast one, and
the pair produced 884 tok/s against 863.

**What B2 rules out, and what it leaves.** The sheet expected queues to matter
through oversubscription: more queues than the firmware maps at once, and the
processes time-sliced. That did not happen in any call. Every runlist mapped two
processes, with 14 queues plain and masked and 8 with one queue each, and
`dmesg` logged no line during any of the three calls. So the queue count
changes how the command processor divides the card between two processes even
when every queue is mapped. Five compute queues against five, the process that
opened the device first loses. Two against two, neither does. Nothing the run
recorded shows how the command processor arbitrates among mapped queues.

`rocm-smi` read the device 100 % busy at a median of 727–749 W in all three
calls. It is device-wide and cannot say which process held the card.

---

## 4. Checkpoint C — the 70B launch

| Log line | Predicted | Logged |
|---|---|---|
| `GPU KV cache size` | ≤ 739 334 tokens, an over-estimate | **458 016** tokens in 69.89 GiB |
| bytes per token of KV | 163 840, `config.json` at one byte | 163 845, from the two logged figures |
| `kv_cache_dtype` | `fp8` | `fp8` |
| linear kernel, attention | recorded, not gated | `ROCmFP8ScaledMMLinearKernel`, `ROCM_ATTN` |

**The bound held, and the gap is accounted for.** vLLM's startup line splits
the 0.94 budget, 180.19 GiB. Weights plus non-torch memory took 81.87 GiB, of
which the weights were 69.3 GiB. Peak activation at a 131 072-token step took
**28.43 GiB**, and what remained was the KV pool. `ceiling.py` subtracted the
weights alone. The activations are the quantity the sheet named as
unmodelled. At one byte per element the pool's bytes per token match the
derivation to within the log's rounding.

**The pool seats about 55 requests of this load, not 64.** The sheet set the
line at 589 824 = 64 × 9 216 tokens. Below that line c = 64 can preempt, and it
did, 31 times. During c = 64's steady state the engine log reads 55–59
requests running, 5–8 waiting, and the KV cache 98–100 % full. Its ceiling was
written for 64 sequences decoding at once.

---

## 5. Block C — the claim at TP = 1

| c | Floor, eff_mem 0.57 | Median ITL | Median TPOT | tok/s per user, 1 / TPOT | ≤ ceiling | tok/s per GPU | ≤ ceiling | Median TTFT |
|---|---|---|---|---|---|---|---|---|
| 4 | 25.7 ms | **27.49 ms** | 33.5 ms | **29.8** | 38.8 | 110 | 155 | 2.1 s |
| 16 | 30.8 ms | **30.95 ms** | 87.2 ms | **11.5** | 32.4 | 172 | 519 | 4.8 s |
| 64 | 51.2 ms | **55.34 ms** | 180.5 ms | **5.5** | 19.5 | 310 | 1 249 | 21.6 s |

**Every measured figure fell below its ceiling, and the sheet's test of the
coefficient passed.** The test was set at 38.8 tok/s per user at c = 4, and the
measurement read 29.8. The median ITL makes the stronger statement. It is the
decode step here, because steps carrying a prefill were 0.27–4.5 % of all gaps,
well inside its validity condition (`docs/GLOSSARY.md`, *ITL*). Against the
floor built on run 2's `eff_mem` of 0.57, the floor falls short of it by:

- 0.3 % at c = 16, an implied `eff_mem` of 0.568;
- 6.3 % at c = 4, an implied 0.534;
- 7.4 % at c = 64, for a floor drawn for 64 sequences. At the 55–59 the pool
  actually seated, it falls short by 11–14 %, an implied 0.49–0.51.

The engine's own 10-s log lines with no prefill in them agree: 27.5 ms at four
running and 31.8 ms at sixteen. **A coefficient fitted on Qwen3-8B BF16 decode
predicted a 70B FP8 decode step on the same droplet type to within 6.3 %**, up
to the concurrency where the pool filled. At a full pool the KV is half of every
step's bytes, and the floor fell 11–14 % short.

**InferenceX's 50 tok/s per user was not reached at TP = 1, as derived.** The
best per-user figure was 29.8 by TPOT. Even the decode step alone, 1 / median
ITL, gives 36.4 at c = 4, under the 41.6 that batch 1 tops out at on run 2's
coefficient. The claim's 1 482 tok/s per GPU at 50 tok/s per user cannot come
from this card at TP = 1. This run does not say which TP it came from.

**What one user sees is set by prefill, not by the step.** TPOT minus median
ITL is the prefill interference:

| c | TPOT − median ITL | Gaps over 3 × the median | Share of decode time in them | Longest gap |
|---|---|---|---|---|
| 4 | 6.0 ms | 0.27 % | 16.7 % | 4.1 s |
| 16 | 56.3 ms | 1.31 % | 63.8 % | 15.4 s |
| 64 | 125.1 ms | 4.46 % | 69.8 % | 54.8 s |

One prompt alone prefilled at ≈ 5 400 tok/s: 7 948 tokens in 1.47 s, the first
request of the c = 4 and c = 16 levels. The waves that followed it ran at the
same rate, 21 984 tokens in 4.15 s and 118 085 in 22.0 s. Under a budget of
131 072 tokens a 7 400-token prompt is never chunked. Every seat that turns
over stops everyone else's decode for about 1.4 s per prompt. The client
starts all c seats at once, and their output lengths sit within ±11 % of one
another. So the seats turn over in waves, and the stops arrive together. At
c = 16, gaps of a few seconds were 1.3 % of the gaps and 64 % of the time spent
decoding. At c = 64 the queue adds the rest. Requests waited for a seat, and
the median TTFT reached 21.6 s.

**At TP = 1, the server line is the problem, not the card.** InferenceX tuned
the line for a sweep up to TP = 8. On one card the same budget costs 28.4 GiB of
pool and multi-second decode stalls. Whether a smaller budget gives both back
on this load is the next run's question. This run did not measure it (§8).

---

## 6. What changes, and where it goes

| Finding | Goes to |
|---|---|
| Two processes sharing an MI300X are equal at `GPU_MAX_HW_QUEUES=1` and unequal by launch order at the default, with no oversubscription | `docs/runbook.md`, beside any deployment that places two engines on one card; the mechanism stays open (§8) |
| Splitting the CUs with `HSA_CU_MASK` halves the decode-step asymmetry, slows each engine's prefill by 68 % at c = 1, and costs the pair 19 % of its throughput | the same place, as the option measured and not taken |
| `eff_mem` 0.57 predicted a 70B FP8 decode step to within 6 % at c ≤ 16 | `docs/SLO.md` §9, as the coefficient's first test on a second model; the full-pool rows (0.49–0.51) beside it |
| A 131 072-token step budget at TP = 1 reserves 28.4 GiB of activations and stalls decode for about 1.4 s per 7 400-token prompt | `docs/symptom-map.md`: TPOT far above median ITL → whole prompts in one step → `max_num_batched_tokens` |
| The 70B is not in `predictions.MODELS` | a separate change, as the sheet said: with a TTFT floor built on this run's 5 400 tok/s, not on `mfu` 0.247 |

---

## 7. Cost, at $1.99/h

≈ 0.8 h of droplet, ≈ $1.6. The load itself took 38 minutes of it, and block C
26. With runs 1–4, the credits have paid ≈ $12.4 of MI300X time.

---

## 8. What this run could not measure

**How the command processor divides the card among mapped queues.** B2 changed
the count, and the asymmetry went with it. Neither the runlist, nor the MQDs,
nor `dmesg` shows dispatch, and `rocm-smi` is device-wide. Per-process kernel
timing needs `rocprofv3`, as the sheet's §7 says.

**What one queue costs a process alone.** B2 ran the pair only. Its 22.2 ms
step is faster than either plain engine together, but no engine ran alone at
`GPU_MAX_HW_QUEUES=1`. B2 also ran in one launch order.

**Whether a smaller step budget restores the 70B's per-user rate.** The
interference in §5 is measured. Its cure at 8 192 or 16 384 tokens, with the
pool it would return, is not.

**The prefill rate as a coefficient.** The 5 400 tok/s in §5 comes from the
TTFTs of three waves, not from a sweep of prompt lengths. The engine log's
10-s averages read up to 13 185 tok/s, but a single 131 072-token step is as
long as the window that averages it, so the log cannot isolate the rate.

**The `top -H` record** of block B is kept and was not read here. Run 4 ruled
the CPU out.

---

## 9. Predicted vs measured — the summary table

Rows the sheet wrote as a test or a bound are marked so and carry no error.
Error is predicted over measured, less one.

| Quantity | Predicted | Measured | Error | Coefficient used | Fitted to this run? | Card |
|---|---|---|---|---|---|---|
| KV pool, Qwen3-8B at 0.45, each engine of three pairs, tokens | 506 525 (runs 3–4) | 506 525 × 6 | 0.00 % | none, a logged figure | no | MI300X |
| KV pool, masked launch | 506 525: a mask changes no allocation | 506 525 / 506 525 | 0.00 % | none | no | MI300X |
| Droplet check, median ITL at c = 1 | 5.04 ms (runs 3–4) | 5.02 ms | +0.40 % | none, a measured figure | no | MI300X |
| B0, first launched over the other, median ITL | 1.90–1.96 (run 4) | 1.92, 8000 slow | inside | none, a measured figure | no | MI300X |
| Masked c = 1 TTFT over plain | ≈ 2; gate ≥ 1.5, a test | 1.68 | test passed | none | no | MI300X |
| CUs per XCD per engine, masked | 19 of 38, disjoint | 19 / 19 on all eight, disjoint | held | none (KFD source) | no | MI300X |
| "Runlist is getting oversubscribed" in B0 | the sheet's candidate, not a prediction | absent in all three calls | — | none | no | MI300X |
| B1 ratio, median ITL | ≤ 1.10 / ≥ 1.6 / between, a decision table | 1.38 | between | none | no | MI300X |
| B2 ratio, median ITL | towards 1 if the queues carry it, a test | 1.00 | test passed | none | no | MI300X |
| KV pool, 70B at 0.94, tokens | ≤ 739 334, a bound | 458 016 | +61.4 %; bound held | none (weights only, no activations) | no | MI300X |
| `kv_cache_dtype` | `fp8` | `fp8` | held | none | no | MI300X |
| Decode step, median ITL, c = 4 | ≥ 25.7 ms | 27.49 ms | −6.3 % | `eff_mem` 0.57 (mi300x-run2) | no | MI300X |
| Decode step, median ITL, c = 16 | ≥ 30.8 ms | 30.95 ms | −0.3 % | `eff_mem` 0.57 (mi300x-run2) | no | MI300X |
| Decode step, median ITL, c = 64 (55–59 seated) | ≥ 51.2 ms | 55.34 ms | −7.4 % | `eff_mem` 0.57 (mi300x-run2) | no | MI300X |
| tok/s per user, c = 4 / 16 / 64 | ≤ 38.8 / 32.4 / 19.5, bounds | 29.8 / 11.5 / 5.5 | bounds held | `eff_mem` 0.57 | no | MI300X |
| tok/s per GPU, c = 4 / 16 / 64 | ≤ 155 / 519 / 1 249, bounds | 110 / 172 / 310 | bounds held | `eff_mem` 0.57 | no | MI300X |
| `eff_mem` transfers to the 70B: c = 4 not above 38.8 tok/s per user | a test | 29.8 | test passed | `eff_mem` 0.57 | no | MI300X |
| 50 tok/s per user unreachable at TP = 1 | derived: ≤ 41.6 at batch 1 | best 29.8 by TPOT, 36.4 by ITL | held | `eff_mem` 0.57 | no | MI300X |

**Every row that predicted a memory figure or a decode step held, from a
coefficient fitted on another model.** The one bound that missed by a margin is
the 70B pool. It held as a bound, and its gap is the activation term the sheet
said it left out. The rows that decide block B were tests, and they read: the
mask reached the hardware, the queues carry the asymmetry, and oversubscription
did not.
