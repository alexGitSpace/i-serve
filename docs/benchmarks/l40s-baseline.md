---
card: l40s
run: 1
date: 2026-08-18
asked: What does the card really deliver against the spec-sheet coefficients?
found: eff_mem 0.83 and mfu 0.439 replace the guesses, and TPOT p99 allows 12 seats where the arithmetic said 32
predicted_with: l40s
fitted: l40s-run1
---

# L40S baseline — run 1, 2026-08-18

The first measurements this repository has. Everything in `docs/SLO.md` was a
prediction until this run faced it; this file is where the predictions were
scored and where two empirical coefficients stopped being guesses.

Raw evidence, verbatim and uninterpreted: `docs/benchmarks/raw/l40s-2026-08-18/`.
The checklist the run followed: `docs/benchmarks/runsheets/l40s-first-run-card.md`,
the on-the-clock extract of `docs/benchmarks/runsheets/l40s-first-run.md`, which
is the sheet that carries dates: its decisions are marked *checked*, *settled*
and *done* on 2026-08-15, three days before the card was rented. Neither sheet was revised against the results below. The
ordering rests on those datelines rather than on anything in this tree that could
prove it: the working papers are attached so the claim can be judged, not taken.
Where a number here disagrees with a derivation elsewhere in the repo, **this
file wins for the L40S and only for the L40S** — a coefficient belongs to a card
and a phase (`docs/SLO.md` §9).

---

## 1. What was run

| | |
|---|---|
| Accelerator | 1 × NVIDIA L40S, 46 068 MiB reported, driver 580.159.04 / CUDA 13.0 |
| Provider | RunPod, on-demand, **$0.99/h**, North American DC |
| Image | `vllm/vllm-openai:v0.27.1`, fingerprint `vllm-0.27.1-d58650c8` |
| Model | `Qwen/Qwen3-8B`, `--dtype auto` → BF16, FlashAttention 2 backend |
| Server flags | `--gpu-memory-utilization 0.90 --max-model-len 9000 --no-enable-prefix-caching` |
| Load | `vllm bench serve`, `--dataset-name random --random-range-ratio 0 --ignore-eos --request-rate inf` |
| Levels | concurrency 1 … 45 at 4 000-token input; input 2 000 / 4 000 / 8 000 at concurrency 4 |
| Output length | 200 tokens, fixed, EOS ignored |
| Failures | zero, at every level |
| Spend | $1.45 against a $2.50 budget |

Twelve levels, 12 JSON results, one 162-sample metrics trace at the top level.

---

## 2. Checkpoint A — the KV pool, measured before any load

vLLM prints the pool it actually carved. That log outranks the derivation
(`docs/SLO.md` §9), and this is the row that says by how much.

| | Tokens |
|---|---|
| Derived: `(48e9 × 0.9 − 16.4e9) / 147 456` | 181 749 |
| **Logged: `GPU KV cache size`** | **168 985** |
| Error of the derivation | **+7.6 %** (7.0 % of the measured value) |

The 12 764-token gap is not noise. It is four terms the derivation ignores, and
they account for it almost exactly:

```
nameplate 48 GB vs 44.39 GiB actually visible to the driver  →  2 054 tokens
non-torch memory beyond the weights, 0.27 GiB                →  1 966
peak activation memory, 1.18 GiB                             →  8 592
CUDA graph pool, 0.57 GiB                                    →  4 151   (partly inside the above)
```

Cross-check from the other side: the log's own `kv cache memory in use is
23.23 GiB` is 169 156 tokens, against 168 985 reported — the remainder is
rounding to `block_size`.

**What this changes.** `gpu_memory_utilization` is not a fraction of the KV
cache; it is a fraction of the *card*, out of which the weights, the activation
peak and the CUDA graphs are paid first. A capacity ceiling derived without those
three terms is optimistic by roughly 7 % on this card — small enough to look
right and large enough to schedule a sequence that does not fit.

**And it was confirmed under load.** At concurrency 45 the in-pod sampler read a
maximum `num_requests_running` of **41**, `kv_cache_usage_perc` of **0.999**, and
`num_preemptions_total` of **4**. The measured pool divided by the operating
context, 168 985 / 4 100, is **41.2**. The engine stopped exactly where the
corrected arithmetic says the shelf ends, and the surplus requests became
preemptions rather than throughput.

---

## 3. Calibrating `achieved_bandwidth` — the decode step

### Read the median ITL, not the TPOT

TPOT is the mean inter-token latency of a request. Under chunked prefill some
decode steps also carry a slice of somebody else's prefill, and those steps are
several times longer. The mean absorbs them; the **median ITL** does not, which
makes it the closest thing this benchmark reports to a pure decode step.

The distance between the two is not an artefact to be discarded — §5 spends it —
but the physics has to be calibrated against the median, or the coefficient
absorbs a scheduling decision.

### The fit

Predicted at 100 % of peak bandwidth:
`(16.4e9 + n × context × 147 456) / 864e9`, with `context` = input + 100 (the
mean context over a 200-token generation) and `n` the number of sequences the
engine actually ran.

| Level | ctx | Predicted @100 % | Median ITL | **Implied `eff_mem`** |
|---|---|---|---|---|
| c=1, in 4 000 | 4 100 | 19.68 ms | 22.70 ms | **0.867** |
| c=4, in 2 000 | 2 100 | 20.42 ms | 24.73 ms | **0.825** |
| c=4, in 4 000 | 4 100 | 21.78 ms | 26.32 ms | **0.827** |
| c=4, in 8 000 | 8 100 | 24.51 ms | 29.52 ms | **0.830** |
| c=8 | 4 100 | 24.58 ms | 29.65 ms | **0.829** |
| c=13 | 4 100 | 28.08 ms | 33.80 ms | **0.831** |
| c=14 | 4 100 | 28.78 ms | 34.60 ms | **0.832** |
| c=16 | 4 100 | 30.18 ms | 36.30 ms | **0.831** |
| c=23 | 4 100 | 35.08 ms | 43.07 ms | **0.814** |
| c=24 | 4 100 | 35.78 ms | 43.96 ms | **0.814** |
| c=32 | 4 100 | 41.37 ms | 50.69 ms | **0.816** |
| c=45 (ran 41) | 4 100 | 47.67 ms | 59.66 ms | **0.799** |

**`eff_mem = 0.83` on the L40S, ±0.02 across the whole sweep.**

Three things that fit is worth more than the number itself:

1. **One coefficient closes both sweeps.** Concurrency moves `n`, prompt length
   moves `context`; they enter `bytes_moved` as a product but arrive from
   different knobs, and a single scalar reconciles all twelve rows. The model was
   right in *shape* — only its constant was wrong.
2. **0.70 was pessimistic by 19 %**, and the coefficient is a divisor, so the
   error passes straight through. It moved `max_num_seqs_from_slo` at 4 000
   tokens from 23 to 32 — a 39 % difference in seats, on a card whose price is
   fixed.
3. **The direction stated in `docs/SLO.md` §9 was inverted.** That section
   expected `eff_mem` to be *worst* at batch 1, where launch and attention
   overhead amortise over nothing. Batch 1 measured **0.867**, the best point in
   the table, and efficiency declines slowly with batch (0.83 in the middle,
   0.80 at the ceiling). The reason is visible once stated: at batch 1 the step
   is dominated by one long streaming read of contiguous weights, which is the
   easiest possible access pattern; adding sequences adds *scattered* KV blocks,
   which are not.

### The residual: how the model still misses

The decline from 0.867 to 0.799 is the part one scalar cannot express. Fitting
the weights read and the KV read as separate efficiencies gives roughly 0.87 for
the streaming weights term and 0.80 for the paged KV term — but a two-parameter
fit to twelve points from one run is over-fitting, and `bench/roofline.py`
deliberately keeps one scalar per accelerator until a second run contradicts it.
What is recorded here is the shape of the residual, not a second coefficient.

---

## 4. Calibrating `mfu` — prefill

`mfu = 0.45` was the assumption this repository started from, taken from general
knowledge and never measured on anything in it.

Measured on the only level with no prefill contention — one request, alone on the
card:

```
FLOP      = 2 × 6.95e9 × 4 000                  = 5.56e13
Predicted = 5.56e13 / 362.05e12 ÷ 0.45          = 341.3 ms
Measured  = median TTFT at c=1, 4 000 tokens    = 350.1 ms
Implied mfu                                     = 0.439
```

**2.5 % from the assumption.** The coefficient survives, and it is now measured
rather than borrowed — on one point, at one prompt length, on this card.

### Prompt length: linear, as predicted

Concurrency fixed at 4:

| Input | TTFT floor @ `mfu` 0.45 | Measured median TTFT | Ratio to floor |
|---|---|---|---|
| 2 000 | 170.6 ms | 391.7 ms | 2.30 |
| 4 000 | 341.3 ms | 674.8 ms | 1.98 |
| 8 000 | 682.5 ms | 1 313.2 ms | 1.92 |

Slope 0.1536 ms/token, and the ratio to the floor is flat rather than climbing —
the runsheet's stop condition ("if the ratio itself climbs steeply, stop here")
did not fire. **TTFT is linear in prompt length**, on real silicon, through two
doublings and across the `max_num_batched_tokens = 2 048` chunk boundary.

The constant offset is the price of sharing: the isolated slope at c=1 is
0.0875 ms/token, so four concurrent prefills cost each request 1.75× the per-token
rate. That is contention, not a change of regime.

### And TPOT barely moved

Over the same 4× in prompt length, median ITL went 24.73 → 29.52 ms, +19 %, all
of it accounted for by the KV term in `bytes_moved`. TTFT over the same range
went ×3.35.

**Prompt length is a TTFT knob. It reaches TPOT only through the KV share of the
traffic, and on this card that share is small.** Settled with evidence rather
than with an explanation.

---

## 5. The 50 ms line — three different answers, and only one of them ships

The interactive TPOT target is 50 ms p99 (`docs/SLO.md` §2). Where the sweep
crosses it depends entirely on which quantity is asked:

| Asked of | Crosses 50 ms at |
|---|---|
| Derivation, `eff_mem = 0.70` (what the runsheet predicted) | 23 / 24 |
| Derivation, `eff_mem = 0.83` (calibrated here) | 32 / 33 |
| **Measured decode step** (median ITL) | **~31** — 43.07 ms at c=23, 50.69 ms at c=32 |
| **Measured TPOT p50** | **13 / 14** — 48.62 ms, then 51.36 ms |
| **Measured TPOT p99** — the SLO metric | **~12** — already 50.87 ms at c=13 |

Once calibrated, the roofline predicts the decode step to within 3 % (32
predicted against ~31 measured). It is a good model of the hardware.

It is not a model of the service. The number an operator can promise is **12**,
2.6× lower, and the whole difference is the engine's scheduler, not the memory
bus. The evidence is the ratio of TPOT to median ITL as load rises:

| c | 1 | 4 | 8 | 13 | 16 | 23 | 32 | 45 |
|---|---|---|---|---|---|---|---|---|
| TPOT p50 / median ITL | 1.00 | 1.10 | 1.28 | 1.44 | 1.53 | 1.72 | 1.91 | 2.06 |

At concurrency 1 there is nobody to interfere and the two are identical. By
concurrency 45 the average decode step a request experiences is twice the length
of the step the hardware performs. The gap is chunked prefill: with
`max_num_batched_tokens = 2 048`, every arriving request injects 2 048-token
prefill chunks into the decode stream, and each injection stretches that step for
**every** sequence in the batch.

**The knob the gap names is `max_num_batched_tokens`**, with
`long_prefill_token_threshold` beside it. Smaller chunks damage a decode step
less and delay prefill more — the TTFT-for-TPOT trade, now with a measured price
on both sides. Run 2 tests it.

---

## 6. What ships

`max_num_seqs = min(capacity, latency)`, both terms measured rather than
derived, at 4 000-token context:

```
capacity :  168 985 / 4 100                     =  41
latency  :  TPOT p99 ≤ 50 ms, measured          =  12      ← binds
```

**The card was booted with the vLLM default, `max_num_seqs = 256`**, which is
what produced the concurrency-45 level: the engine admitted requests until the
KV pool was full, ran 41, queued 39 more, and preempted 4. TTFT p99 at that level
was 23 955 ms. A default is not a decision, and 256 on this card means "admit
until something breaks".

| Class | `max_num_seqs` | Bound by | Note |
|---|---|---|---|
| Interactive, 4 000-token prompts | — | — | unreachable, see below |
| Interactive, short prompts | 12 | latency | needs the TTFT re-measurement of run 2 |
| Batch (200 ms TPOT) | 41 | capacity | TPOT p99 is 132.57 ms at the ceiling, inside the target |

### The interactive class does not fit this card at 4 000 tokens

Not marginally — arithmetically, before any queue exists:

```
prefill floor, 4 000 tokens = 341 ms   >   the entire 300 ms TTFT budget
```

Inverting it gives the longest prompt the budget can hold at zero queue:

```
300 ms × 362.05e12 × 0.45 / (2 × 6.95e9)  ≈  3 516 tokens
```

3 516 tokens spends the whole budget on computation and leaves nothing for
queueing. A workload that must also survive a queue needs prompts well under
that — roughly 1 500 tokens to leave half the budget for waiting. **On the L40S,
the interactive SLO is a statement about prompt length before it is a statement
about concurrency.**

---

## 7. Cost, at $0.99/h

| c | Output tok/s | $/1M output tokens | $/1M total tokens |
|---|---|---|---|
| 1 | 41.1 | 6.688 | 0.3185 |
| 8 | 192.7 | 1.427 | 0.0680 |
| **13** — last level inside 50 ms TPOT p50 | 244.2 | **1.126** | 0.0536 |
| 23 | 293.6 | 0.937 | 0.0446 |
| **32** — throughput optimum | 314.9 | **0.873** | **0.0416** |
| 45 | 311.6 | 0.883 | 0.0420 |

Three readings:

**Throughput saturates and then reverses.** Marginal output per added sequence
falls from 27.5 tok/s at the bottom of the curve to 2.3 tok/s over 24→32, and to
**−0.3 tok/s** over 32→45. The last thirteen sequences bought negative
throughput and a 23-second TTFT tail; past the capacity ceiling, admitting more
work makes the card slower. This is the shape `docs/SLO.md` §5 predicts, measured.

**Holding the latency SLO costs 29 %.** $1.126 per 1M output tokens at the
SLO-respecting point against $0.873 at the throughput optimum. That is the price
of the promise, and it is now a number rather than an argument.

**A full card is 7.7× cheaper per token than one user on it.** $0.0416 per 1M
total tokens at c=32 against $0.3185 at c=1 — the same card, the same model, the
same hour of rent. (An earlier draft of this paragraph said 21×; that is
6.688 / 0.3185, the ratio of output to total tokens at *one* level, not a ratio
between levels — at a fixed geometry the factor between the two denominators is
the same 21 on every row, so the c=1 : c=32 ratio is 7.66 whichever column is
read.) Which is the entire argument for separating the two workload classes
(`docs/SLO.md` §1) rather than serving both from one pool: the interactive class
pays its 29 % at c=13, and only a pool that never has to hold a latency promise
reaches c=32.

---

## 8. What run 1 could not measure

**TTFT against its SLO.** `--request-rate inf` is a closed loop: all N requests
of a level are released at once and a new one starts only as one finishes. Every
TTFT after the first wave therefore contains queue time created by the load
generator, not by the service. That is why TTFT p99 reads 364 ms at concurrency 1
and 23 955 ms at concurrency 45, and why none of those numbers may be compared to
a 300 ms target. **A finite `--request-rate` sweep is the first requirement of
run 2.**

**Decode MFU.** Still unmeasured, and still assumed at the prefill value of 0.45
inside `tpot_floor`. Harmless — it flatters the losing side of a roofline that
memory wins by two orders of magnitude — but unvalidated, exactly as `docs/SLO.md`
§9 says.

**`mfu` at more than one prompt length.** One clean point, at 4 000 tokens. The
prompt-length sweep ran at concurrency 4, where contention is in the number.

**Anything about FP8 KV, prefix caching, or a second replica.** Out of scope by
design; `--no-enable-prefix-caching` was set precisely so this run measures the
uncached path.

**Anything about MI300X.** No coefficient measured here transfers to it.

---

## 9. Predicted vs measured — the summary table

The format `docs/SLO.md` §9 requires: every row names the coefficient that
produced its prediction, whether that coefficient was fitted to this same
measurement, and the accelerator.

| Quantity | Predicted | Measured | Error | Coefficient used | Fitted to this run? | Card |
|---|---|---|---|---|---|---|
| KV pool, tokens | 181 749 | 168 985 | +7.6 % | none (capacity arithmetic) | no | L40S |
| Sequences at 4 000 ctx, capacity | 45 | 41 | +9.8 % | none | no | L40S |
| Decode step, c=1, 4 100 ctx | 28.12 ms | 22.70 ms | +23.9 % | `eff_mem` 0.70 | no | L40S |
| Decode step, c=1, 4 100 ctx | 23.71 ms | 22.70 ms | +4.5 % | `eff_mem` 0.83 | **yes** | L40S |
| Decode step, c=32, 4 100 ctx | 59.10 ms | 50.69 ms | +16.6 % | `eff_mem` 0.70 | no | L40S |
| Decode step, c=32, 4 100 ctx | 49.85 ms | 50.69 ms | −1.7 % | `eff_mem` 0.83 | **yes** | L40S |
| TTFT, c=1, 4 000-token prompt | 341.3 ms | 350.1 ms | −2.5 % | `mfu` 0.45 | no | L40S |
| TTFT slope in prompt length | linear | linear, 0.1536 ms/tok at c=4 | — | `mfu` 0.45 | no | L40S |
| `max_num_seqs` by latency, 50 ms | 23 | ~31 (decode step) | −26 % | `eff_mem` 0.70 | no | L40S |
| `max_num_seqs` by latency, 50 ms | 32 | ~31 (decode step) | +3 % | `eff_mem` 0.83 | **yes** | L40S |
| `max_num_seqs` by latency, 50 ms | 32 | 12 (TPOT p99) | +167 % | `eff_mem` 0.83 | **yes** | L40S |

The four rows marked *fitted* prove nothing on their own — they compare a
prediction against the measurement it was tuned to. They become evidence only
when run 2 faces them without having been used to produce them. The two rows that
already count are the `mfu` row and the last row, and the last row is the one
that matters: **the calibrated model is right about the hardware and wrong about
the service by 2.6×, and the residual is a scheduler.**
