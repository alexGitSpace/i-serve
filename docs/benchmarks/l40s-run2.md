---
card: l40s
run: 2
date: 2026-08-23
asked: Do run 1's coefficients hold on another pod, under real arrivals, other chunk sizes and FP8 KV?
found: eff_mem 0.83 held to 0.01 %, FP8 KV doubles the pool exactly, and past the goodput peak a useful token costs 15× more
predicted_with: l40s-run1
---

# L40S run 2 — the scheduler, the arrival rate, and FP8 KV

Run 1 calibrated the card. This run is the first that faced those coefficients
without having produced them, which is the only way `docs/SLO.md` §9 allows a
fitted number to become a result. It also asked three questions run 1 could not:
what TTFT does under a real arrival process, what `max_num_batched_tokens`
actually buys, and whether FP8 KV doubles the pool.

Raw evidence: `docs/benchmarks/raw/l40s-2026-08-23/`. The checklist the run
followed: `docs/benchmarks/runsheets/l40s-run-2-card.md`; its reasoning and the
predictions written before the card was rented:
`docs/benchmarks/runsheets/l40s-run-2.md`, never revised against the results
below, and dated by itself rather than by anything in this tree.
Where a number here
disagrees with a derivation elsewhere in the repo, **this file wins for the L40S
and only for the L40S** (`docs/SLO.md` §9).

**The one-line result: `eff_mem = 0.83` survived.** Six independent launches on a
different pod, four chunk sizes and two attention backends put the decode step at
c=13 within 1.1 % of each other, and their mean lands on the prediction to 0.01 %.
Everything else in this file is what happened around that.

---

## 1. What was run

| | |
|---|---|
| Accelerator | 1 × NVIDIA L40S, 46 068 MiB reported, driver **550.127.05** / CUDA 13.0 |
| Provider | RunPod, on-demand, $0.99/h, North American DC, pod `70700baa4cfa` |
| Image | `vllm/vllm-openai:v0.27.1`, `vllm --version` 0.27.1 |
| Model | `Qwen/Qwen3-8B`, `--dtype auto` → BF16 |
| Server flags | `--gpu-memory-utilization 0.90 --max-model-len 9000 --no-enable-prefix-caching`, `max_num_seqs` left at the default 256 |
| Structure | vLLM **not PID 1** — the container slept and eight servers were launched by hand, which is what made four `max_num_batched_tokens` settings affordable in one pod |
| Block A | open loop, input 1 500, `--request-rate` 0.5 … 4.0, `--burstiness 1.0`, no `--max-concurrency`, `--goodput ttft:300 tpot:50` |
| Block B | closed loop, input 4 000, c = 13 and 32, `max_num_batched_tokens` 512 / 1 024 / 2 048 / 4 096, plus c = 16 at 512 |
| Block C | closed loop, input 4 000, `--kv-cache-dtype fp8`, c = 13 / 26 / 32, plus two BF16 controls |
| Output length | 200 tokens, fixed, EOS ignored |
| Failures | **1 of 3 026 requests**, at 3.0 req/s, `ServerDisconnectedError` |
| Spend | ≈ $1.44 against a $2.50 budget, 87 minutes of 150 |

22 levels, 8 server configurations, one 4 105-sample metrics trace at 1 Hz across
the whole session.

---

## 2. Checkpoint A′ — the gate fired, and the gate was wrong

The run opened on the same read run 1 took, as a gate: if the KV pool differs,
nothing may be compared to run 1. The tolerance written into the card was
**±500 tokens**.

| Line | Predicted from run 1 | Measured | |
|---|---|---|---|
| `GPU KV cache size` | 168 985 ± 500 | **169 833** | +848, **+0.50 %** |
| `kv cache memory in use` | 23.23 GiB | 23.34 GiB | +0.47 % |
| `Maximum concurrency for 9 000 tokens` | 18.8× | 18.87× | — |
| `max_num_batched_tokens` | 2 048 | 2 048, now passed explicitly | — |
| `max_num_seqs` | 256 | absent from `non-default args` ⇒ default 256 | — |
| attention backend | FlashAttention 2 | `FLASH_ATTN` | — |
| `enable_prefix_caching` | False | False | — |

**The gate fired and the run continued, on three grounds recorded before the next
command was typed.** The bytes-per-token arithmetic is exact — 23.34 GiB /
169 833 = 147 566 B = 144 KiB, against 36 layers × 8 KV heads × 128 × 2 × 2 B =
147 456 B, so the geometry is identical. The difference lives in the free memory
at startup, not in the cache. And the one number the later blocks depend on does
not move: seats at 4 100 context are 169 833 / 4 100 = 41.4 → **41**, exactly run
1's.

**Later in the same run the gate was shown to be indefensible.** Two further
launches of the *identical* configuration reported 176 994 and 169 833 tokens —
a 4.2 % spread on one pod, one image, one flag set, with only the `torch.compile`
cache warm or cold between them. A ±0.3 % gate sits inside the platform's own
run-to-run variation. The 848-token miss was noise, and the tolerance is the
defect. §6 collects this with the rest of what the run learned about its own
instrument.

---

## 3. Block A — the open loop, and the divergence that is the whole point

Run 1 could not compare a single TTFT to a target: `--request-rate inf` is a
closed loop and every TTFT after the first wave contains the load generator's own
backlog (`l40s-baseline.md` §8). Block A is the same card under a Poisson arrival
process at 1 500-token prompts — short enough that the 131.2 ms prefill floor
leaves 169 ms of the 300 ms TTFT budget for queueing, so the question is live.

| Rate | req/s | **goodput** | TTFT p50 | TTFT p99 | TPOT p50 | **TPOT p99** | med ITL | out tok/s | peak conc |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.48 | 0.48 | 175.2 | 272.9 | 25.3 | 27.9 | 24.07 | 96.0 | 8 |
| 1.0 | 0.96 | 0.95 | 181.3 | **314.5** | 27.3 | 33.4 | 25.11 | 192.2 | 16 |
| 1.5 | 1.44 | 1.38 | 185.3 | 375.3 | 31.0 | 37.1 | 26.54 | 287.3 | 22 |
| 2.0 | 1.91 | 1.71 | 194.5 | 437.8 | 35.3 | 43.1 | 28.03 | 382.6 | 29 |
| 2.5 | 2.36 | **1.87** | 207.7 | 502.8 | 41.1 | **51.2** | 30.94 | 471.9 | 37 |
| 3.0 | 2.84 | 0.90 | 235.9 | 626.9 | 54.2 | 67.0 | 36.85 | 567.2 | 57 |
| 4.0 | 3.61 | 0.12 | 461.5 | 1 723.7 | 102.6 | 127.0 | 55.76 | 722.8 | 111 |

**Throughput and goodput separate, and they separate hard.** Output throughput
climbs monotonically over the whole sweep, 96 → 723 tok/s, and never signals
anything. Goodput rises to **1.87 req/s** at an offered 2.5 and then falls off a
cliff: 0.90 at 3.0, 0.12 at 4.0. Between the peak and the top of the sweep the
card emits **1.5× more tokens** while serving **15× fewer requests inside SLO**.
This is the sentence `docs/GLOSSARY.md` has carried from the start, now with a
measured collapse point on this card at this prompt length.

### The three predictions, scored

1. **TPOT p99 crosses 50 ms between 1.5 and 2.0 req/s** — *falsified*. It crosses
   between 2.0 and 2.5 (43.13 → 51.22). The runsheet had stated the direction of
   the expected error in advance: the 1.9× TPOT-to-step multiplier was borrowed
   from 4 000-token prompts, which inject 350 ms of prefill per request against
   131 ms here, so the multiplier should be smaller and the crossing later. It
   was, and it is. The multiplier is not a constant at all — TPOT p99 ÷ median
   ITL runs 1.16 · 1.33 · 1.40 · 1.54 · 1.66 · 1.82 · **2.28** across the sweep.
2. **TTFT p99 crosses 300 ms between 1.5 and 2.5 req/s** — *falsified in the
   other direction*, and this is the unwelcome result. It crosses between 0.5 and
   1.0 (272.9 → 314.5), at an arrival rate a third of the one predicted. The
   corroborating detail is beside it: ITL p99 sits at 122–144 ms while the median
   step is 24–31 ms, at *every* rate including 0.5. Single decode steps are being
   displaced by prefill chunks even on an almost idle card, and the TTFT tail is
   the same phenomenon seen from the other end.
3. **Goodput peaks at 1.2–1.6 req/s and is ~0 by 3.0** — *partly*. The peak is
   higher and later (1.87 at an offered 2.5) and 3.0 still returns 0.90. The
   shape was right; the level was not.

### `eff_mem` under an arrival process

Concurrency is an output here, not an input, so this is a compound test of the
queueing model and the step model together — which is why the runsheet put the
coefficient work in blocks B and C. Taken as such it is still striking:

| Rate | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|---|
| Predicted step at the predicted `n`, ms | 23.70 | 24.74 | 26.07 | 27.84 | 30.28 | 33.91 | 51.25 |
| Measured median ITL, ms | 24.07 | 25.11 | 26.54 | 28.03 | 30.94 | 36.85 | 55.76 |
| Error | −1.6 % | −1.5 % | −1.8 % | −0.7 % | −2.2 % | −8.0 % | −8.1 % |

Within 2.2 % up to the goodput peak, then a one-directional 8 %, the prediction
low every time. The break lands
exactly where the model's assumption does: above 2.5 req/s the server no longer
self-stabilises at the `n` that matches the offered rate, because the pool runs
out of seats. Inverting the measured step for the concurrency it implies gives
3.6 · 6.8 · 11.2 · 15.7 · 24.5 · 42.5 · **100.0** against predictions of
2.5 · 5.7 · 9.7 · 15.1 · 22.5 · 33.5 · 86.3 — the same story.

### The pool ceiling, measured a third way

The in-pod sampler read a maximum `num_requests_running` of **106**, a maximum
`num_requests_waiting` of 5 over block A, and **2** preemptions, with
`kv_cache_usage_perc` touching 0.998. Capacity at this prompt length is
169 833 / 1 600 = 106.1 seats. The engine stopped at 106, queued 5, and preempted
twice; the load generator's own "peak concurrent requests" of 111 is exactly
106 + 5. Three independent counters closing on one number, as at c=45 in run 1.

---

## 4. Block B — `max_num_batched_tokens` buys two seats

Run 1 named this knob from a ratio table and could not price it. The runsheet's
argument, written before the run, was that the knob **redistributes** prefill work
rather than removing it — total prefill is set by the load — so the *mean*
inflation should be invariant and only the *tail* should move.

At c = 13, 4 000-token prompts, run 1's exact geometry:

| MNBT | TTFT p50 | TPOT p50 | **TPOT p99** | med ITL | **ITL p99** |
|---|---|---|---|---|---|
| 512 | 697.5 | 47.26 | **48.00** | 33.99 | **70.63** |
| 1 024 | 640.0 | 49.70 | 50.93 | 33.77 | 114.23 |
| 2 048 | 745.6 | 49.44 | 51.83 | 33.79 | 198.18 |
| 4 096 | 1 079.5 | 49.32 | 53.26 | 33.69 | 364.91 |

At c = 32:

| MNBT | TTFT p50 | TPOT p50 | TPOT p99 | med ITL | ITL p99 |
|---|---|---|---|---|---|
| 512 | **5 080.8** | 80.06 | 80.58 | 79.56 | 87.34 |
| 1 024 | 921.9 | 98.37 | 99.87 | 115.16 | 138.96 |
| 2 048 | 850.3 | 98.71 | 100.43 | 50.73 | 229.58 |
| 4 096 | 1 175.0 | 99.77 | 103.25 | 50.48 | 398.89 |

**Prediction 1 — TPOT p50 invariant within ±15 % — holds at c = 13** (spread
5.2 %) and **breaks at c = 32**, where MNBT 512 sits 19 % below the other three.
The mechanism is visible in the same row: TTFT p50 is 5.1 seconds there. With a
512-token chunk at that load prefill cannot keep up, so it stops competing with
decode and the decode looks fast. The invariance is real but conditional — it
holds while prefill keeps up, and MNBT 512 at c = 32 is the level where it does
not.

**Prediction 2 — ITL p99 monotone in MNBT — holds, and quantitatively.** The
runsheet's upper bound for the worst step was `decode step + chunk compute at
mfu 0.439`, ordered 77.5 : 122.2 : 211.8 : 390.9, and expected to over-count by
about 20 % because a step overlaps the two:

| MNBT | 512 | 1 024 | 2 048 | 4 096 |
|---|---|---|---|---|
| Bound as published, ms | 77.5 | 122.2 | 211.8 | 390.9 |
| Measured ITL p99, ms | 70.63 | 114.23 | 198.18 | 364.91 |
| Bound is high by | +9.7 % | +7.0 % | +6.9 % | +7.1 % |

**Not 20 % but 5–10 %, and flat across an eightfold range of chunk size.** The
worst decode step under chunked prefill *is* the decode step plus the chunk's
compute, to within a small constant. That is the sharpest predictive result of the
run after the coefficient itself, and it is the one that makes `mfu = 0.439`
useful for something other than TTFT.

One correction to the sheet, found while scoring it: the 4 096 row charged 4 083
prefill tokens to a 4 000-token prompt, which no prompt has. The correct bound
there is 383.6 ms and the over-count is **+5.1 %**, not +7.1 %. `bench/measured_run2.py`
computes `min(mnbt − 13, 4000)` and prints the corrected figure; the table above
keeps the published one, because a prediction is scored as it was written.

**Prediction 4 — TTFT p50 ordered 512 > 1 024 > 2 048 > 4 096 — is falsified**,
and the same arithmetic says why. The runsheet's TTFT lower bound is
`chunks × worst step`:

| MNBT | 512 | 1 024 | 2 048 | 4 096 |
|---|---|---|---|---|
| Lower bound, ms | 697 | 489 | 424 | 391 |
| Measured TTFT p50, ms | 697.5 | 640.0 | 745.6 | 1 079.5 |
| Slack above the bound | +0.1 % | +31 % | +76 % | **+176 %** |

The bound is *tight* at MNBT 512 and useless at 4 096. Read as a decomposition it
is unambiguous: at a small chunk size TTFT is entirely the serialisation of one
prompt's own chunks, and at a large one it is overwhelmingly the wait behind
*other* requests' prefills — which grows with the chunk, because a 4 096-token
chunk blocks the queue for 391 ms at a time. Two different quantities produce the
ordering, and they run opposite ways, so no monotone prediction in MNBT could
have been right.

### Prediction 3 — the block's actual question

Run 1 left the seat count at **12** by TPOT p99, against 31 by the decode step.
The three offered readings were: improves toward 31 (pure redistribution, the
knob is the fix), improves to ~18–22 (part of the gap is prefill nothing removes),
or does not move (run 1's attribution was wrong).

TPOT p99 at MNBT 512 reads 48.00 ms at c = 13 — under the SLO, where MNBT 2 048
reads 51.83 and breaches. So the seat count moved. An extra level was added in-run
to bracket it, with the prediction stated first (`TPOT p99 > 50 ms, estimated
53–55`):

| | c = 13 | c = 16 | 50 ms crossing |
|---|---|---|---|
| MNBT 2 048 | 51.83 | — | **c ≈ 12** (run 1 agrees) |
| MNBT 512 | 48.00 | **54.62** | **c ≈ 13.9** |

**The answer is +2 seats, 12 → 14. Below the lowest of the three offered bands.**
And the gain is not a faster card: median ITL at c = 13 is invariant to 0.9 %
across all four chunk sizes (33.99 / 33.77 / 33.79 / 33.69). Every millisecond of
it comes from removing prefill from the tail — ITL p99 70.63 against 198.18 —
which is exactly the quantity the knob was predicted to move, and exactly as small
as the redistribution argument said it would be.

**The price is on the other side and it is not small.** Holding MNBT at 512
raises TTFT p50 at c = 32 from 850 ms to 5 081 ms, a factor of six. Two seats for
a sixfold TTFT regression is not a trade an interactive service takes.

**What ships from this block:** `max_num_batched_tokens` is not the fix for the
2.6× that run 1 found. The gap is prefill work the load creates, and the knob only
decides who waits for it. The fix is elsewhere — fewer prefill tokens (prefix
caching, shorter prompts) or a pool that does not mix the two phases.

---

## 5. Block C — FP8 KV doubles the pool exactly, and is faster besides

| Quantity | Predicted | Measured | |
|---|---|---|---|
| `GPU KV cache size` | ~338 300 (2.00×) | **339 666** | **2.000×**, exact |
| `Maximum concurrency` at 9 000 | 37.7× | 37.74× | — |
| `kv cache memory in use` | unchanged | 23.34 GiB, identical to BF16 | same bytes, half per token |
| Seats at 4 100 context | 82 | **82** (82.8) | exact; BF16 was 41 |
| Decode step, n = 13 | 28.35 ms | 28.91 | −1.9 % |
| **Decode step, n = 26** | **33.83 ms = BF16 at n = 13** | **35.41** | −4.5 % |
| Decode step, n = 32 | 36.36 ms | 38.08 | −4.5 % |
| TTFT, c = 13 | unchanged within 3 % | 745.6 → 682.8 | −8.4 % |
| Attention backend | "may change — read it" | **FLASH_ATTN → FLASHINFER** | it changed |

**The central claim holds within 5 %.** FP8 at 26 sequences costs the same decode
step as BF16 at 13. The pool doubles, the seats double, and the step at twice the
seats is 35.41 ms against 33.79 — 4.8 % apart. `docs/SLO.md` §6's statement that
FP8 KV buys sequences rather than a change of regime is now measured rather than
argued.

**And it is better than the claim, in the way the roofline said it would be.**
At *equal* concurrency the step is faster, because the decode step is bandwidth-
bound and FP8 halves the KV half of the traffic:

```
c=13, ctx 4 100:  weights 16.4 GB + KV 13 × 4 100 × 144 KiB = 7.67 GB  →  24.1 GB
FP8 halves the KV term:            16.4 + 3.84                        →  20.2 GB
ratio 0.84  →  33.79 ms × 0.84 = 28.4 ms predicted,  28.91 measured
```

Measured: 33.79 → 28.91 at c = 13 (−14.4 %) and 50.73 → 38.08 at c = 32
(−24.9 %). The saving grows with concurrency because the KV share of the traffic
does, which is `docs/SLO.md` §5's curve read from a third direction.

### The confounder, and why the first control did not work

FP8 KV **forced a different attention backend**: `FLASH_ATTN` disappeared from the
candidate list entirely and vLLM selected `FLASHINFER`, since FlashAttention 2
does not support an FP8 cache on sm89. Two things changed at once, and "FP8 gives
2× seats" could not be separated from "FlashInfer is faster" without a control.

The card's conditional did not trigger — TTFT moved 8.4 % against a 10 % threshold
— but the log said plainly that the kernel had changed, which is stronger evidence
than the proxy. The control was run anyway, twice:

- `VLLM_ATTENTION_BACKEND=FLASHINFER` was **silently ignored** by vLLM 0.27.1; the
  log shows `Using FLASH_ATTN`. The environment variable no longer selects a
  backend. `--attention-backend FLASHINFER` does, and prints a different line from
  a different code path (`cuda.py:422`, `Using AttentionBackendEnum.FLASHINFER
  backend.`) — grepping for the usual "out of potential backends" line returns
  nothing and reads like a failure.
- With the flag that works, median ITL at c = 13 is **34.05 ms** against 33.79 and
  33.71 on FLASH_ATTN. **The kernel swap costs 0.8 %, which is nothing.**

**The whole FP8 gain is the cache.** The prediction stated before the control ran
was 33.7–33.8 ms if the kernel were neutral; it is.

Accuracy was deliberately not measured: no scale line was logged, so the FP8
scales are uncalibrated and default to 1.0. Nothing here says FP8 KV is free of
quality cost — only that it is free of latency cost.

---

## 6. What the run measured about its own instrument

The failed control turned into a replicate of configuration 1, and that accident
is worth more than the control would have been. Two launches of the identical
server, same pod, same flags:

| | first launch | second launch | Δ |
|---|---|---|---|
| **median ITL, c = 13** | 33.79 | 33.71 | **0.24 %** |
| TPOT p99, c = 13 | 51.83 | 51.45 | 0.73 % |
| output tok/s | 240.47 | 241.34 | 0.36 % |
| **TTFT p50, c = 13** | 745.6 | **919.5** | **+23.3 %** |
| **`GPU KV cache size`** | 169 833 | **176 994** | **+4.2 %** |

A third launch returned 169 833 exactly and TTFT p50 697.4, so the pool figure is
an outlier rather than a new level and TTFT p50 spans 32 % across three identical
runs. Three consequences, each of which changes an instrument rather than a
number:

1. **The checkpoint gate of ±500 tokens is tighter than the platform's own
   variation.** Gate on `kv cache memory in use`, or widen to ±5 %.
2. **TTFT p50 at a fixed concurrency cannot carry a 10 % decision.** The card's
   confounder detector was built on exactly that and had no power. The backend
   line in the log is exact, free, and was right.
3. **The decode step is reproducible to a quarter of a percent.** Of everything
   this run measured, it is the only quantity fit to carry a coefficient — which
   is, fortunately, the one the repository asks it to carry.

### The median ITL stops being a decode step, and the ratio measures the wrong thing

`docs/GLOSSARY.md` treats median ITL as the closest available proxy for a pure
decode step, on the argument that only a minority of steps carry a prefill chunk.
Block B found where that argument expires. At c = 32 the predicted step is
49.85 ms and the measured median ITL reads:

| MNBT | 512 | 1 024 | 2 048 | 4 096 |
|---|---|---|---|---|
| median ITL, ms | 79.56 | 115.16 | 50.73 | 50.48 |
| TPOT p50 ÷ median ITL | 1.006 | **0.854** | 1.946 | 1.976 |

At 2 048 and 4 096 the proxy is intact. At 512 and 1 024 it is not: the chunk is
small enough that prefill lands in the *majority* of scheduler steps, and the
median moves into the contaminated mode. At MNBT 512 and c = 16 the same thing is
already visible — 56.23 ms where the decode step should be ~35.

**And the ratio TPOT ÷ median ITL does not measure how much prefill interference
there is. It measures how unevenly it is spread.** At MNBT 512, c = 32 the ratio
is 1.006 not because interference vanished but because it became universal; a
ratio at or below 1 is a sufficient sign that the median is no longer a decode
step. Run 1 read the ratio as the amount of interference (`l40s-baseline.md` §5).
The conclusion it drew survives — the gap is chunked prefill — but the quantity
was misnamed.

The non-monotonicity at c = 32 (1 024 above 512) is not explained here and is not
explained away; it is an open item.

---

## 7. Cost, at $0.99/h — and the price of a token someone can use

Block A is the first sweep in this repository that can price *goodput*, because
`--goodput` reports it natively. Good output tokens per second are
`goodput req/s × 200`:

| Rate | out tok/s | **$/1M output** | good tok/s | **$/1M good output** |
|---|---|---|---|---|
| 0.5 | 96.0 | 2.864 | 96.0 | 2.865 |
| 1.0 | 192.2 | 1.431 | 190.0 | 1.447 |
| 1.5 | 287.3 | 0.957 | 276.0 | 0.996 |
| 2.0 | 382.6 | 0.719 | 342.0 | 0.804 |
| **2.5** | 471.9 | 0.583 | **374.0** | **0.735** |
| 3.0 | 567.2 | 0.485 | 180.0 | 1.528 |
| 4.0 | 722.8 | **0.380** | 24.0 | **11.458** |

**The two columns point opposite ways past 2.5 req/s.** Raw cost per million
output tokens keeps improving to the top of the sweep, $0.583 → $0.380. Cost per
million tokens that met the SLO bottoms out at **$0.735 at an offered 2.5 req/s**
and then rises **15×**, to $11.46. An operator optimising the left column ships
the right column to customers.

FP8 moves both, because it is a throughput gain at fixed seats: at c = 32,
309.05 → 372.81 output tok/s is +20.6 %, and $0.890 → $0.738 per 1M output
tokens — before counting the doubled pool, which is a capacity argument rather
than a cost one.

---

## 8. What run 2 could not measure

**FP8 accuracy.** Uncalibrated scales, no measurement, no claim. The latency and
capacity results say nothing about output quality.

**Mean concurrency per level in block A.** The sampler recorded the whole session
at 1 Hz but was reduced only to session maxima on the pod; per-level means were
never extracted, so the `eff_mem` fit under an arrival process stays a compound
test against *predicted* `n`. The trace is on the volume.

**Why median ITL at c = 32 is worse at MNBT 1 024 than at 512.** Observed, not
explained.

**A driver-effect control.** Run 1 ran on 580.159.04 and run 2 on 550.127.05. The
decode step reproduced across them to 0.1 %, which bounds any driver effect on
*that* quantity, and says nothing about the others.

**`long_prefill_token_threshold`.** Confirmed to exist and deliberately not used —
a second scheduler knob in one run makes neither attributable.

**Anything about MI300X, prefix caching, or a second replica.** Out of scope by
design, as in run 1.

---

## 9. Predicted vs measured — the summary table

The format `docs/SLO.md` §9 requires. The column that matters this time is the
fourth: run 2 is the first run in which `eff_mem = 0.83` appears as **not fitted**.

| Quantity | Predicted | Measured | Error | Coefficient | Fitted to this run? | Card |
|---|---|---|---|---|---|---|
| KV pool, tokens | 168 985 ± 500 | 169 833 | −0.50 % | none | no | L40S |
| Seats at 4 100 ctx, capacity | 41 | 41 | 0 | none | no | L40S |
| Seats at 1 600 ctx, capacity | 106.1 | 106 running | +0.1 % | none | no | L40S |
| **Decode step, c = 13, 4 100 ctx** | **33.83 ms** | **33.83 ms** (mean of six launches, spread 1.1 %) | **−0.01 %** | `eff_mem` 0.83 | **no** | L40S |
| **Decode step, c = 32, 4 100 ctx** | **49.85 ms** | 50.61 ms (mean of the two uncontaminated levels) | **−1.5 %** | `eff_mem` 0.83 | **no** | L40S |
| Decode step under Poisson arrivals, 0.5–2.5 req/s | 23.70–30.28 ms | 24.07–30.94 ms | −0.7 … −2.2 % | `eff_mem` 0.83 | no | L40S |
| Decode step under Poisson arrivals, 3.0–4.0 req/s | 33.91 / 51.25 ms | 36.85 / 55.76 ms | −8.0 / −8.1 % | `eff_mem` 0.83 | no | L40S |
| Worst decode step vs chunk size, four levels | 77.5 / 122.2 / 211.8 / 390.9 ms | 70.6 / 114.2 / 198.2 / 364.9 ms | +9.7 / +7.0 / +6.9 / +7.1 % | `mfu` 0.439 | no | L40S |
| TTFT p50 at MNBT 512, c = 13 | ≥ 697 ms | 697.5 ms | +0.1 % | `mfu` 0.439 | no | L40S |
| TTFT p50 at MNBT 4 096, c = 13 | ≥ 391 ms | 1 079.5 ms | bound loose by 176 % | `mfu` 0.439 | no | L40S |
| FP8 KV pool | 2.00× | 2.000× | 0 | none | no | L40S |
| FP8 decode step, n = 13 | 28.35 ms | 28.91 ms | −1.9 % | `eff_mem` 0.83, `kv_dtype_bytes` 1 | no | L40S |
| FP8 decode step, n = 26 = BF16 at n = 13 | 33.83 ms | 35.41 ms | −4.5 % | `eff_mem` 0.83, `kv_dtype_bytes` 1 | no | L40S |
| TPOT p99 crossing under arrivals | 1.5–2.0 req/s | 2.0–2.5 req/s | later | `eff_mem` 0.83 × 1.9 | no | L40S |
| TTFT p99 crossing 300 ms under arrivals | 1.5–2.5 req/s | 0.5–1.0 req/s | **earlier** | `mfu` 0.439 | no | L40S |
| Goodput peak | 1.2–1.6 req/s | 1.87 req/s | +17 % | `eff_mem` 0.83 | no | L40S |
| `max_num_seqs` by TPOT p99, MNBT 512 | 18–22, or 31 | **14** | far below | — | no | L40S |
| Attention kernel effect on the decode step | unknown | +0.8 % | — | — | no | L40S |

**`eff_mem = 0.83` is no longer a hypothesis.** It predicted a decode step on a
different pod, under a different driver, at four chunk sizes, under two attention
backends, under both a closed and an open loop, and with the KV dtype halved —
and missed by under 5 % everywhere the median ITL was still a decode step, by
under 2 % wherever the card was not saturated. `mfu = 0.439` earned a second kind
of confirmation: it priced a chunk's compute to within 5–10 % across an eightfold
range of chunk size, having only ever been fitted to one uncontended prefill.

**What is still not a model of the service.** Run 1's headline stands and has been
priced. The gap between the decode step and TPOT p99 is scheduling; the knob
run 1 named moves it by two seats out of nineteen; and the arrival-rate sweep
found the tail breaking earlier than any of it predicted, at a third of the rate.
The card is understood. The queue is not.
