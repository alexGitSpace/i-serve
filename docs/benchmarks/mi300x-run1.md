# MI300X run 1 — 2026-09-27

The first run on the card `docs/SLO.md` was derived for. It did for the MI300X
what run 1 did for the L40S and nothing more: read the KV pool from the startup
log, fit `eff_mem` from the decode step, fit `mfu` from an uncontended prefill,
and find where the seat curve ends. Every prediction it faced was written in
`docs/benchmarks/runsheets/mi300x-run-1.md` from `bench/predictions.py`
tables 9 and 10, at the uncalibrated 0.70 / 0.45 and `memory_bytes` 192e9,
before the droplet existed.

Raw evidence, verbatim: `docs/benchmarks/raw/mi300x-2026-09-27/`. The
arithmetic behind every figure below: `python3 bench/measured_mi300x_run1.py`.
Where a number here disagrees with a derivation elsewhere in the repo, **this
file wins for the MI300X and only for the MI300X** (`docs/SLO.md` §9).

**The run in four sentences**, the runsheet's definition of done:

- The logged pool is **1 123 065** tokens (70 254 blocks), 5.9 % above the
  derived 1 060 655 and 13.9 % above the corrected ~986 000, because the
  derivation took the card as 192e9 B where it holds 192 GiB. Derived at 192 GiB
  it is 2.1 % below on the first launch and 0.75 % on the second: what the pool
  pays before it is carved is a few GiB that moved between two launches of one
  configuration, so the L40S's 7 % transfers neither as a percentage nor as a
  constant.
- `eff_mem` is **0.46** (0.41–0.49 over five rows at batch 1, 8 and 32), a third
  below the 0.70 prior. Batch 1 is the highest, but most of its step does not
  scale with KV, so the scalar may be carrying a per-step cost as well as a
  bandwidth.
- `mfu` is **0.166** at 4 000 tokens, against 0.45 assumed, and TTFT did not
  double with the prompt: ×2.21, then ×2.50.
- **Latency ended the curve, as served**: TPOT p99 crossed 50 ms between c = 8
  and c = 32, through prefill interference rather than the decode step. The
  sheet predicted the opposite — capacity at ~234 with every level inside 50 ms.
  The pool and the cap were never reached: the token budget held the engine at
  ~99 running with the pool a third full, and throughput had flattened by c = 32
  at what prefill compute allows.

---

## 1. What was run

| | |
|---|---|
| Accelerator | 1 × AMD Instinct MI300X VF, gfx942, 205 822 885 888 B reported by `rocm-smi` |
| Provider | AMD Developer Cloud (DigitalOcean GPU droplet), ATL1, on-demand, **$1.99/h**, paid from credits; the same card lists at $2.59/h on DigitalOcean's own price list |
| Host | *Quick Start → ROCm Software*, version 7.14 on Ubuntu 24.04 |
| Image | `vllm/vllm-openai-rocm:v0.27.1`, `vllm --version` 0.27.1+rocm723 |
| Model | `Qwen/Qwen3-8B`, `--dtype auto` → BF16, `ROCM_ATTN` backend, block size 16 |
| Server flags | `--gpu-memory-utilization 0.90 --max-model-len 9000 --no-enable-prefix-caching --max-num-batched-tokens 2048 --max-num-seqs 256` |
| Instrument | `vllm bench sweep serve`, the runsheet's §3 line verbatim; one server per launch, two launches |
| Load | `--dataset-name random --random-range-ratio 0 --ignore-eos --request-rate inf`, 4 000 in / 200 out, `num_prompts = max(20, 4 × c)` |
| Rows | c = 1, 8, 32, 64, 128, 192, 224, 240, 256, 288 at 4 000 tokens; 2 000 and 8 000 tokens at c = 1 |
| Repeats | 3 for c001–c064, 2 for everything after (below) |
| Failures | zero requests failed, in all 28 repeats |
| Clock | droplet 09:55–11:35 UTC, ≈ 1.7 h, ≈ $3.3 against a $5 budget; the invoice outranks this figure |

`--max-num-batched-tokens 2048` is runs 1–3's geometry, not this card's: this
build's default for an API server on a card of 70 GiB or more is 8 192. §5 and §6
are about what that choice did.

**Five departures from the sheet, each decided on the clock:**

1. **The image.** The console lists no *Vanilla ROCm*; *Quick Start → ROCm
   Software* is the plain ROCm stack above the *Quick Start Packages*, and it was
   taken with version 7.14 over the default 10.0, because 7.14 is the host AMD
   itself pairs with vLLM 0.27.1 in its package. Neither the host version nor
   the virtual function was a variable here. The image booted with a container
   named `rocm` running Jupyter Lab on a public port with the GPU devices mapped
   — the thing §0 of the sheet expected of the Quick Start images. It held no
   GPU memory and was stopped before the run's container started.
2. **Checkpoint A missed both figures and the sweep continued.** The stop
   condition reads *stop, account for it, relaunch*. The miss was accounted
   before row c001 ended (§2) — the independent content of that account is the
   card's byte count and 147 456 B per token of KV; the rest is the log's own
   terms — and nothing in the configuration was wrong, so a relaunch would have
   reproduced the same server. The letter of the condition was not followed.
3. **Drop step 1 before the 90-minute mark.** At 10:19 UTC the pace (~0.39 s per
   prompt once saturated) put three repeats past the two-hour mark. Ctrl-C landed
   at 10:22:39, after `c128/run=1.json`, and the identical line relaunched at
   10:23:12 with `--num-runs 2 --resume`. No never-dropped row was touched.
4. **`tee -a` on the relaunch.** The sheet's line uses `tee`, which would have
   truncated `sweep.log` and the startup log inside it.
5. **The `eff_mem` fit set.** The sheet fitted c001–c224; this report fits
   c001–c032 and the two length rows, because above c = 32 the median ITL is no
   longer a decode step (§3). Decided after seeing the data.

---

## 2. Checkpoint A — the pool, and a unit

| | Tokens |
|---|---|
| Derived at `memory_bytes` 192e9: `(192e9 × 0.9 − 16.4e9) / 147 456` | 1 060 655 |
| Derived × 0.93, the L40S's shortfall | ~986 000 |
| **Logged: `GPU KV cache size`, launch 1** | **1 123 065** |
| Logged, launch 2 | 1 138 444 |
| Derived at 192 GiB, `bench/roofline.py` since this run | 1 147 072 |

Outside both ±3 % bands. The log accounts for launch 1 term by term, from the
card `rocm-smi` reports:

```
the card, 205.82e9 B, not 192e9                 →  +84 368 tokens
non-torch memory beyond the weights, 1.76 GiB   →  −12 816
peak activation memory, 1.12 GiB                →   −8 156
                                    accounted      1 124 052
```

The logged figure is not blocks × 16: vLLM prints `max_concurrency ×
max_model_len`, and 1 123 065 inverts to **70 254 blocks = 1 124 064 tokens**.
The account lands 12 tokens from it. The log's `kv cache memory in use is
154.37 GiB` is the same pool from the other side.

**"192 GB" on the MI300X is 192 GiB.** AMD's data sheet prints "192 GB" without
defining the unit; `rocm-smi`'s 205 822 885 888 B is exactly 192 GiB less 320 MiB,
held back on the virtual function. `MI300X.memory_bytes = 192e9` under-counted
the card by 7.2 %, the whole miss and more. It is corrected to `192 * 2**30` in
both instances, and the correction moved `docs/SLO.md` §6: the MI300X's two
derived limits, 8 % apart at 192e9, now tie at 286.

**The CUDA graph pool is not taken from the KV pool on this build.** The pool is
carved at 10:02:39 and the graphs captured at 10:02:44–51; in vLLM v0.27.1 the V2
model runner's `profile_cudagraph_memory()` returns 0, so nothing is set aside
for them. The engine therefore holds 17.03 + 1.12 + 4.12 + 154.37 = 176.64 GiB,
**92.1 % of the card** against the 90 % asked — the log's own hint,
`--kv-cache-memory=161173659956` (150.1 GiB) "to fit into requested memory", is
the same arithmetic.

**What is paid before the pool moves between launches.** Launch 2 logged 0.58 GiB
of non-torch memory and 0.19 GiB of activation against launch 1's 1.76 and 1.12,
with a warm compile cache; its pool came out 15 379 tokens larger. Against the
192 GiB derivation the shortfall is 2.1 % on launch 1 and 0.75 % on launch 2; the
L40S's was 7.6 %, on a pool 6.6× smaller. The terms are a few GiB, set by the
model, the step budget, the platform and the state of the compile cache, and as
a percentage they transfer between neither cards nor launches.
`bench/predictions.py` takes launch 1's 2.1 % for the MI300X tables, the
conservative one.

Derived from the logged pool, the card seats 280 sequences at 4 000 tokens and
267 at 4 200. No seat count was observed: the running count stopped at ~100 (§5)
and nothing was preempted.

---

## 3. `eff_mem` — the decode step

The read is run 1's: the median ITL against the floor at 100 % of peak,
`(16.4e9 + n × context × 147 456) / 5.3e12`, with `context` = input + 100.

| Row | ctx | Floor @ 0.70 | Median ITL | **Implied `eff_mem`** |
|---|---|---|---|---|
| c001-in2000 | 2 100 | 4.50 ms | 6.57 ms | **0.480** |
| c001 | 4 100 | 4.58 ms | 6.61 ms | **0.485** |
| c001-in8000 | 8 100 | 4.74 ms | 6.78 ms | **0.489** |
| c008 | 4 100 | 5.72 ms | 9.72 ms | **0.412** |
| c032 | 4 100 | 9.64 ms | 14.90 ms | **0.453** |

Table 9 printed the floors at 4 000 tokens of context (5.69 and 9.51 ms at c008
and c032); these are at the mean context, as the implied values are.

**`eff_mem = 0.46`**, the mean of five rows at three batch sizes, 0.41–0.49. The
median reproduced to 0.01 ms across repeats at batch 1 and c008, and to 2 % at
c032.

- **0.70 was 52 % above it**, where on the L40S the same prior was 16 % below
  0.83. One number for two cards missed them in opposite directions.
- **Batch 1 is the highest**, 0.48–0.49 at three prompt lengths; c008 sits below
  c032, and nothing in this run says why.
- **Most of the batch-1 step does not scale with KV.** From 2 100 to 8 100 tokens
  of context the step grows 0.21 ms for 0.885 GB more KV — 4.2 TB/s, 0.79 of
  peak, reproduced to 0.01 ms. The remaining ~6.5 ms is either the weights read
  at under half of peak or a per-step cost the scalar absorbs; the sampler (the
  model's `generation_config` imposes temperature 0.6, top-k and top-p), host-side
  scheduling and the virtual function are candidates this run cannot separate.
- **The fit stops at c032.** At c064 the median ITL is 134 ms and from c128 up it
  is 168–170 ms: every step carries a prefill chunk (§5). Five points over batch
  1–32, against run 1's twelve over 1–41; the coefficient is correspondingly
  weaker.

---

## 4. `mfu` — prefill

One request at a time on the card, at three prompt lengths:

| Prompt | Engine steps | Launch | TTFT floor @ 0.45 | Median TTFT | × prev | **Implied `mfu`** |
|---|---|---|---|---|---|---|
| 2 000 | 1 | 2 | 47.3 ms | 116.4 ms | — | 0.183 |
| 4 000 | 2 | 1 | 94.5 ms | 256.9 ms | 2.21× | **0.166** |
| 8 000 | 4 | 2 | 189.1 ms | 642.6 ms | 2.50× | 0.132 |

**`mfu = 0.166` at 4 000 tokens**, the point run 1 took 0.439 from on the L40S.
The spec sheet gives the MI300X 3.6× the L40S's dense BF16; this prefill came
out **1.36×** faster (256.9 ms against 350.1 ms).

**TTFT grew faster than the prompt.** The sheet's rule was that the three must
double in step or something other than prefill is in the TTFT. They did not.
Attention FLOPs, which the floor leaves out, would make the steps 2.08× and
2.16× — part of the gap, not all of it. Two other candidates are in the table
itself: at a 2 048-token budget the prompts are 1, 2 and 4 engine steps, and
every chunk after the first attends over a cached prefix; and two of the three
points ran on the second launch. The check that separates them is the three
lengths again on one launch at a budget of 8 192, where every prefill is one step.

---

## 5. Where the running count settles, and where throughput does

The engine's 10-second log lines (Running, Waiting and KV use are snapshots, so
their maxima are a floor on the peak; prompt tok/s is averaged over the saturated
lines) beside the benchmark's means over repeats:

| c | Running | Waiting | KV use | Prompt tok/s | Median ITL | TPOT p99 | Output tok/s |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 2 | 2.9 % | 7 000 | 9.72 ms | 24.35 ms | 345.9 |
| 32 | 32 | 26 | 11.6 % | 10 199 | 14.90 ms | 76.33 ms | 499.1 |
| 64 | 64 | 53 | 23.4 % | 10 566 | 134.40 ms | 143.63 ms | 501.7 |
| 128 | 101 | 107 | 36.6 % | 10 746 | 168.07 ms | 237.26 ms | 511.2 |
| 192 | 101 | 177 | 36.1 % | 10 839 | 169.12 ms | 236.91 ms | 524.6 |
| 240 | 100 | 229 | 35.8 % | 10 905 | 169.81 ms | 237.30 ms | 530.5 |
| 256 | 101 | 240 | 36.1 % | 10 904 | 170.06 ms | 237.53 ms | 531.7 |
| 288 | 100 | 276 | 35.9 % | 10 917 | 170.18 ms | 237.64 ms | 534.1 |

Zero preemptions in all 28 repeats; the cap of 256 was never reached; the pool
never passed 37 %.

**The token budget set the running count.** A step spends one token per running
sequence on decode and the rest on prefill; a request leaves after 200 decode
steps and enters after 4 000 prefill tokens, so the count settles where the two
rates meet:

```
n = budget × output / (input + output) = 2 048 × 200 / 4 200 = 97.5
```

Logged: 98 or 99 on 324 of the 350 saturated lines, the one or two above the
arithmetic being requests partly prefilled — the scheduler counts a request as
running from its first chunk, and admits a waiting one only while the budget has
tokens left. No other cap in v0.27.1's scheduler applies here. The derivation
is written after the fact; its falsifiable form is that at a budget of 4 096 the
same geometry settles near 195 running.

**Prefill compute set the throughput.** Output was 499 tok/s at c032, with 32
running, and 534 at c288, with ~100: it had flattened before the running count
reached its ceiling. Every row of a 4 000 / 200 closed loop spends 20 prompt
tokens per output token, so output is prompt throughput over 20 at every level,
and the prompt side ran at ~10 900 tok/s once full. Above c032 a higher
concurrency buys a longer queue: median TTFT went from 0.69 s at c032 to 68.6 s
at c288. So the second half of the falsifiable form: at a budget of 4 096, output
should stay within a few percent of 534 while the median ITL roughly doubles. If
output rises materially, the budget was limiting throughput too.

The sheet's c288 prediction held in form and not in cause: the row queued and did
not preempt, but behind ~100 running, not 256.

On the L40S none of this could show: its pool seated 41, below the 97.5 the same
budget and geometry allow.

---

## 6. The 50 ms line, and interference

| Asked of | Crosses 50 ms at |
|---|---|
| Table 9, `eff_mem` 0.70, 192e9 | between 256 and 288 (45.12, then 50.21 ms) — after capacity at ~234, which the sheet said would bind first |
| Roofline with `MI300X_RUN1`, `eff_mem` 0.46 | 178 |
| Measured decode step (median ITL) | not measurable: no decode step above c032 |
| **Measured TPOT p50** | **between 8 and 32** — 18.81 ms, then 55.83 ms |
| **Measured TPOT p99** — the SLO metric | **between 8 and 32** — 24.35 ms, then 76.33 ms |

The sheet spaced its rows for a curve that ended past 200, so there is no row
between 8 and 32; the crossing is placed only to within that gap. The limit that
binds `max_num_seqs` at this geometry is latency, as served — the same verdict
as the L40S's 12 against its 41, reached by a different road.

**The road is interference, and it is prefill cost, not the budget.** TPOT p50
over median ITL was 1.94 at c008 and **3.75 at c032**, against the L40S's 1.91 at
c = 32. The sheet expected it to shrink because prefill would be 3.6× faster; it
was 1.36× faster, while the decode step was 3.4× faster (6.61 ms against
22.70 ms). A model with no budget term reproduces both rows: if each of the other
c − 1 requests prefills once during a request's 200 decode steps,

```
TPOT ≈ median ITL + (c − 1) × TTFT(c = 1) / 200
c008:  9.72 + 7 × 1.285  = 18.71 ms   against 18.81 measured
c032: 14.90 + 31 × 1.285 = 54.72 ms   against 55.83 measured
```

and it crosses 50 ms at c ≈ 29. Written after the run, not a prediction it faced.
It agrees with L40S run 2 §4, where the budget moved the tail and left TPOT p50
unchanged while prefill kept up: the lever on the mean is what a prefill costs —
`mfu`, or prefix caching — and `max_num_batched_tokens` moves where it lands.

The interactive class at 4 000 tokens does fit this card alone: TTFT p99 at c = 1
is 264 ms against a 300 ms budget, where the L40S's 350 ms median already did
not. It leaves 36 ms for a queue, and at c = 8 the median TTFT is 654 ms.

---

## 7. Cost, at $1.99/h

| c | Output tok/s | $/1M output | TPOT p50 | TPOT p99 |
|---|---|---|---|---|
| 1 | 127.3 | 4.343 | 6.61 ms | 6.61 ms |
| **8** — last row inside 50 ms, p50 and p99 | 345.9 | **1.598** | 18.81 ms | 24.35 ms |
| 32 | 499.1 | 1.108 | 55.83 ms | 76.33 ms |
| 128 | 511.2 | 1.081 | 174.57 ms | 237.26 ms |
| **288** — throughput plateau | 534.1 | **1.035** | 175.18 ms | 237.64 ms |

**At the plateau the MI300X costs 19 % more per output token than the L40S at
this configuration**: $1.035 against $0.873, both measured at 4 000 in / 200 out
and a 2 048-token budget, the MI300X at AMD's $1.99/h front-door price.

**Inside the SLO the comparison is within the row gap.** At c = 8 on both cards,
$1.598 against the L40S's $1.427 — 1.80× the throughput for 2.01× the rate (the
MI300X's c008 is $1.549 by its median repeat; the first of three was slower).
The crossings sit between measured rows on both cards. On p50, the §6 model puts the MI300X's near c = 29, ≈ $1.15 by interpolating throughput between c008
and c032, against the L40S's $1.126 at its last row inside 50 ms — parity. On
p99, the SLO metric, linear interpolation puts the crossings near c ≈ 20 and
c ≈ 12.6, ≈ $1.31 against ≈ $1.14 — the MI300X 15 % dearer. Both are derived
figures.

The configuration was chosen for parity with runs 1–3, not for this card, and
§6 names the lever. Whether the card is cheaper configured for itself is MI300X
run 2's question.

---

## 8. What this run could not measure

**Why `mfu` is 0.17.** The candidates it can name — the attention backend,
untuned GEMMs, the chunking of §4, the host stack and the virtual function — are
not separable in this run, and a guess at the cause would be a claim this
repository does not make without a measurement. Run 2 moves one at a time.

**What the batch-1 decode step is made of.** §3's increment says the KV read is
fast and something fixed is not; the weights read and a per-step cost look the
same to a scalar.

**`eff_mem` above batch 32.** Every step above c032 carries a prefill chunk.
A decode-only read at high batch needs output-heavy geometry (short prompts, long
generations) or a larger budget.

**The pool's behaviour under load.** KV use never passed 37 %, so nothing here
confirms that the logged pool is where preemption begins, the way c = 45 did on
the L40S.

**A peak running count.** The sweep has no gauge sampler; the engine's 10-second
lines are snapshots, and their maxima are a floor on the peak.

**FP8 KV, prefix caching, a second card.** Out of scope by design, as the sheet
said.

---

## 9. Predicted vs measured — the summary table

Every row names the coefficient that produced its prediction, whether that
coefficient was fitted to this same run, and the card (`docs/SLO.md` §9). Rows
the sheet wrote as a bound or as "read, not predicted" are marked so and carry no
error.

| Quantity | Predicted | Measured | Error | Coefficient used | Fitted to this run? | Card |
|---|---|---|---|---|---|---|
| KV pool, tokens | 1 060 655 | 1 123 065 (printed; 1 124 064 in blocks) | −5.6 % | none (capacity arithmetic, 192e9) | no | MI300X |
| KV pool after the L40S's shortfall, tokens | ~986 000 | 1 123 065 (launch 1) | −12.2 % | 7 % shortfall, fitted on the L40S | no | MI300X |
| Seats at 4 200 tokens, capacity — the pool row divided by 4 200, not a second test | ~234 | 267 (derived from the logged pool) | −12.4 % | 7 % shortfall | no | MI300X |
| Decode step, c = 1 | 4.58 ms | 6.61 ms | −30.8 % | `eff_mem` 0.70 | no | MI300X |
| Decode step, c = 32 | 9.51 ms | 14.90 ms | −36.2 % | `eff_mem` 0.70 | no | MI300X |
| Decode step, c = 1, refitted | 6.97 ms | 6.61 ms (c001) | +5.5 % | `eff_mem` 0.46 | **yes** | MI300X |
| Decode step, c = 8, refitted | 8.71 ms | 9.72 ms (c008) | −10.4 % | `eff_mem` 0.46 | **yes** | MI300X |
| Decode step, c = 32, refitted | 14.66 ms | 14.90 ms (c032) | −1.6 % | `eff_mem` 0.46 | **yes** | MI300X |
| TTFT, c = 1, 2 000-token prompt | 47.3 ms | 116.4 ms | −59.4 % | `mfu` 0.45 | no | MI300X |
| TTFT, c = 1, 4 000-token prompt | 94.5 ms | 256.9 ms | −63.2 % | `mfu` 0.45 | no | MI300X |
| TTFT, c = 1, 8 000-token prompt | 189.1 ms | 642.6 ms | −70.6 % | `mfu` 0.45 | no | MI300X |
| TTFT per doubling of the prompt | 2.00× / 2.00× | 2.21× / 2.50× | — | none (a ratio) | no | MI300X |
| Running at c = 288 | at most 256, a bound | ~100 (101 max) | bound held | none (`max_num_seqs`) | no | MI300X |
| Preemptions at c = 256 | > 0, a bound | 0 | bound failed | none | no | MI300X |
| `max_num_seqs` by latency, 50 ms | 286 | 8–32 (TPOT p99) | ×9 to ×36 | `eff_mem` 0.70, no interference term | no | MI300X |
| Interference, TPOT p50 / median ITL | read, not predicted (expected to shrink from 2.06) | 3.75 at c = 32 | — | none | no | MI300X |

The three *refitted* rows prove nothing on their own; they become evidence when
MI300X run 2 faces `eff_mem` 0.46 and `mfu` 0.166 without having produced them.
The rows that count already are the unfitted ones. Every unfitted row that
predicted a time predicted it too short. The latency row misses by most, and
most of that miss is not the coefficient: at `eff_mem` 0.46 the same arithmetic
gives 178, and the rest is interference the 286 never contained. The pool, the
one row on the other side, was under-counted by a unit.
