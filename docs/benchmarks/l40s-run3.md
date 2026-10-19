---
card: l40s
run: 3
date: 2026-08-30
asked: How many seats is an 80 % prefix cache hit rate worth?
found: 3× the seats, 12–14 to 32–40, because the decode step itself fell where the derivation said it could not
predicted_with: l40s-run1
---

# L40S run 3 — prefix caching, priced in seats

Runs 1 and 2 measured a card and a service with the feature **off**. This run is
the first that varies the one quantity `docs/SLO.md` §6 could only derive: the
prefix cache hit rate `h`. It is also the first driven by `bench/harness.py`
rather than `vllm bench serve`, because a controlled `h` is not a flag that tool
has.

Raw evidence: `docs/benchmarks/raw/l40s-2026-08-30/` — machine-written this time,
not transcribed: every level is a JSON file the harness produced, beside the two
startup logs and the console output of each sweep. Its reasoning and the
predictions written before the card was rented:
`docs/benchmarks/runsheets/l40s-run-3.md`, never revised against the results
below, and dated by itself rather than by anything in this tree.
Where a number here
disagrees with a derivation elsewhere in the repo, **this file wins for the L40S
and only for the L40S** (`docs/SLO.md` §9).

**The one-line result: `h` = 0.8 is worth 3× the seats — and for a reason §6 said
it could not be.** The seat count went from 12–14 to 32–40, past the 24 ± 2 that
was predicted, because the decode step *fell* at equal concurrency. §6's channel 1
— "a cache saves computing and writing KV, never reading it" — is false on this
card. What §6 got exactly right is the term it was actually arguing about: the
prefill interference scaled by 0.194 against a predicted (1 − `h`) = 0.200.

---

## 1. What was run

| | |
|---|---|
| Accelerator | 1 × NVIDIA L40S, driver **580.126.20**, pod `ff6689f28bec` (RunPod `tlm7f7mjyzi3ot`) |
| Provider | RunPod, on-demand, $0.99/h, no network volume — weights re-downloaded |
| Image | `vllm/vllm-openai:v0.27.1`, `vllm --version` 0.27.1 — the same build as runs 1 and 2 |
| Model | `Qwen/Qwen3-8B`, `--dtype auto` → BF16 |
| Config 1 | `--gpu-memory-utilization 0.90 --max-model-len 9000 --no-enable-prefix-caching` |
| Config 2 | the same, **without** the negative flag; `max_num_batched_tokens` and `max_num_seqs` left at defaults |
| Structure | vLLM **not PID 1**; the pod slept, two servers were launched by hand, both startup logs are files |
| Instrument | `bench/harness.py` on the pod, prompts sent as **token IDs** so a shared prefix is shared by construction |
| Block A | closed loop, 4 000-token prompts, c = 8…28 at `h` = 0 and `h` = 0.8, extended to c = 56 |
| Block B | open loop, Poisson, 1 500-token prompts, 1.5…7 req/s at `h` = 0.8, extended to 13 |
| Block C | closed loop, 4 000 tokens, c = 13, `h` = 0, run on **both** servers |
| Output length | 200 tokens, fixed — comparable with runs 1 and 2 without adjustment |
| Volume | 45 levels, 7 900 requests, 17.2 M prompt tokens, 1.58 M output tokens |
| Failures | **0 of 7 900** |
| Measurement window | 09:47:56 → 10:58:26 UTC, 70.5 min |
| Spend | ≈ 1.7 pod-hours ≈ **$1.65** against a $2.48 budget, of which ~0.25 h was a rejected pod (§2) |

### The pod that was thrown away

The first machine rented held driver **570.124.06** (CUDA 12.8) and the engine
refused to start: `The NVIDIA driver on your system is too old (found version
12080)`. Downgrading the image would have made every cross-run comparison in this
file a comparison of two vLLM builds, so the pod was terminated and redeployed
with a CUDA ≥ 12.9 filter. Cost: about fifteen minutes and no data.

Two deploy facts worth carrying into the next runsheet, both discovered here:

- **`sshd` does not run under a JSON entrypoint override.** Replacing the image's
  entrypoint with `bash -lc "sleep infinity"` also replaces the script that starts
  it, so the direct TCP endpoint — the only harvest path (run 2's postscript, item
  8) — is dead until `openssh-server` is installed and started by hand. The
  RunPod proxy still works and is how that is done.
- **`/workspace` was a network filesystem** (`mfs#…runpod.net`), not a local disk,
  even with no network volume attached. Weights were put on the container's own
  overlay instead; logs and results stayed on `/workspace` and were harvested by
  `scp` before Terminate.

---

## 2. Checkpoint A″ — six facts of eight, and the two that are gone

| Log line | Predicted | Config 1 | Config 2 |
|---|---|---|---|
| `GPU KV cache size` | 169 833 ± 5 % | **168 985** | **176 227** |
| `kv cache memory in use` | 23.34 GiB | 23.23 | 24.22 |
| `Maximum concurrency` | ~18.8× | 18.78 | 19.58 |
| `enable_prefix_caching` | False / True | **False** | **True** |
| `kv_cache_dtype` | auto (BF16) | auto | auto |
| attention backend | FlashAttention 2 | `FLASH_ATTN` | `FLASH_ATTN` |
| `max_num_batched_tokens` | 2 048, from the log | **absent** | **absent** |
| `max_num_seqs` | absent | absent | absent |

Config 1's pool is run 1's figure to the token. The 4.3 % between the two configs
is inside the ±5 % gate and inside the 4.2 % of launch-to-launch noise run 2
measured, so **this run cannot say whether prefix caching costs pool space**; it
can only say the cost is below the instrument.

**The sheet predicted seven keys of eight and got six.** `max_num_batched_tokens`
does not appear in this build's startup log either — run 3's own §7 edit found
`max_num_seqs` missing and assumed the other was safe. Both are now launch-command
facts, and the honest statement about this run's chunk size is that it was *not
passed*, therefore default, and that the decode step measured at c = 13 (§5) is
what makes "the same geometry as runs 1 and 2" a measurement rather than a claim.

**Cascade attention is not in the log — and could not have been.** Grepped,
absent, and no time spent hunting for it: the sheet's instruction. Reading the
source afterwards showed why the grep came back empty. At tag `v0.27.1`,
`vllm/v1/attention/backends/flash_attn.py` emits no cascade log line at all — the
only `logger` calls on that path are the FlashAttention version and a debug line
about KV strides — and the choice is taken per step in
`use_cascade_attention()`, not once at startup, so there is nothing a startup log
could have carried. **The check therefore closed nothing.** What it establishes
is that this engine does not report which attention path it took. Block A
answered the underlying question instead, and the answer was not the one either
hypothesis predicted (§4).

**The two questions only a card could answer**, both from the smoke level:

- `/v1/completions` **accepts a list of token IDs** as `prompt` on this build.
  Had it not, the run's independent variable would not have existed.
- `vllm:prefix_cache_queries` **counts tokens**: a 128-token prefix over a
  256-token prompt read `h` = 0.500 exactly. Every `h` in this file is therefore
  scored against a denominator that was verified, not assumed.

---

## 3. The defect the harness caught, and what it cost

The first uncached sweep reported `h` = 0.664 / 0.854 / 0.872 on levels whose
nominal `h` was **zero**, and the harness invalidated them on its own gate. The
numbers name the cause: 24/36, 36/42 and 42/48 are the share of each level's
prompts that the *previous* level had already sent. `Workload.seed` was one
constant for every level, so a level's prompts were a byte-identical prefix of
the next level's — invisible with prefix caching off, which is how runs 1 and 2
never met it, and a 100 % cache hit with it on. From c = 20 the pool
(176 227 tokens against 48 × 4 200) had evicted the earlier prompts and `h`
returned to a clean zero, which is why the contamination shows as a bulge in the
middle of the sweep rather than a trend.

Fix: `SEED + concurrency` per seat level, `SEED + 1000 + 10×rate` per rate level,
so each level's prompts are its own and the only hits a level can score are the
ones its construction asks for. Block C's control and treatment keep one seed
between them, which is what makes them the same level against two servers.

Cost: one seven-minute sweep re-run, ~$0.12. Both sweeps are kept in the raw
directory — `run3-a-h00` is the contaminated one, `run3-a-h00-clean` is the
measurement — because the defect is the run's clearest evidence that the gate
works: **the level that lies about its own independent variable is the one the
instrument is built to refuse.**

---

## 4. Block A — three seats become nine, and channel 1 is false

Two sweeps, same nine concurrencies, same server, same 4 000-token prompts,
differing only in whether 3 200 of those tokens are shared. Measured `h` was
**0.800 on every cached level**, to three decimals, and 0.000 on every uncached
one.

| c | med ITL `h`=0 | med ITL `h`=0.8 | fall | TPOT p99 `h`=0 | TPOT p99 `h`=0.8 |
|---|---|---|---|---|---|
| 8 | 29.54 | 25.42 | −13.9 % | 39.28 | 27.20 |
| 12 | 32.85 | 27.36 | −16.7 % | **48.71** | 30.79 |
| 14 | 34.46 | 27.60 | −19.9 % | **53.76** | 31.65 |
| 16 | 36.15 | 27.92 | −22.8 % | 58.63 | 32.38 |
| 20 | 40.36 | 28.92 | −28.3 % | 69.15 | 35.14 |
| 22 | 42.02 | 29.23 | −30.5 % | 74.14 | 35.91 |
| 24 | 43.74 | 29.76 | −31.9 % | 79.39 | 37.63 |
| 26 | 45.26 | 33.07 | −26.9 % | 84.67 | 41.14 |
| 28 | 47.21 | 33.58 | −28.9 % | 89.71 | 42.30 |
| 32 | — | 34.39 | — | — | 44.29 |
| 40 | — | 39.03 | — | — | **52.17** |
| 48 | — | 40.23 | — | — | 57.08 |
| 56 | — | 44.05 | — | — | 64.05 |

**The seat count**, on the SLO metric (TPOT p99 ≤ 50 ms), by linear interpolation
between the levels that bracket it:

| | Predicted (§6) | Measured |
|---|---|---|
| `h` = 0 | 13.5 | **12.5** — between 12 and 14 |
| `h` = 0.8 | 24.3 | **37.8** — between 32 and 40 |
| Ratio | 1.8× | **3.0×** |

The uncached number is the one §6 was entitled to be judged on, and it lands
within a seat. The cached number is 56 % higher than predicted, and the rest of
this section is why.

### The decode step moved — the run's sharpest falsification

§6's channel 1 says prefix caching cannot touch the decode step: one physical
copy in HBM, `n` reads per step, so the step is whatever `bytes_moved` says. The
run measured that quantity at nine matched concurrencies against an instrument
reproducible to 0.24 %, and it fell by 14 % to 32 %.

Fitting `med ITL = a + b·n` over each sweep:

| | intercept, ms | slope, ms/seat | implied context per seat |
|---|---|---|---|
| `h` = 0, measured | 22.16 | **0.8952** | 4 354 tokens |
| roofline model, ctx 4 100 | 22.87 | 0.8430 | 4 100 |
| `h` = 0.8, measured (c ≤ 28) | 22.27 | **0.3670** | 1 785 tokens |
| cascade hypothesis (runsheet §5) | 23.53 | 0.1850 | ~900 |

Read across that table: the uncached sweep reproduces the memory-bandwidth model
to 6 % on the slope and 3 % on the intercept — the fourth independent
confirmation of `eff_mem` = 0.83. The cached sweep does not fit it at all. Nor
does it fit the alternative the runsheet wrote down: **cascade attention would
have made the fall twice as large as it was.** The measured per-seat cost
corresponds to ~1 785 tokens of context where a seat's *unique* content is 800
prompt tokens plus at most 200 output.

So the prefix is being read less than once per sequence per step and more than
once per batch. This run cannot say by what mechanism — it has no profiler trace
and the engine logs nothing about it — and naming it would be inventing a
mechanism to fit a slope. What it can say is that **the latency limit is not
invariant to `h`**, which is a fourth channel §6 does not contain, and that the
detector the runsheet built for a binary question returned a third answer.

### Two candidates, from the source and from arithmetic

Everything below this line is **post-run analysis on the committed numbers and on
vLLM's source at the pinned tag — no new measurement.** It is here because the
paragraph above dismissed cascade attention on a grep that §2 has now shown to be
empty by construction.

**Candidate 1 — cascade, decided per step.** `use_cascade_attention()` is not a
mode; it is a cost model evaluated on every step. With the run's geometry
(`n_q` = 32, `n_kv` = 8, prefix 3 200, 142 SMs on an L40S) it reduces to

```
num_prefix_tiles    = ceil(3200 / 128)              = 25
cascade_time        = ceil(32·ceil(n/128) / 142)·25 = 25      for any n <= 128
flash_decoding_time = ceil(n · 8 · 25 / 142)        = ceil(1.408·n)
cascade  <=>  25 < ceil(1.408·n)  <=>  n >= 18
```

so cascade should engage above ~18 requests in a step and not below — plus
unconditionally in any step carrying a prefill chunk, where `query_lens` are not
all 1. **The sweep does not show that step.** Fitting the cached levels in
pieces:

| range | slope, ms/seat | implied context per seat |
|---|---|---|
| `h` = 0, c ≤ 28 | 0.895 | 4 354 tokens |
| `h` = 0.8, c ≤ 16 | 0.316 | **1 535 tokens** |
| `h` = 0.8, 12 ≤ c ≤ 24 | 0.205 | 995 tokens |
| `h` = 0.8, c ≥ 26 | 0.367 | 1 784 tokens |

Below c = 16 the heuristic says cascade is off, which obliges the full 4 000
tokens per seat; measured is 1 535, and there is no discontinuity at 16 → 20.
These are two-to-six-point fits and individually noisy, but the direction is not
in doubt: **cascade alone does not account for the fall.**

**Candidate 2 — L2 residency of the shared prefix.** Attention runs a layer at a
time, so the quantity that meets the cache is per-layer KV: `2 × 8 × 128 × 2` =
**4 096 B per token per layer**.

| | per layer |
|---|---|
| shared prefix, 3 200 tokens | 13.1 MB |
| unique content, ~900 tokens per seat | 3.7 MB |
| L2 on AD102 / L40S | ~96 MB — **datasheet, not measured here** |

The working set `13.1 + 3.7·n` MB crosses 96 MB at **n ≈ 22.5**: below it the
prefix is read from HBM once per layer and served to the other sequences out of
L2; above it the unique stream evicts it. That predicts a feature the table in
this section already contains and this file first read as noise — med ITL at
`h` = 0.8 rises ~0.2 ms/seat across the sweep and jumps **once**, by 3.31 ms
between c = 24 and c = 26, sixteen times the neighbouring increments. At `h` = 0
no such kink exists, and should not: there the working set is 137 MB at the very
first level, over L2 from the start, which is why that sweep reproduces the
memory-bandwidth model to 6 %.

**What splits them**, both on one card and neither needing a profiler:

| experiment | cascade | L2 residency |
|---|---|---|
| `--disable-cascade-attn` at c = 24 and c = 40 | step returns to the `h` = 0 slope | nothing changes |
| prefix 12 800 instead of 3 200, same `h` | saving holds or grows | saving nearly vanishes — the prefix alone is 52 MB per layer |

Until one of those runs, the honest statement is the one this section opened
with: the fall is measured, the mechanism is not.

### The term §6 actually argued about scaled exactly as predicted

Interference — TPOT p50 minus median ITL, the prefill work a request's decode
stream absorbs:

| | fit over the sweep | slope ratio |
|---|---|---|
| `h` = 0 | `I(n) = 1.559 n − 4.52` ms | — |
| `h` = 0.8 | `I(n) = 0.302 n − 1.95` ms | **0.194** |
| predicted | scale by (1 − `h`) | 0.200 |

Run 1's fit, from a different pod, was `1.640 n − 6.36`. The (1 − `h`) scaling is
the assertion §6 made about a *mechanism* — that the interference is prefill work
per unit time, and `h` removes that share of it — and it survives to within 3 %
of its own slope. The seat prediction missed anyway, because it held the decode
step fixed while the measurement moved it.

### The worst step did **not** halve

The runsheet predicted ITL p99 near 106 ms on the cached sweep against 198 ms
measured in run 2, on the argument that the chunk a decode step waits behind is
now 800 tokens rather than 4 000.

| c | 8 | 16 | 24 | 28 | 40 | 56 |
|---|---|---|---|---|---|---|
| ITL p99, `h` = 0 | 179.0 | 198.7 | 212.6 | 219.0 | — | — |
| ITL p99, `h` = 0.8 | 39.9 | 181.2 | 199.0 | 203.5 | 212.5 | 234.2 |

Uncached, the worst step is 179 ms from the very first level — which is
`max_num_batched_tokens` = 2 048 tokens of prefill at `mfu` 0.439, 179.1 ms,
exactly. Cached, it starts far below that and climbs to the same ceiling by
c = 16: 2 048 / 800 = 2.6, so as soon as three requests prefill in the same step
the chunk budget is full again and the worst step is the chunk, not the request.

**Prefix caching halves the mean prefill work and leaves the tail step where the
chunk size put it.** The knob for the tail is still `max_num_batched_tokens`,
priced at two seats for 6× TTFT p50 in run 2 §4 — and the two knobs are therefore
complements, not substitutes.

### TTFT

TTFT p50 fell 25–41 % at matched concurrency (672.5 → 394.0 at c = 8; 793.8 →
590.5 at c = 24). The runsheet set the bar at "more than ~40 % says anything",
because TTFT p50 spans 32 % across identical launches, so only the c = 8 point
clears it and the honest reading is **direction confirmed, magnitude not
measurable on this instrument.** No level in either sweep recorded a single
preemption, and none breached the prefill floor from below.

---

## 5. Block B — the peak moves 3.5×, and `max_num_seqs` is what finally binds

1 500-token prompts, Poisson arrivals, `h` = 0.8 — run 2's block A geometry with
one thing changed, so run 2's curve is the control.

| Offered | goodput | throughput | TTFT p50 | TTFT p99 | TPOT p50 | med ITL | max running |
|---|---|---|---|---|---|---|---|
| 1.5 | 1.44 | 1.44 | 87.1 | 109.4 | 25.02 | 24.23 | 13 |
| 2.5 | 2.08 | 2.08 | 88.5 | 125.8 | 25.59 | 24.76 | 21 |
| 4.0 | 3.65 | 3.65 | 97.6 | 143.3 | 29.01 | 26.59 | 36 |
| 5.0 | 4.51 | 4.51 | 108.4 | 153.8 | 31.42 | 28.31 | 41 |
| 6.0 | 5.81 | 5.81 | 124.0 | 256.0 | 36.69 | 31.40 | 83 |
| 7.0 | **6.57** | 6.57 | 135.0 | 238.6 | 40.48 | 32.91 | 69 |
| 9.0 | 2.59 | 8.35 | 181.3 | **325.2** | **59.26** | 40.07 | 124 |
| 11.0 | 0.30 | 9.09 | 456.8 | 7 590 | 118.28 | 91.52 | **256** |
| 13.0 | 0.00 | 9.15 | 11 402 | 30 612 | 136.82 | 151.06 | **256** |

**Goodput peaks at 6.57 req/s against run 2's 1.87** — 3.5×, and up to 7 req/s
every single request met both SLOs, so goodput and throughput are the same
number. The runsheet predicted "at least 3.5 req/s" and was right about the
direction and low on the size.

**Which SLO breaks first, now that the pool cannot.** At `h` = 0.8 a seat costs
500 tokens and the prefix is stored once, so the pool seats hundreds and never
binds: **no preemptions at any rate, including 13 req/s**. What binds instead is
`max_num_seqs` = 256, reached at 11 req/s with 65 more requests queued behind
it — the first time in three runs that this repository has seen the scheduler's
sequence ceiling, rather than KV space, end a curve. Between 7 and 9 req/s both
latency SLOs go together: TPOT p50 40.5 → 59.3 and TTFT p99 239 → 325.

Two runsheet predictions missed in the same direction and for the same reason:
TPOT p50 was predicted to cross 50 ms between 5 and 6 req/s (measured: between 7
and 9) and TTFT p99 to cross 300 ms between 2.5 and 4.0 (measured: between 7 and
9, against 0.5–1.0 uncached in run 2). Both were derived with the decode step
held fixed. §4 is why they are low.

**The generator kept its schedule.** Maximum lateness was 24.4 ms at 13 req/s,
against a 77 ms mean arrival interval — the open loop stayed open even where the
server was 100 requests behind, so the TTFT figures above are the server's and
not the harness's.

---

## 6. Block C — the feature costs nothing when it buys nothing

One level, c = 13, 4 000-token prompts, `h` = 0, run on both servers within
fourteen minutes of each other on one pod, differing in the flag alone.

| | control (`--no-enable-prefix-caching`) | treatment (caching on) | Δ |
|---|---|---|---|
| median ITL | 33.677 ms | 33.686 ms | **+0.027 %** |
| TPOT p50 | 48.62 | 48.84 | +0.45 % |
| TPOT p99 | 50.62 | 50.88 | +0.51 % |
| ITL p99 | 187.8 | 189.1 | +0.7 % |
| TTFT p50 | 700.8 | 706.1 | +0.76 % |

Predicted: under 1 % on the decode step, because hashing is a per-block CPU cost
on the prefill path while the decode step is a bandwidth quantity. Measured:
0.027 %, which is a ninth of the 0.24 % the quantity reproduces to — i.e. **zero,
as far as this instrument can tell**. The TTFT effect is +0.8 % against a 32 %
instrument spread and is reported as unmeasurable rather than as small.

The operator's version: on this card there is no latency reason to run with
prefix caching off. The remaining cost is pool space held by finished requests'
blocks, which this run could not resolve either (§2).

### The decode step, four launches, three pods

| run 1 | run 2 | run 2 | run 3 control | run 3 treatment |
|---|---|---|---|---|
| 33.79 | 33.71 | 33.80 | **33.677** | **33.686** |

Five measurements of one quantity at c = 13, ctx 4 100, MNBT 2 048, spanning
three pods, two drivers and twelve days: total spread **0.37 %**. This is the
quantity the repository is entitled to hang thresholds on, and run 3 is the
second run to say so.

### Cold start, decomposed — the largest removable term is not the weights

Read out of `raw/l40s-2026-08-30/run3/serve-cache-off.log` and `serve-cache-on.log`
after the run, for a question the run was not designed to answer: what
`controllers/modelwarmup/` would actually remove. Both are launches of vLLM
v0.27.1 on the same pod, their engine-init lines 4 min 14 s apart (`core.py:121`,
09:45:20 and 09:49:34); the first found an empty `~/.cache/vllm` and an empty HF
cache, the second found both warm.

| Term | Log line | Cold | Warm |
|---|---|---|---|
| Download weights, 16.4 GB | `weight_utils.py:530` | 16.54 s | absent |
| Load weights to GPU | `default_loader.py:430` | 2.56 s | 2.78 s |
| `torch.compile` | `monitor.py:53` | 34.95 s | **0.20 s** |
| Profile, KV pool, CUDA graph capture | `core.py:348` minus the row above | 15.11 s | 13.49 s |
| **Total to first served token** | | **69.16 s** | **16.47 s** |

The last row is the floor and the first three rows are the argument. Two things
in the table were not expected.

**The compiled artefacts are 6 MB.** `backends.py:920` reports what was saved:
`37 entries, 3 artifacts, 6283572 bytes total` — 5.99 MiB, against 16.4 GB of
weights (`docs/SLO.md` §3). Per unit of cache carried, the two terms are not
comparable: 5.80 s per MiB for the compilation, 0.00106 s per MiB for the
weights, a ratio of ~5 500. **The expensive part of a cold start is a
computation, not a download**, and its result fits in an image layer.

**The floor is real and it agrees with itself.** Subtracting compilation from
`init engine` leaves 15.11 s cold and 13.49 s warm — two independent readings of
the same quantity, 11 % apart. Nothing on disk removes it: it is profiling, KV
pool construction and CUDA graph capture, and `docs/GLOSSARY.md` already says
why the last of those can never be cached (graphs are captured into GPU memory
and have no on-disk form). Adding the unavoidable 2.6 s read of the weights into
the card, **no amount of warming takes a replica below ~17 s.**

Three qualifications, because the file names invite a wrong attribution:

- **The 52.7 s is not what `cache-off`/`cache-on` measured.** Those names refer to
  the prefix caching flag, the subject of this section. The startup difference is
  an artefact of launch *order* — the second start reused what the first one
  wrote. The run measured it without intending to, and the numbers are only
  admissible as a cold-vs-warm pair, not as an effect of the flag.
- **The caches were inside the container.** `/root/.cache/vllm/...` and the HF
  cache do not survive the pod. Across pods this saving exists only if
  `VLLM_CACHE_ROOT` and the HF cache point at a volume that outlives the
  container (`docs/GLOSSARY.md`, `VLLM_CACHE_ROOT`) — which is the whole design
  question, not a detail.
- **The download rate is this platform's, not a constant.** 16.4 GB in 16.54 s is
  0.99 GB/s ≈ 7.9 Gbit/s, RunPod's network. The two terms trade places at
  16.4 GB / 34.75 s = 472 MB/s: **below ~3.8 Gbit/s the weights dominate the
  compilation**, and on a 1 Gbit/s link the download is 131 s and nothing else
  matters. Which term to attack first is a property of the cluster's network and
  is read, not assumed.

One observation about the cache key, offered as a hypothesis rather than a
result. The AOT directory hash is *identical* across the two launches —
`torch_aot_compile/80b170aaf97ebd5ff45eb73b7c36d2b7d30020ad803455ac0d2f0a3bcbc8c5b2/`
in both — although they differ in `enable_prefix_caching` (`core.py:121` of each
log). So that flag is not part of the key, which is consistent with it changing
the block manager rather than the compiled graph. What the key *does* contain was
not established here: model, dtype, GPU architecture and vLLM version are the
expected members and none of them varied. The cheap test on the next pod is to
change `--max-model-len` and read the hash.

---

## 7. Cost

Both seat counts, converted at $0.99/h into the only figure a customer's price
list cares about — a token someone can actually use, at the largest batch that
holds the SLO:

| | seats | output tok/s | $ per M SLO-respecting output tokens |
|---|---|---|---|
| `h` = 0 | 12 | 233.9 | **$1.18** |
| `h` = 0.8 | 32 | 703.9 | **$0.39** |

3.0× on the seat count is 3.0× on cost, because within the SLO the card is
serving three times the tokens for the same hourly rate. That number is available
to any operator whose traffic repeats — and to no operator whose traffic does
not, which is the point §6 makes about `h` being a property of the workload.

Run cost: ≈ $1.65 of a $2.48 budget, ~$0.25 of it on the rejected pod and ~$0.12
on the re-run sweep. Three runs to date: $1.45 + $1.44 + $1.65 = **$4.54**.

---

## 8. What run 3 could not measure

- **One prefix, not a distribution.** Every request shared the same 3 200 tokens.
  Real traffic has many prefixes of many lengths and the pool evicts between
  them; this run measures the ceiling of what `h` is worth, not what a fleet sees.
- **`h` remains an input.** Nothing here measures the hit rate of real traffic.
- **The mechanism behind the decode-step fall** (§4). Two named candidates now —
  vLLM's per-step cascade decision, and L2 residency of the shared prefix — and
  nothing in this run separates them. The test is a `--disable-cascade-attn` A/B
  at c = 24 and c = 40 plus one sweep at a prefix too large for L2; no profiler
  needed. The 96 MB L2 figure the second candidate rests on is a datasheet
  number and wants a `deviceQuery` on the next pod.
- **Whether prefix caching costs pool space.** The 4.3 % between the two configs
  is inside the launch-to-launch noise (§2).
- **The chunk size, from the log.** Defaults were used and not logged (§2).
- **FP8 KV together with prefix caching.** Both measured, their interaction not.
- **The TTFT tail** (run 2's open item). Block B constrains it — the tail is
  healthy to 7 req/s at `h` = 0.8 — but does not explain run 2's early break.
- **MI300X.** Three runs, three cards' worth of L40S evidence, zero for AMD.

---

## 9. Predicted vs measured — the summary table

Every "predicted" below was written in `docs/benchmarks/runsheets/l40s-run-3.md`
before the card was rented. The sheet is in this repository so the claim can be
checked against it; the ordering rests on its dateline, not on a history that
would prove it.

| # | Prediction | Predicted | Measured | |
|---|---|---|---|---|
| A1 | seat count, `h` = 0 | 12–14 | **12–14** (12.5) | ✅ |
| A1 | seat count, `h` = 0.8 | 22–24 | **32–40** (37.8) | ❌ low by 56 % |
| A2 | decode step unchanged at equal `n` | ±1 % | **−14 % to −32 %** | ❌ falsified |
| A2′ | if cascade engages, step at c = 24 | 27.97 | 29.76 | ❌ between the two (§4) |
| A3 | measured `h` at every cached level | 0.800 ± 0.02 | **0.800** | ✅ |
| A4 | TTFT p50 falls at matched `n` | > 40 % to count | 25–41 % | ~ direction only |
| A5 | no preemptions in either sweep | 0 | **0** in all 45 levels | ✅ |
| A6 | ITL p99 halves under caching | ~106 ms | **199 ms** | ❌ chunk-bound |
| — | interference scales by (1 − `h`) | 0.200 | **0.194** | ✅ |
| B1 | goodput peak | ≥ 3.5 req/s | **6.57** | ✅ (low) |
| B1 | pool is not the collapse mechanism | no preemptions | **0**, `max_num_seqs` binds | ✅ |
| B2 | TPOT p50 crosses 50 ms | 5–6 req/s | **7–9** | ❌ low |
| B3 | TTFT p99 crosses 300 ms | 2.5–4 req/s | **7–9** | ❌ low |
| B4 | generator lateness < 1 interval | — | 24.4 ms of 77 | ✅ |
| C1 | decode-step cost at `h` = 0 | < 1 % | **+0.027 %** | ✅ |
| C2 | TTFT cost at `h` = 0 | unmeasurable | +0.8 %, unmeasurable | ✅ |
| A″ | KV pool | 169 833 ± 5 % | 168 985 / 176 227 | ✅ |
| A″ | startup-log keys present | 7 of 8 | **6 of 8** | ❌ |
| A″ | token-ID prompts accepted | unknown | **yes** | ✅ |
| A″ | `prefix_cache_queries` counts tokens | unknown | **yes**, `h` = 0.500 | ✅ |

Eleven held, seven missed, and the seven have one cause between them: every
prediction that held the decode step fixed while `h` moved came out low. That is
one wrong assumption, not seven wrong numbers, and `docs/SLO.md` §6 has been
corrected accordingly.
