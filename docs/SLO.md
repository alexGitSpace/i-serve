# Service Level Objectives

What this platform promises, and why each number is the number. Every threshold
below is either derived from hardware arithmetic or from user perception — none
of them are picked by feel.

This is the single home for the derivations: other files in the repo link to a
section here instead of repeating a number, so a revised coefficient has to be
changed in exactly one place. Terminology and symbols are defined in
[GLOSSARY.md](GLOSSARY.md); the code that computes these figures is
`bench/roofline.py`.

---

## 1. Workload classes

The stack serves two classes with genuinely different economics. Treating them
as one workload is the most common way to overpay for inference.

**Interactive** — a human is watching text appear. Latency *is* the product.

**Batch** — a document pipeline consuming millions of tokens per day with no one
waiting on any individual response. The real obligation is *"N million tokens
processed by 07:00"*: a throughput-and-deadline commitment, not a latency one.

The per-token threshold for batch exists only as a guard against pathological
stalls. Setting it tightly would cap batch size, cut throughput and raise cost
per million tokens — paying real money to improve a metric no one observes.

---

## 2. Targets

| Class | TTFT p99 | TPOT p99 | The SLO that actually matters |
|---|---|---|---|
| Interactive | 300 ms | 50 ms | latency |
| Batch | 3 000 ms | 200 ms | tokens delivered before deadline |

Percentiles, not means: inter-token *consistency* drives perceived quality more
than raw speed. A steady 25 ms reads better than a jittery 20 ms, and a mean
hides exactly the stalls users notice.

### Why 50 ms for interactive

An adult reads 200–300 words per minute — roughly 4–5 words/s. At ~0.75 words
per token, merely *keeping pace* with a reader needs only ~6 tokens/s, i.e.
160 ms per token. That feels sluggish, because text lands exactly where the eye
already is, with no buffer.

50 ms per token is 20 tokens/s: about three times reading speed, so output stays
comfortably ahead of the reader. Beyond roughly 40 tokens/s the difference stops
being perceptible, and GPU time spent there buys nothing. The upper bound on an
SLO is as much an engineering decision as the lower one.

---

## 3. Baseline for all derivations

| | Value | Source |
|---|---|---|
| Model | Qwen3-8B, BF16 | `config.json` |
| Parameters, total | 8.2e9 | model card — used for memory |
| Parameters, non-embedding | 6.95e9 | model card — used for FLOPs |
| Layers `L` | 36 | `num_hidden_layers` |
| KV heads `n_kv` | 8 | `num_key_value_heads` (GQA, 32 Q heads) |
| `head_dim` | 128 | `head_dim` |
| Accelerator | 1 × AMD Instinct MI300X | — |
| HBM3 capacity | 192 GiB (206.2e9 B) | AMD's ROCm spec table lists 192 GiB; the data sheet prints "192 GB"; `rocm-smi`'s 205 822 885 888 B is 192 GiB less 320 MiB |
| Peak memory bandwidth | 5.3 TB/s | AMD spec table |
| Peak BF16, **dense** | 1.307 PFLOP/s | AMD spec table — *not* the sparsity row |
| Assumed achieved bandwidth | 70% of peak | empirical |
| Assumed MFU | 45% | empirical, prefill |

Derived constants:

```
weights_bytes      = 8.2e9 × 2                    = 16.4 GB
kv_bytes_per_token = 2 × 8 × 128 × 2 × 36         = 147 456 B  (147 KB)
```

147 KB of HBM per token of context. GQA is what keeps it there — without shared
KV heads it would be 590 KB, four times the memory bill for identical output.

---

## 4. Floors

A *floor* is the best the hardware can physically do. Anything measured above it
is queueing, overhead or misconfiguration — never the model itself. The floor is
what turns "TTFT is 800 ms, fix it" from guesswork into a decision.

### TPOT floor — decode, batch = 1, empty context

Taken at zero context, where the KV term vanishes and only the weights are read.
That is what makes it a *floor*: any real sequence carries context and moves more
bytes. §5 prices the same step at a realistic context length.

Each side is divided by its own coefficient *before* the `max()`, so the two
times being compared are both achievable ones — `eff_mem` for memory, `mfu` for
compute. Comparing a derated time against a peak time mixes bases and inflates
the ratio; the notation in [GLOSSARY.md](GLOSSARY.md) (`t_mem_real`,
`t_compute_real`) is the same convention.

```
t_mem     = 16.4e9 / 5.3e12               ÷ 0.70  =  4.42 ms   → 226 tok/s
t_compute = 2 × 6.95e9 × 1 / 1.307e15     ÷ 0.45  =  0.024 ms
max()                                             =  4.42 ms   memory-bound, 187×
```

187× is not "somewhat memory-bound". During decode the matrix cores idle 99.5% of
the time and the accelerator behaves as an expensive memory bus. Every batching
mechanism in this stack exists to reclaim that idle capacity.

The 187× is a *lower* bound on how memory-bound decode is. `mfu = 0.45` was
measured on prefill, where the matmuls are large; a decode step multiplies a
matrix by a single vector per sequence and reaches nowhere near that utilisation.
Using the prefill coefficient therefore overstates the compute side, and a
measured decode MFU would only widen the gap — see §9.

### TTFT floor — prefill, 2 000-token prompt

```
t_mem     = (16.4e9 + 2000 × 147 456) / 5.3e12  ÷ 0.70  =  4.50 ms
t_compute = 2 × 6.95e9 × 2000 / 1.307e15        ÷ 0.45  = 47.27 ms
max()                                                   = 47.3 ms   compute-bound, 10.5×
```

The memory side carries the KV the prompt *writes*, which an earlier version of
this section omitted. It is 0.29 GB against 16.4 GB of weights here and does not
move the verdict, but it grows linearly with the prompt: at a 32 000-token
reasoning prompt it is 4.72 GB, and stating it is what keeps the two phases
symmetric — decode reads KV, prefill writes it. Not modelled either way:
attention re-reading the KV it has already written as the prompt is consumed.

Same model, same silicon, inverted roofline. Prefill and decode are physically
different regimes, which is why they get separate tuning knobs.

### The same floors on the measurement card

Everything above assumes MI300X (§3). The first measurements are taken on a rented
**NVIDIA L40S** — 48 GB GDDR6, 864 GB/s, 362 TFLOP/s BF16 **dense** (the datasheet
prints "362 / 733", and 733 is the sparsity figure). The MI300X runs come later, on
credits reserved for them; the reason for two cards, and the requirement that every
predicted-vs-measured row name the one it used, are in §9.

| | MI300X, derived | L40S, derived | **L40S, run 1** |
|---|---|---|---|
| Memory | 192 GiB @ 5.3 TB/s | 48 GB @ 864 GB/s | 44.39 GiB visible |
| BF16 dense | 1.307 PFLOP/s | 362 TFLOP/s | — |
| `achieved_bandwidth` | 0.70 assumed | 0.70 assumed | **0.83 measured** |
| `mfu` | 0.45 assumed | 0.45 assumed | **0.439 measured** |
| TPOT floor, batch 1, empty context | 4.42 ms | 27.1 ms | **22.9 ms** |
| TTFT floor, 2 000-token prompt | 47.3 ms | 170.6 ms | 174.8 ms |
| Queue share of the 300 ms TTFT budget | 84% | 43% | 42% |
| KV space at `gpu_memory_utilization = 0.9` | 169 GB | 26.8 GB | **24.9 GB** |
| Sequences at 4 000 context — memory allows | 286 | 45 | **41** |
| Sequences at 4 000 context — a 50 ms TPOT allows | 286 | 23 | **32** decode step, **12** as served |
| Which limit binds | a tie, counted as latency | latency, by 2× | latency, by 3.4× |

The third column is measurement, not derivation, and it is **L40S only** — the
MI300X column keeps its uncalibrated coefficients until a run on that card
faces them (§9). Every figure in it, and the difference between 32 and 12 in the
second-to-last row, is derived and argued in
[benchmarks/l40s-baseline.md](benchmarks/l40s-baseline.md).

The last row is why this card was chosen for the first runs rather than a
cheaper one.
On MI300X the two derived limits coincide at 286 (§6), and a coincidence
teaches nothing about which one is which. On L40S the SLO permits 23 sequences
while memory holds 45, so `max_num_seqs` has to be *derived* rather than read off
the capacity — and `min(SLO limit, memory limit)` becomes a decision instead of a
tie. The choice paid off in a way the derivation did not anticipate: measured,
the two limits are **12 and 41**, further apart than predicted and for a reason
no coefficient covers (§9).

An RTX 4090 was rejected on the same arithmetic: at 1 008 GB/s its decode floor is
better than the L40S's (23.2 ms), but 165 TFLOP/s dense puts its TTFT floor at
374 ms for a 2 000-token prompt — above the *entire* 300 ms budget, making the
interactive class physically unreachable. An L4 fails earlier still: 300 GB/s gives
a 78 ms TPOT floor at batch 1, against a 50 ms target.

### What the floors mean for the targets

```
TTFT budget   300 ms  =  47 ms prefill  +  253 ms of everything else   (84%)
```

The overwhelming majority of the TTFT budget is queue wait, not computation.
**TTFT is a provisioning problem before it is an optimisation problem.** This is
the direct justification for autoscaling on `vllm:num_requests_waiting` rather
than on GPU utilisation: the metric that predicts an SLO breach is queue depth,
and GPU utilisation saturates long before latency degrades.

### What the queue budget makes the threshold — and what it does not

The scaler's `threshold` is a queue depth per replica, so the budget above has to
be converted into one. On the measurement card, at the interactive operating
point (§6: 12 seats as served, 200 output tokens at the 50 ms TPOT target):

```
queue budget       300 ms - 174.8 ms  =  125.2 ms
service time/seat  200 tokens x 50 ms =   10.0 s
completion rate    12 seats / 10.0 s  =    1.20 /s      <- all seats busy
a seat frees every 1 / 1.20 /s        =    833 ms
tolerable depth    125.2 ms / 833 ms  =    0.15 requests
```

The last line is the finding: **the tolerable depth is below one request.** A
single queued request has already spent 833 ms waiting — nearly seven times the
whole queue budget — so no positive integer threshold preserves interactive TTFT.
Queue depth is not an early warning here; by the time the gauge reads 1, the
request it represents has missed.

Three consequences, and they are the reason the threshold is a policy number
rather than a derived one:

1. **Autoscaling protects goodput, not per-request TTFT.** What a replica buys is
   a shorter *overload*, not a faster queued request. Per-request TTFT is bought
   with headroom — `minReplicaCount` above the mean load — which is a cost
   decision (§7), not a scaler setting.
2. **The control loop is minutes, not milliseconds.** Scrape interval, plus
   KEDA's 30 s default poll, plus the HPA's own window, plus a pod that loads
   16 GB of weights before it serves anything. A burst shorter than that loop
   cannot be answered by scaling at all — it can only be answered by seats that
   already exist.
3. **The trade is depth against churn.** Accepting 1 s of added queue wait puts
   the threshold at 1.2; 5 s at 6.0; a full service time at 12.0, which is the
   seat count itself. Below ~1 the scaler reacts to every transient; at 12 it
   waits until a replica's worth of work is already queued.

The manifests take `threshold: "1"` — the smallest value the arithmetic permits
anyone to defend — and bound the resulting eagerness with a scale-down
stabilisation window and a replica cap, not with a larger threshold. Derivation
of the seat count is §6; the manifests are `deploy/keda/`.

### The alert rule

§10 asked for three things: which SLI, which window, which burn rate.

**SLI.** The share of requests whose TTFT is inside the budget, read from vLLM's
`vllm:time_to_first_token_seconds` histogram as good over total — not
`histogram_quantile(0.99, …)`. The histogram's bucket edges are 0.25 s and 0.5 s
with nothing at 0.3, so a p99 evaluated at the target would be a linear
interpolation between two bucket counts, a number the engine never measured.
The rule reads the `le="0.25"` bucket instead: stricter than the target by
50 ms, never looser. The p99 target restated as a ratio is 99 % good events and
a 1 % error budget.

**Burn rate.** The error share divided by the 1 % allowed. A burn rate of 1
spends the budget exactly over its period; 14.4 spends 2 % of a 30-day budget in
one hour, the conventional page threshold. The period is a policy number: §2
sets a target, not a budget period, and 30 days stands until a business owner
sets one.

**Window.** Two, and both must hold: 1 h says the burn is real, 5 m says it is
still happening, so the page resolves when the cause does. `for: 2m` keeps one
evaluation from paging on its own.

**What `kind` can test.** None of the above — the stub exports no histogram. The
rule that can be tested is the consequence this section already derived: a
queued request has already missed the budget, so `vllm:num_requests_waiting`
at or above 1 on any one replica *after the scaler has had time to act* is a
breach the scaler did not clear. Its `for` window is longer than the control
loop on either instrument — 5 s scrape, 15 s HPA sync, and a pod start of
seconds on `kind` or 69 s cold on the L40S ([benchmarks/l40s-run3.md](benchmarks/l40s-run3.md) §6).
The rules are `deploy/observability/prometheus/rules.yml`; the queue rule was
watched firing on `kind` and is written up in `deploy/observability/README.md`.

---

## 5. The batching trade-off

Weights are read once per forward pass regardless of batch size; KV is read per
sequence. That asymmetry is the entire economics of serving.

Decode step, 4 000 tokens of context per sequence:

| | batch 1 | batch 64 |
|---|---|---|
| KV in `bytes_moved` | 0.59 GB | 37.75 GB |
| `bytes_moved` | 16.99 GB | 54.15 GB |
| TPOT | 4.58 ms | 14.6 ms |
| tok/s per user | 218 | 68 |
| **tok/s aggregate** | **218** | **4 384** |
| memory : compute ratio | 194× | 9.6× |

20× the throughput for 3.2× the per-user latency. Tuning is the search for a
point on this curve; the SLO is what says where to stop.

Both columns carry the 4 000 tokens of context stated above, which is why the
batch-1 figure here is 4.58 ms rather than the 4.42 ms floor in §4: a floor
assumes an empty cache, an operating point does not. The KV term is only 3.5% of
the traffic at batch 1 and 70% of it at batch 64 — that swing, not either number
alone, is what batching actually is.

---

## 6. Concurrency ceiling

```
KV space   = 192 GiB × 0.9 (gpu_memory_utilization) − 16.4 GB = 169 GB
           = 169e9 / 147 456                                  ≈ 1.15 M KV tokens
             ≈ 286 sequences at 4 000 tokens of context

at that fill:  bytes_moved = 185.1 GB  →  34.9 ms  ÷ 0.70  =  49.9 ms TPOT
```

Two conclusions:

**Memory capacity and the interactive latency budget bind at the same point.**
A full card lands at 49.9 ms TPOT against a 50 ms target. (Derived at 192e9 B,
the two sat 8% apart.) Raising
`gpu_memory_utilization` therefore buys nothing — the SLO breaks at the same
moment the memory runs out. It buys nothing on the L40S either, for the opposite
reason: there the latency limit is reached at 23 sequences while 45 already fit,
so the spare seats are idle capacity, not a constraint. Measured on the L40S the
gap is wider still, 12 against 41 — the idle seats are real, and paying for more
capacity would buy more of them.

**`max_num_seqs` is derived, not discovered** — the `min()` of a latency limit
and a capacity limit, never trial and error. Which of the two binds is a
property of the card and the target rather than a general rule: on MI300X
the two tie (286 fit, 286 the SLO permits, and a tie counts as latency-bound),
on L40S latency binds (23 permitted against 45 that fit; measured, 12 against
41). Quoting one of the two
without the other is how a card ends up configured against the constraint it is
not actually against — and quoting either without saying whether it is derived
or measured is how a run-1 coefficient ends up in a run-2 decision (§9).

The batch class cannot exploit its looser 200 ms threshold on a single
accelerator: memory binds first, at the same 1.15 M KV tokens. Its lever is
capacity, not scheduling.

### Highest-leverage knob: FP8 KV cache

Halving KV bytes per token doubles the number of tokens that fit in the same
169 GB, while `bytes_moved` at full occupancy — and therefore TPOT — stays put.

```
FP8 KV:  73 728 B/token  →  2.29 M KV tokens  →  ~573 sequences at 4 k context
         TPOT unchanged at ~49.9 ms, aggregate throughput ~2×
```

TPOT holds because `bytes_moved` at the new ceiling is the one it replaced:
2 × 286 × 4 000 × 73 728 is the same 169 GB of KV as 286 × 4 000 × 147 456 (the
ceiling itself floors to 573).

**What FP8 KV cannot do is change which limit binds.** Both limits have the form
`n = (X − weights) / (context_len × kv_bytes_per_token)`, differing only in `X`
— `tpot_target × bandwidth × eff_mem` for latency, `memory_bytes ×
gpu_memory_utilization` for capacity. `kv_bytes_per_token` divides both, so
halving it doubles both and cancels out of their ratio, as does `context_len`:

```
ratio = (X_latency − weights) / (X_memory − weights)
```

MI300X therefore stays tied at FP8, and the L40S stays
latency-bound by the same 2×. The terms left in that expression name the knobs
that *do* move it: the weights (quantise them), the bandwidth (a different
card), or the target itself.

Both halves were falsifiable predictions. **Run 2 tested them on the L40S and
both hold** ([benchmarks/l40s-run2.md](benchmarks/l40s-run2.md) §5):
`--kv-cache-dtype fp8` moved the logged pool from 169 833 to 339 666 tokens —
2.000×, exact — seats at 4 100 context from 41 to 82, and the decode step at 26
sequences came out 4.8% from the step at 13 sequences in BF16, which is the "same
TPOT at twice the occupancy" claim within the run's own noise. Which limit binds
did not move: latency still binds, 65 against 90 at FP8 where it was 32 against
45, and the ratio is 1.38 against 1.41.

Two things the derivation above did not anticipate, both from the same run:

- **FP8 KV is also faster at equal concurrency**, by 14% at 13 sequences and 25%
  at 32. The claim in this section is about the *ceiling*, where the byte count is
  held constant; below the ceiling the halved KV term simply reduces
  `bytes_moved`, so the same arithmetic that predicts the ceiling predicts the
  speed-up, and did — to under 5% at three concurrencies.
- **On this card FP8 KV forces a different attention kernel.** FlashAttention 2
  does not support an FP8 cache on sm89, so vLLM silently selected FlashInfer. A
  control at BF16 with the kernel forced measured the kernel worth 0.8%, so the
  gain is the cache — but "change one thing" was violated by the engine, not by
  the operator, and only the startup log said so.

### The knob that could close the rest: prefix caching

Run 2 left nineteen seats unaccounted for and one candidate eliminated. The
decode step permits **31** sequences at 4 000 tokens of context; the number an
operator can promise against a 50 ms TPOT is **12**, and
`max_num_batched_tokens` buys two of the missing nineteen at six times the TTFT
p50 ([benchmarks/l40s-run2.md](benchmarks/l40s-run2.md) §4). The knob only
decides *who waits* for prefill work the load creates. Closing the rest requires
removing the work.

**Prefix caching has three channels, and two of them are irrelevant on this
card.** It reuses the KV of a shared prompt prefix across requests, so:

1. ~~It does **not** move the latency limit.~~ **Measured false, run 3**
   ([benchmarks/l40s-run3.md](benchmarks/l40s-run3.md) §4). The argument was that
   a sequence reads its whole context at every decode step whether or not another
   sequence stored the same bytes — one physical copy in HBM, `n` reads per step,
   so caching saves *computing and writing* KV and never reading it. The decode
   step is the one quantity here reproducible to 0.24 %, and at nine matched
   concurrencies it **fell by 14–32 %** when 3 200 of 4 000 prompt tokens were
   shared. Its slope went from 0.895 to 0.367 ms per seat, against 0.843
   predicted: the per-seat cost behaves as ~1 785 tokens of context where a seat's
   unique content is ~900. Cascade attention would have taken it to ~900 tokens
   and 0.185 ms per seat, so **the fall is real and neither hypothesis fits it**.
   The mechanism is unidentified and is §10's first open item; what this section
   may no longer assume is that the latency limit is independent of `h`.
2. It **does** move the capacity limit, and hard: the shared blocks are stored
   once instead of once per sequence. With a 3 200-token prefix shared by every
   request, the pool's 169 833 logged tokens seat `(169 833 − 3 200) / 900 ≈ 185`
   sequences instead of 41. On the L40S that is worth nothing — latency binds at
   a third of even the old figure — and on the MI300X, where the two derived
   limits tie (§6), it would be the whole story. The same flag is a capacity knob
   or a no-op depending on the card.
3. It removes **prefill work**, which is the term that actually separates 12 from
   31. This is the channel the section was built on, and it is the one that came
   through the measurement intact: run 3 fitted the interference at `h` = 0.8 to
   0.302 `n` against 1.559 `n` at `h` = 0 — a ratio of **0.194 where (1 − `h`) =
   0.200**.

**The third channel, priced.** The interference is measurable directly as TPOT
p50 minus median ITL — the decode step a request experiences minus the step the
hardware performs ([benchmarks/l40s-baseline.md](benchmarks/l40s-baseline.md)
§5), at `max_num_batched_tokens` 2 048 and 4 000-token prompts:

| c | 13 | 23 | 32 |
|---|---|---|---|
| Decode step (median ITL), ms | 33.99 | 43.07 | 50.69 |
| Inflation (TPOT p50 − median ITL), ms | 14.96 | 31.01 | 46.13 |

Linear in `n` to within 1% across the range that matters: `I(n) ≈ 1.640 n −
6.36` ms. A cache hit rate `h` — the share of prompt tokens served from the
cache — removes that share of the prefill work, so the served step becomes

```
TPOT(n, h)  =  step(n) + (1 − h) × I(n)
step(n)     =  (weights + n × context_len × kv_per_token) / (bandwidth × eff_mem)
            =  22.87 + 0.843 n   ms      (L40S, eff_mem = 0.83, 4 100 tokens)
```

Solving `TPOT(n, h) = 50 ms`:

| Cache hit rate `h` | 0 | 0.25 | 0.50 | 0.70 | 0.80 | 0.90 | 0.95 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| Seats at 50 ms TPOT p50 | **13.5** | 15.4 | 18.2 | 21.8 | **24.3** | 27.6 | 29.7 | **32.2** |
| Of the 18.7 missing, closed | 0% | 10% | 25% | 44% | 58% | 75% | 87% | 100% |
| **Measured, run 3 (TPOT p99)** | **12.5** | — | — | — | **37.8** | — | — | — |

**The two measured cells are the whole of what run 3 changed here.** The `h` = 0
cell lands within a seat of the row above it, which is the model working. The
`h` = 0.8 cell is 56 % higher than the row predicts, and the row is low for one
reason: it holds `step(n)` fixed while `h` moves, and channel 1 above says why
that seemed safe. The interference half of the formula was right to 3 %; the
step half was wrong, in the direction that gives an operator *more* than the
arithmetic promised. Until the mechanism behind that is identified (§10), **this
table is a floor at high `h`, not an estimate**, and the honest way to quote it
is "at least this many seats".

The model is stated on the TPOT **p50** basis because that is where three
measured points exist; the SLO metric is p99 and sits about one seat lower
throughout (measured: p50 crosses between 13 and 14, p99 at ~12). Its own
zero-cache prediction, 13.5 seats, reproduces that measured p50 crossing — which
is the only validation it has, and it is a validation against the data it was
fitted to.

**Three things an operator reads off that row.**

- **The payoff is convex in `h`.** Caching half the prompt closes a quarter of
  the gap, not half of it, because the interference term shrinks and the decode
  step does not. Below roughly `h = 0.5` this knob is not the answer to a seat
  count; above 0.8 it is most of the answer.
- **`h` is a property of the traffic, not of the card.** Every other knob in this
  document is decided by hardware and a target. This one is decided by whether
  customers send the same system prompt — which makes it the first entry here
  that cannot be derived at all, only measured against a real workload, and the
  first whose value a *product* decision can change.
- **It is the only candidate that moves TTFT and TPOT the same way.**
  `max_num_batched_tokens` trades one for the other, measured at two seats for
  6× TTFT p50. Prefix caching removes prefill work outright, so the same `h` that
  buys seats also cuts the prefill component of TTFT by `(1 − h)`. There is no
  trade to price, only a hit rate to measure.

**What it costs — measured, run 3 §6.** The same level (c = 13, 4 000-token
prompts, `h = 0`) on two servers on one pod, differing in the flag alone, put the
decode step at **33.677 against 33.686 ms: +0.027 %**, a ninth of the 0.24 % the
quantity reproduces to. TTFT p50 moved +0.8 % against a 32 % instrument spread
and is therefore unmeasurable rather than small. **On this card there is no
latency reason to serve with prefix caching off.** The cost that remains
unmeasured is pool space: retained blocks occupy the pool after their request
finishes, and the 4.3 % between run 3's two logged pools is inside the 4.2 % of
launch-to-launch noise (§9).

**The fourth channel, now measured and still unnamed.** Point 1 above holds only
if each sequence reads the shared prefix separately. vLLM's V1 engine has a
*cascade attention* path that reads a shared prefix once for the whole batch,
which would move the latency limit too. Run 3 built a detector for it — the
decode step at nine matched concurrencies, against an instrument reproducible to
0.24 % — and got a third answer: the step fell, but by half of what cascade
attention predicts. This build logs nothing about cascade at all — the path is
chosen per step, not at startup — so the question was never the log's to answer.
Two candidates are now written down and neither is measured: vLLM's per-step
cascade cost model, and L2 residency of the shared prefix
([benchmarks/l40s-run3.md](benchmarks/l40s-run3.md) §4 carries the arithmetic and
the experiment that splits them). **A fourth channel exists, it is worth roughly
thirteen seats on this card, and this document cannot yet say which of the two it
is.**

**The verdict on the open item.** Of the three candidates for the remaining
seats, all attack the same quantity — prefill work per unit time in the decode
stream — and differ only in what they charge for it. A **prompt-length cap**
removes the work by refusing it: free in engineering, paid by the product.
**Separating prefill and decode pools** removes the interference without removing
the work, and costs a second accelerator, so it is not a single-card option and
belongs beside the two-replica step. **Prefix caching** removes the work itself,
free on this card, at an effectiveness set entirely by traffic the operator does
not control. It is therefore first to measure and last to promise.

**What run 3 settled, and what it opened.** The prediction was 24 ± 2 seats at
`h ≈ 0.8` against 13/14 uncached, with a measured count near 14 falsifying the
interference model rather than the flag. Measured: **12.5 and 37.8**
([benchmarks/l40s-run3.md](benchmarks/l40s-run3.md)). The interference model
survived — it was the only half of the prediction that faced the card on its own
terms — and the flag beat its own forecast by thirteen seats. Three consequences
this section now carries:

- **The seat payoff is no longer convex-and-bounded.** It was 1.8× on paper and
  is 3.0× measured, which in the only unit a price list uses is $1.18 → $0.39 per
  million SLO-respecting output tokens at $0.99/h.
- **The tail is a different knob.** Prefix caching halves the mean prefill work
  and leaves ITL p99 where `max_num_batched_tokens` put it: the worst step is the
  chunk budget, 179 ms at 2 048 tokens, in both regimes. The two knobs are
  complements, not substitutes.
- **What binds at high `h` is `max_num_seqs`.** With the prefix stored once the
  pool seats hundreds; run 3 raised the arrival rate until 256 sequences ran with
  65 queued, and recorded **zero preemptions at every rate**. The `min()` of a
  latency and a capacity limit that opens this section is unchanged — but at
  `h = 0.8` the default 256 is a third limit and is reached before either.

---

## 7. Cost

```
cost_per_1M_tokens = accelerator_hourly_rate / (aggregate_tokens_per_sec × 3600) × 1e6
```

Rate is deliberately left as a parameter — it is the one input that varies by
provider and contract rather than by physics. `bench/roofline.py` carries the
formula as `cost_per_1m_tokens`.

**Which tokens go in the denominator is the whole question.** The formula is one
line; the figure it produces moves by an order of magnitude with the choice of
what counts as a delivered token, and each of the three L40S runs forced one
choice:

- **All tokens, prompt included** — what a price list quotes. Prompt tokens are
  cheap per unit because prefill is compute-bound and batched (§3), so this
  denominator flatters any long-prompt workload.
- **Output tokens** — what the throughput optimum maximises and what
  `vllm bench serve` reports as throughput. It keeps improving past the point
  where customers stop being served.
- **SLO-respecting output tokens** — `goodput req/s × output length`, good output
  tokens per second. The only denominator an operator with a promise may
  optimise: it is the number that falls when latency targets fail while the
  other two keep rising.

**Measured, L40S at $0.99/h, Qwen3-8B BF16, 200 output tokens per request**, three
runs on one build. The geometry column is why the rows cannot be read across
naively: seats at 4 000-token prompts and rates at 1 500-token prompts are two
workloads on the same card, and run 1 had no goodput instrument — its SLO point
is a TPOT *median*, run 3's is a p99.

| Run | Operating point | Geometry | $/1M output | $/1M SLO-respecting output |
|---|---|---|---|---|
| 1 | c = 1, one user on the card | 4 000-token prompts, closed loop | 6.688 | — |
| 1 | c = 13, last level inside 50 ms TPOT p50 | same | **1.126** | — |
| 1 | c = 32, throughput optimum | same | **0.873** | — |
| 2 | 2.5 req/s, the goodput peak | 1 500-token prompts, Poisson, `h` = 0 | 0.583 | **0.735** |
| 2 | 4.0 req/s, top of the sweep | same | **0.380** | **11.458** |
| 2 | c = 32, BF16 → FP8 KV | 4 000-token prompts, closed loop | 0.890 → **0.738** | — |
| 3 | 12 seats, `h` = 0 | 4 000-token prompts, closed loop, TPOT p99 ≤ 50 ms | — | **1.18** |
| 3 | 32 seats, `h` = 0.8 | same | — | **0.39** |

Sources: [benchmarks/l40s-baseline.md](benchmarks/l40s-baseline.md) §7,
[benchmarks/l40s-run2.md](benchmarks/l40s-run2.md) §7,
[benchmarks/l40s-run3.md](benchmarks/l40s-run3.md) §7. Run 2's BF16 control at
c = 32 is $0.890 against run 1's $0.873 at the same point — 2 %, the
launch-to-launch spread every comparison in this table sits on.

Three readings, and the order is the order an operator needs them in:

1. **The promise has a price, and past the peak that price has the opposite sign
   to the raw figure.** Run 1: holding 50 ms TPOT costs **29 %** against the
   throughput optimum at the same geometry. Run 2: from 2.5 to 4.0 req/s the raw
   cost per output token *fell* 35 % while the cost per token that met the SLO
   *rose* 15×. An operator who reports the left column at 4 req/s reports $0.38
   for tokens nobody received. So the cost figure this repository reports is the
   SLO-respecting one; the other two columns are diagnostics.
2. **Two knobs, priced in dollars.** FP8 KV: **−17 %** per output token at equal
   seats (c = 32), a throughput gain — before its doubling of the pool, which is
   a capacity argument (§6), not a cost one. Prefix caching at `h` = 0.8:
   **−67 %** per SLO-respecting token, 3.0× on the seat count and therefore 3.0×
   on cost — the largest single lever measured on this card, and the one that
   belongs to the workload rather than the hardware: `h` is a property of the
   traffic (§6). At `h` = 0 the feature costs +0.027 % on the decode step, so it
   is on by default and the saving is whatever the traffic repeats.
3. **One user and a full card are 7.7× apart per token**, on the same hour of
   rent — the same 7.66 in either column; the run 1 write-up says why an earlier
   draft read 21×. This is the cost argument for
   §1's two classes and for two pools rather than one: the interactive class
   pays reading 1's 29 % at c = 13, and only a pool with no latency promise to
   keep reaches c = 32.

**Not yet priced:**

- **The MI300X row.** Both inputs — the rate and the throughput — arrive with
  the first run on that card; the formula does not change.
- **Headroom.** `minReplicaCount` above the mean load buys per-request TTFT (§4)
  and is paid in idle card-hours. Runs 1–3 measured one card, never a fleet, so
  this is a parameter of the scaler's configuration, not a figure.
- **Reasoning tokens.** A `$/1M` figure that does not separate reasoning from
  answering tokens prices the wrong thing (§8, §10).

---

## 8. Known gap: reasoning workloads

Everything above assumes non-thinking generation. The default model, Qwen3-8B,
ships a thinking mode, so this gap is live rather than hypothetical — the
thresholds in §2 were derived as if it did not exist.

**TTFT stops being a proxy for perceived latency.** The first token the engine
emits is the first *reasoning* token. Prefill is unchanged, the server reports a
healthy TTFT, and `vllm bench serve` agrees — while the user watches a spinner
for minutes. The metric does not break; its meaning does.

**TTFAT** (Time To First Answering Token) is the metric that survives, measuring
time to the first *visible* token. It is only measurable if the server separates
reasoning output from final output — vLLM does this with `--reasoning-parser`,
which supports Qwen3.

**The phase-to-metric mapping breaks.** §4 rests on prefill → TTFT and decode →
TPOT. With reasoning, decode feeds both: the reasoning phase lands in TTFAT and
the answering phase in TPOT. The clean one-to-one correspondence is gone.

**Concurrency collapses.** Reasoning tokens enter the KV cache like any others.
The §6 ceiling assumed 4 000 tokens of context per sequence; a reasoning request
can reach 32 000. At that length the same 1.15 M KV tokens support roughly
**35 concurrent sequences instead of 286** — an eightfold reduction, and the neat
coincidence between memory capacity and latency budget no longer holds.

**TPOT is not stationary within a request.** `bytes_moved` grows as the trace
grows, so each decode step is slower than the last. The derivations above
silently assume a fixed context length; over a long generation that assumption
decays continuously.

**Cost accounting has to split the token stream.** Reasoning tokens are paid for
and never seen. A `$/1M tokens` figure that does not separate reasoning from
answering tokens misstates the economics of every reasoning workload.

None of this is measured yet. The reasoning-mode scenario — the reasoning
parser enabled so TTFAT is observable, and a thinking-token budget tested
against the concurrency ceiling — is deliberately sequenced *after* the base
non-thinking predicted-vs-measured cycle closes: stretch scope, not part of
the first measurement pass. The open items in §10 hold the list.

---

## 9. Assumptions, and how they get validated

Two numbers here are empirical rather than derived, and they carry the error bars
of the whole document. **Both were measured on the L40S in run 1** (2026-08-18,
[benchmarks/l40s-baseline.md](benchmarks/l40s-baseline.md)) **and both were then
faced by run 2, which did not produce them** (2026-08-23,
[benchmarks/l40s-run2.md](benchmarks/l40s-run2.md)). That second step is what this
section requires before a fitted number counts, and it has now happened once.

### `achieved_bandwidth` — 0.70 assumed, 0.83 measured, 0.83 confirmed

Fitted against the **median ITL** of twelve benchmark levels, which is the
closest thing `vllm bench serve` reports to a pure decode step; the spread across
all twelve is ±0.02. The assumption was pessimistic by 19%, and since the
coefficient enters as a divisor the error passes straight through to every decode
floor and to `max_num_seqs`.

Two things about it that matter more than the value:

- **It belongs to a card and a phase.** 0.83 is L40S × decode × BF16 KV ×
  vLLM 0.27.1 × FlashAttention 2. The MI300X figures in this document keep 0.70
  and stay unvalidated; transplanting the measurement would produce a table that
  looks calibrated and is not.
- **The direction stated in earlier versions of this section was wrong.** It
  expected efficiency to be *worst* at batch 1, where launch and attention
  overhead amortise over nothing. Batch 1 measured 0.867, the best point in the
  sweep, and efficiency falls slowly to 0.80 at the capacity ceiling. A decode
  step at batch 1 is one long streaming read of contiguous weights; adding
  sequences adds scattered KV blocks, and it is the scattered half that costs.
  The one-scalar model in `bench/roofline.py` stands — a two-parameter fit to one
  run's twelve points would be over-fitting — but the residual now has a known
  shape and a known sign.

### `mfu` — 0.45 assumed, 0.439 measured on the L40S

Fitted against the single level with no prefill contention: one request, alone on
the card, 4 000-token prompt. Within 2.5% of the assumption, which is a
confirmation rather than a correction.

**Run 2 gave it a second, different confirmation, and a use.** The worst decode
step under chunked prefill should be the decode step plus the compute of the
prefill chunk sharing it — a quantity that needs `mfu`, at chunk sizes of 499 to
4 000 tokens, none of them the 4 000-token uncontended prefill it was fitted to.
Predicted 77.5 / 122.2 / 211.8 / 383.6 ms against measured ITL p99 of 70.6 / 114.2
/ 198.2 / 364.9. It is an upper bound at every chunk size, high by 5–10%, and the
excess barely drifts across an eightfold range. One point at one prompt length
turns out to price a chunk.

Unchanged: it is still one *utilisation* measurement, and still applied to the
compute side of decode where it does not belong.

Unchanged: `mfu` is still applied to the compute side of *decode*, where it does
not belong — a decode step is a matrix-vector product per sequence and cannot
reach prefill utilisation. The error is one-directional and harmless, flattering
the losing side of a roofline that memory wins by two orders of magnitude, but a
measured decode MFU would replace it rather than confirm it.

### What run 1 also corrected, which was not a coefficient at all

The concurrency ceiling in §6 subtracts only the weights from
`memory_bytes × gpu_memory_utilization`. Three further terms are paid before the
KV pool is carved — non-torch allocations, the peak activation, and the CUDA
graph pool — and on the L40S they cost 7% of the derived pool. `gpu_memory_utilization` is a
fraction of the *card*, not of the cache. The measured
breakdown is in the baseline write-up; the ceiling here is left as the clean
arithmetic it is, with the caveat that it is an upper bound and the startup log
is the number to act on.

### The rules for the table that scores all of this

**Every row must record which coefficient produced its prediction, whether that
coefficient was itself fitted to this measurement, and which accelerator the row
was taken on.** Without the first two columns the table silently compares a
prediction against the measurement it was fitted to, which always agrees and
proves nothing. Without the third, two sets of coefficients merge into one: both
are properties of a specific card, the first runs are taken on a cheap rental
rather than on the MI300X these derivations assume, and an `eff_mem` fitted there
says nothing about this document's baseline.

Run 1 was therefore reported against the *uncalibrated* 0.70 and 0.45, with the
recalibrated rows marked as fitted. **A revised coefficient may only be validated
by a later run it did not see** — which made the 0.83 rows of run 1 a hypothesis
for run 2 rather than a result.

**Run 2 was that run, and 0.83 came through it.** Every row of its
predicted-vs-measured table is unfitted. The load-bearing one: six BF16 launches
at 13 sequences — four chunk sizes, two attention backends, three separate starts
of one configuration, a different pod and a different driver — put the decode step
within 1.1% of each other, and their mean is 0.01% from the prediction. At 32
sequences the miss is 1.5%; under a Poisson arrival process it is under 2.2% up to
the goodput peak; with the KV dtype halved it is under 5% at three concurrencies.
The coefficient is now a measured property of L40S × decode × vLLM 0.27.1, no
longer a fit. The MI300X still has none.

Two limits on that statement, both from the same run. It degrades to a
one-directional 8% once the card saturates and the queueing model behind the
prediction stops holding. And it is only checkable where the **median ITL is still
a decode step**: at `max_num_batched_tokens` 512 or 1 024 with 32 sequences,
prefill occupies the majority of scheduler steps and the median moves into them,
reading 79.6 and 115.2 ms against a 49.8 ms step. The proxy this section calibrates
against has a validity condition, and run 2 found its edge.

**Run 3 faced them a third time, and added a limit of a different kind.** Its
uncached sweep at nine concurrencies reproduces the step model to 6% on the slope
and 3% on the intercept, and its c = 13 control lands 0.3% from the same level in
runs 1 and 2 — five measurements of one quantity across three pods, two drivers
and twelve days, total spread **0.37%**. What run 3 found is not a coefficient
that drifted but **a term the model does not contain**: with a shared prefix the
measured slope is 0.367 ms per seat where the model says 0.843, so `eff_mem` is
sound and `context_len` is not the context the hardware is paying for. A
coefficient validated at `h` = 0 may not be carried into `h` > 0 without saying so
(§6, channel 1).

A cheap early check, and the one that caught the 7%: vLLM logs the actual KV
cache size and available block count at startup. That log outranks any derivation
in this repository. **What it carries is build-dependent**: run 3's vLLM 0.27.1
logged neither `max_num_batched_tokens` nor `max_num_seqs`, and a gate that reads
a key the log stopped emitting does not fire and does not complain — which is why
`bench/vllm_metrics.unread_startup_facts` now reports the misses by name.

**But it is not repeatable to better than a few per cent, which run 2 learned the
expensive way.** Three launches of one configuration on one pod, differing only in
whether the `torch.compile` cache was warm, logged 169 833 / 176 994 / 169 833
tokens — a 4.2% spread, because the pool is what is left after a memory-profiling
pass whose peak moves. A reproducibility gate of ±500 tokens was written into the
run-2 card and fired on a difference that was noise. **Gate on `kv cache memory in
use`, or on ±5%, and never on a token count to three digits.** The same run found
median TTFT at fixed concurrency spanning 32% across three identical launches,
which disqualifies it from carrying any threshold decision; the decode step, over
the same three, held to 0.24%.

## 10. Open items

- [x] Validate floors against `vllm bench serve` — run 1, 2026-08-18, L40S only
- [x] Calibrate `eff_mem` and `mfu` from measurement, recording per row which
      coefficient was used and whether it was fitted to that same run (§9) —
      done for the L40S; **still open for the MI300X**
- [x] Re-face the calibrated 0.83 with a run that did not produce it (§9) —
      run 2, 2026-08-23; it holds to 0.01% at 13 sequences and under 5% wherever
      the median ITL is still a decode step
- [x] Measure TTFT against its target under a **finite** `--request-rate`; run 1
      was a closed loop and cannot answer it — run 2 §3. TTFT p99 crosses 300 ms
      between **0.5 and 1.0 req/s** at 1 500-token prompts, and goodput peaks at
      1.87 req/s and collapses by 3.0
- [x] Price the `max_num_batched_tokens` trade: the 2.6× between the decode step
      and TPOT as served is scheduling, not bandwidth — run 2 §4. The knob moves
      the seat count from 12 to **14** and costs 6× the TTFT p50 at 32 sequences.
      **It is not the fix**, which reopens the next item rather than closing it
- [x] Confirm FP8 KV doubles concurrency at constant TPOT — run 2 §5, 2.000× on
      the pool, 41 → 82 seats, and faster at equal concurrency besides
- [x] **Find what does close the decode-step-to-TPOT-p99 gap**, now that the
      chunk size is measured to be worth two seats of nineteen: prefix caching,
      a prompt-length cap, or separating the prefill and decode pools — answered
      by arithmetic in §6, 2026-08-26. All three attack prefill work per unit
      time; prefix caching is the one that removes it on a single card, and its
      effect is set by the cache hit rate, not by the hardware. **Derived, not
      measured** — the three items below are what a measurement of it costs
- [x] **Measure the seat count against a known cache hit rate** — run 3,
      2026-08-30. §6 predicted 24 ± 2 seats at `h` ≈ 0.8 against 13/14 uncached;
      measured **12.5 and 37.8** on TPOT p99, with `h` = 0.800 on every cached
      level and zero preemptions in all 45 levels. The three conditions run 2's
      noise imposed were all met — `h` read per level from counter increments,
      the pool gated at ±5 %, TTFT below the floor treated as an invalidation —
      and the harness invalidated three levels of its own accord when a shared
      seed made one sweep's prompts a prefix of the next's
      ([benchmarks/l40s-run3.md](benchmarks/l40s-run3.md) §3)
- [x] **Read whether cascade attention engages** — run 3, 2026-08-30, and the
      answer is neither yes nor no. The startup log says nothing about cascade on
      this build, so the detector carried it: the decode step **fell 14–32 % at
      matched concurrency**, where channel 1 predicts 0 % and cascade attention
      predicts twice the fall measured. §6's channel 1 is false and its
      replacement is unnamed
- [ ] **Identify the mechanism behind the decode-step fall at `h` > 0** — the item
      the one above turned into. The measured per-seat cost at `h` = 0.8 behaves
      as ~1 785 tokens of context where a seat's unique content is ~900, so the
      shared prefix is read less than once per sequence and more than once per
      batch. Two candidates, neither measured, both testable without a profiler:
      a `--disable-cascade-attn` A/B at c = 24 and c = 40, and one sweep at a
      prefix too large to sit in L2
      ([benchmarks/l40s-run3.md](benchmarks/l40s-run3.md) §4). Worth ~13 seats on
      this card; **until it is named, §6's seat table is a floor at high `h`, not
      an estimate**
- [x] **Measure the `h = 0` overhead** of prefix caching — run 3 §6, on one pod,
      one level, two servers differing in the flag alone: **+0.027 %** on the
      decode step and +0.8 % on TTFT p50 against a 32 % instrument spread. There
      is no latency reason to serve with the feature off. What the run could not
      measure is its cost in pool space: 4.3 % between the two logged pools,
      inside 4.2 % of launch-to-launch noise
- [ ] Explain the TTFT tail breaking at a third of the predicted arrival rate
      (run 2 §3) — the queueing model predicts the *step* well and the *tail*
      badly. Run 3 constrains it without explaining it: at `h` = 0.8 the tail is
      healthy to 7 req/s and TTFT p99 crosses 300 ms at 8.4, against 0.5–1.0
      uncached — so whatever breaks the tail early is prefill work, not queueing
- [ ] **Re-derive the third limit.** At `h` = 0.8 neither the latency limit nor
      the pool ends block B's curve: `max_num_seqs` = 256 does, with 65 requests
      queued behind it (run 3 §5). §6 derives the seat count as a `min()` of two
      limits and this is a third, entering exactly where prefix caching sends the
      other two away
- [ ] **Price prefix caching against a distribution of prefixes**, not one. Run 3
      shared a single 3 200-token prefix across every request, which measures the
      ceiling of what `h` is worth; a fleet holds several prefixes and evicts
      between them. The sheet that faces it is written and unreviewed:
      [benchmarks/runsheets/mi300x-run-3.md](benchmarks/runsheets/mi300x-run-3.md),
      2026-09-19, where the working set is the swept variable and the routing
      policy decides how much of it each replica has to hold
- [ ] Re-derive `max_num_seqs` for the interactive class at 1 500 tokens, the
      prompt length run 2 showed is the only one this card can serve interactively
- [x] Define the alert rule: which SLI, which window, which burn rate — §4
      "The alert rule", 2026-09-03. Good-share from the TTFT histogram at the
      0.25 s edge, burn rate 14.4 over 1 h and 5 m; the queue proxy it also
      derives **fired on `kind`**, the histogram rule waits for a card
- [x] Attach a cost figure once throughput is measured — §7, consolidated from
      runs 1–3 on 2026-09-03: **$0.735 → $0.39 per 1M SLO-respecting output
      tokens** on the L40S, the promise priced at 29 % and the two knobs in
      dollars. The figure is the SLO-respecting one; raw cost per output token
      moves the opposite way past the goodput peak. Consolidating it exposed one
      arithmetic error in run 1 §7 (21× was the token-count factor, the real
      ratio is 7.7×), now corrected in both places. **Still open for the
      MI300X**, and for headroom — a fleet cost this repository has not measured
- [ ] **`--max-model-len 9000` is undefended, and the origin of the number is
      unrecovered.** It is set in `deploy/manifests/base/deployment.yaml` and
      nothing in this repository derives it. Two things are computed *from* it:
      the edge's `proxy-read-timeout: 480`, which is 9 000 tokens × the 50 ms
      TPOT target = 450 s rounded up (`deploy/manifests/base/ingress.yaml`, and
      that one is measured on `kind`), and the terminationGracePeriod comment in
      the same deployment. So the constant is load-bearing in both directions and
      is the one number here that cannot say where it came from — in a repository
      whose entire claim is that its numbers can. Recorded rather than
      back-derived on 2026-09-11: a derivation invented now would read exactly
      like the ones that are real. Deriving it properly means choosing the
      context length the interactive class is promised, then checking it against
      the KV pool at the seat count §4 ships — separate work, and it moves
      `proxy-read-timeout` with it
- [ ] Measure TTFAT with `--reasoning-parser` and set a target for it (§8)
- [ ] Re-derive the concurrency ceiling for realistic reasoning context lengths
- [ ] Split cost accounting into reasoning and answering tokens
