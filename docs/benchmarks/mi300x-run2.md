# MI300X run 2 — 2026-09-27

The card configured for itself, one lever at a time: five server configurations
in one sweep, each against `b2048`, this run's replay of MI300X run 1's server
line. Every prediction it faced was written in
`docs/benchmarks/runsheets/mi300x-run-2.md` from `bench/predictions.py` table 12,
at run 1's fitted `eff_mem` 0.46 and `mfu` 0.166, before the droplet existed.

Raw evidence, verbatim: `docs/benchmarks/raw/mi300x-2026-09-27/run2/`. The
arithmetic behind every figure below: `python3 bench/measured_mi300x_run2.py`.
Where a number here disagrees with a derivation elsewhere in the repo, **this
file wins for this droplet and only for it** (`docs/SLO.md` §9).

**The run in five sentences**, the runsheet's definition of done:

- **Run 1's coefficients did not hold, and the reason is not in the
  configuration.** Run 1's server line, on a second droplet of the same product,
  ran the decode step 16–22 % faster and prefill 30–36 % faster, with the same
  image and a KV pool identical to the token. The predictions at 0.46 / 0.166
  were 13–34 % too long on the decode step and 25–58 % too long on TTFT. Refitted
  here: `eff_mem` 0.57 and `mfu` 0.247.
- **TPOT p99 did not cross 50 ms by c = 32 on any configuration** (40.8–47.5 ms),
  against a TPOT p50 crossing the model placed at c = 30. It crossed somewhere
  between 32 and 256, so the sheet's "within 8 seats" was not met.
- **The token-budget ceiling held at all three budgets**: running at c256 was
  99, 196 and 252 against 97.5, 195 and the cap of 256. The median ITL at c256 rose
  ×2.28 from `b2048` to `b4096`. Output did not stay flat: 800, 710 and 869 tok/s
  at 2 048, 4 096 and 8 192. TPOT p50 at c008–c032 stayed put at 4 096 and fell
  9–20 % at 8 192.
- **Chunked prefill under `ROCM_ATTN` was the excess in TTFT, and two levers
  remove it**: at 4 000 tokens, `b8192`, `b4096` and `aiter-fa` each cut TTFT by
  10 %, to `mfu` 0.274–0.275. `b8192`'s second doubling ratio fell to ×2.22, under the
  sheet's ×2.33. `triton` was 15 % slower.
- **Every configuration read at concurrency beat the L40S**: $0.631–0.779 per 1M
  output tokens at the c256 plateau against $0.873, and $0.706–0.778 at c032,
  still inside 50 ms, against the L40S's ≈ $1.14 at its crossing. On run 1's
  droplet the same server line cost $1.035 at its plateau. The verdict belongs to
  the droplet as much as to the card.

---

## 1. What was run

| | |
|---|---|
| Accelerator | 1 × AMD Instinct MI300X VF, gfx942, 205 822 885 888 B reported by `rocm-smi` — the same figure as run 1 |
| Provider | AMD Developer Cloud, ATL1, on-demand, **$1.990/h** on the console at creation, paid from credits |
| Host | *Quick Start → ROCm Software*, version 7.14 on Ubuntu 24.04, as run 1. Kernel `6.8.0-138-generic`, amdgpu `6.19.14`, Intel Xeon Platinum 8568Y+ with 20 vCPUs — recorded here and not in run 1 |
| Image | `vllm/vllm-openai-rocm:v0.27.1`, digest `sha256:bb44b39a…f0e7`, `vllm --version` 0.27.1+rocm723, `AITER_BRANCH: v0.1.19`, `import aiter` succeeded |
| Model | `Qwen/Qwen3-8B`, snapshot `b968826d`, `--dtype auto` → BF16 |
| Server line | run 1's: `--gpu-memory-utilization 0.90 --max-model-len 9000 --no-enable-prefix-caching --max-num-seqs 256`, and per serve row a budget and a backend |
| Serve rows | `b2048` (`ROCM_ATTN`), `b8192` (`ROCM_ATTN`), `triton` (2 048, `TRITON_ATTN`), `aiter-fa` (2 048, `ROCM_AITER_FA`), `b4096` (`ROCM_ATTN`), launched in that order, one server each |
| Instrument | `vllm bench sweep serve`, the runsheet's §3 line verbatim, one launch, no `--resume` |
| Load | 4 000 in / 200 out, `--random-range-ratio 0 --ignore-eos --request-rate inf`, `num_prompts = max(20, 4 × c)`; c = 1, 8, 16, 24, 32, 256, and c = 1 at 2 000 and 8 000 tokens |
| Repeats | 2 for all 40 cells, 80 in all |
| Failures | zero requests failed, zero preemptions, in all 80 repeats |
| Clock | droplet created ≈ 15:00 UTC, sweep 15:07–16:38, harvest done 16:41, then destroyed; ≈ 1.7 h, ≈ $3.4 against ≈ $5.1 expected. The invoice outranks this figure |

**`aiter-fa` is read at c = 1 only.** Its c008–c256 cells ran and are in the raw
capture, and are not measurements: on this build `ROCM_AITER_FA` misreads a batch
that mixes prompts, as the runsheet sets out under *The five serve rows*. They
are in no table below.

**Four departures from the sheet:**

1. **A host snapshot and the image digest were captured**
   (`host-under-load.txt`, `image-digest.txt`), not asked for. They were taken
   after `b2048` came in faster than run 1 (§3). Under load at c008 the card read
   1 579 MHz sclk, 742 W, 73 °C junction, "Performance Level: auto".
2. **`aiter-fa` compiled one kernel at launch**:
   `[aiter] finish build … pa_v1_…, cost 22.2 s`. The sheet expected no build.
   The line is not a gate, and the launch still took only 79 s.
3. **The sheet's rule for the kernel read was replaced.** It said a kernel was
   the cost if `triton` or `aiter-fa` moved TTFT at 4 000 by more than `b2048`
   moved from run 1's 256.9 ms. `b2048` moved 33 %, a gap between droplets and
   not noise, so the rule would pass no lever. The levers are read against
   `b2048` on this launch sequence instead. Decided after seeing the data.
4. **One registry instance, not one per lever.** The sheet gave a lever that
   moved `mfu` its own `Accelerator`. `MI300X_RUN2` carries `b2048`, the row
   comparable with run 1, and the levers' `mfu` stays in §4. Decided in the
   write-up.

---

## 2. Checkpoint A — five startup logs

| Serve row | Backend line | Budget | KV pool, tokens | vs `b2048` | Peak activation | CUDA graphs | Ready in |
|---|---|---|---|---|---|---|---|
| `b2048` | `Overriding with ROCM_ATTN` | 2 048 | **1 123 065** | — | 1.12 GiB | 4.12 GiB | 72 s |
| `b8192` | `Overriding with ROCM_ATTN` | 8 192 | 1 122 410 | −0.06 % | 1.21 GiB | 4.12 GiB | 53 s |
| `triton` | `TRITON_ATTN (selected via --attention-backend)` | 2 048 | 1 123 065 | 0 | 1.12 GiB | 4.05 GiB | 57 s |
| `aiter-fa` | `ROCM_AITER_FA (selected via --attention-backend)` | 2 048 | 1 123 065 | 0 | 1.12 GiB | 4.11 GiB | 79 s |
| `b4096` | `Overriding with ROCM_ATTN` | 4 096 | 1 122 777 | −0.03 % | 1.15 GiB | 4.12 GiB | 53 s |

*Ready in* runs from `non-default args` to `Starting vLLM server`.

- **`b2048`'s pool is run 1's launch 1 to the token.** 1 123 065 on both
  droplets, against table 12's 1 122 983 (192 GiB less run 1's 2.1 %). The gate
  was ±5 %; the miss is 0.01 %.
- **A larger budget costs activation, and little else.** 8 192 paid 0.09 GiB
  more than 2 048, which is 655 tokens of pool. `b8192`'s pool is 47 210 tokens
  above 256 × 4 200, so c256 could not preempt there, and did not.
- **Both forced rows passed.** Each logged its `selected via` line, then
  `Starting vLLM server` with no `Traceback` between; there is none anywhere in
  the log. `non-default args` carried the budget and, on the two forced rows,
  `attention_backend`, and nothing else new.

---

## 3. Run 1's coefficients, faced — and a second droplet

The read is run 1's: the median ITL of `b2048` against the floor at
`MI300X_RUN1`, context = input + 100.

| Row | Step @ 0.46 | Median ITL | Error | Implied `eff_mem` | Run 1's median ITL | Change |
|---|---|---|---|---|---|---|
| c001-in2000 | 6.85 ms | 5.14 ms | +33.4 % | 0.614 | 6.57 ms | −21.8 % |
| c001 | 6.97 ms | 5.22 ms | +33.7 % | 0.615 | 6.61 ms | −21.1 % |
| c001-in8000 | 7.22 ms | 5.40 ms | +33.8 % | 0.615 | 6.78 ms | −20.4 % |
| c008 | 8.71 ms | 7.64 ms | +14.0 % | 0.525 | 9.72 ms | −21.4 % |
| c016 | 10.69 ms | 9.07 ms | +17.9 % | 0.542 | — | — |
| c024 | 12.68 ms | 11.18 ms | +13.4 % | 0.522 | — | — |
| c032 | 14.66 ms | 12.46 ms | +17.6 % | 0.541 | 14.90 ms | −16.3 % |

| Row | TTFT @ 0.166 | TTFT p50 | Error | Implied `mfu` | Run 1's TTFT p50 |
|---|---|---|---|---|---|
| c001-in2000 | 128.1 ms | 80.9 ms | +58.3 % | 0.263 | 116.4 ms |
| c001 | 256.3 ms | 172.0 ms | +49.0 % | **0.247** | 256.9 ms |
| c001-in8000 | 512.5 ms | 410.8 ms | +24.8 % | 0.207 | 642.6 ms |

**Every prediction was too long.** These are the first MI300X rows to face a
coefficient not fitted to them, and they missed by more than the spread run 1's
own five rows showed (0.41–0.49).

**The same server line ran faster on this droplet.** Two repeats agree to 0.04 ms
on every decode row, so the gap is not noise, and nothing in the launch
differed. The image is the same, the host version is the same, the flags are the
same, and the startup log reports the same memory terms: 17.03 GiB consumed,
1.12 GiB activation, 4.12 GiB graphs, the same pool. What differs is the
droplet. Run 1 recorded no clocks, host CPU, kernel or driver, so this run
cannot say which of them it was.

**The change is in the part of the step that does not scale with KV.** At batch 1,
2 100 → 8 100 tokens of context adds 0.26 ms here and 0.21 ms on run 1, for the
same 0.885 GB. Extrapolated to zero KV the step is 5.05 ms here against 6.50 ms,
and the weights alone at peak bandwidth take 3.09 ms. The two-digit ITLs make the
increment coarse, but not coarse enough to hold a 1.45 ms difference. Run 1 §8
could not tell a slow weights read from a per-step cost, and this run cannot
either. It shows only that whichever it is, it is 22 % smaller on this droplet.

**Refitted on this run**: `eff_mem` **0.57**, the mean of seven rows at batch
1–32 (0.52–0.62), and `mfu` **0.247** at 4 000 tokens. They go into
`bench/roofline.py` as `MI300X_RUN2`, beside `MI300X_RUN1` and not over it.
**Two droplets, two fits**: which of them a third droplet resembles is a question
for the third droplet. Until then, any MI300X figure in this repository that
leans on either fit carries a ±20 % that neither run can narrow.

---

## 4. `mfu` — what moved it

One request at a time, three prompt lengths, per serve row:

| Serve row | 2 000 | 4 000 | 8 000 | × 2k→4k | × 4k→8k | `mfu` at 4 000 | vs `b2048` at 4 000 |
|---|---|---|---|---|---|---|---|
| `b2048` | 80.9 ms | 172.0 ms | 410.8 ms | 2.12× | **2.39×** | 0.247 | — |
| `b8192` | 80.6 ms | 155.2 ms | 344.6 ms | 1.92× | **2.22×** | 0.274 | −9.8 % |
| `b4096` | 82.2 ms | 154.9 ms | 386.7 ms | 1.88× | 2.50× | 0.275 | −9.9 % |
| `aiter-fa` | 76.6 ms | 154.5 ms | 327.5 ms | 2.02× | 2.12× | 0.275 | −10.2 % |
| `triton` | 89.0 ms | 197.9 ms | 520.4 ms | 2.22× | 2.63× | 0.215 | +15.1 % |

Attention FLOPs, which the floor leaves out, make the two doubling ratios 2.08×
and 2.16×. The sheet's line for chunking was 2.33×, the midpoint to run 1's 2.50×.

**Chunking made the excess.** `b8192`'s second ratio is 2.22×, under 2.33×, and
`b2048`'s is 2.39×, not under it: the sheet's test, passed. The other rows say
the same thing without being asked. At 4 096 a 4 000-token prompt is one step and
costs what it costs at 8 192 (154.9 against 155.2 ms). An 8 000-token prompt is
two steps there, and the 2.50× comes back. Every `ROCM_ATTN` prompt that is
chunked costs more than one that is not, and by more the more chunks it has:
410.8, 386.7 and 344.6 ms for four, two and one.

**AITER FA does not pay for chunks.** At a budget of 2 048 its prompts are
chunked as `b2048`'s are, and its TTFT is `b8192`'s: 154.5 ms at 4 000, 327.5 ms
at 8 000, doubling ratios 2.02× and 2.12×, at the attention-FLOP line. That
points to `ROCM_ATTN`'s prefill kernel over a cached prefix as the cost, rather
than chunking as such. Two rows point there; neither isolates it.

**Triton is slower at every length**, most at 8 000, and its decode step is
slower too: 5.48 ms at c001 against 5.22, and 16.6 against 11.2 ms at c024.

What none of the levers reaches is most of the gap to the prior: 0.275 against
0.45.

---

## 5. What the budget moved

At c256, from the engine's 10-second lines. Running is taken over lines within
10 % of the row's maximum, and the maximum is a floor on the peak. KV use is the
maximum line.

| Serve row | Budget | Ceiling | Running, median (range) | KV use | Median ITL | TTFT p50 | Output tok/s |
|---|---|---|---|---|---|---|---|
| `b2048` | 2 048 | 97.5 | **99** (98–100) | 36.4 % | 115.8 ms | 38.5 s | 800.1 |
| `b4096` | 4 096 | 195.0 | **196** (183–200) | 72.9 % | 263.6 ms | 17.9 s | 709.5 |
| `b8192` | 8 192 | 390.1 | **252** (239–256) | 94.3 % | 334.2 ms | 1.7 s | 869.2 |
| `triton` | 2 048 | 97.5 | 99 (98–101) | 36.6 % | 109.2 ms | 36.0 s | 875.4 |

**The ceiling held.** At 2 048 and 4 096 the running count sits one or two above
the arithmetic, as it did on run 1, for the requests partly prefilled. At 8 192 the
cap and the client's 256 tie, and 252 is all that says. Zero preemptions in all
80 repeats. At 8 192 the pool reached 94 %, the first time an MI300X
configuration here came near filling it.

**The median ITL at c256 roughly doubled**, ×2.28 from `b2048` to `b4096`, as the
sheet's falsifiable form said.

**Output did not stay flat.** The sheet said within 5 % of run 1's 531.7 tok/s.
Against run 1 every row is 33–65 % higher, which is the droplet (§3). Against
this run's own `b2048`, 4 096 lost 11 % and 8 192 gained 9 %. One candidate for
the loss is arithmetic, not measured: at 196 running a 4 096-token step leaves
3 900 tokens for prefill, less than one prompt. So nearly every prompt runs as
3 900 tokens and then 100, and §4 shows a chunked `ROCM_ATTN` prompt costs more.
This run does not separate that from anything else.

**TPOT p50 across budgets, which the model said would coincide:**

| Serve row | c008 | c016 | c024 | c032 |
|---|---|---|---|---|
| `b2048` | 12.82 ms | 21.98 ms | 29.49 ms | 37.37 ms |
| `b4096` | 11.86 ms | 21.53 ms | 29.71 ms | 39.84 ms |
| `b8192` | 10.87 ms | 17.70 ms | 25.92 ms | 33.97 ms |

At 4 096 it did coincide, to within −7…+7 %. At 8 192 it did not: 9–20 % lower.
The model carries no budget term, but it does carry TTFT alone, and that is what
8 192 changed (§6).

---

## 6. The 50 ms line, and interference

**TPOT p99** — the SLO metric:

| Serve row | c001 | c008 | c016 | c024 | c032 | c256 |
|---|---|---|---|---|---|---|
| `b2048` | 5.29 | 20.21 | 24.44 | 35.40 | **47.52** | 143.62 |
| `b8192` | 5.21 | 18.27 | 22.84 | 32.25 | 40.84 | 304.80 |
| `triton` | 5.49 | 19.84 | 22.49 | 35.09 | 43.26 | 110.69 |
| `b4096` | 5.21 | 19.54 | 24.37 | 32.58 | 42.85 | 284.85 |

Milliseconds. **Every row is inside 50 ms at c032.** The rows were spaced to
place a crossing run 1's droplet put between 8 and 32, and this droplet moved it
past 32. Where it falls between 32 and 256 is not measured.

**Table 12's TPOT p50 was too long by the same droplet.** It predicted 17.68,
29.91, 42.15 and 54.38 ms at c008–c032, crossing 50 ms at c = 30. `b2048`
measured 12.82, 21.98, 29.49 and 37.37 ms: errors of +38 %, +36 %, +43 %, +46 %.

**The model's form held when fed this run's own inputs.** Run 1's
`TPOT ≈ median ITL + (c − 1) × TTFT(c = 1) / 200`, with each serve row's own
median ITL and c001 TTFT:

| Serve row | TTFT c001 | c008 | c016 | c024 | c032 |
|---|---|---|---|---|---|
| `b2048` | 172.0 ms | +6.5 % | −0.1 % | +5.0 % | +4.7 % |
| `b4096` | 154.9 ms | +10.1 % | −4.0 % | −2.4 % | −8.4 % |
| `b8192` | 155.2 ms | +20.1 % | +16.7 % | +11.8 % | +7.4 % |
| `triton` | 197.9 ms | +18.9 % | +16.1 % | +16.2 % | +17.0 % |

Model over measured TPOT p50, as errors. At run 1's budget and kernel it is
within 7 %; at 8 192 and under Triton it runs 7–20 % long. It is still a model
written after run 1, and it was not a prediction this run faced as such: table
12 fed it floors, and those were the droplet's miss.

TPOT p50 over the median ITL at c032 was 3.00, 3.19, 2.74 and 2.28 in the order
above, against run 1's 3.75.

**TTFT p99 misses 300 ms at every loaded row, on every configuration**: 2.3–2.8 s
at c008, 5.9–7.5 s at c032, 161–348 ms at c = 1. In a closed loop at
`--request-rate inf` the first c prompts arrive together and queue behind each
other's prefill. This run does not separate that burst from the steady state.
The pricing below is at the TPOT
line, as run 1's was, and not at the interactive class as a whole.

---

## 7. Cost, at $1.99/h

| Serve row | c032 tok/s | $/1M at c032 | TPOT p99 at c032 | c256 tok/s | **$/1M at c256** |
|---|---|---|---|---|---|
| `b2048` | 750.4 | 0.737 | 47.52 ms | 800.1 | 0.691 |
| `b8192` | 783.3 | **0.706** | 40.84 ms | 869.2 | 0.636 |
| `triton` | 710.8 | 0.778 | 43.26 ms | 875.4 | **0.631** |
| `b4096` | 728.9 | 0.758 | 42.85 ms | 709.5 | 0.779 |

**At the plateau every row clears the L40S's $0.873**, and the break-even at
633 tok/s. The cheapest are `triton` and `b8192`, 28 % below the L40S. The two
reach it by opposite roads: `triton` holds 99 running, while `b8192` holds 252
with a 334 ms median ITL.

**Inside 50 ms TPOT p99 the MI300X is 32–38 % cheaper.** c032 is the last
measured row on every configuration, and the crossing lies past it, so c032's
figure bounds the cost at the crossing from above. `b8192` gives $0.706 against
the L40S's ≈ $1.144 at its interpolated crossing (c ≈ 12.6), a derived figure.

**The same card on run 1's droplet was 19 % dearer than the L40S**, at the
same server line and price ($1.035 at its plateau, c288, run 1 §7). Between the two
MI300X runs the configuration moved the plateau between −11 % and +9 % against
`b2048`. The droplet moved it by 50 %. For an operator the first lever on this card is
which droplet is under it. With two droplets this run can show a gap and give no
distribution.

---

## 8. What this run could not measure

**Why the droplets differ.** Run 1 recorded no clocks, CPU, kernel or driver,
and this run has no second droplet to set its own record against. What it
recorded is in `host-under-load.txt` for the next run to compare.

**Where TPOT p99 crosses 50 ms.** Past c032 on every configuration; the sweep
had no row between 32 and 256.

**`aiter-fa` at concurrency, and the levers together.** The first waits on the
batch-order mismatch (runsheet, *Not in this run*). AITER FA at a budget of 8 192
or 4 096 was never launched.

**Why 4 096 lost output.** §5 gives arithmetic that fits and no measurement that
separates it.

**What is left of the gap to `mfu` 0.45.** The best row here is 0.275. The GEMM
path, which no lever here touches on gfx942, is next in the sheet's list.

**The batch-1 step.** It is 22 % smaller than run 1's and still 1.6× the weights
read at peak.

---

## 9. Predicted vs measured — the summary table

Every row names the coefficient that produced its prediction, whether that
coefficient was fitted to this same run, and the card (`docs/SLO.md` §9). Rows
at `b2048` unless named. Rows the sheet wrote as a bound or a test are marked so
and carry no error.

| Quantity | Predicted | Measured | Error | Coefficient used | Fitted to this run? | Card |
|---|---|---|---|---|---|---|
| KV pool, tokens | ~1 123 000 | 1 123 065 | −0.01 % | 2.1 % shortfall, fitted on MI300X run 1 | no | MI300X |
| Decode step, c = 1, 2 000-token prompt | 6.85 ms | 5.14 ms | +33.4 % | `eff_mem` 0.46 | no | MI300X |
| Decode step, c = 1 | 6.97 ms | 5.22 ms | +33.7 % | `eff_mem` 0.46 | no | MI300X |
| Decode step, c = 1, 8 000-token prompt | 7.22 ms | 5.40 ms | +33.8 % | `eff_mem` 0.46 | no | MI300X |
| Decode step, c = 8 | 8.71 ms | 7.64 ms | +14.0 % | `eff_mem` 0.46 | no | MI300X |
| Decode step, c = 16 | 10.69 ms | 9.07 ms | +17.9 % | `eff_mem` 0.46 | no | MI300X |
| Decode step, c = 24 | 12.68 ms | 11.18 ms | +13.4 % | `eff_mem` 0.46 | no | MI300X |
| Decode step, c = 32 | 14.66 ms | 12.46 ms | +17.6 % | `eff_mem` 0.46 | no | MI300X |
| TTFT, c = 1, 2 000-token prompt | 128.1 ms | 80.9 ms | +58.3 % | `mfu` 0.166 | no | MI300X |
| TTFT, c = 1, 4 000-token prompt | 256.3 ms | 172.0 ms | +49.0 % | `mfu` 0.166 | no | MI300X |
| TTFT, c = 1, 8 000-token prompt | 512.5 ms | 410.8 ms | +24.8 % | `mfu` 0.166 | no | MI300X |
| TPOT p50, c = 8 | 17.68 ms | 12.82 ms | +37.9 % | `eff_mem` 0.46, `mfu` 0.166, run 1's interference model | no | MI300X |
| TPOT p50, c = 32 | 54.38 ms | 37.37 ms | +45.5 % | `eff_mem` 0.46, `mfu` 0.166, run 1's interference model | no | MI300X |
| TPOT p50 crosses 50 ms | at c = 30 | past 32 (37.37 ms at c = 32) | — | the same | no | MI300X |
| Running at c = 256, `b2048` | 98 | 99 (98–100) | −1.0 % | none (token-budget ceiling, 97.5) | no | MI300X |
| Running at c = 256, `b4096` | 195 | 196 (183–200) | −0.5 % | none (token-budget ceiling) | no | MI300X |
| Running at c = 256, `b8192` | 256, the cap and the client tied | 252 (239–256) | — | none | no | MI300X |
| Median ITL at c = 256, `b4096` over `b2048` | ~2× | 2.28× | — | none (a ratio) | no | MI300X |
| Output at c = 256, `b4096` | 531.7 ± 5 % | 709.5 tok/s | −25.1 % | run 1's plateau, measured on run 1's droplet | no | MI300X |
| TTFT, second doubling ratio, `b8192` | below 2.33× if chunking made run 1's 2.50×, a test | 2.22× | test passed | none (a ratio) | no | MI300X |
| TPOT p50, c = 8–32, `b8192` against `b2048` | unchanged | 9–20 % lower | — | run 1's interference model, no budget term | no | MI300X |
| TPOT p99 crossing, placed within 8 seats | between two rows | past 32 on every row | not placed | none | no | MI300X |
| Plateau, every row read at concurrency | above 633 tok/s to beat the L40S, a bound | 709.5–875.4 tok/s | bound held | L40S run 1's $0.873 | no | MI300X |
| Decode step, c = 1, refitted | 5.63 ms | 5.22 ms | +7.9 % | `eff_mem` 0.57 | **yes** | MI300X |
| Decode step, c = 32, refitted | 11.83 ms | 12.46 ms | −5.1 % | `eff_mem` 0.57 | **yes** | MI300X |
| TTFT, c = 1, 4 000-token prompt, refitted | 172.2 ms | 172.0 ms | +0.1 % | `mfu` 0.247 | **yes** | MI300X |

**The rows that held carried no droplet speed.** The pool and the two running
counts are memory and scheduler arithmetic, and they held to within 1 %; the
ITL ratio came in at 2.28 against ~2. Every row that predicted a time missed in one
direction, too long, by the gap between two droplets. The output row missed for
the same reason; read against this run's own `b2048` it would have missed by
+13 %, the other way.
The three refitted rows prove nothing on their own. They become evidence when a
later run faces 0.57 and 0.247 on a droplet that did not produce them.
