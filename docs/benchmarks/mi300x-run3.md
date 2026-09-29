# MI300X run 3 — 2026-09-29

Two engines of one model on one card, and the prefix router in front of them.
Every prediction the run faced was written in
`docs/benchmarks/runsheets/mi300x-run-3.md` from `bench/predictions.py`
table 11, before the droplet existed.

Raw evidence, verbatim: `docs/benchmarks/raw/mi300x-2026-09-29/run3/`. The
arithmetic behind every figure below: `python3 bench/measured_mi300x_run3.py`.
Where a number here disagrees with a derivation elsewhere in the repo, **this
file wins for this droplet and only for it** (`docs/SLO.md` §9).

**The run in five sentences.** The runsheet's definition of done was not met,
and the fourth sentence says why:

- **The fleet's memory bill held.** The solo pool logged 1 119 644 tokens,
  each engine of the pair 506 525, and the second copy of the weights cost
  106 594 tokens against 111 220 derived. The serial launch gate worked: the
  two engines logged the same available memory to the hundredth of a GiB.
- **At `h` = 0 the pair out-produced the single engine**: 862 tok/s against 768
  at 64 seats. The decode-step row of block A was not faced: at 4 000 unique
  tokens the median ITL is 83 ms against a 17.9 ms floor, which is interference,
  not a decode step.
- **The rotation model held where it could be read.** Clean levels gave 0.784
  at N = 31 under affinity and 0.024 and 0.000 past round robin's cliff, against
  0.800, 0 and 0. The uniform model's 0.278 at N = 255 did not.
- **Block B's arm comparison is void at every N, for two instrument defects.**
  Both arms at one N sent identical requests, so the second arm hit the first
  arm's bodies (§5.1). And one warmup pass under round robin left the first
  measured pass cold (§5.2). Both are fixed in `bench/`, with tests, and block B
  goes to MI300X run 4.
- **The two engines were not equal.** Engine 8000 served each token 2.8–3.7×
  slower than 8001 at an equal share of the load, and slower in both arms. Its requests stayed in
  flight longer and bounded loads moved them to 8001, so the 60 % split warning
  fired, and it was a symptom of the pair, not of the ring (§6).

---

## 1. What was run

| | |
|---|---|
| Accelerator | 1 × AMD Instinct MI300X VF, gfx942, 205 822 885 888 B reported by `rocm-smi` |
| Provider | AMD Developer Cloud, ATL1, on-demand, **$1.990/h** on the console at creation, paid from credits |
| Host | *Quick Start → ROCm Software* 7.14 on Ubuntu 24.04. Kernel `6.8.0-138-generic`, amdgpu `6.19.14`, Intel Xeon Platinum 8568Y+, 20 vCPUs: the same record as run 2's droplet |
| Image | `vllm/vllm-openai-rocm:v0.27.1`, digest `sha256:bb44b39a…f0e7`, `vllm --version` 0.27.1+rocm723 |
| Model | `Qwen/Qwen3-8B`, snapshot `b968826d`, `--dtype auto` → BF16 |
| Server line | `--max-model-len 9000 --max-num-batched-tokens 2048`, prefix caching on by default; `--gpu-memory-utilization` 0.90 solo, 0.45 each for the pair on ports 8000 and 8001 |
| Router | `prefix-router:dev` built on the droplet, `-policy prefix` or `round_robin`, 128 vnodes, key 512 bytes, bounded load 1.25; relaunched for every level |
| Instrument | `bench/harness.py`, the runsheet's lines, one call per level in block B |
| Load | 4 000 in / 200 out. Block A: unique prompts at c = 1, 64 solo and 2 × 32. Block B: 64 closed-loop seats, a 3 200-token prefix, N = 31 … 511 in strict rotation, `max(256, 4N)` requests |
| Failures | zero failed requests and zero preemptions in every level |
| Clock | droplet created ≈ 08:57 UTC, load 09:06–09:49, harvest 09:50, destroyed ≈ 09:53; ≈ 0.93 h, ≈ $1.9 against ≈ $4 expected. The invoice outranks this figure |

The droplet check read a median ITL of **5.04 ms** at c = 1, against 6.61 ms on
run 1's droplet and 5.22 ms on run 2's, so every predicted time below uses
`FIT = mi300x-run2`.

---

## 2. Checkpoint A — three startup logs

| Log line | Predicted | Measured |
|---|---|---|
| `GPU KV cache size`, solo at 0.90 | 1 123 065, logged by runs 1–2 | **1 119 644** (−0.3 %) |
| `GPU KV cache size`, each engine at 0.45 | ~507 050 corrected | **506 525** and **506 525** (−0.1 %) |
| `Available KV cache memory`, 8000 against 8001 | equal | 69.62 and 69.62 GiB |
| The pair's pools summed, against the solo pool | −111 220 derived | **−106 594** |
| `Graph capturing finished … took` | read, not predicted | 4.24 GiB in each of the three launches |
| backend, prefix caching, budget | `ROCM_ATTN`, True, 2 048 | the same in all three |

The second engine was started only once the first answered `/health`, and
the two logs agree to the hundredth, so the early-gate failure the runsheet's
third pass found (§0) did not happen. Checkpoint A also caught a defect in the
instrument. The harness printed `max_num_batched_tokens` and `attention_backend`
as NOT FOUND on this log, because its patterns matched CUDA's wording and a
harvested summary line, never a raw ROCm log. Both are fixed in
`bench/vllm_metrics.py` and tested against `engine-8000.log`. `max_num_seqs`
is not printed at its default, as before.

---

## 3. Block A — the fleet's bill at `h` = 0

| | TPOT p50 | TPOT p99 | median ITL | tok/s |
|---|---|---|---|---|
| solo, c = 64 | 75.30 ms | 86.96 ms | 83.43 ms | 767.9 |
| engine 8000, c = 32 | 74.38 ms | 114.49 ms | 48.87 ms | 391.7 |
| engine 8001, c = 32 | 58.19 ms | 81.84 ms | 26.77 ms | 470.1 |

**The pair produced 862 tok/s against the solo engine's 768**, 12 % more at
the same 64 seats. The sheet predicted the bill as a decode step, 17.92 →
23.35 ms at `eff_mem` 0.57, and that row was not faced. With every request
prefilling 4 000 unique tokens through a 2 048-token budget, the solo median ITL
is 83 ms, 4.7× the floor. A step that carries a prefill chunk is not a decode
step, and the fit is not read from it (`docs/adding-a-run.md` §3). What the row
does show is consistent with each engine owning its own budget: two engines move
twice the prefill tokens per step. It does not separate that from anything else.
The two calls overlapped for 41 of 49 s (the raw README).

---

## 4. Block B — what the clean levels measured

At each N the arm that ran first met a cache no identical request had touched,
and those five levels are the run's block B. The second arm at each N is not
read (§5.1).

| N | Arm, first at its N | `h` | Uniform | Rotation | Whole-prefix hits |
|---|---|---|---|---|---|
| 31 | prefix | **0.784** | 0.800 | 0.800 | 251 of 256 |
| 63 | round_robin | 0.603, defect 2 | 0.800 | 0.800 | 193 of 256 |
| 127 | prefix | **0.748** | 0.800 | 0.800 | 475 of 508 |
| 255 | round_robin | **0.024** | 0.278 | 0 | 30.3, fractional |
| 511 | prefix | **0.000** | 0.278 | 0 | 0 |

**Every request of a clean level hit its whole prefix or nothing** until
eviction began: the hit counter divided by 3 200 is an integer at N = 31, 63
and 127. At N = 31 the misses number exactly the five requests bounded
loads moved off their prefix's owner (§6). At N = 127 there are 33 misses
against 56 moved requests, so the gap to 0.800 needs no eviction on the owner.
Past round robin's cliff the rotation model's zero held and the uniform
model's share did not. Round robin at N = 127 and affinity at N = 255 are the
two levels that would have separated the arms, and both ran second.

---

## 5. Two defects in the instrument

### 5.1 The second arm hit the first arm's bodies

A level's seed is `SEED + N` and not the arm's, so both arms at one N send
byte-identical requests. And vLLM keeps a finished request's body and output
hashed beside its prefix (runsheet §0, second pass). A second-arm request that
lands on the engine the first arm sent it to therefore finds itself whole in
the cache: up to 3 984 tokens hit instead of 3 200.

| N | Second arm | Same engine as in the first | TTFT p50, same / other | `h` | `h` if every same-engine request hit whole |
|---|---|---|---|---|---|
| 31 | round_robin | 50.8 % | **151 / 308 ms** | 0.859 | 0.900 |
| 63 | prefix | 51.6 % | 430 / 468 ms | **0.901** | **0.901** |

The last column holds only while nothing is evicted. At N = 63 it matches to
the third decimal. At N = 31 it overstates by 0.04, which this run does not
account for. In every second-arm level the hit counter over 3 200 is
fractional, against the integers of §4. The fix is `--reset-cache`, a `POST
/reset_prefix_cache` to every engine before each level's warmup. It needs
`VLLM_SERVER_DEV_MODE=1`, and the harness raises on refusal
(`bench/tests/test_harness.py`).

### 5.2 One warmup pass under round robin warmed the wrong engine

The warmup sends one request per prefix through the router. Under round robin
with N odd, those N requests flip the cursor's parity, so every request of the
first measured pass meets its prefix on the engine that did not warm it. At
N = 63, where round robin ran first: `h` **0.6031**, against 0.8 × 193 / 256 =
**0.6031** if the first pass is wholly cold, and a first-pass TTFT p50 of
4 604 ms against 260 ms after it. The fix is `--warmup-passes 2`: the second
pass lands each prefix on the other engine. It changes nothing under affinity.

---

## 6. The pair was not symmetric

| N | TPOT p50 rr, 8000 / 8001 | TPOT p50 prefix, 8000 / 8001 | Prefixes 8001 owns | On the owner | Moved to 8001 / 8000 |
|---|---|---|---|---|---|
| 31 | 55.1 / 17.7 ms | 58.5 / 25.7 ms | 65 % | 98.0 % | 4 / 1 |
| 63 | 77.8 / 27.6 ms | 54.1 / 19.9 ms | 51 % | 84.8 % | 36 / 3 |
| 127 | 85.5 / 30.0 ms | 63.9 / 27.7 ms | 56 % | 89.0 % | 49 / 7 |
| 255 | 124.9 / 33.6 ms | 76.7 / 50.5 ms | 47 % | 94.0 % | 61 / 0 |
| 511 | 122.8 / 35.2 ms | 112.8 / 56.3 ms | 53 % | 94.3 % | 114 / 2 |

**Under round robin each engine takes exactly half of every level, and 8000
is still 2.8–3.7× slower per token.** Block A shows the same, 48.9 against
26.8 ms median ITL at 32 seats each. The two launches had the same flags, the
same pool and the same graphs, and nothing in the logs tells them apart.

**The ring is not the cause.** The owners are computed offline by a port of
`router/key.go` and `ring.go` in the read-out, checked once against the Go
code on all 987 prefixes (2026-09-29). The ring gives 8001 51.3 % of the key space, and the ownership
column is that share sampled N times. What pushes the split past 60 % is the
moved requests, and nearly all of them move one way. A slower engine holds its
requests longer, its in-flight count reaches 1.25 × the fleet mean first, and
the bound sends the excess to 8001. The router did what `router/README.md` §4
says it does. What bounded loads cost in `h` on an asymmetric pair is the
N = 31 row of §4: as many misses as moved requests.

**Why 8000 is slower is not known.** The candidates are launch order, CPU
contention between the two engine processes on 20 vCPUs, and how the card
schedules two processes' queues. The run recorded nothing that separates them,
so it goes to MI300X run 4's block 0.

---

## 7. Cost, at $1.99/h

≈ 0.93 h of droplet, ≈ $1.9. The load itself took 43 minutes of it. With runs 1
and 2 the credits have paid ≈ $8.6 of MI300X time.

---

## 8. What this run could not measure

**The arm comparison.** Void at every N (§5). The seat and TTFT arithmetic of
the runsheet's §5 has nothing to read.

**The fleet's decode-step bill.** Not faced, because at `h` = 0 with unique
prompts the median ITL is not a decode step (§3). A level with the prefill
removed would face it: a shared prefix at `h` near 1, or output-dominated
requests.

**Why the pair is asymmetric** (§6), and so whether a symmetric pair would have
tripped the split warning at all.

**The 0.04 at N = 31** in §5.1's second row.

---

## 9. Predicted vs measured — the summary table

Rows the sheet wrote as a test or a bound are marked so and carry no error.
Error is predicted over measured, less one.

| Quantity | Predicted | Measured | Error | Coefficient used | Fitted to this run? | Card |
|---|---|---|---|---|---|---|
| KV pool, solo at 0.90, tokens | 1 123 065 (logged, runs 1–2) | 1 119 644 | +0.31 % | none, a logged figure | no | MI300X |
| KV pool, each engine at 0.45, tokens | ~507 050 | 506 525 | +0.10 % | 2.1 % shortfall, fitted on MI300X run 1 | no | MI300X |
| Available KV memory, 8000 against 8001 | equal, a test | 69.62 / 69.62 GiB | test passed | none | no | MI300X |
| The second copy of the weights, tokens | 111 220 | 106 594 | +4.34 % | none (memory arithmetic) | no | MI300X |
| Decode step at 64 seats, one engine → two | 17.92 → 23.35 ms | not faced | — | `eff_mem` 0.57 | no | MI300X |
| `h`, prefix, N = 31 | 0.800 | 0.784 | +1.99 % | none (the construction) | no | MI300X |
| `h`, prefix at N = 31 ≈ 0.8, the stop condition | within 0.05, a test | 0.784 | test passed | none | no | MI300X |
| `h`, round_robin, N = 63 | 0.800 | 0.603 | not scored: defect 2 | none | no | MI300X |
| `h`, prefix, N = 127 | 0.800 (both models) | 0.748 | +6.95 % | 89 retained, room in seats | no | MI300X |
| `h`, round_robin, N = 255, rotation | 0 | 0.024 | — (a zero) | 89 retained, rotation | no | MI300X |
| `h`, round_robin, N = 255, uniform | 0.278 | 0.024 | missed | 89 retained, uniform | no | MI300X |
| `h`, prefix, N = 511, rotation | 0 | 0.000 | held | 89 retained, rotation | no | MI300X |
| Second-arm levels, all five N | table 11 | 0.000–0.901 | not scored: defect 1 | — | no | MI300X |
| One engine takes ≤ 60 % of a level, a bound | ≤ 60 % | 64–66 % at N = 31, 63, 127 under affinity | bound broken; cause §6 | none | no | MI300X |

**The memory rows held, and the time row was not reached.** The pool, the
per-engine pool and the weights' bill are arithmetic in bytes, and they held to
within 5 %. The hit-rate rows held where the level was clean and the model
could be read. The one the sheet called decisive, the arms at N = 127 and 255,
was lost to the instrument, not to the card.
