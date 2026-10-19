---
card: mi300x
run: 4
date: 2026-10-17
asked: Why was the pair unequal, and where does affinity routing beat round robin?
found: the engine launched first is the slow one, and with the instrument fixed the two routings part at 127 prefixes
predicted_with: mi300x-run2
---

# MI300X run 4 — 2026-10-17

Two engines of one model on one card, the prefix router in front of them, and
the instrument fixed since run 3. Every prediction the run faced was written in
`docs/benchmarks/runsheets/mi300x-run-4.md` before the droplet existed, and the
scripts that ran were committed beside it.

Raw evidence, verbatim: `docs/benchmarks/raw/mi300x-2026-10-17/run4/`. The
arithmetic behind every figure below: `python3 bench/measured_mi300x_run4.py`.
Where a number here disagrees with a derivation elsewhere in the repo, **this
file wins for this droplet and only for it** (`docs/SLO.md` §9).

**The run in five sentences.** The runsheet's definition of done was met:

- **The slow engine is the one launched first.** Alone, the two engines served
  12.43 and 12.34 ms per token against 12.46 predicted. Together, the one
  started first served 52.2 ms against 27.5. Relaunched in the other order, the
  slowness moved with it, 53.3 against 27.2. No CPU thread separates them, and
  the mechanism is below what the run recorded (§3).
- **Both fixes held.** Every level ran after a cache reset and two warmup
  passes. Round robin inside its room read 0.800 exactly at N = 31 and 63,
  where run 3 read 0.859 and 0.603.
- **With bounded loads off, affinity at N = 127 read 0.800**, the rotation
  model's figure, with every request on its prefix's owner. With the bound on,
  each miss was the first visit of a prefix moved to the other engine: 20 of
  20 at N = 31, 22 of 22 at N = 63 (§4).
- **The two arms parted at N = 127 and 255**, as the sheet expected, but neither
  level read the zero it predicted. Round robin at 127 read 0.467, decaying
  across the level, and affinity at 255 read 0.352. A replay through a model of
  vLLM's block pool reproduces ten of the eleven levels within 0.02, and
  places the rotation's cliff at 120 prefixes per engine, not 89 (§5).
- **The first launch on an empty compile cache logged a pool 2.9 % smaller**:
  the engine that compiled the graph had 1.98 GiB less for KV than the one that
  loaded it. Relaunched on the warm cache, both matched run 3 to the token (§2).

---

## 1. What was run

| | |
|---|---|
| Accelerator | 1 × AMD Instinct MI300X VF, gfx942, 205 822 885 888 B reported by `rocm-smi` |
| Provider | AMD Developer Cloud, on-demand, **$1.99/h** on the create page, paid from credits; about $90 of credit left at creation, read off *Billing* by the operator |
| Host | *Quick Start → ROCm Software* 7.14 on Ubuntu 24.04. Kernel `6.8.0-138-generic`, amdgpu `6.19.14`, Intel Xeon Platinum 8568Y+, 20 vCPUs: the same record as runs 2 and 3 |
| Image | `vllm/vllm-openai-rocm:v0.27.1`, digest `sha256:bb44b39a…f0e7`, `vllm --version` 0.27.1+rocm723 |
| Model | `Qwen/Qwen3-8B`, snapshot `b968826d`, `--dtype auto` → BF16 |
| Server line | `--gpu-memory-utilization 0.45 --max-model-len 9000 --max-num-batched-tokens 2048` on ports 8000 and 8001, prefix caching on by default, `VLLM_SERVER_DEV_MODE=1` |
| Router | `prefix-router:dev` built on the droplet, 128 vnodes, key 512 bytes, bounded load 1.25 or 1 (off); relaunched for every level |
| Instrument | `bench/harness.py` with `--reset-cache` on every call and `--warmup-passes 2` on every block-B level; `top -H`, `ps -L` and `rocm-smi` during block 0b |
| Load | 4 000 in / 200 out. Block 0: `fleet-bill-half`, 96 unique prompts at c = 32 per engine. Block B: 64 closed-loop seats, a 3 200-token prefix, N = 31 … 511 in strict rotation, `max(256, 4N)` requests |
| Failures | zero failed requests and zero preemptions in every level |
| Clock | droplet created ≈ 09:28 UTC, load 09:35–10:27, harvest 10:28–10:31, destroyed after it; ≈ 1.1 h, ≈ $2.2 against ≈ $2.6 expected. The invoice outranks this figure |

The droplet check read **5.04 ms** at c = 1 on both launches, as on run 3's
droplet, so every predicted time uses `FIT = mi300x-run2`.

---

## 2. Checkpoint A — six startup logs

| Launch | 8000 | 8001 | Graph |
|---|---|---|---|
| first, 8000 first | **492 074** tokens, 67.64 GiB | 506 525, 69.62 GiB | 8000 compiled it, 8001 loaded it |
| second, 8000 first | 506 525, 69.62 GiB | 506 525, 69.62 GiB | both loaded |
| 0c's relaunch, 8001 first | 506 525, 69.62 GiB | 506 525, 69.62 GiB | both loaded |

The first launch failed the sheet's equal-memory row. The row's remedy is a
relaunch, and the flags were not changed. The row names an early launch gate
as the cause, but that was not it: 8000 started on an empty card, and its log
is the only one of the six that compiled the `torch.compile` graph (11.1 s)
instead of loading it. **Compiling cost 1.98 GiB of KV memory, 14 451 tokens or
2.9 % of the pool**, inside the ±5 % gate on tokens and outside the
equal-memory row. Every later launch loaded the cached graph and logged run 3's
pool to the token. The dev-mode warning, `Route: /reset_prefix_cache` and an
idle `{"success":true}` were present on all six. Backend `ROCM_ATTN`, budget
2 048 and prefix caching were the same as in runs 1–3.

---

## 3. Block 0 — why the pair was unequal

| Step | 8000, median ITL | 8001, median ITL | Reading |
|---|---|---|---|
| 0a, each alone, the other idle | 12.43 ms | 12.34 ms | equal, +0.7 %; predicted 12.46 |
| 0b, together, 8000 launched first | **52.23 ms** | 27.51 ms | run 3's asymmetry: 48.9 / 26.8 |
| 0c, together, relaunched 8001 first | 27.16 ms | **53.32 ms** | the slowness moved with launch order |

**Alone, the pair is equal.** So the difference needs both engines busy: it is
contention, not a property of either process. **Together, the engine started
first is the slower one, 1.90× and 1.96× the other**, and when the order is
reversed the slowness moves to the other port. Port and process identity are
ruled out. The sheet's decision table sends this branch to block B on the
relaunched pair. Its slow engine is 8001.

The fast engine served 27.2–27.5 ms against ≈ 24.9 derived for a card
alternating fairly between two equal loads. Together the pair produced 856 tok/s,
against 758 for either engine alone at the same 32 seats: the second engine
added 13 %. The slow engine's call ran
49.4 s against 41.1, so its last 8 s were alone and its median is, if anything,
flattered.

**What the CPU record shows, and what it rules out.** Each EngineCore had two
threads busy in every 5-s interval of the level, 78–100 % of a core and mostly
above 98 %, on both engines and in both orders. Each of the four threads stayed on its own core for the whole
level: 17, 11, 7 and 12 in 0b, then 12, 11, 4 and 17 in 0c. No two busy threads
shared a core, and the fast engine's threads were as busy as the slow one's.
The sheet's CPU branch therefore did not apply, and `pin.sh` was not run.
`rocm-smi` read the device 100 % busy, at 546–752 W, in every sample while
both calls ran.
It cannot say which process held the card.

**Why the first-launched engine loses is not known.** Launch order reaches
nothing in the server line, and the two launches logged the same pool and
the same graphs. What the order does change is which process opened the device
first. How the card arbitrates two processes' work is the remaining candidate,
and nothing this run recorded can see it.

---

## 4. Block B — both arms clean

| Level | `h` | 8000 / 8001 | Requests 8000 / 8001 | Rotation | Hits / 3 200 | Prefixes moved | TTFT p50 |
|---|---|---|---|---|---|---|---|
| prefix, N = 31 | **0.738** | 0.682 / 0.800 | 136 / 120 | 0.800 | 236 | 20 | 529 ms |
| round_robin, N = 31 | **0.800** | 0.800 / 0.800 | 128 / 128 | 0.800 | 256 | — | 352 ms |
| round_robin, N = 63 | **0.800** | 0.800 / 0.800 | 128 / 128 | 0.800 | 256 | — | 333 ms |
| prefix, N = 63 | **0.731** | 0.682 / 0.800 | 149 / 107 | 0.800 | 234 | 22 | 637 ms |
| prefix, N = 127 | **0.713** | 0.652 / 0.800 | 297 / 211 | 0.800 | 453 | 54 | 430 ms |
| prefix, N = 127, bound off | **0.800** | 0.800 / 0.800 | 224 / 284 | 0.800 | 508 | 0 | 257 ms |
| round_robin, N = 127 | **0.467** | 0.461 / 0.472 | 254 / 254 | 0 | 296.26 | — | 380 ms |
| prefix, N = 255 | **0.352** | 0.029 / 0.734 | 553 / 467 | 0 | 448.49 | 39 | 460 ms |
| round_robin, N = 255 | **0.026** | 0.022 / 0.029 | 510 / 510 | 0 | 32.70 | — | 449 ms |
| prefix, N = 511 | **0.000** | 0.000 / 0.000 | 1 174 / 870 | 0 | 0 | 136 | 592 ms |
| round_robin, N = 511 | **0.000** | 0.000 / 0.000 | 1 021 / 1 023 | 0 | 0 | — | 429 ms |

Every level recorded `harness_cache_reset` 1 and `harness_warmup_passes` 2,
and the stop rule did not fire. **Round robin inside its room hit every
prefix on both engines**: 256 of 256 at N = 31 and 63, where run 3's defects
read 0.859 and 0.603.

**The bound costs affinity one miss per prefix it moves, not one per request.**
Bounded loads moved 47 requests at N = 31, all of them from the slow 8001 to
8000. They moved 20 distinct prefixes, and the level missed 20 times. At N =
63 they moved 24 requests and 22 prefixes, and the level missed 22 times. A
prefix moved once misses on the other engine, and every later move of it hits
the copy it left there. At N = 127 there are 55 misses against 54 moved
prefixes. With the bound off, every request went to its owner, and `h` read
0.800 with the split at the ring's share, 71 of 127 prefixes owned by 8001.
As on run 3, the bound moved load off the slow engine, and here the slow
engine is the first-launched 8001 (§3).

**Block C, TTFT arm against arm.** The sheet's bound on the difference, 137.8–
205.0 ms at N = 127, assumed `h_prefix − h_rr` = 0.8. The measured difference
in `h` was 0.247, and round robin's TTFT p50 was 50 ms *lower* than
affinity's. The column is printed and not scored. In a closed loop the harness
flags TTFT as no service metric, and on an unequal pair the arms differ in how
much of the load lands on the slow engine. Affinity with the bound off had the
lowest TTFT at N = 127 by 123 ms, with the highest `h` of the three.

---

## 5. What the rotation model got wrong at the edge

Two levels did not read the zero the sheet predicted: round robin at N = 127,
0.467, and affinity at N = 255, 0.352. Neither level passes the whole-prefix
test: hits over 3 200 are 296.26 and 448.49. A model of the block pool, read
at `v0.27.1`, was replayed with each level's requests, in their real order
and on their real engine. In the model, a finished request's blocks join the
free queue tail-first (`single_type_kv_cache_manager.free`), so its output
and body leave before its prefix, and a prefix's tail before its head. A hit
takes its block out of the queue (`block_pool.touch`), and allocation evicts
from the head. The replay matched ten of the eleven levels within 0.02 in
total and 0.025 per engine:

| Level | `h` run | `h` replay | 8000 run / replay | 8001 run / replay |
|---|---|---|---|---|
| prefix, N = 31 / 63 | 0.738 / 0.731 | 0.738 / 0.731 | 0.682 / 0.682 | 0.800 / 0.800 |
| prefix, N = 127 | 0.713 | 0.713 | 0.652 / 0.668 | 0.800 / 0.777 |
| prefix, N = 127, bound off | 0.800 | 0.800 | 0.800 / 0.800 | 0.800 / 0.800 |
| **round_robin, N = 127** | **0.467** | **0.337** | 0.461 / 0.328 | 0.472 / 0.347 |
| prefix, N = 255 | 0.352 | 0.339 | 0.029 / 0.006 | 0.734 / 0.734 |
| round_robin, N = 255 | 0.026 | 0.021 | 0.022 / 0.021 | 0.029 / 0.021 |

Round robin at 31 and 63 and both arms at 511 match exactly.

**The cliff is at 120 prefixes per engine, not 89.** Table 11's room is the
whole requests a pool keeps *beside* 32 live seats. Under strict rotation,
though, the live seats are the newest members of the window, so the window
that must fit is the room plus the seats: the pool over one request's
4 208 tokens, **120.4**. A synthetic rotation through the same model holds
0.800 for every cycle up to W = 120 prefixes per engine. At W = 121 it falls
to 0.740, at W = 122 to zero after the first cycle, and W = 127 is zero from
the start. For an even split, no zero in table 11 moves: round robin's 127 per
engine and affinity's 128 at N = 255 are both past 120. But both deciding
levels sit within eight prefixes of the edge.

**Affinity at N = 255 is a split, not an average.** The ring gives 8001 121
of the 255 prefixes, and bounded loads moved 34 of its requests away. Its reuse
window was 117 requests at the median, inside 120: 0.734, a steady state over
all four cycles. 8000 owned 134 and received the moved ones. Its window was
138, and it read 0.029. The rotation model split N evenly between the engines.
The ring does not.

**Round robin at N = 127 was a transient.** Its window is 127, past the cliff,
so its steady state is zero. Two sequential warmup passes, with no live seats,
left each engine holding nearly all 127 prefixes, and the measured requests
used that up. Per cycle of 127 requests the replay reads 0.80, 0.24, 0.31,
0.00. The run's TTFT p50 rose with it, 194, 380 and 516 ms over cycles 2–4,
the first cycle carrying the closed loop's start. The level's 0.467 is a
residue whose size depends on the level's length. The replay is 0.13 low on
that one level, and a cliff this steep amplifies any error in footprint or
timing: in the synthetic rotation, one prefix per engine is the difference
between 0.800 and 0.740.

---

## 6. What changes, and where it goes

| Finding | Goes to |
|---|---|
| The rotation's window is the room plus the live seats, a cliff at 120 per engine on this pool | `bench/predictions.py` table 11 and `retained_room`, a separate change; no predicted zero moves for an even split |
| Affinity's per-engine set is the ring's split, not N / 2 | the same change: `ring_owner` is already ported in `bench/measured_mi300x_run3.py` |
| Under bounded loads, affinity misses once per moved prefix | `router/README.md` §4, beside the bound's description |
| On a card shared by two engines, the first launched is the slower one under load | an open question; the next pair should be one engine per card, or a run that records per-process GPU time |
| A first launch on an empty compile cache logs a smaller pool | the runsheet template's checkpoint A: warm the cache, or expect the first engine low |

---

## 7. Cost, at $1.99/h

≈ 1.1 h of droplet, ≈ $2.2. The load itself took 52 minutes of it. With runs
1–3, the credits have paid ≈ $10.8 of MI300X time.

---

## 8. What this run could not measure

**The mechanism behind launch order** (§3). The run separated launch order
from port, process identity and CPU. What it could not see is how the card
shares itself between two processes.

**Round robin's steady state at N = 127.** The level was four cycles long, and
the replay says it reaches zero by the fourth. A longer level would read the
steady state, not the warmup's residue.

**The fleet's decode-step bill**, and **run 3's 0.04 at N = 31**: the sheet's
§7, unchanged.

---

## 9. Predicted vs measured — the summary table

Rows the sheet wrote as a test or a bound are marked so and carry no error.
Error is predicted over measured, less one.

| Quantity | Predicted | Measured | Error | Coefficient used | Fitted to this run? | Card |
|---|---|---|---|---|---|---|
| KV pool, each engine at 0.45, tokens | 506 525 (run 3) | 506 525 / 506 525 | 0.00 % | none, a logged figure | no | MI300X |
| KV pool, first launch on an empty compile cache | 506 525 (run 3) | 492 074 (8000) | +2.94 % | none, a logged figure | no | MI300X |
| Available KV memory, 8000 against 8001 | equal, a test | 67.64 / 69.62 GiB first launch; 69.62 / 69.62 after | failed, then passed | none | no | MI300X |
| Droplet check, median ITL at c = 1 | 5.04 ms (run 3) | 5.04 ms | 0.00 % | none, a measured figure | no | MI300X |
| Median ITL, one engine alone at c = 32 | 12.46 ms | 12.43 ms (8000) / 12.34 ms (8001) | +0.21 % | none, run 2's measured c032 | no | MI300X |
| 8000 alone against 8001 alone | within 5 %, a test | +0.7 % | test passed | none | no | MI300X |
| Median ITL, each engine, both at c = 32 | ≈ 24.9 ms | 27.5 ms (fast) / 52.2 ms (first launched) | −9.48 % / missed | none, twice the alone step | no | MI300X |
| `h`, round_robin, N = 31 | 0.800 | 0.800 | 0.00 % | none (the construction) | no | MI300X |
| `h`, round_robin, N = 63 | 0.800 | 0.800, 256 of 256 | 0.00 % | 89 retained, rotation | no | MI300X |
| `h`, prefix, N = 31 | 0.800 | 0.738 | +8.47 % | none (the construction) | no | MI300X |
| `h`, prefix, N = 63 | 0.800 | 0.731 | +9.40 % | 89 retained, rotation | no | MI300X |
| `h`, prefix, N = 127 | 0.800 | 0.713 | +12.14 % | 89 retained, rotation | no | MI300X |
| `h`, prefix, N = 127, bounded loads off | 0.800 | 0.800 (bound off) | 0.00 % | 89 retained, rotation | no | MI300X |
| `h`, round_robin, N = 127 | 0 | 0.467 | missed: a transient (§5) | 89 retained, rotation | no | MI300X |
| `h`, prefix, N = 255 | 0 | 0.352 | missed: the split (§5) | 89 retained, rotation | no | MI300X |
| `h`, round_robin, N = 255 | 0 | 0.026 | — (a zero) | 89 retained, rotation | no | MI300X |
| `h`, both arms, N = 511 | 0 | 0.000 / 0.000 | held | 89 retained, rotation | no | MI300X |
| Stop rule: `cache_reset` 1, `warmup_passes` 2, no `h` over 0.81, round robin ≥ 0.78 inside its room | a test, every level | eleven of eleven | test passed | none | no | MI300X |
| TTFT p50, round_robin minus prefix, N = 127 | ≤ 137.8–205.0 ms, a bound | −50 ms | not scored: closed loop, unequal pair | `mfu` 0.247 / 0.166 | no | MI300X |

**The memory rows held once the compile cache was warm. The hit-rate rows held
wherever the window fit, and missed at the two levels that sit at its edge.**
The misses came from the model's room and its even split, not from the
instrument. The time rows are the pair's: equal alone, unequal together, and
the inequality follows launch order.
