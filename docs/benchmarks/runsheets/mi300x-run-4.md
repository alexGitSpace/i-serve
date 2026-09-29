# Runsheet — MI300X run 4: the router arm, measured cleanly, and an unequal pair

MI300X run 3 asked what sending a request to the replica that already holds its
prefix buys, and did not get an answer. Two defects in the instrument voided
the arm comparison at every working set, and the pair of engines it routed
over turned out not to be a pair of equals (`docs/benchmarks/mi300x-run3.md`
§5–§6). **This sheet asks run 3's question again with the instrument fixed.
First it asks the one run 3 left: why engine 8000 served each token 2.8–3.7×
slower than 8001.**

**Status: written 2026-09-29, before any card existed. Not yet reviewed.**
`docs/adding-a-run.md` §1 wants review before renting, and that includes one
pass against the platform, image and framework defaults, not only against this
repository.

**Cost, flagged up front.** AMD Developer Cloud, 1 × MI300X at **$1.99/h**
on-demand, as read off the console on 2026-09-29 (the price at droplet creation
wins and is written here when read). Expected clock **≈ 1.3 h ≈ $2.6**, session
budget **≤ 2 h ≈ $4**, hard stop. The credits expire **2026-10-18**; runs 1–3
spent ≈ $8.6 of them.

**Every predicted figure below comes from `bench/predictions.py` table 11 or
from a measured row it names.** Where this sheet disagrees with a measured
number, `docs/benchmarks/` wins.

---

## What this run is not

- **Not block A again.** The fleet's memory bill held on run 3 (§2 there), and
  its decode-step row needs a level without prefill, which is a different
  scenario (§7).
- **Not a new routing property.** The ring was checked offline in run 3 against
  every prefix the harness sends (§6 there); nothing here re-measures it.
- **Not the replica question.** Two engines on one card, as in run 3.

## What run 3 established, and which of it transfers

| From run 3 | Status here |
|---|---|
| Pools: solo 1 119 644, each engine 506 525, serial launch equalises them | Transfers as checkpoint A's reference: same card, flags and image |
| The droplet check, 5.04 ms: `FIT = mi300x-run2` | Re-read: a different droplet may be rented |
| Clean `h`: 0.784 at N = 31 under affinity, 0.024 and 0.000 past round robin's cliff | The rotation model's first facing; this run faces it on both arms |
| Every clean non-evicting level hit whole prefixes or nothing | The test this run's levels must pass: hits over 3 200 an integer until eviction |
| 8000 slower than 8001 at an equal share, with nothing in the logs to tell them apart | Open, and §3 is for it |
| Bounded loads moved 2–15 % of affinity's requests, nearly all towards 8001 | A symptom of the pair; whether it survives a symmetric pair is §3's consequence |

---

## 0 · Before the card (no credits spent)

- [x] **`--reset-cache`** in `bench/harness.py`: `POST /reset_prefix_cache` to
      every metrics endpoint before each level's warmup, raised on 404 or
      refusal, recorded as `harness_cache_reset` per level. Read at
      `v0.27.1`: the route exists only under `VLLM_SERVER_DEV_MODE=1`, it
      refuses while any block is held, and it clears the local hit-rate window
      without touching the Prometheus counters the harness takes deltas of.
- [x] **`--warmup-passes 2`**: under round robin over two engines with N odd,
      the second pass seeds each prefix on the other engine. Under affinity it
      re-hits the owner and changes nothing. Recorded as
      `harness_warmup_passes`.
- [x] **The startup-log patterns** read `max_num_batched_tokens` and the
      backend from a raw ROCm log (`bench/tests/test_harness.py`, against run
      3's `engine-8000.log`).
- [ ] **The reset on a live engine**, not only in the source: §1's first gate.
      A 404 means the variable did not reach the engine, and nothing after it
      can run.
- [x] `bench/harness.py --dry-run --scenario router-arm-n127 --accelerator
      mi300x-run2 --reset-cache --warmup-passes 2` exits 0 off-card, and so
      does `fleet-bill-half`.
- [ ] The review passes, one of them external: the image's default
      environment (does `VLLM_SERVER_DEV_MODE` survive `docker run -e`?), the
      router's `-bounded-load 1` meaning "off" (`router/main.go`), and what
      `rocm-smi` and `ps` can see from the host of a process in the container.
- [ ] Read the hourly price off the console and write it into the header.

---

## 1 · Droplet up, and the pair (~15 min, mostly download)

The droplet, image, container, `tmux` session and `scp` of `bench/` and
`router/` are MI300X run 3's §1 verbatim, with `run4` for `run3`
(`docs/benchmarks/runsheets/mi300x-run-3.md`). The router image is built on
the droplet. No solo engine: block A is not repeated.

The pair is launched as in run 3 §3, serially, and **with dev mode on**:

```
export VLLM_SERVER_DEV_MODE=1
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --dtype auto \
  --gpu-memory-utilization 0.45 --max-model-len 9000 \
  --max-num-batched-tokens 2048 > /workspace/run4/engine-8000.log 2>&1 &
until curl -sf 127.0.0.1:8000/health; do sleep 2; done
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8001 --dtype auto \
  --gpu-memory-utilization 0.45 --max-model-len 9000 \
  --max-num-batched-tokens 2048 > /workspace/run4/engine-8001.log 2>&1 &
until curl -sf 127.0.0.1:8001/health; do sleep 2; done
```

**Gate: the reset answers on both engines**, idle:

```
for p in 8000 8001; do curl -s -X POST 127.0.0.1:$p/reset_prefix_cache; echo; done
# {"success":true} twice. A 404 is dev mode missing: stop, relaunch with it
```

Checkpoint A is run 3's: both pools within 5 % of 506 525 and equal to each
other; `Available KV cache memory` equal; `ROCM_ATTN`, prefix caching True,
budget 2 048. Then the droplet check, run 3 §3's c = 1 level, against 8001 at
c = 1: `python3 bench/harness.py --scenario fleet-bill --port 8001 …` reads its
first level and is stopped after it. The reading is 5.04 ms on run 3's droplet
and 5.22 ms on run 2's. Set `FIT` from it.

---

## 2 · Checkpoint A

As run 3 §2, with one row added:

| Log line | Predicted | On a miss |
|---|---|---|
| `GPU KV cache size`, each engine | 506 525 (run 3) | outside 5 %: stop and account for it from the two memory lines |
| `Available KV cache memory`, 8000 against 8001 | equal | the launch gate was early: kill both, relaunch per §1 |
| `POST /reset_prefix_cache` | `{"success":true}` on both | 404: dev mode did not reach the engine |

---

## 3 · Block 0 — why the pair is unequal (~12 min)

Run 3 gave the same flags, pool and graphs to both engines, and 8000 was still
slower at an equal share. Three hypotheses, one cheap level each.
`fleet-bill-half` is run 3's block-A level: 96 unique 4 000-token prompts at
c = 32, about 45 s.

| Step | What runs | If 8000 is slower here | If not |
|---|---|---|---|
| 0a | `fleet-bill-half` on 8000 **alone**, 8001 idle; then on 8001 alone | the difference is per-process, not contention: a launch-order or placement effect that holds with the card to itself | the difference needs both engines busy: contention |
| 0b | both together, as run 3 did, sampling `ps -eo pid,psr,pcpu,comm` and `rocm-smi --showuse` every 5 s on the host | read the samples: an EngineCore at ~100 % of one core, or two on one core, is the CPU | — |
| 0c | kill both, wait for `rocm-smi` to show the card empty, relaunch **8001 first**, then 0b again | slowness stayed on 8000: the port or process identity, which nothing in the config reaches | slowness moved to 8001, the first launched: launch order |

```
python3 bench/harness.py --scenario fleet-bill-half --accelerator $FIT \
  --port 8000 --out /workspace/run4/results/alone-8000      # 0a, then --port 8001 into alone-8001
# 0b and 0c: run 3's two concurrent calls, into together-<order>-800x
```

**Predicted, if the pair is symmetric:**

| Quantity | Predicted | Source |
|---|---|---|
| Median ITL, one engine alone at c = 32 | **12.46 ms**, run 2's c032 median ITL at the same budget and backend; floor 11.83 ms at the run-2 fit | `mi300x-run2.md` §3, table 11's fit |
| 8000 alone against 8001 alone | equal within 5 % | the null hypothesis |
| Median ITL, each engine, both at c = 32 together | **≈ 24.9 ms each**, twice the alone step, if the card alternates fairly | derived, stated as such; run 3 measured 48.9 / 26.8 |

**Decision for block B.** If 0c moves the slowness with launch order, block B
runs on whatever order is faster and says so. If 0b points at CPU, pin each
engine's processes to disjoint cores with `taskset` (8000 on 0–9, 8001 on
10–19), repeat 0b once, and run block B pinned if it evens the pair. If nothing
evens the pair, block B runs on the unequal pair, and §4's affinity column is
read against the moved-request share it records.

---

## 4 · Block B — the router arm, both arms clean (~45 min)

Run 3's ten calls, in run 3's order, with the two fixes on every call. `arm()`
takes an optional third argument for the router's bounded load:

```
arm() {   # arm <prefix|round_robin> <031|063|127|255|511> [bounded-load]
  docker rm -f router >/dev/null 2>&1
  docker run -d --name router --network host prefix-router:dev \
    -listen :8080 -upstreams http://127.0.0.1:8000,http://127.0.0.1:8001 \
    -policy $1 -dial-timeout 250ms -bounded-load ${3:-1.25} >/dev/null
  until curl -sf localhost:8080/v1/models >/dev/null; do sleep 1; done
  docker exec -w /workspace/i-serve run4 python3 bench/harness.py \
    --scenario router-arm-n$2 --accelerator $FIT --port 8080 --expect-policy $1 \
    --metrics-endpoint 127.0.0.1:8000 --metrics-endpoint 127.0.0.1:8001 \
    --reset-cache --warmup-passes 2 \
    --startup-log /workspace/run4/engine-8000.log --reference-pool 506525 \
    --out /workspace/run4/results/$1-n$2${3:+-bl$3}
  docker logs router > /workspace/run4/router-$1-n$2${3:+-bl$3}.log 2>&1
}

arm prefix 031;      arm round_robin 031
arm round_robin 063; arm prefix 063
arm prefix 127;      arm round_robin 127
arm round_robin 255; arm prefix 255
arm prefix 511;      arm round_robin 511
arm prefix 127 1     # bounded loads off: what the bound cost affinity on this pair
```

The arm order is kept, though the reset removes the bias it was there for, so
that the ten levels differ from run 3's in the two flags and nothing else.

**Predicted, rotation model** (table 11: room 89 requests per engine; the
harness draws prefixes in strict rotation):

| N | `h` round_robin | `h` prefix | Round robin's per-engine set / affinity's |
|---|---|---|---|
| 31 | 0.800 | 0.800 | 31 / 16 |
| 63 | 0.800 | 0.800 | 63 / 32 |
| 127 | **0** | **0.800** | 127 / 64 |
| 255 | 0 | **0** | 255 / 128 |
| 511 | 0 | 0 | 511 / 256 |

**On an unequal pair, affinity's column falls by the requests bounded loads
move**: run 3 measured 98.0 / 84.8 / 89.0 % on the owner at N = 31 / 63 / 127,
and at N = 31 the misses equalled the moved requests exactly. The
`-bounded-load 1` level at N = 127 measures that directly: with the bound off,
every request goes to its owner and `h` should be 0.800. The split then shows
what the bound was protecting against.

**Read per level:** `h` summed and per engine; hits over 3 200 (an integer until
eviction, as on run 3's clean levels); `harness_cache_reset` 1 and
`harness_warmup_passes` 2 in every JSON; the `X-Router-Policy` counts; the split.

**Stop condition.** Stop if any of the first four levels does not measure `h`
≈ 0.8. Each fix shows in its own level. Round robin at N = 31 and affinity at
N = 63 run second at their N and read above 0.8 if the reset did nothing: run 3
measured 0.859 and 0.901. Round robin at N = 63 runs first and reads 0.603 if
the second warmup pass did nothing.

---

## 5 · Block C — what the hit rate buys (no extra card time)

Run 3's runsheet §5, unchanged: TTFT compared arm against arm at the same N,
never against a floor. The upper bound at N = 127 under the rotation is
137.8–205.0 ms (table 11). Seats stay *not derivable*: no MI300X entry in
`INTERFERENCE_FITS`.

---

## 6 · Budget, drop order, and what "done" means

| | |
|---|---|
| Expected clock | ≈ 1.3 h: 15 min up, 12 min block 0, 45 min block B, the rest harvest and destroy |
| Hard stop | **2 h ≈ $4** of the credit balance |
| Drop order | The `-bounded-load 1` level first, then N = 511, then 31. **Never N = 63, 127 or 255**: 63 is the fixes' stop condition, 127 the level where the arms differ under the rotation, 255 where the two order models differ |
| Done | block 0's three steps read against the decision table; the arm pair at N = 63, 127 and 255 with `cache_reset` and `warmup_passes` recorded and the whole-prefix test passed |
| Not done | any level without `harness_cache_reset` = 1: it is run 3's defect, not a measurement |

Everything lands in `docs/benchmarks/raw/mi300x-<date>/run4/` with its README,
read once for a credential before staging, and the file count checked across
`git add` (`docs/adding-a-run.md` §2).

---

## 7 · Not in this run, and where each goes

- **The fleet's decode-step bill.** Run 3 could not face it at `h` = 0 because
  prefill filled the step. It needs a level whose steps carry no prefill: a
  scenario of its own.
- **The 0.04 at N = 31** that run 3's contamination model left over
  (`mi300x-run3.md` §5.1). The reset removes the case, not the question.
- **Zipf-distributed prefix popularity, `-key-bytes`, two cards**: run 3's
  runsheet §7, unchanged.

---

## 8 · If it goes sideways

| Symptom | First move |
|---|---|
| A level raises on the reset | Dev mode is missing, or a request is still holding blocks. `--settle` gives stragglers time; a 404 is §1's gate failing late |
| Round robin at N = 63 reads 0.603 | The warmup ran one pass: read `harness_warmup_passes` in the JSON |
| Any level's `h` exceeds 0.800 | The reset did not happen: read `harness_cache_reset` |
| 0c's relaunch leaves the card non-empty | Run 3 §3's `rocm-smi` wait; a process that will not release it means rebooting the droplet |
| The clock passes 1.7 h | Harvest what exists and destroy the droplet |
