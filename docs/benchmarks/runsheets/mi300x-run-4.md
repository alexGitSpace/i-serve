# Runsheet — MI300X run 4: the router arm, measured cleanly, and an unequal pair

MI300X run 3 asked what sending a request to the replica that already holds its
prefix buys, and did not get an answer. Two defects in the instrument voided
the arm comparison at every working set, and the pair of engines it routed
over turned out not to be a pair of equals (`docs/benchmarks/mi300x-run3.md`
§5–§6). **This sheet asks run 3's question again with the instrument fixed.
First it asks the one run 3 left: why engine 8000 served each token 2.8–3.7×
slower than 8001.**

**Status: written 2026-09-29, reviewed 2026-10-17 in two passes, before any
card existed.** The internal pass checked the sheet against `bench/` and
`router/`, and the external pass against vLLM `v0.27.1`, its ROCm image and the
scripts run 3 actually ran. Between them they found three defects that would
have voided a measurement and four that would have lost evidence or left a
branch that could not run. Each is fixed where it lived, and §0 lists them.

**The commands are scripts.** Every stage below is a file in
`docs/benchmarks/raw/mi300x-2026-10-17/run4/`, copied to
`/workspace/i-serve/run4/` beside `bench/` and `router/`. Run 3 typed the
sheet's blocks into files on the day and lost checks on the way (§0, I4);
here the files exist first and the sheet points at them. A block printed below
is an excerpt, and the file wins.

**Cost, flagged up front.** AMD Developer Cloud, 1 × MI300X at **$1.99/h**
on-demand, as read off the console on 2026-09-29 and again on 2026-10-17 on the
create page: 20 vCPU, 240 GB RAM, 192 GB VRAM, run 3's plan. The price at
droplet creation wins. Expected clock **≈ 1.3 h ≈ $2.6**, session
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
- [x] `bench/harness.py --dry-run` exits 0 off-card on every harness line the
      scripts send: the five `router-arm` levels with block B's flags, and
      `fleet-bill-half` with `--reset-cache`.
- [x] The review, both passes (2026-10-17). What the external pass verified:
      - At `v0.27.1`, `POST /reset_prefix_cache` is attached only when
        `VLLM_SERVER_DEV_MODE` parses as a true integer, and it answers
        `{"success": bool}`. It is false while any block is held
        (`serve/dev/cache/api_router.py`, `envs.py`, `v1/core/block_pool.py`).
        Enabling it logs `SECURITY WARNING: Development endpoints are enabled!`
        (`serve/__init__.py`), and §1's gate reads that line.
      - Run 3's engines were served by the Python `api_server.py`, the one that
        carries the dev routes, though the tag also ships a Rust server under
        `rust/src/server` (run 3's `engine-8000.log`).
      - `docker/Dockerfile.rocm` sets no `VLLM_SERVER_DEV_MODE`, and the engines
        inherit it from the container shell's `export`, so whether it survives
        `docker run -e` does not arise.
      - `-bounded-load 1` is "off" (`router/ring.go`: `c <= 1`), and the bound
        applies only under `-policy prefix` (`router/router.go`).
      - On this droplet `rocm-smi` on the host reports device-wide busy and power
        figures and the KFD processes under host PIDs named `python` (run 3's
        record). It cannot attribute card time to an engine.
- [x] What the review found, and where each fix went:

      | # | Defect | Fix |
      |---|---|---|
      | B1 | `fleet-bill-half`'s seed is fixed, so 0b replayed the 96 prompts 0a had left whole in both pools (≈ 403k tokens against 506 525) and 0c met them cold: run 3's §5.1 again | `--reset-cache` on every block-0 call (§3) |
      | B2 | 0b's CPU record could not see a busy core: `ps`'s `pcpu` is a lifetime average, `psr` without `-L` is one thread, `comm` does not tell the engines apart, and host PIDs are not the container's | `top -H` per 5 s and `ps -L` for `psr`, inside the container, over each engine's process tree (§3) |
      | B3 | The stop rule wanted `h` ≈ 0.8 from affinity, which the same sheet says an unequal pair lowers by the moved requests, so it would have stopped a run §3 tells to proceed | Stop only on what each fix shows (§4) |
      | I1 | 0c's relaunch truncated `engine-800x.log`, checkpoint A's evidence | Relaunch logs are `-r2` (§3) |
      | I2 | "Runs on whatever order is faster" chose between two unequal pairs | Record which port is slow and carry on (§3) |
      | I3 | `taskset` on a pid pins one thread of one process | `taskset -a` over each engine's process tree (§3) |
      | I4 | Run 3 ran scripts (its raw `s1`–`s3`, `arm.sh`), not the sheet's blocks, and run 4's `arm()` had lost their `FIT` guard, captured output and `ARMDONE` line, and the launch its death check | The scripts exist before the card (header) |
- [x] The hourly price, off the create page on 2026-10-17: $1.99/h, unchanged.
- [ ] The credit balance and the **time of day** the credits expire, off
      *Billing*, written into the header.

---

## 1 · Droplet up, and the pair (~15 min, mostly download)

The droplet, image, container and `tmux` session are MI300X run 3's §1
verbatim, with `run4` for `run3` (`docs/benchmarks/runsheets/mi300x-run-3.md`).
`scp` carries `bench/`, `router/` and this run's scripts to
`/workspace/i-serve/` and `/workspace/i-serve/run4/`. The router image is built
on the droplet. No solo engine: block A is not repeated. The container's
commands run in `tmux` window 0 and the host's in window 1.

```
bash /workspace/i-serve/run4/s1.sh        # container; progress in /workspace/run4/stage
```

`s1.sh` checks that `top`, `pgrep` and `taskset` exist in the image, downloads
the weights and launches the pair as run 3 §3 did: serially, each engine's
death caught while `/health` is awaited, and **with dev mode on**, exported in
`lib.sh`. **Then it gates each engine** on three things: the dev-mode warning
in its log, `Route: /reset_prefix_cache` in its route listing, and
`{"success":true}` from an idle reset into `reset-gate.txt`. A failed gate
leaves `GATE_<what>_<port>` in `stage` and stops. Checkpoint A's lines go to
`checkpoint-a.txt`.

The droplet check is run 3 §3's c = 1 level against 8001, killed once its
JSON exists. It prints one line, and the reading is 5.04 ms on run 3's droplet
and 5.22 ms on run 2's. Set `FIT` from it, **in both windows**: the host's
`arm.sh` does not see the container's shell, and refuses to start without it.

---

## 2 · Checkpoint A

As run 3 §2, with two rows added:

| Log line | Predicted | On a miss |
|---|---|---|
| `GPU KV cache size`, each engine | 506 525 (run 3) | outside 5 %: stop and account for it from the two memory lines |
| `Available KV cache memory`, 8000 against 8001 | equal | the launch gate was early: kill both, relaunch per §1 |
| `SECURITY WARNING: Development endpoints are enabled!`, and `Route: /reset_prefix_cache` | in both logs | absent: dev mode did not reach the engine; `export` it in this shell and relaunch |
| `POST /reset_prefix_cache` | `{"success":true}` on both | 404: dev mode did not reach the engine. `false` while idle: a request still holds blocks |

---

## 3 · Block 0 — why the pair is unequal (~15 min, with a relaunch)

Run 3 gave the same flags, pool and graphs to both engines, and 8000 was still
slower at an equal share. Three hypotheses, one cheap level each.
`fleet-bill-half` is run 3's block-A level: 96 unique 4 000-token prompts at
c = 32, about 45 s.

**Every call resets its engine's cache first.** `fleet-bill-half` has one
fixed seed, and 96 × 4 200 tokens fit in either pool, so without the reset 0b
would replay prompts 0a had left whole in both engines, and 0c, after the
relaunch, would meet them cold. The reset needs no shared prefix to act on
(`run_level` resets before it looks at the workload).

| Step | What runs | If 8000 is slower here | If not |
|---|---|---|---|
| 0a | `fleet-bill-half` on 8000 **alone**, 8001 idle; then on 8001 alone | the difference is per-process, not contention: a launch-order or placement effect that holds with the card to itself | the difference needs both engines busy: contention |
| 0b | both together, as run 3 did, with each engine's process tree recorded (below) | an EngineCore thread at ~100 % of a core, or both engines' busy threads sharing cores, is the CPU | — |
| 0c | kill both, confirm the card empty on the host, relaunch **8001 first**, then 0b again | slowness stayed on 8000: the port or process identity, which nothing in the config reaches | slowness moved to 8001, the first launched: launch order |

```
bash /workspace/i-serve/run4/hostrec.sh        # host, window 1, before s2; left running
bash /workspace/i-serve/run4/s2.sh $FIT        # container: 0a, then 0b as together-8000first
bash /workspace/i-serve/run4/s3.sh $FIT        # container: 0c, then 0b as together-8001first
# s3 waits at 0C_WAIT_CARD: on the host, run
#   rocm-smi --showmeminfo vram --showpids
# until ~0.3 GB is used and no KFD process is listed, then touch /workspace/run4/card-empty
```

**What 0b records, and what it cannot see.** Inside the container, over the
pids of both engines' process trees (an APIServer and its EngineCore each):
`top -b -H -d 5` gives `%CPU` per thread per 5-s interval, and `ps -L -o
pid,tid,psr,comm` every 5 s gives the core each thread last ran on.
`ps-<tag>.txt` maps each pid to its `--port`. `ps`'s own `pcpu` is not read:
it is a lifetime average, and an engine up ten minutes cannot show a core
pinned for one 45-s level. On the host, `rocm-smi` every 5 s gives busy %,
power and clocks for the **whole device**. It cannot say which engine had the
card, only that the card was busy.

The relaunch writes `engine-8001-r2.log` and `engine-8000-r2.log`. The first
launch's two logs are checkpoint A's evidence and stay as they are. The
relaunch passes the same gates, and its pools go to `checkpoint-a.txt` beside
the first launch's.

**Predicted, if the pair is symmetric:**

| Quantity | Predicted | Source |
|---|---|---|
| Median ITL, one engine alone at c = 32 | **12.46 ms**, run 2's c032 median ITL at the same budget and backend; the harness prints a floor of 11.99 ms at the run-2 fit (4 200 tokens of context; `mi300x-run2.md` gives 11.83 ms at input + 100) | `mi300x-run2.md` §3, table 11's fit |
| 8000 alone against 8001 alone | equal within 5 % | the null hypothesis |
| Median ITL, each engine, both at c = 32 together | **≈ 24.9 ms each**, twice the alone step, if the card alternates fairly | derived, stated as such; run 3 measured 48.9 / 26.8 |

The 12.46 ms came from `vllm bench serve` over 128 prompts with prefix caching
off at 0.90. Here it is the harness over 96 prompts at 0.45, so the absolute
comparison is soft. The comparison that decides, 8000 against 8001, uses one
instrument on both sides and is not.

**Decision for block B.**

- **Launch order.** If 0c moves the slowness with launch order, no order gives
  an equal pair, so there is nothing to choose. Block B runs on the relaunched
  pair, and the write-up names the slow port.
- **CPU.** If 0b points at CPU, pin each engine's whole process tree to
  disjoint cores, 8000 on 0–9 and 8001 on 10–19. Pinning one pid would pin one
  thread of one process, so the script uses `taskset -a` over the tree, and
  threads created later inherit the mask. Then 0b runs once more:

  ```
  bash /workspace/i-serve/run4/pin.sh $FIT engine-8000-r2.log engine-8001-r2.log   # together-pinned
  ```

  If that evens the pair, block B runs pinned.
- **Neither.** If nothing evens the pair, block B runs on the unequal pair,
  and §4's affinity column is read against the moved-request share it records.

---

## 4 · Block B — the router arm, both arms clean (~45 min)

Run 3's ten calls, in run 3's order, with the two fixes on every call, then
one with bounded loads off. `arm.sh` is run 3's `arm.sh`, with three changes:
the two fixes, a third argument for the router's bounded load, and each level
checked against the stop rule below as soon as its JSON lands. `blockb.sh`
runs the eleven calls and stops at the first level that fails the rule:

```
docker build -t prefix-router:dev /workspace/i-serve/router                       # host, once
FIT=$FIT LOG=engine-8000-r2.log bash /workspace/i-serve/run4/blockb.sh            # host
#   LOG: the live engine-8000 startup log -- engine-8000.log if 0c was dropped
```

| Order | Calls |
|---|---|
| 1–4 | `prefix 031`, `round_robin 031`, `round_robin 063`, `prefix 063` |
| 5–8 | `prefix 127`, `round_robin 127`, `round_robin 255`, `prefix 255` |
| 9–10 | `prefix 511`, `round_robin 511` |
| 11 | `prefix 127 1`: bounded loads off, what the bound cost affinity on this pair |

Progress, exit codes and each level's check go to `arm-progress`. To drop a
level (§6), interrupt `blockb.sh`, source `arm.sh` in the host shell and call
`arm` by hand.

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

**Stop condition: on what each fix shows, never on affinity's `h` alone.**
Affinity below 0.8 is what this sheet predicts for an unequal pair: run 3's
84.8 % on the owner at N = 63 would read ≈ 0.68. A rule asking it for 0.8
would stop the run that §3's last branch tells to proceed. `arm.sh` stops at
any level where:

| Condition | What it means | Run 3's reading |
|---|---|---|
| `harness_cache_reset` ≠ 1 or `harness_warmup_passes` ≠ 2 | a flag did not reach the harness | — |
| `h` > 0.81, any arm | the reset did not take: a second arm hit the first arm's bodies | 0.859 round robin N = 31, 0.901 affinity N = 63 |
| round robin at N ≤ 63 with `h` < 0.78 | one warmup pass: round robin carries no bound, and inside its room every prefix should be seeded on both engines | 0.603 at N = 63 |

The rule was exercised off-card on run 3's readings and on 0.68 for affinity:
it stops on the first three and lets the fourth through.

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
| Expected clock | ≈ 1.3 h: 15 min up, 15 min block 0, 45 min block B, the rest harvest and destroy |
| Hard stop | **2 h ≈ $4** of the credit balance |
| Drop order | The `-bounded-load 1` level first, then N = 511, then 31. **Never N = 63, 127 or 255**: 63 is the fixes' stop condition, 127 the level where the arms differ under the rotation, 255 where the two order models differ |
| Done | block 0's three steps read against the decision table; the arm pair at N = 63, 127 and 255 with `cache_reset` and `warmup_passes` recorded and the whole-prefix test passed |
| Not done | any level without `harness_cache_reset` = 1: it is run 3's defect, not a measurement |

Everything lands in `docs/benchmarks/raw/mi300x-2026-10-17/run4/` beside the
scripts that produced it, with its README, read once for a credential before
staging, and the file count checked across `git add` (`docs/adding-a-run.md`
§2). If the run takes place on another date, the directory is renamed to it.

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
| `stage` reads `TOOL_MISSING <tool>` | `apt-get install -y procps util-linux` in the container, then `s1.sh` again |
| `stage` reads `GATE_<what>_<port>` | Dev mode did not reach that engine, or the idle reset was refused. Read `reset-gate.txt` and the log's last lines |
| `stage` reads `E<port>_DIED` | The engine's log tail; an early launch gate shows as 8001 short of memory |
| A level raises on the reset | Dev mode is missing, or a request is still holding blocks. `--settle` gives stragglers time; a 404 is §1's gate failing late |
| `blockb.sh` stops | `arm-progress` names the level and the condition, and §4's table says which fix failed |
| `arm.sh` exits at once with `FIT` or `LOG` | Set both in the host's shell |
| 0c's relaunch leaves the card non-empty | Run 3 §3's `rocm-smi` wait; a process that will not release it means rebooting the droplet |
| The clock passes 1.7 h | Harvest what exists and destroy the droplet |
