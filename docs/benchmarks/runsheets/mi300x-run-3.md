# Runsheet — MI300X run 3: the router arm, and what a fleet costs on one card

`router/` routes on cache locality and has been watched doing it on `kind`, and
not one number from that day says the seat count moves: the routing property and
the seat effect are different measurements, and the second one needs a card and a
prediction written before it (`router/README.md` §8). **This sheet is that
prediction.** It asks one question — *what does sending a request to the replica
that already holds its prefix buy, and what does having two replicas cost in the
first place?* — and it asks it in the regime where the answer is not obviously
"nothing": a card where capacity binds — a premise MI300X run 1 has put in question
(status, below).

**Status: written 2026-09-19, before any card exists and before run 1 has been
taken. Not reviewed** — `docs/adding-a-run.md` §1 says a sheet is reviewed and
committed before renting, and neither has happened for this one.

**Re-derived 2026-09-27, after MI300X run 1** (`docs/benchmarks/mi300x-run1.md`):
the figures below are table 11 at `memory_bytes` = 192 GiB, with run 1's larger
measured pool shortfall, 2.1 %, in the retained-prefix count, where they were
192e9 and the L40S's 7 %. Four of run 1's findings bear on this sheet, and the
review has to weigh them before renting:

- **The capacity limit is no longer ahead of latency by 8 %.** At one context
  the two derived limits tie. Table 11 prints capacity by the clean arithmetic at
  a 4 200-token seat, 273 (~267 after the 2.1 %, which it applies only to the
  retained-prefix count), against 286 by latency at a 4 000-token read — a
  context mismatch as much as a gap.
- **Latency as served bound far below both**: at `h` = 0, TPOT p99 crossed 50 ms
  between 8 and 32 seats, consistent with prefill interference at an `mfu` of
  0.166 (run 1 §6). At `h` = 0.8 the linear part of each prefill scales by
  800 / 4 000; the attention part does not, because the 800 new tokens still
  attend over the whole prompt. Whether 32 seats per engine sit inside the target
  is the question this sheet's seat effect now depends on.
- **The token-budget ceiling does not apply here**: at 800 prefilled tokens it is
  2 048 × 200 / 1 000 ≈ 410 per engine, above the cap and far above block B's 32.
- **Each engine holds its CUDA graphs outside its share**: vLLM 0.27.1 does not
  set memory aside for them. Run 1 captured 4.12 GiB at `max_num_seqs` 256; this
  sheet's serve lines leave it at the card's default of 1024, which captures more
  decode graphs, so read each engine's `Graph capturing finished … took` line —
  the pair holds 2 × (86.3 GiB + that), not 172.5 GiB.

Blocks A and B stay as written until the review decides.

**Runs are numbered per card**, so "run 3" is this one and the L40S's third run
is named with its card wherever both appear below. Run 2 is
`mi300x-run-2.md`, the card configured for itself: the token budget and the
attention kernel, written 2026-09-27 after run 1 took FP8 KV off it. If the
credit window closes before run 2, this one keeps its number — a gap in the
numbering is cheaper than a sheet whose name stops matching the run that faced
it.

**Run order: after MI300X run 2. Re-derived from it on 2026-09-27**
(`docs/benchmarks/mi300x-run2.md`). Five of its findings change this sheet, and
the review weighs them with run 1's four:

- **Two droplets, two fits.** Run 1's server line ran the decode step 21 %
  faster and prefill 33 % faster on run 2's droplet, and fitted 0.57 / 0.247
  there against 0.46 / 0.166. Which droplet this run gets is not known before
  it is rented. Table 11 now prints every figure that needs a coefficient at
  both fits, and §3 opens with a c = 1 level that says which one applies.
  Capacity needs no coefficient, and both droplets logged the same pool,
  1 123 065 tokens.
- **Latency as served depends on the droplet too.** At `h` = 0 run 1's droplet
  crossed 50 ms TPOT p99 between 8 and 32 seats; run 2's was still inside at 32
  (47.5 ms). Block B holds 32 seats per engine at `h` = 0.8, with less prefill
  per request. Two engines time-slicing one card is not measured on either
  droplet.
- **The budget stays at 2 048.** Run 2 found chunked prefill under `ROCM_ATTN`
  cost ~10 % of TTFT at 4 000 tokens. At `h` = 0.8 a request prefills 800 new
  tokens, one chunk at any budget, so block B is untouched. Block A's unique
  prompts are chunked, as they were when both fits were measured.
- **The backend stays `ROCM_ATTN`**, the automatic choice. `ROCM_AITER_FA`
  misreads a mixed batch on the V2 model runner (`mi300x-run-2.md`, *The five
  serve rows*), and `TRITON_ATTN` was slower.
- **The interference model was faced once.** Fed each serve row's own median
  ITL and c = 1 TTFT, it came within 7 % of TPOT p50 at run 1's budget and
  kernel (run 2 §6). It is still not in `INTERFERENCE_FITS` (§5).

**Every predicted figure below comes from `bench/predictions.py` table 11**,
added 2026-09-19; re-run it if the module has changed since. Where this sheet
disagrees with a measured number, `docs/benchmarks/` wins.

**The instrument is `bench/harness.py`**, not `vllm bench sweep`: a controlled
prefix cache hit rate is the one thing the sweep cannot be asked for
(`bench/predictions.py`, module docstring), and `h` is this run's dependent
variable. Prefix caching is **on** in both arms, which run 3 measured to be free
when it buys nothing (+0.027 % on the decode step, `docs/SLO.md` §6).

**Cost, flagged up front.** AMD Developer Cloud, 1 × MI300X at **$1.99/h**
assumed (the console price at droplet creation wins and is written here when
read). Expected clock **≈ 2 h**, session budget **≤ 3 h ≈ $6**, hard stop. The
credits expire **2026-10-18**; runs 1 and 2 spent ≈ $6.7 of them.

---

## What this run is not

- **Not a routing-correctness run.** Which replica a prompt lands on, how evenly
  the ring spreads, what a stale fleet costs: measured on `kind` on 2026-09-19,
  `deploy/router/README.md`. Nothing here re-measures it.
- **Not the replica question.** Two engines *on one card* is not two cards. What
  it can and cannot settle is §3.
- **Not a second model, and not FP8.** One model, BF16, `docs/adding-a-run.md`
  §7 and run 2 respectively.

## What earlier work established, and which of it transfers

| From | Status here |
|---|---|
| Run 3 (L40S): `h` = 0.8 is worth 12.5 → 37.8 seats | One prefix shared by **every** request — the ceiling of what `h` is worth. This run prices a *distribution* of prefixes, which is the open item `docs/SLO.md` §10 holds, and the first thing that can make `h` fall below its nominal value |
| Run 3: the `h` = 0 overhead of prefix caching is +0.027 % | Transfers as a decision, not a number: both arms serve with the cache on, so the arms differ in the routing policy alone |
| `kind`, 2026-09-19: the router routes, and a stale fleet costs 29 % of requests | Routing behaviour only. It says nothing about TTFT or seats, which is why this sheet exists (`router/README.md` §8) |
| MI300X `eff_mem`, `mfu`, and the prefill interference | **Two fits from two droplets**: run 1's 0.46 / 0.166 and run 2's 0.57 / 0.247. Quote each with the run that produced it, never the L40S's (`docs/SLO.md` §9). Which one applies here is read in §3 |
| §6: which limit binds is a property of the card | Derived at the prior, the MI300X's two limits tie at a common context, and a second engine subtracts the same weights from both, so the tie holds (asserted in `bench/tests/test_roofline.py`). At either fit latency binds first. Measured, latency as served binds first at `h` = 0 on both droplets, at a seat count that differs between them — see the status note |

---

## 0 · Before the card (no credits spent)

Writing this sheet found three defects that would each have let the run spend
credits and measure nothing **without erroring**. They are prerequisites, not
risks, and the first one was verified rather than argued. **All four boxes below
were closed on 2026-09-19**, in the commit that follows this one — the sheet is
left showing what it found, because a runsheet that quietly matches the code it
fixed cannot be judged.

- [x] **The router cannot read the prompts the harness sends.** `bench/loadgen.py`
      sends `"prompt": [151, 2934, …]` — an array of token ids — and
      `router/key.go`'s `writeStringOrArray` unmarshals a string or an array of
      *strings*, so the body parses, no key is built, and every request takes the
      `no-prompt` fallback to `round_robin`. Both arms would be the control.
      Verified 2026-09-19 against `cacheKey` itself, not by reading it. The fix
      is to teach the key the number-array shape, with a test: for this harness
      token ids are the *better* key, since the BPE boundary approximation of
      `router/README.md` §2 disappears when the ids are what is hashed.
      **Done**: `writeStringOrArray` reads `[]json.Number`, three tests in
      `router/key_test.go`.
- [x] **A gate that makes the above impossible to miss again.** The harness
      records `X-Router-Policy` per request, and a level is invalid unless every
      response on the prefix arm carries `prefix` and every response on the
      control carries `round_robin` — the shape `check_hit_rate` already has.
      **Done**: `--expect-policy`, which invalidates the level rather than
      warning it, and `Record.policy` carrying the header into the
      per-request artefacts.
- [x] **There is no control arm yet.** `round_robin` is a fallback in this
      router, not a policy that can be asked for (`router/README.md` §1), and a
      control that reaches the engines by a different door differs in hops
      rather than in policy. Add `-policy prefix|round_robin`. **Done**, and
      the control still reads the body and computes the key before discarding
      it, so the arms differ in the decision and not in the work.
- [x] **Metrics through the router are one replica's, picked by the fallback.**
      `bench/harness.py` scrapes `/metrics` at the same `host:port` it loads,
      which behind the router is an unkeyed path and therefore a round-robin
      choice. Add a repeatable `--metrics-endpoint`, scraped directly from each
      engine and summed, so `hit_rate()` reads the fleet and not a sample of it.
      **Done**: `--metrics-endpoint`, `scrape_fleet()` and `delta_fleet()`,
      which sum the counters before taking the ratio — an average of two
      engines' ratios would give an idle engine an equal vote.
- [ ] Three scenarios in `bench/scenarios/`: `router-arm`, five levels at
      `num_prefixes` 32 / 64 / 128 / 256 / 512, `prompt_tokens` 4 000,
      `prefix_tokens` 3 200, `concurrency` 64, `num_prompts` = 4 × N and never
      below 256; and `fleet-bill` / `fleet-bill-half`, one level each of unique
      prompts at concurrency 64 and 32 for block A. `fleet-bill` opens with a
      level at concurrency 1, 20 unique 4 000-token prompts: the droplet check
      of §3.
- [ ] `python3 bench/predictions.py` — table 11 open beside the terminal.
- [x] Runs 1 and 2 taken, `mi300x-run1` and `mi300x-run2` in `ACCELERATORS`.
- [ ] An MI300X entry in `INTERFERENCE_FITS`, or the review's decision not to
      register one. Without it §5's seat line stays *not derivable*, and that is
      a correct answer rather than a missing one.
- [ ] `bench/harness.py --dry-run --scenario router-arm --accelerator mi300x-run1`
      and the same with `mi300x-run2` exit 0 off-card.
- [ ] The router image built **on the droplet** (`docker build -t
      prefix-router:dev router`): a Mac builds arm64 and the droplet is amd64.
- [ ] Read the hourly price off the console and write it into the header above.

---

## 1 · Droplet up, and the single engine first (~25 min, mostly download)

Same droplet as run 1 — one card, not eight; destroy, never stop; everything
under `/workspace` and copied off before the destroy (`mi300x-run-1.md` §1). The
container is the same image with the same device flags, plus `--network host` so
the router can reach the engines, and the repository's `bench/` and `router/`
copied in by `scp`.

```
docker run -it --rm --name run3 --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc host \
  -v /workspace:/workspace -v /workspace/hf:/root/.cache/huggingface \
  --entrypoint bash vllm/vllm-openai-rocm:v0.27.1
```

**The single engine runs first**, because block A's control has to be measured
and not derived:

```
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --dtype auto \
  --gpu-memory-utilization 0.90 --max-model-len 9000 \
  --max-num-batched-tokens 2048 2>&1 | tee /workspace/run3/engine-solo.log
```

No `--enable-prefix-caching`: it is the V1 default, and the log line is the
proof rather than the flag.

Record the host as run 2 did, into `/workspace/run3/host-under-load.txt`, while
§3's first level runs: `rocm-smi --showclocks --showpower --showperflevel
--showuse --showtemp`, the CPU model, `nproc`, `uname -r` and
`/sys/module/amdgpu/version`. The droplet is a variable (run 2 §3), and this is
the record that places a third one against the second.

---

## 2 · Checkpoint A — three startup logs against the arithmetic

Read the solo log now and the pair's two logs after §3 relaunches; the gates are
the same and the comparison between them is the whole of block A's first row.

```
grep -E 'GPU KV cache size|Maximum concurrency|prefix_caching|max_num_batched_tokens|block_size|backend|dtype|non-torch|activation' \
  /workspace/run3/engine-*.log
```

| Log line | Predicted | Source | On a miss |
|---|---|---|---|
| `GPU KV cache size`, solo at 0.90 | **1 123 065**, logged by runs 1 and 2 on two droplets; **1 147 072** derived | runs 1–2, table 11 | Within §9's 5 % of the logged figure is a pass: the same card, flags and image, and run 1's two launches differed by 1.4 %. The two figures are 2.1 % apart, so the two-band rule run 1 used no longer separates them |
| `GPU KV cache size`, each engine of the pair at 0.45 | **517 926** derived, **~507 000** corrected | table 11 | Within 5 % of the corrected figure |
| The pair's two pools summed | **1 035 852** derived, against the solo engine's own figure | table 11 | The difference, **111 220 tokens**, *is* the second copy of the weights. A sum that does not show it means the flag was not obeyed and §3 has nothing to measure |
| `prefix_caching` | **True** everywhere | V1 default | If false, relaunch: every level in §4 asks for a hit rate |
| `max_num_batched_tokens` | **2 048** everywhere | the launch line | A different value moves the interference and makes run 1's fit inapplicable |
| attention backend, `block_size`, dtype | `Overriding with ROCM_ATTN`, as both runs logged | runs 1–2 | Not gates, except the backend: one that differs invalidates both fits, and that is a stop rather than a note |

---

## 3 · Block A — the fleet's bill (~15 min)

The one block that needs no router: the same load against **one** engine at
`0.90` and against **two** at `0.45` each, unique prompts, `h` = 0.

**The droplet check comes first.** `fleet-bill`'s opening level is one request at
a time. Its median ITL at 4 000 tokens was 6.61 ms on run 1's droplet and
5.22 ms on run 2's. Set `FIT=mi300x-run1` if it reads nearer the first and
`FIT=mi300x-run2` if nearer the second, and write the figure down. If it sits
far from both, the droplet is a third kind: keep `FIT=mi300x-run2`, note it, and
read block A against both rows of the table below.

```
# against the solo engine
python3 bench/harness.py --scenario fleet-bill --accelerator $FIT \
  --host 127.0.0.1 --port 8000 --startup-log /workspace/run3/engine-solo.log \
  --out /workspace/run3/results/solo
```

Then stop it and bring the pair up **serially — the second only once the first
has logged its KV pool**:

```
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --dtype auto \
  --gpu-memory-utilization 0.45 --max-model-len 9000 \
  --max-num-batched-tokens 2048 2>&1 | tee /workspace/run3/engine-8000.log &
# wait for "GPU KV cache size" in that log, then the same at --port 8001
```

The ordering is load-bearing, and checkpoint A is what tests it: vLLM sizes its
pool from the memory it finds free at start-up, so two engines racing each other
through that measurement is a way to get two pools that do not add up to the
card. Read both logs rather than assuming the flag was obeyed. Then the same
level against the pair, half its concurrency to each engine:

```
python3 bench/harness.py --scenario fleet-bill-half --accelerator $FIT \
  --port 8000 --startup-log /workspace/run3/engine-8000.log \
  --out /workspace/run3/results/pair-8000 &
python3 bench/harness.py --scenario fleet-bill-half --accelerator $FIT \
  --port 8001 --startup-log /workspace/run3/engine-8001.log \
  --out /workspace/run3/results/pair-8001 &
```

Predicted (table 11), at both droplets' fits where a row needs one:

| Quantity | One engine | Two engines | Difference |
|---|---|---|---|
| KV pool, tokens | 1 147 072 | 1 035 852 | **−111 220 (−9.7 %)** |
| Seats at a 4 200-token seat, capacity | 273 | 246 | **−27** |
| Seats at 50 ms, latency floor, 4 000 tokens, run 1 / run 2 fit | 178 / 228 | 151 / 200 | **−27 / −28** |
| Decode step at 64 seats fleet-wide, `eff_mem` 0.46 | 22.21 ms | 28.94 ms | **+6.73 ms (+30 %)** |
| Decode step at 64 seats fleet-wide, `eff_mem` 0.57 | 17.92 ms | 23.35 ms | **+5.43 ms (+30 %)** |

One sentence holds all five rows: **both limits have the form
`(X − weights) / (context × kv_per_token)`, so a second copy of the weights costs
the same seats in each, and a second weights read costs the step
`weights / (bandwidth × eff_mem)`.**

**What this cannot settle.** Two engines on one card time-slice that card, so the
measured rise in the decode step is the duplicated read **plus** whatever running
two processes on one accelerator costs. A rise larger than the fit's own figure,
+6.73 or +5.43 ms by the droplet check, is therefore not evidence against the
arithmetic, and this run cannot split the two. The open
item as originally posed — replicas duplicating the weights read — is about two
*cards* and stays open (`mi300x-run-1.md`, *Not in this run*). What this block
does settle is what a single-card fleet actually pays, which is the arrangement
§4 routes over.

---

## 4 · Block B — the hit rate each policy can hold (~50 min)

Five working sets × two policies, at 64 seats across the fleet, 32 per engine,
prompts of 4 000 tokens behind a shared prefix of 3 200 — the construction that
measured `h` = 0.800 on every cached level of run 3.

The model, from table 11: a replica retains about **116** prefixes after the
shortfall and the live sequences. Under `round_robin` a replica sees all N
prefixes; under affinity it sees N/R. So the same pool holds twice the working
set, and what that is worth depends on where N falls:

| Prefixes N | Retained, rr / prefix | `h` rr | `h` prefix | Seats recovered |
|---|---|---|---|---|
| 32 | 32 / 16 | 0.800 | 0.800 | 24.4 |
| 64 | 64 / 32 | 0.800 | 0.800 | 48.8 |
| 128 | 116 / 64 | 0.728 | 0.800 | 79.9 |
| 256 | 116 / 116 | 0.364 | 0.728 | 0.0 |
| 512 | 116 / 116 | 0.182 | 0.364 | 0.0 |

The router goes in front of the pair, and the arm is the flag on it:

```
docker build -t prefix-router:dev router      # on the droplet: it is amd64
docker run -d --name router --network host prefix-router:dev \
  -listen :8080 -upstreams http://127.0.0.1:8000,http://127.0.0.1:8001 \
  -policy prefix -dial-timeout 250ms
```

```
python3 bench/harness.py --scenario router-arm --accelerator $FIT \
  --port 8080 --expect-policy prefix \
  --metrics-endpoint 127.0.0.1:8000 --metrics-endpoint 127.0.0.1:8001 \
  --startup-log /workspace/run3/engine-8000.log \
  --out /workspace/run3/results/prefix
```

Then `docker rm -f router`, relaunch it with `-policy round_robin`, and send the
same scenario with `--expect-policy round_robin` to `--out …/round-robin`. The load goes to `:8080` both times and
the counters are read from the engines directly — which is the point of
`--metrics-endpoint`, since `/metrics` through the router is an unkeyed path and
therefore one replica chosen by the fallback.

**Two regimes, and locating the boundary is the point.** While both policies
retain everything, affinity's saving is *space* — it buys seats, at 0.76 each
per prefix, and it has to clear block A's 27-seat bill before the arrangement is
worth anything at all: **below N ≈ 35, two engines on one card are a loss no
routing policy recovers.** Once `round_robin` is evicting, both policies hold the
same 116 prefixes, there is no space left to differ over, and the whole difference
moves into `h`.

**The order the prefixes are asked for is a decision, and it is made here.** The
`h` columns above assume any prefix is as likely to be asked for next as any
other. The harness draws them in strict rotation, which is LRU's worst case: a
prefix comes round again only after every other one has evicted it, so the hit
rate does not fall to a share, it falls to **zero** — `round_robin` past N = 116
and affinity past N = 233. Both cliffs assume vLLM evicts least-recently-used
blocks, which this repository has *not* read at the pinned tag
(`vllm/v1/core/block_pool.py`): it is the one input to this block that is
neither measured nor derived, and reading it costs nothing and no credits. That is the grid this run sends, unchanged, and it
means the measured spread between the arms is the **friendliest case for the
router**, not a general figure. Real traffic is neither: it is Zipf-shaped, and
sits between the two models. Both predictions are written down; the run faces the
rotation one.

**Alternate the arm order across the working sets** — prefix first at 32, 128 and
512, `round_robin` first at 64 and 256. Each level warms its own prefixes, but
the arm that runs second starts against a cache the first arm shaped, and
alternating is what keeps that bias from lining up with the policy.

**Read per level:** measured `h` summed over both engines; the
`X-Router-Policy` counts; the split of requests between the two engines. If one
engine takes more than 60 % of a level, bounded loads engaged — affinity was
traded for balance, by design (`router/README.md` §4), and `h` is not the only
thing that moved.

**Stop condition.** If the prefix arm at N = 32 does not measure `h` ≈ 0.8, stop
and fix the instrument: at that working set everything is retained under either
policy and the number is not about the card.

---

## 5 · Block C — what the hit rate buys (no extra card time)

Read off block B's levels; nothing new is sent.

**TTFT.** Prefix caching removes prefill work outright, so the prefill component
of TTFT falls by `(1 − h)` — the one knob that moves TTFT and TPOT the same way
(`docs/SLO.md` §6). At 4 000 tokens a lone prefill takes **256.3 ms** at run 1's
fit and **172.2 ms** at run 2's (table 11). The predicted difference between the
arms is that times `(h_prefix − h_rr)` per request: **62.7–93.3 ms** at N = 256
under the uniform model, **137.8–205.0 ms** at N = 128 under the rotation the run
actually sends. Both are upper bounds: a hit removes the linear part of a
prefill, and the 800 new tokens still attend over the whole prompt. Compared arm against arm at
the same load, never against the floor: queueing sits on top of it and is the
larger term (`docs/SLO.md` §4).

**Seats: not derivable on this card until a fit is registered.**
`INTERFERENCE_FITS` has no MI300X entry on purpose — lending the L40S's line to a
different memory system and a different attention backend would print a seat
count with no run behind any part of it. Run 1 measured the interference and wrote
a model from two rows after the fact (`docs/benchmarks/mi300x-run1.md` §6), and
run 2 faced it within 7 % at run 1's budget and kernel (`mi300x-run2.md` §6).
Whether it is registered is the review's decision. Once a fit is registered:

```
python3 bench/predictions.py --what-if --accelerator $FIT --hit-rate <measured h>
```

once per arm, and the difference between the two is the seat effect of the
routing policy. Those numbers belong in the report, never back into this sheet.

---

## 6 · Budget, drop order, and what "done" means

| | |
|---|---|
| Expected clock | ≈ 2 h — 25 min up, 15 min block A, 50 min block B, the rest harvest and destroy |
| Hard stop | **3 h ≈ $6** of the credit balance |
| Drop order | N = 512 first (it only halves an already-broken hit rate), then N = 32 (it predicts no difference), then N = 64. **Never N = 128 or 256** — those two are what separate the two order models |
| Done | both logged pools read against table 11; block A's four rows faced; the arm pair at N = 128 and N = 256 measured with `h` per engine and the policy header counted |
| Not done | anything where `X-Router-Policy` was not checked. A level without it is a level that may have run the control twice |

Everything lands in `docs/benchmarks/raw/mi300x-<date>/` with its README, read
once for a credential before staging, and the file count checked across
`git add` (`docs/adding-a-run.md` §2).

---

## 7 · Not in this run, and where each goes

- **The real replica question** — weights duplicated across two *cards*, not two
  processes. Needs the 8-card droplet at $15.92/h; its own decision.
- **The Pod watch, and the retry to a different replica** — both priced on
  `kind` on 2026-09-19 and deliberately not taken (`router/README.md` §8). This
  run pins its fleet, so neither is on the path.
- **`-key-bytes`** — 512 bytes is a guess with a shape. Fitting it needs prompts
  that share a prefix of *varying* length, which is a different scenario.
- **Zipf-distributed prefix popularity** — the model between this sheet's two,
  and the one real traffic sits at. A generator change, and worth its own arm
  once the two bounds are measured.
- **FP8 KV** — not scheduled: at a 2 048-token budget neither MI300X run filled
  the pool (`mi300x-run-2.md`, *Not in this run*). It halves `kv_per_token`, which
  divides both limits and cancels out of their ratio (`docs/SLO.md` §6), so it moves every number here
  and none of the conclusions.

---

## 8 · If it goes sideways

| Symptom | First move |
|---|---|
| Both arms measure the same `h` | The key defect is back. `X-Router-Policy` on any response says which of the five ran; `no-prompt` means the body shape, not the router |
| The second engine's pool is far below the first's | The launch race. Kill both, relaunch strictly serially, and re-read checkpoint A — every figure in §3 is against the pair, so one wrong pool invalidates the block |
| OOM at launch | `--gpu-memory-utilization 0.42` each, note it, and re-derive §2 and §3 at the new share before continuing; the seat figures are not comparable across shares |
| One engine takes almost everything | Bounded loads at 1.25, or a ring imbalance. Read the request split before the hit rate: a policy that concentrated the load is not the policy the table predicts |
| `h` on the prefix arm sits below nominal at every N | The warmup is not warm, or the counter window includes it. Run 3 hit exactly this and the fix was the per-level seed offset (`bench/scenarios/prefix_sweep.py`) |
| The router returns 502s | An engine died, and the fleet is a flag: nothing re-reads it. Check both engines before blaming the router (`deploy/router/README.md` §1) |
| The clock passes 2.5 h | Harvest what exists and destroy the droplet. A partial block B with its `h` values is a result; an over-run is a credit the window does not have |
