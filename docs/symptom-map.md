# Symptom map — what a breaching metric means before any knob is touched

An operator's decision tree: seven symptoms, and under each the number to read
first, the branches it splits into, and the traps that make a right number read
wrong. One line per node. Every node ends with a pointer to its home — the
section of [SLO.md](SLO.md), a benchmark report, or a `deploy/` README where
the number was derived or measured — and the line itself only reminds. A bold
word on a node is its evidence: **measured** on a rented card, **seen** once in
a run, **watched** or **watched firing** on `kind`, **checked** against the live
objects, **run once** as a procedure; a node with no bold word is derived and
waits for a run. The trailing comment on each line is its stable id, used by
[symptom-map.json](symptom-map.json), which the advisor on the site walks as
rules over the dashboard's readings.

**Before every branch: compute the limit before concluding.** Every time this
map was walked against a real log, the failure was the same — a number named
without the division that produces it, `max_num_seqs` 256 on a card whose
logged pool seats 41. Divide first.

The runbook hands off here twice — from an alert ([runbook.md](runbook.md#slo-breach--from-an-alert-to-a-branch))
and from row 3 of the morning dashboard ([runbook.md](runbook.md#morning-triage--five-minutes-on-one-dashboard)).
Both arrive at the first two symptoms.

---

## TTFT p99 breaching

- First number: the gap above the TTFT floor. The floor is prefill compute; everything above it is queue + overhead ([SLO.md §4](SLO.md#4-floors)) <!-- ttft.first.gap -->
  - The alert fired on a canary Pod → roll back first, diagnose second: the rule names the track, and one delete fails what the canary holds ([runbook.md](runbook.md#roll-back)) <!-- ttft.canary.rollback -->
  - Gap is queue (`num_requests_waiting` growing) → a provisioning problem, not a tuning one: replicas (KEDA). Mind the scale-up delay — the pod's cold start sits inside it, 69 s **measured**, and it is the one term removable rather than tunable ([architecture.md §4](architecture.md#4-the-control-loop-and-why-it-is-minutes)) <!-- ttft.queue.replicas -->
    - …and the replica will not arrive in time: the loop is scrape + HPA sync + weight load, minutes. A burst shorter than that is answered only by seats that already exist ([SLO.md §4](SLO.md#what-the-queue-budget-makes-the-threshold--and-what-it-does-not)) <!-- ttft.queue.loop -->
    - The alert for it: a queue standing on *one* replica for longer than that loop — `for: 2m` clears scrape + HPA + a 69 s cold start — `pending` → `firing` → resolved in one evaluation each side. The histogram SLI at the 0.25 s edge is loaded and waits for a card. **watched firing** ([observability README](../deploy/observability/README.md#observed-on-kind-2026-09-07)) <!-- ttft.queue.alert -->
  - Gap is prefill itself (prompts grew) → TTFT scales linearly in prompt length, **measured** → `chunked prefill` (buys everyone's TPOT smoothness for a little TTFT), `prefix caching` where prefixes repeat, or cap the prompt ([run 1 §4](benchmarks/l40s-baseline.md#4-calibrating-mfu--prefill)) <!-- ttft.prefill.prompts -->
    - Before any of that: does the prompt fit the budget at all? 3 516 tokens spends all 300 ms on an L40S with zero queue ([run 1 §6](benchmarks/l40s-baseline.md#6-what-ships)) <!-- ttft.prefill.budget -->
    - Chunk size is a TTFT knob only under load: at c = 13 the four chunk sizes span 69 % of TTFT p50 with no ordering, at c = 32 a 512-token chunk costs 6× ([run 2 §4](benchmarks/l40s-run2.md#4-block-b--max_num_batched_tokens-buys-two-seats)) <!-- ttft.prefill.chunk-size -->
  - The tail goes first, and early: under Poisson arrivals TTFT p99 crossed 300 ms between 0.5 and 1.0 req/s on a card whose median was still 181 ms. **measured** ([run 2 §3](benchmarks/l40s-run2.md#3-block-a--the-open-loop-and-the-divergence-that-is-the-whole-point)) <!-- ttft.tail.first -->
  - Trap: TTFT *below* its floor is not a fast card — it is prefix caching, and the floor to compare against is the one at the tokens left uncached, prompt × (1 − `h`). Coded as a gate; no level has breached it ([bench/harness.py](../bench/harness.py)) <!-- ttft.trap.below-floor -->
  - Trap: the number came from a closed-loop bench — then it is the generator's queue, not the service's, and no knob applies ([GLOSSARY.md: closed-loop load](GLOSSARY.md)) <!-- ttft.trap.closed-loop -->
- Trap: the calm queue gauge — a mean cannot clear a tail; read the p99, not the gauge ([SLO.md §2](SLO.md#2-targets)) <!-- ttft.trap.calm-gauge -->
- Trap: the preemption cascade — KV full → evictions → re-prefill → the TTFT tail explodes while TPOT looks merely warm. **seen**: c = 45 on an L40S ran 41, waited 39, cache 0.999, 4 preemptions, TTFT p99 23 955 ms ([run 1 §2](benchmarks/l40s-baseline.md#2-checkpoint-a--the-kv-pool-measured-before-any-load)) <!-- ttft.trap.preemption-cascade -->

## TPOT p99 breaching

- First number: the floor **at the operating point**, not the batch-1 floor — a floor within a few percent of the observed step means there is nothing left to tune, and run 1's fitted step sits 4.5 % from its floor ([run 1 §3](benchmarks/l40s-baseline.md#3-calibrating-achieved_bandwidth--the-decode-step)) <!-- tpot.first.floor-at-op -->
- Second number: median ITL beside TPOT. Equal → it really is the decode step; TPOT far above → prefill is being injected into it, and the knob is `max_num_batched_tokens` ([run 1 §5](benchmarks/l40s-baseline.md#5-the-50-ms-line--three-different-answers-and-only-one-of-them-ships)) <!-- tpot.second.median-itl -->
- Floor ≈ measurement → physics: fewer sequences (`max_num_seqs` down), fewer bytes (FP8 KV), or a faster memory bus — each with its cost line ([SLO.md §5–6](SLO.md#6-concurrency-ceiling)) <!-- tpot.physics.fewer-bytes -->
- Floor ≪ measurement → overhead or misconfiguration: scheduler pressure, prefill interference — `chunked prefill` is the knob that trades it ([GLOSSARY.md: prefill interference](GLOSSARY.md)) <!-- tpot.overhead.interference -->
  - And it is priced: `max_num_batched_tokens` 2 048 → 512 buys 2 seats of the 19 missing (12 → 14) and costs 6× TTFT p50 at c = 32. Not the fix. **measured** ([run 2 §4](benchmarks/l40s-run2.md#4-block-b--max_num_batched_tokens-buys-two-seats)) <!-- tpot.priced.mnbt -->
  - What the chunk *does* set exactly: the worst step = decode step + chunk compute, an upper bound high by only 5–10 % across 512…4 096 ([run 2 §4](benchmarks/l40s-run2.md#4-block-b--max_num_batched_tokens-buys-two-seats)) <!-- tpot.chunk.worst-step -->
  - The candidate that *removes* the work instead of moving it: prefix caching, worth 25 seats at `h` = 0.8 — 12.5 → 37.8, three times the prediction's 1.8×. **measured** ([run 3 §4](benchmarks/l40s-run3.md#4-block-a--three-seats-become-nine-and-channel-1-is-false)) <!-- tpot.cache.seats -->
    - …but only p50: ITL p99 stays at ~199 ms in both regimes, because the worst step is the chunk budget (2 048 tok = 179 ms), not the prompt. Mean and tail are different knobs ([run 3 §4](benchmarks/l40s-run3.md#4-block-a--three-seats-become-nine-and-channel-1-is-false)) <!-- tpot.cache.p50-only -->
    - …and once the pool stops binding, the ceiling is `max_num_seqs` = 256: zero preemptions at 13 req/s, 65 queued behind the sequence limit ([run 3 §5](benchmarks/l40s-run3.md#5-block-b--the-peak-moves-35-and-max_num_seqs-is-what-finally-binds)) — on the MI300X it was predicted to bind *before* the pool at c = 288, and never did: `running` stopped at ~100, the token-budget ceiling below ([MI300X run 1 §5](benchmarks/mi300x-run1.md#5-where-the-running-count-settles-and-where-throughput-does)) <!-- tpot.cache.max-num-seqs-binds -->
  - …and once every step is a full budget, more concurrency buys only queue: `running` settles near `max_num_batched_tokens` × output / (input + output) — 98–99 against 97.5 at 4 000 / 200 on the MI300X, pool 36 % full, zero preemptions, median ITL flat at the full-budget step. The count is **measured**; the formula, derived after that run, was faced by MI300X run 2 at a budget of 4 096 — 196 running against 195 ([MI300X run 1 §5](benchmarks/mi300x-run1.md#5-where-the-running-count-settles-and-where-throughput-does), [run 2 §5](benchmarks/mi300x-run2.md#5-what-the-budget-moved)) <!-- tpot.overhead.token-budget-ceiling -->
  - …and the third option, separating prefill and decode pools, is not a single-card move: it removes the interference without removing the work, and costs a second accelerator ([SLO.md §6](SLO.md#6-concurrency-ceiling)) <!-- tpot.option.disaggregation -->
- Trap: median ITL stops being the decode step once prefill lands in most steps — 79.6 ms against a 49.8 ms step at `max_num_batched_tokens` 512, c = 32. Ratio TPOT / median ITL ≤ 1 is the tell ([run 2 §6](benchmarks/l40s-run2.md#6-what-the-run-measured-about-its-own-instrument)) <!-- tpot.trap.median-itl-contaminated -->
- Cause vs symptom: context growth moves the KV term *within* the same memory-bound regime — decode did not "become" memory-bound, it always was ([GLOSSARY.md: memory-bound](GLOSSARY.md)) <!-- tpot.cause.context-growth -->

## OOM / preemptions climbing

- It is an admission failure, not a crash: the KV cache cannot seat another sequence ([GLOSSARY.md: preemption](GLOSSARY.md)) <!-- oom.admission -->
- First read: the startup log's KV size and block count against the derived ceiling — the log outranks the arithmetic ([SLO.md §9](SLO.md#9-assumptions-and-how-they-get-validated)) <!-- oom.first.startup-log -->
- `gpu_memory_utilization` is rarely the lever: derived, on the MI300X the SLO breaks at the same point memory runs out, and measured it broke first, with the pool under 37 % — on the L40S the headroom is real and idle, 41 seats against 12 the SLO permits ([run 1 §6](benchmarks/l40s-baseline.md#6-what-ships)) <!-- oom.gmu.rarely -->
- And the pool is not "budget − weights": activations and non-torch memory are paid first — a few GiB that moved from 2.88 to 0.77 between two launches of one config, 7 % of an L40S pool and 0.75–2.1 % of an MI300X one. Read the log, never the derivation ([SLO.md §9](SLO.md#9-assumptions-and-how-they-get-validated)) <!-- oom.pool.not-budget-minus-weights -->
- Real levers, in cost order: shorter context · FP8 KV (~2× the tokens fit) · `max_num_seqs` down · replicas ([SLO.md §6](SLO.md#6-concurrency-ceiling)) <!-- oom.levers.cost-order -->
- FP8 KV buys seats, never a change of regime — it doubles both limits, so whichever one bound still binds. **measured**: pool ×2.000, 41 → 82 seats, latency still binds ([run 2 §5](benchmarks/l40s-run2.md#5-block-c--fp8-kv-doubles-the-pool-exactly-and-is-faster-besides)) <!-- oom.fp8.doubles-both -->
  - …and it is faster at equal n besides, −14 % at 13 and −25 % at 32, because it halves the KV half of `bytes_moved` ([run 2 §5](benchmarks/l40s-run2.md#5-block-c--fp8-kv-doubles-the-pool-exactly-and-is-faster-besides)) <!-- oom.fp8.faster -->
- Trap: FP8 KV changed the attention backend on this card (FA2 → FlashInfer). Two changes, one flag — only the startup log says so ([run 2 §5](benchmarks/l40s-run2.md#5-block-c--fp8-kv-doubles-the-pool-exactly-and-is-faster-besides)) <!-- oom.trap.backend-changed -->

## Cost per 1M tokens too high

- Aggregate tok/s is the denominator: a card far below its latency-permitted batch is buying idle bandwidth ([SLO.md §7](SLO.md#7-cost)) <!-- cost.denominator.aggregate -->
- Same card, two operating points: $0.59 against $7.73 per 1M derived — measured, $0.873 against $6.688, and the cheap one is still the *slower* one per user. **measured** ([run 1 §7](benchmarks/l40s-baseline.md#7-cost-at-099h)) <!-- cost.two-points -->
- First ask which denominator the figure has — all / output / SLO-respecting tokens — and which geometry; two $ numbers divide only when both match. The 21× that did not: total/output at one level, 7.7× between levels ([SLO.md §7](SLO.md#7-cost)) <!-- cost.first.which-denominator -->
- Past the capacity ceiling more sequences buy *negative* throughput: 32 → 45 cost −3.3 tok/s and a 24 s TTFT tail ([run 1 §7](benchmarks/l40s-baseline.md#7-cost-at-099h)) <!-- cost.past-ceiling.negative -->
- Replicas are not free throughput: a second replica reads the weights a second time, so the aggregate grows by less than the replica count and the cost per token rises. Untested here — two cards, open since the first runsheet ([MI300X runsheet](benchmarks/runsheets/mi300x-run-1.md#not-in-this-run-and-where-each-goes)) <!-- cost.replicas.not-free -->
- Reasoning workloads: the invisible half of the stream is billed — split the accounting before trusting any $/1M figure ([SLO.md §8](SLO.md#8-known-gap-reasoning-workloads)) <!-- cost.reasoning.invisible-half -->

## Throughput healthy, users angry

- Goodput is the honest metric: count only requests inside SLO — raw throughput keeps looking fine while every user breaches ([GLOSSARY.md: goodput](GLOSSARY.md)) <!-- goodput.honest-metric -->
- Measured collapse: goodput peaks at 1.87 req/s and is 0.12 by 4.0, while output throughput climbs 472 → 723 tok/s the whole way. **measured** ([run 2 §3](benchmarks/l40s-run2.md#3-block-a--the-open-loop-and-the-divergence-that-is-the-whole-point)) <!-- goodput.measured.collapse -->
- The two cost columns point opposite ways past the peak: $/1M output falls 0.583 → 0.380, $/1M *inside SLO* rises 0.735 → 11.46 ([run 2 §7](benchmarks/l40s-run2.md#7-cost-at-099h--and-the-price-of-a-token-someone-can-use)) <!-- goodput.cost.opposite -->
- Thinking mode: healthy TTFT, minutes of spinner — measure TTFAT (`--reasoning-parser`) ([SLO.md §8](SLO.md#8-known-gap-reasoning-workloads)) <!-- goodput.thinking.ttfat -->

## Deployed, but not serving

The Kubernetes branch; nothing here needs a GPU to rehearse.

- Pod `Pending` and no event about it → an unsatisfiable resource request. Extended resources (`nvidia.com/gpu`) live in `limits` only and are never overcommitted, so a missing device plugin is silence, not an error ([GLOSSARY.md: extended resource](GLOSSARY.md)) <!-- k8s.pending.device-plugin -->
- Container restart-looping while the weights load → readiness/liveness firing before the engine exists. `startupProbe` suspends both until it passes; liveness slower than readiness, so a wedged engine leaves rotation before it is killed ([GLOSSARY.md: startupProbe](GLOSSARY.md)) <!-- k8s.restart-loop.startup-probe -->
- Service resolves but nothing answers → the selector matched no labels. Read the **EndpointSlice**, not the Service: an empty one is the only place the mismatch shows ([GLOSSARY.md: EndpointSlice](GLOSSARY.md)) <!-- k8s.service.endpointslice -->
- Rollout stuck with the new Pod `Pending` on a fixed GPU pool → `RollingUpdate`'s surge Pod is waiting for the accelerator the old Pod still holds. `Recreate` accepts the gap; zero-downtime is a canary with real spare capacity ([runbook.md](runbook.md#canary-rollout-of-a-model-version)) <!-- k8s.rollout.recreate -->
- Rollout takes exactly `terminationGracePeriodSeconds` every time → the process is not handling SIGTERM. PID 1 discards unhandled signals; the drain is readiness-first, `/health` failing before the listener closes ([manifests README](../deploy/manifests/README.md#graceful-shutdown-and-the-bug-that-made-it-visible)) <!-- k8s.termination.sigterm -->
- Replicas all land on one node while others sit empty → a PersistentVolume's node affinity, not the scheduler. RWO is per node; the volume decides placement ([pvc.yaml](../deploy/manifests/base/pvc.yaml)) <!-- k8s.placement.pv-affinity -->
- Pods created and killed with no scaling event and no error anywhere → two writers of `spec.replicas`, a controller re-rendering a Deployment the HPA also scales. Not a conflict, a flap ([ModelWarmup note §4](../controllers/modelwarmup/README.md#4-failure-modes-each-with-the-signal-that-shows-it)) <!-- k8s.flap.two-writers -->
- Replica count oscillates under steady load → the trigger metric is destroyed by the response to it. Widen the scale-down stabilization window; a bigger threshold only moves the oscillation. **watched** ([KEDA README](../deploy/keda/README.md#observed-on-kind--including-the-failure)) <!-- k8s.oscillation.stabilization-window -->
- Stable replicas added under a canary, all idle → the scaler's sum includes the canary's queue. Select the track the ScaledObject scales. **watched** ([KEDA README](../deploy/keda/README.md#observed-beside-a-canary--the-scaler-reading-the-wrong-track)) <!-- k8s.canary.scaler-track -->
- A lone 503 at scale-in → the draining Pod refuses completions while still on the endpoint list, and the edge does not retry 503 by design. Observed, not yet a knob ([manifests README](../deploy/manifests/README.md#graceful-shutdown-and-the-bug-that-made-it-visible)) <!-- k8s.scale-in.503 -->
- A generation dies at a suspiciously round wall-clock time (60 s, 504) → the edge, not the engine. `proxy-read-timeout` is a gap between reads, so it only bites the non-streaming path; derive it from `--max-model-len` × the TPOT target. **measured** ([ingress.yaml](../deploy/manifests/base/ingress.yaml)) <!-- k8s.timeout.proxy-read -->
- 503 from the edge in the seconds after a deploy or a scale-out → the route exists and its endpoint set is empty. Distinguish by code: 503 no live endpoint, 404 no route at all, **empty reply** (curl exit 52) no controller on the node whose ports are mapped. **measured** ([running on kind](running-on-kind.md)) <!-- k8s.edge.status-codes -->
- Hit rate falls as replicas are added → requests are landing away from the replica holding their prefix. The policy is the edge's `round_robin` over Pod addresses, so no Service field reaches it — kube-proxy is not in the path, and prefix-aware routing is a Gateway API step. **checked** ([architecture.md §1](architecture.md#1-the-path)) <!-- k8s.hit-rate.routing -->
  - The decision that node is missing: a consistent-hash ring over the prompt's leading bytes, bounded so one hot prefix cannot pin a replica. Adding one replica then moves 20% of keys where `hash mod n` moves 80%. **measured** ([router](../router/README.md)) <!-- k8s.hit-rate.affinity -->
    - …and it routes on a cluster, on a second host in front of the same Pods, so the two policies can be compared over one fleet. **watched** ([router on kind](../deploy/router/README.md)) <!-- k8s.hit-rate.affinity-on-cluster -->
    - …and what it is worth depends on the *working set*: while every prefix fits on every replica, affinity buys free pool; once the set outgrows a replica's room, round robin makes each one hold all of it and the cost lands on `h` instead. On one card a second engine costs 27 seats before routing returns any, so below ~35 distinct prefixes the fleet is the wrong move. **derived** ([runsheet, MI300X run 3 §4](benchmarks/runsheets/mi300x-run-3.md)) <!-- k8s.hit-rate.working-set -->
    - A new replica takes no traffic at all, or a share of requests 502s forever → the router was told its fleet once, at start-up. Measured on a scale-out: none of 24 requests to the new replica and no error anywhere; on a scale-in: 29% to an address that is gone, permanently. The knob is a Pod watch, not a bigger timeout. **measured** ([router on kind §1](../deploy/router/README.md#1-the-flags-bill-measured-on-kind-2026-09-19)) <!-- k8s.hit-rate.stale-fleet -->
- A request hangs for a suspiciously round 30 s and then 502s → something in the path is dialling an address nothing claims. A removed Pod address blackholes rather than refusing, and Go's default transport spends its full 30 s dial on it — 100× the TTFT budget, to produce an error. Bound the dial; it does not rescue the request, it returns the failure inside the budget. **measured** ([router on kind §2](../deploy/router/README.md#2-what-a-departed-replica-costs-per-request)) <!-- k8s.dial.blackhole -->
- Capacity is occupied and nobody receives the output → an edge timeout shorter than the generation. `running` and the HPA both see useful work; the loss is only visible from the client side, which is the case for the client-side SLO view. **measured** ([manifests README](../deploy/manifests/README.md#the-edge-and-the-two-ways-it-wastes-a-card)) <!-- k8s.timeout.invisible-loss -->

## After any deliberate change

Config, model version, or card.

- Re-read the startup log before any latency — a new version changes the arithmetic; a seat count off the prediction is a roll-back with no traffic spent ([runbook.md](runbook.md#canary-rollout-of-a-model-version)) <!-- change.startup-log.first -->
  - …but do not gate on it too tightly: the same config relaunched moved the logged pool 4.2 %. Gate on `kv cache memory in use`, or ±5 % ([run 2 §6](benchmarks/l40s-run2.md#6-what-the-run-measured-about-its-own-instrument)) <!-- change.gate.band -->
  - …unless the gate must *separate* two hypotheses: the MI300X's two pool figures are 7 % apart, overlap at ±5 % and not at ±3 %; the gap between the bands is a recorded outcome, not a failure ([MI300X runsheet §2](benchmarks/runsheets/mi300x-run-1.md#2--checkpoint-a--the-startup-log-against-the-arithmetic)). The log then landed outside both, on a unit the derivation had wrong — account for it first, and relaunch only if the account finds the configuration at fault ([MI300X run 1 §2](benchmarks/mi300x-run1.md#2-checkpoint-a--the-pool-and-a-unit)) <!-- change.gate.separate-hypotheses -->
- Pick a gate metric by its own reproducibility first: TTFT p50 spans 32 % across identical launches, the decode step 0.24 % ([run 2 §6](benchmarks/l40s-run2.md#6-what-the-run-measured-about-its-own-instrument)) <!-- change.gate.reproducibility -->
- Confirm the batch actually filled before believing the row: `num_requests_running` sampled *during* the level, never inferred from latency ([run 1 §2](benchmarks/l40s-baseline.md#2-checkpoint-a--the-kv-pool-measured-before-any-load)) <!-- change.batch.filled -->
- One predicted-vs-measured row per changed figure, coefficient and card named ([SLO.md §9](SLO.md#the-rules-for-the-table-that-scores-all-of-this)) <!-- change.row.per-figure -->
- Canary before fleet: the rule names the track; roll-back is one delete and fails what the canary holds — do it at `pending`. **run once** ([runbook.md](runbook.md#roll-back)) <!-- change.canary.first -->
