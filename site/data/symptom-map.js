window.SYMPTOM_MAP = {
 "version": 1,
 "source": "docs/symptom-map.md",
 "inputs": {
  "alert": {
   "type": "enum",
   "values": [
    "none",
    "QueueBeyondTTFTBudget",
    "TTFTBudgetBurning"
   ],
   "panel": "row 1, Alerts"
  },
  "track": {
   "type": "enum",
   "values": [
    "stable",
    "canary"
   ],
   "panel": "the Pod label the alert names"
  },
  "waiting": {
   "type": "number",
   "panel": "row 3, Queue per replica (waiting)",
   "on_kind": true
  },
  "running": {
   "type": "number",
   "panel": "row 3, Seats in use per replica (running)",
   "on_kind": true
  },
  "max_num_seqs": {
   "type": "number",
   "panel": "the engine's flag; vLLM defaults to 256",
   "on_kind": true
  },
  "kv_usage": {
   "type": "number",
   "panel": "row 3, KV cache usage per replica, 0..1",
   "on_kind": false
  },
  "preemptions_per_s": {
   "type": "number",
   "panel": "row 3, Preemptions / s",
   "on_kind": false
  },
  "ttft_p99_ms": {
   "type": "number",
   "panel": "row 2, TTFT p50 / p99",
   "on_kind": false
  },
  "tpot_p99_ms": {
   "type": "number",
   "panel": "row 2, TPOT p50 / p99",
   "on_kind": false
  },
  "median_itl_ms": {
   "type": "number",
   "panel": "bench only: the harness's medITL column",
   "on_kind": false
  },
  "replicas": {
   "type": "number",
   "panel": "row 3, Replicas scraped, per track",
   "on_kind": true
  },
  "max_replicas": {
   "type": "number",
   "panel": "the ScaledObject's maxReplicaCount; 4 here",
   "on_kind": true
  },
  "targets_down": {
   "type": "number",
   "panel": "row 3, Scrape targets down",
   "on_kind": true
  },
  "changed_recently": {
   "type": "bool",
   "panel": "was a config, model or card changed since the last quiet hour",
   "on_kind": true
  },
  "ttft_target_ms": {
   "type": "number",
   "panel": "from the calculator",
   "on_kind": true
  },
  "tpot_target_ms": {
   "type": "number",
   "panel": "from the calculator",
   "on_kind": true
  }
 },
 "derived": {
  "ttft_floor_ms": "the calculator's TTFT floor at the uncached tokens of its prompt",
  "tpot_floor_at_running_ms": "the calculator's decode-step floor at `running` seats",
  "prompt_fits_budget": "ttft_floor at the full prompt <= ttft_target_ms",
  "ttft_gap_ratio": "ttft_p99_ms / ttft_floor_ms",
  "tpot_gap_ratio": "tpot_p99_ms / tpot_floor_at_running_ms",
  "itl_ratio": "tpot_p99_ms / median_itl_ms"
 },
 "symptoms": [
  {
   "id": "change",
   "title": "After any deliberate change",
   "anchor": "#after-any-deliberate-change",
   "active_when": {
    "all": [
     {
      "field": "changed_recently",
      "op": "eq",
      "value": true
     }
    ]
   },
   "requires": []
  },
  {
   "id": "k8s",
   "title": "Deployed, but not serving",
   "anchor": "#deployed-but-not-serving",
   "active_when": {
    "all": [
     {
      "field": "targets_down",
      "op": "gt",
      "value": 0
     }
    ]
   },
   "requires": []
  },
  {
   "id": "ttft",
   "title": "TTFT p99 breaching",
   "anchor": "#ttft-p99-breaching",
   "active_when": {
    "any": [
     {
      "field": "alert",
      "op": "ne",
      "value": "none"
     },
     {
      "field": "ttft_p99_ms",
      "op": "gt",
      "value": {
       "field": "ttft_target_ms"
      }
     }
    ]
   },
   "requires": []
  },
  {
   "id": "tpot",
   "title": "TPOT p99 breaching",
   "anchor": "#tpot-p99-breaching",
   "active_when": {
    "all": [
     {
      "field": "tpot_p99_ms",
      "op": "gt",
      "value": {
       "field": "tpot_target_ms"
      }
     }
    ]
   },
   "requires": [
    "tpot_p99_ms"
   ]
  },
  {
   "id": "oom",
   "title": "OOM / preemptions climbing",
   "anchor": "#oom--preemptions-climbing",
   "active_when": {
    "any": [
     {
      "field": "preemptions_per_s",
      "op": "gt",
      "value": 0
     },
     {
      "field": "kv_usage",
      "op": "gte",
      "value": 0.95
     }
    ]
   },
   "requires": [
    "kv_usage",
    "preemptions_per_s"
   ]
  }
 ],
 "nodes": [
  {
   "id": "change.startup-log.first",
   "symptom": "change",
   "kind": "probe",
   "text": "Re-read the startup log before any latency — a new version changes the arithmetic; a seat count off the prediction is a roll-back with no traffic spent (runbook.md)",
   "evidence": null,
   "source": "runbook.md#canary-rollout-of-a-model-version",
   "requires": []
  },
  {
   "id": "change.gate.band",
   "symptom": "change",
   "kind": "continuation",
   "text": "…but do not gate on it too tightly: the same config relaunched moved the logged pool 4.2 %. Gate on `kv cache memory in use`, or ±5 % (run 2 §6)",
   "evidence": null,
   "source": "benchmarks/l40s-run2.md#6-what-the-run-measured-about-its-own-instrument",
   "parent": "change.startup-log.first",
   "requires": []
  },
  {
   "id": "change.canary.first",
   "symptom": "change",
   "kind": "branch",
   "text": "Canary before fleet: the rule names the track; roll-back is one delete and fails what the canary holds — do it at `pending`. **run once** (runbook.md)",
   "evidence": "run once",
   "source": "runbook.md#roll-back",
   "when": {
    "all": []
   },
   "knobs": [
    {
     "name": "canary weight",
     "source": "runbook.md#canary-rollout-of-a-model-version"
    }
   ],
   "next_number": "the logged KV pool against the prediction, at pending, before any traffic",
   "requires": []
  },
  {
   "id": "change.batch.filled",
   "symptom": "change",
   "kind": "trap",
   "text": "Confirm the batch actually filled before believing the row: `num_requests_running` sampled *during* the level, never inferred from latency (run 1 §2)",
   "evidence": null,
   "source": "benchmarks/l40s-baseline.md#2-checkpoint-a--the-kv-pool-measured-before-any-load",
   "requires": []
  },
  {
   "id": "k8s.service.endpointslice",
   "symptom": "k8s",
   "kind": "probe",
   "text": "Service resolves but nothing answers → the selector matched no labels. Read the **EndpointSlice**, not the Service: an empty one is the only place the mismatch shows (GLOSSARY.md: EndpointSlice)",
   "evidence": null,
   "source": "GLOSSARY.md",
   "requires": []
  },
  {
   "id": "k8s.edge.status-codes",
   "symptom": "k8s",
   "kind": "branch",
   "text": "503 from the edge in the seconds after a deploy or a scale-out → the route exists and its endpoint set is empty. Distinguish by code: 503 no live endpoint, 404 no route at all, **empty reply** (curl exit 52) no controller on the node whose ports are mapped. **measured** (running on kind)",
   "evidence": "measured",
   "source": "running-on-kind.md",
   "when": {
    "all": []
   },
   "knobs": [],
   "next_number": "the edge's status code: 503, 404 or an empty reply",
   "requires": []
  },
  {
   "id": "k8s.restart-loop.startup-probe",
   "symptom": "k8s",
   "kind": "trap",
   "text": "Container restart-looping while the weights load → readiness/liveness firing before the engine exists. `startupProbe` suspends both until it passes; liveness slower than readiness, so a wedged engine leaves rotation before it is killed (GLOSSARY.md: startupProbe)",
   "evidence": null,
   "source": "GLOSSARY.md",
   "requires": []
  },
  {
   "id": "ttft.first.gap",
   "symptom": "ttft",
   "kind": "probe",
   "text": "First number: the gap above the TTFT floor. The floor is prefill compute; everything above it is queue + overhead (SLO.md §4)",
   "evidence": null,
   "source": "SLO.md#4-floors",
   "requires": []
  },
  {
   "id": "ttft.canary.rollback",
   "symptom": "ttft",
   "kind": "branch",
   "text": "The alert fired on a canary Pod → roll back first, diagnose second: the rule names the track, and one delete fails what the canary holds (runbook.md)",
   "evidence": null,
   "source": "runbook.md#roll-back",
   "when": {
    "all": [
     {
      "field": "alert",
      "op": "ne",
      "value": "none"
     },
     {
      "field": "track",
      "op": "eq",
      "value": "canary"
     }
    ]
   },
   "knobs": [
    {
     "name": "roll back",
     "source": "runbook.md#roll-back"
    }
   ],
   "next_number": "nothing before the roll-back; the canary's pool figure afterwards",
   "requires": []
  },
  {
   "id": "ttft.queue.replicas",
   "symptom": "ttft",
   "kind": "branch",
   "text": "Gap is queue (`num_requests_waiting` growing) → a provisioning problem, not a tuning one: replicas (KEDA). Mind the scale-up delay — the pod's cold start sits inside it, 69 s **measured**, and it is the one term removable rather than tunable (architecture.md §4)",
   "evidence": "measured",
   "source": "architecture.md#4-the-control-loop-and-why-it-is-minutes",
   "when": {
    "all": [
     {
      "field": "waiting",
      "op": "gt",
      "value": 0
     }
    ]
   },
   "knobs": [
    {
     "name": "replicas (KEDA)",
     "source": "../deploy/keda/README.md"
    }
   ],
   "next_number": "replicas against maxReplicaCount: at the cap this is capacity, not the scaler",
   "requires": []
  },
  {
   "id": "ttft.queue.loop",
   "symptom": "ttft",
   "kind": "continuation",
   "text": "…and the replica will not arrive in time: the loop is scrape + HPA sync + weight load, minutes. A burst shorter than that is answered only by seats that already exist (SLO.md §4)",
   "evidence": null,
   "source": "SLO.md#what-the-queue-budget-makes-the-threshold--and-what-it-does-not",
   "parent": "ttft.queue.replicas",
   "requires": []
  },
  {
   "id": "ttft.queue.alert",
   "symptom": "ttft",
   "kind": "continuation",
   "text": "The alert for it: a queue standing on *one* replica for longer than that loop — `for: 2m` clears scrape + HPA + a 69 s cold start — `pending` → `firing` → resolved in one evaluation each side. The histogram SLI at the 0.25 s edge is loaded and waits for a card. **watched firing** (observability README)",
   "evidence": "watched firing",
   "source": "../deploy/observability/README.md#observed-on-kind-2026-09-07",
   "parent": "ttft.queue.replicas",
   "requires": []
  },
  {
   "id": "ttft.prefill.budget",
   "symptom": "ttft",
   "kind": "branch",
   "text": "Before any of that: does the prompt fit the budget at all? 3 516 tokens spends all 300 ms on an L40S with zero queue (run 1 §6)",
   "evidence": null,
   "source": "benchmarks/l40s-baseline.md#6-what-ships",
   "when": {
    "all": [
     {
      "field": "waiting",
      "op": "eq",
      "value": 0
     },
     {
      "field": "prompt_fits_budget",
      "op": "eq",
      "value": false
     }
    ]
   },
   "knobs": [
    {
     "name": "prompt cap",
     "source": "SLO.md#4-floors"
    },
    {
     "name": "prefix caching",
     "source": "SLO.md#the-knob-that-could-close-the-rest-prefix-caching"
    }
   ],
   "next_number": "the TTFT floor at the full prompt against the target: over it, no queue fix exists",
   "requires": []
  },
  {
   "id": "ttft.prefill.prompts",
   "symptom": "ttft",
   "kind": "branch",
   "text": "Gap is prefill itself (prompts grew) → TTFT scales linearly in prompt length, **measured** → `chunked prefill` (buys everyone's TPOT smoothness for a little TTFT), `prefix caching` where prefixes repeat, or cap the prompt (run 1 §4)",
   "evidence": "measured",
   "source": "benchmarks/l40s-baseline.md#4-calibrating-mfu--prefill",
   "when": {
    "all": [
     {
      "field": "waiting",
      "op": "eq",
      "value": 0
     },
     {
      "field": "ttft_gap_ratio",
      "op": "lt",
      "value": 1.5
     }
    ]
   },
   "requires": [
    "ttft_p99_ms"
   ],
   "knobs": [
    {
     "name": "chunked prefill",
     "source": "GLOSSARY.md"
    },
    {
     "name": "prefix caching",
     "source": "SLO.md#the-knob-that-could-close-the-rest-prefix-caching"
    },
    {
     "name": "prompt cap",
     "source": "SLO.md#4-floors"
    }
   ],
   "next_number": "prompt length over time, and h: the floor moves with prompt x (1 - h)"
  },
  {
   "id": "ttft.prefill.chunk-size",
   "symptom": "ttft",
   "kind": "branch",
   "text": "Chunk size is a TTFT knob only under load: at c = 13 the four chunk sizes span 69 % of TTFT p50 with no ordering, at c = 32 a 512-token chunk costs 6× (run 2 §4)",
   "evidence": null,
   "source": "benchmarks/l40s-run2.md#4-block-b--max_num_batched_tokens-buys-two-seats",
   "when": {
    "all": [
     {
      "field": "waiting",
      "op": "eq",
      "value": 0
     },
     {
      "field": "ttft_gap_ratio",
      "op": "gte",
      "value": 1.5
     },
     {
      "field": "running",
      "op": "gt",
      "value": 0
     }
    ]
   },
   "requires": [
    "ttft_p99_ms"
   ],
   "knobs": [
    {
     "name": "max_num_batched_tokens",
     "source": "benchmarks/l40s-run2.md#4-block-b--max_num_batched_tokens-buys-two-seats"
    }
   ],
   "next_number": "running against max_num_seqs: the prefill is waiting behind other requests' chunks"
  },
  {
   "id": "ttft.tail.first",
   "symptom": "ttft",
   "kind": "continuation",
   "text": "The tail goes first, and early: under Poisson arrivals TTFT p99 crossed 300 ms between 0.5 and 1.0 req/s on a card whose median was still 181 ms. **measured** (run 2 §3)",
   "evidence": "measured",
   "source": "benchmarks/l40s-run2.md#3-block-a--the-open-loop-and-the-divergence-that-is-the-whole-point",
   "parent": "ttft.prefill.chunk-size",
   "requires": []
  },
  {
   "id": "ttft.trap.below-floor",
   "symptom": "ttft",
   "kind": "trap",
   "text": "Trap: TTFT *below* its floor is not a fast card — it is prefix caching, and the floor to compare against is the one at the tokens left uncached, prompt × (1 − `h`). Coded as a gate; no level has breached it (bench/harness.py)",
   "evidence": null,
   "source": "../bench/harness.py",
   "when": {
    "all": [
     {
      "field": "ttft_gap_ratio",
      "op": "lt",
      "value": 1.0
     }
    ]
   },
   "requires": [
    "ttft_p99_ms"
   ]
  },
  {
   "id": "ttft.trap.closed-loop",
   "symptom": "ttft",
   "kind": "trap",
   "text": "Trap: the number came from a closed-loop bench — then it is the generator's queue, not the service's, and no knob applies (GLOSSARY.md: closed-loop load)",
   "evidence": null,
   "source": "GLOSSARY.md",
   "requires": []
  },
  {
   "id": "ttft.trap.calm-gauge",
   "symptom": "ttft",
   "kind": "trap",
   "text": "Trap: the calm queue gauge — a mean cannot clear a tail; read the p99, not the gauge (SLO.md §2)",
   "evidence": null,
   "source": "SLO.md#2-targets",
   "when": {
    "all": [
     {
      "field": "waiting",
      "op": "eq",
      "value": 0
     },
     {
      "field": "alert",
      "op": "eq",
      "value": "TTFTBudgetBurning"
     }
    ]
   },
   "requires": []
  },
  {
   "id": "ttft.trap.preemption-cascade",
   "symptom": "ttft",
   "kind": "trap",
   "text": "Trap: the preemption cascade — KV full → evictions → re-prefill → the TTFT tail explodes while TPOT looks merely warm. **seen**: c = 45 on an L40S ran 41, waited 39, cache 0.999, 4 preemptions, TTFT p99 23 955 ms (run 1 §2)",
   "evidence": "seen",
   "source": "benchmarks/l40s-baseline.md#2-checkpoint-a--the-kv-pool-measured-before-any-load",
   "when": {
    "all": [
     {
      "field": "preemptions_per_s",
      "op": "gt",
      "value": 0
     },
     {
      "field": "kv_usage",
      "op": "gte",
      "value": 0.9
     }
    ]
   },
   "requires": [
    "preemptions_per_s",
    "kv_usage"
   ]
  },
  {
   "id": "tpot.first.floor-at-op",
   "symptom": "tpot",
   "kind": "probe",
   "text": "First number: the floor **at the operating point**, not the batch-1 floor — a floor within a few percent of the observed step means there is nothing left to tune, and run 1's fitted step sits 4.5 % from its floor (run 1 §3)",
   "evidence": null,
   "source": "benchmarks/l40s-baseline.md#3-calibrating-achieved_bandwidth--the-decode-step",
   "requires": []
  },
  {
   "id": "tpot.second.median-itl",
   "symptom": "tpot",
   "kind": "probe",
   "text": "Second number: median ITL beside TPOT. Equal → it really is the decode step; TPOT far above → prefill is being injected into it, and the knob is `max_num_batched_tokens` (run 1 §5)",
   "evidence": null,
   "source": "benchmarks/l40s-baseline.md#5-the-50-ms-line--three-different-answers-and-only-one-of-them-ships",
   "requires": []
  },
  {
   "id": "tpot.cache.max-num-seqs-binds",
   "symptom": "tpot",
   "kind": "branch",
   "text": "…and once the pool stops binding, the ceiling is `max_num_seqs` = 256: zero preemptions at 13 req/s, 65 queued behind the sequence limit (run 3 §5) — on the MI300X it was predicted to bind *before* the pool at c = 288, and never did: `running` stopped at ~100, the token-budget ceiling below (MI300X run 1 §5)",
   "evidence": null,
   "source": "benchmarks/mi300x-run1.md#5-where-the-running-count-settles-and-where-throughput-does",
   "when": {
    "all": [
     {
      "field": "running",
      "op": "gte",
      "value": {
       "field": "max_num_seqs"
      }
     },
     {
      "field": "waiting",
      "op": "gt",
      "value": 0
     }
    ]
   },
   "knobs": [
    {
     "name": "max_num_seqs",
     "source": "SLO.md#6-concurrency-ceiling"
    }
   ],
   "next_number": "which limit set max_num_seqs on this card -- capacity or latency -- from the calculator",
   "requires": []
  },
  {
   "id": "tpot.physics.fewer-bytes",
   "symptom": "tpot",
   "kind": "branch",
   "text": "Floor ≈ measurement → physics: fewer sequences (`max_num_seqs` down), fewer bytes (FP8 KV), or a faster memory bus — each with its cost line (SLO.md §5–6)",
   "evidence": null,
   "source": "SLO.md#6-concurrency-ceiling",
   "when": {
    "all": [
     {
      "field": "tpot_gap_ratio",
      "op": "lt",
      "value": 1.1
     }
    ]
   },
   "knobs": [
    {
     "name": "max_num_seqs",
     "source": "SLO.md#6-concurrency-ceiling"
    },
    {
     "name": "FP8 KV cache",
     "source": "SLO.md#highest-leverage-knob-fp8-kv-cache"
    }
   ],
   "next_number": "the decode step at the seat count you want, against the target: the floor is the answer",
   "requires": []
  },
  {
   "id": "tpot.overhead.interference",
   "symptom": "tpot",
   "kind": "branch",
   "text": "Floor ≪ measurement → overhead or misconfiguration: scheduler pressure, prefill interference — `chunked prefill` is the knob that trades it (GLOSSARY.md: prefill interference)",
   "evidence": null,
   "source": "GLOSSARY.md",
   "when": {
    "all": [
     {
      "field": "tpot_gap_ratio",
      "op": "gte",
      "value": 1.1
     }
    ]
   },
   "knobs": [
    {
     "name": "chunked prefill",
     "source": "GLOSSARY.md"
    },
    {
     "name": "prefix caching",
     "source": "SLO.md#the-knob-that-could-close-the-rest-prefix-caching"
    }
   ],
   "next_number": "TPOT against median ITL: the gap between them is the interference",
   "requires": []
  },
  {
   "id": "tpot.priced.mnbt",
   "symptom": "tpot",
   "kind": "continuation",
   "text": "And it is priced: `max_num_batched_tokens` 2 048 → 512 buys 2 seats of the 19 missing (12 → 14) and costs 6× TTFT p50 at c = 32. Not the fix. **measured** (run 2 §4)",
   "evidence": "measured",
   "source": "benchmarks/l40s-run2.md#4-block-b--max_num_batched_tokens-buys-two-seats",
   "parent": "tpot.overhead.interference",
   "requires": []
  },
  {
   "id": "tpot.cache.seats",
   "symptom": "tpot",
   "kind": "continuation",
   "text": "The candidate that *removes* the work instead of moving it: prefix caching, worth 25 seats at `h` = 0.8 — 12.5 → 37.8, three times the prediction's 1.8×. **measured** (run 3 §4)",
   "evidence": "measured",
   "source": "benchmarks/l40s-run3.md#4-block-a--three-seats-become-nine-and-channel-1-is-false",
   "parent": "tpot.overhead.interference",
   "requires": []
  },
  {
   "id": "tpot.option.disaggregation",
   "symptom": "tpot",
   "kind": "continuation",
   "text": "…and the third option, separating prefill and decode pools, is not a single-card move: it removes the interference without removing the work, and costs a second accelerator (SLO.md §6)",
   "evidence": null,
   "source": "SLO.md#6-concurrency-ceiling",
   "parent": "tpot.overhead.interference",
   "requires": []
  },
  {
   "id": "tpot.trap.median-itl-contaminated",
   "symptom": "tpot",
   "kind": "trap",
   "text": "Trap: median ITL stops being the decode step once prefill lands in most steps — 79.6 ms against a 49.8 ms step at `max_num_batched_tokens` 512, c = 32. Ratio TPOT / median ITL ≤ 1 is the tell (run 2 §6)",
   "evidence": null,
   "source": "benchmarks/l40s-run2.md#6-what-the-run-measured-about-its-own-instrument",
   "when": {
    "all": [
     {
      "field": "itl_ratio",
      "op": "lte",
      "value": 1.0
     }
    ]
   },
   "requires": [
    "median_itl_ms"
   ]
  },
  {
   "id": "tpot.cause.context-growth",
   "symptom": "tpot",
   "kind": "trap",
   "text": "Cause vs symptom: context growth moves the KV term *within* the same memory-bound regime — decode did not \"become\" memory-bound, it always was (GLOSSARY.md: memory-bound)",
   "evidence": null,
   "source": "GLOSSARY.md",
   "requires": []
  },
  {
   "id": "oom.admission",
   "symptom": "oom",
   "kind": "probe",
   "text": "It is an admission failure, not a crash: the KV cache cannot seat another sequence (GLOSSARY.md: preemption)",
   "evidence": null,
   "source": "GLOSSARY.md",
   "requires": []
  },
  {
   "id": "oom.first.startup-log",
   "symptom": "oom",
   "kind": "probe",
   "text": "First read: the startup log's KV size and block count against the derived ceiling — the log outranks the arithmetic (SLO.md §9)",
   "evidence": null,
   "source": "SLO.md#9-assumptions-and-how-they-get-validated",
   "requires": []
  },
  {
   "id": "oom.levers.cost-order",
   "symptom": "oom",
   "kind": "branch",
   "text": "Real levers, in cost order: shorter context · FP8 KV (~2× the tokens fit) · `max_num_seqs` down · replicas (SLO.md §6)",
   "evidence": null,
   "source": "SLO.md#6-concurrency-ceiling",
   "when": {
    "all": []
   },
   "knobs": [
    {
     "name": "max_model_len",
     "source": "SLO.md#6-concurrency-ceiling"
    },
    {
     "name": "FP8 KV cache",
     "source": "SLO.md#highest-leverage-knob-fp8-kv-cache"
    },
    {
     "name": "max_num_seqs",
     "source": "SLO.md#6-concurrency-ceiling"
    },
    {
     "name": "replicas (KEDA)",
     "source": "../deploy/keda/README.md"
    }
   ],
   "next_number": "the logged KV pool against the derived ceiling, then seats x context against it",
   "requires": []
  },
  {
   "id": "oom.fp8.doubles-both",
   "symptom": "oom",
   "kind": "continuation",
   "text": "FP8 KV buys seats, never a change of regime — it doubles both limits, so whichever one bound still binds. **measured**: pool ×2.000, 41 → 82 seats, latency still binds (run 2 §5)",
   "evidence": "measured",
   "source": "benchmarks/l40s-run2.md#5-block-c--fp8-kv-doubles-the-pool-exactly-and-is-faster-besides",
   "parent": "oom.levers.cost-order",
   "requires": []
  },
  {
   "id": "oom.gmu.rarely",
   "symptom": "oom",
   "kind": "trap",
   "text": "`gpu_memory_utilization` is rarely the lever: derived, on the MI300X the SLO breaks at the same point memory runs out, and measured it broke first, with the pool under 37 % — on the L40S the headroom is real and idle, 41 seats against 12 the SLO permits (run 1 §6)",
   "evidence": null,
   "source": "benchmarks/l40s-baseline.md#6-what-ships",
   "requires": []
  },
  {
   "id": "oom.pool.not-budget-minus-weights",
   "symptom": "oom",
   "kind": "trap",
   "text": "And the pool is not \"budget − weights\": activations and non-torch memory are paid first — a few GiB that moved from 2.88 to 0.77 between two launches of one config, 7 % of an L40S pool and 0.75–2.1 % of an MI300X one. Read the log, never the derivation (SLO.md §9)",
   "evidence": null,
   "source": "SLO.md#9-assumptions-and-how-they-get-validated",
   "requires": []
  },
  {
   "id": "oom.trap.backend-changed",
   "symptom": "oom",
   "kind": "trap",
   "text": "Trap: FP8 KV changed the attention backend on this card (FA2 → FlashInfer). Two changes, one flag — only the startup log says so (run 2 §5)",
   "evidence": null,
   "source": "benchmarks/l40s-run2.md#5-block-c--fp8-kv-doubles-the-pool-exactly-and-is-faster-besides",
   "when": {
    "all": [
     {
      "field": "changed_recently",
      "op": "eq",
      "value": true
     }
    ]
   },
   "requires": []
  }
 ],
 "not_in_rules": [
  "cost-per-1m-tokens-too-high",
  "throughput-healthy-users-angry"
 ]
};
