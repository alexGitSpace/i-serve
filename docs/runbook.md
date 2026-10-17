# Runbook

> Written alongside the alerting rules and the dashboard. Five of
> its six procedures exist and two have been run; the sixth is described by what
> it must contain, so `README.md` does not promise a file that is absent.

Operational procedures. Each one is written to be followed at 03:00 by someone who
did not build the stack, which is the only test of a runbook that matters. The
objects every procedure names are the ones under `deploy/`; the numbers it quotes
are derived in [SLO.md](SLO.md) and are not restated here.

## SLO breach — from an alert to a branch

Two rules read the TTFT objective (`deploy/observability/prometheus/rules.yml`;
which SLI, window and burn rate: [SLO.md](SLO.md) §4). The alert that fired says
which branch this is:

- **`QueueBeyondTTFTBudget` naming a `vllm-canary-…` Pod.** A canary that cannot
  hold its share. Go to *Roll back*, below. Nothing else is wrong.
- **`QueueBeyondTTFTBudget` naming a stable Pod.** A queue the autoscaler did not
  clear. Read the replica count against `maxReplicaCount` first: at the cap this
  is capacity — a pool with no accelerator to land on — and below it the scaler
  itself is the fault. Either way the gap is the *queue* term of §4, and the
  answer is replicas, not configuration.
- **`TTFTBudgetBurning` with `waiting` at zero.** Requests are admitted at once
  and still miss the budget: the gap is the *floor* term of §4, and only a
  configuration or hardware change moves it. The decision tree from here is
  [symptom-map.md](symptom-map.md), *TTFT p99 breaching* — it opens with the
  number to read first, the gap above the floor.

## Canary rollout of a model version

The version is whatever changes the two numbers of [SLO.md](SLO.md) §3 — seats in
the KV pool and the decode step: an image tag, a `--model` id, a flag set. On
`kind` it is the two env values in
`deploy/manifests/overlays/kind-canary/patch-version.yaml`, and the whole
procedure was run there once (`deploy/manifests/README.md`, "Canary, observed on
kind"). The split is by weight at the edge, not by replica count: the canary is a
second Deployment with its own Service behind an Ingress ingress-nginx treats as
a weighted alternative for the same `/v1` rule
(`deploy/manifests/overlays/kind-canary/`).

**Before traffic.**

1. **Capacity.** The canary Pod needs an accelerator the stable track is not
   using. On a fixed pool that means lowering the stable `maxReplicaCount` by one
   for the duration, or the canary sits `Pending` and the rollout tests nothing.
   On `kind` there is nothing to reserve.
2. **Write the version** into `patch-version.yaml` and nothing else in that
   directory. The weight stays at 10.
3. **Apply and wait:**

        kubectl apply -k deploy/manifests/overlays/kind-canary
        kubectl -n llm rollout status deployment/vllm-canary

4. **Read the startup log before reading any latency.** vLLM logs the KV cache
   size and block count it actually got; the stub prints its seats and step. Put
   that number against what §3's arithmetic predicts for this version. A seat
   count that is not the predicted one is a roll-back *now*, with no traffic
   spent — the version is not what it was thought to be.
5. **Confirm it is scraped.** A target with `service="vllm-canary"` must be `up`
   in Prometheus, or the canary will queue in silence and the signal below never
   arrives.

**The signal.** `QueueBeyondTTFTBudget` naming a `vllm-canary-…` Pod. It is the
one rule that exists on `kind`, and on a card it is the faster of the two: a
canary that queues has already missed the budget for every request in that queue.
Watch it for at least the rule's `for` window plus the time a queue takes to form
at the canary's share of the arrival rate — 2.5 min on `kind` at one request per
second; on a card, the burn-rate windows of §4 set the floor at one hour.

**Decide.**

- **Fired, and the Pod is the canary's → Roll back.** Prefer to do it at
  `pending`: the cost of a roll-back is the queue standing on the canary, and
  the queue is deeper at `firing`.
- **Nothing fired after the window → Promote.** In this order, because the
  stable rollout is `strategy: Recreate` and is a gap:
  1. Set the canary weight to 100 — the canary now carries everything on its
     one replica, which is degraded capacity and not downtime.
  2. Write the version into the stable overlay and apply it; wait for
     `rollout status deployment/vllm`. The gap is the old Pod's drain plus the
     new Pod's weight load — 69 s cold on the L40S ([benchmarks/l40s-run3.md](benchmarks/l40s-run3.md) §6) — and the canary is the only replica serving through it.
  3. `kubectl delete -k deploy/manifests/overlays/kind-canary`. The rule
     reverts to the stable backend; the canary's in-flight requests finish
     inside its drain.

  Not yet run; the capacity arithmetic above is derived, not measured.

## Roll back

One command, and it is the same command whether the canary is healthy or not:

    kubectl delete -k deploy/manifests/overlays/kind-canary

What follows, measured once on `kind`: the edge reverts to the stable backend
within seconds — requests drawn for the canary in that window fall through to
stable and succeed; the alert clears when the canary Service leaves discovery,
about half a minute in; the Pod itself takes the readiness withdrawal plus the
drain to exit and **every request still in its queue at that point fails.** A
roll-back does not rescue the requests the canary already holds; it stops it
taking more. That is the argument for rolling back at `pending`.

Rolling back a version that was already *promoted* is a stable rollout in the
other direction — the previous version written back into the stable overlay and
applied — and pays the same `Recreate` gap the promotion did, with no canary to
carry traffic through it unless one is applied first.

## Morning triage — five minutes on one dashboard

The dashboard is `vLLM — SLO` (`deploy/observability/grafana/`; reaching it is
one port-forward, in [running-on-kind.md](running-on-kind.md)). It is read **top to bottom and
never bottom up**, because its three rows are ordered by the question they
answer, and the last row is only a question when the first row says so. Read it
once with the *Track* variable at *All*; if anything is red, once more per track
— a canary and a stable fleet share a dashboard and must not share a diagnosis.

**On `kind` the order inverts, and this is its one exception.** The stub exports
two gauges and no histogram, so row 1's shares and row 2 are *No data* by
construction (`deploy/observability/README.md`). Row 3's entry condition — *read
only when row 1 is not healthy* — therefore cannot be established from the
numbers: only the alert table can still trip it, so with no alert firing a reader
who follows the order above walks away thinking the stack is fine when nothing
about the promise has been measured at all. On `kind`, read **row 3 first** — queue per
replica, seats, replicas scraped — and then row 1's **alert table**, which does
work, because the queue rule stands on a gauge the stub genuinely serves. Five
panels of sixteen light here; that is the fixture, not the procedure. On a card
the order above is the right one.

**Row 1 — is the promise kept.** Two shares, a burn rate and the alert table.
*Healthy:* TTFT and TPOT shares at or above 99 %, burn rate under 1 on the hour,
the table empty. Anything in the table ends triage: go to *SLO breach*, above.
A burn rate between 1 and 14.4 with no alert is budget being spent faster than
it is earned and not yet a page — write it down and let the hour window say by
lunch whether it was a blip. Two things this row cannot say: the TTFT share is
stricter than the SLO by 50 ms, so a share of 98.5 % is not yet a proven
breach of the 300 ms target ([SLO.md](SLO.md) §4); and on `kind` the shares are
*No data* by construction (`deploy/observability/README.md`).

**Row 2 — what the users get, and its price.** Goodput first, tokens per second
last, and the order is the point: a server can look busy while every user is
outside their latency target. *Healthy:* a thin goodput band that tracks the
*completed* line. The band opening — upper far above lower — is the two SLIs
failing on *different* requests, which no server-side number can resolve and
the client-side `--goodput` can; *completed* running well above the band is the
service busy and failing. On `$/1M`, the SLO-respecting line is the figure to
report and the other is a diagnostic; the two diverging is [SLO.md](SLO.md) §7's
first reading happening live. A fall in `h` with nothing deployed is the traffic
changing, not the stack — the same seats serve fewer users at a lower hit rate.

**Row 3 — where the gap is.** Read only when row 1 is not healthy; it decomposes
a breach into the two terms of [SLO.md](SLO.md) §4.

- *Queue on any replica* — the queue term. Replicas scraped at
  `maxReplicaCount`: capacity, a pool with no accelerator to land on. Below it:
  the scaler, or a Pod that never joined — *targets down* names it, and its
  startup log is read before anything else.
- *Queue at zero, seats below the ceiling, KV usage well under 0.9, row 1 red*
  — the floor term. Only a configuration or hardware change moves it; the
  decision tree is [symptom-map.md](symptom-map.md) — *TTFT p99 breaching* if
  row 1's red share is TTFT, *TPOT p99 breaching* if it is TPOT, and each opens
  with the number to read first.
- *KV usage near 1.0 with preemptions* — the pool, not the queue: the engine
  admitted more than it could hold ([SLO.md](SLO.md) §6). The OOM procedure
  below is the continuation, when written.
- *Seats flat at the ceiling with a queue beside them* — the seat count is
  binding; whether that is capacity or the latency limit is a property of the
  card ([SLO.md](SLO.md) §6), and the answer is in the startup log's pool size,
  not on this dashboard.

What was seen when this was run on `kind`, with the breach the alerting rules
were tested on: `deploy/observability/README.md`, "Observed on kind,
2026-09-07".

## Two engines on one card

Two engine processes share one MI300X whenever a card hosts two replicas. At
the default, the process that opened the device first serves each token about
twice as slow as the other at an equal share, and no engine flag or CPU
placement reaches it ([benchmarks/mi300x-run4.md](benchmarks/mi300x-run4.md) §3).

1. **Set `GPU_MAX_HW_QUEUES=1` in each engine's environment**, before it
   starts: HIP reads it when the process opens the device. In a Deployment it
   is an `env` entry on the engine container.
2. **Confirm it arrived** in every process of the engine, the spawned
   EngineCore included: `tr '\0' '\n' < /proc/<pid>/environ | grep
   '^GPU_MAX_HW_QUEUES='`. Print the one variable and never the whole environ,
   which carries the HF token. On the host, root reads each process's compute
   queues in `/sys/kernel/debug/kfd/mqds`: two per process at 1, five at the
   default.
3. **Read both engines' median ITL under an equal load.** Equal within a few
   per cent is the pass: measured, 22.21 against 22.19 ms
   ([benchmarks/mi300x-run5.md](benchmarks/mi300x-run5.md) §3). Unequal with
   the variable in place is outside what has been measured. Go to
   [symptom-map.md](symptom-map.md), *TPOT p99 breaching*.

**Not `HSA_CU_MASK`.** Splitting the CUs between the two engines was measured
and not taken. It removes about half of the decode-step asymmetry, slows each
engine's prefill, and costs the pair a fifth of its throughput (the same
report, §3).

Measured once on a droplet, with the pair launched in one order; not run on
`kind`, where the stub opens no device. What one queue costs an engine alone on
the card is not measured.

## Not yet written

**OOM.** What vLLM does when the KV cache cannot admit a sequence, why
`gpu_memory_utilization` is not the lever it appears to be ([SLO.md](SLO.md) §6),
and what to check first — the KV cache size vLLM logs at startup. Waits on a
card. The branch exists — [symptom-map.md](symptom-map.md), *OOM / preemptions
climbing* — the procedure does not.
