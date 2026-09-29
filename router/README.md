# `prefix-router` — one hop that routes on cache locality

The pull is [`../docs/architecture.md`](../docs/architecture.md) §6: the edge
balances `round_robin` over Pod addresses, and for this traffic that is close to
the worst available policy. The size of the gap is the largest number in this
repository attached to a component that does not exist — run 3 measured **12.5
seats at a prefix-cache hit rate of 0 against 37.8 at 0.8**
([`../docs/SLO.md`](../docs/SLO.md) §6).

This directory is that component: a binary, its tests, and the decisions it had
to make — which is the part worth defending. Since 2026-09-19 it also runs **on
`kind`, in the request path**, behind the edge and in front of the stub Pods —
on one host, `router.localhost`, while the engine's own route keeps
`round_robin` so the two policies can be compared over one set of Pods. What
that took, and what it cost, is [`../deploy/router/`](../deploy/router/);
`architecture.md` §6 records the change to the drawing. The replica set is
still a flag, and §8 is where that is priced.

---

## 1. What it decides

One decision per request, and its name is returned in a header so the decision
is visible from outside without a debugger.

| `X-Router-Policy` | When | What it means for the cache |
|---|---|---|
| `prefix` | a prompt was read from a keyed route | the request goes to the replica that took its prefix last time |
| `round_robin` | `-policy round_robin`: the key was computed and thrown away | the control arm of a measurement. The body is still read, so the two arms differ in the decision and not in the work |
| `no-prompt` | the body parsed but held no prompt this router reads | `round_robin` — the `h` = 0 column |
| `unreadable-body` | the client stopped sending | `round_robin`; the upstream produces the error, not the router |
| `oversized-body` | the body is past `-max-body` | `round_robin`, and the body is streamed rather than held |
| `unkeyed-path` | anything but `POST` to a completions route | `round_robin`; those bodies are never read |

Four of the six mean *this request will not find a warm prefix*, and they are
not interchangeable: `round_robin` is a policy that was chosen, the other three
are a router that could not route. An operator who cannot tell them apart cannot
tell a cold workload from a broken router — and a control arm reporting
`no-prompt` where it should report `round_robin` is the body-shape defect of
2026-09-19, still visible from outside
([`../docs/benchmarks/runsheets/mi300x-run-3.md`](../docs/benchmarks/runsheets/mi300x-run-3.md) §0).

## 2. The key is bytes, and the claim that makes that legal

vLLM's prefix cache matches **blocks of tokens, from token zero**
([`../docs/GLOSSARY.md`](../docs/GLOSSARY.md), *Prefix caching*). A router that
wants a hit must group requests by their leading tokens, and this one has no
tokenizer: shipping a second copy of the model's vocabulary means keeping it
equal to the engine's, and a wrong tokenizer routes confidently to the wrong
replica — worse than not routing.

So the key is a hash of the first `-key-bytes` **bytes** of the prompt, plus the
model name. The claim:

> Two prompts sharing a byte prefix share every token of that prefix except, at
> most, the one token straddling the cut.

BPE is deterministic and reads left to right, so an identical byte run produces
an identical token run up to the boundary. At a 16-token block size that costs
at most one block of the shared prefix — **an error in the safe direction**: it
understates the shared prefix, never overstates it.

What the key does *not* claim is that the prefix is still in the cache. That
depends on eviction, which lives on the replica. Affinity raises the probability
of a hit; it does not produce one.

## 3. A ring, not `hash % n`

Adding one replica is the event KEDA produces, repeatedly, under exactly the
load that makes locality worth having (`../deploy/keda/README.md`). Under
modulo, nearly every key lands somewhere new — a fleet-wide cache miss timed to
arrive with the traffic. Under a consistent-hash ring, only the arc the newcomer
claims moves.

`ring_test.go` measures both over 20 000 keys, growing a fleet from four
replicas to five:

    4 -> 5 replicas: ring moves 20.4% of keys, modulo moves 79.9%

20 % is the ideal — the share the newcomer is entitled to. The argument is the
test, not this paragraph.

**Virtual nodes** are what make that true rather than intended. Four replicas
with one ring point each cut the space into four arcs of unequal size; the
measured relative spread of traffic was **1.12 at one point per replica and
0.083 at 128**, against a 1/√128 = 0.088 expectation.

That test also found a real defect. FNV-1a diffuses a late-byte difference
upward by one multiplication only, so `addr#0` … `addr#127` agreed in their high
bits — and a ring is ordered by exactly those bits. All 128 points of a replica
landed in one arc: a ring with the balance of a single point, measured spread
0.53, and a fifth replica taking 8.5 % of the keys instead of 20 %. The fix is
an avalanche step after the hash. **A hash function is not interchangeable with
another hash function** when what you use is its ordering.

## 4. Bounded loads, because affinity alone is a way to overload one replica

One popular system prompt is one key, and every request carrying it wants the
same engine. Left alone, the router would queue a workload behind one replica
while three sat idle — and queueing is the largest term in the TTFT budget, by a
margin that [`../docs/SLO.md`](../docs/SLO.md) §4 gives per card.

So a replica at capacity is skipped and the walk continues around the ring.
Capacity is `-bounded-load` × the fleet's mean in-flight count, plus one so that
an idle fleet is routable:

    limit = int(c * (total_in_flight + 1) / n) + 1

`c` = 1 is strict balance and no affinity at all; `c` large is affinity with no
protection. The default, 1.25, allows a replica a quarter more than its share
before traffic spills. Measured in `ring_test.go`: 40 requests on one hot key
land on **4 of 4 replicas, the busiest holding 13**, where the unbounded ring
gives it all 40.

## 5. Concurrency: three objects, three different rules

| What | Rule | Why |
|---|---|---|
| the ring | immutable; replaced wholesale by `atomic.Pointer` | a request loads it once and uses that snapshot to the end, so a scale event mid-stream cannot move a request already being served |
| the in-flight counter | `atomic.Int64` on the replica, **carried across rebuilds** | a rebuild happens under a scale event, which is under load; a reset would tell §4 that every surviving replica is idle |
| the round-robin cursor | `atomic.Uint64` on the ring, reset by a rebuild | `round_robin` has no memory worth preserving |

The counter is raised before the first byte leaves and released by `defer` after
the last one is copied — "in flight" covers the whole stream, which for a
completion is most of its life. `router_test.go` asserts it returns to zero
under fifty concurrent requests, and the suite runs under `-race`.

## 6. Failure modes, each with the signal that shows it

1. **A leaked in-flight count.** §4's cap tightens with every request until the
   ring routes by load alone — affinity silently gone, no error anywhere.
   Signal: `prefix` policy still reported while the busiest replica never
   exceeds the mean. Guard: the `defer`, and the mutation-checked test.
2. **The body read twice, or not put back.** Routing requires reading the
   payload; a proxy that inspects payloads and forwards a truncated one fails
   silently at the engine, as a malformed request. Guard: the upstream's
   received bytes are compared to the sent bytes, in both the buffered and the
   oversized path.
3. **A buffered stream.** A proxy that batches writes converts a good TPOT into
   a bad one without the engine changing at all, and no test that reads the
   whole response can see it. Signal: ITL measured at the client far above ITL
   at the engine. Guard: `FlushInterval: -1`, and a test whose upstream holds
   the second frame until the client has read the first.
4. **A hot key that is not hot at the engine.** The router bounds by *its own*
   in-flight count, which is a proxy for load and not a measurement of it. A
   replica slow for another reason — a neighbour on the same card, a long
   generation — is not detected. This is the honest limit of §4 and the reason
   the engine's own queue depth belongs in the decision eventually.

## 7. Running it, against engines this repository already has

Two ways, and this section is the second of them. On a cluster — the edge in
front, the stub Pods behind, the fleet filled from the EndpointSlice — the
commands and what to expect are [`../deploy/router/README.md`](../deploy/router/README.md)
§3. Below is the same binary with no cluster at all, which is still the fastest
way to see a policy change.

The tests need nothing at all:

    cd router
    go test -race ./...
    go build -o bin/prefix-router .

Three of them print measurements rather than `PASS` — the two in §3 and the one
in §4 — and `-race` is the point of the run rather than a precaution (§5).

Seeing it route needs engines to route *to*, and
[`../deploy/manifests/overlays/kind/stub/`](../deploy/manifests/overlays/kind/stub/)
is one: it carries vLLM's API contract, streams SSE and serves no model
([`../deploy/manifests/README.md`](../deploy/manifests/README.md)). Two of those
and the router are the whole demonstration — no cluster, no card, nothing
installed:

    DRAIN_DELAY_S=0 PORT=8001 python3 ../deploy/manifests/overlays/kind/stub/server.py &
    DRAIN_DELAY_S=0 PORT=8002 python3 ../deploy/manifests/overlays/kind/stub/server.py &
    ./bin/prefix-router -upstreams http://127.0.0.1:8001,http://127.0.0.1:8002 -listen :8099 &

    # one prompt, four times -- one replica
    for i in 1 2 3 4; do
      curl -s -D - -o /dev/null -X POST localhost:8099/v1/completions \
        -H 'content-type: application/json' \
        -d '{"model":"Qwen/Qwen3-8B","prompt":"you are a careful assistant. summarise.","max_tokens":2}' \
      | grep -i '^x-router'
    done

    # twelve prompts, all different -- the fleet
    for i in $(seq 1 12); do
      curl -s -D - -o /dev/null -X POST localhost:8099/v1/completions \
        -H 'content-type: application/json' \
        -d "{\"model\":\"Qwen/Qwen3-8B\",\"prompt\":\"unrelated question $i\",\"max_tokens\":2}" \
      | grep -i '^x-router-upstream'
    done | sort | uniq -c

**Read the policy, not the address.** Which replica a given prompt gets is the
hash's business and differs per machine; the property is `prefix` on all four of
the first loop with one address repeated four times, and **two** lines out of
the second. One line out of the second — every prompt on one replica — is the
defect in §3 come back.

A body with no prompt in it is the fastest way to see the other half:

    curl -s -D - -o /dev/null -X POST localhost:8099/v1/completions \
      -H 'content-type: application/json' -d '{"model":"Qwen/Qwen3-8B"}' | grep -i '^x-router'

`no-prompt`, and the request went round the ring by turn: the router names which
of the five policies ran rather than implying a hit it did not get (§1).

Tear down with `kill` on the three jobs. `DRAIN_DELAY_S=0` shortens the stub's
own drain so its port frees at once; the router has the same behaviour on
`-drain`, which is why killing it does not free `:8099` for five seconds — that
is §5's ordering working, not a hang.

Flags: `-key-bytes` (512; 0 routes only exact repeats), `-bounded-load` (1.25),
`-vnodes` (128), `-max-body` (1 MiB — 4 000-token prompts are ~16 kB, so the
cap is there for the request that is not a prompt at all), `-drain` (5 s between
failing readiness and closing the listener, because withdrawing readiness and
withdrawing traffic are one action — `../deploy/ingress/README.md`),
`-dial-timeout` (250 ms — how long to spend discovering that an upstream is
gone; the default transport's 30 s is two orders of magnitude past the TTFT
budget, measured in `../deploy/router/README.md` §2).

`/-/healthz` is the router's own; every other path is forwarded.

## 8. What this has not established

- **Anything about TTFT or seats.** Every number above is about *routing* —
  which replica, how evenly, how often it changes. Not one of them says the
  seat count moves. That measurement needs the fleet on `kind` and then a card,
  and the prediction has to be written before the run
  ([`../docs/adding-a-run.md`](../docs/adding-a-run.md)). MI300X run 3 faced
  that prediction on 2026-09-29, and its arm comparison was void: the
  instrument let the second arm hit the first arm's requests
  ([`../docs/benchmarks/mi300x-run3.md`](../docs/benchmarks/mi300x-run3.md) §5).
  The same run showed the router routing as this file says, bounded loads
  included, on a pair whose engines were not equal (§6 there). The re-run is
  [`../docs/benchmarks/runsheets/mi300x-run-4.md`](../docs/benchmarks/runsheets/mi300x-run-4.md).
- **That the byte prefix buys the block alignment §2 argues it does.** The
  argument is about BPE, and BPE was not run. The test is one prompt pair
  tokenized by the engine's own tokenizer, compared block by block.
- **What `-key-bytes` should be.** 512 bytes ≈ 128 tokens ≈ 8 blocks is a guess
  with a shape, not a fit. Too short and unrelated prompts share a replica; too
  long and a system prompt with a per-user line in it never matches.
- **Retrying a connection error on a different replica.** The edge already drops
  the `timeout` half of `proxy-next-upstream` because a retried generation
  re-spends accelerator time (`../deploy/ingress/README.md`). The connection
  half is the recoverable one, and retrying it here is a second routing decision
  that needs its own test. What the `kind` run added is the bill for not having
  it: with a stale fleet, every request in the departed replica's share of the
  ring fails, and `-dial-timeout` only decides how fast. A retry would convert
  those into successes at the cost of the affinity they were routed for —
  which is the trade to write down before writing the code.
- **Where the replica set comes from.** It is still a flag — but the flag is no
  longer only an inelegance, it has a price, and the price was measured on
  `kind` on 2026-09-19: a fleet grown 3 → 4 left the new replica with **none**
  of 24 requests and raised no error anywhere, and a fleet shrunk 3 → 2 sent
  **29 %** of requests to an address that no longer existed, permanently
  (`../deploy/router/README.md` §1). A Pod watch is the answer, and
  `setUpstreams` is already safe to call under load — which is the whole reason
  the ring is built the way §5 says.
- **Whether this should be a Gateway API endpoint picker instead.** The serving
  gateways build on the Inference Extension for exactly this decision
  (`../docs/GLOSSARY.md`, *Gateway API*), and a router that duplicates it is a
  liability the day the cluster grows one.
