# Glossary

Vocabulary and notation used across this repo. This is a **reference**: it exists
to be looked up, not memorised, and it is deliberately short — one or two lines
per term, enough to read [SLO.md](SLO.md) and the benchmark reports without a
detour. Where a term carries an argument rather than just a meaning, the argument
lives in [SLO.md](SLO.md) and this file points at it.

Terms marked ⏳ are introduced later in the project and are listed so the picture
is complete.

**Maintenance rule.** Every term, abbreviation, metric, formula symbol or named
configuration knob used anywhere in this repo has an entry here, one or two lines,
definition only — an argument belongs in [SLO.md](SLO.md) and the entry links to
the section. A term used in prose elsewhere with no entry here is a defect in this
file.

---

## Notation

Every symbol used in [SLO.md](SLO.md), `bench/roofline.py` and the benchmark
reports. Work in SI base units throughout — bytes, bytes/s, FLOP, FLOP/s,
seconds — and convert only for display.

**SI** (Système International d'Unités) — the metric system, whose prefixes are
powers of ten: `k` = 1e3, `M` = 1e6, `G` = 1e9, `T` = 1e12. **IEC** binary prefixes
are the other convention — `KiB` = 1024, `GiB` = 2³⁰ — and the two differ by 7.4%
per `G`. Vendor bandwidth figures are decimal by construction, so this repo works
in SI throughout and converts only for display. Memory *capacity* is the ambiguous
one: vLLM's startup log settles it ([SLO.md](SLO.md) §9).

### Read from the model

| Symbol | Meaning | Where from |
|---|---|---|
| `P_total` | total parameter count — drives **memory**, since every weight is resident | model card |
| `P_compute` | non-embedding parameter count — drives **FLOPs**, since the input embedding is a lookup rather than a matrix multiply | model card |
| `bytes_per_param` | bytes per stored weight: BF16 and FP16 = 2, FP8 = 1 | `config.json` → `torch_dtype` |
| `L` | number of transformer layers — each keeps its own KV cache | `config.json` → `num_hidden_layers` |
| `n_q` | query heads per layer | `config.json` → `num_attention_heads` |
| `n_kv` | key/value heads per layer. Under GQA, `n_q / n_kv` query heads share one KV head — that ratio is exactly how much KV memory GQA saves | `config.json` → `num_key_value_heads` |
| `head_dim` | width of one attention head | `config.json` → `head_dim`, else `hidden_size / n_q` |

### Read from the hardware

| Symbol | Meaning | Where from |
|---|---|---|
| `BW_peak` | peak memory bandwidth, bytes/s | vendor spec table |
| `FLOPS_dense` | peak FLOP/s at the serving precision, **dense** row | vendor spec table — never the sparsity row |

### Chosen by the workload

| Symbol | Meaning |
|---|---|
| `batch` | sequences running concurrently in one forward pass |
| `context_len` | tokens of context held per sequence |
| `B_tokens` | tokens processed in **one forward pass**. Prefill: the prompt length. Decode: equal to `batch`, one token per sequence. This single variable is what distinguishes the two phases |
| `h` | prefix cache hit rate — the share of prompt tokens served from the cache instead of being computed. A property of the traffic, not of the card, and the only quantity in this repo an operator cannot derive ([SLO.md](SLO.md) §6) |

### Empirical coefficients

| Symbol | Meaning |
|---|---|
| `eff_mem` | fraction of `BW_peak` actually achieved. Applies to the memory side only. 0.70 assumed everywhere; **0.83 measured on the L40S**, decode |
| `mfu` | fraction of `FLOPS_dense` actually achieved. Applies to the compute side only. 0.45 assumed everywhere; **0.439 measured on the L40S**, prefill |

Neither appears on a datasheet. They are the error bars of every estimate in this
repo — see [SLO.md](SLO.md) §9 for how they get calibrated. A measured value
belongs to one card **and** one phase and never transfers to another of either.

### Derived

```
weights_bytes      = P_total × bytes_per_param

kv_bytes_per_token = 2 × n_kv × head_dim × bytes_per_param × L
                     └ K and V are cached separately

kv_bytes_total     = kv_bytes_per_token × batch × context_len

bytes_moved        = weights_bytes + kv_bytes_total
                     device-memory traffic in one forward pass. Decode *reads*
                     the KV of every resident sequence; prefill *writes* the KV
                     of the prompt it is consuming — kv_bytes_per_token × B_tokens
                     — so the term is present in both phases and means the
                     opposite direction in each

t_mem_ideal        = bytes_moved / BW_peak
t_compute_ideal    = 2 × P_compute × B_tokens / FLOPS_dense
                     └ FLOPS_PER_MAC: one multiply–accumulate = 2 FLOP
t_step_ideal       = max(t_mem_ideal, t_compute_ideal)          ← roofline

t_mem_real         = t_mem_ideal     / eff_mem
t_compute_real     = t_compute_ideal / mfu

achieved_bandwidth = BW_peak × eff_mem
                     the denominator of every decode floor; the peak row of a
                     vendor table is never it
```

Weights are read **once per forward pass** whatever `B_tokens` is; KV is read per
sequence. That asymmetry is the whole economics of batching, worked through with
numbers in [SLO.md](SLO.md) §5.

---

## Request phases

**Token** — the unit a model reads and writes: a word piece from a fixed
vocabulary, on average about three quarters of an English word. Prompt
lengths, answer lengths, speeds and prices in this repository are all counted
in tokens, never in words.

**Autoregressive generation** — the model produces one token at a time; each new
token is appended to the input and fed back in. A response of N tokens therefore
costs N sequential forward passes, not one.

**Prefill** — the first forward pass, which processes the entire prompt at once and
produces the first output token. All prompt tokens are computed in parallel, so
prefill is typically **compute-bound**.

**Decode** — every subsequent forward pass, producing exactly one token per pass
per sequence. Very little arithmetic per pass, but the full weight matrix is read
from memory each time, so decode is typically **memory-bound**.

---

## Latency and throughput metrics

**Seat** — one conversation held at the same time as the others; the unit
concurrency, ceilings and cost per seat are counted in.

**TTFT** (Time To First Token) — from request arrival to the first token reaching
the client. Dominated by queue wait plus prefill. What a user perceives as "did it
hang?".

**TTFB** (Time To First Byte) — a transport measurement, and **not** TTFT: the
first byte of an HTTP response is the start of its *headers*, which a proxy
forwards long before any token exists. Useful for locating a hop, useless as a
token-latency instrument.

**SSE** (Server-Sent Events, `Content-Type: text/event-stream`) — the transport a
streaming completion arrives over: one event per token over a single response
body held open. **TTFT is only observable through it** — a non-streaming
completion has exactly one arrival, so TTFT and total latency are the same
number at the client.

**Chunked transfer encoding** — an HTTP/1.1 body of unknown length, sent as
size-prefixed frames ending in a zero-length one. What lets a response begin
before its length is known, and therefore what SSE rides on.

**TTFAT** (Time To First Answering Token) — for reasoning models, time to the first
token the user actually sees, i.e. the first token *after* the chain of thought.
Only measurable if the server separates reasoning output from final output. Why
TTFT stops working here: [SLO.md](SLO.md) §8.

**ITL** (Inter-Token Latency) — the gap between two consecutive output tokens, one
measurement per gap. Its **median** is the closest a load test gets to the cost of
a single decode step, because it is unmoved by the minority of steps that carry
something else.

**TPOT** (Time Per Output Token) — the *mean* ITL of one request. Determines
perceived "typing speed"; `1 / TPOT` = tokens per second for one stream, and it is
TPOT, not median ITL, that an SLO is written against, because a user experiences
the mean and not the median.

The two are equal only when nothing interferes. Under chunked prefill a decode step
that also carries a prefill chunk is several times longer, the mean absorbs those
steps and the median does not, so TPOT drifts above median ITL as load rises —
measured at 1.00× alone on the card and 2.06× at the capacity ceiling
([benchmarks/l40s-baseline.md](benchmarks/l40s-baseline.md) §5). Reporting one and
calling it the other is how a scheduling cost gets charged to the memory bus.

**The median ITL is a decode step only while prefill occupies a minority of steps**,
and that condition is a knob away from failing: at `max_num_batched_tokens` 512 with
32 sequences the median reads 79.6 ms against a 49.8 ms step, because a small chunk
puts prefill into most steps and the median moves in with it. The tell is the ratio
itself falling to 1 or below — which also means the ratio measures how *unevenly*
prefill is spread, not how much of it there is
([benchmarks/l40s-run2.md](benchmarks/l40s-run2.md) §6).

**End-to-end latency** — `TTFT + TPOT × output_tokens`. The number a batch pipeline
actually cares about.

**Throughput**, also **aggregate tok/s** — total tokens per second the server emits
across all concurrent requests. One decode step emits one token per sequence, so
at a steady operating point it is `batch / TPOT` — which is why batching multiplies
it while dividing `1 / TPOT`, the rate one user sees. Maximising it alone is
misleading: a server can have excellent throughput while every individual user is
far outside their latency target.

**`$/1M tokens`** — the unit customers are quoted in and the denominator that makes
throughput matter: `hourly_rate / (aggregate tok/s × 3600) × 1e6`. Derived from a
floor it is a *lower bound* on cost, since a real run sits above the floor. The
rate is a contract figure, not a hardware one — [SLO.md](SLO.md) §7.

**Goodput** — throughput counting **only** requests that met their SLO. The honest
operator metric: it collapses the moment latency targets start failing, whereas raw
throughput keeps looking healthy. Measurable directly: `vllm bench serve
--goodput ttft:300 tpot:50` reports it in requests per second, with the values in
**milliseconds** and the keys limited to `ttft`, `tpot` and `e2el` — so the SLO the
number is filtered against is written into the command rather than reconstructed
afterwards.

**SLO-respecting output token** — an output token belonging to a request that met
every SLO in force, so the cost denominator built from it is `goodput req/s × output
length`, not the throughput the server reports. Same formula as `$/1M tokens`,
different denominator: past the goodput peak the two figures move in opposite
directions, which is why a cost figure quoted without naming its denominator is
not a figure — [SLO.md](SLO.md) §7.

**p50 / p95 / p99** — percentiles. `TTFT p99 = 500 ms` means 99% of requests get
their first token within 500 ms. SLOs are written against tail percentiles, not
averages, because averages hide the queue.

**Little's law** — for a queue at steady state, `items = arrival rate x time in
system`. Used here in the other direction: with all seats busy a seat frees every
`service time / seats`, so a tolerable wait converts into a tolerable queue depth
— the step that turns a TTFT budget into an autoscaling threshold
([SLO.md](SLO.md) §4).

---

## Reliability vocabulary

**SLI** (Service Level Indicator) — a measured quantity, e.g. observed TTFT p99.

**SLO** (Service Level Objective) — the internal target for an SLI, e.g. "TTFT p99
≤ 300 ms". What this platform is engineered against.

**SLA** (Service Level Agreement) — a contractual promise to a customer, with
penalties. Always looser than the SLO, so there is room to react before money is
involved.

**Error budget** — the allowed share of SLO violations over a window. Spending it
slowly is fine; burning it fast is what an alert should fire on.

**Burn rate** — the observed error share divided by the share the SLO allows. 1
spends the error budget exactly over its period; 14.4 spends 2 % of a 30-day
budget in one hour, the conventional page threshold. Which SLI it is computed
on here: [SLO.md](SLO.md) §4, "The alert rule".

**Multi-window burn-rate alert** — one SLI, two windows that must agree: the long
one (1 h) establishes that the burn is real, the short one (5 m) that it is still
happening, so the page resolves when the cause does.

**Alerting rule** (`for:`, `pending`, `firing`) — a PromQL expression Prometheus
evaluates every `evaluation_interval`. A non-empty result puts the alert in
`pending`; it becomes `firing` once the result has stayed non-empty for `for:`.
The state is exposed as the `ALERTS` series and at `/alerts`; delivering it to a
person is Alertmanager's job, which this repo does not run.

**Alertmanager** ⏳ — the Prometheus component that receives firing alerts and
routes, groups and silences them towards a person. Absent here: "fires" in this
repo means the rule's state, not a page received.

**Staleness** — how Prometheus retires a series. A target that leaves service
discovery has its series marked stale at once, so a rule reading them clears at
the next evaluation; a series whose target is still listed but stops answering is
stale only after five minutes. Why a rolled-back canary's alert clears with the
Service, not the Pod: `deploy/manifests/README.md`.

**`evaluation_interval`** — how often Prometheus evaluates `rule_files`; the
granularity of every `for:` clause. At 15 s a 2 m window is eight checks, and an
alert resolves at the first evaluation after its expression turns empty.

---

## Hardware and performance model

**L40S** — NVIDIA's 48 GB data-centre GPU, the card runs 1–3 rented. **MI300X** —
AMD's accelerator with 192 GiB of HBM, the card
[MI300X run 1](benchmarks/mi300x-run1.md) rented; the byte count and why it is
GiB are in [SLO.md](SLO.md) §3.

**Virtual function (VF)** — the slice of a physical GPU a hypervisor hands to one
virtual machine. A cloud droplet's MI300X is one: `rocm-smi` names it
"MI300X VF".

**`rocm-smi`** — AMD's command-line monitor for ROCm devices: product, memory,
the processes holding the GPU through its kernel driver (KFD). The `nvidia-smi`
of this card.

**sclk** — the GPU's shader (compute) clock as `rocm-smi --showclocks` reports it,
the current level and its frequency. Beside it, *Performance Level* says whether
the driver picks the level (`auto`) or it is pinned. A snapshot under load, not a
rated figure.

**Image digest** — the content hash a container registry gives one exact image
(`repo@sha256:…`). A tag such as `v0.27.1` can be moved to other contents; the
digest cannot, so it is what says two runs pulled the same bytes.

**gfx942** — the ISA target name of the MI300 series, which `rocm-smi` prints and
ROCm kernels are compiled for; gfx950 is the MI350 series'. Both are **CDNA 3** or
later, CDNA being AMD's data-centre GPU architecture line, and several ROCm code
paths are gated on that generation rather than on the ISA name.

**Coefficient provenance** — whether an empirical coefficient is *measured* on
the card it is used for or is a *prior* carried from a spec sheet. A property of
the pairing, not of the number: 0.70 is a prior on MI300X and was one on the L40S
until run 1 replaced it. Named because a floor built on a fit and a floor built
on an assumption are the same shape on the page and are not the same claim, so
the status travels with the value wherever it prints.

**Predicted only** — of a *model* what *prior* is of a coefficient: an
architecture the calculator prices from its `config.json` figures while this
stack never serves it and no run here has measured it. Nothing a run produced is
lent to it — no measured point is drawn over it and the prefill interference fit
is withheld, so its service seat count reads *not derivable* — because pricing a
model is not serving it ([audience.md](audience.md#changing-the-model)).

**Prior** — a coefficient carried from a spec sheet or an assumption rather than
fitted on the card it is used for; `l40s` and `mi300x` in `bench/roofline.py`
carry `0.70` / `0.45` as priors. A floor printed from a prior is the shape of an
answer, not its size, and every page that prints one says so — see *Coefficient
provenance*.

**Roofline** — execution time is bounded by whichever is slower, arithmetic or
memory movement: `time = max(compute_time, memory_time)`. All tuning in this repo
comes down to identifying which side binds.

**Compute-bound** — the arithmetic units are the bottleneck and the memory system
has spare capacity. Prefill on a long prompt.

**Memory-bound** — moving bytes is the bottleneck and the arithmetic units are
mostly idle. Decode at small batch sizes.

**Floor** — the best achievable value of a metric on given hardware, derived from
physics rather than measured. Anything above it is queueing, overhead or
misconfiguration. Used to check that an SLO target is physically reachable before
committing to it; the floors for this stack are in [SLO.md](SLO.md) §4.

**HBM** (High Bandwidth Memory) — the on-package memory holding weights and KV
cache. Its **memory bandwidth** (TB/s) sets the decode floor.

**GDDR6** — the alternative memory technology on graphics-derived accelerators;
the L40S carries 48 GB of GDDR6 with ECC rather than HBM. Nothing in the roofline
depends on which technology it is — only on capacity and bandwidth — which is why
`bench/roofline.py` names the field `memory_bytes` and not `hbm_bytes`. A field
naming a technology would be false for half the cards this stack is run on.

**FLOPS, dense vs sparse** — vendors publish both. The "sparse" figure assumes
structured sparsity, is usually exactly 2× the dense one, and does not apply to LLM
inference. Always derive from the **dense** number.

**MAC** (multiply–accumulate) — one multiplication plus one addition into a
running sum, the single operation a matrix multiply is built from. A matmul of
`P` weights over `B` tokens performs `P × B` MACs.

**`FLOPS_PER_MAC`** — the constant 2, because one MAC counts as two floating-point
operations. It is where the leading 2 in `2 × P_compute × B_tokens` comes from,
and it is arithmetic rather than a property of the model or the accelerator:
`bench/roofline.py` names it as a module constant so it cannot be mistaken for a
dtype or a coefficient.

**MFU** (Model FLOPs Utilization) — the fraction of peak dense FLOPS actually
achieved. Realistic prefill values are 40–50%, which is why a floor derived from
peak must be divided by MFU to become a usable estimate.

**Arithmetic intensity** — FLOPs performed per byte moved. Low intensity puts a
workload on the memory-bound side of the roofline, high intensity on the
compute-bound side. Batching raises the arithmetic intensity of decode, which is
exactly why it works.

**SRAM** — static on-die memory. Two to three orders of magnitude more bandwidth
per byte than HBM and roughly two orders more expensive per byte, so architectures
built on it trade capacity for bandwidth. See
[accelerator-landscape.md](accelerator-landscape.md) §3 for what that trade costs.

**LPDDR** — low-power commodity DRAM used by capacity-first accelerators. Large
capacity per watt and per dollar, bandwidth roughly an order of magnitude below
HBM — the opposite trade to SRAM.

**Wafer-scale** — an accelerator built as one undiced wafer rather than a die, so
its memory is entirely on-wafer SRAM. Cerebras only.

**ASIC** (Application-Specific Integrated Circuit) — silicon fixed to one workload
class rather than programmable. In this market the word usually means a
hyperscaler's in-house part (TPU, Trainium) rather than a merchant GPU.

**Near-memory computing** — placing compute next to the memory array to cut the
bytes crossing the bus. Pays off in proportion to data reuse, so it helps prefill
far more than decode.

**`sweep_time`** — derived: the time to read an accelerator's entire memory once,
`memory_capacity / memory_bandwidth`. A single-number stand-in for decode
suitability across cards; defined, tabulated and bounded in
[accelerator-landscape.md](accelerator-landscape.md) §2.

**Scale-up domain** — the set of accelerators sharing one coherent high-bandwidth
fabric (NVLink, UALink, Infinity Fabric), across which a model may be sharded at
full speed. Its **size** is the count of accelerators in it, and it is the hardware
limit on how wide a model can be split before traffic falls back to the network.

**Scale-out** — the network *between* scale-up domains (Ethernet, InfiniBand),
typically an order of magnitude slower per accelerator. Any parallelism strategy
that crosses this boundary pays for it.

**NVLink** — NVIDIA's scale-up fabric. NVL72 is a 72-GPU domain; the per-GPU
bidirectional figure (1.8 TB/s on Blackwell, 3.6 TB/s on Rubin) is what an
all-to-all between shards actually runs at.

**Expert parallelism (EP)** ⏳ — sharding an MoE model's experts across
accelerators rather than replicating them. "Wide EP" spreads them across a whole
scale-up domain, which frees memory per accelerator for KV cache at the cost of an
all-to-all per token.

**Disaggregated prefill/decode** ⏳ — running the two phases on separate
accelerator pools, so each is sized for the limit that actually binds it
([model-anatomy.md](model-anatomy.md) level 1) instead of sharing one card that is
wrong for both.

---

## Model internals

**Parameters (P)** — the weight count. Bytes of weights = `P × bytes_per_param`.
The rule of thumb `2 × P × B` gives forward-pass FLOPs for `B` tokens.

**BF16 / FP16 / FP8** — numeric formats: 2, 2 and 1 bytes per value. BF16 and FP16
differ in exponent range, not size. Serving in FP8 halves both weight bytes and KV
bytes, moving the memory-bound floor down.

**KV cache** — the stored keys and values of every previous token, so each decode
step does not recompute the whole prefix. Grows linearly with
`batch × context_length`, and eventually rivals or exceeds the weights in size.

**GQA** (Grouped-Query Attention) — several query heads share one key/value head,
so the KV cache stores `n_kv` head pairs per layer instead of `n_q`. The saving is
exactly the ratio `n_q / n_kv` — for the default model, 4× the memory bill for
identical output ([SLO.md](SLO.md) §3). GQA shares KV heads *within* a sequence;
reusing KV *between* requests is prefix caching, a different mechanism.

**MHA / MQA** — the two ends of the same spectrum. MHA (multi-head) gives every
query head its own KV head, `n_kv = n_q`: maximum cache, no sharing. MQA
(multi-query) collapses all query heads onto a single KV head, `n_kv = 1`: minimum
cache, some quality cost. GQA sits between them and is what most current models
ship.

**head_dim** — width of one attention head, `hidden_size / num_attention_heads`
unless stated explicitly in `config.json`.

**hidden_size** — width of the vector carried from layer to layer, one per token.
`config.json` → `hidden_size`. **vocab_size** is the number of distinct tokens, and
sets the width of the two vocabulary-sized matrices below.

**Embedding table** — `vocab_size × hidden_size` weights turning token ids into
vectors. Accessed by **gather** — the rows named by the input ids are looked up —
so it costs no matrix multiply, which is why it is excluded from `P_compute`.

**`lm_head`** — the `hidden_size × vocab_size` matrix at the end of the stack,
turning the final hidden state into logits over the vocabulary. Same size as the
embedding table and, in models that do not tie the two, a separate tensor. Unlike
the table it is a real matmul.

**Q / K / V / O projections** — the four weight matrices of an attention block:
three producing queries, keys and values from the hidden state, one projecting the
attention output back to `hidden_size`. Only K and V outputs are cached.

**MLP block** — the feed-forward half of a decoder layer, roughly two thirds of its
parameters. It holds no cache and is pure weight traffic.

**Decoder layer** — one repeated block of the stack: an attention block plus an MLP
block. `L` of them in sequence, each with its own KV cache. Drawn in
[model-anatomy.md](model-anatomy.md).

**RMSNorm and residual connection** — the normalisation applied before each block
and the addition of each block's output back onto the hidden state. Named only so
their absence from the byte accounting is deliberate: both move a negligible share
of the bytes.

**Context length** — prompt plus generated tokens for one sequence. The direct
multiplier on that sequence's KV footprint.

**Batch size** — how many sequences the engine runs concurrently in one forward
pass. Raising it costs almost nothing on the compute side and a great deal on the
memory side, because the extra sequences bring their own KV. The ceiling is set by
the TPOT target, not by how much memory happens to be free ([SLO.md](SLO.md) §6).

---

## Serving mechanics ⏳

**Continuous batching** — the scheduler adds and removes sequences from the running
batch at every decode step instead of waiting for a whole batch to finish. Keeps
the GPU busy under uneven request lengths.

**Chunked prefill** — splitting a long prefill into pieces so it interleaves with
decode steps of other requests, trading a little TTFT for a large drop in TPOT
jitter for everyone else.

**Prefix caching** — reusing the KV cache of a shared prompt prefix (system prompt,
few-shot examples) across requests, removing that part of prefill entirely. On by
default in vLLM's V1 engine, and turned off with `--no-enable-prefix-caching`.
Repeating a prompt is a cache hit, so a benchmark that sends the same prompts twice
measures the cache and not the engine — which is why a measured TTFT *below* its
floor points here first. What it does and does not buy — three channels, only
one of which matters on a latency-bound card — is [SLO.md](SLO.md) §6.

**Prefix cache hit rate** (`h`) — the share of a prompt's tokens found in the
cache. It sets how much prefill work disappears, and on the L40S `h` = 0.8 was
measured worth 3.0× the seat count and 3.0× the cost of a token a customer can
use ([SLO.md](SLO.md) §6). vLLM's V1 engine exposes no gauge for it — the
V0 `vllm:gpu_prefix_cache_hit_rate` was replaced by two counters,
`vllm:prefix_cache_queries` and `vllm:prefix_cache_hits`, both counted in
**tokens**, so their ratio over a window is exactly `h` as defined here:
`rate(vllm:prefix_cache_hits[5m]) / rate(vllm:prefix_cache_queries[5m])`.

**Nominal hit rate** — the `h` a benchmark workload is *built* to produce: the
block-aligned shared prefix divided by the prompt length. Distinct from the
measured `h` above, which comes from the engine's counters over one level's
window, and reported beside it — the two are independent routes to one quantity,
so a gap between them is a defect in the workload rather than a finding about the
engine (`bench/scenarios/`).

**Working set** (of prefixes) — how many *distinct* prompt prefixes the traffic
touches in a window. `h` is what one prefix is worth; the working set is how many
of them a replica is asked to keep at once, and it decides whether prefix-aware
routing buys anything: a set small enough to fit on every replica makes affinity
a no-op ([SLO.md](SLO.md) §6, channel 2; the arithmetic is table 11 of
`bench/predictions.py`).

**LRU** (least recently used) — the eviction order a cache uses when it must
free space: the block untouched for longest goes first. It is why a working set
larger than the pool costs hit rate rather than memory, and why a workload that
asks for its prefixes in strict rotation measures the worst case — each one is
evicted exactly before it is next needed. vLLM's block pool is LRU, read at the
pinned tag: a finished request's hashed blocks — body and output as well as the
prefix — go to the tail, eviction takes the head
([runsheet, MI300X run 3 §0](benchmarks/runsheets/mi300x-run-3.md)).

**Zipf distribution** — the popularity shape real prompt traffic has: a few
prefixes carry most requests and a long tail carries the rest. Between a uniform
draw and a strict rotation, which is why a benchmark that uses either states
which of the two it measured ⏳.

**Prefill interference** — the amount by which a request's average decode step
exceeds the step the hardware performs, because prefill chunks of *other*
requests land in the same steps. Measured directly as TPOT minus median ITL, and
the quantity that separates the 31 seats the decode step permits from the 12 an
L40S can promise ([benchmarks/l40s-baseline.md](benchmarks/l40s-baseline.md) §5).
`max_num_batched_tokens` redistributes it; prefix caching removes it in
proportion to `h`. The fitted line belongs to an engine, a chunk size, a card
**and a model** — `INTERFERENCE_FITS` and `INTERFERENCE_MODEL` in
`bench/predictions.py` — and is withheld from any other pairing rather than
lent to it.

**Cascade attention** — an attention path that reads a KV prefix shared by the
whole batch **once** per step rather than once per sequence. Named here because
whether the shared prefix is read once or `n` times decides whether prefix caching
also moves the latency limit. Not a mode but a per-step decision: vLLM 0.27.1
re-evaluates a cost model in `use_cascade_attention()` on every step, logs
nothing about the outcome, and can be forbidden it entirely with
`--disable-cascade-attn`. The one measurement this repository has of the
underlying question came out between once and `n` times: at `h` = 0.8 the decode
step fell by half of what a once-per-batch read predicts
([benchmarks/l40s-run3.md](benchmarks/l40s-run3.md) §4).

**L2 residency** — the case where KV a whole batch shares is read from HBM once
per layer and served to the remaining sequences out of the GPU's L2 cache, so the
per-sequence cost falls without any change of attention path. Bounded by the
cache: it holds while the layer's working set fits and stops when the batch's
unique KV evicts the shared part, which makes it a *concurrency-dependent*
saving, unlike cascade attention. Attention runs a layer at a time, so the
quantity to compare against L2 is `2 · n_kv · head_dim · dtype_bytes` per token —
4 096 B for the repo's default model, not the 147 456 B of all 36 layers.

**Preemption** — the scheduler taking the KV cache back from a sequence that is
already running, because the pool cannot hold every admitted sequence at once. The
work is not lost but it is repeated: the sequence is either recomputed from its
prompt or swapped out and back. Counted by vLLM as `vllm:num_preemptions_total`,
and the symptom is TPOT and TTFT p99 rising together at a concurrency the pool
cannot seat.

**Queue depth** — requests admitted but not yet running. Exposed by vLLM as
`vllm:num_requests_waiting`, and the correct autoscaling signal for a GPU service
([SLO.md](SLO.md) §4).

**FP8 KV cache** — storing keys and values at one byte per value instead of two,
independently of the weight precision (`kv_cache_dtype` in vLLM). It is
compression of what is *stored*, so it needs no FP8 arithmetic support — a
compute capability requirement belongs to FP8 *computation* (W8A8 weights and
activations), not to this. What it buys, and why "TPOT falls" and "twice as many
sequences fit" are answers to different questions: [SLO.md](SLO.md) §6.

**`max_num_seqs`** — cap on sequences running concurrently in one engine. Derived,
not discovered: the `min()` of a latency limit (the largest batch whose TPOT floor
fits the target) and a capacity limit (the sequences whose KV fits). Both are
memory limits — one counts bytes the card holds, the other bytes per second it can
move inside the target — which is why "memory-bound" alone never says which one an
operator is against. Which binds is a property of the card and the target, not a
general rule ([SLO.md](SLO.md) §6). Unset, vLLM picks 256 — or 1024 on a card
with 70 GiB or more (`get_batch_defaults` in `vllm/engine/arg_utils.py`), which is
why the MI300X runs pin it.

**Third limit** — `max_num_seqs` itself, when the configured value is reached
before either limit `max_num_seqs` is derived from: the latency limit or the
capacity limit ([SLO.md](SLO.md) §6). Run 3 met it at `h` = 0.8 on the L40S, 65
requests queued behind 256 with the pool and the SLO both idle; on the MI300X at
4 000 tokens the pinned 256 was predicted to sit *between* the pool and the
latency limit, and MI300X run 1 never reached it — the token-budget ceiling
bound first ([benchmarks/mi300x-run1.md](benchmarks/mi300x-run1.md) §5). The symptom is
`num_requests_waiting` > 0 with `num_requests_running` pinned at the cap and no
preemptions. Open item in [SLO.md](SLO.md) §10.

**`gpu_memory_utilization`** — fraction of GPU memory vLLM may claim; what is left
after weights becomes KV cache space. Applied once, at startup: the engine profiles
a forward pass, then carves the remainder into KV blocks. It is not a live dial,
and neither is any other engine knob — changing one means restarting the server,
while the load-side parameters (concurrency, prompt and output length) belong to
the client and change per run.

**`max_model_len`** — the longest sequence the engine will accept, prompt plus
generation. It is the per-sequence claim on the KV cache, so the concurrency the
startup log reports is the KV pool divided by it: doubling it roughly halves the
sequences that fit, at an unchanged pool.

**`max_num_batched_tokens`** — the token budget of one engine step, prefill and
decode together. It is what chunked prefill spends: a prompt longer than the budget
is consumed over several steps, so this knob decides how much a long prefill delays
everyone else's decode.

**Token-budget ceiling** — the running count an engine settles at when a queue
stands and every step is a full `max_num_batched_tokens` budget:
`n = budget × output / (input + output)`, counting only the tokens that are
prefilled. It binds when it is smaller than both the pool's seat count and
`max_num_seqs`. Derived after MI300X run 1 to fit its running count
([benchmarks/mi300x-run1.md](benchmarks/mi300x-run1.md) §5), and faced by
MI300X run 2 at budgets of 2 048 and 4 096
([benchmarks/mi300x-run2.md](benchmarks/mi300x-run2.md) §5).

**`long_prefill_token_threshold`** — a cap on how many tokens of a *single* long
prefill may enter one step, sitting beside `max_num_batched_tokens` rather than
replacing it: the first bounds the step, the second bounds one request's share of
it. Two knobs on the same step, which is why moving both in one run makes neither
attributable.

**`block_size`** — the number of tokens of KV allocated at once, 16 by default. The
KV pool is handed out in blocks rather than tokens, so a sequence's claim is rounded
up to a whole block, and the pool is reported in tokens only because a block is a
fixed number of them.

**`ignore_eos`** — a per-request flag that keeps generation going to `max_tokens`
instead of stopping at the end-of-sequence token. It turns output length from an
output into an *input*, which is what makes two levels comparable: without it each
level measures a different number of decode steps. Load-testing only — no
production request would set it.

**`--kv-cache-dtype`** — the dtype the KV cache is *stored* in, independent of the
weights: `auto` follows the model, `fp8` stores one byte per element. It halves
`kv_bytes_per_token`, which doubles both the pool and the KV half of `bytes_moved`'s
divisor, so it buys sequences and speed and cannot change which limit binds
([SLO.md](SLO.md) §6). Uncalibrated scales default to 1.0; the accuracy cost is a
separate question this repo has not measured.

**Attention backend** — the kernel family vLLM uses for attention, chosen at startup
from what the card, the dtype and the build support, and printed in the startup log.
On sm89 with a BF16 cache the choice is `FLASH_ATTN` (FlashAttention 2); an FP8 cache
removes it from the candidate list and `FLASHINFER` is selected instead. On ROCm
the first candidate is `ROCM_ATTN`, ahead of `ROCM_AITER_FA` (a candidate only
with `VLLM_ROCM_USE_AITER=1`, forceable without it), `ROCM_AITER_UNIFIED_ATTN`
(when AMD's `aiter` package is found, on CDNA 3 or later) and `TRITON_ATTN`
(`vllm/platforms/rocm.py`). What each runs at `v0.27.1`: `ROCM_ATTN` prefills
through a Triton prefix-prefill kernel (`context_attention_fwd`) and decodes
through a HIP paged-attention kernel; `ROCM_AITER_FA` uses *AITER*'s flash
attention for prefill and its paged attention for decode; `TRITON_ATTN` runs both
through one Triton kernel (`unified_attention`). It matters
because a run that changes the cache dtype has silently changed the kernel too —
measured worth 0.8% of the decode step on the L40S
([benchmarks/l40s-run2.md](benchmarks/l40s-run2.md) §5), but measured rather than
assumed.

**`--attention-backend`** — forces that choice instead of leaving it to selection,
which is how a kernel change is turned into a control. It replaced the environment
variable `VLLM_ATTENTION_BACKEND`, which vLLM 0.27.1 accepts and ignores. When the
backend is forced vLLM logs `Using AttentionBackendEnum.X backend.` — on ROCm,
`Using X backend (selected via --attention-backend).` — and *not* the usual `out of
potential backends` line, so grepping for the latter alone reads like a failure. On
ROCm a forced backend that fails validation raises at startup rather than falling
back.

**AITER** — AMD's kernel library for ROCm, the `aiter` package, built into vLLM's
ROCm image with prebuilt kernels for gfx942 and gfx950. `VLLM_ROCM_USE_AITER=1`
(off by default) lets vLLM choose its kernels across several families at once;
`--attention-backend ROCM_AITER_FA` uses its flash attention alone, without the
variable. On gfx942 the variable does not take the BF16 dense *GEMM*s
(`is_tgemm_enabled` requires gfx950, `vllm/_aiter_ops.py`); it switches AITER's
RMSNorm, sampler and fused rotary / KV-write paths on, and alone it still leaves
attention on `ROCM_ATTN`.

**Triton** — a Python-embedded language and compiler for GPU kernels, compiled at
run time for the card it finds; vLLM writes its portable kernels in it, which is
what lets one kernel serve both CUDA and ROCm.

**GEMM** — general matrix multiply: the weight products that are almost all of a
dense model's FLOPs, and so almost all of prefill's time. On ROCm vLLM sends BF16
GEMMs to PyTorch's own matrix-multiply path (hipBLASLt or rocBLAS) unless a
kernel library takes them.

**Model runner (V1 / V2)** — the vLLM component that turns a scheduled step into
tensors and calls the model. `v0.27.1` runs Qwen3-8B on V2 by default
(`VLLM_USE_V2_MODEL_RUNNER` overrides it). The two order a step's requests
differently, and an attention backend written for one order can misread the
other ([benchmarks/runsheets/mi300x-run-2.md](benchmarks/runsheets/mi300x-run-2.md)).

**`enforce_eager`** — disables CUDA graph capture, running each kernel launch from
Python instead of replaying a recorded graph. It saves the memory a captured graph
holds and shortens startup, and it costs decode latency, because launch overhead is
paid per layer per step and a decode step is small. Set by several hosted vLLM
templates by default; a latency measurement taken with it on attributes framework
overhead to the hardware.

**CUDA graph pool** — the GPU memory holding the captured graphs that
`enforce_eager` would have disabled. It is reserved out of the
`gpu_memory_utilization` budget *before* the KV cache is sized, so capturing
graphs necessarily shrinks the reported KV pool at a fixed utilisation. vLLM logs
the amount it took, which is what makes the difference between a derived pool and
a logged one an accounted quantity rather than a residual.

**Activation memory** — the transient tensors a forward pass needs beyond the
weights, sized by vLLM at startup by profiling one pass at the largest batch it
will schedule. Like the CUDA graph pool it is paid out of the
`gpu_memory_utilization` budget *before* the KV cache is carved, which is why a
capacity ceiling derived as "budget minus weights" is optimistic — by 7% on the
L40S ([SLO.md](SLO.md) §9).

**`--kv-cache-memory`** — sets the KV pool in bytes directly, instead of letting it
fall out of `gpu_memory_utilization` minus everything else. vLLM prints the value
that would reproduce the pool it just chose, which makes the pool reproducible
across driver and version changes that move the other terms.

**`vllm:num_requests_running`** — sequences the engine is decoding right now, as
opposed to `vllm:num_requests_waiting`. Its ceiling under saturation is the seat
count the KV pool actually provides, so it is the cheapest live check of a derived
capacity ceiling: at concurrency 45 it read 41 on the L40S, against 41.2 predicted
from the logged pool.

**`vllm:kv_cache_usage_perc`** — the fraction of the KV pool currently allocated;
named `vllm:gpu_cache_usage_perc` in earlier versions. Reading 1.0 alongside a
non-zero `num_requests_waiting` is the definition of the capacity ceiling being hit,
and it is the state preemption follows from.

**`vllm:num_requests_waiting`** — sequences admitted by the API and not yet
decoding: the queue in front of the seats. The autoscaler's input and the
proxy SLI of the queue alert, because a request that is queued at all has
already missed the interactive TTFT budget ([SLO.md](SLO.md) §4).

**`vllm:time_to_first_token_seconds`**, **`vllm:request_time_per_output_token_seconds`**
— the engine's two latency histograms behind the two targets of [SLO.md](SLO.md)
§2: TTFT per request, and the per-request *mean* time per output token. Read as
`_bucket{le=…}` over `_count`; at `v0.27.1` the TTFT edges nearest the 300 ms
target are 0.25 and 0.5 s, and the TPOT histogram has an edge exactly at 0.05 s.
A third, `vllm:inter_token_latency_seconds`, is ITL rather than TPOT.

**`vllm:request_success_total`**, **`vllm:generation_tokens_total`** — the
engine's completion counter (labelled by `finished_reason`) and its output-token
counter. vLLM declares them without the suffix; the Prometheus client appends
`_total` in the exposition, so a PromQL query names the suffixed form and a
`/metrics` dump the bare one. The same holds for `num_preemptions`,
`prefix_cache_queries` and `prefix_cache_hits`.

**`Peak concurrent requests`** — a line in the `vllm bench serve` summary, and a
*client-side* count: requests the load generator had in flight. It is not
`vllm:num_requests_running`, which counts what the engine is decoding; the difference
is the queue. Measured at the pool ceiling: 111 in flight = 106 running + 5 waiting
([benchmarks/l40s-run2.md](benchmarks/l40s-run2.md) §3). Reading the client's number
as the server's overstates the seats a card has.

**`VLLM_CACHE_ROOT`** — where vLLM writes its on-disk caches, `~/.cache/vllm` by
default. It holds the `torch.compile` / Inductor artefacts and similar, so
pointing it at persistent storage saves the recompile on the next container. It
does **not** hold CUDA graphs, which are captured into GPU memory at every
startup and have no on-disk form.

**Cold start** — the interval from a serving container starting to its first
served token: fetching the weights, loading them onto the card, `torch.compile`,
then profiling, KV pool construction and CUDA graph capture. Decomposed and
measured on the L40S in
([benchmarks/l40s-run3.md](benchmarks/l40s-run3.md) §6).

**AOT compilation cache** — the `torch.compile` artefacts vLLM writes under
`VLLM_CACHE_ROOT` and reloads on a later start with the same key, logged as
`Directly load AOT compilation from path …`. The key is a hash whose members are
only partly established
([benchmarks/l40s-run3.md](benchmarks/l40s-run3.md) §6).

---

## Kubernetes and deployment

Named in `README.md` and the stubs above before the work that introduces them, so
the entries exist as placeholders and get deepened when it happens.

**Pod** — in Kubernetes, the smallest schedulable unit: one or more containers
that share a network namespace and storage and are always placed on one node. **A
RunPod "pod" is not one of these** — it is a single rented container on a GPU host,
the word borrowed rather than the concept. The collision matters wherever a
rented host and a cluster are discussed in the same sentence.

**vLLM** — the inference server this stack deploys; the stub below stands in
for it on `kind`.

**Stub** — the container that replaces vLLM on `kind`: it carries the engine's
API shape, health semantics, metric names and drain behaviour, and serves no
model.

**`kind`** (Kubernetes in Docker) — a cluster running inside local containers, with
no GPU. Used here to debug manifests and controller logic for free, so that GPU
time is spent only on runs that need a GPU.

**Ingress** — the entry point that routes external traffic into the cluster: a
set of host/path rules pointing at Services. The first hop of the request path in
[architecture.md](architecture.md), and the hop where client-observed TTFT
starts. **The object is inert on its own** — it is a rule, and something has to
be reading it (see **ingress controller**). Frozen at `networking.k8s.io/v1`;
its successor is the **Gateway API**.

**KEDA** (Kubernetes Event-Driven Autoscaling) — scales replicas on an external
metric rather than on CPU. Here that metric is `vllm:num_requests_waiting`; the
reason it is the right one is [SLO.md](SLO.md) §4.

**Karpenter** ⏳ — provisions cluster nodes on demand in response to unschedulable
pods. The layer below KEDA: KEDA asks for a replica, Karpenter finds the GPU node
to put it on.

**GPU device plugin** ⏳ — the component that makes GPUs visible to the scheduler as
a requestable resource, so a pod can ask for one the way it asks for memory.

**Operator** — a controller that manages an application through a **reconcile
loop**: read desired state, read actual state, act to close the gap, repeat.
**CRD** (Custom Resource Definition) is how the desired state gets its own
Kubernetes object type. Whether this repo should have one, and what it would look
like, is `controllers/modelwarmup/README.md`.

**Level-triggered** — the controller is handed only the *name* of an object and
re-reads everything else, so it converges from any state and a dropped event
costs nothing. **Edge-triggered** is the opposite: it acts on the event itself
and needs every one delivered. `Reconcile` receives a `NamespacedName` and no
event, which is what makes idempotence a contract rather than a courtesy.

**`Reconcile` outcomes** — the four exits of a `controller-runtime` reconcile.
A returned **error** requeues with exponential backoff (5 ms doubling to a
1000 s ceiling) and the returned `Result` is ignored; **`RequeueAfter`** requeues
after a stated delay and resets the failure count; **`Requeue`** requeues with
backoff; a zero `Result` and no error forgets the item until the next event.
**`reconcile.TerminalError`** marks a failure that will not fix itself and is not
requeued at all.

**Informer** — the watch-and-cache a controller runs per watched type: one `LIST`
at start, then a long-lived `WATCH`, held in process memory. A cached client's
`Get` reads that memory rather than the API server, so a read can be stale and
needs no `get` RBAC verb — only `list` and `watch`.

**`ownerReference`** — a field in a child object's metadata naming its owner. It
drives cascade deletion by the garbage collector, and is the reverse pointer a
controller follows to enqueue the owner when a child changes.

**Status subresource** — a separate `/status` endpoint for an object's status, so
that writes to `status` cannot alter `spec` and writes to the object cannot alter
`status`, each with its own RBAC verbs. Enabled by
`+kubebuilder:subresource:status`; with it, `metadata.generation` increments only
on `spec` changes.

**`observedGeneration`** — the `metadata.generation` a status was computed for.
Without it a fresh status cannot be told from a stale one, which is the
difference between a status a person can read and one a rollout can act on.

**Finalizer** — a string on an object that blocks its deletion until the
controller removes it; the mechanism for cleaning up state *outside* the
cluster's object graph, which `ownerReferences` and garbage collection do not
reach.

**Server-Side Apply** — a write mode in which a client sends only the fields it
owns and the API server records ownership in `metadata.managedFields`; fields the
owner stops sending are removed. A read-merge-with-override-write can add and
change a field but never delete one.

**Deployment** — declares a Pod template plus a replica count; its controller
reconciles actual to desired. Its label **selector is immutable** after creation,
which is why the labels a manifest stamps into it are chosen once.

**Service** (ClusterIP) — a stable virtual IP and DNS name
(`<name>.<namespace>.svc.cluster.local`) in front of a label-selected set of
Pods. It selects by **label, never by name**: a wrong selector yields a Service
with no backends rather than an error.

**EndpointSlice** — the resolved list of Pod addresses behind a Service, and the
only place to see whether a selector actually matched.

**`sessionAffinity`** (Service) — pins a client address to one Pod for traffic
that goes *through* the Service, and it is kube-proxy that enforces it. It has no
effect on requests an ingress controller sends to Pod addresses itself
([architecture.md](architecture.md) §1).

**control plane** — the four processes that decide what runs, none of which run
the workload: `kube-apiserver` (the only door — every change is an HTTP request to
it), `etcd` (stores desired state), `kube-scheduler` (chooses the node),
`kube-controller-manager` (the reconcile loop). Under `kind` all four are static
Pods inside the control-plane container. Kubernetes has **no GUI**; the API is the
interface and `kubectl` is a client.

**startupProbe / readinessProbe / livenessProbe** — three probes with three jobs:
tolerate a slow start, gate traffic, restart the container. While a startupProbe
is still failing the other two are **suspended** — the reason a multi-minute
weight load is not a restart loop. Liveness is set slower than readiness so a
wedged engine leaves rotation before it is killed.

**`strategy: Recreate`** — a rollout that terminates the old Pods before creating
new ones, as opposed to `RollingUpdate`, which starts a surge Pod first. On a
fixed accelerator pool the surge Pod has no free GPU to land on, so the rolling
update never completes; zero-downtime there is a canary with real spare capacity,
which is [runbook.md](runbook.md)'s subject ⏳.

**Extended resource** (`nvidia.com/gpu`, `amd.com/gpu`) — a countable resource
advertised to the scheduler by a GPU device plugin. Declared in `limits` only:
Kubernetes mirrors the value into `requests` and forbids overcommit, so an
accelerator is never split between Pods. An unsatisfiable request leaves a Pod
`Pending` indefinitely rather than failing.

**kustomize** — layering of manifests as a **base** plus **overlays** that patch
it, built into `kubectl` (`apply -k`). `configMapGenerator` folds a file into a
ConfigMap whose generated name carries a content hash, which is what makes
editing that file roll the Deployment instead of leaving stale content in a Pod
that never restarted. `nameSuffix` renames every
resource in a build and rewrites the references between them — Ingress backend,
ConfigMap mount, Service port — which is how two copies of one overlay share a
namespace (`deploy/manifests/overlays/kind-canary`).

**CNI** (Container Network Interface) — the plugin that gives Pods routable IPs
and makes Pod-to-Pod traffic work; `kindnet` in `kind`, and the reason a fresh
cluster reports `NotReady` for its first few seconds.

**ScaledObject** — KEDA's custom resource: a scale target, a floor and a cap, and
one or more triggers. Above `minReplicaCount: 0` KEDA does not scale anything
itself — it registers an **external metric** and the HPA runs the loop, which is
why `pollingInterval` applies only when the floor is zero.

**HPA** (HorizontalPodAutoscaler) — the controller that sets a replica count from
a metric: `desired = ceil(metric / threshold)`. An external metric of type
`AverageValue` is reported *per replica*, so `1250m` across four replicas is five
of whatever is being counted.

**Stabilization window** (`behavior.scaleUp|scaleDown.stabilizationWindowSeconds`)
— how long the HPA must see a smaller desired count before acting on it. Set
asymmetrically for a model server: adding a replica costs a weight load, removing
one throws that load away.

**PersistentVolumeClaim / PersistentVolume** — a request for storage, and the
volume bound to it. **Access modes are per node, not per Pod**: `ReadWriteOnce`
permits any number of Pods on *one* node; the per-Pod guarantee is
`ReadWriteOncePod`.

**`volumeBindingMode: WaitForFirstConsumer`** — the volume is bound only when a
Pod needs it, and for local storage the resulting PV carries a **node affinity**
to that node. Every later replica is then scheduled there too: the volume becomes
a placement constraint, and free accelerators on other nodes are unreachable.

**Job** (batch) — a Pod run to completion. Immutable once created, so changing one
means deleting it first; used here to populate the weights volume before any
server starts.

**`terminationGracePeriodSeconds`** — the window between SIGTERM and SIGKILL. It
buys a **drain** only if the process handles the signal, and the drain is
readiness-first: fail `/health` while still serving, so the Service withdraws the
endpoint before the socket closes.

**PID 1** — a container's entrypoint. The kernel does not apply **default signal
dispositions** to PID 1, so a signal with no installed handler is discarded: an
unhandled SIGTERM turns every rollout into the full grace period followed by
SIGKILL.

**`$patch: delete`** — the kustomize patch directive that removes a base resource
from an overlay, as opposed to modifying it.

**`HF_HUB_OFFLINE`** — forbids the Hugging Face client any network call, so a
pod whose weights volume is incomplete fails immediately instead of hanging on a
download inside the startup probe's budget.

**Ingress controller** — the process that reads Ingress objects and configures a
proxy from them. **ingress-nginx** is the one used here; it holds its backend set
in an in-process Lua table fed by an endpoints watch, so a replica change costs no
config reload.

**IngressClass** / **`ingressClassName`** — which controller owns an Ingress.
Omitted, the object is claimed by nobody and reports nothing: created, and not a
route.

**`hostPort`** — a container port published on its node's own network namespace,
bypassing Service and kube-proxy. Exclusive per node, so one Pod per node can
hold it.

**`extraPortMappings`** (`kind`) — publishes a node container's port on the host.
A docker-level publish, independent of anything listening inside: a connection to
an unserved mapping is accepted and then answered with nothing (curl exit 52),
which is not the same signature as a refusal.

**Admission webhook** — a service the API server calls to validate or mutate an
object before persisting it. **`failurePolicy`** decides what an unreachable
webhook means: `Fail` rejects the object, `Ignore` admits it **unvalidated**.
Both appear in this stack, and the second is the quieter failure — see the
bring-up order in [running-on-kind.md](running-on-kind.md).

**Default backend** — where an ingress controller sends a request matching no
rule. A 404 from it therefore means "no route", as against a **503**, which means
a route whose endpoint set is empty.

**`proxy-buffering`** (ingress-nginx annotation) — whether nginx decouples the
rate it reads upstream from the rate it writes to the client. `off` is this
controller's own default and is pinned here deliberately; what it does and does
not do to a stream was measured in
[`deploy/manifests/README.md`](../deploy/manifests/README.md).

**`proxy-read-timeout`** (ingress-nginx annotation) — the gap allowed between two
successful reads from upstream, not a total. Streaming resets it every decode
step; a non-streaming completion does not, which is why its value is derived from
`--max-model-len` x the TPOT target.

**`proxy-next-upstream`** (ingress-nginx annotation) — which upstream failures
are retried against another replica. Default `error timeout`; the `timeout` half
re-spends a whole generation's accelerator time, and is dropped here.

**`proxy_ignore_client_abort`** (nginx directive) — default `off`, so a client
disconnect closes the upstream connection. What makes an abandoned generation
cancellable at all.

**`load-balance`** (ingress-nginx) — the policy for choosing between a route's
endpoints. Default `round_robin`, applied by the controller itself over Pod
addresses rather than by kube-proxy over a ClusterIP. The default is a fallback
inside the controller's own Lua — **`DEFAULT_LB_ALG`**, used when neither the
annotation nor the ConfigMap names a policy.

**Gateway API** ⏳ — the successor to Ingress: routing as a set of typed
resources with a role split between cluster operator and application owner. Its
**Inference Extension** ⏳ is the LLM-specific part, and the reason the serving
gateways (llm-d, AIBrix, vLLM production-stack) build on it rather than on
Ingress: it can route on model identity and cache locality, which an Ingress
cannot express.

**Prefix-aware routing** (also *affinity*) — choosing the replica that already
holds a request's prompt prefix in its KV cache, instead of choosing by turn. What it is worth is
the gap between the `h` = 0 and `h` = 0.8 columns of [SLO.md](SLO.md) §6; the
implementation and its limits are `router/README.md`.

**Consistent hashing** — placing replicas and keys on one hash ring and giving a
key to the first replica at or after it, so that adding a replica moves only the
arc it claims rather than remapping every key as `hash mod n` does.

**Virtual node** (`-vnodes`) — one of the many ring positions a single replica
occupies. The traffic imbalance between replicas falls with roughly 1/√vnodes;
with one position each, four replicas divide the space into four arcs of
unequal size.

**Bounded loads** (`-bounded-load`, `c`) — the rule that keeps affinity from
overloading one replica: a key's owner is skipped while its in-flight count is
at or above `c` x the fleet mean, and the walk continues around the ring.
`c` = 1 is balance with no affinity; large `c` is affinity with no protection.

**Avalanche step** — a final mixing pass over a hash value, so that inputs
differing in their last bytes differ in their high bits too. Required wherever a
hash is used for *ordering* rather than equality — a consistent-hash ring is
sorted by the high bits, and FNV-1a without this step clusters every position of
one replica into a single arc (`router/README.md` §3).

**`-key-bytes`** (`prefix-router`) — how many leading bytes of the prompt form
the routing key; 0 means the whole prompt, which routes only exact repeats. The
router carries no tokenizer, so the key is bytes and not tokens, and the
approximation this rests on is `router/README.md` §2.

**BPE** (byte-pair encoding) — the tokenizer family these models use: a
deterministic left-to-right merge of byte pairs, which is why an identical byte
prefix yields an identical token prefix up to the token straddling the cut
(`router/README.md` §2).

**`X-Router-Policy`** (`prefix-router`) — the response header naming which of the
six routing policies produced the choice: `prefix`, `round_robin` when the
control arm asked for it, or one of four reasons the request *fell back* to
round robin. The list, and why four reasons and not one, is `router/README.md`
§1; the gate that reads it is `bench/harness.py --expect-policy`.

**`-policy`** (`prefix-router`) — which routing policy to apply to keyed
requests: `prefix`, or `round_robin` as a deliberate choice rather than a
fallback. It exists so that a measurement can compare two policies over one
fleet through one hop, differing in the policy alone
(`docs/benchmarks/runsheets/mi300x-run-3.md` §0).

**`X-Router-Upstream`** (`prefix-router`) — the response header naming the
replica the request was sent to. Read together with `X-Router-Policy`: the
address alone cannot distinguish a prefix hit from a turn taken.

**FNV-1a** — a fast non-cryptographic hash, one XOR and one multiply per byte.
Adequate for equality, and not for ordering without an avalanche step, which is
why a consistent-hash ring built straight on it clusters.

**`-max-body`** (`prefix-router`) — the largest request body the router will hold
in memory in order to read a prompt from it. Past it the request is streamed
through and routed `round_robin`: the bound on what inspecting a payload costs.

**`-drain`** (`prefix-router`) — the gap between failing readiness and closing
the listener on SIGTERM, so that whatever is in front stops sending before the
socket goes. The same ordering the stub uses on `DRAIN_DELAY_S`.

**`-dial-timeout`** (`prefix-router`) — how long the router will spend
discovering that an upstream is gone. A removed Pod address blackholes rather
than refusing, so the default 30 s of Go's `http.DefaultTransport` is spent in
full before the 502: measured 30.006 s against 0.026 s to a live replica, which
is 100× the TTFT budget ([SLO.md](SLO.md) §1). It does not make the request
succeed — there is no retry — it makes the failure arrive inside the budget.

**Blackhole** (of an address) — a destination that neither answers nor refuses,
so a connection attempt to it waits out its own timeout rather than failing. A
Pod address removed from a cluster network behaves this way, which is why a
stale upstream costs a full dial timeout and not a refused connection.

**RTO** (retransmission timeout) — how long TCP waits for an acknowledgement
before resending a segment. Its initial value is 1 s (RFC 6298 §2.1), so a dial
timeout below one second fails a connection whose first SYN was dropped instead
of recovering it — the trade `-dial-timeout` makes.

**`imagePullPolicy`** — when the kubelet fetches a container image. `Never`
requires the image to be on the node already, which is how a locally built
image is used without a registry, and makes a forgotten load an explicit
`ErrImageNeverPull` rather than a stale image silently serving.

**`kind load docker-image`** — copies an image from the host daemon into every
`kind` node. The counterpart to `imagePullPolicy: Never`, and the reason a
compiled component costs `kind` two commands that an interpreted one does not.

**`scratch`** (container base image) — the empty base: the image contains only
what is copied into it. Legal for a `CGO_ENABLED=0` Go binary, which needs no
libc, and it removes the shell — so a container built on it cannot be exec'd
into, and its behaviour has to be observable from outside.

**`paused-replicas`** (`autoscaling.keda.sh/paused-replicas`, annotation on a
ScaledObject) — pins the target Deployment at a replica count and stops KEDA
acting on the metric, without deleting the ScaledObject. The supported way to
hold a fleet still while something other than autoscaling is under examination.

**`FlushInterval`** (Go `httputil.ReverseProxy`) — how often a proxied response
is flushed to the client; `-1` flushes after every write. The Go counterpart of
nginx's `proxy-buffering off`, and required for the same reason: a batched
stream makes ITL a property of the proxy.

---

## Reasoning models

All six terms below matter because they break assumptions the rest of this repo
rests on; the consequences are worked through in [SLO.md](SLO.md) §8.

**Thinking mode** (reasoning mode) — the model emits a chain of thought before its
answer. Qwen3 supports it, toggleable per request.

**Reasoning phase / answering phase** — the two halves of decode under thinking
mode. The reasoning phase produces internal tokens the user does not see; the
answering phase produces the visible reply.

**Reasoning tokens** — tokens generated during the reasoning phase. Billed,
occupying KV cache exactly like visible tokens, and invisible to the user.

**Thinking budget** — a cap on how many reasoning tokens a request may spend before
being forced to answer. The main lever for keeping reasoning workloads inside a
latency and cost envelope.

**Reasoning parser** — server-side splitting of the output stream into
`reasoning_content` and `content` (`--reasoning-parser` in vLLM, Qwen3 supported).
Without it TTFAT cannot be measured, because the server cannot tell which token was
the first visible one.

---

## Operations

**Run** — one paid session on a rented card, numbered in order per card (L40S
runs 1–3, MI300X run 1 so far); each is preceded by a runsheet and followed by a
report.

**Sweep** — a series of benchmark runs that steps **one** parameter through a range
while every other input is held fixed, so the resulting curve is attributable to
that parameter alone. Two are planned for the first run
(`docs/benchmarks/runsheets/l40s-first-run.md`): concurrency at fixed prompt
length, which moves the `batch` factor of `bytes_moved`, and prompt length at fixed concurrency,
which moves the `context` factor. Moving both at once measures nothing.

**`vllm bench sweep`** ⏳ — the engine's own sweep driver, and a different thing
from *Sweep* above: given `--serve-cmd` and `--bench-cmd` it runs the Cartesian
product of `--serve-params` × `--bench-params`, `--num-runs` times each (default
3), resumable with `--resume`. Subcommands at `v0.27.1`: `serve`,
`serve_workload`, `startup`, `plot`, `plot_pareto` — no SLA-driven search among
them. An override from a params file *replaces* the flag if the base command
carries it and is *appended* otherwise; the subcommand's flags are listed only by
`--help=all`. Mechanics and rules: `docs/instrument-vllm-bench-sweep.md`.

**`--dry-run`** (`vllm bench sweep`) — expands the grid and prints every server and
benchmark command without running any; needs no GPU and writes nothing, so the
"existing experiment" guard never fires under it. `bench/sweep/dry-run.sh` runs
it in vLLM's CPU image.

**Serve row** (`vllm bench sweep`) — one entry of `--serve-params`: one server
launch, kept up across every *benchmark row*, one entry of `--bench-params`. Its key names the `SERVE--<key>` part
of each results directory.

**`--link-vars a=b`** (`vllm bench sweep`) — keeps only the rows of the Cartesian
product where serve key `a` equals bench key `b`, e.g.
`max_num_seqs=max_concurrency` turns a grid into its diagonal.

**`--after-bench-cmd`** (`vllm bench sweep`) — the one hook, run after each
benchmark **in place of** the cache reset, not in addition to it. A hook that
reads `/metrics` must reset the caches itself if the next run is meant cold. It
runs under `subprocess.run(check=True)`: a non-zero exit aborts the sweep, so its
last command must be one that cannot fail.

**`--server-ready-timeout`** (`vllm bench sweep`) — seconds the sweep waits for
the spawned server's `/health` to answer 200 before raising `TimeoutError` and
killing it. Default **300** at `v0.27.1`; a cold launch includes any weight
download, `torch.compile` and graph capture, so a measurement run sets it well
above the slowest launch it can imagine.

**`--show-stdout`** (`vllm bench sweep`) — forwards the spawned server's stdout;
without it that stdout goes to `/dev/null`, and vLLM logs to stdout, so the
startup log — the KV pool line [SLO.md](SLO.md) §9 puts above every derivation —
is lost. Mandatory, piped through `tee`.

**`VLLM_SERVER_DEV_MODE`** — environment variable (`=1`) that enables the engine's
development endpoints, `/reset_*_cache` among them. `vllm bench sweep` sets it on
the server it spawns; a server started by hand needs it set to be resettable.

**`--workload-var`, `--workload-iters`** ⏳ — the two knobs of `vllm bench sweep
serve_workload`: which variable carries the load (`request_rate` or
`max_concurrency`), and how many levels are run. The levels are chosen by the
tool — serial inference, then batch inference, then the remainder spread
uniformly between those two.

**Exposition format** — Prometheus's scrape format: `# HELP` and `# TYPE` lines
above `name{labels} value`, served as `text/plain; version=0.0.4`. A **gauge** may
move either way (queue depth), a **counter** only rises (tokens generated). The
label set is part of the contract: a scaler's query selects on it, so a renamed
label breaks autoscaling rather than a dashboard.

**`kubernetes_sd_configs`** (`role: endpoints`) — Prometheus discovering one
target per Pod address behind a Service, rather than scraping the Service's
virtual IP. Scraping the VIP samples one arbitrary replica per scrape; discovery
plus `sum()` in the query is what makes a fleet-wide number.

**Dashboard as code** — the dashboard is a JSON file in the repository
(`deploy/observability/grafana/dashboards/`) that Grafana loads at start-up,
reviewed as a diff and rebuilt on every Pod start; an edit made in the browser
is lost by design. The alternative, a dashboard that lives only in Grafana's
database, cannot be diffed, reviewed or reproduced on a fresh cluster.

**Provisioning** (Grafana) — the file-based configuration Grafana reads from
`/etc/grafana/provisioning/` at start: `datasources/` names where the data comes
from, `dashboards/` names a *provider* that points at a directory of dashboard
JSON. Both are `apiVersion: 1` YAML; `allowUiUpdates: false` on the provider is
what makes the file win over the UI.

**`$__rate_interval`** — a Grafana variable for the window of a `rate()`: at
least four times the datasource's scrape interval, so the window always holds
more than one sample. Not used in this repository's dashboard, whose windows are
the fixed 5 m and 1 h of the alerting rules so that a panel and the rule that
pages read the same number.

**Goodput band** — the server-side bound on goodput when the SLIs arrive as
separate histograms: completions × the smaller of the good shares is an upper
bound, completions × (share₁ + share₂ − 1), floored at zero, a lower one. Which
requests met *both* targets is not in either histogram; the exact figure is the
client's `--goodput`. A thin band is a healthy service.

**`histogram_quantile`** — PromQL's quantile over a histogram's `_bucket` series.
It assumes samples are spread uniformly inside a bucket and interpolates
linearly between edges, so a quantile evaluated at a value that is not a bucket
edge is an estimate the engine never measured. Read the bucket at the nearest
edge instead when the number has to be defended ([SLO.md](SLO.md) §4).

**`/reset_*_cache`** ⏳ — the engine's cache-reset endpoints: `/reset_prefix_cache`,
`/reset_mm_cache`, `/reset_encoder_cache` at `v0.27.1`, available only under
`VLLM_SERVER_DEV_MODE=1`. `vllm bench sweep` calls all three between benchmark
runs unless `--after-bench-cmd` is given, so by default every level of a sweep
starts from a cold cache.

**Pareto frontier** ⏳ — the set of configurations that no other configuration
beats on both axes at once. For serving the two axes are tokens/s per user and
tokens/s per GPU, and `vllm bench sweep plot_pareto` draws the frontier from
sweep results.

**Closed-loop load** — a load generator that keeps exactly N requests in flight,
starting a new one only as one finishes. `vllm bench serve --request-rate inf` with
`--max-concurrency N` is this. It measures the server *at* a chosen concurrency,
which is what a capacity or a decode-step question wants — and it makes every
latency that contains queue time uninterpretable, because the queue is the
generator's own backlog rather than the service's. A TTFT p99 from a closed loop
may never be compared to a TTFT target.

**Open-loop load** — requests issued at an arrival *rate* independent of how fast
the server answers, set with a finite `--request-rate`. Concurrency becomes an
output rather than an input, queueing becomes the service's own, and TTFT becomes
a number an SLO can be checked against. It is also the only way to find the
arrival rate at which goodput collapses.

**`--burstiness`** — the shape of an open-loop arrival process in `vllm bench
serve`, taking effect only when `--request-rate` is finite. At the default 1.0
arrivals are Poisson; below 1.0 they clump, above 1.0 they even out toward a fixed
interval. It matters because a mean arrival rate inside capacity still breaches a
tail SLO if the arrivals clump, so burstiness and rate are two separate statements
about the same load.

**Gated model** — weights whose download requires accepting a licence and being
approved by the publisher. Relevant here because a gated default model would break
the "clone and run in one evening" promise (`README.md`).

**OOM** (out of memory) ⏳ — on a serving GPU this is usually not a crash but an
admission failure: the KV cache cannot fit another sequence. What to check first
is in [runbook.md](runbook.md), and why raising `gpu_memory_utilization` is not the
fix is [SLO.md](SLO.md) §6.

**Canary rollout** — sending a small share of traffic to a new version and
promoting or rolling back on a metric rather than on inspection. Here the share
is a canary weight at the edge, the metric is an alerting rule that names the
track, and *promotion* is the stable track taking the canary's version while the
canary carries the traffic. Procedure: [runbook.md](runbook.md).

**Canary weight** — the share of one Ingress rule's requests that ingress-nginx
sends to a second Ingress marked `canary: "true"` for the same host and path,
`nginx.ingress.kubernetes.io/canary-weight`, out of 100. Drawn per request, not
per client: a streamed completion stays on the track it landed on for its whole
life, and the client's next request is drawn again. At most one canary Ingress
per rule.

**Track** — one of the two copies of the serving Deployment behind a single
path, *stable* or *canary*, each with its own Service. Told apart by
`app.kubernetes.io/name` on the Pods and, in every metric, by the `service`
label Prometheus stamps from the Service name — which is what lets a rule name
the track and the autoscaler read only the one it scales.

**Droplet** — DigitalOcean's name for a VM; a *GPU Droplet* is one with a card
attached and direct SSH, no cluster in between. Billed per second from creation
until **destroyed** — a powered-off droplet keeps its disk, RAM and address
reserved and keeps billing. Module: `deploy/terraform/`.

**Droplet check** — the first level of an MI300X run: one request at a time,
4 000 tokens, read as a median ITL against the figure each earlier droplet
logged, to say whose fitted coefficients apply. Two droplets of one product
have differed by 21 % on the decode step, and a run cannot choose its droplet.

**AMD Developer Cloud** — DigitalOcean GPU Droplets behind AMD's own front door:
the console at `amd.digitalocean.com`, the API at `api-amd.digitalocean.com`.
Same API shape, different host, different contract ($1.99/h against $2.59/h for
the same MI300X), and plan slugs with a `-devcloud` suffix. A token issued by
that console is valid only against that host. MI300X region: ATL1.

**Account credit** (DigitalOcean) — a balance applied to the *next invoice*
rather than a prepaid wallet: usage bills as normal and the credit settles the
bill afterwards, so the account still needs a payment method. It carries an
expiry date and does not apply to **Add-Ons** — paid extras billed beside a
resource; a GPU Droplet is not one.

**Slug** (size slug, image slug, region slug) — the string the API accepts for a
plan (`gpu-mi300x1-192gb`), a base image (`gpu-amd-base`) or a datacenter
(`atl1`). Read from the API — `doctl compute size list`, `image list` — never
from memory: the same plan has a different slug under the AMD front door.

**Terraform provider** — the plugin that maps one API's objects to Terraform
resources; here `digitalocean/digitalocean`, pinned `~> 2.100` (2.100.x only).
Its `api_endpoint` argument moves every call to another host — the one
AMD-specific line in `deploy/terraform/`. The token is read from
`DIGITALOCEAN_TOKEN`, never written into a tracked file.

**`validate` → `plan` → `apply` → `destroy`** (Terraform) ⏳ — the ladder, and
what each step needs: `validate` is offline (syntax, types, references);
`plan` needs a token and creates nothing; `apply` creates and starts billing;
`destroy` is the only exit for a GPU droplet. `deploy/terraform/` has climbed to
`validate`; the first `plan` waits for the credits.

**cloud-init** (`user_data`) ⏳ — the first-boot script a droplet runs once, as
root, *after* SSH is up, so the operator can log in while it works.
`cloud-init status --wait` blocks until it has finished and is the gate before
the first `docker run`. Changing it forces a new droplet. Rendered here by
`templatefile()` with the vLLM image tag; `deploy/terraform/cloud-init.yaml`.

**`doctl`** — DigitalOcean's CLI; `--api-url` selects the host, and without
`https://api-amd.digitalocean.com` it talks to a paid DigitalOcean account, not
to the credits.

**`--dry-run`** (`bench/harness.py`) — a different flag from the `vllm bench
sweep` one above: it expands the scenario and prints each level's predicted
prefill floor against the chosen accelerator, sending no request and needing no
server, cluster or card. The floors come from `bench/roofline.py`.

**`--accelerator`** (`bench/harness.py`, `bench/predictions.py`) — which
`Accelerator` instance the floors and gates are computed against: `l40s-run1`
and `mi300x-run1` carry coefficients fitted to each card's run 1, `l40s` and
`mi300x` carry the unvalidated priors. Which is which matters more than the value; `docs/SLO.md` §9. The names
come from the `ACCELERATORS` registry in `bench/roofline.py`, which is where a
card is added.

**`--model`** (`bench/predictions.py`) — which architecture the floors are
computed for: `qwen3-8b`, the model this stack serves and the only one any run
here has measured, or a second entry priced but never served — see *Predicted
only*. The names come from the `MODELS` registry in `bench/predictions.py`, and
every model in it must be dense, GQA and full-attention, because the formulas
above assume exactly that (*Architectures that break the standard arithmetic*).

**`--what-if`** (`bench/predictions.py`) — prints one operating point of the
reader's choosing — card, context length, prompt length, TPOT and TTFT targets,
`gpu_memory_utilization`, KV dtype, hourly rate — instead of the twelve fixed
tables. The tables take no parameters on purpose: `docs/SLO.md` quotes their
rows, so a flag that moved them would be a flag that edits a derivation. Every
line it prints is still a floor.

**`--hit-rate`** (`bench/predictions.py --what-if`) — the prefix cache hit rate
`h` the operating point is priced at, 0..1. Moves the uncached TTFT floor
(prompt × (1 − `h`) tokens) and the service seat count, and nothing a decode
step reads.

**`--json`** (`bench/predictions.py --what-if`, `bench/harness.py --dry-run`) —
the same answer as data: the operating point as the dict `what_if_point()`
returns, or the dry-run plan with its per-level verdicts. Both are the forms the
site's export and its golden grid are built from.

**`--metrics-endpoint`** (`bench/harness.py`) — an engine to scrape `/metrics`
from, repeatable, separate from the address the load is sent to. Needed the
moment a router sits in front of a fleet: the load endpoint is then one hop, and
the counters live on several engines behind it
(`docs/benchmarks/runsheets/mi300x-run-3.md` §0).

**`--reference-pool`** (`bench/harness.py`) — the logged KV pool, in tokens,
that a level's pool gate compares the engine's startup log against. Defaults to
the L40S's run-1 pool, so any other card must pass its own.

**`--expect-policy`** (`bench/harness.py`) — the `X-Router-Policy` every
response of a level must carry, or the level is invalid. Turns a routing arm
from something believed into something checked: an arm that silently fell back
to round robin produces a complete, plausible level otherwise.

**`--no-warmup`** (`bench/harness.py`) — skips the request that seeds a level's
shared prefix. Only for measuring a cold cache deliberately: without the seed the
`h` a level measures is not the `h` its construction intends, and the harness
stops rather than reporting one for the other.

**`--warmup-passes`** (`bench/harness.py`) — how many times the warmup is sent.
One pass seeds a prefix on the one engine the router picks; under round robin
over R engines it takes R passes, with N coprime to R, for every prefix to be
seeded on every engine.

**`--reset-cache`** (`bench/harness.py`) — `POST /reset_prefix_cache` to every
metrics endpoint before a level's warmup, so no earlier level's requests can be
hit. Needs `VLLM_SERVER_DEV_MODE=1`; a refusal is raised, not skipped.

**Stub fixtures** (`MAX_NUM_SEQS`, `SIM_DECODE_MS`) — the seat count and simulated
milliseconds per token of the `kind` stub
(`deploy/manifests/overlays/kind/patch-stub.yaml`). They set the behaviour a
laptop can watch and measure nothing; no figure derived from them belongs in
`docs/benchmarks/`.

**GitHub Pages** — GitHub's static hosting: the files under `site/` served at
`https://<account>.github.io/<repository>/`, deployed by
`.github/workflows/pages.yml` once the repository's Pages source is set to
GitHub Actions. Public for a public repository; a private repository cannot
enable it on a Free plan.

**`PAGES_ENABLED`** — a repository variable (Settings → Secrets and variables →
Actions) the deploy job of `.github/workflows/pages.yml` is gated on, so the
private working repository skips the deploy instead of failing it. Set to `true`
on the public repository only.

**Commit guards** — the two refusals `.githooks/pre-commit` adds to the
regeneration it performs, both judged over the index so that a commit cannot be
wrong rather than be caught after a red build: no status glyph — ✅, ⚠️ or ⏳ —
in `symptom-map.md` or `symptom-map.json`, because the map is a decision tree and
not a progress board; and no non-English text in repository content
(`CONTRIBUTING.md`, *What a reviewer enforces*). The `guards` job of
`.github/workflows/bench.yml` runs the same two over a checkout, because hooks
are not carried by `git clone` and a contributor's branch is checked there and
nowhere else.

**Symptom map** — the operator's decision tree in `symptom-map.md`: seven
breaching symptoms, and under each the number to read first, the branches it
splits into and the traps; one line per node, each ending in a stable id
(`<!-- ttft.queue.replicas -->`) and a pointer to where its number lives.
`symptom-map.json` is the subset the site's advisor evaluates as rules, held
equal to the markdown by a test.

**Evidence marks** (symptom map) — the bold word on a node saying how it is
known: **measured** on a rented card, **seen** once in a run, **watched** or
**watched firing** on `kind`, **checked** against the live objects, **run once**
as a procedure. A node with no bold word is derived and waits for a run.

**Advisor** — the page on the Pages site that evaluates `symptom-map.json` as
rules over the dashboard's readings and ends at a branch of the symptom map, the
number to read next, the knobs and the traps. It reads no metric; a reading
left empty is *No data*, and a rule over it says "cannot say" rather than a
verdict.

**Calculator** — the page that runs the arithmetic of `bench/roofline.py` in the
browser for a card, a target, a prompt length, a hit rate and a rate of the
reader's own. Its numbers are floors, as the model's are; the two presets are
the derived classes of [SLO.md](SLO.md) §2, and any other target is the reader's
contract.

**Parity test** (golden grid) — a test that a second implementation reproduces a
first one's outputs on fixed inputs. Here: `bench/export_site_data.py` writes
`site/data/golden.json`, rows of `predictions.what_if_point()` over a grid of
inputs, and `site/selftest.js` runs `site/calc.js` on every row and compares —
in CI under `node`, and in the page on every load, as the badge.

**Runsheet** — the per-run checklist written before a run: its commands, its
predicted numbers, its stop conditions and its budget. Ships unedited beside
the report (`benchmarks/runsheets/`), so the before-the-fact claim can be
judged rather than taken.

**Predicted-vs-measured table** (§9 table) — one row per prediction a run faced,
naming the coefficient that produced it, whether that coefficient was fitted to
the same run, and the card. Rules in [SLO.md](SLO.md) §9; the front-page chart
is a drawn selection of the runs' tables.

**Stall guard** — the per-token threshold of the batch class, named for what it
is: a catch for a pathological stall, not a promise anyone reads. Setting it
tightly caps the batch size and raises cost per million tokens to improve a
metric nobody observes, and loosening it buys only what the *room* limit has
left to give — nothing at all on a card where capacity already binds
([SLO.md](SLO.md#1-workload-classes) §1, [§6](SLO.md#6-concurrency-ceiling)).

**Workload class** — interactive or batch: a pair of TTFT / TPOT targets and the
SLO that actually binds each ([SLO.md](SLO.md) §1–2). Both are derived there;
a class added without a derivation would be a target picked by feel. In code,
`SLO_CLASSES` in `bench/predictions.py`.

---

## Architectures that break the standard arithmetic

The rules of thumb above (`2 × P × B` for FLOPs, `2 × n_kv × head_dim × bytes × L`
for KV per token) assume a dense, GQA, full-attention transformer. Four families
break them. Check which one you are holding before reaching for a formula.

**MoE (Mixture of Experts)** — only a subset of experts fires per token, so the
parameter count splits in two. Memory must use **total** parameters, since every
expert is resident in HBM; FLOPs must use **active** parameters. Config fields:
`num_experts`, `num_experts_per_tok`. Using total params for FLOPs overestimates
prefill time by roughly the expert ratio.

**MLA (Multi-head Latent Attention)** — DeepSeek-V2 and later compress keys and
values into a low-rank latent instead of storing them per head. The KV formula does
not apply; size is driven by `kv_lora_rank`. The cache is dramatically smaller,
which changes the concurrency ceiling entirely.

**Sliding window attention** — Mistral and Gemma bound attention to a fixed window,
so KV stops growing once context exceeds `sliding_window`. The linear
`KV ~ batch × context` relation holds only up to the window size.

**Quantization** — `bytes_per_param` is not fixed: FP8 weights are 1 byte. KV can
also be quantized independently of the weights (`kv_cache_dtype` in vLLM), so the
weight term and the KV term of `bytes_moved` may carry different multipliers.

In all four cases the check is the same one that applies to every derivation here:
vLLM's startup log outranks the arithmetic ([SLO.md](SLO.md) §9).
