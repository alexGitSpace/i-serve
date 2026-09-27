# Who this is for, and what it will and will not give you

This repository is a slice of an LLM inference operator's platform, built small
on purpose. It is written for **one reader**: an engineer who wants to understand
the economics and behaviour of LLM inference, and who checks that understanding
by changing something and watching what moves.

Everything here is arranged for that reader. If you are skimming to judge whether
the work is real, the first route below is the short one; if you are looking for
a single formula or a manifest to copy, the headings in [SLO.md](SLO.md) and the
`README.md` under each `deploy/` directory are stable anchors and you can deep
link to them.

## Four routes

### Thirty seconds, in a browser

The calculator and the advisor, on the Pages site: the arithmetic of
`bench/roofline.py` in vanilla JavaScript, for a card, a target, a prompt length,
a hit rate and a rate of your own, with the measured points of runs 1–3 drawn
over it where the sliders sit inside the runs' geometry; and
[symptom-map.md](symptom-map.md) walked as rules over the readings you type in
from the dashboard. Its claim to be this repository's arithmetic is a parity
check — a golden grid Python writes, re-run by the page on every load and by CI
before every deploy (`site/README.md`).

The honest limit in one sentence: it reads nothing — no cluster, no metric — so
it prices a target and asks you for the numbers; it cannot see whether you are
meeting the target.

### Five minutes, nothing installed

Run the performance model. It needs no GPU, no cluster, no server and no
third-party packages — `bench/` is Python 3.10+ standard library only:

    python3 bench/harness.py --scenario smoke --accelerator l40s-run1 --dry-run
    python3 bench/harness.py --scenario smoke --accelerator mi300x  --dry-run

The two commands print the same workload against two different accelerators, and
the prefill floor changes because the arithmetic in `bench/roofline.py` changed
with the card. That arithmetic is the centre of this repository; the rest of it
exists to find out where the arithmetic is wrong.

Read the coefficients in the header line. They are not interchangeable — see
*Measured, derived, assumed* below.

Then change something that is not the card. `bench/predictions.py` prints ten
fixed tables with no arguments — the operating points this repository argues
about — and one operating point of your own with `--what-if`:

    python3 bench/predictions.py --what-if --accelerator l40s-run1
    python3 bench/predictions.py --what-if --accelerator mi300x
    python3 bench/predictions.py --what-if --context-len 32000 --kv-dtype-bytes 1

The line worth watching is `max_num_seqs`, because it names which constraint you
are actually against:

      max_num_seqs          31   bound by latency, gap 1.39x
      max_num_seqs          273  bound by latency, gap 1.00x

The first card runs out of *time* — it cannot re-read that much KV every token
and still land inside the TPOT target — and on the second, time and *room* run
out at the same seat: the gap is 1.00, a tie. That is a property of a card
against a target, never a rule about GPUs, and it is the distinction
[SLO.md](SLO.md) §4 and §6 exist to draw. `--help` lists the rest:
context length, prompt length, both targets, `gpu_memory_utilization`, the KV
dtype and the hourly rate.

Every line it prints is a floor — the best the hardware could do. A real run
lands above it, and the size of that gap is what [benchmarks/](benchmarks/)
measures.

Then, if you read one document: [SLO.md](SLO.md) §4 (the floors) and §6 (the
concurrency ceiling).

### An evening, on your laptop

Bring the whole stack up on a CPU-only `kind` cluster — ingress, queue-depth
autoscaling, Prometheus, Grafana and a vLLM-shaped workload — with no
accelerator and nothing to pay for. The command order is in
[running-on-kind.md](running-on-kind.md).

What you can learn this way is **what the system does**, not what the numbers
are. That distinction is not a disclaimer; it is the design, and the next
section is the contract.

### Deep

The derivations are the point, and they are long. Reading order:

1. [model-anatomy.md](model-anatomy.md) — where the bytes come from.
2. [SLO.md](SLO.md) — targets, floors, ceiling, cost, and §9 on how each
   assumption gets validated or falsified.
3. [benchmarks/](benchmarks/) — three L40S runs, each with a predicted-vs-measured
   table and a section on what the run could **not** measure.
4. [architecture.md](architecture.md) — the request path, and where each SLO is
   won or lost along it.
5. [runbook.md](runbook.md) — canary rollout, morning triage.
6. [symptom-map.md](symptom-map.md) — where the runbook's row 3 continues: seven
   symptoms, the number to read first, the branches, the traps.
7. [adding-a-run.md](adding-a-run.md) — if you want to add to it.

[GLOSSARY.md](GLOSSARY.md) defines every term, symbol and config knob used above.

## What is real on `kind`, and what is not

The container on `kind` is not vLLM. It is a stub
(`deploy/manifests/overlays/kind/stub/server.py`) that reproduces the engine's
*platform contract* — its API shape, its health semantics, its metric names and
labels, its drain behaviour — and serves no model.

**Real, and worth watching:** a FIFO waiting queue in front of a bounded set of
seats; back-pressure expressed as latency rather than a 429; server-sent events,
which is the only shape in which TTFT is observable at all; readiness-first
draining on `SIGTERM`; and, above the stub, every Kubernetes object in this
repository behaving exactly as it would on a GPU node — ingress routing, KEDA
scaling on queue depth, Prometheus scraping per pod, an alert moving from
pending to firing, a canary splitting traffic at the edge, and — on a second
host only — a prefix router choosing a replica by prompt and holding to it
(`deploy/router/`). What that last one establishes is *which replica*, and
nothing else: no TTFT, no seat count, no cache hit rate. Those need a card.

**Fixtures, which mean nothing:** the stub's seat count and its simulated
milliseconds per token (`MAX_NUM_SEQS`, `SIM_DECODE_MS` in
`overlays/kind/patch-stub.yaml`). Changing them changes the behaviour you can
watch and carries no information about any accelerator. No number produced on
`kind` appears in [benchmarks/](benchmarks/), and none should be quoted.

**Absent, deliberately:** latency histograms, KV cache utilisation, preemption
counters and token counters. The stub exports two gauges and nothing else,
because a fabricated histogram would light eleven dashboard panels and mean
nothing, where an absent series is merely absent. Which five panels of sixteen
light on `kind` is the table in `deploy/observability/README.md`.

**Never run at all:** `deploy/manifests/base/` is the GPU-node shape — the
engine's image, the Job that fetches the weights, the accelerator request — and
no card has ever been sent it. The runs in [benchmarks/](benchmarks/) were made
against a server started by hand, with no Kubernetes anywhere in the picture
([architecture.md](architecture.md) §6). So what is unvalidated here is not the
stub, which is honest about what it is; it is the *pairing*. The engine and
these manifests have never run together.

**Consequence worth stating plainly:** of the three questions the repository
promises to answer, exactly one — *what happens on a traffic spike* — can be
demonstrated on `kind`, and it has two continuations: [symptom-map.md](symptom-map.md),
which carries an alert from the runbook into a branch, and the advisor on the
Pages site, which walks the same tree as rules. Neither reads a metric. On
`kind` the histogram panels are *No data*, so the advisor is asking for numbers
that do not exist there; on a card you read them off row 3 of the dashboard and
answer. *Are we inside SLO* still needs the engine's histograms — the calculator
prices a target and cannot say whether it is met — and *what does a million
tokens cost* needs a card. Those two are answered in [benchmarks/](benchmarks/),
on rented hardware.

### Generating load on `kind`

`bench/harness.py` is a measurement instrument for a real engine and will not
work against the stub: it counts a token by the arrival of a non-empty text
field, and the stub refuses to invent one. Use the shell loop in
`deploy/observability/README.md` instead, which is how the queue breach and the
1 → 4 → 1 scaling in that file were produced.

## Changing the model

The served model is `Qwen/Qwen3-8B`, chosen so that nothing in the quick-start
path needs an account, a token or an approval.

**The calculator is the exception, and only because it deploys nothing.** The
page on the Pages site can price a second architecture — it does arithmetic over
`config.json` figures and nothing else — so it offers one, marked *predicted
only*, for the comparison the next paragraph is about. What it does **not** do
is lend that model anything a run measured: the interference fit belongs to the
model it was measured decoding, so the second architecture's *service* answer is
"not derivable", and no measured point is ever drawn over it. Pricing a model is
not serving it.

Swapping the **served** model is **not a string substitution**, and this repository
deliberately does not offer a `MODEL_ID` that pretends otherwise. Three strings name the
model, and six numbers are *derived* from it:

| What | Where | Derived from |
|---|---|---|
| `--max-model-len` | `deploy/manifests/base/deployment.yaml` | context, KV bytes per token, pool size |
| `proxy-read-timeout` | `deploy/manifests/base/ingress.yaml` | `--max-model-len` × the TPOT target |
| PVC size | `deploy/manifests/base/pvc.yaml` | weight bytes |
| KEDA threshold | `deploy/keda/base/scaledobject.yaml` | the concurrency ceiling, [SLO.md](SLO.md) §4 |
| Queue alert threshold | `deploy/observability/prometheus/rules.yml` | the same ceiling |
| `Model(...)` | `bench/roofline.py` | `config.json`: layers, KV heads, head dim, parameters |

### Gated weights, and why the default is not one

| Model | Licence | Access | Role here |
|---|---|---|---|
| `Qwen/Qwen3-8B` | Apache 2.0 | open | default, used by every quick-start path |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 Community | gated, needs `HF_TOKEN` | documented alternative |
| `Qwen/Qwen2.5-7B-Instruct` | Apache 2.0 | open | the calculator's second architecture; never served here, never measured |

Some weights — `meta-llama/*` among them — are *gated*: the publisher wants an
accepted licence and an approved request before the download works. Serving one
here is fine, but it adds prerequisites to whoever runs the stack: a Hugging Face
account, an approved request, and an `HF_TOKEN` in the cluster. That is three
things standing between `git clone` and a running service, and it is the whole
reason the default is an open model instead.

One consequence for reading results: every report in `docs/benchmarks/` names the
model that produced it, because architecture differences change the arithmetic.
Figures from two models are not interchangeable, however similar the parameter
counts look. The calculator's second model is there to make that concrete rather
than to be believed: Qwen2.5-7B carries four KV heads over 28 layers against
Qwen3-8B's eight over 36, so its KV per token is 2.6× smaller and the same L40S
at the same 50 ms target seats 85 of it against 31 — two models of the same
advertised size, a seat count that differs by 2.7×. The parameter count was
never the number that decided it.

A last limit on that list: every model on it is dense, GQA and full-attention,
because the formulas in `bench/roofline.py` assume exactly that. The four
families that break them are named in [GLOSSARY.md](GLOSSARY.md), *Architectures
that break the standard arithmetic*; adding one of those to the page would
produce confident wrong numbers, which is worse than offering no choice at all.

A knob that changed the three strings and left the six numbers pointing at the
old model would produce a stack that comes up, serves, autoscales — and is wrong
about every figure it reports. The arithmetic to recompute all six lives in
`bench/roofline.py` and is not yet wired to the manifests; until it is, changing
the model means changing both, and the table above is the checklist.

## Measured, derived, assumed

Three kinds of number live here and they are never mixed:

- **Measured** — produced by a run, with raw evidence committed under
  `docs/benchmarks/raw/`. On the accelerator side only `l40s-run1` carries
  measured coefficients: `eff_mem = 0.83` from twelve decode levels, `mfu = 0.439`
  from one uncontended prefill.
- **Derived** — computed from architecture and vendor specifications. Every floor
  and ceiling in [SLO.md](SLO.md) is derived, and each is a prediction until a run
  faces it.
- **Assumed** — a prior standing in for a measurement not yet taken. The `l40s`
  and `mi300x` entries in `bench/roofline.py` carry `0.70` and `0.45`, marked
  `unvalidated` at the line. Numbers computed from them are the shape of an
  answer, not its size, and comparing an assumed card against a measured one
  compares two different kinds of thing.

Adding an accelerator is one run and one registry edit, and two of its six
fields can only come from the run: [adding-a-run.md](adding-a-run.md). That is
why this repository knows three accelerators rather than thirty.

Where a measurement and a derivation disagree, the measurement wins and the
derivation is corrected in place; [SLO.md](SLO.md) §9 records how each assumption
is meant to fail.
