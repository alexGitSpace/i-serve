# Contributing

This repository is a ledger of inference efficiency: predictions faced by
measurements, kept small on purpose. A contribution is welcome when it adds a
row to that ledger or corrects one; the shape of the work is fixed, and this
page says what it is.

## What a contribution is

- **A run.** The unit of contribution here is one run on one card: a runsheet
  with its predictions written *before* the card is rented, the raw results
  after, two fitted coefficients, and one row per prediction in a §9 table. The
  whole procedure, and why runs are what accumulate here:
  [docs/adding-a-run.md](docs/adding-a-run.md).
- **A probe.** One engine, one weight file, one device you own: the ceiling
  read from the file's bytes and registered before the measurement, the
  measurement with the engine's version and the host's state beside it, and
  the share of ceiling in the read-out. The template is the Mac sheet
  ([docs/benchmarks/runsheets/mac-probe-1.md](docs/benchmarks/runsheets/mac-probe-1.md)),
  including the way it stopped.
- **A corrected number, with the evidence that corrects it.** Measurement
  outranks derivation ([docs/SLO.md §9](docs/SLO.md#9-assumptions-and-how-they-get-validated)):
  a derivation a run has shown wrong is corrected in place, the §9 table gains
  the row that showed it, and nothing is quietly re-fitted.
- **A documentation fix.** A dangling link, a stale state cell in
  [docs/layout.md](docs/layout.md), a term used without an entry in
  [docs/GLOSSARY.md](docs/GLOSSARY.md).

## What it is not

- **A model without a probe.** A model enters the ledger measured, on a named
  engine and device; a `config.json` alone is priced by the calculator as a
  prediction and marked as one. Swapping the model the `kind` half serves is
  still not a string substitution
  ([docs/audience.md](docs/audience.md#changing-the-model)), and that half no
  longer grows.
- **A public claim as a row.** A vendor's or a provider's figure is checked on
  paper against the ceiling and lives in its own table, never in the ledger: a
  row is taken with a run or a probe. Where a claim omits the precision, the
  context or the batch, the verdict is *condition not disclosed*, not a number.
- **A card without a run.** A new `Accelerator(...)` with spec-sheet coefficients
  is welcome as a *prior* — `provenance="prior, unvalidated; no run on this card"`
  — and it will be labelled as one on every page it prints. It becomes a
  measured card only through a run.
- **A stub that fabricates a histogram.** The `kind` stub exports two gauges and
  nothing else, so that eleven dark dashboard panels stay dark rather than lit
  by numbers that mean nothing.
- **A number produced on `kind` quoted as a measurement.** Anywhere.

## What CI enforces

`.github/workflows/bench.yml`, on every push, installing nothing:

- `bench/` is Python 3.10+ standard library only — the tests run on 3.10 and
  3.14 with no `pip install`, and a dependency that became necessary is what
  the workflow fails on;
- the committed front-page chart is what its data produces;
- every report in `docs/benchmarks/` registers its run in its front matter,
  and the run journal and the front page's runs table are what `bench/runs.py`
  produces;
- the committed `site/data/` is what `bench/export_site_data.py` produces, and
  `site/calc.js` agrees with Python on every golden row;
- the committed ledger is what `bench/ledger.py` produces from the modules
  that score each run;
- every node of `docs/symptom-map.json` is a line of `docs/symptom-map.md`, and
  every link in either resolves.

## What a reviewer enforces

The conventions in `CLAUDE.md`, in short: one home per fact, and other files
link to it; SI base units in derivations; dense FLOPS rows, never the sparsity
row; architecture from `config.json`; a derived floor never reported as a
measurement; repository content in English.

One consequence of the first of those is worth spelling out, because it is the
failure mode a repository of pointers actually has: **a fact that moves is
followed by a `grep` for its old address.** CI cannot catch this. A pointer into
a file that still exists resolves cleanly while naming a fact that left it, and
that is how three files came to say the bring-up order lived in `README.md`
after it had moved to [docs/running-on-kind.md](docs/running-on-kind.md).

And one rule about the evidence rather than the prose: **no file carries a
credential.** Raw logs are captured terminal, so a key can arrive in a pull
request without anyone typing it. `.githooks/no-secrets.sh` refuses the commit,
runs again on a push over every object being sent, and runs in CI on a branch
no hook ever saw; `bench/tests/test_no_secrets.py` is what keeps its patterns
from going quietly inert.

## How to send it

1. **The runsheet first, as its own pull request, before the run.** In the
   public repository the merge date is the dateline, and a sheet merged before
   its results exist is a stronger before-the-fact claim than any sentence in
   the report. The MI300X sheet in `docs/benchmarks/runsheets/` is the
   template.
2. **The results and the report as a second pull request**, with the raw
   evidence under `docs/benchmarks/raw/`, the §9 rows, the coefficient fit, the
   registry edit, the report's front matter, and the regenerated chart, site
   data, run journal and ledger (`.githooks/pre-commit` does all four if
   enabled). A probe follows the same two steps, with a single-stream
   geometry, and its sheet records the device's thermal state beside each
   row.

## Licence

Apache-2.0, the licence of the served model. Contributions are accepted under
the same.
