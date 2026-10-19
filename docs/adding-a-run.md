# Adding a run — or a card

The checklist for the one kind of contribution this repository is built to
receive, and the argument for why it is that one.

## 0. Why runs are what accumulate here

Every card in `ACCELERATORS` (`bench/roofline.py`) is a prior until a run makes
it measured, and every floor and ceiling in [SLO.md](SLO.md) is a prediction
until a run faces it. The repository is a **ledger of predictions faced by
measurements** — the chart on the front page is that ledger drawn — and what a
run adds is not a number but a row where a prediction could have been wrong.

A coefficient a run fitted is a hypothesis until a *later* run it did not see
confirms it ([SLO.md §9](SLO.md#9-assumptions-and-how-they-get-validated)):
`eff_mem` 0.83 was fitted on run 1 and stood on runs 2 and 3, which is the
only reason the front page may quote a floor built on it. That is why one run
is never enough, why the same card is worth a second run, and why the unit of
contribution is a run rather than a card. Two of an accelerator's six fields
cannot come from a datasheet at all: `achieved_bandwidth` and `mfu` are what a
run produces, and that is why this repository knows three cards rather than
thirty.

What a run on a new card answers first is narrow, and should be: does the
arithmetic transfer? The MI300X sheet
([benchmarks/runsheets/mi300x-run-1.md](benchmarks/runsheets/mi300x-run-1.md))
does for that card what run 1 did for the L40S and nothing more — twelve
levels, two coefficients, a §9 table — and its fitted figures become the
hypothesis for MI300X run 2.

## 1. Before the card: the runsheet

Written before any money is spent, and shipped unedited beside the report so the
before-the-fact claim can be judged rather than taken. The template is the
reviewed MI300X sheet; its parts, in order:

- what earlier runs established, and which of it transfers to this card;
- numbered steps with a clock, and the budget in hours × the hourly rate read
  off the provider's console that morning;
- **Checkpoint A** — the startup log against the arithmetic: the KV pool the
  engine logs outranks the pool this repository derives, and the gate is ±5 %,
  because two launches of one config differed by 4.2 %
  ([benchmarks/l40s-run2.md §6](benchmarks/l40s-run2.md#6-what-the-run-measured-about-its-own-instrument));
- **predictions, row by row, from `python3 bench/predictions.py`** — never typed
  by hand; `--what-if --accelerator <key>` for the operating points the run
  will visit, with the coefficients' provenance printed beside them;
- stop conditions, the drop order if the clock runs short, and what "done"
  means;
- what this run cannot answer, and where each of those questions goes.

Commit the sheet before renting. In the public repository, open it as its own
pull request ([CONTRIBUTING.md](../CONTRIBUTING.md)).

## 2. On the card: what to capture

Everything lands in `benchmarks/raw/<card>-<date>/`, with a README in the shape
of [raw/l40s-2026-08-30/README.md](benchmarks/raw/l40s-2026-08-30/README.md):
one table of paths, and what each directory is.

- **The startup log, always.** It is the primary record
  ([SLO.md §9](SLO.md#9-assumptions-and-how-they-get-validated)) and the one
  file that can invalidate a whole sweep.
- **The harness's own output.** `bench/harness.py --out results/<run>` writes
  `run.json`, one JSON per level and one `-metrics.json` of Prometheus deltas
  per level; `bench/measured_run3.py` reads those files directly, with no
  hand-typed derivative to verify against. `vllm bench sweep` is the other
  instrument, and [instrument-vllm-bench-sweep.md](instrument-vllm-bench-sweep.md)
  says where it belongs.
- **Read what you captured before you stage it.** A log is written by the
  tools, not by you, and two ordinary moves put a credential in one: a
  benchmark client prints its own arguments, so a `--header "Authorization:
  Bearer …"` ends up in the `tee`d output beside the numbers; and a shell
  that dumps `/proc/1/environ` to find one variable dumps all of them.
  `.githooks/no-secrets.sh` refuses such a commit, but it is the second line of
  defence — pass the key as `$VAR`, and print a verdict rather than a value.
- **Count files before and after `git add`.** `.gitignore` ignores `*.log` and
  `results/` and then negates both under `benchmarks/raw/` — because nine
  startup logs were once dropped silently, on the day the volume holding the
  only other copy was deleted. A file count caught it; an eye did not.

## 3. After: two coefficients

- `achieved_bandwidth` (`eff_mem`) — from the **median ITL** across the sweep's
  levels, not the TPOT: the median step is the closest a benchmark reports to a
  decode step alone, and under chunked prefill the TPOT carries other requests'
  prefill. The fit stops being valid where the median ITL stops being a decode
  step ([benchmarks/l40s-run2.md §6](benchmarks/l40s-run2.md#6-what-the-run-measured-about-its-own-instrument)).
- `mfu` — from one uncontended prefill, a single request alone on the card.

Report both against the **uncalibrated prior first** — the §9 rule — so the
table shows what the spec-sheet assumption predicted and by how much the card
disagreed. One `bench/measured_<run>.py` per run, never merged into another's:
each read-out names the files it reads and the geometry they were taken at.

## 4. The §9 table, and the drawing

One row per prediction the run faced: the prediction, the measurement, the
coefficient that produced the prediction, whether that coefficient was fitted to
this same run, and the card. The rules are in
[SLO.md §9, *The rules for the table*](SLO.md#the-rules-for-the-table-that-scores-all-of-this)
and are not restated here; the shape is §9 of any report in
[benchmarks/](benchmarks/).

Then the picture: add the run's two-sided rows to `bench/plot_predicted_vs_measured.py`
with both cells copied verbatim from the table, and run the generator. The test
asserts each row still matches one line of the report, and CI diffs the
committed SVG against what the generator produces.

And the registration: the report opens with its front matter, which is how the
run reaches the chart, the site and the run journal. The shape, from
[benchmarks/mi300x-run5.md](benchmarks/mi300x-run5.md):

```yaml
---
card: mi300x
run: 5
date: 2026-10-17
asked: What makes two engines on one card equal, and does a 70B obey run 2's coefficient?
found: one hardware queue per process ends the asymmetry, and the 70B's decode step held to run 2's eff_mem at TP = 1
predicted_with: mi300x-run2
---
```

`card` is the card's prior key, `predicted_with` and `fitted` are
`ACCELERATORS` keys, and `fitted` is there only if the run produced
coefficients. `asked` and `found` are one sentence each, in a reader's words:
they are what the site and the front page show for the run. Plain text only —
GitHub reads the block as YAML, so a value may not contain `: ` or ` #`, and
`bench/runs.py` refuses one that does. Then `python3 bench/runs.py` rewrites
[benchmarks/README.md](benchmarks/README.md) and the front page's runs table.

## 5. The registry edit

One `Accelerator(...)` in `bench/roofline.py` and one key in `ACCELERATORS`:

- four fields from the vendor's table, **dense rows and never the sparsity row**
  — `name`, `memory_bytes`, `peak_bandwidth`, `peak_flops`, in SI base units;
- two from the run — `achieved_bandwidth`, `mfu`;
- `provenance`, in the reader's words: `"measured (run N, YYYY-MM-DD)"`, and
  what later runs did to it.

When a prior for the card already exists, add a **separate instance** rather
than editing the prior's numbers — `L40S` and `L40S_RUN1` are the precedent — so
a prediction can still be printed against either, and the §9 table can say
which one produced each row. Keys are lowercase and stable: a rename breaks
every runsheet that prints a command. `HOURLY_RATES` in `bench/predictions.py`
needs the card's rate and where it was read; `INTERFERENCE_FITS` needs nothing
unless the run measured TPOT minus median ITL across seats.

Then regenerate what the numbers feed: `python3 bench/export_site_data.py`
(the calculator's card table and its golden grid) — `.githooks/pre-commit` does
it on commit if enabled. Nothing else is needed for the site: a new card appears
in the sentence's card list, its runs counted and linked into the journal, a new fit becomes
the card's coefficients in use once a run trusts it and stays choosable under
*More assumptions* after that, and the golden grid gains its rows. A run's measured
points appear on the seats picture once `measured_<run>.py` feeds them into
`export_site_data.measured_points()` with the geometry they were taken at.

## 6. What else a run touches

- A node of [symptom-map.md](symptom-map.md) whose claim the run measured gains
  its bold evidence word and a link to the report section.
- The [SLO.md §10](SLO.md#10-open-items) item the run closes, and any §6 or §9
  derivation the run showed wrong, corrected in place.
- [GLOSSARY.md](GLOSSARY.md), for any term the report uses that has no entry.

## 7. Out of scope: another model

Adding a second model is deliberately not sought. Three strings name the model
and six numbers are derived from it, and the arithmetic to recompute those six
is not wired to the manifests
([audience.md](audience.md#changing-the-model)). Beyond the mechanics, a second
model of the same class produces a difference the arithmetic already predicts;
the comparisons worth a rented card are between configurations of one model —
KV dtype, chunk size, prefix caching, the card itself. If that decision is ever
reversed, the shape is the one the cards already have: a `MODELS` registry in
`bench/roofline.py` beside `ACCELERATORS`, a `--model` flag, a loop in the
export and a select on the site — and a run per model, because `eff_mem` and
`mfu` are properties of a model on a card, not of the card.

## Checklist

- [ ] runsheet written, predictions from `bench/predictions.py`, committed before renting
- [ ] hourly rate read off the console and written into the sheet
- [ ] startup log captured; Checkpoint A read against the derivation at ±5 %
- [ ] harness or sweep output under `benchmarks/raw/<card>-<date>/`, with its README,
      read once for a credential before staging (§2)
- [ ] file count before and after `git add` agrees
- [ ] `eff_mem` from the median ITL, `mfu` from one uncontended prefill, both reported against the prior
- [ ] `bench/measured_<run>.py` reads the raw files; the report's §9 table quotes it
- [ ] two-sided rows added to `bench/plot_predicted_vs_measured.py`; chart regenerated
- [ ] the report's front matter written; `python3 bench/runs.py` run
- [ ] `Accelerator(...)` and `ACCELERATORS` key added; `HOURLY_RATES` entry; `site/data/` regenerated
- [ ] symptom-map nodes, §10 items and glossary entries the run touched
