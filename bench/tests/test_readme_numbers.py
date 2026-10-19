"""The front page quotes numbers it does not own; this is what stops them drifting.

README.md and docs/layout.md restate figures whose home is bench/ and docs/SLO.md,
the one exemption from one-fact-one-file that site/ also enjoys under its own
guards. A bare "12" or "0.39" is a substring of half the page, so every probe
carries prose either side of the number, is built by f-string from the module
that owns it, and declares how many times it must occur (the trap is documented
in test_plot_predicted_vs_measured.py). A right number inside a wrong sentence
is beyond it: only a reading catches a direction.
"""

# Run directly on a pod without pytest, this file gets only its own directory on
# the path; conftest.py is not consulted, so the two inserts live here.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))


import contextlib
import io
import json
import re

import export_site_data as export
import harness
import measured_run3 as run3
import plot_predicted_vs_measured as plot
import predictions
import roofline
import test_symptom_map as symptom_map_tests


# --- the sources, each read from the module that owns it ----------------------

FRONT_PAGES = ("README.md", "docs/layout.md")


def _readme() -> str:
    """Both pages as one line: a probe must not depend on where a paragraph broke."""
    return re.sub(r"\s+", " ", " ".join(
        (export.ROOT / name).read_text(encoding="utf-8") for name in FRONT_PAGES))


def _claims() -> list[tuple[str, int, str]]:
    """(probe, how many times it must occur, who owns the number)."""
    rows = run3.slo_seat_rows()
    cold, cached = rows["h0"], rows["h80"]
    seats_cold = int(cold["harness_load"])
    seats_cached = int(cached["harness_load"])
    cost_cold = run3.dollars_per_1m(cold["output_throughput"])
    cost_cached = run3.dollars_per_1m(cached["output_throughput"])
    tpot = int(run3.TPOT_TARGET_MS)
    rate = predictions.HOURLY_RATES["l40s-run1"][0]

    ceiling = predictions.what_if_point()["seats"]["max_num_seqs"]

    # Section 6's row at h = 0.8, and what run 3 measured against it.
    row_h80 = roofline.seats_under_prefill_interference(
        predictions.QWEN3_8B, predictions.L40S_RUN1, 4100, predictions.TPOT_TARGET,
        predictions.L40S_RUN1_INTERFERENCE_SLOPE,
        predictions.L40S_RUN1_INTERFERENCE_INTERCEPT, 0.8)
    measured_h80 = export.model_data()["measured"]["seat_crossings"]["h80"]
    over_row = round((measured_h80 / row_h80 - 1) * 100)

    # The same miss against the runsheet's two-sided prediction, which is where
    # the drawing takes it from.
    seat_pred = next(p for p in plot.PREDICTIONS
                     if p.run == "l40s-3" and p.label == "seat count, h = 0.8")
    over_midpoint = round((seat_pred.ratio - 1) * 100)
    lo, hi = re.findall(r"\d+", seat_pred.pred_text)

    floors = {}
    for key in ("l40s-run1", "mi300x"):
        plan = harness.dry_run_plan(harness.SCENARIOS["seats-cached"],
                                    harness.ACCELERATORS[key], harness.SLOTargets())
        got = {round(level["ttft_floor_uncached_ms"], 1) for level in plan["levels"]}
        assert len(got) == 1, f"{key}: {len(got)} distinct prefill floors, expected one"
        floors[key] = got.pop()

    golden = len(export.golden_grid())
    md_nodes = len(symptom_map_tests.markdown_nodes())
    json_nodes = len(json.loads(export.SYMPTOM_MAP.read_text(encoding="utf-8"))["nodes"])

    return [
        (f"seats **{seats_cold} people with a cold cache and {seats_cached} when "
         f"80 % of each prompt repeats** — measured, TPOT p99 ≤ {tpot} ms", 1,
         "measured_run3.slo_seat_rows()"),
        (f"says {ceiling} seats before arriving prompts are priced in", 1,
         "predictions.what_if_point()['seats']['max_num_seqs']"),
        (f"**${cost_cached:.2f} per 1M output tokens that met the SLO** — "
         f"L40S at ${rate:.2f}/h", 1,
         "measured_run3.dollars_per_1m() and predictions.HOURLY_RATES"),
        (f"{seats_cached} seats, TPOT p99 ≤ {tpot} ms", 1,
         "measured_run3.slo_seat_rows()['h80']"),
        (f"hold {seats_cold} seats instead of {seats_cached}, and the figure is "
         f"**${cost_cold:.2f}**", 1,
         "measured_run3.slo_seat_rows()['h0']"),
        (f"came out **{over_row} % higher** than the §6 row predicts", 1,
         "roofline.seats_under_prefill_interference(..., h=0.8) vs the run-3 crossing"),
        (f"{over_midpoint} % above the midpoint of the {lo}–{hi} range", 1,
         "plot_predicted_vs_measured.PREDICTIONS, 'seat count, h = 0.8'"),
        (f"parity {golden}/{golden} rows", 1,
         "export_site_data.golden_grid()"),
        (f"for the {md_nodes} nodes", 1,
         "symptom_map_tests.markdown_nodes()"),
        (f"{md_nodes} nodes; the JSON's {json_nodes} nodes", 1,
         "symptom_map_tests.markdown_nodes() and docs/symptom-map.json"),
        (f"moves from **{floors['l40s-run1']} ms** to **{floors['mi300x']} ms**", 1,
         "harness.dry_run_plan(...)['levels'][*]['ttft_floor_uncached_ms']"),
    ]


# --- the tests ----------------------------------------------------------------

def test_every_headline_number_is_the_one_python_produces():
    """The exemption from one-fact-one-file rests on this test."""
    readme = _readme()
    problems = []
    for probe, want, source in _claims():
        got = readme.count(probe)
        if got != want:
            problems.append(f"{source}: README carries {got} copies of {probe!r}, "
                            f"expected {want}")
    assert not problems, ("README.md has drifted from bench/. Re-read the source "
                          "and fix the prose, never the other way round:\n  "
                          + "\n  ".join(problems))


def test_the_header_a_reader_types_is_what_the_harness_prints():
    """The code block under *Five minutes* is stdout, not a transcription."""
    readme = _readme()
    problems = []
    for key in ("l40s-run1", "mi300x"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            harness.dry_run(harness.SCENARIOS["seats-cached"],
                            harness.ACCELERATORS[key], harness.SLOTargets())
        for line in out.getvalue().splitlines()[:2]:
            line = re.sub(r"\s+", " ", line).strip()
            if readme.count(line) != 1:
                problems.append(f"{key}: README does not carry {line!r} exactly once")
    assert not problems, ("the quick-start block no longer matches the harness's "
                          "own header:\n  " + "\n  ".join(problems))


def test_a_mutated_value_is_actually_caught():
    """Perturb each probe's first digit; a probe that still matches checks nothing."""
    readme = _readme()
    problems = []
    for probe, _want, source in _claims():
        mutant = re.sub(r"\d", lambda m: str((int(m.group()) + 1) % 10), probe, count=1)
        if mutant == probe:
            problems.append(f"{source}: probe carries no digit: {probe!r}")
        elif readme.count(mutant) != 0:
            problems.append(f"{source}: the mutated probe still matches: {mutant!r}")
    assert not problems, "a probe that survives mutation guards nothing:\n  " + \
                         "\n  ".join(problems)


def test_no_probe_is_a_bare_number():
    """Twelve characters of prose either side, or the probe is not a probe."""
    problems = []
    for probe, _want, source in _claims():
        if len(re.sub(r"[\d.,$%/]", "", probe)) < 12:
            problems.append(f"{source}: too little context around the number: {probe!r}")
    assert not problems, "\n  ".join(problems)


def test_the_symptom_count_is_quoted_from_the_map_not_invented():
    """"seven symptoms" has no Python home, so it is held to the map's own sentence.

    Counting the map's headings gives seven only by coincidence, so the words
    are compared, not the arithmetic.
    """
    readme = _readme()
    map_md = re.sub(r"\s+", " ",
                    (export.ROOT / "docs/symptom-map.md").read_text(encoding="utf-8"))
    phrase = "seven symptoms"
    assert phrase in map_md, f"docs/symptom-map.md no longer says {phrase!r}"
    assert readme.count(phrase) == 1, \
        f"README.md carries {readme.count(phrase)} copies of {phrase!r}, expected 1"


def test_every_picture_and_document_the_front_page_names_exists():
    missing = []
    for name in FRONT_PAGES:
        page = export.ROOT / name
        missing += [f"{name}: {target}"
                    for target in re.findall(r"\]\(([^)#]+)\)", page.read_text(encoding="utf-8"))
                    if not target.startswith(("http://", "https://", "#"))
                    and not (page.parent / target).exists()]
    assert not missing, "README.md links to things that are not there:\n  " + \
                        "\n  ".join(missing)


if __name__ == "__main__":
    # The same runner its siblings carry, for the same reason: a rented pod has
    # no pytest, and a test file that does nothing when executed is a trap.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}\n      {exc or test.__doc__}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
