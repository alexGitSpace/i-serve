"""The committed ledger is what bench/ledger.py produces, and its rows say what the reports say.

No pytest dependency:

    python3 bench/tests/test_ledger.py
"""

# Run directly on a pod without pytest, this file gets only its own directory on
# the path; conftest.py is not consulted, so the insert lives here.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import json
import statistics

import ledger
import runs
from roofline import ACCELERATORS, decode_step_bytes

ROWS = ledger.rows()


def by_run(run_id):
    return [r for r in ROWS if r["run"] == run_id]


def test_the_committed_ledger_is_what_the_generator_produces():
    for path, text in ledger.outputs().items():
        assert path.read_text(encoding="utf-8") == text, \
            f"{path.relative_to(ledger.ROOT)} is stale: python3 bench/ledger.py"


def test_every_line_is_one_json_object_with_the_schema():
    keys = ("run", "date", "card", "host", "engine", "engine_version", "model", "revision",
            "weights", "regime", "mode", "point", "ceiling", "measured", "share", "faced",
            "sheet", "sheet_commit", "raw")
    for line in ledger.OUT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        assert all(k in r for k in keys), sorted(set(keys) - set(r))
        assert tuple(r["host"]) == ledger.HOST_FIELDS
        assert r["regime"] in ledger.REGIMES and r["mode"] in ledger.MODES
        for f in r["faced"]:
            assert f["claim"] in ledger.CLAIMS and f["status"] in ledger.STATUSES


def test_every_row_belongs_to_a_registered_run():
    journal = {r.id for r in runs.runs()}
    assert {r["run"] for r in ROWS} <= journal, {r["run"] for r in ROWS} - journal


def test_a_coefficient_fitted_to_its_own_run_is_a_hypothesis_and_nothing_else():
    fitted = {r.id: r.fitted for r in runs.runs()}
    for r in ROWS:
        for f in r["faced"]:
            assert f["fitted_here"] == (f["with"] == fitted[r["run"]]), (r["run"], f)
            assert (f["status"] == "hypothesis") == f["fitted_here"], (r["run"], f)


def test_share_is_the_ceiling_over_the_measurement():
    """The one definition, recomputed from the row's own bytes and the card's peak."""
    for r in ROWS:
        peak = ACCELERATORS[r["card"]].peak_bandwidth
        assert abs(r["ceiling"]["step_bytes"] / peak * 1e3 - r["ceiling"]["step_ms"]) < 1e-3
        assert abs(r["ceiling"]["step_ms"] / r["measured"]["step_ms"] - r["share"]) < 1e-3
        assert 0 < r["share"] <= 1, (r["run"], r["share"])


def test_the_shares_are_the_coefficients_the_reports_fitted():
    """L40S run 1 fitted 0.83, MI300X run 1 0.46 and run 2 0.57: the ledger reads the same."""
    assert abs(statistics.mean(r["share"] for r in by_run("l40s-1")) - 0.83) < 0.01
    assert abs(statistics.mean(r["share"] for r in by_run("mi300x-1")) - 0.46) < 0.01
    assert abs(statistics.mean(r["share"] for r in by_run("mi300x-2")) - 0.57) < 0.01


def test_the_verdicts_are_the_reports():
    """0.83 survived runs 2-3; run 1's MI300X fit missed on run 2's droplet; run 5's floor held."""
    for run_id in ("l40s-2", "l40s-3"):
        assert {f["status"] for r in by_run(run_id) for f in r["faced"]
                if f["with"] == "l40s-run1"} == {"survived"}, run_id
    assert {f["status"] for r in by_run("mi300x-2") for f in r["faced"]
            if f["with"] == "mi300x-run1"} == {"falsified"}
    assert {f["status"] for r in by_run("mi300x-5") for f in r["faced"]} == {"survived"}


def test_a_bound_survives_only_when_the_step_does_not_cross_it():
    assert ledger.status("bound", False, 25.7, 27.49) == "survived"
    assert ledger.status("bound", False, 25.7, 25.0) == "falsified"
    assert ledger.status("point", False, 100.0, 104.0) == "survived"
    assert ledger.status("point", False, 100.0, 106.0) == "falsified"
    assert ledger.status("point", True, 100.0, 100.0) == "hypothesis"


def test_the_ceiling_bytes_are_the_rooflines():
    """A row's bytes are decode_step_bytes at its point, for the model the module names."""
    import measured
    r = by_run("l40s-1")[0]
    assert r["ceiling"]["step_bytes"] == int(
        decode_step_bytes(measured.LEDGER["model"], r["point"]["batch"], r["point"]["context"]))


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
