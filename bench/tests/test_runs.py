"""Every report registers its run, and every page that lists runs is what bench/runs.py produces.

No pytest dependency:

    python3 bench/tests/test_runs.py
"""

# Run directly on a pod without pytest, this file gets only its own directory on
# the path; conftest.py is not consulted, so the insert lives here.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import runs
from roofline import ACCELERATORS


def test_every_report_is_a_run():
    reports = [p for p in runs.REPORTS.glob("*.md") if p != runs.JOURNAL]
    assert len(runs.runs()) == len(reports), [p.name for p in reports]


def test_the_committed_pages_are_what_the_journal_produces():
    for path, text in runs.outputs().items():
        assert path.read_text(encoding="utf-8") == text, \
            f"{path.relative_to(runs.ROOT)} is stale: python3 bench/runs.py"


def test_a_report_without_front_matter_is_refused():
    assert runs.front_matter("# MI300X run 6\n\nNo block above.\n") is None


def test_a_value_github_would_not_draw_is_refused():
    # Each would turn the table GitHub draws over the report into an error box.
    for value in ("ratio: 1.38", "a comment # here", "#3", "- a list", "'quoted'", "ends with:"):
        try:
            runs.front_matter(f"---\nfound: {value}\n---\n")
        except ValueError:
            continue
        raise AssertionError(f"accepted {value!r}")
    plain = "one queue per process ends the asymmetry, and the 70B's step held at TP = 1"
    assert runs.front_matter(f"---\nfound: {plain}\n---\n") == {"found": plain}


def test_every_card_runs_on_measured_coefficients_once_a_run_exists():
    for name, card in runs.cards().items():
        assert card["fit"] in ACCELERATORS, name
        measured = ACCELERATORS[card["fit"]].provenance.startswith("measured")
        assert measured == bool(card["runs"]), name


def test_a_stale_readme_block_is_detected():
    page = runs.FRONT_PAGE.read_text(encoding="utf-8")
    block = runs.readme_block(runs.runs())
    assert runs.with_block(page, block) == page
    stale = page.replace(block, block.replace("| 5 |", "| 4 |").replace("[5]", "[4]"))
    assert stale != page and runs.with_block(stale, block) != stale


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
