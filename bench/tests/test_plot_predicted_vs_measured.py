"""The picture is a selection of the reports' tables; this is what stops it drifting.

`bench/plot_predicted_vs_measured.py` restates numbers that already live in
section 9 of the three benchmark reports, which the repository otherwise
forbids -- one fact, one file. The duplication is allowed only because it is
checked, so the check has to be worth the exemption, and the obvious version is
not: searching the whole report for a bare "2.00" passes even after the cell is
edited, because "2.00" is a substring of the measured "2.000×" beside it, and
"| 45 " occurs in three unrelated tables. Measured on the first attempt: 59 of
80 such probes matched more than one place in their file.

What is checked instead: **one row of section 9 -- the same row -- contains both
cells**. Editing either value moves it out of that row and the test fails.
"""

# Two ways in, and both have to work. `pytest bench/` gets bench/ on the import
# path from tests/conftest.py; `python3 bench/tests/test_roofline.py` on a rented
# pod, where pytest is not installed, gets only this directory. The three lines
# below are what make the second one work, and they are here rather than in
# conftest.py for exactly that reason.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import re

import plot_predicted_vs_measured as plot


def section_nine(path: str) -> list[str]:
    """The table rows of a report's section 9, and nothing else."""
    body = (plot.ROOT / path).read_text(encoding="utf-8")
    match = re.search(r"^## 9\..*?(?=^## |\Z)", body, re.M | re.S)
    assert match, f"{path}: no section 9"
    return [line for line in match.group(0).splitlines()
            if line.startswith("|") and set(line) - set("|-: ")]


def test_each_prediction_still_matches_one_row_of_its_report():
    """Both cells, on one row, in section 9. The exemption rests on this test."""
    rows = {source: section_nine(source) for source in
            {p.source for p in plot.PREDICTIONS}}
    problems = []
    for p in plot.PREDICTIONS:
        hits = [r for r in rows[p.source]
                if p.pred_text in r and p.meas_text in r]
        if len(hits) != 1:
            problems.append(f"{p.source} run {p.run} {p.label!r}: "
                            f"{len(hits)} rows carry both "
                            f"{p.pred_text!r} and {p.meas_text!r}")
    assert not problems, "section 9 no longer matches the drawing:\n  " + \
                         "\n  ".join(problems)


def test_no_probe_is_satisfied_by_its_own_neighbour():
    """The "2.00" trap: a probe contained in the other cell checks nothing.

    If `pred_text` is a substring of `meas_text` then editing the predicted
    value cannot fail the row test above, because the measured cell keeps
    satisfying it.
    """
    swallowed = [f"run {p.run} {p.label!r}: {p.pred_text!r} vs {p.meas_text!r}"
                 for p in plot.PREDICTIONS
                 if p.pred_text != p.meas_text
                 and (p.pred_text in p.meas_text or p.meas_text in p.pred_text)]
    assert not swallowed, "probe swallowed by its neighbour:\n  " + \
                          "\n  ".join(swallowed)


def test_a_mutated_cell_is_actually_caught():
    """The guard is exercised, not just declared.

    Every real row is perturbed in memory and must stop matching. Without this,
    a future rewrite could weaken the probes back to substrings and nothing
    would say so.
    """
    rows = {source: section_nine(source) for source in
            {p.source for p in plot.PREDICTIONS}}
    survived = []
    for p in plot.PREDICTIONS:
        mutated = [r.replace(p.pred_text, p.pred_text.replace("|", "!", 1))
                   for r in rows[p.source]]
        if any(p.pred_text in r and p.meas_text in r for r in mutated):
            survived.append(f"run {p.run} {p.label!r}")
    assert not survived, "a mutated predicted cell still matched:\n  " + \
                         "\n  ".join(survived)


def test_the_committed_svg_is_what_the_data_produces():
    """Editing the table without regenerating is the failure this catches."""
    on_disk = plot.OUT.read_text(encoding="utf-8")
    assert on_disk == plot.svg(), (
        "docs/benchmarks/predicted-vs-measured.svg is stale -- "
        "run python3 bench/plot_predicted_vs_measured.py")


def test_no_dot_is_drawn_outside_the_axis():
    """A point off the canvas is a silently missing miss, and misses are the point.

    A ratio beyond the axis is drawn at the edge as an arrowhead instead, so the
    check is that every such row got one and every arrowhead sits on the plot.
    """
    off = [p for p in plot.PREDICTIONS if not plot.on_axis(p.ratio)]
    arrows = re.findall(r'<polygon points="([^"]*)"', plot.svg())
    assert len(arrows) == len(off), \
        f"{len(off)} rows off the axis, {len(arrows)} arrowheads drawn"
    for points in arrows:
        for x in (float(pair.split(",")[0]) for pair in points.split()):
            assert plot.PLOT_L - 7 <= x <= plot.PLOT_R + 7, points


def test_no_prediction_divides_by_zero():
    """Section 9 has rows like "no preemptions | 0 | 0" that cannot be a ratio.

    They are left out by hand; this says so out loud rather than waiting for a
    ZeroDivisionError from whoever adds the next row.
    """
    zeros = [p.label for p in plot.PREDICTIONS if p.predicted == 0]
    assert not zeros, f"a prediction of zero has no ratio: {zeros}"


def test_no_label_is_drawn_off_the_canvas():
    """A clipped label is invisible in exactly the case that matters: the misses.

    Estimated, not measured -- the renderer is the browser's -- so the width per
    character is scaled by the font size the element actually carries and chosen
    generously, to fail before a real clip does.
    """
    body = plot.svg()
    overflow = []
    for tag, text in re.findall(r"<text([^>]*)>([^<]*)</text>", body):
        attrs = dict(re.findall(r'(\S+)="([^"]*)"', tag))
        size = float(attrs.get("font-size", plot.BASE_FONT))
        width = plot.CHAR_W * (size / plot.BASE_FONT) * len(text)
        x = float(attrs["x"])
        anchor = attrs.get("text-anchor", "start")
        left = {"end": x - width, "middle": x - width / 2}.get(anchor, x)
        if left < 0 or left + width > plot.W:
            overflow.append((text, round(left), round(left + width)))
    assert not overflow, f"outside 0..{plot.W}: {overflow}"


def test_the_reports_named_exist():
    for source in {p.source for p in plot.PREDICTIONS}:
        assert (plot.ROOT / source).is_file(), source


if __name__ == "__main__":
    # The same runner its two siblings carry, for the same reason: a rented pod
    # has no pytest, and a test file that does nothing when executed is a trap.
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
