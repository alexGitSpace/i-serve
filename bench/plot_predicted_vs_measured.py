"""Draw every prediction the runs made, as measured / predicted, on a log axis.

The front page's picture: one row per prediction, grouped by run, sorted by ratio;
a ratio because the predictions come in seven units. Rows are a selection of each
report's section 9, and bench/tests/test_plot_predicted_vs_measured.py holds
`pred_text` / `meas_text` to one row of it. Drawn: two-sided values only, a
predicted range at its midpoint, an off-axis ratio as an arrowhead at the edge;
hollow means the coefficient was fitted to that run. The ratio is measured /
predicted, the reports' Error column the reverse; where a report gives two
measured values the dot is the first. The pre-commit hook, that test and CI keep
the committed SVG equal to svg().
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass

# Resolved from this file, not from the working directory: the generator and its
# test are run from both the repository root and from bench/.
ROOT = pathlib.Path(__file__).resolve().parent.parent

BASELINE = "docs/benchmarks/l40s-baseline.md"
RUN2 = "docs/benchmarks/l40s-run2.md"
RUN3 = "docs/benchmarks/l40s-run3.md"
MI300X_RUN1 = "docs/benchmarks/mi300x-run1.md"

OUT = ROOT / "docs/benchmarks/predicted-vs-measured.svg"


@dataclass(frozen=True)
class Prediction:
    run: int
    label: str
    predicted: float
    measured: float
    pred_text: str          # verbatim from the report's section 9
    meas_text: str          # verbatim from the report's section 9
    source: str
    fitted: bool = False    # coefficient fitted to this same run

    @property
    def ratio(self) -> float:
        return self.measured / self.predicted


PREDICTIONS: tuple[Prediction, ...] = (
    # Each probe carries enough delimiters to match one section-9 row and no
    # other: a bare "2.00" is a substring of "2.000×" on the same line.

    # --- run 1, 2026-08-18 ---------------------------------------------------
    Prediction(1, "KV pool", 181749, 168985,
               "| 181 749 |", "| 168 985 |", BASELINE),
    Prediction(1, "seats by capacity, 4 000 ctx", 45, 41,
               "| 45 |", "| 41 |", BASELINE),
    Prediction(1, "decode step c=1, eff_mem 0.70", 28.12, 22.70,
               "| 28.12 ms |", "| 22.70 ms |", BASELINE),
    Prediction(1, "decode step c=1, eff_mem 0.83", 23.71, 22.70,
               "| 23.71 ms |", "| 22.70 ms |", BASELINE, fitted=True),
    Prediction(1, "decode step c=32, eff_mem 0.70", 59.10, 50.69,
               "| 59.10 ms |", "| 50.69 ms |", BASELINE),
    Prediction(1, "decode step c=32, eff_mem 0.83", 49.85, 50.69,
               "| 49.85 ms |", "| 50.69 ms |", BASELINE, fitted=True),
    Prediction(1, "TTFT c=1, 4 000-token prompt", 341.3, 350.1,
               "| 341.3 ms |", "| 350.1 ms |", BASELINE),
    Prediction(1, "max_num_seqs by latency, eff_mem 0.70", 23, 31,
               "| 23 |", "| ~31 (decode step) |", BASELINE),
    Prediction(1, "max_num_seqs by latency, eff_mem 0.83", 32, 31,
               "| 32 |", "| ~31 (decode step) |", BASELINE, fitted=True),
    Prediction(1, "max_num_seqs against TPOT p99", 32, 12,
               "| 32 |", "| 12 (TPOT p99) |", BASELINE, fitted=True),

    # --- run 2, 2026-08-23 ---------------------------------------------------
    Prediction(2, "KV pool", 168985, 169833,
               "| 168 985 ± 500 |", "| 169 833 |", RUN2),
    Prediction(2, "seats by capacity, 4 100 ctx", 41, 41,
               "| 41 | 41 |", "| 41 | 41 |", RUN2),
    Prediction(2, "seats by capacity, 1 600 ctx", 106.1, 106,
               "| 106.1 |", "| 106 running |", RUN2),
    Prediction(2, "decode step c=13", 33.83, 33.83,
               "| **33.83 ms** |", "**33.83 ms** (mean of six launches", RUN2),
    Prediction(2, "decode step c=32", 49.85, 50.61,
               "| **49.85 ms** |", "50.61 ms (mean of the two uncontaminated", RUN2),
    Prediction(2, "decode step, Poisson 0.5 req/s", 23.70, 24.07,
               "| 23.70–30.28 ms |", "| 24.07–30.94 ms |", RUN2),
    Prediction(2, "decode step, Poisson 2.5 req/s", 30.28, 30.94,
               "| 23.70–30.28 ms |", "| 24.07–30.94 ms |", RUN2),
    Prediction(2, "decode step, Poisson 3.0 req/s", 33.91, 36.85,
               "| 33.91 / 51.25 ms |", "| 36.85 / 55.76 ms |", RUN2),
    Prediction(2, "decode step, Poisson 4.0 req/s", 51.25, 55.76,
               "| 33.91 / 51.25 ms |", "| 36.85 / 55.76 ms |", RUN2),
    Prediction(2, "worst step, chunk 1", 77.5, 70.6,
               "| 77.5 / 122.2 / 211.8 / 390.9 ms |",
               "| 70.6 / 114.2 / 198.2 / 364.9 ms |", RUN2),
    Prediction(2, "worst step, chunk 2", 122.2, 114.2,
               "| 77.5 / 122.2 / 211.8 / 390.9 ms |",
               "| 70.6 / 114.2 / 198.2 / 364.9 ms |", RUN2),
    Prediction(2, "worst step, chunk 3", 211.8, 198.2,
               "| 77.5 / 122.2 / 211.8 / 390.9 ms |",
               "| 70.6 / 114.2 / 198.2 / 364.9 ms |", RUN2),
    Prediction(2, "worst step, chunk 4", 390.9, 364.9,
               "| 77.5 / 122.2 / 211.8 / 390.9 ms |",
               "| 70.6 / 114.2 / 198.2 / 364.9 ms |", RUN2),
    Prediction(2, "FP8 KV pool multiplier", 2.00, 2.000,
               "| 2.00× |", "| 2.000× |", RUN2),
    Prediction(2, "FP8 decode step, n=13", 28.35, 28.91,
               "| 28.35 ms |", "| 28.91 ms |", RUN2),
    Prediction(2, "FP8 decode step, n=26", 33.83, 35.41,
               "| 33.83 ms |", "| 35.41 ms |", RUN2),
    Prediction(2, "TPOT p99 crossing", 1.75, 2.25,
               "| 1.5–2.0 req/s |", "| 2.0–2.5 req/s |", RUN2),
    Prediction(2, "TTFT p99 crossing 300 ms", 2.0, 0.75,
               "| 1.5–2.5 req/s |", "| 0.5–1.0 req/s |", RUN2),
    Prediction(2, "goodput peak", 1.4, 1.87,
               "| 1.2–1.6 req/s |", "| 1.87 req/s |", RUN2),
    Prediction(2, "max_num_seqs by TPOT p99", 20, 14,
               "| 18–22, or 31 |", "| **14** |", RUN2),

    # --- run 3, 2026-08-30 ---------------------------------------------------
    Prediction(3, "seat count, h = 0", 13, 12.5,
               "| 12–14 |", "| **12–14** (12.5) |", RUN3),
    Prediction(3, "seat count, h = 0.8", 23, 37.8,
               "| 22–24 |", "| **32–40** (37.8) |", RUN3),
    Prediction(3, "decode step at c=24, cascade", 27.97, 29.76,
               "| 27.97 |", "| 29.76 |", RUN3),
    Prediction(3, "measured h at cached levels", 0.800, 0.800,
               "| 0.800 ± 0.02 |", "| **0.800** |", RUN3),
    Prediction(3, "ITL p99 under caching", 106, 199,
               "| ~106 ms |", "| **199 ms** |", RUN3),
    Prediction(3, "interference scales by (1 - h)", 0.200, 0.194,
               "| 0.200 |", "| **0.194** |", RUN3),
    Prediction(3, "TPOT p50 crossing 50 ms", 5.5, 8,
               "| 5–6 req/s |", "| **7–9** |", RUN3),
    Prediction(3, "TTFT p99 crossing 300 ms", 3.25, 8,
               "| 2.5–4 req/s |", "| **7–9** |", RUN3),
    # Two measured pools; the dot is the first, the launch the prediction was
    # made for. Both travel in the cell text so the choice stays visible.
    Prediction(3, "KV pool", 169833, 168985,
               "| 169 833 ± 5 % |", "| 168 985 / 176 227 |", RUN3),
    Prediction(3, "startup-log keys present", 7, 6,
               "| 7 of 8 |", "| **6 of 8** |", RUN3),

    # --- MI300X run 1, 2026-09-27 --------------------------------------------
    Prediction(4, "KV pool", 1060655, 1123065,
               "| 1 060 655 |", "| 1 123 065 (printed; 1 124 064 in blocks) |", MI300X_RUN1),
    Prediction(4, "KV pool after the L40S's 7 %", 986410, 1123065,
               "| ~986 000 |", "| 1 123 065 (launch 1) |", MI300X_RUN1),
    Prediction(4, "decode step c=1, eff_mem 0.70", 4.58, 6.61,
               "| 4.58 ms |", "| 6.61 ms |", MI300X_RUN1),
    Prediction(4, "decode step c=32, eff_mem 0.70", 9.51, 14.90,
               "| 9.51 ms |", "| 14.90 ms |", MI300X_RUN1),
    Prediction(4, "decode step c=1, eff_mem 0.46", 6.97, 6.61,
               "| 6.97 ms |", "| 6.61 ms (c001) |", MI300X_RUN1, fitted=True),
    Prediction(4, "decode step c=8, eff_mem 0.46", 8.71, 9.72,
               "| 8.71 ms |", "| 9.72 ms (c008) |", MI300X_RUN1, fitted=True),
    Prediction(4, "decode step c=32, eff_mem 0.46", 14.66, 14.90,
               "| 14.66 ms |", "| 14.90 ms (c032) |", MI300X_RUN1, fitted=True),
    Prediction(4, "TTFT c=1, 2 000-token prompt", 47.3, 116.4,
               "| 47.3 ms |", "| 116.4 ms |", MI300X_RUN1),
    Prediction(4, "TTFT c=1, 4 000-token prompt", 94.5, 256.9,
               "| 94.5 ms |", "| 256.9 ms |", MI300X_RUN1),
    Prediction(4, "TTFT c=1, 8 000-token prompt", 189.1, 642.6,
               "| 189.1 ms |", "| 642.6 ms |", MI300X_RUN1),
    Prediction(4, "TTFT doubling, 2 000 to 4 000", 2.00, 2.21,
               "| 2.00× / 2.00× |", "| 2.21× / 2.50× |", MI300X_RUN1),
    Prediction(4, "TTFT doubling, 4 000 to 8 000", 2.00, 2.50,
               "| 2.00× / 2.00× |", "| 2.21× / 2.50× |", MI300X_RUN1),
    # Left out: the seats row is the pool row divided by 4 200, not a second
    # test; running at c = 288 was a bound (it held); and the latency limit was
    # measured as a range, 8-32 -- a midpoint would invent a measured value.
)

@dataclass(frozen=True)
class Run:
    """One rented card on one date; the card is named per band, not in the title."""

    label: str
    date: str
    card: str


RUNS: dict[int, Run] = {
    1: Run("run 1", "2026-08-18", "NVIDIA L40S"),
    2: Run("run 2", "2026-08-23", "NVIDIA L40S"),
    3: Run("run 3", "2026-08-30", "NVIDIA L40S"),
    4: Run("MI300X run 1", "2026-09-27", "AMD Instinct MI300X"),
}

# --- geometry ----------------------------------------------------------------
# One named row per prediction: GitHub renders the picture inside an <img>, so a
# <title> label never shows and stacked dots would carry no name.
W = 860
MARGIN = 16
LABEL_R = 258                       # right edge of the name column
PLOT_L, PLOT_R = 272, 772
VALUE_R = W - MARGIN                # right edge of the ratio column

ROW = 15                            # one prediction
GROUP_HEAD = 27                     # a run's header line
GROUP_GAP = 9                       # air above a header, except the first

TITLE_Y = 28
SUBTITLE_Y = 49
LEGEND_Y = 72
TICK_TOP_Y = 99                     # tick labels above the plot
DIR_Y = 117                         # the two direction legends and the band label
PLOT_TOP = 128

# The axis is symmetric in log space -- 1/3.2 = 0.3125 -- so 1x sits exactly in
# the middle and "left = high, right = low" is a fair split rather than an
# artefact of the bounds.
X_MAX = 3.2
X_MIN = 1 / X_MAX
TICKS = (0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
TOLERANCE = 0.10                    # the +-10 % band

BASE_FONT = 11.5                    # the size CHAR_W was calibrated at
CHAR_W = 6.4                        # generous width per character at BASE_FONT
ROW_FONT = 11
SMALL_FONT = 10.5

# Every colour clears 3:1 on both GitHub surfaces, #ffffff and #0d1117: an <img>
# cannot ask which it landed on. HIT and MISS are a CVD-safe warm/cool pair
# (worst case dE 30.5, protanopia), and position against the band repeats them.
HIT = "#2f81f7"                     # 3.75 on white, 5.05 on #0d1117
MISS = "#cc7000"                    # 3.57 on white, 5.30 on #0d1117
INK = "#6e7781"                     # 4.55 / 4.16 -- labels and axis text
DIM = "#8b949e"                     # hairlines, stripes, the band fill


def on_axis(ratio: float) -> bool:
    return X_MIN <= ratio <= X_MAX


def x_of(ratio: float) -> float:
    ratio = min(max(ratio, X_MIN), X_MAX)
    lo, hi = math.log10(X_MIN), math.log10(X_MAX)
    t = (math.log10(ratio) - lo) / (hi - lo)
    return PLOT_L + t * (PLOT_R - PLOT_L)


def layout() -> tuple[list[tuple[int, float, list[tuple[Prediction, float]]]], float]:
    """Where every header and row sits, rows sorted by ratio inside their run."""
    groups: list[tuple[int, float, list[tuple[Prediction, float]]]] = []
    y = float(PLOT_TOP)
    for run in sorted(RUNS):
        rows = sorted((p for p in PREDICTIONS if p.run == run),
                      key=lambda p: p.ratio)
        if groups:
            y += GROUP_GAP
        head_y, y = y, y + GROUP_HEAD
        placed = []
        for p in rows:
            placed.append((p, y + ROW / 2))
            y += ROW
        groups.append((run, head_y, placed))
    return groups, y


def escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))


def is_hit(p: Prediction) -> bool:
    return abs(p.ratio - 1) <= TOLERANCE


def text(x: float, y: float, body: str, *, size: float = ROW_FONT,
         fill: str = INK, anchor: str = "start", weight: str = "") -> str:
    bold = f' font-weight="{weight}"' if weight else ""
    end = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" '
            f'font-size="{size}"{end}{bold}>{escape(body)}</text>')


def svg() -> str:
    groups, plot_bottom = layout()
    tick_bottom_y = plot_bottom + 22
    axis_title_y = plot_bottom + 45
    height = plot_bottom + 60

    hits = sum(1 for p in PREDICTIONS if is_hit(p))
    fitted = sum(1 for p in PREDICTIONS if p.fitted)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height:.0f}" '
        f'width="{W}" height="{height:.0f}" font-family="system-ui,-apple-system,'
        f'Segoe UI,Roboto,sans-serif" role="img" '
        f'aria-label="One row per prediction of L40S runs 1 to 3 and MI300X run 1, '
        f'each drawn as measured divided by predicted on a logarithmic axis">',
        '<title>Predicted vs measured — L40S runs 1–3, MI300X run 1</title>',
        text(MARGIN, TITLE_Y, "Every prediction the runs made, against what the "
             "card did", size=15.5, fill=INK, weight="600"),
        text(MARGIN, SUBTITLE_Y,
             f"{len(PREDICTIONS)} predictions · {hits} within ±10 % · "
             f"{fitted} had their coefficient fitted to the very run they "
             f"predict", size=SMALL_FONT, fill=INK),
    ]

    # --- legend: shape and colour both named, so neither has to be guessed ---
    x = MARGIN + 5
    for label, colour, hollow in (("within ±10 %", HIT, False),
                                  ("outside ±10 %", MISS, False),
                                  ("coefficient fitted to that run", INK, True)):
        mark = (f'fill="none" stroke="{colour}" stroke-width="1.8"' if hollow
                else f'fill="{colour}"')
        parts.append(f'<circle cx="{x:.1f}" cy="{LEGEND_Y - 3.5:.1f}" r="3.8" '
                     f'{mark}/>')
        parts.append(text(x + 9, LEGEND_Y, label, size=SMALL_FONT, fill=INK))
        x += 9 + CHAR_W * (SMALL_FONT / BASE_FONT) * len(label) + 24

    # --- the axis, stated twice: above the first row and below the last ------
    # A reader at row forty should not have to scroll back up to find out what
    # the horizontal position means.
    for tick in TICKS:
        tx = x_of(tick)
        mark = f"{tick:g}×"
        for ty in (TICK_TOP_Y, tick_bottom_y):
            parts.append(text(tx, ty, mark, size=SMALL_FONT, fill=INK,
                              anchor="middle"))
        if tick != 1.0:
            parts.append(
                f'<line x1="{tx:.1f}" y1="{PLOT_TOP}" x2="{tx:.1f}" '
                f'y2="{plot_bottom:.1f}" stroke="{DIM}" stroke-width="1" '
                f'opacity="0.35"/>')

    # the +-10 % band, its two edges, and the line of a perfect prediction
    x_lo, x_hi = x_of(1 - TOLERANCE), x_of(1 + TOLERANCE)
    parts.append(
        f'<rect x="{x_lo:.1f}" y="{PLOT_TOP}" width="{x_hi - x_lo:.1f}" '
        f'height="{plot_bottom - PLOT_TOP:.1f}" fill="{DIM}" opacity="0.15"/>')
    for edge in (x_lo, x_hi):
        parts.append(
            f'<line x1="{edge:.1f}" y1="{PLOT_TOP}" x2="{edge:.1f}" '
            f'y2="{plot_bottom:.1f}" stroke="{DIM}" stroke-width="1" '
            f'opacity="0.5"/>')
    parts.append(
        f'<line x1="{x_of(1):.1f}" y1="{PLOT_TOP}" x2="{x_of(1):.1f}" '
        f'y2="{plot_bottom:.1f}" stroke="{INK}" stroke-width="1.3"/>')

    # the three legends that say what the axis means -- the band in the middle,
    # and which side of it is which kind of wrong
    parts.append(text(x_of(1), DIR_Y, "±10 %", size=10, fill=INK,
                      anchor="middle"))
    parts.append(text(x_lo - 16, DIR_Y, "← predicted high · the card came in under",
                      size=SMALL_FONT, fill=INK, anchor="end"))
    parts.append(text(x_hi + 16, DIR_Y, "predicted low · the card came in over →",
                      size=SMALL_FONT, fill=INK))
    parts.append(text((PLOT_L + PLOT_R) / 2, axis_title_y,
                      "measured ÷ predicted · logarithmic, so equal distances "
                      "are equal factors", size=SMALL_FONT, fill=INK,
                      anchor="middle"))

    # --- the runs --------------------------------------------------------
    for run, head_y, placed in groups:
        meta = RUNS[run]
        if run != min(RUNS):
            parts.append(
                f'<line x1="{MARGIN}" y1="{head_y - 4:.1f}" x2="{VALUE_R}" '
                f'y2="{head_y - 4:.1f}" stroke="{DIM}" stroke-width="1" '
                f'opacity="0.45"/>')
        parts.append(text(MARGIN, head_y + 18,
                          f"{meta.label} · {meta.date} · {meta.card}",
                          size=11.5, fill=INK, weight="600"))
        inside = sum(1 for p, _ in placed if is_hit(p))
        parts.append(text(VALUE_R, head_y + 18,
                          f"{inside} of {len(placed)} within ±10 %",
                          size=SMALL_FONT, fill=INK, anchor="end"))

        for i, (p, y) in enumerate(placed):
            # A faint stripe every other row: the name is 250 px from its dot
            # and the eye needs a rail to cross that on.
            if i % 2:
                parts.append(
                    f'<rect x="{MARGIN}" y="{y - ROW / 2:.1f}" '
                    f'width="{VALUE_R - MARGIN}" height="{ROW}" fill="{DIM}" '
                    f'opacity="0.07"/>')

            colour = HIT if is_hit(p) else MISS
            cx = x_of(p.ratio)
            baseline = x_of(1)

            # The deviation bar. Length is the error, which is the one thing a
            # reader should be able to compare between rows without reading a
            # number; it stops short of the dot so the two marks stay separate.
            if abs(cx - baseline) > 6:
                stop = cx - math.copysign(5.0, cx - baseline)
                parts.append(
                    f'<line x1="{baseline:.1f}" y1="{y:.1f}" x2="{stop:.1f}" '
                    f'y2="{y:.1f}" stroke="{colour}" stroke-width="2" '
                    f'stroke-linecap="round" opacity="0.75"/>')

            mark = (f'fill="none" stroke="{colour}" stroke-width="1.8"'
                    if p.fitted else f'fill="{colour}"')
            if on_axis(p.ratio):
                parts.append(f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="3.8" {mark}/>')
            else:
                # Off the axis: an arrowhead at the edge, pointing the way it went.
                tip = 6.0 if p.ratio > X_MAX else -6.0
                parts.append(
                    f'<polygon points="{cx + tip:.1f},{y:.1f} {cx - tip:.1f},{y - 5:.1f} '
                    f'{cx - tip:.1f},{y + 5:.1f}" {mark}/>')

            parts.append(text(LABEL_R, y + 3.8, p.label, fill=INK, anchor="end"))
            parts.append(text(VALUE_R, y + 3.8, f"{p.ratio:.2f}×",
                              size=SMALL_FONT, fill=INK, anchor="end"))

    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> None:
    OUT.write_text(svg(), encoding="utf-8")
    hits = sum(1 for p in PREDICTIONS if is_hit(p))
    print(f"{OUT.relative_to(ROOT)}: {len(PREDICTIONS)} predictions, {hits} within "
          f"±{TOLERANCE:.0%}, {len(PREDICTIONS) - hits} outside")
    for p in sorted(PREDICTIONS, key=lambda q: q.ratio)[:3]:
        print(f"  lowest  {p.ratio:5.2f}x  run {p.run}  {p.label}")
    for p in sorted(PREDICTIONS, key=lambda q: q.ratio)[-3:]:
        print(f"  highest {p.ratio:5.2f}x  run {p.run}  {p.label}")


if __name__ == "__main__":
    main()
