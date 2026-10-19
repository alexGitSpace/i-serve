"""The run journal: every report in docs/benchmarks/ registers its run in its front matter.

The chart, the site and the two pages below list runs from here, so a report
without front matter fails all of them. The fields: docs/GLOSSARY.md, *Front
matter*; the procedure: docs/adding-a-run.md section 4.

    python3 bench/runs.py        # docs/benchmarks/README.md, and README.md's runs block
"""

from __future__ import annotations

import pathlib
import re
import sys
from dataclasses import dataclass

from roofline import ACCELERATORS

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS = ROOT / "docs" / "benchmarks"
JOURNAL = REPORTS / "README.md"
FRONT_PAGE = ROOT / "README.md"
BEGIN, END = "<!-- runs:begin -->", "<!-- runs:end -->"

REQUIRED = ("card", "run", "date", "asked", "found", "predicted_with")
OPTIONAL = ("fitted",)

# GitHub parses the block as YAML, and a value YAML reads as something other
# than plain text turns the table over the report into an error box.
UNSAFE = re.compile(r": | #|:$|^[-?:,\[\]{}#&*!|>'\"%@`]")


@dataclass(frozen=True)
class Run:
    card: str               # the prior's key in ACCELERATORS: "l40s", "mi300x"
    number: int             # in order per card
    date: str
    asked: str
    found: str
    predicted_with: str     # the ACCELERATORS key its predictions were printed from
    fitted: str | None      # the key of the coefficients it produced, if any
    report: str             # relative to the repository root

    @property
    def id(self) -> str:
        return f"{self.card}-{self.number}"

    @property
    def label(self) -> str:
        return f"{short(self.card)} run {self.number}"

    @property
    def trusts(self) -> str:
        """The coefficients a reader should take from this run: its own fit, or the one it faced."""
        return self.fitted or self.predicted_with


def short(card: str) -> str:
    return card.upper()


def card_of(key: str) -> str:
    """"mi300x-run2" -> "mi300x": a fit's key is its card's prior key plus the run."""
    return re.sub(r"-run\d+$", "", key)


def front_matter(text: str) -> dict[str, str] | None:
    """The `key: value` lines between the two `---` that open a report, or None."""
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not match:
        return None
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(": ")
        if not sep or key in fields or UNSAFE.search(value):
            raise ValueError(f"not one plain `key: value` line: {line!r}")
        fields[key] = value
    return fields


def runs() -> list[Run]:
    """Every report's run, oldest first. A report without front matter is an error."""
    found: list[Run] = []
    for path in sorted(REPORTS.glob("*.md")):
        if path == JOURNAL:
            continue
        rel = path.relative_to(ROOT).as_posix()
        fields = front_matter(path.read_text(encoding="utf-8"))
        if fields is None:
            raise ValueError(f"{rel}: no front matter (docs/adding-a-run.md section 4)")
        missing = set(REQUIRED) - set(fields)
        unknown = set(fields) - set(REQUIRED) - set(OPTIONAL)
        if missing or unknown:
            raise ValueError(f"{rel}: missing {sorted(missing)}, unknown {sorted(unknown)}")
        run = Run(card=fields["card"], number=int(fields["run"]), date=fields["date"],
                  asked=fields["asked"], found=fields["found"],
                  predicted_with=fields["predicted_with"],
                  fitted=fields.get("fitted"), report=rel)
        # Every key names a registry entry of this same card.
        for key in (run.card, run.predicted_with, run.fitted):
            if key is not None and (key not in ACCELERATORS or card_of(key) != run.card):
                raise ValueError(f"{rel}: {key!r} is not one of {run.card}'s ACCELERATORS keys")
        found.append(run)
    found.sort(key=lambda r: (r.date, r.card, r.number))
    for card in {r.card for r in found}:
        numbers = [r.number for r in found if r.card == card]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"{card}: runs numbered {numbers}, expected 1..{len(numbers)} by date")
    return found


def span(numbers: list[int]) -> str:
    """[2, 3] -> "runs 2–3"; [2] -> "run 2"."""
    if len(numbers) == 1:
        return f"run {numbers[0]}"
    if numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"runs {numbers[0]}–{numbers[-1]}"
    return "runs " + ", ".join(map(str, numbers))


def cards(journal: list[Run] | None = None) -> dict[str, dict]:
    """Each card in ACCELERATORS once: its runs, and its options with the set in use first.

    In use: docs/GLOSSARY.md, *Coefficients in use*. The rest follow newest fit
    first, the prior last.
    """
    journal = runs() if journal is None else journal
    out: dict[str, dict] = {}
    for key in ACCELERATORS:
        card = card_of(key)
        if card in out:
            continue
        mine = [r for r in journal if r.card == card]
        fit = mine[-1].trusts if mine else card
        newest = {r.fitted: r.number for r in mine if r.fitted}
        others = sorted((k for k in ACCELERATORS if card_of(k) == card and k != fit),
                        key=lambda k: -newest.get(k, 0))
        fitted_by = next((r for r in mine if r.fitted == fit), None)
        out[card] = {
            "name": ACCELERATORS[card].name,
            "short": short(card),
            "fit": fit,
            "options": [fit] + others,
            "fitted_by": fitted_by.number if fitted_by else None,
            "faced_by": [r.number for r in mine
                         if r.predicted_with == fit and r is not fitted_by],
            "runs": [r.id for r in mine],
            # Where the card's rows sit in the journal: GitHub's anchor for its heading.
            "journal": "docs/benchmarks/README.md#"
                       + re.sub(r"[^a-z0-9]+", "-", ACCELERATORS[card].name.lower()).strip("-"),
        }
    return out


def coefficients(card: dict) -> str:
    """The front page's cell: the coefficients in use, the run that fitted them, the runs that faced them."""
    a = ACCELERATORS[card["fit"]]
    text = f"eff_mem {a.achieved_bandwidth:g}, mfu {a.mfu:g}"
    if card["fitted_by"] is None:
        return text + ": the spec sheet's, no run has fitted them"
    text += f": run {card['fitted_by']}'s fit"
    return text + (f", faced by {span(card['faced_by'])}" if card["faced_by"] else ", not yet faced")


def journal_md(journal: list[Run]) -> str:
    """docs/benchmarks/README.md: one section per card, one row per run."""
    lines = [
        "# The runs",
        "",
        "Every run on a rented card: what it asked, what it found, and which",
        "coefficients its predictions came from. Generated by `python3 bench/runs.py`",
        "from the front matter of each report, so edit the report, never this page",
        "([adding-a-run.md](../adding-a-run.md) §4).",
        "",
        "Every prediction the runs faced, drawn: [predicted-vs-measured.svg](predicted-vs-measured.svg).",
        "The sheet written before each run: [runsheets/](runsheets/). The evidence, verbatim: [raw/](raw/).",
    ]
    for card in cards(journal).values():
        mine = [r for r in journal if r.id in card["runs"]]
        if not mine:
            continue
        lines += ["", f"## {card['name']}", "",
                  f"Coefficients in use: `{card['fit']}` — {coefficients(card)}.", "",
                  "| Run | Date | Asked | Found | Coefficients |",
                  "|---|---|---|---|---|"]
        for r in mine:
            coef = f"predicted with `{r.predicted_with}`"
            if r.fitted:
                coef += f"; fitted `{r.fitted}`"
            name = pathlib.PurePosixPath(r.report).name
            lines.append(f"| [{r.number}]({name}) | {r.date} | {r.asked} | {r.found} | {coef} |")
    return "\n".join(lines) + "\n"


def readme_block(journal: list[Run]) -> str:
    """The front page's runs table, between its markers: one row per card."""
    lines = [BEGIN,
             "| Card | Runs | Coefficients in use | Latest run |",
             "|---|---|---|---|"]
    for card in cards(journal).values():
        mine = [r for r in journal if r.id in card["runs"]]
        if not mine:
            continue
        last = mine[-1]
        lines.append(f"| {card['name']} | [{len(mine)}]({card['journal']}) "
                     f"| {coefficients(card)} "
                     f"| [run {last.number}]({last.report}), {last.date}: {last.found} |")
    return "\n".join(lines + [END])


def with_block(page: str, block: str) -> str:
    """The front page with its runs block replaced; refuses a page without markers."""
    head, sep1, rest = page.partition(BEGIN)
    _, sep2, tail = rest.partition(END)
    if not (sep1 and sep2):
        raise ValueError(f"README.md has no {BEGIN} ... {END} block")
    return head + block + tail


def outputs() -> dict[pathlib.Path, str]:
    """Every file this module writes, as path -> content. The test reads this."""
    journal = runs()
    page = FRONT_PAGE.read_text(encoding="utf-8")
    return {JOURNAL: journal_md(journal),
            FRONT_PAGE: with_block(page, readme_block(journal))}


def main() -> None:
    # --check-readme: is the README on stdin (the staged one, from the hook) current?
    if sys.argv[1:] == ["--check-readme"]:
        page = sys.stdin.read()
        try:
            raise SystemExit(0 if with_block(page, readme_block(runs())) == page else 1)
        except ValueError as exc:        # no markers: stale by definition
            raise SystemExit(f"bench/runs.py: {exc}")
    for path, text in outputs().items():
        path.write_text(text, encoding="utf-8")
    journal = runs()
    print(f"{len(journal)} runs on {len({r.card for r in journal})} cards: "
          f"{JOURNAL.relative_to(ROOT)}, {FRONT_PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
