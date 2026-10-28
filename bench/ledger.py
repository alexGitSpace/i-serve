"""The ledger: every decode step a run measured, beside the ceiling it faced.

One line of docs/benchmarks/ledger.jsonl per point, from the LEDGER and
ledger_points() of each contributing measured_*.py (docs/adding-a-run.md
section 4). The share and every status are computed here, once, so no module
scores itself. The fields: docs/GLOSSARY.md, *Ledger*.

    python3 bench/ledger.py        # docs/benchmarks/ledger.jsonl
"""

from __future__ import annotations

import importlib
import json
import pathlib

import runs
from roofline import ACCELERATORS, decode_step_bytes, tpot_floor

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "benchmarks" / "ledger.jsonl"

# The modules with rows: MI300X runs 3 and 4 measured a pair of engines and the
# router, and faced no decode step against the ceiling, so they have none.
MODULES = (
    "measured",
    "measured_run2",
    "measured_run3",
    "measured_mi300x_run1",
    "measured_mi300x_run2",
    "measured_mi300x_run5",
)

# A point prediction survives inside this band: the checkpoint gate's +-5 %
# (docs/adding-a-run.md section 1), the one tolerance the runs already use.
TOLERANCE = 0.05
NOT_RECORDED = "not recorded"
HOST_FIELDS = ("provider", "instance", "kernel", "driver", "thermal")
REGIMES = ("in", "out", "not disclosed")
MODES = ("single-stream", "seats")
CLAIMS = ("point", "bound")
STATUSES = ("hypothesis", "survived", "falsified")


def status(claim: str, fitted_here: bool, predicted_ms: float, measured_ms: float) -> str:
    """SLO.md section 9's rule: a coefficient fitted to the run it faces proves nothing."""
    if fitted_here:
        return "hypothesis"
    if claim == "bound":                    # a floor: the step may not be shorter
        return "survived" if measured_ms >= predicted_ms else "falsified"
    return "survived" if abs(predicted_ms - measured_ms) / measured_ms <= TOLERANCE \
        else "falsified"


def relative(path: str) -> str:
    """A module's absolute raw path, as the ledger prints it."""
    return pathlib.Path(path).resolve().relative_to(ROOT).as_posix()


def row(run: runs.Run, conditions: dict, point: dict) -> dict:
    """One ledger line: the point's ceiling at peak bandwidth, its share, and what it faced."""
    model = conditions["model"]
    batch, context = point["batch"], point["context"]
    step_bytes = decode_step_bytes(model, batch, context)
    at_peak_ms = step_bytes / ACCELERATORS[run.card].peak_bandwidth * 1e3
    faced = []
    for key, claim in point["faced"]:
        if runs.card_of(key) != run.card:
            raise ValueError(f"{run.id}: faced {key!r}, not one of {run.card}'s coefficients")
        if claim not in CLAIMS:
            raise ValueError(f"{run.id}: claim {claim!r} is not one of {CLAIMS}")
        predicted = tpot_floor(model, ACCELERATORS[key], batch, context).seconds * 1e3
        fitted_here = key == run.fitted
        faced.append({
            "with": key,
            "claim": claim,
            "predicted_ms": round(predicted, 3),
            "fitted_here": fitted_here,
            "status": status(claim, fitted_here, predicted, point["step_ms"]),
        })
    for name in ("regime", "mode"):
        allowed = REGIMES if name == "regime" else MODES
        if conditions[name] not in allowed:
            raise ValueError(f"{run.id}: {name} {conditions[name]!r} is not one of {allowed}")
    line = {
        "run": run.id,
        "date": run.date,
        "card": run.card,
        "host": {f: conditions["host"].get(f, NOT_RECORDED) for f in HOST_FIELDS},
        "engine": conditions["engine"],
        "engine_version": conditions["engine_version"],
        "model": model.name,
        "revision": conditions.get("revision", NOT_RECORDED),
        "weights": {
            "bits_per_weight": 8 * model.weight_dtype_bytes,
            "bytes": int(model.weights_bytes),
            # config.json and the checkpoint's parameter count, not a file on disk
            "bytes_from": "config",
        },
        "regime": conditions["regime"],
        "mode": conditions["mode"],
        "point": {"batch": batch, "context": context, "prompt": point["prompt"]},
        "ceiling": {
            "step_bytes": int(step_bytes),
            "step_ms": round(at_peak_ms, 3),
            "tok_s": round(batch / at_peak_ms * 1e3, 1),
        },
        "measured": {"step_ms": round(point["step_ms"], 3), "repeats": point.get("repeats", 1)},
        "share": round(at_peak_ms / point["step_ms"], 3),
        "faced": faced,
        "sheet": conditions["sheet"],
        "sheet_commit": conditions["sheet_commit"],
        "raw": relative(conditions["raw"]),
    }
    if "note" in point:
        line["note"] = point["note"]
    for path in (line["sheet"], line["raw"]):
        if not (ROOT / path).exists():
            raise ValueError(f"{run.id}: {path} does not exist")
    return line


def rows() -> list[dict]:
    """Every contributing module's points, in journal order."""
    journal = {r.id: r for r in runs.runs()}
    out: list[dict] = []
    for name in MODULES:
        module = importlib.import_module(name)
        run = journal[module.LEDGER["run"]]
        out += [row(run, module.LEDGER, point) for point in module.ledger_points()]
    return out


def outputs() -> dict[pathlib.Path, str]:
    """Every file this module writes, as path -> content. The test reads this."""
    return {OUT: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows())}


def main() -> None:
    for path, text in outputs().items():
        path.write_text(text, encoding="utf-8")
    lines = rows()
    print(f"{len(lines)} rows from {len({r['run'] for r in lines})} runs: "
          f"{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
