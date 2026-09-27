"""Export the performance model as data for `site/`, and the grid that proves the port.

Every number the page shows lives in Python and is written to site/data/model.json;
the formulas exist twice, held equal by site/data/golden.json (what_if_point() over
a grid) and site/selftest.js. The files stay byte-stable, carry no Infinity, and
each .json has a .js twin that loads from file:// -- site/README.md has the why.

    python3 bench/export_site_data.py
"""

from __future__ import annotations

import json
import pathlib

import measured
import measured_mi300x_run1
import measured_run2
import measured_run3
import plot_predicted_vs_measured as plot
from predictions import (DEFAULT_MODEL, GMU, HOURLY_RATES, INTERFERENCE_FITS,
                         INTERFERENCE_MODEL, MODELS, SLO_CLASSES,
                         TPOT_TARGET, TTFT_TARGET, what_if_point)
from roofline import ACCELERATORS, FLOPS_PER_MAC

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE_DATA = ROOT / "site" / "data"
SYMPTOM_MAP = ROOT / "docs" / "symptom-map.json"

REPORTS = {1: plot.BASELINE, 2: plot.RUN2, 3: plot.RUN3, 4: plot.MI300X_RUN1}

# The point every one-factor-at-a-time row departs from: this repository's own
# operating point, 4 000 in, 200 out, at the cached hit rate run 3 measured.
BASE = dict(accelerator="l40s-run1", model=DEFAULT_MODEL,
            prompt_tokens=4000, output_tokens=200,
            hit_rate=0.8, tpot_target=TPOT_TARGET, ttft_target=TTFT_TARGET,
            kv_dtype_bytes=2, gpu_memory_utilization=GMU, hourly_rate=None)


def _point(**inputs) -> dict:
    """what_if_point() from the page's vocabulary: output tokens, not context."""
    kwargs = dict(inputs)
    kwargs["context_len"] = kwargs.pop("prompt_tokens") + kwargs.pop("output_tokens")
    kwargs["prompt_tokens"] = inputs["prompt_tokens"]
    return what_if_point(**kwargs)


def model_data() -> dict:
    """Every constant the page needs, from the modules that own them."""
    accelerators = {}
    for key, accel in ACCELERATORS.items():
        rate, rate_provenance = HOURLY_RATES[key]
        accelerators[key] = {
            "name": accel.name,
            "memory_bytes": accel.memory_bytes,
            "peak_bandwidth": accel.peak_bandwidth,
            "peak_flops": accel.peak_flops,
            "achieved_bandwidth": accel.achieved_bandwidth,
            "mfu": accel.mfu,
            "provenance": accel.provenance,
            "measured": accel.provenance.startswith("measured"),
            "hourly_rate": rate,
            "hourly_rate_provenance": rate_provenance,
        }
    interference = {
        key: {"slope_s_per_seat": slope, "intercept_s": intercept,
              "provenance": provenance}
        for key, (slope, intercept, provenance) in INTERFERENCE_FITS.items()
    }
    models = {
        key: {
            "name": m.name,
            "params_total": m.params_total,
            "params_non_embedding": m.params_non_embedding,
            "num_layers": m.num_layers,
            "num_kv_heads": m.num_kv_heads,
            "head_dim": m.head_dim,
            "weight_dtype_bytes": m.weight_dtype_bytes,
            "kv_dtype_bytes": m.kv_dtype_bytes,
            "kv_bytes_per_token": m.kv_bytes_per_token,
            # Which of the two the page may draw a measured point over, and
            # lend the interference fit to. Exactly one, and it says so here
            # rather than in the JavaScript, so the fact has one home.
            "measured": key == INTERFERENCE_MODEL,
            "served_by_this_stack": key == DEFAULT_MODEL,
            "provenance": ("config.json and the model card; docs/model-anatomy.md"
                           if key == DEFAULT_MODEL else
                           "config.json and the model card; never served or measured here, "
                           "docs/audience.md"),
        }
        for key, m in MODELS.items()
    }
    return {
        "models": models,
        "accelerators": accelerators,
        "interference": interference,
        "slo_classes": dict(SLO_CLASSES, source="docs/SLO.md#2-targets"),
        "defaults": {
            "accelerator": "l40s-run1",
            "model": DEFAULT_MODEL,
            "prompt_tokens": 4000,
            "output_tokens": 200,
            "gpu_memory_utilization": GMU,
            "kv_dtype_bytes": 2,
            "hit_rate": 0.0,
        },
        "constants": {"FLOPS_PER_MAC": FLOPS_PER_MAC},
        "runs": {
            str(n): {"label": run.label, "date": run.date, "card": run.card,
                     "report": REPORTS[n]}
            for n, run in plot.RUNS.items()
        },
        "measured": measured_points(),
        "generated_by": "bench/export_site_data.py",
    }


def measured_points() -> dict:
    """Every run's levels, each set carrying the geometry the page hides it outside of."""
    # The model every run was taken on, so the page reads it from the data
    # instead of carrying a copy of the fact in draw.js.
    run1 = list(measured.levels())
    run2_a = measured_run2.block("A")
    run3_h0 = measured_run3.levels("run3-a-h00-clean")
    run3_h80 = measured_run3.levels("run3-a-h80") + measured_run3.levels("run3-a-h80-ext")
    run3_rates = measured_run3.levels("run3-b-h80") + measured_run3.levels("run3-b-h80-ext")
    mi_rows, mi_gauges = measured_mi300x_run1.rows(), measured_mi300x_run1.gauges()

    def seats(levels):
        return [{
            "concurrency": lv["harness_load"],
            "p50_tpot_ms": lv["p50_tpot_ms"],
            "p99_tpot_ms": lv["p99_tpot_ms"],
            "median_itl_ms": lv["median_itl_ms"],
            "output_throughput": lv["output_throughput"],
            "request_goodput": lv["request_goodput"],
            "measured_hit_rate": lv["harness_measured_hit_rate"],
            "preemptions": lv["harness_preemptions"],
        } for lv in levels]

    return {
        "model": INTERFERENCE_MODEL,
        "run1_decode": {
            "run": 1, "mode": "closed", "accelerator": "l40s",
            "prompt_tokens": 4000, "output_tokens": 200, "kv_dtype_bytes": 2,
            "gpu_memory_utilization": GMU, "hit_rate": 0.0,
            "max_num_batched_tokens": 2048,
            "levels": [{
                "asked": lv["asked"], "ran": lv["ran"],
                "decode_step_ms": lv["decode_step_ms"],
                "tpot_p50_ms": lv["tpot_p50_ms"], "tpot_p99_ms": lv["tpot_p99_ms"],
                "ttft_p50_ms": lv["ttft_p50_ms"], "output_tps": lv["output_tps"],
            } for lv in run1],
        },
        "run2_rates": {
            "run": 2, "mode": "poisson", "accelerator": "l40s",
            "prompt_tokens": 1500, "output_tokens": 200, "kv_dtype_bytes": 2,
            "gpu_memory_utilization": GMU, "hit_rate": 0.0,
            "max_num_batched_tokens": 2048,
            "levels": [{
                "rate": lv["rate"], "goodput": lv["goodput"],
                "request_throughput": lv["req_thr"], "output_tps": lv["out_tps"],
                "ttft_p99_ms": lv["ttft_p99_ms"], "tpot_p50_ms": lv["tpot_p50_ms"],
                "tpot_p99_ms": lv["tpot_p99_ms"], "peak_concurrency": lv["peak_conc"],
            } for lv in run2_a],
        },
        "run3_seats": {
            "run": 3, "mode": "closed", "accelerator": "l40s",
            "prompt_tokens": 4000, "output_tokens": 200, "kv_dtype_bytes": 2,
            "gpu_memory_utilization": GMU, "max_num_batched_tokens": 2048,
            "series": [
                {"hit_rate": 0.0, "levels": seats(run3_h0)},
                {"hit_rate": 0.8, "levels": seats(run3_h80)},
            ],
        },
        "run3_rates": {
            "run": 3, "mode": "poisson", "accelerator": "l40s",
            "prompt_tokens": 1500, "output_tokens": 200, "kv_dtype_bytes": 2,
            "gpu_memory_utilization": GMU, "hit_rate": 0.8,
            "max_num_batched_tokens": 2048,
            "levels": [{
                "rate": lv["harness_load"],
                "request_goodput": lv["request_goodput"],
                "request_throughput": lv["request_throughput"],
                "output_throughput": lv["output_throughput"],
                "ttft_p99_ms": lv["p99_ttft_ms"], "tpot_p50_ms": lv["p50_tpot_ms"],
                "max_running": lv["harness_max_num_requests_running"],
                "max_waiting": lv["harness_max_num_requests_waiting"],
            } for lv in run3_rates],
        },
        # Prefix caching off, so one cold series; the median ITL is a decode
        # step only through c = 32 (docs/benchmarks/mi300x-run1.md section 3).
        "mi300x_run1_seats": {
            "run": 4, "mode": "closed", "accelerator": "mi300x",
            "prompt_tokens": 4000, "output_tokens": 200, "kv_dtype_bytes": 2,
            "gpu_memory_utilization": GMU, "hit_rate": 0.0,
            "max_num_batched_tokens": measured_mi300x_run1.MAX_NUM_BATCHED_TOKENS,
            "decode_step_through": measured_mi300x_run1.DECODE_STEP_THROUGH,
            "levels": [{
                "concurrency": r["c"],
                "p50_tpot_ms": r["tpot_p50_ms"], "p99_tpot_ms": r["tpot_p99_ms"],
                "median_itl_ms": r["itl_ms"], "output_throughput": r["output_tps"],
                "max_running": max(x["running"] for x in mi_gauges[name]),
            } for name, r in mi_rows.items() if "-in" not in name],
        },
        # Where TPOT p99 crossed 50 ms, linear between two levels: the two
        # figures docs/SLO.md section 6 quotes as 12.5 and 37.8.
        "seat_crossings": {
            "tpot_p99_target_ms": measured_run3.TPOT_TARGET_MS,
            "h0": measured_run3.crossing(run3_h0, "p99_tpot_ms", measured_run3.TPOT_TARGET_MS),
            "h80": measured_run3.crossing(run3_h80, "p99_tpot_ms", measured_run3.TPOT_TARGET_MS),
        },
    }


def golden_grid() -> list[dict]:
    """Inputs and the answer, over a grid the page's sliders can reach.

    A core grid over the regime-changing inputs (where a floor() disagreeing on
    23.999... would show), one-factor steps from this repository's operating
    point, and the edges where the answer is a sentence rather than a number.
    """
    rows: list[dict] = []

    def add(**inputs):
        rows.append({"inputs": inputs, "expected": _point(**inputs)})

    for accelerator in sorted(ACCELERATORS):
        for prompt in (512, 2000, 4000, 8000, 32000):
            for hit_rate in (0.0, 0.5, 0.8):
                for tpot in (0.050, 0.200):
                    for kv in (2, 1):
                        add(**dict(BASE, accelerator=accelerator, prompt_tokens=prompt,
                                   hit_rate=hit_rate, tpot_target=tpot,
                                   kv_dtype_bytes=kv))
    for output in (1, 1000, 4096):
        add(**dict(BASE, output_tokens=output))
    for ttft in (0.100, 3.000):
        add(**dict(BASE, ttft_target=ttft))
    for gmu in (0.50, 0.95):
        add(**dict(BASE, gpu_memory_utilization=gmu))
    for rate in (0.50, 5.00):
        add(**dict(BASE, hourly_rate=rate))
    add(**dict(BASE, hit_rate=1.0))
    add(**dict(BASE, hit_rate=0.0))
    for accelerator in ("l40s", "mi300x"):
        add(**dict(BASE, accelerator=accelerator))
    # The edges, and the second model over the regimes that decide its answer,
    # so its "service: not derivable" is held to Python too.
    for prompt in (512, 4000, 32000):
        for tpot in (0.050, 0.200):
            for kv in (2, 1):
                add(**dict(BASE, model="qwen2.5-7b", prompt_tokens=prompt,
                           tpot_target=tpot, kv_dtype_bytes=kv))
    for accelerator in ("l40s", "mi300x-run1", "mi300x"):
        add(**dict(BASE, model="qwen2.5-7b", accelerator=accelerator))
    add(**dict(BASE, model="qwen2.5-7b", hit_rate=0.0))
    add(**dict(BASE, tpot_target=0.020))
    add(**dict(BASE, gpu_memory_utilization=0.30))
    add(**dict(BASE, accelerator="mi300x", prompt_tokens=32000, hit_rate=0.0))
    return rows


def dumps(payload) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"


def dumps_rows(rows: list[dict]) -> str:
    """One row per line, so a diff names the operating point that moved."""
    body = ",\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows)
    return "[\n" + body + "\n]\n"


def js_twin(name: str, json_text: str) -> str:
    return f"window.{name} = {json_text.rstrip()};\n"


def outputs() -> dict[str, str]:
    """Every file this module writes, as path -> content. The test reads this."""
    model_json = dumps(model_data())
    golden_json = dumps_rows(golden_grid())
    symptom_json = SYMPTOM_MAP.read_text(encoding="utf-8")
    json.loads(symptom_json)                       # a twin of invalid JSON is worse than none
    return {
        "model.json": model_json,
        "model.js": js_twin("MODEL_DATA", model_json),
        "golden.json": golden_json,
        "golden.js": js_twin("GOLDEN", golden_json),
        "symptom-map.js": js_twin("SYMPTOM_MAP", symptom_json),
    }


def main() -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    files = outputs()
    for name, text in files.items():
        (SITE_DATA / name).write_text(text, encoding="utf-8")
    rows = len(json.loads(files["golden.json"]))
    print(f"wrote {len(files)} files to {SITE_DATA.relative_to(ROOT)}/: "
          f"{rows} golden rows, {len(ACCELERATORS)} accelerators")


if __name__ == "__main__":
    main()
