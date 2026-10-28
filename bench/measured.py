"""Predicted vs measured, scored against a real run -- roofline.py TODO 6.

Reads the raw results of one `vllm bench serve` sweep and prints what the model
predicted beside what the card did. SLO.md section 9 sets the format and the
reason for it: every row names the coefficient that produced its prediction,
whether that coefficient was itself fitted to this same measurement, and the
accelerator. Drop any of the three and the table quietly compares a prediction
against its own fit, which always agrees.

Separate from predictions.py for the same reason predictions.py is separate from
roofline.py: this file changes when a run happens, that one when an interesting
operating point moves, and roofline.py only when the physics does.

The argument is docs/benchmarks/l40s-baseline.md; this file is the arithmetic
behind it, so a figure quoted there fails here rather than drifting silently.

    python3 bench/measured.py
"""

import json
import os

from roofline import (
    L40S,
    L40S_RUN1,
    QWEN3_8B,
    kv_cache_tokens,
    tpot_floor,
    ttft_floor,
)

RUN = "l40s-2026-08-18"
RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "benchmarks", "raw", RUN, "results.jsonl",
)

GMU = 0.90
TPOT_TARGET = 0.050         # SLO.md section 2, interactive

# From the startup log, which outranks any derivation here (SLO.md section 9).
# pod-console.log lines 39, 49, 50 and 95 of the raw capture.
LOGGED_KV_TOKENS = 168_985
LOGGED_MAX_CONCURRENCY = 18.78      # at max_model_len 9 000
LOGGED_WEIGHTS_GIB = 15.27
LOGGED_KV_GIB = 23.23

# The generation was 200 tokens with EOS ignored, so context grows from the input
# length to input + 200 over the run. The mean is what a decode step sees on
# average, and it is the number every prediction below is taken at.
OUTPUT_TOKENS = 200

# At the top level the engine never ran the 45 sequences asked for: the KV pool
# seats fewer. concurrency/c45-metrics-maxima.txt, num_requests_running.
RAN_INSTEAD = {45: 41}


def levels():
    """The run's twelve results, each reduced to what a prediction needs."""
    with open(RESULTS) as f:
        for line in f:
            r = json.loads(line)
            prompts = r["num_prompts"]
            yield {
                "asked": r["max_concurrency"],
                "ran": RAN_INSTEAD.get(r["max_concurrency"], r["max_concurrency"]),
                "input_len": r["total_input_tokens"] // prompts,
                "context": r["total_input_tokens"] // prompts + OUTPUT_TOKENS // 2,
                # median ITL, not TPOT: TPOT is a request's mean inter-token
                # latency, and under chunked prefill some of its steps also carry
                # a slice of another request's prefill. The median step is the
                # closest this benchmark reports to a decode step alone, and the
                # gap between the two is a scheduler's, not the memory bus's.
                "decode_step_ms": r["median_itl_ms"],
                "tpot_p50_ms": r["median_tpot_ms"],
                "tpot_p99_ms": r["p99_tpot_ms"],
                "ttft_p50_ms": r["median_ttft_ms"],
                "output_tps": r["output_throughput"],
                "total_tps": r["total_token_throughput"],
            }


def implied_bandwidth_efficiency(model, accel, level) -> float:
    """What fraction of peak bandwidth the measured decode step actually reached.

    tpot_floor divides by accel.achieved_bandwidth, so multiplying it back out
    recovers the time at 100% of peak -- and that time over the measured one is
    the coefficient the card earned, independent of whatever was assumed.
    """
    at_peak = tpot_floor(
        model, accel, level["ran"], level["context"]
    ).seconds * accel.achieved_bandwidth
    return at_peak / (level["decode_step_ms"] / 1000)


def implied_mfu(model, accel, level) -> float:
    """Same trick on the compute side, against the measured TTFT.

    Only meaningful where the request had the card to itself: with other prefills
    in flight the number absorbs contention and stops being a utilisation.
    """
    at_peak = ttft_floor(
        model, accel, level["input_len"]
    ).seconds * accel.mfu
    return at_peak / (level["ttft_p50_ms"] / 1000)


def rule(width: int = 104) -> None:
    print("-" * width)


def checkpoint_a() -> None:
    print()
    print("CHECKPOINT A -- the KV pool, read from the startup log before any load")
    print("The one row taken without a benchmark, and the one the run failed worst.")
    print()
    derived = kv_cache_tokens(QWEN3_8B, L40S, GMU)
    error = (derived - LOGGED_KV_TOKENS) / LOGGED_KV_TOKENS * 100
    print(f"  derived  kv_cache_tokens : {derived:>12,.0f}")
    print(f"  logged   GPU KV cache    : {LOGGED_KV_TOKENS:>12,}")
    print(f"  the derivation is        : {error:>+11.1f}%")
    print()
    print("  The gap is three terms paid before the cache is carved -- non-torch")
    print("  allocations, the activation peak and the CUDA graph pool -- plus the")
    print("  card showing 44.39 GiB where the nameplate says 48 GB. Breakdown in")
    print("  docs/benchmarks/l40s-baseline.md section 2.")
    print()
    for ctx in (4000, 9000):
        print(f"  sequences at {ctx:>5} context : "
              f"derived {derived / ctx:>5.1f}   measured {LOGGED_KV_TOKENS / ctx:>5.1f}")
    print(f"  vLLM's own line at 9 000 : {LOGGED_MAX_CONCURRENCY:>5.2f}x")


def decode_table() -> None:
    print()
    print("DECODE STEP -- predicted against the measured median ITL")
    print("Two prediction columns, because a coefficient fitted to this run is not")
    print("evidence about it. 0.70 was the assumption; 0.83 is what these rows fitted.")
    print()
    print(f"{'level':>14} {'ctx':>6} {'@0.70':>9} {'err':>8} "
          f"{'@0.83*':>9} {'err':>8} {'measured':>9} {'implied':>8}")
    print(f"{'':>14} {'':>6} {'ms':>9} {'':>8} {'ms':>9} {'':>8} {'ms':>9} {'eff':>8}")
    rule()
    for lv in levels():
        pred_old = tpot_floor(QWEN3_8B, L40S, lv["ran"], lv["context"]).seconds * 1000
        pred_new = tpot_floor(QWEN3_8B, L40S_RUN1, lv["ran"], lv["context"]).seconds * 1000
        got = lv["decode_step_ms"]
        name = f"c={lv['asked']}, in {lv['input_len']}"
        print(f"{name:>14} {lv['context']:>6} "
              f"{pred_old:>9.2f} {(pred_old - got) / got * 100:>+7.1f}% "
              f"{pred_new:>9.2f} {(pred_new - got) / got * 100:>+7.1f}% "
              f"{got:>9.2f} "
              f"{implied_bandwidth_efficiency(QWEN3_8B, L40S, lv):>8.3f}")
    rule()
    print("  * fitted to these same rows -- a hypothesis for run 2, not a result.")
    print("    Coefficient: achieved_bandwidth. Card: L40S. Phase: decode.")


def prefill_table() -> None:
    print()
    print("PREFILL -- predicted against the measured median TTFT")
    print("Only the first row is a utilisation. Every other level shares the card,")
    print("so its 'implied mfu' is contention wearing a coefficient's clothes.")
    print()
    print(f"{'level':>14} {'prompt':>7} {'@0.45':>9} {'err':>8} {'measured':>9} {'implied':>8}")
    print(f"{'':>14} {'tokens':>7} {'ms':>9} {'':>8} {'ms':>9} {'mfu':>8}")
    rule()
    for lv in levels():
        pred = ttft_floor(QWEN3_8B, L40S, lv["input_len"]).seconds * 1000
        got = lv["ttft_p50_ms"]
        name = f"c={lv['asked']}, in {lv['input_len']}"
        alone = "" if lv["asked"] == 1 else "  (contended)"
        print(f"{name:>14} {lv['input_len']:>7} "
              f"{pred:>9.1f} {(pred - got) / got * 100:>+7.1f}% "
              f"{got:>9.1f} "
              f"{implied_mfu(QWEN3_8B, L40S, lv):>8.3f}{alone}")
    rule()


def the_line() -> None:
    """Where the sweep crosses the 50 ms TPOT target -- four different answers."""
    print()
    print("THE 50 ms LINE -- what 'max_num_seqs by latency' means depends on the question")
    print()
    at_4000 = [lv for lv in levels() if lv["input_len"] == 4000]
    at_4000.sort(key=lambda lv: lv["asked"])

    def crossing(key):
        below = None
        for lv in at_4000:
            if lv[key] > TPOT_TARGET * 1000:
                return below, lv
            below = lv
        return below, None

    for key, label in (
        ("decode_step_ms", "measured decode step (median ITL)"),
        ("tpot_p50_ms", "measured TPOT p50"),
        ("tpot_p99_ms", "measured TPOT p99  <- the SLO metric"),
    ):
        lo, hi = crossing(key)
        lo_s = f"c={lo['asked']} at {lo[key]:.2f} ms" if lo else "already over at c=1"
        hi_s = f"c={hi['asked']} at {hi[key]:.2f} ms" if hi else "never crosses"
        print(f"  {label:<38} {lo_s:>22}  ->  {hi_s}")
    print()
    print("  The distance between the first line and the last is not bandwidth.")
    print("  It is chunked prefill injected into other requests' decode steps:")
    print(f"  {'c':>4} {'TPOT p50 / median ITL':>24}")
    for lv in at_4000:
        print(f"  {lv['asked']:>4} {lv['tpot_p50_ms'] / lv['decode_step_ms']:>24.2f}")


def cost_table(hourly_rate: float = 0.99) -> None:
    print()
    print(f"COST -- measured throughput at ${hourly_rate:.2f}/h")
    print("Measured, so unlike every $/1M figure in predictions.py these are not floors.")
    print()
    print(f"{'level':>14} {'out tok/s':>10} {'tot tok/s':>10} "
          f"{'$/1M out':>10} {'$/1M total':>11}")
    rule()
    for lv in levels():
        name = f"c={lv['asked']}, in {lv['input_len']}"
        print(f"{name:>14} {lv['output_tps']:>10.1f} {lv['total_tps']:>10.1f} "
              f"{hourly_rate / (lv['output_tps'] * 3600) * 1e6:>10.3f} "
              f"{hourly_rate / (lv['total_tps'] * 3600) * 1e6:>11.4f}")
    rule()



# The ledger's view of this run (bench/ledger.py): the conditions its rows carry,
# and every decode step it measured with the coefficients each one faced.
LEDGER = {
    "run": "l40s-1",
    "model": QWEN3_8B,
    "engine": "vLLM",
    "engine_version": "0.27.1",
    "regime": "in",
    "mode": "seats",
    "host": {"provider": "RunPod, North American DC", "driver": "580.159.04 / CUDA 13.0"},
    "sheet": "docs/benchmarks/runsheets/l40s-first-run.md",
    # The public history starts at this commit, after the run; the sheet's own
    # date and the report's say which came first.
    "sheet_commit": "f7172a6",
    "raw": RESULTS,
}


def ledger_points():
    """Every level's median ITL: the prior faced, and L40S_RUN1 fitted -- a hypothesis here."""
    for lv in levels():
        yield {
            "batch": lv["ran"],
            "context": lv["context"],
            "prompt": lv["input_len"],
            "step_ms": lv["decode_step_ms"],
            "faced": [("l40s", "point"), ("l40s-run1", "point")],
        }


if __name__ == "__main__":
    print(f"Run: {RUN}   raw evidence: docs/benchmarks/raw/{RUN}/")
    print(f"Argument and conclusions: docs/benchmarks/l40s-baseline.md")
    checkpoint_a()
    decode_table()
    prefill_table()
    the_line()
    cost_table()
    print()
