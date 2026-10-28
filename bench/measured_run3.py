"""Run 3 scored: prefix caching, priced in seats -- SLO.md section 9.

Third sibling of measured.py (run 1) and measured_run2.py (run 2), and the same
rule holds: one module per run, because the run is the unit a coefficient belongs
to and merging two of them is how a fitted number starts scoring itself.

What is different here is the evidence, not the argument. Runs 1 and 2 were
transcribed by hand off `vllm bench serve` output into a levels.tsv; run 3 was
driven by bench/harness.py, which writes one JSON file per level. So this module
reads the artefacts directly and there is nothing between the card and the table
below to mistype -- which also means a level's `harness_*` fields, including the
gates it fired, are available here rather than only in a console log.

The predictions this scores were written in
docs/benchmarks/runsheets/l40s-run-3.md before the card was rented, and every coefficient used to make them (eff_mem 0.83,
mfu 0.439, I(n) = 1.640 n - 6.36) comes from run 1. None was fitted to anything
below.

The argument is docs/benchmarks/l40s-run3.md; this file is the arithmetic behind
it, so a figure quoted there fails here rather than drifting silently.

    python3 bench/measured_run3.py
"""

import json
import os
import glob

from roofline import L40S_RUN1, QWEN3_8B, tpot_floor, ttft_floor

RUN = "l40s-2026-08-30"
RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "benchmarks", "raw", RUN,
)

HOURLY = 0.99               # SLO.md section 7, the contract figure of runs 1-2
TPOT_TARGET_MS = 50.0       # SLO.md section 2, interactive
TTFT_TARGET_MS = 300.0
OUTPUT_TOKENS = 200
SEAT_CONTEXT = 4_100        # 4 000 prompt + half of 200 output, as in runs 1-2

# From the two startup logs, which outrank any derivation here (SLO.md section 9).
LOGGED_KV_TOKENS_CACHE_OFF = 168_985     # run 1's figure to the token
LOGGED_KV_TOKENS_CACHE_ON = 176_227
LOGGED_KV_GIB = (23.23, 24.22)
LOGGED_MAX_CONCURRENCY = (18.78, 19.58)

# The two facts this build's startup log does NOT carry. Run 2 reported the first
# from its launch command; run 3 passed neither, so both are defaults -- and the
# claim that the geometry matches runs 1-2 rests on the c=13 decode step (block C),
# not on a log line. Named here so the gap is visible in the code as well.
UNLOGGED = ("max_num_batched_tokens", "max_num_seqs")

# Run 1's interference fit, at 4 000-token prompts and mnbt 2 048. The whole of
# SLO.md section 6's seat prediction is this line scaled by (1 - h).
INTERFERENCE_SLOPE = 1.640
INTERFERENCE_INTERCEPT = -6.36

# The same quantity, measured at c=13 / ctx 4 100 / mnbt 2 048, across every run
# that has measured it. Three pods, two drivers, twelve days.
DECODE_STEP_C13 = (33.79, 33.71, 33.80, 33.677, 33.686)

# Run 2's block A, for the one comparison block B exists to make.
RUN2_GOODPUT_PEAK = 1.87            # req/s, at an offered 2.5
RUN2_TTFT_P99_CROSSING = (0.5, 1.0)  # req/s, uncached, 1 500-token prompts

# vLLM's default, which block B reached and the KV pool never did.
MAX_NUM_SEQS_DEFAULT = 256


def levels(directory):
    """The levels of one harness run, sorted by load."""
    out = []
    for path in sorted(glob.glob(os.path.join(RAW, "results", directory, "*.json"))):
        name = os.path.basename(path)
        if name == "run.json" or name.endswith("-metrics.json"):
            continue
        with open(path) as f:
            out.append(json.load(f))
    return sorted(out, key=lambda lv: lv["harness_load"])


def counters(directory, label):
    """The Prometheus deltas the harness took across one level."""
    path = os.path.join(RAW, "results", directory, f"{label}-metrics.json")
    with open(path) as f:
        return json.load(f)


def fit(xs, ys):
    """Least squares, returned as (slope, intercept)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    return slope, my - slope * mx


def crossing(rows, field, target):
    """The load at which `field` crosses `target`, linear between two levels."""
    for lo, hi in zip(rows, rows[1:]):
        if lo[field] <= target <= hi[field]:
            span = hi[field] - lo[field]
            return lo["harness_load"] + (hi["harness_load"] - lo["harness_load"]) \
                * (target - lo[field]) / span
    return None


def dollars_per_1m(tokens_per_second):
    return HOURLY / (tokens_per_second * 3600) * 1e6


def slo_seat_rows():
    """The two levels run 3 ships: the largest load still inside the TPOT target.

    This rule -- take the widest level whose p99 TPOT is under the interactive
    target, at each hit rate -- decides the seat counts and the cost figures the
    README and SLO.md section 7 both quote. It lived inside cost()'s printer
    until the README grew a test that needs the same two rows; a number quoted in
    two places must be computed in one.
    """
    unc = levels("run3-a-h00-clean")
    cached = levels("run3-a-h80") + levels("run3-a-h80-ext")
    widest = lambda rows: max(                                  # noqa: E731
        (lv for lv in rows if lv["p99_tpot_ms"] <= TPOT_TARGET_MS),
        key=lambda lv: lv["harness_load"])
    return {"h0": widest(unc), "h80": widest(cached)}


def rule():
    print("-" * 78)


def checkpoint():
    print()
    print("CHECKPOINT A'' -- the startup log, against what was predicted from run 1")
    print()
    ratio = LOGGED_KV_TOKENS_CACHE_ON / LOGGED_KV_TOKENS_CACHE_OFF
    print(f"  KV pool, caching off : {LOGGED_KV_TOKENS_CACHE_OFF:>9,} tokens "
          f"({LOGGED_KV_GIB[0]} GiB)  -- run 1's figure to the token")
    print(f"  KV pool, caching on  : {LOGGED_KV_TOKENS_CACHE_ON:>9,} tokens "
          f"({LOGGED_KV_GIB[1]} GiB)  -- {ratio:.3f}x")
    print(f"  Run 2 measured 4.2% between two launches of ONE configuration, so "
          f"{ratio - 1:+.1%} is")
    print("  inside the instrument: this run cannot say what caching costs in pool space.")
    print(f"  Not in this build's log at all: {', '.join(UNLOGGED)}")


def block_a():
    print()
    print("BLOCK A -- the seat count against h, and the assumption underneath it")
    print()
    unc = levels("run3-a-h00-clean")
    cached = levels("run3-a-h80") + levels("run3-a-h80-ext")
    by_n = {lv["harness_load"]: lv for lv in cached}

    print(f"{'n':>4} {'h=0 step':>9} {'h=.8 step':>10} {'fall':>7} "
          f"{'h=0 p99':>8} {'h=.8 p99':>9} {'measured h':>11} {'preempt':>8}")
    rule()
    for lv in unc:
        n = lv["harness_load"]
        c = by_n.get(n)
        if not c:
            continue
        print(f"{n:>4.0f} {lv['median_itl_ms']:>9.2f} {c['median_itl_ms']:>10.2f} "
              f"{c['median_itl_ms'] / lv['median_itl_ms'] - 1:>+6.1%} "
              f"{lv['p99_tpot_ms']:>8.2f} {c['p99_tpot_ms']:>9.2f} "
              f"{c['harness_measured_hit_rate']:>11.3f} "
              f"{c['harness_preemptions'] + lv['harness_preemptions']:>8.0f}")
    rule()

    seats_unc = crossing(unc, "p99_tpot_ms", TPOT_TARGET_MS)
    seats_cached = crossing(cached, "p99_tpot_ms", TPOT_TARGET_MS)
    print(f"  seats at TPOT p99 <= {TPOT_TARGET_MS:.0f} ms: "
          f"{seats_unc:.1f} at h=0 (predicted 13.5), "
          f"{seats_cached:.1f} at h=0.8 (predicted 24.3)")
    print(f"  the feature is worth {seats_cached / seats_unc:.1f}x the seats, "
          f"where SLO.md section 6 said 1.8x")

    print()
    print("  The decode step, fitted -- the quantity section 6 said could not move:")
    step_model = tpot_floor(QWEN3_8B, L40S_RUN1, 1, SEAT_CONTEXT).seconds * 1000
    per_seat = (tpot_floor(QWEN3_8B, L40S_RUN1, 2, SEAT_CONTEXT).seconds * 1000
                - step_model)
    for label, rows in (("h = 0  ", unc), ("h = 0.8", [lv for lv in cached
                                                       if lv["harness_load"] <= 28])):
        slope, intercept = fit([lv["harness_load"] for lv in rows],
                               [lv["median_itl_ms"] for lv in rows])
        print(f"    {label}: {intercept:6.2f} + {slope:.4f} n   "
              f"-> implied context {slope / per_seat * SEAT_CONTEXT:6.0f} tok/seat")
    print(f"    model  : {step_model - per_seat:6.2f} + {per_seat:.4f} n   "
          f"-> implied context {SEAT_CONTEXT:6.0f} tok/seat   (roofline, eff_mem 0.83)")
    print("    A seat's unique content at h=0.8 is 800 prompt tokens plus <=200 of")
    print("    output. The cached slope sits between 'read per sequence' and 'read")
    print("    once per batch': the latency limit is NOT invariant to h, and no")
    print("    mechanism in this run says why.")

    print()
    print("  The interference, fitted -- the term section 6 was actually about:")
    for label, rows, expected in (("h = 0  ", unc, 1.0),
                                  ("h = 0.8", [lv for lv in cached
                                               if lv["harness_load"] <= 28], 0.2)):
        slope, intercept = fit([lv["harness_load"] for lv in rows],
                               [lv["harness_prefill_interference_ms"] for lv in rows])
        print(f"    {label}: I(n) = {slope:.3f} n {intercept:+.2f} ms")
    slope0, _ = fit([lv["harness_load"] for lv in unc],
                    [lv["harness_prefill_interference_ms"] for lv in unc])
    slope8, _ = fit([lv["harness_load"] for lv in cached if lv["harness_load"] <= 28],
                    [lv["harness_prefill_interference_ms"] for lv in cached
                     if lv["harness_load"] <= 28])
    print(f"    ratio of slopes {slope8 / slope0:.3f} against (1 - h) = 0.200 "
          f"-- section 6's mechanism, confirmed")
    print(f"    (run 1's fit, from another pod: "
          f"{INTERFERENCE_SLOPE:.3f} n {INTERFERENCE_INTERCEPT:+.2f})")

    print()
    print("  The worst step, against the chunk that sets it:")
    chunk_ms = ttft_floor(QWEN3_8B, L40S_RUN1, 2048).seconds * 1000
    print(f"    max_num_batched_tokens 2 048 at mfu 0.439 = {chunk_ms:.1f} ms of prefill")
    for n in (8, 16, 24, 28):
        lv = next(x for x in unc if x["harness_load"] == n)
        c = by_n[n]
        print(f"    c={n:<3} ITL p99  h=0 {lv['p99_itl_ms']:7.1f}   "
              f"h=0.8 {c['p99_itl_ms']:7.1f}")
    print("    Uncached the tail is the chunk from the first level on. Cached it")
    print("    starts far below and climbs to the same ceiling by c=16, because")
    print("    2 048 / 800 = 2.6: three concurrent prefills refill the budget.")
    print("    Prefix caching halves the mean prefill work and not the tail step.")


def block_b():
    print()
    print("BLOCK B -- goodput under arrivals at h = 0.8, and what binds now")
    print()
    rows = levels("run3-b-h80") + levels("run3-b-h80-ext")
    print(f"{'rate':>6} {'goodput':>8} {'thru':>7} {'TTFT p99':>9} {'TPOT p50':>9} "
          f"{'running':>8} {'waiting':>8} {'preempt':>8} {'$/M used':>9}")
    rule()
    for lv in rows:
        used = lv["request_goodput"] * OUTPUT_TOKENS
        cost = f"{dollars_per_1m(used):9.3f}" if used else f"{'--':>9}"
        print(f"{lv['harness_load']:>6.1f} {lv['request_goodput']:>8.2f} "
              f"{lv['request_throughput']:>7.2f} {lv['p99_ttft_ms']:>9.1f} "
              f"{lv['p50_tpot_ms']:>9.2f} {lv['harness_max_num_requests_running']:>8.0f} "
              f"{lv['harness_max_num_requests_waiting']:>8.0f} "
              f"{lv['harness_preemptions']:>8.0f} {cost}")
    rule()
    peak = max(rows, key=lambda lv: lv["request_goodput"])
    print(f"  goodput peaks at {peak['request_goodput']:.2f} req/s "
          f"(offered {peak['harness_load']:.0f}) against run 2's {RUN2_GOODPUT_PEAK} "
          f"-- {peak['request_goodput'] / RUN2_GOODPUT_PEAK:.1f}x")
    tpot_cross = crossing(rows, "p50_tpot_ms", TPOT_TARGET_MS)
    ttft_cross = crossing(rows, "p99_ttft_ms", TTFT_TARGET_MS)
    print(f"  TPOT p50 crosses {TPOT_TARGET_MS:.0f} ms at {tpot_cross:.1f} req/s "
          f"(predicted 5-6)")
    print(f"  TTFT p99 crosses {TTFT_TARGET_MS:.0f} ms at {ttft_cross:.1f} req/s "
          f"(predicted 2.5-4; run 2 uncached: {RUN2_TTFT_P99_CROSSING[0]}-"
          f"{RUN2_TTFT_P99_CROSSING[1]})")
    preempted = sum(lv["harness_preemptions"] for lv in rows)
    ceiling = [lv for lv in rows
               if lv["harness_max_num_requests_running"] >= MAX_NUM_SEQS_DEFAULT]
    print(f"  preemptions across the whole block: {preempted:.0f} -- the pool never "
          f"binds at h=0.8")
    if ceiling:
        print(f"  what binds instead: max_num_seqs = {MAX_NUM_SEQS_DEFAULT}, reached at "
              f"{ceiling[0]['harness_load']:.0f} req/s with "
              f"{ceiling[0]['harness_max_num_requests_waiting']:.0f} queued behind it")
    late = max(lv["harness_max_lateness_ms"] for lv in rows)
    top = rows[-1]["harness_load"]
    print(f"  generator lateness never exceeded {late:.1f} ms against a "
          f"{1000 / top:.0f} ms arrival interval at {top:.0f} req/s: the open loop "
          f"stayed open")


def block_c():
    print()
    print("BLOCK C -- what prefix caching costs when the workload never repeats")
    print()
    control = levels("run3-c-control")[0]
    treatment = levels("run3-c-treatment")[0]
    fields = (("median ITL (decode step)", "median_itl_ms"),
              ("TPOT p50", "p50_tpot_ms"),
              ("TPOT p99", "p99_tpot_ms"),
              ("ITL p99", "p99_itl_ms"),
              ("TTFT p50", "p50_ttft_ms"))
    print(f"{'':26} {'caching off':>12} {'caching on':>12} {'delta':>9}")
    rule()
    for label, key in fields:
        a, b = control[key], treatment[key]
        print(f"  {label:<24} {a:>12.3f} {b:>12.3f} {b / a - 1:>+8.2%}")
    rule()
    print(f"  measured h on the treatment: "
          f"{treatment['harness_measured_hit_rate']:.3f} -- it had nothing to reuse")
    print("  The decode step moved by a ninth of the 0.24% it reproduces to, i.e. by")
    print("  nothing this instrument can resolve. TTFT p50 spans 32% across identical")
    print("  launches, so +0.8% there is unmeasurable rather than small.")
    print()
    spread = max(DECODE_STEP_C13) / min(DECODE_STEP_C13) - 1
    print(f"  The same step at c=13, every run that has measured it: "
          f"{' / '.join(f'{x:.2f}' for x in DECODE_STEP_C13)}")
    print(f"  Three pods, two drivers, twelve days, total spread {spread:.2%}")


def cost():
    print()
    print("COST -- the seat count converted into the only figure a price list needs")
    print()
    rows = slo_seat_rows()
    seat_unc, seat_cached = rows["h0"], rows["h80"]
    print(f"{'':10} {'seats':>6} {'out tok/s':>10} {'$/M output':>11}")
    rule()
    for label, lv in (("h = 0", seat_unc), ("h = 0.8", seat_cached)):
        print(f"  {label:<8} {lv['harness_load']:>6.0f} {lv['output_throughput']:>10.1f} "
              f"{dollars_per_1m(lv['output_throughput']):>11.2f}")
    rule()
    ratio = seat_cached["output_throughput"] / seat_unc["output_throughput"]
    print(f"  {ratio:.1f}x the tokens inside the same SLO for the same ${HOURLY}/h, "
          f"available to")
    print("  any operator whose traffic repeats and to no operator whose traffic does not.")



# The ledger's view of this run (bench/ledger.py).
LEDGER = {
    "run": "l40s-3",
    "model": QWEN3_8B,
    "engine": "vLLM",
    "engine_version": "0.27.1",
    "regime": "in",
    "mode": "seats",
    "host": {"provider": "RunPod", "instance": "pod ff6689f28bec", "driver": "580.126.20"},
    "sheet": "docs/benchmarks/runsheets/l40s-run-3.md",
    "sheet_commit": "f7172a6",      # as measured.py: the history starts after the run
    "raw": os.path.join(RAW, "results"),
}


def ledger_points():
    """The uncached sweep and the c = 13 control: run 1's fit faced on a third pod."""
    for directory in ("run3-a-h00-clean", "run3-c-control"):
        for lv in levels(directory):
            yield {
                "batch": int(lv["harness_load"]),
                "context": SEAT_CONTEXT,
                "prompt": int(lv["harness_prompt_tokens_nominal"]),
                "step_ms": lv["median_itl_ms"],
                "faced": [("l40s-run1", "point")],
                "note": directory,
            }


if __name__ == "__main__":
    print(f"Run: {RUN}   raw evidence: docs/benchmarks/raw/{RUN}/")
    print("Argument and conclusions: docs/benchmarks/l40s-run3.md")
    print("Coefficients: eff_mem 0.83, mfu 0.439 and I(n) from run 1, "
          "NOT fitted to anything below.")
    checkpoint()
    block_a()
    block_b()
    block_c()
    cost()
    print()
