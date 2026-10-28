"""Run 2 scored against the coefficients it did not produce -- SLO.md section 9.

Sibling of measured.py, which holds run 1 and only run 1. A new run gets a new
module for the same reason a new run gets a new file under docs/benchmarks/: the
run is the unit that a coefficient belongs to, and merging two of them into one
loader is how a fitted number quietly starts scoring itself.

What makes this run different from run 1, and the whole reason it exists:
`eff_mem = 0.83` and `mfu = 0.439` were fitted to run 1's twelve levels. Here they
face 22 levels they never saw, on a different pod under a different driver, at
four chunk sizes, under two attention backends, and with the KV dtype halved.
Every prediction printed below is therefore *unfitted* -- the column run 1's table
could not have.

The argument is docs/benchmarks/l40s-run2.md; this file is the arithmetic behind
it, so a figure quoted there fails here rather than drifting silently.

    python3 bench/measured_run2.py
"""

import os

from dataclasses import replace

from roofline import (
    L40S_RUN1,
    QWEN3_8B,
    tpot_floor,
    ttft_floor,
)

RUN = "l40s-2026-08-23"
LEVELS_TSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "benchmarks", "raw", RUN, "levels.tsv",
)

HOURLY = 0.99               # SLO.md section 7, the same contract figure as run 1
TPOT_TARGET = 0.050         # SLO.md section 2, interactive
OUTPUT_TOKENS = 200

# The same model with a one-byte KV cache. roofline.Model already carries
# kv_dtype_bytes for exactly this, so FP8 needs no new physics -- which is itself
# the block C prediction: halve the bytes, halve the KV term, nothing else moves.
QWEN3_8B_FP8_KV = replace(QWEN3_8B, kv_dtype_bytes=1)

# From the startup logs, which outrank any derivation here (SLO.md section 9).
# raw/l40s-2026-08-23/startup-lines.txt.
LOGGED_KV_TOKENS_BF16 = 169_833
LOGGED_KV_TOKENS_FP8 = 339_666
LOGGED_KV_GIB = 23.34                   # identical under both dtypes
LOGGED_MAX_CONCURRENCY_BF16 = 18.87     # at max_model_len 9 000
LOGGED_MAX_CONCURRENCY_FP8 = 37.74

# The same configuration relaunched, twice more, with nothing changed but a warm
# torch.compile cache. This is the run's measurement of its own reproducibility
# and the reason the checkpoint gate is called a defect in the write-up.
LOGGED_KV_TOKENS_RELAUNCHES = (169_833, 176_994, 169_833)

# Whole-session maxima from the 1 Hz sampler, 4 105 samples.
METRICS_MAX_RUNNING = 106
METRICS_MAX_WAITING = 30
METRICS_MAX_PREEMPTIONS = 2
METRICS_MAX_KV_USAGE = 0.998211

# Block A's operating points are an *output*, not an input: with a finite
# --request-rate and no --max-concurrency the server settles at whatever n makes
# its completion rate match the arrival rate. These are the n the runsheet solved
# for before the run, and comparing a step predicted at them against the measured
# step tests the queueing model and the step model together -- which is why the
# coefficient work lives in blocks B and C.
PREDICTED_N_AT_RATE = {
    0.5: 2.5, 1.0: 5.7, 1.5: 9.7, 2.0: 15.1, 2.5: 22.5, 3.0: 33.5, 4.0: 86.3,
}

# Chunk sizes per configuration tag, so a block B row knows its own knob.
MNBT = {
    "bf16-512": 512, "bf16-512b": 512, "bf16-1024": 1024,
    "bf16-2048": 2048, "bf16-4096": 4096,
    "fp8-2048": 2048, "bf16-ctl": 2048, "bf16-fi": 2048,
}

# Where the median ITL is no longer a decode step, established in the write-up
# section 6: at c=32 a 512- or 1024-token chunk puts prefill into the majority of
# scheduler steps and the median moves into the contaminated mode. Listed rather
# than detected, because the detection rule (TPOT / median ITL <= 1) is a finding
# of this run and not yet something to filter data with.
MEDIAN_ITL_CONTAMINATED = {("bf16-512", 32), ("bf16-1024", 32), ("bf16-512b", 16)}


def _num(s):
    return None if s == "-" else float(s)


def levels():
    """The 22 rows of raw/<run>/levels.tsv, typed."""
    with open(LEVELS_TSV) as f:
        header = None
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            row = dict(zip(header, fields))
            tag = row["tag"]
            config = tag.rsplit("-c", 1)[0] if row["block"] != "A" else "bf16-2048"
            yield {
                "tag": tag,
                "block": row["block"],
                "config": config,
                "mnbt": MNBT.get(config, 2048),
                "fp8": config.startswith("fp8"),
                "input_len": int(row["input_len"]),
                # The mean context a decode step sees over a 200-token generation.
                "context": int(row["input_len"]) + OUTPUT_TOKENS // 2,
                "rate": None if row["rate"] == "inf" else float(row["rate"]),
                "conc": _num(row["max_conc"]),
                "failed": int(row["failed"]),
                "req_thr": float(row["req_thr"]),
                "goodput": _num(row["goodput"]),
                "out_tps": float(row["out_tps"]),
                "total_tps": float(row["total_tps"]),
                "peak_conc": int(row["peak_conc"]),
                "ttft_p50_ms": float(row["ttft_p50"]),
                "ttft_p99_ms": float(row["ttft_p99"]),
                "tpot_p50_ms": float(row["tpot_p50"]),
                "tpot_p99_ms": float(row["tpot_p99"]),
                # median ITL, not TPOT -- see measured.py for why, and section 6
                # of the write-up for where the distinction stops working.
                "decode_step_ms": float(row["itl_p50"]),
                "itl_p99_ms": float(row["itl_p99"]),
            }


def block(name):
    return [lv for lv in levels() if lv["block"] == name]


def model_for(level):
    return QWEN3_8B_FP8_KV if level["fp8"] else QWEN3_8B


def predicted_step_ms(level, batch=None):
    """The unfitted prediction: run 1's eff_mem, this run's geometry."""
    n = batch if batch is not None else level["conc"]
    return tpot_floor(
        model_for(level), L40S_RUN1, n, level["context"]).seconds * 1000


def dollars_per_1m(tokens_per_s, hourly=HOURLY):
    return hourly / (tokens_per_s * 3600) * 1e6 if tokens_per_s else float("inf")


def rule(width: int = 96) -> None:
    print("-" * width)


def checkpoint() -> None:
    print()
    print("CHECKPOINT A' -- the pool, and the gate that was too tight")
    print()
    print(f"  run 1 logged             : {168_985:>12,} tokens")
    print(f"  run 2 logged             : {LOGGED_KV_TOKENS_BF16:>12,} tokens"
          f"   ({(LOGGED_KV_TOKENS_BF16 - 168_985) / 168_985 * 100:+.2f}%)")
    print(f"  the card's gate was      : {'+-500':>12} tokens  (+-0.30%)")
    print()
    lo, hi = min(LOGGED_KV_TOKENS_RELAUNCHES), max(LOGGED_KV_TOKENS_RELAUNCHES)
    print(f"  the identical config, relaunched: {LOGGED_KV_TOKENS_RELAUNCHES}")
    print(f"  run-to-run spread on one pod    : {(hi - lo) / lo * 100:.1f}%"
          f"   -- fourteen times the gate")
    print()
    per_token = LOGGED_KV_GIB * 1024 ** 3 / LOGGED_KV_TOKENS_BF16
    print(f"  bytes per token, from the log   : {per_token:>10,.0f} B")
    print(f"  bytes per token, from config    : {QWEN3_8B.kv_bytes_per_token:>10,.0f} B"
          f"   -- the geometry is identical")
    print()
    for ctx, label in ((4100, "4 100 (block B and C)"), (1600, "1 600 (block A)")):
        print(f"  seats at ctx {label:<24}: "
              f"{LOGGED_KV_TOKENS_BF16 / ctx:>6.1f}")
    print(f"  ...and the sampler's maximum num_requests_running: "
          f"{METRICS_MAX_RUNNING}")
    print(f"  ...with {METRICS_MAX_WAITING} waiting and "
          f"{METRICS_MAX_PREEMPTIONS} preemptions, kv usage "
          f"{METRICS_MAX_KV_USAGE:.3f}")
    print()
    print(f"  FP8 pool logged          : {LOGGED_KV_TOKENS_FP8:>12,} tokens"
          f"   ({LOGGED_KV_TOKENS_FP8 / LOGGED_KV_TOKENS_BF16:.3f}x)")
    print(f"  seats at 4 100 ctx, FP8  : {LOGGED_KV_TOKENS_FP8 / 4100:>12.1f}")


def unfitted_decode_table() -> None:
    print()
    print("DECODE STEP -- run 1's eff_mem 0.83 against levels it never saw")
    print("No column here is fitted. That is the whole point of the run.")
    print()
    print(f"{'level':>16} {'mnbt':>6} {'n':>5} {'predicted':>10} {'measured':>9} "
          f"{'err':>8}  note")
    rule()
    for lv in sorted(block("B") + block("C"), key=lambda l: (l["conc"], l["tag"])):
        pred = predicted_step_ms(lv)
        got = lv["decode_step_ms"]
        flag = ("  median ITL is not a decode step here"
                if (lv["config"], int(lv["conc"])) in MEDIAN_ITL_CONTAMINATED else "")
        print(f"{lv['tag']:>16} {lv['mnbt']:>6} {int(lv['conc']):>5} "
              f"{pred:>10.2f} {got:>9.2f} {(pred - got) / got * 100:>+7.1f}%{flag}")
    rule()

    clean = [lv for lv in block("B") + block("C")
             if int(lv["conc"]) == 13 and not lv["fp8"]]
    mean = sum(lv["decode_step_ms"] for lv in clean) / len(clean)
    pred = tpot_floor(QWEN3_8B, L40S_RUN1, 13, 4100).seconds * 1000
    steps = [lv["decode_step_ms"] for lv in clean]
    print(f"  {len(clean)} BF16 launches at c=13 -- four chunk sizes, two attention")
    print(f"  backends, three separate starts of the same configuration:")
    print(f"    spread {max(steps) - min(steps):.2f} ms "
          f"({(max(steps) - min(steps)) / min(steps) * 100:.1f}%); "
          f"mean {mean:.4f} ms against {pred:.4f} predicted "
          f"({(pred - mean) / mean * 100:+.2f}%)")


def block_a_table() -> None:
    print()
    print("BLOCK A -- the open loop: throughput climbs, goodput collapses")
    print()
    print(f"{'rate':>5} {'req/s':>6} {'goodput':>8} {'TTFTp99':>8} {'TPOTp99':>8} "
          f"{'medITL':>7} {'pred':>7} {'err':>7} {'$/1M out':>9} {'$/1M good':>10}")
    rule()
    for lv in block("A"):
        if lv["tag"] == "A-warm":
            continue
        n = PREDICTED_N_AT_RATE[lv["rate"]]
        pred = predicted_step_ms(lv, batch=n)
        got = lv["decode_step_ms"]
        good_tps = lv["goodput"] * OUTPUT_TOKENS
        print(f"{lv['rate']:>5} {lv['req_thr']:>6} {lv['goodput']:>8} "
              f"{lv['ttft_p99_ms']:>8.1f} {lv['tpot_p99_ms']:>8.2f} "
              f"{got:>7.2f} {pred:>7.2f} {(pred - got) / got * 100:>+6.1f}% "
              f"{dollars_per_1m(lv['out_tps']):>9.3f} "
              f"{dollars_per_1m(good_tps):>10.3f}")
    rule()
    peak = max(block("A")[1:], key=lambda lv: lv["goodput"])
    top = block("A")[-1]
    print(f"  goodput peaks at {peak['goodput']} req/s (offered {peak['rate']}); "
          f"at {top['rate']} it is {top['goodput']}")
    print(f"  over that stretch output throughput goes "
          f"{peak['out_tps']:.0f} -> {top['out_tps']:.0f} tok/s "
          f"({top['out_tps'] / peak['out_tps']:.2f}x) while goodput falls "
          f"{peak['goodput'] / top['goodput']:.0f}x")
    print(f"  and the cost of a token a customer can use goes "
          f"${dollars_per_1m(peak['goodput'] * OUTPUT_TOKENS):.3f} -> "
          f"${dollars_per_1m(top['goodput'] * OUTPUT_TOKENS):.2f} per 1M")


def block_b_table() -> None:
    print()
    print("BLOCK B -- what max_num_batched_tokens moves, and what it does not")
    print("The worst-step bound is decode step + chunk compute at mfu 0.439.")
    print()
    print(f"{'mnbt':>6} {'chunks':>7} {'bound':>8} {'ITL p99':>8} {'err':>7}   "
          f"{'TTFT>=':>7} {'TTFT p50':>9} {'slack':>7}   {'TPOT p99':>9} {'medITL':>7}")
    rule()
    step13 = tpot_floor(QWEN3_8B, L40S_RUN1, 13, 4100).seconds * 1000
    at13 = {lv["mnbt"]: lv for lv in block("B")
            if int(lv["conc"]) == 13}
    for mnbt in (512, 1024, 2048, 4096):
        lv = at13[mnbt]
        # the prefill tokens that share the step with 13 decode tokens, and the
        # number of steps a 4 000-token prompt therefore needs
        chunk_tokens = min(mnbt - 13, 4000)
        chunks = -(-4000 // chunk_tokens)
        chunk_ms = ttft_floor(QWEN3_8B, L40S_RUN1, chunk_tokens).seconds * 1000
        bound = step13 + chunk_ms
        ttft_bound = bound * chunks
        print(f"{mnbt:>6} {chunks:>7} {bound:>8.1f} {lv['itl_p99_ms']:>8.2f} "
              f"{(bound - lv['itl_p99_ms']) / lv['itl_p99_ms'] * 100:>+6.1f}%   "
              f"{ttft_bound:>7.0f} {lv['ttft_p50_ms']:>9.1f} "
              f"{(lv['ttft_p50_ms'] - ttft_bound) / ttft_bound * 100:>+6.0f}%   "
              f"{lv['tpot_p99_ms']:>9.2f} {lv['decode_step_ms']:>7.2f}")
    rule()
    print("  The bound over-counts the tail by 5-10% across an eightfold range of")
    print("  chunk size -- the runsheet expected ~20% -- and it bounds TTFT tightly")
    print("  only at 512. Above that TTFT is the wait behind OTHER requests'")
    print("  prefills, which the bound omits, so no monotone TTFT prediction in")
    print("  mnbt could have been right.")
    print("  (The runsheet's own 4 096 row charged 4 083 prefill tokens to a 4 000-")
    print("   token prompt and so published 390.9 ms, +7.1%, where 383.6 is correct.)")
    print()
    lo = at13[512]
    hi = next(lv for lv in block("B") if lv["tag"] == "bf16-512b-c16")
    crossing = 13 + (16 - 13) * (50.0 - lo["tpot_p99_ms"]) / \
        (hi["tpot_p99_ms"] - lo["tpot_p99_ms"])
    print(f"  TPOT p99 at mnbt 512: {lo['tpot_p99_ms']:.2f} ms at c=13, "
          f"{hi['tpot_p99_ms']:.2f} ms at c=16  ->  50 ms at c={crossing:.1f}")
    print(f"  Run 1 put the seat count at 12 with mnbt 2048. The knob buys "
          f"{crossing - 12:.0f} seats.")
    c32 = {lv["mnbt"]: lv for lv in block("B") if int(lv["conc"]) == 32}
    print(f"  It costs TTFT p50 at c=32: {c32[2048]['ttft_p50_ms']:.0f} ms at 2048 "
          f"-> {c32[512]['ttft_p50_ms']:.0f} ms at 512 "
          f"({c32[512]['ttft_p50_ms'] / c32[2048]['ttft_p50_ms']:.1f}x)")


def block_c_table() -> None:
    print()
    print("BLOCK C -- FP8 KV: the pool doubles, and the step is faster besides")
    print()
    bf16 = {int(lv["conc"]): lv for lv in block("B")
            if lv["config"] == "bf16-2048"}
    fp8 = {int(lv["conc"]): lv for lv in block("C") if lv["fp8"]}
    print(f"{'n':>4} {'BF16 step':>10} {'FP8 pred':>9} {'FP8 meas':>9} {'err':>7} "
          f"{'FP8 vs BF16':>12}")
    rule()
    for n in sorted(fp8):
        lv = fp8[n]
        pred = predicted_step_ms(lv)
        got = lv["decode_step_ms"]
        ref = bf16.get(n)
        vs = f"{got / ref['decode_step_ms'] - 1:+.1%}" if ref else "-"
        base = f"{ref['decode_step_ms']:.2f}" if ref else "-"
        print(f"{n:>4} {base:>10} "
              f"{pred:>9.2f} {got:>9.2f} {(pred - got) / got * 100:>+6.1f}% {vs:>12}")
    rule()
    claim = fp8[26]["decode_step_ms"]
    base = bf16[13]["decode_step_ms"]
    print(f"  The claim: FP8 at n=26 costs what BF16 costs at n=13.")
    print(f"    {claim:.2f} ms against {base:.2f} ms  ->  {claim / base - 1:+.1%}")
    print()
    ctl = {lv["tag"]: lv for lv in block("C") if not lv["fp8"]}
    print("  The confounder: FP8 forced FLASHINFER, so the kernel changed too.")
    print(f"    BF16 + FLASH_ATTN : {bf16[13]['decode_step_ms']:.2f} and "
          f"{ctl['bf16-ctl-c13']['decode_step_ms']:.2f} ms")
    print(f"    BF16 + FLASHINFER : {ctl['bf16-fi-c13']['decode_step_ms']:.2f} ms"
          f"   -- the kernel is worth "
          f"{ctl['bf16-fi-c13']['decode_step_ms'] / bf16[13]['decode_step_ms'] - 1:+.1%}")
    print(f"    FP8  + FLASHINFER : {fp8[13]['decode_step_ms']:.2f} ms"
          f"   -- the cache is worth "
          f"{fp8[13]['decode_step_ms'] / bf16[13]['decode_step_ms'] - 1:+.1%}")
    print()
    print(f"  Throughput at c=32: {bf16[32]['out_tps']:.2f} -> "
          f"{fp8[32]['out_tps']:.2f} out tok/s "
          f"({fp8[32]['out_tps'] / bf16[32]['out_tps'] - 1:+.1%}), "
          f"${dollars_per_1m(bf16[32]['out_tps']):.3f} -> "
          f"${dollars_per_1m(fp8[32]['out_tps']):.3f} per 1M")



# The ledger's view of this run (bench/ledger.py).
LEDGER = {
    "run": "l40s-2",
    "model": QWEN3_8B,
    "engine": "vLLM",
    "engine_version": "0.27.1",
    "regime": "in",
    "mode": "seats",
    "host": {"provider": "RunPod, North American DC", "instance": "pod 70700baa4cfa",
             "driver": "550.127.05 / CUDA 13.0"},
    "sheet": "docs/benchmarks/runsheets/l40s-run-2.md",
    "sheet_commit": "f7172a6",      # as measured.py: the history starts after the run
    "raw": LEVELS_TSV,
}


def ledger_points():
    """The six BF16 launches at c = 13 whose median ITL is a decode step: run 1's fit, faced."""
    for lv in sorted(block("B") + block("C"), key=lambda l: l["tag"]):
        if int(lv["conc"]) == 13 and not lv["fp8"]:
            yield {
                "batch": 13,
                "context": lv["context"],
                "prompt": lv["input_len"],
                "step_ms": lv["decode_step_ms"],
                "faced": [("l40s-run1", "point")],
                "note": lv["tag"],
            }


if __name__ == "__main__":
    print(f"Run: {RUN}   raw evidence: docs/benchmarks/raw/{RUN}/")
    print("Argument and conclusions: docs/benchmarks/l40s-run2.md")
    print("Coefficients: eff_mem 0.83 and mfu 0.439, fitted to run 1, "
          "NOT to anything below.")
    checkpoint()
    unfitted_decode_table()
    block_a_table()
    block_b_table()
    block_c_table()
    print()
