"""MI300X run 1 scored -- a sibling of measured.py, one module per run.

Reads the sweep's per-repeat JSON and the engine's 10-second log lines in
sweep.log, the only gauges the sweep kept. The predictions are the runsheet's;
the argument is docs/benchmarks/mi300x-run1.md; bench/tests/test_roofline.py
holds MI300X_RUN1 to these files.

    python3 bench/measured_mi300x_run1.py
"""

import glob
import json
import math
import os
import re
import statistics
from dataclasses import replace

import measured
from roofline import (
    MI300X,
    MI300X_RUN1,
    QWEN3_8B,
    kv_cache_tokens,
    token_budget_ceiling,
    tpot_with_interference,
    ttft_floor,
    tpot_floor,
)

RUN = "mi300x-2026-09-27"
RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "benchmarks", "raw", RUN, "run1",
)
RESULTS = os.path.join(RAW, "results", "mi300x-run-1")

GMU = 0.90
HOURLY = 1.99               # the console, 2026-09-27, on-demand
TPOT_TARGET_MS = 50.0       # SLO.md section 2, interactive
OUTPUT_TOKENS = 200
MAX_MODEL_LEN = 9000
BLOCK_SIZE = 16
GIB = 2 ** 30
CARD_BYTES = 205_822_885_888    # rocm-smi, the VF: 192 GiB less 320 MiB

# From the startup log of each launch, which outranks any derivation here.
# `GPU KV cache size` is max_concurrency x max_model_len, not blocks x 16.
LOGGED_KV_TOKENS = (1_123_065, 1_138_444)       # launch 1, launch 2 (after --resume)
LOGGED_MAX_CONCURRENCY = (124.79, 126.49)       # at max_model_len 9 000
LOGGED_WEIGHTS_GIB = 15.27
LOGGED_CONSUMED_GIB = (17.03, 15.85)            # weights + non-torch, per launch
LOGGED_ACTIVATION_GIB = (1.12, 0.19)            # peak activation, per launch
LOGGED_CUDAGRAPH_GIB = 4.12
LOGGED_SHARE_GIB = 172.52                       # 0.90 of what the engine saw

# Set in the serve command; the build's default budget here is 8 192 (report §1).
MAX_NUM_BATCHED_TOKENS = 2048
MAX_NUM_SEQS = 256

# Rows whose median ITL is still a decode step: at c064 and above every step
# carries a prefill chunk (docs/benchmarks/mi300x-run1.md section 3).
DECODE_ROWS = ("c001", "c008", "c032", "c001-in2000", "c001-in8000")
DECODE_STEP_THROUGH = 32

# L40S run 1 at the same geometry, docs/benchmarks/l40s-baseline.md section 7.
L40S_COST = {"c8": 1.427, "c13_p50": 1.126, "plateau": 0.873}

GAUGE = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: "
    r"([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+)%")


def rows():
    """Every row of the sweep: its repeats, and the means a prediction needs."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RESULTS, "SERVE--*"))):
        runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "run=*.json")))]
        first = runs[0]
        mean = lambda k: statistics.mean(r[k] for r in runs)   # noqa: E731
        input_len = first["total_input_tokens"] // first["num_prompts"]
        out[first["_benchmark_name"]] = {
            "c": first["max_concurrency"],
            "repeats": len(runs),
            "prompts": first["num_prompts"],
            "input_len": input_len,
            # input + half the output: the mean context a decode step reads
            "context": input_len + OUTPUT_TOKENS // 2,
            "itl_ms": mean("median_itl_ms"),
            "itl_each": [r["median_itl_ms"] for r in runs],
            "tpot_p50_ms": mean("median_tpot_ms"),
            "tpot_p99_ms": mean("p99_tpot_ms"),
            "ttft_p50_ms": mean("median_ttft_ms"),
            "ttft_p99_ms": mean("p99_ttft_ms"),
            "output_tps": mean("output_throughput"),
            "output_each": [r["output_throughput"] for r in runs],
            "failed": sum(r["failed"] for r in runs),
        }
    return out


def gauges():
    """The engine's 10-second log lines, grouped by the row that was running.

    Running, Waiting and KV usage are snapshots; the throughputs are interval averages.
    """
    out, current = {}, None
    with open(os.path.join(RAW, "sweep.log"), errors="replace") as f:
        for line in f:
            if "Namespace(subparser=" in line and "bench_type='serve'" in line:
                c = int(re.search(r"max_concurrency=(\d+)", line).group(1))
                n = re.search(r"random_input_len=(\d+)", line).group(1)
                current = f"c{c:03d}" + ("" if n == "4000" else f"-in{n}")
                out.setdefault(current, [])
            m = GAUGE.search(line)
            if m and current:
                prompt, gen, running, waiting, kv = map(float, m.groups())
                out[current].append({"prompt_tps": prompt, "gen_tps": gen,
                                     "running": running, "waiting": waiting,
                                     "kv_pct": kv})
    return out


def preemptions():
    """num_preemptions_total after every repeat, from the after-bench hook."""
    with open(os.path.join(RAW, "metrics-after.log")) as f:
        return [float(line.split()[-1]) for line in f
                if line.startswith("vllm:num_preemptions_total")]


def saturated_running(gauge):
    """Running counts on the lines where the engine was full, c128 and up."""
    return [x["running"] for name in ("c128", "c192", "c224", "c240", "c256", "c288")
            for x in gauge[name] if x["running"] >= 90]


def logged_blocks(tokens):
    """The KV blocks behind a `GPU KV cache size` line, which is
    num_blocks / cdiv(max_model_len, block_size) x max_model_len."""
    per_request = math.ceil(MAX_MODEL_LEN / BLOCK_SIZE)
    return round(tokens * per_request / MAX_MODEL_LEN)


def at_peak_ms(fn, *args):
    """A floor multiplied back out of its coefficient: the time at 100 % of peak."""
    return fn(QWEN3_8B, MI300X, *args).seconds * 1000


def implied_eff_mem(row):
    return at_peak_ms(tpot_floor, row["c"], row["context"]) * MI300X.achieved_bandwidth \
        / row["itl_ms"]


def implied_mfu(row):
    return at_peak_ms(ttft_floor, row["input_len"]) * MI300X.mfu / row["ttft_p50_ms"]


def fitted_eff_mem(data):
    """What MI300X_RUN1 carries: the mean over the decode rows."""
    return statistics.mean(implied_eff_mem(data[n]) for n in DECODE_ROWS)


def model_crossing(data, target_ms=TPOT_TARGET_MS):
    """Where tpot_with_interference crosses the target, ITL linear from c008 to c032."""
    alone = data["c001"]["ttft_p50_ms"]
    lo, hi = data["c008"], data["c032"]
    for c in range(lo["c"], hi["c"] + 1):
        itl = lo["itl_ms"] + (c - lo["c"]) * (hi["itl_ms"] - lo["itl_ms"]) / (hi["c"] - lo["c"])
        if tpot_with_interference(itl, c, alone, OUTPUT_TOKENS) > target_ms:
            return c
    return None


def dollars_per_1m(tokens_per_second):
    return HOURLY / (tokens_per_second * 3600) * 1e6


def interpolate(points, target, key="tpot_p99_ms", load="c", value="output_tps"):
    """(load, value) where `key` crosses target, linear between two measured rows."""
    for lo, hi in zip(points, points[1:]):
        if lo[key] <= target < hi[key]:
            t = (target - lo[key]) / (hi[key] - lo[key])
            return lo[load] + t * (hi[load] - lo[load]), lo[value] + t * (hi[value] - lo[value])
    return None


def p99_crossings(data):
    """TPOT p99 = 50 ms on both cards, by linear interpolation: derived, inside the row gap."""
    mi = interpolate([data[n] for n in ("c001", "c008", "c032")], TPOT_TARGET_MS)
    l40s_rows = sorted((lv for lv in measured.levels() if lv["input_len"] == 4000),
                       key=lambda lv: lv["asked"])
    l40s = interpolate(l40s_rows, TPOT_TARGET_MS, load="asked")
    return {"mi300x": (mi[0], dollars_per_1m(mi[1])),
            "l40s": (l40s[0], measured_rate(l40s[1]))}


def measured_rate(tokens_per_second, hourly=0.99):
    return hourly / (tokens_per_second * 3600) * 1e6


def seats_from_blocks(context_len, tokens=LOGGED_KV_TOKENS[0]):
    """Sequences the logged pool holds, counted in whole blocks as vLLM does."""
    return logged_blocks(tokens) // math.ceil(context_len / BLOCK_SIZE)


def busy_time_share(data, gauge, name="c032"):
    """Prefill (at the c = 1 rate) plus decode time per wall second: derived, not measured."""
    g = gauge[name]
    busy = [x for x in g if x["running"] >= 0.9 * max(y["running"] for y in g)]
    prompt = statistics.mean(x["prompt_tps"] for x in busy)
    gen = statistics.mean(x["gen_tps"] for x in busy)
    solo = 4000 / (data["c001"]["ttft_p50_ms"] / 1000)
    prefill = prompt / solo
    decode = gen / data[name]["c"] * data[name]["itl_ms"] / 1000
    return prefill, decode


def rule(width=96):
    print("-" * width)


def checkpoint_a():
    print()
    print("CHECKPOINT A -- the KV pool against the arithmetic")
    print()
    per_token = QWEN3_8B.kv_bytes_per_token
    # The prediction the runsheet faced was made at 192e9; roofline.py now says 192 GiB.
    derived = kv_cache_tokens(QWEN3_8B, replace(MI300X, memory_bytes=192e9), GMU)
    corrected = kv_cache_tokens(QWEN3_8B, MI300X, GMU)
    logged = LOGGED_KV_TOKENS[0]
    pool = [logged_blocks(t) * BLOCK_SIZE for t in LOGGED_KV_TOKENS]
    print(f"  derived, memory_bytes = 192e9          : {derived:>12,.0f}   "
          f"{(logged - derived) / derived:+.1%} from it")
    print(f"  derived x 0.93 (the L40S's shortfall)  : {derived * 0.93:>12,.0f}   "
          f"{(logged - derived * 0.93) / (derived * 0.93):+.1%} from it")
    print(f"  logged, launch 1                       : {logged:>12,}   "
          f"= {logged_blocks(logged):,} blocks = {pool[0]:,} tokens")
    print(f"  logged, launch 2                       : {LOGGED_KV_TOKENS[1]:>12,}   "
          f"= {logged_blocks(LOGGED_KV_TOKENS[1]):,} blocks, "
          f"{LOGGED_KV_TOKENS[1] / logged - 1:+.1%}")
    print(f"  derived at 192 GiB, roofline.py now    : {corrected:>12,.0f}   "
          f"launch 1 {(corrected - logged) / corrected:.1%} below it, "
          f"launch 2 {(corrected - LOGGED_KV_TOKENS[1]) / corrected:.2%}")
    print()
    print("  Launch 1, term by term, from the card rocm-smi reports:")
    true_card = (CARD_BYTES * GMU - QWEN3_8B.weights_bytes) / per_token
    non_torch = (LOGGED_CONSUMED_GIB[0] - LOGGED_WEIGHTS_GIB) * GIB / per_token
    activation = LOGGED_ACTIVATION_GIB[0] * GIB / per_token
    accounted = true_card - non_torch - activation
    print(f"    the card, {CARD_BYTES / 1e9:.2f}e9 B, not 192e9          : "
          f"{true_card - derived:>+10,.0f} tokens")
    print(f"    non-torch beyond the weights, "
          f"{LOGGED_CONSUMED_GIB[0] - LOGGED_WEIGHTS_GIB:.2f} GiB  : {-non_torch:>+10,.0f}")
    print(f"    peak activation, {LOGGED_ACTIVATION_GIB[0]} GiB               : {-activation:>+10,.0f}")
    print(f"    accounted                              : {accounted:>10,.0f}   "
          f"against {pool[0]:,} in blocks")
    print(f"  launch 2 paid {LOGGED_CONSUMED_GIB[1] - LOGGED_WEIGHTS_GIB:.2f} GiB non-torch "
          f"and {LOGGED_ACTIVATION_GIB[1]} GiB activation: the terms move between launches")
    in_use = LOGGED_CONSUMED_GIB[0] + LOGGED_ACTIVATION_GIB[0] + LOGGED_CUDAGRAPH_GIB \
        + pool[0] * per_token / GIB
    print(f"  CUDA graphs, {LOGGED_CUDAGRAPH_GIB} GiB, are captured after the pool and not taken from it:")
    print(f"    {in_use:.2f} GiB in use against a {LOGGED_SHARE_GIB} GiB share, "
          f"{in_use / (CARD_BYTES / GIB):.1%} of the card")
    print(f"  seats in whole blocks: {seats_from_blocks(4000)} at 4 000 tokens, "
          f"{seats_from_blocks(4200)} at 4 200 -- derived from the logged pool, not observed")


def decode_table(data):
    print()
    print("DECODE STEP -- the median ITL against the floor at eff_mem 0.70")
    print("Only the rows where the median step is still a decode step.")
    print()
    print(f"{'row':>12} {'ctx':>5} {'floor@0.70':>11} {'@0.46*':>8} {'median ITL':>11} "
          f"{'implied eff_mem':>16}")
    rule()
    for name in DECODE_ROWS:
        r = data[name]
        floor = tpot_floor(QWEN3_8B, MI300X, r["c"], r["context"]).seconds * 1000
        fit = tpot_floor(QWEN3_8B, MI300X_RUN1, r["c"], r["context"]).seconds * 1000
        print(f"{name:>12} {r['context']:>5} {floor:>9.2f}ms {fit:>6.2f}ms {r['itl_ms']:>9.2f}ms "
              f"{implied_eff_mem(r):>16.3f}")
    rule()
    values = [implied_eff_mem(data[n]) for n in DECODE_ROWS]
    print(f"  mean {fitted_eff_mem(data):.3f} over five rows at three batch sizes (1, 8, 32), "
          f"range {min(values):.3f}-{max(values):.3f}")
    print(f"  * MI300X_RUN1.achieved_bandwidth = {MI300X_RUN1.achieved_bandwidth}, "
          "fitted to these rows -- a hypothesis for run 2")
    small, large = data["c001-in2000"], data["c001-in8000"]
    extra_kv = (large["context"] - small["context"]) * QWEN3_8B.kv_bytes_per_token
    rate = extra_kv / ((large["itl_ms"] - small["itl_ms"]) / 1000)
    print(f"  batch 1, 2 100 -> 8 100 context: +{extra_kv / 1e9:.3f} GB of KV for "
          f"+{large['itl_ms'] - small['itl_ms']:.2f} ms, {rate / 1e12:.1f} TB/s "
          f"({rate / MI300X.peak_bandwidth:.2f} of peak)")
    print("  so most of the batch-1 step does not scale with KV: the weights read at")
    print("  under half of peak, or a per-step cost this run cannot separate from it.")


def prefill_table(data):
    print()
    print("PREFILL -- median TTFT at c = 1 against table 10's floor at mfu 0.45")
    print()
    print(f"{'row':>12} {'prompt':>7} {'steps':>6} {'launch':>7} {'floor@0.45':>11} "
          f"{'TTFT p50':>10} {'x prev':>7} {'implied mfu':>12}")
    rule()
    prev = None
    for name, launch in (("c001-in2000", 2), ("c001", 1), ("c001-in8000", 2)):
        r = data[name]
        floor = ttft_floor(QWEN3_8B, MI300X, r["input_len"]).seconds * 1000
        steps = math.ceil(r["input_len"] / MAX_NUM_BATCHED_TOKENS)
        step = f"{r['ttft_p50_ms'] / prev:.2f}x" if prev else "--"
        print(f"{name:>12} {r['input_len']:>7} {steps:>6} {launch:>7} {floor:>9.1f}ms "
              f"{r['ttft_p50_ms']:>8.1f}ms {step:>7} {implied_mfu(r):>12.3f}")
        prev = r["ttft_p50_ms"]
    rule()
    print(f"  MI300X_RUN1.mfu = {MI300X_RUN1.mfu} -- the 4 000-token point, the one run 1 used")
    print("  Predicted 2.00x per doubling. Attention FLOPs, which the floor leaves out,")
    d = QWEN3_8B.num_layers * 4096        # layers x (query heads x head_dim)
    linear = lambda n: 2 * QWEN3_8B.params_non_embedding * n     # noqa: E731
    attn = lambda n: 2 * n * n * d                                # noqa: E731
    total = lambda n: linear(n) + attn(n)                         # noqa: E731
    print(f"  would make it {total(4000) / total(2000):.2f}x and "
          f"{total(8000) / total(4000):.2f}x. The rest is unresolved: the prompts are")
    print("  1, 2 and 4 engine steps, and two of the three points ran on the second launch.")


def the_curve(data, gauge):
    print()
    print("WHERE THE CURVE ENDS -- the engine's gauges (max over 10-second snapshots),")
    print("prompt tok/s averaged over the saturated lines, the rest means over repeats")
    print()
    print(f"{'row':>6} {'running':>8} {'waiting':>8} {'KV %':>6} {'prompt tok/s':>13} "
          f"{'median ITL':>11} {'TPOT p50':>9} {'TPOT p99':>9} {'out tok/s':>10}")
    rule()
    for name, r in data.items():
        if "-in" in name:
            continue
        g = gauge[name]
        busy = [x for x in g if x["running"] >= 0.9 * max(y["running"] for y in g)]
        print(f"{name:>6} {max(x['running'] for x in g):>8.0f} "
              f"{max(x['waiting'] for x in g):>8.0f} {max(x['kv_pct'] for x in g):>6.1f} "
              f"{statistics.mean(x['prompt_tps'] for x in busy):>13.0f} "
              f"{r['itl_ms']:>9.2f}ms {r['tpot_p50_ms']:>7.2f}ms {r['tpot_p99_ms']:>7.2f}ms "
              f"{r['output_tps']:>10.1f}")
    rule()
    p = preemptions()
    saturated = saturated_running(gauge)
    print(f"  preemptions: {int(max(p))} across all {len(p)} repeats; "
          f"max_num_seqs {MAX_NUM_SEQS} never reached")
    print(f"  running, c128 and up, saturated lines: {min(saturated):.0f}-{max(saturated):.0f}, "
          f"median {statistics.median(saturated):.0f}")
    n_star = token_budget_ceiling(MAX_NUM_BATCHED_TOKENS, 4000, OUTPUT_TOKENS)
    print(f"  the token budget sets that count: {MAX_NUM_BATCHED_TOKENS} x {OUTPUT_TOKENS} / "
          f"(4 000 + {OUTPUT_TOKENS}) = {n_star:.1f} decoding, plus one or two mid-prefill")
    print(f"  throughput had already flattened by c032: {data['c032']['output_tps']:.0f} "
          f"tok/s at 32 running, {data['c288']['output_tps']:.0f} at ~100 -- prefill")
    print("  compute, of which every row spends 20 tokens per output token.")
    prefill, decode = busy_time_share(data, gauge)
    print(f"  c032, derived: prefill at the solo rate {prefill:.2f} of wall time, decode "
          f"{decode:.2f}, together {prefill + decode:.2f}")


def interference(data):
    print()
    print("INTERFERENCE -- TPOT p50 over the median ITL, and a model without a budget")
    print()
    alone = data["c001"]["ttft_p50_ms"]
    for name in ("c001", "c008", "c032"):
        r = data[name]
        model = tpot_with_interference(r["itl_ms"], r["c"], alone, OUTPUT_TOKENS)
        print(f"  {name}  ratio {r['tpot_p50_ms'] / r['itl_ms']:.2f}   "
              f"TPOT p50 {r['tpot_p50_ms']:6.2f} ms   ITL + (c-1) x TTFT_1 / 200 = {model:6.2f} ms")
    print(f"  the model crosses {TPOT_TARGET_MS:.0f} ms at c = {model_crossing(data)} "
          "(TPOT p50); derived, not measured")
    print("  The L40S reached 1.91 at c=32 and 2.06 at c=45 (run 1).")


def cost(data):
    print()
    print(f"COST -- measured output throughput at ${HOURLY}/h")
    print()
    print(f"{'row':>6} {'out tok/s':>10} {'$/1M out':>9} {'TPOT p50':>9} {'TPOT p99':>9}")
    rule(50)
    for name in ("c001", "c008", "c032", "c064", "c128", "c288"):
        r = data[name]
        print(f"{name:>6} {r['output_tps']:>10.1f} {dollars_per_1m(r['output_tps']):>9.3f} "
              f"{r['tpot_p50_ms']:>9.2f} {r['tpot_p99_ms']:>9.2f}")
    rule(50)
    c8, top = data["c008"], data["c288"]
    print(f"  c008 by median repeat: {dollars_per_1m(statistics.median(c8['output_each'])):.3f} "
          "(run 0 was slower than runs 1-2)")
    print(f"  at c = 8 on both cards: {dollars_per_1m(c8['output_tps']):.3f} against the L40S's "
          f"{L40S_COST['c8']}")
    print(f"  plateau: {dollars_per_1m(top['output_tps']):.3f} against {L40S_COST['plateau']}, "
          f"{dollars_per_1m(top['output_tps']) / L40S_COST['plateau'] - 1:+.0%}")
    x = p99_crossings(data)
    print(f"  TPOT p99 = {TPOT_TARGET_MS:.0f} ms, interpolated: MI300X c = {x['mi300x'][0]:.1f} at "
          f"{x['mi300x'][1]:.3f}, L40S c = {x['l40s'][0]:.1f} at {x['l40s'][1]:.3f}, "
          f"{x['mi300x'][1] / x['l40s'][1] - 1:+.0%}")


def every_repeat(data):
    print()
    print("EVERY REPEAT -- the complete record; summary.csv folded 24 of these 28")
    print()
    for name, r in data.items():
        print(f"  {name:>12}  x{r['repeats']}  median ITL "
              + " / ".join(f"{x:.2f}" for x in r["itl_each"])
              + f"  failed {r['failed']}")


if __name__ == "__main__":
    print(f"Run: {RUN}   raw evidence: docs/benchmarks/raw/{RUN}/")
    print("Argument and conclusions: docs/benchmarks/mi300x-run1.md")
    print("Predictions: tables 9 and 10 at eff_mem 0.70, mfu 0.45, not fitted to anything below.")
    data, gauge = rows(), gauges()
    checkpoint_a()
    decode_table(data)
    prefill_table(data)
    the_curve(data, gauge)
    interference(data)
    cost(data)
    every_repeat(data)
    print()
