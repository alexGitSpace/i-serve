"""MI300X run 2 scored -- a sibling of measured_mi300x_run1.py, one module per run.

Five serve rows on one card, each launched once and kept up across eight
benchmark rows. Reads the per-repeat JSON, the startup log of each launch and
the engine's 10-second lines in sweep.log. The predictions are table 12 of
bench/predictions.py at MI300X_RUN1; the argument is
docs/benchmarks/mi300x-run2.md.

    python3 bench/measured_mi300x_run2.py
"""

import ast
import glob
import json
import math
import os
import re
import statistics

import measured_mi300x_run1 as run1
from roofline import (
    MI300X,
    MI300X_RUN1,
    QWEN3_8B,
    token_budget_ceiling,
    tpot_floor,
    tpot_with_interference,
    ttft_floor,
)

RUN = "mi300x-2026-09-27"
RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "benchmarks", "raw", RUN, "run2",
)
RESULTS = os.path.join(RAW, "results", "mi300x-run-2")

HOURLY = 1.99               # the console, 2026-09-27, on-demand
TPOT_TARGET_MS = 50.0       # SLO.md section 2, interactive
PROMPT_TOKENS = 4000
OUTPUT_TOKENS = 200
SATURATED = "c256"
with open(os.path.join(RAW, "mi300x-run-2-serve.json")) as f:
    SERVE = json.load(f)

# File order, which is the order the sweep launched them in.
SERVE_ROWS = ("b2048", "b8192", "triton", "aiter-fa", "b4096")
# ROCM_AITER_FA misreads a mixed batch on the V2 model runner; only its
# single-request rows are measurements (runsheet mi300x-run-2, *The five serve rows*).
SINGLE_REQUEST_ONLY = {"aiter-fa"}
AT_CONCURRENCY = tuple(s for s in SERVE_ROWS if s not in SINGLE_REQUEST_ONLY)
LENGTH_ROWS = ("c001-in2000", "c001", "c001-in8000")
LOADED_ROWS = ("c008", "c016", "c024", "c032")
# The median ITL is still a decode step through c032 on every row read at concurrency.
DECODE_ROWS = ("c001-in2000", "c001", "c001-in8000") + LOADED_ROWS

GIB = 2 ** 30
RUN1_TTFT_4000_MS = 256.9   # run 1's median at c001, the figure the sheet's lever read uses

ARGS = re.compile(r"non-default args: (\{.*\})")
LOGGED = {
    "kv_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "graph_gib": re.compile(r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB"),
    "activation_gib": re.compile(r"([\d.]+) GiB for peak activation"),
    "consumed_gib": re.compile(r"Actual usage is ([\d.]+) GiB for consumed memory"),
    "build_s": re.compile(r"finish build \S+, cost ([\d.]+)s"),
    "backend": re.compile(r"Using (\w+) backend \(selected via|Overriding with (\w+) out of"),
}
STAMP = re.compile(r"\d\d-\d\d (\d\d):(\d\d):(\d\d)")


def serve_name(args):
    """The serve row a launch's `non-default args` belong to."""
    backend = args.get("attention_backend")
    if backend == "TRITON_ATTN":
        return "triton"
    if backend == "ROCM_AITER_FA":
        return "aiter-fa"
    return f"b{args['max_num_batched_tokens']}"


def seconds(line):
    h, m, s = map(int, STAMP.search(line).groups())
    return 3600 * h + 60 * m + s


def launches():
    """Each server launch as its startup log reports it, keyed by serve row."""
    out, current = {}, None
    with open(os.path.join(RAW, "sweep.log"), errors="replace") as f:
        for line in f:
            m = ARGS.search(line)
            if m:
                args = ast.literal_eval(m.group(1))     # vLLM prints a Python dict
                current = out.setdefault(serve_name(args), {"args": args, "start": seconds(line)})
                continue
            if current is None:
                continue
            for key, pattern in LOGGED.items():
                m = pattern.search(line)
                if m and key not in current:
                    value = next(g for g in m.groups() if g)
                    current[key] = value if key == "backend" else float(value.replace(",", ""))
            if "Starting vLLM server" in line and "ready_s" not in current:
                current["ready_s"] = seconds(line) - current["start"]
            if "Traceback" in line:
                current["traceback"] = True
    return out


def rows():
    """Every (serve, bench) cell: its repeats, and the means a prediction needs."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RESULTS, "SERVE--*"))):
        runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "run=*.json")))]
        serve, bench = re.match(r"SERVE--(.+)-BENCH--(.+)", os.path.basename(d)).groups()
        first = runs[0]
        mean = lambda k: statistics.mean(r[k] for r in runs)   # noqa: E731
        input_len = first["total_input_tokens"] // first["num_prompts"]
        out[serve, bench] = {
            "c": first["max_concurrency"],
            "repeats": len(runs),
            "input_len": input_len,
            "context": input_len + OUTPUT_TOKENS // 2,
            "itl_ms": mean("median_itl_ms"),
            "itl_each": [r["median_itl_ms"] for r in runs],
            "tpot_p50_ms": mean("median_tpot_ms"),
            "tpot_p99_ms": mean("p99_tpot_ms"),
            "ttft_p50_ms": mean("median_ttft_ms"),
            "ttft_p99_ms": mean("p99_ttft_ms"),
            "output_tps": mean("output_throughput"),
            "failed": sum(r["failed"] for r in runs),
        }
    return out


def gauges():
    """The engine's 10-second lines, grouped by (serve, bench) row.

    The sweep's own markers reach the log in late blocks; the server's and the
    client's lines arrive in order, so a launch's `non-default args` and each
    client's Namespace line say which row a gauge line belongs to.
    """
    out, serve, bench = {}, None, None
    with open(os.path.join(RAW, "sweep.log"), errors="replace") as f:
        for line in f:
            m = ARGS.search(line)
            if m:
                serve = serve_name(ast.literal_eval(m.group(1)))
            if "Namespace(subparser=" in line and "bench_type='serve'" in line:
                c = int(re.search(r"max_concurrency=(\d+)", line).group(1))
                n = re.search(r"random_input_len=(\d+)", line).group(1)
                bench = f"c{c:03d}" + ("" if n == str(PROMPT_TOKENS) else f"-in{n}")
            m = run1.GAUGE.search(line)
            if m and bench:
                prompt, gen, running, waiting, kv = map(float, m.groups())
                out.setdefault((serve, bench), []).append(
                    {"prompt_tps": prompt, "gen_tps": gen, "running": running,
                     "waiting": waiting, "kv_pct": kv})
    return out


def preemptions():
    with open(os.path.join(RAW, "metrics-after.log")) as f:
        return [float(line.split()[-1]) for line in f
                if line.startswith("vllm:num_preemptions_total")]


def saturated_running(gauge, serve):
    """Running on the c256 lines within 10 % of the row's maximum; the maximum is a floor on the peak."""
    g = gauge[serve, SATURATED]
    top = max(x["running"] for x in g)
    return [x["running"] for x in g if x["running"] >= 0.9 * top]


def predicted_step_ms(row, accel=MI300X_RUN1):
    return tpot_floor(QWEN3_8B, accel, row["c"], row["context"]).seconds * 1000


def predicted_ttft_ms(row, accel=MI300X_RUN1):
    return ttft_floor(QWEN3_8B, accel, row["input_len"]).seconds * 1000


def error(predicted, measured):
    """SLO.md section 9's sign: negative when the prediction was too short."""
    return (predicted - measured) / measured


def implied_eff_mem(row):
    return predicted_step_ms(row, MI300X) * MI300X.achieved_bandwidth / row["itl_ms"]


def implied_mfu(row):
    return predicted_ttft_ms(row, MI300X) * MI300X.mfu / row["ttft_p50_ms"]


def fitted_eff_mem(data, serve="b2048"):
    return statistics.mean(implied_eff_mem(data[serve, n]) for n in DECODE_ROWS)


def modelled_tpot_ms(data, serve, bench):
    """Run 1's interference model, fed this row's own median ITL and c001 TTFT."""
    r = data[serve, bench]
    alone = data[serve, "c001"]["ttft_p50_ms"]
    return tpot_with_interference(r["itl_ms"], r["c"], alone, OUTPUT_TOKENS)


def dollars_per_1m(tokens_per_second):
    return HOURLY / (tokens_per_second * 3600) * 1e6


def rule(width=96):
    print("-" * width)


def checkpoint_a(launch):
    print()
    print("CHECKPOINT A -- five startup logs")
    print()
    print(f"{'serve row':>10} {'backend':>14} {'budget':>7} {'KV pool':>11} {'vs b2048':>9} "
          f"{'activation':>11} {'graphs':>8} {'build':>7} {'ready':>7}")
    rule()
    base = launch["b2048"]["kv_tokens"]
    for name in SERVE_ROWS:
        x = launch[name]
        build = f"{x['build_s']:.0f} s" if "build_s" in x else "-"
        print(f"{name:>10} {x['backend']:>14} {x['args']['max_num_batched_tokens']:>7} "
              f"{x['kv_tokens']:>11,.0f} {x['kv_tokens'] / base - 1:>+9.2%} "
              f"{x['activation_gib']:>8.2f} GiB {x['graph_gib']:>5.2f} GiB {build:>7} "
              f"{x['ready_s']:>5.0f} s")
    rule()
    print(f"  b2048 against run 1's launch 1: {base:,.0f} against "
          f"{run1.LOGGED_KV_TOKENS[0]:,}, {base / run1.LOGGED_KV_TOKENS[0] - 1:+.2%}")
    print(f"  tracebacks in the log: {sum(x.get('traceback', False) for x in launch.values())}")
    floor = 256 * (PROMPT_TOKENS + OUTPUT_TOKENS)
    print(f"  b8192's pool against 256 x {PROMPT_TOKENS + OUTPUT_TOKENS} = {floor:,}: "
          f"{launch['b8192']['kv_tokens'] - floor:+,.0f} tokens")


def faced(data):
    print()
    print("RUN 1'S COEFFICIENTS FACED -- b2048, run 1's server line, at MI300X_RUN1 "
          f"({MI300X_RUN1.achieved_bandwidth} / {MI300X_RUN1.mfu})")
    print()
    old = run1.rows()
    print(f"{'row':>12} {'ctx':>5} {'step @0.46':>11} {'median ITL':>11} {'error':>7} "
          f"{'implied eff_mem':>16} {'run 1 ITL':>10} {'change':>7}")
    rule()
    for name in DECODE_ROWS:
        r = data["b2048", name]
        step = predicted_step_ms(r)
        then = (f"{old[name]['itl_ms']:>8.2f}ms {r['itl_ms'] / old[name]['itl_ms'] - 1:>+7.1%}"
                if name in old else f"{'-':>10} {'-':>7}")
        print(f"{name:>12} {r['context']:>5} {step:>9.2f}ms {r['itl_ms']:>9.2f}ms "
              f"{error(step, r['itl_ms']):>+7.1%} {implied_eff_mem(r):>16.3f} {then}")
    rule()
    values = [implied_eff_mem(data["b2048", n]) for n in DECODE_ROWS]
    print(f"  refitted on this run: eff_mem {fitted_eff_mem(data):.3f} "
          f"over seven rows at batch 1-32, range {min(values):.3f}-{max(values):.3f}")
    for label, small, large in (("run 2", data["b2048", "c001-in2000"], data["b2048", "c001-in8000"]),
                                ("run 1", old["c001-in2000"], old["c001-in8000"])):
        extra_kv = (large["context"] - small["context"]) * QWEN3_8B.kv_bytes_per_token
        grow = large["itl_ms"] - small["itl_ms"]
        rate = extra_kv / (grow / 1000)
        fixed = small["itl_ms"] - small["context"] * QWEN3_8B.kv_bytes_per_token / rate * 1000
        print(f"  {label}, batch 1, 2 100 -> 8 100 context: +{grow:.2f} ms for "
              f"+{extra_kv / 1e9:.3f} GB, {rate / 1e12:.1f} TB/s; at zero KV the step "
              f"would be {fixed:.2f} ms")
    weights = QWEN3_8B.weights_bytes / MI300X.peak_bandwidth * 1000
    print(f"  the weights alone at 100 % of peak: {weights:.2f} ms")
    print()
    print(f"{'row':>12} {'TTFT @0.166':>12} {'TTFT p50':>10} {'error':>7} {'implied mfu':>12} "
          f"{'run 1 TTFT':>11}")
    rule()
    for name in LENGTH_ROWS:
        r = data["b2048", name]
        ttft = predicted_ttft_ms(r)
        print(f"{name:>12} {ttft:>10.1f}ms {r['ttft_p50_ms']:>8.1f}ms "
              f"{error(ttft, r['ttft_p50_ms']):>+7.1%} {implied_mfu(r):>12.3f} "
              f"{old[name]['ttft_p50_ms']:>9.1f}ms")
    rule()


def interference(data):
    print()
    print("INTERFERENCE -- TPOT p50 against run 1's model with each row's own "
          "median ITL and c001 TTFT")
    print()
    print(f"{'serve row':>10} {'TTFT c001':>10} " + " ".join(
        f"{b + ' meas/model':>22}" for b in LOADED_ROWS))
    rule(110)
    for serve in AT_CONCURRENCY:
        cells = []
        for bench in LOADED_ROWS:
            r = data[serve, bench]
            model = modelled_tpot_ms(data, serve, bench)
            cells.append(f"{r['tpot_p50_ms']:>7.2f} / {model:>5.2f} {error(model, r['tpot_p50_ms']):>+6.1%}")
        print(f"{serve:>10} {data[serve, 'c001']['ttft_p50_ms']:>8.1f}ms " + " ".join(
            f"{c:>22}" for c in cells))
    rule(110)
    print("  table 12's column, fed MI300X_RUN1's floors, predicted "
          + " / ".join(f"{p:.2f}" for p in (17.68, 29.91, 42.15, 54.38)) + " ms")
    print()
    print(f"{'serve row':>10} " + " ".join(f"{b:>8}" for b in ("c001",) + LOADED_ROWS + (SATURATED,))
          + "   TPOT p99, ms")
    rule(70)
    for serve in AT_CONCURRENCY:
        print(f"{serve:>10} " + " ".join(
            f"{data[serve, b]['tpot_p99_ms']:>8.2f}" for b in ("c001",) + LOADED_ROWS + (SATURATED,)))
    rule(70)
    print()
    print(f"{'serve row':>10} " + " ".join(f"{b:>8}" for b in ("c001",) + LOADED_ROWS)
          + "   TTFT p99, ms; the interactive target is 300")
    rule(70)
    for serve in AT_CONCURRENCY:
        print(f"{serve:>10} " + " ".join(
            f"{data[serve, b]['ttft_p99_ms']:>8.0f}" for b in ("c001",) + LOADED_ROWS))
    rule(70)
    print("  TPOT p50 over the median ITL at c032: " + ", ".join(
        f"{s} {data[s, 'c032']['tpot_p50_ms'] / data[s, 'c032']['itl_ms']:.2f}"
        for s in AT_CONCURRENCY) + "; run 1 3.75")
    print(f"  every row read at concurrency is inside {TPOT_TARGET_MS:.0f} ms at c032 and "
          f"past it at {SATURATED}: the crossing is not placed within 8 seats")


def budget(data, gauge):
    print()
    print("WHAT THE BUDGET MOVED -- c256, the engine's gauges on the saturated lines")
    print()
    print(f"{'serve row':>10} {'budget':>7} {'ceiling':>8} {'running':>12} {'KV %':>6} "
          f"{'median ITL':>11} {'TTFT p50':>10} {'out tok/s':>10} {'vs 531.7':>9}")
    rule()
    run1_plateau = run1.rows()[SATURATED]["output_tps"]
    for serve in AT_CONCURRENCY:
        r = data[serve, SATURATED]
        b = SERVE[serve]["max_num_batched_tokens"]
        running = saturated_running(gauge, serve)
        kv = max(x["kv_pct"] for x in gauge[serve, SATURATED])
        print(f"{serve:>10} {b:>7} {token_budget_ceiling(b, PROMPT_TOKENS, OUTPUT_TOKENS):>8.1f} "
              f"{statistics.median(running):>5.0f} ({min(running):.0f}-{max(running):.0f}) "
              f"{kv:>6.1f} {r['itl_ms']:>9.2f}ms {r['ttft_p50_ms'] / 1000:>8.2f} s "
              f"{r['output_tps']:>10.1f} {r['output_tps'] / run1_plateau - 1:>+9.1%}")
    rule()
    p = preemptions()
    print(f"  preemptions: {int(max(p))} across all {len(p)} repeats")
    print(f"  median ITL at c256, b4096 over b2048: "
          f"x{data['b4096', SATURATED]['itl_ms'] / data['b2048', SATURATED]['itl_ms']:.2f}")


def prefill(data):
    print()
    print("PREFILL -- median TTFT at c = 1, three lengths, per serve row")
    print()
    print(f"{'serve row':>10} " + " ".join(f"{n:>12}" for n in LENGTH_ROWS)
          + f" {'x 2k->4k':>9} {'x 4k->8k':>9} {'mfu @4000':>10}")
    rule()
    for serve in SERVE_ROWS:
        t = [data[serve, n]["ttft_p50_ms"] for n in LENGTH_ROWS]
        print(f"{serve:>10} " + " ".join(f"{x:>10.1f}ms" for x in t)
              + f" {t[1] / t[0]:>8.2f}x {t[2] / t[1]:>8.2f}x "
              f"{implied_mfu(data[serve, 'c001']):>10.3f}")
    rule()
    d = QWEN3_8B.num_layers * 4096        # layers x (query heads x head_dim)
    total = lambda n: 2 * QWEN3_8B.params_non_embedding * n + 2 * n * n * d   # noqa: E731
    print(f"  attention FLOPs alone: {total(4000) / total(2000):.2f}x and "
          f"{total(8000) / total(4000):.2f}x; the sheet's line for chunking is 2.33x")
    base = data["b2048", "c001"]["ttft_p50_ms"]
    print(f"  b2048 moved TTFT at 4 000 from run 1's {RUN1_TTFT_4000_MS} ms by "
          f"{base / RUN1_TTFT_4000_MS - 1:+.1%}; against b2048 on this run:")
    for serve in SERVE_ROWS[1:]:
        print(f"    {serve:>9}: {data[serve, 'c001']['ttft_p50_ms'] / base - 1:+.1%}")


def cost(data):
    print()
    print(f"COST -- measured output throughput at ${HOURLY}/h")
    print()
    print(f"{'serve row':>10} {'c032 tok/s':>11} {'$/1M':>7} {'TPOT p99':>9} "
          f"{'c256 tok/s':>11} {'$/1M':>7}")
    rule(64)
    for serve in AT_CONCURRENCY:
        lo, hi = data[serve, "c032"], data[serve, SATURATED]
        print(f"{serve:>10} {lo['output_tps']:>11.1f} {dollars_per_1m(lo['output_tps']):>7.3f} "
              f"{lo['tpot_p99_ms']:>9.2f} {hi['output_tps']:>11.1f} "
              f"{dollars_per_1m(hi['output_tps']):>7.3f}")
    rule(64)
    l40s = run1.p99_crossings(run1.rows())["l40s"]
    even = HOURLY / (run1.L40S_COST["plateau"] * 3600) * 1e6
    print(f"  plateau break-even with the L40S's ${run1.L40S_COST['plateau']}: {even:.0f} tok/s")
    print(f"  L40S at its TPOT p99 crossing, interpolated: c = {l40s[0]:.1f}, ${l40s[1]:.3f}")
    print("  c032 is the last measured row inside 50 ms; the crossing lies past it, so")
    print("  its $/1M is an upper bound on the cost at the crossing")


def every_repeat(data):
    print()
    print("EVERY REPEAT -- median ITL per repeat, and failures")
    print()
    for (serve, bench), r in data.items():
        print(f"  {serve:>9} {bench:>12}  x{r['repeats']}  "
              + " / ".join(f"{x:.2f}" for x in r["itl_each"]) + f"  failed {r['failed']}")


if __name__ == "__main__":
    print(f"Run: {RUN}   raw evidence: docs/benchmarks/raw/{RUN}/run2/")
    print("Argument and conclusions: docs/benchmarks/mi300x-run2.md")
    print("Predictions: table 12 at MI300X_RUN1, fitted by run 1, not by anything below.")
    data, gauge = rows(), gauges()
    checkpoint_a(launches())
    faced(data)
    interference(data)
    budget(data, gauge)
    prefill(data)
    cost(data)
    every_repeat(data)
    print()
