"""MI300X run 5 scored -- a sibling of measured_mi300x_run4.py, one module per run.

Block B splits the card between the pair, then holds each process to one hardware
queue; block C faces InferenceX's Llama-3.3-70B FP8 claim at TP = 1 with the
ceiling ceiling.py printed before the card. The predictions are the runsheet's
(docs/benchmarks/runsheets/mi300x-run-5.md); the argument is
docs/benchmarks/mi300x-run5.md.

    python3 bench/measured_mi300x_run5.py
"""

import collections
import contextlib
import io
import json
import os
import re
import runpy
import statistics

from roofline import ACCELERATORS, decode_step_bytes, kv_cache_tokens, tpot_floor

RUN = "mi300x-2026-10-17"
RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "benchmarks", "raw", RUN, "run5",
)
# The model and context exactly as written before the card; its printout is silenced.
with contextlib.redirect_stdout(io.StringIO()):
    CEILING = runpy.run_path(os.path.join(RAW, "ceiling.py"))
MODEL, CONTEXT, GMU = CEILING["LLAMA_70B_FP8"], CEILING["CONTEXT"], CEILING["GMU"]
FIT = ACCELERATORS["mi300x-run2"]

HOURLY = 1.99                       # the create page, 2026-10-17
TAGS = ("plain", "masked", "q1")    # B0, B1, B2; each pair launched 8000 first
LEVELS = (4, 16, 64)
LONG_GAP = 3                        # an ITL this many times the level's median carries a prefill


def read(*parts):
    with open(os.path.join(RAW, *parts), encoding="utf-8") as f:
        return f.read()


def level(directory, name):
    return json.loads(read("results", directory, f"{name}.json"))


def rule(width=96):
    print("-" * width)


def checkpoint_a():
    print("CHECKPOINT A -- each Qwen3-8B launch's pool, and whether it compiled its graph")
    rule()
    for log in ("engine-warm.log", "engine-8000.log", "engine-8001.log",
                "engine-8000-masked.log", "engine-8001-masked.log",
                "engine-8000-q1.log", "engine-8001-q1.log"):
        text = read(log)
        tokens = re.search(r"GPU KV cache size: ([\d,]+) tokens", text).group(1)
        gib = re.search(r"Available KV cache memory: ([\d.]+) GiB", text).group(1)
        graph = "loaded" if "Directly load the compiled graph" in text else "compiled"
        print(f"  {log:<24}{tokens:>10} tokens  {gib:>6} GiB  graph {graph}")
    print("  runs 3-4: 506,525 tokens and 69.62 GiB on each engine")
    print()


def droplet_check():
    ttft = {}
    for d in ("droplet-check", "droplet-check-masked"):
        j = level(d, "fleet-c001-h00")
        ttft[d] = j["median_ttft_ms"]
        print(f"DROPLET CHECK ({d}) -- c = 1 on 8001: median ITL {j['median_itl_ms']:.2f} ms, "
              f"median TTFT {j['median_ttft_ms']:.1f} ms")
    print(f"  run 4 5.04 ms: FIT = mi300x-run2; the mask's TTFT ratio "
          f"{ttft['droplet-check-masked'] / ttft['droplet-check']:.2f}, gate at 1.5")
    print()


def block_b():
    print("BLOCK B -- fleet-bill-half on both engines at once, c = 32 each, 8000 launched first")
    rule()
    print(f"{'':>14}{'med ITL':>9}{'TPOT p50':>10}{'TPOT p99':>10}{'TTFT p50':>10}"
          f"{'tok/s':>8}{'seconds':>9}")
    for tag in TAGS:
        for e in ("8000", "8001"):
            j = level(f"{tag}-8000first-{e}", "fleet-c032-h00")
            print(f"{tag + ', ' + e:>14}{j['median_itl_ms']:>9.2f}{j['median_tpot_ms']:>10.2f}"
                  f"{j['p99_tpot_ms']:>10.2f}{j['median_ttft_ms']:>10.0f}"
                  f"{j['output_throughput']:>8.1f}{j['duration']:>9.1f}")
    for tag in TAGS:
        j = [level(f"{tag}-8000first-{e}", "fleet-c032-h00") for e in ("8000", "8001")]
        itl, tpot = (j[0][k] / j[1][k] for k in ("median_itl_ms", "median_tpot_ms"))
        print(f"  {tag}: 8000 over 8001 {itl:.2f}x by median ITL, {tpot:.2f}x by median TPOT; "
              f"the pair {sum(x['output_throughput'] for x in j):.0f} tok/s")
    print("  run 4: alone 12.43 / 12.34 ms; together, first launched 1.90x and 1.96x")
    print()


def mqds(tag):
    """Each process's queues from the host's debugfs dump: [(kind, dwords)] per PID."""
    out, cur = collections.defaultdict(list), None
    for line in read(f"mqds-{tag}-8000first.txt").splitlines():
        if line.startswith("Process"):
            pid = line.split()[1]
        elif "queue on device" in line:
            cur = []
            out[pid].append((line.split()[0], cur))
        elif re.match(r"\s+[0-9a-f]{8}:", line):
            cur.extend(int(w, 16) for w in line.split(":", 1)[1].split())
    return out


def cu_mask(dwords, xccs=8, stride=512):
    # COMPUTE_STATIC_THREAD_MGMT_SE0-SE3 at dwords 23, 24, 26, 27 of each XCC's
    # v9 MQD; GC 9.4.3 keeps one MQD per XCC, `stride` dwords apart.
    return tuple(tuple(dwords[x * stride + i] for i in (23, 24, 26, 27)) for x in range(xccs))


UNMASKED = cu_mask([0xFFFFFFFF] * 8 * 512)


def queues():
    print("QUEUES -- the host's MQDs, runlist and dmesg at each paired call")
    rule()
    host = read("host.txt").split("\n")
    print(f"  sched_policy {host[-3]}, hws_max_conc_proc {host[-2]}")
    for tag in TAGS:
        procs = mqds(tag)
        rls = read(f"rls-{tag}-8000first.txt")
        dmesg = read(f"dmesg-{tag}-8000first.txt").strip().splitlines()
        # The lower host PID opened the device first: 8000, launched first.
        kinds = {pid: collections.Counter(k for k, _ in qs) for pid, qs in procs.items()}
        masks = {pid: {cu_mask(q) for k, q in qs if k == "Compute"}
                 for pid, qs in procs.items()}
        print(f"  {tag}: " + "; ".join(
            f"pid {pid} {c['Compute']} compute + {c['SDMA']} SDMA queues, CU mask "
            + ("none" if masks[pid] == {UNMASKED} else
               f"{[sum(bin(w).count('1') for w in x) for x in next(iter(masks[pid]))]} per XCC")
            + ("" if len(masks[pid]) == 1 else " (queues differ)")
            for pid, c in sorted(kinds.items())))
        a, b = (next(iter(masks[pid])) for pid in sorted(masks))
        disjoint = all(x & y == 0 for xa, xb in zip(a, b) for x, y in zip(xa, xb))
        # PM4 headers: MAP_PROCESS (opcode 0xA1) and MAP_QUEUES (0xA2).
        print(f"    masks disjoint: {disjoint}; runlist maps {rls.count('c013a100')} processes "
              f"and {rls.count('c005a200')} queues; dmesg lines during the call: {len(dmesg)}")
    print()


def checkpoint_c():
    text = read("engine-70b.log")
    tokens = int(re.search(r"GPU KV cache size: ([\d,]+) tokens", text).group(1).replace(",", ""))
    gib = float(re.search(r"Available KV cache memory: ([\d.]+) GiB", text).group(1))
    profile = re.search(r"Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ "
                        r"non-torch\), ([\d.]+) GiB for peak activation, and ([\d.]+) GiB "
                        r"for CUDAGraph memory", text).groups()
    ceiling = kv_cache_tokens(MODEL, FIT, GMU)
    print("CHECKPOINT C -- the 70B's pool against ceiling.py")
    rule()
    print(f"  logged {tokens:,} tokens in {gib} GiB: {gib * 2**30 / tokens:,.0f} B per token, "
          f"derived {MODEL.kv_bytes_per_token:,.0f}")
    print(f"  ceiling {ceiling:,.0f} tokens: the log is {tokens / ceiling - 1:+.1%}; "
          f"c = 64 at 9 216 tokens needs {64 * 9216:,}")
    loading = re.search(r"Model loading took ([\d.]+) GiB", text).group(1)
    print(f"  model loading {loading} GiB; weights + non-torch {profile[0]} GiB, "
          f"peak activation {profile[1]} GiB, CUDA graphs {profile[2]} GiB")
    kv, linear, attention = (re.search(p, text).group(1) for p in (
        r"kv_cache_dtype=(\w+)", r"Selected (\w+) for Fp8LinearMethod", r"Overriding with (\w+)"))
    print(f"  kv_cache_dtype={kv}; {linear}; attention {attention}")
    print()


def preemptions(c):
    count = [float(re.search(r"num_preemptions_total\{[^}]*\} ([\d.]+)",
                             read(f"claim-c{c}-metrics-{side}.txt")).group(1))
             for side in ("before", "after")]
    return count[1] - count[0]


def block_c():
    print("BLOCK C -- the median ITL is the decode step; TPOT carries other requests' prefills")
    rule()
    print(f"{'c':>4}{'floor':>8}{'med ITL':>9}{'error':>8}{'eff_mem':>9}{'TPOT p50':>10}"
          f"{'1/TPOT':>8}{'1/ITL':>7}{'ceiling':>9}{'tok/s GPU':>11}{'ceiling':>9}"
          f"{'long gaps':>11}{'of time':>9}{'preempt':>9}{'TTFT p50':>10}")
    for c in LEVELS:
        j = json.loads(read("results", "claim", f"c{c}.json"))
        floor = tpot_floor(MODEL, FIT, c, CONTEXT).seconds * 1e3
        itl = j["median_itl_ms"]
        # eff_mem at which the memory side of the floor equals the measured step.
        eff = decode_step_bytes(MODEL, c, CONTEXT) / (FIT.peak_bandwidth * itl / 1e3)
        gaps = [g for r in j["itls"] for g in r]
        long = [g for g in gaps if g > LONG_GAP * itl / 1e3]
        print(f"{c:>4}{floor:>8.1f}{itl:>9.2f}{floor / itl - 1:>+8.1%}{eff:>9.3f}"
              f"{j['median_tpot_ms']:>10.1f}{1e3 / j['median_tpot_ms']:>8.1f}{1e3 / itl:>7.1f}"
              f"{1e3 / floor:>9.1f}{j['output_throughput']:>11.0f}{c * 1e3 / floor:>9.0f}"
              f"{len(long) / len(gaps):>11.2%}{sum(long) / sum(gaps):>9.1%}"
              f"{preemptions(c):>9.0f}{j['median_ttft_ms']:>10.0f}")
    print(f"  floor at eff_mem {FIT.achieved_bandwidth} (mi300x-run2) and context {CONTEXT}; "
          f"a long gap is over {LONG_GAP}x the level's median ITL")
    # c = 64 seated only what the pool held: the engine log's running count, 55-59.
    itl = json.loads(read("results", "claim", "c64.json"))["median_itl_ms"]
    for running in (55, 59):
        floor = tpot_floor(MODEL, FIT, running, CONTEXT).seconds * 1e3
        eff = decode_step_bytes(MODEL, running, CONTEXT) / (FIT.peak_bandwidth * itl / 1e3)
        print(f"  c = 64 at {running} running: floor {floor:.1f} ms, error {floor / itl - 1:+.1%}, "
              f"eff_mem {eff:.3f}")
    print()


def decode_intervals():
    """vLLM's 10-s log lines with no prompt tokens: the decode step as the engine saw it."""
    print("DECODE-ONLY INTERVALS -- engine log lines with no prefill, running / generation tok/s")
    rule()
    steps = collections.defaultdict(list)
    for m in re.finditer(r"Avg prompt throughput: 0\.0 tokens/s, Avg generation throughput: "
                         r"([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
                         r"GPU KV cache usage: ([\d.]+)%", read("engine-70b.log")):
        gen, running, waiting, kv = float(m[1]), int(m[2]), int(m[3]), float(m[4])
        if gen > 0 and running > 1:
            steps[running].append((running / gen * 1e3, waiting, kv))
    # A bucket of one or two lines is a level's edge, where the count changed mid-interval.
    for running, rows in sorted((r, v) for r, v in steps.items() if len(v) >= 3):
        print(f"  running {running:>2}: {len(rows)} intervals, step "
              f"{statistics.median(s for s, _, _ in rows):.1f} ms, waiting "
              f"{sorted({w for _, w, _ in rows})}, KV {max(k for _, _, k in rows):.1f} % at most")
    print()


def cost():
    up = read("droplet-up.txt").strip()
    harvest = read("harvest-start.txt").strip()
    print(f"COST -- work directory made {up}; harvest began {harvest}; "
          f"the droplet's lifetime is in the report, at ${HOURLY}/h")


if __name__ == "__main__":
    checkpoint_a()
    droplet_check()
    block_b()
    queues()
    checkpoint_c()
    block_c()
    decode_intervals()
    cost()
