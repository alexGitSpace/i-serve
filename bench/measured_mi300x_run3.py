"""MI300X run 3 scored -- a sibling of measured_mi300x_run2.py, one module per run.

A fleet of two engines on one card, and a prefix router in front of it. Reads
the three startup logs and the harness's per-level and per-request artefacts.
The predictions are table 11 of bench/predictions.py; the argument is
docs/benchmarks/mi300x-run3.md.

    python3 bench/measured_mi300x_run3.py
"""

import collections
import json
import os
import re
import statistics

from predictions import (FLEET_CONCURRENCY, FLEET_REPLICAS, GMU, POOL_SHORTFALL,
                         SHARED_PREFIX_TOKENS, fleet_model, retained_room)
from roofline import MI300X, MI300X_RUN2, QWEN3_8B, kv_cache_tokens, tpot_floor
from scenarios.fleet import PROMPT_TOKENS, ROUTER_ARM, WORKING_SETS

RUN = "mi300x-2026-09-29"
RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "benchmarks", "raw", RUN, "run3",
)
RESULTS = os.path.join(RAW, "results")

HOURLY = 1.99                    # the console at creation, 2026-09-29
LOGGED_SOLO_RUNS_1_2 = 1_123_065  # both earlier droplets' pool, mi300x-run2.md section 2
MEDIAN_ITL_C1_MS = {"mi300x-run1": 6.61, "mi300x-run2": 5.22}  # runsheet section 3
NOMINAL_H = SHARED_PREFIX_TOKENS / PROMPT_TOKENS
# Both arms at one N send identical requests; this is the order the ten calls ran in.
FIRST_ARM = {31: "prefix", 63: "round_robin", 127: "prefix",
             255: "round_robin", 511: "prefix"}
FULL_HIT = (PROMPT_TOKENS - 1) // 16 * 16   # a whole cached prompt still recomputes its last token
ENGINES = ("8000", "8001")


def startup(name):
    """Pool, the memory it was sized from, and the graphs held outside it."""
    with open(os.path.join(RAW, f"engine-{name}.log"), encoding="utf-8") as f:
        text = f.read()
    grab = lambda pattern: float(re.search(pattern, text).group(1).replace(",", ""))
    return {"tokens": grab(r"GPU KV cache size: ([\d,]+) tokens"),
            "available_gib": grab(r"Available KV cache memory: ([\d.]+) GiB"),
            "graphs_gib": grab(r"Graph capturing finished .*?took ([\d.]+) GiB")}


def level(directory, name):
    with open(os.path.join(RESULTS, directory, f"{name}.json")) as f:
        return json.load(f)


def increments(directory, name):
    with open(os.path.join(RESULTS, directory, f"{name}-metrics.json")) as f:
        return json.load(f)


def requests(directory, name):
    with open(os.path.join(RESULTS, directory, f"{name}-requests.jsonl")) as f:
        return [json.loads(line) for line in f]


def engine_of(record):
    return record["upstream"][-4:]


def hash64(text):
    """router/ring.go's hash: FNV-1a 64, then MurmurHash3's finalizer."""
    mask = (1 << 64) - 1
    x = 0xcbf29ce484222325
    for byte in text.encode():
        x = ((x ^ byte) * 0x100000001b3) & mask
    for multiplier in (0xff51afd7ed558ccd, 0xc4ceb9fe1a85ec53):
        x ^= x >> 33
        x = (x * multiplier) & mask
    return x ^ (x >> 33)


def ring_owner(prompt_ids, model="Qwen/Qwen3-8B", key_bytes=512, vnodes=128):
    """The engine router/key.go and ring.pick choose for a prompt, bounded loads off."""
    body = ""
    for token in prompt_ids:
        if len(body) >= key_bytes:
            break
        body += f"{token}\x1d"
    key = hash64(f"{model}\0{body[:key_bytes]}")
    points = sorted((hash64(f"http://127.0.0.1:{e}#{i}"), e)
                    for e in ENGINES for i in range(vnodes))
    return next((e for h, e in points if h >= key), points[0][1])


def median_ms(values):
    return statistics.median(values) * 1e3


def rule(width=96):
    print("-" * width)


def checkpoint_a():
    solo, pair = startup("solo"), [startup(e) for e in ENGINES]
    derived_solo = kv_cache_tokens(QWEN3_8B, MI300X, GMU)
    derived_each = kv_cache_tokens(QWEN3_8B, MI300X, GMU / FLEET_REPLICAS)
    derived_bill = derived_solo - kv_cache_tokens(fleet_model(QWEN3_8B, FLEET_REPLICAS),
                                                  MI300X, GMU)
    print("CHECKPOINT A -- three startup logs")
    rule()
    print(f"  solo at {GMU}: {solo['tokens']:,.0f} tokens, "
          f"{solo['tokens'] / LOGGED_SOLO_RUNS_1_2 - 1:+.1%} on runs 1-2's logged "
          f"{LOGGED_SOLO_RUNS_1_2:,}; graphs {solo['graphs_gib']} GiB")
    corrected = derived_each * (1 - POOL_SHORTFALL)
    for name, e in zip(ENGINES, pair):
        print(f"  engine {name} at {GMU / FLEET_REPLICAS}: {e['tokens']:,.0f} tokens, "
              f"{e['tokens'] / corrected - 1:+.1%} on ~{corrected:,.0f} corrected; "
              f"available {e['available_gib']} GiB, graphs {e['graphs_gib']} GiB")
    bill = solo["tokens"] - sum(e["tokens"] for e in pair)
    print(f"  the second copy of the weights: {bill:,.0f} tokens measured, "
          f"{derived_bill:,.0f} derived")
    print()


def droplet_check():
    c1 = level("solo", "fleet-c001-h00")["median_itl_ms"]
    nearest = min(MEDIAN_ITL_C1_MS, key=lambda k: abs(MEDIAN_ITL_C1_MS[k] - c1))
    print(f"DROPLET CHECK -- median ITL at c = 1, 4 000 tokens: {c1:.2f} ms "
          f"(run 1 {MEDIAN_ITL_C1_MS['mi300x-run1']}, run 2 "
          f"{MEDIAN_ITL_C1_MS['mi300x-run2']}): FIT = {nearest}")
    print()


def block_a():
    one = tpot_floor(QWEN3_8B, MI300X_RUN2, FLEET_CONCURRENCY, PROMPT_TOKENS).seconds * 1e3
    two = tpot_floor(fleet_model(QWEN3_8B, FLEET_REPLICAS), MI300X_RUN2,
                     FLEET_CONCURRENCY, PROMPT_TOKENS).seconds * 1e3
    print(f"BLOCK A -- h = 0, {FLEET_CONCURRENCY} seats: one engine against two")
    rule()
    print(f"{'':>18}{'TPOT p50':>10}{'TPOT p99':>10}{'med ITL':>10}{'tok/s':>9}{'seconds':>9}")
    rows = [("solo, c = 64", level("solo", "fleet-c064-h00"))]
    rows += [(f"engine {e}, c = 32", level(f"pair-{e}", "fleet-c032-h00")) for e in ENGINES]
    for label, d in rows:
        print(f"{label:>18}{d['median_tpot_ms']:>10.2f}{d['p99_tpot_ms']:>10.2f}"
              f"{d['median_itl_ms']:>10.2f}{d['output_throughput']:>9.1f}{d['duration']:>9.1f}")
    fleet = sum(d["output_throughput"] for _, d in rows[1:])
    print(f"  the pair's output: {fleet:.1f} tok/s against the solo engine's "
          f"{rows[0][1]['output_throughput']:.1f}")
    print(f"  decode-step floor, eff_mem {MI300X_RUN2.achieved_bandwidth}: {one:.2f} -> "
          f"{two:.2f} ms; the solo median ITL is {rows[0][1]['median_itl_ms']:.1f} ms, "
          f"so at h = 0 it is not a decode step and the row is not faced")
    print()


def block_b():
    room = retained_room()
    print(f"BLOCK B -- h per arm; room {room:.0f} requests per engine; "
          f"the first arm at each N is the clean one")
    rule()
    print(f"{'N':>4} {'arm':<12}{'order':>7}{'h':>7}{'h 8000':>8}{'h 8001':>8}"
          f"{'to 8000':>9}{'to 8001':>9}{'uniform':>9}{'rotation':>10}{'hits/prefix':>13}")
    for n in WORKING_SETS:
        for arm in (FIRST_ARM[n], ({"prefix", "round_robin"} - {FIRST_ARM[n]}).pop()):
            d = level(f"{arm}-n{n:03d}", f"router-n{n:03d}")
            per_engine = n if arm == "round_robin" else n / FLEET_REPLICAS
            uniform = NOMINAL_H * min(1.0, room / per_engine)
            rotation = NOMINAL_H if per_engine <= room else 0.0
            order = "first" if arm == FIRST_ARM[n] else "second"
            # An integer here means every request hit its whole prefix or nothing.
            whole = increments(f"{arm}-n{n:03d}", f"router-n{n:03d}")["vllm:prefix_cache_hits"] \
                / SHARED_PREFIX_TOKENS
            print(f"{n:>4} {arm:<12}{order:>7}{d['harness_measured_hit_rate']:>7.3f}"
                  f"{d['harness_hit_rate_127.0.0.1:8000']:>8.3f}"
                  f"{d['harness_hit_rate_127.0.0.1:8001']:>8.3f}"
                  f"{d['harness_upstream_http://127.0.0.1:8000']:>9.0f}"
                  f"{d['harness_upstream_http://127.0.0.1:8001']:>9.0f}"
                  f"{uniform:>9.3f}{rotation:>10.3f}{whole:>13.2f}")
    print()


def contamination():
    """The second arm meets the first arm's identical requests, bodies included."""
    print("DEFECT 1 -- the second arm at one N hits the first arm's bodies")
    rule()
    print(f"{'N':>4} {'second arm':<12}{'same engine':>12}{'TTFT same':>11}"
          f"{'TTFT other':>11}{'h':>7}{'h if same = full':>18}")
    for n in WORKING_SETS:
        first = FIRST_ARM[n]
        second = ({"prefix", "round_robin"} - {first}).pop()
        before = {r["index"]: engine_of(r) for r in requests(f"{first}-n{n:03d}", f"router-n{n:03d}")}
        after = requests(f"{second}-n{n:03d}", f"router-n{n:03d}")
        same = [r for r in after if engine_of(r) == before[r["index"]]]
        other = [r for r in after if engine_of(r) != before[r["index"]]]
        share = len(same) / len(after)
        full = (share * FULL_HIT + (1 - share) * SHARED_PREFIX_TOKENS) / PROMPT_TOKENS
        h = level(f"{second}-n{n:03d}", f"router-n{n:03d}")["harness_measured_hit_rate"]
        print(f"{n:>4} {second:<12}{share:>12.3f}{median_ms(r['ttft'] for r in same):>11.1f}"
              f"{median_ms(r['ttft'] for r in other):>11.1f}{h:>7.3f}{full:>18.3f}")
    print("  the last column holds only while nothing is evicted: N = 31 and 63")
    print()


def warmup_parity():
    """One warmup pass of N (odd) requests flips round_robin's parity."""
    n = 63
    recs = sorted(requests(f"round_robin-n{n:03d}", f"router-n{n:03d}"), key=lambda r: r["index"])
    h = level(f"round_robin-n{n:03d}", f"router-n{n:03d}")["harness_measured_hit_rate"]
    print(f"DEFECT 2 -- round_robin at N = {n}, first at its N: one warmup pass")
    rule()
    print(f"  TTFT p50, first pass {median_ms(r['ttft'] for r in recs[:n]):.0f} ms, "
          f"later passes {median_ms(r['ttft'] for r in recs[n:]):.0f} ms")
    print(f"  h {h:.4f} measured, {NOMINAL_H * (len(recs) - n) / len(recs):.4f} if the "
          f"first pass is wholly cold")
    print()


def asymmetry():
    """Engine 8000 serves slower at an equal share, and bounded loads move its excess."""
    print("THE PAIR IS NOT SYMMETRIC -- TPOT p50 per engine (ms), each prefix's "
          "owner on the ring, and the requests bounded loads moved off it")
    rule()
    print(f"{'N':>4}{'rr 8000':>9}{'rr 8001':>9}{'prefix 8000':>13}{'prefix 8001':>13}"
          f"{'8001 owns':>11}{'on owner':>10}{'to 8001':>9}{'to 8000':>9}")
    levels = {w.num_prefixes: w for w in ROUTER_ARM}
    for n in WORKING_SETS:
        tpot = {}
        for arm in ("round_robin", "prefix"):
            recs = requests(f"{arm}-n{n:03d}", f"router-n{n:03d}")
            for e in ENGINES:
                tpot[arm, e] = median_ms(r["tpot"] for r in recs if engine_of(r) == e)
        owner = [ring_owner(req.prompt_ids) for req in levels[n].build()[:n]]
        moved = collections.Counter(engine_of(r) for r in recs
                                    if engine_of(r) != owner[r["index"] % n])
        on_owner = 1 - sum(moved.values()) / len(recs)
        print(f"{n:>4}{tpot['round_robin', '8000']:>9.1f}{tpot['round_robin', '8001']:>9.1f}"
              f"{tpot['prefix', '8000']:>13.1f}{tpot['prefix', '8001']:>13.1f}"
              f"{owner.count('8001') / n:>11.0%}{on_owner:>10.1%}"
              f"{moved['8001']:>9}{moved['8000']:>9}")
    points = sorted((hash64(f"http://127.0.0.1:{e}#{i}"), e) for e in ENGINES for i in range(128))
    arc = sum((h - points[i - 1][0]) % (1 << 64) for i, (h, e) in enumerate(points) if e == "8001")
    print(f"  the ring gives 8001 {arc / (1 << 64):.1%} of the key space: the ownership "
          f"column is that share sampled N times, and the excess is the moved requests")
    print()


def cost():
    first = min(r["sent"] for r in requests("solo", "fleet-c001-h00"))
    last = max(r["sent"] + r["latency"]
               for r in requests("round_robin-n511", "router-n511"))
    print(f"COST -- load on the card {(last - first) / 60:.0f} min; the droplet's "
          f"lifetime is in the report, at ${HOURLY}/h")


if __name__ == "__main__":
    checkpoint_a()
    droplet_check()
    block_a()
    block_b()
    contamination()
    warmup_parity()
    asymmetry()
    cost()
