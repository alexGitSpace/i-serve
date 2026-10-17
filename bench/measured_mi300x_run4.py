"""MI300X run 4 scored -- a sibling of measured_mi300x_run3.py, one module per run.

Block 0 asks why one engine of the pair was slower; block B repeats run 3's
router arm with the cache reset and two warmup passes. Reads the six startup
logs, the CPU and device records of block 0, and the harness's artefacts. The
predictions are the runsheet's (docs/benchmarks/runsheets/mi300x-run-4.md);
the argument is docs/benchmarks/mi300x-run4.md.

    python3 bench/measured_mi300x_run4.py
"""

import collections
import json
import os
import re

from measured_mi300x_run3 import ring_owner
from predictions import FLEET_REPLICAS, SHARED_PREFIX_TOKENS, retained_room
from scenarios.fleet import PROMPT_TOKENS, ROUTER_ARM, WORKING_SETS

RUN = "mi300x-2026-10-17"
RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "benchmarks", "raw", RUN, "run4",
)
RESULTS = os.path.join(RAW, "results")

HOURLY = 1.99                       # the create page, 2026-10-17
NOMINAL_H = SHARED_PREFIX_TOKENS / PROMPT_TOKENS
ENGINES = ("8000", "8001")
# Which engine each launch started first: s1.sh, then s3.sh.
LAUNCHES = (("first launch", "together-8000first", "8000", "", ""),
            ("relaunch, 8001 first", "together-8001first", "8001", "-r2", "-r2"))
ARMS = ("prefix-n031", "round_robin-n031", "round_robin-n063", "prefix-n063",
        "prefix-n127", "round_robin-n127", "round_robin-n255", "prefix-n255",
        "prefix-n511", "round_robin-n511", "prefix-n127-bl1")


def read(*parts):
    with open(os.path.join(RAW, *parts), encoding="utf-8") as f:
        return f.read()


def level(directory, name):
    return json.loads(read("results", directory, f"{name}.json"))


def requests(directory, name):
    return [json.loads(line) for line in read("results", directory,
                                              f"{name}-requests.jsonl").splitlines()]


def rule(width=96):
    print("-" * width)


def checkpoint_a():
    print("CHECKPOINT A -- each launch's pool, and whether it compiled its graph")
    rule()
    for log in ("engine-8000-cold.log", "engine-8001-cold.log", "engine-8000.log",
                "engine-8001.log", "engine-8001-r2.log", "engine-8000-r2.log"):
        text = read(log)
        tokens = re.search(r"GPU KV cache size: ([\d,]+) tokens", text).group(1)
        gib = re.search(r"Available KV cache memory: ([\d.]+) GiB", text).group(1)
        graph = "loaded" if "Directly load the compiled graph" in text else "compiled"
        print(f"  {log:<22}{tokens:>10} tokens  {gib:>6} GiB  graph {graph}")
    print("  run 3: 506,525 tokens and 69.62 GiB on each engine")
    print()


def droplet_check():
    for d in ("check-cold", "check"):
        itl = level(d, "fleet-c001-h00")["median_itl_ms"]
        print(f"DROPLET CHECK ({d}) -- median ITL at c = 1: {itl:.2f} ms "
              f"(run 3 5.04, run 2 5.22, run 1 6.61): FIT = mi300x-run2")
    print()


def block_0():
    print("BLOCK 0 -- fleet-bill-half, c = 32 per engine, each call after a cache reset")
    rule()
    print(f"{'':>28}{'med ITL':>9}{'TPOT p50':>10}{'TPOT p99':>10}{'tok/s':>8}{'seconds':>9}")
    rows = [("alone, 8000", "alone-8000"), ("alone, 8001", "alone-8001")]
    for label, tag, first, *_ in LAUNCHES:
        rows += [(f"{tag[9:]}, {e}{' (first)' if e == first else ''}", f"{tag}-{e}")
                 for e in ENGINES]
    for label, d in rows:
        j = level(d, "fleet-c032-h00")
        print(f"{label:>28}{j['median_itl_ms']:>9.2f}{j['median_tpot_ms']:>10.2f}"
              f"{j['p99_tpot_ms']:>10.2f}{j['output_throughput']:>8.1f}{j['duration']:>9.1f}")
    alone = [level(f"alone-{e}", "fleet-c032-h00")["median_itl_ms"] for e in ENGINES]
    print(f"  alone: 8000 against 8001 {alone[0] / alone[1] - 1:+.1%}; predicted 12.46 ms, "
          f"equal within 5 %")
    for label, tag, first, *_ in LAUNCHES:
        itl = {e: level(f"{tag}-{e}", "fleet-c032-h00")["median_itl_ms"] for e in ENGINES}
        other = ({"8000", "8001"} - {first}).pop()
        print(f"  {label}: the first-launched {first} at {itl[first] / itl[other]:.2f}x "
              f"the other's median ITL; predicted ~24.9 ms each")
    print()


def cpu_and_device():
    """Busy threads per 5-s interval, the cores they ran on, and the card as a whole."""
    print("BLOCK 0b -- threads over 50 % of a core, and rocm-smi for the whole device")
    rule()
    for label, tag, first, *_ in LAUNCHES:
        port = {}
        for line in read(f"ps-{tag}.txt").splitlines()[1:]:
            pid, ppid, args = line.split(None, 2)
            m = re.search(r"--port (\d+)", args)
            port[pid] = m.group(1) if m else port.get(ppid, "")
        owner, cores = {}, collections.defaultdict(set)
        for line in read(f"psr-{tag}.txt").splitlines():
            f = line.split()
            if len(f) == 4 and f[0].isdigit():
                owner[f[1]] = port.get(f[0], "")
                cores[f[1]].add(f[2])
        busy = collections.defaultdict(list)
        for block in read(f"top-{tag}.txt").split("top - ")[2:]:
            for line in block.splitlines():
                f = line.split()
                if len(f) > 8 and f[0].isdigit() and float(f[8]) > 50:
                    busy[f[0]].append(float(f[8]))
        use = [int(x) for x in re.findall(r"GPU use \(%\): (\d+)", read(f"rocm-smi-{tag}.txt"))]
        watts = [float(x) for x in re.findall(r"Package Power \(W\): ([\d.]+)",
                                             read(f"rocm-smi-{tag}.txt"))]
        print(f"  {label}:")
        for tid, pcts in sorted(busy.items(), key=lambda kv: owner.get(kv[0], "")):
            print(f"    thread {tid} of engine {owner.get(tid, '?')}: {min(pcts):.1f}-"
                  f"{max(pcts):.1f} % in {len(pcts)} intervals, cores {sorted(cores[tid])}")
        print(f"    device busy {sorted(set(use))} %, power {min(watts):.0f}-{max(watts):.0f} W")
    print()


def block_b():
    room = retained_room()
    levels = {w.num_prefixes: w for w in ROUTER_ARM}
    owners = {n: [ring_owner(r.prompt_ids) for r in levels[n].build()[:n]] for n in WORKING_SETS}
    print(f"BLOCK B -- h per arm, room {room:.0f} requests per engine; every level after a "
          f"reset and two warmup passes")
    rule()
    print(f"{'level':<18}{'h':>7}{'h 8000':>8}{'h 8001':>8}{'to 8000':>9}{'to 8001':>9}"
          f"{'rotation':>10}{'hits/3200':>11}{'on owner':>10}{'moved to 8000/8001':>20}"
          f"{'prefixes moved':>16}")
    for tag in ARMS:
        arm, n = tag.split("-n")[0], int(tag.split("-n")[1][:3])
        d = level(tag, f"router-n{n:03d}")
        assert d["harness_cache_reset"] == 1 and d["harness_warmup_passes"] == 2, tag
        hits = json.loads(read("results", tag, f"router-n{n:03d}-metrics.json"))
        per_engine = n if arm == "round_robin" else n / FLEET_REPLICAS
        rotation = NOMINAL_H if per_engine <= room else 0.0
        owner_col = moved_col = first_col = ""
        if arm == "prefix":
            recs = requests(tag, f"router-n{n:03d}")
            moved = collections.Counter(r["upstream"][-4:] for r in recs
                                        if r["upstream"][-4:] != owners[n][r["index"] % n])
            owner_col = f"{1 - sum(moved.values()) / len(recs):.1%}"
            moved_col = f"{moved['8000']} / {moved['8001']}"
            # A prefix misses once on the engine it is moved to, then hits its copy there.
            first_col = len({r["index"] % n for r in recs
                             if r["upstream"][-4:] != owners[n][r["index"] % n]})
        print(f"{tag:<18}{d['harness_measured_hit_rate']:>7.3f}"
              f"{d['harness_hit_rate_127.0.0.1:8000']:>8.3f}"
              f"{d['harness_hit_rate_127.0.0.1:8001']:>8.3f}"
              f"{d['harness_upstream_http://127.0.0.1:8000']:>9.0f}"
              f"{d['harness_upstream_http://127.0.0.1:8001']:>9.0f}"
              f"{rotation:>10.3f}{hits['vllm:prefix_cache_hits'] / SHARED_PREFIX_TOKENS:>11.2f}"
              f"{owner_col:>10}{moved_col:>20}{first_col:>16}")
    print("  cache_reset 1 and warmup_passes 2 asserted on every level")
    print()


class BlockPool:
    """vLLM v0.27.1's free-block queue for one engine, in 16-token blocks; the rules
    it copies and where each is read: docs/benchmarks/mi300x-run4.md section 5."""

    def __init__(self, blocks):
        self.free = collections.OrderedDict((b, None) for b in range(blocks))
        self.by_hash, self.hash_of, self.ref, self.serial = {}, {}, collections.Counter(), 0

    def _pop(self):
        b, h = self.free.popitem(last=False)
        if h is not None and self.by_hash.get(h) == b:
            del self.by_hash[h]
        self.ref[b] += 1
        return b

    def admit(self, prefix, prompt_blocks, prefix_blocks):
        hit = []
        while len(hit) < prefix_blocks and (prefix, len(hit)) in self.by_hash:
            hit.append(self.by_hash[prefix, len(hit)])
        for b in hit:
            if self.ref[b] == 0:
                del self.free[b]
            self.ref[b] += 1
        blocks = hit + [self._pop() for _ in range(prompt_blocks - len(hit))]
        for i, b in enumerate(blocks[len(hit):], len(hit)):
            self.serial += 1
            self.hash_of[b] = (prefix, i) if i < prefix_blocks else ("unique", self.serial)
            self.by_hash.setdefault(self.hash_of[b], b)
        return blocks, len(hit)

    def grow(self, blocks, full):
        b = self._pop()
        self.serial += 1
        self.hash_of[b] = ("unique", self.serial) if full else None
        if full:
            self.by_hash[self.hash_of[b]] = b
        blocks.append(b)

    def release(self, blocks):
        for b in reversed(blocks):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free[b] = self.hash_of.get(b)
                if self.free[b] is None:
                    self.free.move_to_end(b, last=False)


def replay(tag, pool_tokens=506_525, block=16, output_tokens=200):
    """One level's measured requests, in their real order and engine, through BlockPool.

    Prompt blocks at `sent`, one output block per 16 tokens decoded, all freed at
    `sent + latency`; the two warmup passes run first, one request at a time.
    """
    arm, n = tag.split("-n")[0], int(tag.split("-n")[1][:3])
    recs = requests(tag, f"router-n{n:03d}")
    engines = sorted({r["upstream"] for r in recs})
    pools = {e: BlockPool(pool_tokens // block) for e in engines}
    prompt_blocks, prefix_blocks = PROMPT_TOKENS // block, SHARED_PREFIX_TOKENS // block
    out_blocks = -(-output_tokens // block)
    usual = collections.defaultdict(collections.Counter)
    for r in recs:
        usual[r["index"] % n][r["upstream"]] += 1
    for i in range(2 * n):
        e = engines[i % 2] if arm == "round_robin" else usual[i % n].most_common(1)[0][0]
        pools[e].release(pools[e].admit(i % n, prompt_blocks, prefix_blocks)[0])
    events = []
    for r in recs:
        events += [(r["sent"], 2, r, 0), (r["sent"] + r["latency"], 0, r, 0)]
        events += [(r["sent"] + r["ttft"] + j * block * r["tpot"], 1, r, j)
                   for j in range(out_blocks)]
    live, hits, prompts = {}, collections.Counter(), collections.Counter()
    cycle_hits, cycle_prompts = collections.Counter(), collections.Counter()
    for _, kind, r, j in sorted(events, key=lambda ev: ev[:2]):
        e = r["upstream"]
        if kind == 2:
            live[r["index"]], hit = pools[e].admit(r["index"] % n, prompt_blocks, prefix_blocks)
            hits[e] += hit * block
            prompts[e] += r["prompt_tokens"]
            cycle_hits[r["index"] // n] += hit * block
            cycle_prompts[r["index"] // n] += r["prompt_tokens"]
        elif kind == 1 and r["index"] in live:
            pools[e].grow(live[r["index"]], j < out_blocks - 1)
        elif kind == 0:
            pools[e].release(live.pop(r["index"]))
    return (sum(hits.values()) / sum(prompts.values()),
            [hits[e] / prompts[e] for e in engines],
            [cycle_hits[c] / cycle_prompts[c] for c in sorted(cycle_prompts)])


def replayed():
    print("BLOCK B REPLAYED -- each level through a model of the block pool, against the run")
    rule()
    print(f"{'level':<18}{'h run':>7}{'h model':>9}{'8000 run/model':>18}{'8001 run/model':>18}")
    for tag in ARMS:
        n = int(tag.split("-n")[1][:3])
        d = level(tag, f"router-n{n:03d}")
        h, per, _ = replay(tag)
        print(f"{tag:<18}{d['harness_measured_hit_rate']:>7.3f}{h:>9.3f}"
              f"{d['harness_hit_rate_127.0.0.1:8000']:>11.3f} / {per[0]:.3f}"
              f"{d['harness_hit_rate_127.0.0.1:8001']:>11.3f} / {per[1]:.3f}")
    print()


def transient():
    """h per cycle of N requests in the replay, beside the run's TTFT p50 per cycle."""
    print("PER CYCLE -- the model's h, and the run's TTFT p50, cycle by cycle (N requests each)")
    rule()
    for tag in ("round_robin-n127", "prefix-n255", "round_robin-n255"):
        n = int(tag.split("-n")[1][:3])
        cycles = collections.defaultdict(list)
        for r in requests(tag, f"router-n{n:03d}"):
            cycles[r["index"] // n].append(r["ttft"] * 1e3)
        model = replay(tag)[2]
        print(f"  {tag:<18} model h {' '.join(f'{h:.2f}' for h in model)};  TTFT p50 "
              f"{' '.join(f'{sorted(v)[len(v) // 2]:.0f}' for _, v in sorted(cycles.items()))} ms")
    print("  the first cycle also carries the closed loop's start: 64 requests at once")
    print()


def cliff(pool_tokens=506_525, block=16, seats=32, cycles=6):
    """Strict rotation over W prefixes on one engine, after one warmup visit each."""
    request_blocks = (PROMPT_TOKENS + 200 + block - 1) // block
    print(f"THE CLIFF -- synthetic rotation, {seats} seats, {cycles} cycles; the pool holds "
          f"{pool_tokens / (request_blocks * block):.1f} whole requests")
    rule()
    for w in (89, 117, 120, 121, 122, 127):
        pool, inflight, per = BlockPool(pool_tokens // block), collections.deque(), []
        for p in range(w):
            pool.release(pool.admit(p, PROMPT_TOKENS // block, SHARED_PREFIX_TOKENS // block)[0])
        for _ in range(cycles):
            hits = 0
            for p in range(w):
                if len(inflight) == seats:
                    pool.release(inflight.popleft())
                blocks, hit = pool.admit(p, PROMPT_TOKENS // block, SHARED_PREFIX_TOKENS // block)
                for k in range(request_blocks - PROMPT_TOKENS // block):
                    pool.grow(blocks, k < request_blocks - PROMPT_TOKENS // block - 1)
                inflight.append(blocks)
                hits += hit * block
            per.append(hits / (PROMPT_TOKENS * w))
        print(f"  W = {w:>3} prefixes per engine: h per cycle {' '.join(f'{h:.3f}' for h in per)}")
    print(f"  the window that must fit is the room table 11 prints plus the {seats} live seats: "
          f"they are the newest members of the rotation")
    print()


def cost():
    first = min(r["sent"] for r in requests("check-cold", "fleet-c001-h00"))
    last = max(r["sent"] + r["latency"] for r in requests("prefix-n127-bl1", "router-n127"))
    print(f"COST -- load on the card {(last - first) / 60:.0f} min, first request to last; "
          f"the droplet's lifetime is in the report, at ${HOURLY}/h")


if __name__ == "__main__":
    checkpoint_a()
    droplet_check()
    block_0()
    cpu_and_device()
    block_b()
    replayed()
    transient()
    cliff()
    cost()
