"""The load harness: run a scenario against a server and write down what happened.

It orchestrates bench/scenarios, loadgen, stats and vllm_metrics, and adds what
none of them can: a verdict on whether each level is valid, printed beside the
row it judges, because a gate in a runsheet is read too late
(docs/benchmarks/l40s-run2.md section 6).

    python3 bench/harness.py --scenario seats-cached --out results/run3-cached \
        --startup-log /workspace/run3/serve-prefix-on.log
    python3 bench/harness.py --scenario seats-cached --dry-run   # no server needed
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict

from loadgen import Endpoint, Record, run_closed_loop, run_open_loop, send_one
from roofline import ACCELERATORS, QWEN3_8B, max_num_seqs_from_slo, tpot_floor, ttft_floor
from scenarios import Workload
from scenarios.prefix_sweep import SCENARIOS
from stats import LevelStats, SLOTargets, as_vllm_json, summarize, with_flag
from vllm_metrics import (delta, delta_fleet, hit_rate, pool_gate,
                          read_startup_log, scrape_fleet,
                          scrape, unread_startup_facts)

# Run 1's logged pool on the L40S, the reference the +-5% gate compares against
# (docs/benchmarks/l40s-baseline.md). A flag rather than a constant the moment a
# different card is used -- which is what --reference-pool is for.
DEFAULT_REFERENCE_POOL = 168_985


def _metrics_endpoints(values: list[str] | None,
                       ep: Endpoint) -> tuple[tuple[str, int], ...]:
    """`HOST:PORT` strings into pairs, or the load endpoint when none is given.

    Strict on purpose: a typo would silently read another engine's counters.
    """
    if not values:
        return ((ep.host, ep.port),)
    out = []
    for value in values:
        host, _, port = value.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"--metrics-endpoint {value!r}: expected HOST:PORT")
        out.append((host, int(port)))
    return tuple(out)


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_records(path: str, records: list[Record]) -> None:
    """Per-request lines, because a percentile is not evidence."""
    with open(path, "w", encoding="utf-8") as handle:
        for rec in records:
            row = asdict(rec)
            row["itls"] = [round(gap, 6) for gap in rec.itls]
            row["tpot"] = rec.tpot
            handle.write(json.dumps(row) + "\n")


def check_prefill_floor(stats: LevelStats, workload: Workload,
                        measured_h: float | None, accel) -> LevelStats:
    """Compare TTFT p50 with the floor of the tokens that were actually prefilled.

    The binding floor is at prompt_tokens x (1 - h measured), not at the prompt:
    comparing against the full prompt would invalidate every cached level. Below
    it no tokens were computed, so the workload repeats whole prompts.
    """
    share = measured_h if measured_h is not None else workload.nominal_hit_rate
    uncached = max(1, round(workload.prompt_tokens * (1.0 - share)))
    floor = ttft_floor(QWEN3_8B, accel, uncached).seconds
    full = ttft_floor(QWEN3_8B, accel, workload.prompt_tokens).seconds

    extra = {
        "prefill_floor_uncached_ms": floor * 1e3,
        "prefill_floor_full_prompt_ms": full * 1e3,
        "uncached_prompt_tokens": float(uncached),
    }
    if stats.ttft["p50"] < floor:
        return with_flag(
            stats, extra=extra,
            invalid=(f"ttft p50 {stats.ttft['p50'] * 1e3:.1f} ms is below the "
                     f"{floor * 1e3:.1f} ms prefill floor of {uncached} uncached "
                     f"tokens: the cache is being measured, not the engine"))
    return with_flag(stats, extra=extra)


def check_hit_rate(stats: LevelStats, workload: Workload,
                   measured_h: float | None) -> LevelStats:
    """Score the engine's counters against the h the prompts were built for.

    Construction and counters must agree to a point or two; a gap is a defect in
    the run, caught while the pod is still rented.
    """
    nominal = workload.nominal_hit_rate
    extra = {"nominal_hit_rate": nominal}
    if measured_h is None:
        note = ("no prefix-cache counters in /metrics: prefix caching is off on "
                "this server" if nominal == 0 else
                f"workload asks for h = {nominal:.2f} but the server reports no "
                f"prefix-cache queries at all -- caching is off")
        return with_flag(stats, warning=note, extra=extra)

    extra["measured_hit_rate"] = measured_h
    if abs(measured_h - nominal) > 0.05:
        return with_flag(
            stats, extra=extra,
            invalid=(f"measured h {measured_h:.3f} against nominal {nominal:.3f}: "
                     f"the workload is not producing the hit rate it claims"))
    return with_flag(stats, extra=extra)


def check_policy(stats: LevelStats, records: list[Record],
                 expected: str | None) -> LevelStats:
    """Was this level routed the way the arm says it was?

    Invalidates rather than warns: a level routed by the wrong policy measures
    something else (docs/benchmarks/runsheets/mi300x-run-3.md section 0).
    """
    if expected is None:
        return stats

    counts: dict[str, int] = {}
    for rec in records:
        if rec.ok:
            counts[rec.policy or "none"] = counts.get(rec.policy or "none", 0) + 1
    extra = {f"policy_{name}": float(n) for name, n in counts.items()}
    stats = with_flag(stats, extra=extra)

    wrong = sum(n for name, n in counts.items() if name != expected)
    if not counts:
        return stats
    if wrong:
        summary = ", ".join(f"{name} {n}" for name, n in sorted(counts.items()))
        return with_flag(stats, invalid=(
            f"{wrong} of {sum(counts.values())} responses were not routed "
            f"`{expected}` ({summary}): the arm is not the arm it claims"))
    return stats


async def sample_gauges(metrics: tuple[tuple[str, int], ...],
                        interval: float, into: dict) -> None:
    """Poll the engine's gauges while a level runs, keeping the peaks.

    Only the engine can say the batch reached the concurrency asked for. Off by
    default, since each poll is a round trip to the server under test; it runs in
    a thread so a blocking call cannot land in the ITL distribution.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            snapshots = await asyncio.to_thread(scrape_fleet, metrics)
        except OSError:
            continue                     # a refused scrape is not a failed level
        # Counts add across a fleet and a fraction does not: two engines at 40 %
        # of their pools are a fleet at 40 %, not at 80 %. Summing that column
        # would report a pool pressure no engine is under.
        for name, combine in (("vllm:num_requests_running", sum),
                              ("vllm:num_requests_waiting", sum),
                              ("vllm:gpu_cache_usage_perc", max)):
            values = [s.get(name) for s in snapshots]
            values = [v for v in values if v is not None]
            if values:
                key = f"max_{name.split(':')[1]}"
                into[key] = max(into.get(key, 0.0), combine(values))


async def run_level(workload: Workload, ep: Endpoint, targets: SLOTargets,
                    accel, settle: float, warm: bool, max_in_flight: int | None,
                    sample_interval: float = 0.0,
                    metrics: tuple[tuple[str, int], ...] = (),
                    expect_policy: str | None = None,
                    ) -> tuple[LevelStats, list[Record], dict]:
    """One level end to end: settle, warm the cache, scrape, load, scrape, judge.

    The order is the measurement: settle so stragglers are out of the step, warm
    before the first scrape so seeding misses fall outside the counter window.
    """
    # The load goes to one address; the counters come from the engines. The two
    # are the same thing only when nothing sits in front of them.
    metrics = metrics or ((ep.host, ep.port),)

    if settle > 0:
        await asyncio.sleep(settle)

    if warm and workload.prefix_tokens:
        for req in workload.warmup():
            record = await send_one(ep, req, time.perf_counter())
            if not record.ok:
                raise RuntimeError(
                    f"{workload.name}: warmup request failed ({record.error}); "
                    f"a cold prefix would make every h below meaningless")

    before = await asyncio.to_thread(scrape_fleet, metrics)
    requests = workload.build()
    peaks: dict[str, float] = {}
    sampler = (asyncio.create_task(sample_gauges(metrics, sample_interval, peaks))
               if sample_interval > 0 else None)
    try:
        if workload.mode == "closed":
            records, duration = await run_closed_loop(ep, requests, workload.concurrency)
        else:
            records, duration = await run_open_loop(
                ep, requests, workload.request_rate, seed=workload.seed,
                max_in_flight=max_in_flight)
    finally:
        if sampler is not None:
            sampler.cancel()
    after = await asyncio.to_thread(scrape_fleet, metrics)

    increments = delta_fleet(before, after)
    measured_h = hit_rate(increments)

    stats = summarize(workload.name, workload.mode, records, duration, targets)
    stats = check_hit_rate(stats, workload, measured_h)
    stats = check_prefill_floor(stats, workload, measured_h, accel)
    stats = check_policy(stats, records, expect_policy)

    load = workload.concurrency if workload.mode == "closed" else workload.request_rate
    extra = {
        "load": float(load),
        "prompt_tokens_nominal": float(workload.prompt_tokens),
        "prefix_tokens": float(workload.prefix_tokens),
        "preemptions": increments.get("vllm:num_preemptions", 0.0),
    }
    extra.update(peaks)
    stats = with_flag(stats, extra=extra)

    # A closed loop that never filled its seats measured a smaller batch than
    # the level is labelled with, and every figure in the row then belongs to
    # that smaller number instead.
    running = peaks.get("max_num_requests_running")
    if workload.mode == "closed" and running is not None and running < load:
        stats = with_flag(stats, warning=(
            f"num_requests_running peaked at {running:.0f} against a requested "
            f"concurrency of {load:.0f}"))
    return stats, records, increments


HEADER = (f"{'level':<20} {'load':>6} {'ok/f':>8} {'h':>6} "
          f"{'TTFT p50':>9} {'TTFT p99':>9} {'TPOT p50':>9} {'TPOT p99':>9} "
          f"{'medITL':>8} {'interf':>8} {'good/s':>7} {'tok/s':>8}")


def format_row(stats: LevelStats) -> str:
    ms = 1e3
    h = stats.extra.get("measured_hit_rate")
    return (f"{stats.label:<20} {stats.extra.get('load', 0):>6.4g} "
            f"{stats.completed:>4}/{stats.failed:<3} "
            f"{(f'{h:.3f}' if h is not None else '--'):>6} "
            f"{stats.ttft['p50'] * ms:>9.1f} {stats.ttft['p99'] * ms:>9.1f} "
            f"{stats.tpot['p50'] * ms:>9.2f} {stats.tpot['p99'] * ms:>9.2f} "
            f"{stats.itl['p50'] * ms:>8.2f} {stats.prefill_interference * ms:>8.2f} "
            f"{stats.goodput:>7.2f} {stats.output_throughput:>8.1f}")


def seat_verdict(levels: list[LevelStats], targets: SLOTargets) -> str:
    """The largest closed-loop concurrency whose TPOT p99 still met the target.

    Reported as an interval between the last pass and the first failure: the
    crossing lies between two integers, and one number would be unmeasured.
    """
    closed = [s for s in levels if s.mode == "closed" and s.completed]
    if not closed:
        return "seats: not measured -- this scenario has no closed-loop level"
    passing = [s for s in closed if s.tpot["p99"] <= targets.tpot]
    failing = [s for s in closed if s.tpot["p99"] > targets.tpot]
    if not passing:
        return (f"seats: none -- TPOT p99 exceeds {targets.tpot * 1e3:.0f} ms at "
                f"every concurrency measured")
    best = max(passing, key=lambda s: s.extra.get("load", 0))
    if not failing:
        return (f"seats: >= {best.extra['load']:.0f} -- the sweep never breached "
                f"TPOT p99 {targets.tpot * 1e3:.0f} ms, so the crossing is above it")
    first_fail = min(failing, key=lambda s: s.extra.get("load", 0))
    return (f"seats: between {best.extra['load']:.0f} and "
            f"{first_fail.extra['load']:.0f} at TPOT p99 <= "
            f"{targets.tpot * 1e3:.0f} ms")


def dry_run_plan(levels: tuple[Workload, ...], accel,
                 targets: SLOTargets) -> dict:
    """The plan and its derivable floors, as data, judged against the targets.

    The verdict column: ok (neither floor over its target), TTFT>, TPOT>, both. A
    floor over its target proves the level cannot pass with these coefficients;
    one under it says nothing. Poisson levels carry no TPOT floor: a rate is not a
    batch. A dict, so --json and the tests can read it.
    """
    provenance = accel.provenance or "provenance not stated; treat as a prior"
    rows = []
    for workload in levels:
        share = workload.nominal_hit_rate
        uncached = max(1, round(workload.prompt_tokens * (1 - share)))
        floor_unc = ttft_floor(QWEN3_8B, accel, uncached).seconds
        floor_full = ttft_floor(QWEN3_8B, accel, workload.prompt_tokens).seconds
        if workload.mode == "closed":
            load = workload.concurrency
            # prompt + output, the seat's reserved context, as --what-if does.
            step = tpot_floor(QWEN3_8B, accel, workload.concurrency,
                              workload.prompt_tokens + workload.output_tokens).seconds
            tpot_over = step > targets.tpot
        else:
            load, step, tpot_over = workload.request_rate, None, None
        rows.append({
            "name": workload.name,
            "mode": workload.mode,
            "load": load,
            "requests": workload.num_prompts,
            "prompt_tokens": workload.prompt_tokens,
            "prefix_tokens": workload.prefix_tokens,
            "output_tokens": workload.output_tokens,
            "nominal_hit_rate": share,
            "uncached_tokens": uncached,
            "ttft_floor_uncached_ms": floor_unc * 1e3,
            "ttft_floor_full_ms": floor_full * 1e3,
            "tpot_floor_ms": None if step is None else step * 1e3,
            "verdict": {"ttft_over": floor_unc > targets.ttft, "tpot_over": tpot_over},
        })
    # One context for the whole plan: the longest seat, i.e. the smallest count.
    context = max(w.prompt_tokens + w.output_tokens for w in levels)
    return {
        "accelerator": {
            "name": accel.name,
            "eff_mem": accel.achieved_bandwidth,
            "mfu": accel.mfu,
            "provenance": provenance,
        },
        "slo": {"ttft_ms": targets.ttft * 1e3, "tpot_ms": targets.tpot * 1e3},
        "seats_by_latency": max_num_seqs_from_slo(QWEN3_8B, accel, context, targets.tpot),
        "seat_context_tokens": context,
        "levels": rows,
        "totals": {
            "levels": len(levels),
            "requests": sum(w.num_prompts for w in levels),
            "output_tokens": sum(w.num_prompts * w.output_tokens for w in levels),
        },
    }


def _verdict_text(verdict: dict) -> str:
    ttft, tpot = verdict["ttft_over"], verdict["tpot_over"]
    if ttft and tpot:
        return "both"
    if ttft:
        return "TTFT>"
    if tpot:
        return "TPOT>"
    return "ok"


def dry_run(levels: tuple[Workload, ...], accel, targets: SLOTargets) -> None:
    """Print dry_run_plan() without touching a server.

    The first two lines are quoted by README.md and docs/audience.md and held
    byte-identical by a test: the provenance travels with the coefficients.
    """
    plan = dry_run_plan(levels, accel, targets)
    card = plan["accelerator"]
    print(f"accelerator: {card['name']}")
    print(f"coefficients: eff_mem {card['eff_mem']}, mfu {card['mfu']} "
          f"-- {card['provenance']}")
    print(f"slo: TTFT p99 <= {plan['slo']['ttft_ms']:.0f} ms, TPOT p99 <= "
          f"{plan['slo']['tpot_ms']:.0f} ms   (--slo-ttft-ms, --slo-tpot-ms)\n")
    print(f"{'level':<20} {'mode':>8} {'load':>6} {'reqs':>5} {'prompt':>7} "
          f"{'prefix':>7} {'nominal h':>10} {'floor@unc':>10} {'floor@full':>11} "
          f"{'TPOT floor':>11} {'vs SLO':>7}")
    for row in plan["levels"]:
        step = "--" if row["tpot_floor_ms"] is None else f"{row['tpot_floor_ms']:.2f}"
        print(f"{row['name']:<20} {row['mode']:>8} {row['load']:>6.4g} "
              f"{row['requests']:>5} {row['prompt_tokens']:>7} "
              f"{row['prefix_tokens']:>7} {row['nominal_hit_rate']:>10.3f} "
              f"{row['ttft_floor_uncached_ms']:>10.1f} "
              f"{row['ttft_floor_full_ms']:>11.1f} {step:>11} "
              f"{_verdict_text(row['verdict']):>7}")
    totals = plan["totals"]
    print(f"\n{totals['levels']} levels, {totals['requests']} requests, "
          f"{totals['output_tokens']:,} output tokens")
    print(f"the decode step alone permits {plan['seats_by_latency']} seats at "
          f"{plan['slo']['tpot_ms']:.0f} ms ({plan['seat_context_tokens']:,} tokens "
          f"of context per seat); every closed level above that is predicted to "
          f"breach\nbefore a single prefill lands in its steps")
    print("floors are milliseconds, from bench/roofline.py; 'unc' is the floor at "
          "the tokens\nleft to prefill once the cache serves its share -- the gate "
          "a level is judged against.\n'vs SLO' compares floors with targets: a "
          "floor over its target cannot pass, a floor under\nit proves nothing. "
          "A closed-loop TTFT is not a service metric either way; the harness "
          "flags it so")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="smoke", choices=sorted(SCENARIOS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--out", default=None,
                        help="directory for this run's artefacts")
    parser.add_argument("--accelerator", default="l40s-run1",
                        choices=sorted(ACCELERATORS),
                        help="which card, and which coefficients with it. "
                             "l40s-run1 is the L40S with the coefficients run 1 "
                             "fitted and runs 2-3 failed to break -- the default, "
                             "and the only fit a later run has faced. l40s is "
                             "the same card on spec-sheet priors, kept so a "
                             "prediction can be printed against either. "
                             "mi300x-run1 carries the MI300X's run-1 fit; mi300x "
                             "keeps its priors for contrast. The header prints "
                             "which you got")
    parser.add_argument("--slo-ttft-ms", type=float, default=300.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=50.0)
    parser.add_argument("--settle", type=float, default=5.0,
                        help="seconds of quiet between levels, so one level's "
                             "stragglers are not in the next level's step")
    parser.add_argument("--max-in-flight", type=int, default=None,
                        help="safety valve for open-loop levels; if it engages "
                             "the level is no longer open-loop and says so")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip seeding the shared prefix -- only for "
                             "measuring a cold cache deliberately")
    parser.add_argument("--sample-gauges", type=float, default=0.0,
                        metavar="SECONDS",
                        help="poll num_requests_running/waiting while each level "
                             "runs, keeping the peaks; costs one HTTP round trip "
                             "per sample against the server being measured")
    parser.add_argument("--metrics-endpoint", action="append", default=None,
                        metavar="HOST:PORT",
                        help="an engine to read /metrics from, repeatable. "
                             "Default: the load endpoint, which is right only "
                             "when nothing sits in front of it -- behind a "
                             "router /metrics is an unkeyed path and the answer "
                             "is one replica picked by the fallback")
    parser.add_argument("--expect-policy", default=None,
                        choices=("prefix", "round_robin"),
                        help="fail any level whose responses did not all carry "
                             "this X-Router-Policy. The gate for a two-arm "
                             "routing measurement: without it an arm that "
                             "silently fell back looks like a result")
    parser.add_argument("--startup-log", default=None,
                        help="vLLM server log, for the KV pool and config gates")
    parser.add_argument("--reference-pool", type=float, default=DEFAULT_REFERENCE_POOL)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the floors it will be "
                             "judged against, then exit -- no server, no "
                             "cluster, no card, no dependencies")
    parser.add_argument("--json", action="store_true",
                        help="with --dry-run: print the plan as JSON instead "
                             "of text. Meaningless on a live run, which "
                             "already writes run.json and one JSON per level")
    args = parser.parse_args(argv)

    levels = SCENARIOS[args.scenario]
    accel = ACCELERATORS[args.accelerator]
    # Built before the dry-run branch, not after it, since 2026-09-13: from the
    # day the harness was written the two --slo flags were parsed and then
    # thrown away on the one path a reader without a card can take.
    targets = SLOTargets(ttft=args.slo_ttft_ms / 1e3, tpot=args.slo_tpot_ms / 1e3)
    if args.json and not args.dry_run:
        parser.error("--json means nothing without --dry-run: a live run "
                     "writes run.json and one JSON per level into --out")
    if args.dry_run:
        if args.json:
            print(json.dumps(dry_run_plan(levels, accel, targets), indent=2,
                             sort_keys=True, allow_nan=False))
        else:
            dry_run(levels, accel, targets)
        return 0

    ep = Endpoint(host=args.host, port=args.port, model=args.model,
                  api_key=args.api_key)
    try:
        metrics = _metrics_endpoints(args.metrics_endpoint, ep)
    except ValueError as exc:
        parser.error(str(exc))
    out = args.out or os.path.join(
        "results", f"{time.strftime('%Y%m%d-%H%M%S')}-{args.scenario}")
    os.makedirs(out, exist_ok=True)

    run_meta: dict = {
        "scenario": args.scenario,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "endpoint": {"host": ep.host, "port": ep.port, "model": ep.model},
        "metrics_endpoints": [f"{host}:{port}" for host, port in metrics],
        "expect_policy": args.expect_policy,
        "accelerator": accel.name,
        # The console header says where the coefficients came from; so must
        # the file, or a results/ directory read months later cannot say
        # whether its floors stand on a fit or on a spec sheet.
        "coefficients": {
            "achieved_bandwidth": accel.achieved_bandwidth,
            "mfu": accel.mfu,
            "provenance": accel.provenance or "not stated",
        },
        "slo": {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms},
    }

    # The startup log first, before a single request: it is the one fact that can
    # invalidate the whole sweep, and this is the cheapest moment to learn it.
    if args.startup_log:
        facts = read_startup_log(args.startup_log)
        missing = unread_startup_facts(facts)
        run_meta["startup_log"] = facts
        run_meta["startup_log_unread"] = list(missing)
        print(f"startup log: {facts}")
        if missing:
            # Loud, and before any level: every gate below reads one of these by
            # name, and a key that is absent disables its gate in silence.
            print(f"  -> NOT FOUND in the log: {', '.join(missing)}")
            print("  -> a gate that reads a missing fact does not fire and does "
                  "not complain; fix the pattern in bench/vllm_metrics.py before "
                  "spending a level on it")
        if "kv_cache_tokens" in facts:
            ok, verdict = pool_gate(facts["kv_cache_tokens"], args.reference_pool)
            run_meta["pool_gate"] = verdict
            print(verdict)
            if not ok:
                print("  -> this pod is not comparable with the reference run; "
                      "results are still recorded, and any seat count from them "
                      "belongs to this pod alone")
        if str(facts.get("enable_prefix_caching")) == "False" and \
                any(w.hit_rate_target for w in levels):
            print("  -> the server has prefix caching OFF and this scenario asks "
                  "for a hit rate: stop and relaunch, or the run measures nothing")
            return 2

    print(f"\n{HEADER}")
    collected: list[LevelStats] = []
    for workload in levels:
        stats, records, increments = asyncio.run(run_level(
            workload, ep, targets, accel, args.settle,
            warm=not args.no_warmup, max_in_flight=args.max_in_flight,
            sample_interval=args.sample_gauges, metrics=metrics,
            expect_policy=args.expect_policy))
        collected.append(stats)
        print(format_row(stats))
        for note in stats.invalid:
            print(f"  INVALID  {note}")
        for note in stats.warnings:
            print(f"  warning  {note}")

        _write_json(os.path.join(out, f"{workload.name}.json"), as_vllm_json(stats))
        _write_json(os.path.join(out, f"{workload.name}-metrics.json"), increments)
        _write_records(os.path.join(out, f"{workload.name}-requests.jsonl"), records)

    print()
    print(seat_verdict(collected, targets))
    flagged = [s.label for s in collected
               if any(not note.startswith("ttft-not-a-service-metric")
                      for note in s.invalid)]
    if flagged:
        print(f"levels carrying an INVALID flag: {', '.join(flagged)}")

    run_meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_meta["levels"] = [s.label for s in collected]
    run_meta["seat_verdict"] = seat_verdict(collected, targets)
    _write_json(os.path.join(out, "run.json"), run_meta)
    print(f"\nartefacts: {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
