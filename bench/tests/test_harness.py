"""The load harness, exercised end to end without a GPU -- or a server.

Sibling of bench/tests/test_roofline.py, and it holds the same rule from the other
side: that file asserts every figure docs/SLO.md publishes, this one asserts
every behaviour the harness is trusted for while a card is rented. The reason is
the same in both cases -- run 2 paid real money for two gates that were wrong,
and a gate is only worth having if something checks that it fires.

The centrepiece is a fake vLLM: 120 lines of asyncio that speak HTTP/1.1 with
chunked transfer, stream SSE token events on a timer, and emulate a prefix cache
well enough to move `vllm:prefix_cache_hits` the way the real one does. That is
what makes a 400-line async client debuggable on a laptop -- everything that can
be debugged without a card is debugged without one -- and it is
the only way to test the paths that matter -- a failed request, a stream that
ends early, a counter that goes backwards -- since none of them can be produced
on demand against a real server.

What the fake is *not*: a scheduler. It does not batch, queue, or slow down
under concurrency, so nothing here asserts a latency figure. Timings are checked
for shape only (a TTFT exists, there are output_tokens - 1 gaps), and every
number this repository publishes still comes from a rented card.

No dependency on pytest, which is not installed here -- plain functions, plain
asserts, and a runner at the bottom:

    python3 bench/tests/test_harness.py
"""

# Two ways in, and both have to work. `pytest bench/` gets bench/ on the import
# path from tests/conftest.py; `python3 bench/tests/test_roofline.py` on a rented
# pod, where pytest is not installed, gets only this directory. The three lines
# below are what make the second one work, and they are here rather than in
# conftest.py for exactly that reason.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import asyncio
import json
import math
import os
import time

from loadgen import (Endpoint, Record, Request, poisson_offsets,
                     run_closed_loop, run_open_loop, send_one)
import contextlib
import io

import harness
from harness import (check_hit_rate, check_prefill_floor, dry_run,
                     dry_run_plan, seat_verdict)
from roofline import L40S_RUN1
from scenarios import BLOCK_SIZE, Workload
from harness import SCENARIOS
from stats import (SLOTargets, as_vllm_json, max_concurrent, percentile,
                   summarize)
from vllm_metrics import (Snapshot, delta, hit_rate, parse_prometheus,
                          pool_gate, read_startup_log, unread_startup_facts)

# Resolved from this file, and not by counting "..": these tests moved down one
# directory once already, and the two raw-evidence paths below went with them.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


# --- the fake vLLM -----------------------------------------------------------

class FakeVLLM:
    """An OpenAI-compatible streaming endpoint plus a /metrics page.

    The prefix cache emulation follows vLLM's actual rule rather than a
    convenient one, because the harness's whole independent variable depends on
    it: KV is hashed in blocks of BLOCK_SIZE, a block's hash chains onto its
    predecessor's, and hits stop at the first miss -- a cache hit on a *prefix*
    is not a hit on a scattered subset. Getting this wrong in the fake would let
    a broken workload pass its test and fail on the card.

    Queries are counted over block-aligned prompt tokens, which is the one place
    this deliberately approximates: the real engine's denominator may include the
    trailing partial block. The difference at 1 500-token prompts is 12 tokens,
    0.8% of h -- an order of magnitude inside the 5-point gate the harness
    applies, and the reason that gate is 5 points and not 1.
    """

    def __init__(self, ttft: float = 0.004, itl: float = 0.002,
                 fail_after: int | None = None, truncate: bool = False,
                 dev_mode: bool = True):
        self.ttft = ttft
        self.itl = itl
        self.fail_after = fail_after
        self.truncate = truncate
        self.dev_mode = dev_mode
        self.served = 0
        self.in_flight = 0
        self.seen_blocks: set[int] = set()
        self.counters = {
            "vllm:prefix_cache_queries": 0.0,
            "vllm:prefix_cache_hits": 0.0,
            "vllm:num_preemptions": 0.0,
        }
        self.gauges = {"vllm:num_requests_running": 0.0,
                       "vllm:num_requests_waiting": 0.0}
        self.server: asyncio.Server | None = None
        self.port = 0

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        self.server.close()
        await self.server.wait_closed()

    def _account(self, prompt_ids: list[int]) -> None:
        aligned = len(prompt_ids) - len(prompt_ids) % BLOCK_SIZE
        self.counters["vllm:prefix_cache_queries"] += aligned
        chain, still_hitting, hits = 0, True, 0
        for start in range(0, aligned, BLOCK_SIZE):
            block = tuple(prompt_ids[start:start + BLOCK_SIZE])
            chain = hash((chain, block))
            if still_hitting and chain in self.seen_blocks:
                hits += BLOCK_SIZE
            else:
                still_hitting = False
            self.seen_blocks.add(chain)
        self.counters["vllm:prefix_cache_hits"] += hits

    async def _handle(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        request_line = await reader.readline()
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode().partition(":")
            headers[name.strip().lower()] = value.strip()

        path = request_line.decode().split()[1] if request_line else "/"
        if path == "/metrics":
            body = ("".join(f"{name}_total{{model_name=\"fake\"}} {value}\n"
                            for name, value in self.counters.items())
                    + "".join(f"{name}{{model_name=\"fake\"}} {value}\n"
                              for name, value in self.gauges.items())).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                         b"Content-Length: " + str(len(body)).encode()
                         + b"\r\nConnection: close\r\n\r\n" + body)
            await writer.drain()
            writer.close()
            return

        body = await reader.readexactly(int(headers.get("content-length", 0)))
        if path.startswith("/reset_prefix_cache"):
            # vLLM mounts the route only in dev mode, and clears every hash.
            if self.dev_mode:
                self.seen_blocks.clear()
            reply = (b"HTTP/1.1 200 OK\r\nContent-Length: 17\r\n\r\n{\"success\": true}"
                     if self.dev_mode else
                     b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            writer.write(reply)
            await writer.drain()
            writer.close()
            return
        payload = json.loads(body)
        self.served += 1

        if self.fail_after is not None and self.served > self.fail_after:
            message = b'{"error":"simulated overload"}'
            writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: "
                         + str(len(message)).encode() + b"\r\n\r\n" + message)
            await writer.drain()
            writer.close()
            return

        self._account(payload["prompt"])
        self.in_flight += 1
        self.gauges["vllm:num_requests_running"] = float(self.in_flight)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                     b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
        await writer.drain()

        def chunk(data: bytes) -> bytes:
            return f"{len(data):x}\r\n".encode() + data + b"\r\n"

        wanted = payload["max_tokens"]
        emitted = wanted // 2 if self.truncate else wanted
        for index in range(emitted):
            await asyncio.sleep(self.ttft if index == 0 else self.itl)
            event = {"choices": [{"text": "x", "index": 0, "finish_reason": None}]}
            writer.write(chunk(f"data: {json.dumps(event)}\n\n".encode()))
            await writer.drain()

        usage = {"choices": [], "usage": {"prompt_tokens": len(payload["prompt"]),
                                          "completion_tokens": emitted}}
        writer.write(chunk(f"data: {json.dumps(usage)}\n\n".encode()))
        writer.write(chunk(b"data: [DONE]\n\n"))
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        writer.close()
        self.in_flight -= 1
        self.gauges["vllm:num_requests_running"] = float(self.in_flight)


def with_server(coro, **kwargs):
    """Run one async test against a freshly started fake server."""
    async def wrapper():
        server = FakeVLLM(**kwargs)
        await server.start()
        try:
            return await coro(server, Endpoint(host="127.0.0.1", port=server.port,
                                               model="fake"))
        finally:
            await server.stop()
    return asyncio.run(wrapper())


# --- statistics --------------------------------------------------------------

def test_percentile_matches_numpy_linear():
    """The convention, not just the value: numpy's `linear`, as vllm bench uses.

    p50 of four sorted values is the midpoint of the middle two, and p99 of ten
    lands 99% of the way along the last interval. A nearest-rank implementation
    would return 3.0 and 10.0 here, which is how a harness and a published table
    end up one sample apart at exactly the tail the SLO is written against.
    """
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert abs(percentile(list(range(1, 11)), 99) - 9.91) < 1e-9
    assert percentile([7], 99) == 7
    assert math.isnan(percentile([], 50))


def _record(index=0, ttft=0.1, gaps=(0.02, 0.02), sent=0.0):
    return Record(index=index, prompt_tokens=100, scheduled=sent, sent=sent,
                  ttft=ttft, itls=list(gaps), latency=ttft + sum(gaps),
                  output_tokens=len(gaps) + 1)


def test_tpot_is_the_mean_gap_and_undefined_below_two_tokens():
    """docs/GLOSSARY.md: TPOT is the mean ITL of one request.

    The single-token case returns None rather than 0. A zero would be averaged
    into a percentile and would drag a level's TPOT toward a number no request
    ever experienced.
    """
    rec = _record(ttft=0.1, gaps=(0.02, 0.04))
    assert abs(rec.tpot - 0.03) < 1e-12
    single = Record(index=1, prompt_tokens=10, scheduled=0.0, sent=0.0,
                    ttft=0.1, itls=[], latency=0.1, output_tokens=1)
    assert single.tpot is None


def test_goodput_counts_only_requests_inside_every_threshold():
    """Goodput is not throughput with a discount: one breach removes the request.

    The level below has three requests, one of which breaches TTFT and one TPOT,
    so exactly one is good. Run 2 measured this collapsing to 0.12 req/s while
    output throughput was still climbing (docs/benchmarks/l40s-run2.md section 3)
    -- the whole point of the metric is that it can fall while the other rises.
    """
    records = [
        _record(0, ttft=0.100, gaps=(0.02, 0.02)),      # good
        _record(1, ttft=0.400, gaps=(0.02, 0.02)),      # TTFT breach
        _record(2, ttft=0.100, gaps=(0.09, 0.09)),      # TPOT breach
    ]
    stats = summarize("t", "poisson", records, duration=1.0, targets=SLOTargets())
    assert stats.completed == 3
    assert stats.goodput == 1.0
    assert stats.request_throughput == 3.0


def test_closed_loop_ttft_is_labelled_not_a_service_metric():
    """The trap both maps carry: a closed loop reports its own queue.

    The number is still computed -- it is comparable between levels of the same
    shape -- but it leaves summarize() already flagged, so no table can print it
    against the 300 ms target without the flag travelling alongside.
    """
    stats = summarize("t", "closed", [_record()], 1.0, SLOTargets())
    assert any("not-a-service-metric" in note for note in stats.invalid)
    assert not summarize("t", "poisson", [_record()], 1.0, SLOTargets()).invalid


def test_failed_requests_are_counted_never_dropped():
    records = [_record(0), Record(index=1, prompt_tokens=100, scheduled=0.0,
                                  sent=0.0, error="HTTP 503")]
    stats = summarize("t", "poisson", records, 1.0, SLOTargets())
    assert (stats.completed, stats.failed) == (1, 1)
    assert any("1 of 2 requests failed" in note for note in stats.warnings)


def test_max_concurrent_is_a_sweep_line_not_a_count():
    """Two requests that do not overlap are concurrency 1, not 2."""
    sequential = [_record(0, sent=0.0, gaps=(0.01,)), _record(1, sent=5.0, gaps=(0.01,))]
    overlapping = [_record(0, sent=0.0, gaps=(0.01,)), _record(1, sent=0.05, gaps=(0.01,))]
    assert max_concurrent(sequential) == 1
    assert max_concurrent(overlapping) == 2


def test_prefill_interference_is_tpot_minus_median_itl():
    """The quantity seats_under_prefill_interference() is fitted against.

    Computed on the level rather than by each caller, so the line fitted in a
    write-up and the line fitted in code cannot come from two definitions.
    """
    records = [_record(0, ttft=0.1, gaps=(0.030, 0.070))]   # median 0.05, mean 0.05
    stats = summarize("t", "closed", records, 1.0, SLOTargets())
    assert abs(stats.prefill_interference - 0.0) < 1e-12
    records = [_record(0, ttft=0.1, gaps=(0.030, 0.030, 0.030, 0.150))]
    stats = summarize("t", "closed", records, 1.0, SLOTargets())
    assert abs(stats.prefill_interference - (0.06 - 0.03)) < 1e-9


def test_vllm_json_keeps_the_tool_s_own_key_names():
    """A level from this harness must be readable by run 2's loader.

    Same keys, same milliseconds, so a figure quoted in docs/benchmarks/ can be
    found by one name in either kind of artefact -- and the harness's own
    additions are namespaced, so nothing can mistake one for a vLLM output.
    """
    payload = as_vllm_json(summarize("t", "poisson", [_record()], 2.0, SLOTargets()))
    for key in ("p99_ttft_ms", "median_itl_ms", "mean_tpot_ms", "request_goodput",
                "output_throughput", "total_input_tokens"):
        assert key in payload, key
    assert payload["harness_mode"] == "poisson"
    assert abs(payload["p50_ttft_ms"] - 100.0) < 1e-9


# --- workloads ---------------------------------------------------------------

def test_prefix_is_floored_onto_a_block_boundary():
    """h = 0.803 of 1 500 tokens is 1 204 requested and 1 200 cacheable.

    The nominal hit rate is recomputed from what survives the flooring, never
    from what was asked for: the engine caches whole blocks, so a prefix that
    ends mid-block would report a lower h than the flag claimed and the gap
    would be charged to the engine.
    """
    workload = Workload(name="w", prompt_tokens=1500, output_tokens=8,
                        num_prompts=4, concurrency=2, hit_rate_target=0.803)
    assert workload.prefix_tokens == 1200
    assert workload.prefix_tokens % BLOCK_SIZE == 0
    assert abs(workload.nominal_hit_rate - 0.8) < 1e-12
    assert workload.unique_tokens == 300


def test_every_request_shares_the_prefix_and_nothing_else():
    workload = Workload(name="w", prompt_tokens=320, output_tokens=4,
                        num_prompts=6, concurrency=2, hit_rate_target=0.5)
    requests = workload.build()
    prefix = requests[0].prompt_ids[:workload.prefix_tokens]
    bodies = set()
    for req in requests:
        assert req.prompt_ids[:workload.prefix_tokens] == prefix
        assert len(req.prompt_ids) == 320
        bodies.add(req.prompt_ids[workload.prefix_tokens:])
    assert len(bodies) == 6, "unique bodies must actually be unique"


def test_warmup_seeds_the_prefix_without_seeding_any_measured_body():
    """Otherwise one request of the level gets a full hit and lifts h.

    The bias is (1 - h) / num_prompts, small and with a known sign, which
    docs/SLO.md section 9 says to remove rather than to model.
    """
    workload = Workload(name="w", prompt_tokens=320, output_tokens=4,
                        num_prompts=6, concurrency=2, hit_rate_target=0.5)
    measured_bodies = {r.prompt_ids[workload.prefix_tokens:] for r in workload.build()}
    for warm in workload.warmup():
        assert warm.index < 0
        assert warm.prompt_ids[:workload.prefix_tokens] == \
            workload.build()[0].prompt_ids[:workload.prefix_tokens]
        assert warm.prompt_ids[workload.prefix_tokens:] not in measured_bodies


def test_workload_rejects_shapes_that_cannot_answer_anything():
    for kwargs in (
        dict(mode="closed", concurrency=None),           # no batch to hold
        dict(mode="poisson", request_rate=None),         # no arrivals
        dict(output_tokens=1),                           # no ITL, so no TPOT
        dict(hit_rate_target=1.0),                       # nothing left to prefill
        dict(num_prefixes=99),                           # more prefixes than requests
    ):
        base = dict(name="w", prompt_tokens=64, output_tokens=4, num_prompts=4,
                    mode="closed", concurrency=2)
        base.update(kwargs)
        try:
            Workload(**base)
        except ValueError:
            continue
        raise AssertionError(f"accepted an impossible workload: {kwargs}")


def test_scenarios_are_all_constructible():
    """Every named scenario is validated at import, so a typo cannot survive
    until the pod is rented. 510 requests is the cached seat sweep."""
    assert set(SCENARIOS) >= {"seats-cached", "seats-uncached",
                              "seats-cached-extended", "goodput-cached",
                              "goodput-bridge", "overhead", "smoke"}
    assert sum(w.num_prompts for w in SCENARIOS["seats-cached"]) == 510
    for levels in SCENARIOS.values():
        for workload in levels:
            assert workload.build()[0].prompt_tokens == workload.prompt_tokens


def test_each_block_runs_at_the_length_of_the_measurement_it_faces():
    """The defect the run-3 runsheet caught: one prompt length for two questions.

    The seat blocks are scored against docs/SLO.md section 6, whose 13/14 against
    24 +- 2 is derived at 4 000-token prompts and whose interference model was
    fitted there; at 1 500 tokens both crossings move to 31 and 61 seats, outside
    the very grid section 6's numbers chose. The goodput block is scored against
    run 2's curve, which is at 1 500. Asserted here because nothing else does:
    a single constant satisfied every other test in this file
    (docs/benchmarks/runsheets/l40s-run-3.md section 0).
    """
    for name in ("seats-uncached", "seats-cached", "seats-cached-extended",
                 "overhead"):
        for workload in SCENARIOS[name]:
            assert workload.prompt_tokens == 4_000, name
    for name in ("goodput-cached", "goodput-bridge"):
        for workload in SCENARIOS[name]:
            assert workload.prompt_tokens == 1_500, name

    # And the prefix lands on a block boundary exactly, so nominal h is the
    # number that was asked for rather than a rounded one.
    cached = SCENARIOS["seats-cached"][0]
    assert cached.prefix_tokens == 3_200
    assert cached.nominal_hit_rate == 0.8


def test_no_two_levels_of_a_sweep_send_the_same_prompts():
    """Run 3's defect: one seed for every level, on a server that remembers.

    `Workload.build` is deterministic in `seed`, and with one seed shared across a
    sweep a short level's prompts are a byte-identical prefix of the next level's.
    With prefix caching off that is invisible, which is how runs 1 and 2 never met
    it; with it on, every earlier level is a 100% cache hit for the later one, and
    run 3 measured h = 0.664 / 0.854 / 0.872 -- exactly 24/36, 36/42, 42/48 -- on
    levels whose nominal h was zero (docs/benchmarks/l40s-run3.md section 3).

    The rule the fix encodes: on a server that retains state between levels, sweep
    order is part of the measurement unless every level's input is its own.
    """
    for name, workloads in SCENARIOS.items():
        seen = {}
        for workload in workloads:
            first = workload.build()[0].prompt_ids
            body = first[workload.prefix_tokens:]
            assert body not in seen, (
                f"{name}: {workload.name} sends the same prompt body as "
                f"{seen[body]}, so on a caching server one level's h is the "
                f"other level's traffic")
            seen[body] = workload.name

    # Block C is the exception that proves it: the same level against two server
    # configurations must be byte-identical, or the pair measures the flag plus
    # a prompt change.
    overhead = SCENARIOS["overhead"][0]
    assert overhead.build()[0].prompt_ids == overhead.build()[0].prompt_ids


# --- the engine's counters ---------------------------------------------------

def test_prometheus_parse_folds_total_and_sums_series():
    text = ("# HELP vllm:prefix_cache_hits tokens\n"
            "# TYPE vllm:prefix_cache_hits counter\n"
            "vllm:prefix_cache_hits_total{model_name=\"a\"} 10.0\n"
            "vllm:prefix_cache_hits_total{model_name=\"b\"} 5.0\n"
            "vllm:num_requests_running 3.0\n")
    values = parse_prometheus(text)
    assert values["vllm:prefix_cache_hits"] == 15.0
    assert values["vllm:num_requests_running"] == 3.0


def test_hit_rate_is_a_window_and_none_means_the_feature_is_off():
    """The distinction the open item in docs/SLO.md section 10 turns on.

    h is the ratio of *increments* across one level, not a session average: the
    cold first level and the warm last one would otherwise be mixed into a
    number describing neither. And no queries at all is not h = 0 -- it is a
    server with prefix caching disabled, which is a different finding.
    """
    before = Snapshot(0.0, {"vllm:prefix_cache_queries": 1000.0,
                            "vllm:prefix_cache_hits": 100.0})
    after = Snapshot(1.0, {"vllm:prefix_cache_queries": 2000.0,
                           "vllm:prefix_cache_hits": 900.0})
    assert abs(hit_rate(delta(before, after)) - 0.8) < 1e-12
    assert hit_rate({}) is None


def test_a_counter_going_backwards_is_raised_not_clamped():
    """A restart mid-level invalidates every cross-level comparison."""
    before = Snapshot(0.0, {"vllm:prefix_cache_queries": 1000.0})
    after = Snapshot(1.0, {"vllm:prefix_cache_queries": 12.0})
    try:
        delta(before, after)
    except ValueError as exc:
        assert "restarted" in str(exc)
        return
    raise AssertionError("a counter reset was absorbed silently")


def test_startup_log_is_read_from_run_2_s_actual_log():
    """Parsed against the real evidence, not a fixture written to pass.

    docs/benchmarks/raw/l40s-2026-08-23/startup-lines.txt is what vLLM 0.27.1
    printed on the card. If a future version renames a line, this fails here
    rather than in the first minute of a rented pod.
    """
    path = os.path.join(REPO_ROOT, "docs", "benchmarks",
                        "raw", "l40s-2026-08-23", "startup-lines.txt")
    facts = read_startup_log(path)
    assert facts["kv_cache_tokens"] == 169_833
    assert facts["max_num_batched_tokens"] == 2048
    assert facts["max_concurrency"] == 18.87
    assert facts["enable_prefix_caching"] == "False"
    assert facts["attention_backend"] == "FLASH_ATTN"
    # And the fact that gate exists for: run 2's pod is 0.5% from run 1's logged
    # pool, well inside the 5% the platform's own relaunch noise demands.
    assert pool_gate(facts["kv_cache_tokens"], 168_985)[0] is True


def test_a_renamed_log_line_is_reported_rather_than_silently_ungated():
    """A gate that reads a missing fact does not fire and does not complain.

    read_startup_log returns only what matched, and bench/harness.py gates the
    KV pool only when the key is present -- so a vLLM release that renames one
    line runs the sweep ungated. This asserts the miss is *named*, which is what
    turns a silent hole into a first-minute fix (l40s-run-3.md section 3).
    """
    assert unread_startup_facts({}) != ()
    complete = {"kv_cache_tokens": 1.0, "kv_cache_gib": 1.0, "max_concurrency": 1.0,
                "max_num_batched_tokens": 1.0, "max_num_seqs": 1.0,
                "enable_prefix_caching": "True", "kv_cache_dtype": "auto",
                "attention_backend": "FLASH_ATTN"}
    assert unread_startup_facts(complete) == ()
    assert "kv_cache_tokens" in unread_startup_facts(
        {k: v for k, v in complete.items() if k != "kv_cache_tokens"})

    # Against the real evidence, seven of the eight patterns match -- and the
    # eighth is the first thing this function found: `max_num_seqs` appears in
    # no harvested startup log. Run 2 reported it as "256 (default, untouched)"
    # from the launch command, not from the engine, which is a different claim
    # and a weaker one. Asserted as it is rather than as it should be: if a
    # future version prints it, this fails and the assertion is what gets
    # updated; if any of the other seven stops matching, it fails for the reason
    # the function exists.
    path = os.path.join(REPO_ROOT, "docs", "benchmarks",
                        "raw", "l40s-2026-08-23", "startup-lines.txt")
    assert unread_startup_facts(read_startup_log(path)) == ("max_num_seqs",)


def test_pool_gate_is_looser_than_the_platform_s_own_noise():
    """4.2% between two launches of one identical config, so the gate is 5%.

    Run 2's checkpoint gate asked for +-0.3% of this quantity and would have
    failed on noise. A gate tighter than the measurement is not a strict gate,
    it is a broken one (docs/benchmarks/l40s-run2.md section 6).
    """
    assert pool_gate(168_985 * 1.042, 168_985)[0] is True
    assert pool_gate(168_985 * 1.061, 168_985)[0] is False


# --- the harness's own judgements --------------------------------------------

def test_a_ttft_below_the_uncached_prefill_floor_invalidates_the_level():
    """The gate that catches a harness measuring its own repeated prompts.

    At h = 0.8 of a 1 500-token prompt only 300 tokens are prefilled, so the
    floor that binds is the 300-token one, not the 1 500-token one. A level
    faster than *that* has not prefilled the tokens at all, which means the
    workload is repeating whole prompts (docs/GLOSSARY.md, prefix caching).
    """
    # Built here rather than taken from SCENARIOS: this asserts the gate, and a
    # test of a gate should not fail because a run changed its prompt length.
    workload = Workload(name="t", prompt_tokens=1_500, output_tokens=200,
                        num_prompts=24, concurrency=8, hit_rate_target=0.8)
    honest = summarize("t", "closed", [_record(ttft=0.080)], 1.0, SLOTargets())
    checked = check_prefill_floor(honest, workload, 0.8, L40S_RUN1)
    assert not [n for n in checked.invalid if "prefill floor" in n]

    impossible = summarize("t", "closed", [_record(ttft=0.004)], 1.0, SLOTargets())
    checked = check_prefill_floor(impossible, workload, 0.8, L40S_RUN1)
    assert [n for n in checked.invalid if "prefill floor" in n]
    # And the floor it was judged against is the uncached one, not the full prompt.
    assert checked.extra["uncached_prompt_tokens"] == 300.0
    assert checked.extra["prefill_floor_uncached_ms"] < \
        checked.extra["prefill_floor_full_prompt_ms"]


def test_measured_h_far_from_nominal_h_invalidates_the_level():
    """Two routes to one quantity; disagreement is a defect in the workload."""
    workload = SCENARIOS["seats-cached"][0]
    stats = summarize("t", "closed", [_record()], 1.0, SLOTargets())
    def hit_flags(measured):
        return [n for n in check_hit_rate(stats, workload, measured).invalid
                if "nominal" in n]

    assert not hit_flags(0.79), "a 1-point gap is the measurement, not a defect"
    assert hit_flags(0.42), "a 38-point gap is a workload that lost its prefix"
    warned = check_hit_rate(stats, workload, None)
    assert any("caching is off" in note for note in warned.warnings)


def test_seat_verdict_reports_an_interval_between_two_measured_levels():
    """The crossing sits between two integers, and only those two were measured.

    Reported rather than eyeballed, because the read-out is the standing failure:
    compute the limit before concluding.
    """
    def level(load, tpot_p99):
        stats = summarize(f"c{load}", "closed",
                          [_record(gaps=(tpot_p99, tpot_p99))], 1.0, SLOTargets())
        from stats import with_flag
        return with_flag(stats, extra={"load": float(load)})

    verdict = seat_verdict([level(12, 0.040), level(16, 0.061)], SLOTargets())
    assert "between 12 and 16" in verdict
    assert "none" in seat_verdict([level(12, 0.061)], SLOTargets())
    assert ">= 16" in seat_verdict([level(12, 0.040), level(16, 0.045)], SLOTargets())


# --- arrivals ----------------------------------------------------------------

# ---------------------------------------------------------------------------
# The dry run. Until 2026-09-13 it returned before SLOTargets was built, so the
# two --slo flags were accepted and ignored on the one path a reader without a
# card can take. These three hold the fix and the two lines README.md quotes.
# ---------------------------------------------------------------------------


def test_dry_run_verdict_flips_with_the_slo_flags():
    """The extended seat sweep runs above the 31 seats the decode step permits at
    50 ms, so every one of its levels is TPOT> against the interactive target
    and ok against the batch one. A dry run that printed the same column for
    both targets would be the old behaviour in a new column.
    """
    levels = SCENARIOS["seats-cached-extended"]
    tight = dry_run_plan(levels, L40S_RUN1, SLOTargets(ttft=0.300, tpot=0.050))
    loose = dry_run_plan(levels, L40S_RUN1, SLOTargets(ttft=3.000, tpot=0.200))
    assert tight["seats_by_latency"] == 31, tight["seats_by_latency"]
    assert all(row["verdict"]["tpot_over"] for row in tight["levels"]), \
        [(r["name"], r["verdict"]) for r in tight["levels"]]
    assert not any(row["verdict"]["tpot_over"] for row in loose["levels"])
    assert tight["slo"] == {"ttft_ms": 300.0, "tpot_ms": 50.0}
    # A rate is not a batch: Poisson levels carry no decode-step floor.
    poisson = dry_run_plan(SCENARIOS["goodput-cached"], L40S_RUN1, SLOTargets())
    assert all(row["tpot_floor_ms"] is None and row["verdict"]["tpot_over"] is None
               for row in poisson["levels"])


def test_dry_run_header_first_two_lines_are_the_ones_the_readme_quotes():
    """README.md and docs/audience.md print these two lines to the reader as
    the thing to read first. The SLO line goes third so they stay put.
    """
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        dry_run(SCENARIOS["seats-cached"], L40S_RUN1, SLOTargets())
    lines = out.getvalue().splitlines()
    assert lines[0] == "accelerator: NVIDIA L40S (run 1 coefficients)", lines[0]
    assert lines[1] == ("coefficients: eff_mem 0.83, mfu 0.439 -- measured "
                        "(run 1, 2026-08-18); survived runs 2-3"), lines[1]
    assert lines[2].startswith("slo: TTFT p99 <= 300 ms, TPOT p99 <= 50 ms"), lines[2]


def test_dry_run_json_is_only_a_dry_run_form():
    """A live run already writes run.json and one JSON per level; a second
    --json there would be a second copy of the same facts.
    """
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            harness.main(["--json"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("--json without --dry-run was accepted")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = harness.main(["--dry-run", "--json", "--scenario", "smoke",
                             "--slo-tpot-ms", "200"])
    assert code == 0
    plan = json.loads(out.getvalue())
    assert plan["slo"]["tpot_ms"] == 200.0
    assert {"accelerator", "slo", "seats_by_latency", "levels", "totals"} <= set(plan)


def test_poisson_offsets_are_monotone_with_the_right_mean_gap():
    offsets = poisson_offsets(rate=2.0, count=20_000, seed=7)
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    assert all(gap >= 0 for gap in gaps)
    assert abs(sum(gaps) / len(gaps) - 0.5) < 0.02
    # Exponential, not constant: the coefficient of variation of an exponential
    # is 1, and it is the coincident arrivals that build the queue a TTFT tail
    # is made of. A constant-interval generator would land near 0 here.
    mean = sum(gaps) / len(gaps)
    sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
    assert 0.9 < sd / mean < 1.1
    assert poisson_offsets(2.0, 10, 7) == poisson_offsets(2.0, 10, 7), "not seeded"


# --- end to end, against the fake ---------------------------------------------

def test_one_request_streams_and_is_timed():
    async def body(server, ep):
        req = Request(index=0, prompt_ids=tuple(range(1000, 1064)),
                      max_tokens=6, prefix_tokens=0)
        return await send_one(ep, req, time.perf_counter())

    rec = with_server(body, ttft=0.02, itl=0.005)
    assert rec.ok and rec.error is None
    assert rec.output_tokens == 6
    assert len(rec.itls) == 5, "one gap fewer than tokens, by definition"
    assert rec.usage_output_tokens == 6
    assert rec.ttft >= 0.02 and rec.latency > rec.ttft


def test_an_http_error_becomes_a_failed_record_not_an_exception():
    """A load generator that raises stops measuring; one that retries measures
    its retry policy. It records the failure and goes on."""
    async def body(server, ep):
        req = Request(index=0, prompt_ids=tuple(range(1000, 1032)),
                      max_tokens=4, prefix_tokens=0)
        return await send_one(ep, req, time.perf_counter())

    rec = with_server(body, fail_after=0)
    assert not rec.ok
    assert "503" in rec.error


def test_a_truncated_stream_is_recorded_with_the_tokens_that_arrived():
    async def body(server, ep):
        req = Request(index=0, prompt_ids=tuple(range(1000, 1032)),
                      max_tokens=10, prefix_tokens=0)
        return await send_one(ep, req, time.perf_counter())

    rec = with_server(body, truncate=True)
    assert rec.ok and rec.output_tokens == 5


def test_closed_loop_holds_exactly_the_concurrency_it_was_given():
    async def body(server, ep):
        workload = Workload(name="w", prompt_tokens=64, output_tokens=4,
                            num_prompts=12, mode="closed", concurrency=3)
        return await run_closed_loop(ep, workload.build(), 3)

    records, duration = with_server(body, ttft=0.01, itl=0.002)
    assert len(records) == 12 and all(r.ok for r in records)
    assert max_concurrent(records) <= 3
    assert duration > 0


def test_open_loop_arrives_on_schedule_regardless_of_completions():
    """The property that makes TTFT meaningful: arrivals are an input.

    The check is the harness's own lateness, which is also what the run reports:
    a generator that cannot keep up has silently become a closed loop, and the
    only way to know is to have measured it.
    """
    async def body(server, ep):
        workload = Workload(name="w", prompt_tokens=64, output_tokens=4,
                            num_prompts=20, mode="poisson", request_rate=50.0)
        return await run_open_loop(ep, workload.build(), 50.0, seed=3)

    records, duration = with_server(body, ttft=0.05, itl=0.002)
    assert len(records) == 20 and all(r.ok for r in records)
    assert max(r.lateness for r in records) < 0.050
    # 20 arrivals at 50/s take ~0.4 s while each request takes ~0.06 s: an open
    # loop finishes in arrival time, a closed loop would take 20 x 0.06 = 1.2 s.
    assert duration < 1.0


def test_the_hit_rate_the_engine_reports_matches_the_one_built_into_the_prompts():
    """The end-to-end assertion this whole harness exists for.

    A workload constructed at h = 0.75 is sent to a server that hashes blocks
    the way vLLM does, and the counters come back saying 0.75. Nominal and
    measured are two independent routes to the number the seat count will be
    plotted against; if they can disagree, the run has no independent variable.
    """
    async def body(server, ep):
        workload = Workload(name="w", prompt_tokens=256, output_tokens=3,
                            num_prompts=8, mode="closed", concurrency=2,
                            hit_rate_target=0.75)
        for req in workload.warmup():
            await send_one(ep, req, time.perf_counter())
        before = Snapshot(0.0, dict(server.counters))
        await run_closed_loop(ep, workload.build(), 2)
        after = Snapshot(1.0, dict(server.counters))
        return workload, hit_rate(delta(before, after))

    workload, measured = with_server(body)
    assert measured is not None
    assert abs(measured - workload.nominal_hit_rate) < 0.05, \
        f"measured {measured:.3f} against nominal {workload.nominal_hit_rate:.3f}"


def test_without_the_warmup_the_first_request_of_each_prefix_misses():
    """The bias the warmup removes, demonstrated rather than asserted in prose."""
    async def body(server, ep):
        workload = Workload(name="w", prompt_tokens=256, output_tokens=3,
                            num_prompts=8, mode="closed", concurrency=2,
                            hit_rate_target=0.75)
        before = Snapshot(0.0, dict(server.counters))
        await run_closed_loop(ep, workload.build(), 2)
        after = Snapshot(1.0, dict(server.counters))
        return workload, hit_rate(delta(before, after))

    workload, measured = with_server(body)
    assert measured < workload.nominal_hit_rate
    assert measured > workload.nominal_hit_rate * 0.8


def test_gauge_sampling_sees_the_batch_the_client_thinks_it_opened():
    """The check no latency can make: did the server actually seat n requests?

    Run 1 confirmed 41 seats by watching num_requests_running reach 41, and saw
    its preemption cascade as num_requests_waiting 39 beside it. A level whose
    batch never filled is labelled with a concurrency it did not run at.
    """
    from harness import run_level
    from stats import SLOTargets as Targets

    async def body(server, ep):
        workload = Workload(name="w", prompt_tokens=64, output_tokens=6,
                            num_prompts=9, mode="closed", concurrency=3)
        return await run_level(workload, ep, Targets(), L40S_RUN1, settle=0.0,
                               warm=False, max_in_flight=None,
                               sample_interval=0.01)

    stats, records, _ = with_server(body, ttft=0.03, itl=0.01)
    assert len(records) == 9
    assert stats.extra["max_num_requests_running"] >= 2, stats.extra
    assert not [w for w in stats.warnings if "peaked at" in w]


def test_a_fleets_counters_are_summed_before_the_ratio_is_taken():
    """Two engines that served different amounts do not get equal votes.

    hit_rate() divides hits by queries, so summing the counters first gives the
    fleet's token-weighted h; averaging two engines' ratios would give the idle
    one the same weight as the busy one.
    """
    from vllm_metrics import Snapshot, delta_fleet, hit_rate

    def snap(hits, queries, t):
        return Snapshot(values={"vllm:prefix_cache_hits": hits,
                                "vllm:prefix_cache_queries": queries}, at=t)

    before = (snap(0, 0, 0.0), snap(0, 0, 0.0))
    after = (snap(900, 1000, 1.0), snap(0, 1000, 1.0))   # one busy, one cold
    increments = delta_fleet(before, after)

    assert increments["vllm:prefix_cache_hits"] == 900
    assert hit_rate(increments) == 0.45                  # not (0.9 + 0.0) / 2


def test_a_half_read_fleet_is_not_a_measurement():
    """Mismatched snapshot counts mean one engine was not read, and its tokens
    are missing from a ratio that will be reported as the fleet's."""
    from vllm_metrics import Snapshot, delta_fleet

    one = (Snapshot(values={}, at=0.0),)
    two = (Snapshot(values={}, at=1.0), Snapshot(values={}, at=1.0))
    try:
        delta_fleet(one, two)
    except ValueError:
        return
    raise AssertionError("a fleet read twice at different widths was accepted")


def test_metrics_endpoints_default_to_the_load_endpoint():
    """Which is correct exactly when nothing sits in front of the engine -- the
    shape of every run before the router existed."""
    from harness import _metrics_endpoints
    from loadgen import Endpoint

    ep = Endpoint(host="127.0.0.1", port=8000)
    assert _metrics_endpoints(None, ep) == (("127.0.0.1", 8000),)
    assert _metrics_endpoints(["127.0.0.1:8000", "127.0.0.1:8001"], ep) == \
        (("127.0.0.1", 8000), ("127.0.0.1", 8001))
    for bad in ("8000", "host:", "host:port"):
        try:
            _metrics_endpoints([bad], ep)
        except ValueError:
            continue
        raise AssertionError(f"accepted --metrics-endpoint {bad!r}")


def test_a_level_routed_by_the_wrong_policy_is_invalid_not_noisy():
    """The gate that would have caught the token-id body shape.

    Every request falling back to round_robin while the arm believes it is
    routing by prefix is not a noisy prefix measurement -- it is a control arm
    wearing the treatment's name, and nothing else in the level says so
    (docs/benchmarks/runsheets/mi300x-run-3.md section 0).
    """
    from harness import check_policy
    from stats import SLOTargets as Targets, summarize

    def records(policies):
        out = []
        for i, policy in enumerate(policies):
            rec = Record(index=i, prompt_tokens=4, scheduled=0.0)
            rec.sent, rec.ttft, rec.policy = 0.0, 0.01, policy
            rec.latency, rec.itls, rec.output_tokens = 0.05, [0.01, 0.01], 3
            out.append(rec)
        return out

    # A closed-loop level already carries one invalid flag of its own
    # (ttft-not-a-service-metric), so the assertions are about the flag this
    # gate adds, not about the count.
    blank = summarize("w", "closed", [], 1.0, Targets())

    def routing_flags(stats):
        return [note for note in stats.invalid if "routed" in note]

    assert not routing_flags(check_policy(blank, records(["prefix"] * 4), "prefix"))
    assert not routing_flags(check_policy(blank, records(["no-prompt"] * 4), None))

    fell_back = check_policy(blank, records(["no-prompt"] * 4), "prefix")
    assert routing_flags(fell_back), "a level that never routed by prefix passed"
    assert fell_back.extra["policy_no-prompt"] == 4.0, fell_back.extra


def test_round_robin_over_an_even_rotation_is_affinity():
    """Under the rotation an even N pins each prefix to one engine, so every working set is odd (runsheet mi300x-run-3 section 0)."""
    from scenarios.fleet import WORKING_SETS

    def engines_per_prefix(n):
        seen = {}
        for i in range(4 * n):
            seen.setdefault(i % n, set()).add(i % 2)
        return {len(engines) for engines in seen.values()}

    assert engines_per_prefix(128) == {1}
    for n in WORKING_SETS:
        assert engines_per_prefix(n) == {2}, n


def test_one_warmup_pass_under_round_robin_leaves_the_first_pass_cold():
    """With N odd, one warmup pass leaves every first-pass prefix cold under round robin."""
    from scenarios.fleet import WORKING_SETS

    def cold_first_pass(n, passes, cursor):
        warmed = set()
        for _ in range(passes):
            for prefix in range(n):
                warmed.add((prefix, cursor % 2))
                cursor += 1
        return sum((i, (cursor + i) % 2) not in warmed for i in range(n))

    for n in WORKING_SETS:
        for start in (0, 1):
            assert cold_first_pass(n, 1, start) == n, n
            assert cold_first_pass(n, 2, start) == 0, n


def test_the_second_arm_at_one_working_set_hits_the_first_arms_bodies():
    """Without a reset the second arm at one N hits the first arm's identical requests."""
    from harness import run_level
    from stats import SLOTargets as Targets

    workload = Workload(name="w", prompt_tokens=256, output_tokens=3,
                        num_prompts=8, mode="closed", concurrency=2,
                        hit_rate_target=0.75, num_prefixes=2)

    async def twice(server, ep, reset):
        for _ in range(2):
            stats, _, _ = await run_level(workload, ep, Targets(), L40S_RUN1,
                                          settle=0.0, warm=True, max_in_flight=None,
                                          reset=reset)
        return stats.extra["measured_hit_rate"]

    stale = with_server(lambda s, ep: twice(s, ep, False))
    clean = with_server(lambda s, ep: twice(s, ep, True))
    assert stale > workload.nominal_hit_rate + 0.05, stale
    assert abs(clean - workload.nominal_hit_rate) < 0.05, clean


def test_a_reset_the_engine_does_not_serve_is_raised_not_skipped():
    """A 404 from /reset_prefix_cache (no VLLM_SERVER_DEV_MODE=1) stops the level instead of running it warm."""
    import urllib.error
    from harness import run_level
    from stats import SLOTargets as Targets

    workload = Workload(name="w", prompt_tokens=64, output_tokens=3,
                        num_prompts=2, mode="closed", concurrency=1)

    async def body(server, ep):
        await run_level(workload, ep, Targets(), L40S_RUN1, settle=0.0,
                        warm=False, max_in_flight=None, reset=True)

    try:
        with_server(body, dev_mode=False)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
        return
    raise AssertionError("a reset answered 404 and the level ran anyway")


def test_the_split_between_replicas_and_each_engines_h_are_recorded():
    """The done-criteria ask for h per engine and the request split; both land in extra."""
    from harness import check_split, per_engine_hit_rates
    from stats import SLOTargets as Targets, summarize
    from vllm_metrics import Snapshot

    def rec(i, upstream):
        r = Record(index=i, prompt_tokens=4, scheduled=0.0)
        r.sent, r.ttft, r.upstream = 0.0, 0.01, upstream
        r.latency, r.itls, r.output_tokens = 0.05, [0.01, 0.01], 3
        return r

    blank = summarize("w", "closed", [], 1.0, Targets())
    even = check_split(blank, [rec(i, f"e{i % 2}") for i in range(10)])
    assert even.extra["upstream_e0"] == 5.0 and not even.warnings, even
    skewed = check_split(blank, [rec(i, "e0" if i < 7 else "e1") for i in range(10)])
    assert any("70%" in w for w in skewed.warnings), skewed.warnings
    assert check_split(blank, [rec(0, None)]) is blank

    def snap(hits, queries):
        return Snapshot(values={"vllm:prefix_cache_hits": hits,
                                "vllm:prefix_cache_queries": queries}, at=0.0)

    pair = (("127.0.0.1", 8000), ("127.0.0.1", 8001))
    rates = per_engine_hit_rates(pair, (snap(0, 0), snap(0, 0)),
                                 (snap(800, 1000), snap(0, 1000)))
    assert rates == {"hit_rate_127.0.0.1:8000": 0.8, "hit_rate_127.0.0.1:8001": 0.0}
    assert per_engine_hit_rates(pair[:1], (snap(0, 0),), (snap(1, 1),)) == {}


TESTS = [value for name, value in sorted(globals().items())
         if name.startswith("test_") and callable(value)]


def serve_forever(port: int, ttft: float, itl: float,
                  fail_after: int | None, truncate: bool) -> None:
    """Run the fake vLLM as a standalone server, so the real CLI can face it.

    The tests drive the harness in-process, which proves the pieces; this proves
    the *program* -- argument parsing, the gates printing, the artefacts landing
    on disk -- against something that answers like vLLM and costs nothing. It is
    also how a gate is demonstrated rather than asserted: a fake that answers in
    a millisecond trips the prefill-floor check, which is exactly what a workload
    repeating whole prompts would do on a real card.

        python3 bench/tests/test_harness.py --serve --port 8123
        python3 bench/harness.py --scenario smoke --port 8123 --model fake

    What it cannot do is stand in for a measurement: it has no scheduler, no
    batching and no memory, so every latency it produces is the sleep it was
    told to take.
    """
    async def run() -> None:
        server = FakeVLLM(ttft=ttft, itl=itl, fail_after=fail_after,
                          truncate=truncate)
        server.server = await asyncio.start_server(server._handle, "127.0.0.1", port)
        server.port = port
        print(f"fake vLLM on 127.0.0.1:{port} -- ttft {ttft * 1e3:.0f} ms, "
              f"itl {itl * 1e3:.0f} ms; ^C to stop", flush=True)
        async with server.server:
            await server.server.serve_forever()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="the harness's offline tests")
    parser.add_argument("--serve", action="store_true",
                        help="run the fake vLLM instead of the tests")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--ttft-ms", type=float, default=120.0)
    parser.add_argument("--itl-ms", type=float, default=40.0)
    parser.add_argument("--fail-after", type=int, default=None,
                        help="answer 503 after this many requests")
    parser.add_argument("--truncate", action="store_true",
                        help="end every stream at half the tokens asked for")
    args = parser.parse_args()
    if args.serve:
        serve_forever(args.port, args.ttft_ms / 1e3, args.itl_ms / 1e3,
                      args.fail_after, args.truncate)
        return 0

    failures = []
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failures.append((test.__name__, str(exc) or "assertion failed"))
        except Exception as exc:                      # noqa: BLE001 -- a broken
            failures.append((test.__name__,           # test is a failing test
                             f"{type(exc).__name__}: {exc}"))
    for name, message in failures:
        print(f"FAIL  {name}\n      {message}")
    print(f"{len(TESTS) - len(failures)}/{len(TESTS)} harness tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
