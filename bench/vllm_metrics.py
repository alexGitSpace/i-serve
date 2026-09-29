"""The engine's own numbers: /metrics counters around a level, and the startup log.

h is a ratio of two token counters over a window (docs/GLOSSARY.md), taken per
level, never per session (docs/SLO.md section 10); the logged KV pool outranks
every derivation (docs/SLO.md section 9). What each fact means for a level is
bench/harness.py.
"""

import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# Everything else vLLM exports is dropped: a snapshot is written into the
# run's artefacts, and a hundred series per level is not readable evidence.
COUNTERS = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:num_preemptions",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:request_success",
)
GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:gpu_prefix_cache_hit_rate",   # V0 only; absent on V1, kept as a probe
)

_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+"
                     r"(?P<value>[-+0-9.eEnaN]+)\s*$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum every series of a metric into one number per metric name.

    One server serves one model, so the sum across label sets is the series;
    a histogram's _bucket lines would not survive it, so none is in COUNTERS or
    GAUGES. A _total suffix is folded onto the base name, as the docs spell it.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name.endswith("_total"):
            name = name[: -len("_total")]
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + value
    return out


@dataclass(frozen=True)
class Snapshot:
    """Everything /metrics said at one instant, plus when that instant was."""

    at: float
    values: dict[str, float]

    def get(self, name: str) -> float | None:
        return self.values.get(name)


def scrape(host: str = "127.0.0.1", port: int = 8000,
           timeout: float = 5.0) -> Snapshot:
    """One blocking GET /metrics, parsed.

    Never on the event loop: a blocked loop cannot timestamp an arriving token,
    and the stall lands in the ITL distribution.
    """
    url = f"http://{host}:{port}/metrics"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        text = response.read().decode("utf-8", "replace")
    return Snapshot(at=time.perf_counter(), values=parse_prometheus(text))


def reset_prefix_cache(host: str = "127.0.0.1", port: int = 8000,
                       attempts: int = 10, pause: float = 1.0,
                       timeout: float = 5.0) -> None:
    """POST /reset_prefix_cache, retried while a request still holds blocks.

    Raises on a 404 (no VLLM_SERVER_DEV_MODE=1) and when every attempt is
    refused. Why a level needs it: docs/GLOSSARY.md, --reset-cache.
    """
    url = f"http://{host}:{port}/reset_prefix_cache"
    for _ in range(attempts):
        request = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if b'"success":true' in response.read().replace(b" ", b""):
                return
        time.sleep(pause)
    raise RuntimeError(f"{url}: still refused after {attempts} attempts -- "
                       f"a request is holding blocks")


def delta(before: Snapshot, after: Snapshot) -> dict[str, float]:
    """Counter increments across a window. Gauges are taken from `after`.

    A counter that went down means the engine restarted mid-level; raised, not
    clamped, because every cross-level comparison assumes one engine process.
    """
    out: dict[str, float] = {}
    for name in COUNTERS:
        start, end = before.values.get(name), after.values.get(name)
        if start is None or end is None:
            continue
        if end < start:
            raise ValueError(
                f"{name} fell from {start:.0f} to {end:.0f}: the engine restarted "
                f"during this level, so no counter increment across it is meaningful"
            )
        out[name] = end - start
    for name in GAUGES:
        if name in after.values:
            out[name] = after.values[name]
    return out


def scrape_fleet(endpoints: tuple[tuple[str, int], ...]) -> tuple[Snapshot, ...]:
    """One GET /metrics per engine, in order, never through a router; raises if refused.

    Why not the router: docs/benchmarks/runsheets/mi300x-run-3.md section 0."""
    return tuple(scrape(host, port) for host, port in endpoints)


def delta_fleet(before: tuple[Snapshot, ...],
                after: tuple[Snapshot, ...]) -> dict[str, float]:
    """The fleet's increments: each engine's delta, summed per counter.

    Summed before dividing, so hit_rate() is token-weighted, not a mean of ratios.
    """
    if len(before) != len(after):
        raise ValueError(f"{len(before)} snapshots before, {len(after)} after")
    totals: dict[str, float] = {}
    for b, a in zip(before, after):
        for name, value in delta(b, a).items():
            totals[name] = totals.get(name, 0.0) + value
    return totals


def hit_rate(increments: dict[str, float]) -> float | None:
    """`h` over the window: hit tokens divided by queried tokens.

    None, not 0.0, when nothing was queried: "the cache served none of it" and
    "prefix caching is off, so the counters do not exist" are different findings.
    """
    queries = increments.get("vllm:prefix_cache_queries")
    hits = increments.get("vllm:prefix_cache_hits")
    if not queries:
        return None
    return hits / queries if hits is not None else None


# --- the startup log ---------------------------------------------------------
# Wording from docs/benchmarks/raw/l40s-2026-08-23/startup-lines.txt and
# mi300x-2026-09-29/run3/engine-8000.log; a renamed line fails one regex here.

_LOG_PATTERNS = {
    "kv_cache_tokens": re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.I | re.S),
    "kv_cache_gib": re.compile(r"kv cache memory in use is\s*([\d.]+)\s*GiB", re.I),
    "max_concurrency": re.compile(r"Maximum concurrency[\s\S]*?([\d.]+)x", re.I),
    # The quote is how "non-default args" prints the key in a raw log.
    "max_num_batched_tokens": re.compile(r"max_num_batched_tokens'?[=:]\s*([\d]+)"),
    "max_num_seqs": re.compile(r"max_num_seqs[=:]\s*([\d]+)"),
    "enable_prefix_caching": re.compile(r"enable_prefix_caching[=:]\s*'?(True|False)'?"),
    "kv_cache_dtype": re.compile(r"kv_cache_dtype[=:]\s*'?([\w]+)'?"),
    # CUDA announces its choice; ROCm reports it as an override (MI300X runs 1-3).
    "attention_backend": re.compile(
        r"Using (\w+) attention|Overriding with (\w+) out of potential backends", re.I),
}


def read_startup_log(path: str) -> dict[str, float | str]:
    """Pull the facts a run is gated on out of a vLLM startup log.

    Each can move between launches or under one flag
    (docs/benchmarks/l40s-run2.md sections 5-6).
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    found: dict[str, float | str] = {}
    for key, pattern in _LOG_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        raw = next(group for group in match.groups() if group is not None)
        if key in ("enable_prefix_caching", "kv_cache_dtype", "attention_backend"):
            found[key] = raw
        else:
            found[key] = float(raw.replace(",", ""))
    return found


def unread_startup_facts(facts: dict[str, float | str]) -> tuple[str, ...]:
    """Gated facts the log did not yield: a renamed line would remove a gate in silence.

    docs/benchmarks/runsheets/l40s-run-2.md, postscript 3-4."""
    return tuple(key for key in _LOG_PATTERNS if key not in facts)


def pool_gate(logged_tokens: float, reference_tokens: float,
              tolerance: float = 0.05) -> tuple[bool, str]:
    """Is this pod's KV pool the same one the reference run measured?

    5 % sits just above one config's launch-to-launch noise
    (docs/benchmarks/l40s-run2.md section 6).
    """
    ratio = logged_tokens / reference_tokens
    ok = abs(ratio - 1.0) <= tolerance
    verdict = (
        f"KV pool {logged_tokens:,.0f} tokens against reference {reference_tokens:,.0f} "
        f"({ratio:.3f}x, gate +-{tolerance:.0%}): {'PASS' if ok else 'FAIL'}"
    )
    return ok, verdict
