"""Load generation against an OpenAI-compatible server: arrivals, SSE, timings.

The I/O half of the load harness: it puts requests on a socket and timestamps
what comes back, and knows nothing about what the numbers mean. Aggregation is
bench/stats.py, the engine's counters bench/vllm_metrics.py, what to run
bench/scenarios/. Three decisions this module made: standard library only, so
it runs inside the vllm-openai image without adding a dependency; prompts are
sent as token IDs, so prompt length and a shared prefix are exact by
construction (docs/GLOSSARY.md); no retries, because a retry would be a second
arrival and deform the arrival process.
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Endpoint:
    """Where the server is, and how patient the client is with it.

    timeout is per request and large on purpose: a request queued behind 40
    others is late, which is a measurement, not an error.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    path: str = "/v1/completions"
    model: str = "Qwen/Qwen3-8B"
    api_key: str | None = None
    timeout: float = 600.0


@dataclass(frozen=True)
class Request:
    """One prompt, already tokenised, with the shape that produced it.

    prefix_tokens is carried so the nominal hit rate can be stated without
    re-inspecting the prompt; the engine's measured h is scored against it.
    """

    index: int
    prompt_ids: tuple[int, ...]
    max_tokens: int
    prefix_tokens: int

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_ids)


@dataclass
class Record:
    """One request's timings, as the client saw them.

    Seconds from time.perf_counter, converted for display only. `sent` is when
    the bytes reached the socket; the connection is opened before it, so TTFT
    excludes TCP setup and stays comparable to `vllm bench serve`.
    `scheduled - sent` is the harness's own lateness, recorded rather than
    asserted away: an open loop that cannot keep up silently becomes a closed
    one (docs/GLOSSARY.md), and only a measurement shows it.
    """

    index: int
    prompt_tokens: int
    scheduled: float
    sent: float = 0.0
    ttft: float | None = None
    itls: list[float] = field(default_factory=list)
    latency: float | None = None
    output_tokens: int = 0
    usage_output_tokens: int | None = None
    error: str | None = None
    # X-Router-Policy, if something in front of the engine set it (docs/GLOSSARY.md).
    # None is the ordinary case of loading an engine directly, not a failed read.
    policy: str | None = None
    # X-Router-Upstream: which engine answered, the only per-request record of
    # the split between replicas (docs/benchmarks/runsheets/mi300x-run-3.md §4).
    upstream: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.ttft is not None

    @property
    def lateness(self) -> float:
        return self.sent - self.scheduled

    @property
    def tpot(self) -> float | None:
        """Mean ITL of this request, the SLO quantity (docs/GLOSSARY.md).

        None for a single-token response: there is no gap to average, and 0 or
        the TTFT would be a lie that percentiles smooth into plausibility.
        """
        if not self.ok or self.latency is None or self.output_tokens < 2:
            return None
        return (self.latency - self.ttft) / (self.output_tokens - 1)


class _ProtocolError(Exception):
    """The server said something that is not HTTP/1.1 + SSE as expected."""


async def _read_headers(reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
    status_line = await reader.readline()
    if not status_line:
        raise _ProtocolError("connection closed before a status line")
    parts = status_line.decode("latin-1").split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise _ProtocolError(f"bad status line: {status_line!r}")
    status = int(parts[1])

    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        name, _, value = line.decode("latin-1").partition(":")
        headers[name.strip().lower()] = value.strip()
    return status, headers


async def _iter_body(reader: asyncio.StreamReader, headers: dict[str, str]):
    """Yield raw body bytes as they arrive, unwrapping chunked framing.

    vLLM streams SSE chunked; the Content-Length branch serves the test server
    and error responses.
    """
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = await reader.readline()
            if not size_line:
                raise _ProtocolError("connection closed mid-chunk")
            size = int(size_line.strip().split(b";")[0] or b"0", 16)
            if size == 0:
                await reader.readline()          # the trailing CRLF
                return
            yield await reader.readexactly(size)
            await reader.readexactly(2)          # the CRLF after the chunk
    elif "content-length" in headers:
        remaining = int(headers["content-length"])
        while remaining > 0:
            data = await reader.read(min(65536, remaining))
            if not data:
                return
            remaining -= len(data)
            yield data
    else:
        while True:
            data = await reader.read(65536)
            if not data:
                return
            yield data


def _build_http_request(ep: Endpoint, req: Request) -> bytes:
    """The wire format, assembled by hand because there is no client library.

    ignore_eos makes output length an input (docs/GLOSSARY.md); temperature 0
    because a run comparing configurations should not also be sampling.
    """
    body = json.dumps({
        "model": ep.model,
        "prompt": list(req.prompt_ids),
        "max_tokens": req.max_tokens,
        "min_tokens": req.max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    lines = [
        f"POST {ep.path} HTTP/1.1",
        f"Host: {ep.host}:{ep.port}",
        "Content-Type: application/json",
        "Accept: text/event-stream",
        # One connection per request: keep-alive would hide a stalled socket.
        "Connection: close",
        f"Content-Length: {len(body)}",
    ]
    if ep.api_key:
        lines.append(f"Authorization: Bearer {ep.api_key}")
    head = ("\r\n".join(lines) + "\r\n\r\n").encode()
    return head + body


async def send_one(ep: Endpoint, req: Request, scheduled: float) -> Record:
    """Send one request, timestamping every token event that comes back.

    One SSE chunk can carry several events; they share a timestamp and one ITL
    reads zero, which is why the median ITL is the decode-step proxy
    (docs/GLOSSARY.md).
    """
    rec = Record(index=req.index, prompt_tokens=req.prompt_tokens, scheduled=scheduled)
    writer = None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ep.host, ep.port), timeout=ep.timeout)

        rec.sent = time.perf_counter()
        writer.write(_build_http_request(ep, req))
        await writer.drain()

        status, headers = await asyncio.wait_for(_read_headers(reader), ep.timeout)
        rec.policy = headers.get("x-router-policy")
        rec.upstream = headers.get("x-router-upstream")
        if status != 200:
            detail = b"".join([c async for c in _iter_body(reader, headers)])
            rec.error = f"HTTP {status}: {detail[:200].decode('utf-8', 'replace')}"
            return rec

        last = rec.sent
        buffer = b""
        async for chunk in _iter_body(reader, headers):
            now = time.perf_counter()
            buffer += chunk
            while b"\n" in buffer:
                line, _, buffer = buffer.partition(b"\n")
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[len(b"data:"):].strip()
                if payload == b"[DONE]":
                    continue
                event = json.loads(payload)

                usage = event.get("usage")
                if usage:
                    rec.usage_output_tokens = usage.get("completion_tokens")

                choices = event.get("choices") or []
                if not choices or not choices[0].get("text"):
                    # A usage-only or finish frame is not a token, so not an ITL sample.
                    continue

                if rec.ttft is None:
                    rec.ttft = now - rec.sent
                else:
                    rec.itls.append(now - last)
                last = now
                rec.output_tokens += 1

        if rec.ttft is None:
            rec.error = "stream ended before any token"
        else:
            rec.latency = last - rec.sent
        return rec

    except asyncio.TimeoutError:
        rec.error = f"timeout after {ep.timeout:.0f} s"
        return rec
    except (OSError, _ProtocolError, json.JSONDecodeError) as exc:
        rec.error = f"{type(exc).__name__}: {exc}"
        return rec
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass


def poisson_offsets(rate: float, count: int, seed: int) -> list[float]:
    """Arrival offsets in seconds for a Poisson process of the given rate.

    Exponential gaps: constant intervals never produce the coincident arrivals
    that build a queue (docs/benchmarks/l40s-run2.md section 3). Seeded, so a
    level can be re-faced with a changed configuration.
    """
    if rate <= 0:
        raise ValueError("a Poisson arrival rate must be positive")
    rng = random.Random(seed)
    offsets, t = [], 0.0
    for _ in range(count):
        offsets.append(t)
        t += rng.expovariate(rate)
    return offsets


async def run_open_loop(
        ep: Endpoint, requests: list[Request], rate: float,
        seed: int = 0, max_in_flight: int | None = None) -> tuple[list[Record], float]:
    """Fire requests at wall-clock arrival times, regardless of what is finished.

    The only mode in which TTFT is the service's (docs/SLO.md section 9).
    max_in_flight is a safety valve, not a load parameter: once it engages the
    loop is no longer open, so the deferred count is reported, not absorbed.
    """
    offsets = poisson_offsets(rate, len(requests), seed)
    gate = asyncio.Semaphore(max_in_flight) if max_in_flight else None
    records: list[Record] = []
    start = time.perf_counter()

    async def one(req: Request, offset: float) -> None:
        scheduled = start + offset
        delay = scheduled - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        if gate is not None:
            async with gate:
                records.append(await send_one(ep, req, scheduled))
        else:
            records.append(await send_one(ep, req, scheduled))

    await asyncio.gather(*(one(r, o) for r, o in zip(requests, offsets)))
    duration = time.perf_counter() - start
    records.sort(key=lambda r: r.index)
    return records, duration


async def run_closed_loop(
        ep: Endpoint, requests: list[Request],
        concurrency: int) -> tuple[list[Record], float]:
    """Hold exactly `concurrency` requests in flight until the list is exhausted.

    The mode that measures the engine: batch size is the independent variable.
    Its TTFT is the generator's own backlog, not a service metric, and
    bench/stats.py carries that flag so a table cannot print it against a target.
    """
    if concurrency < 1:
        raise ValueError("closed-loop concurrency must be at least 1")
    queue: asyncio.Queue[Request] = asyncio.Queue()
    for req in requests:
        queue.put_nowait(req)
    records: list[Record] = []
    start = time.perf_counter()

    async def worker() -> None:
        while True:
            try:
                req = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            # scheduled == sent: in a closed loop the arrival is the previous completion.
            records.append(await send_one(ep, req, time.perf_counter()))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    duration = time.perf_counter() - start
    records.sort(key=lambda r: r.index)
    return records, duration
