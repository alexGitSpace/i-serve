"""Public claims, checked on paper against the ceiling and kept out of the ledger.

A claim is a figure someone else published; this repository did not measure it,
so it never becomes a ledger row (CONTRIBUTING.md). Each is checked with the
arithmetic the ledger uses: bytes per step from a weight file's header
(bench/models/weight_files.json), bandwidth from a passport (bench/passports.py).
The verdicts are GLOSSARY.md's, *Claims table*.

    python3 bench/claims.py        # docs/claims.md
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from passports import PASSPORTS
from roofline import FLOPS_PER_MAC, MI300X, kv_bytes_per_token, weights_of

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "claims.md"
FILES = json.loads((ROOT / "bench" / "models" / "weight_files.json").read_text(encoding="utf-8"))
COLLECTED = "2026-10-17"      # every claim and quote below was read on this day

VERDICTS = ("within ceiling", "over ceiling", "above floor", "below floor",
            "condition not disclosed", "out of regime", "not checked")


def read_bytes(key: str) -> int:
    """One decode step's weight read: roofline.Weights.read, every role but the embedding table."""
    return int(weights_of(FILES[key]["roles"]).read(1.0))


def resident_bytes(key: str) -> int:
    """What the weights occupy: every tensor, the embedding included."""
    return int(weights_of(FILES[key]["roles"]).resident)


def kv_per_token(key: str, dtype_bytes: int) -> int:
    a = FILES[key]["attention"]
    return kv_bytes_per_token(a["layers"], a["kv_heads"], a["head_dim"], dtype_bytes)


def flops_per_token(key: str) -> int:
    """The compute side: every weight but the embedding is multiplied once per token."""
    roles = FILES[key]["roles"]
    return FLOPS_PER_MAC * (roles["decoder"]["weights"] + roles["lm_head"]["weights"])


@dataclass(frozen=True)
class Source:
    claimant: str
    url: str
    published: str


@dataclass(frozen=True)
class Stream:
    """A single-stream decode rate on a named weight file: share at zero context, a lower bound."""
    id: str
    source: Source
    device: str
    engine: str
    model: str
    claimed: float                # tok/s
    file: str                     # weight_files.json key; the page names the file or the precision
    stand_in: bool = False        # the precision is published, the file is not: a file of it stands in

    def ceiling(self) -> float:
        return PASSPORTS[self.device].peak_bandwidth / read_bytes(self.file)

    def share(self) -> float:
        return self.claimed / self.ceiling()

    def verdict(self) -> str:
        return "within ceiling" if self.share() <= 1 else "over ceiling"


@dataclass(frozen=True)
class Undisclosed:
    """A single-stream rate whose precision is not published: the bytes per step it allows."""
    id: str
    source: Source
    device: str
    engine: str
    model: str
    claimed: float
    candidates: tuple[str, ...]   # files of the model at the precisions it could be

    def step_at_most(self) -> float:
        return PASSPORTS[self.device].peak_bandwidth / self.claimed

    def fits(self) -> list[str]:
        return [f for f in self.candidates if read_bytes(f) <= self.step_at_most()]

    def verdict(self) -> str:
        return "condition not disclosed"


@dataclass(frozen=True)
class Checked:
    """A claim whose check is its own arithmetic: a batch, a capacity or a price."""
    id: str
    source: Source
    claim: str
    check: str
    verdict: str


@dataclass(frozen=True)
class Refused:
    id: str
    source: Source
    claim: str
    why: str
    verdict: str                  # "out of regime" or "not checked"


OLLAMA_SPARK = Source("Ollama (engine vendor)", "https://ollama.com/blog/nvidia-spark-performance",
                      "2025-10-23")
LMSYS_SPARK = Source("LMSYS (SGLang team)", "https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/",
                     "2025-10-13")
ATOMIC = Source("Atomic Chat blog (reviewer)", "https://atomic.chat/blog/guides/ollama-vs-llamacpp",
                "2026-06-26")
LLAMACPP_4167 = Source("llama.cpp maintainers' table", "https://github.com/ggml-org/llama.cpp/discussions/4167",
                       "maintained; setup updated 2026-08-25")
LLM_TRACKER = Source("llm-tracker (community)",
                     "https://llm-tracker.info/AMD-Strix-Halo-(Ryzen-AI-Max+-395)-GPU-Performance",
                     "2025-05-17")
EDGE_LLM = Source("NVIDIA (hardware vendor)",
                  "https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/"
                  "performance-benchmarks.html",
                  "undated; v0.10.0 results")
SPARK_TRTLLM = Source("NVIDIA (hardware vendor)",
                      "https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-"
                      "intensive-ai-tasks", "2025-10-24")
JETSON = Source("NVIDIA (hardware vendor)", "https://developer.nvidia.com/embedded/jetson-benchmarks",
                "undated")
INFERENCEX = Source("SemiAnalysis InferenceX (analyst)",
                    "https://inferencex.semianalysis.com/run/llama-3-3-70b-on-mi300x", "run 2025-10-29")
SPHERON = Source("Spheron (GPU cloud)",
                 "https://www.spheron.network/blog/how-many-concurrent-users-can-one-h100-serve-a-kv-cache-sizi/",
                 "undated")
DEEPINFRA_70B = Source("DeepInfra (API provider)",
                       "https://deepinfra.com/meta-llama/Llama-3.3-70B-Instruct-Turbo", "live page")
TOGETHER_70B = Source("Together (API provider)", "https://www.together.ai/models/llama-3-3-70b", "live page")
DEEPINFRA_32B = Source("DeepInfra (API provider)", "https://deepinfra.com/Qwen/Qwen3-32B", "live page")
MLPERF = Source("NVIDIA (MLPerf Inference v6.1)",
                "https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference",
                "undated")
GROQ = Source("Groq (API provider)", "https://console.groq.com/docs/models", "live page")
CEREBRAS = Source("Cerebras (hardware vendor)",
                  "https://www.cerebras.ai/news/hugging-face-partners-with-cerebras-to-give-developers-"
                  "access-to-industry-s-fastest-ai-inference", "2025-03-11")

# Each row's words on its page, verbatim; a table row's cells joined by " ¦ ".
# A row read off the same table as another says so instead of repeating it.
QUOTES = {
    "c01": "NVIDIA DGX Spark ¦ llama3.1 ¦ 8B ¦ q4_K_M ¦ 7.614k ¦ 38.02",
    "c02": "With SGLang at batch 1, it achieved 7,991 tps prefill / 20.5 tps decode",
    "c03": "llama.cpp: 249 tok/s",
    "c03o": "Ollama: 108 tok/s (re-read 2026-11-03: \"one GGUF file for both tools\")",
    "c04": "M4 Max ¦ 546 ¦ 40 ¦ 31.64 ¦ 54.05 ¦ 83.06",
    "c04q": "the c04 row, its Q4_0 cell",
    "c05": "Vulkan ¦ 881.71 ± 1.71 ¦ 52.22 ± 0.05",
    "c06": "Qwen3-8B ¦ Vanilla ¦ INT4 AWQ ¦ 1 ¦ 61 ¦ 32.2",
    "c07": "MI300X sustains 1,482 tokens/s per GPU serving Llama 3.3 70B with vLLM in FP8",
    "c08": "each user's KV cache costs ~0.52GB (0.13MB x 4,096) ... 56GB / 0.52GB ≈ 107 concurrent users",
    "c09": "$0.10 ... $0.32 ... fp8",
    "c10": "$1.04 / 1M tokens",
    "c11": "Offline ¦ 171,114 tokens/sec ¦ 8x B300",
    "c12": "llama-3.1-8b-instant ... 560 (T/SEC)",
    "c13": "NVIDIA DGX Spark ¦ gpt-oss ¦ 120B ¦ MXFP4 ¦ 1.169k ¦ 41.14",
    "c14": "NVIDIA DGX Spark ¦ gemma3 ¦ 27B ¦ q4_K_M ¦ 834.1 ¦ 10.83",
    "c15": "Qwen3-8B ¦ EAGLE3 ¦ NVFP4 ¦ 1 ¦ 61 ¦ 153.7 ¦ 4.16",
    "c15v": "Qwen3-8B LLM Vanilla NVFP4 1 61 27.7 2,208.9 - - - 47.3 (re-read 2026-11-03)",
    "x01": "NVIDIA DGX Spark ¦ qwen3 ¦ 32B ¦ q4_K_M ¦ 705.0 ¦ 9.411",
    "x02": "scaling up linearly to 7,949 tps / 368 tps at batch 32",
    "x03": "Llama 3.1 8B ¦ NVFP4 ¦ TRT-LLM ¦ 10256.9 ¦ 38.65",
    "x04a": "Llama 3.3 70B ¦ 4.7 ¦ 12.6",
    "x04b": "the x04a table, its Llama 3.1 8B row",
    "x04c": "the x04a table, its Qwen 3 32B row",
    "x05": "$0.08 ... $0.28 ... / 1M tokens ... fp8",
    "x06": "Cerebras recently announced industry-leading speeds for Llama 3.3 70B, achieving over "
           "2,200 tokens per second",
}

STREAMS = (
    Stream("c01", OLLAMA_SPARK, "dgx-spark", "Ollama v0.12.6", "Llama-3.1-8B Q4_K_M", 38.02,
           "llama-3.1-8b-instruct.q4_k_m.ollama"),
    Stream("c02", LMSYS_SPARK, "dgx-spark", "SGLang", "Llama-3.1-8B FP8", 20.5,
           "llama-3.1-8b-instruct-fp8.nvidia", stand_in=True),
    Stream("c03", ATOMIC, "rtx-5090", "llama.cpp 3466812", "Llama-3.1-8B Q4_K_M", 249,
           "llama-3.1-8b-instruct.q4_k_m.lmstudio"),
    Stream("c03o", ATOMIC, "rtx-5090", "Ollama v0.33.2", "Llama-3.1-8B Q4_K_M", 108,
           "llama-3.1-8b-instruct.q4_k_m.lmstudio"),
    Stream("c04", LLAMACPP_4167, "m4-max-40c", "llama.cpp Metal", "Llama-2-7B F16", 31.64,
           "llama-2-7b.f16.ggml-org"),
    Stream("c04q", LLAMACPP_4167, "m4-max-40c", "llama.cpp Metal", "Llama-2-7B Q4_0", 83.06,
           "llama-2-7b.q4_0.ggml-org"),
    Stream("c05", LLM_TRACKER, "strix-halo", "llama.cpp Vulkan", "Llama-2-7B Q4_0", 52.22,
           "llama-2-7b.q4_0.thebloke"),
    Stream("c06", EDGE_LLM, "jetson-agx-orin-64gb", "TensorRT Edge-LLM 0.10.0", "Qwen3-8B INT4 AWQ", 32.2,
           "qwen3-8b-awq", stand_in=True),
    Stream("x01", OLLAMA_SPARK, "dgx-spark", "Ollama v0.12.6", "Qwen3-32B Q4_K_M", 9.411,
           "qwen3-32b.q4_k_m.ollama"),
    Stream("x03", SPARK_TRTLLM, "dgx-spark", "TensorRT-LLM", "Llama-3.1-8B NVFP4", 38.65,
           "llama-3.1-8b-instruct-nvfp4.nvidia", stand_in=True),
    Stream("c15v", EDGE_LLM, "jetson-agx-thor", "TensorRT Edge-LLM 0.10.0", "Qwen3-8B NVFP4", 47.3,
           "qwen3-8b-nvfp4.nvidia", stand_in=True),
)

UNDISCLOSED = (
    Undisclosed("x04a", JETSON, "jetson-agx-thor", "vLLM", "Llama-3.3-70B", 4.7,
                ("llama-3.3-70b-instruct.q4_k_m.lmstudio", "llama-3.3-70b-instruct-fp8.nvidia")),
    Undisclosed("x04b", JETSON, "jetson-agx-thor", "vLLM", "Llama-3.1-8B", 41.3,
                ("llama-3.1-8b-instruct.q4_k_m.lmstudio", "llama-3.1-8b-instruct-nvfp4.nvidia",
                 "llama-3.1-8b-instruct-fp8.nvidia")),
    Undisclosed("x04c", JETSON, "jetson-agx-thor", "vLLM", "Qwen3-32B", 13.19,
                ("qwen3-32b.q4_k_m.ollama", "qwen3-32b-fp8")),
)


def lmsys_batch() -> Checked:
    """368 tok/s at batch 32: the context it allows, at either KV width."""
    f, batch, claimed = "llama-3.1-8b-instruct-fp8.nvidia", 32, 368
    bandwidth = PASSPORTS["dgx-spark"].peak_bandwidth
    at_zero = claimed / (batch * bandwidth / read_bytes(f))
    most = {b: (batch * bandwidth / claimed - read_bytes(f)) / (batch * kv_per_token(f, b)) for b in (2, 1)}
    return Checked(
        "x02", LMSYS_SPARK, "DGX Spark, SGLang, Llama-3.1-8B FP8: 368 tok/s at batch 32",
        f"share at least {at_zero:.2f}, the share at zero context; the context is not published, "
        f"and the rate is possible only below {most[2]:,.0f} tokens per sequence with a BF16 KV cache, "
        f"{most[1]:,.0f} with FP8",
        "condition not disclosed")


def inferencex() -> Checked:
    """1,482 tok/s per GPU at 50 tok/s per user, ISL 8k / OSL 1k: which tokens, and on how many cards."""
    f, per_user, isl, osl = "llama-3.3-70b-instruct-fp8.nvidia", 50, 8192, 1024
    seat = kv_per_token(f, 2) * (isl + osl // 2)        # KV read per sequence at its mean context
    billed = 0.95 / 0.18e-6 / 3600                       # tok/s their own $0.18/1M at $0.95/h pays for

    def share(output_per_gpu: float, tp: int) -> float:
        seats = output_per_gpu / per_user * tp           # sequences across the TP group
        return (read_bytes(f) + seats * seat) / (tp * MI300X.peak_bandwidth) * per_user

    output = 1482 * osl / (isl + osl)
    return Checked(
        "c07", INFERENCEX, "MI300X, vLLM, Llama-3.3-70B FP8: 1,482 tok/s per GPU at 50 tok/s per user",
        f"their price bills {billed:,.0f} tok/s per GPU, so the figure counts input and output; "
        f"output is {output:.0f} tok/s, {output / per_user:.1f} sequences per GPU. Share {share(output, 1):.2f} "
        f"at TP 1, {share(output, 8):.2f} at TP 8 (BF16 KV); TP is not published. Read as output tokens, "
        f"the figure is over the ceiling at TP 1 ({share(1482, 1):.2f}) and TP 2 ({share(1482, 2):.2f})",
        "condition not disclosed")


def spheron() -> Checked:
    """107 users of Llama-3.1-8B BF16 at 4K on one H100: a capacity, checked as one."""
    f, context, gmu = "llama-3.1-8b-instruct.unsloth", 4096, 0.90
    seat = kv_per_token(f, 2) * context
    seats = {name: (mem * gmu - resident_bytes(f)) / seat for name, mem in (("GB", 80e9), ("GiB", 80 * 2**30))}
    return Checked(
        "c08", SPHERON, "H100, Llama-3.1-8B BF16: 107 concurrent users at 4K context",
        f"(80 GB × {gmu:.2f} − {resident_bytes(f) / 1e9:.2f} GB) / ({kv_per_token(f, 2):,} B × {context:,}) "
        f"= {seats['GB']:.0f} seats, {seats['GiB']:.0f} if the 80 GB is GiB. A capacity, not a service: "
        f"no TPOT target is set, and a target lowers it ([SLO.md §6](SLO.md#6-concurrency-ceiling))",
        "within ceiling")


def price(id: str, source: Source, claim: str, f: str, inp: float, out: float, hourly: float,
          rate_from: str) -> Checked:
    """Input priced at prefill on 100 % of an H100's dense FP8, at the claimant's own hourly rate."""
    tok_s = PASSPORTS["h100-sxm"].dense_fp8_flops / flops_per_token(f)
    floor = hourly / 3600 / tok_s * 1e6
    return Checked(
        id, source, claim,
        f"floor ${floor:.3f}/1M: {flops_per_token(f) / 1e9:.0f} GFLOP per token on 100 % of an H100's dense "
        f"FP8 at {rate_from} ${hourly:.2f}/h. Input {inp / floor:.1f}× the floor, output {out / floor:.1f}×",
        "above floor" if inp >= floor and out >= floor else "below floor")


CHECKED = (
    lmsys_batch(),
    inferencex(),
    spheron(),
    price("c09", DEEPINFRA_70B, "DeepInfra, Llama-3.3-70B FP8: $0.10 / $0.32 per 1M",
          "llama-3.3-70b-instruct.q4_k_m.lmstudio", 0.10, 0.32, 2.20, "DeepInfra's"),
    price("c10", TOGETHER_70B, "Together, Llama-3.3-70B FP8: $1.04 / $1.04 per 1M",
          "llama-3.3-70b-instruct.q4_k_m.lmstudio", 1.04, 1.04, 3.99, "Together's"),
    price("x05", DEEPINFRA_32B, "DeepInfra, Qwen3-32B FP8: $0.08 / $0.28 per 1M",
          "qwen3-32b.q4_k_m.ollama", 0.08, 0.28, 2.20, "DeepInfra's"),
)

REFUSED = (
    Refused("c15", EDGE_LLM, "Jetson AGX Thor, Qwen3-8B NVFP4 + EAGLE3: 153.7 tok/s",
            "speculative decoding accepts ~4 tokens per weight pass; against the one-pass ceiling it "
            "would read as a share of 2.90", "out of regime"),
    Refused("c13", OLLAMA_SPARK, "DGX Spark, Ollama, gpt-oss-120b MXFP4: 41.14 tok/s",
            "sliding-window attention on alternate layers, which the ceiling does not model", "out of regime"),
    Refused("c14", OLLAMA_SPARK, "DGX Spark, Ollama, Gemma 3 27B Q4_K_M: 10.83 tok/s",
            "sliding-window attention, which the ceiling does not model", "out of regime"),
    Refused("c12", GROQ, "Groq LPU, Llama-3.1-8B: 560 tok/s per user",
            "weights held in on-chip SRAM across many chips; no device-memory ceiling applies", "out of regime"),
    Refused("x06", CEREBRAS, "Cerebras CS-3, Llama-3.3-70B: 2,200 tok/s",
            "weights held in wafer-scale SRAM; no device-memory ceiling applies", "out of regime"),
    Refused("c11", MLPERF, "8× B300, Llama-3.1-8B: 171,114 tok/s, MLPerf Offline",
            "no B300 passport, and the dataset's token lengths are not on the page", "not checked"),
)


def fmt(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}"


def markdown() -> str:
    every = [*STREAMS, *UNDISCLOSED, *CHECKED, *REFUSED]
    for c in every:
        assert (c.verdict() if callable(c.verdict) else c.verdict) in VERDICTS, c.id
    assert sorted(QUOTES) == sorted(c.id for c in every), "a row without its quote, or a quote without its row"
    claims = {c.id[:3] for c in every}     # c03o, x04a: a second row of one published figure
    lines = [
        "# Claims, checked on paper",
        "",
        "<!-- Generated by bench/claims.py; edit that, never this. -->",
        "",
        f"Figures other people published, checked against the ceiling with the ledger's arithmetic "
        f"and kept out of the ledger: nothing here was measured in this repository "
        f"([CONTRIBUTING.md](../CONTRIBUTING.md)). {len(every)} rows from {len(claims)} published "
        f"figures, collected {COLLECTED}. Bytes per step are read from each weight file's header "
        f"(`bench/models/weight_files.json`), bandwidth from the device's passport "
        f"(`bench/passports.py`); the verdicts are [GLOSSARY.md](GLOSSARY.md)'s, *Claims table*.",
        "",
        "## Single stream",
        "",
        "The ceiling is at zero context, so each share is a lower bound: the KV a real context adds "
        "lowers the ceiling. *Stand-in* marks a page that publishes the precision but not the file.",
        "",
        "| | Device | Engine | Model | Weights read per step | Claimed, tok/s | Ceiling, tok/s | Share | Verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for s in sorted(STREAMS, key=lambda s: s.share()):
        weights = f"{read_bytes(s.file) / 1e9:.2f} GB" + (" (stand-in)" if s.stand_in else "")
        lines.append(f"| {s.id} | {PASSPORTS[s.device].name} | {s.engine} | {s.model} | {weights} | "
                     f"{fmt(s.claimed, 2)} | {fmt(s.ceiling())} | {s.share():.2f} | {s.verdict()} |")
    lines += [
        "",
        "## Precision not disclosed",
        "",
        "The page names no precision, so no share is computed; the claim bounds the bytes a step can "
        "read, and the files of that model are tested against the bound.",
        "",
        "| | Device | Engine | Model | Claimed, tok/s | Step reads at most | Files that fit | Verdict |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for u in UNDISCLOSED:
        fit = ", ".join(f"`{f}` ({read_bytes(f) / 1e9:.1f} GB)" for f in u.fits()) or "none"
        lines.append(f"| {u.id} | {PASSPORTS[u.device].name} | {u.engine} | {u.model} | {fmt(u.claimed, 2)} | "
                     f"{u.step_at_most() / 1e9:.1f} GB | {fit} | {u.verdict()} |")
    lines += [
        "", "## Seats and prices", "",
        "A price is checked against a floor: prefill at 100 % of the card's dense FP8, at the claimant's "
        "own hourly rate. For output the ratio is an upper bound on the margin, since one card does not "
        "batch decode to its compute limit.",
        "", "| | Claim | Check | Verdict |", "|---|---|---|---|"]
    lines += [f"| {c.id} | {c.claim} | {c.check} | {c.verdict} |" for c in CHECKED]
    lines += ["", "## Refused", "", "| | Claim | Why | Verdict |", "|---|---|---|---|"]
    lines += [f"| {r.id} | {r.claim} | {r.why} | {r.verdict} |" for r in REFUSED]
    lines += ["", "## Sources", "", "| | Claimant | Published | Quoted |", "|---|---|---|---|"]
    for c in sorted(every, key=lambda c: c.id):
        lines.append(f"| {c.id} | [{c.source.claimant}]({c.source.url}) | {c.source.published} | "
                     f"{QUOTES[c.id]} |")
    return "\n".join(lines) + "\n"


def outputs() -> dict[pathlib.Path, str]:
    """Every file this module writes, as path -> content. The test reads this."""
    return {OUT: markdown()}


def main() -> None:
    for path, text in outputs().items():
        path.write_text(text, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
