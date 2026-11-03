"""Device passports: the datasheet, and nothing a run fitted (GLOSSARY.md, *Device passport*).

Dense rows only: where a vendor prints only the sparsity row, `note` says how
the dense figure was obtained, or the field is None.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passport:
    name: str
    memory_bytes: float           # as printed: vendor GB x 1e9
    peak_bandwidth: float         # bytes/s
    dense_bf16_flops: float | None
    dense_fp8_flops: float | None
    source: str
    quote: str                    # the vendor's words the figures were read from
    retrieved: str
    note: str = ""


PASSPORTS = {
    "dgx-spark": Passport(
        name="NVIDIA DGX Spark (GB10)",
        memory_bytes=128e9, peak_bandwidth=273e9,
        dense_bf16_flops=None, dense_fp8_flops=None,
        source="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
        quote="Tensor Performance Up to 1 PFLOP FP4; 128 GB LPDDR5x; Memory Bandwidth 273 GB/s",
        retrieved="2026-10-17",
        note="no BF16 or FP8 figure published; only FP4 with sparsity",
    ),
    "rtx-5090": Passport(
        name="NVIDIA GeForce RTX 5090",
        memory_bytes=32e9, peak_bandwidth=1792e9,
        dense_bf16_flops=209.5e12, dense_fp8_flops=419e12,
        source="https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/"
               "nvidia-rtx-blackwell-gpu-architecture.pdf",
        quote="Peak BF16 Tensor TFLOPS with FP32 Accumulate 209.5/419; 32 GB GDDR7; "
              "Memory Bandwidth 1792 GB/sec",
        retrieved="2026-10-17",
        note="FLOPS at FP32 accumulate",
    ),
    "m4-max-40c": Passport(
        name="Apple M4 Max (16-core CPU / 40-core GPU)",
        memory_bytes=128e9, peak_bandwidth=546e9,
        dense_bf16_flops=None, dense_fp8_flops=None,
        source="https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/",
        quote="M4 Max supports up to 128GB of fast unified memory and up to 546GB/s of "
              "memory bandwidth",
        retrieved="2026-10-17",
        note="Apple publishes no GPU FLOPS",
    ),
    "strix-halo": Passport(
        name="AMD Ryzen AI Max+ 395 (Strix Halo, Radeon 8060S)",
        memory_bytes=128e9, peak_bandwidth=256e9,
        dense_bf16_flops=60e12, dense_fp8_flops=None,
        source="https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/"
               "amd-ryzen-ai-max-plus-395.html",
        quote="256-bit LPDDR5x; Max. Memory 128 GB; LPDDR5x-8000; FP16 AI Compute Up to 60 TFLOPS",
        retrieved="2026-10-17",
        note="bandwidth not printed by AMD: 256 bit x 8000 MT/s / 8 = 256 GB/s, from the "
             "printed interface",
    ),
    "jetson-agx-orin-64gb": Passport(
        name="NVIDIA Jetson AGX Orin 64GB",
        memory_bytes=64e9, peak_bandwidth=204.8e9,
        dense_bf16_flops=43e12, dense_fp8_flops=None,
        source="https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/",
        quote="GPU Tensor Core FP16 Performance 85 SPARSE FP16 TFLOPs | 43 DENSE FP16 TFLOPs; "
              "64GB LPDDR5 204.8GB/s",
        retrieved="2026-10-17",
        note="the FP16 row; BF16 not printed separately",
    ),
    "jetson-agx-thor": Passport(
        name="NVIDIA Jetson AGX Thor (T5000, 128GB)",
        memory_bytes=128e9, peak_bandwidth=273e9,
        dense_bf16_flops=258e12, dense_fp8_flops=517e12,
        source="https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/",
        quote="FP8 Dense 517 TFLOPS; FP16 Dense 258 TFLOPS; up to 128GB of DRAM with 273 GB/s",
        retrieved="2026-10-17",
        note="the FP16 row; BF16 not printed separately",
    ),
    "h100-sxm": Passport(
        name="NVIDIA H100 SXM5 80GB",
        memory_bytes=80e9, peak_bandwidth=3.35e12,
        dense_bf16_flops=989.5e12, dense_fp8_flops=1979e12,
        source="https://www.nvidia.com/en-us/data-center/h100/",
        quote="BFLOAT16 Tensor Core* 1,979 teraFLOPS; FP8 Tensor Core* 3,958 teraFLOPS; "
              "3.35TB/s; * With sparsity",
        retrieved="2026-10-17",
        note="the page prints the sparsity row only; dense is half of it, per its footnote",
    ),
}
