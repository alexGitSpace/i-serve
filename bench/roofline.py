"""Roofline floors for LLM inference: what the hardware can physically do.

Inputs are config.json and the vendor spec table; a floor is a ceiling on
performance, not a forecast (derivations: docs/SLO.md). A library that
bench/predictions.py prints and bench/tests/test_roofline.py checks, layered
specs -> traffic -> roofline -> floors -> inversions -> money.

Owed before quantised weights are benchmarked: the dtype byte counts as float.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

FLOPS_PER_MAC = 2

# A decode step reads each recurrent state and writes it back: vLLM's kernel
# updates it in place (SLO.md section 11).
STATE_PASSES = 2


@dataclass(frozen=True)
class Experts:
    """The routed experts of an MoE FFN: resident in full, read as routed.

    Shared experts are not here -- every token reads them, so they are ordinary
    weights. params from bench/models/checkpoints.json.
    """

    params: float                 # every routed expert of every layer
    count: int                    # config.json: num_experts
    per_token: int                # config.json: num_experts_per_tok

    def share_read(self, tokens: int) -> float:
        """Expected share a pass over `tokens` tokens reads, routing uniform (SLO.md section 11)."""
        return 1 - (1 - self.per_token / self.count) ** tokens


# A dense FFN is the MoE whose routed experts weigh nothing.
NO_EXPERTS = Experts(params=0.0, count=1, per_token=1)


@dataclass(frozen=True)
class LinearAttention:
    """Gated DeltaNet layers: a fixed state per sequence where attention keeps KV per token.

    Shapes as vLLM v0.27.1 allocates them (mamba_utils.py,
    gated_delta_net_state_shape).
    """

    num_layers: int               # config.json: layer_types == "linear_attention"
    num_k_heads: int              # config.json: linear_num_key_heads
    num_v_heads: int              # config.json: linear_num_value_heads
    head_k_dim: int               # config.json: linear_key_head_dim
    head_v_dim: int               # config.json: linear_value_head_dim
    conv_kernel: int              # config.json: linear_conv_kernel_dim
    state_dtype_bytes: int = 4    # config.json: mamba_ssm_dtype float32, which vLLM adopts
    conv_dtype_bytes: int = 2     # the model's dtype

    @property
    def bytes_per_sequence(self) -> int:
        """A k x v matrix per value head, plus the conv window over q, k and v, per layer."""
        recurrent = self.num_v_heads * self.head_k_dim * self.head_v_dim * self.state_dtype_bytes
        channels = 2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        conv = (self.conv_kernel - 1) * channels * self.conv_dtype_bytes
        return self.num_layers * (recurrent + conv)


# Full attention in every layer is the hybrid with no linear layers.
NO_LINEAR_ATTENTION = LinearAttention(0, 0, 0, 0, 0, conv_kernel=1)


@dataclass(frozen=True)
class Model:
    """Architecture from config.json plus the checkpoint's parameter counts.

    Memory reads every weight, compute skips the embedding lookup. Empty experts
    and linear are dense full attention (SLO.md section 11).
    """

    name: str
    params_total: float           # memory side: all weights are read
    params_non_embedding: float   # compute side: embeddings contribute no FLOPs
    num_layers: int               # config.json: num_hidden_layers
    num_kv_heads: int             # config.json: num_key_value_heads (GQA groups)
    head_dim: int                 # config.json: head_dim
    weight_dtype_bytes: int = 2   # BF16
    kv_dtype_bytes: int = 2       # BF16 KV cache; 1 for FP8
    experts: Experts = NO_EXPERTS
    linear: LinearAttention = NO_LINEAR_ATTENTION
    provenance: str = ""          # where the figures came from, read by the printers

    @property
    def weights_bytes(self) -> float:
        """Resident, and read in full by any pass that routes to every expert."""
        return self.params_total * self.weight_dtype_bytes

    def weights_read_bytes(self, tokens: int) -> float:
        """Read by one forward pass over `tokens` tokens: all but the experts none of them routed to."""
        skipped = self.experts.params * (1 - self.experts.share_read(tokens))
        return (self.params_total - skipped) * self.weight_dtype_bytes

    @property
    def params_active(self) -> float:
        """Compute side: P_active in GLOSSARY.md's notation."""
        unrouted = 1 - self.experts.per_token / self.experts.count
        return self.params_non_embedding - self.experts.params * unrouted

    @property
    def attention_layers(self) -> int:
        """The layers that keep KV; the rest keep a state."""
        return self.num_layers - self.linear.num_layers

    @property
    def kv_bytes_per_token(self) -> float:
        """Read once per sequence per decode step. The leading 2 is K and V."""
        return (
            2
            * self.num_kv_heads
            * self.head_dim
            * self.kv_dtype_bytes
            * self.attention_layers
        )

    @property
    def state_bytes_per_sequence(self) -> int:
        """Held by every sequence whatever its length; 0 under full attention."""
        return self.linear.bytes_per_sequence

    @property
    def state_tokens(self) -> float:
        """The state in tokens of this model's KV: what a seat holds beyond its context."""
        return self.state_bytes_per_sequence / self.kv_bytes_per_token


@dataclass(frozen=True)
class Accelerator:
    """Vendor spec table, dense rows only -- never the sparsity row."""

    name: str
    memory_bytes: float
    peak_bandwidth: float         # bytes/s
    peak_flops: float             # FLOP/s, BF16 dense
    achieved_bandwidth: float     # empirical, see SLO.md section 9
    mfu: float                    # empirical, prefill

    # Where the coefficients came from, in words: "measured (run 1, ...)" or
    # "prior, unvalidated". Read by the printers only; empty by default so
    # that no provenance is asserted unasked.
    provenance: str = ""


class Roofline(NamedTuple):
    seconds: float
    bound_by: str               # "memory" | "compute"
    ratio: float                # how many times the binding side exceeds the other


class Concurrency(NamedTuple):
    """What max_num_seqs should be set to, and which constraint decided it.

    Both limits are kept: their gap says whether a knob is worth touching
    (SLO.md section 4, last row). "capacity", not "memory": both limits are
    memory limits, one in bytes and one in bytes per second.
    """

    sequences: int              # what ships: the smaller of the two
    bound_by: str               # "capacity" | "latency"
    by_capacity: int            # bytes the card holds
    by_latency: int             # bytes per second, inside the TPOT target
    ratio: float                # how far apart the two limits are; inf if either is 0


QWEN3_8B = Model(
    name="Qwen3-8B",
    params_total=8.2e9,
    params_non_embedding=6.95e9,
    num_layers=36,
    num_kv_heads=8,
    head_dim=128,
    provenance="config.json and the model card; docs/model-anatomy.md",
)

# Priced, never served or measured here. Parameters: bench/models/checkpoints.json
# as served text-only (SLO.md section 11).

# Qwen3-8B's FFN (4096 / 12288) under another attention: 8 of 32 layers keep KV.
QWEN3_5_9B = Model(
    name="Qwen3.5-9B",
    params_total=8_953_803_264,
    params_non_embedding=6_919_565_824,
    num_layers=32,
    num_kv_heads=4,
    head_dim=256,
    linear=LinearAttention(num_layers=24, num_k_heads=16, num_v_heads=32,
                           head_k_dim=128, head_v_dim=128, conv_kernel=4),
    provenance="config.json and the safetensors headers at c202236; "
               "never served or measured here, docs/audience.md",
)

# Qwen3.8-Max's architecture (qwen3_5_moe_text) at a size one card holds.
QWEN3_6_35B_A3B = Model(
    name="Qwen3.6-35B-A3B",
    params_total=34_660_610_688,
    params_non_embedding=33_643_491_968,
    num_layers=40,
    num_kv_heads=2,
    head_dim=256,
    experts=Experts(params=32_212_254_720, count=256, per_token=8),
    linear=LinearAttention(num_layers=30, num_k_heads=16, num_v_heads=32,
                           head_k_dim=128, head_v_dim=128, conv_kernel=4),
    provenance="config.json and the safetensors headers at 995ad96; "
               "never served or measured here, docs/audience.md",
)

MI300X = Accelerator(
    name="AMD Instinct MI300X",
    memory_bytes=192 * 2**30,   # "192 GB" is GiB: rocm-smi shows 191.69 GiB (MI300X run 1)
    peak_bandwidth=5.3e12,
    peak_flops=1.307e15,
    achieved_bandwidth=0.70,    # unvalidated; SLO.md section 9, not measured on this card
    mfu=0.45,                   # unvalidated; prefill figure, not measured on this card
    provenance="prior, unvalidated; MI300X run 1 measured 0.46 / 0.166 against it",
)

L40S = Accelerator(
    name="NVIDIA L40S",
    memory_bytes=48e9,
    peak_bandwidth=864e9,
    peak_flops=362.05e12,
    achieved_bandwidth=0.70,    # unvalidated; SLO.md section 9, not measured on this card
    mfu=0.45,                   # unvalidated; prefill figure, not measured on this card
    provenance="prior, unvalidated; the spec-sheet card, kept for contrast",
)

# Same card, coefficients fitted by run 1 (docs/benchmarks/l40s-baseline.md). A
# separate instance, so a prediction can be printed against either and the
# section 9 table can say which produced a row (SLO.md section 9).
L40S_RUN1 = Accelerator(
    name="NVIDIA L40S (run 1 coefficients)",
    memory_bytes=48e9,
    peak_bandwidth=864e9,
    peak_flops=362.05e12,
    achieved_bandwidth=0.83,    # measured, decode, median ITL, twelve levels
    mfu=0.439,                  # measured, prefill, one uncontended request
    provenance="measured (run 1, 2026-08-18); survived runs 2-3",
)

# Same card as MI300X, coefficients fitted by MI300X run 1
# (docs/benchmarks/mi300x-run1.md). A separate instance, so the two differ only
# in what the run measured.
MI300X_RUN1 = Accelerator(
    name="AMD Instinct MI300X (run 1 coefficients)",
    memory_bytes=192 * 2**30,
    peak_bandwidth=5.3e12,
    peak_flops=1.307e15,
    achieved_bandwidth=0.46,    # measured, decode, median ITL, five levels, 0.41-0.49
    mfu=0.166,                  # measured, prefill, one uncontended 4 000-token request
    provenance="measured (MI300X run 1, 2026-09-27); MI300X run 2, on another droplet, "
               "ran 21 % faster decode and 33 % faster prefill at the same server line",
)

# Same card, same server line, a second droplet: MI300X run 2's b2048 row
# (docs/benchmarks/mi300x-run2.md section 3). Beside run 1's, not over it: the
# two droplets disagree, and neither is the card's.
MI300X_RUN2 = Accelerator(
    name="AMD Instinct MI300X (run 2 coefficients)",
    memory_bytes=192 * 2**30,
    peak_bandwidth=5.3e12,
    peak_flops=1.307e15,
    achieved_bandwidth=0.57,    # measured, decode, median ITL, seven levels, 0.52-0.62
    mfu=0.247,                  # measured, prefill, one uncontended 4 000-token request
    provenance="measured (MI300X run 2, 2026-09-27), on a second droplet; every later "
               "MI300X run predicted from it",
)


# Cards by name, for every --accelerator flag. A card is physics, so it lives
# here and not in a caller. Keys are stable: a rename breaks every runsheet
# that prints a command.
ACCELERATORS = {
    "l40s-run1": L40S_RUN1,
    "l40s": L40S,
    "mi300x-run2": MI300X_RUN2,
    "mi300x-run1": MI300X_RUN1,
    "mi300x": MI300X,
}


def seat_step_bytes(model: Model, context_len: int) -> float:
    """What one more sequence adds to a decode step: its KV read, its state read and written."""
    return (
        model.kv_bytes_per_token * context_len
        + STATE_PASSES * model.state_bytes_per_sequence
    )


def decode_step_bytes(model: Model, batch_size: int, context_len: int) -> float:
    """Bytes moved per decode step: weights once, each sequence's share; every decode figure derives from it."""
    return (
        model.weights_read_bytes(batch_size)
        + seat_step_bytes(model, context_len) * batch_size
    )


def kv_cache_tokens(
        model: Model, accel: Accelerator,
        gpu_memory_utilization: float) -> float:
    """Tokens of KV that fit in what is left after the weights.

    No default for gpu_memory_utilization: a server knob, never an assumption.
    An over-estimate -- activations are not modelled, and the startup log
    outranks it (SLO.md section 9).
    """
    kv_space = accel.memory_bytes * gpu_memory_utilization - model.weights_bytes
    if kv_space < 0:
        # A deployment error, not an operating point: raise rather than return 0.
        raise ValueError(
            f"{model.name} weights ({model.weights_bytes / 1e9:.1f} GB) exceed "
            f"the {gpu_memory_utilization:.0%} share of {accel.name} "
            f"({accel.memory_bytes * gpu_memory_utilization / 1e9:.1f} GB): "
            f"no KV cache remains"
        )
    return kv_space / model.kv_bytes_per_token


def roofline(memory: float, compute: float) -> Roofline:
    """Compare two achievable times; the larger one binds."""
    if memory > compute:
        seconds, bound_by, other = memory, "memory", compute
    else:
        seconds, bound_by, other = compute, "compute", memory

    # inf, not ZeroDivisionError: at batch 0 the compute side vanishes while
    # the weights are still read, and a sweep must not crash at that point.
    ratio = seconds / other if other > 0 else math.inf
    return Roofline(seconds, bound_by, ratio)


def tpot_floor(
        model: Model, accel: Accelerator,
        batch_size: int, context_len: int) -> Roofline:
    """Roofline of one decode step; each side over its own coefficient."""
    memory = decode_step_bytes(model, batch_size, context_len) / \
                (accel.peak_bandwidth * accel.achieved_bandwidth)
    compute = FLOPS_PER_MAC * model.params_active * batch_size / \
                (accel.peak_flops * accel.mfu)
    return roofline(memory, compute)


def prefill_bytes(model: Model, prompt_tokens: int) -> float:
    """Bytes moved while prefilling one prompt: weights once, KV and state written once.

    Not modelled: attention re-reading the KV already written (SLO.md section 4).
    """
    return (
        model.weights_read_bytes(prompt_tokens)
        + model.kv_bytes_per_token * prompt_tokens
        + model.state_bytes_per_sequence
    )


def ttft_floor(
        model: Model, accel: Accelerator, prompt_tokens: int) -> Roofline:
    """The same roofline as decode, with the prompt length as B_tokens.

    Prefill computes every prompt token at once, so the compute side scales
    with prompt_tokens where decode's scales with batch_size. Which side won is
    reported, not assumed: a weaker matrix engine flips it (SLO.md section 4).
    """
    memory = prefill_bytes(model, prompt_tokens) / \
                (accel.peak_bandwidth * accel.achieved_bandwidth)
    compute = FLOPS_PER_MAC * model.params_active * prompt_tokens / \
                (accel.peak_flops * accel.mfu)
    return roofline(memory, compute)


def max_num_seqs_from_slo(
        model: Model, accel: Accelerator,
        context_len: int, tpot_target: float) -> int:
    """Largest batch whose TPOT floor still fits the latency budget.

    Latency only; capacity is concurrency_ceiling and the shipped value is
    max_num_seqs. Returns 0 when the weights read alone exceeds the budget.
    """
    if context_len <= 0:
        raise ValueError(
            "context_len must be positive: with no KV to read, the memory side "
            "does not bound the batch at all and the inversion is undefined"
        )

    byte_budget = tpot_target * accel.peak_bandwidth * accel.achieved_bandwidth
    per_seq = seat_step_bytes(model, context_len)

    # by_bandwidth: the memory *bus* inside a time budget. "memory" already
    # names the capacity limit in concurrency_ceiling.
    by_bandwidth = (byte_budget - model.weights_bytes) / per_seq

    # Every expert read and none bracket the limit; dense weights make it one
    # point, MoE is bisected inside it (SLO.md section 11).
    lo = max(math.floor(by_bandwidth), 0)
    hi = math.floor((byte_budget - model.weights_read_bytes(0)) / per_seq)
    if hi > lo:
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if model.weights_read_bytes(mid) + per_seq * mid <= byte_budget:
                lo = mid
            else:
                hi = mid - 1
        by_bandwidth = lo

    by_compute = (tpot_target * accel.peak_flops * accel.mfu) / \
                    (FLOPS_PER_MAC * model.params_active)

    # floor, never round: rounding up ships a breached SLO. max(_, 0): a target
    # below the batch-1 floor makes by_bandwidth negative, and a negative seat
    # count can travel as far as a config file.
    return max(math.floor(min(by_bandwidth, by_compute)), 0)


def seats_under_prefill_interference(
        model: Model, accel: Accelerator,
        context_len: int, tpot_target: float,
        interference_slope: float, interference_intercept: float,
        hit_rate: float = 0.0) -> float:
    """Seats a service can promise once prefill lands inside decode steps.

    slope and intercept are seconds per seat, fitted on one run and valid over
    its concurrency range only (l40s-baseline.md section 5); hit_rate scales the
    interference term alone (SLO.md section 6). A float: callers compare it
    against a measured crossing between two integers.
    """
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError("hit_rate is a share of prompt tokens, so 0 <= h <= 1")

    step_per_seat = seat_step_bytes(model, context_len) / \
                    (accel.peak_bandwidth * accel.achieved_bandwidth)
    step_at_zero = model.weights_bytes / \
                    (accel.peak_bandwidth * accel.achieved_bandwidth)

    uncached = 1.0 - hit_rate
    # step(n) + uncached * (slope*n + intercept) = tpot_target, solved for n.
    denominator = step_per_seat + uncached * interference_slope
    if denominator <= 0:
        raise ValueError(
            "a non-positive slope means the served step does not grow with the "
            "batch, and the seat count is unbounded rather than large"
        )
    return (tpot_target - step_at_zero - uncached * interference_intercept) / denominator


def concurrency_ceiling(
        model: Model, accel: Accelerator,
        context_len: int, gpu_memory_utilization: float) -> int:
    """Sequences whose KV fits at a fixed context length; capacity only.

    context_len is the length *reserved* per sequence, so this is the
    steady-state seat count; a recurrent state adds its own tokens' worth. Returns
    0 when less than one fits -- a real answer, unlike the weights not fitting,
    which kv_cache_tokens raises on.
    """
    if context_len <= 0:
        raise ValueError(
            "context_len must be positive: at zero reserved context every "
            "sequence is free and the seat count is unbounded"
        )
    return math.floor(kv_cache_tokens(model, accel, gpu_memory_utilization) /
                      (context_len + model.state_tokens))


def max_num_seqs(
        model: Model, accel: Accelerator,
        context_len: int, tpot_target: float,
        gpu_memory_utilization: float) -> Concurrency:
    """The number an operator actually sets, and which constraint set it.

    Five arguments, not a struct: context_len is workload, tpot_target a
    promise (SLO.md section 2), gpu_memory_utilization a server knob, and no
    caller holds all three in one place.
    """
    by_latency = max_num_seqs_from_slo(model, accel, context_len, tpot_target)
    by_capacity = concurrency_ceiling(model, accel, context_len, gpu_memory_utilization)

    # A tie is latency-bound: the SLO is the promise that was made.
    bound_by = "capacity" if by_capacity < by_latency else "latency"
    sequences = min(by_capacity, by_latency)

    # From the shipped integers, not the unfloored limits: the gap an operator
    # can act on is between seats they can fill (a wobble at small counts,
    # SLO.md section 4). inf when nothing can be served at this point.
    ratio = max(by_capacity, by_latency) / sequences if sequences > 0 else math.inf

    return Concurrency(sequences, bound_by, by_capacity, by_latency, ratio)


def aggregate_tokens_per_sec(
        model: Model, accel: Accelerator,
        batch_size: int, context_len: int) -> float:
    """Output tokens per second across the batch -- still a floor, never a forecast."""
    return batch_size / tpot_floor(model, accel, batch_size, context_len).seconds


def cost_per_1m_tokens(
        aggregate_tokens_per_sec: float, hourly_rate: float) -> float:
    """SLO.md section 7: hourly rate over tokens delivered per hour.

    Both are arguments: the rate is a contract, not physics, and one function
    prices a measured run and a predicted one alike.
    """
    return hourly_rate / (aggregate_tokens_per_sec * 3600) * 1e6


def token_budget_ceiling(budget: int, prompt_tokens: int, output_tokens: int) -> float:
    """GLOSSARY.md, *Token-budget ceiling*: running count when every step is full."""
    return budget * output_tokens / (prompt_tokens + output_tokens)


def tpot_with_interference(step: float, batch_size: int, ttft_alone: float,
                           output_tokens: int) -> float:
    """Run 1's interference model (docs/benchmarks/mi300x-run1.md section 6), in the unit of its inputs."""
    return step + (batch_size - 1) * ttft_alone / output_tokens
