"""Every figure docs/SLO.md and the benchmark reports publish, as an assertion.

The rule: a number is published in a doc or asserted here, preferably both, and a
test that fails after a coefficient changes means the doc and the expectation
change together. Tolerances are per assertion: integers match exactly, milliseconds
to the doc's rounding. No pytest dependency; `pytest bench/` also works.

    python3 bench/tests/test_roofline.py
"""

# Run directly on a pod without pytest, this file gets only its own directory on
# the path; conftest.py is not consulted, so the insert lives here.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import math

from dataclasses import replace

import measured
import measured_mi300x_run1 as mi300x_run1
import measured_run2
import predictions

from roofline import (
    ACCELERATORS,
    FLOPS_PER_MAC,
    L40S,
    L40S_RUN1,
    MI300X,
    MI300X_RUN1,
    QWEN3_8B,
    Accelerator,
    aggregate_tokens_per_sec,
    concurrency_ceiling,
    cost_per_1m_tokens,
    decode_step_bytes,
    kv_cache_tokens,
    max_num_seqs,
    max_num_seqs_from_slo,
    prefill_bytes,
    roofline,
    seats_under_prefill_interference,
    tpot_floor,
    ttft_floor,
)

MS = 1e-3
INTERACTIVE_TPOT = 0.050    # section 2
INTERACTIVE_TTFT = 0.300    # section 2
GMU = 0.9                   # the gpu_memory_utilization every derivation assumes


def close(actual: float, expected: float, tol: float) -> bool:
    """Absolute tolerance, because the documented figures are rounded decimals."""
    return abs(actual - expected) <= tol


# --- section 3: derived constants ------------------------------------------

def test_weights_bytes():
    # 8.2e9 parameters at BF16. Exact: this is a multiplication, not an estimate.
    assert QWEN3_8B.weights_bytes == 16.4e9


def test_kv_bytes_per_token():
    # 2 (K and V) x 8 kv heads x 128 head_dim x 2 bytes x 36 layers.
    assert QWEN3_8B.kv_bytes_per_token == 147_456


def test_gqa_is_the_whole_saving():
    """The same model with MHA pays 4x for identical output (section 3)."""
    mha = replace(QWEN3_8B, num_kv_heads=32)
    assert mha.kv_bytes_per_token == 589_824
    assert mha.kv_bytes_per_token == 4 * QWEN3_8B.kv_bytes_per_token


def test_fp8_kv_halves_the_kv_term_and_leaves_the_weights_alone():
    """The two dtype fields scale different terms -- the reason they are separate."""
    fp8 = replace(QWEN3_8B, kv_dtype_bytes=1)
    assert fp8.kv_bytes_per_token == QWEN3_8B.kv_bytes_per_token / 2
    assert fp8.weights_bytes == QWEN3_8B.weights_bytes


# --- section 4: floors ------------------------------------------------------

def test_tpot_floor_is_taken_at_empty_context():
    """4.42 ms on MI300X, and it is the weights read and nothing else."""
    floor = tpot_floor(QWEN3_8B, MI300X, batch_size=1, context_len=0)
    assert close(floor.seconds, 4.42 * MS, 0.01 * MS)
    assert floor.bound_by == "memory"
    # 187x, and the document is explicit that this is a *lower* bound, because
    # mfu = 0.45 was measured on prefill and flatters the compute side.
    assert close(floor.ratio, 187, 1)


def test_tpot_floor_on_the_measurement_card():
    """27.1 ms on L40S -- the row that caught 864 typed as 846."""
    floor = tpot_floor(QWEN3_8B, L40S, batch_size=1, context_len=0)
    assert close(floor.seconds, 27.1 * MS, 0.05 * MS)


def test_ttft_floor_inverts_the_roofline():
    """47.3 ms on MI300X for a 2 000-token prompt, compute-bound by 10.5x."""
    floor = ttft_floor(QWEN3_8B, MI300X, prompt_tokens=2000)
    assert close(floor.seconds, 47.3 * MS, 0.05 * MS)
    assert floor.bound_by == "compute"
    assert close(floor.ratio, 10.5, 0.1)


def test_ttft_floor_on_the_measurement_card():
    """170.6 ms on L40S, 57% of the entire 300 ms budget before any queueing."""
    floor = ttft_floor(QWEN3_8B, L40S, prompt_tokens=2000)
    assert close(floor.seconds, 170.6 * MS, 0.1 * MS)
    assert floor.seconds < INTERACTIVE_TTFT


def test_prefill_carries_the_kv_it_writes():
    """The term section 4 omitted: small at 2 000 tokens, 29% of traffic at 32 000."""
    assert prefill_bytes(QWEN3_8B, 0) == QWEN3_8B.weights_bytes
    assert close(prefill_bytes(QWEN3_8B, 2000), 16.69e9, 0.01e9)
    assert close(prefill_bytes(QWEN3_8B, 32000), 21.12e9, 0.01e9)


def test_the_rejected_cards_are_rejected_by_arithmetic():
    """Section 4's two counterexamples: the RTX 4090 fails on the TTFT floor, the L4 on the batch-1 TPOT floor."""
    rtx4090 = Accelerator(
        name="NVIDIA RTX 4090", memory_bytes=24e9, peak_bandwidth=1008e9,
        peak_flops=165.2e12, achieved_bandwidth=0.70, mfu=0.45,
    )
    l4 = Accelerator(
        name="NVIDIA L4", memory_bytes=24e9, peak_bandwidth=300e9,
        peak_flops=121e12, achieved_bandwidth=0.70, mfu=0.45,
    )

    assert tpot_floor(QWEN3_8B, rtx4090, 1, 0).seconds < tpot_floor(QWEN3_8B, L40S, 1, 0).seconds
    assert ttft_floor(QWEN3_8B, rtx4090, 2000).seconds > INTERACTIVE_TTFT
    assert tpot_floor(QWEN3_8B, l4, 1, 0).seconds > INTERACTIVE_TPOT


# --- section 5: the batching trade-off --------------------------------------

def test_batching_table():
    """20x the throughput for 3.2x the latency -- the headline of section 5."""
    one = tpot_floor(QWEN3_8B, MI300X, batch_size=1, context_len=4000)
    many = tpot_floor(QWEN3_8B, MI300X, batch_size=64, context_len=4000)

    assert close(one.seconds, 4.58 * MS, 0.01 * MS)
    assert close(many.seconds, 14.6 * MS, 0.05 * MS)
    assert close(many.ratio, 9.6, 0.1)

    assert close(aggregate_tokens_per_sec(QWEN3_8B, MI300X, 1, 4000), 218, 1)
    assert close(aggregate_tokens_per_sec(QWEN3_8B, MI300X, 64, 4000), 4384, 5)


def test_the_kv_term_is_what_batching_actually_moves():
    """3.5% of the traffic at batch 1, 70% at batch 64 (section 5)."""
    for batch, share in ((1, 0.035), (64, 0.70)):
        total = decode_step_bytes(QWEN3_8B, batch, 4000)
        kv = total - QWEN3_8B.weights_bytes
        assert close(kv / total, share, 0.01)


def test_identical_bytes_means_identical_tpot_and_a_64x_cost_gap():
    """batch x context is what the memory side sees; cost per token is not."""
    wide = tpot_floor(QWEN3_8B, MI300X, batch_size=64, context_len=1000)
    deep = tpot_floor(QWEN3_8B, MI300X, batch_size=1, context_len=64000)
    assert close(wide.seconds, deep.seconds, 1e-9)

    wide_rate = aggregate_tokens_per_sec(QWEN3_8B, MI300X, 64, 1000)
    deep_rate = aggregate_tokens_per_sec(QWEN3_8B, MI300X, 1, 64000)
    assert close(wide_rate / deep_rate, 64, 0.01)


# --- section 6: the concurrency ceiling -------------------------------------

def test_kv_pool_matches_the_startup_log_prediction():
    """1.15 M KV tokens on MI300X at 192 GiB; its startup log printed 1.12 M (section 9)."""
    assert close(kv_cache_tokens(QWEN3_8B, MI300X, GMU), 1.147e6, 0.001e6)
    assert close(kv_cache_tokens(QWEN3_8B, L40S, GMU), 0.1817e6, 0.001e6)


def test_seats_at_four_thousand_context():
    """286 and 45 -- section 4's table."""
    assert concurrency_ceiling(QWEN3_8B, MI300X, 4000, GMU) == 286
    assert concurrency_ceiling(QWEN3_8B, L40S, 4000, GMU) == 45


def test_a_full_card_lands_just_inside_the_target():
    """49.9 ms against 50 -- 0.2% of headroom, which is 'fits exactly', not 'fits'."""
    floor = tpot_floor(QWEN3_8B, MI300X, batch_size=286, context_len=4000)
    assert close(floor.seconds, 49.9 * MS, 0.1 * MS)
    assert floor.seconds < INTERACTIVE_TPOT


def test_fp8_kv_doubles_the_seats_at_constant_tpot():
    """Section 6's falsifiable prediction, and an FP8 KV run exists to break it."""
    fp8 = replace(QWEN3_8B, kv_dtype_bytes=1)

    assert close(kv_cache_tokens(fp8, MI300X, GMU), 2.294e6, 0.001e6)
    assert concurrency_ceiling(fp8, MI300X, 4000, GMU) == 573

    base = tpot_floor(QWEN3_8B, MI300X, 286, 4000).seconds
    doubled = tpot_floor(fp8, MI300X, 2 * 286, 4000).seconds
    # Not approximately equal -- identical. Twice the sequences at half the bytes
    # each is the same bytes_moved, and TPOT is a function of bytes_moved alone.
    assert close(doubled, base, 1e-9)


def test_fp8_kv_cannot_change_which_limit_binds():
    """Both limits divide by kv_bytes_per_token, so it cancels out of their ratio."""
    fp8 = replace(QWEN3_8B, kv_dtype_bytes=1)
    for accel in (MI300X, L40S):
        before = max_num_seqs(QWEN3_8B, accel, 4000, INTERACTIVE_TPOT, GMU)
        after = max_num_seqs(fp8, accel, 4000, INTERACTIVE_TPOT, GMU)
        assert before.bound_by == after.bound_by
        # within one seat: each count is floored separately
        assert abs(after.by_capacity - 2 * before.by_capacity) <= 1


# --- the inversions ---------------------------------------------------------

def test_latency_limit_round_trip():
    """The property that makes an inversion checkable: n fits, n+1 does not."""
    cases = [
        (L40S,   4000, 0.050),
        (L40S,   4000, 0.030),
        (MI300X, 4000, 0.050),
        (MI300X, 32000, 0.050),
    ]
    for accel, ctx, target in cases:
        n = max_num_seqs_from_slo(QWEN3_8B, accel, ctx, target)
        assert n >= 1, f"{accel.name} at {ctx}/{target}: nothing qualifies"
        assert tpot_floor(QWEN3_8B, accel, n, ctx).seconds <= target
        assert tpot_floor(QWEN3_8B, accel, n + 1, ctx).seconds > target


def test_an_unreachable_target_returns_zero_not_a_negative():
    """A 20 ms target on L40S: the weights read alone costs 27.1 ms."""
    assert max_num_seqs_from_slo(QWEN3_8B, L40S, 4000, 0.020) == 0


def test_which_limit_binds_is_a_property_of_the_card():
    """Section 4's last row: a tie on MI300X, which counts as latency; latency by 2x on L40S."""
    mi300x = max_num_seqs(QWEN3_8B, MI300X, 4000, INTERACTIVE_TPOT, GMU)
    assert mi300x.sequences == 286
    assert mi300x.by_capacity == mi300x.by_latency
    assert mi300x.bound_by == "latency"
    assert close(mi300x.ratio, 1.00, 0.001)

    l40s = max_num_seqs(QWEN3_8B, L40S, 4000, INTERACTIVE_TPOT, GMU)
    assert l40s.sequences == 23
    assert l40s.bound_by == "latency"
    assert close(l40s.ratio, 1.96, 0.01)


def test_reasoning_length_context_collapses_concurrency():
    """Section 8: 35 sequences instead of 286, an eightfold reduction."""
    assert concurrency_ceiling(QWEN3_8B, MI300X, 32000, GMU) == 35
    assert max_num_seqs(QWEN3_8B, MI300X, 32000, INTERACTIVE_TPOT, GMU).sequences == 35


# --- the runsheet levels ----------------------------------------------------
# docs/benchmarks/runsheets/l40s-first-run.md quotes a floor per level: a sheet
# that drifted from the module fails here, before the card is rented.

def test_runsheet_concurrency_sweep_floors():
    """Step 4: eight levels at 4 000 context, and the crossing between 23 and 24."""
    expected = {
        1: 28.09, 4: 31.02, 8: 34.92, 16: 42.72,
        23: 49.55, 24: 50.52, 32: 58.32, 45: 71.00,
    }
    for batch, ms in expected.items():
        floor = tpot_floor(QWEN3_8B, L40S, batch, 4000)
        assert close(floor.seconds, ms * MS, 0.01 * MS), f"level {batch}"
        assert floor.bound_by == "memory"

    assert tpot_floor(QWEN3_8B, L40S, 23, 4000).seconds <= INTERACTIVE_TPOT
    assert tpot_floor(QWEN3_8B, L40S, 24, 4000).seconds > INTERACTIVE_TPOT


def test_the_top_level_of_the_sweep_must_preempt():
    """A seat costs input + output, which turns section 4's 45 into 43 at 4 000 in and 200 out."""
    assert concurrency_ceiling(QWEN3_8B, L40S, 4000, GMU) == 45
    assert concurrency_ceiling(QWEN3_8B, L40S, 4200, GMU) == 43
    for level in (23, 24, 32):
        assert level <= 43
    assert 45 > 43


def test_runsheet_length_sweep_floors():
    """Step 5: TTFT doubles with the prompt, TPOT at batch 4 barely moves."""
    ttft = {2000: 170.6, 4000: 341.3, 8000: 682.5}
    tpot = {2000: 29.07, 4000: 31.02, 8000: 34.92}
    for prompt in (2000, 4000, 8000):
        measured_ttft = ttft_floor(QWEN3_8B, L40S, prompt).seconds
        measured_tpot = tpot_floor(QWEN3_8B, L40S, 4, prompt).seconds
        assert close(measured_ttft, ttft[prompt] * MS, 0.05 * MS), prompt
        assert close(measured_tpot, tpot[prompt] * MS, 0.01 * MS), prompt

    # The shape, not the numbers: prefill is linear in prompt length, decode
    # carries the prompt only as one batch's share of one step's KV traffic.
    assert close(ttft[4000] / ttft[2000], 2.0, 0.01)
    assert close(ttft[8000] / ttft[4000], 2.0, 0.01)
    assert tpot[8000] / tpot[2000] < 1.25


def test_the_length_sweep_fits_under_the_serving_limit():
    """9 000 covers 8 000 in + 200 out with slack; 8 192 does not (step 3)."""
    longest = 8000 + 200
    assert longest > 8192
    assert longest < 9000
    # And the pool still seats enough sequences for checkpoint B to be readable:
    # the logged maximum concurrency roughly halves against boot #1's 4 096.
    at_4096 = kv_cache_tokens(QWEN3_8B, L40S, GMU) / 4096
    at_9000 = kv_cache_tokens(QWEN3_8B, L40S, GMU) / 9000
    assert close(at_4096, 44.4, 0.1)
    assert close(at_9000, 20.2, 0.1)


# --- section 7: cost --------------------------------------------------------

def test_cost_is_the_rate_divided_by_tokens_per_hour():
    """The formula of section 7, checked against arithmetic done by hand."""
    assert close(cost_per_1m_tokens(4384, 2.00), 0.1267, 0.0001)


def test_batching_divides_a_fixed_bill():
    """The same card, the same hour: 26x the cost per token at batch 1."""
    alone = cost_per_1m_tokens(aggregate_tokens_per_sec(QWEN3_8B, MI300X, 1, 4000), 2.00)
    full = cost_per_1m_tokens(aggregate_tokens_per_sec(QWEN3_8B, MI300X, 286, 4000), 2.00)
    assert close(alone / full, 26, 0.5)


# --- guards -----------------------------------------------------------------

def test_a_model_that_does_not_fit_raises_rather_than_returning_zero():
    """Weights above the budget is a deployment error, not an operating point."""
    tiny = Accelerator(
        name="16 GB card", memory_bytes=16e9, peak_bandwidth=600e9,
        peak_flops=100e12, achieved_bandwidth=0.70, mfu=0.45,
    )
    try:
        kv_cache_tokens(QWEN3_8B, tiny, GMU)
    except ValueError as exc:
        assert "16.4 GB" in str(exc)
    else:
        raise AssertionError("expected ValueError: 16.4 GB of weights in 14.4 GB")


def test_zero_context_is_rejected_by_both_inversions():
    """Different reasons, both undefined: no KV to read, and no seats to divide."""
    for call in (
        lambda: max_num_seqs_from_slo(QWEN3_8B, L40S, 0, INTERACTIVE_TPOT),
        lambda: concurrency_ceiling(QWEN3_8B, L40S, 0, GMU),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on context_len = 0")


def test_a_vanishing_side_reports_infinity_rather_than_dividing_by_zero():
    """Batch 0: the weights are still read, and nothing computes them."""
    assert roofline(memory=1.0, compute=0.0).ratio == math.inf
    assert tpot_floor(QWEN3_8B, L40S, batch_size=0, context_len=0).ratio == math.inf


def test_flops_per_mac_is_arithmetic_not_a_dtype():
    """It equalled kv_dtype_bytes at BF16, and FP8 KV would have halved compute."""
    assert FLOPS_PER_MAC == 2
    fp8 = replace(QWEN3_8B, kv_dtype_bytes=1)
    compute_side = ttft_floor(fp8, MI300X, 2000)
    assert close(compute_side.seconds, ttft_floor(QWEN3_8B, MI300X, 2000).seconds, 0.1 * MS)


# --- Run 1, L40S -- docs/benchmarks/l40s-baseline.md -------------------------
# Measurements: these fail when the document quotes a number the raw evidence
# under docs/benchmarks/raw/ does not support.

def test_run1_the_derived_kv_pool_overstates_the_logged_one():
    """The startup log outranks the derivation, and by 7.6% here (SLO.md 9)."""
    derived = kv_cache_tokens(QWEN3_8B, L40S, GMU)
    logged = measured.LOGGED_KV_TOKENS
    assert logged < derived, "the derivation must be the optimistic one"
    assert close((derived - logged) / logged * 100, 7.6, 0.1)


def test_run1_the_measured_pool_reproduces_vllms_own_concurrency_line():
    """168 985 / 9 000 must give back the 18.78x vLLM printed, independently."""
    assert close(measured.LOGGED_KV_TOKENS / 9000, measured.LOGGED_MAX_CONCURRENCY, 0.01)


def test_run1_uncalibrated_bandwidth_is_pessimistic_at_every_single_level():
    """0.70 predicted a slower step than the card took at all twelve levels: a coefficient, not noise."""
    for level in measured.levels():
        predicted = tpot_floor(
            QWEN3_8B, L40S, level["ran"], level["context"]).seconds * 1000
        assert predicted > level["decode_step_ms"], level


def test_run1_calibrated_bandwidth_predicts_every_decode_step_within_5_percent():
    """0.83 fitted to these rows, so this asserts the fit, not the physics."""
    for level in measured.levels():
        predicted = tpot_floor(
            QWEN3_8B, L40S_RUN1, level["ran"], level["context"]).seconds * 1000
        error = abs(predicted - level["decode_step_ms"]) / level["decode_step_ms"]
        assert error < 0.05, (level, predicted, error)


def test_run1_the_implied_bandwidth_efficiency_is_one_scalar_not_a_curve():
    """+-0.04 across four thousand tokens of context and forty-one sequences."""
    effs = [measured.implied_bandwidth_efficiency(QWEN3_8B, L40S, lv)
            for lv in measured.levels()]
    assert max(effs) - min(effs) < 0.08, effs
    assert close(sum(effs) / len(effs), 0.83, 0.01)


def test_run1_the_best_efficiency_is_at_batch_one_inverting_the_old_assumption():
    """SLO.md 9 expected batch 1 to be the worst point. It is the best."""
    by_level = {lv["asked"]: measured.implied_bandwidth_efficiency(QWEN3_8B, L40S, lv)
                for lv in measured.levels() if lv["input_len"] == 4000}
    assert by_level[1] == max(by_level.values())
    assert by_level[45] == min(by_level.values())


def test_run1_prefill_confirms_the_mfu_assumption_on_the_uncontended_level():
    """One request alone on the card: 341.3 ms predicted, 350.1 ms measured."""
    alone = next(lv for lv in measured.levels()
                 if lv["asked"] == 1 and lv["input_len"] == 4000)
    predicted = ttft_floor(QWEN3_8B, L40S, alone["input_len"]).seconds * 1000
    assert abs(predicted - alone["ttft_p50_ms"]) / alone["ttft_p50_ms"] < 0.03
    assert close(measured.implied_mfu(QWEN3_8B, L40S, alone), 0.439, 0.001)


def test_run1_ttft_is_linear_in_prompt_length_and_tpot_is_not():
    """Closed on the bench rather than by argument: 4x the prompt, 3.35x the TTFT."""
    sweep = {lv["input_len"]: lv for lv in measured.levels() if lv["asked"] == 4}
    ttft_ratio = sweep[8000]["ttft_p50_ms"] / sweep[2000]["ttft_p50_ms"]
    step_ratio = sweep[8000]["decode_step_ms"] / sweep[2000]["decode_step_ms"]
    assert close(ttft_ratio, 3.35, 0.05), ttft_ratio      # 4x the prompt, 3.35x the TTFT
    assert close(step_ratio, 1.19, 0.02), step_ratio      # 4x the prompt, 1.19x the step


def test_run1_calibration_does_not_change_which_limit_binds_on_this_card():
    """0.70 -> 0.83 buys nine seats and leaves latency binding, 41 against 32."""
    before = max_num_seqs(QWEN3_8B, L40S, 4000, 0.050, GMU)
    after = max_num_seqs(QWEN3_8B, L40S_RUN1, 4000, 0.050, GMU)
    assert before.bound_by == after.bound_by == "latency"
    assert before.by_latency == 23 and after.by_latency == 32


def test_run1_the_service_number_is_far_below_the_calibrated_decode_number():
    """Right about the card, wrong about the service: 32 seats by the decode step, 13 by TPOT p99."""
    at_4000 = {lv["asked"]: lv for lv in measured.levels() if lv["input_len"] == 4000}
    assert max_num_seqs(QWEN3_8B, L40S_RUN1, 4000, 0.050, GMU).by_latency == 32
    assert at_4000[13]["tpot_p99_ms"] > 50.0
    assert at_4000[8]["tpot_p99_ms"] < 50.0
    assert at_4000[45]["tpot_p50_ms"] / at_4000[45]["decode_step_ms"] > 2.0


# --- Run 2, L40S -- docs/benchmarks/l40s-run2.md -----------------------------
# 0.83 and 0.439 were fitted to run 1, not to any level here: each assertion is
# a prediction facing a measurement that did not produce it (SLO.md section 9).


def _r2(tag):
    return next(lv for lv in measured_run2.levels() if lv["tag"] == tag)


def test_run2_the_calibrated_coefficient_survives_a_run_it_did_not_see():
    """Six BF16 launches at c=13 on another pod and driver, none of them fitted to, hold 0.83."""
    steps = [lv["decode_step_ms"] for lv in measured_run2.levels()
             if lv["conc"] == 13 and not lv["fp8"]]
    assert len(steps) == 6, steps
    assert (max(steps) - min(steps)) / min(steps) < 0.02, steps
    predicted = tpot_floor(QWEN3_8B, L40S_RUN1, 13, 4100).seconds * 1000
    mean = sum(steps) / len(steps)
    assert abs(predicted - mean) / mean < 0.005, (predicted, mean)


def test_run2_the_coefficient_also_holds_at_the_other_concurrency():
    """c=32, only where the median ITL is still a decode step (l40s-run2.md section 6 names the exclusions)."""
    predicted = tpot_floor(QWEN3_8B, L40S_RUN1, 32, 4100).seconds * 1000
    for tag in ("bf16-2048-c32", "bf16-4096-c32"):
        got = _r2(tag)["decode_step_ms"]
        assert abs(predicted - got) / got < 0.02, (tag, predicted, got)


def test_run2_the_median_itl_stops_being_a_decode_step_at_small_chunks():
    """At c=32 a small chunk puts prefill in most steps, and TPOT p50 / median ITL falls to 1 or below."""
    predicted = tpot_floor(QWEN3_8B, L40S_RUN1, 32, 4100).seconds * 1000
    for tag in ("bf16-512-c32", "bf16-1024-c32"):
        lv = _r2(tag)
        assert lv["decode_step_ms"] > predicted * 1.5, tag
        assert lv["tpot_p50_ms"] / lv["decode_step_ms"] <= 1.01, tag
    for tag in ("bf16-2048-c32", "bf16-4096-c32"):
        lv = _r2(tag)
        assert lv["tpot_p50_ms"] / lv["decode_step_ms"] > 1.9, tag


def test_run2_fp8_kv_doubles_the_pool_exactly():
    """2.000x, and the seats double with it. No coefficient involved."""
    ratio = (measured_run2.LOGGED_KV_TOKENS_FP8
             / measured_run2.LOGGED_KV_TOKENS_BF16)
    assert close(ratio, 2.0, 0.001), ratio
    assert int(measured_run2.LOGGED_KV_TOKENS_BF16 / 4100) == 41
    assert int(measured_run2.LOGGED_KV_TOKENS_FP8 / 4100) == 82


def test_run2_fp8_at_twice_the_seats_costs_what_bf16_cost_at_half():
    """SLO.md section 6's claim, measured: 2x sequences at the same decode step."""
    fp8_26 = _r2("fp8-2048-c26")["decode_step_ms"]
    bf16_13 = _r2("bf16-2048-c13")["decode_step_ms"]
    assert abs(fp8_26 - bf16_13) / bf16_13 < 0.05, (fp8_26, bf16_13)


def test_run2_fp8_is_predicted_by_halving_the_kv_term_and_nothing_else():
    """No new physics: the same roofline with kv_dtype_bytes = 1."""
    for tag, n in (("fp8-2048-c13", 13), ("fp8-2048-c26", 26), ("fp8-2048-c32", 32)):
        got = _r2(tag)["decode_step_ms"]
        predicted = tpot_floor(
            measured_run2.QWEN3_8B_FP8_KV, L40S_RUN1, n, 4100).seconds * 1000
        assert abs(predicted - got) / got < 0.05, (tag, predicted, got)


def test_run2_the_attention_kernel_is_not_the_reason_fp8_is_faster():
    """The confounder control. FP8 forced FLASHINFER; the kernel is worth 0.8%."""
    flash_attn = _r2("bf16-2048-c13")["decode_step_ms"]
    flashinfer = _r2("bf16-fi-c13")["decode_step_ms"]
    fp8 = _r2("fp8-2048-c13")["decode_step_ms"]
    assert abs(flashinfer - flash_attn) / flash_attn < 0.02
    assert (flash_attn - fp8) / flash_attn > 0.10


def test_run2_max_num_batched_tokens_moves_the_tail_and_not_the_step():
    """At c=13 the decode step is invariant to the chunk size while ITL p99 rises with it."""
    at13 = {lv["mnbt"]: lv for lv in measured_run2.block("B") if lv["conc"] == 13}
    steps = [at13[m]["decode_step_ms"] for m in (512, 1024, 2048, 4096)]
    assert (max(steps) - min(steps)) / min(steps) < 0.01, steps
    tails = [at13[m]["itl_p99_ms"] for m in (512, 1024, 2048, 4096)]
    assert tails == sorted(tails), tails
    assert tails[-1] / tails[0] > 5, tails


def test_run2_the_worst_step_is_the_decode_step_plus_the_chunk():
    """mfu 0.439, fitted to one uncontended prefill, prices a chunk to 10%."""
    step = tpot_floor(QWEN3_8B, L40S_RUN1, 13, 4100).seconds * 1000
    at13 = {lv["mnbt"]: lv for lv in measured_run2.block("B") if lv["conc"] == 13}
    for mnbt in (512, 1024, 2048, 4096):
        chunk = min(mnbt - 13, 4000)
        bound = step + ttft_floor(QWEN3_8B, L40S_RUN1, chunk).seconds * 1000
        got = at13[mnbt]["itl_p99_ms"]
        assert got < bound, (mnbt, bound, got)          # it is a bound
        assert (bound - got) / got < 0.10, (mnbt, bound, got)   # and a tight one


def test_run2_the_scheduler_knob_buys_two_seats_and_costs_six_times_the_ttft():
    """Block B's question, answered below the lowest offered band: 12 seats became 14."""
    lo, hi = _r2("bf16-512-c13"), _r2("bf16-512b-c16")
    assert lo["tpot_p99_ms"] < 50.0 < hi["tpot_p99_ms"]
    crossing = 13 + 3 * (50.0 - lo["tpot_p99_ms"]) / \
        (hi["tpot_p99_ms"] - lo["tpot_p99_ms"])
    assert 13.5 < crossing < 14.5, crossing
    assert _r2("bf16-2048-c13")["tpot_p99_ms"] > 50.0        # run 1's 12 reproduced
    price = _r2("bf16-512-c32")["ttft_p50_ms"] / _r2("bf16-2048-c32")["ttft_p50_ms"]
    assert price > 5.0, price


def test_run2_throughput_and_goodput_point_opposite_ways_past_the_peak():
    """The block A headline, and the only place cost per SLO-respecting token exists."""
    a = {lv["rate"]: lv for lv in measured_run2.block("A") if lv["tag"] != "A-warm"}
    peak = max(a.values(), key=lambda lv: lv["goodput"])
    top = a[4.0]
    assert peak["rate"] == 2.5 and close(peak["goodput"], 1.87, 0.01)
    assert top["out_tps"] > peak["out_tps"] * 1.5           # throughput still climbing
    assert top["goodput"] < peak["goodput"] / 15            # goodput gone
    cheap = measured_run2.dollars_per_1m(peak["goodput"] * 200)
    ruinous = measured_run2.dollars_per_1m(top["goodput"] * 200)
    assert ruinous / cheap > 15, (cheap, ruinous)
    assert measured_run2.dollars_per_1m(top["out_tps"]) < \
        measured_run2.dollars_per_1m(peak["out_tps"])       # the misleading column


def test_run2_the_pool_ceiling_closes_on_one_number_three_ways():
    """Capacity arithmetic, the engine's own gauge, and the load generator."""
    seats = measured_run2.LOGGED_KV_TOKENS_BF16 / 1600
    assert close(seats, 106.1, 0.1), seats
    assert measured_run2.METRICS_MAX_RUNNING == 106
    assert measured_run2.METRICS_MAX_PREEMPTIONS > 0        # the ceiling was reached
    assert _r2("A-r4.0")["peak_conc"] == 106 + 5            # running + waiting


def test_run2_the_checkpoint_gate_was_tighter_than_the_platforms_own_variation():
    """The +-500-token gate was tighter than three launches of one configuration (SLO.md section 9)."""
    pools = measured_run2.LOGGED_KV_TOKENS_RELAUNCHES
    spread = (max(pools) - min(pools)) / min(pools)
    assert spread > 0.04, pools
    assert spread > 500 / 168_985 * 10, (spread, "the gate is an order of magnitude tighter")


def test_run2_ttft_p50_cannot_carry_a_ten_percent_decision():
    """TTFT p50 across three identical BF16 launches is too noisy to carry a ten-percent decision."""
    ttfts = [_r2(t)["ttft_p50_ms"]
             for t in ("bf16-2048-c13", "bf16-ctl-c13", "bf16-fi-c13")]
    assert (max(ttfts) - min(ttfts)) / min(ttfts) > 0.30, ttfts
    steps = [_r2(t)["decode_step_ms"]
             for t in ("bf16-2048-c13", "bf16-ctl-c13", "bf16-fi-c13")]
    assert (max(steps) - min(steps)) / min(steps) < 0.02, steps


# --- Prefix caching -- docs/SLO.md section 6 --------------------------------
# Derived: these hold the published table to the code that prints it.

SLOPE = predictions.L40S_RUN1_INTERFERENCE_SLOPE
INTERCEPT = predictions.L40S_RUN1_INTERFERENCE_INTERCEPT


def _seats(h):
    return seats_under_prefill_interference(
        QWEN3_8B, L40S_RUN1, 4100, INTERACTIVE_TPOT, SLOPE, INTERCEPT, h)


def test_prefix_cache_at_zero_reproduces_run_1s_measured_crossing():
    """At h = 0 the model lands between run 1's 13 and 14 seats, the data it was fitted to."""
    assert 13.0 < _seats(0.0) < 14.0, _seats(0.0)


def test_prefix_cache_at_full_hit_rate_is_the_decode_step_limit():
    """h = 1 removes all prefill, leaving max_num_seqs_from_slo's answer."""
    hardware = max_num_seqs_from_slo(QWEN3_8B, L40S_RUN1, 4100, INTERACTIVE_TPOT)
    assert abs(_seats(1.0) - hardware) < 1.0, (_seats(1.0), hardware)


def test_prefix_cache_payoff_is_convex_in_the_hit_rate():
    """Each equal step in h buys strictly more seats than the last (SLO.md section 6)."""
    gains = [_seats(h + 0.2) - _seats(h) for h in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert all(b > a for a, b in zip(gains, gains[1:])), gains


def test_prefix_cache_table_matches_the_document():
    """The four figures docs/SLO.md section 6 publishes in its table."""
    assert close(_seats(0.00), 13.5, 0.05), _seats(0.00)
    assert close(_seats(0.50), 18.2, 0.05), _seats(0.50)
    assert close(_seats(0.80), 24.3, 0.05), _seats(0.80)
    assert close(_seats(1.00), 32.2, 0.05), _seats(1.00)


def test_prefix_cache_does_not_touch_the_decode_step():
    """Section 6 channel 1: with the interference off, the seat count is the same at any hit rate."""
    flat = [seats_under_prefill_interference(
        QWEN3_8B, L40S_RUN1, 4100, INTERACTIVE_TPOT, SLOPE, 0.0, h)
        for h in (0.0, 0.5, 1.0)]
    steps = [decode_step_bytes(QWEN3_8B, int(n), 4100) for n in flat]
    assert steps[0] < steps[1] < steps[2], steps       # more seats, more bytes
    assert flat[0] < flat[1] < flat[2], flat


def test_prefix_cache_rejects_a_hit_rate_outside_zero_to_one():
    """h is a share. 1.2 would return a seat count above the hardware limit."""
    for bad in (-0.1, 1.1):
        try:
            _seats(bad)
        except ValueError:
            continue
        raise AssertionError(f"hit_rate {bad} was accepted")


# --- The --what-if flag -- bench/predictions.py ------------------------------
# The two properties a flag can break on its own: it computes what the fixed
# tables do, and it cannot reach them.


# --- table 11: a fleet on one card --------------------------------------------
# Two engines on one MI300X (docs/benchmarks/runsheets/mi300x-run-3.md):
# capacity arithmetic, no interference fit, outranked by the startup log.


def test_fleet_model_multiplies_the_weights_and_leaves_the_flops_alone():
    """Two engines read two copies of the weights and compute one token once."""
    fleet = predictions.fleet_model(QWEN3_8B, 2)
    assert fleet.params_total == 2 * QWEN3_8B.params_total
    assert fleet.params_non_embedding == QWEN3_8B.params_non_embedding
    assert fleet.kv_bytes_per_token == QWEN3_8B.kv_bytes_per_token


def test_a_second_engine_costs_exactly_one_more_copy_of_the_weights():
    """The fleet's bill on both limits is 16.4 GB divided by that limit's seat (SLO.md section 6)."""
    seat_capacity = QWEN3_8B.kv_bytes_per_token * 4200
    seat_latency = QWEN3_8B.kv_bytes_per_token * 4000
    fleet = predictions.fleet_model(QWEN3_8B, 2)

    capacity_bill = concurrency_ceiling(QWEN3_8B, MI300X, 4200, 0.90) - \
        concurrency_ceiling(fleet, MI300X, 4200, 0.90)
    latency_bill = max_num_seqs_from_slo(QWEN3_8B, MI300X, 4000, INTERACTIVE_TPOT) - \
        max_num_seqs_from_slo(fleet, MI300X, 4000, INTERACTIVE_TPOT)

    assert abs(capacity_bill - QWEN3_8B.weights_bytes / seat_capacity) < 1.0, capacity_bill
    assert abs(latency_bill - QWEN3_8B.weights_bytes / seat_latency) < 1.0, latency_bill
    assert (capacity_bill, latency_bill) == (27, 28), (capacity_bill, latency_bill)


def test_the_fleet_keeps_the_tie_on_this_card():
    """A second engine subtracts the same weights from both limits, so the MI300X's tie holds."""
    fleet = predictions.fleet_model(QWEN3_8B, 2)
    seats = max_num_seqs(fleet, MI300X, 4200, INTERACTIVE_TPOT, 0.90)
    assert seats.by_capacity == seats.by_latency, seats
    assert seats.bound_by == "latency", seats


def test_affinity_repays_the_fleet_bill_only_past_a_working_set():
    """Below ~35 distinct prefixes, at 0.76 seats each, affinity cannot repay the second engine's bill."""
    per_prefix = (2 - 1) * predictions.SHARED_PREFIX_TOKENS / 4200
    break_even = 27 / per_prefix
    assert 35.0 < break_even < 36.0, break_even


def test_affinity_stops_buying_space_once_both_policies_fill_the_pool():
    """Past the pool's room the saving moves from seats into h, and is not paid twice."""
    room, replicas = 96.0, 2
    nominal = 0.8

    def freed(n):
        return replicas * (min(n, room) - min(n / replicas, room))

    def hit_rates(n):
        return (nominal * min(1.0, room / n), nominal * min(1.0, room * replicas / n))

    assert freed(64) > 0 and hit_rates(64) == (nominal, nominal)
    assert freed(512) == 0
    rr, prefix = hit_rates(512)
    assert prefix == 2 * rr > 0, (rr, prefix)


def test_what_if_agrees_with_table_3_on_the_same_operating_point():
    """Table 3 and --what-if call the same function and must print the same seats."""
    seats = max_num_seqs(QWEN3_8B, MI300X, 4000, INTERACTIVE_TPOT, 0.9)
    assert seats.bound_by == "latency", seats
    assert seats.sequences == 286, f"table 3 row moved: {seats.sequences}"


def test_what_if_defaults_reserve_room_to_generate():
    """The default context reserves the 200 output tokens this repository measures at."""
    import inspect

    defaults = inspect.signature(predictions.what_if).parameters
    ctx = defaults["context_len"].default
    prompt = defaults["prompt_tokens"].default
    assert ctx == 4200 and prompt == 4000, (ctx, prompt)
    assert ctx > prompt, "the default seat has no room for a single output token"


def test_what_if_parameters_cannot_reach_the_fixed_tables():
    """A --what-if parameter passed without --what-if fails rather than editing the fixed tables."""
    # argparse prints its usage and its message to stderr before it exits, and
    # a suite that looks like it crashed while passing is a suite people stop
    # reading. Swallowed here rather than globally: this is the one test whose
    # subject is the error text.
    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()) as noise:
            predictions.main(["--accelerator", "mi300x"])
    except SystemExit as exc:
        assert exc.code != 0, "a parameter without --what-if exited clean"
        assert "--what-if" in noise.getvalue(), noise.getvalue()
        return
    raise AssertionError("--accelerator without --what-if printed the tables")


def test_what_if_names_every_card_the_harness_can_be_given():
    """One accelerator registry, shared by the harness and --what-if."""
    import harness

    assert harness.ACCELERATORS is ACCELERATORS
    assert set(ACCELERATORS) == {"l40s-run1", "l40s", "mi300x-run1", "mi300x"}


# --- MI300X run 1 -------------------------------------------------------------
# docs/benchmarks/mi300x-run1.md quotes these; bench/measured_mi300x_run1.py reads
# them from docs/benchmarks/raw/mi300x-2026-09-27/, so a figure that drifts from
# the evidence fails here.

def test_mi300x_run1_the_logged_pool_is_whole_blocks_and_vllms_own_line():
    """1 123 065 is max_concurrency x 9 000, and inverts to 70 254 blocks of 16."""
    for tokens, concurrency in zip(mi300x_run1.LOGGED_KV_TOKENS,
                                   mi300x_run1.LOGGED_MAX_CONCURRENCY):
        assert close(tokens / 9000, concurrency, 0.01)
        blocks = tokens * math.ceil(9000 / 16) / 9000
        assert abs(blocks - round(blocks)) < 0.05, blocks
    assert mi300x_run1.logged_blocks(mi300x_run1.LOGGED_KV_TOKENS[0]) == 70_254


def test_mi300x_run1_the_unit_was_the_miss():
    """At 192e9 the derivation fell 5.6 % short of the log; at 192 GiB it is 2.1 % over."""
    logged = mi300x_run1.LOGGED_KV_TOKENS[0]
    at_192e9 = kv_cache_tokens(QWEN3_8B, replace(MI300X, memory_bytes=192e9), GMU)
    assert at_192e9 < logged
    assert close((kv_cache_tokens(QWEN3_8B, MI300X, GMU) - logged) / logged * 100, 2.1, 0.1)
    assert close(predictions.POOL_SHORTFALL,
                 1 - logged / kv_cache_tokens(QWEN3_8B, MI300X, GMU), 0.001)


def test_mi300x_run1_the_fitted_coefficients_are_the_raw_files():
    """0.46 is the mean of five decode rows; 0.166 is the uncontended 4 000-token prefill."""
    data = mi300x_run1.rows()
    assert close(MI300X_RUN1.achieved_bandwidth, mi300x_run1.fitted_eff_mem(data), 0.01)
    assert close(MI300X_RUN1.mfu, mi300x_run1.implied_mfu(data["c001"]), 0.001)


def test_mi300x_run1_the_prior_predicted_every_decode_step_too_fast():
    """0.70 was optimistic at every row where the median ITL is a decode step."""
    data = mi300x_run1.rows()
    for name in mi300x_run1.DECODE_ROWS:
        row = data[name]
        predicted = tpot_floor(QWEN3_8B, MI300X, row["c"], row["context"]).seconds * 1000
        assert predicted < row["itl_ms"], (name, predicted, row["itl_ms"])


def test_mi300x_run1_running_settles_at_the_token_budget_not_the_pool_or_the_cap():
    """~99 running against 97.5 derived, the cap 256 unreached, nothing preempted."""
    saturated = mi300x_run1.saturated_running(mi300x_run1.gauges())
    ceiling = mi300x_run1.equilibrium_running(4000, 200)
    assert abs(sorted(saturated)[len(saturated) // 2] - ceiling) <= 2, ceiling
    assert max(saturated) < mi300x_run1.MAX_NUM_SEQS
    assert max(mi300x_run1.preemptions()) == 0


def test_mi300x_run1_latency_as_served_crossed_between_8_and_32():
    """TPOT p99 under 50 ms at c008 and over it at c032: latency bound, not the pool."""
    data = mi300x_run1.rows()
    assert data["c008"]["tpot_p99_ms"] < 50 < data["c032"]["tpot_p99_ms"]


def test_mi300x_run1_interference_needs_no_budget_term():
    """ITL + (c - 1) x TTFT(c=1) / 200 lands within 3 % of TPOT p50 at c008 and c032."""
    data = mi300x_run1.rows()
    alone = data["c001"]["ttft_p50_ms"]
    for name in ("c008", "c032"):
        row = data[name]
        model = mi300x_run1.tpot_with_interference(row["itl_ms"], row["c"], alone)
        assert abs(model - row["tpot_p50_ms"]) / row["tpot_p50_ms"] < 0.03, (name, model)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}\n      {exc or test.__doc__}")
        else:
            print(f"ok    {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
