"""Every figure docs/SLO.md publishes, as an assertion.

Why this file exists, stated plainly: in three separate sessions the code and
the document disagreed, and every time it was caught by eye. `864` typed as
`846` is 2.1% and does not look wrong. The batch-1 column was computed at zero
context under a header saying 4 000. The prefill memory side omitted the KV the
prompt writes. None of those is a subtle bug; all three survived review because
nothing forced the two files to meet.

So the rule this file encodes: **a number is published in docs/SLO.md or it is
asserted here, and preferably both.** Since run 1 the same rule covers
docs/benchmarks/l40s-baseline.md, whose measured figures are held by the
run-1 block near the bottom of this file, and since run 2 the same again for
docs/benchmarks/l40s-run2.md in the block after it. A test that fails after a coefficient
changes is doing its job -- update the document and the expectation together,
in the same commit, which is exactly the step that kept being skipped.

Tolerances are stated per assertion rather than globally, because they mean
different things: an exact integer (seat counts, byte counts) is arithmetic and
must match exactly, while a millisecond figure is compared against a document
that rounds to one or two decimals.

No dependency on pytest, which is not installed here -- plain functions, plain
asserts, and a runner at the bottom:

    python3 bench/tests/test_roofline.py

Written so that `pytest bench/` works verbatim the day pytest is installed: the
names begin with test_, the assertions are bare, and nothing uses a fixture.
"""

# Two ways in, and both have to work. `pytest bench/` gets bench/ on the import
# path from tests/conftest.py; `python3 bench/tests/test_roofline.py` on a rented
# pod, where pytest is not installed, gets only this directory. The three lines
# below are what make the second one work, and they are here rather than in
# conftest.py for exactly that reason.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import math

from dataclasses import replace

import measured
import measured_run2
import predictions

from roofline import (
    ACCELERATORS,
    FLOPS_PER_MAC,
    L40S,
    L40S_RUN1,
    MI300X,
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
    """Section 4's two counterexamples, kept executable so the choice stays checkable.

    The RTX 4090 has *better* decode bandwidth than the L40S and is still
    unusable: 165 TFLOP/s dense puts its TTFT floor above the whole budget, so
    the interactive class cannot exist on it. The L4 fails one step earlier, on
    the TPOT floor at batch 1.
    """
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
    """1.15 M KV tokens on MI300X at 192 GiB; the startup log outranks it (section 9)."""
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
    """The property that makes an inversion checkable: n fits, n+1 does not.

    Printed as a column in bench/predictions.py; asserted here, because an
    assert is the version that fails a build rather than needing to be read.
    """
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
#
# The sweeps of docs/benchmarks/runsheets/l40s-first-run.md quote a floor per
# level, and a runsheet is read beside a running pod where nothing re-derives
# anything. So the
# levels are asserted here for the same reason every other published figure is:
# a sheet that has drifted from the module should fail a test in step 0, before
# the card is rented, rather than be noticed at the pod.

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
    """A seat costs input + output, which is what turns 45 into 43.

    Section 4 seats 45 sequences at 4 000 tokens; the sweep sends 4 000 in and
    200 out, so a seat holds 4 200 and the pool seats 43. Levels 24 and 32 stay
    inside capacity and breach only the latency target -- the two failures are
    separate, and the sweep is arranged to show them one at a time.
    """
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


# ---------------------------------------------------------------------------
# Run 1, L40S, 2026-08-18 -- docs/benchmarks/l40s-baseline.md
#
# These hold measurements, not derivations, so they fail for a different reason
# than everything above: not "the code drifted from the document" but "the
# document quoted a number the raw evidence does not support". The evidence is
# committed under docs/benchmarks/raw/, which is what makes them re-runnable.
# ---------------------------------------------------------------------------

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
    """0.70 predicted a slower step than the card took, twelve times out of twelve.

    The direction is the finding: a one-sided error is a coefficient, a
    two-sided one would have been noise.
    """
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
    """0.70 -> 0.83 buys nine seats and leaves latency binding, 41 against 32.

    The same property the FP8 test asserts, met from the other side: a
    coefficient moves a limit, and moving a limit is not moving the verdict.
    """
    before = max_num_seqs(QWEN3_8B, L40S, 4000, 0.050, GMU)
    after = max_num_seqs(QWEN3_8B, L40S_RUN1, 4000, 0.050, GMU)
    assert before.bound_by == after.bound_by == "latency"
    assert before.by_latency == 23 and after.by_latency == 32


def test_run1_the_service_number_is_far_below_the_calibrated_decode_number():
    """The headline: the model is right about the card and wrong about the service.

    Calibrated, the roofline permits 32 sequences inside a 50 ms decode step.
    Measured TPOT p99 -- the metric SLO.md 2 actually promises -- breaches at 13.
    The residual is chunked prefill, not bandwidth, so no coefficient can absorb it.
    """
    at_4000 = {lv["asked"]: lv for lv in measured.levels() if lv["input_len"] == 4000}
    assert max_num_seqs(QWEN3_8B, L40S_RUN1, 4000, 0.050, GMU).by_latency == 32
    assert at_4000[13]["tpot_p99_ms"] > 50.0
    assert at_4000[8]["tpot_p99_ms"] < 50.0
    assert at_4000[45]["tpot_p50_ms"] / at_4000[45]["decode_step_ms"] > 2.0


# ---------------------------------------------------------------------------
# Run 2 -- L40S, 2026-08-23. docs/benchmarks/l40s-run2.md.
#
# The difference from the run-1 block above is the whole reason these exist:
# eff_mem 0.83 and mfu 0.439 were fitted to run 1 and NOT to any level here, so
# every assertion below is a prediction facing a measurement that did not
# produce it -- the condition SLO.md section 9 sets before a coefficient may be
# called a result.
# ---------------------------------------------------------------------------


def _r2(tag):
    return next(lv for lv in measured_run2.levels() if lv["tag"] == tag)


def test_run2_the_calibrated_coefficient_survives_a_run_it_did_not_see():
    """The headline. Six BF16 launches at c=13, none of them fitted to.

    Four chunk sizes, two attention backends and three separate starts of the
    same configuration, on a different pod under a different driver. If 0.83 were
    an artefact of run 1's twelve levels this is where it would show.
    """
    steps = [lv["decode_step_ms"] for lv in measured_run2.levels()
             if lv["conc"] == 13 and not lv["fp8"]]
    assert len(steps) == 6, steps
    assert (max(steps) - min(steps)) / min(steps) < 0.02, steps
    predicted = tpot_floor(QWEN3_8B, L40S_RUN1, 13, 4100).seconds * 1000
    mean = sum(steps) / len(steps)
    assert abs(predicted - mean) / mean < 0.005, (predicted, mean)


def test_run2_the_coefficient_also_holds_at_the_other_concurrency():
    """c=32, and only where the median ITL is still a decode step.

    bf16-512 and bf16-1024 are excluded by name rather than by a filter: which
    levels are contaminated is a finding of this run (write-up section 6), not
    yet a rule to select data with.
    """
    predicted = tpot_floor(QWEN3_8B, L40S_RUN1, 32, 4100).seconds * 1000
    for tag in ("bf16-2048-c32", "bf16-4096-c32"):
        got = _r2(tag)["decode_step_ms"]
        assert abs(predicted - got) / got < 0.02, (tag, predicted, got)


def test_run2_the_median_itl_stops_being_a_decode_step_at_small_chunks():
    """Where the proxy expires, held so the glossary claim cannot drift back.

    At c=32 a 512- or 1024-token chunk puts prefill into the majority of steps.
    The tell is TPOT p50 / median ITL falling to 1 or below: not less
    interference, but interference so even that the median sits inside it.
    """
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
    """Prediction 1 and 2 of block B, at c=13, in one assertion.

    The decode step is invariant to the chunk size to under 1%, while ITL p99
    rises monotonically with it. Whatever the knob does, it does not do it by
    making the card faster.
    """
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
    """Block B's actual question, answered below the lowest offered band.

    Run 1 left the seat count at 12 by TPOT p99. The three candidate answers were
    ~31, ~18-22, or no movement. It is 14.
    """
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
    """Why the +-500-token gate is a defect and not a finding.

    Three launches of one configuration on one pod, differing only in whether the
    torch.compile cache was warm.
    """
    pools = measured_run2.LOGGED_KV_TOKENS_RELAUNCHES
    spread = (max(pools) - min(pools)) / min(pools)
    assert spread > 0.04, pools
    assert spread > 500 / 168_985 * 10, (spread, "the gate is an order of magnitude tighter")


def test_run2_ttft_p50_cannot_carry_a_ten_percent_decision():
    """The card's confounder detector had no power, and this is why.

    Three identical BF16 launches, same concurrency, same prompts.
    """
    ttfts = [_r2(t)["ttft_p50_ms"]
             for t in ("bf16-2048-c13", "bf16-ctl-c13", "bf16-fi-c13")]
    assert (max(ttfts) - min(ttfts)) / min(ttfts) > 0.30, ttfts
    steps = [_r2(t)["decode_step_ms"]
             for t in ("bf16-2048-c13", "bf16-ctl-c13", "bf16-fi-c13")]
    assert (max(steps) - min(steps)) / min(steps) < 0.02, steps


# ---------------------------------------------------------------------------
# Prefix caching -- docs/SLO.md section 6. Derived, not measured: the only
# check the model has is that it reproduces run 1's zero-cache crossing, and
# these assertions hold the published table to the code that prints it.
# ---------------------------------------------------------------------------

SLOPE = predictions.L40S_RUN1_INTERFERENCE_SLOPE
INTERCEPT = predictions.L40S_RUN1_INTERFERENCE_INTERCEPT


def _seats(h):
    return seats_under_prefill_interference(
        QWEN3_8B, L40S_RUN1, 4100, INTERACTIVE_TPOT, SLOPE, INTERCEPT, h)


def test_prefix_cache_at_zero_reproduces_run_1s_measured_crossing():
    """The model's one validation, and it is against the data it was fitted to.

    Run 1 measured TPOT p50 at 48.62 ms with 13 sequences and 51.36 with 14
    (docs/benchmarks/l40s-baseline.md section 5), so the crossing is between the
    two. A model that could not land there would not be worth extrapolating.
    """
    assert 13.0 < _seats(0.0) < 14.0, _seats(0.0)


def test_prefix_cache_at_full_hit_rate_is_the_decode_step_limit():
    """h = 1 removes all prefill work, leaving the step the hardware performs.

    That is max_num_seqs_from_slo's answer, measured at ~31 and derived at 32:
    the two must agree, or the interference term is contaminating the step.
    """
    hardware = max_num_seqs_from_slo(QWEN3_8B, L40S_RUN1, 4100, INTERACTIVE_TPOT)
    assert abs(_seats(1.0) - hardware) < 1.0, (_seats(1.0), hardware)


def test_prefix_cache_payoff_is_convex_in_the_hit_rate():
    """Half the prompt cached buys a quarter of the gap, not half of it.

    The operator-facing claim of section 6, and the reason a hit rate below
    ~0.5 is not an answer to a seat count. Convexity, not a number, so it is
    asserted as a shape: each equal step in h buys strictly more than the last.
    """
    gains = [_seats(h + 0.2) - _seats(h) for h in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert all(b > a for a, b in zip(gains, gains[1:])), gains


def test_prefix_cache_table_matches_the_document():
    """The four figures docs/SLO.md section 6 publishes in its table."""
    assert close(_seats(0.00), 13.5, 0.05), _seats(0.00)
    assert close(_seats(0.50), 18.2, 0.05), _seats(0.50)
    assert close(_seats(0.80), 24.3, 0.05), _seats(0.80)
    assert close(_seats(1.00), 32.2, 0.05), _seats(1.00)


def test_prefix_cache_does_not_touch_the_decode_step():
    """Section 6 channel 1: caching saves computing KV, never reading it.

    The same seat count must come out whatever the hit rate, once the
    interference term is switched off -- if it does not, the implementation is
    quietly scaling the bytes a decode step moves.
    """
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


# ---------------------------------------------------------------------------
# The --what-if flag -- bench/predictions.py, added 2026-09-11 for route 1 of
# docs/audience.md. Nothing here re-checks the arithmetic, which the sections
# above already hold to docs/SLO.md; these assert the two properties a flag can
# break on its own: that it computes the same thing the fixed tables do, and
# that it cannot reach them.
# ---------------------------------------------------------------------------


# --- table 11: a fleet on one card --------------------------------------------
#
# The arrangement the router arm runs on (docs/benchmarks/runsheets/mi300x-run-3.md):
# two engines sharing one MI300X, each with half the memory share. Nothing here
# needs an interference fit, which is why these figures may be printed for a
# card no run has touched -- they are capacity arithmetic, and the startup log
# outranks every one of them on the day.


def test_fleet_model_multiplies_the_weights_and_leaves_the_flops_alone():
    """Two engines read two copies of the weights and compute one token once.

    The whole content of predictions.fleet_model, and the one way it could be
    wrong that no printed row would reveal: multiplying params_non_embedding
    too would inflate the compute side of every floor, silently, by the replica
    count.
    """
    fleet = predictions.fleet_model(QWEN3_8B, 2)
    assert fleet.params_total == 2 * QWEN3_8B.params_total
    assert fleet.params_non_embedding == QWEN3_8B.params_non_embedding
    assert fleet.kv_bytes_per_token == QWEN3_8B.kv_bytes_per_token


def test_a_second_engine_costs_exactly_one_more_copy_of_the_weights():
    """The fleet's bill, on both limits, is 16.4 GB divided by a seat.

    Both limits have the form (X - weights) / (context x kv_per_token) and
    differ only in X (docs/SLO.md section 6), so a replica that adds `weights`
    to the numerator's subtrahend costs the same seats in each -- at whatever
    context that limit counts a seat in. Asserted rather than read off the
    table, because it is the sentence the table's first block is for.
    """
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
    """Below ~35 distinct prefixes the second engine is a loss on one card.

    Affinity stores each prefix once instead of once per replica, which is
    (R - 1) x N x prefix_tokens of pool -- 0.76 seats per prefix at 3 200
    tokens in a 4 200-token seat. It has to clear the 27-seat bill above before
    the arrangement is worth anything at all, and that is the first number the
    runsheet asks the run to face.
    """
    per_prefix = (2 - 1) * predictions.SHARED_PREFIX_TOKENS / 4200
    break_even = 27 / per_prefix
    assert 35.0 < break_even < 36.0, break_even


def test_affinity_stops_buying_space_once_both_policies_fill_the_pool():
    """Past the pool's room the gain changes form, from seats to hit rate.

    The regime change table 11 exists to locate: while every prefix is retained
    under both policies the saving is space; once round_robin is evicting, both
    policies hold `room` prefixes, there is no space to differ over, and the
    difference appears in h instead. A model that kept paying seats there would
    double-count the same saving.
    """
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
    """A second path to a number is a second chance to get it wrong.

    Table 3 prints max_num_seqs for the points the document argues about;
    --what-if prints it for the reader's. They call the same function, and this
    fails the moment one of them starts passing something different -- a target
    in milliseconds where the other passes seconds, say.
    """
    seats = max_num_seqs(QWEN3_8B, MI300X, 4000, INTERACTIVE_TPOT, 0.9)
    assert seats.bound_by == "latency", seats
    assert seats.sequences == 286, f"table 3 row moved: {seats.sequences}"


def test_what_if_defaults_reserve_room_to_generate():
    """4 000 tokens of prompt in 4 000 tokens of context is a seat that cannot
    answer. The default context length is the seat cost this repository
    measures at -- 4 000 in, 200 out -- and a default that quietly drops the
    output would price a workload nobody runs.
    """
    import inspect

    defaults = inspect.signature(predictions.what_if).parameters
    ctx = defaults["context_len"].default
    prompt = defaults["prompt_tokens"].default
    assert ctx == 4200 and prompt == 4000, (ctx, prompt)
    assert ctx > prompt, "the default seat has no room for a single output token"


def test_what_if_parameters_cannot_reach_the_fixed_tables():
    """docs/SLO.md quotes the eleven tables' rows, so a parameter that moved them
    would be a parameter that edits a derivation. Passing one without
    --what-if has to fail loudly rather than print the unchanged tables.
    """
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
    """One registry, in bench/roofline.py since 2026-09-11. Two copies is how
    a card gets added to one tool and not the other.
    """
    import harness

    assert harness.ACCELERATORS is ACCELERATORS
    assert set(ACCELERATORS) == {"l40s-run1", "l40s", "mi300x"}


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
