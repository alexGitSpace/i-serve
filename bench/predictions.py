"""Every number bench/roofline.py predicts, printed as twelve numbered tables.

Split from roofline.py because it changes after every run, and the module only
when the arithmetic does. A runsheet's step 0 opens these tables beside the card;
each row names the docs/SLO.md section it holds, and bench/tests/test_roofline.py
asserts the same figures.

    python3 bench/predictions.py
    python3 bench/predictions.py --what-if --accelerator mi300x --context-len 32000
"""

import argparse
import json
import math
import os
import textwrap
from dataclasses import replace

from roofline import (
    ACCELERATORS,
    L40S,
    L40S_RUN1,
    MI300X,
    MI300X_RUN1,
    Model,
    QWEN2_5_7B,
    QWEN3_8B,
    aggregate_tokens_per_sec,
    concurrency_ceiling,
    cost_per_1m_tokens,
    decode_step_bytes,
    kv_cache_tokens,
    max_num_seqs,
    max_num_seqs_from_slo,
    prefill_bytes,
    seats_under_prefill_interference,
    token_budget_ceiling,
    tpot_floor,
    tpot_with_interference,
    ttft_floor,
)

TTFT_TARGET = 0.300         # section 2, interactive
TPOT_TARGET = 0.050         # section 2, interactive
GMU = 0.90                  # the gpu_memory_utilization the runsheet serves at

# MI300X run 1's larger launch-to-derivation gap at 192 GiB (the other was 0.75 %,
# docs/benchmarks/mi300x-run1.md section 2); tables 9 and 11 take it off, and the
# engine's log outranks both.
POOL_SHORTFALL = 0.021

# Two engines on one card: the smallest fleet a router can route over, and the
# only one a single-GPU droplet holds. The prefix is run 3's construction
# (bench/scenarios/prefix_sweep.py).
FLEET_REPLICAS = 2
SHARED_PREFIX_TOKENS = 3_200

# Fixed, not swept: table 11's variable is the working set, and at 32 seats per
# engine both sit far inside their latency limits, so the rows show the pool
# filling -- the quantity routing moves.
FLEET_CONCURRENCY = 64

# docs/SLO.md section 2's two classes, read from here by the calculator and
# bench/export_site_data.py. No third class without a derivation there; a
# reader's own target is --tpot-ms / --ttft-ms.
SLO_CLASSES = {
    "interactive": {"ttft_s": TTFT_TARGET, "tpot_s": TPOT_TARGET},
    "batch": {"ttft_s": 3.000, "tpot_s": 0.200},
}

# Prefill interference fitted to run 1 (docs/benchmarks/l40s-baseline.md
# section 5): an interpolation over 13..32 seats at one chunk size and arrival
# pattern, not a law -- so it is named for its run and never reused silently.
L40S_RUN1_INTERFERENCE_SLOPE = 1.640e-3       # seconds of extra step per seat
L40S_RUN1_INTERFERENCE_INTERCEPT = -6.36e-3   # seconds
L40S_RUN1_INTERFERENCE_PROVENANCE = (
    "measured (run 1, 2026-08-18): TPOT p50 minus median ITL at c = 13/23/32, "
    "4 000-token prompts, max_num_batched_tokens 2 048; its (1 - h) scaling "
    "confirmed by run 3, slope ratio 0.194 against 0.200")

# Dense, GQA, full-attention models only: the formulas are not general
# (GLOSSARY.md, "Architectures that break the standard arithmetic"). Qwen2.5-7B
# is for comparison; Qwen3-8B is the one served and measured.
MODELS = {
    "qwen3-8b": QWEN3_8B,
    "qwen2.5-7b": QWEN2_5_7B,
}
DEFAULT_MODEL = "qwen3-8b"

# The fit was measured decoding Qwen3-8B; what_if_point() withholds it elsewhere.
INTERFERENCE_MODEL = "qwen3-8b"

# A scheduler quantity, not a bandwidth one, so both L40S entries share the fit.
# No MI300X entry on purpose: what_if_point() says "not derivable" instead.
INTERFERENCE_FITS = {
    "l40s-run1": (L40S_RUN1_INTERFERENCE_SLOPE, L40S_RUN1_INTERFERENCE_INTERCEPT,
                  L40S_RUN1_INTERFERENCE_PROVENANCE),
    "l40s": (L40S_RUN1_INTERFERENCE_SLOPE, L40S_RUN1_INTERFERENCE_INTERCEPT,
             L40S_RUN1_INTERFERENCE_PROVENANCE),
}

# Contract figures, not physics: each provenance string says where its rate came
# from, and the rate enters every $/1M figure as a plain multiplier.
L40S_HOURLY = 0.99
MI300X_HOURLY = 1.99
L40S_HOURLY_PROVENANCE = ("RunPod console, 2026-08-15, On-Demand, 1x L40S 48 GB "
                          "(48 GB VRAM, 62 GB RAM, 16 vCPU)")
MI300X_HOURLY_PROVENANCE = ("AMD Developer Cloud console, 2026-09-27, on-demand, "
                            "1x MI300X; paid from credits")

# A default per --accelerator key, with its provenance; --hourly-rate overrides
# it, and a test asserts every card in ACCELERATORS has one.
HOURLY_RATES = {
    "l40s-run1": (L40S_HOURLY, L40S_HOURLY_PROVENANCE),
    "l40s": (L40S_HOURLY, L40S_HOURLY_PROVENANCE),
    "mi300x-run2": (MI300X_HOURLY, MI300X_HOURLY_PROVENANCE),
    "mi300x-run1": (MI300X_HOURLY, MI300X_HOURLY_PROVENANCE),
    "mi300x": (MI300X_HOURLY, MI300X_HOURLY_PROVENANCE),
}

PROSE_WIDTH = 88            # comfortable reading width, independent of the table


def table_header(title: str, blurb: str, columns: str = "") -> None:
    """Title, what the table answers, then the column row -- above every table.

    Rules match the column row's width; prose wraps narrower, to be read.
    """
    width = len(columns) if columns else PROSE_WIDTH
    print()
    print(title)
    print("=" * width)
    print(textwrap.fill(blurb, width=min(width, PROSE_WIDTH)))
    print()
    if columns:
        print(columns)
        print("-" * len(columns))


def sizes() -> None:
    """The two derived constants of section 3, and the GQA counterfactual."""
    table_header(
        f"MODEL: {QWEN3_8B.name}",
        "Every table below is built from these two numbers. Weights are read "
        "once per decode step no matter how many sequences are in flight; KV "
        "per token is multiplied by batch and by context, so it is the only "
        "term that grows. That is why GQA -- 8 KV heads, not 32 -- is the "
        "whole saving: 4x off the term that grows, paid for on the side of "
        "the roofline that is not binding. docs/SLO.md section 3.",
    )
    print(f"  weights          {QWEN3_8B.weights_bytes / 1e9:.1f} GB")
    # SI units throughout: vendor bandwidth figures are SI, so GB/s divides
    # cleanly into GB. Mixing in GiB is where roofline arithmetic quietly drifts.
    print(f"  KV per token     {QWEN3_8B.kv_bytes_per_token / 1e3:.0f} KB")

    # replace(), not a __dict__ copy: it goes through __init__, so a future
    # validator runs on the counterfactual too.
    mha = replace(QWEN3_8B, name="hypothetical MHA", num_kv_heads=32)
    print(f"  KV per token if MHA (n_kv=32)  {mha.kv_bytes_per_token / 1e3:.0f} KB")


def decode_table() -> None:
    """TPOT floors at operating points each card can actually hold.

    A floor says what the physics permits, not what the card can seat, so no row
    sits past a card's pool.
    """
    cases = [
        # accel, batch, context_len
        (MI300X,  64, 4000),   # section 5, the batching table
        (MI300X, 286, 4000),   # section 6, the full card: 49.9 ms against 50
        (L40S,    45, 4000),   # its capacity limit -- 71 ms, well past the SLO
        (L40S,    23, 4000),   # its latency limit -- 49.55 ms, what ships
        (L40S,     1, 4000),
        (L40S,     1,    0),   # section 4: the floor is taken at empty context
    ]
    table_header(
        "TABLE 1: TPOT floors -- one decode step, at points each card can hold",
        "A decode step moves weights + batch x context x KV-per-token, and the "
        "floor is those bytes divided by the card's derated bandwidth. 'ratio' "
        "is the memory side over the compute side: decode is memory-bound by "
        "two orders of magnitude, and no batch size on these cards changes "
        "that. The last row is the same card at empty context -- the batch-1 "
        "floor of section 4, which is lower than the floor at any real "
        "operating point. docs/SLO.md sections 4-5.",
        f"{'accelerator':<20}{'batch':>6}{'ctx':>6}{'bytes/step':>12}"
        f"{'TPOT floor':>12}{'bound by':>10}{'ratio':>8}",
    )
    for accel, batch, ctx in cases:
        floor = tpot_floor(QWEN3_8B, accel, batch, ctx)
        gb = decode_step_bytes(QWEN3_8B, batch, ctx) / 1e9
        print(f"{accel.name:<20}{batch:>6}{ctx:>6}{gb:>9.2f} GB"
              f"{floor.seconds * 1e3:>9.2f} ms{floor.bound_by:>10}"
              f"{floor.ratio:>7.1f}x")


def latency_limit_table() -> None:
    """max_num_seqs_from_slo with the round trip that checks it: n fits, n+1 breaks.

    Printed, not asserted, because an assert disappears under -O; the assert
    lives in bench/tests/test_roofline.py.
    """
    cases = [
        # accel, context_len, tpot_target -- the latency half of section 6
        (L40S,   4000, 0.050),   # 23 by latency; 45 fit, so latency binds
        (L40S,   4000, 0.030),   # same card, tighter target: 2 sequences
        (MI300X, 4000, 0.050),   # 286 by latency; 286 fit, a tie
        (L40S,   4000, 0.020),   # below the batch-1 floor: no batch qualifies
    ]
    table_header(
        "TABLE 2: max_num_seqs from the latency side, and the round trip",
        "How many sequences the TPOT target permits, checked by inverting the "
        "answer: the floor at n must fit inside the target and the floor at "
        "n+1 must break it. The boundary is one sequence wide, which is why "
        "the result is floored and never rounded, and why a target below the "
        "batch-1 floor answers 0 rather than a negative number. "
        "docs/SLO.md section 6.",
        f"{'accelerator':<20}{'ctx':>6}{'TPOT target':>13}{'max_num_seqs':>14}"
        f"{'floor at n':>13}{'floor at n+1':>15}{'round trip':>13}",
    )
    for accel, ctx, target in cases:
        n = max_num_seqs_from_slo(QWEN3_8B, accel, ctx, target)

        at_n = tpot_floor(QWEN3_8B, accel, n, ctx).seconds if n >= 1 else None
        at_next = tpot_floor(QWEN3_8B, accel, n + 1, ctx).seconds
        ok = (at_n is None or at_n <= target) and at_next > target

        cell_n = f"{at_n * 1e3:.2f} ms" if at_n is not None else "n/a"
        print(f"{accel.name:<20}{ctx:>6}{target * 1e3:>10.0f} ms{n:>14}"
              f"{cell_n:>13}{at_next * 1e3:>12.2f} ms"
              f"{'ok' if ok else 'BROKEN':>13}")


def shipping_table() -> None:
    """Both limits side by side, and which one an operator is actually against."""
    fp8_kv = replace(QWEN3_8B, name="Qwen3-8B FP8 KV", kv_dtype_bytes=1)
    cases = [
        # model, accel, context_len, tpot_target, gpu_memory_utilization
        (QWEN3_8B, L40S,    4000, 0.050, 0.9),   # section 4: latency binds, by 2x
        (QWEN3_8B, MI300X,  4000, 0.050, 0.9),   # section 4: a tie, counted as latency
        (QWEN3_8B, MI300X, 32000, 0.050, 0.9),   # section 8: reasoning-length context
        (fp8_kv,   MI300X,  4000, 0.050, 0.9),   # section 6: the falsifiable ~573
    ]
    table_header(
        "TABLE 3: both limits on max_num_seqs, and which one an operator is against",
        "Both are memory limits, and they count different things: 'by "
        "capacity' is how many sequences' KV the card physically holds, 'by "
        "latency' how many it can re-read every single token and still land "
        "inside the TPOT target. max_num_seqs is the smaller of the two; 'gap' "
        "is their ratio. On L40S latency binds at half the seats that fit "
        "(1.96x), on MI300X the two coincide (1.00x, and a tie is latency-bound) -- a property of "
        "the card and the target, never a rule about GPUs. The FP8 KV row "
        "doubles both limits and leaves the gap untouched. "
        "docs/SLO.md sections 4 and 6.",
        f"{'model':<18}{'accelerator':<20}{'ctx':>7}{'gmu':>6}"
        f"{'KV tokens':>12}{'by capacity':>13}{'by latency':>12}"
        f"{'max_num_seqs':>14}{'bound by':>10}{'gap':>7}",
    )
    for model, accel, ctx, target, gmu in cases:
        c = max_num_seqs(model, accel, ctx, target, gmu)
        pool = kv_cache_tokens(model, accel, gmu)
        print(f"{model.name:<18}{accel.name:<20}{ctx:>7}{gmu:>6.2f}"
              f"{pool / 1e6:>9.2f} M{c.by_capacity:>13}{c.by_latency:>12}"
              f"{c.sequences:>14}{c.bound_by:>10}{c.ratio:>6.2f}x")


def prefill_table() -> None:
    """TTFT floors, and how much of the 300 ms budget each one has already spent."""
    cases = [
        # accel, prompt_tokens
        (MI300X,  2000),   # section 4: 47.3 ms of a 300 ms budget
        (L40S,    2000),   # section 4: 170.6 ms of the same budget
        (L40S,     512),   # a short prompt, where prefill is barely compute-bound
        (L40S,   32000),   # section 8: a reasoning-length prompt, budget gone
    ]
    table_header(
        "TABLE 4: TTFT floors, and what prefill alone has already spent",
        "The last column is the floor as a share of the 300 ms interactive "
        "budget of section 2 -- what is gone before a single token of queue or "
        "overhead is counted. 'ratio' shows prefill is compute-bound by prompt "
        "length, not by nature: at 512 tokens it is 1.6x, nearly balanced, and "
        "a short-prompt workload is a different machine. The bytes column "
        "includes the KV that prefill writes for decode to read. "
        "docs/SLO.md sections 4 and 8.",
        f"{'accelerator':<20}{'prompt':>8}{'bytes':>10}{'TTFT floor':>13}"
        f"{'bound by':>10}{'ratio':>8}{'of 300 ms budget':>19}",
    )
    for accel, prompt in cases:
        floor = ttft_floor(QWEN3_8B, accel, prompt)
        gb = prefill_bytes(QWEN3_8B, prompt) / 1e9
        share = floor.seconds / TTFT_TARGET
        print(f"{accel.name:<20}{prompt:>8}{gb:>7.2f} GB"
              f"{floor.seconds * 1e3:>10.1f} ms{floor.bound_by:>10}"
              f"{floor.ratio:>7.1f}x{share:>18.0%}")


def cost_table() -> None:
    """The same floors as money: a floor over a rate, so a lower bound."""
    cases = [
        # accel, batch, context_len, hourly_rate
        (L40S,    23, 4000, L40S_HOURLY),     # what ships on L40S: its latency limit
        (L40S,     1, 4000, L40S_HOURLY),     # one user alone on the card
        (MI300X, 286, 4000, MI300X_HOURLY),   # what ships on MI300X: both limits, tied
        (MI300X,   1, 4000, MI300X_HOURLY),
    ]
    table_header(
        "TABLE 5: the same floors as money",
        "Batching divides a fixed hourly bill across more streams, so the same "
        "card at two operating points differs 13x in cost per token -- and the "
        "cheap point is also the slower one per user, which is the trade an "
        "SLO exists to settle. Every figure is a floor divided by a rate, so "
        "it is a lower bound; the rates are contract assumptions, not physics, "
        "and are re-checked before being quoted. docs/SLO.md section 7.",
        f"{'accelerator':<20}{'batch':>6}{'ctx':>6}{'$/h':>7}{'TPOT':>10}"
        f"{'tok/s/user':>12}{'tok/s total':>13}{'$/1M tokens':>14}",
    )
    for accel, batch, ctx, rate in cases:
        tpot = tpot_floor(QWEN3_8B, accel, batch, ctx).seconds
        total = aggregate_tokens_per_sec(QWEN3_8B, accel, batch, ctx)
        print(f"{accel.name:<20}{batch:>6}{ctx:>6}{rate:>7.2f}"
              f"{tpot * 1e3:>7.2f} ms{1 / tpot:>12.0f}{total:>13.0f}"
              f"{cost_per_1m_tokens(total, rate):>13.3f}")


def sweep_concurrency_table() -> None:
    """One floor per level of the L40S concurrency sweep -- runsheet step 4.

    Tables 1-5 print the points docs/SLO.md argues about; this prints the levels a
    run sends. A seat costs 4 200 tokens, input plus output, so the top level is
    predicted to run out of seats; whether the engine preempts or queues is left
    to the run.
    """
    ctx, out = 4000, 200
    levels = (1, 4, 8, 16, 23, 24, 32, 45)
    seats = concurrency_ceiling(QWEN3_8B, L40S, ctx + out, GMU)
    seats_ignoring_output = concurrency_ceiling(QWEN3_8B, L40S, ctx, GMU)

    table_header(
        "TABLE 6: the concurrency sweep, level by level (runsheet step 4)",
        f"L40S at {ctx} tokens of context: the seven levels runsheet step 4 "
        f"sends, plus the concurrency 4 that step 5 holds fixed while it moves "
        f"the prompt instead. "
        f"'inside 50 ms' is the interactive TPOT target of section 2: the "
        f"crossing sits between 23 and 24, one sequence wide, which is what "
        f"makes the pair worth two levels of GPU time. 'KV seat' counts the "
        f"{ctx} input plus the {out} output tokens a seat actually holds, so "
        f"the pool seats {seats} sequences and not the "
        f"{seats_ignoring_output} of section 4 -- every level above {seats} "
        f"runs out of seats, which the engine answers by preempting a running "
        f"sequence or by making the next one wait. "
        f"docs/SLO.md sections 4 and 6.",
        f"{'concurrency':>12}{'bytes/step':>12}{'TPOT floor':>12}"
        f"{'inside 50 ms':>14}{'KV seat @ 4 200':>17}",
    )
    for batch in levels:
        floor = tpot_floor(QWEN3_8B, L40S, batch, ctx)
        gb = decode_step_bytes(QWEN3_8B, batch, ctx) / 1e9
        inside = "yes" if floor.seconds <= TPOT_TARGET else "no"
        fits = "fits" if batch <= seats else "no seat"
        print(f"{batch:>12}{gb:>9.2f} GB{floor.seconds * 1e3:>9.2f} ms"
              f"{inside:>14}{fits:>17}")

    # The two figures the startup log prints, which is where the runsheet's
    # checkpoints A and B compare a derivation against the engine's own answer.
    # vLLM divides the pool by max_model_len, not by the length actually sent.
    pool = kv_cache_tokens(QWEN3_8B, L40S, GMU)
    print()
    print(f"  KV pool {pool:,.0f} tokens -> logged maximum concurrency "
          f"{pool / 4096:.1f}x at max_model_len 4 096 (checkpoint A), "
          f"{pool / 9000:.1f}x at 9 000 (checkpoint B)")


def sweep_length_table(accel=L40S, batch: int = 4, number: int = 7) -> None:
    """Prompt length at a fixed batch -- runsheet step 5, and the MI300X's mfu read.

    Batch 1 is the only level whose TTFT is a prefill and nothing else; the
    defaults reproduce table 7. 'max_model_len needed' adds the 200 output tokens,
    which a limit set to the prompt alone would reject.
    """
    out = 200
    prompts = (2000, 4000, 8000)

    table_header(
        f"TABLE {number}: the prompt-length sweep, level by level (runsheet step 5)",
        f"{accel.name} at concurrency {batch}, the three lengths the runsheet sends. "
        f"Both floors are predictions the measurement must sit above; the "
        f"measured TTFT sits further above its floor than the TPOT does, "
        f"because at concurrency {batch} the requests queue behind each other's "
        f"prefills and that queue is not in any floor. docs/SLO.md section 4.",
        f"{'input':>8}{'TTFT floor':>13}{'x prev':>9}{'TPOT floor':>13}"
        f"{'x prev':>9}{'max_model_len needed':>22}",
    )
    prev_ttft = prev_tpot = None
    for prompt in prompts:
        ttft = ttft_floor(QWEN3_8B, accel, prompt).seconds
        tpot = tpot_floor(QWEN3_8B, accel, batch, prompt).seconds
        ttft_x = f"{ttft / prev_ttft:.2f}x" if prev_ttft else "--"
        tpot_x = f"{tpot / prev_tpot:.2f}x" if prev_tpot else "--"
        print(f"{prompt:>8}{ttft * 1e3:>10.1f} ms{ttft_x:>9}"
              f"{tpot * 1e3:>10.2f} ms{tpot_x:>9}{prompt + out:>22}")
        prev_ttft, prev_tpot = ttft, tpot


def prefix_cache_table() -> None:
    """What a prefix cache hit rate is worth in seats -- docs/SLO.md section 6.

    h is a property of the traffic, not the card. No MI300X row: the table
    prices seats through an interference fit, and that card has none.
    """
    table_header(
        "TABLE 8: prefix cache hit rate against the seat count",
        "Prefix caching removes prefill work rather than redistributing it, so "
        "it scales the interference term by (1 - h) and leaves the decode step "
        "alone. h = 0 reproduces run 1's measured TPOT p50 crossing between 13 "
        "and 14, which is the only check this model has. The SLO metric is p99 "
        "and sits about one seat lower throughout. Unvalidated against a run "
        "that varies h -- docs/SLO.md section 10.",
        f"{'hit rate h':>12}{'seats at 50 ms':>17}{'seats gained':>15}"
        f"{'of the gap closed':>20}",
    )
    at_zero = seats_under_prefill_interference(
        QWEN3_8B, L40S_RUN1, 4100, TPOT_TARGET,
        L40S_RUN1_INTERFERENCE_SLOPE, L40S_RUN1_INTERFERENCE_INTERCEPT, 0.0)
    at_one = seats_under_prefill_interference(
        QWEN3_8B, L40S_RUN1, 4100, TPOT_TARGET,
        L40S_RUN1_INTERFERENCE_SLOPE, L40S_RUN1_INTERFERENCE_INTERCEPT, 1.0)

    for h in (0.0, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 1.0):
        seats = seats_under_prefill_interference(
            QWEN3_8B, L40S_RUN1, 4100, TPOT_TARGET,
            L40S_RUN1_INTERFERENCE_SLOPE, L40S_RUN1_INTERFERENCE_INTERCEPT, h)
        gained = seats - at_zero
        share = gained / (at_one - at_zero)
        print(f"{h:>12.2f}{seats:>17.2f}{gained:>15.2f}{share:>19.0%}")


def mi300x_sweep_table() -> None:
    """One floor per level of the MI300X calibration sweep -- runsheet mi300x-run-1.

    The runsheet froze this table at 192e9 and a 7 % shortfall; the run measured
    both (docs/benchmarks/mi300x-run1.md). Uncalibrated on purpose (SLO.md
    section 9); max_num_seqs pinned at 256, the third limit of section 10.
    """
    ctx, out = 4000, 200
    levels = (1, 8, 32, 64, 128, 192, 224, 240, 256, 288)
    seq_cap = 256          # pinned in bench/sweep/mi300x-run-1-serve.json

    pool = kv_cache_tokens(QWEN3_8B, MI300X, GMU)
    seats_derived = concurrency_ceiling(QWEN3_8B, MI300X, ctx + out, GMU)
    seats_corrected = int(pool * (1 - POOL_SHORTFALL) // (ctx + out))
    by_latency = max_num_seqs_from_slo(QWEN3_8B, MI300X, ctx, TPOT_TARGET)

    table_header(
        "TABLE 9: the MI300X calibration sweep, level by level",
        f"MI300X at {ctx} tokens of context, uncalibrated eff_mem "
        f"{MI300X.achieved_bandwidth} and mfu {MI300X.mfu}. The pool seats "
        f"{seats_derived} sequences of {ctx + out} tokens by the clean "
        f"arithmetic and about {seats_corrected} once the measured "
        f"{POOL_SHORTFALL:.1%} startup-log shortfall is applied; the latency "
        f"limit at 50 ms is {by_latency}. These are floors, and the run showed "
        f"what they leave out: the token budget held the engine near 100 "
        f"running sequences and TPOT p99 crossed 50 ms between 8 and 32 "
        f"(docs/benchmarks/mi300x-run1.md section 5). The runsheet's copy was "
        f"printed at 192e9 bytes and a 7% shortfall. docs/SLO.md sections 6 "
        f"and 10.",
        f"{'concurrency':>12}{'bytes/step':>12}{'TPOT floor':>12}"
        f"{'inside 50 ms':>14}{'KV seat @ 4 200':>17}{'vs max_num_seqs':>17}",
    )
    for batch in levels:
        floor = tpot_floor(QWEN3_8B, MI300X, batch, ctx)
        gb = decode_step_bytes(QWEN3_8B, batch, ctx) / 1e9
        inside = "yes" if floor.seconds <= TPOT_TARGET else "no"
        if batch <= seats_corrected:
            fits = "fits"
        elif batch <= seats_derived:
            fits = "fits?"       # inside the clean arithmetic, outside the corrected pool
        else:
            fits = "no seat"
        queue = "queues" if batch > seq_cap else "runs"
        print(f"{batch:>12}{gb:>9.2f} GB{floor.seconds * 1e3:>9.2f} ms"
              f"{inside:>14}{fits:>17}{queue:>17}")

    print()
    print(f"  KV pool {pool:,.0f} tokens derived, ~{pool * (1 - POOL_SHORTFALL):,.0f} "
          f"after the {POOL_SHORTFALL:.1%} shortfall -> maximum concurrency "
          f"{pool / 9000:.1f}x derived, ~{pool * (1 - POOL_SHORTFALL) / 9000:.1f}x "
          f"corrected, at max_model_len 9 000 (logged: docs/benchmarks/mi300x-run1.md)")


def fleet_model(model: Model, replicas: int) -> Model:
    """The aggregate a card sees when `replicas` engines serve one model on it.

    Only params_total moves -- each engine reads its own weights, a token's FLOPs
    and KV bytes do not -- so it holds for engines on ONE accelerator only.
    """
    if replicas < 1:
        raise ValueError("a fleet has at least one engine")
    return replace(model,
                   name=f"{model.name} x{replicas} on one card",
                   params_total=model.params_total * replicas)


def fleet_router_table() -> None:
    """What a second engine costs and what affinity buys back -- runsheet mi300x-run-3.

    Capacity arithmetic only, so no interference fit is needed on the MI300X. The
    working set sets the regime: below the pool's room for prefixes round_robin
    costs space, above it the cost moves into h (docs/SLO.md section 6).
    """
    ctx, prompt = 4200, 4000
    replicas = FLEET_REPLICAS
    prefix = SHARED_PREFIX_TOKENS
    nominal_h = prefix / prompt
    per_engine = FLEET_CONCURRENCY // replicas
    working_sets = (32, 64, 128, 256, 512)

    fleet = fleet_model(QWEN3_8B, replicas)
    pool_one = kv_cache_tokens(QWEN3_8B, MI300X, GMU)
    pool_fleet = kv_cache_tokens(fleet, MI300X, GMU)
    cap_one = concurrency_ceiling(QWEN3_8B, MI300X, ctx, GMU)
    cap_fleet = concurrency_ceiling(fleet, MI300X, ctx, GMU)
    lat_one = max_num_seqs_from_slo(QWEN3_8B, MI300X, prompt, TPOT_TARGET)
    lat_fleet = max_num_seqs_from_slo(fleet, MI300X, prompt, TPOT_TARGET)

    # The K of the hit-rate model below: prefixes one engine can still hold once
    # its live seats have reserved theirs. A prediction; the log outranks it.
    pool_per_engine = kv_cache_tokens(QWEN3_8B, MI300X, GMU / replicas)
    live = per_engine * ctx
    room = (pool_per_engine * (1 - POOL_SHORTFALL) - live) / prefix

    table_header(
        "TABLE 11: a fleet on one card, and what prefix affinity buys back",
        f"{replicas} engines at gpu_memory_utilization {GMU / replicas:.2f} "
        f"each against one at {GMU:.2f}, MI300X, uncalibrated eff_mem "
        f"{MI300X.achieved_bandwidth} and mfu {MI300X.mfu}, seats of {ctx} "
        f"reserved tokens. A second engine reads and stores a second copy of "
        f"the {QWEN3_8B.weights_bytes / 1e9:.1f} GB of weights, so the fleet "
        f"pays for them twice on both limits -- that is the bill below, and it "
        f"is charged whatever the routing policy is. What the policy decides is "
        f"the second half: round_robin puts every prefix on every engine, "
        f"affinity puts it on one. Capacity arithmetic throughout, so no "
        f"interference fit is borrowed from another card. Capacity counts a "
        f"{ctx}-token seat and latency a {prompt}-token read, which is why "
        f"'binds' says capacity where one context would tie. docs/SLO.md "
        f"section 6, channel 2.",
        f"{'arrangement':>14}{'KV pool':>14}{'latency seats':>16}"
        f"{'capacity seats':>17}{'binds':>12}",
    )
    for label, pool, lat, cap in (
            ("1 engine", pool_one, lat_one, cap_one),
            (f"{replicas} engines", pool_fleet, lat_fleet, cap_fleet)):
        binds = "capacity" if cap <= lat else "latency"
        print(f"{label:>14}{pool:>14,.0f}{lat:>16}{cap:>17}{binds:>12}")

    bill = cap_one - cap_fleet
    recovered_per_prefix = (replicas - 1) * prefix / ctx
    break_even = bill / recovered_per_prefix

    print()
    print(f"  the bill: {bill} seats of capacity (a {ctx}-token seat) and "
          f"{lat_one - lat_fleet} of latency ({prompt} tokens of context), "
          f"both of them the second {QWEN3_8B.weights_bytes / 1e9:.1f} GB "
          f"divided by what a seat costs in that limit")
    # The same bill in block A's unit: unlike a seat count, this step difference
    # compares directly against a median ITL at matched total concurrency.
    step_one = tpot_floor(QWEN3_8B, MI300X, FLEET_CONCURRENCY, prompt)
    step_fleet = tpot_floor(fleet, MI300X, FLEET_CONCURRENCY, prompt)
    print(f"  at {FLEET_CONCURRENCY} seats across the fleet the decode step "
          f"goes {step_one.seconds * 1e3:.2f} -> {step_fleet.seconds * 1e3:.2f} ms "
          f"(+{(step_fleet.seconds - step_one.seconds) * 1e3:.2f} ms, "
          f"+{step_fleet.seconds / step_one.seconds - 1:.0%}), the second "
          f"weights read at eff_mem {MI300X.achieved_bandwidth}")
    print(f"  one engine holds {pool_per_engine:,.0f} tokens derived, "
          f"~{pool_per_engine * (1 - POOL_SHORTFALL):,.0f} after the "
          f"{POOL_SHORTFALL:.1%} shortfall; at {per_engine} live seats that "
          f"leaves room for {room:.0f} retained prefixes of {prefix:,} tokens")
    print()
    print(f"  the working set, at {FLEET_CONCURRENCY} seats across the fleet "
          f"({per_engine} per engine) and a nominal h of {nominal_h:.2f}:")
    print()
    print(f"{'prefixes N':>12}{'per engine: rr / prefix':>26}{'h rr':>10}"
          f"{'h prefix':>11}{'seats recovered':>18}")
    for n in working_sets:
        seen_rr = min(n, room)
        seen_prefix = min(n / replicas, room)
        h_rr = nominal_h * min(1.0, room / n)
        h_prefix = nominal_h * min(1.0, room * replicas / n)
        # Zero once both policies saturate the pool: the saving moves into h.
        recovered = replicas * (seen_rr - seen_prefix) * prefix / ctx
        print(f"{n:>12}{f'{seen_rr:.0f} / {seen_prefix:.0f}':>26}{h_rr:>10.3f}"
              f"{h_prefix:>11.3f}{recovered:>18.1f}")

    print()
    print(f"  affinity repays the fleet's {bill}-seat bill at N >= "
          f"{break_even:.1f} prefixes ({recovered_per_prefix:.2f} seats each): "
          f"below that, two engines on one card are a loss that no routing "
          f"policy recovers")
    print(f"  the h columns assume a prefix is as likely to be asked for next "
          f"as any other. The harness draws them in strict rotation instead "
          f"(bench/scenarios/__init__.py), which is LRU's worst case: both "
          f"columns fall to 0 rather than to a share, round_robin past N = "
          f"{room:.0f} and affinity past N = {room * replicas:.0f}")


RUN2_SWEEP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "sweep", "mi300x-run-2-{}.json")


def run2_sweep() -> tuple[dict, dict]:
    """The serve and bench rows runsheet mi300x-run-2 launches, as the sweep reads them."""
    with open(RUN2_SWEEP.format("serve")) as serve, open(RUN2_SWEEP.format("bench")) as bench:
        return json.load(serve), json.load(bench)


def interference_tpot(accel, batch: int, prompt: int, output: int) -> float:
    """TPOT in seconds by run 1's interference model, fed floors at the mean context."""
    step = tpot_floor(QWEN3_8B, accel, batch, prompt + output // 2).seconds
    alone = ttft_floor(QWEN3_8B, accel, prompt).seconds
    return tpot_with_interference(step, batch, alone, output)


def run2_running(budget: int, limits: dict, prompt: int, output: int) -> tuple[float, str]:
    """A saturated row's running count, and every limit that sets it: a row cannot split a tie."""
    limits = {"token budget": token_budget_ceiling(budget, prompt, output), **limits}
    low = min(limits.values())
    return low, " = ".join(k for k, v in limits.items() if v == low)


def mi300x_run2_table() -> None:
    """Run 1's coefficients faced, and what each lever should move -- runsheet mi300x-run-2.

    MI300X_RUN1 throughout (docs/SLO.md section 9); rows and caps come from the
    sweep's JSON, run 1's figures from its read-out.
    """
    import measured_mi300x_run1 as run1    # the read-out, not a copy of its numbers

    prompt, out = 4000, 200
    accel = MI300X_RUN1
    serve, bench = run2_sweep()
    budgets = sorted({row["max_num_batched_tokens"] for row in serve.values()})
    (seq_cap,) = {row["max_num_seqs"] for row in serve.values()}
    levels = sorted(row["max_concurrency"] for row in bench.values()
                    if "random_input_len" not in row)
    lengths = sorted(row["random_input_len"] for row in bench.values()
                     if "random_input_len" in row)
    saturated, levels = levels[-1], levels[:-1]

    pool = kv_cache_tokens(QWEN3_8B, accel, GMU) * (1 - POOL_SHORTFALL)
    seats = int(pool // (prompt + out))
    alone = ttft_floor(QWEN3_8B, accel, prompt).seconds

    table_header(
        "TABLE 12: MI300X run 2 -- run 1's coefficients faced, and the levers",
        f"MI300X at eff_mem {accel.achieved_bandwidth} and mfu {accel.mfu}, both "
        f"fitted by run 1 and faced here for the first time. The decode step is "
        f"taken at the mean context, {prompt} + {out // 2}; TPOT p50 adds the "
        f"other seats' prefills spread over {out} decode steps, the model run 1 "
        f"wrote after the fact, which carries no budget term. So it predicts the "
        f"same TPOT p50 at every max_num_batched_tokens for as long as prefill "
        f"costs the same, and the running count is what the budget moves. "
        f"docs/benchmarks/mi300x-run1.md sections 5-6.",
        f"{'row':>14}{'decode step':>14}{'TTFT alone':>13}"
        f"{'TPOT p50':>11}{'inside 50 ms':>15}",
    )
    for batch in levels:
        step = tpot_floor(QWEN3_8B, accel, batch, prompt + out // 2).seconds
        tpot = interference_tpot(accel, batch, prompt, out)
        ttft = f"{alone * 1e3:.1f} ms" if batch == 1 else "-"
        inside = "yes" if tpot <= TPOT_TARGET else "no"
        print(f"{f'c{batch:03d}':>14}{step * 1e3:>11.2f} ms{ttft:>13}"
              f"{tpot * 1e3:>8.2f} ms{inside:>15}")
    for length in lengths:
        step = tpot_floor(QWEN3_8B, accel, 1, length + out // 2).seconds
        ttft = ttft_floor(QWEN3_8B, accel, length).seconds
        print(f"{f'c001-in{length}':>14}{step * 1e3:>11.2f} ms"
              f"{f'{ttft * 1e3:.1f} ms':>13}{'-':>11}{'-':>15}")
    crossing = next(c for c in range(1, seq_cap + 1)
                    if interference_tpot(accel, c, prompt, out) > TPOT_TARGET)
    print()
    print(f"  TPOT p50 crosses 50 ms at c = {crossing}; run 1 placed it near 29 "
          f"from measured rows. p99 is not modelled: run 1's crossed between "
          f"8 and 32, near 20 by interpolation")

    print()
    print(f"{'budget':>10}{'prefill steps':>15}{'ceiling':>10}"
          f"{f'running at c{saturated}':>18}{'set by':>30}")
    print("-" * 83)
    for budget in budgets:
        steps = "/".join(str(math.ceil(n / budget)) for n in sorted((*lengths, prompt)))
        running, binds = run2_running(
            budget, {"max_num_seqs": seq_cap, "concurrency": saturated,
                     "KV pool": seats}, prompt, out)
        print(f"{budget:>10}{steps:>15}"
              f"{token_budget_ceiling(budget, prompt, out):>10.1f}"
              f"{running:>18.0f}{binds:>30}")
    print()
    print(f"  prefill steps for {', '.join(str(n) for n in sorted((*lengths, prompt)))}"
          f"-token prompts. The pool seats {seats} at {prompt + out} tokens after "
          f"the {POOL_SHORTFALL:.1%} shortfall at the smallest budget; a larger "
          f"budget raises the peak activation it pays for, so checkpoint A is "
          f"read per serve row")

    print()
    print(f"{'mfu':>10}{'TTFT alone, 4 000':>20}{'seats inside 50 ms, p50':>26}")
    print("-" * 56)
    for mfu in (accel.mfu, 0.20, 0.25, 0.30, MI300X.mfu):
        faster = replace(accel, mfu=mfu)
        ttft = ttft_floor(QWEN3_8B, faster, prompt).seconds
        inside = max(c for c in range(1, seq_cap + 1)
                     if interference_tpot(faster, c, prompt, out) <= TPOT_TARGET)
        print(f"{mfu:>10.3f}{ttft * 1e3:>17.1f} ms{inside:>26}")

    l40s = run1.L40S_COST["plateau"]
    plateau = run1.rows()[f"c{saturated:03d}"]["output_tps"]
    even = MI300X_HOURLY / (l40s * 3600) * 1e6
    print()
    print(f"  the plateau breaks even with the L40S's ${l40s}/1M at {even:.0f} "
          f"output tok/s, {even / plateau - 1:+.1%} on run 1's {plateau:.1f} at "
          f"c{saturated}, ${MI300X_HOURLY}/h")


def _finite(value: float | None) -> float | None:
    """inf -> None: a ratio is inf at zero seats, and JSON has no Infinity."""
    return None if value is None or math.isinf(value) else value


def what_if_point(accelerator: str = "l40s-run1",
                  model: str = DEFAULT_MODEL,
                  context_len: int = 4200,
                  prompt_tokens: int = 4000,
                  tpot_target: float = TPOT_TARGET,
                  ttft_target: float = TTFT_TARGET,
                  gpu_memory_utilization: float = GMU,
                  kv_dtype_bytes: int = 2,
                  hourly_rate: float | None = None,
                  hit_rate: float = 0.0) -> dict:
    """One operating point, as data. Everything else is a rendering of this.

    The keys are a contract frozen by bench/tests/test_predictions_json.py: the
    site's JavaScript reads them, and a renamed key there is a blank cell, not an
    error. "error" is a sentence rather than an exception because a slider can be
    dragged past any operating point. Every figure is a floor.
    """
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError("hit_rate is a share of prompt tokens, so 0 <= h <= 1")
    accel = ACCELERATORS[accelerator]
    model_key, base = model, MODELS[model]
    model = base if kv_dtype_bytes == 2 else replace(
        base, name=f"{base.name} FP8 KV", kv_dtype_bytes=kv_dtype_bytes)
    # The rate is a contract figure, not physics, so it follows the card only as
    # a default: any rate can be passed, and on reserved capacity it should be.
    if hourly_rate is None:
        hourly_rate = HOURLY_RATES[accelerator][0]

    point: dict = {
        "inputs": {
            "accelerator": accelerator,
            "model": model_key,
            "context_len": context_len,
            "prompt_tokens": prompt_tokens,
            "tpot_target_s": tpot_target,
            "ttft_target_s": ttft_target,
            "gpu_memory_utilization": gpu_memory_utilization,
            "kv_dtype_bytes": kv_dtype_bytes,
            "hourly_rate": hourly_rate,
            "hit_rate": hit_rate,
        },
        "model_name": model.name,
        "accelerator_name": accel.name,
        "coefficients": {
            "eff_mem": accel.achieved_bandwidth,
            "mfu": accel.mfu,
            "provenance": accel.provenance,
        },
        "kv_pool_tokens": None,
        "seats": None,
        "tpot_floor_at_max_num_seqs": None,
        "ttft_floor": None,
        "ttft_floor_uncached": None,
        "aggregate_tokens_per_sec": None,
        "cost_per_1m_output_tokens": None,
        "service": None,
        "error": None,
    }

    # Prefill first: it needs no KV pool, so it has an answer even where the
    # decode side has none.
    prefill = ttft_floor(model, accel, prompt_tokens)
    # the floor bench/harness.py's check_prefill_floor gates a level on
    uncached = max(1, round(prompt_tokens * (1.0 - hit_rate)))
    prefill_uncached = ttft_floor(model, accel, uncached)
    point["ttft_floor"] = {
        "seconds": prefill.seconds,
        "bound_by": prefill.bound_by,
        "ratio": _finite(prefill.ratio),
        "share_of_target": prefill.seconds / ttft_target,
    }
    point["ttft_floor_uncached"] = {
        "seconds": prefill_uncached.seconds,
        "uncached_tokens": uncached,
        "share_of_target": prefill_uncached.seconds / ttft_target,
    }

    try:
        pool = kv_cache_tokens(model, accel, gpu_memory_utilization)
    except ValueError as exc:
        point["error"] = str(exc)
        return point
    seats = max_num_seqs(model, accel, context_len, tpot_target,
                         gpu_memory_utilization)
    point["kv_pool_tokens"] = pool
    point["seats"] = {
        "by_capacity": seats.by_capacity,
        "by_latency": seats.by_latency,
        "max_num_seqs": seats.sequences,
        "bound_by": seats.bound_by,
        "gap": _finite(seats.ratio) if seats.sequences else None,
    }
    if seats.sequences == 0:
        point["error"] = (
            f"Not one sequence fits: {context_len:,} tokens of reserved context "
            f"against a {pool / 1e6:.2f} M token pool, or a TPOT target below the "
            f"batch-1 decode step. Lower the context length, loosen the target, "
            f"raise gpu_memory_utilization, or take a bigger card -- there is no "
            f"operating point here to price.")
        return point

    decode = tpot_floor(model, accel, seats.sequences, context_len)
    total = aggregate_tokens_per_sec(model, accel, seats.sequences, context_len)
    point["tpot_floor_at_max_num_seqs"] = {
        "seconds": decode.seconds,
        "bound_by": decode.bound_by,
        "ratio": _finite(decode.ratio),
        "share_of_target": decode.seconds / tpot_target,
    }
    point["aggregate_tokens_per_sec"] = total
    point["cost_per_1m_output_tokens"] = cost_per_1m_tokens(total, hourly_rate)

    fit = INTERFERENCE_FITS.get(accelerator) if model_key == INTERFERENCE_MODEL else None
    if fit is not None:
        slope, intercept, provenance = fit
        # At context_len, not table 8's mean occupancy of 4 100: ~0.3 seat apart
        # by convention. Floored, never rounded, and the pool must still seat them.
        by_interference = seats_under_prefill_interference(
            model, accel, context_len, tpot_target, slope, intercept, hit_rate)
        shipped = max(math.floor(by_interference), 0)
        bound_by = "interference"
        if seats.by_capacity < shipped:
            shipped, bound_by = seats.by_capacity, "capacity"
        service: dict = {
            "seats_at_hit_rate": by_interference,
            "seats_shipped": shipped,
            "bound_by": bound_by,
            "tpot_floor_at_seats_s": None,
            "aggregate_tokens_per_sec": None,
            "cost_per_1m_output_tokens": None,
            "fit": {"slope_s_per_seat": slope, "intercept_s": intercept,
                    "provenance": provenance},
            "note": ("a floor, not an estimate: at h = 0.8 run 3 measured 37.8 "
                     "seats against 24.3 predicted (docs/SLO.md section 6)"
                     if hit_rate > 0.5 else None),
        }
        if shipped > 0:
            step = tpot_floor(model, accel, shipped, context_len).seconds
            rate = aggregate_tokens_per_sec(model, accel, shipped, context_len)
            service["tpot_floor_at_seats_s"] = step
            service["aggregate_tokens_per_sec"] = rate
            service["cost_per_1m_output_tokens"] = cost_per_1m_tokens(rate, hourly_rate)
        point["service"] = service
    return point


def what_if(accelerator: str = "l40s-run1",
            model: str = DEFAULT_MODEL,
            context_len: int = 4200,
            prompt_tokens: int = 4000,
            tpot_target: float = TPOT_TARGET,
            ttft_target: float = TTFT_TARGET,
            gpu_memory_utilization: float = GMU,
            kv_dtype_bytes: int = 2,
            hourly_rate: float | None = None,
            hit_rate: float = 0.0) -> None:
    """One operating point, named on the command line, answered three ways.

    Tables 3, 4 and 5 on the reader's own point (route 1 of docs/audience.md). A
    printer only: every number comes from what_if_point(), and the first nine
    lines, which docs/audience.md quotes, are kept byte-identical by a test.
    """
    point = what_if_point(accelerator=accelerator, model=model,
                          context_len=context_len, prompt_tokens=prompt_tokens,
                          tpot_target=tpot_target, ttft_target=ttft_target,
                          gpu_memory_utilization=gpu_memory_utilization,
                          kv_dtype_bytes=kv_dtype_bytes,
                          hourly_rate=hourly_rate, hit_rate=hit_rate)
    hourly_rate = point["inputs"]["hourly_rate"]

    table_header(
        f"WHAT IF: {point['model_name']} on {point['accelerator_name']}",
        f"One operating point: {context_len:,} tokens of context reserved per "
        f"seat, a {prompt_tokens:,}-token prompt, gpu_memory_utilization "
        f"{gpu_memory_utilization:.2f}, against a TPOT target of "
        f"{tpot_target * 1e3:.0f} ms and a TTFT target of "
        f"{ttft_target * 1e3:.0f} ms. Floors, not forecasts: docs/SLO.md "
        "sections 4, 6 and 7 derive every line below, and the coefficients "
        "printed first decide how much of the card the arithmetic is allowed "
        "to assume.",
    )

    coef = point["coefficients"]
    print(f"  {'coefficients':<22}eff_mem {coef['eff_mem']:.2f}, "
          f"mfu {coef['mfu']:.3f} -- {coef['provenance']}")
    if point["seats"] is None:
        print(f"\n  {point['error']}")
        print()
        return
    seats = point["seats"]
    print(f"  {'KV pool':<22}{point['kv_pool_tokens'] / 1e6:.2f} M tokens")
    print(f"  {'seats by capacity':<22}{seats['by_capacity']}")
    print(f"  {'seats by latency':<22}{seats['by_latency']}")
    gap = f", gap {seats['gap']:.2f}x" if seats["gap"] is not None else ""
    print(f"  {'max_num_seqs':<22}{seats['max_num_seqs']}"
          f"   bound by {seats['bound_by']}{gap}")

    if point["error"]:
        print(f"\n  {point['error']}")
        print()
        return

    decode = point["tpot_floor_at_max_num_seqs"]
    prefill = point["ttft_floor"]
    n = seats["max_num_seqs"]
    print(f"  {'TPOT floor':<22}{decode['seconds'] * 1e3:.2f} ms at "
          f"{n} seats   bound by {decode['bound_by']}, "
          f"{decode['ratio']:.1f}x   ({decode['share_of_target']:.0%} of target)")
    print(f"  {'TTFT floor':<22}{prefill['seconds'] * 1e3:.1f} ms at "
          f"{prompt_tokens:,} tokens   bound by {prefill['bound_by']}, "
          f"{prefill['ratio']:.1f}x   ({prefill['share_of_target']:.0%} of target)")
    print(f"  {'aggregate':<22}{point['aggregate_tokens_per_sec']:,.0f} tok/s total, "
          f"{1 / decode['seconds']:.0f} tok/s per seat")
    print(f"  {'cost':<22}${point['cost_per_1m_output_tokens']:.3f} / 1M "
          f"tokens at ${hourly_rate:.2f}/h")

    # The service view: what can be promised once prefill lands inside decode steps.
    h = f"h={hit_rate:.2f}"
    unc = point["ttft_floor_uncached"]
    print(f"  {f'TTFT floor at {h}':<22}{unc['seconds'] * 1e3:.1f} ms at "
          f"{unc['uncached_tokens']:,} uncached tokens   "
          f"({unc['share_of_target']:.0%} of target)")
    service = point["service"]
    if service is None:
        print(f"  {f'seats at {h}':<22}not derivable: no interference fit on this "
              f"card (docs/SLO.md section 6)")
    else:
        print(f"  {f'seats at {h}':<22}{service['seats_at_hit_rate']:.1f} by prefill "
              f"interference -> ship {service['seats_shipped']}"
              + ("" if service["bound_by"] == "interference"
                 else ", capped by capacity")
              + f"   fit: {service['fit']['provenance']}")
        if service["note"]:
            print(f"  {'':<22}{service['note']}")
        if service["seats_shipped"] > 0:
            print(f"  {'service':<22}{service['aggregate_tokens_per_sec']:,.0f} tok/s, "
                  f"${service['cost_per_1m_output_tokens']:.3f} / 1M tokens at "
                  f"{service['seats_shipped']} seats")
        else:
            print(f"  {'service':<22}no seat meets the target once interference "
                  f"is priced in")
    print()


def main(argv=None) -> int:
    """No arguments prints the twelve tables; --what-if prints one chosen point.

    The tables take no flags, because docs/SLO.md quotes their rows.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--what-if", action="store_true",
                        help="print one operating point instead of the twelve "
                             "tables, from the parameters below")
    parser.add_argument("--accelerator", choices=sorted(ACCELERATORS),
                        help="which card, and which coefficients with it "
                             "(default l40s-run1, the only entry backed by a "
                             "measurement)")
    parser.add_argument("--model", choices=sorted(MODELS),
                        help="which architecture to price (default qwen3-8b, "
                             "the one this stack serves and the only one any "
                             "run here has measured)")
    parser.add_argument("--context-len", type=int,
                        help="tokens of KV reserved per seat: prompt plus "
                             "the output it will grow (default 4200, this "
                             "repository's own point -- 4000 in, 200 out)")
    parser.add_argument("--prompt-tokens", type=int,
                        help="prompt length the TTFT floor is taken at "
                             "(default 4000)")
    parser.add_argument("--tpot-ms", type=float,
                        help="TPOT target in milliseconds (default 50)")
    parser.add_argument("--ttft-ms", type=float,
                        help="TTFT target in milliseconds (default 300)")
    parser.add_argument("--gpu-memory-utilization", type=float,
                        help="the vLLM knob, as served (default 0.90)")
    parser.add_argument("--kv-dtype-bytes", type=int, choices=(1, 2),
                        help="2 for a BF16 KV cache, 1 for FP8 (default 2)")
    parser.add_argument("--hourly-rate", type=float,
                        help="$/h for the cost line; a contract figure, not "
                             "physics (default: the card's rate in this file)")
    parser.add_argument("--hit-rate", type=float,
                        help="prefix cache hit rate h, 0..1: the share of the "
                             "prompt the cache serves. Moves the uncached TTFT "
                             "floor and the service seat count, and nothing "
                             "else -- a decode step reads the same bytes "
                             "whatever h is (default 0)")
    parser.add_argument("--json", action="store_true",
                        help="print the operating point as JSON instead of "
                             "text -- the same dict the site's golden grid is "
                             "rows of")
    args = parser.parse_args(argv)

    # A parameter without --what-if is an error, not a silent print of the fixed
    # tables; --json is store_true, so it is False, not None, when absent.
    point = {k: v for k, v in vars(args).items()
             if k not in ("what_if", "json") and v is not None}
    if (point or args.json) and not args.what_if:
        flags = [f"--{k.replace('_', '-')}" for k in point] + \
                (["--json"] if args.json else [])
        parser.error(f"{', '.join(flags)} "
                     f"{'mean' if len(flags) > 1 else 'means'} nothing without "
                     "--what-if: the twelve tables are fixed operating points "
                     "and take no parameters")

    if args.what_if:
        kwargs = {k: v for k, v in point.items()
                  if k not in ("tpot_ms", "ttft_ms")}
        if args.tpot_ms is not None:
            kwargs["tpot_target"] = args.tpot_ms / 1e3
        if args.ttft_ms is not None:
            kwargs["ttft_target"] = args.ttft_ms / 1e3
        if args.json:
            # allow_nan=False: an inf that slipped past _finite() is a bug, and
            # a crash here is better than a file JSON.parse rejects later.
            print(json.dumps(what_if_point(**kwargs), indent=2, sort_keys=True,
                             allow_nan=False))
        else:
            what_if(**kwargs)
        return 0

    sizes()
    for table in (decode_table, latency_limit_table, shipping_table,
                  prefill_table, cost_table, sweep_concurrency_table,
                  sweep_length_table, prefix_cache_table,
                  mi300x_sweep_table):
        table()
    sweep_length_table(MI300X, batch=1, number=10)
    fleet_router_table()
    mi300x_run2_table()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
