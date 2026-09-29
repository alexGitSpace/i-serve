"""MI300X run 3's levels: a fleet's bill on one card, and the router arm.

Data only; predictions are table 11 of bench/predictions.py, the procedure is
docs/benchmarks/runsheets/mi300x-run-3.md. Output is 200 tokens, as everywhere.
"""

from . import Workload

SEED = 20261001
PROMPT_TOKENS = 4_000
OUTPUT_TOKENS = 200

# Odd on purpose: under the rotation an even N makes round_robin affinity
# (runsheet section 0).
WORKING_SETS = (31, 63, 127, 255, 511)


def _router_level(num_prefixes: int) -> Workload:
    """64 seats across the fleet, h = 0.8 per prompt, four passes of the rotation."""
    return Workload(
        name=f"router-n{num_prefixes:03d}",
        prompt_tokens=PROMPT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        num_prompts=max(256, 4 * num_prefixes),
        mode="closed",
        concurrency=64,
        hit_rate_target=0.8,
        num_prefixes=num_prefixes,
        # One seed per working set, or N = 31's prefixes are the first 31 of
        # N = 63's and a level inherits the one before it.
        seed=SEED + num_prefixes,
    )


def _unique_level(concurrency: int, num_prompts: int) -> Workload:
    return Workload(
        name=f"fleet-c{concurrency:03d}-h00",
        prompt_tokens=PROMPT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        num_prompts=num_prompts,
        mode="closed",
        concurrency=concurrency,
        # Distinct seeds: with the cache on, c = 1's prompts would otherwise
        # be c = 64's first twenty, and served from it.
        seed=SEED + 10_000 + concurrency,
    )


ROUTER_ARM = tuple(_router_level(n) for n in WORKING_SETS)

# Block A. c = 1 first: its median ITL says which droplet this is (section 3).
# The pair splits c = 64 as two harnesses at c = 32, same request count in all.
FLEET_BILL = (_unique_level(1, 20), _unique_level(64, 192))
FLEET_BILL_HALF = (_unique_level(32, 96),)

# One scenario per working set too: the runsheet alternates which arm goes
# first per N, and that needs the router relaunched between levels.
SCENARIOS: dict[str, tuple[Workload, ...]] = {
    "router-arm": ROUTER_ARM,
    **{level.name.replace("router-", "router-arm-"): (level,) for level in ROUTER_ARM},
    "fleet-bill": FLEET_BILL,
    "fleet-bill-half": FLEET_BILL_HALF,
}
