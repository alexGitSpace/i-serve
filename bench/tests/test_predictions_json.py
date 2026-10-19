"""What --what-if promises the page: a dict whose shape and figures do not move.

Since 2026-09-13 `bench/predictions.py --what-if` is a printer over
`what_if_point()`, and the calculator on the Pages site is a JavaScript port
checked against rows of that same function (`site/data/golden.json`). Two
things can therefore break silently: a key renamed here becomes `undefined` in
the browser, which is a blank cell and not an error; and a change to the text
printer could drift from the dict it claims to render. This file holds both,
plus the nine lines docs/audience.md quotes to the reader, byte for byte.

No dependency on pytest, which is not installed here -- plain functions, plain
asserts, and a runner at the bottom:

    python3 bench/tests/test_predictions_json.py
"""

# Two ways in, and both have to work. `pytest bench/` gets bench/ on the import
# path from tests/conftest.py; `python3 bench/tests/test_roofline.py` on a rented
# pod, where pytest is not installed, gets only this directory. The three lines
# below are what make the second one work, and they are here rather than in
# conftest.py for exactly that reason.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import contextlib
import io
import json
import math
import re

import predictions
from predictions import (HOURLY_RATES, INTERFERENCE_FITS, SLO_CLASSES,
                         what_if, what_if_point)
from roofline import ACCELERATORS, L40S_RUN1, QWEN3_8B, max_num_seqs_from_slo

ROOT = _pathlib.Path(__file__).resolve().parent.parent.parent

# The contract with site/calc.js. A key added here must be added there; a key
# removed here breaks the page in silence, so this set is spelled out rather
# than derived.
POINT_KEYS = {
    "inputs", "model_name", "accelerator_name", "coefficients", "kv_pool_tokens",
    "state_tokens_per_seat", "seats", "tpot_floor_at_max_num_seqs", "ttft_floor",
    "ttft_floor_uncached", "aggregate_tokens_per_sec", "cost_per_1m_output_tokens",
    "service", "error",
}
INPUT_KEYS = {
    "accelerator", "model", "context_len", "prompt_tokens", "tpot_target_s",
    "ttft_target_s", "gpu_memory_utilization", "kv_dtype_bytes", "hourly_rate",
    "hit_rate",
}
SERVICE_KEYS = {
    "seats_at_hit_rate", "seats_shipped", "bound_by", "tpot_floor_at_seats_s",
    "aggregate_tokens_per_sec", "cost_per_1m_output_tokens", "fit", "note",
}

# The nine lines docs/audience.md quotes from, for the default point. Held as
# text, not recomputed: the point of the test is that the *rendering* did not
# move, and a recomputation would follow any drift it was meant to catch.
NINE_LINES = """\
  coefficients          eff_mem 0.83, mfu 0.439 -- measured (run 1, 2026-08-18); survived runs 2-3
  KV pool               0.18 M tokens
  seats by capacity     43
  seats by latency      31
  max_num_seqs          31   bound by latency, gap 1.39x
  TPOT floor            49.64 ms at 31 seats   bound by memory, 18.3x   (99% of target)
  TTFT floor            349.8 ms at 4,000 tokens   bound by compute, 14.8x   (117% of target)
  aggregate             624 tok/s total, 20 tok/s per seat
  cost                  $0.440 / 1M tokens at $0.99/h"""


def _printed(**kwargs) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        what_if(**kwargs)
    return out.getvalue()


def test_what_if_point_carries_exactly_the_keys_the_page_reads():
    point = what_if_point()
    assert set(point) == POINT_KEYS, set(point) ^ POINT_KEYS
    assert set(point["inputs"]) == INPUT_KEYS, set(point["inputs"]) ^ INPUT_KEYS
    assert set(point["service"]) == SERVICE_KEYS, set(point["service"]) ^ SERVICE_KEYS
    assert set(point["seats"]) == {"by_capacity", "by_latency", "max_num_seqs",
                                   "bound_by", "gap"}
    assert set(point["coefficients"]) == {"eff_mem", "mfu", "provenance"}
    assert set(point["tpot_floor_at_max_num_seqs"]) == {
        "seconds", "bound_by", "ratio", "share_of_target", "experts_read_share"}


def test_the_first_nine_lines_are_the_ones_audience_md_quotes():
    """docs/audience.md pastes the max_num_seqs line; the rest of the block is
    what a reader sees first. Byte-identical to what --what-if printed before
    it grew the service view.
    """
    lines = _printed().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  coefficients"))
    assert "\n".join(lines[start:start + 9]) == NINE_LINES, \
        "\n".join(lines[start:start + 9])


def test_the_text_is_a_rendering_of_the_dict():
    """Every number the printer shows is in the dict, at the precision shown.

    The check that stops the two from drifting: a printer that computed one
    figure of its own would print a number the page could not reproduce.
    """
    point = what_if_point(hit_rate=0.8)
    text = _printed(hit_rate=0.8)
    seats, service = point["seats"], point["service"]
    expected = [
        f"{point['kv_pool_tokens'] / 1e6:.2f} M tokens",
        f"seats by capacity     {seats['by_capacity']}",
        f"seats by latency      {seats['by_latency']}",
        f"max_num_seqs          {seats['max_num_seqs']}   bound by {seats['bound_by']}",
        f"{point['tpot_floor_at_max_num_seqs']['seconds'] * 1e3:.2f} ms at",
        f"{point['ttft_floor']['seconds'] * 1e3:.1f} ms at",
        f"{point['aggregate_tokens_per_sec']:,.0f} tok/s total",
        f"${point['cost_per_1m_output_tokens']:.3f} / 1M",
        f"{point['ttft_floor_uncached']['seconds'] * 1e3:.1f} ms at "
        f"{point['ttft_floor_uncached']['uncached_tokens']:,} uncached tokens",
        f"{service['seats_at_hit_rate']:.1f} by prefill interference -> ship "
        f"{service['seats_shipped']}",
        f"${service['cost_per_1m_output_tokens']:.3f} / 1M tokens at "
        f"{service['seats_shipped']} seats",
    ]
    missing = [e for e in expected if e not in text]
    assert not missing, missing


def test_the_state_and_expert_lines_are_renderings_too_and_only_where_they_apply():
    """A dense model prints neither; the hybrid MoE prints both, from the dict."""
    kwargs = dict(model="qwen3.6-35b-a3b", accelerator="mi300x-run2")
    point, text = what_if_point(**kwargs), _printed(**kwargs)
    decode = point["tpot_floor_at_max_num_seqs"]
    assert f"{point['state_tokens_per_seat']:,.0f} tokens' worth of KV per seat" in text
    assert (f"{decode['experts_read_share']:.0%} per step at "
            f"{point['seats']['max_num_seqs']} seats") in text
    dense = _printed()
    assert "recurrent state" not in dense and "experts read" not in dense
    assert what_if_point()["state_tokens_per_seat"] is None


def test_hit_rate_scales_only_the_interference_term():
    """docs/SLO.md section 6: caching removes prefill work and leaves the bytes a
    decode step reads untouched. So h moves the service seats and the uncached
    floor, and nothing on the hardware side.
    """
    cold, warm, full = (what_if_point(hit_rate=h) for h in (0.0, 0.8, 1.0))
    for key in ("kv_pool_tokens", "seats", "tpot_floor_at_max_num_seqs",
                "ttft_floor", "aggregate_tokens_per_sec", "cost_per_1m_output_tokens"):
        assert cold[key] == warm[key] == full[key], key
    assert cold["service"]["seats_at_hit_rate"] < warm["service"]["seats_at_hit_rate"] \
        < full["service"]["seats_at_hit_rate"]
    # With nothing left to prefill, the interference is gone and the service
    # limit is the continuous latency limit itself -- the unfloored 31.x that
    # max_num_seqs_from_slo floors to 31.
    assert math.floor(full["service"]["seats_at_hit_rate"]) == \
        max_num_seqs_from_slo(QWEN3_8B, L40S_RUN1, 4200, 0.050) == 31
    assert cold["ttft_floor_uncached"]["seconds"] == cold["ttft_floor"]["seconds"]
    assert warm["ttft_floor_uncached"]["uncached_tokens"] == 800
    assert warm["service"]["note"] and cold["service"]["note"] is None


def test_a_card_without_a_fit_has_no_service_view():
    point = what_if_point(accelerator="mi300x")
    assert point["service"] is None
    assert "mi300x" not in INTERFERENCE_FITS
    assert "not derivable" in _printed(accelerator="mi300x")


def test_no_operating_point_is_a_sentence_not_an_exception():
    """A slider can be dragged to a target below the batch-1 step or to a memory
    share the weights do not fit in. Both are answers, and both are JSON.
    """
    too_tight = what_if_point(tpot_target=0.020)
    assert too_tight["seats"]["max_num_seqs"] == 0
    assert too_tight["seats"]["gap"] is None
    assert too_tight["error"] and too_tight["tpot_floor_at_max_num_seqs"] is None
    no_room = what_if_point(gpu_memory_utilization=0.30)
    assert no_room["seats"] is None and "no KV cache remains" in no_room["error"]
    assert no_room["ttft_floor"] is not None, "prefill needs no pool"
    for point in (too_tight, no_room, what_if_point(), what_if_point(accelerator="mi300x")):
        json.dumps(point, allow_nan=False)      # raises on a stray inf


def test_slo_classes_are_the_two_rows_of_section_two():
    body = (ROOT / "docs/SLO.md").read_text(encoding="utf-8")
    section = re.search(r"^## 2\. Targets.*?(?=^## )", body, re.M | re.S).group(0)
    assert re.search(r"^\| Interactive \| 300 ms \| 50 ms \|", section, re.M), section
    assert re.search(r"^\| Batch \| 3 000 ms \| 200 ms \|", section, re.M), section
    assert SLO_CLASSES == {"interactive": {"ttft_s": 0.300, "tpot_s": 0.050},
                           "batch": {"ttft_s": 3.000, "tpot_s": 0.200}}


def test_the_interference_fit_is_never_lent_to_a_model_it_was_not_measured_on():
    """The fit is a property of an engine, a chunk size, a card AND a model. It
    was measured decoding Qwen3-8B; a second architecture on the same card gets
    "not derivable", the same answer the MI300X gets, and for the same reason.
    """
    from predictions import INTERFERENCE_MODEL, MODELS
    measured = what_if_point(model=INTERFERENCE_MODEL)
    assert measured["service"] is not None and measured["service"]["seats_shipped"] > 0
    for key in MODELS:
        if key == INTERFERENCE_MODEL:
            continue
        point = what_if_point(model=key)
        assert point["service"] is None, key
        if point["seats"] is None:                          # does not fit the card at all
            assert "exceed" in point["error"], key
            continue
        assert point["seats"]["max_num_seqs"] > 0, key      # the floors still hold
        assert "not derivable" in _printed(model=key)


def test_a_second_architecture_changes_the_seat_count_on_the_same_card():
    """The attention layout, not "an 8B model", sets the seat count (docs/audience.md, SLO.md section 11)."""
    from roofline import QWEN3_5_9B
    assert QWEN3_8B.kv_bytes_per_token / QWEN3_5_9B.kv_bytes_per_token == 4.5
    a = what_if_point(model="qwen3-8b")["seats"]["max_num_seqs"]
    b = what_if_point(model="qwen3.5-9b")["seats"]["max_num_seqs"]
    assert b > 2 * a, (a, b)


def test_every_card_has_a_rate_and_a_sentence_saying_where_it_came_from():
    assert set(HOURLY_RATES) == set(ACCELERATORS)
    for key, (rate, provenance) in HOURLY_RATES.items():
        assert rate > 0 and provenance, key
    assert what_if_point(accelerator="mi300x")["inputs"]["hourly_rate"] == \
        HOURLY_RATES["mi300x"][0]
    assert what_if_point(hourly_rate=5.0)["inputs"]["hourly_rate"] == 5.0


def test_json_flag_prints_the_point_and_nothing_else():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = predictions.main(["--what-if", "--hit-rate", "0.8", "--json"])
    assert code == 0
    point = json.loads(out.getvalue())
    assert point == what_if_point(hit_rate=0.8)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            predictions.main(["--json"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("--json without --what-if printed the tables")


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
