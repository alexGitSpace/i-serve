"""The committed claims table is what bench/claims.py produces, and no claim reaches the ledger.

No pytest dependency:

    python3 bench/tests/test_claims.py
"""

# Run directly without pytest, this file gets only its own directory on the
# path; conftest.py is not consulted, so the insert lives here.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import claims
import ledger
from passports import PASSPORTS

EVERY = [*claims.STREAMS, *claims.UNDISCLOSED, *claims.CHECKED, *claims.REFUSED]


def test_the_committed_table_is_what_the_generator_produces():
    for path, text in claims.outputs().items():
        assert path.read_text(encoding="utf-8") == text, \
            f"{path.relative_to(claims.ROOT)} is stale: python3 bench/claims.py"


def test_twenty_one_published_figures_each_once():
    ids = [c.id for c in EVERY]
    assert len(ids) == len(set(ids)), ids
    assert len({i[:3] for i in ids}) == 21


def test_every_file_and_device_is_on_record():
    for s in claims.STREAMS:
        assert s.file in claims.FILES and s.device in PASSPORTS, s.id
    for u in claims.UNDISCLOSED:
        assert u.device in PASSPORTS and all(f in claims.FILES for f in u.candidates), u.id


def test_a_claim_is_never_a_ledger_row():
    """A claim stays in its own table (CONTRIBUTING.md, *A public claim as a row*)."""
    urls = {c.source.url for c in EVERY}
    for row in ledger.rows():
        assert not any(u in (row["sheet"], row["raw"]) for u in urls), row["run"]
    assert "claims" not in ledger.MODULES


def test_the_bytes_reproduce_the_shares_the_check_was_read_at():
    """Shares as first read on 2026-10-17 from file sizes; the tensor tables agree within 0.01."""
    first = {"c01": 0.64, "c02": 0.60, "c03": 0.64, "c03o": 0.28, "c04": 0.77, "c04q": 0.57,
             "c05": 0.77, "c06": 0.76, "x01": 0.67, "x03": 0.70, "c15v": 0.89}
    for s in claims.STREAMS:
        assert abs(s.share() - first[s.id]) <= 0.015, (s.id, round(s.share(), 3))


def test_no_disclosed_single_stream_claim_crosses_the_ceiling():
    assert all(s.verdict() == "within ceiling" for s in claims.STREAMS)


def test_a_step_reads_every_weight_but_the_embedding_unless_the_head_is_tied():
    roles = claims.FILES["llama-3.1-8b-instruct.q4_k_m.ollama"]["roles"]
    assert claims.read_bytes("llama-3.1-8b-instruct.q4_k_m.ollama") == \
        roles["decoder"]["bytes"] + roles["lm_head"]["bytes"]


if __name__ == "__main__":
    # The same runner its siblings carry, for the same reason: a rented pod has
    # no pytest, and a test file that does nothing when executed is a trap.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}\n      {exc or test.__doc__}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
