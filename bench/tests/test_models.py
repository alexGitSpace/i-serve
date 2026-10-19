"""Every model the calculator offers is its checkpoint, and one the formulas cover.

The rule is docs/audience.md's; the snapshots are GLOSSARY.md, *Checkpoint
snapshot*. No pytest dependency:

    python3 bench/tests/test_models.py
"""

# Run directly on a pod without pytest, this file gets only its own directory on
# the path; conftest.py is not consulted, so the insert lives here.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))


import json
import math

from predictions import DEFAULT_MODEL, GMU, MODELS
from roofline import ACCELERATORS, kv_cache_tokens

SNAPSHOTS = _pathlib.Path(__file__).resolve().parent.parent / "models"
CHECKPOINTS = json.loads((SNAPSHOTS / "checkpoints.json").read_text(encoding="utf-8"))


def _text_config(key: str) -> dict:
    """A multimodal checkpoint nests the language model's keys under text_config."""
    config = json.loads((SNAPSHOTS / f"{key}.config.json").read_text(encoding="utf-8"))
    return config.get("text_config", config)


def test_every_model_has_a_snapshot_and_every_snapshot_a_model():
    configs = {p.name.removesuffix(".config.json") for p in SNAPSHOTS.glob("*.config.json")}
    assert configs == set(MODELS) == set(CHECKPOINTS), (configs, set(MODELS), set(CHECKPOINTS))


def test_every_model_is_one_the_formulas_cover():
    """GLOSSARY.md, *Architectures that break the standard arithmetic*: refused, not mispriced."""
    for key in MODELS:
        c = _text_config(key)
        assert not c.get("sliding_window") and not c.get("use_sliding_window"), key
        assert "kv_lora_rank" not in c and "q_lora_rank" not in c, key          # MLA
        assert set(c.get("layer_types", [])) <= {"full_attention", "linear_attention"}, key
        assert not c.get("mlp_only_layers") and c.get("decoder_sparse_step", 1) == 1, key
        assert c.get("dtype", c.get("torch_dtype")) == "bfloat16", key
        assert "quantization_config" not in c, key


def test_every_literal_is_its_config_json():
    for key, m in MODELS.items():
        c = _text_config(key)
        assert (m.num_layers, m.num_kv_heads, m.head_dim) == \
            (c["num_hidden_layers"], c["num_key_value_heads"], c["head_dim"]), key
        layers = c.get("layer_types", [])
        assert m.linear.num_layers == layers.count("linear_attention"), key
        if m.linear.num_layers:
            assert (m.linear.num_k_heads, m.linear.num_v_heads, m.linear.head_k_dim,
                    m.linear.head_v_dim, m.linear.conv_kernel) == \
                (c["linear_num_key_heads"], c["linear_num_value_heads"],
                 c["linear_key_head_dim"], c["linear_value_head_dim"],
                 c["linear_conv_kernel_dim"]), key
            assert c["mamba_ssm_dtype"] == "float32" and m.linear.state_dtype_bytes == 4, key
        assert (m.experts.count, m.experts.per_token) == \
            ((c["num_experts"], c["num_experts_per_tok"]) if "num_experts" in c else (1, 1)), key


def test_every_parameter_count_is_its_checkpoint_served_text_only():
    """Embedding, lm_head and decoder, exact -- except Qwen3-8B's, the model card's three figures."""
    for key, m in MODELS.items():
        p = CHECKPOINTS[key]["parameters"]
        experts = p.get("routed_experts", 0)
        served = p["embedding"] + p["lm_head"] + p["decoder"] + experts
        assert math.isclose(m.params_total, served, rel_tol=2e-3), (key, m.params_total, served)
        assert math.isclose(m.params_non_embedding, p["decoder"] + experts, rel_tol=2e-3), key
        assert m.experts.params == experts, key
        if key != DEFAULT_MODEL:
            assert (m.params_total, m.params_non_embedding) == (served, p["decoder"] + experts), key
            assert CHECKPOINTS[key]["revision"][:7] in m.provenance, key


def test_the_routed_experts_weigh_what_config_json_says():
    """Three matrices of hidden x moe_intermediate per expert per layer: the header agrees exactly."""
    for key, m in MODELS.items():
        if not m.experts.params:
            continue
        c = _text_config(key)
        derived = (c["num_hidden_layers"] * c["num_experts"] * 3
                   * c["hidden_size"] * c["moe_intermediate_size"])
        assert derived == CHECKPOINTS[key]["parameters"]["routed_experts"], key


def test_every_model_fits_one_card_at_least():
    """The calculator prices one card; a model no card holds is not a comparison (Qwen3.8-Max)."""
    for key, m in MODELS.items():
        fits = []
        for accel in ACCELERATORS.values():
            try:
                kv_cache_tokens(m, accel, GMU)
                fits.append(accel.name)
            except ValueError:
                pass
        assert fits, key


if __name__ == "__main__":
    # The same runner its siblings carry, for the same reason: a rented pod has
    # no pytest, and a test file that does nothing when executed is a trap.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except (AssertionError, KeyError) as exc:
            failed += 1
            print(f"FAIL  {test.__name__}\n      {exc or test.__doc__}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
