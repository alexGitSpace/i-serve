"""Snapshot one checkpoint: its config.json, and its parameters split by role.

GLOSSARY.md, *Checkpoint snapshot*. Headers only, a Range request per shard, so
no weights are downloaded.

    python3 bench/models/snapshot.py qwen3.5-9b Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a
"""

import json
import math
import pathlib
import struct
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
CHECKPOINTS = HERE / "checkpoints.json"


def fetch(url: str, byte_range: str | None = None) -> bytes:
    headers = {"Range": f"bytes={byte_range}"} if byte_range else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        return response.read()


def role(name: str) -> str:
    """Which part of the checkpoint a tensor belongs to, by its name."""
    if name.startswith("mtp."):
        return "mtp"                  # the speculative head vLLM skips unless asked
    if ".visual." in name or name.startswith("visual."):
        return "vision"               # the tower --language-model-only skips
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if ".mlp.experts." in name:
        return "routed_experts"       # shared experts say shared_expert, not experts
    return "decoder"                  # attention, linear attention, dense FFN, router, norms


def parameters(base: str) -> dict[str, int]:
    """Every tensor in every shard, its element count summed under its role."""
    try:
        index = json.loads(fetch(base + "model.safetensors.index.json"))
        shards = sorted(set(index["weight_map"].values()))
    except urllib.error.HTTPError:
        shards = ["model.safetensors"]
    roles: dict[str, int] = {}
    for shard in shards:
        # A safetensors file opens with an 8-byte little-endian header length.
        length = struct.unpack("<Q", fetch(base + shard, "0-7"))[0]
        header = json.loads(fetch(base + shard, f"8-{7 + length}"))
        header.pop("__metadata__", None)
        for name, tensor in header.items():
            roles[role(name)] = roles.get(role(name), 0) + math.prod(tensor["shape"])
    return dict(sorted(roles.items()))


def main(key: str, repo: str, revision: str) -> None:
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    config = json.loads(fetch(base + "config.json"))
    (HERE / f"{key}.config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoints = json.loads(CHECKPOINTS.read_text(encoding="utf-8")) if CHECKPOINTS.exists() else {}
    checkpoints[key] = {"repo": repo, "revision": revision, "parameters": parameters(base)}
    CHECKPOINTS.write_text(json.dumps(checkpoints, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"{key}: {sum(checkpoints[key]['parameters'].values()):,} parameters at {revision[:7]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
