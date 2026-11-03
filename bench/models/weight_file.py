"""Snapshot one weight file: the bytes each tensor role holds, read from its header.

GLOSSARY.md, *Weight file snapshot*. GGUF or safetensors, headers only, so no
weights are downloaded. A URL ending in "/" is a safetensors checkpoint; any
other is one GGUF file.

    python3 bench/models/weight_file.py llama-3.1-8b-instruct.q4_k_m.ollama \
        https://registry.ollama.ai/v2/library/llama3.1/blobs/sha256:667b0c19...
"""

import json
import math
import pathlib
import struct
import sys

from snapshot import fetch, role, tensors

HERE = pathlib.Path(__file__).resolve().parent
WEIGHT_FILES = HERE / "weight_files.json"

# ggml type id -> (bytes per block, weights per block): the block structs in
# ggml-common.h. A type missing here stops the snapshot rather than guess.
GGML_BLOCKS = {
    0: (4, 1),        # F32
    1: (2, 1),        # F16
    2: (18, 32),      # Q4_0: one F16 scale + 32 nibbles
    3: (20, 32),      # Q4_1
    6: (22, 32),      # Q5_0
    7: (24, 32),      # Q5_1
    8: (34, 32),      # Q8_0
    10: (84, 256),    # Q2_K
    11: (110, 256),   # Q3_K
    12: (144, 256),   # Q4_K
    13: (176, 256),   # Q5_K
    14: (210, 256),   # Q6_K
    30: (2, 1),       # BF16
}

# safetensors dtype -> bytes per stored element. Packed formats (AWQ's I32,
# NVFP4's U8) store several weights per element, so only bytes are kept for
# them, never a parameter count.
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
               "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1}

# GGUF metadata value types (gguf.md): scalars by struct code; 8 string, 9 array.
GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                10: "Q", 11: "q", 12: "d"}


class Prefix:
    """A cursor over a remote file's first bytes; reading past them fetches as many again."""

    def __init__(self, url: str):
        self.url, self.data, self.pos = url, b"", 0

    def take(self, n: int) -> bytes:
        while self.pos + n > len(self.data):
            start = len(self.data)
            self.data += fetch(self.url, f"{start}-{max(2 * start, 1 << 20) - 1}")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, code: str):
        return struct.unpack("<" + code, self.take(struct.calcsize("<" + code)))[0]

    def string(self) -> str:
        return self.take(self.unpack("Q")).decode()

    def value(self, kind: int):
        if kind == 8:
            return self.string()
        if kind == 9:
            inner, count = self.unpack("I"), self.unpack("Q")
            return [self.value(inner) for _ in range(count)]
        return self.unpack(GGUF_SCALARS[kind])


def gguf_role(name: str) -> str:
    """GGUF's tensor names: llama.cpp's convert script renames the HF ones."""
    if name == "token_embd.weight":
        return "embedding"
    if name == "output.weight":
        return "lm_head"
    return "decoder"


def gguf(url: str) -> dict:
    """Bytes and weights per role, each tensor's size checked against its slot in the file."""
    head = Prefix(url)
    if head.take(4) != b"GGUF":
        raise ValueError(f"{url}: not a GGUF file")
    head.unpack("I")                                        # version
    count, kv_count = head.unpack("Q"), head.unpack("Q")
    meta = {}
    for _ in range(kv_count):
        key = head.string()
        meta[key] = head.value(head.unpack("I"))
    table = []
    for _ in range(count):
        name = head.string()
        dims = [head.unpack("Q") for _ in range(head.unpack("I"))]
        kind, offset = head.unpack("I"), head.unpack("Q")
        if kind not in GGML_BLOCKS:
            raise ValueError(f"{name}: ggml type {kind} is not in GGML_BLOCKS")
        block_bytes, block_weights = GGML_BLOCKS[kind]
        table.append((offset, name, math.prod(dims), math.prod(dims) // block_weights * block_bytes))
    # The tensors sit back to back, each padded to the alignment: a slot shorter
    # than its tensor, or longer by a whole alignment, means a block size is wrong.
    alignment = meta.get("general.alignment", 32)
    table.sort()
    for (offset, name, _, size), (following, *_) in zip(table, table[1:]):
        if not size <= following - offset < size + alignment:
            raise ValueError(f"{name}: {size} bytes in a {following - offset}-byte slot")
    roles: dict[str, dict[str, int]] = {}
    for _, name, weights, size in table:
        entry = roles.setdefault(gguf_role(name), {"bytes": 0, "weights": 0})
        entry["bytes"] += size
        entry["weights"] += weights
    # The KV geometry, under the architecture's own prefix; key_length is absent
    # when the head dimension is the embedding split evenly over the heads.
    arch = meta["general.architecture"]
    attention = {
        "layers": meta[f"{arch}.block_count"],
        "kv_heads": meta[f"{arch}.attention.head_count_kv"],
        "head_dim": meta.get(f"{arch}.attention.key_length")
        or meta[f"{arch}.embedding_length"] // meta[f"{arch}.attention.head_count"],
    }
    return {"format": "gguf", "attention": attention, "roles": dict(sorted(roles.items()))}


def safetensors(base: str) -> dict:
    """Bytes per role from the data offsets, checked against dtype times shape."""
    roles: dict[str, dict[str, int]] = {}
    for name, tensor in tensors(base):
        start, end = tensor["data_offsets"]
        if end - start != math.prod(tensor["shape"]) * DTYPE_BYTES[tensor["dtype"]]:
            raise ValueError(f"{name}: offsets disagree with {tensor['dtype']} {tensor['shape']}")
        entry = roles.setdefault(role(name), {"bytes": 0})
        entry["bytes"] += end - start
    c = json.loads(fetch(base + "config.json"))
    c = c.get("text_config", c)
    attention = {
        "layers": c["num_hidden_layers"],
        "kv_heads": c["num_key_value_heads"],
        "head_dim": c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
    }
    return {"format": "safetensors", "attention": attention, "roles": dict(sorted(roles.items()))}


def main(key: str, url: str) -> None:
    snapshot = {"url": url, **(safetensors(url) if url.endswith("/") else gguf(url))}
    files = json.loads(WEIGHT_FILES.read_text(encoding="utf-8")) if WEIGHT_FILES.exists() else {}
    files[key] = snapshot
    WEIGHT_FILES.write_text(json.dumps(files, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(r["bytes"] for r in snapshot["roles"].values())
    print(f"{key}: {total:,} bytes in {len(snapshot['roles'])} roles")


if __name__ == "__main__":
    main(*sys.argv[1:])
