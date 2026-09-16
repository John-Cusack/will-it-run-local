"""Synthetic GGUF construction, so tests never need a multi-GB download."""
import struct

import pytest


@pytest.fixture
def isolated_runtime(monkeypatch):
    """Make an omitted hardware/network mock fail before doing real work."""
    import os
    import subprocess
    import sys
    import urllib.request

    def unexpected(*args, **kwargs):
        pytest.fail("test attempted an unmocked process, signal or network request")

    monkeypatch.setattr(subprocess, "run", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    monkeypatch.setattr(urllib.request, "urlopen", unexpected)
    monkeypatch.setattr(os, "killpg", unexpected)
    monkeypatch.setitem(sys.modules, "numpy", None)

(T_UINT32, T_FLOAT32, T_STRING, T_ARRAY, T_UINT64) = 4, 6, 8, 9, 10


def _kv_string(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def _value(vtype, v):
    if vtype == T_STRING:
        return _kv_string(v.encode())
    if vtype == T_UINT32:
        return struct.pack("<I", v)
    if vtype == T_UINT64:
        return struct.pack("<Q", v)
    if vtype == T_FLOAT32:
        return struct.pack("<f", v)
    raise ValueError(vtype)


def write_gguf(path, kv, tensors):
    """kv: {key: (type, value)}; tensors: [(name, dims, type_id)]."""
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)
    out += struct.pack("<QQ", len(tensors), len(kv))
    for k, (vt, v) in kv.items():
        out += _kv_string(k.encode())
        out += struct.pack("<I", vt)
        if vt == T_ARRAY:
            et, items = v
            out += struct.pack("<I", et) + struct.pack("<Q", len(items))
            for it in items:
                out += _value(et, it)
        else:
            out += _value(vt, v)
    off = 0
    for name, dims, tid in tensors:
        out += _kv_string(name.encode())
        out += struct.pack("<I", len(dims))
        out += struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<I", tid)
        out += struct.pack("<Q", off)
        off += 32
    path.write_bytes(bytes(out))
    return str(path)


@pytest.fixture
def build_gguf():
    """The GGUF writer, as a fixture.

    Tests must not `from tests.conftest import ...`: that only resolves when
    the working directory happens to be on sys.path, and any other installed
    package named `tests` shadows it.
    """
    return write_gguf


@pytest.fixture
def moe_model(tmp_path):
    """A small MoE with a shape that makes the arithmetic checkable by hand.

    2 layers, 8 experts, 2 active. Expert tensors are f32 so bytes are exact:
    each is 4*4*8 = 128 elements -> 512 bytes, of which 2/8 is read per token.
    """
    kv = {
        "general.architecture": (T_STRING, "toymoe"),
        "toymoe.block_count": (T_UINT32, 2),
        "toymoe.embedding_length": (T_UINT32, 4),
        "toymoe.expert_count": (T_UINT32, 8),
        "toymoe.expert_used_count": (T_UINT32, 2),
        "toymoe.attention.head_count": (T_UINT32, 2),
        "toymoe.attention.head_count_kv": (T_UINT32, 1),
        "toymoe.attention.key_length": (T_UINT32, 4),
        "toymoe.attention.value_length": (T_UINT32, 4),
        "tokenizer.ggml.model": (T_STRING, "gpt2"),
        "tokenizer.ggml.tokens": (T_ARRAY, (T_STRING, ["a", "b", "c"])),
        "tokenizer.ggml.bos_token_id": (T_UINT32, 0),
        "tokenizer.ggml.eos_token_id": (T_UINT32, 1),
    }
    tensors = []
    for li in range(2):
        # routed experts: trailing dim is the expert count
        tensors += [(f"blk.{li}.ffn_up_exps.weight", (4, 4, 8), 0),
                    (f"blk.{li}.ffn_down_exps.weight", (4, 4, 8), 0),
                    # always-read: attention and the shared expert
                    (f"blk.{li}.attn_q.weight", (4, 4), 0),
                    (f"blk.{li}.ffn_up_shexp.weight", (4, 4), 0)]
    tensors.append(("token_embd.weight", (4, 3), 0))
    tensors.append(("output.weight", (4, 3), 0))
    return write_gguf(tmp_path / "toymoe.gguf", kv, tensors)
