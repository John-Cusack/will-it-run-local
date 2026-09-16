import struct

import pytest

from wirl import gguf
from wirl.ggml_types import bits_per_weight, tensor_nbytes


def test_reads_metadata_and_tensors(moe_model):
    g = gguf.read(moe_model)
    assert g.arch == "toymoe"
    assert g.n_layer == 2
    assert g.n_expert == 8
    assert g.n_expert_used == 2
    assert g.is_moe
    assert len(g.tensors) == 10


def test_rejects_non_gguf(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not a GGUF file"):
        gguf.read(str(p))


def test_vocab_signature_is_content_based(tmp_path, moe_model, build_gguf):
    """Two models with the same token count but different tokens must differ.

    Comparing n_vocab alone would call these compatible and produce garbage.
    """
    g1 = gguf.read(moe_model)
    kv = {
        "general.architecture": (8, "toymoe"),
        "tokenizer.ggml.model": (8, "gpt2"),
        "tokenizer.ggml.tokens": (9, (8, ["a", "b", "X"])),
        "tokenizer.ggml.bos_token_id": (4, 0),
        "tokenizer.ggml.eos_token_id": (4, 1),
    }
    other = build_gguf(tmp_path / "other.gguf", kv, [])
    g2 = gguf.read(other)
    assert g1.vocab_sig()["n_vocab"] == g2.vocab_sig()["n_vocab"]
    assert g1.vocab_sig()["vocab_sha256_16"] != g2.vocab_sig()["vocab_sha256_16"]


@pytest.mark.parametrize("tid,dims,expect", [
    (0, (256, 2), 2048),        # f32: 4 bytes each
    (1, (256, 2), 1024),        # f16
    (12, (256, 1), 144),        # q4_K: one 256-block
    (39, (256, 1), 8 * 17),     # mxfp4: eight 32-blocks
    (8, (32, 4), 4 * 34),       # q8_0
])
def test_tensor_sizes_match_ggml(tid, dims, expect):
    assert tensor_nbytes(tid, dims) == expect


def test_removed_types_are_rejected():
    """Deprecated types must raise, not silently price as zero."""
    with pytest.raises(ValueError):
        tensor_nbytes(31, (256, 1))     # removed Q4_0_4_4


def test_bits_per_weight():
    assert bits_per_weight(39) == pytest.approx(4.25)   # mxfp4
    assert bits_per_weight(12) == pytest.approx(4.5)    # q4_K
    assert bits_per_weight(1) == pytest.approx(16.0)    # f16


def test_shard_expansion_requires_all_parts(tmp_path):
    p = tmp_path / "m-00001-of-00003.gguf"
    p.write_bytes(b"GGUF")
    with pytest.raises(FileNotFoundError, match="shard 2 of 3"):
        gguf.shard_paths(str(p))


def test_unsharded_path_passes_through(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUF")
    assert gguf.shard_paths(str(p)) == [str(p)]
