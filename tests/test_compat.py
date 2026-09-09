"""Draft-model vetting. The failures here are all ones that only show up
after a multi-GB download completes, which is why they are worth catching."""
import pytest

from wirl import compat

TARGET = {"arch": "deepseek4", "n_vocab": 129280, "tok_model": "gpt2",
          "tok_pre": "joyai-llm", "bos": 0, "eos": 1,
          "vocab_sha256_16": "a31e23b927c61c03"}


def _draft(**over):
    d = dict(TARGET, arch="dflash")
    d.update(over)
    return d


def test_matching_pair_is_compatible():
    r = compat.compare(TARGET, _draft())
    assert r["compatible"]
    assert not r["problems"] and not r["warnings"]


def test_different_vocabulary_is_rejected():
    r = compat.compare(TARGET, _draft(vocab_sha256_16="ffffffffffffffff"))
    assert not r["compatible"]
    assert "vocabulary mismatch" in r["problems"][0]


def test_same_token_count_different_tokens_is_still_rejected():
    """The trap: identical n_vocab, different content. Only the hash catches it."""
    r = compat.compare(TARGET, _draft(vocab_sha256_16="0000000000000000"))
    assert not r["compatible"]
    assert r["target"]["n_vocab"] == r["draft"]["n_vocab"]


def test_unregistered_architecture_warns():
    """Three of four candidate drafters on HuggingFace declared architectures
    llama.cpp does not register and could never have loaded."""
    r = compat.compare(TARGET, _draft(arch="deepseek4-dspark"))
    assert r["compatible"]          # vocab is fine
    assert any("deepseek4-dspark" in w for w in r["warnings"])


def test_eos_mismatch_warns_without_blocking():
    r = compat.compare(TARGET, _draft(eos=2))
    assert r["compatible"]
    assert any("eos token id differs" in w for w in r["warnings"])


def test_known_draft_archs_includes_registered_ones():
    assert "dflash" in compat.KNOWN_DRAFT_ARCHS
    assert "llama" in compat.KNOWN_DRAFT_ARCHS


def test_resolve_url_shape():
    u = compat.resolve_url("owner/repo", "a/b.gguf")
    assert u == "https://huggingface.co/owner/repo/resolve/main/a/b.gguf"
