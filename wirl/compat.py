"""Draft-model compatibility checking, without downloading the draft model.

Speculative decoding pairs a small draft model with a large target. llama.cpp
requires the two to share a vocabulary exactly, and requires the draft's
architecture to be one it actually implements. Neither is discoverable from a
model card, and both fail only after the download completes.

A GGUF stores its metadata -- including the full token list -- at the head of
the file, so an HTTP Range request for the first few MB is enough to answer
both questions. On the reference machine this ruled out three of four candidate
drafters, which declared architectures llama.cpp does not register and could
never have loaded.
"""

from __future__ import annotations

import json
import urllib.request

HF_API = "https://huggingface.co/api/models"


def resolve_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def list_gguf(repo: str) -> list:
    """GGUF files in a HuggingFace repo with their sizes."""
    req = urllib.request.Request(f"{HF_API}/{repo}/tree/main?recursive=1",
                                 headers={"User-Agent": "will-it-run-local/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        tree = json.load(r)
    out = []
    for f in tree:
        if f.get("type") == "file" and f["path"].lower().endswith(".gguf"):
            size = (f.get("lfs") or {}).get("size") or f.get("size") or 0
            out.append({"path": f["path"], "size": size})
    return sorted(out, key=lambda x: x["path"])


# Architectures llama.cpp registers for draft models. A draft declaring
# anything else cannot load, regardless of how well-formed the file is.
KNOWN_DRAFT_ARCHS = {"dflash", "llama", "qwen2", "qwen3", "gemma", "gemma2",
                     "gemma3", "phi3", "deepseek2", "deepseek4", "mistral"}


def compare(target_sig: dict, draft_sig: dict) -> dict:
    """Decide whether a draft can pair with a target, and say why not."""
    problems = []
    warnings = []

    if target_sig["vocab_sha256_16"] != draft_sig["vocab_sha256_16"]:
        problems.append(
            f"vocabulary mismatch: target {target_sig['vocab_sha256_16']} vs "
            f"draft {draft_sig['vocab_sha256_16']}. Speculative decoding "
            "requires identical vocabularies; llama.cpp will refuse to load.")
    if target_sig["n_vocab"] != draft_sig["n_vocab"]:
        problems.append(f"token count differs: {target_sig['n_vocab']} vs "
                        f"{draft_sig['n_vocab']}.")
    for k in ("bos", "eos"):
        if target_sig[k] != draft_sig[k]:
            warnings.append(f"{k} token id differs: {target_sig[k]} vs {draft_sig[k]}.")
    if target_sig["tok_pre"] != draft_sig["tok_pre"]:
        warnings.append(f"tokenizer pre-tokenizer differs: "
                        f"{target_sig['tok_pre']} vs {draft_sig['tok_pre']}.")

    da = draft_sig["arch"]
    if da not in KNOWN_DRAFT_ARCHS:
        warnings.append(
            f"draft declares architecture '{da}', which is not in this tool's "
            "list of architectures llama.cpp registers. Confirm with "
            "`llama-server --list-architectures` or by grepping "
            "src/llama-arch.cpp before downloading -- an unregistered "
            "architecture fails at load time with 'unknown model architecture'.")

    return {"compatible": not problems, "problems": problems,
            "warnings": warnings, "target": target_sig, "draft": draft_sig}


def check_remote_sig(repo: str, filename: str) -> dict:
    """Vocabulary fingerprint of a remote GGUF, header bytes only."""
    from .gguf import read_header_only
    return read_header_only(resolve_url(repo, filename)).vocab_sig()


def check_remote(target_path: str, repo: str, filename: str) -> dict:
    """Vet a HuggingFace-hosted draft against a local target. No full download."""
    from .gguf import read_header_only, read_one
    tgt = read_one(target_path, want_tensors=False).vocab_sig()
    drf = read_header_only(resolve_url(repo, filename)).vocab_sig()
    res = compare(tgt, drf)
    res["source"] = f"{repo}/{filename}"
    return res


def check_local(target_path: str, draft_path: str) -> dict:
    from .gguf import read_one
    tgt = read_one(target_path, want_tensors=False).vocab_sig()
    drf = read_one(draft_path, want_tensors=False).vocab_sig()
    res = compare(tgt, drf)
    res["source"] = draft_path
    return res
