"""Find a speculative-decoding draft model for a target, and vet it.

Speculative decoding is the single largest free win available on a
CPU-offloaded MoE -- it was worth +33% on the reference machine -- but only if
you can find a draft model that llama.cpp will actually load with your target.
Three things have to line up, and none of them are on a model card:

  1. an identical vocabulary (llama.cpp refuses otherwise)
  2. an architecture llama.cpp registers (several published drafters declare
     architectures that do not exist in any build and can never load)
  3. small enough that drafting is cheaper than the tokens it saves

A GGUF keeps its metadata at the head of the file, so all three can be checked
with a few MB of HTTP Range requests rather than by downloading candidates.

Two families of candidate are searched:

  purpose-built drafters -- DSpark/DFlash for DeepSeek, EAGLE/MTP heads, and
  anything published with "draft" in the name. These are trained against a
  specific target and give the highest acceptance rates.

  small siblings -- a smaller model from the same family, which usually shares
  a tokeniser (Qwen3-0.6B drafting for Qwen3-32B, say). Lower acceptance, but
  far more widely available.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from . import compat

HF_SEARCH = "https://huggingface.co/api/models"

DRAFTER_KEYWORDS = ("dspark", "drafter", "draft", "eagle", "mtp", "dflash")

# Parameter counts small enough to be worth drafting with, in billions.
DRAFT_MAX_B = 4.0


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "will-it-run-local/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def search_hf(query, limit=20):
    q = urllib.parse.urlencode({"search": query, "limit": limit,
                                "filter": "gguf", "sort": "downloads",
                                "direction": -1})
    try:
        return _get(f"{HF_SEARCH}?{q}")
    except Exception:                                        # noqa: BLE001
        return []


def _family_terms(model_name: str):
    """Reduce a model filename to searchable family terms.

    'DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf'
      -> 'DeepSeek-V4-Flash'
    """
    n = re.sub(r"\.gguf$", "", model_name, flags=re.I)
    n = re.sub(r"-\d{5}-of-\d{5}$", "", n)
    # strip quantisation and packaging suffixes
    n = re.sub(r"[-_](UD[-_])?(IQ|Q)\d+[A-Z0-9_]*$", "", n, flags=re.I)
    n = re.sub(r"[-_](BF16|F16|F32|MXFP4|GGUF|abliterated|instruct|chat)$", "",
               n, flags=re.I)
    parts = [p for p in re.split(r"[-_]", n) if p]
    return parts


def candidate_queries(gguf) -> list:
    """Search terms most likely to surface a compatible drafter."""
    import os
    parts = _family_terms(os.path.basename(gguf.path))
    family = "-".join(parts[:3]) if parts else gguf.arch
    short = "-".join(parts[:2]) if len(parts) >= 2 else family
    out = [f"{family} draft", f"{family} dspark", f"{short} draft"]
    # A smaller sibling of the same family.
    out.append(short)
    seen, uniq = set(), []
    for q in out:
        if q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    return uniq


def _looks_like_drafter(repo_id: str) -> bool:
    low = repo_id.lower()
    return any(k in low for k in DRAFTER_KEYWORDS)


def find_candidates(gguf, limit_repos=12, verbose=True):
    """Repos that might contain a usable drafter. Metadata only, no downloads."""
    seen, cands = set(), []
    for q in candidate_queries(gguf):
        for m in search_hf(q, limit=limit_repos):
            rid = m.get("id") or m.get("modelId")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            cands.append({"repo": rid,
                          "downloads": m.get("downloads", 0),
                          "purpose_built": _looks_like_drafter(rid)})
    # Purpose-built drafters first, then by popularity.
    cands.sort(key=lambda c: (not c["purpose_built"], -c["downloads"]))
    return cands


def vet(target_sig, repo, max_files=4, max_gb=None, verbose=True):
    """Read headers from a repo and return every file that could pair.

    Returns a list of dicts, each with the file, its size, and the verdict.
    """
    out = []
    try:
        files = compat.list_gguf(repo)
    except Exception as e:                                   # noqa: BLE001
        return [{"repo": repo, "error": str(e)}]
    # Smallest first: a drafter must be small to be worth anything.
    files = sorted(files, key=lambda f: f["size"])
    for f in files[:max_files]:
        if max_gb and f["size"] > max_gb * 1e9:
            continue
        try:
            sig = compat.check_remote_sig(repo, f["path"])
        except Exception as e:                               # noqa: BLE001
            out.append({"repo": repo, "file": f["path"], "size": f["size"],
                        "error": str(e)})
            continue
        res = compat.compare(target_sig, sig)
        out.append({"repo": repo, "file": f["path"], "size": f["size"],
                    "sig": sig, "compatible": res["compatible"],
                    "problems": res["problems"], "warnings": res["warnings"]})
    return out
