"""Candidate discovery using fake metadata responses, never downloads."""
import io
import json
from types import SimpleNamespace
import urllib.parse

import pytest

from wirl import USER_AGENT, compat, drafters, gguf

pytestmark = pytest.mark.usefixtures("isolated_runtime")


def signature(**changes):
    return dict(arch="qwen3", n_vocab=2, tok_model="gpt2", tok_pre="qwen",
                bos=0, eos=1, vocab_sha256_16="same", **changes)


def test_search_request_and_failure(monkeypatch):
    requests = []
    response = io.BytesIO(json.dumps([{"id": "owner/draft"}]).encode())

    def opened(req, **kw):
        requests.append((req, kw))
        return response

    monkeypatch.setattr(drafters.urllib.request, "urlopen", opened)
    assert drafters.search_hf("Qwen draft", limit=3) == [{"id": "owner/draft"}]
    req, options = requests[0]
    assert req.get_header("User-agent") == USER_AGENT and options == {"timeout": 30}
    assert urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query) == {
        "search": ["Qwen draft"], "limit": ["3"], "filter": ["gguf"],
        "sort": ["downloads"], "direction": ["-1"]}
    assert response.closed
    monkeypatch.setattr(drafters, "_get", lambda *a: (_ for _ in ()).throw(OSError("offline")))
    assert drafters.search_hf("Qwen") == []


@pytest.mark.parametrize("path,arch,expected", [
    ("DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf", "deepseek4",
     ["DeepSeek-V4-Flash draft", "DeepSeek-V4-Flash dspark", "DeepSeek-V4 draft", "DeepSeek-V4"]),
    ("Qwen3-0.6B-F16.GGUF", "qwen3", ["Qwen3-0.6B draft", "Qwen3-0.6B dspark", "Qwen3-0.6B"]),
    ("Solo.gguf", "llama", ["Solo draft", "Solo dspark", "Solo"]),
    (".gguf", "llama", ["llama draft", "llama dspark", "llama"]),
])
def test_candidate_queries_remove_packaging_and_duplicates(path, arch, expected):
    assert drafters.candidate_queries(SimpleNamespace(path=path, arch=arch)) == expected


def test_candidates_deduplicate_and_rank_purpose_built_first(monkeypatch):
    monkeypatch.setattr(drafters, "candidate_queries", lambda g: ["one", "two"])
    rows = [{"id": "owner/sibling", "downloads": 1000},
            {"modelId": "owner/draft", "downloads": 20}, {},
            {"id": "owner/EAGLE", "downloads": 100}, {"id": "owner/unknown"}]
    calls = []
    monkeypatch.setattr(drafters, "search_hf", lambda q, limit: calls.append((q, limit)) or rows)
    result = drafters.find_candidates(None, limit_repos=5)
    assert [c["repo"] for c in result] == ["owner/EAGLE", "owner/draft", "owner/sibling", "owner/unknown"]
    assert [c["purpose_built"] for c in result] == [True, True, False, False]
    assert calls == [("one", 5), ("two", 5)]


def test_vetting_size_limits_failures_and_incompatible_files(monkeypatch):
    monkeypatch.setattr(compat, "list_gguf", lambda repo: [
        {"path": "large.gguf", "size": 3e9}, {"path": "bad.gguf", "size": 1e8},
        {"path": "wrong.gguf", "size": 2e8}, {"path": "good.gguf", "size": 3e8}])
    calls = []

    def check(repo, filename):
        calls.append(filename)
        if filename == "bad.gguf":
            raise EOFError("short header")
        sig = signature()
        if filename == "wrong.gguf":
            sig["n_vocab"] = 3
        return sig

    monkeypatch.setattr(compat, "check_remote_sig", check)
    result = drafters.vet(signature(), "owner/repo", max_gb=1)
    assert calls == ["bad.gguf", "wrong.gguf", "good.gguf"]
    assert result[0]["error"] == "short header"
    assert not result[1]["compatible"] and "token count differs" in result[1]["problems"][0]
    assert result[2]["compatible"] and not result[2]["warnings"]
    calls.clear()
    assert len(drafters.vet(signature(), "owner/repo", max_files=1)) == 1
    assert calls == ["bad.gguf"]
    monkeypatch.setattr(compat, "list_gguf", lambda repo: (_ for _ in ()).throw(OSError("offline")))
    assert drafters.vet(signature(), "owner/repo") == [{"repo": "owner/repo", "error": "offline"}]


def test_remote_tree_filters_files_and_prefers_lfs_sizes(monkeypatch):
    tree = [{"type": "directory", "path": "fake.gguf"}, {"type": "file", "path": "README.md"},
            {"type": "file", "path": "z.GGUF", "size": 10, "lfs": {"size": 100}},
            {"type": "file", "path": "a.gguf", "size": 20},
            {"type": "file", "path": "b.gguf", "lfs": None}]
    calls = []
    monkeypatch.setattr(compat.urllib.request, "urlopen", lambda req, **kw:
                        calls.append((req, kw)) or io.BytesIO(json.dumps(tree).encode()))
    assert compat.list_gguf("owner/repo") == [
        {"path": "a.gguf", "size": 20}, {"path": "b.gguf", "size": 0}, {"path": "z.GGUF", "size": 100}]
    assert calls[0][0].full_url.endswith("/owner/repo/tree/main?recursive=1")
    assert calls[0][0].get_header("User-agent") == USER_AGENT
    assert calls[0][1] == {"timeout": 60}


def test_local_and_remote_checks_read_only_headers(monkeypatch):
    calls = []
    sig = signature()
    monkeypatch.setattr(gguf, "read_one", lambda path, want_tensors:
                        calls.append((path, want_tensors)) or SimpleNamespace(vocab_sig=lambda: sig))
    monkeypatch.setattr(gguf, "read_header_only", lambda path:
                        calls.append((path, False)) or SimpleNamespace(vocab_sig=lambda: sig))
    assert compat.check_remote_sig("owner/repo", "draft.gguf") == sig
    assert compat.check_remote("target.gguf", "owner/repo", "draft.gguf")["source"] == "owner/repo/draft.gguf"
    assert compat.check_local("target.gguf", "draft.gguf")["source"] == "draft.gguf"
    assert all(not tensors for _, tensors in calls)
    assert calls[0][0] == "https://huggingface.co/owner/repo/resolve/main/draft.gguf"


def test_pre_tokeniser_and_bos_differences_warn():
    draft = signature()
    draft.update(bos=5, tok_pre="other")
    result = compat.compare(signature(), draft)
    assert result["compatible"]
    assert len(result["warnings"]) == 2
