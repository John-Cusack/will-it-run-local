"""Server lifecycle and timings using fake processes and HTTP responses."""
from contextlib import contextmanager
import io
import json
import signal
import subprocess
from types import SimpleNamespace
import urllib.error

import pytest

from wirl import runner, search, tune

pytestmark = pytest.mark.usefixtures("isolated_runtime")


def test_config_labels_flags_and_empty_statistics():
    cfg = runner.RunConfig("model", threads=None, flash_attn=False, ctx=8192, draft_model="draft")
    assert "--flash-attn" not in cfg.argv("fake")
    assert "ctx=8192" in cfg.label() and "nmax=1" in cfg.label() and "t=auto" in cfg.label()
    result = runner.RunResult(cfg, [], [], 0, 0, True)
    assert result.mean == result.spread_pct == 0
    assert runner.RunResult(cfg, [0, 0], [], 0, 0, True).spread_pct == 0


def test_server_discovery_sources_and_absence(monkeypatch):
    monkeypatch.setattr(runner.os.path, "exists", lambda p: p in ("explicit", "environment"))
    assert runner.find_server("explicit") == "explicit"
    assert runner.find_server("missing") is None
    monkeypatch.setenv("WIRL_LLAMA_SERVER", "environment")
    assert runner.find_server() == "environment"
    monkeypatch.setenv("WIRL_LLAMA_SERVER", "missing")
    monkeypatch.setattr(runner.shutil, "which", lambda p: "on-path")
    assert runner.find_server() == "on-path"
    monkeypatch.setattr(runner.shutil, "which", lambda p: None)
    monkeypatch.setattr(runner.os.path, "exists", lambda p: p == "/opt/llama.cpp/bin/llama-server")
    assert runner.find_server() == "/opt/llama.cpp/bin/llama-server"
    monkeypatch.setattr(runner.os.path, "exists", lambda p: False)
    assert runner.find_server() is None


def test_readiness_exit_timeout_and_retry(monkeypatch):
    ticks = iter(range(100))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    assert runner._wait_health("h", 1, SimpleNamespace(poll=lambda: 1), 20) is None
    assert runner._wait_health("h", 1, SimpleNamespace(poll=lambda: None), 0) is None
    statuses = iter([None, 404, 200])
    requests = []

    def opened(url, **kw):
        requests.append(url)
        status = next(statuses)
        if status is None:
            raise urllib.error.URLError("not bound")
        response = io.BytesIO()
        response.status = status
        return response

    monkeypatch.setattr(runner.urllib.request, "urlopen", opened)
    outputs = iter([(None, None, 0), (5, 10, 1)])
    monkeypatch.setattr(runner, "_generate", lambda *a, **kw: next(outputs))
    assert runner._wait_health("h", 1, SimpleNamespace(poll=lambda: None), 20) is not None
    assert requests == ["http://h:1/health"]*3


def test_prompt_construction_and_completion_request(monkeypatch):
    assert runner.build_prompt(0) == runner.BENCH_PROMPT
    prompt = runner.build_prompt(1000)
    assert len(prompt.split("\n")[0].split()) == 750
    requests = []
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda req, **kw:
                        requests.append((req, kw)) or io.BytesIO(json.dumps({"timings": {
                            "predicted_per_second": 10, "prompt_per_second": 20, "prompt_n": 30}}).encode()))
    assert runner._generate("host", 1, 5, prompt="history", timeout=9) == (10, 20, 30)
    req, options = requests[0]
    body = json.loads(req.data)
    assert req.full_url == "http://host:1/v1/chat/completions" and options == {"timeout": 9}
    assert body["messages"][0]["content"] == "history"
    assert body["max_tokens"] == 5 and not body["stream"] and not body["cache_prompt"]
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"{}"))
    assert runner._generate("host", 1, 1) == (None, None, None)


@pytest.mark.parametrize("ready", [True, False])
def test_server_closes_log_and_stops_after_readiness(monkeypatch, tmp_path, ready, capsys):
    monkeypatch.setattr(runner, "_check_port", lambda *a: None)
    proc = SimpleNamespace(poll=lambda: 7)
    launches, stops = [], []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: launches.append((argv, kw)) or proc)
    monkeypatch.setattr(runner, "_wait_health", lambda *a: 1 if ready else None)
    monkeypatch.setattr(runner, "_stop", lambda p: stops.append(p))
    if ready:
        with runner.server(runner.RunConfig("model"), "fake", log_dir=tmp_path) as started:
            assert started == 1 and not launches[0][1]["stdout"].closed
        assert "ready in 1s" in capsys.readouterr().out
    else:
        with pytest.raises(runner.ServerFailed, match="exit=7"):
            with runner.server(runner.RunConfig("model"), "fake", log_dir=tmp_path):
                pytest.fail("unready server yielded")
    assert stops == [proc] and launches[0][1]["stdout"].closed


def test_run_config_startup_failure_and_partial_timings(monkeypatch, capsys):
    @contextmanager
    def failed(*a):
        raise runner.ServerFailed("no VRAM")
        yield

    monkeypatch.setattr(runner, "server", failed)
    result = runner.run_config(runner.RunConfig("model"), "fake")
    assert not result.ok and result.error == "no VRAM"

    @contextmanager
    def running(*a):
        yield 2

    monkeypatch.setattr(runner, "server", running)
    monkeypatch.setattr(runner, "gpu_used_bytes", lambda uuid: 1)
    monkeypatch.setattr(runner, "_generate", lambda *a: (0, 0, 1))
    result = runner.run_config(runner.RunConfig("model"), "fake", reps=1)
    assert result.samples == result.prefill == []
    assert "rep0: 0.00 tok/s" in capsys.readouterr().out


@pytest.mark.parametrize("fallback", [False, True])
def test_stop_escalates_and_waits_without_real_signals(monkeypatch, fallback):
    calls = []
    proc = SimpleNamespace(pid=42, poll=lambda: None,
                           terminate=lambda: calls.append("terminate"), kill=lambda: calls.append("kill"),
                           wait=lambda timeout: calls.append(("wait", timeout)))
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: pid)

    def killpg(pid, sig):
        calls.append((pid, sig))
        if fallback:
            raise OSError("no process group")

    monkeypatch.setattr(runner.os, "killpg", killpg)
    monkeypatch.setattr(runner.time, "sleep", lambda s: calls.append(("sleep", s)))
    runner._stop(proc, grace=0)
    assert (42, signal.SIGTERM) in calls and (42, signal.SIGKILL) in calls
    assert ("wait", 30) in calls and calls[-1] == ("sleep", 3)
    assert ("terminate" in calls) == fallback and ("kill" in calls) == fallback


def test_stop_already_exited_and_graceful_exit(monkeypatch):
    runner._stop(SimpleNamespace(poll=lambda: 0))
    polls = iter([None, None, 0])
    calls = []
    proc = SimpleNamespace(pid=42, poll=lambda: next(polls))
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(runner.os, "killpg", lambda *a: calls.append(a))
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0)
    monkeypatch.setattr(runner.time, "sleep", lambda s: calls.append(s))
    runner._stop(proc, grace=10)
    assert calls == [(42, signal.SIGTERM), .5, 3]


def test_search_log_without_result_and_verbose_failed_dense_edge(monkeypatch, capsys):
    log = search.SearchLog()
    cfg = runner.RunConfig("model")
    log.add("skipped", cfg, None, note="reason")
    assert log.attempts[0]["note"] == "reason" and not log.attempts[0]["ok"]
    monkeypatch.setattr(search, "run_config", lambda cfg, *a, **kw:
                        runner.RunResult(cfg, [], [], 0, 0, False, "no fit"))
    assert search.find_ngl_edge(cfg, "fake", 0, 2, log=log) is None
    assert "does not start" in capsys.readouterr().out


@pytest.mark.parametrize("verbose", [True, False])
def test_depth_profile_skips_full_context_and_failed_requests(monkeypatch, verbose, capsys):
    calls = []

    def generate(host, port, tokens, prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("request failed")
        return (None, None, None) if len(calls) == 1 else (5, 10, 50)

    monkeypatch.setattr(runner, "_generate", generate)
    points = search.profile_depth(runner.RunConfig("model", ctx=100), "host", 1,
                                  depths=[0, 20, 50, 90], verbose=verbose)
    assert len(calls) == 3 and len(points) == 2
    assert points[0].ttft_s == 0 and points[1].ttft_s == 5
    text = capsys.readouterr().out
    assert ("request failed" in text) == verbose
    assert search.summarise_depth([], 100) == ""
    assert "Decode falls" not in search.summarise_depth(points[:1], 100)
    assert "Decode falls" not in search.summarise_depth(points, 100)


def test_draft_sweep_stops_on_failure_and_regression(monkeypatch, capsys):
    calls = []

    def run(cfg, *a, **kw):
        calls.append(cfg.draft_n_max)
        return runner.RunResult(cfg, [10 if cfg.draft_n_max == 1 else 8], [], 1, 1, True)

    monkeypatch.setattr(tune, "run_config", run)
    assert len(tune.sweep_draft_depth(runner.RunConfig("model"), "fake")) == 2
    assert calls == [1, 2] and "deeper will not help" in capsys.readouterr().out
    monkeypatch.setattr(tune, "run_config", lambda cfg, *a, **kw:
                        runner.RunResult(cfg, [], [], 0, 0, False, "failed"))
    results = tune.sweep_draft_depth(runner.RunConfig("model"), "fake")
    assert len(results) == 1 and "FAILED" in tune.summarise(results)
    assert tune.repeatability(results) == ""


def test_thread_sweep_and_unstable_flags(monkeypatch):
    monkeypatch.setattr(tune, "run_config", lambda cfg, *a, **kw:
                        runner.RunResult(cfg, [8, 12], [], 1, 1, True))
    results = tune.sweep_threads(runner.RunConfig("model"), "fake", [2, 4])
    assert [r.config.threads for r in results] == [2, 4]
    assert "40.0%" in tune.summarise(results)
    assert tune.flag_unstable(results) and tune.flag_unstable(results, threshold=50) == []
    assert "Unstable" in tune.flag_unstable(results)[0]
