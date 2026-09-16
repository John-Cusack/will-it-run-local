"""Command construction and emitted artefacts."""
import pytest

from wirl import emit
from wirl.runner import RunConfig


def test_argv_includes_the_essential_flags():
    a = RunConfig(model="/m.gguf", n_cpu_moe=43, threads=32).argv("llama-server")
    assert "--n-cpu-moe" in a and a[a.index("--n-cpu-moe") + 1] == "43"
    assert a[a.index("--threads") + 1] == "32"
    assert a[a.index("--threads-batch") + 1] == "32"


def test_k_cache_defaults_to_f16():
    a = RunConfig(model="/m.gguf").argv("llama-server")
    assert a[a.index("--cache-type-k") + 1] == "f16"


def test_draft_flags_only_appear_with_a_draft_model():
    assert "--spec-draft-model" not in RunConfig(model="/m.gguf").argv("s")
    a = RunConfig(model="/m.gguf", draft_model="/d.gguf", draft_n_max=2).argv("s")
    assert a[a.index("--spec-draft-model") + 1] == "/d.gguf"
    assert a[a.index("--spec-draft-n-max") + 1] == "2"


def test_launch_script_pins_to_physical_cores():
    cfg = RunConfig(model="/m.gguf", n_cpu_moe=43)
    s = emit.launch_script(cfg, "/bin/llama-server", cpus=list(range(64)))
    assert "taskset -c 0-63" in s
    assert s.startswith("#!/usr/bin/env bash")
    assert "--n-cpu-moe 43" in s


def test_launch_script_uses_explicit_topology_or_no_pinning():
    cfg = RunConfig("m")
    assert "taskset -c 0,2,4,16-19" in emit.launch_script(cfg, "s", cpus=[0, 2, 4, 16, 17, 18, 19])
    assert "taskset" not in emit.launch_script(cfg, "s")


def test_launch_script_quotes_awkward_paths():
    cfg = RunConfig(model="/models/my model.gguf")
    s = emit.launch_script(cfg, "/bin/llama-server")
    assert "'/models/my model.gguf'" in s


def test_systemd_unit_has_restart_and_generous_start_timeout():
    u = emit.systemd_unit("/run.sh", "test", notes="chosen by measurement")
    assert "Restart=on-failure" in u
    assert "TimeoutStartSec=900" in u
    assert "# chosen by measurement" in u
    assert "ExecStart=/run.sh" in u


def test_written_script_is_executable(tmp_path):
    import os
    p = emit.write(str(tmp_path / "run.sh"), "#!/bin/sh\ntrue\n", 0o755)
    assert os.access(p, os.X_OK)


def test_benchmark_lock_is_exclusive(tmp_path, monkeypatch):
    """Concurrent benchmarks must fail loudly rather than produce noise."""
    import wirl.lock as L
    monkeypatch.setattr(L, "LOCK_PATH", str(tmp_path / "l"))
    with L.benchmark_lock():
        with pytest.raises(L.BenchmarkBusy, match="another benchmark"):
            with L.benchmark_lock():
                pass


def test_lock_is_released_after_use(tmp_path, monkeypatch):
    import wirl.lock as L
    monkeypatch.setattr(L, "LOCK_PATH", str(tmp_path / "l"))
    with L.benchmark_lock():
        pass
    with L.benchmark_lock():
        pass


def _ok(label_cfg, mean):
    from wirl.runner import RunResult
    return RunResult(label_cfg, [mean], [], 0, 0.0, True)


def test_repeatability_reports_cross_launch_spread():
    """A config measured in several phases is an accidental control: it gives
    the real uncertainty on every other comparison in the table."""
    from wirl import tune
    cfg = RunConfig(model="/m", n_cpu_moe=43, threads=32)
    out = tune.repeatability([_ok(cfg, 9.90), _ok(cfg, 10.01), _ok(cfg, 10.08)])
    assert "measured 3x" in out
    assert "1.8%" in out
    assert "noise, not findings" in out


def test_repeatability_silent_when_nothing_repeats():
    from wirl import tune
    a = RunConfig(model="/m", n_cpu_moe=42)
    b = RunConfig(model="/m", n_cpu_moe=43)
    assert tune.repeatability([_ok(a, 9.9), _ok(b, 10.0)]) == ""


def test_readiness_requires_a_real_generation(monkeypatch):
    """/health returns 200 while the model is still loading, and a completion
    then fails with 503 "Loading model". Trusting /health alone declares a
    server ready seconds after launch and invalidates every timing after it.
    """
    import wirl.runner as R

    class FakeProc:
        def poll(self):
            return None

    calls = {"n": 0}

    class FakeResp:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(R.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    monkeypatch.setattr(R.time, "sleep", lambda s: None)

    def gen(host, port, n, prompt=None, timeout=3600):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 Loading model")
        return 10.0, 50.0, 5

    monkeypatch.setattr(R, "_generate", gen)
    assert R._wait_health("h", 1, FakeProc(), timeout=60) is not None
    assert calls["n"] == 3, "should have retried until generation succeeded"


@pytest.mark.parametrize("uuid", [None, "GPU-one"])
def test_server_pins_selected_uuid(monkeypatch, uuid):
    from wirl import runner
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "inherited")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: calls.append(kw) or object())
    monkeypatch.setattr(runner, "_wait_health", lambda *a: 1.0)
    monkeypatch.setattr(runner, "_stop", lambda *a: None)
    with runner.server(RunConfig("m", gpu_uuid=uuid), "s", verbose=False):
        pass
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == (uuid or "inherited")
    assert runner.os.environ["CUDA_VISIBLE_DEVICES"] == "inherited"


def test_vram_reads_selected_uuid(monkeypatch):
    from wirl import probe, runner
    monkeypatch.setattr(probe, "gpu_info", lambda: [
        {"index": 0, "uuid": "GPU-zero", "vram_used": 10},
        {"index": 1, "uuid": "GPU-one", "vram_used": 20}])
    assert runner.gpu_used_bytes("GPU-one") == 20
    assert runner.gpu_used_bytes() == 10
    assert runner.gpu_used_bytes("GPU-missing") == 0


def test_foreign_users_filter_selected_gpu(monkeypatch):
    from wirl import lock, probe
    monkeypatch.setattr(probe, "gpu_processes", lambda: [
        {"pid": "42", "name": "other-server", "vram_mb": "1000", "gpu_uuid": "GPU-zero"},
        {"pid": "43", "name": "selected-server", "vram_mb": "2000", "gpu_uuid": "GPU-one"}])
    assert [p["pid"] for p in lock.foreign_gpu_users(gpu_uuid="GPU-one")] == [43]
    assert len(lock.foreign_gpu_users()) == 2


def test_launcher_exports_measured_gpu():
    script = emit.launch_script(RunConfig("m", gpu_uuid="GPU-one"), "s")
    assert "export CUDA_VISIBLE_DEVICES=GPU-one" in script
    assert script.index("export CUDA_VISIBLE_DEVICES") < script.index("exec ")
    assert "CUDA_VISIBLE_DEVICES" not in emit.launch_script(RunConfig("m"), "s")


def test_run_config_measures_selected_gpu(monkeypatch):
    from contextlib import contextmanager
    from wirl import runner
    seen = []

    @contextmanager
    def fake_server(*a, **kw):
        yield 1.0

    monkeypatch.setattr(runner, "server", fake_server)
    monkeypatch.setattr(runner, "_generate", lambda *a: (10.0, 20.0, 1))
    monkeypatch.setattr(runner, "gpu_used_bytes", lambda uuid=None: seen.append(uuid) or 123)
    result = runner.run_config(RunConfig("m", gpu_uuid="GPU-one"), "s", reps=1, verbose=False)
    assert result.ok and result.peak_vram == 123
    assert seen and all(uuid == "GPU-one" for uuid in seen)


def test_read_only_lock_is_a_clear_error(tmp_path, monkeypatch):
    import os
    from wirl import lock
    if os.geteuid() == 0:
        pytest.skip("root bypasses file permissions")
    path = tmp_path / "lock"
    path.write_text("existing")
    path.chmod(0o444)
    monkeypatch.setattr(lock, "LOCK_PATH", str(path))
    with pytest.raises(lock.LockUnavailable, match="owner:.*WIRL_LOCK"):
        with lock.benchmark_lock():
            pass


def test_existing_lock_is_opened_without_create(tmp_path, monkeypatch):
    import os
    from wirl import lock
    path = tmp_path / "lock"
    path.write_text("existing")
    inode = path.stat().st_ino
    monkeypatch.setattr(lock, "LOCK_PATH", str(path))
    real_open, flags = os.open, []

    def opened(path, opts, *a):
        flags.append(opts)
        return real_open(path, opts, *a)

    monkeypatch.setattr(lock.os, "open", opened)
    with lock.benchmark_lock():
        assert path.stat().st_ino == inode
    assert flags == [os.O_RDWR]


def test_new_lock_is_shared_despite_umask(tmp_path, monkeypatch):
    import os
    from wirl import lock
    path = tmp_path / "lock"
    monkeypatch.setattr(lock, "LOCK_PATH", str(path))
    previous = os.umask(0o077)
    try:
        with lock.benchmark_lock():
            assert path.stat().st_mode & 0o777 == 0o666
    finally:
        os.umask(previous)


def test_lock_creation_race_retries_plain_open(tmp_path, monkeypatch):
    import os
    from wirl import lock
    path = tmp_path / "lock"
    monkeypatch.setattr(lock, "LOCK_PATH", str(path))
    real_open, flags = os.open, []

    def opened(filename, opts, *a):
        flags.append(opts)
        if len(flags) == 1:
            path.write_text("created by another process")
            raise FileNotFoundError(filename)
        return real_open(filename, opts, *a)

    monkeypatch.setattr(lock.os, "open", opened)
    with lock.benchmark_lock():
        pass
    assert flags == [os.O_RDWR, os.O_RDWR | os.O_CREAT | os.O_EXCL, os.O_RDWR]
