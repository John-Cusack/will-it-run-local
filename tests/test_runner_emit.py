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
    s = emit.launch_script(cfg, "/bin/llama-server", physical_cores=64)
    assert "taskset -c 0-63" in s
    assert s.startswith("#!/usr/bin/env bash")
    assert "--n-cpu-moe 43" in s


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
