"""Probe compilation failures and cache behaviour, without running a probe."""
import subprocess
from pathlib import Path

import pytest

from wirl import membw


def test_compile_failure_reports_compiler_reason(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CC", "/bin/false")
    monkeypatch.setattr(membw, "cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(membw, "_numpy_stream", lambda *a: None)
    monkeypatch.setattr(membw.subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 1, "", "synthetic compiler diagnostic"))
    assert membw.build_probe(force=True) is None
    assert "compile failed" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="compile failed.*synthetic compiler diagnostic"):
        membw.measure(threads=1, gib=1)


def test_missing_compiler_is_distinguished(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(membw.shutil, "which", lambda *a: None)
    monkeypatch.setattr(membw, "cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(membw, "_numpy_stream", lambda *a: None)
    assert membw.build_probe() is None
    assert "no C compiler" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="no C compiler"):
        membw.measure(threads=1, gib=1)


@pytest.fixture
def compiler(monkeypatch, tmp_path):
    monkeypatch.setenv("CC", "mock-cc")
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(membw, "cache_dir", lambda: str(cache))
    source = tmp_path / "probe.c"
    source.write_text("first source")
    monkeypatch.setattr(membw, "CSRC", str(source))
    calls = []

    def compile(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"complete binary")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(membw.subprocess, "run", compile)
    return source, cache, calls


def test_source_content_has_its_own_cache_entry(compiler, monkeypatch, tmp_path):
    source, cache, calls = compiler
    first = membw.build_probe()
    second_source = tmp_path / "other.c"
    second_source.write_text("second source")
    monkeypatch.setattr(membw, "CSRC", str(second_source))
    second = membw.build_probe(force=True)
    assert first != second
    assert Path(first).exists() and Path(second).exists()
    # Identical source in another installation should share the first entry.
    duplicate = tmp_path / "duplicate.c"
    duplicate.write_bytes(source.read_bytes())
    monkeypatch.setattr(membw, "CSRC", str(duplicate))
    assert membw.build_probe() == first
    assert len(calls) == 2


def test_correct_cache_entry_skips_compilation(compiler, monkeypatch):
    first = membw.build_probe()

    def unexpected(*a, **kw):
        pytest.fail("compiler invoked for an existing content hash")

    monkeypatch.setattr(membw.subprocess, "run", unexpected)
    assert membw.build_probe() == first


def test_compiler_command_is_part_of_hash(compiler, monkeypatch):
    first = membw.build_probe()
    monkeypatch.setenv("CC", "mock-cc -fno-omit-frame-pointer")
    second = membw.build_probe()
    assert first != second
    assert compiler[2][-1][:2] == ["mock-cc", "-fno-omit-frame-pointer"]


def test_failed_rebuild_keeps_complete_binary_and_cleans_temp(compiler, monkeypatch):
    first = membw.build_probe()

    def failed(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"partial binary")
        return subprocess.CompletedProcess(cmd, 1, "", "compile error")

    monkeypatch.setattr(membw.subprocess, "run", failed)
    assert membw.build_probe(force=True) is None
    assert Path(first).read_bytes() == b"complete binary"
    assert list(compiler[1].iterdir()) == [Path(first)]


def test_concurrent_compiles_use_separate_temporary_outputs(compiler, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    barrier, outputs = threading.Barrier(2), []

    def compile(cmd, **kw):
        outputs.append(cmd[-1])
        Path(cmd[-1]).write_bytes(b"complete binary")
        barrier.wait(timeout=2)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(membw.subprocess, "run", compile)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: membw.build_probe(force=True), range(2)))
    assert results[0] == results[1] and results[0] is not None
    assert len(set(outputs)) == 2
    assert all(output != results[0] for output in outputs)
    assert Path(results[0]).read_bytes() == b"complete binary"
    assert list(compiler[1].iterdir()) == [Path(results[0])]
