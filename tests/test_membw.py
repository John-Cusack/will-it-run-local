"""Probe compilation failures and cache behaviour, without running a probe."""
import subprocess

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
