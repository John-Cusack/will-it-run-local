"""Bandwidth defaults, parser and failures without real allocations."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wirl import membw, probe

pytestmark = pytest.mark.usefixtures("isolated_runtime")


@pytest.fixture(autouse=True)
def idle_memory(monkeypatch):
    monkeypatch.setattr(probe, "mem_info", lambda: {"free": 100 << 30, "available": 100 << 30})
    monkeypatch.setattr(probe, "swap_activity", lambda: {"pswpin": 0, "pswpout": 0})
    monkeypatch.setattr("resource.getrusage", lambda who: SimpleNamespace(ru_majflt=0))


def test_cache_directory_respects_xdg_and_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert membw.cache_dir() == str(tmp_path / "xdg" / "will-it-run-local")
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setattr(membw.os.path, "expanduser", lambda p: str(tmp_path / "home"))
    assert membw.cache_dir() == str(tmp_path / "home" / "will-it-run-local")


@pytest.mark.parametrize("kind,expected", [("platform", "Linux only"), ("command", "invalid compiler command"),
                                          ("source", "cannot read bandwidth probe source"), ("process", "compile failed")])
def test_build_failure_diagnostics(monkeypatch, tmp_path, kind, expected):
    monkeypatch.setattr(membw, "PACKAGED_PROBE", str(tmp_path / "absent"))
    monkeypatch.setattr(membw, "CSRC", str(tmp_path / "missing.c"))
    monkeypatch.setenv("CC", "fake-cc")
    if kind == "platform":
        monkeypatch.setattr(membw.sys, "platform", "darwin")
    elif kind == "command":
        monkeypatch.setenv("CC", '"unterminated')
    elif kind == "process":
        source = tmp_path / "source.c"
        source.write_text("fake source")
        monkeypatch.setattr(membw, "CSRC", str(source))
        monkeypatch.setattr(membw, "cache_dir", lambda: str(tmp_path))
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(OSError("missing compiler")))
    path, reason = membw._build_probe()
    assert path is None and expected in reason
    assert not list(tmp_path.glob(".membw-*"))


@pytest.mark.parametrize("stdout,stderr,detail", [("compiler stdout", "", "compiler stdout"), ("", "", "exit 7")])
def test_compiler_nonzero_without_stderr(monkeypatch, tmp_path, stdout, stderr, detail):
    source = tmp_path / "source.c"
    source.write_text("fake source")
    monkeypatch.setattr(membw, "CSRC", str(source))
    monkeypatch.setattr(membw, "PACKAGED_PROBE", str(tmp_path / "absent"))
    monkeypatch.setattr(membw, "cache_dir", lambda: str(tmp_path))
    monkeypatch.setenv("CC", "fake-cc")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 7, stdout, stderr))
    assert detail in membw._build_probe()[1]


def test_result_statistics_empty_single_and_spread():
    empty = membw.BwResult("stream", 1, 1, [], "fake")
    assert empty.best == empty.median == empty.spread_pct == 0
    assert membw.BwResult("stream", 1, 1, [0, 0], "fake").spread_pct == 0
    single = membw.BwResult("stream", 1, 1, [4], "fake")
    assert single.best == single.median == 4 and single.spread_pct == 0
    result = membw.BwResult("stream", 1, 1, [2, 4, 6], "fake")
    assert result.best == 6 and result.median == 4 and result.spread_pct == 100


def test_numpy_fallback_uses_fake_array_and_clock(monkeypatch):
    assert membw._numpy_stream(1, 1) is None
    sizes = []
    fake = SimpleNamespace(uint64="uint64", ones=lambda n, dtype:
                           sizes.append((n, dtype)) or SimpleNamespace(sum=lambda: 12))
    monkeypatch.setitem(sys.modules, "numpy", fake)
    ticks = iter([0, 1, 2, 4])
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    result = membw._numpy_stream(1, 2)
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert sizes == [(1<<27, "uint64")]
    assert result.samples == pytest.approx([1.073741824, .536870912])
    assert result.threads == 1 and "lower bound" in result.method


def test_measure_defaults_cores_output_and_failures(monkeypatch):
    monkeypatch.setattr(probe, "cpu_info", lambda: {"physical": 4})
    monkeypatch.setattr(probe, "mem_info", lambda: {"available": 100<<30, "free": 100<<30})
    monkeypatch.setattr(membw, "_build_probe", lambda: ("fake-probe", None))
    calls = []
    text = "banner\nstream 0 1 2000000000\nstream 1 0 2000000000\n"
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        calls.append((cmd, kw)) or subprocess.CompletedProcess(cmd, 0, text, ""))
    result = membw.measure(cores=[2, 4])
    assert result.samples == [2] and result.threads == 4 and result.gib == 32
    assert calls[0][0] == ["fake-probe", "stream", "4", "32", "3", "2,4"]
    assert calls[0][1]["timeout"] == 900
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "failed\n"))
    with pytest.raises(RuntimeError, match="bandwidth probe failed: failed"):
        membw.measure(threads=1, gib=1)


def test_measure_fallback_and_both_modes(monkeypatch):
    fallback = membw.BwResult("stream", 1, 1, [1], "lower bound")
    monkeypatch.setattr(membw, "_build_probe", lambda: (None, "no compiler"))
    monkeypatch.setattr(membw, "_numpy_stream", lambda *a: fallback)
    assert membw.measure(threads=1, gib=1) is fallback
    calls = []
    monkeypatch.setattr(membw, "measure", lambda *a: calls.append(a) or a[0])
    assert membw.measure_both(2, 4, 5) == {"stream": "stream", "gather": "gather"}
    assert calls == [("stream", 2, 4, 5), ("gather", 2, 4, 5)]
