"""Reject memory pressure without allocating buffers or starting a probe."""
import subprocess

import pytest

from wirl import membw, probe

pytestmark = pytest.mark.usefixtures("isolated_runtime")


@pytest.fixture(autouse=True)
def fake_machine(monkeypatch):
    monkeypatch.setattr(probe, "mem_info", lambda: {"available": 100 << 30, "free": 100 << 30})
    monkeypatch.setattr(probe, "swap_activity", lambda: {"pswpin": 10, "pswpout": 20})
    monkeypatch.setattr(membw, "_build_probe", lambda: ("fake-probe", None))


@pytest.mark.parametrize("free,available,expected", [(100, 100, 32), (8, 25, 2), (100, 6, 2)])
def test_default_buffer_leaves_unused_ram_headroom(monkeypatch, free, available, expected):
    monkeypatch.setattr(probe, "mem_info", lambda: {"free": free << 30, "available": available << 30})
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "stream 0 1 2000000000\n", ""))
    result = membw.measure(threads=1)
    assert result.gib == expected
    assert calls[0][3] == str(expected)


@pytest.mark.parametrize("free,available,gib", [(0, 25, None), (1, 25, None),
                                              (100, 1, None), (8, 25, 8)])
def test_insufficient_unused_ram_refuses_before_launch(monkeypatch, free, available, gib):
    monkeypatch.setattr(probe, "mem_info", lambda: {"free": free << 30, "available": available << 30})
    with pytest.raises(RuntimeError, match="unused RAM"):
        membw.measure(threads=1, gib=gib)


@pytest.mark.parametrize("gib", [0, -1])
def test_invalid_buffer_size_is_rejected(gib):
    with pytest.raises(ValueError, match="positive"):
        membw.measure(threads=1, gib=gib)


@pytest.mark.parametrize("page_in,page_out", [(1, 0), (0, 1), (7, 9)])
def test_concurrent_swapping_invalidates_c_results(monkeypatch, page_in, page_out):
    values = iter([{"pswpin": 10, "pswpout": 20},
                   {"pswpin": 10 + page_in, "pswpout": 20 + page_out}])
    monkeypatch.setattr(probe, "swap_activity", lambda: next(values))
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 0, "stream 0 1 2000000000\n", ""))
    with pytest.raises(RuntimeError, match=f"swapping.*{page_in} pages in.*{page_out} pages out"):
        membw.measure(threads=1, gib=1)


def test_historical_swap_use_does_not_invalidate_idle_measurement(monkeypatch):
    monkeypatch.setattr(probe, "mem_info", lambda:
                        {"available": 25 << 30, "free": 8 << 30, "swap_used": 1 << 30})
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 0, "stream 0 1 2000000000\n", ""))
    result = membw.measure(threads=1, gib=1)
    assert result.samples == [2] and result.method == "c-probe"


@pytest.mark.parametrize("swapping", [False, True])
def test_numpy_fallback_checks_swap_activity(monkeypatch, swapping):
    values = iter([{"pswpin": 10, "pswpout": 20},
                   {"pswpin": 11 if swapping else 10, "pswpout": 20}])
    monkeypatch.setattr(probe, "swap_activity", lambda: next(values))
    monkeypatch.setattr(membw, "_build_probe", lambda: (None, "no compiler"))
    fallback = membw.BwResult("stream", 1, 1, [2], "fake numpy")
    monkeypatch.setattr(membw, "_numpy_stream", lambda *args: fallback)
    if swapping:
        with pytest.raises(RuntimeError, match="swapping"):
            membw.measure(threads=1, gib=1)
    else:
        assert membw.measure(threads=1, gib=1) is fallback
