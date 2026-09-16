"""Acceptance decisions and collection against fake probes, never hardware."""
import itertools
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
import runpy

import pytest

from wirl import membw, probe

pytestmark = pytest.mark.usefixtures("isolated_runtime")


@pytest.fixture
def comparison(monkeypatch):
    result = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/compare_probe_bandwidth.py"))
    monkeypatch.setitem(result["main"].__globals__, "benchmark_lock", lambda **kwargs: nullcontext())
    return result


def test_containment_is_unreliable_even_for_equal_distributions():
    assignments = list(itertools.combinations(range(6), 3))
    contained = 0
    for local in assignments:
        prebuilt = [v for v in range(6) if v not in local]
        contained += min(local) <= min(prebuilt) and max(prebuilt) <= max(local)
    assert len(assignments) == 20 and contained == 4


@pytest.mark.parametrize("local,prebuilt,passed", [
    ([99, 100, 101], [98.9, 100, 101.1], True),
    ([100, 100, 102], [99, 99, 99], True),
    ([98, 100, 102], [107, 108, 109], False),
    ([100, 100, 100], [100, 100, 100], True),
    ([100, 100, 100], [100.1, 100.1, 100.1], False),
    ([99, 100, 101], [70, 100, 130], False),
    ([70, 100, 130], [99, 100, 101], False),
])
def test_compare_effect_with_spread_and_reject_instability(comparison, local, prebuilt, passed):
    result = comparison["compare_samples"]([local] * 3, [prebuilt] * 3)
    assert result["passed"] is passed
    assert result["local_median"] == sorted(local)[1]
    assert result["prebuilt_median"] == sorted(prebuilt)[1]


@pytest.mark.parametrize("samples", [[], [0], [-1], [float("inf")], [float("nan")]])
def test_invalid_samples_are_rejected(comparison, samples):
    with pytest.raises(ValueError, match="finite, positive"):
        comparison["compare_samples"]([samples] * 3, [[1]] * 3)


@pytest.mark.parametrize("count", [2, 4])
def test_wrong_launch_count_is_rejected(comparison, count):
    with pytest.raises(ValueError, match="three launches"):
        comparison["compare_samples"]([[1]] * count, [[1]] * 3)


@pytest.fixture
def fake_probes(monkeypatch, tmp_path):
    packaged, local, source = [tmp_path / n for n in ("packaged", "local", "source.c")]
    for path in (packaged, local):
        path.write_bytes(b"fake probe")
        path.chmod(0o755)
    source.write_text("fake source")
    monkeypatch.setattr(membw, "PACKAGED_PROBE", str(packaged))
    monkeypatch.setattr(membw, "CSRC", str(source))
    monkeypatch.setattr(membw, "_buffer_gib", lambda gib: 2)
    monkeypatch.setattr(membw, "build_probe", lambda force=False: str(local) if force else str(packaged))
    monkeypatch.setattr(probe, "physical_core_cpus", lambda: [2, 4])
    monkeypatch.setattr(probe, "cpu_info", lambda: {"model": "fake CPU", "flags": set()})
    monkeypatch.setattr(probe, "mem_info", lambda: {"free": 8 << 30, "available": 25 << 30})
    monkeypatch.setattr(probe, "swap_activity", lambda: {"pswpin": 10, "pswpout": 20})
    monkeypatch.setattr("os.getloadavg", lambda: (0, 0, 0))
    calls = []

    def measure(path, mode, threads, gib, reps, cores):
        calls.append((path, mode, threads, gib, reps, cores))
        return membw.BwResult(mode, threads, gib, [99, 100, 101], "fake")

    monkeypatch.setattr(membw, "_measure_probe", measure)
    return packaged, local, calls


@pytest.mark.parametrize("old_cache", [None, "existing-cache"])
def test_collect_alternates_and_preserves_cache(comparison, fake_probes, monkeypatch, old_cache):
    packaged, local, calls = fake_probes
    if old_cache is None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_CACHE_HOME", old_cache)
    result = comparison["collect"]()
    assert result["passed"] and len(result["runs"]) == 12
    assert [c[0] for c in calls] == [str(packaged), str(local)] * 6
    assert [c[1] for c in calls] == ["stream"] * 6 + ["gather"] * 6
    assert all(c[2:] == (2, 2, 3, [2, 4]) for c in calls)
    assert comparison["os"].environ.get("XDG_CACHE_HOME") == old_cache
    assert result["swap_delta"] == {"pswpin": 0, "pswpout": 0}


def test_swap_guard_aborts_further_launches(comparison, fake_probes, monkeypatch):
    calls = []
    def fail(*args):
        calls.append(args)
        raise RuntimeError("swapping detected")
    monkeypatch.setattr(membw, "_measure_probe", fail)
    result = comparison["collect"]()
    assert not result["passed"] and result["error"] == "swapping detected"
    assert len(calls) == 1 and result["runs"] == []


@pytest.mark.parametrize("counter,passed", [("pswpin", True), ("pswpout", False)])
def test_global_swap_attribution(comparison, fake_probes, monkeypatch, counter, passed):
    counts = {"pswpin": 10, "pswpout": 20}
    original = membw._measure_probe
    def measure(*args):
        result = original(*args)
        counts[counter] += 1
        return result
    monkeypatch.setattr(membw, "_measure_probe", measure)
    monkeypatch.setattr(probe, "swap_activity", lambda: counts.copy())
    result = comparison["collect"]()
    assert result["passed"] is passed
    assert result["swap_delta"][counter] == 12


@pytest.mark.parametrize("samples", [[], [100, 101]])
def test_incomplete_probe_output_aborts_comparison(comparison, fake_probes, monkeypatch, samples):
    monkeypatch.setattr(membw, "_measure_probe", lambda *args:
                        membw.BwResult("stream", 2, 2, samples, "fake"))
    result = comparison["collect"]()
    assert not result["passed"] and "samples; expected 3" in result["error"]
    assert not result["runs"]


@pytest.mark.parametrize("problem", ["package", "topology", "compiler", "reps"])
def test_preflight_failure_never_measures(comparison, fake_probes, monkeypatch, problem):
    if problem == "package":
        monkeypatch.setattr(membw, "PACKAGED_PROBE", "absent-probe")
    elif problem == "topology":
        monkeypatch.setattr(probe, "physical_core_cpus", lambda: None)
    elif problem == "compiler":
        monkeypatch.setattr(membw, "build_probe", lambda **kw: None)
    error = ValueError if problem == "reps" else RuntimeError
    with pytest.raises(error):
        comparison["collect"](reps=0 if problem == "reps" else 3)
    assert not fake_probes[2]


@pytest.mark.parametrize("passed", [False, True])
def test_main_saves_results_and_returns_acceptance_status(comparison, monkeypatch, tmp_path, passed):
    comparison["main"].__globals__["collect"] = lambda *args: {"passed": passed, "runs": [], "summary": {}}
    output = tmp_path / "results.json"
    assert comparison["main"](["--output", str(output)]) == (0 if passed else 1)
    assert json.loads(output.read_text())["passed"] is passed


def test_main_retains_preflight_diagnostic(comparison, tmp_path):
    def fail(*args):
        raise RuntimeError("not enough unused RAM")
    comparison["main"].__globals__["collect"] = fail
    output = tmp_path / "results.json"
    assert comparison["main"](["--output", str(output)]) == 1
    assert json.loads(output.read_text()) == {"passed": False, "error": "not enough unused RAM"}


def test_busy_lock_refuses_before_any_measurement(comparison, monkeypatch, tmp_path):
    from wirl.lock import BenchmarkBusy
    @contextmanager
    def busy(**kwargs):
        raise BenchmarkBusy("another benchmark holds the lock")
        yield
    def unexpected(*args):
        pytest.fail("comparison started despite an occupied benchmark lock")
    monkeypatch.setitem(comparison["main"].__globals__, "benchmark_lock", busy)
    monkeypatch.setitem(comparison["main"].__globals__, "collect", unexpected)
    output = tmp_path / "results.json"
    assert comparison["main"](["--output", str(output)]) == 1
    assert "another benchmark" in json.loads(output.read_text())["error"]
