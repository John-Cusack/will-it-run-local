"""Empirical search: the VRAM boundary is found by launching, not predicting."""
from dataclasses import dataclass, field

import pytest

from wirl import search
from wirl.runner import RunConfig


@dataclass
class FakeResult:
    config: RunConfig
    ok: bool
    samples: list = field(default_factory=list)
    peak_vram: int = 0
    error: str = None
    spread_pct: float = 0.0

    @property
    def mean(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


def _fake_runner(min_starting, tps_at=None, vram_at=None):
    """A server that starts only at or above `min_starting`."""
    calls = []

    def run(cfg, binary, **kw):
        calls.append(cfg.n_cpu_moe)
        n = cfg.n_cpu_moe
        if n < min_starting:
            return FakeResult(cfg, False, error="unable to allocate CUDA0 buffer")
        tps = (tps_at or {}).get(n, 10.0)
        vram = (vram_at or {}).get(n, 20 << 30)
        return FakeResult(cfg, True, [tps], vram)

    run.calls = calls
    return run


def test_finds_the_lowest_config_that_starts(monkeypatch):
    run = _fake_runner(min_starting=41)
    monkeypatch.setattr(search, "run_config", run)
    base = RunConfig(model="/m.gguf")
    edge = search.find_vram_edge(base, "srv", 37, 43, verbose=False)
    assert edge == 41


def test_bisection_is_cheap(monkeypatch):
    """A 0..43 range must cost a handful of launches, not 44."""
    run = _fake_runner(min_starting=20)
    monkeypatch.setattr(search, "run_config", run)
    search.find_vram_edge(RunConfig(model="/m.gguf"), "srv", 0, 43, verbose=False)
    assert len(run.calls) <= 7


def test_returns_none_when_nothing_starts(monkeypatch):
    run = _fake_runner(min_starting=99)
    monkeypatch.setattr(search, "run_config", run)
    edge = search.find_vram_edge(RunConfig(model="/m.gguf"), "srv", 0, 43,
                                 verbose=False)
    assert edge is None


def test_measures_from_the_edge_upward_only(monkeypatch):
    """Configs below the measured edge are known to fail; do not retry them."""
    run = _fake_runner(min_starting=41)
    monkeypatch.setattr(search, "run_config", run)
    res = search.measure_around(RunConfig(model="/m.gguf"), "srv", 41, span=2,
                                verbose=False)
    assert [r.config.n_cpu_moe for r in res] == [41, 42, 43]
    assert all(n >= 41 for n in run.calls)


def test_measure_respects_the_layer_count(monkeypatch):
    run = _fake_runner(min_starting=0)
    monkeypatch.setattr(search, "run_config", run)
    res = search.measure_around(RunConfig(model="/m.gguf"), "srv", 42, span=3,
                                verbose=False, max_layer=43)
    assert max(r.config.n_cpu_moe for r in res) == 43


def _r(n, tps, vram_mib):
    return FakeResult(RunConfig(model="/m", n_cpu_moe=n), True, [tps],
                      vram_mib << 20)


def test_prefers_headroom_over_the_last_two_percent():
    """The fastest config on the reference machine sat at 97% of VRAM and
    later failed to allocate. Headroom wins."""
    total = 24576 << 20
    results = [_r(41, 10.21, 23360), _r(42, 10.10, 23834), _r(43, 9.89, 20570)]
    best, why = search.pick_recommended(results, total, 3 << 30)
    assert best.config.n_cpu_moe == 43
    assert "gives up" in why


def test_falls_back_when_nothing_leaves_headroom():
    total = 24576 << 20
    results = [_r(41, 10.21, 24000), _r(42, 10.10, 23900)]
    best, why = search.pick_recommended(results, total, 3 << 30)
    assert best.config.n_cpu_moe == 41
    assert "no config left headroom" in why


def test_failed_runs_are_never_recommended():
    total = 24576 << 20
    bad = FakeResult(RunConfig(model="/m", n_cpu_moe=40), False, error="boom")
    results = [bad, _r(43, 9.89, 20570)]
    best, _ = search.pick_recommended(results, total, 3 << 30)
    assert best.config.n_cpu_moe == 43


def test_no_successful_runs_returns_nothing():
    bad = FakeResult(RunConfig(model="/m", n_cpu_moe=40), False, error="boom")
    best, _ = search.pick_recommended([bad], 24576 << 20)
    assert best is None


def test_vram_byte_counts_are_integers():
    """nvidia-smi returns decimal strings; these values get bit-shifted and
    formatted as integers throughout, so a float here is a crash later."""
    from wirl import probe
    for g in probe.gpu_info():
        for k in ("vram_total", "vram_used", "vram_free"):
            assert isinstance(g[k], int), f"{k} is {type(g[k]).__name__}"


# --- dense models: the knob and the direction both change -------------------

def _fake_ngl_runner(max_starting):
    """A server that starts only at or below `max_starting` -ngl."""
    calls = []

    def run(cfg, binary, **kw):
        calls.append(cfg.n_gpu_layers)
        if cfg.n_gpu_layers > max_starting:
            return FakeResult(cfg, False, error="unable to allocate CUDA0 buffer")
        return FakeResult(cfg, True, [10.0], 20 << 30)

    run.calls = calls
    return run


def test_dense_edge_is_the_largest_ngl_that_starts(monkeypatch):
    """--n-gpu-layers runs the opposite way to --n-cpu-moe: higher means more
    on the GPU. Bisecting in the wrong direction silently returns the slowest
    configuration that works instead of the fastest."""
    run = _fake_ngl_runner(max_starting=28)
    monkeypatch.setattr(search, "run_config", run)
    edge = search.find_ngl_edge(RunConfig(model="/m.gguf"), "srv", 0, 40,
                                verbose=False)
    assert edge == 28


def test_dense_edge_never_emits_n_cpu_moe(monkeypatch):
    """--n-cpu-moe is meaningless on a dense model and must not be passed."""
    captured = []

    def run(cfg, binary, **kw):
        captured.append(cfg)
        return FakeResult(cfg, True, [10.0], 1)

    monkeypatch.setattr(search, "run_config", run)
    search.find_ngl_edge(RunConfig(model="/m.gguf"), "srv", 0, 4, verbose=False)
    assert captured, "no launches happened"
    for cfg in captured:
        assert cfg.n_cpu_moe is None
        assert "--n-cpu-moe" not in cfg.argv("srv")


def test_dense_measures_downward_from_the_edge(monkeypatch):
    """Below the edge is the safe side for -ngl, so that is where headroom is."""
    run = _fake_ngl_runner(max_starting=30)
    monkeypatch.setattr(search, "run_config", run)
    res = search.measure_around(RunConfig(model="/m.gguf"), "srv", 30, span=2,
                                verbose=False, moe=False)
    assert [r.config.n_gpu_layers for r in res] == [30, 29, 28]


def test_moe_still_measures_upward(monkeypatch):
    run = _fake_runner(min_starting=41)
    monkeypatch.setattr(search, "run_config", run)
    res = search.measure_around(RunConfig(model="/m.gguf"), "srv", 41, span=2,
                                verbose=False, moe=True, max_layer=43)
    assert [r.config.n_cpu_moe for r in res] == [41, 42, 43]


def test_label_reports_the_knob_actually_in_use():
    assert "ncmoe=43" in RunConfig(model="/m", n_cpu_moe=43).label()
    dense = RunConfig(model="/m", n_cpu_moe=None, n_gpu_layers=28)
    assert "ngl=28" in dense.label()
    assert "ncmoe" not in dense.label()


# --- long-context reporting -------------------------------------------------

def _dp(prompt_n, prefill, decode, ttft):
    return search.DepthPoint(prompt_n, prompt_n, prefill, decode, ttft)


def test_flat_decode_is_reported_as_unaffected_not_as_a_zero_percent_fall():
    out = search.summarise_depth(
        [_dp(23, 17, 10.15, 1.4), _dp(7509, 63, 10.83, 119.7)], 16384)
    assert "unaffected by context depth" in out
    assert "falls 0%" not in out


def test_real_decode_regression_is_reported():
    out = search.summarise_depth(
        [_dp(23, 17, 10.0, 1.4), _dp(7509, 63, 6.0, 20.0)], 16384)
    assert "falls 40%" in out


def test_slow_first_token_is_called_out():
    """A two-minute wait matters more than a few percent of decode."""
    out = search.summarise_depth(
        [_dp(23, 17, 10.15, 1.4), _dp(7509, 63, 10.83, 119.7)], 16384)
    assert "120 seconds to the first token" in out
    assert "COLD case" in out
    assert "cache-ram" in out


def test_fast_first_token_is_not_nagged_about():
    out = search.summarise_depth(
        [_dp(23, 17, 10.1, 1.4), _dp(512, 200, 10.2, 2.6)], 16384)
    assert "seconds to the first token" not in out


# --- machines with no GPU at all --------------------------------------------

def _no_gpu(monkeypatch):
    """nvidia-smi ignores CUDA_VISIBLE_DEVICES, so the only honest way to test
    the no-GPU path is to remove the GPU at the source."""
    from wirl import probe
    monkeypatch.setattr(probe, "gpu_info", lambda: [])
    monkeypatch.setattr(probe, "gpu_processes", lambda: [])


def test_auto_refuses_without_a_gpu_and_says_why(monkeypatch, capsys):
    from wirl import cli
    _no_gpu(monkeypatch)
    import pytest as _pytest
    with _pytest.raises(SystemExit) as e:
        cli.main(["auto", "/nonexistent-but-never-read.gguf", "--mem-bandwidth", "20"])
    # It must fail on something explicable, not a traceback.
    assert e.value.code != 0


def test_plan_falls_back_to_a_cpu_only_estimate(monkeypatch, capsys, moe_model):
    from wirl import cli
    _no_gpu(monkeypatch)
    cli.main(["plan", moe_model, "--mem-bandwidth", "20"])
    out = capsys.readouterr().out
    assert "none -- CPU only" in out
    assert "CPU only:" in out
    # and it must not pretend the estimate is trustworthy
    assert "well below this" in out


def test_auto_pins_selected_gpu(monkeypatch, tmp_path, moe_model):
    from wirl import cli, doctor, lock, probe
    from wirl.runner import RunResult
    monkeypatch.setattr(cli, "find_server", lambda *a: "mock-server")
    monkeypatch.setattr(lock, "LOCK_PATH", str(tmp_path / "lock"))
    monkeypatch.setattr(probe, "cpu_info", lambda: {"model": "test", "physical": 64})
    monkeypatch.setattr(probe, "gpu_info", lambda: [
        {"index": 1, "uuid": "GPU-one", "name": "test", "vram_total": 24 << 30}])
    checks, calls = [], []
    monkeypatch.setattr(doctor, "run_all", lambda *a, **kw: checks.append(kw) or [])

    def run(cfg, binary, **kw):
        calls.append(cfg)
        return RunResult(cfg, [10.0], [], 100, 1.0, True)

    monkeypatch.setattr(search, "run_config", run)
    out = tmp_path / "run.sh"
    assert cli.main(["auto", moe_model, "--gpu", "1", "--mem-bandwidth", "20",
                     "--no-depth", "--emit", str(out)]) == 0
    assert checks[0]["gpu_index"] == 1
    assert calls and all(c.gpu_uuid == "GPU-one" for c in calls)
    assert "export CUDA_VISIBLE_DEVICES=GPU-one" in out.read_text()
