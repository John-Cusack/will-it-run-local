"""CLI sweeps with synthetic headers and entirely mocked server launches."""
import pytest

from wirl import cli, predict, probe, tune
from wirl.runner import RunConfig, RunResult


@pytest.fixture
def dense_model(tmp_path, build_gguf):
    kv = {"general.architecture": (8, "dense"),
          "dense.block_count": (4, 4), "dense.embedding_length": (4, 8),
          "dense.attention.head_count": (4, 2)}
    tensors = [(f"blk.{i}.attn_q.weight", (4096, 32768), 0) for i in range(4)]
    return build_gguf(tmp_path / "dense.gguf", kv, tensors)


@pytest.fixture
def tune_calls(monkeypatch, tmp_path):
    import wirl.lock as lock
    monkeypatch.setattr(lock, "LOCK_PATH", str(tmp_path / "lock"))
    monkeypatch.setattr(cli, "find_server", lambda *a: "mock-server")
    monkeypatch.setattr(cli, "foreign_gpu_users", lambda **kw: [])
    monkeypatch.setattr(probe, "gpu_info", lambda: [
        {"index": 0, "name": "RTX 3090", "vram_total": 24 << 30,
         "vram_used": 0, "uuid": "GPU-test"}])
    calls = []

    def run(cfg, binary, **kw):
        calls.append(cfg)
        speed = 20 if cfg.n_gpu_layers == 2 else 10
        return RunResult(cfg, [speed], [], 1 << 30, 1.0, True)

    monkeypatch.setattr(tune, "run_config", run)
    return calls


def test_dense_tune_sweeps_partial_offload(dense_model, tune_calls, capsys):
    assert cli.main(["tune", dense_model, "--vram", "1.5", "--headroom", "0",
                     "--mem-bandwidth", "20"]) == 0
    assert tune_calls
    values = [c.n_gpu_layers for c in tune_calls]
    assert values == sorted(set(values)) and len(values) > 1
    assert all(c.n_cpu_moe is None for c in tune_calls)
    assert all("--n-cpu-moe" not in c.argv("s") for c in tune_calls)
    assert "Measured sweep: --n-gpu-layers" in capsys.readouterr().out


def test_moe_tune_keeps_descending_candidates(moe_model, tune_calls):
    assert cli.main(["tune", moe_model, "--mem-bandwidth", "20"]) == 0
    assert [c.n_cpu_moe for c in tune_calls] == [2, 1, 0]
    assert all(c.n_gpu_layers == 99 for c in tune_calls)


def test_dense_draft_depth_keeps_winning_offload(dense_model, moe_model, tune_calls):
    assert cli.main(["tune", dense_model, "--vram", "1.5", "--headroom", "0",
                     "--mem-bandwidth", "20", "--draft", moe_model,
                     "--depth-sweep"]) == 0
    depth_calls = tune_calls[-3:]
    assert [c.draft_n_max for c in depth_calls] == [1, 2, 3]
    assert all(c.n_gpu_layers == 2 and c.n_cpu_moe is None for c in depth_calls)


def test_dense_failed_verdict_names_dense_knob():
    out = " ".join(predict.verdict(None, None, 20e9, knob="--n-gpu-layers"))
    assert "--n-gpu-layers" in out and "--n-cpu-moe" not in out


@pytest.mark.parametrize("moe,candidates", [(True, [3, 2, 1]), (False, [1, 2, 3])])
def test_offload_sweep_stops_at_first_failure(monkeypatch, moe, candidates):
    calls = []

    def run(cfg, binary, **kw):
        calls.append(cfg)
        return RunResult(cfg, [], [], 0, 0, False, "allocation failed")

    monkeypatch.setattr(tune, "run_config", run)
    results = tune.sweep_offload(RunConfig("m"), "s", candidates, moe)
    assert len(results) == len(calls) == 1


@pytest.mark.parametrize("index", [3, -1])
def test_invalid_gpu_choice_is_an_error(monkeypatch, moe_model, index):
    monkeypatch.setattr(probe, "gpu_info", lambda: [
        {"index": 0, "name": "only GPU", "uuid": "GPU-zero", "vram_total": 24 << 30}])
    with pytest.raises(SystemExit, match="available GPUs:.*0.*only GPU"):
        cli.main(["plan", moe_model, "--gpu", str(index), "--mem-bandwidth", "20"])


def test_tune_selects_physical_gpu(monkeypatch, moe_model, tune_calls):
    seen = []
    monkeypatch.setattr(probe, "gpu_info", lambda: [
        {"index": 1, "name": "selected", "uuid": "GPU-one", "vram_total": 24 << 30},
        {"index": 0, "name": "other", "uuid": "GPU-zero", "vram_total": 24 << 30}])
    monkeypatch.setattr(cli, "foreign_gpu_users", lambda **kw: seen.append(kw) or [])
    assert cli.main(["tune", moe_model, "--gpu", "1", "--mem-bandwidth", "20"]) == 0
    assert seen == [{"gpu_uuid": "GPU-one"}]
    assert all(c.gpu_uuid == "GPU-one" for c in tune_calls)
