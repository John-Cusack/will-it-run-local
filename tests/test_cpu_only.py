"""F2/F3/F4 regression tests: CPU-only is a supported path.

F2: `recommend` on a GPU-less box routes through `cpu_only_tps` and prints
    real CPU figures -- never 0.0 rows.
F3: `plan` refuses to price unpriceable toys, and its prose matches the 0.70
    constant.
F4: `auto`/`tune` on a GPU-less box print next steps; `tune` runs a CPU
    thread-count sweep whose null result is the deliverable, and never says
    "this GPU" when no GPU was probed.
"""
import re
import pytest
from wirl import cli, model, predict, probe, recommend


def _no_gpu(monkeypatch):
    monkeypatch.setattr(probe, "gpu_info", lambda: [])
    monkeypatch.setattr(probe, "gpu_processes", lambda: [])


def _toy_cost(moe_model):
    from wirl import gguf
    g = gguf.read(moe_model)
    return g, model.build(g)


def test_cpu_only_pricable_rejects_the_toy(moe_model):
    _, mc = _toy_cost(moe_model)
    assert not predict.cpu_only_pricable(mc)


# --- F2 ---------------------------------------------------------------------

def test_evaluate_cpu_only_reports_a_real_figure(tmp_path):
    g, mc = _toy_cost(_bigmoe_model(tmp_path))
    cand = recommend.Candidate(name="Q4_K_M", files=["m.gguf"], size=10,
                               bpw=4.9)
    cand._cost, cand._gguf = mc, g
    recommend.evaluate(cand, 20e9, 1.0, 0, 64 << 30, 8192)
    assert cand.fits_at_all
    assert cand.tps == pytest.approx(predict.cpu_only_tps(mc, 20e9))
    assert cand.tps > 0
    assert cand.confidence == "calibrated"
    assert cand.knob == "--n-cpu-moe" and cand.knob_value == mc.n_layer


def test_choose_names_a_quant_on_cpu_only(tmp_path):
    g, mc = _toy_cost(_bigmoe_model(tmp_path))
    cands = []
    for name, bpw in (("Q4_K_M", 4.9), ("Q6_K", 6.2)):
        c = recommend.Candidate(name=name, files=[f"{name}.gguf"], size=10,
                                bpw=bpw)
        c._cost, c._gguf = mc, g
        recommend.evaluate(c, 20e9, 1.0, 0, 64 << 30, 8192)
        cands.append(c)
    best, _why = recommend.choose(cands, min_tps=5.0)
    assert best is not None and best.tps > 0


def test_evaluate_cpu_only_refuses_the_toy(moe_model):
    g, mc = _toy_cost(moe_model)
    cand = recommend.Candidate(name="Q4_K_M", files=["m.gguf"], size=10)
    cand._cost, cand._gguf = mc, g
    recommend.evaluate(cand, 20e9, 1.0, 0, 64 << 30, 8192)
    assert not cand.fits_at_all
    assert "too small to price" in cand.note


def test_no_fit_at_this_ctx_is_unusable_not_zero(moe_model):
    """GPU path with nothing fitting must mark --, never a 0.0 row."""
    g, mc = _toy_cost(moe_model)
    cand = recommend.Candidate(name="Q4_K_M", files=["m.gguf"], size=10)
    cand._cost, cand._gguf = mc, g
    recommend.evaluate(cand, 20e9, 500e9, 1.0, 64 << 30, 16384)
    assert not cand.fits_at_all
    assert cand.tps == 0.0
    assert "context length" in cand.note


def test_recommend_end_to_end_on_cpu_only(monkeypatch, capsys, tmp_path):
    from wirl import compat
    _no_gpu(monkeypatch)
    g, mc = _toy_cost(_bigmoe_model(tmp_path))
    monkeypatch.setattr(compat, "list_gguf",
                        lambda repo: [{"path": "m-Q4_K_M.gguf", "size": 16e9}])

    def _fake_load(repo, cand, timeout_note=None):
        cand._cost, cand._gguf = mc, g
        cand.bpw = 4.9
        return cand

    monkeypatch.setattr(recommend, "load_remote", _fake_load)
    assert cli.main(["recommend", "someone/toy-GGUF",
                     "--mem-bandwidth", "20"]) == 0
    out = capsys.readouterr().out
    assert "Download" in out and "Q4_K_M" in out
    table = out.split("What you can run")[1].split("Recommendation")[0]
    assert "  --  " not in table
    assert re.search(r"\d+\.\d", table)


# --- F3 ---------------------------------------------------------------------


def test_plan_refuses_the_toy_on_cpu_only(monkeypatch, capsys, moe_model):
    _no_gpu(monkeypatch)
    cli.main(["plan", moe_model, "--mem-bandwidth", "20"])
    out = capsys.readouterr().out
    assert "none -- CPU only" in out
    assert "too small to price" in out
    assert "tok/s ceiling" not in out


def _bigmoe_model(tmp_path):
    """Same shape as the toy but ~40 MB per token: pricable, still header-only."""
    from tests.conftest import (T_ARRAY, T_FLOAT32, T_STRING, T_UINT32,
                                build_gguf)
    D = 512
    kv = {
        "general.architecture": (T_STRING, "toymoe"),
        "toymoe.block_count": (T_UINT32, 2),
        "toymoe.embedding_length": (T_UINT32, D),
        "toymoe.expert_count": (T_UINT32, 8),
        "toymoe.expert_used_count": (T_UINT32, 2),
        "toymoe.attention.head_count": (T_UINT32, 2),
        "toymoe.attention.head_count_kv": (T_UINT32, 1),
        "toymoe.attention.key_length": (T_UINT32, 4),
        "toymoe.attention.value_length": (T_UINT32, 4),
        "tokenizer.ggml.model": (T_STRING, "gpt2"),
        "tokenizer.ggml.tokens": (T_ARRAY, (T_STRING, ["a", "b", "c"])),
        "tokenizer.ggml.bos_token_id": (T_UINT32, 0),
        "tokenizer.ggml.eos_token_id": (T_UINT32, 1),
    }
    tensors = []
    for li in range(2):
        tensors += [(f"blk.{li}.ffn_up_exps.weight", (D, D, 8), T_FLOAT32),
                    (f"blk.{li}.ffn_down_exps.weight", (D, D, 8), T_FLOAT32),
                    (f"blk.{li}.attn_q.weight", (D, D), T_FLOAT32),
                    (f"blk.{li}.ffn_up_shexp.weight", (D, D), T_FLOAT32)]
    tensors.append(("token_embd.weight", (D, 3), T_FLOAT32))
    tensors.append(("output.weight", (D, 3), T_FLOAT32))
    return build_gguf(tmp_path / "bigmoe.gguf", kv, tensors)


def test_plan_prices_a_real_model_with_the_measured_derate(monkeypatch, capsys,
                                                           tmp_path):
    _no_gpu(monkeypatch)
    path = _bigmoe_model(tmp_path)
    cli.main(["plan", path, "--mem-bandwidth", "20"])
    out = capsys.readouterr().out
    assert "tok/s ceiling" in out
    assert "0.0 tok/s" not in out
    # prose matches the 0.70 constant, not the old "about half"
    assert "0.70" in out or "70%" in out
    assert "about half" not in out


# --- F4 ---------------------------------------------------------------------


def test_auto_points_cpu_only_at_plan_and_tune(monkeypatch, capsys, moe_model):
    _no_gpu(monkeypatch)
    monkeypatch.setattr(cli, "find_server", lambda explicit=None: "/bin/true")
    with pytest.raises(SystemExit) as e:
        cli.main(["auto", moe_model, "--mem-bandwidth", "20"])
    assert e.value.code != 0
    msg = str(e.value.code)
    assert "wirl plan" in msg
    assert "wirl tune" in msg
    assert "Scope" in msg


def _cpu_sweep(monkeypatch, means):
    from wirl import tune
    cfgs = []

    def _fake_sweep(base, binary, counts, reps=3, n_tokens=400, log_dir=None):
        from wirl.runner import RunResult
        out = []
        for t, m in zip(counts, means):
            import copy
            cfg = copy.copy(base)
            cfg.threads = t
            cfgs.append(t)
            out.append(RunResult(config=cfg, samples=[m, m * 1.001],
                                 prefill=[], peak_vram=0, startup_s=1.0,
                                 ok=True, error=None))
        return out

    monkeypatch.setattr(tune, "sweep_threads", _fake_sweep)
    return cfgs


def test_tune_cpu_only_reports_the_null_result(monkeypatch, capsys, tmp_path):
    _no_gpu(monkeypatch)
    monkeypatch.setattr(cli, "find_server", lambda explicit=None: "/bin/true")
    path = _bigmoe_model(tmp_path)
    counts = _cpu_sweep(monkeypatch, [10.0, 10.05, 9.98])
    assert cli.main(["tune", path, "--mem-bandwidth", "20"]) == 0
    out = capsys.readouterr().out
    assert "CPU-only" in out
    assert "Null result" in out
    assert "this GPU" not in out
    # The point count follows this box's core count (2 on small CI VMs,
    # 3 on a 16-core workstation); the sweep must run either way.
    assert len(counts) >= 2


def test_tune_cpu_only_measures_when_too_small_to_price(monkeypatch, capsys,
                                                        moe_model):
    _no_gpu(monkeypatch)
    monkeypatch.setattr(cli, "find_server", lambda explicit=None: "/bin/true")
    _cpu_sweep(monkeypatch, [10.0, 10.05, 9.98])
    assert cli.main(["tune", moe_model, "--mem-bandwidth", "20"]) == 0
    out = capsys.readouterr().out
    assert "too small to price" in out
    assert "Winner" in out
    assert "this GPU" not in out
