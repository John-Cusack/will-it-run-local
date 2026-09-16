"""RAM pre-flight requirements include the KV cache and GPU allocation costs."""
import pytest

from wirl import cli, doctor, gguf, model, predict, probe


@pytest.fixture
def dense_cost():
    g = gguf.GGUF("synthetic.gguf", 3, {
        "general.architecture": "dense", "dense.block_count": 4,
        "dense.embedding_length": 8, "dense.attention.head_count": 2}, [])
    layers = [model.LayerCost(i, 0, 0, 512 << 20) for i in range(4)]
    return g, model.ModelCost(4, 0, 0, layers, 0, 0, 2 << 30)


def test_moe_min_ram_tracks_offload(moe_model):
    g = gguf.read(moe_model)
    mc = model.build(g)
    kv = model.kv_cache_bytes(g, 8192)
    budgets = [0, 1] + [predict.vram_needed(mc, g, n, 8192) for n in range(mc.n_layer, -1, -1)]
    needs = [predict.min_ram_needed(mc, g, b, 8192) for b in budgets]
    assert needs[0] == needs[1] == mc.total_bytes + kv
    assert needs == sorted(needs, reverse=True)
    assert needs[-2] == mc.layers[0].expert_bytes
    assert needs[-1] == 0


def test_dense_min_ram_includes_cpu_kv(dense_cost):
    g, mc = dense_cost
    kv = model.kv_cache_bytes(g, 8192)
    budgets = [0, 1] + [predict.vram_needed_dense(mc, g, n, 8192) for n in range(5)]
    needs = [predict.min_ram_needed(mc, g, b, 8192) for b in budgets]
    assert needs[0] == needs[1] == mc.total_bytes + kv
    assert needs == sorted(needs, reverse=True)
    assert needs[4] == mc.total_bytes - mc.resident_vram_dense(2) + kv / 2
    assert needs[-1] == 0


@pytest.mark.parametrize("dense", [False, True])
def test_draft_reserves_vram_and_fallback_ram(moe_model, dense_cost, dense):
    dg = gguf.read(moe_model)
    dmc = model.build(dg)
    g, mc = dense_cost if dense else (dg, dmc)
    boundary = (predict.vram_needed_dense(mc, g, 2, 8192) if dense
                else predict.vram_needed(mc, g, 1, 8192))
    without = predict.min_ram_needed(mc, g, boundary, 8192)
    with_draft = predict.min_ram_needed(mc, g, boundary, 8192, draft_mc=dmc, draft_g=dg)
    assert with_draft > without
    fallback = predict.min_ram_needed(mc, g, 0, 8192, draft_mc=dmc, draft_g=dg)
    assert fallback == mc.total_bytes + model.kv_cache_bytes(g, 8192) + dmc.total_bytes + model.kv_cache_bytes(dg, 8192)


@pytest.mark.parametrize("command", ["auto", "doctor"])
def test_cli_uses_offloaded_ram_requirement(monkeypatch, dense_cost, command):
    g, mc = dense_cost
    monkeypatch.setattr(cli, "_load", lambda *a: (g, mc))
    monkeypatch.setattr(cli, "find_server", lambda *a: "mock-server")
    monkeypatch.setattr(probe, "cpu_info", lambda: {"model": "test", "physical": 16})
    monkeypatch.setattr(probe, "gpu_info", lambda: [
        {"index": 0, "uuid": "GPU-zero", "name": "test", "vram_total": 24 << 30}])
    seen = []

    class StopPreflight(Exception):
        pass

    def checks(*a, **kw):
        seen.append((a, kw))
        raise StopPreflight

    monkeypatch.setattr(doctor, "run_all", checks)
    args = (["auto", "synthetic.gguf", "--mem-bandwidth", "20"] if command == "auto"
            else ["doctor", "--model", "synthetic.gguf"])
    args += ["--vram", "1.5", "--ctx", "8192"]
    with pytest.raises(StopPreflight):
        cli.main(args)
    need = seen[0][1]["ram_need"]
    assert need == predict.min_ram_needed(mc, g, int(1.5 * (1 << 30)), 8192)
    assert need < mc.total_bytes
    assert doctor.check_ram_for_model({"available": int(1.2 * (1 << 30))}, need).status == doctor.OK
    assert doctor.check_ram_for_model({"available": 512 << 20}, need).status == doctor.FAIL
