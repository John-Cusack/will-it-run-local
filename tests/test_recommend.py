"""Choosing a quantisation for a specific machine.

The failure mode being guarded against is a tool that gives everyone the same
answer, or that confidently reports a number it has no basis for.
"""
import pytest

from wirl import recommend
from wirl.recommend import Candidate, choose, group_shards, quality_score


def _c(name, bpw, tps, fits=True, size=None, share=0.9):
    return Candidate(name=name, files=[f"{name}.gguf"],
                     size=size or int(bpw * 1e9), bpw=bpw, tps=tps,
                     fits_at_all=fits, cpu_share=share)


def test_shards_are_grouped_into_one_candidate():
    files = [{"path": "m-00001-of-00002.gguf", "size": 40},
             {"path": "m-00002-of-00002.gguf", "size": 20},
             {"path": "solo-Q4_K_M.gguf", "size": 15}]
    g = group_shards(files)
    assert len(g) == 2
    sharded = [x for x in g if len(x.files) == 2][0]
    assert sharded.size == 60


def test_quant_name_is_extracted_from_filename():
    files = [{"path": "Qwen3-30B-A3B-Q4_K_M.gguf", "size": 1},
             {"path": "Qwen3-30B-UD-IQ2_XXS.gguf", "size": 1},
             {"path": "BF16/Qwen3-BF16-00001-of-00002.gguf", "size": 1}]
    names = {c.name for c in group_shards(files)}
    assert "Q4_K_M" in names
    assert "UD-IQ2_XXS" in names
    assert "BF16" in names


def test_does_not_recommend_bf16_over_a_good_quant():
    """Maximising bits-per-weight degenerates into always recommending BF16.

    Quality gains flatten around Q6_K while bytes-per-token keeps growing, and
    on a bandwidth-bound machine those bytes are paid directly in tokens/sec.
    """
    cands = [_c("Q4_K_M", 4.9, 60), _c("Q6_K", 6.6, 40),
             _c("Q8_0", 8.5, 25), _c("BF16", 16.0, 12)]
    best, _ = choose(cands, min_tps=5)
    assert best.name == "Q6_K"


def test_quality_saturates():
    assert quality_score(16.0) == quality_score(8.5) == pytest.approx(6.5)
    assert quality_score(4.9) == pytest.approx(4.9)


def test_speed_floor_is_respected():
    cands = [_c("Q4_K_M", 4.9, 20), _c("Q6_K", 6.6, 3)]
    best, why = choose(cands, min_tps=5)
    assert best.name == "Q4_K_M"
    assert "above 5" in why


def test_falls_back_to_fastest_when_nothing_meets_the_floor():
    cands = [_c("Q4_K_M", 4.9, 2.0), _c("Q6_K", 6.6, 1.0)]
    best, why = choose(cands, min_tps=5)
    assert best.name == "Q4_K_M"
    assert "fastest" in why


def test_returns_nothing_when_nothing_fits():
    best, why = choose([_c("BF16", 16.0, 0, fits=False)], min_tps=5)
    assert best is None and why is None


def test_models_that_do_not_fit_are_excluded_not_ranked():
    cands = [_c("Q4_K_M", 4.9, 30), _c("Q8_0", 8.5, 0, fits=False)]
    best, _ = choose(cands, min_tps=5)
    assert best.name == "Q4_K_M"


@pytest.mark.parametrize("share,expected", [
    (0.95, "calibrated"),   # the regime measured on the reference machine
    (0.80, "calibrated"),
    (0.60, "mixed"),        # GPU term matters and is not calibrated
    (0.10, "ceiling"),      # GPU-bound: report no figure at all
])
def test_confidence_tracks_how_much_is_read_from_ram(share, expected, monkeypatch):
    """The tool must not present a GPU-bound roofline as a prediction.

    Calibration came from runs at ~92% CPU share, where the GPU term is a
    rounding error. At low CPU share the roofline is several times optimistic.
    """
    class FakeBest:
        n_cpu_moe = 4
        tps = 100.0
        vram_bytes = 1
        cpu_ms = share * 100
        gpu_ms = (1 - share) * 100

    class FakeCost:
        is_moe = True
        n_layer = 8
        total_bytes = 10

        def resident_vram_weights(self, n):
            return 5

    cand = Candidate(name="x", files=[], size=10)
    cand._cost = FakeCost()
    cand._gguf = object()
    monkeypatch.setattr(recommend.predict, "best_fit",
                        lambda *a, **k: (FakeBest(), []))
    monkeypatch.setattr(recommend.model, "kv_cache_bytes", lambda *a, **k: 0)
    recommend.evaluate(cand, 45e9, 936e9, 24e9, 200e9, 8192)
    assert cand.confidence == expected


def _cpu_candidate(moe_model):
    from wirl import gguf, model
    cand = Candidate(name="Q4_K_M", files=["model.gguf"], size=1)
    cand._gguf = gguf.read(moe_model)
    cand._cost = model.build(cand._gguf)
    return cand


def test_cpu_only_recommendation_includes_kv(moe_model):
    from wirl import model, predict
    cand = _cpu_candidate(moe_model)
    need = cand._cost.total_bytes + model.kv_cache_bytes(cand._gguf, 8192)
    recommend.evaluate(cand, 20e9, 0, 0, need, 8192)
    assert cand.fits_at_all
    assert cand.tps == predict.cpu_only_tps(cand._cost, 20e9)
    assert cand.knob == "--n-gpu-layers" and cand.knob_value == 0
    assert cand.ram_needed == need
    assert cand.vram == 0 and cand.cpu_share == 1.0
    assert cand.confidence == "calibrated"


def test_cpu_only_recommendation_refuses_swap(moe_model):
    from wirl import model
    cand = _cpu_candidate(moe_model)
    need = cand._cost.total_bytes + model.kv_cache_bytes(cand._gguf, 8192)
    recommend.evaluate(cand, 20e9, 0, 0, need - 1, 8192)
    assert not cand.fits_at_all
    assert "swap" in cand.note


def test_recommend_cli_without_gpu(monkeypatch, capsys, moe_model):
    from wirl import cli, compat, probe
    from test_search import _no_gpu
    _no_gpu(monkeypatch)
    monkeypatch.setattr(probe, "mem_info", lambda: {"available": 64 << 30})
    monkeypatch.setattr(probe, "cpu_info", lambda: {"physical": 16})
    monkeypatch.setattr(compat, "list_gguf", lambda repo: [{"path": "Q4_K_M.gguf", "size": 1}])

    def load(repo, cand):
        source = _cpu_candidate(moe_model)
        cand._gguf, cand._cost = source._gguf, source._cost

    monkeypatch.setattr(recommend, "load_remote", load)
    assert cli.main(["recommend", "some/repo", "--mem-bandwidth", "20"]) == 0
    out = capsys.readouterr().out
    assert "Download" in out and "--n-gpu-layers 0" in out
