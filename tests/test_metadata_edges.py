"""GGUF, cost and recommendation edge cases with tiny synthetic headers."""
import io
import struct

import pytest

from wirl import ggml_types, gguf, model, predict, recommend

pytestmark = pytest.mark.usefixtures("isolated_runtime")


def test_remote_reader_refetches_reuses_buffer_and_reports_eof(monkeypatch):
    data = b"abcdefghij"
    requests = []
    reader = gguf._Reader("https://example.invalid/file", chunk=4)
    monkeypatch.setattr(reader, "_fetch", lambda start, length:
                        requests.append((start, length)) or data[start:start+length])
    assert reader.read(2) == b"ab"
    assert reader.read(2) == b"cd"
    assert requests == [(0, 4)]
    assert reader.read(5) == b"efghi"
    with pytest.raises(EOFError, match="wanted 2, got 1 at 9"):
        reader.read(2)
    assert reader.pos == 9
    reader.close()


@pytest.mark.parametrize("type_id,fmt,value", [
    (0, "<B", 255), (1, "<b", -1), (2, "<H", 65535), (3, "<h", -2),
    (4, "<I", 42), (5, "<i", -3), (6, "<f", 1.25), (7, "<?", True),
    (10, "<Q", 1<<40), (11, "<q", -(1<<40)), (12, "<d", 2.5),
])
def test_metadata_scalar_types(type_id, fmt, value):
    assert gguf._read_value(io.BytesIO(struct.pack(fmt, value)), type_id) == value


def test_empty_arrays_and_replacement_text():
    assert gguf._read_value(io.BytesIO(struct.pack("<IQ", 8, 0)), 9) == []
    assert gguf._read_value(io.BytesIO(struct.pack("<Q", 1)+b"\xff"), 8) == "\ufffd"


def test_tensor_properties_quant_mix_and_header_only(moe_model):
    g = gguf.read(moe_model)
    tensor = g.tensors[0]
    assert tensor.n_elements == 128 and tensor.type == "f32"
    assert g.quant_mix() == {"f32": g.total_bytes}
    header = gguf.read_header_only(moe_model)
    assert header.n_embd == 4 and header.tensors == []
    assert header.vocab_sig() == g.vocab_sig()


def test_multishard_read_combines_all_tensor_tables(tmp_path, build_gguf):
    paths = []
    for i in (1, 2):
        paths.append(build_gguf(tmp_path / f"m-{i:05d}-of-00002.gguf",
                                {"general.architecture": (8, "dense")}, [(f"output{i}", (4,), 0)]))
    g = gguf.read(paths[0])
    assert [t.name for t in g.tensors] == ["output1", "output2"]
    assert g.kv["_wirl.shards"] == 2 and g.path == paths[0]


def test_read_remote_uses_one_url_and_no_shard_files(monkeypatch, moe_model):
    data = open(moe_model, "rb").read()
    monkeypatch.setattr(gguf._Reader, "_fetch", lambda self, start, length: data[start:start+length])
    g = gguf.read("https://example.invalid/m-00001-of-00002.gguf")
    assert len(g.tensors) == 10 and "_wirl.shards" not in g.kv


def test_unknown_reserved_and_padded_tensor_sizes(monkeypatch):
    assert ggml_types.type_name(999) == "unknown(999)"
    assert ggml_types.bits_per_weight(999) == 0
    assert ggml_types.tensor_nbytes(2, (33, 2)) == 72
    monkeypatch.setitem(ggml_types.GGML_TYPES, 999, ("reserved", 0, 0))
    with pytest.raises(ValueError, match="has no size"):
        ggml_types.tensor_nbytes(999, (32,))


def test_missing_layers_are_priced_as_non_layer_weights():
    g = gguf.GGUF("tiny", 3, {}, [gguf.Tensor("blk.5.weight", (4,), 0)])
    mc = model.build(g)
    assert mc.non_layer_bytes == mc.bytes_per_token == 16
    assert mc.kv_fraction_on_gpu(1) == 0
    assert not g.is_moe and g.n_expert_used == 0
    assert g.vocab_sig()["n_vocab"] == 0


@pytest.mark.parametrize("heads,kv_heads,expected", [([2, 4], [1, 2], 32), ([0], [1], 0), ([], [], 0)])
def test_per_layer_attention_head_lists(heads, kv_heads, expected):
    g = gguf.GGUF("hybrid", 3, {"general.architecture": "hybrid", "hybrid.block_count": 1,
                 "hybrid.embedding_length": 16, "hybrid.attention.head_count": heads,
                 "hybrid.attention.head_count_kv": kv_heads}, [])
    assert model.kv_cache_bytes(g, 1) == expected


def test_mla_compressed_cache_uses_latent_and_rope():
    g = gguf.GGUF("mla", 3, {"general.architecture": "mla", "mla.block_count": 2,
                 "mla.attention.kv_lora_rank": 8, "mla.rope.dimension_count": 4,
                 "mla.attention.compress_ratios": [1, 4]}, [])
    assert model.kv_cache_bytes(g, 10) == 2*10*12*2*.625


def test_zero_bandwidth_and_unfittable_dense_model(moe_model):
    g = gguf.read(moe_model)
    mc = model.build(g)
    assert predict.predict_tps(mc, 1, 0, 0) == (0, 0, 0)
    assert predict.cpu_only_tps(mc, 0) == 0
    assert predict.best_fit(mc, g, 1, 1, 0, 1)[0] is None
    points = predict.dense_curve(mc, g, 0, 0, 0, 1)
    assert all(p.tps == 0 for p in points)
    assert predict.best_fit_dense(mc, g, 1, 1, 0, 1)[0] is None
    budget = predict.vram_needed_dense(mc, g, 0, 1)
    assert predict.best_fit_dense(mc, g, 1, 1, budget, 1, headroom=1e20)[0] is not None


def test_verdict_without_timing_and_with_intermediate_ratio():
    best = predict.Prediction(1, 1, 1, 1, True, 10, 1, 1)
    assert predict.verdict(None, best, 1) == []
    assert predict.verdict(None, best, 1, measured_tps=7) == []
    best.tps = best.cpu_ms = best.gpu_ms = 0
    assert "GPU-bound" in predict.verdict(None, best, 1, measured_tps=1)[0]


def test_remote_candidate_loads_every_shard(monkeypatch, moe_model):
    original = gguf.read(moe_model)
    calls = []

    def read(url, want_tensors):
        calls.append((url, want_tensors))
        return gguf.GGUF(url, 3, original.kv, [original.tensors[len(calls)-1]])

    monkeypatch.setattr(gguf, "read_one", read)
    cand = recommend.Candidate("test", ["a.gguf", "b.gguf"], 1024)
    assert recommend.load_remote("owner/repo", cand) is cand
    assert cand.n_params == 256 and cand.bpw == 32 and cand.is_moe
    assert [t.name for t in cand._gguf.tensors] == [t.name for t in original.tensors[:2]]
    assert calls == [("https://huggingface.co/owner/repo/resolve/main/a.gguf", True),
                     ("https://huggingface.co/owner/repo/resolve/main/b.gguf", True)]
    assert recommend._bpw(0, 0) == 0
    assert recommend._short_name("quant/model.gguf") == "quant"


@pytest.mark.parametrize("dense,budget,ram,best,expected", [
    (False, 10, 1, False, "larger than"), (False, 100, 100, False, "does not fit"),
    (False, 100, 1, True, "would swap"), (True, 100, 100, True, ""),
])
def test_candidate_fit_failure_and_dense_dispatch(monkeypatch, dense, budget, ram, best, expected):
    g = gguf.GGUF("tiny", 3, {}, [])
    mc = model.ModelCost(1, 0 if dense else 2, 1, [model.LayerCost(0, 16, 8, 0)], 0, 0, 16)
    cand = recommend.Candidate("test", ["tiny"], 16)
    cand._gguf, cand._cost = g, mc
    prediction = predict.Prediction(1, 1, 1, 10, True, 1, 0, 0) if best else None
    for name in ("best_fit", "best_fit_dense"):
        monkeypatch.setattr(predict, name, lambda *a, **kw: (prediction, []))
    recommend.evaluate(cand, 1, 1, budget, ram, 1)
    assert expected in cand.note
    if dense:
        assert cand.knob == "--n-gpu-layers" and cand.fits_vram
