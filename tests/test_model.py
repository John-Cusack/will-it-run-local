"""Cost model arithmetic, checked against numbers you can verify by hand."""
import pytest

from wirl import gguf, model


def test_expert_tensors_are_discounted_by_routing(moe_model):
    """Only n_expert_used of n_expert experts are read per token."""
    g = gguf.read(moe_model)
    mc = model.build(g)
    layer = mc.layers[0]
    # two expert tensors, each 4*4*8 f32 = 512 bytes
    assert layer.expert_bytes == 2 * 512
    # 2 of 8 experts fire -> a quarter is read
    assert layer.expert_bytes_per_tok == 2 * 128


def test_shared_expert_is_not_discounted(moe_model):
    """`_shexp` fires on every token and must not be treated as routed."""
    g = gguf.read(moe_model)
    mc = model.build(g)
    # attn_q (64 B) + ffn_up_shexp (64 B) are both always-read
    assert mc.layers[0].other_bytes == 128


def test_bytes_per_token_is_less_than_model_size(moe_model):
    g = gguf.read(moe_model)
    mc = model.build(g)
    assert mc.bytes_per_token < mc.total_bytes
    per_layer = 2 * 128 + 128
    # token_embd is indexed rather than streamed, so only `output` counts
    assert mc.bytes_per_token == 2 * per_layer + 48


def test_split_moves_bytes_between_pools(moe_model):
    g = gguf.read(moe_model)
    mc = model.build(g)
    all_cpu = mc.split(2)
    all_gpu = mc.split(0)
    assert all_cpu[0] > 0 and all_gpu[0] == 0
    # nothing is lost or duplicated by the split
    assert sum(all_cpu) == sum(all_gpu) == mc.bytes_per_token


def test_resident_vram_grows_as_layers_move_to_gpu(moe_model):
    g = gguf.read(moe_model)
    mc = model.build(g)
    assert mc.resident_vram_weights(2) < mc.resident_vram_weights(1)
    assert mc.resident_vram_weights(0) == mc.total_bytes


def test_kv_cache_scales_with_context(moe_model):
    g = gguf.read(moe_model)
    a = model.kv_cache_bytes(g, 1024)
    b = model.kv_cache_bytes(g, 2048)
    assert b == 2 * a
    # 2 layers * 1024 ctx * 1 kv head * (4*2 + 4*2) bytes
    assert a == 2 * 1024 * 1 * 16


def test_kv_cache_quantisation_shrinks_it(moe_model):
    g = gguf.read(moe_model)
    f16 = model.kv_cache_bytes(g, 1024, "f16", "f16")
    q8 = model.kv_cache_bytes(g, 1024, "f16", "q8_0")
    assert q8 < f16


def test_dense_model_reads_everything(tmp_path):
    """With no experts, bytes/token is essentially the whole model."""
    from tests.conftest import build_gguf
    kv = {"general.architecture": (8, "dense"),
          "dense.block_count": (4, 1),
          "dense.embedding_length": (4, 8)}
    t = [("blk.0.attn_q.weight", (8, 8), 0), ("output.weight", (8, 4), 0)]
    g = gguf.read(build_gguf(tmp_path / "d.gguf", kv, t))
    mc = model.build(g)
    assert not mc.is_moe
    assert mc.bytes_per_token == mc.total_bytes


def test_dense_offload_moves_whole_layers(tmp_path):
    """For a dense model the knob is --n-gpu-layers, and it moves everything."""
    from tests.conftest import build_gguf
    kv = {"general.architecture": (8, "dense"), "dense.block_count": (4, 4),
          "dense.embedding_length": (4, 8)}
    t = [(f"blk.{i}.attn_q.weight", (8, 8), 0) for i in range(4)]
    t.append(("output.weight", (8, 4), 0))
    g = gguf.read(build_gguf(tmp_path / "d.gguf", kv, t))
    mc = model.build(g)

    none_offloaded = mc.split_dense(0)
    all_offloaded = mc.split_dense(4)
    assert none_offloaded[1] == 0 or none_offloaded[0] > all_offloaded[0]
    assert all_offloaded[0] == 0
    # nothing is created or lost by the split
    for n in range(5):
        assert sum(mc.split_dense(n)) == mc.bytes_per_token


def test_dense_vram_grows_with_offloaded_layers(tmp_path):
    from tests.conftest import build_gguf
    kv = {"general.architecture": (8, "dense"), "dense.block_count": (4, 4),
          "dense.embedding_length": (4, 8)}
    t = [(f"blk.{i}.attn_q.weight", (8, 8), 0) for i in range(4)]
    g = gguf.read(build_gguf(tmp_path / "d.gguf", kv, t))
    mc = model.build(g)
    vram = [mc.resident_vram_dense(n) for n in range(5)]
    assert vram == sorted(vram)
    assert vram[0] == 0
    assert mc.kv_fraction_on_gpu(2) == 0.5
