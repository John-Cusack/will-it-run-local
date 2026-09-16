"""Turn a GGUF tensor table into a cost model.

Two numbers drive every recommendation this tool makes:

  bytes_per_token   how much weight data a single decode step must read
  where those bytes live   VRAM at ~900 GB/s, or system RAM at ~50 GB/s

Autoregressive decode of one token touches every weight it needs exactly once,
so decode speed on a large model is a memory-bandwidth problem, not a compute
problem. For a dense model bytes_per_token is simply the model size. For a
mixture-of-experts model only n_expert_used of n_expert routed experts fire per
layer per token, so the routed tensors are discounted by that ratio -- which is
why a 156 GB MoE can decode faster than a 70 GB dense model.

Everything here is derived from tensor names and shapes. No per-architecture
special cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Routed-expert tensors carry an "_exps" suffix in llama.cpp's naming scheme and
# a trailing dimension equal to the expert count. Shared experts ("_shexp") fire
# on every token and are deliberately NOT in this pattern.
EXPERT_RE = re.compile(r"_exps(\.|$)")
BLK_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass
class LayerCost:
    idx: int
    expert_bytes: int        # all experts resident, i.e. what offloading moves
    expert_bytes_per_tok: int  # only the routed fraction actually read
    other_bytes: int         # attention, shared expert, norms: always read


@dataclass
class ModelCost:
    n_layer: int
    n_expert: int
    n_expert_used: int
    layers: list
    non_layer_bytes: int         # embeddings, output head, final norm
    non_layer_bytes_per_tok: int
    total_bytes: int

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 1

    @property
    def bytes_per_token(self) -> int:
        """Weight bytes read for one decoded token, wherever they live."""
        return (self.non_layer_bytes_per_tok
                + sum(l.expert_bytes_per_tok + l.other_bytes for l in self.layers))

    def split(self, n_cpu_moe: int):
        """Bytes read per token from each pool for a given --n-cpu-moe value.

        llama.cpp's --n-cpu-moe N keeps the routed experts of the first N layers
        in system RAM and puts everything else on the GPU. N therefore trades
        VRAM against DRAM traffic, one layer at a time, and is the single most
        important tuning knob on a machine that cannot hold the whole model.
        """
        n = max(0, min(n_cpu_moe, self.n_layer))
        cpu = sum(l.expert_bytes_per_tok for l in self.layers[:n])
        gpu = (self.non_layer_bytes_per_tok
               + sum(l.other_bytes for l in self.layers)
               + sum(l.expert_bytes_per_tok for l in self.layers[n:]))
        return cpu, gpu

    # --- dense models -----------------------------------------------------
    #
    # A dense model has no routed experts, so --n-cpu-moe does nothing. The
    # knob is --n-gpu-layers: whole layers move to the GPU, weights and KV
    # cache together.

    def split_dense(self, n_gpu_layers: int):
        """Bytes read per token from each pool for a given --n-gpu-layers."""
        n = max(0, min(n_gpu_layers, self.n_layer))
        on_gpu = self.layers[self.n_layer - n:] if n else []
        gpu = sum(l.expert_bytes_per_tok + l.other_bytes for l in on_gpu)
        cpu = self.bytes_per_token - gpu
        if n >= self.n_layer:
            # The output head is offloaded along with the last layer.
            return 0, self.bytes_per_token
        return cpu - self.non_layer_bytes_per_tok, gpu + self.non_layer_bytes_per_tok

    def resident_vram_dense(self, n_gpu_layers: int) -> int:
        n = max(0, min(n_gpu_layers, self.n_layer))
        on_gpu = self.layers[self.n_layer - n:] if n else []
        total = sum(l.expert_bytes + l.other_bytes for l in on_gpu)
        if n >= self.n_layer:
            total += self.non_layer_bytes
        return total

    def kv_fraction_on_gpu(self, n_gpu_layers: int) -> float:
        if not self.n_layer:
            return 0.0
        return max(0, min(n_gpu_layers, self.n_layer)) / self.n_layer

    def resident_vram_weights(self, n_cpu_moe: int) -> int:
        """Weight bytes the GPU must actually hold (not per token)."""
        n = max(0, min(n_cpu_moe, self.n_layer))
        return (self.non_layer_bytes
                + sum(l.other_bytes for l in self.layers)
                + sum(l.expert_bytes for l in self.layers[n:]))


def build(g) -> ModelCost:
    n_layer = g.n_layer or 0
    n_expert = g.n_expert or 0
    n_used = g.n_expert_used or 0
    frac = (n_used / n_expert) if n_expert else 0.0

    per_layer = {i: {"expert": 0, "expert_tok": 0, "other": 0} for i in range(n_layer)}
    non_layer = 0
    non_layer_tok = 0

    for t in g.tensors:
        nb = t.nbytes
        m = BLK_RE.match(t.name)
        if not m:
            non_layer += nb
            # The output head and embeddings are read once per token. The
            # embedding table is indexed, not streamed, so only a row is
            # actually touched -- but it is small relative to the head, and
            # llama.cpp reads the head in full to produce logits.
            non_layer_tok += nb if "token_embd" not in t.name else 0
            continue
        li = int(m.group(1))
        if li >= n_layer:
            non_layer += nb
            non_layer_tok += nb
            continue
        if EXPERT_RE.search(t.name) and n_expert:
            per_layer[li]["expert"] += nb
            per_layer[li]["expert_tok"] += int(nb * frac)
        else:
            per_layer[li]["other"] += nb

    layers = [LayerCost(i, per_layer[i]["expert"], per_layer[i]["expert_tok"],
                        per_layer[i]["other"]) for i in range(n_layer)]
    return ModelCost(n_layer, n_expert, n_used, layers,
                     non_layer, non_layer_tok, g.total_bytes)


# --- KV cache -------------------------------------------------------------

def kv_cache_bytes(g, ctx: int, k_type: str = "f16", v_type: str = "f16") -> int:
    """Size of the KV cache at a given context length.

    Handles both standard GQA and the compressed latent cache (MLA) used by
    DeepSeek-style models, where what is cached is the low-rank KV projection
    plus a small RoPE component rather than full K and V.
    """
    bpe = {"f16": 2, "bf16": 2, "f32": 4, "q8_0": 34 / 32, "q5_1": 24 / 32,
           "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32}
    kb = bpe.get(k_type, 2)
    vb = bpe.get(v_type, 2)
    n_layer = g.n_layer or 0

    # Some architectures compress the KV cache per layer. DeepSeek-V4-Flash
    # publishes attention.compress_ratios: one entry per layer, where N means
    # that layer keeps roughly 1/N of the tokens (0 = keep everything). On the
    # reference model 20 of 44 layers keep 1/128 and 21 keep 1/4, so the real
    # cache is ~19% of what the naive per-layer arithmetic predicts -- about
    # 16 KiB/token instead of 86. Ignoring this over-estimates KV by 5x and
    # makes long contexts look unaffordable when they are nearly free.
    ratios = g.a("attention.compress_ratios")
    layer_scale = 1.0
    if isinstance(ratios, list) and ratios:
        eff = sum(1.0 if r in (0, 1) else 1.0 / r for r in ratios)
        layer_scale = eff / len(ratios)

    lora = g.a("attention.kv_lora_rank")
    if lora:
        rope = g.a("rope.dimension_count") or g.a("attention.key_length_mla") or 64
        # MLA caches one compressed latent vector per token per layer.
        return int(n_layer * ctx * (lora + rope) * kb * layer_scale)

    n_head_kv = g.a("attention.head_count_kv") or g.a("attention.head_count") or 0
    if isinstance(n_head_kv, list):
        n_head_kv = max(n_head_kv)
    n_head = g.a("attention.head_count") or 1
    if isinstance(n_head, list):
        n_head = max(n_head)
    n_embd = g.n_embd or 0
    k_len = g.a("attention.key_length") or (n_embd // n_head if n_head else 0)
    v_len = g.a("attention.value_length") or k_len
    return int(n_layer * ctx * n_head_kv * (k_len * kb + v_len * vb) * layer_scale)
