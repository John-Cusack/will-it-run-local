"""Answer "what can *I* run?" instead of "how do I run this file I already have".

A GGUF stores its metadata and its complete tensor table at the head of the
file, so a few MB of HTTP Range requests is enough to price a model exactly --
including which of its tensors are routed experts -- without downloading it.
Reading an 18 GB model this way takes about two seconds.

That makes it possible to compare every quantisation in a repository against
the machine actually in front of you, and to say which one to download, before
spending an hour and 150 GB finding out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import compat, gguf, model, predict

SHARD_RE = re.compile(r"^(?P<stem>.*)-(?P<idx>\d{5})-of-(?P<tot>\d{5})\.gguf$")

# Bits per weight is the honest quality proxy within a model: more bits means
# less quantisation error. But the returns flatten hard. Around Q6_K the output
# is already very close to the unquantised model, and Q8_0 and BF16 buy almost
# nothing for inference while doubling the bytes read per token -- which on a
# bandwidth-bound machine is paid back directly in tokens per second.
#
# So quality is scored as bpw capped at this point, and ties are broken on
# speed. Without the cap, "best quality that is fast enough" always degenerates
# into "recommend BF16", which is the wrong answer for essentially everyone.
QUALITY_SATURATION_BPW = 6.5


def _bpw(mc_total_bytes, n_params):
    return (mc_total_bytes * 8.0 / n_params) if n_params else 0.0


def quality_score(bpw: float) -> float:
    return min(bpw, QUALITY_SATURATION_BPW)


@dataclass
class Candidate:
    name: str            # display name, e.g. "Q4_K_M"
    files: list          # repo-relative paths (>1 for a sharded model)
    size: int            # on-disk bytes, summed across shards
    arch: str = ""
    n_layer: int = 0
    is_moe: bool = False
    bytes_per_token: int = 0
    n_params: int = 0
    bpw: float = 0.0
    # filled in by evaluate()
    fits_vram: bool = False
    fits_at_all: bool = False
    knob: str = ""
    knob_value: int = 0
    tps: float = 0.0
    vram: int = 0
    ram_needed: int = 0
    note: str = ""
    # How far to trust the number, by how much of the token is spent on the CPU:
    #   calibrated  >=80%  the regime the model was measured in (92% on the
    #                      reference machine), where the GPU term is a rounding
    #                      error and cannot hide a mistake
    #   mixed      30-80%  both sides matter; the GPU term is an uncalibrated
    #                      roofline, so the total is optimistic by an unknown
    #                      amount
    #   ceiling     <30%   GPU-bound. Nothing here was measured in that regime
    #                      and the roofline is several times too optimistic, so
    #                      no figure is reported at all.
    confidence: str = "calibrated"
    cpu_share: float = 0.0


def group_shards(files) -> list:
    """Collapse `-00001-of-00005` sets into one candidate."""
    groups = {}
    for f in files:
        base = f["path"].rsplit("/", 1)[-1]
        m = SHARD_RE.match(base)
        prefix = f["path"][:len(f["path"]) - len(base)]
        key = prefix + (m.group("stem") if m else base)
        g = groups.setdefault(key, {"paths": [], "size": 0})
        g["paths"].append(f["path"])
        g["size"] += f["size"]
    out = []
    for key, g in groups.items():
        out.append(Candidate(name=_short_name(key), files=sorted(g["paths"]),
                             size=g["size"]))
    return sorted(out, key=lambda c: c.size)


def _short_name(key: str) -> str:
    """Pull the quantisation label out of a filename, e.g. '...-Q4_K_M.gguf'."""
    base = key.rsplit("/", 1)[-1].replace(".gguf", "")
    m = re.search(r"((?:UD-)?(?:IQ|Q)\d+[A-Z0-9_]*|BF16|F16|F32|MXFP4|bf16|f16)$", base)
    if m:
        return m.group(1)
    # Some repos put the quant in a directory instead of the filename.
    parts = key.split("/")
    if len(parts) > 1:
        return parts[-2]
    return base


def load_remote(repo: str, cand: Candidate, timeout_note=None) -> Candidate:
    """Read headers for one candidate and fill in its cost figures."""
    tensors = []
    first = None
    for path in cand.files:
        g = gguf.read_one(compat.resolve_url(repo, path), want_tensors=True)
        if first is None:
            first = g
        tensors.extend(g.tensors)
    first.tensors = tensors
    mc = model.build(first)
    cand.arch = first.arch
    cand.n_layer = mc.n_layer
    cand.is_moe = mc.is_moe
    cand.bytes_per_token = mc.bytes_per_token
    cand.n_params = sum(t.n_elements for t in tensors)
    cand.bpw = _bpw(mc.total_bytes, cand.n_params)
    cand._gguf = first
    cand._cost = mc
    return cand


def evaluate(cand: Candidate, bw_cpu, bw_gpu, vram_budget, ram_available, ctx,
             headroom=3.0e9) -> Candidate:
    """Find this candidate's best configuration on this machine."""
    mc, g = cand._cost, cand._gguf
    kv = model.kv_cache_bytes(g, ctx)

    if vram_budget <= 0:
        cand.knob, cand.knob_value = "--n-gpu-layers", 0
        cand.tps = predict.cpu_only_tps(mc, bw_cpu)
        cand.vram, cand.cpu_share = 0, 1.0
        cand.confidence = "calibrated"
        cand.ram_needed = mc.total_bytes + kv
        cand.fits_at_all = cand.ram_needed <= ram_available
        if not cand.fits_at_all:
            cand.note = "would swap: not enough system RAM"
        return cand

    # Can it be held at all, across VRAM and RAM together?
    cand.fits_at_all = mc.total_bytes + kv < (vram_budget + ram_available)
    if not cand.fits_at_all:
        cand.note = "larger than VRAM + RAM combined"
        return cand

    if mc.is_moe:
        best, _ = predict.best_fit(mc, g, bw_cpu, bw_gpu, vram_budget, ctx,
                                   headroom=headroom)
        cand.knob = "--n-cpu-moe"
    else:
        best, _ = predict.best_fit_dense(mc, g, bw_cpu, bw_gpu, vram_budget, ctx,
                                         headroom=min(headroom, 1.0e9))
        cand.knob = "--n-gpu-layers"
    if best is None:
        cand.note = "does not fit at this context length"
        return cand

    cand.knob_value = best.n_cpu_moe
    cand.tps = best.tps
    cand.vram = best.vram_bytes
    total_ms = best.cpu_ms + best.gpu_ms
    share = (best.cpu_ms / total_ms) if total_ms else 0.0
    cand.cpu_share = share
    cand.confidence = ("calibrated" if share >= 0.80
                       else "mixed" if share >= 0.30
                       else "ceiling")
    cand.ram_needed = max(0, mc.total_bytes - mc.resident_vram_weights(best.n_cpu_moe)
                          if mc.is_moe else mc.total_bytes - mc.resident_vram_dense(best.n_cpu_moe))

    if cand.ram_needed > ram_available:
        cand.note = "would swap: not enough system RAM"
        cand.fits_at_all = False
        return cand

    # Entirely on the GPU is worth calling out: it is a different speed class.
    if mc.is_moe:
        cand.fits_vram = best.n_cpu_moe == 0
    else:
        cand.fits_vram = best.n_cpu_moe >= mc.n_layer
    return cand


def choose(cands, min_tps=5.0):
    """Recommend the best-quality option that still runs fast enough.

    Quality rises with bits per weight, so the right choice is the largest
    quantisation that still clears the user's speed floor -- not the fastest
    option, which is always the most damaged one.
    """
    usable = [c for c in cands if c.fits_at_all and c.tps >= min_tps]
    if not usable:
        runnable = [c for c in cands if c.fits_at_all and c.tps > 0]
        if not runnable:
            return None, None
        return max(runnable, key=lambda c: c.tps), "fastest that runs at all"
    # Highest quality, then fastest among equals -- so once quality saturates,
    # the smaller and faster file wins.
    best = max(usable, key=lambda c: (quality_score(c.bpw), c.tps))
    return best, f"best quality above {min_tps:g} tok/s"
