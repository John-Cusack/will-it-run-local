"""Roofline prediction and configuration fitting.

The decode model is deliberately simple, because the physics is simple:

    seconds_per_token = cpu_bytes / cpu_bandwidth + gpu_bytes / gpu_bandwidth

Every weight needed for a token is read exactly once, from whichever pool it
lives in. Compute barely matters -- see the thread sweep in README.md, where
going from 24 to 64 threads changed throughput by 0.9%.

Calibrated against measured runs on an EPYC 7B12 + RTX 3090 the absolute error
on this model is ~1.5%, which is smaller than run-to-run variance on the same
machine. Do not mistake that for universality: it is a starting point that the
`tune` command then verifies by measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

# nvidia-smi reports allocation, which includes the CUDA context and llama.cpp's
# compute buffers on top of weights and KV cache. Calibrated against measured
# runs; the fit command verifies empirically rather than trusting this.
CUDA_OVERHEAD = 0.40e9

# Fraction of measured streaming bandwidth that llama.cpp actually achieves.
# Measured 0.80-0.84 across configs on the reference machine.
DEFAULT_EFFICIENCY = 0.80


@dataclass
class Prediction:
    n_cpu_moe: int
    cpu_bytes_per_tok: int
    gpu_bytes_per_tok: int
    vram_bytes: int
    fits: bool
    tps: float
    cpu_ms: float
    gpu_ms: float


def predict_tps(mc, n_cpu_moe, bw_cpu, bw_gpu, efficiency=DEFAULT_EFFICIENCY):
    """Predicted decode tokens/sec for one --n-cpu-moe setting."""
    cpu_b, gpu_b = mc.split(n_cpu_moe)
    t_cpu = cpu_b / (bw_cpu * efficiency) if bw_cpu else 0.0
    t_gpu = gpu_b / bw_gpu if bw_gpu else 0.0
    total = t_cpu + t_gpu
    return (1.0 / total if total else 0.0), t_cpu * 1e3, t_gpu * 1e3


def vram_needed(mc, g, n_cpu_moe, ctx, k_type="f16", v_type="f16",
                draft_mc=None, draft_g=None, draft_ctx=None):
    """Predicted VRAM allocation, matching what nvidia-smi will report."""
    from .model import kv_cache_bytes

    total = mc.resident_vram_weights(n_cpu_moe)
    total += kv_cache_bytes(g, ctx, k_type, v_type)
    if draft_mc is not None and draft_g is not None:
        total += draft_mc.total_bytes
        total += kv_cache_bytes(draft_g, draft_ctx or ctx, k_type, v_type)
    return total + CUDA_OVERHEAD


def dense_curve(mc, g, bw_cpu, bw_gpu, vram_budget, ctx,
                efficiency=DEFAULT_EFFICIENCY, k_type="f16", v_type="f16"):
    """Predicted throughput and VRAM for every --n-gpu-layers value.

    For a dense model the knob is --n-gpu-layers, not --n-cpu-moe: there are no
    routed experts to leave behind, so whole layers move, taking their share of
    the KV cache with them.
    """
    from .model import kv_cache_bytes
    kv_total = kv_cache_bytes(g, ctx, k_type, v_type)
    out = []
    for n in range(0, mc.n_layer + 1):
        cpu_b, gpu_b = mc.split_dense(n)
        t_cpu = cpu_b / (bw_cpu * efficiency) if bw_cpu else 0.0
        t_gpu = gpu_b / bw_gpu if bw_gpu else 0.0
        total = t_cpu + t_gpu
        tps = 1.0 / total if total else 0.0
        vr = (mc.resident_vram_dense(n)
              + kv_total * mc.kv_fraction_on_gpu(n) + CUDA_OVERHEAD)
        out.append(Prediction(n, cpu_b, gpu_b, vr, vr <= vram_budget,
                              tps, t_cpu * 1e3, t_gpu * 1e3))
    return out


def best_fit_dense(mc, g, bw_cpu, bw_gpu, vram_budget, ctx, headroom=1.0e9,
                   efficiency=DEFAULT_EFFICIENCY, k_type="f16", v_type="f16"):
    pts = dense_curve(mc, g, bw_cpu, bw_gpu, vram_budget, ctx, efficiency,
                      k_type, v_type)
    ok = [p for p in pts if p.vram_bytes <= vram_budget - headroom]
    if not ok:
        ok = [p for p in pts if p.fits]
    if not ok:
        return None, pts
    return max(ok, key=lambda p: p.tps), pts


def curve(mc, g, bw_cpu, bw_gpu, vram_budget, ctx, efficiency=DEFAULT_EFFICIENCY,
          k_type="f16", v_type="f16", draft_mc=None, draft_g=None):
    """Predicted throughput and VRAM for every --n-cpu-moe value."""
    out = []
    for n in range(mc.n_layer, -1, -1):
        tps, cms, gms = predict_tps(mc, n, bw_cpu, bw_gpu, efficiency)
        vr = vram_needed(mc, g, n, ctx, k_type, v_type, draft_mc, draft_g)
        cpu_b, gpu_b = mc.split(n)
        out.append(Prediction(n, cpu_b, gpu_b, vr, vr <= vram_budget,
                              tps, cms, gms))
    return out


def best_fit(mc, g, bw_cpu, bw_gpu, vram_budget, ctx, headroom=3.0e9,
             efficiency=DEFAULT_EFFICIENCY, k_type="f16", v_type="f16",
             draft_mc=None, draft_g=None):
    """Pick the fastest --n-cpu-moe that leaves a headroom margin.

    Headroom is not superstition. Compute buffers grow with prompt length, and
    a config that fits an empty context can fail with `unable to allocate
    CUDA0 buffer` several thousand tokens into a real conversation. On the
    reference machine the peak-throughput config sat at 97% of VRAM and the
    recommended one at 84%, for a 2% throughput cost.
    """
    pts = curve(mc, g, bw_cpu, bw_gpu, vram_budget, ctx, efficiency,
                k_type, v_type, draft_mc, draft_g)
    ok = [p for p in pts if p.vram_bytes <= vram_budget - headroom]
    if not ok:
        # Nothing fits with headroom; fall back to anything that fits at all.
        ok = [p for p in pts if p.fits]
    if not ok:
        return None, pts
    return max(ok, key=lambda p: p.tps), pts


# Running with no GPU at all (-ngl 0) lands well below the pure-bandwidth
# roofline: with nothing offloaded, router and attention work is no longer
# overlapped with expert reads and stops being free. Measured 2.21 tok/s against
# a 3.11 tok/s roofline on the reference machine, i.e. ~0.7.
CPU_ONLY_DERATE = 0.70


def cpu_only_tps(mc, bw_cpu, efficiency=DEFAULT_EFFICIENCY, derate=CPU_ONLY_DERATE):
    """Throughput with nothing on a GPU.

    Unlike a hybrid split this reads *every* byte from system RAM -- including
    attention and the shared expert, which --n-cpu-moe never moves.
    """
    if not bw_cpu:
        return 0.0
    seconds = mc.bytes_per_token / (bw_cpu * efficiency)
    return (1.0 / seconds) * derate if seconds else 0.0


def verdict(mc, best, bw_cpu, measured_tps=None, knob="--n-cpu-moe"):
    """Say plainly whether tuning is worth the user's time.

    An honest 'stop, you are already at the wall' is more useful than another
    sweep. This is the output that would have saved the most time on the
    reference machine.
    """
    lines = []
    if best is None:
        lines.append(f"DOES NOT FIT: no {knob} value fits this GPU at this "
                     "context length. Reduce --ctx-size, drop the draft model, "
                     "or use a smaller quantisation.")
        return lines

    cpu_share = best.cpu_ms / (best.cpu_ms + best.gpu_ms) if (best.cpu_ms + best.gpu_ms) else 0
    over_roofline = bool(measured_tps and best.tps and measured_tps > best.tps * 1.1)
    if cpu_share > 0.8 and not over_roofline:
        lines.append(
            f"Bandwidth-bound: {cpu_share*100:.0f}% of each token is spent reading "
            f"experts from system RAM at {bw_cpu/1e9:.0f} GB/s. Thread count and "
            "most other flags will not move this.")
        lines.append(
            "The only large wins available are: more VRAM (offload more layers), "
            "faster RAM, or a smaller quantisation.")
    if measured_tps:
        ratio = measured_tps / best.tps if best.tps else 0
        if cpu_share < 0.30:
            # GPU-bound. The bandwidth roofline ignores kernel launch overhead,
            # attention and sampling, all of which dominate once a model fits
            # in VRAM: 340 tok/s is 2.9 ms per token, which is not a bandwidth
            # number. This tool is calibrated for the CPU-offload regime, so it
            # says nothing rather than raising a false alarm.
            lines.append(
                f"Decode is GPU-bound ({(1-cpu_share)*100:.0f}% of each token "
                "is read from VRAM), where the bandwidth roofline does not "
                f"apply -- it ignores per-token fixed costs. Measured "
                f"{measured_tps:.2f} tok/s is the only meaningful figure here.")
        elif ratio > 1.1:
            # Beating the roofline does not mean the machine is fast; it means
            # the roofline was wrong. The model charges every routed expert to
            # DRAM on every token, but with a large last-level cache and skewed
            # top-k routing, hot experts stay resident and are not re-read.
            # Measured 10.43 tok/s against a 5.17 tok/s roofline on the
            # reference machine -- 202% -- for exactly this reason.
            lines.append(
                f"Measured {measured_tps:.2f} tok/s EXCEEDS the {best.tps:.2f} "
                f"tok/s roofline ({ratio*100:.0f}%). That means the estimate is "
                "wrong, not that the machine is exceptional.")
            lines.append(
                "The usual cause is expert cache reuse: the cost model charges "
                "every routed expert to DRAM on every token, but hot experts "
                "stay in last-level cache and are not re-read. Trust the "
                "measurement; treat the roofline as a floor for this model.")
        elif ratio > 0.9:
            lines.append(
                f"Measured {measured_tps:.2f} tok/s is {ratio*100:.0f}% of the "
                f"{best.tps:.2f} tok/s roofline. There is essentially nothing "
                "left to tune -- stop here.")
        elif ratio < 0.6:
            lines.append(
                f"Measured {measured_tps:.2f} tok/s is only {ratio*100:.0f}% of the "
                f"{best.tps:.2f} tok/s roofline. Something is wrong: check swap, "
                "run `wirl doctor`, and confirm nothing else is using the machine.")
    return lines
