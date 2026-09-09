"""Find the best configuration by running it, not by predicting it.

The prediction model in predict.py is used for exactly one thing here: choosing
which configurations are worth launching, so a sweep is a handful of runs
instead of dozens. Every number this module reports comes from a server that
actually started and actually generated tokens.

The VRAM boundary in particular is found empirically. A predicted boundary is a
guess about allocator behaviour, compute buffer growth and driver overhead; the
real one is wherever `llama-server` stops starting, and that is cheap to find
by bisection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .runner import RunConfig, run_config


@dataclass
class SearchLog:
    """Everything that was tried, so the result can be audited afterwards."""
    attempts: list = field(default_factory=list)

    def add(self, kind, cfg, result, note=""):
        self.attempts.append({"kind": kind, "label": cfg.label(),
                              "ok": result.ok if result else False,
                              "tps": result.mean if result and result.ok else 0.0,
                              "vram": result.peak_vram if result else 0,
                              "note": note or (result.error if result and not result.ok else "")})


def probe_fit(cfg, binary, log=None, startup_timeout=1200, log_dir=None):
    """Does this configuration actually start? One launch, no generation.

    Cheaper than a full measurement and answers the only question that matters
    when bracketing the VRAM boundary.
    """
    r = run_config(cfg, binary, reps=1, n_tokens=8,
                   startup_timeout=startup_timeout, log_dir=log_dir, verbose=False)
    if log:
        log.add("fit-probe", cfg, r)
    return r


def _bisect_edge(base, binary, lo, hi, apply, flag, prefer_low, log=None,
                 log_dir=None, verbose=True):
    """Bisect a monotonic knob for the extreme value that still starts.

    `prefer_low=True` for --n-cpu-moe, where *lower* puts more work on the GPU
    and costs more VRAM. `prefer_low=False` for --n-gpu-layers, where *higher*
    does. Getting that direction wrong silently returns the safest possible
    configuration instead of the fastest, so the two are kept explicit.
    """
    best_ok = None
    if verbose:
        print(f"  bracketing the VRAM boundary between {flag} {lo} and {hi}",
              flush=True)
    while lo <= hi:
        mid = (lo + hi) // 2
        cfg = copy.copy(base)
        apply(cfg, mid)
        if verbose:
            print(f"    trying {flag} {mid} ... ", end="", flush=True)
        r = probe_fit(cfg, binary, log, log_dir=log_dir)
        if r.ok:
            if verbose:
                print(f"starts ({r.peak_vram >> 20} MiB)", flush=True)
            best_ok = mid
            if prefer_low:
                hi = mid - 1      # push more onto the GPU
            else:
                lo = mid + 1
        else:
            if verbose:
                print("does not start", flush=True)
            if prefer_low:
                lo = mid + 1
            else:
                hi = mid - 1
    return best_ok


def _set_ncmoe(cfg, v):
    cfg.n_cpu_moe = v
    cfg.n_gpu_layers = 99


def _set_ngl(cfg, v):
    cfg.n_cpu_moe = None          # meaningless on a dense model
    cfg.n_gpu_layers = v


def find_vram_edge(base: RunConfig, binary, lo, hi, log=None, log_dir=None,
                   verbose=True):
    """Smallest --n-cpu-moe that still starts (mixture-of-experts models)."""
    return _bisect_edge(base, binary, lo, hi, _set_ncmoe, "--n-cpu-moe",
                        prefer_low=True, log=log, log_dir=log_dir,
                        verbose=verbose)


def find_ngl_edge(base: RunConfig, binary, lo, hi, log=None, log_dir=None,
                  verbose=True):
    """Largest --n-gpu-layers that still starts (dense models).

    --n-cpu-moe does nothing on a dense model: there are no routed experts to
    leave behind. Bisecting it would launch several identical servers and call
    the resulting noise a result.
    """
    return _bisect_edge(base, binary, lo, hi, _set_ngl, "--n-gpu-layers",
                        prefer_low=False, log=log, log_dir=log_dir,
                        verbose=verbose)


def measure_around(base: RunConfig, binary, centre, span=2, reps=3, n_tokens=400,
                   log=None, log_dir=None, verbose=True, max_layer=None,
                   moe=True):
    """Measure properly at the empirical boundary and on the safe side of it.

    Values past `centre` are known not to start, so they are not attempted.
    The ones measured trade throughput for VRAM headroom, which is the trade
    the user actually has to make.
    """
    results = []
    if moe:
        hi = centre + span if max_layer is None else min(centre + span, max_layer)
        values, apply, flag = range(centre, hi + 1), _set_ncmoe, "--n-cpu-moe"
    else:
        lo = max(0, centre - span)
        values, apply, flag = range(centre, lo - 1, -1), _set_ngl, "--n-gpu-layers"
    for n in values:
        cfg = copy.copy(base)
        apply(cfg, n)
        if verbose:
            print(f"  measuring {flag} {n}", flush=True)
        r = run_config(cfg, binary, reps=reps, n_tokens=n_tokens,
                       log_dir=log_dir, verbose=verbose)
        if log:
            log.add("measure", cfg, r)
        results.append(r)
    return results


def pick_recommended(results, vram_total, headroom_bytes=3 << 30):
    """Choose from measured results, preferring headroom over the last 2%.

    The fastest configuration measured on the reference machine sat at 97% of
    VRAM and failed to allocate partway through a long conversation. This picks
    the fastest one that still leaves room, and falls back to the fastest
    overall if none does.
    """
    ok = [r for r in results if r.ok and r.samples]
    if not ok:
        return None, ""
    roomy = [r for r in ok if r.peak_vram <= vram_total - headroom_bytes]
    if not roomy:
        return max(ok, key=lambda r: r.mean), "fastest measured (no config left headroom)"
    best = max(roomy, key=lambda r: r.mean)
    fastest = max(ok, key=lambda r: r.mean)
    if best is not fastest and fastest.mean > 0:
        cost = (fastest.mean - best.mean) / fastest.mean * 100
        return best, (f"fastest with headroom; gives up {cost:.1f}% against "
                      f"{fastest.config.label()} to keep "
                      f"{(vram_total - best.peak_vram) >> 20} MiB free")
    return best, "fastest measured"


# ---------------------------------------------------------------------------
# Phase C: what a real conversation actually feels like.
#
# Everything above measures decode with an almost-empty context, which is the
# easiest case and not the one anybody uses. These probes run against a server
# that is already up, so they cost seconds rather than another startup, and
# they answer the two questions that decide whether a setup is usable:
#
#   how long before the first token, on a prompt the size of real chat history
#   how much does decode slow down once that history is in the KV cache
# ---------------------------------------------------------------------------

@dataclass
class DepthPoint:
    requested: int
    prompt_n: int
    prefill_tps: float
    decode_tps: float
    ttft_s: float


def profile_depth(cfg, host, port, depths=(0, 2048, 8192), gen_tokens=120,
                  verbose=True):
    """Measure prefill and decode against a live server at several history sizes."""
    from .runner import _generate, build_prompt

    out = []
    for d in depths:
        if d and d > cfg.ctx * 0.85:
            continue                       # leave room for the generation
        prompt = build_prompt(d)
        try:
            dec, pre, pn = _generate(host, port, gen_tokens, prompt=prompt)
        except Exception as e:             # noqa: BLE001
            if verbose:
                print(f"    depth {d}: failed ({e})", flush=True)
            continue
        pn = pn or d
        ttft = (pn / pre) if (pre and pn) else 0.0
        pt = DepthPoint(d, pn, pre or 0.0, dec or 0.0, ttft)
        out.append(pt)
        if verbose:
            print(f"    history {pn:>6} tok: prefill {pt.prefill_tps:7.1f} tok/s "
                  f"({ttft:5.1f}s to first token), decode {pt.decode_tps:5.2f} tok/s",
                  flush=True)
    return out


def summarise_depth(points, ctx):
    if not points:
        return ""
    rows = ["| history | prefill | time to first token | decode |",
            "|---|---|---|---|"]
    for p in points:
        rows.append(f"| {p.prompt_n} tok | {p.prefill_tps:.0f} tok/s | "
                    f"{p.ttft_s:.1f} s | {p.decode_tps:.2f} tok/s |")
    base = points[0].decode_tps
    deepest = max(points, key=lambda p: p.prompt_n)
    rows.append("")
    if base and len(points) > 1:
        drop = (base - deepest.decode_tps) / base * 100
        if drop > 2:
            rows.append(f"Decode falls {drop:.0f}% between an empty context and "
                        f"{deepest.prompt_n} tokens of history.")
        else:
            rows.append(f"Decode is unaffected by context depth "
                        f"({base:.2f} -> {deepest.decode_tps:.2f} tok/s at "
                        f"{deepest.prompt_n} tokens). The KV cache is on the GPU, "
                        "where re-reading it is cheap.")
    if deepest.ttft_s > 20:
        rows.append("")
        rows.append(f"**{deepest.ttft_s:.0f} seconds to the first token** at "
                    f"{deepest.prompt_n} tokens of history. On a long "
                    "conversation this, not decode speed, is what you will "
                    "actually feel. Prompt caching (`--cache-reuse`, or a client "
                    "that reuses the prefix) is the lever that matters here.")
    return "\n".join(rows)
