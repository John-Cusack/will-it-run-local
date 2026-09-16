"""Measured sweeps.

Prediction gets you to a starting configuration in seconds. This confirms it,
and it exists because prediction is not measurement: the model here is accurate
to a few percent on the machine it was calibrated on, and nobody should trust
that transfers.

Rules, each learned the hard way:
  * one configuration at a time, under a lock
  * several repetitions per configuration, with the spread reported
  * search the cheap axis first and stop when it stops paying
"""

from __future__ import annotations

import copy

from .runner import RunConfig, RunResult, run_config
from .search import set_ncmoe, set_ngl


def sweep_offload(base: RunConfig, binary, candidates, moe, reps=3, n_tokens=400,
                log_dir=None) -> list:
    """Walk towards more GPU offload until it stops fitting.

    Descending --n-cpu-moe or ascending --n-gpu-layers puts more on the GPU, so
    throughput improves monotonically until allocation fails. The interesting
    part is not the peak but where the peak sits relative to the VRAM ceiling.
    """
    results = []
    for n in candidates:
        cfg = copy.copy(base)
        (set_ncmoe if moe else set_ngl)(cfg, n)
        print(f"  {cfg.label()}", flush=True)
        r = run_config(cfg, binary, reps=reps, n_tokens=n_tokens, log_dir=log_dir)
        results.append(r)
        if not r.ok:
            print(f"    FAILED: {r.error}", flush=True)
            # Both candidate orders increase VRAM; later values will also fail.
            break
    return results


def sweep_draft_depth(base: RunConfig, binary, depths=(1, 2, 3), reps=3,
                      n_tokens=400, log_dir=None) -> list:
    """Sweep --spec-draft-n-max shallow-first, and stop when it stops helping.

    Deeper drafting is not better on a sparse MoE. Accepting k tokens per pass
    only pays if verifying k tokens costs less than k single-token passes, and
    on a top-k-of-many router it largely does not: each additional draft token
    routes to mostly different experts, so expert reads scale close to linearly
    with the batch. On the reference machine depth 1 beat depth 5 by 33%, and
    depth 5 was slower than not drafting at all.
    """
    results = []
    best = 0.0
    for d in depths:
        cfg = copy.copy(base)
        cfg.draft_n_max = d
        print(f"  draft depth n_max={d}", flush=True)
        r = run_config(cfg, binary, reps=reps, n_tokens=n_tokens, log_dir=log_dir)
        results.append(r)
        if not r.ok:
            break
        if r.mean < best * 0.98:
            print("    slower than the previous depth; deeper will not help. "
                  "Stopping.", flush=True)
            break
        best = max(best, r.mean)
    return results


def sweep_threads(base: RunConfig, binary, counts, reps=3, n_tokens=400,
                  log_dir=None) -> list:
    """Sweep thread count.

    Included for completeness and because people expect it, but on a
    bandwidth-bound model it is usually a null result: 24 vs 64 threads differed
    by 0.9% on the reference machine, less than the run-to-run spread. If this
    sweep is flat, that is itself the finding -- it confirms you are memory
    bound and should stop tuning the CPU side.
    """
    results = []
    for t in counts:
        cfg = copy.copy(base)
        cfg.threads = t
        print(f"  threads={t}", flush=True)
        results.append(run_config(cfg, binary, reps=reps, n_tokens=n_tokens,
                                  log_dir=log_dir))
    return results


def summarise(results) -> str:
    rows = ["| config | tok/s | spread | peak VRAM | startup |",
            "|---|---|---|---|---|"]
    for r in results:
        if not r.ok:
            rows.append(f"| {r.config.label()} | FAILED | | | {r.error or ''} |")
            continue
        rows.append(f"| {r.config.label()} | {r.mean:.2f} | {r.spread_pct:.1f}% | "
                    f"{r.peak_vram / (1 << 20):.0f} MiB | {r.startup_s:.0f}s |")
    return "\n".join(rows)


def repeatability(results) -> str:
    """Report the spread of any configuration measured more than once.

    A sweep often re-measures the same configuration in different phases. That
    is an accidental control, and a valuable one: it gives the cross-launch
    variance, which is the real uncertainty on every other comparison in the
    table. Differences smaller than this are not results.
    """
    from collections import defaultdict
    by_label = defaultdict(list)
    for r in results:
        if r.ok and r.samples:
            by_label[r.config.label()].append(r.mean)
    repeated = {k: v for k, v in by_label.items() if len(v) > 1}
    if not repeated:
        return ""
    lines = []
    worst = 0.0
    for label, means in repeated.items():
        spread = (max(means) - min(means)) / (sum(means) / len(means)) * 100
        worst = max(worst, spread)
        vals = ", ".join(f"{m:.2f}" for m in means)
        lines.append(f"  {label} measured {len(means)}x: {vals} tok/s "
                     f"({spread:.1f}% apart)")
    lines.insert(0, "Same configuration, separate launches:")
    lines.append(f"So treat differences below about {worst:.0f}% in the table "
                 "above as noise, not findings.")
    return "\n".join(lines)


def flag_unstable(results, threshold=8.0) -> list:
    """Call out configurations whose repetitions disagree.

    A wide spread means the machine, not the configuration, is the variable.
    Comparing configurations in that state produces confident nonsense.
    """
    bad = [r for r in results if r.ok and r.spread_pct > threshold]
    if not bad:
        return []
    msg = [f"Unstable measurements ({threshold:.0f}%+ spread within a single config):"]
    for r in bad:
        msg.append(f"  {r.config.label()}: {[round(s, 2) for s in r.samples]} tok/s")
    msg.append("Do not compare configurations until this settles. Check swap, "
               "background load, thermals, and page-cache state.")
    return msg
