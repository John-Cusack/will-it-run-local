"""Measured memory bandwidth.

The single most useful number for predicting local LLM decode speed, and the
one nobody measures. Vendor figures assume every channel is populated and
clocked at spec; a machine with a dead DIMM, a downclocked channel, or an
unbalanced NUMA layout can deliver less than half of them, and no amount of
config tuning recovers that.

Compiles a small C probe on first use and caches the binary.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass

CSRC = os.path.join(os.path.dirname(__file__), "csrc", "membw.c")


def cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = os.path.join(base, "will-it-run-local")
    os.makedirs(d, exist_ok=True)
    return d


def build_probe(force=False) -> str | None:
    """Compile the probe. Returns the binary path, or None if no compiler."""
    out = os.path.join(cache_dir(), "membw")
    if os.path.exists(out) and not force:
        if os.path.getmtime(out) >= os.path.getmtime(CSRC):
            return out
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None
    cmd = [cc, "-O2", "-pthread", CSRC, "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"warning: could not build bandwidth probe:\n{p.stderr}", file=sys.stderr)
        return None
    return out


@dataclass
class BwResult:
    mode: str
    threads: int
    gib: int
    samples: list          # GB/s per repetition
    method: str

    @property
    def best(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def spread_pct(self) -> float:
        if len(self.samples) < 2 or not self.median:
            return 0.0
        return 100.0 * (max(self.samples) - min(self.samples)) / self.median


def _numpy_stream(gib: int, reps: int) -> BwResult | None:
    """Fallback when no C compiler exists. A lower bound, not a measurement."""
    try:
        import time

        import numpy as np
    except ImportError:
        return None
    n = (gib << 30) // 8
    a = np.ones(n, dtype=np.uint64)
    samples = []
    for _ in range(reps):
        t0 = time.monotonic()
        int(a.sum())
        el = time.monotonic() - t0
        samples.append((n * 8) / el / 1e9)
    return BwResult("stream", 1, gib, samples, "numpy (single-threaded lower bound)")


def measure(mode="stream", threads=None, gib=None, reps=3, cores=None) -> BwResult:
    """Run the bandwidth probe.

    The buffer must be far larger than last-level cache or the result measures
    cache, not memory. It must also fit comfortably in free RAM: swapping
    during the probe produces a number that means nothing.
    """
    from .probe import cpu_info, mem_info

    if threads is None:
        threads = cpu_info()["physical"]
    if gib is None:
        avail_gib = mem_info()["available"] // (1 << 30)
        gib = max(2, min(32, int(avail_gib * 0.35)))

    bin_path = build_probe()
    if not bin_path:
        r = _numpy_stream(gib, reps)
        if r:
            return r
        raise RuntimeError("no C compiler and no numpy: cannot measure bandwidth")

    cmd = [bin_path, mode, str(threads), str(gib), str(reps)]
    if cores:
        cmd.append(",".join(str(c) for c in cores))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"bandwidth probe failed: {p.stderr.strip()}")

    samples = []
    for line in p.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 4:
            el, nbytes = float(parts[2]), float(parts[3])
            if el > 0:
                samples.append(nbytes / el / 1e9)
    return BwResult(mode, threads, gib, samples, "c-probe")


def measure_both(threads=None, gib=None, reps=3) -> dict:
    return {"stream": measure("stream", threads, gib, reps),
            "gather": measure("gather", threads, gib, reps)}
