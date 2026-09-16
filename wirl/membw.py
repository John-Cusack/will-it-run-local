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
import hashlib
import json
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass

CSRC = os.path.join(os.path.dirname(__file__), "csrc", "membw.c")
PACKAGED_PROBE = os.path.join(os.path.dirname(__file__), "_bin", "membw")


def cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = os.path.join(base, "will-it-run-local")
    os.makedirs(d, exist_ok=True)
    return d


def _build_probe(force=False) -> tuple:
    """Return (binary, failure reason), keeping diagnostics with the result."""
    if sys.platform != "linux":
        return None, "will-it-run-local supports Linux only"
    if not force and os.path.isfile(PACKAGED_PROBE) and os.access(PACKAGED_PROBE, os.X_OK):
        return PACKAGED_PROBE, None
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None, "no C compiler found (tried cc, gcc and clang)"
    try:
        command = shlex.split(cc) + ["-O2", "-pthread"]
    except ValueError as e:
        return None, f"invalid compiler command: {e}"
    try:
        with open(CSRC, "rb") as source:
            digest = hashlib.sha256(source.read() + b"\0" + json.dumps(command).encode()).hexdigest()[:12]
    except OSError as e:
        return None, f"cannot read bandwidth probe source: {e}"
    directory = cache_dir()
    out = os.path.join(directory, f"membw-{digest}")
    if os.path.isfile(out) and not force:
        return out, None
    pending = None
    try:
        # Separate installs with identical source share the cache; separate
        # compiles never write over a binary another benchmark may be running.
        fd, pending = tempfile.mkstemp(prefix=f".membw-{digest}-", dir=directory)
        os.close(fd)
        cmd = command + [CSRC, "-o", pending]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            detail = (p.stderr or p.stdout or f"exit {p.returncode}").strip()
            return None, f"bandwidth probe compile failed ({cc}): {detail}"
        os.chmod(pending, 0o755)
        os.replace(pending, out)
        return out, None
    except OSError as e:
        return None, f"bandwidth probe compile failed ({cc}): {e}"
    finally:
        if pending is not None and os.path.exists(pending):
            os.unlink(pending)


def build_probe(force=False) -> str | None:
    """Prefer the packaged probe; force=True explicitly recompiles its source."""
    binary, reason = _build_probe(force)
    if reason:
        print(f"warning: {reason}", file=sys.stderr)
    return binary


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


def _buffer_gib(gib=None) -> int:
    from .probe import mem_info

    memory = mem_info()
    # The 8 GiB allocation on a cache-filled 64 GiB host triggered swapping
    # despite 25 GiB MemAvailable. Leave headroom in genuinely unused RAM too.
    room = min(memory["available"], memory["free"]) // (1 << 30)
    budget = int(room * 0.35)
    if gib is None:
        gib = min(32, budget)
        if gib < 2:
            raise RuntimeError("not enough unused RAM for a 2 GiB bandwidth buffer; "
                               "let memory activity settle before measuring")
    if gib <= 0:
        raise ValueError("bandwidth buffer size must be positive")
    if gib > budget:
        raise RuntimeError(f"{gib} GiB bandwidth buffer exceeds unused RAM headroom "
                           f"({budget} GiB budget); choose a smaller buffer")
    return gib


def _check_swap(before):
    from .probe import swap_activity

    after = swap_activity()
    page_in = after["pswpin"] - before["pswpin"]
    page_out = after["pswpout"] - before["pswpout"]
    if page_in > 0 or page_out > 0:
        raise RuntimeError(f"bandwidth measurement invalid: swapping detected "
                           f"({page_in} pages in, {page_out} pages out); "
                           "let memory activity settle before measuring")


def _measure_probe(bin_path, mode, threads, gib, reps, cores=None) -> BwResult:
    """Use the same guards for packaged/source acceptance measurements."""
    from .probe import swap_activity

    gib = _buffer_gib(gib)
    cmd = [bin_path, mode, str(threads), str(gib), str(reps)]
    if cores:
        cmd.append(",".join(str(c) for c in cores))
    before = swap_activity()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"bandwidth probe failed: {p.stderr.strip()}")
    _check_swap(before)

    samples = []
    for line in p.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 4:
            el, nbytes = float(parts[2]), float(parts[3])
            if el > 0:
                samples.append(nbytes / el / 1e9)
    return BwResult(mode, threads, gib, samples, "c-probe")


def measure(mode="stream", threads=None, gib=None, reps=3, cores=None) -> BwResult:
    """Run the bandwidth probe.

    The buffer must be far larger than last-level cache or the result measures
    cache, not memory. It must also fit comfortably in free RAM: swapping
    during the probe produces a number that means nothing.
    """
    from .probe import cpu_info, swap_activity

    if threads is None:
        threads = cpu_info()["physical"]
    gib = _buffer_gib(gib)

    bin_path, reason = _build_probe()
    if not bin_path:
        before = swap_activity()
        r = _numpy_stream(gib, reps)
        if r:
            _check_swap(before)
            return r
        raise RuntimeError(f"{reason}; no numpy fallback: cannot measure bandwidth")
    return _measure_probe(bin_path, mode, threads, gib, reps, cores)


def measure_both(threads=None, gib=None, reps=3) -> dict:
    return {"stream": measure("stream", threads, gib, reps),
            "gather": measure("gather", threads, gib, reps)}
