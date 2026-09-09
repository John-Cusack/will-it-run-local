"""Exclusion for benchmarking.

Concurrent benchmarks do not merely add noise -- they can do lasting damage.
On the reference machine two benchmarks timed GPU kernels at the same time
while a Triton-based runtime was autotuning. Triton picks kernels by timing
candidates, and under contention it picked badly, then wrote those choices to
a persistent cache. The bad configuration survived restarts and reboots and
cost 38% throughput until it was found days later.

So: take a lock, refuse to run beside other GPU work, and never assume a
number obtained under contention is merely noisy.
"""

from __future__ import annotations

import errno
import fcntl
import os
import sys
from contextlib import contextmanager

LOCK_PATH = os.environ.get("WIRL_LOCK", "/tmp/will-it-run-local.benchmark.lock")


class BenchmarkBusy(RuntimeError):
    pass


@contextmanager
def benchmark_lock(wait=False):
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        flags = fcntl.LOCK_EX if wait else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(fd, flags)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                try:
                    with open(LOCK_PATH) as f:
                        holder = f.read().strip()
                except OSError:
                    holder = "unknown"
                raise BenchmarkBusy(
                    f"another benchmark holds {LOCK_PATH} (owner: {holder}). "
                    "Concurrent benchmarks produce invalid numbers; wait for it "
                    "to finish or pass --wait.") from None
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} argv={' '.join(sys.argv)}\n".encode())
        os.fsync(fd)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def foreign_gpu_users(min_mb=200, ignore_pids=()):
    """Processes holding meaningful GPU memory, excluding our own.

    Desktop compositors and browsers take tens of MB and are harmless. An
    inference server holding gigabytes is not: it will both skew the timing
    and make VRAM fitting meaningless.
    """
    from .probe import gpu_processes
    out = []
    for p in gpu_processes():
        try:
            mb = int(float(p["vram_mb"]))
            pid = int(p["pid"])
        except (ValueError, KeyError):
            continue
        if mb >= min_mb and pid not in ignore_pids:
            out.append({"pid": pid, "name": p["name"].split()[0], "vram_mb": mb})
    return out
