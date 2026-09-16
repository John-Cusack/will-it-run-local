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
import os
import sys
from contextlib import contextmanager

LOCK_PATH = os.environ.get("WIRL_LOCK", "/tmp/will-it-run-local.benchmark.lock")


class BenchmarkBusy(RuntimeError):
    pass


class LockUnavailable(BenchmarkBusy):
    """The shared lock file cannot be opened by this user."""


def _open_lock():
    try:
        try:
            # O_CREAT on another user's file in sticky /tmp is rejected by
            # protected_regular, even when the file itself is world-writable.
            return os.open(LOCK_PATH, os.O_RDWR)
        except FileNotFoundError:
            try:
                fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o666)
            except FileExistsError:
                return os.open(LOCK_PATH, os.O_RDWR)
            try:
                os.fchmod(fd, 0o666)
            except OSError:
                os.close(fd)
                raise
            return fd
    except PermissionError:
        owner = "unknown"
        try:
            import pwd
            uid = os.stat(LOCK_PATH).st_uid
            owner = str(uid)
            try:
                owner = f"{pwd.getpwuid(uid).pw_name} (uid {uid})"
            except KeyError:
                pass
        except OSError:
            pass
        raise LockUnavailable(
            f"cannot open benchmark lock {LOCK_PATH} (owner: {owner}). "
            "Ask its owner to make it writable, or set WIRL_LOCK to a shared "
            "writable lock path for everyone benchmarking this machine.") from None


@contextmanager
def benchmark_lock(wait=False):
    import fcntl
    fd = _open_lock()
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


def foreign_gpu_users(min_mb=200, ignore_pids=(), gpu_uuid=None):
    """Processes holding meaningful GPU memory, excluding our own.

    Desktop compositors and browsers take tens of MB and are harmless. An
    inference server holding gigabytes is not: it will both skew the timing
    and make VRAM fitting meaningless.
    """
    from .probe import gpu_processes
    out = []
    for p in gpu_processes():
        if gpu_uuid is not None and p.get("gpu_uuid") != gpu_uuid:
            continue
        try:
            mb = int(float(p["vram_mb"]))
            pid = int(p["pid"])
        except (ValueError, KeyError):
            continue
        if mb >= min_mb and pid not in ignore_pids:
            out.append({"pid": pid, "name": p["name"].split()[0], "vram_mb": mb})
    return out
