"""Lock diagnostics and filtering against fake failures and process records."""
import errno
import fcntl
import os
import pwd

import pytest

from wirl import lock, probe

pytestmark = pytest.mark.usefixtures("isolated_runtime")


def test_failed_creation_permissions_close_descriptor(monkeypatch):
    calls = []

    def opened(path, flags, *args):
        if not flags & os.O_CREAT:
            raise FileNotFoundError(path)
        return 123

    monkeypatch.setattr(lock.os, "open", opened)
    monkeypatch.setattr(lock.os, "close", lambda fd: calls.append(fd))
    monkeypatch.setattr(lock.os, "fchmod", lambda *a: (_ for _ in ()).throw(OSError("failed chmod")))
    with pytest.raises(OSError, match="failed chmod"):
        lock._open_lock()
    assert calls == [123]


@pytest.mark.parametrize("stat_fails", [True, False])
def test_inaccessible_lock_handles_unknown_owners(monkeypatch, tmp_path, stat_fails):
    path = tmp_path / "lock"
    path.write_text("existing")
    monkeypatch.setattr(lock, "LOCK_PATH", str(path))
    monkeypatch.setattr(lock.os, "open", lambda *a: (_ for _ in ()).throw(PermissionError("denied")))
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: (_ for _ in ()).throw(KeyError(uid)))
    if stat_fails:
        stat = os.stat
        monkeypatch.setattr(lock.os, "stat", lambda p, *a, **kw:
                            (_ for _ in ()).throw(OSError("gone")) if p == str(path) else stat(p, *a, **kw))
    with pytest.raises(lock.LockUnavailable, match="owner: " + ("unknown" if stat_fails else str(os.getuid()))):
        lock._open_lock()


@pytest.mark.parametrize("error", [errno.EAGAIN, errno.EIO])
def test_flock_failure_closes_lock_and_reports_holder_errors(monkeypatch, tmp_path, error):
    path = tmp_path / "lock"
    monkeypatch.setattr(lock, "LOCK_PATH", str(path))
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    monkeypatch.setattr(lock, "_open_lock", lambda: descriptor)
    calls = []

    def flock(fd, flags):
        calls.append(flags)
        if flags != fcntl.LOCK_UN:
            raise OSError(error, "flock failure")

    monkeypatch.setattr(fcntl, "flock", flock)
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError("holder unreadable")))
    with pytest.raises(lock.BenchmarkBusy if error == errno.EAGAIN else OSError):
        with lock.benchmark_lock(wait=True):
            pytest.fail("failed flock yielded")
    assert calls == [fcntl.LOCK_EX, fcntl.LOCK_UN]
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_foreign_gpu_users_skip_ignored_small_and_malformed(monkeypatch):
    rows = [dict(pid="1", name="server --args", vram_mb="1000.5", gpu_uuid="GPU-one"),
            dict(pid="2", name="self", vram_mb="500", gpu_uuid="GPU-one"),
            dict(pid="3", name="desktop", vram_mb="20", gpu_uuid="GPU-one"),
            dict(pid="bad", name="bad", vram_mb="N/A"), {}]
    monkeypatch.setattr(probe, "gpu_processes", lambda: rows)
    assert lock.foreign_gpu_users(ignore_pids=[2]) == [{"pid": 1, "name": "server", "vram_mb": 1000}]
    assert lock.foreign_gpu_users(gpu_uuid="GPU-other") == []
