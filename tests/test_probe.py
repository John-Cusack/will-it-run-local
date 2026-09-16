"""Hardware parsing against reported text, without querying real devices."""
from wirl import probe
import pytest


def test_gpu_info_parses_uuid(monkeypatch):
    calls = []
    monkeypatch.setattr(probe.shutil, "which", lambda name: name)

    def run(cmd):
        calls.append(cmd)
        return "1, RTX 3090, 24576, 1234, 23342, 8.6, 595.84, 4, 16, 9751, 350, GPU-one\n"

    monkeypatch.setattr(probe, "_run", run)
    gpu = probe.gpu_info()[0]
    assert gpu["uuid"] == "GPU-one" and gpu["index"] == 1
    assert gpu["vram_used"] == 1234 << 20
    assert calls[0][1].endswith(",uuid")


def test_gpu_processes_parse_uuid(monkeypatch):
    monkeypatch.setattr(probe.shutil, "which", lambda name: name)
    calls = []

    def run(cmd):
        calls.append(cmd)
        return "42, llama-server, 1234, GPU-one\n"

    monkeypatch.setattr(probe, "_run", run)
    assert probe.gpu_processes() == [
        {"pid": "42", "name": "llama-server", "vram_mb": "1234", "gpu_uuid": "GPU-one"}]
    assert calls[0][1].endswith(",gpu_uuid")


@pytest.mark.parametrize("text,cpus", [("0", [0]), ("0-3,8", [0, 1, 2, 3, 8]),
                                      ("0,2,4-6", [0, 2, 4, 5, 6]), ("", [])])
def test_cpu_lists_round_trip(text, cpus):
    assert probe.parse_cpulist(text) == cpus
    assert probe.format_cpulist(cpus) == text
    assert probe.format_cpulist(list(reversed(cpus)) + cpus) == text


def _topology(monkeypatch, siblings, allowed, older=False):
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda pid: set(allowed))

    def read(path, default=None):
        cpu = int(path.split("/cpu")[-1].split("/")[0])
        if older and path.endswith("core_cpus_list"):
            return default
        return siblings.get(cpu, default)

    monkeypatch.setattr(probe, "_read", read)


def test_server_topology_uses_one_thread_per_core(monkeypatch):
    siblings = {cpu: f"{cpu % 64},{cpu % 64 + 64}" for cpu in range(128)}
    _topology(monkeypatch, siblings, range(128))
    assert probe.physical_core_cpus() == list(range(64))


def test_hybrid_topology_uses_all_e_cores(monkeypatch):
    siblings = {cpu: f"{cpu // 2 * 2}-{cpu // 2 * 2 + 1}" for cpu in range(16)}
    siblings.update({cpu: str(cpu) for cpu in range(16, 32)})
    _topology(monkeypatch, siblings, range(32))
    cpus = list(range(0, 16, 2)) + list(range(16, 32))
    assert probe.physical_core_cpus() == cpus
    assert probe.format_cpulist(cpus) == "0,2,4,6,8,10,12,14,16-31"


def test_topology_respects_affinity_and_older_kernel(monkeypatch):
    _topology(monkeypatch, {0: "0-1", 1: "0-1", 2: "2-3", 3: "2-3", 4: "4"},
              [1, 2, 3, 4], older=True)
    assert probe.physical_core_cpus() == [1, 2, 4]


def test_unreadable_topology_does_not_guess(monkeypatch):
    _topology(monkeypatch, {}, [0, 1])
    assert probe.physical_core_cpus() is None
