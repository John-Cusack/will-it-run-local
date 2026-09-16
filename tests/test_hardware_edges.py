"""Kernel inventories and pre-flight checks against fake machine state."""
import io
import subprocess
from types import SimpleNamespace

import pytest

from wirl import doctor, lock, probe, runner

pytestmark = pytest.mark.usefixtures("isolated_runtime")


def test_read_and_command_failures(monkeypatch, tmp_path):
    path = tmp_path / "kernel-value"
    path.write_text(" value\n")
    assert probe._read(path) == "value"
    assert probe._read(tmp_path / "absent", "default") == "default"
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "answer", ""))
    assert probe._run(["fake"]) == "answer"
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "failure"))
    assert probe._run(["fake"]) is None
    for error in (OSError("absent"), subprocess.TimeoutExpired("fake", 1)):
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(error))
        assert probe._run(["fake"]) is None


@pytest.mark.parametrize("text", ["3-2", "1-2-3", "-1", "x"])
def test_invalid_cpu_lists(text):
    with pytest.raises(ValueError):
        probe.parse_cpulist(text)


@pytest.mark.parametrize("siblings", ["1", "invalid"])
def test_invalid_topology_is_not_guessed(monkeypatch, siblings):
    monkeypatch.setattr(probe.os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setattr(probe, "_read", lambda *a: siblings)
    assert probe.physical_core_cpus() is None


@pytest.mark.parametrize("reported", [True, False])
def test_cpu_inventory_counts_socket_core_pairs_and_numa(monkeypatch, reported):
    text = "\n".join(f"model name: Test CPU\nphysical id: {s}\ncore id: {c}\nflags: avx2 fma avx512f"
                     for s in range(2) for c in range(2)) if reported else "core id: 0"
    monkeypatch.setattr(probe, "_read", lambda path, default=None:
                        text if path == "/proc/cpuinfo" else "performance")
    monkeypatch.setattr(probe.os, "cpu_count", lambda: 8 if reported else None)
    monkeypatch.setattr(probe.os.path, "isdir", lambda path: reported)
    monkeypatch.setattr(probe.os, "listdir", lambda path: ["node0", "node1", "online"])
    info = probe.cpu_info()
    assert info["physical"] == (4 if reported else 0)
    assert info["sockets"] == (2 if reported else 1)
    assert info["numa_nodes"] == (2 if reported else 1)
    assert info["isa"]["avx2"] == reported
    assert info["governor"] == "performance"


@pytest.mark.parametrize("edac", [True, False])
def test_memory_and_swap_inventory(monkeypatch, edac):
    values = {"/proc/meminfo": "MemTotal: 100 kB\nMemAvailable: 70 kB\nSwapTotal: 40 kB\nSwapFree: 30 kB\nEmpty:",
              "/sys/devices/system/edac/mc/mc0/size_mb": "128",
              "/sys/devices/system/edac/mc/mc1/size_mb": "unknown",
              "/proc/vmstat": "pswpin 3\npswpout 7\npgfault 999"}
    monkeypatch.setattr(probe, "_read", lambda path, default=None: values.get(path, default))
    monkeypatch.setattr(probe.os.path, "isdir", lambda path: edac)
    monkeypatch.setattr(probe.os, "listdir", lambda path: ["mc0", "mc1", "mc2"])
    info = probe.mem_info()
    assert info == dict(total=100*1024, available=70*1024, swap_total=40*1024,
                        swap_free=30*1024, swap_used=10*1024, edac_total=128*1024*1024 if edac else 0)
    assert probe.swap_activity() == {"pswpin": 3, "pswpout": 7}


def test_gpu_absence_failed_queries_and_malformed_rows(monkeypatch):
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)
    assert probe.gpu_info() == [] and probe.gpu_processes() == []
    monkeypatch.setattr(probe.shutil, "which", lambda name: "fake-smi")
    for output in (None, "", "\n"):
        monkeypatch.setattr(probe, "_run", lambda *a: output)
        assert probe.gpu_info() == [] and probe.gpu_processes() == []
    monkeypatch.setattr(probe, "_run", lambda *a: "short,row\nN/A, Test GPU, N/A, N/A, N/A, unknown, driver, N/A, N/A, N/A, N/A, GPU-test")
    gpu = probe.gpu_info()[0]
    assert gpu["index"] is None and gpu["vram_total"] == 0 and gpu["pcie_gen"] is None
    monkeypatch.setattr(probe, "_run", lambda *a: "short,row")
    assert probe.gpu_processes() == []
    assert probe.gpu_bandwidth_hint({"name": "RTX 3090"}) == 936e9
    assert probe.gpu_bandwidth_hint({}) is None


def test_probe_composes_inventory(monkeypatch):
    for name in ("cpu_info", "mem_info", "gpu_info", "gpu_processes"):
        monkeypatch.setattr(probe, name, lambda name=name: name)
    assert probe.probe() == dict(cpu="cpu_info", mem="mem_info", gpus="gpu_info", gpu_procs="gpu_processes")


def test_doctor_governors_isa_cache_and_capacity():
    assert doctor.check_swap(dict(swap_used=1<<30)).status == doctor.WARN
    assert doctor.check_dimm_population(dict(edac_total=0, total=1)).status == doctor.OK
    assert doctor.check_gpu_free([], []).status == doctor.WARN
    gpu = dict(index=1, name="test", vram_total=8<<30, vram_free=7<<30, driver="mock")
    assert doctor.check_gpu_free([gpu], [], gpu_index=0).status == doctor.FAIL
    assert doctor.check_gpu_free([gpu], [], gpu_index=1).status == doctor.OK
    assert doctor.check_governor({}).status == doctor.OK
    assert doctor.check_governor(dict(governor="powersave")).status == doctor.WARN
    assert doctor.check_governor(dict(governor="performance")).status == doctor.OK
    isa = dict(avx2=True, fma=True, avx512f=False, avx512_bf16=False, amx_int8=False)
    assert "No AVX-512" in doctor.check_isa(dict(isa=isa, model="test")).detail
    isa["avx512f"] = True
    assert "No AVX-512" not in doctor.check_isa(dict(isa=isa, model="test")).detail
    assert doctor.check_k_cache_quant("f16", {}).status == doctor.OK
    assert doctor.check_ram_for_model(dict(available=100), 95).status == doctor.WARN


@pytest.mark.parametrize("text,status", [(None, "ok"), ("always madvise [never]", "ok"),
                                         ("[always] madvise never", "warn"), ("madvise", "ok")])
def test_thp_reading(monkeypatch, text, status):
    monkeypatch.setattr(doctor.os.path, "exists", lambda p: text is not None)
    monkeypatch.setattr("builtins.open", lambda *a, **kw: io.StringIO(text))
    assert doctor.check_thp().status == status


@pytest.mark.parametrize("files", [[], ["kernel1", "kernel2"]])
def test_autotune_cache_inventory(monkeypatch, files):
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.setattr(doctor.os.path, "isdir", lambda p: p.endswith("/cache"))
    monkeypatch.setattr(doctor.os, "walk", lambda p: [(p, [], files)])
    check = doctor.check_stale_autotune()
    assert check.status == (doctor.WARN if files else doctor.OK)
    if files:
        assert "2 files" in check.detail


def test_disk_failures_and_free_space(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: SimpleNamespace(free=100))
    assert doctor.check_disk("fake", 101).status == doctor.FAIL
    assert doctor.check_disk("fake", 100).status == doctor.OK
    assert doctor.check_disk("fake", 0).status == doctor.OK
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: (_ for _ in ()).throw(OSError("missing")))
    assert doctor.check_disk("fake", 100).status == doctor.OK


@pytest.mark.parametrize("banner,library,ldd,status", [
    ("build 123\nCUDA", False, "", "ok"), ("version 123", True, "", "ok"),
    ("", False, "libcuda.so", "ok"), ("", False, "libc.so", "warn"),
    (None, False, None, "warn"),
])
def test_server_backend_detection(monkeypatch, banner, library, ldd, status):
    monkeypatch.setattr(runner, "find_server", lambda explicit: "/fake/bin/llama-server")
    monkeypatch.setattr(doctor.os.path, "exists", lambda p: library and p.endswith("../lib/libggml-cuda.so"))
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        text = banner if cmd[0] != "ldd" else ldd
        if text is None:
            raise OSError("unavailable")
        return subprocess.CompletedProcess(cmd, 0, text, "")

    monkeypatch.setattr(subprocess, "run", run)
    result = doctor.check_llama_server()
    assert result.status == status
    if banner:
        assert banner.splitlines()[0] in result.detail
    assert calls[0] == ["/fake/bin/llama-server", "--version"]


def test_prompt_log_absent_normal_and_unreadable(monkeypatch, tmp_path):
    assert doctor.check_prompt_cache(tmp_path / "absent").status == doctor.OK
    path = tmp_path / "server.log"
    path.write_text("prompt cache enabled")
    assert doctor.check_prompt_cache(path).detail == "no prompt-cache warnings in the server log."
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError("denied")))
    assert doctor.check_prompt_cache(path).detail == "could not read server log."


@pytest.mark.parametrize("selected", [True, False])
def test_all_checks_pass_ram_requirement_and_selected_uuid(monkeypatch, selected):
    monkeypatch.setattr(probe, "cpu_info", lambda: {})
    monkeypatch.setattr(probe, "mem_info", lambda: {"available": 100})
    monkeypatch.setattr(probe, "gpu_info", lambda: [{"index": 2, "uuid": "GPU-two"}] if selected else [])
    users = []
    monkeypatch.setattr(lock, "foreign_gpu_users", lambda **kw: users.append(kw) or [])
    names = ["llama_server", "isa", "gpu_free", "swap", "dimm_population", "governor", "thp",
             "stale_autotune", "k_cache_quant", "prompt_cache"]
    for name in names:
        monkeypatch.setattr(doctor, "check_"+name, lambda *a, name=name: doctor.Check(name, "ok", "mock"))
    checks = doctor.run_all(ram_need=50, gpu_index=2)
    assert [ch.name for ch in checks] == names + ["ram-capacity"]
    assert users == ([{"gpu_uuid": "GPU-two"}] if selected else [])
    assert len(doctor.run_all()) == 10
