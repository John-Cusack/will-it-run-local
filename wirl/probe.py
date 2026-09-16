"""Hardware inventory.

Deliberately reports *measured* or *reported-by-the-kernel* values and never
nameplate specifications. A CPU's advertised memory bandwidth is a theoretical
peak that assumes every channel is populated and running at its rated clock;
real machines routinely deliver a fraction of it, and the gap is exactly what
determines whether a model is usable.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess


def _read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def parse_cpulist(text: str) -> list:
    cpus = set()
    for part in text.split(",") if text.strip() else []:
        ends = part.strip().split("-")
        start, end = int(ends[0]), int(ends[-1])
        if len(ends) > 2 or start < 0 or end < start:
            raise ValueError(f"invalid CPU list: {text}")
        cpus.update(range(start, end + 1))
    return sorted(cpus)


def format_cpulist(cpus) -> str:
    ordered = sorted(set(cpus))
    ranges = []
    i = 0
    while i < len(ordered):
        start = end = ordered[i]
        i += 1
        while i < len(ordered) and ordered[i] == end + 1:
            end = ordered[i]
            i += 1
        ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges)


def physical_core_cpus() -> list | None:
    """One allowed logical CPU per core, without guessing sibling numbering."""
    try:
        allowed = sorted(os.sched_getaffinity(0))
        seen, cpus = set(), []
        for cpu in allowed:
            base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
            text = _read(f"{base}/core_cpus_list")
            if text is None:
                text = _read(f"{base}/thread_siblings_list")
            if text is None:
                return None
            siblings = set(parse_cpulist(text))
            if cpu not in siblings:
                return None
            if cpu not in seen:
                cpus.append(cpu)
                seen.update(siblings)
        return cpus
    except (AttributeError, OSError, ValueError):
        return None


def cpu_info() -> dict:
    """Physical vs logical cores, ISA features, NUMA topology."""
    info = {"model": None, "logical": os.cpu_count() or 0, "physical": None,
            "sockets": None, "numa_nodes": 1, "flags": set()}

    txt = _read("/proc/cpuinfo", "")
    pairs = set()
    phys_ids = set()
    cur_phys = cur_core = None
    for line in txt.splitlines():
        if line.startswith("model name") and not info["model"]:
            info["model"] = line.split(":", 1)[1].strip()
        elif line.startswith("physical id"):
            cur_phys = line.split(":", 1)[1].strip()
            phys_ids.add(cur_phys)
        elif line.startswith("core id"):
            cur_core = line.split(":", 1)[1].strip()
            if cur_phys is not None:
                pairs.add((cur_phys, cur_core))
        elif line.startswith("flags") and not info["flags"]:
            info["flags"] = set(line.split(":", 1)[1].split())
    if pairs:
        info["physical"] = len(pairs)
    info["sockets"] = len(phys_ids) or 1
    if not info["physical"]:
        info["physical"] = info["logical"]

    nodes = [d for d in os.listdir("/sys/devices/system/node")
             if re.fullmatch(r"node\d+", d)] if os.path.isdir("/sys/devices/system/node") else []
    info["numa_nodes"] = len(nodes) or 1

    f = info["flags"]
    info["isa"] = {
        "avx2": "avx2" in f,
        "fma": "fma" in f,
        "avx512f": "avx512f" in f,
        "avx512_bf16": "avx512_bf16" in f,
        "amx_int8": "amx_int8" in f,
    }
    gov = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["governor"] = gov
    return info


def mem_info() -> dict:
    mi = {}
    for line in _read("/proc/meminfo", "").splitlines():
        k, _, v = line.partition(":")
        mi[k] = int(v.split()[0]) * 1024 if v.split() else 0
    out = {
        "total": mi.get("MemTotal", 0),
        "free": mi.get("MemFree", 0),
        "available": mi.get("MemAvailable", 0),
        "swap_total": mi.get("SwapTotal", 0),
        "swap_free": mi.get("SwapFree", 0),
    }
    out["swap_used"] = out["swap_total"] - out["swap_free"]

    # EDAC knows the physically installed capacity. A large gap against
    # MemTotal means a DIMM or an entire channel is not being seen -- which
    # costs bandwidth proportional to the missing channels, silently.
    edac = 0
    base = "/sys/devices/system/edac/mc"
    if os.path.isdir(base):
        for mc in os.listdir(base):
            v = _read(os.path.join(base, mc, "size_mb"))
            if v and v.isdigit():
                edac += int(v) * 1024 * 1024
    out["edac_total"] = edac
    return out


def gpu_info() -> list:
    """Per-GPU state via nvidia-smi. Empty list if there is no NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return []
    fields = ("index,name,memory.total,memory.used,memory.free,compute_cap,"
              "driver_version,pcie.link.gen.current,pcie.link.width.current,"
              "clocks.max.memory,power.limit,uuid")
    out = _run(["nvidia-smi", f"--query-gpu={fields}",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 12:
            continue

        def num(x, cast=float):
            try:
                return cast(x)
            except ValueError:
                return None
        gpus.append({
            # Byte counts are ints: they get shifted and formatted as integers
            # all over the codebase, and nvidia-smi hands back decimal strings.
            "index": num(p[0], int), "name": p[1],
            "vram_total": int((num(p[2]) or 0) * 1024 * 1024),
            "vram_used": int((num(p[3]) or 0) * 1024 * 1024),
            "vram_free": int((num(p[4]) or 0) * 1024 * 1024),
            "compute_cap": p[5], "driver": p[6],
            "pcie_gen": num(p[7], int), "pcie_width": num(p[8], int),
            "mem_clock_mhz": num(p[9]), "power_limit_w": num(p[10]),
            "uuid": p[11],
        })
    return gpus


def gpu_processes() -> list:
    """Anything already using the GPU. Benchmarking next to these is invalid."""
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                "--format=csv,noheader,nounits"])
    if not out or not out.strip():
        return []
    procs = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            procs.append({"pid": p[0], "name": p[1], "vram_mb": p[2], "gpu_uuid": p[3]})
    return procs


def gpu_bandwidth_hint(gpu: dict) -> float | None:
    """Rough VRAM bandwidth in bytes/s, used only to price the GPU side.

    The GPU half of a hybrid split is never the bottleneck by a wide margin, so
    an approximate figure is sufficient here; the CPU side is measured properly.
    """
    name = (gpu.get("name") or "").lower()
    table = {  # GB/s
        "3060": 360, "3070": 448, "3080": 760, "3090": 936,
        "4060": 272, "4070": 504, "4080": 717, "4090": 1008,
        "5070": 672, "5080": 960, "5090": 1792,
        "a4000": 448, "a5000": 768, "a6000": 768,
        "a100": 1555, "h100": 3350, "l40": 864, "v100": 900,
        "titan rtx": 672, "2080": 448,
    }
    for k, v in table.items():
        if k in name:
            return v * 1e9
    return None


def swap_activity() -> dict:
    """Cumulative page-in/page-out counters, for before/after comparison."""
    out = {"pswpin": 0, "pswpout": 0}
    for line in _read("/proc/vmstat", "").splitlines():
        k, _, v = line.partition(" ")
        if k in out:
            out[k] = int(v)
    return out


def probe() -> dict:
    return {"cpu": cpu_info(), "mem": mem_info(),
            "gpus": gpu_info(), "gpu_procs": gpu_processes()}
