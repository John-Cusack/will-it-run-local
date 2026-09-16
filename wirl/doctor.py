"""Pre-flight checks.

These are not generic system hygiene. Each one corresponds to a specific
failure that cost real time on the reference machine, in the order of how much
time it cost. Most of them are invisible in normal operation: the model still
loads, still answers, and is simply slower or wrong than it should be.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _gib(n):
    return n / (1 << 30)


def check_swap(mem) -> Check:
    used = mem["swap_used"]
    if used > 4 << 30:
        return Check("swap", FAIL,
                     f"{_gib(used):.1f} GiB of swap in use.",
                     "A model whose weights get swapped decodes at disk speed. "
                     "Free memory or `sudo swapoff -a && sudo swapon -a` before "
                     "measuring anything; a first run after swapping shows "
                     "wildly wrong throughput until it drains.")
    if used > 512 << 20:
        return Check("swap", WARN, f"{_gib(used):.2f} GiB of swap in use.",
                     "Small, but re-check after loading the model.")
    return Check("swap", OK, "no meaningful swap in use.")


def check_dimm_population(mem) -> Check:
    """A missing DIMM costs bandwidth silently and is never reported anywhere.

    EDAC reports the capacity the memory controller was told about; MemTotal
    reports what the kernel actually has. A large gap means a module or a whole
    channel is not being used, and decode speed scales with populated channels.
    """
    edac, total = mem["edac_total"], mem["total"]
    if not edac:
        return Check("dimm-population", OK, "no EDAC data (not an error).")
    gap = edac - total
    # The kernel always reserves some memory, so a few percent is normal.
    if gap > max(8 << 30, int(0.10 * edac)):
        return Check("dimm-population", WARN,
                     f"EDAC reports {_gib(edac):.0f} GiB installed but the kernel "
                     f"sees {_gib(total):.0f} GiB -- a {_gib(gap):.0f} GiB gap.",
                     "That gap is about the size of one or more DIMMs. If a "
                     "memory channel is unpopulated you lose bandwidth in "
                     "proportion, and no software setting recovers it. Confirm "
                     "with `sudo dmidecode -t memory` and compare the number of "
                     "populated slots against your CPU's channel count.")
    return Check("dimm-population", OK,
                 f"EDAC {_gib(edac):.0f} GiB vs kernel {_gib(total):.0f} GiB: consistent.")


def check_gpu_free(gpus, foreign, gpu_index=0) -> Check:
    if not gpus:
        return Check("gpu", WARN, "no NVIDIA GPU detected.",
                     "CPU-only inference on a large MoE is roughly 4x slower "
                     "than a hybrid split. Expect single-digit tokens/sec.")
    g = next((g for i, g in enumerate(gpus) if g.get("index", i) == gpu_index), None)
    if g is None:
        return Check("gpu", FAIL, f"GPU index {gpu_index} is not available.")
    if foreign:
        who = ", ".join(f"{p['name']} (pid {p['pid']}, {p['vram_mb']} MB)"
                        for p in foreign)
        return Check("gpu-exclusive", FAIL,
                     f"other processes are using {g['name']} (GPU {gpu_index}): {who}",
                     "Stop them before measuring. VRAM fitting will be wrong "
                     "and timings will be contended.")
    return Check("gpu-exclusive", OK,
                 f"{g['name']}, {_gib(g['vram_total']):.0f} GiB, "
                 f"{_gib(g['vram_free']):.1f} GiB free, driver {g['driver']}.")


def check_governor(cpu) -> Check:
    gov = cpu.get("governor")
    if gov is None:
        return Check("cpu-governor", OK, "no cpufreq governor exposed.")
    if gov in ("powersave", "conservative"):
        return Check("cpu-governor", WARN, f"governor is '{gov}'.",
                     "Prefill and prompt processing suffer. "
                     "`sudo cpupower frequency-set -g performance` if it matters; "
                     "decode on a bandwidth-bound model barely notices.")
    return Check("cpu-governor", OK, f"governor is '{gov}'.")


def check_isa(cpu) -> Check:
    isa = cpu["isa"]
    if not isa["avx2"]:
        return Check("cpu-isa", FAIL, "no AVX2.",
                     "llama.cpp will be very slow. This machine is not suited "
                     "to CPU inference of large models.")
    bits = [k for k in ("avx2", "fma", "avx512f", "avx512_bf16", "amx_int8") if isa[k]]
    detail = f"{cpu['model']}: {', '.join(bits)}."
    if not isa["avx512f"]:
        return Check("cpu-isa", OK, detail + " No AVX-512 -- expected on Zen 2 "
                     "and earlier; not a problem for bandwidth-bound decode.")
    return Check("cpu-isa", OK, detail)


def check_k_cache_quant(cfg_k: str, cpu) -> Check:
    """Quantised K cache is a well-known correctness hazard on some builds.

    On the reference machine `--cache-type-k q8_0` produced garbage output
    (llama.cpp issue #25382) while V quantisation was fine. It fails silently:
    the server starts, tokens flow, and the text is wrong.
    """
    if cfg_k not in ("f16", "bf16", "f32"):
        return Check("kv-cache-type", WARN,
                     f"--cache-type-k is '{cfg_k}'.",
                     "Quantising the K cache has produced corrupt output on "
                     "some architectures (llama.cpp #25382) and fails silently. "
                     "Verify output quality explicitly, or leave K at f16 and "
                     "quantise V only.")
    return Check("kv-cache-type", OK, f"K cache is {cfg_k}.")


def check_thp() -> Check:
    p = "/sys/kernel/mm/transparent_hugepage/enabled"
    if not os.path.exists(p):
        return Check("transparent-hugepages", OK, "not exposed.")
    with open(p) as f:
        val = f.read().strip()
    cur = val[val.find("[") + 1:val.find("]")] if "[" in val else val
    if cur == "always":
        return Check("transparent-hugepages", WARN, "THP is 'always'.",
                     "With a >100 GB mmap'd model, khugepaged compaction can "
                     "cause multi-second stalls. 'madvise' is usually calmer.")
    return Check("transparent-hugepages", OK, f"THP is '{cur}'.")


def check_stale_autotune() -> Check:
    """Kernel autotuning caches persist bad choices made under contention.

    Triton and torchinductor time candidate kernels and write the winner to
    disk. If that timing happened while the machine was busy, the cache holds a
    slow kernel permanently -- across restarts and reboots. This cost 38%
    throughput on the reference machine and survived every other fix.
    """
    hits = []
    for d in ("~/.triton/cache", "~/.cache/torchinductor",
              "/tmp/torchinductor_" + (os.environ.get("USER") or "root")):
        p = os.path.expanduser(d)
        if os.path.isdir(p):
            n = sum(len(files) for _, _, files in os.walk(p))
            if n:
                hits.append(f"{p} ({n} files)")
    if hits:
        return Check("autotune-cache", WARN,
                     "kernel autotune caches present: " + "; ".join(hits),
                     "If throughput regressed for no clear reason, move these "
                     "aside (do not delete -- rename, so you can compare) and "
                     "re-measure. A cache written while the machine was busy "
                     "pins a slow kernel indefinitely.")
    return Check("autotune-cache", OK, "no kernel autotune caches found.")


def check_ram_for_model(mem, model_bytes, gpu_bytes) -> Check:
    need = max(0, model_bytes - gpu_bytes)
    avail = mem["available"]
    if need > avail:
        return Check("ram-capacity", FAIL,
                     f"needs ~{_gib(need):.0f} GiB in RAM but only "
                     f"{_gib(avail):.0f} GiB is available.",
                     "It will swap and be unusably slow. Use a smaller "
                     "quantisation or offload more to the GPU.")
    if need > 0.9 * avail:
        return Check("ram-capacity", WARN,
                     f"needs ~{_gib(need):.0f} GiB of {_gib(avail):.0f} GiB available.",
                     "Very little margin. The page cache will thrash.")
    return Check("ram-capacity", OK,
                 f"~{_gib(need):.0f} GiB needed in RAM, {_gib(avail):.0f} GiB available.")


def check_disk(path, need_bytes) -> Check:
    try:
        free = shutil.disk_usage(path).free
    except OSError:
        return Check("disk", OK, "could not stat path.")
    if need_bytes and free < need_bytes:
        return Check("disk", FAIL,
                     f"{_gib(free):.0f} GiB free at {path}, need {_gib(need_bytes):.0f} GiB.",
                     "Download will fail partway.")
    return Check("disk", OK, f"{_gib(free):.0f} GiB free at {path}.")


def check_llama_server(explicit=None) -> Check:
    """Is there a usable llama-server, and was it built with CUDA?

    The most likely first failure for anyone cloning this: no binary, or one
    built CPU-only, which produces a confusing error deep inside a sweep
    instead of a clear message up front.
    """
    from .runner import find_server
    binary = find_server(explicit)
    if not binary:
        return Check("llama.cpp", FAIL, "no llama-server binary found.",
                     "Build llama.cpp with CUDA, then either put llama-server "
                     "on PATH, set WIRL_LLAMA_SERVER=/path/to/llama-server, or "
                     "pass --llama-server. Build with: cmake -B build "
                     "-DGGML_CUDA=ON && cmake --build build -j --target "
                     "llama-server")

    # llama-server writes its banner to stderr, so both streams are needed.
    blob = ""
    try:
        p = subprocess.run([binary, "--version"], capture_output=True, text=True,
                           timeout=30)
        blob = (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError):
        pass
    version = ""
    for line in blob.splitlines():
        if "version" in line.lower() or "build" in line.lower():
            version = line.strip()
            break

    # llama-server prints its detected backends on startup, not for --version,
    # so fall back to checking that a CUDA backend library was linked or built.
    cuda = ("CUDA" in blob or "cuda" in blob)
    if not cuda:
        d = os.path.dirname(binary)
        for probe in ("libggml-cuda.so", "../lib/libggml-cuda.so"):
            if os.path.exists(os.path.join(d, probe)):
                cuda = True
                break
        if not cuda:
            # A statically linked CUDA build leaves no .so; look for the symbol.
            try:
                ldd = subprocess.run(["ldd", binary], capture_output=True,
                                     text=True, timeout=20).stdout or ""
            except (OSError, subprocess.SubprocessError):
                ldd = ""
            cuda = "cuda" in ldd.lower()

    detail = f"{binary}" + (f" ({version})" if version else "")
    if not cuda:
        return Check("llama.cpp", WARN, detail + " -- could not confirm CUDA support.",
                     "If this binary is CPU-only, every GPU offload flag is "
                     "silently ignored and the sweep will measure nothing "
                     "useful. Rebuild with -DGGML_CUDA=ON, or continue if you "
                     "know it is a GPU build.")
    return Check("llama.cpp", OK, detail + ", CUDA support detected.")


def check_prompt_cache(server_log=None) -> Check:
    """Prompt caching is the difference between a usable chat and a slow one.

    It is on by default, so this mostly exists to surface the one silent
    failure: `--cache-reuse` is disabled without error on any model whose
    context cannot KV-shift (sliding-window attention, some latent-cache
    designs). If you set it and never read the log, you will believe it is
    working.
    """
    if not server_log or not os.path.exists(server_log):
        return Check("prompt-cache", OK,
                     "prompt caching is on by default (--cache-prompt).",
                     "")
    try:
        with open(server_log, errors="replace") as f:
            blob = f.read()
    except OSError:
        return Check("prompt-cache", OK, "could not read server log.")
    if "cache_reuse is not supported by this context" in blob:
        return Check("prompt-cache", WARN,
                     "--cache-reuse was requested but llama.cpp disabled it.",
                     "This context cannot KV-shift, so the flag does nothing. "
                     "Plain prefix caching still works. To keep several "
                     "conversations warm, raise --cache-ram (default 8192 MiB) "
                     "instead.")
    return Check("prompt-cache", OK, "no prompt-cache warnings in the server log.")


def run_all(model_bytes=None, gpu_bytes=0, cache_type_k="f16", path=".",
            llama_server=None, server_log=None, gpu_index=0) -> list:
    from .lock import foreign_gpu_users
    from .probe import cpu_info, gpu_info, mem_info

    cpu, mem, gpus = cpu_info(), mem_info(), gpu_info()
    selected = next((g for g in gpus if g["index"] == gpu_index), None)
    foreign = foreign_gpu_users(gpu_uuid=selected["uuid"]) if selected else []
    checks = [
        check_llama_server(llama_server),
        check_isa(cpu),
        check_gpu_free(gpus, foreign, gpu_index),
        check_swap(mem),
        check_dimm_population(mem),
        check_governor(cpu),
        check_thp(),
        check_stale_autotune(),
        check_k_cache_quant(cache_type_k, cpu),
        check_prompt_cache(server_log),
    ]
    if model_bytes:
        checks.append(check_ram_for_model(mem, model_bytes, gpu_bytes))
        checks.append(check_disk(path, 0))
    return checks
