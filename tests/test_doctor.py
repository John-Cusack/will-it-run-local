"""Pre-flight checks. Each corresponds to a specific silent failure."""
from wirl import doctor

GiB = 1 << 30


def _mem(**over):
    m = {"total": 216 * GiB, "available": 200 * GiB, "swap_total": 8 * GiB,
         "swap_free": 8 * GiB, "swap_used": 0, "edac_total": 0}
    m.update(over)
    return m


def test_heavy_swap_use_is_blocking():
    c = doctor.check_swap(_mem(swap_used=6 * GiB))
    assert c.status == doctor.FAIL


def test_no_swap_use_is_ok():
    assert doctor.check_swap(_mem()).status == doctor.OK


def test_missing_dimm_is_detected():
    """EDAC sees 256 GiB installed, the kernel sees 216: one DIMM is dark.

    This is the check that found a dead memory channel on the reference
    machine after it had cost hours of misattributed benchmarking.
    """
    c = doctor.check_dimm_population(_mem(edac_total=256 * GiB, total=216 * GiB))
    assert c.status == doctor.WARN
    assert "40 GiB gap" in c.detail


def test_normal_kernel_reservation_is_not_flagged():
    c = doctor.check_dimm_population(_mem(edac_total=256 * GiB, total=254 * GiB))
    assert c.status == doctor.OK


def test_no_edac_data_is_not_an_error():
    assert doctor.check_dimm_population(_mem()).status == doctor.OK


def test_busy_gpu_is_blocking():
    gpus = [{"name": "RTX 3090", "vram_total": 24 * GiB, "vram_free": 2 * GiB,
             "driver": "595.84"}]
    busy = [{"pid": 1, "name": "llama-server", "vram_mb": 21090}]
    assert doctor.check_gpu_free(gpus, busy).status == doctor.FAIL
    assert doctor.check_gpu_free(gpus, []).status == doctor.OK


def test_quantised_k_cache_is_flagged():
    """--cache-type-k q8_0 produced corrupt output on the reference machine
    (llama.cpp #25382) while the server appeared entirely healthy."""
    cpu = {"isa": {}, "model": "x"}
    assert doctor.check_k_cache_quant("q8_0", cpu).status == doctor.WARN
    assert doctor.check_k_cache_quant("f16", cpu).status == doctor.OK


def test_missing_avx2_is_blocking():
    cpu = {"model": "old", "isa": {"avx2": False, "fma": False, "avx512f": False,
                                   "avx512_bf16": False, "amx_int8": False}}
    assert doctor.check_isa(cpu).status == doctor.FAIL


def test_zen2_without_avx512_is_fine():
    cpu = {"model": "EPYC 7B12", "isa": {"avx2": True, "fma": True,
                                         "avx512f": False, "avx512_bf16": False,
                                         "amx_int8": False}}
    c = doctor.check_isa(cpu)
    assert c.status == doctor.OK
    assert "No AVX-512" in c.detail


def test_model_larger_than_ram_is_blocking():
    c = doctor.check_ram_for_model(_mem(available=100 * GiB), 156 * GiB, 0)
    assert c.status == doctor.FAIL


def test_offloading_to_gpu_can_make_it_fit():
    c = doctor.check_ram_for_model(_mem(available=100 * GiB), 156 * GiB, 80 * GiB)
    assert c.status == doctor.OK


def test_missing_llama_server_is_blocking(monkeypatch):
    """The most likely first failure for anyone cloning the repo."""
    import wirl.runner
    monkeypatch.setattr(wirl.runner, "find_server", lambda e=None: None)
    c = doctor.check_llama_server()
    assert c.status == doctor.FAIL
    assert "GGML_CUDA=ON" in c.fix


def test_llama_server_found_is_reported(monkeypatch):
    import wirl.runner
    monkeypatch.setattr(wirl.runner, "find_server", lambda e=None: "/bin/true")
    c = doctor.check_llama_server()
    assert c.status in (doctor.OK, doctor.WARN)
    assert "/bin/true" in c.detail


def test_silently_disabled_cache_reuse_is_surfaced(tmp_path):
    """llama.cpp disables --cache-reuse without an error on models whose
    context cannot KV-shift. Set it, never read the log, and you will believe
    it is working."""
    log = tmp_path / "srv.log"
    log.write_text("W srv load_model: cache_reuse is not supported by this "
                   "context, it will be disabled\n")
    c = doctor.check_prompt_cache(str(log))
    assert c.status == doctor.WARN
    assert "--cache-ram" in c.fix


def test_clean_server_log_passes(tmp_path):
    log = tmp_path / "srv.log"
    log.write_text("srv load_model: loaded\n")
    assert doctor.check_prompt_cache(str(log)).status == doctor.OK


def test_no_log_is_not_an_error():
    assert doctor.check_prompt_cache(None).status == doctor.OK


def test_gpu_check_describes_selected_card():
    gpus = [{"index": i, "name": f"card {i}", "vram_total": 24 * GiB,
             "vram_free": 20 * GiB, "driver": "test"} for i in range(2)]
    check = doctor.check_gpu_free(gpus, [], gpu_index=1)
    assert "card 1" in check.detail and "card 0" not in check.detail


def test_doctor_filters_contention_to_selected_gpu(monkeypatch):
    from wirl import lock, probe
    gpus = [{"index": i, "uuid": f"GPU-{i}", "name": f"card {i}",
             "vram_total": 24 * GiB, "vram_free": 20 * GiB, "driver": "test"}
            for i in range(2)]
    monkeypatch.setattr(probe, "gpu_info", lambda: gpus)
    monkeypatch.setattr(probe, "cpu_info", lambda: {})
    monkeypatch.setattr(probe, "mem_info", _mem)
    for name in ("check_llama_server", "check_isa", "check_swap", "check_dimm_population",
                 "check_governor", "check_thp", "check_stale_autotune", "check_k_cache_quant",
                 "check_prompt_cache"):
        monkeypatch.setattr(doctor, name, lambda *a: doctor.Check("mock", doctor.OK, ""))
    seen = []
    monkeypatch.setattr(lock, "foreign_gpu_users", lambda **kw: seen.append(kw) or [])
    checks = doctor.run_all(gpu_index=1)
    assert seen == [{"gpu_uuid": "GPU-1"}]
    assert "card 1" in next(c.detail for c in checks if c.name == "gpu-exclusive")
