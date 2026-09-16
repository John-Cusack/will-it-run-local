"""Command behaviour on a fake machine; no servers, downloads or benchmarks."""
from contextlib import contextmanager
import copy
import runpy
import sys
from types import SimpleNamespace

import pytest

from wirl import cli, compat, doctor, drafters, gguf, membw, model, predict, probe, recommend, runner, search, tune

pytestmark = pytest.mark.usefixtures("isolated_runtime")


@pytest.fixture
def machine(monkeypatch, tmp_path, moe_model):
    monkeypatch.chdir(tmp_path)
    g = gguf.read(moe_model)
    g.kv.update({"_wirl.shards": 2, "toymoe.context_length": 32768})
    cpu = dict(model="Fake CPU", physical=8, logical=16, sockets=1, numa_nodes=2,
               isa=dict(avx2=True, fma=True), governor="performance")
    mem = dict(total=256<<30, available=200<<30, swap_used=0, swap_total=8<<30, edac_total=256<<30)
    gpu = dict(index=0, uuid="GPU-zero", name="RTX 3090", vram_total=24<<30,
               vram_free=20<<30, compute_cap="8.6", driver="fake", pcie_gen=4, pcie_width=16)
    state = SimpleNamespace(g=g, mc=model.build(g), cpu=cpu, mem=mem,
                            gpus=[gpu, dict(gpu, index=1, name="Unknown", uuid="GPU-one")],
                            calls=[], bw_calls=[], locks=[], busy_at=None, edge=1,
                            failure=False, ctx_failure=False, server_failure=False,
                            samples=[10, 10], checks=[], root=tmp_path, profile=[])
    monkeypatch.setattr(cli, "_load", lambda *a: (state.g, state.mc))
    monkeypatch.setattr(cli, "find_server", lambda *a: "fake-server")
    monkeypatch.setattr(probe, "cpu_info", lambda: state.cpu)
    monkeypatch.setattr(probe, "mem_info", lambda: state.mem)
    monkeypatch.setattr(probe, "gpu_info", lambda: state.gpus)
    monkeypatch.setattr(probe, "gpu_processes", lambda: [])
    monkeypatch.setattr(probe, "physical_core_cpus", lambda: list(range(0, 16, 2)))
    monkeypatch.setattr(cli, "foreign_gpu_users", lambda **kw: [])
    monkeypatch.setattr(doctor, "run_all", lambda **kw: state.checks)

    def bandwidth(mode, **kw):
        state.bw_calls.append((mode, kw))
        return membw.BwResult(mode, 8, 2, state.samples, "fake probe")

    monkeypatch.setattr(membw, "measure", bandwidth)

    @contextmanager
    def locked(wait=False):
        state.locks.append(wait)
        if state.busy_at == len(state.locks):
            raise cli.BenchmarkBusy("fake busy lock")
        yield

    @contextmanager
    def server(*a, **kw):
        if state.server_failure:
            raise runner.ServerFailed("fake server failure")
        yield 1

    def run(cfg, *a, **kw):
        state.calls.append((copy.copy(cfg), kw))
        failed = state.failure or (state.ctx_failure and cfg.ctx == 32768)
        return runner.RunResult(cfg, [] if failed else list(state.samples), [], 100, 1,
                                not failed, "fake failure" if failed else None)

    monkeypatch.setattr(cli, "benchmark_lock", locked)
    monkeypatch.setattr(runner, "server", server)
    for module in (runner, search, tune):
        monkeypatch.setattr(module, "run_config", run)
    for name in ("find_vram_edge", "find_ngl_edge"):
        monkeypatch.setattr(search, name, lambda *a, **kw: state.edge)
    monkeypatch.setattr(search, "profile_depth", lambda *a, **kw: state.profile)
    return state


def dense(machine):
    machine.g.kv["toymoe.expert_count"] = 0
    machine.g.kv["toymoe.expert_used_count"] = 0
    machine.mc = model.build(machine.g)


def test_load_reports_missing_invalid_and_valid_files(monkeypatch, tmp_path, moe_model):
    assert cli._load(moe_model)[1].is_moe
    with pytest.raises(SystemExit, match="error:"):
        cli._load(str(tmp_path / "missing"))
    path = tmp_path / "invalid"
    path.write_bytes(b"NOPE")
    with pytest.raises(SystemExit, match="error reading draft model"):
        cli._load(str(path), "draft model")


@pytest.mark.parametrize("gpu", [0, 1, 2])
def test_gpu_choice_and_missing_index(machine, gpu):
    args = SimpleNamespace(gpu=gpu, vram=12)
    if gpu == 2:
        with pytest.raises(SystemExit, match="available GPUs: 0: RTX 3090, 1: Unknown"):
            cli._gpu_choice(args)
    else:
        selected, budget, bw = cli._gpu_choice(args)
        assert selected["index"] == gpu and budget == 12<<30
        assert bw == (936e9 if gpu == 0 else 500e9)


@pytest.mark.parametrize("samples", [[10, 10], [5, 15]])
def test_bandwidth_helper_and_command_warn_on_spread(machine, samples, capsys):
    machine.samples = samples
    bw, description = cli._bandwidth(SimpleNamespace(mem_bandwidth=None, bw_reps=2))
    assert bw == max(samples)*1e9 and "measured" in description
    assert cli.main(["bandwidth", "--threads", "2", "--gib", "1"]) == 0
    text = capsys.readouterr().out
    assert ("unstable:" in text) == (samples[0] != samples[1])
    assert ("Repetitions disagree" in text) == (samples[0] != samples[1])
    assert [mode for mode, kw in machine.bw_calls] == ["stream", "stream", "gather"]


@pytest.mark.parametrize("present,measure", [(True, True), (True, False), (False, False)])
def test_probe_output_and_optional_bandwidth(machine, present, measure, capsys):
    if not present:
        machine.gpus.clear()
        machine.mem["edac_total"] = 0
        machine.cpu["isa"] = {}
    assert cli.main(["probe"] + ([] if measure else ["--no-bandwidth"])) == 0
    text = capsys.readouterr().out
    assert "Fake CPU" in text and "swap:" in text
    assert ("EDAC reports" in text) == present
    assert ("none detected (CPU-only" in text) == (not present)
    assert bool(machine.bw_calls) == measure


@pytest.mark.parametrize("moe,measure", [(True, True), (False, False)])
def test_inspect_dense_and_moe_metadata(machine, moe, measure, capsys):
    if not moe:
        dense(machine)
        machine.g.kv.pop("_wirl.shards")
        machine.g.kv.pop("toymoe.context_length")
    assert cli.main(["inspect", "model"] + ([] if measure else ["--no-bandwidth"])) == 0
    text = capsys.readouterr().out
    assert "bytes per token" in text and "KV cache" in text
    assert ("experts        " in text) == moe
    assert ("across 2 shards" in text) == moe
    assert ("this machine (measured)" in text) == measure


@pytest.mark.parametrize("moe,draft", [(True, True), (False, True), (False, False)])
def test_plan_selects_model_knob_and_explains_draft_limits(machine, moe, draft, capsys):
    if not moe:
        dense(machine)
    argv = ["plan", "model", "--mem-bandwidth", "20"] + (["--draft", "draft"] if draft else [])
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert ("--n-cpu-moe" if moe else "--n-gpu-layers") in text and "Start here" in text
    assert ("draft-model VRAM is not modelled" in text) == (draft and not moe)


@pytest.mark.parametrize("command", ["plan", "auto"])
def test_incompatible_draft_stops_before_hardware(machine, monkeypatch, command, capsys):
    monkeypatch.setattr(compat, "compare", lambda *a: {"compatible": False, "problems": ["vocab mismatch"]})
    argv = [command, "model", "--draft", "draft"]
    if command == "plan":
        with pytest.raises(SystemExit) as exit:
            cli.main(argv)
        assert exit.value.code == 1
    else:
        assert cli.main(argv) == 1
    assert "vocab mismatch" in capsys.readouterr().out and not machine.bw_calls and not machine.calls


@pytest.mark.parametrize("fits", [True, False])
def test_plan_limits_large_tables_and_handles_no_prediction(machine, monkeypatch, fits, capsys):
    points = [predict.Prediction(n, 0, 0, 1, fits, 1, 0, 0) for n in range(20)]
    monkeypatch.setattr(predict, "curve", lambda *a, **kw: points)
    monkeypatch.setattr(predict, "best_fit", lambda *a, **kw: (points[0] if fits else None, points))
    assert cli.main(["plan", "model", "--mem-bandwidth", "20"]) == 0
    text = capsys.readouterr().out
    assert ("recommended" in text) == fits
    assert ("DOES NOT FIT" in text) == (not fits)
    assert len([line for line in text.splitlines() if "GiB" in line and ("NO" in line or "yes" in line)]) == (13 if fits else 7)


@pytest.mark.parametrize("command", ["auto", "tune"])
def test_missing_server_is_clear(machine, monkeypatch, command):
    monkeypatch.setattr(cli, "find_server", lambda *a: None)
    with pytest.raises(SystemExit, match="could not find llama-server"):
        cli.main([command, "model"])


def test_auto_blocked_preflight_and_missing_gpu(machine, capsys):
    machine.checks = [doctor.Check("swap", "fail", "busy")]
    assert cli.main(["auto", "model", "--mem-bandwidth", "20"]) == 1
    assert "Refusing to sweep" in capsys.readouterr().out and not machine.calls
    machine.gpus.clear()
    with pytest.raises(SystemExit, match="no GPU detected"):
        cli.main(["auto", "model"])


@pytest.mark.parametrize("moe,ctx_failure", [(True, False), (False, True)])
def test_auto_sweeps_contexts_and_emits_measured_launcher(machine, moe, ctx_failure, capsys):
    if not moe:
        dense(machine)
    machine.ctx_failure = ctx_failure
    machine.samples = [8, 12]
    machine.checks = [doctor.Check("swap", "fail", "busy")]
    machine.profile = [search.DepthPoint(0, 10, 5, 10, 2)]
    out = machine.root / "auto.sh"
    assert cli.main(["auto", "model", "--draft", "draft", "--depth-sweep", "--thread-sweep",
                     "--ctx2", "32768", "--mem-bandwidth", "20", "--emit", str(out), "--force"]) == 0
    text = capsys.readouterr().out
    assert "Same configuration, separate launches" in text and "Unstable measurements" in text
    assert "Long-context behaviour" in text and "VRAM boundary" in out.read_text()
    assert out.with_suffix(".sh.service").exists()
    assert ("does not run at ctx" in text) == ctx_failure
    counts = [cfg.threads for cfg, kw in machine.calls if cfg.threads]
    assert counts == [3, 4, 6, 8]


@pytest.mark.parametrize("failure", ["edge", "measure", "lock"])
def test_auto_boundary_measurement_and_lock_failures(machine, failure, capsys):
    machine.edge = None if failure == "edge" else 1
    machine.failure = failure == "measure"
    machine.busy_at = 1 if failure == "lock" else None
    if failure == "edge":
        assert cli.main(["auto", "model", "--mem-bandwidth", "20"]) == 1
        assert "Nothing in that range started" in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit, match="no configuration completed" if failure == "measure" else "fake busy lock"):
            cli.main(["auto", "model", "--draft", "draft", "--depth-sweep", "--thread-sweep", "--mem-bandwidth", "20"])


@pytest.mark.parametrize("mode", ["depth-lock", "depth-server", "context-lock", "no-prediction"])
def test_auto_skipped_followup_checks_and_missing_prediction(machine, monkeypatch, mode, capsys):
    machine.busy_at = 2 if mode in ("depth-lock", "context-lock") else None
    machine.server_failure = mode == "depth-server"
    if mode == "no-prediction":
        dense(machine)
        monkeypatch.setattr(predict, "best_fit_dense", lambda *a, **kw: (None, []))
    argv = ["auto", "model", "--mem-bandwidth", "20", "--ctx2", "32768"]
    if mode == "context-lock":
        argv.append("--no-depth")
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert ("skipped:" in text) == (mode != "no-prediction")
    assert (machine.root / "run-llama.sh").exists()


@pytest.mark.parametrize("moe,emit_script", [(True, True), (False, False)])
def test_tune_emits_winner_and_stops_failed_draft_sweep(machine, moe, emit_script, capsys):
    if not moe:
        dense(machine)
    machine.samples = [8, 12]
    out = machine.root / "tune.sh"
    argv = ["tune", "model", "--draft", "draft", "--depth-sweep", "--mem-bandwidth", "20"]
    if emit_script:
        argv += ["--emit", str(out)]
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert "Winner" in text and "Unstable measurements" in text and "separate launches" in text
    assert out.exists() == emit_script


@pytest.mark.parametrize("failure", ["foreign", "prediction", "lock", "results"])
def test_tune_refuses_invalid_benchmarks(machine, monkeypatch, failure, capsys):
    if failure == "foreign":
        monkeypatch.setattr(cli, "foreign_gpu_users", lambda **kw: [dict(name="other", pid=1, vram_mb=500)])
        assert cli.main(["tune", "model"]) == 1
        assert "refusing to benchmark" in capsys.readouterr().out
        assert cli.main(["tune", "model", "--force", "--mem-bandwidth", "20"]) == 0
        return
    if failure == "prediction":
        machine.gpus.clear()
    machine.busy_at = 1 if failure == "lock" else None
    machine.failure = failure == "results"
    with pytest.raises(SystemExit, match={"prediction": "no --n-cpu-moe configuration", "lock": "fake busy lock",
                                          "results": "no configuration completed"}[failure]):
        cli.main(["tune", "model", "--draft", "draft", "--depth-sweep", "--mem-bandwidth", "20"])


@pytest.mark.parametrize("failure", [False, True])
def test_doctor_exit_status_and_model_checks(machine, failure, capsys):
    machine.checks = [doctor.Check("check", "fail" if failure else "warn", "detail")]
    assert cli.main(["doctor"] + (["--model", "model"] if failure else [])) == int(failure)
    assert ("1 blocking" in capsys.readouterr().out) == failure


@pytest.mark.parametrize("error", [KeyboardInterrupt, BrokenPipeError])
def test_main_handles_interruption_and_broken_pipe(monkeypatch, capsys, error):
    monkeypatch.setattr(cli, "cmd_probe", lambda args: (_ for _ in ()).throw(error()))
    assert cli.main(["probe"]) == (130 if error is KeyboardInterrupt else 0)
    assert ("interrupted" in capsys.readouterr().err) == (error is KeyboardInterrupt)


def test_cli_module_entrypoint_uses_real_parser(machine, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wirl", "doctor"])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as exit:
            runpy.run_module("wirl.cli", run_name="__main__")
    assert exit.value.code == 0


@pytest.mark.parametrize("failure,pattern", [("listing", "could not list"), ("empty", "no GGUF files"),
                                            ("filter", "no quantisation matching"), ("headers", "could not read any")])
def test_recommend_reports_remote_and_filter_failures(machine, monkeypatch, failure, pattern):
    def files(repo):
        if failure == "listing":
            raise OSError("offline")
        return [] if failure == "empty" else [{"path": "Q4_K_M.gguf", "size": 1}]

    monkeypatch.setattr(compat, "list_gguf", files)
    monkeypatch.setattr(recommend, "load_remote", lambda *a: (_ for _ in ()).throw(EOFError("short header")))
    argv = ["recommend", "owner/repo", "--mem-bandwidth", "20"]
    if failure == "filter":
        argv += ["--filter", "absent"]
    with pytest.raises(SystemExit, match=pattern):
        cli.main(argv)


@pytest.mark.parametrize("mode", ["ceiling", "mixed", "calibrated", "slow", "none", "filter"])
def test_recommend_displays_fit_confidence_and_download_choice(machine, monkeypatch, mode, capsys):
    names = ["IQ2_XXS", "Q4_K_M", "Q6_K", "Q8_0", "BAD"]
    monkeypatch.setattr(compat, "list_gguf", lambda repo:
                        [{"path": name+".gguf", "size": (i+1)*1000000000} for i, name in enumerate(names)])

    def load(repo, cand):
        if cand.name == "BAD":
            raise EOFError("short header")

    def evaluate(cand, *a, **kw):
        cand.fits_at_all = cand.name != "IQ2_XXS" and mode != "none"
        cand.note = "not enough RAM"
        cand.bpw = {"IQ2_XXS": 2, "Q4_K_M": 4, "Q6_K": 6, "Q8_0": 8}[cand.name]
        cand.tps = 2 if mode == "slow" else 10
        cand.confidence = {"Q4_K_M": "calibrated", "Q6_K": "mixed", "Q8_0": "ceiling"}.get(cand.name, "calibrated")
        cand.fits_vram = cand.name == "Q8_0"
        cand.knob, cand.knob_value = "--n-cpu-moe", 1
        cand.ram_needed = 3<<30
        if mode in ("mixed", "calibrated", "slow") and cand.name == "Q8_0":
            cand.fits_at_all = False
        if mode in ("calibrated", "slow") and cand.name == "Q6_K":
            cand.fits_at_all = False

    monkeypatch.setattr(recommend, "load_remote", load)
    monkeypatch.setattr(recommend, "evaluate", evaluate)
    argv = ["recommend", "owner/repo", "--mem-bandwidth", "20", "--ram", "64"]
    if mode == "filter":
        argv += ["--filter", "Q4_K"]
    assert cli.main(argv) == (1 if mode == "none" else 0)
    text = capsys.readouterr().out
    assert "What you can run" in text
    if mode == "none":
        assert "Nothing in this repository fits" in text
    else:
        assert "hf download owner/repo" in text
        expected = "Q8_0" if mode == "ceiling" else "Q6_K" if mode == "mixed" else "Q4_K_M"
        assert "Download "+expected in text
        assert ("comfortably fast" in text) == (mode == "ceiling")
    if mode != "filter":
        assert "short header" in text and "not enough RAM" in text


@pytest.mark.parametrize("verbose", [True, False])
def test_find_draft_displays_good_warned_and_incompatible_candidates(machine, monkeypatch, verbose, capsys):
    monkeypatch.setattr(drafters, "find_candidates", lambda g: [
        {"repo": "owner/draft", "purpose_built": True}, {"repo": "owner/sibling", "purpose_built": False}])

    def vet(sig, repo, **kw):
        base = dict(repo=repo, file="good.gguf", size=100000000, sig=sig,
                    compatible=True, problems=[], warnings=[])
        return [{"error": "offline"}, base, dict(base, file="warn.gguf", warnings=["tokeniser differs"]),
                dict(base, file="bad.gguf", compatible=False, problems=["vocab mismatch"])]

    monkeypatch.setattr(drafters, "vet", vet)
    assert cli.main(["find-draft", "model"] + (["--verbose"] if verbose else [])) == 0
    text = capsys.readouterr().out
    assert "Smallest compatible" in text and "purpose-built" in text and "sibling" in text
    assert "tokeniser differs" in text
    assert ("vocab mismatch" in text) == verbose
    assert "hf download owner/draft good.gguf" in text


@pytest.mark.parametrize("candidates", [False, True])
def test_find_draft_reports_no_candidates_or_no_compatible_headers(machine, monkeypatch, candidates, capsys):
    monkeypatch.setattr(drafters, "find_candidates", lambda g:
                        [{"repo": "owner/draft", "purpose_built": True}] if candidates else [])
    monkeypatch.setattr(drafters, "vet", lambda *a, **kw: [{"error": "offline"}])
    assert cli.main(["find-draft", "model"]) == 1
    assert ("Nothing compatible found" if candidates else "No candidate repositories found") in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["list", "remote", "local", "incompatible"])
def test_check_draft_listing_and_verdicts(machine, monkeypatch, mode, capsys):
    sig = machine.g.vocab_sig()
    draft = dict(sig)
    if mode == "incompatible":
        draft.update(vocab_sha256_16="different", bos=5)
    result = compat.compare(sig, draft)
    monkeypatch.setattr(compat, "list_gguf", lambda repo: [{"path": "draft.gguf", "size": 1000000000}])
    monkeypatch.setattr(compat, "check_remote", lambda *a: result)
    monkeypatch.setattr(compat, "check_local", lambda *a: result)
    argv = ["check-draft", "target"]
    if mode in ("list", "remote"):
        argv += ["--repo", "owner/repo"]
        if mode == "remote":
            argv += ["--file", "draft.gguf"]
    else:
        argv += ["--draft", "draft.gguf"]
    assert cli.main(argv) == int(mode == "incompatible")
    text = capsys.readouterr().out
    assert ("Re-run with --file" if mode == "list" else "NOT COMPATIBLE" if mode == "incompatible" else "COMPATIBLE") in text
    if mode == "incompatible":
        assert "vocabulary mismatch" in text and "warning:" in text
