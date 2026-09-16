# Implementation plan: PyPI release and pre-release fixes

Written 2026-09-16. Line numbers refer to commit `3c6015a` plus the Phase 1
changes, which are in the working tree but not yet committed.

## Goal

Publish `will-it-run-local` on PyPI as a command-line tool, installed with
`pipx install will-it-run-local` or `uv tool install will-it-run-local`.

Before the first release, fix the bugs that only appear on hardware unlike the
reference machine. Those are exactly the people PyPI brings.

## Follow-up scope (2026-09-16)

The user authorised continuing the remaining phases. Phase 4 is now active.
The earlier restrictions on real hardware benchmarks, llama-server/service
operations, pushing and tagging remain in force. Implement and test the local
packaging/release preparation, and leave the hardware acceptance measurements
and external account setup explicitly pending.

**Coverage status: measured, not 100%.** Phase 4 brings the suite to 187 passing
tests. The Python package has 69.00% line and 54.12% branch coverage, leaving
730 statements and 334 branch outcomes unexecuted (Python 3.12.3 measurement).
Added development-only coverage tooling, CI JSON/HTML artifacts on 3.14 and a
concrete gap inventory in `docs/test-coverage.md`. Full line/branch coverage is
an outstanding workstream; no paths are excluded to inflate the result. The
Phase 1–3 baseline is retained in that document for comparison.
The latest totals also reproduce on Python 3.14.2. Both edited workflows pass
actionlint 1.7.12 locally; their remote jobs remain pending until authorised
pushing/CI execution.

## Coverage follow-up

**Status: in progress.** The user asked to continue the remaining work. Close
the Python line and branch gaps with behavioural tests and retain all hardware,
network and publication restrictions. Hardware acceptance is separate.

### C1: drafter discovery and compatibility

**Status: complete.** Ten new fake-response tests cover search requests and
failure, filename/family reduction, candidate deduplication/ranking, size/file
limits, per-file/repository failures, remote trees, header-only checks and
tokeniser warnings. Both modules now have 100% line and branch coverage.
The suite passes 197 tests; overall coverage is 72.40% lines and 56.87% branches.
No runtime changes or departures. Added an opt-in fixture that rejects any
unmocked process, network request or signal in these tests.

### C2: hardware discovery, pre-flight, bandwidth and locks

**Status: complete.** Forty-six new mocked cases cover kernel inventories,
missing/invalid GPU queries, topology validation, server backend detection,
cache/log inventories, bandwidth defaults/output/errors and fake numpy,
permission/owner/flock failures and foreign-process filtering. All five modules
have 100% line and branch coverage; plain pytest passes 243 tests. No runtime
changes, predictor figure changes or departures. No real hardware measurements.

### C3: GGUF metadata, costs, predictions, recommendations and output

**Status: complete.** Thirty-six new cases cover scalar/array metadata,
remote buffer reuse/refetch/EOF, complete shard tables, tensor properties,
missing/per-layer/MLA metadata, zero-bandwidth/no-fit predictions, candidate
loading/failure/dense paths, colour/wrapping, generated files and entry-point
help. These modules now have 100% line and branch coverage; plain pytest
passes 279 tests. Published figures in `tests/test_predict.py` are unchanged.

**Departures:** the long-word wrapping regression failed on the original code;
fixed its extra blank first line. Removed redundant empty-list checks after
head-count values already normalise empty lists with `or`, and the impossible
empty-parent check after `abspath`. Empty metadata and nested-file tests retain
the same behaviour, without artificial fixtures for unreachable branches.

### C4: server lifecycle, searches and sweeps

**Status: complete.** Fifteen new isolated cases cover executable discovery,
readiness retry/timeout/exit, prompt and completion payloads, log/teardown
handling, failed/partial measurements, graceful and escalated cleanup, failed
dense bisection, skipped/failed context requests, draft regressions, thread
sweeps and instability reports. All three modules have 100% line and branch
coverage; plain pytest passes 294 tests. No runtime changes or departures.
Every subprocess and signal is fake, including the SIGTERM/SIGKILL cases.

### C5: command flows and error reporting

**Status: complete.** Fifty-nine fake-machine cases cover command success and
failure output, missing files/server/GPU selections, incompatible drafts,
pre-flight refusal/force, table limits, failed searches and sweeps, context
skips/failures, emitted scripts, recommendation confidence and download choices,
drafter verdicts, interruption/broken pipes and the module entry point.
The CLI now has 100% line and branch coverage. Plain pytest passes 353 tests;
the full Python package covers 2354/2354 statements and 726/726 branch outcomes,
with no exclusions. No runtime changes or departures in this item; all
bandwidth, server, network and signal operations remain mocked.

## Decisions

| decision | why |
|---|---|
| Ship a CLI tool, not a library | the modules change with every calibration; nobody should import `wirl.predict` and expect it to hold still, and the README should not suggest they can |
| Version 0.1.0, `Development Status :: 3 - Alpha` | tested on one machine, which the README already says |
| Bundle the unchanged probe in Linux x86_64/aarch64 wheels; retain source compilation | the follow-up activates Phase 4 so prebuilt/Docker llama.cpp users need no compiler |
| Linux only, declared in the metadata | the code depends on `fcntl`, `/proc`, `/sys`, `pthread_setaffinity_np`, `taskset` and systemd |
| PyPI trusted publishing, no API tokens | nothing to leak or rotate |

## Order of work

| phase | contents | blocks the release? |
|---|---|---|
| 1 | CI and packaging | yes; done, not committed |
| 2 | five correctness bugs | yes |
| 3 | hardening | no, except 3.1 and 3.2, which are recommended |
| | **release 0.1.0** | |
| 4 | prebuilt probe in platform wheels | no; start when triggered |

The Phase 2 items don't depend on each other; the only dependency in the plan
is that 3.3 needs 2.3. Land each item as its own commit, with its tests.

---

## Phase 1: CI and packaging (done, not committed)

**Status: complete locally.** Committed with the existing Phase 1 changes after
plain `pytest -q` passed (109 tests) in a throwaway editable-install venv.
The system pytest initially could not import the uninstalled project; installing
`.[dev]` fixed the environment. Remote CI and release setup are outside this run.

### What was wrong

- **CI failed on every push since it was added.** Six tests did
  `from tests.conftest import build_gguf`. That only resolves when the working
  directory is on `sys.path` (as under `python -m pytest`), and breaks whenever
  another package named `tests` is importable. CI runs plain `pytest`.
- **The sdist left out `tests/conftest.py`**, so the suite could not run from it.
- **Metadata:** the licence used the deprecated TOML-table form plus a licence
  classifier. The version was written in both `pyproject.toml` and
  `wirl/__init__.py`, and hard-coded as `will-it-run-local/0.1` in three
  User-Agent strings. The build warned that `wirl.csrc` was not a listed package.
- **README:** it said CI had not run and that there were 92 tests (both false),
  and its relative `examples/` link would break on the PyPI page.

### What changed

| file | change |
|---|---|
| `tests/conftest.py`, `tests/test_model.py`, `tests/test_gguf.py` | the writer is renamed `write_gguf` and exposed as the `build_gguf` fixture; the six tests take the fixture instead of importing |
| `MANIFEST.in` (new) | `recursive-include tests *.py` |
| `pyproject.toml` | `setuptools>=77`; `license = "MIT"` with `license-files`; version read from `wirl.__version__`; package discovery that includes `wirl.csrc`; classifiers for Alpha, NVIDIA CUDA, Linux, Python 3 only, Benchmark |
| `wirl/__init__.py` | `USER_AGENT`, built from `__version__`, used by `gguf.py`, `compat.py` and `drafters.py` |
| `README.md` | CI status badge; removed the false CI bullet and the test count; absolute `examples/` link |
| `.github/workflows/ci.yml` | matrix 3.9 / 3.11 / 3.14; new `package` job (below); callable from other workflows |
| `.github/workflows/release.yml` (new) | on a `v*` tag: run CI, check the tag equals `v` + `wirl.__version__`, build, publish through trusted publishing in the `pypi` environment |

The `package` job builds the wheel and sdist, runs `twine check --strict`,
installs the wheel into a clean venv, runs the sdist's tests against it from
outside the source tree, and compiles the probe from the installed package.

### Verified locally

- The `test` job's steps in fresh venvs on 3.9, 3.11 and 3.14: 109 passed, the
  probe compiles, every subcommand's `--help` works.
- The `package` job's shell steps, run unchanged: pass.
- Negative control: an sdist built without `MANIFEST.in` gives 94 passed and 15
  errors, so the job does catch that mistake.
- Metadata-Version 2.4, every classifier valid, both workflow files pass
  GitHub's workflow schema.

### Remaining

1. Commit and push, then confirm both CI jobs pass on GitHub. Nothing above
   counts until they do.
2. The one-time setup in the [release checklist](#release-checklist).

---

## Phase 2: correctness bugs (block the release)

### 2.1 `wirl recommend` says nothing fits on a machine with no GPU

**Status: complete.** Added CPU-only estimates, KV-inclusive RAM accounting,
swap rejection and README calibration caveat. Three regression tests failed
before the fix and pass afterwards; plain `pytest -q`: 112 passed.

**Departure:** handle CPU-only before the generic combined-pool rejection.
Otherwise insufficient RAM returns before the specified swap note can be set.
An exact RAM fit is accepted, consistent with the planned `>` swap check.

**Problem.** With no GPU, `cmd_recommend` sets the VRAM budget to 0
(`cli.py:512`). `recommend.evaluate` then calls `predict.best_fit` or
`best_fit_dense`, which both add `CUDA_OVERHEAD` (0.4 GB) to every
configuration, so nothing fits. Reproduced: a 36 MiB model with 64 GiB of RAM
is reported as "does not fit at this context length" at 0 tok/s, and the
command ends with "Nothing in this repository fits this machine."

**Change** (`wirl/recommend.py`, `evaluate`). After the existing
`fits_at_all` check, treat `vram_budget <= 0` as CPU-only and return early:

- `knob, knob_value = "--n-gpu-layers", 0`
- `tps = predict.cpu_only_tps(mc, bw_cpu)`
- `vram = 0`, `cpu_share = 1.0`, `confidence = "calibrated"` (what the existing
  80%-or-more rule gives)
- `ram_needed = mc.total_bytes + kv`: with nothing offloaded the KV cache is in
  RAM too, whereas the GPU paths leave it out because it lives on the GPU there
- if `ram_needed > ram_available`: `note = "would swap: not enough system RAM"`
  and `fits_at_all = False`

The CLI needs no change: its table already prints `f"{knob} {knob_value}"` and
"N GiB in RAM".

`CPU_ONLY_DERATE` comes from a single measurement. Say so in the README's
"Known limitations", next to the other calibration constants.

**Tests** (`tests/test_recommend.py`, following the `FakeCost` pattern in
`test_confidence_tracks_how_much_is_read_from_ram`):

- budget 0 with plenty of RAM: `fits_at_all`, `tps == predict.cpu_only_tps(...)`,
  `knob_value == 0`, and `ram_needed` includes the KV cache
- budget 0 with less RAM than model + KV: not runnable, and the note mentions swap
- CLI: with `_no_gpu(monkeypatch)` from `tests/test_search.py`, and
  `compat.list_gguf` / `recommend.load_remote` monkeypatched to serve the
  `moe_model` fixture, `wirl recommend some/repo --mem-bandwidth 20` recommends
  it and prints `--n-gpu-layers 0`

**Done when** a machine with no GPU gets a recommendation and a figure.

### 2.2 `wirl tune` treats dense models as mixture-of-experts

**Status: complete.** Dense tuning predicts and sweeps ascending GPU-layer
counts, MoE keeps descending expert counts, and draft depths copy the winner.
Offload setters are public and verdicts name the selected knob. Six new cases
cover both CLI paths, winner preservation, verdicts and first-failure stopping;
five failed before the fix (the MoE control already passed).
Plain `pytest -q`: 118 passed. Followed the plan; no existing tests changed.

**Problem.** `cmd_tune` always predicts with `predict.best_fit`
(`cli.py:734`) and sweeps `--n-cpu-moe` (`cli.py:741`, `:754`). On a dense
model `--n-cpu-moe` does nothing: every value gives the same predicted split
(verified). So:

- a dense model that fits entirely in VRAM launches several identical servers
  and reports the noise as a sweep
- a dense model that doesn't fit exits with "no configuration fits this GPU",
  although partial offload works: for the same synthetic model and budget,
  `best_fit_dense` puts 2 of 4 layers on the GPU

`cmd_auto` already branches on `mc.is_moe` (`cli.py:299`–`314`); `tune` never
got that fix. Its draft-depth step makes the same assumption (`cli.py:766`
copies only `n_cpu_moe`).

**Change.**

- `wirl/cli.py`, `cmd_tune`: copy `cmd_auto`'s branch. MoE stays as it is.
  For dense models, predict with
  `predict.best_fit_dense(..., headroom=min(args.headroom * 1e9, 1e9))`, and use
  ascending candidates `range(max(0, start - span), min(n_layer, start + span) + 1)`,
  i.e. towards more GPU. Headings and messages use `knob`.
- `wirl/tune.py`: replace `sweep_ncmoe` with
  `sweep_offload(base, binary, candidates, moe, ...)`, which applies
  `search._set_ncmoe` or `search._set_ngl`. Make those two public (`set_ncmoe`,
  `set_ngl`). Keep stopping at the first failure: in both cases the candidate
  list moves towards more VRAM, so later ones would fail too.
- Draft depth: `b2 = copy.copy(win.config)`, instead of copying `base` and
  setting `n_cpu_moe`. Drop the local `import copy`; `cli.py` already imports
  it at the top.
- `predict.verdict`'s "DOES NOT FIT" text names `--n-cpu-moe`; take the knob
  name as a parameter.

**Tests** (new `tests/test_tune.py`). Monkeypatch `wirl.tune.run_config` like
`_fake_runner` in `tests/test_search.py`, plus `cli.find_server`,
`cli.foreign_gpu_users`, `probe.gpu_info` (one 24 GiB GPU) and
`wirl.lock.LOCK_PATH` (into `tmp_path`).

- dense model larger than `--vram`: no "no configuration fits"; every launched
  config has `n_cpu_moe is None`; the `n_gpu_layers` values are distinct and
  ascending
- MoE model: the launched configs are the same as today (descending `n_cpu_moe`)
- dense with `--draft --depth-sweep`: the depth configs keep the winner's
  `n_gpu_layers`

Large synthetic models cost nothing. Tensor sizes come from the dimensions in
the header and the file holds no weights, so
`(f"blk.{i}.attn_q.weight", (1024, 1024), 0)` prices at 4 MiB in a file of a few
hundred bytes.

**Done when** `wirl tune` on a dense model sweeps `--n-gpu-layers` and never
passes `--n-cpu-moe`.

### 2.3 Measurements are wrong on machines with several GPUs

**Status: complete.** UUIDs flow through discovery, auto/tune launches, VRAM
reads, contention checks and emitted launchers; doctor accepts `--gpu`.
Fourteen new cases cover parsers, process environment, selected VRAM/contention,
launcher output, invalid/negative indices, auto/tune and doctor's selected card.
The first eleven failed before the fix; plain `pytest -q`: 132 passed.

**Refinement:** select by the reported physical `index`, rather than a list
offset, including when discovery order differs. Auto's contention check is in
`doctor.run_all`, so passing its GPU index applies the UUID filter there.

**Problem.**

- `RunConfig.argv` (`runner.py:48`) and `server()` (`runner.py:223`) never
  restrict devices. By default llama-server splits layers across every visible
  GPU, so the VRAM boundary found by bisection is for all GPUs combined.
- `run_config` records VRAM with `gpu_used_bytes()` (`runner.py:252`), which
  reads GPU 0 only.
- `--gpu N` changes only the budget and bandwidth used for prediction
  (`cli.py:37`–`44`), and silently falls back to GPU 0 when N is out of range
  (`cli.py:41`).
- `foreign_gpu_users` counts processes on every GPU, and
  `doctor.check_gpu_free` describes GPU 0 (`doctor.py:83`), so a job on another
  GPU blocks the sweep.
- The README's "Known limitations" says "With several present it uses index 0
  unless told otherwise". It doesn't.

**Change.** Pin the llama-server process to one physical GPU, by UUID.

- `wirl/probe.py`
  - `gpu_info`: append `uuid` to the end of the `--query-gpu` fields, so
    existing positions don't move, and return it
  - `gpu_processes`: query `pid,process_name,used_memory,gpu_uuid` and return
    `gpu_uuid`
- `wirl/runner.py`
  - add `RunConfig.gpu_uuid: str | None = None`
  - `server()` passes `env` to `Popen`: a copy of `os.environ` with
    `CUDA_VISIBLE_DEVICES=<uuid>` when a UUID is set
  - `gpu_used_bytes(uuid=None)` finds the GPU by UUID (`None` keeps today's
    GPU-0 behaviour); `run_config` passes `cfg.gpu_uuid`
- `wirl/lock.py`: `foreign_gpu_users(..., gpu_uuid=None)` filters to that GPU
- `wirl/doctor.py`: `check_gpu_free` describes the selected GPU; `run_all`
  takes `gpu_index`
- `wirl/cli.py`
  - `_gpu_choice` exits with an error listing the available GPUs when `--gpu` is
    out of range
  - `auto` and `tune` set `gpu_uuid` on the base `RunConfig` and pass it to the
    foreign-process check
  - `doctor` gains `--gpu`
- `wirl/emit.py`: when `cfg.gpu_uuid` is set, the launcher exports
  `CUDA_VISIBLE_DEVICES` before `exec`, with a comment saying it pins the card
  that was measured
- README: replace the "Single GPU only" bullet. Each run uses one GPU, chosen
  with `--gpu` (the `nvidia-smi` index), and llama-server is restricted to it.
  Splitting a model across GPUs is not modelled.

Why a UUID and not an index: `nvidia-smi` numbers GPUs in PCI bus order, but
CUDA numbers them fastest-first unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set. So
`CUDA_VISIBLE_DEVICES=1` can name a different card from `nvidia-smi`'s GPU 1. A
UUID can't be misread, and it still names the right card in the emitted
launcher if the order changes later. `nvidia-smi` ignores
`CUDA_VISIBLE_DEVICES`, so the indices it reports stay physical.

**Tests.**

- `gpu_info` and `gpu_processes` parse UUIDs (monkeypatch `probe._run` and
  `shutil.which`)
- `server()` passes `CUDA_VISIBLE_DEVICES=<uuid>` to `Popen` (monkeypatch
  `subprocess.Popen` to record its arguments, `_wait_health` to return 1.0,
  and `_stop` to do nothing)
- `foreign_gpu_users(gpu_uuid=...)` ignores processes on other GPUs
- the launcher has the export line when `gpu_uuid` is set, and not otherwise
- `wirl plan model.gguf --gpu 3` on a one-GPU machine exits with a clear error

**Done when** `--gpu 1` launches on, measures VRAM of, and checks contention
for GPU 1 only, and the emitted launcher runs on the same card.

### 2.4 The launcher's CPU pinning is wrong on many desktop CPUs

**Status: complete.** Launchers use an explicit affinity-aware sibling list
and omit pinning when topology is unavailable. Sweeps scale by the specified
fractions, preserving 24/32/48/64 on the reference machine. Twelve new cases
cover CPU-list round trips, server/hybrid/limited/older/missing topology,
launcher pinning and 64/16/1-core counts; these and the updated signature test
failed before the fix. Plain `pytest -q`: 144 passed.

**Refinement:** the thread-count expression is in `tune.thread_counts` so its
reference-machine behaviour can be tested directly. The existing launcher
test now supplies CPU IDs, since core count alone cannot identify siblings.

**Problem.** `emit.launch_script` writes `taskset -c 0-{physical_cores - 1}`
(`emit.py:22`), which assumes hyperthreads are numbered after every physical
core. That holds on the reference EPYC, where core N's second thread is CPU
N+64, but not in general. On Intel desktop CPUs with performance and efficiency
cores, each performance core's two threads are numbered next to each other. On
a 13900K (8 P-cores × 2 threads = CPUs 0–15, then 16 E-cores = CPUs 16–31),
`0-23` pins both threads of every P-core plus 8 of the 16 E-cores: the opposite
of what the comment above it says. It also ignores any CPU restriction the
process already has (cgroups, containers).

Related, found while writing this plan: `auto --thread-sweep` sweeps
`{24, 32, 48, physical}` (`cli.py:376`). On a 16-core machine three of those
four exceed the core count, and each costs a server launch.

**Change.**

- `wirl/probe.py`
  - `parse_cpulist("0-3,8")` → `[0, 1, 2, 3, 8]`, and
    `format_cpulist([0, 1, 2, 3, 8])` → `"0-3,8"`
  - `physical_core_cpus()`: go through `os.sched_getaffinity(0)` in order and,
    for each CPU, read `/sys/devices/system/cpu/cpuN/topology/core_cpus_list`
    (or `thread_siblings_list` on older kernels). Keep the first allowed CPU of
    each sibling set. Return `None` if the topology can't be read.
- `wirl/emit.py`: `launch_script(cfg, binary, cpus=None, ...)` replaces
  `physical_cores` and writes `taskset -c <format_cpulist(cpus)>`. When `cpus`
  is `None` it doesn't pin at all, rather than guessing.
- `wirl/cli.py` (`:474`, `:808`): pass `probe.physical_core_cpus()`.
- Thread sweep: `counts = sorted({max(1, round(p * f)) for f in (0.375, 0.5, 0.75, 1.0)})`
  with `p = cpu["physical"]`. That is exactly `{24, 32, 48, 64}` on the
  reference machine, so its published results stay reproducible, and
  `{6, 8, 12, 16}` on 16 cores.

**Tests.**

- `parse_cpulist` / `format_cpulist` round-trip, with single CPUs and ranges
- `physical_core_cpus` against a fake sysfs (monkeypatch `probe._read` and
  `os.sched_getaffinity`):
  - server layout (second threads at N+64) → `0-63`
  - hybrid layout (`0-1`, `2-3`, … then single E-cores) → `0,2,4,…,14,16-31`
  - affinity limited to some CPUs → only those
- update `test_launch_script_pins_to_physical_cores` for the new signature
- thread counts for 64 and 16 physical cores

**Done when** a generated launcher pins exactly one logical CPU per physical
core, matching `lscpu -e=CPU,CORE`, on both layouts.

### 2.5 `wirl auto` refuses models bigger than free RAM, even when RAM plus VRAM is enough

**Status: complete.** Extracted dense VRAM accounting and added the
most-offloaded RAM requirement, including CPU-side dense KV. Auto and doctor
use it; doctor accepts context and VRAM options. Six new cases cover monotonic
MoE/dense requirements, drafts, CPU fallback and both CLI pre-flights with
explicit context/VRAM overrides. These and the two updated doctor-signature
tests failed before the fix. Plain `pytest -q`: 150 passed; the measured
figures in `tests/test_predict.py` are unchanged.

**Departure:** reserve draft weights + KV for dense pre-flight too, and include
them in the CPU fallback's RAM requirement. The proposed fallback omitted a
supplied draft even when neither model can be offloaded. Dense speed prediction
remains unchanged; this reservation only informs the RAM pre-flight check.
The two existing doctor tests pass the computed RAM need directly, retaining
their original blocking/offload scenarios under the new signature.

**Problem.** `cmd_auto` runs the pre-flight checks with `gpu_bytes=0`
(`cli.py:278`), so `doctor.check_ram_for_model` counts the whole model against
system RAM. Whenever the model is bigger than available RAM that is a blocking
FAIL, and `auto` won't sweep without `--force`. That's the tool's core case: a
model too big for either pool alone.

`wirl doctor --model` makes the opposite mistake. It assumes a full GPU's
worth of weights is offloaded (`cli.py:663`), ignoring the KV cache, CUDA
overhead and headroom.

**Change.**

- `wirl/predict.py`
  - move the inline VRAM expression in `dense_curve` (`predict.py:82`) into
    `vram_needed_dense(mc, g, n_gpu_layers, ctx, k_type, v_type)` and use it
    there. This shouldn't change behaviour, and the existing tests cover it.
  - add `min_ram_needed(mc, g, vram_budget, ctx, k_type="f16", v_type="f16", draft_mc=None, draft_g=None)`:
    the system RAM needed at the most-offloaded configuration whose predicted
    VRAM fits the budget, with no headroom
    - MoE: `total_bytes - resident_vram_weights(n)` for the smallest `n` that fits
    - dense: `total_bytes - resident_vram_dense(n)`, plus the KV-cache share left
      on the CPU, for the largest `n` that fits
    - no GPU, or nothing fits: `total_bytes + kv`

  The best case is deliberate. A FAIL here stops `auto` before it launches
  anything, and `auto` exists to find the real boundary by launching. A
  prediction should only rule a model out when even the best case can't fit.
- `wirl/doctor.py`: `check_ram_for_model(mem, ram_need)` takes the requirement
  directly, and `run_all(ram_need=None, ...)` replaces `model_bytes, gpu_bytes`.
  Its message says the figure is for the most-offloaded configuration that fits
  in VRAM.
- `wirl/cli.py`: `auto` and `doctor --model` both compute `ram_need` with
  `min_ram_needed`. `doctor` gains `--ctx` (default 16384) and the GPU options,
  because the answer depends on both.

**Tests.**

- `min_ram_needed` on the `moe_model` fixture: a large budget leaves only the
  CPU-side experts; a zero budget gives model + KV; the result never rises as
  the budget grows
- the dense equivalent, including the KV cache left on the CPU
- update `test_model_larger_than_ram_is_blocking` and
  `test_offloading_to_gpu_can_make_it_fit` for the new signature
- `cmd_auto` passes `run_all` a requirement smaller than the model when VRAM is
  available (monkeypatch `doctor.run_all` to record its arguments, then stop)

**Done when** a model that is bigger than free RAM but fits in RAM plus VRAM
passes pre-flight in both commands, and one bigger than RAM plus VRAM still
fails.

---

## Phase 3: hardening

### 3.1 The benchmark lock fails for a second user (recommended before release)

**Status: complete.** Open existing locks without `O_CREAT`, create exclusively
with explicit shared permissions, retry creation races, and report inaccessible
locks as `LockUnavailable` with owner and `WIRL_LOCK` guidance. Four new tests
(permissions, inode/plain open, restrictive umask, creation race) failed before
the fix; plain `pytest -q`: 154 passed. Followed the plan. The real permission
test skips under root, as specified; it ran on this machine.

**Problem.** `benchmark_lock` opens `/tmp/will-it-run-local.benchmark.lock`
with `O_CREAT` (`lock.py:31`), outside the `try`. The requested `0o666` is cut
down by the umask, usually to `0644`. Worse, with `fs.protected_regular` set to
1 or more (it is 2 on the reference machine), the kernel refuses an `O_CREAT`
open of another user's file in a sticky, world-writable directory like `/tmp`,
whatever its permissions. So the second user of a shared machine gets a
`PermissionError` traceback.

**Change.** Open with `O_RDWR` first. If the file doesn't exist, create it with
`O_CREAT | O_EXCL` and `fchmod(fd, 0o666)`, and retry the plain open if another
process created it in the meantime. Turn a `PermissionError` into
`LockUnavailable`, a subclass of `BenchmarkBusy`, that names the file's owner
and mentions `WIRL_LOCK`. The existing `except BenchmarkBusy` handlers in
`cli.py` then print it cleanly.

**Tests.** A read-only (`0o444`) lock file in `tmp_path` raises
`LockUnavailable`, not `PermissionError` (skip when running as root). An
existing lock file is opened, not replaced.

### 3.2 Remote header reads trust the server to honour Range (recommended before release)

**Status: complete.** Reject non-206 responses before reading and cap each read
at the requested length. Two fake-response tests failed before the fix and
verify ignored Range rejection without a read, response closure, the Range
header and bounded 206 reads. Plain `pytest -q`: 156 passed. Followed the plan.

**Problem.** `_Reader._fetch` (`gguf.py:49`–`56`) sends a Range header and then
reads the whole response. A server or proxy that ignores Range replies `200`
with the entire file, and that gets read into memory: tens of GB.

**Change.** Raise an error unless `r.status == 206`, and read at most `length`
bytes.

**Tests.** A fake `urlopen` response with status 200 raises; a 206 response
with extra bytes returns only `length`.

### 3.3 "Peak VRAM" is not a peak (needs 2.3)

**Status: complete.** A background thread samples the selected UUID at a
one-second interval during requests and retains the maximum, alongside
post-request readings. The thread is stopped and joined before server teardown,
including request failure. Two transient-spike/cleanup cases failed before the
fix; a third covers telemetry errors. Plain `pytest -q`: 159 passed.

**Refinement:** a sampling exception makes the run unsuccessful rather than
presenting missing telemetry as spare headroom. This remains a sampled maximum;
allocations shorter than the interval can still be missed.

**Problem.** `run_config` samples `nvidia-smi` after each request finishes
(`runner.py:252`), so "peak VRAM" is really the VRAM in use after generation.
Anything allocated during a request and released before it ends is missed, and
the headroom logic in `search.pick_recommended` relies on this number.

**Change.** While requests run, a background thread samples
`gpu_used_bytes(cfg.gpu_uuid)` about once a second and keeps the maximum.

**Tests.** A fake `gpu_used_bytes` that spikes only while `_generate` is
running: the spike is recorded.

### 3.4 Clear errors off Linux

**Status: complete.** Defer `fcntl` until locking, reject non-Linux commands
after argparse, and preserve compilation failure reasons in bandwidth errors.
Six tests cover darwin rejection, help/version, import without fcntl, mocked
`CC=/bin/false` diagnostics and missing compiler; four failed before the fix,
while help/version controls already passed. Plain `pytest -q`: 165 passed.

**Refinement:** a shared `_build_probe` returns path + reason; public
`build_probe` keeps its existing path/None return for CI and reports the reason,
while `measure` uses the same result directly without global error state.

**Execution incident:** the first platform negative control lacked a mocked
command dispatch and ran real stream/gather bandwidth probes. They completed;
no llama-server was launched or service changed. Dispatch is now mocked, and
all subsequent platform checks avoid hardware work.

**Problem.** `lock.py:17` imports `fcntl` at module level, and `cli.py` imports
`lock` at startup, so `wirl --help` crashes with `ModuleNotFoundError` on
Windows. On macOS the probe fails to compile (`cpu_set_t` and
`pthread_setaffinity_np` are glibc-only), and `membw.measure` then reports "no C
compiler" (`membw.py:110`), which is false.

**Change.**

- Import `fcntl` inside `benchmark_lock`.
- When `sys.platform != "linux"`, `cli.main` exits with "will-it-run-local
  supports Linux only". `--help` and `--version` still work, because argparse
  handles them first.
- `build_probe` reports why it produced no binary (no compiler, or compile
  failed, with the compiler's error), and `measure` uses that reason.

**Tests.** With `sys.platform` monkeypatched to `"darwin"`,
`cli.main(["probe"])` exits with the message. With `CC=/bin/false`, the error
says the compile failed.

### 3.5 Probe binary cache: rebuild loops and a compile race

**Status: complete.** Cache by source-content and compiler-command hash,
compile to unique temporary outputs, and atomically replace complete executable
binaries. Five tests cover distinct/shared source content, cache reuse, compiler
options, failed rebuild preservation/cleanup and concurrent compiles; four
failed before the fix, while cache reuse already passed.
Plain `pytest -q`: 170 passed.

**Refinement:** exclude install-specific source/output paths from the command
hash so identical installs share a binary; split `CC` into argv to support
compiler wrappers/options and hash the command actually invoked.

**Problem.** `build_probe` (`membw.py:31`) caches a single binary at
`~/.cache/will-it-run-local/membw` and reuses it if it's newer than the source.
Two installs (say, pipx and a development checkout) keep rebuilding over each
other. Two `wirl` processes compiling at once write to the same path while one
of them may already be running it.

**Change.** Name the binary by a hash of the source and the compiler command
(`membw-<sha256[:12]>`). Compile to a temporary file in the same directory, then
`os.replace` it into place.

**Tests.** Two different sources give two binaries. An existing binary with the
right hash is reused without invoking the compiler (monkeypatch
`subprocess.run`).

### 3.6 A stale server on the benchmark port

**Status: complete.** Check the resolved host/port before creating logs or
calling `Popen`, with an old-server hint for an occupied port. Two held-socket
IPv4/IPv6 cases failed before the fix; a third checks invalid-address errors.
Plain `pytest -q`: 173 passed. The existing mocked launch/environment test now
uses port 0 so the bind check never depends on a real benchmark port being free.

**Departure:** use `SO_REUSEADDR` so closed sweep connections in TIME_WAIT don't
cause a false refusal. Report non-occupancy bind errors with their real reason
rather than claiming every failure is an old server. Resolve IPv4/IPv6 addresses
and check each. The check cannot eliminate the bind-to-launch race completely.

**Problem.** Servers start with `start_new_session=True` (`runner.py:223`).
If `wirl` is killed outright (SIGKILL, or an SSH session dropping), llama-server
keeps running and keeps port 38080. A later run's readiness check
(`_wait_health`, `runner.py:125`) can then get answers from the old server and
record a start that never happened. Today `auto` and `tune` refuse to run while
other processes are using the GPU, which catches this unless `--force` is
passed.

**Change.** Before `Popen`, try to bind the host and port. If that fails, raise
`ServerFailed` saying the port is in use and an old llama-server may still be
running.

**Tests.** With a socket holding the port, `server()` raises `ServerFailed` with
that message and never calls `Popen`.

### 3.7 Small corrections

**Status: complete.** Dense launcher notes name `--n-gpu-layers`; removed the
unused results marker and the ineffective disk check/path parameter from
`run_all` and its caller. One new mocked dense-auto launcher regression failed
before the note fix and also verifies the emitted CPU list. Plain `pytest -q`:
174 passed. Followed the plan; no existing tests changed for this item.

**Final verification:** all 12 requested items are complete, each in its own
local commit after a passing plain `pytest -q`; Phase 1 has its separate initial
commit. Throwaway venvs installed with `pip install -e '.[dev]'` ran plain pytest
from the repo root: Python 3.9.25 (pytest 8.4.2), 174 passed; Python 3.14.2 (the
newest installed interpreter, pytest 9.1.1), 174 passed. Python 3.12.3 also passes
all 174 tests. A read-only `lscpu -e=CPU,CORE,SOCKET` comparison confirms the
detected CPU list contains the first allowed logical CPU of each physical core
on this machine. The 64-core sweep and published predictor measurements remain
unchanged. Phase 4 and the release checklist were not undertaken; nothing was
pushed or tagged. The execution incident is recorded under 3.4.

- `cli.py:471`: the notes in the emitted launcher say "found empirically at
  --n-cpu-moe" even for dense models; use `knob`.
- `cli.py:385`: remove the unused `_RESULTS_MARK = None`.
- `doctor.py:308`: `check_disk(path, 0)` can never fail, because the amount
  needed is 0 and the model is already on disk by then. Remove it and the `path`
  argument from `run_all`.

---

## Phase 4: prebuilt bandwidth probe in platform wheels

**Status: implementation complete; hardware/ARM acceptance pending.** Added
Linux-only build hooks compiling the unchanged C source with `-O2 -pthread`,
Python-independent platform wheel tags, packaged-probe selection, a source-only
sdist, and native x86_64/aarch64 cibuildwheel CI. The release workflow collects
those tested wheels alongside the sdist. README describes the packaged probe
and the optional numpy fallback. No external settings, tags or pushes changed.

**Tests:** thirteen new mocked tests cover compilation flags and executable
permissions, absent compiler/stale output cleanup, strict release builds,
failed compilation, Python-independent tags, platlib routing, unsupported OS
and architecture, editable installs, packaged selection without cache/compiler,
unusable packaged files, explicit source rebuilds and mocked measurements.
The packaged-selection and measurement regressions failed without the runtime
fix; the platlib regression failed before the distribution fix. Existing
compilation tests explicitly hide a packaged binary so the same assertions run
against both checkouts and installed wheels; none were weakened or removed.

**Integration:** the real x86_64 wheel builds in manylinux_2_28 and auditwheel
repairs it to `py3-none` manylinux tags, including compatibility with glibc 2.17.
Installation in a network-disabled Debian slim container with no cc/gcc/clang
passes the loader/selection smoke check without allocating benchmark buffers.
CI additionally performs actual stream/gather checks on isolated runners and
in compiler-free containers. Local Python 3.9.25, 3.12.3 and 3.14.2 suites pass
187 tests with the development extra. The spike details are in
`docs/phase4-spike.md` and `tools/spike_probe_wheel.py`.
The same repaired wheel also passes all 187 source-archive tests in fresh
Python 3.9/3.14 venvs from outside the checkout; imports were checked to resolve
to site-packages, and both interpreters start the bundled executable's usage
path. Strict twine checks pass for the wheel and source archive.

**Departures:** marking `root_is_pure=False` alone placed the executable in
`.data/purelib`, which auditwheel rejected in the first real wheel build. A
binary `Distribution` also selects the platlib installation scheme; this keeps
the executable in the wheel root without introducing a Python extension.
Release CI uses `WIRL_REQUIRE_PROBE=1` to fail rather than publish a source-only
wheel after compiler discovery fails. `build_probe(force=True)` still compiles
source for the existing CI check; normal calls prefer the packaged executable.
The fallback to numpy belongs to `measure`, preserving the existing public
path-or-None result of `build_probe`.

**Done-when checks still pending:** native ARM execution and the alternating
reference-machine bandwidth comparison (three runs per probe per mode), plus
the actual compiler-free container measurements in CI. No ARM runner is
available locally. The earlier prohibition on real bandwidth runs still
applies here, so the local container check deliberately exercises usage only.
Do not claim Phase 4 fully accepted until those measurements pass.

**Start when** issues report "no C compiler", or before promoting the tool to
people who don't build llama.cpp themselves. People running llama.cpp from
Docker or prebuilt binaries may have no compiler, and the numpy fallback is no
substitute: numpy isn't a dependency, and the fallback is single-threaded and
gives a lower bound, not a measurement.

**Goal.** On Linux x86_64 and aarch64, `pip install` needs no compiler.
Elsewhere, installation fails with a clear message.

**Design.**

- Keep `wirl/csrc/membw.c`. At wheel build time, compile it to
  `wirl/_bin/membw` with `-O2 -pthread` and no `-march=native`, because the wheel
  has to run on any x86_64 or aarch64 CPU.
- Add a small `setup.py` next to `pyproject.toml` (setuptools reads both):
  - a custom `build_py` that compiles the probe into the build directory
  - on anything but Linux, stop with "will-it-run-local supports Linux only",
    so installing the sdist on macOS or Windows fails with that message
  - on Linux without a compiler (an sdist install), warn and build without the
    binary; the compile-on-first-run path still covers that case
  - a custom `bdist_wheel` that marks the wheel as platform-specific with the
    tag `py3-none-<platform>`. Nothing links against Python, so one wheel per
    architecture serves every Python version.
- Release workflow: use cibuildwheel with one build per architecture (e.g.
  `CIBW_BUILD="cp312-manylinux_*"`), and build aarch64 on GitHub's native ARM
  runners rather than under emulation. cibuildwheel runs `auditwheel repair` by
  default, which applies the manylinux tag. Keep publishing the sdist.
- At runtime, `membw.build_probe()` tries, in order: the packaged binary, the
  cached compile from source, numpy.

**Rejected:** rewriting the probe as a C extension module. It would run a
multi-GiB memory benchmark inside the Python process, and tie the wheels to
CPython versions for no benefit.

**Check in a short spike before committing to this:**

- `auditwheel` accepts a wheel whose only compiled file is a standalone
  executable rather than an extension module, and tags it manylinux
- the aarch64 wheel's probe runs (`pthread_setaffinity_np` and `cpu_set_t` are
  glibc)
- musllinux has not been looked at; leave it out at first

**Done when:**

- on the reference machine, the prebuilt probe's stream and gather results fall
  within the run-to-run spread of a locally compiled one (3 runs each,
  alternating), which is the standard the tool applies to everything else
- installing the wheel in a container with no compiler still measures bandwidth

---

## Release checklist

**Status: pending; no publication actions taken.** Local platform-wheel and
sdist preparation is implemented under Phase 4. Account/environment setup,
main-branch CI, README publication instructions, tagging, pushing and PyPI
installation checks remain untouched under the earlier release restriction.

### One-time setup

1. On PyPI, under your account's publishing settings, add a pending publisher:
   owner `John-Cusack`, repository `will-it-run-local`, workflow `release.yml`,
   environment `pypi`. This also reserves the name. Neither `will-it-run-local`
   nor `wirl` was taken on 2026-09-16.
2. On GitHub, under Settings → Environments, create `pypi`. Optionally require
   your approval, so a pushed tag waits for a click before publishing.
3. Optional dry run on TestPyPI. It needs its own pending publisher on
   test.pypi.org, and `repository-url: https://test.pypi.org/legacy/` on the
   publish step.

### Each release

1. Phase 2 merged, and CI green on `main`.
2. For the first release, rewrite the README's "Install" section to lead with
   `pipx install will-it-run-local` (or `uv tool install will-it-run-local`), plus
   `uvx --from will-it-run-local wirl doctor` to try it without installing. Move
   the clone instructions under "Development".
3. Set `__version__` in `wirl/__init__.py`, the only place the version is kept.
4. `git tag v0.1.0 && git push origin v0.1.0`. The workflow refuses to publish
   if the tag and `__version__` disagree.
5. After it publishes, in a clean environment, run
   `pipx install will-it-run-local`, `wirl --version` and `wirl doctor`. Check
   that the README renders on the PyPI page.

---

## Open questions

- **Python 3.9 floor.** 3.9 reached end of life in October 2025, and pytest 9
  dropped it; on 3.9, CI resolves pytest 8.4.2. Nothing in the code needs 3.9,
  so raising the floor to 3.10 would cost nothing today.
- **Thread placement on hybrid CPUs.** On Intel CPUs with performance and
  efficiency cores, llama.cpp threads that land on efficiency cores may slow
  decoding, and the thread sweep in 2.4 doesn't tell the two apart. Measure it
  on such a machine before changing anything.

## Out of scope

- splitting one model across several GPUs (2.3 adds choosing one GPU, nothing more)
- ROCm, Apple Silicon, Windows and macOS
- a stable Python API
