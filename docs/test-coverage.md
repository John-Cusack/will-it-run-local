# Test coverage

**Current status: 100% Python line and branch coverage.** On 2026-09-16 the
completed coverage follow-up passes 353 tests and covers all 2354 statements
and 726 branch outcomes across every module in `wirl`. No source files,
statements or error paths are excluded. The small statement-count reduction
comes from simplifying an unreachable empty-parent guard, rather than hiding
code from coverage.
Fresh editable-install venvs on Python 3.9.25 and 3.14.2 reproduce these exact
totals and pass plain `pytest -q`. The extracted CI test/JSON gate also passes
locally on 3.14.2. Python 3.12.3 passes the same suite and coverage totals.

The follow-up adds 166 tests/cases covering the gaps below. The tests exposed
and fixed an extra blank first line when wrapping a long word. Two redundant
empty-list guards after head-count normalisation were also simplified. Existing
tests and the published measurements in `tests/test_predict.py` are unchanged.

## Historical measurements

Measured on 2026-09-16 after Phases 1–3: 174 passing tests, **68.45% line
coverage** (1610/2352 statements) and **53.03% branch coverage** (385/726).
There are 742 unexecuted statements and 341 unexecuted branch outcomes. No
source files or error paths were excluded from this measurement.

After Phase 4's mocked packaging/runtime tests: **187 tests pass**, with
**69.00% line coverage** (1625/2355 statements) and **54.12% branch coverage**
(394/728). At that point, reaching 100% required covering **730 statements** and
**334 branch outcomes**, chiefly in the areas listed below. Both historical
measurements use Python 3.12.3; the latest totals also reproduce exactly on
Python 3.14.2. CI collects JSON/HTML reports on Python 3.14.

## Running checks

Install the development extra and use the real pytest entry point from the
repository root, matching CI:

```bash
pip install -e '.[dev]'
pytest -q
pytest -q --cov=wirl --cov-branch --cov-report=term-missing --cov-report=html
```

The HTML report is written to `htmlcov/index.html`. These are Python execution
metrics: the C probe and GitHub workflow shell steps require separate build and
integration checks. Even 100% line and branch coverage cannot establish the
accuracy of a bandwidth prediction on unmeasured hardware.
CI uploads `coverage.json` and the HTML report as the `python-coverage` artifact.
Its Python 3.14 check applies `--cov-fail-under=100` and independently asserts
zero missing statements, zero missing branch outcomes and zero excluded lines
in JSON. The combined coverage percentage also includes branches; the separate
statement/branch fields establish that both totals are exactly 100%.

## Closed coverage gaps

| area | missing statements in the baseline | scenarios to add |
|---|---:|---|
| CLI | 323 | every command's success/failure output; incompatible drafts; blocked pre-flight; depth, context and thread sweeps; emitted scripts; interruptions and broken pipes |
| hardware discovery | 66 | fake `/proc`/sysfs inventories, missing or malformed values, absent/failed nvidia-smi, bandwidth hints, swap counters |
| drafter discovery | 55 | mocked HuggingFace search/tree responses, filtering, size limits, incompatible candidates, per-repository failures |
| runner | 49 | executable discovery, readiness timeout/exit/retry, fake completions, prompt construction, log handling, SIGTERM/SIGKILL cleanup and failed/partial measurements |
| bandwidth | 49 | mocked output parsing, defaults, invalid compiler/source, numpy fallback using a tiny fake array, probe subprocess failure |
| doctor | 47 | fake governors/THP/cache inventories, low RAM margin, log errors, compiler/backend detection failures |
| recommendation | 30 | remote header/shard loading, dense splits, no-fit and swap paths, filename/quality edge cases |
| compatibility | 25 | remote list/header responses, missing vocabulary metadata, local/remote incompatible draft results |
| GGUF reader | 21 | chunk refetch/EOF, all metadata scalar types, tensor-free headers, multi-shard reads and vocabulary properties |
| search | 21 | failure direction, recorded attempts, mocked long-context profiling and context-report branches |
| sweeps | 17 | draft-depth failure/regression, thread sweeps, failed summaries and instability thresholds |
| lock | 12 | permission/owner lookup failures, flock errors, ignored/small/malformed GPU process records |
| model/predict/report/type table/entry point | 27 | degenerate metadata, no-fit/zero-bandwidth cases, coloured output/wrapped checks, unknown types and subprocess entry-point help |

These gaps are now covered by behavioural assertions, using tiny synthetic
headers, fake files, HTTP responses, arrays, clocks, processes and signals.
The new tests use an opt-in `isolated_runtime` fixture that fails any omitted
process/network/signal mock and disables real numpy allocations. Hardware
commands, network requests, model loading and llama-server launches remain
mocked, including in negative controls.

## Still required outside Python coverage

The user authorised the real Phase 4 bandwidth checks. The installed wheel
successfully measures stream and gather in a network-disabled container with
no compiler. Three alternating launches per probe per mode were also recorded
on the available Ryzen 5950X host. The 8 GiB attempt detected swapping and
failed gather's median comparison; the 2 GiB repeat passes both median range
checks with no new swap-outs, but concurrent page-ins and the differing host
leave idle reference-machine acceptance pending. Every-sample containment
fails and is recorded separately. All samples and the method are retained in
[`phase4-bandwidth-acceptance.md`](phase4-bandwidth-acceptance.md).

Native ARM execution and remote CI also remain pending. The workflow implements
the isolated-runner/container checks, but has not been pushed or run remotely.
PyPI/GitHub setup, tags and publication remain pending under the earlier release
restriction. These manual C integration checks do not change Python coverage
or make pytest depend on a compiler, hardware measurements or the network.
