# Test coverage

Measured on 2026-09-16 after Phases 1–3: 174 passing tests, **68.45% line
coverage** (1610/2352 statements) and **53.03% branch coverage** (385/726).
There are 742 unexecuted statements and 341 unexecuted branch outcomes. No
source files or error paths were excluded from this measurement.

After Phase 4's mocked packaging/runtime tests: **187 tests pass**, with
**69.00% line coverage** (1625/2355 statements) and **54.12% branch coverage**
(394/728). Reaching 100% still requires covering **730 statements** and
**334 branch outcomes**, chiefly in the same areas listed below. Both local
measurements use Python 3.12.3; the latest totals also reproduce exactly on
Python 3.14.2. CI collects JSON/HTML reports on Python 3.14.

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
CI uploads `coverage.json` and the HTML report as the `python-coverage` artifact
without applying a threshold the suite does not yet meet. The combined coverage
percentage also includes branches; use the separate statement/branch fields
in JSON when assessing progress towards 100%.

## Work required for 100%

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

Add behavioural assertions for the missed paths, then rerun the report and
close the remaining branch outcomes. Keep all published predictor figures
unchanged. Use fake files and processes throughout; hardware commands, network
requests, model loading and llama-server launches must be mocked, including in
negative controls. Do not add exclusions or weaken assertions to obtain 100%.

Once all measured gaps are closed, add `--cov-fail-under=100` to a dedicated CI
coverage check and explicitly verify both line and branch totals in the JSON
report. The current suite does not meet that threshold; no false 100% gate has
been enabled.
