# Phase 4 packaging spike

`tools/spike_probe_wheel.py` compiles the unchanged C source with `-O2 -pthread`,
puts the standalone executable in a minimal platform wheel, and asks auditwheel
to inspect and repair it. It never runs a memory benchmark.

```bash
pip install auditwheel patchelf
python tools/spike_probe_wheel.py --output /tmp/wirl-spike --plat manylinux_2_38_x86_64
```

Verified on this machine on 2026-09-16 with auditwheel 6.8.2:

- The wheel contains an executable, without a Python extension module.
- Auditwheel recognises it as a platform wheel and repairs it successfully.
- Repair preserves `py3-none` and changes the platform to
  `manylinux_2_38_x86_64`.
- Asking for `manylinux_2_34_x86_64` fails because the local compiler emits
  references to glibc 2.38. Relabelling a local build would be incorrect; release
  wheels must be compiled inside a manylinux image.

The planned release image is `manylinux_2_28`, on native x86_64 and aarch64
runners. Auditwheel's repair support and cibuildwheel's Linux repair command
are documented by [auditwheel](https://github.com/pypa/auditwheel) and
[cibuildwheel](https://cibuildwheel.pypa.io/en/stable/options/#repair-wheel-command).
GitHub documents the native
[`ubuntu-24.04-arm` runner](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

ARM execution and the alternating reference-machine bandwidth comparison
remain acceptance checks. No ARM runner is available locally, and the user has
not lifted the restriction on real bandwidth benchmarks.

## Actual setuptools/cibuildwheel build

The first full build exposed an extra requirement: setting a platform wheel
tag and `Root-Is-Purelib: false` alone still sent the package to
`.data/purelib`, because setuptools considered the distribution pure Python.
Auditwheel rejected the executable in that directory. `BinaryDistribution`
now selects the platlib scheme, and a mocked regression verifies that routing.

With cibuildwheel 4.2.1 and the manylinux_2_28 x86_64 image, the repaired wheel
contains `wirl/_bin/membw` with executable permissions and these tags:

```text
py3-none-manylinux_2_17_x86_64
py3-none-manylinux2014_x86_64
py3-none-manylinux_2_28_x86_64
```

This is auditwheel's measured compatibility result, rather than a manually
applied older tag. The source archive contains the C source, build hooks and
tests, and excludes generated binaries. Twine's strict metadata check passes.

The safe local build overrides CI's measurement command:

```bash
CIBW_TEST_COMMAND='python {project}/tools/check_installed_probe.py && pytest -q {project}/tests' \
  cibuildwheel --platform linux --output-dir /tmp/wirl-wheels
```

The default checker executes only the C program's argument-error path, which
returns before any allocation. The built wheel installs and starts that path
in a network-disabled `python:3.12-slim-bookworm` container, with actual compiler
absence checked separately. CI uses `--measure` on isolated native runners,
including a compiler-free container, to exercise both kernels. Those CI runs
and the reference-machine comparison are still pending.
