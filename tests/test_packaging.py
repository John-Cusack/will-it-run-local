"""Build hooks against a fake compiler; no network or binary execution."""
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
from setuptools import Distribution
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel


@pytest.fixture
def hooks(monkeypatch):
    import setuptools
    monkeypatch.setattr(setuptools, "setup", lambda **kw: None)
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))


def test_build_compiles_baseline_executable(hooks, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.setenv("CC", "mock-cc -fno-omit-frame-pointer")

    def compile(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"built probe")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", compile)
    command = hooks["BuildPy"](Distribution())
    command.build_lib = str(tmp_path)
    command.run()
    binary = tmp_path / "wirl" / "_bin" / "membw"
    assert binary.read_bytes() == b"built probe"
    assert binary.stat().st_mode & 0o111
    assert calls[0][:4] == ["mock-cc", "-fno-omit-frame-pointer", "-O2", "-pthread"]
    assert "-march=native" not in calls[0]


def test_missing_compiler_removes_stale_binary(hooks, monkeypatch, tmp_path):
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("WIRL_REQUIRE_PROBE", raising=False)
    monkeypatch.setattr(hooks["shutil"], "which", lambda name: None)
    target = tmp_path / "wirl" / "_bin" / "membw"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous architecture")
    command = hooks["BuildPy"](Distribution())
    command.build_lib = str(tmp_path)
    with pytest.warns(UserWarning, match="no C compiler"):
        command.run()
    assert not target.exists()


def test_release_build_requires_compiler(hooks, monkeypatch, tmp_path):
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setenv("WIRL_REQUIRE_PROBE", "1")
    monkeypatch.setattr(hooks["shutil"], "which", lambda name: None)
    command = hooks["BuildPy"](Distribution())
    command.build_lib = str(tmp_path)
    with pytest.raises(RuntimeError, match="no C compiler"):
        command.run()


def test_compile_failure_cannot_make_a_release_wheel(hooks, monkeypatch, tmp_path):
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.setenv("CC", "mock-cc")
    def failed(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"partial output")
        return subprocess.CompletedProcess(cmd, 1, "", "synthetic build failure")

    monkeypatch.setattr(subprocess, "run", failed)
    command = hooks["BuildPy"](Distribution())
    command.build_lib = str(tmp_path)
    with pytest.raises(RuntimeError, match="compile failed.*synthetic build failure"):
        command.run()
    assert not (tmp_path / "wirl" / "_bin" / "membw").exists()


def test_wheel_has_no_python_abi_tag(hooks, monkeypatch):
    monkeypatch.setattr(bdist_wheel, "finalize_options", lambda self: None)
    monkeypatch.setattr(bdist_wheel, "get_tag", lambda self: ("cp312", "cp312", "linux_aarch64"))
    command = hooks["PlatformWheel"](Distribution())
    command.finalize_options()
    assert not command.root_is_pure
    assert command.get_tag() == ("py3", "none", "linux_aarch64")


def test_binary_distribution_installs_in_platlib(monkeypatch, tmp_path):
    import setuptools
    from setuptools.command.install import install
    from setuptools.warnings import SetuptoolsDeprecationWarning
    options = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kw: options.update(kw))
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))
    distribution = options.get("distclass", Distribution)()
    # Wheel creation uses this command internally; no legacy install is run.
    with pytest.warns(SetuptoolsDeprecationWarning, match="setup.py install is deprecated"):
        command = install(distribution)
    command.install_purelib = str(tmp_path / "purelib")
    command.install_platlib = str(tmp_path / "platlib")
    command.ensure_finalized()
    assert Path(command.install_lib) == Path(command.install_platlib)


def test_sdist_metadata_rejects_non_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(SystemExit, match="will-it-run-local supports Linux only"):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))


def test_editable_install_keeps_runtime_compilation(hooks, monkeypatch, tmp_path):
    monkeypatch.setattr(build_py, "run", lambda self: None)

    def unexpected(*a, **kw):
        pytest.fail("editable install unexpectedly invoked the compiler")

    monkeypatch.setattr(subprocess, "run", unexpected)
    command = hooks["BuildPy"](Distribution())
    command.build_lib = str(tmp_path)
    command.editable_mode = True
    command.run()
    assert not (tmp_path / "wirl" / "_bin" / "membw").exists()


def test_sdist_metadata_rejects_unsupported_architecture(hooks, monkeypatch):
    monkeypatch.setattr(hooks["platform"], "machine", lambda: "riscv64")
    with pytest.raises(SystemExit, match="supports Linux x86_64 and aarch64 only"):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))
