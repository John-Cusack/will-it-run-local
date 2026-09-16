"""Build a Python-independent Linux probe alongside the CLI."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import warnings

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel


if sys.platform != "linux":
    sys.exit("will-it-run-local supports Linux only")
if platform.machine() not in ("x86_64", "aarch64"):
    sys.exit("will-it-run-local supports Linux x86_64 and aarch64 only")


class BuildPy(build_py):
    def run(self):
        super().run()
        if self.editable_mode:
            return
        target = Path(self.build_lib) / "wirl" / "_bin" / "membw"
        # Reusing a build directory must not carry an old architecture's binary
        # into a source-only wheel when the compiler has disappeared.
        if target.exists():
            target.unlink()
        compiler = (os.environ.get("CC") or shutil.which("cc")
                    or shutil.which("gcc") or shutil.which("clang"))
        if not compiler:
            reason = "no C compiler: wheel has no prebuilt bandwidth probe; runtime compilation requires a compiler"
            if os.environ.get("WIRL_REQUIRE_PROBE") == "1":
                raise RuntimeError(reason)
            warnings.warn(reason)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).resolve().parent / "wirl" / "csrc" / "membw.c"
        # Keep the same measured kernels and optimisation level; native ISA
        # flags would make this wheel unusable on older machines.
        command = shlex.split(compiler) + ["-O2", "-pthread", str(source), "-o", str(target)]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except OSError as e:
            raise RuntimeError(f"bandwidth probe compile failed ({compiler}): {e}") from None
        if result.returncode:
            if target.exists():
                target.unlink()
            detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
            raise RuntimeError(f"bandwidth probe compile failed ({compiler}): {detail}")
        target.chmod(0o755)


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # The executable must be installed as platform data. A wheel tag alone
        # leaves it in purelib, which auditwheel rejects even for a valid ELF.
        return True


setup(distclass=BinaryDistribution,
      cmdclass={"build_py": BuildPy, "bdist_wheel": PlatformWheel})
