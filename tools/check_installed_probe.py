"""Verify a wheel's prebuilt probe without a compiler.

The default only runs the usage path, which allocates no benchmark buffer.
--measure is for isolated CI runners, never a machine serving a model.
"""
from __future__ import annotations

import argparse
import os
import subprocess

from wirl import membw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    os.environ.pop("CC", None)
    membw.shutil.which = lambda name: None
    binary = membw.build_probe()
    if binary != membw.PACKAGED_PROBE:
        raise RuntimeError("installed wheel has no usable prebuilt bandwidth probe")
    result = subprocess.run([binary], capture_output=True, text=True)
    if result.returncode != 2 or "usage: membw" not in result.stderr:
        raise RuntimeError(f"prebuilt probe could not start: {result.stderr}")
    print(f"prebuilt probe starts without a compiler: {binary}")
    if args.measure:
        for mode in ("stream", "gather"):
            result = membw.measure(mode, threads=1, gib=1, reps=1)
            if not result.samples or result.best <= 0:
                raise RuntimeError(f"prebuilt {mode} probe produced no measurement")
            print(f"{mode}: {result.best:.2f} GB/s")


if __name__ == "__main__":
    main()
