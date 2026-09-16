"""Check standalone-probe wheel repair without running a memory benchmark.

Run inside a manylinux build image with auditwheel installed, or choose a
platform compatible with the local compiler's glibc using --plat.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import platform
from pathlib import Path
import shlex
import subprocess
import tempfile
from zipfile import ZipFile, ZipInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plat", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "wirl" / "csrc" / "membw.c"
    architecture = platform.machine()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        binary = Path(temporary) / "membw"
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                       ["-O2", "-pthread", str(source), "-o", str(binary)], check=True)
        info = "wirl_probe_spike-0.0.0.dist-info"
        files = {
            "wirl/_bin/membw": binary.read_bytes(),
            f"{info}/METADATA": b"Metadata-Version: 2.1\nName: wirl-probe-spike\nVersion: 0.0.0\n",
            f"{info}/WHEEL": ("Wheel-Version: 1.0\nGenerator: local-spike\n"
                              f"Root-Is-Purelib: false\nTag: py3-none-linux_{architecture}\n").encode(),
        }
    record = io.StringIO()
    writer = csv.writer(record)
    for name, data in files.items():
        writer.writerow((name, "", len(data)))
    writer.writerow((f"{info}/RECORD", "", ""))
    files[f"{info}/RECORD"] = record.getvalue().encode()
    wheel = args.output / f"wirl_probe_spike-0.0.0-py3-none-linux_{architecture}.whl"
    with ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            entry = ZipInfo(name)
            entry.external_attr = (0o100755 if name.endswith("/membw") else 0o100644) << 16
            archive.writestr(entry, data)
    subprocess.run(["auditwheel", "show", str(wheel)], check=True)
    subprocess.run(["auditwheel", "repair", "--plat", args.plat, "-w",
                    str(args.output / "repaired"), str(wheel)], check=True)


if __name__ == "__main__":
    main()
