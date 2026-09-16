"""Compare installed and locally compiled probes on authorised hardware.

Three alternating launches per probe and mode, using the normal three
repetitions. Keep raw samples; reject timed faults, swap-outs and instability.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

from wirl import membw, probe
from wirl.lock import benchmark_lock


def compare_samples(local, prebuilt):
    if len(local) != 3 or len(prebuilt) != 3:
        raise ValueError("comparison needs three launches per probe")
    for runs in (local, prebuilt):
        if any(not run or any(not math.isfinite(v) or v <= 0 for v in run) for run in runs):
            raise ValueError("comparison needs finite, positive samples in every launch")
    local_values = [v for run in local for v in run]
    prebuilt_values = [v for run in prebuilt for v in run]
    local_medians = [statistics.median(run) for run in local]
    prebuilt_medians = [statistics.median(run) for run in prebuilt]
    local_median = statistics.median(local_medians)
    prebuilt_median = statistics.median(prebuilt_medians)
    local_span = max(local_values) - min(local_values)
    local_spread = 100 * local_span / local_median
    prebuilt_spread = 100 * (max(prebuilt_values) - min(prebuilt_values)) / prebuilt_median
    delta = abs(prebuilt_median - local_median)
    # Requiring every new point inside three baseline extrema rejects 80% of
    # equal continuous distributions. Compare the effect with measured noise,
    # as tune does, and refuse wide noise rather than letting it mask a shift.
    within_spread = delta <= local_span
    stable = max(local_spread, prebuilt_spread) <= 10
    return {"local_launch_medians": local_medians, "prebuilt_launch_medians": prebuilt_medians,
            "local_median": local_median, "prebuilt_median": prebuilt_median,
            "median_difference_gb_per_second": delta,
            "median_difference_pct": 100 * delta / local_median,
            "local_span_gb_per_second": local_span,
            "local_spread_pct": local_spread, "prebuilt_spread_pct": prebuilt_spread,
            "within_local_spread": within_spread, "stable": stable,
            "passed": within_spread and stable}


def snapshot():
    return {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "memory": probe.mem_info(), "swap_activity": probe.swap_activity(),
            "load_average": list(os.getloadavg())}


def collect(gib=None, reps=3):
    packaged = membw.PACKAGED_PROBE
    if not os.path.isfile(packaged) or not os.access(packaged, os.X_OK):
        raise RuntimeError("install a platform wheel outside the checkout before comparing probes")
    cores = probe.physical_core_cpus()
    if not cores:
        raise RuntimeError("cannot determine one allowed CPU per physical core")
    gib = membw._buffer_gib(gib)
    if reps < 1:
        raise ValueError("repetitions must be positive")
    previous_cache = os.environ.get("XDG_CACHE_HOME")
    with tempfile.TemporaryDirectory(prefix="wirl-comparison-") as cache:
        os.environ["XDG_CACHE_HOME"] = cache
        try:
            local = membw.build_probe(force=True)
            if not local:
                raise RuntimeError("cannot compile the local reference probe")
            paths = {"prebuilt": packaged, "local": local}
            cpu = probe.cpu_info()
            cpu.pop("flags", None)
            data = {"host": cpu, "parameters": {"threads": len(cores), "cores": cores,
                                               "gib": gib, "reps": reps},
                    "source_sha256": hashlib.sha256(Path(membw.CSRC).read_bytes()).hexdigest(),
                    "binary_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                      for name, path in paths.items()},
                    "criterion": "absolute median difference <= local raw peak-to-peak spread; "
                                 "both raw spreads <= 10%; no new swap-outs or timed major faults",
                    "before": snapshot(), "runs": [], "summary": {}, "passed": False}
            for mode in ("stream", "gather"):
                for launch in range(1, 4):
                    for name, path in paths.items():
                        before = snapshot()
                        try:
                            result = membw._measure_probe(path, mode, len(cores), gib, reps, cores)
                            if len(result.samples) != reps:
                                raise RuntimeError(f"probe returned {len(result.samples)} samples; expected {reps}")
                        except RuntimeError as error:
                            data["error"] = str(error)
                            data["failed_launch"] = {"mode": mode, "probe": name, "launch": launch,
                                                     "before": before, "after": snapshot()}
                            data["after"] = snapshot()
                            return data
                        data["runs"].append({"mode": mode, "probe": name, "launch": launch,
                                             "samples": result.samples, "before": before,
                                             "after": snapshot()})
                        print(f"{mode} {name} launch {launch}: "
                              + ", ".join(f"{v:.3f}" for v in result.samples) + " GB/s", flush=True)
                values = {name: [r["samples"] for r in data["runs"]
                                 if r["mode"] == mode and r["probe"] == name] for name in paths}
                data["summary"][mode] = compare_samples(values["local"], values["prebuilt"])
            data["after"] = snapshot()
            delta = {k: data["after"]["swap_activity"][k] - data["before"]["swap_activity"][k]
                     for k in ("pswpin", "pswpout")}
            data["swap_delta"] = delta
            data["passed"] = (all(s["passed"] for s in data["summary"].values())
                              and delta["pswpout"] <= 0)
            return data
        finally:
            if previous_cache is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = previous_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="save all samples and diagnostics as JSON")
    parser.add_argument("--gib", type=int, help="optional buffer, subject to unused RAM headroom")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--wait", action="store_true", help="wait for the shared benchmark lock")
    args = parser.parse_args(argv)
    try:
        with benchmark_lock(wait=args.wait):
            data = collect(args.gib, args.reps)
    except (RuntimeError, ValueError) as error:
        data = {"passed": False, "error": str(error)}
    Path(args.output).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data.get("summary", {}), indent=2))
    if not data["passed"]:
        print(data.get("error", "comparison failed: check spreads and swap counters in JSON"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
