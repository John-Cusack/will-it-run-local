# Phase 4 bandwidth acceptance checks

Measured on 2026-09-16 after the user explicitly authorised the bandwidth tests.
No llama-server was launched and no existing service was stopped or restarted.
All samples, probe identities, raw output and swap counters are retained in
[`phase4-bandwidth-results.json`](phase4-bandwidth-results.json).

**Current status: corrected host comparison and compiler-free measurements
pass.** The initial attempts below are retained as history. Sections under
“Diagnosis and fixes” describe the current guards and comparison protocol.
The EPYC reference and native ARM execution remain pending.

## Host and method

The available host reports an AMD Ryzen 9 5950X, 16 physical cores, 32 logical
CPUs, one NUMA node, 64 MiB of L3 and 62.71 GiB of kernel-visible RAM. This
differs from the documented reference EPYC 7B12 with 64 cores and 215 GiB RAM.
These results do not replace its published measurements or predictor tests.
Other applications remained running; the initial load averages were
1.29 / 1.38 / 1.15.

Installed the repaired manylinux x86_64 wheel in a throwaway Python 3.14 venv
and ran from outside the checkout. Normal `build_probe()` selected its packaged
executable; `build_probe(force=True)` compiled the identical installed C source
with `/usr/bin/cc` (GCC 11.5.0), `-O2 -pthread` and an isolated temporary cache.
Neither build uses `-march=native`. The prebuilt executable's compiler comments
include GCC 14.2.1 and the manylinux GCC 8.5.0 startup objects.

For each mode, ran prebuilt, local, prebuilt, local, prebuilt, local, sequentially.
Each launch made one measurement, with 16 threads pinned to CPUs 0–15 (one
allowed logical CPU per physical core). Both binaries received identical
arguments. Buffer allocation and page faults occur before the C timing window.
GB/s uses the probe's reported logical byte count divided by elapsed seconds.

Declared the numerical comparison before running: the prebuilt median must lie
in the inclusive minimum/maximum range of the three local samples. Also recorded
whether **every** prebuilt sample lies in that range, rather than concealing
outliers. That stricter condition fails in both modes in both attempts.

## First attempt: 8 GiB, invalid for acceptance

The tool's default buffer calculation selected 8 GiB from 25.20 GiB available
RAM. Despite this margin, the host recorded 162350 swap-out pages and 30942
swap-in pages during the comparison. These are system-wide counters; they
establish concurrent swapping, not which process owned the affected pages.
The buffer allocations coincided with increased memory pressure. This attempt
cannot establish acceptance, including the stream result whose median passed.

| mode | prebuilt samples (GB/s) | local samples (GB/s) | prebuilt / local median | median in local range |
|---|---|---|---|---|
| stream | 26.87, 28.35, 28.36 | 27.88, 28.45, 28.66 | 28.35 / 28.45 | yes |
| gather | 21.24, 22.16, 21.52 | 22.72, 22.22, 22.05 | 21.52 / 22.22 | no |

Gather's median was 3.12% lower than local, whose peak-to-peak spread was 3.00%.
Retained this failed comparison rather than treating it as passing noise.

## Second attempt: 2 GiB, median comparisons pass

Reduced both buffers to 2 GiB to avoid further allocation pressure. This is
32 times the combined L3 capacity and keeps the working set larger than cache.
Retained the same threads, affinity, launch order and numerical criterion.
This is a documented departure from the automatic buffer size, motivated by
the measured swapping; the runtime defaults and C kernels are unchanged.

| mode | prebuilt samples (GB/s) | local samples (GB/s) | prebuilt / local median | local range (GB/s) | median in local range |
|---|---|---|---|---|---|
| stream | 28.10, 28.38, 27.72 | 27.88, 28.33, 27.97 | 28.10 / 27.97 | 27.88–28.33 | yes |
| gather | 23.19, 22.09, 22.32 | 22.01, 23.11, 21.75 | 22.32 / 22.01 | 21.75–23.11 | yes |

The median differences were +0.46% for stream and +1.39% for gather. Local
peak-to-peak spreads were 1.63% and 6.20%. There were zero additional swap-out
pages, but 567 swap-in pages, so this shared-host run still does not establish
an idle-machine comparison. Passing these median checks is evidence of similar
measurements on this host, not full Phase 4 acceptance.

## Compiler-free wheel measurements: pass

Ran the existing CI container command using `python:3.12-slim-bookworm`, with
network access disabled and wheel/checker mounts read-only. Installed the wheel
using `pip --no-index --no-deps`, asserted that cc/gcc/clang were absent, then
ran `python /checks/check_installed_probe.py --measure`.

The checker selected the packaged executable and successfully measured:

| mode | threads | buffer | repetitions | bandwidth |
|---|---:|---:|---:|---:|
| stream | 1 | 1 GiB | 1 | 17.42 GB/s |
| gather | 1 | 1 GiB | 1 | 4.38 GB/s |

This passes the installation/measurement Done-when check. The one-thread,
smaller-buffer results are a functional smoke check and are not compared with
the 16-thread host numbers. The container exited successfully; there were no
new swap-outs and 164 system-wide swap-in pages during this check.

To reproduce on authorised hardware, install the same wheel outside the
checkout, obtain the two paths from `build_probe()` and
`build_probe(force=True)` with `CC=/usr/bin/cc` and a throwaway `XDG_CACHE_HOME`,
then alternate these commands three times per mode:

```text
<prebuilt-path> <stream-or-gather> 16 2 1 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
<local-path>    <stream-or-gather> 16 2 1 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

Adapt the core list and buffer to the actual host; record swap counters and
retain failures. The container command is checked into
`.github/workflows/ci.yml` under “Measure from a wheel in a container without a
compiler”. The wheel SHA-256 and container image ID are in the results JSON.

## Diagnosis and fixes

Investigation found no cgroup memory.high/memory.max limits or reclaim events,
and ordinary VM watermarks with swappiness 60. The first allocation started
with only 758 MiB unused RAM. The kernel's
[MemAvailable documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html)
describes an estimate including reclaimable cache, not an unused-memory
reservation. Allocating 8 GiB coincided with reclaim, swap-outs and pressure
despite the estimated 25 GiB margin. Inventory now reports MemFree too, and
the buffer budget retains the 35% fraction while bounding both available and
unused RAM. Unsafe explicit buffers and default buffers below 2 GiB are refused
before measurement. The new automatic buffer on this host is 2 GiB.

The original prebuilt and local stream/gather inner-loop instruction bytes
match exactly (15 and 59 bytes respectively); hashes and bytes are retained in
the JSON. This points away from a compiler regression as the cause of failure.

Strict containment was an unreliable acceptance rule. For three points from
each of two equal independent continuous distributions, every assignment of
three “local” labels among the six sorted points is equally likely. There
are 20 assignments; both extremes are local in only four. Requiring all
prebuilt points inside the local range therefore rejects 80% of equal
distributions. The old median-in-range gate rejects eight of the 20 assignments.
Neither is a sound interpretation of “differences below measured spread are
noise”. A mocked test retains the enumeration.

The reusable `tools/compare_probe_bandwidth.py` now makes three alternating
launches per binary per mode, using the normal three repetitions. It retains
all nine samples per probe per mode, compares median launch medians, and
requires their absolute difference to be at most the local raw peak-to-peak
spread. Either probe's raw spread above the existing 10% bandwidth instability
threshold invalidates the comparison. This is an engineering noise check
matching `tune`, not a statistical proof of equivalence. Genuine shifts larger
than noise, unstable and incomplete output have explicit rejection tests.

The first guarded rerun aborted after one global page-in with zero swap-outs.
That partial attempt is also retained. A system-wide counter cannot establish
that the measured workload faulted. The final C probe checks major faults in
its own process across each timing window using
[getrusage(RUSAGE_SELF)](https://www.man7.org/linux/man-pages/man2/getrusage.2.html),
which includes all its threads. Both diagnostic calls sit outside the clock
interval, so neither worker kernel nor byte count changes. The numpy fallback
gets equivalent scoped checks. New global swap-outs still invalidate allocation
pressure; background page-in counts remain recorded without an invented
numeric allowance.

A temporary LD_PRELOAD shim simulated a major-fault increment without actually
faulting host memory. The old C probe returned measurements in both modes;
the new probe exited with an invalid-result diagnostic instead. The check also
simulated failures of each getrusage call, which correctly fail closed. The
shim source and all six integration outcomes are retained in JSON. Ordinary
pytest uses only mocked processes/counters/arrays and needs no compiler or
hardware measurements.

### Updated wheel comparison: pass

Rebuilt/repaired and installed the new manylinux x86_64 wheel outside the
checkout. Ran the tool with its default 2 GiB buffer, 16 pinned physical-core
threads and three repetitions per launch. Every repetition is retained; no
first sample was discarded. Zero new swap-outs occurred and the probe detected
no timed major faults. Four background page-in pages are recorded.

| mode | prebuilt median (GB/s) | local median (GB/s) | median difference | local raw spread | result |
|---|---:|---:|---:|---:|---|
| stream | 28.929 | 28.950 | 0.07% | 3.10% | pass |
| gather | 21.533 | 21.834 | 1.38% | 8.06% | pass |

The updated wheel also passes the compiler-free, network-disabled container
measurements: stream 21.18 GB/s and gather 4.52 GB/s (one thread, 1 GiB,
one repetition), with zero swap-ins or swap-outs during that smoke check.

For future runs, install a platform wheel in a throwaway venv and invoke the
checked-in tool from outside the checkout on authorised hardware:

```bash
<venv>/bin/python <repo>/tools/compare_probe_bandwidth.py --output /tmp/comparison.json
```

Its exit status reflects numerical/stability/fault acceptance and its JSON
contains all samples and partial-failure diagnostics. Its source compilation
uses a throwaway cache and restores the caller's XDG_CACHE_HOME setting.
It takes the shared benchmark lock; an occupied-lock integration check
confirms that it refuses before compilation or measurement. `--wait` opts
into waiting for the current benchmark to finish.

## Remaining acceptance

The idle EPYC reference-machine comparison and native aarch64 wheel execution
remain pending. No local ARM runner is available. Remote workflow execution,
publication setup, tags and pushing have not been authorised or performed.

Plain `pytest -q` from the repository root passes all 402 tests on Python
3.9.25, 3.12.3 and 3.14.2 in throwaway development venvs after these checks.
Python line and branch coverage remains exactly 100% (2386 statements and 736
branch outcomes), with no exclusions. The rebuilt wheel's installed container
suite passes 400 and skips the existing real-permission check under root.
The source archive includes the comparison tool and all tests, and passes
all 402 against the installed repaired wheel on 3.14.2 outside the checkout.
