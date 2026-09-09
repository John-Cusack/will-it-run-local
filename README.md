# will-it-run-local

**You have a CPU and one consumer GPU. The model you want needs ten times your
VRAM, and you are not spending five figures on an RTX 6000 Pro to run it.**

**You probably don't have to.** The reference machine here runs a **156 GB**
model on a **24 GB** RTX 3090 at **~10 tokens/sec**, by keeping the parts that
are rarely read in system RAM. `wirl` works out the fastest way to do that on
*your* hardware — and, just as usefully, tells you when to stop trying.

That headline works because the model is a *mixture of experts*: only 2.3% of
its weights are read per token, so most of it can sit in cheap RAM. A dense
model of that size would crawl. `wirl` handles both, and says plainly which one
you are holding — including when the honest answer is "not on this hardware".

Ollama and LM Studio answer *"will it fit"*. This answers *"what is the fastest
configuration that fits"* — a different question, and one nobody automates.

**It answers it by measuring, not by estimating.** One command probes your
hardware, finds the real VRAM limit by launching the server until it stops
starting, measures throughput at each surviving configuration, and writes you a
launcher.

```console
$ wirl auto DeepSeek-V4-Flash-Q4-mxfp4.gguf --draft dspark-Q8_0.gguf \
           --depth-sweep --thread-sweep --ctx 16384

4. VRAM boundary (measured)
---------------------------
    trying --n-cpu-moe 41 ... does not start
    trying --n-cpu-moe 42 ... starts (23971 MiB)
  smallest --n-cpu-moe that starts: 42

Measured results
----------------
| config               | tok/s | spread | peak VRAM |
|----------------------|-------|--------|-----------|
| ncmoe=42 t=32 nmax=1 |  9.89 |  1.3%  | 23963 MiB |
| ncmoe=43 t=32 nmax=1 |  9.90 |  4.2%  | 20699 MiB |
| ncmoe=43 t=32 nmax=2 | 10.43 |  5.2%  | 20735 MiB |
| ncmoe=43 t=32 nmax=3 |  9.80 |  6.8%  | 20760 MiB |
| ncmoe=43 t=24 nmax=1 | 10.01 |  2.6%  | 20721 MiB |
| ncmoe=43 t=48 nmax=1 |  9.90 |  3.5%  | 20699 MiB |
| ncmoe=43 t=64 nmax=1 |  9.92 |  5.0%  | 20699 MiB |

Same configuration, separate launches:
  ncmoe=43 t=32 nmax=1 measured 3x: 9.90, 10.01, 10.08 tok/s (1.8% apart)
So treat differences below about 2% in the table above as noise, not findings.

Long-context behaviour (measured)
---------------------------------
| history | prefill | time to first token | decode |
|---------|---------|---------------------|--------|
| 23 tok  | 17 tok/s|   1.4 s             | 10.15  |
| 1887 tok| 59 tok/s|  32.0 s             | 10.60  |
| 7509 tok| 63 tok/s| 119.7 s             | 10.83  |

Decode is unaffected by context depth (10.15 -> 10.83 tok/s at 7509 tokens).
The KV cache is on the GPU, where re-reading it is cheap.

**120 seconds to the first token** at 7509 tokens of history. On a long
conversation this, not decode speed, is what you will actually feel.
```

Every number there came from a server that started and generated tokens. There
is a cost model in here, and it is accurate, but it is used for exactly one
thing: **deciding which handful of configurations are worth launching**, so a
sweep is four runs instead of forty-four.

---

## Why this exists

I spent two days getting one 284B mixture-of-experts model running well on one
machine. Almost none of that time went where I expected.

The obvious knobs — how many layers to offload, how many threads — took about
twenty minutes and are the only things existing tools help with. The time went
to a set of failures that were all **silent**: the server started, tokens came
out, and the number was simply wrong.

- A kernel autotuning cache, written while two benchmarks ran at once, pinned a
  slow kernel **permanently — across restarts and reboots**. Cost: 38%
  throughput, found three days later.
- A memory channel was dead. Nothing reports this. Bandwidth was halved and
  every measurement taken afterwards was quietly invalid.
- A draft model downloaded at 7 GB declared an architecture llama.cpp does not
  implement, and could never have loaded.
- Deeper speculative decoding made things **slower**, opposite to the default
  and to every piece of advice I could find.
- A trailing space in a config URL — `http://host:30001/v1 ` — produced only
  "no models available".

Each check in this tool exists because one of those cost real time. The one
output I would most have wanted on day one was not a config file. It was a
sentence: *you are at 84% of your memory bandwidth, tuning buys you nothing,
your DIMM is dead, go fix that instead.*

## Install

```bash
git clone https://github.com/John-Cusack/will-it-run-local
cd will-it-run-local
pip install -e .
```

Python 3.9+. **No runtime dependencies** — this runs on machines that are
already short on memory and should never be the reason an environment breaks.
A C compiler is used to build the bandwidth probe on first run; without one it
falls back to numpy and says so.

## Commands

| | measures? | |
|---|---|---|
| **`wirl auto MODEL`** | **yes** | **The main command.** Probe, check, sweep, emit a launcher |
| `wirl tune MODEL` | yes | Just the sweep, if you want to drive it yourself |
| `wirl probe` | yes | What the machine actually is, including bandwidth |
| `wirl bandwidth` | yes | Bandwidth only |
| `wirl doctor` | — | The silent failures, before they cost you a day |
| `wirl inspect MODEL` | — | Bytes per token, expert routing, roofline |
| `wirl check-draft` | — | Vet a draft model **before downloading it** |
| `wirl recommend REPO` | — | Which quantisation to download, for *your* machine |
| `wirl find-draft MODEL` | — | Search for a compatible speculative-decoding drafter |
| `wirl plan MODEL` | no | Estimate only. Prefer `auto` |

---

## How it works

### 1. Decode speed is a bandwidth problem, not a compute problem

Generating one token reads every weight it needs **exactly once**. So:

```
seconds_per_token  =  cpu_bytes / cpu_bandwidth  +  gpu_bytes / gpu_bandwidth
```

That is the entire model. It is accurate to a few percent, and it explains why
almost every knob people agonise over does nothing.

For a dense model, `bytes_per_token` is the whole file. For a mixture of
experts, only the routed experts that fire are read — which is why a 156 GB MoE
decodes faster than a 70 GB dense model:

```console
$ wirl inspect DeepSeek-V4-Flash-Q4-mxfp4.gguf
  architecture   deepseek4
  size           155.97 GB
  experts        256 routed, 6 active per token (2.3%)

  bytes per token   11.192 GB (7.2% of the file)
  per layer         3.423 GB of experts resident, 80 MB read per token
                    153 MB attention/shared/norms, read every token
```

This comes from the **tensor table**, not from an architecture lookup — routed
experts are identified by llama.cpp's `_exps` naming and their expert-count
dimension. A model family this tool has never seen is priced correctly, and
shared experts (`_shexp`, which fire on *every* token) are correctly excluded
from the discount.

### 2. Measure bandwidth — but know what it does and does not predict.

This machine's CPU advertises 204.8 GB/s. It delivers **45**, and on a bad boot
36. Two 40-line C kernels settle it in seconds:

```console
$ wirl bandwidth        # reference machine, idle, model resident in page cache
  stream    45.3 GB/s peak    45.0 median   spread  0.7%   (64 threads, 32 GiB)
  gather    15.3 GB/s peak    15.1 median   spread  1.3%   (64 threads, 32 GiB)
```

`stream` is the optimistic bound and predicts reading a contiguous expert
tensor. `gather` defeats the prefetcher and exposes real DRAM latency.

Run it while something else is using the machine and you get 22.6 GB/s on the
same hardware — which is the point. If repetitions disagree, the machine is not
stable and **no inference benchmark taken right now means anything**; `wirl`
says so rather than reporting a mean.

**A caveat found by measurement, not reasoning.** Later in this machine's life
the probe read 21 GB/s while the model sustained ~36 GB/s of expert traffic —
i.e. inference *beat* the roofline by 2x. Three different probe kernels agreed
on 21 GB/s, so the probe was right and the cost model was wrong: it charges
every routed expert to DRAM on every token, but with 256 MB of L3 and skewed
top-6-of-256 routing, hot experts stay cached and are never re-read. `wirl` now
detects this and says the estimate is wrong rather than claiming you are at the
wall. Treat the roofline as a floor for MoE models, not a ceiling.

### 3. `--n-cpu-moe` is the knob that matters, and it is found by launching

It sets how many leading layers keep their experts in system RAM. Each layer
moved to the GPU takes VRAM and removes DRAM traffic.

`wirl auto` finds the boundary **empirically**, by bisecting on whether the
server actually starts. That costs about three launches for a 0–43 range. A
predicted boundary is a guess about allocator behaviour, compute-buffer growth
and driver overhead; the real one is wherever `llama-server` stops starting,
and that is cheap to find.

Then it measures throughput at the boundary and a few configs above it, because
the trade you actually have to make is throughput against VRAM headroom — and
that trade needs real numbers on both sides.

#### Launches are expensive; requests are not

A server start costs 20–140 s. A request costs seconds. That asymmetry decides
the whole design:

| axis | needs a relaunch? | how it is handled |
|---|---|---|
| `--n-cpu-moe` | yes | bisect for the edge (~3 launches), then measure a few above it |
| `--spec-draft-n-max` | yes | shallow-first, stop as soon as it stops paying |
| `--threads` | yes | opt-in; usually a null result |
| `--ctx-size` | yes | an input, not a sweep — but `--ctx2` measures a second value |
| **prefill speed** | **no** | measured on the winning server, free |
| **decode with a full context** | **no** | measured on the winning server, free |

Those last two are the ones that decide whether a setup is actually pleasant to
use, and they are the ones most benchmarks omit — including every number in the
"Findings" section below, which were all taken with a ~20-token prompt.
`wirl auto` measures decode at 0, 2K and 8K tokens of history, and reports
time-to-first-token at each, because "10 tok/s" means something very different
when the prompt took 50 seconds to process.

#### Why there is a cost model at all

Only to decide what to launch. Sweeping `--n-cpu-moe` 0–43 blind, three
repetitions each, is over a hundred server startups. The model narrows that to a
bracket, and every reported number still comes from a real run.

It earns that job by being accurate. Predicted VRAM comes from the tensor table
plus the KV cache — including the compressed-latent (MLA) layout DeepSeek-style
models use — rather than a rule of thumb:

For a **dense** model there are no routed experts to leave behind, so
`--n-cpu-moe` does nothing and `wirl` switches to `--n-gpu-layers` instead —
whole layers move to the GPU, taking their share of the KV cache with them. The
arithmetic is the same; only the knob changes.

**Checked against measurement on the reference machine:**

| `--n-cpu-moe` | predicted | measured | | predicted VRAM | measured VRAM |
|---|---|---|---|---|---|
| 43 | 9.61 tok/s | **9.89** | | 20.15 GiB | **20.09** |
| 42 | 9.81 tok/s | **10.10** | | 23.34 GiB | **23.28** |
| 37 | — | *failed to allocate* | | 39.27 GiB | *predicted not to fit* ✓ |

Within 3% on throughput and 0.3% on VRAM, and it predicted the allocation
failure. These are regression tests in `tests/test_predict.py`, so the model
cannot drift away from reality unnoticed.

That is good enough to aim a sweep with — and **not** good enough to ship as an
answer. It is calibrated on one machine, in one regime (92% of each token read
from system RAM). `wirl auto` reports it alongside the measured result purely so
you can see how far off it was on *your* hardware, and says outright which of
the two to trust.

### 4. Headroom is not superstition

The fastest fitting config sat at **97%** of VRAM and died several thousand
tokens into a real conversation. Compute buffers grow with context. `wirl` keeps
3 GB back by default, which cost 2% throughput and bought a server that stays up.

---

## Findings this encodes

Things measured here that I could not find written down anywhere.

### Deeper speculative decoding stops paying quickly on sparse MoE

Everyone assumes more draft tokens is better. llama.cpp's default is 3. Two
measurements on the same machine, on different model/drafter pairs:

| `--spec-draft-n-max` | Q4_K_XL + Q2_K drafter | MXFP4 + Q8_0 drafter |
|---|---|---|
| *off* | 6.65 | — |
| 1 | **8.20** | 10.01 |
| 2 | 8.03 | **10.43** |
| 3 | 7.42 | 9.80 |
| 5 | 6.18 | — |

**The optimum is not the same for both pairs** — 1 on the first, 2 on the
second — and the gap between them is only a few percent either way. What *is*
consistent is the shape: it peaks shallow and falls off, and on the first pair
depth 5 was slower than not drafting at all.

Why: verifying a 2.46-token batch cost 331 ms against 150 ms for a single token
— 2.2× the cost for 2.46× the tokens. With top-6-of-256 routing, each extra
draft token pulls in *mostly different* experts, so expert reads scale almost
linearly with batch size. Nothing amortises.

The practical lesson is not "use depth 1". It is that this is worth 4% and the
right value is model-specific, so **measure it** — `wirl auto --depth-sweep`
searches shallow-first and stops as soon as depth stops paying. On the first
pair I originally recorded depth 1 as the winner over depth 2 on a 0.05 tok/s
difference, which was well inside the noise and should not have been called.

### Decode speed is the wrong thing to optimise for chat

The headline number everyone quotes is decode throughput at an empty context.
On the reference machine that is ~10 tok/s and it **does not change at all**
between an empty context and 7,500 tokens of history. What does change is the
wait before the first token:

| history | time to first token |
|---|---|
| 23 tokens | 1.4 s |
| 1,887 tokens | 32 s |
| 7,509 tokens | **120 s** |

Two minutes of silence before a reply starts, on a conversation that is not
even especially long. Prefill plateaus around 63 tok/s because a 512-token
batch routes to essentially every one of the 256 experts, so each batch re-reads
the whole expert set.

**But that is the cold number, and it is easy to mislead yourself with it.**
Those probes send a fresh prompt every time. llama.cpp caches prompts by
default, so a real chat pays it only once. Measured on a conversation that grows
the way a chat client actually sends one:

| turn | tokens prefilled | prefill |
|---|---|---|
| 1 (long paste) | 5,195 | 79.2 s |
| 2 | 37 | 4.1 s |
| 3 | 10 | 0.7 s |
| 4 | 10 | 0.7 s |

I originally reported the 120 s as the headline daily-use problem. It is not —
that reading came from my own benchmark passing `cache_prompt: false`. What
actually still costs full price is narrower: **the first long prompt**, and
**editing near the start of a conversation**, which diverges the prefix and
forces a full reprocess (3,900 tokens / 59 s in one test).

Two flags matter here, and they are not interchangeable:

- **`--cache-ram`** (default 8192 MiB) holds finished prompt caches in host RAM
  so switching between conversations does not re-prefill. The default already
  covers ~6 full 16K conversations on the reference model. Raising it to 24 GiB
  measured **slower** — 9.79 vs 10.25 tok/s over 6 runs each, −4.5%, consistent
  in direction across two tests, cause not established. So: only raise it if you
  actually keep more chats than the default holds, and A/B it when you do.
- **`--cache-reuse`** salvages a diverged prefix by KV-shifting. It is
  **silently disabled** on any model whose context cannot shift — sliding-window
  attention, for one. On the reference model llama.cpp logged
  `cache_reuse is not supported by this context, it will be disabled` and
  carried on. `wirl doctor --server-log` checks for exactly that.

The general lesson stands even though my first conclusion did not: measure
prefill at a realistic prompt length **and with caching behaving as your client
will**, or you will fix a problem you do not have.

### Thread count is a null result

| threads | Q4_K_XL pair | MXFP4 pair |
|---|---|---|
| 24 | 8.81 | 10.01 |
| **32** | **8.89** | **10.08** |
| 48 | 8.83 | 9.90 |
| 64 | 8.84 | 9.92 |

0.9% and 1.8% spread — both smaller than the run-to-run variance *within* a
single setting, on two different model pairs measured weeks apart.
When this sweep is flat, that **is** the finding: you are memory bound, and 32
of your 64 cores are free for something else.

### Vet a draft model before downloading it

A GGUF stores its metadata — including the full token list — at the head of the
file. An HTTP Range request for the first few MB answers both compatibility
questions without downloading the model:

```console
$ wirl check-draft target.gguf --repo dev7a/DeepSeek-V4-Flash-DSpark-Drafter-GGUF \
                               --file ...-Q2_K-Q8_0-dflash.gguf
  fetching header over HTTP Range...
  target  deepseek4    vocab  129280 sha a31e23b927c61c03
  draft   dflash       vocab  129280 sha a31e23b927c61c03

  COMPATIBLE -- vocabularies match exactly.
```

It hashes the **token list**, not the token count: two models can share a count
and differ in content, and that pair produces garbage rather than an error. It
also flags architectures llama.cpp does not register — which ruled out three of
four candidate drafters, none of which could ever have loaded.

### CPU-only costs about 4×

Measured 2.21 tok/s with no GPU against 8.79 for a hybrid split. The prediction
is not simply "all bytes at RAM speed" — with nothing offloaded, router and
attention work stops being overlapped with expert reads and stops being free.
`wirl` derates by 0.70 to match measurement, and says so in the code.

---

## Measurement discipline

The part people skip, and the part that decides whether the answer is real.

**Never benchmark two things at once.** This is not about noise. On the
reference machine, two concurrent benchmarks caused a Triton-based runtime to
autotune under contention, pick bad kernels, and **write them to a persistent
cache**. It survived restarts and reboots and cost 38% throughput until it was
found days later, after the wrong conclusion had already been drawn about which
runtime was faster.

So `wirl tune`:

- takes a global `flock`, and refuses rather than queues
- refuses to start when another process holds GPU memory (`--force` to override)
- runs each config several times and **reports the spread, not just the mean**
- flags configs whose repetitions disagree by more than 8% and tells you not to
  compare anything until it settles
- always tears the server down, and waits for VRAM to actually be released

That last point matters more than it sounds. An identical command line produced
12.13 tok/s and 10.05 tok/s hours apart on this machine. Only the spread made
that visible; a single number would have been reported as fact.

### `wirl doctor`

```console
$ wirl doctor
  [OK  ] cpu-isa: AMD EPYC 7B12: avx2, fma. No AVX-512 -- expected on Zen 2 and
         earlier; not a problem for bandwidth-bound decode.
  [FAIL] gpu-exclusive: other processes are using the GPU: llama-server (21090 MB)
  [WARN] dimm-population: EDAC reports 256 GiB installed but the kernel sees
         216 GiB -- a 40 GiB gap.
         That gap is about the size of one or more DIMMs. If a memory channel
         is unpopulated you lose bandwidth in proportion, and no software
         setting recovers it.
  [WARN] kv-cache-type: --cache-type-k is 'q8_0'.
         Quantising the K cache has produced corrupt output on some
         architectures (llama.cpp #25382) and fails silently.
```

Every check maps to a real failure:

| check | the failure it catches |
|---|---|
| `dimm-population` | a dead memory channel, reported by nothing else |
| `gpu-exclusive` | benchmarking beside other GPU work |
| `autotune-cache` | a stale Triton/inductor cache pinning a slow kernel |
| `swap` | weights in swap, decoding at disk speed |
| `kv-cache-type` | `--cache-type-k q8_0` producing silently corrupt output |
| `ram-capacity` | a model that will thrash rather than fail cleanly |
| `transparent-hugepages` | khugepaged stalls on a >100 GB mmap |

---

## Output

`wirl tune --emit ./run-llama.sh` writes a launch script and a systemd unit,
with the reasoning in comments so the numbers are still explicable in six
months:

```bash
#!/usr/bin/env bash
# Generated by will-it-run-local.
# Re-run `wirl tune` after any hardware or llama.cpp change.
#
# Chosen by measurement: 9.89 tok/s over 3 runs (spread 6.3%), 20570 MiB peak VRAM.
# Roofline for this model on this machine was 9.61 tok/s at 45 GB/s measured RAM bandwidth.
set -euo pipefail

# Pin to physical cores only. Hyperthread siblings add
# contention without bandwidth on a memory-bound workload.
exec taskset -c 0-63 /home/john/llama.cpp/build/bin/llama-server \
  --model /srv/llm/models/.../DeepSeek-V4-Flash-Q4-mxfp4-0731.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 16384 \
  --n-gpu-layers 99 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --parallel 1 \
  --no-warmup \
  --n-cpu-moe 43 \
  --threads 32 \
  --threads-batch 32 \
  --flash-attn on \
  --spec-draft-model /srv/llm/models/.../dspark-Q8_0.gguf \
  --spec-draft-ngl 99 \
  --spec-draft-n-max 1
```

alongside a systemd user unit with `Restart=on-failure`, a 900 s start timeout
for a cold page cache, and `OOMScoreAdjust=-500` so the kernel does not reach
for the process that just mapped 150 GB.

## Scope

**In:** llama.cpp, GGUF, a single NVIDIA GPU, Linux, CPU+GPU hybrid, both
mixture-of-experts and dense models, with or without speculative decoding.

**Out, deliberately:** vLLM, SGLang, TGI, multi-GPU, tensor parallelism, ROCm,
Apple Silicon, Windows, fine-tuning, and wiring up a chat UI. Trying to cover
every runtime is how tools like this die.

**Tested on:** one machine (EPYC 7B12 + RTX 3090, Linux 7.0), against one MoE
model and one dense model. Everything else is untested — if you run it
somewhere different, the `probe` output and a `tune` result in an issue are the
most useful thing you can send.

### Known limitations

Stated plainly, because a tool about honest measurement should be honest about
itself:

- **The calibration constants come from one machine**, and that machine has a
  dead memory channel. `DEFAULT_EFFICIENCY`, `CPU_ONLY_DERATE` and
  `CUDA_OVERHEAD` in `predict.py` are fitted to it. The *measured* paths do not
  depend on them; the estimates do.
- **The cost model under-predicts MoE throughput**, sometimes by 2x, because it
  charges every routed expert to DRAM on every token and ignores cache reuse.
  `wirl auto` detects and reports this rather than pretending otherwise, but
  the roofline is a floor for MoE, not a ceiling.
- **Nothing is calibrated for the GPU-bound regime.** Once a model fits in VRAM,
  per-token cost is kernel launch and attention, not bandwidth. The tool says
  "fast" and declines to give a number rather than quoting a roofline that is
  several times optimistic.
- **`wirl recommend` has not been validated end-to-end** — its predictions come
  from remote headers and have never been checked against downloading the
  recommended file and measuring it.
- **Single GPU only.** With several present it uses index 0 unless told
  otherwise; multi-GPU splits are not modelled at all.
- **CI is configured but has not yet run** on GitHub, only locally.

### What you need before this is any use

| | |
|---|---|
| `llama-server` built with CUDA | `wirl doctor` checks for it and tells you how |
| a GGUF on disk | `wirl recommend <hf-repo>` will tell you which one to download |
| a free GPU | `wirl auto` refuses to sweep next to other GPU work |

## A worked example

[`examples/`](examples/) holds the actual running configuration from the
reference machine — the launcher, the systemd unit, and the probe scripts —
annotated with the measurement behind every flag, so the reasoning survives
being read six months later.

It also includes `model-qa.py`, which checks the model still answers correctly,
keeps reasoning out of the reply, and does not leak tool-call markup. **Run
something like it after every config change.** A faster server that produces
worse text is not an improvement, and throughput benchmarks cannot see the
difference.

## Reference machine

Every measured number here comes from one machine, and is labelled where it
appears. Treat them as evidence that the method works, not as your numbers.

```
CPU     AMD EPYC 7B12, 64 physical cores, Zen 2, AVX2+FMA, no AVX-512
RAM     215 GiB of 256 installed (one channel dark), ~45 GB/s measured
GPU     RTX 3090, 24 GiB, SM_86
Model   DeepSeek-V4-Flash-0731, 43 layers, 256 experts, top-6, MXFP4, 156 GB
Runtime llama.cpp f114f91
```

The bandwidth figure on that machine drifted from 107 GB/s to 36 GB/s across
three boots with no configuration change, which is why bandwidth is measured on
every run rather than cached. That CPU is being replaced.

## Development

```bash
pip install -e '.[dev]'
pytest                    # 92 tests, no model files or GPU required
```

Tests use synthetic GGUFs built in-process and mock every server launch. The regression tests in
`tests/test_predict.py` assert the predictor still reproduces the measurements
above, so a change that looks better on paper but drifts from reality fails.

The GGML type table in `wirl/ggml_types.py` is generated from llama.cpp itself
via `tools/dump_ggml_types.c` rather than copied from documentation — deprecated
type IDs are deliberately absent so a file using one raises instead of being
silently priced as zero bytes.

## Contributing

Measurements from other hardware are the most useful contribution — especially
Intel with AVX-512, Apple Silicon unified memory, and multi-channel DDR5, all of
which will have different bandwidth-to-compute ratios and may break assumptions
baked in here. Open an issue with `wirl probe` output and a measured `wirl tune`
result.

## Licence

MIT
