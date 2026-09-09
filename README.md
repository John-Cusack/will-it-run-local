# will-it-run-local

**Will this model run on my machine, how fast, and what should I stop tuning?**

Ollama and LM Studio answer *"will it fit"*. This answers *"what is the fastest
configuration that fits, and is it worth tuning at all"* — which is a different
question, and one nobody automates.

It works from first principles: read the GGUF tensor table, measure what the
machine actually delivers, and do the arithmetic. Then it verifies the answer by
measurement, because a prediction that has not been checked is a guess with
decimal places.

```console
$ wirl plan DeepSeek-V4-Flash-Q4-mxfp4.gguf --draft dspark-Q8_0.gguf --ctx 16384

  --n-cpu-moe   tok/s     VRAM     fits    of which CPU
      43        9.61   20.15 GiB   yes      92%  <-- recommended
      42        9.81   23.34 GiB   yes      92%
      41       10.02   26.52 GiB    NO      92%

Verdict
-------
  Bandwidth-bound: 92% of each token is spent reading experts from system RAM at
  45 GB/s. Thread count and most other flags will not move this.

  The only large wins available are: more VRAM (offload more layers), faster
  RAM, or a smaller quantisation.
```

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

| | |
|---|---|
| `wirl probe` | What the machine actually is, including **measured** bandwidth |
| `wirl inspect MODEL` | Bytes per token, expert routing, roofline |
| `wirl plan MODEL` | Recommended config, without running anything |
| `wirl doctor` | The silent failures, before they cost you a day |
| `wirl check-draft` | Vet a draft model **before downloading it** |
| `wirl tune MODEL` | Measure real configs, emit a launcher and systemd unit |
| `wirl bandwidth` | Bandwidth only |

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

### 2. Measure bandwidth. Never trust the spec sheet.

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

### 3. `--n-cpu-moe` is the knob that matters

It sets how many leading layers keep their experts in system RAM. Each layer
moved to the GPU takes VRAM and removes DRAM traffic. Predicted VRAM comes from
the tensor table plus the KV cache — including the compressed-latent (MLA)
layout DeepSeek-style models use — so it is exact rather than a rule of thumb.

For a **dense** model there are no routed experts to leave behind, so
`--n-cpu-moe` does nothing and `wirl` switches to `--n-gpu-layers` instead —
whole layers move to the GPU, taking their share of the KV cache with them. The
arithmetic is the same; only the knob changes.

**Validated against measurement on the reference machine:**

| `--n-cpu-moe` | predicted | measured | | predicted VRAM | measured VRAM |
|---|---|---|---|---|---|
| 43 | 9.61 tok/s | **9.89** | | 20.15 GiB | **20.09** |
| 42 | 9.81 tok/s | **10.10** | | 23.34 GiB | **23.28** |
| 37 | — | *failed to allocate* | | 39.27 GiB | *predicted not to fit* ✓ |

Within 3% on throughput and 0.3% on VRAM, and it predicted the allocation
failure. These are regression tests in `tests/test_predict.py`, so the model
cannot drift away from reality unnoticed.

### 4. Headroom is not superstition

The fastest fitting config sat at **97%** of VRAM and died several thousand
tokens into a real conversation. Compute buffers grow with context. `wirl` keeps
3 GB back by default, which cost 2% throughput and bought a server that stays up.

---

## Findings this encodes

Things measured here that I could not find written down anywhere.

### Deeper speculative decoding loses on sparse MoE

Everyone assumes more draft tokens is better. llama.cpp's default is 3.

| `--spec-draft-n-max` | tok/s | acceptance |
|---|---|---|
| **1** | **8.20** | 0.725 |
| 2 | 8.03 | 0.593 |
| 3 | 7.42 | 0.490 |
| 5 | 6.18 | 0.356 |
| *off* | *6.65* | — |

**Depth 5 is slower than not drafting at all.** Why: verifying a 2.46-token
batch cost 331 ms against 150 ms for a single token — 2.2× the cost for 2.46×
the tokens. With top-6-of-256 routing, each extra draft token pulls in *mostly
different* experts, so expert reads scale almost linearly with batch size.
Nothing amortises. `wirl tune --depth-sweep` searches shallow-first and stops
as soon as depth stops paying.

### Thread count is a null result

| threads | tok/s |
|---|---|
| 64 | 8.84 |
| 48 | 8.83 |
| **32** | **8.89** |
| 24 | 8.81 |

0.9% spread — smaller than the run-to-run variance *within* a single setting.
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

**In:** llama.cpp, GGUF, single GPU, Linux, CPU+GPU hybrid MoE and dense models.

**Out:** vLLM, SGLang, TGI, multi-GPU, tensor parallelism, ROCm, Windows,
fine-tuning. These are deliberate. Trying to cover every runtime is how tools
like this die.

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
pytest                    # 59 tests, no model files or GPU required
```

Tests use synthetic GGUFs built in-process. The regression tests in
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
