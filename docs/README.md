# Reference sweeps

Raw, unedited output from `wirl auto` on the reference machine
(AMD EPYC 7B12, 64 cores / RTX 3090 24 GiB / Linux 7.0, llama.cpp f114f91).

Kept verbatim so the claims in the top-level README can be checked against
what the tool actually printed, including the parts that were wrong at the
time and have since been fixed.

| file | what it is |
|---|---|
| `reference-sweep-moe.log` | DeepSeek-V4-Flash-0731 MXFP4 (156 GB, 43 layers, 256 experts, top-6) with a 10.9 GB DSpark drafter. Full scope: VRAM bisection, throughput, draft depth, thread count, long-context profile, and a second context length. |
| `reference-sweep-dense.log` | Qwen3-1.7B Q4_K_M (1.1 GB, 28 layers, dense) — the dense `--n-gpu-layers` path. |

Two bugs were found by these runs and fixed afterwards, so the logs still show
the old output:

- the MoE log ends with *"Measured 10.43 tok/s is 202% of the 5.17 tok/s
  roofline. There is essentially nothing left to tune"*. Beating the roofline
  means the estimate is wrong, not that the machine is at its limit.
- the dense run (an earlier version of `reference-sweep-dense.log`) said
  *"Something is wrong: check swap"* for a model sitting entirely in VRAM,
  where the bandwidth roofline does not apply at all.

Both now print an accurate explanation instead. See `tests/test_predict.py`.
