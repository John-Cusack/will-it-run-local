# Worked example: the machine this tool was built on

A real, running configuration rather than a template — every number in the
comments came from `wirl auto`, and the whole thing is what the reference
machine actually serves from.

```
CPU     AMD EPYC 7B12, 64 physical cores, Zen 2 (AVX2+FMA, no AVX-512)
RAM     215 GiB of 256 installed (one channel dark)
GPU     RTX 3090, 24 GiB, SM_86
Model   DeepSeek-V4-Flash-0731 abliterated, MXFP4, 156 GB
        43 layers, 256 experts, top-6
Draft   matched DSpark drafter, Q8_0, 10.9 GB  (arch `dflash`)
Serving ~10.4 tok/s decode, 20.7 GB VRAM, 3.4 GB headroom
```

## `reference-config/`

| file | |
|---|---|
| `llamacpp-abl.sh` | the launcher, with every flag chosen by measurement |
| `llamacpp.service` | systemd **user** unit; comments record why each value was picked and what the alternatives measured |

Both are annotated with the measurements behind each choice, so the reasoning
survives when someone (including me, in six months) asks why `n_max` is 2.

Install:

```bash
cp llamacpp.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llamacpp
# so it survives logout:
sudo loginctl enable-linger "$USER"
```

### The bind address is the security decision

The unit binds this host's **Tailscale** address (`100.66.109.44`), not
`0.0.0.0`. llama.cpp takes a single `--host`, and `0.0.0.0` would also listen on
the LAN interface. Binding the tailscale0 address instead means the API is
reachable from tailnet devices and from containers on the Docker bridge (which
route to it through the host), while the LAN gets nothing:

```
192.168.0.135:30001 -> refused      # LAN
172.18.0.1:30001    -> refused      # docker bridge
127.0.0.1:30001     -> refused      # loopback
100.66.109.44:30001 -> 200          # tailnet only
```

Change the address to `127.0.0.1` unless you have the same arrangement, and
verify with `ss -tlnp` and a curl against each interface rather than assuming.

**Tailscale Funnel is deliberately not used** — that publishes a service to the
public internet. This is tailnet-only.

### Authentication

`--api-key-file` points at a `0600` file rather than putting the key on the
command line or in the unit, both of which any local user can read via `/proc`
or `systemctl cat`. Without a key the server returns `401`.

## `probes/`

Small scripts that answer questions `wirl` does not, kept because each one
found something.

| script | what it establishes |
|---|---|
| `model-qa.py` | 7 checks that the model still answers correctly, keeps reasoning out of the reply, and does not leak tool-call markup. Run it after **any** config change — a faster server that produces worse text is not an improvement. |
| `prefill-probe.py` | time-to-first-token as a chat client really behaves. Showed a growing conversation prefills 5195 tokens on turn 1 and then 37, 10, 10 — i.e. prompt caching already works and the scary cold number is not what you live with. |
| `switch-probe.py` | whether alternating between two conversations destroys the cache. It does not: `--cache-ram` keeps both warm. |
| `divergence-probe.py` | where prefix caching stops helping. Editing near the END reprocessed 488 tokens; editing near the START reprocessed 3900. That is the case `--cache-reuse` exists for, and the case it cannot help here. |

A caution about `model-qa.py`, learned the hard way: its first version capped
test 3 at 300 tokens, the model ran past that showing its working, and a
truncated reply scored as a wrong answer. A harness that reports a regression
that is not there is worse than no harness.
