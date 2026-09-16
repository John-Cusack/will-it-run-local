"""Launch llama-server, measure it honestly, shut it down.

Reported throughput comes from llama.cpp's own `timings.predicted_per_second`
in the completion response, not from wall-clock around an HTTP call, so it
excludes connection setup and prompt processing.

Every configuration is run several times and the spread is reported alongside
the mean. A single number hides the thing you most need to know: on the
reference machine an identical command line produced 12.13 tok/s and 10.05
tok/s hours apart, and only the spread made that visible.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field

BENCH_PROMPT = ("Write a detailed technical description of how a four-stroke "
                "internal combustion engine works. Be thorough.")


@dataclass
class RunConfig:
    model: str
    n_cpu_moe: int | None = None
    n_gpu_layers: int = 99
    threads: int | None = None
    ctx: int = 16384
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: bool = True
    draft_model: str | None = None
    draft_n_max: int = 1
    draft_ngl: int = 99
    host: str = "127.0.0.1"
    port: int = 38080
    extra: list = field(default_factory=list)
    gpu_uuid: str | None = None

    def argv(self, binary: str) -> list:
        a = [binary, "--model", self.model, "--host", self.host,
             "--port", str(self.port), "--ctx-size", str(self.ctx),
             "--n-gpu-layers", str(self.n_gpu_layers),
             "--cache-type-k", self.cache_type_k,
             "--cache-type-v", self.cache_type_v,
             "--parallel", "1", "--no-warmup"]
        if self.n_cpu_moe is not None:
            a += ["--n-cpu-moe", str(self.n_cpu_moe)]
        if self.threads:
            a += ["--threads", str(self.threads), "--threads-batch", str(self.threads)]
        if self.flash_attn:
            a += ["--flash-attn", "on"]
        if self.draft_model:
            a += ["--spec-draft-model", self.draft_model,
                  "--spec-draft-ngl", str(self.draft_ngl),
                  "--spec-draft-n-max", str(self.draft_n_max)]
        return a + self.extra

    def label(self) -> str:
        bits = [f"ncmoe={self.n_cpu_moe}" if self.n_cpu_moe is not None
                else f"ngl={self.n_gpu_layers}",
                f"t={self.threads or 'auto'}"]
        if self.draft_model:
            bits.append(f"nmax={self.draft_n_max}")
        if self.ctx != 16384:
            bits.append(f"ctx={self.ctx}")
        return " ".join(bits)


@dataclass
class RunResult:
    config: RunConfig
    samples: list                  # tok/s per repetition
    prefill: list
    peak_vram: int
    startup_s: float
    ok: bool
    error: str | None = None

    @property
    def mean(self):
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def spread_pct(self):
        if len(self.samples) < 2 or not self.mean:
            return 0.0
        return 100.0 * (max(self.samples) - min(self.samples)) / self.mean


def find_server(explicit=None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    env = os.environ.get("WIRL_LLAMA_SERVER")
    if env and os.path.exists(env):
        return env
    which = shutil.which("llama-server")
    if which:
        return which
    for p in ("~/llama.cpp/build/bin/llama-server",
              "~/src/llama.cpp/build/bin/llama-server",
              "/usr/local/bin/llama-server", "/opt/llama.cpp/bin/llama-server"):
        p = os.path.expanduser(p)
        if os.path.exists(p):
            return p
    return None


def gpu_used_bytes(uuid=None) -> int:
    from .probe import gpu_info
    for g in gpu_info():
        if (g.get("uuid") == uuid if uuid is not None else g["index"] == 0):
            return g["vram_used"]
    return 0


def _wait_health(host, port, proc, timeout):
    """Wait until the server can actually generate, not merely until it binds.

    /health returns 200 while the model is still loading, and a completion
    request at that point comes back as
    `{"error": {"message": "Loading model", "code": 503}}`. Trusting /health
    alone reports a server ready seconds after launch and makes every timing
    that follows wrong -- and, worse, makes a slow-loading configuration look
    like a fast one. So the readiness check is a real generation.
    """
    url = f"http://{host}:{port}/health"
    t0 = time.monotonic()
    bound = False
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            return None
        if not bound:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    bound = r.status == 200
            except (urllib.error.URLError, OSError):
                pass
        if bound:
            try:
                tps, _, _ = _generate(host, port, 1, timeout=60)
                if tps is not None:
                    return time.monotonic() - t0
            except Exception:                                # noqa: BLE001
                pass                                         # still loading
        time.sleep(2.0)
    return None


# Filler used to build long prompts. Real prose rather than repeated tokens, so
# the tokeniser behaves as it would on actual chat history.
_FILLER = (
    "The engine management system continuously samples intake air temperature, "
    "manifold pressure, and crankshaft position, then adjusts injector pulse "
    "width and spark advance to hold the mixture near stoichiometric. Under "
    "transient load the controller leans on a feed-forward model because the "
    "oxygen sensor lags the event it is meant to correct. "
)


def build_prompt(approx_tokens):
    """A prompt of roughly `approx_tokens` tokens.

    The exact count does not need to be hit: the server reports what it
    actually processed, and that measured figure is what gets recorded.
    """
    if approx_tokens <= 0:
        return BENCH_PROMPT
    words = _FILLER.split()
    # ~0.75 words per token for English prose.
    need = max(1, int(approx_tokens * 0.75))
    out = []
    while len(out) < need:
        out.extend(words)
    return " ".join(out[:need]) + "\n\nSummarise the passage above in detail."


def _generate(host, port, n_tokens, prompt=None, timeout=3600):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt or BENCH_PROMPT}],
        "max_tokens": n_tokens, "temperature": 1.0, "top_p": 1.0,
        "stream": False, "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    t = d.get("timings", {})
    return (t.get("predicted_per_second"), t.get("prompt_per_second"),
            t.get("prompt_n"))


class ServerFailed(RuntimeError):
    """The server never became healthy. Almost always VRAM."""


@contextmanager
def server(cfg: RunConfig, binary: str, startup_timeout=1200, log_dir=None,
           verbose=True):
    """Run a server for the duration of the block, then always tear it down.

    Exposed separately from run_config because launches are expensive (20-140 s)
    while requests are cheap. Anything that can be answered by asking a running
    server -- prefill speed, decode with a long history -- should be, rather
    than paying for another startup.
    """
    argv = cfg.argv(binary)
    logf = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logf = open(os.path.join(
            log_dir, f"llama-{cfg.n_cpu_moe}-{cfg.threads}-{cfg.draft_n_max}-"
                     f"{cfg.ctx}-{int(time.time())}.log"), "wb")
    env = os.environ.copy()
    if cfg.gpu_uuid:
        env["CUDA_VISIBLE_DEVICES"] = cfg.gpu_uuid
    proc = subprocess.Popen(argv, stdout=logf or subprocess.DEVNULL,
                            stderr=subprocess.STDOUT, start_new_session=True, env=env)
    try:
        started = _wait_health(cfg.host, cfg.port, proc, startup_timeout)
        if started is None:
            rc = proc.poll()
            raise ServerFailed(f"server did not become healthy (exit={rc}); "
                               f"see log in {log_dir or 'devnull'}")
        if verbose:
            print(f"    ready in {started:.0f}s", flush=True)
        yield started
    finally:
        _stop(proc)
        if logf:
            logf.close()


def run_config(cfg: RunConfig, binary: str, reps=3, n_tokens=400,
               startup_timeout=1200, log_dir=None, verbose=True) -> RunResult:
    """Start a server, measure it `reps` times, stop it. Always cleans up."""
    samples, prefill, peak = [], [], 0
    try:
        with server(cfg, binary, startup_timeout, log_dir, verbose) as started:
            for i in range(reps):
                tps, pre, _ = _generate(cfg.host, cfg.port, n_tokens)
                if tps:
                    samples.append(tps)
                if pre:
                    prefill.append(pre)
                peak = max(peak, gpu_used_bytes(cfg.gpu_uuid))
                if verbose:
                    print(f"    rep{i}: {tps:.2f} tok/s", flush=True)
            return RunResult(cfg, samples, prefill, peak, started, True)
    except ServerFailed as e:
        return RunResult(cfg, [], [], 0, 0.0, False, str(e))
    except Exception as e:                                  # noqa: BLE001
        return RunResult(cfg, samples, prefill, peak, 0.0, False,
                         f"{type(e).__name__}: {e}")


def _stop(proc, grace=60):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except OSError:
        proc.terminate()
    t0 = time.monotonic()
    while time.monotonic() - t0 < grace:
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=30)
    # VRAM is not released the instant the process dies; the next config will
    # mis-fit if we start it too early.
    time.sleep(3.0)
