"""Minimal GGUF reader: metadata KV, tensor table, multi-shard, and remote.

Everything this tool predicts is derived from the tensor table, not from an
architecture-specific formula. That means a model family we have never seen is
priced correctly as long as it follows llama.cpp's tensor naming.

The remote path uses HTTP Range requests so a drafter can be vetted for
vocabulary and architecture compatibility before downloading tens of GB of it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import urllib.request
from dataclasses import dataclass, field

from .ggml_types import tensor_nbytes, type_name

GGUF_MAGIC = b"GGUF"

(T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL,
 T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64) = range(13)

_FMT = {T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
        T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
        T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d"}


class _Reader:
    """Sequential byte source over a local file or an HTTP Range endpoint."""

    def __init__(self, src, chunk=8 << 20):
        self.pos = 0
        self.chunk = chunk
        if isinstance(src, str) and src.startswith(("http://", "https://")):
            self.url = src
            self.fh = None
            self.buf = b""
            self.buf_start = 0
        else:
            self.url = None
            self.fh = open(src, "rb")

    def _fetch(self, start, length):
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{start + length - 1}",
                     "User-Agent": "will-it-run-local/0.1"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    def read(self, n):
        if self.fh is not None:
            b = self.fh.read(n)
        else:
            end = self.pos + n
            if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
                want = max(n, self.chunk)
                self.buf = self._fetch(self.pos, want)
                self.buf_start = self.pos
            off = self.pos - self.buf_start
            b = self.buf[off:off + n]
        if len(b) != n:
            raise EOFError(f"short read: wanted {n}, got {len(b)} at {self.pos}")
        self.pos += n
        return b

    def close(self):
        if self.fh is not None:
            self.fh.close()


@dataclass
class Tensor:
    name: str
    dims: tuple
    type_id: int

    @property
    def type(self) -> str:
        return type_name(self.type_id)

    @property
    def nbytes(self) -> int:
        return tensor_nbytes(self.type_id, self.dims)

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass
class GGUF:
    path: str
    version: int
    kv: dict = field(repr=False)
    tensors: list = field(repr=False)

    # ---- convenience accessors -------------------------------------------

    @property
    def arch(self) -> str:
        return self.kv.get("general.architecture", "unknown")

    def a(self, suffix, default=None):
        """Read an architecture-scoped key, e.g. a('block_count')."""
        return self.kv.get(f"{self.arch}.{suffix}", default)

    @property
    def n_layer(self):
        return self.a("block_count")

    @property
    def n_embd(self):
        return self.a("embedding_length")

    @property
    def n_expert(self):
        return self.a("expert_count") or 0

    @property
    def n_expert_used(self):
        return self.a("expert_used_count") or 0

    @property
    def is_moe(self) -> bool:
        return bool(self.n_expert and self.n_expert > 1)

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def quant_mix(self) -> dict:
        out = {}
        for t in self.tensors:
            out[t.type] = out.get(t.type, 0) + t.nbytes
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def vocab_sig(self) -> dict:
        """Fingerprint used to decide whether a drafter can pair with a target.

        llama.cpp requires an identical vocabulary for speculative decoding.
        Comparing token counts alone is not enough -- two models can share a
        count and differ in content -- so we hash the token list itself.
        """
        toks = self.kv.get("tokenizer.ggml.tokens", [])
        h = hashlib.sha256("\x00".join(toks).encode("utf-8", "replace")).hexdigest()
        return {
            "arch": self.arch,
            "n_vocab": len(toks),
            "tok_model": self.kv.get("tokenizer.ggml.model"),
            "tok_pre": self.kv.get("tokenizer.ggml.pre"),
            "bos": self.kv.get("tokenizer.ggml.bos_token_id"),
            "eos": self.kv.get("tokenizer.ggml.eos_token_id"),
            "vocab_sha256_16": h[:16],
        }


def _read_value(r: _Reader, t: int):
    if t == T_STRING:
        (n,) = struct.unpack("<Q", r.read(8))
        return r.read(n).decode("utf-8", "replace")
    if t == T_ARRAY:
        (et,) = struct.unpack("<I", r.read(4))
        (n,) = struct.unpack("<Q", r.read(8))
        return [_read_value(r, et) for _ in range(n)]
    fmt = _FMT[t]
    return struct.unpack(fmt, r.read(struct.calcsize(fmt)))[0]


def read_one(src, want_tensors=True) -> GGUF:
    """Parse a single GGUF file (local path or URL)."""
    r = _Reader(src)
    try:
        if r.read(4) != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {src}")
        (version,) = struct.unpack("<I", r.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", r.read(16))

        kv = {}
        for _ in range(n_kv):
            (n,) = struct.unpack("<Q", r.read(8))
            key = r.read(n).decode("utf-8", "replace")
            (t,) = struct.unpack("<I", r.read(4))
            kv[key] = _read_value(r, t)

        tensors = []
        if want_tensors:
            for _ in range(n_tensors):
                (n,) = struct.unpack("<Q", r.read(8))
                name = r.read(n).decode("utf-8", "replace")
                (ndim,) = struct.unpack("<I", r.read(4))
                dims = struct.unpack(f"<{ndim}Q", r.read(8 * ndim))
                (tid,) = struct.unpack("<I", r.read(4))
                struct.unpack("<Q", r.read(8))  # data offset, unused here
                tensors.append(Tensor(name, dims, tid))
        return GGUF(str(src), version, kv, tensors)
    finally:
        r.close()


_SHARD_RE = re.compile(r"^(?P<stem>.*)-(?P<idx>\d{5})-of-(?P<tot>\d{5})\.gguf$")


def shard_paths(path: str):
    """Expand a first-shard path into every shard of the set.

    A sharded model only carries its full tensor table across all files, so
    pricing the first shard alone under-reports the model by 4-5x.
    """
    base = os.path.basename(path)
    m = _SHARD_RE.match(base)
    if not m:
        return [path]
    d = os.path.dirname(path)
    tot = int(m.group("tot"))
    out = []
    for i in range(1, tot + 1):
        p = os.path.join(d, f"{m.group('stem')}-{i:05d}-of-{tot:05d}.gguf")
        if not os.path.exists(p):
            raise FileNotFoundError(f"shard {i} of {tot} missing: {p}")
        out.append(p)
    return out


def read(path: str) -> GGUF:
    """Parse a model, following shards when the filename declares them."""
    paths = shard_paths(path) if not str(path).startswith("http") else [path]
    first = read_one(paths[0])
    for p in paths[1:]:
        first.tensors.extend(read_one(p).tensors)
    first.path = paths[0]
    if len(paths) > 1:
        first.kv["_wirl.shards"] = len(paths)
    return first


def read_header_only(src) -> GGUF:
    """Metadata without the tensor table -- enough for a vocab/arch check."""
    return read_one(src, want_tensors=False)
