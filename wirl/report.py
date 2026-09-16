"""Human-readable output helpers."""

from __future__ import annotations

import sys

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
YEL = "\033[33m"
GRN = "\033[32m"
OFF = "\033[0m"


def _colour():
    return sys.stdout.isatty() and "NO_COLOR" not in __import__("os").environ


def c(text, code):
    return f"{code}{text}{OFF}" if _colour() else text


def head(text):
    print()
    print(c(text, BOLD))
    print(c("-" * len(text), DIM))


def gib(n):
    return n / (1 << 30)


def gb(n):
    return n / 1e9


STATUS_COLOUR = {"ok": GRN, "warn": YEL, "fail": RED}


def print_checks(checks, show_ok=True):
    for ch in checks:
        if ch.status == "ok" and not show_ok:
            continue
        tag = c(f"{ch.status.upper():4}", STATUS_COLOUR.get(ch.status, ""))
        print(f"  [{tag}] {ch.name}: {ch.detail}")
        if ch.fix and ch.status != "ok":
            for line in _wrap(ch.fix, 74):
                print(f"         {c(line, DIM)}")


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if line and len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def para(text, width=78, indent="  "):
    for line in _wrap(text, width):
        print(indent + line)
