#!/usr/bin/env python3
"""Where prefix caching stops helping.

Prefix caching only reuses an exact common prefix. If a conversation is edited
near the beginning -- or a client injects a system prompt that changes -- the
prefix diverges early and everything after it is reprocessed, even though it is
mostly identical. --cache-reuse is supposed to salvage that via KV shifting.
"""
import sys
sys.path.insert(0, "/home/john/bench")
from prefill_probe import ask, big

BASE = big(3000)
cases = [
    ("baseline (cold)",        BASE),
    ("same again",             BASE),
    ("edit near the END",      BASE[:-200] + " The turbocharger spools quickly. "),
    ("edit near the START",    "Note: revised. " + BASE),
]
for label, body in cases:
    r = ask([{"role": "user", "content": body + "\n\nSay OK."}], max_tokens=6)
    print(f"  {label:22} prompt_n={r['prompt_n']:>6}  prefill={r['prompt_ms']/1000:6.1f}s")
