#!/usr/bin/env python3
"""Does switching between two conversations destroy the cache?

With --parallel 1 there is a single slot, so a second conversation evicts the
first. Whether returning to the first costs a full re-prefill decides how
painful multi-conversation use is.
"""
import sys
sys.path.insert(0, "/home/john/bench")
from prefill_probe import ask, big

A = [{"role": "user", "content": big(3000) + "\n\nSay 'A'."}]
B = [{"role": "user", "content": big(3000).replace("engine", "turbine") + "\n\nSay 'B'."}]
for label, conv in (("A cold", A), ("B cold", B), ("A again", A), ("B again", B)):
    r = ask(conv, max_tokens=8)
    print(f"  {label:10} prompt_n={r['prompt_n']:>6}  prefill={r['prompt_ms']/1000:6.1f}s")
