#!/usr/bin/env python3
"""Measure time-to-first-token the way a chat client actually behaves.

The sweep measured prefill with cache_prompt=false, i.e. every request
re-processes the whole prompt. Real clients send a growing conversation, and
llama.cpp caches the previous prompt by default, so only the new tokens should
need processing. This checks whether that actually happens.
"""
import json, sys, time, urllib.request

BASE = "http://172.18.0.1:30001/v1/chat/completions"
FILLER = ("The engine management system samples intake air temperature, manifold "
          "pressure and crankshaft position, then trims injector pulse width and "
          "spark advance to hold the mixture near stoichiometric. ")

def ask(messages, max_tokens=40, cache=True):
    body = {"model": "deepseek-v4-flash", "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.1, "stream": False,
            "cache_prompt": cache}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    wall = time.monotonic() - t0
    t = d.get("timings", {})
    return {"prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
            "pps": t.get("prompt_per_second"), "wall": wall,
            "content": d["choices"][0]["message"]["content"]}

def big(nwords):
    w = FILLER.split()
    out = []
    while len(out) < nwords:
        out.extend(w)
    return " ".join(out[:nwords])

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"
    if mode == "cold":
        # Same long prompt twice: does the second one hit the cache?
        msg = [{"role": "user", "content": big(5000) + "\n\nName one component mentioned."}]
        for label in ("first (cold)", "second (identical)"):
            r = ask(msg)
            print(f"  {label:20} prompt_n={r['prompt_n']:>6} "
                  f"prefill={r['prompt_ms']/1000:7.1f}s  wall={r['wall']:6.1f}s")
    else:
        # A conversation that grows, the way a chat UI sends it.
        conv = [{"role": "user", "content": big(4000) + "\n\nSummarise in one line."}]
        for turn in range(1, 5):
            r = ask(conv)
            print(f"  turn {turn}: prompt_n={r['prompt_n']:>6}  "
                  f"prefill={r['prompt_ms']/1000:7.1f}s  wall={r['wall']:6.1f}s")
            conv.append({"role": "assistant", "content": r["content"]})
            conv.append({"role": "user", "content": "Expand on that briefly."})
