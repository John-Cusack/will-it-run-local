#!/usr/bin/env python3
"""Quality probe for the running model. Checks capability, not compliance."""
import json, sys, time, urllib.request

BASE = "http://172.18.0.1:30001/v1/chat/completions"

def ask(messages, max_tokens=400, thinking=False, tools=None, label=""):
    body = {"model": "deepseek-v4-flash", "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.3, "top_p": 0.95,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    el = time.monotonic() - t0
    m = d["choices"][0]["message"]
    t = d.get("timings", {})
    return {
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning_content") or "",
        "tool_calls": m.get("tool_calls"),
        "finish": d["choices"][0].get("finish_reason"),
        "tps": t.get("predicted_per_second", 0.0),
        "n": t.get("predicted_n"),
        "secs": el,
    }

def show(label, r, expect=None):
    print(f"\n{'='*70}\n### {label}   [{r['n']} tok @ {r['tps']:.1f} tok/s, {r['secs']:.0f}s]")
    if r["reasoning"]:
        print(f"--- reasoning ({len(r['reasoning'])} chars) ---")
        print("  " + r["reasoning"][:300].replace("\n", "\n  "))
    if r["tool_calls"]:
        print("--- tool_calls ---")
        print("  " + json.dumps(r["tool_calls"], indent=2)[:400].replace("\n", "\n  "))
    print("--- answer ---")
    print("  " + (r["content"][:1200].replace("\n", "\n  ") or "(empty)"))
    if expect:
        ok = expect(r)
        print(f"--- verdict: {'PASS' if ok else 'FAIL'}")
    # leak checks always
    c = r["content"]
    leaks = [s for s in ("<think>", "</think>", "DSML", "<|", "｜") if s in c]
    if leaks:
        print(f"--- !! MARKUP LEAK: {leaks}")

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city.",
    "parameters": {"type": "object",
                   "properties": {"location": {"type": "string"},
                                  "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["location"]}}}]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "1"):
        show("1. FACTUAL (Australia capital - common trap)",
             ask([{"role": "user", "content": "What is the capital of Australia? Answer in one sentence."}], 120),
             lambda r: "canberra" in r["content"].lower())

    if which in ("all", "2"):
        show("2. TRICK LOGIC (all but 9)",
             ask([{"role": "user", "content": "A farmer has 17 sheep. All but 9 run away. How many are left? Give the number and one line of justification."}], 200),
             lambda r: "9" in r["content"])

    if which in ("all", "3"):
        show("3. MULTI-STEP ARITHMETIC",
             # 300 was too tight: the model shows its working in markdown and
             # got truncated mid-answer, which reads as a wrong answer.
             ask([{"role": "user", "content": "A train leaves at 14:35 and the journey takes 2 hours 50 minutes. It is then delayed by 40 minutes. What time does it arrive? Show your steps."}], 600),
             lambda r: "18:05" in r["content"] or "6:05" in r["content"])

    if which in ("all", "4"):
        show("4. STRICT INSTRUCTION FOLLOWING",
             ask([{"role": "user", "content": "Reply with exactly three words, no punctuation, describing the ocean."}], 60),
             lambda r: 2 <= len(r["content"].strip().split()) <= 4)

    if which in ("all", "5"):
        show("5. CODE",
             ask([{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number iteratively. Code only, no explanation."}], 300),
             lambda r: "def " in r["content"])

    if which in ("all", "6"):
        show("6. REASONING MODE ENABLED (separation check)",
             ask([{"role": "user", "content": "I have 3 boxes. Box A has twice as many balls as Box B. Box C has 5 fewer than Box A. Total is 45. How many in each?"}],
                 700, thinking=True),
             lambda r: bool(r["reasoning"]) and bool(r["content"]))

    if which in ("all", "7"):
        show("7. TOOL CALLING (no markup leak)",
             ask([{"role": "user", "content": "Use the get_weather tool to find the weather in Paris. Do not answer from memory."}],
                 300, tools=TOOLS),
             lambda r: bool(r["tool_calls"]))
