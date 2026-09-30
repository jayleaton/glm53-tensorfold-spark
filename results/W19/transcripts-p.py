#!/usr/bin/env python3
"""W9: fixed-prompt transcripts for the RoCE engine A/B: 6 prompts x (greedy, sampled seed 1234 T 0.8 top_p 0.95),
256 tokens, thinking off, one at a time, then the same 4 of them concurrently (batch rounds). Usage: transcripts.py OUT"""
import hashlib, json, sys, threading, urllib.request
P = ["Explain in detail how a transformer decoder generates text, step by step.",
     "Write a Python function that merges overlapping intervals, with tests.",
     "Summarise the causes of the French Revolution in five bullet points.",
     "What is 17*23? Show the arithmetic, then give three other multiplication facts.",
     "Write a short story about a lighthouse keeper who finds a message in a bottle.",
     "Compare TCP and UDP for a real-time multiplayer game; recommend one."]
def req(p, sampled):
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": p}], "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}
    body.update({"temperature": 0.8, "top_p": 0.95, "seed": 1234} if sampled else {"temperature": 0, "top_p": 1})
    r = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=900))
    t = d["choices"][0]["message"]["content"]
    return {"sha": hashlib.sha256(t.encode()).hexdigest()[:16], "tokens": d["usage"]["completion_tokens"], "text": t}
out = {"alone": {}, "together": {}}
for i, p in enumerate(P):
    for s in (False, True):
        out["alone"][f"{i}-{'s' if s else 'g'}"] = req(p, s)
got = {}
def go(k, i, s): got[k] = req(P[i], s)
ts = [threading.Thread(target=go, args=(f"{i}-{'s' if s else 'g'}", i, s)) for i, s in ((0, False), (1, True), (2, False), (3, True))]
[t.start() for t in ts]; [t.join() for t in ts]
out["together"] = got
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("alone", {k: v["sha"] for k, v in out["alone"].items()})
print("together == alone:", {k: v["sha"] == out["alone"][k]["sha"] for k, v in got.items()})
