#!/usr/bin/env python3
"""W12: lone requests one after another (a fresh prompt each: the batcher gives a lone fresh request the least recently
admitted free slot, so 8 cover slots 0-3 twice): slot, decode tok/s, round kinds. slots.py OUT [N]"""
import json, sys, time, urllib.request
TOPICS = ["the history of the printing press", "how a heat pump works", "the life cycle of a star",
          "why bridges use expansion joints", "how vaccines train the immune system", "the causes of ocean tides",
          "how a jet engine produces thrust", "the chemistry of bread baking"]
out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
rows = []
for i in range(n):
    body = {"model": "GLM-5.3-Flash-EXL3", "max_tokens": 384, "temperature": 0, "top_p": 1, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"[w12 slot probe {i}] Explain {TOPICS[i % len(TOPICS)]} in detail."}]}
    r = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=900))
    tf = d.get("tensorfold", {}) or {}
    nt = d["usage"]["completion_tokens"]
    row = dict(i=i, slot=tf.get("slot"), tokens=nt, decode_tps=round((nt - 1) / tf["decode_s"], 2) if tf.get("decode_s") else None,
               tpr=tf.get("tokens_per_round"), kinds=tf.get("round_kinds"))
    rows.append(row)
    print(json.dumps(row), flush=True)
    time.sleep(1)
by = {}
for r in rows:
    by.setdefault(r["slot"], []).append(r["decode_tps"])
print("per slot:", {k: v for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))})
json.dump(rows, open(out, "w"), indent=1)
