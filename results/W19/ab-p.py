#!/usr/bin/env python3
"""Per-request knob A/B: unique filler prompt, non-streaming, engine stats. Usage: ab.py OUT ctx1,ctx2 knobsets-json"""
import json, sys, time, urllib.request
FILLER = ("The quick brown fox jumps over the lazy dog while the committee reviews quarterly logistics, "
          "inventory forecasts, and the maintenance schedule for the northern warehouse. ")
out, ctxs, sets = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], json.loads(sys.argv[3])
maxtok = int(sys.argv[4]) if len(sys.argv) > 4 else 256
def req(prompt, knobs):
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": prompt}], "max_tokens": maxtok,
            "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}
    if knobs: body["tf_knobs"] = knobs
    r = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=1800))
    tf = d.get("tensorfold", {})
    n = d["usage"]["completion_tokens"]; pt = d["usage"]["prompt_tokens"]
    dr = tf.get("drafters", "")
    return dict(wall=round(time.time() - t0, 2), prompt_tokens=pt, cached=tf.get("cached"), prefill_s=tf.get("prefill_s"),
                prefill_tps=round(pt / tf["prefill_s"], 1) if tf.get("prefill_s") else None, tokens=n,
                decode_tps=round((n - 1) / tf["decode_s"], 1) if tf.get("decode_s") else None,
                tpr=tf.get("tokens_per_round"), rounds=tf.get("rounds"), m=dr.count("m"), f=dr.count("f"),
                drafters=dr, keeps=tf.get("keeps"), kinds=tf.get("round_kinds"), knobs=knobs, t=time.strftime("%T"),
                sha=tf.get("sha256"))
res = []
for n in ctxs:
    for name, knobs in sets.items():
        tag = f"[ab-{n}-{time.time_ns()}] "
        prompt = tag + FILLER * max(1, n // 32) + "\n\nIn one sentence, what is the committee reviewing? Then count from 1 to 100."
        cold = req(prompt, knobs)
        warm = req(prompt, knobs)
        row = dict(ctx=n, name=name, cold=cold, warm=warm); res.append(row)
        print(f"{cold['t']} ctx {n} {name:10s} prompt {cold['prompt_tokens']} prefill {cold['prefill_s']}s {cold['prefill_tps']} tok/s | "
              f"cold dec {cold['decode_tps']} tpr {cold['tpr']} m/f {cold['m']}/{cold['f']} | warm dec {warm['decode_tps']} tpr {warm['tpr']} m/f {warm['m']}/{warm['f']} sha {cold['sha']}/{warm['sha']}", flush=True)
        json.dump(res, open(out, "w"), indent=1)
