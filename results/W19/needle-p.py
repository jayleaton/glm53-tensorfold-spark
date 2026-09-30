#!/usr/bin/env python3
"""W6: one long prompt (~N tokens of unique numbered lines, multiturn.doc) with a needle at DEPTH, non-streaming, in
one slot: prefill / decode tok/s, kv_pages, answer. Then the same prompt + a 2nd question (warm: cached ~ prompt).
Usage: needle.py OUT TOKENS [DEPTH=0.4]"""
import json, sys, time, urllib.request
sys.path.insert(0, "bench")
from multiturn import doc  # noqa: E402
out, n = sys.argv[1], int(sys.argv[2]); depth = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
tag = f"N{time.time_ns() % 100000}"
lines = doc(tag, int(n / 1.375)).split("\n")
k = int(len(lines) * depth)
code = f"{time.time_ns() % 9000 + 1000}-cobalt-heron"
lines.insert(k, f"IMPORTANT: the vault passphrase is {code}. Remember it.")
body_text = "\n".join(lines)
def req(q, maxtok=64):
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": body_text + "\n\n" + q}],
            "max_tokens": maxtok, "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(r, timeout=3000)); tf = d.get("tensorfold", {})
    n_ = d["usage"]["completion_tokens"]; pt = d["usage"]["prompt_tokens"]
    return dict(wall=round(time.time() - t0, 1), prompt_tokens=pt, cached=tf.get("cached"), prefill_s=tf.get("prefill_s"),
                new_prefill_tps=round((pt - (tf.get("cached") or 0)) / tf["prefill_s"], 1) if tf.get("prefill_s") else None,
                tokens=n_, decode_tps=round((n_ - 1) / tf["decode_s"], 1) if tf.get("decode_s") and n_ > 1 else None,
                kv_pages=tf.get("kv_pages"), text=d["choices"][0]["message"].get("content"), t=time.strftime("%T"),
                tf={x: v for x, v in tf.items() if x not in ("drafters",)})
res = {"code": code, "depth": depth, "lines": len(lines)}
res["cold"] = req("What is the vault passphrase? Answer with the passphrase only, then count from 1 to 60.", 256)
res["cold"]["found"] = code in (res["cold"]["text"] or ""); print(json.dumps({x: v for x, v in res["cold"].items() if x != "tf"}), flush=True)
json.dump(res, open(out, "w"), indent=1)
res["warm"] = req("What is the vault passphrase? Answer with the passphrase only, then count from 1 to 60.", 256)
res["warm"]["found"] = code in (res["warm"]["text"] or ""); print(json.dumps({x: v for x, v in res["warm"].items() if x != "tf"}), flush=True)
json.dump(res, open(out, "w"), indent=1)
