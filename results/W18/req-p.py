#!/usr/bin/env python3
"""W7 request driver (rank 0 host, prod-style API on 127.0.0.1:8001).
  req.py prefill OUT TOKENS MAXTOK [knobs-json]   one cold prompt of ~TOKENS (ab.py's unique filler), non-streaming
  req.py decode OUT N MAXTOK                       N concurrent short natural prompts (thinking off, greedy)
Appends one JSON line per request to OUT (engine stats: prefill_s, pieces, decode_s, tokens_per_round, ...)."""
import json, sys, threading, time, urllib.request
FILLER = ("The quick brown fox jumps over the lazy dog while the committee reviews quarterly logistics, "
          "inventory forecasts, and the maintenance schedule for the northern warehouse. ")
TOPICS = ["the history of the printing press", "how a heat pump works", "the life cycle of a star",
          "why bridges use expansion joints", "how vaccines train the immune system", "the causes of ocean tides"]


def req(prompt, maxtok, knobs=None):
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": prompt}], "max_tokens": maxtok,
            "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}
    if knobs:
        body["tf_knobs"] = knobs
    r = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=1800))
    tf = d.get("tensorfold", {})
    pt, n = d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
    return dict(t0=round(t0, 3), wall=round(time.time() - t0, 3), prompt_tokens=pt, tokens=n, knobs=knobs,
                prefill_s=tf.get("prefill_s"), prefill_tps=round(pt / tf["prefill_s"], 1) if tf.get("prefill_s") else None,
                decode_tps=round((n - 1) / tf["decode_s"], 1) if tf.get("decode_s") else None,
                tf={k: v for k, v in tf.items() if k not in ("drafters",)}, drafters=tf.get("drafters"))


def main():
    mode, out = sys.argv[1], sys.argv[2]
    rows = []
    if mode == "prefill":
        n, maxtok = int(sys.argv[3]), int(sys.argv[4])
        knobs = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
        prompt = f"[w7-{n}-{time.time_ns()}] " + FILLER * max(1, n // 32) + \
            "\n\nIn one sentence, what is the committee reviewing? Then count from 1 to 100."
        rows.append(req(prompt, maxtok, knobs))
    else:
        k, maxtok = int(sys.argv[3]), int(sys.argv[4])
        res = [None] * k

        def one(i):
            res[i] = req(f"[w7d-{i}-{time.time_ns()}] Write a detailed, plain-prose explanation of {TOPICS[i % len(TOPICS)]}.",
                         maxtok)
        th = [threading.Thread(target=one, args=(i,)) for i in range(k)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        rows = res
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        print(time.strftime("%T"), mode, r["prompt_tokens"], "prefill", r["prefill_s"], r["prefill_tps"], "tok/s | tokens",
              r["tokens"], "decode", r["decode_tps"], "tok/s tpr", r["tf"].get("tokens_per_round"), "pieces",
              r["tf"].get("pieces"), flush=True)


main()
