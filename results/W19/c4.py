#!/usr/bin/env python3
"""W15: multi-arrival first tokens (patches/0540 §2, GLM53_TF_EMIT_FIRST): K concurrent short chat requests released
together (RigMark's concurrency shape: ~130-token prompts, reasoning low, 256 tokens), ROUNDS rounds, for K in 2 and 4.

    c4.py OUT.json [ROUNDS]        (C4_THINK=0: thinking off)

Per request: client TTFT (first streamed reasoning or content) and the 0300 request-log line (queue_s, prefill_s,
first_s, pieces, slot). With EMIT_FIRST each first token leaves when its own piece ends (the 2nd-4th of a round
~0.9 / 1.3 / 1.7 s in W13's timing) instead of all at the round's end (W13 b5: 1.91 s for all three). Stdlib only."""
import json, os, statistics, sys, threading, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
LOG = os.environ.get("LOG", "$HOME/.cache/glm53-tf/sessions/requests.jsonl")
MODEL = "GLM-5.3-Flash-EXL3"
TOPICS = ["the history of the printing press", "how a heat pump works", "the life cycle of a star",
          "why bridges use expansion joints", "how vaccines train the immune system", "the causes of ocean tides",
          "how a compiler turns source code into machine code", "why the sky is blue at noon and red at sunset"]


def one(i, k, rnd, res, go):
    prompt = (f"[w17-c{k}-{rnd}-{i}-{time.time_ns()}] You are writing for a general audience. Explain {TOPICS[(i + rnd) % len(TOPICS)]} "
              "in clear, plain prose. Cover the main idea first, then two or three supporting details, then one common "
              "misconception and why it is wrong. Keep the tone friendly and avoid lists; write connected paragraphs. "
              "Do not use headings. Aim for a thorough answer.")
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"reasoning_effort": "low"} if os.environ.get("C4_THINK", "1") != "0" else {"enable_thinking": False}}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    go.wait()
    t0 = time.perf_counter(); first = None; usage = None
    with urllib.request.urlopen(r, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ch = json.loads(line[5:])
            usage = ch.get("usage") or usage
            for c in ch.get("choices", []):
                d = c.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning") or d.get("reasoning_content")):
                    first = time.perf_counter()
    res[i] = dict(i=i, ttft=round(first - t0, 3) if first else None, wall=round(time.perf_counter() - t0, 3),
                  prompt_tokens=(usage or {}).get("prompt_tokens"), tag=prompt[:40])


def main():
    out = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rows = []
    for k in (2, 4):
        for rnd in range(rounds):
            before = len(open(LOG, "rb").read().splitlines())
            res = [None] * k; go = threading.Event()
            th = [threading.Thread(target=one, args=(i, k, rnd, res, go)) for i in range(k)]
            for t in th:
                t.start()
            time.sleep(0.2); go.set()
            for t in th:
                t.join()
            time.sleep(0.5)
            logs = [json.loads(l) for l in open(LOG, "rb").read().splitlines()[before:]]
            for r in res:
                lg = next((x for x in logs if x.get("prompt") == r["prompt_tokens"] and x.get("kind") == "chat"
                           and not x.get("_used")), None)
                if lg:
                    lg["_used"] = True
                r["log"] = {x: (lg or {}).get(x) for x in ("n", "queue_s", "prefill_s", "first_s", "pieces", "slot", "decode_tokens")}
                r.update(k=k, round=rnd)
            rows += res
            print(f"{time.strftime('%T')} C{k} round {rnd}: client ttft " + " ".join(f"{r['ttft']}" for r in sorted(res, key=lambda r: r['ttft'] or 9)) +
                  " | log first_s " + " ".join(f"{r['log']['first_s']}" for r in sorted(res, key=lambda r: r['log']['first_s'] or 9)) +
                  " | queue_s " + " ".join(f"{r['log']['queue_s']}" for r in sorted(res, key=lambda r: r['log']['first_s'] or 9)), flush=True)
            time.sleep(1)
    summ = {}
    for k in (2, 4):
        rs = [r for r in rows if r["k"] == k]
        t = [r["ttft"] for r in rs if r["ttft"] is not None]
        f = [r["log"]["first_s"] for r in rs if r["log"]["first_s"] is not None]
        summ[k] = dict(ttft_median=statistics.median(t), ttft_max=max(t), first_s_median=statistics.median(f), first_s_max=max(f),
                       first_s_sorted=sorted(f))
        print(f"SUMMARY C{k}: per-stream client TTFT median {summ[k]['ttft_median']} s (max {summ[k]['ttft_max']}); "
              f"log first_s median {summ[k]['first_s_median']} (max {summ[k]['first_s_max']}); all {summ[k]['first_s_sorted']}", flush=True)
    json.dump(dict(rows=rows, summary=summ), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
