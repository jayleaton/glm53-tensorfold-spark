#!/usr/bin/env python3
"""W11 request driver (rank 0 host, prod API on 127.0.0.1:8001; standard library only). Non-streaming, so the engine's
per-request stats (``tensorfold``: rounds, keeps, depths, drafters, round_kinds, decode_s) come back whole.

  w11req.py one OUT NAME MAXTOK [TEMP] [SEED]   one request of a named prompt (see PROMPTS), thinking off
  w11req.py conc OUT N MAXTOK                    N concurrent prose requests (W7's dec4 topics), greedy
  w11req.py accept OUT [MAXTOK]                  the acceptance set: every PROMPTS entry, greedy and sampled
                                                 (T 1, top-k 20, top-p 0.95, seed 1234), + 2 thinking-on prose
  w11req.py acceptfix OUT [MAXTOK]               the same prompts (thinking off) with fixed-depth drafting, no
                                                 policy cut: tf_policy "4" (MTP, 4 drafts every round) and "f7"
                                                 (DFlash2, a 7-draft block every round): untruncated a_j

One JSON line a request is appended to OUT."""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench"))
from glmbench import KIT_PROMPTS, TF_PROMPTS, TWEET_PROMPTS  # noqa: E402

P = {p["name"]: p["prompt"] for p in KIT_PROMPTS}
EDIT_SRC = (REPO / "bench" / "glmbench.py").read_text()[:6000]     # glmbench's own edit source
TOPICS = ["the history of the printing press", "how a heat pump works", "the life cycle of a star",
          "why bridges use expansion joints", "how vaccines train the immune system", "the causes of ocean tides"]

# name -> (category, prompt)
PROMPTS = {
    "chat": ("prose", TF_PROMPTS[1]["prompt"]),
    "essay": ("prose", P["essay"]),
    "hashmap": ("prose", P["hashmap"]),
    "tides": ("prose", "Write a detailed, plain-prose explanation of the causes of ocean tides."),
    "letter": ("prose", "Write a warm, detailed letter to a friend describing a week-long hiking trip through the "
                        "Scottish Highlands: the weather, the people you met, the food, and what you learned."),
    "code-lru": ("code", TWEET_PROMPTS[1]["prompt"]),
    "code-fib": ("code", TF_PROMPTS[0]["prompt"]),
    "code-go": ("code", "Implement a thread-safe bounded blocking queue in Go with Put, Take, TryPut and Close, "
                        "full doc comments, and table-driven tests."),
    "code-ts": ("code", "Write a TypeScript React component for a paginated, sortable data table with a search box, "
                        "using hooks, with prop types and a short usage example."),
    "agent-rename": ("agent", f"Here is a Python file:\n\n```python\n{EDIT_SRC}\n```\n\nRename the function `cell` to "
                              "`run_cell` everywhere and output the complete updated file. Output only the code."),
    "agent-log": ("agent", f"Here is a Python file:\n\n```python\n{EDIT_SRC}\n```\n\nChange every `print(` call to "
                           "`log(` and output the complete updated file. Output only the code."),
    "agent-diff": ("agent", f"Here is a Python file:\n\n```python\n{EDIT_SRC}\n```\n\nAdd a `--timeout` command-line "
                            "option (seconds, default 1800) and use it in Client.post. Reply with a unified diff only."),
    "agent-json": ("agent", TWEET_PROMPTS[2]["prompt"]),
    "agent-plan": ("agent", "You are a coding agent working in a Python repository with src/ and tests/. The user "
                            "asks: 'the CSV importer crashes on empty lines'. Write your plan as a numbered list of "
                            "concrete steps (files to open, commands to run, the fix, tests to add), then the exact "
                            "shell commands you would run first, each in a fenced block."),
}


def req(prompt, maxtok, temp=0.0, seed=None, think=False, policy=None):
    body = {"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": prompt}], "max_tokens": maxtok,
            "temperature": temp, "chat_template_kwargs": {"enable_thinking": bool(think)}}
    if temp > 0:
        body.update(top_k=20, top_p=0.95, seed=1234 if seed is None else seed)
    else:
        body["top_p"] = 1
    if policy:
        body["tf_policy"] = policy
    r = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=3600))
    tf = d.get("tensorfold", {})
    n = d["usage"]["completion_tokens"]
    return dict(t0=round(t0, 3), wall=round(time.time() - t0, 3), prompt_tokens=d["usage"]["prompt_tokens"], tokens=n,
                temp=temp, think=think, policy=policy, decode_s=tf.get("decode_s"),
                decode_tps=round((n - 1) / tf["decode_s"], 2) if tf.get("decode_s") else None, tf=tf)


def show(r):
    tf = r["tf"]
    print(time.strftime("%T"), r.get("name"), r.get("cat"), f"T={r['temp']}", "tokens", r["tokens"], "decode",
          r["decode_tps"], "tok/s, tpr", tf.get("tokens_per_round"), "rounds", tf.get("rounds"), "kinds",
          json.dumps(tf.get("round_kinds")), flush=True)


def main():
    mode, out = sys.argv[1], sys.argv[2]
    rows = []
    if mode == "one":
        name, maxtok = sys.argv[3], int(sys.argv[4])
        temp = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        seed = int(sys.argv[6]) if len(sys.argv) > 6 else None
        cat, prompt = PROMPTS[name]
        r = req(prompt, maxtok, temp, seed)
        r.update(name=name, cat=cat)
        rows.append(r)
    elif mode == "conc":
        k, maxtok = int(sys.argv[3]), int(sys.argv[4])
        res = [None] * k

        def one(i):
            res[i] = req(f"Write a detailed, plain-prose explanation of {TOPICS[i % len(TOPICS)]}.", maxtok)
            res[i].update(name=f"conc{k}-{i}", cat="prose4")
        th = [threading.Thread(target=one, args=(i,)) for i in range(k)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        rows = res
    elif mode == "accept":
        maxtok = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        for name, (cat, prompt) in PROMPTS.items():
            mt = 1024 if cat == "agent" and name != "agent-plan" else maxtok
            for temp in (0.0, 1.0):
                r = req(prompt, mt, temp)
                r.update(name=name, cat=cat)
                show(r)
                with open(out, "a") as f:
                    f.write(json.dumps(r) + "\n")
        for name in ("essay", "tides"):
            r = req(PROMPTS[name][1], 1024, 0.0, think=True)
            r.update(name=name + "-think", cat="prose-think")
            show(r)
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
        return
    elif mode == "acceptfix":
        maxtok = int(sys.argv[3]) if len(sys.argv) > 3 else 384
        for pol in ("4", "f7"):
            for name, (cat, prompt) in PROMPTS.items():
                mt = 768 if cat == "agent" and name != "agent-plan" else maxtok
                for temp in (0.0, 1.0):
                    r = req(prompt, mt, temp, policy=pol)
                    r.update(name=name, cat=cat)
                    show(r)
                    with open(out, "a") as f:
                        f.write(json.dumps(r) + "\n")
        return
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        show(r)


main()
