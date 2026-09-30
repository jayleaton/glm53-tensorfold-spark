#!/usr/bin/env python3
"""W17: patches/0560 (GLM53_TF_MULTI_PREFILL) on the real model, MULTI-PREFILL.md §9 step 3.

    mpf.py group OUT.json TAG [ROUNDS]   grouped cold, then each prompt alone (resumed at n - 64 from the group's
                                         snapshot); variants greedy / seeded T 1 x thinking off / low; 4 ~130-token
                                         prompts released together ROUNDS times (default 10 for greedy-off, 3 others);
                                         then a burst of 8 prompts of 200-3,000 tokens, twice, and each alone
    mpf.py alone OUT.json IN.json        every prompt of IN (a `group` run) sent alone on a later / restarted server:
                                         the request log's `cached` says whether it was computed cold; replies vs IN

Each reply's sha is over reasoning + "\\x00" + content (64 tokens, ignore_eos). The 0300 request log (LOG) supplies
`multi` (members of the group forward), `cached`, `queue_s`, `pieces` of the lines each phase appended. Stdlib only."""
import hashlib, json, os, sys, threading, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
LOG = os.environ.get("LOG", "$HOME/.cache/glm53-tf/sessions/requests.jsonl")
MODEL = "GLM-5.3-Flash-EXL3"
TOPICS = ["the printing press", "a heat pump", "the life cycle of a star", "bridge expansion joints",
          "how vaccines train immunity", "ocean tides", "a compiler", "why the sky is blue"]
VARIANTS = {"greedy-off": ({"temperature": 0, "top_p": 1}, False), "greedy-low": ({"temperature": 0, "top_p": 1}, True),
            "seed-off": ({"temperature": 1.0, "top_p": 0.95, "seed": 7}, False),
            "seed-low": ({"temperature": 1.0, "top_p": 0.95, "seed": 7}, True)}


def short_prompt(tag, i):
    return (f"[{tag}-{i}] You are writing for a general audience. Explain {TOPICS[i % len(TOPICS)]} in clear, plain "
            "prose. Cover the main idea first, then two or three supporting details, then one common misconception and "
            "why it is wrong. Keep the tone friendly and avoid lists; write connected paragraphs. Do not use headings.")


def long_prompt(tag, i, words):
    body = " ".join(f"Record {k}: item {(k * 37 + i) % 1000} weighs {(k * 13 + i * 7) % 97} kg and ships to bay "
                    f"{(k * 5 + i) % 23}." for k in range(words // 14 + 1))
    return f"[{tag}-L{i}] Here is a log.\n{body}\nWhich bay receives the heaviest item in the log? Explain briefly."


def req(prompt, variant):
    extra, think = VARIANTS[variant]
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "ignore_eos": True,
            "chat_template_kwargs": {"reasoning_effort": "low"} if think else {"enable_thinking": False}, **extra}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=900) as resp:
        d = json.load(resp)
    m = d["choices"][0]["message"]
    text = (m.get("reasoning") or m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
    return dict(sha=hashlib.sha256(text.encode()).hexdigest()[:16], wall=round(time.perf_counter() - t0, 3),
                prompt_tokens=d.get("usage", {}).get("prompt_tokens"))


def logpos():
    try:
        return len(open(LOG, "rb").read().splitlines())
    except OSError:
        return 0


def logsince(n):
    try:
        return [json.loads(l) for l in open(LOG, "rb").read().splitlines()[n:]]
    except OSError:
        return []


def brief(lines):
    return [{k: x.get(k) for k in ("prompt", "cached", "multi", "queue_s", "prefill_s", "first_s", "pieces", "slot")}
            for x in lines if x.get("kind") == "chat"]


def together(prompts, variant):
    res = [None] * len(prompts); go = threading.Event()

    def run(i):
        go.wait(); res[i] = req(prompts[i], variant)
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    for t in th:
        t.start()
    time.sleep(0.3); go.set()
    for t in th:
        t.join()
    return res


def group(out, tag, rounds):
    rows, ok_all = [], True
    for variant in VARIANTS:
        for rnd in range(rounds if variant == "greedy-off" else min(3, rounds)):
            prompts = [short_prompt(f"{tag}-{variant}-{rnd}", i) for i in range(4)]
            n0 = logpos(); g = together(prompts, variant); time.sleep(0.3); lg = brief(logsince(n0))
            n1 = logpos(); a = [req(p, variant) for p in prompts]; time.sleep(0.3); la = brief(logsince(n1))
            same = [x["sha"] == y["sha"] for x, y in zip(g, a)]
            ok_all &= all(same)
            rows.append(dict(kind="short", variant=variant, round=rnd, prompts=prompts, grouped=[x["sha"] for x in g],
                             alone=[x["sha"] for x in a], same=same, log_grouped=lg, log_alone=la))
            print(f"{time.strftime('%T')} {variant} r{rnd}: grouped == alone {sum(same)}/4 | multi "
                  f"{[x['multi'] for x in lg]} cached {[x['cached'] for x in lg]} queue {[x['queue_s'] for x in lg]} "
                  f"| alone cached {[x['cached'] for x in la]}", flush=True)
    sizes = [200, 450, 700, 1000, 1400, 1900, 2400, 3000]
    for variant in ("greedy-off", "seed-low"):
        prompts = [long_prompt(f"{tag}-{variant}", i, w) for i, w in enumerate(sizes)]
        sends = []
        for s in range(2):
            n0 = logpos(); g = together(prompts, variant); time.sleep(0.3); sends.append((g, brief(logsince(n0))))
        n1 = logpos(); a = [req(p, variant) for p in prompts]; time.sleep(0.3); la = brief(logsince(n1))
        same = [x["sha"] == y["sha"] == z["sha"] for x, y, z in zip(sends[0][0], sends[1][0], a)]
        ok_all &= all(same)
        rows.append(dict(kind="burst", variant=variant, prompts=prompts, send1=[x["sha"] for x in sends[0][0]],
                         send2=[x["sha"] for x in sends[1][0]], alone=[x["sha"] for x in a], same=same,
                         log_send1=sends[0][1], log_send2=sends[1][1], log_alone=la))
        print(f"{time.strftime('%T')} burst {variant}: send1 == send2 == alone {sum(same)}/8 | send1 multi "
              f"{[x['multi'] for x in sends[0][1]]} prompt {[x['prompt'] for x in sends[0][1]]} | send2 cached "
              f"{[x['cached'] for x in sends[1][1]]} multi {[x['multi'] for x in sends[1][1]]}", flush=True)
    n = sum(len(r["same"]) for r in rows); k = sum(sum(r["same"]) for r in rows)
    grouped = sum(1 for r in rows for x in (r.get("log_grouped") or r.get("log_send1")) if (x.get("multi") or 0) >= 2)
    print(f"SUMMARY group: replies equal {k}/{n}; cold sends in a group forward (multi >= 2): {grouped}; all {ok_all}", flush=True)
    json.dump(dict(rows=rows, ok=ok_all, equal=k, total=n), open(out, "w"), indent=1)


def alone(out, inp):
    src = json.load(open(inp)); rows = []; k = n = cold = 0
    for r in src["rows"]:
        ref = r.get("grouped") or r.get("send1")
        for p, s in zip(r["prompts"], ref):
            n0 = logpos(); a = req(p, r["variant"]); time.sleep(0.2); lg = brief(logsince(n0))
            c = (lg[0]["cached"] if lg else None)
            n += 1; k += a["sha"] == s; cold += (c == 0)
            rows.append(dict(variant=r["variant"], kind=r["kind"], ref=s, sha=a["sha"], same=a["sha"] == s, cached=c))
    print(f"SUMMARY alone: {k}/{n} replies == the grouped cold replies; computed cold (cached 0) {cold}/{n}", flush=True)
    json.dump(dict(rows=rows, equal=k, total=n, cold=cold), open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "group":
        group(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 10)
    else:
        alone(sys.argv[2], sys.argv[3])
