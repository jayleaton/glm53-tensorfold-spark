#!/usr/bin/env python3
"""W19: patches/0600 checks against a running server (docs/UPSTREAM-PORTS.md section 4, items 2-5). ports.py OUT.json

 1 disconnect: a non-streamed request (max_tokens 32768, "count to 100000", thinking off) whose client closes after
   5 s: seconds from the close until /health shows requests_running 0 and inflight 0 (bar <= ~1 s), and the request
   log's finish for it; then 4 such at once, then 4 normal requests (all 4 admitted at once: queue_s);
 2 queued: 4 requests (greedy, 256 tokens) alone, then the same 4 with a 5th ~100k-token prompt whose client closes
   after 2 s: the 4 replies' token hashes unchanged, the 5th never answered;
 3 errors: "temperature": true streamed -> 400 JSON before headers; chat_template_kwargs "x" -> 400; a non-UTF-8 body
   -> 400; kill -USR1 on both ranks -> stacks in the logs, both ranks keep serving (a greedy request after);
 4 image URL (vision on): an https image URL == the same image as a data: URL (prompt_tokens, reply hash; greedy);
   http://, https://127.0.0.1/, https://169.254.169.254/ -> 400 with no URL in the message;
 5 /health: completion_tokens_total grows during a reply; rounds_total / drafted_total / accepted_total after it.
"""
import base64, json, socket, subprocess, sys, threading, time, urllib.error, urllib.request

HOST, PORT = "127.0.0.1", 8001
B = f"http://{HOST}:{PORT}"
M = "GLM-5.3-Flash-EXL3"
OUT = sys.argv[1]
res = {}


def save():
    json.dump(res, open(OUT, "w"), indent=1, default=str)


def health():
    return json.load(urllib.request.urlopen(B + "/health", timeout=10))


def post(body, timeout=600, raw=None, headers=None):
    data = raw if raw is not None else json.dumps(body).encode()
    r = urllib.request.Request(B + "/v1/chat/completions", data=data,
                               headers=headers or {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(txt)
        except ValueError:
            return e.code, txt


def chat(content, maxtok=256, think=False, **kw):
    b = {"model": M, "messages": [{"role": "user", "content": content}], "max_tokens": maxtok, "temperature": 0,
         "top_p": 1, "chat_template_kwargs": {"enable_thinking": think}}
    b.update(kw)
    return b


def raw_send_close(body, after_s):
    """POST body on a raw socket, close the socket after after_s seconds; returns the close time."""
    data = json.dumps(body).encode()
    s = socket.create_connection((HOST, PORT))
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              + f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    time.sleep(after_s)
    s.close()
    return time.time()


def wait_idle(t_close, limit=15.0):
    first = None
    while time.time() - t_close < limit:
        h = health()
        if h.get("requests_running", 1) == 0 and h.get("inflight", 1) == 0:
            return round(time.time() - t_close, 3), h
        time.sleep(0.05)
    return None, health()


def reqlog_tail(n=12):
    try:
        out = subprocess.run(["docker", "exec", "glm53-tf-r0", "tail", "-n", str(n), "/sessions/requests.jsonl"],
                             capture_output=True, text=True, timeout=20).stdout
        return [json.loads(l) for l in out.splitlines() if l.strip()]
    except Exception as e:  # noqa: BLE001
        return [str(e)]


COUNT = "Count from 1 to 100000, one number a line. Do not stop early."

# 1 disconnect -----------------------------------------------------------------------------------------------------
t = raw_send_close(chat(COUNT, 32768), 5.0)
dt, h = wait_idle(t)
lg = [d for d in reqlog_tail(6) if isinstance(d, dict)]
res["disconnect_one"] = {"freed_s": dt, "health": h, "reqlog_last": lg[-1:] if lg else None}
print(f"1a non-streamed disconnect: slot free {dt} s after the close; last request log: "
      f"{ {k: lg[-1].get(k) for k in ('finish', 'decode_tokens', 'slot')} if lg else None}", flush=True)
save()
ts = []
th = [threading.Thread(target=lambda: ts.append(raw_send_close(chat(COUNT, 32768), 5.0))) for _ in range(4)]
[x.start() for x in th]
[x.join() for x in th]
dt4, h4 = wait_idle(max(ts))
lg = [d for d in reqlog_tail(8) if isinstance(d, dict)]
fin = [(d.get("finish"), d.get("decode_tokens")) for d in lg[-4:]]
outs = [None] * 4


def norm(i):
    outs[i] = post(chat(f"Name three rivers in country number {i + 1} of the G7, one line.", 64))


th = [threading.Thread(target=norm, args=(i,)) for i in range(4)]
[x.start() for x in th]
[x.join() for x in th]
q = [o[1].get("tensorfold", {}).get("queued_s") if o and o[0] == 200 else None for o in outs]
res["disconnect_four"] = {"freed_s": dt4, "finish": fin, "normal_queue_s": q, "normal_status": [o[0] for o in outs]}
print(f"1b 4 disconnects: all slots free {dt4} s after the last close; finishes {fin}; then 4 normal: status "
      f"{[o[0] for o in outs]} queue_s {q}", flush=True)
save()

# 2 queued / prefilling client gone ---------------------------------------------------------------------------------
TOP = ["the history of the printing press", "how a heat pump works", "the life cycle of a star",
       "why bridges use expansion joints"]


def four(extra=False):
    got = [None] * 4

    def one(i):
        got[i] = post(chat(f"Write a detailed plain-prose explanation of {TOP[i]}.", 256))
    th = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    [x.start() for x in th]
    tc = None
    if extra:
        time.sleep(0.5)
        big = chat("[w19-queued-" + str(time.time_ns()) + "] " + "The quick brown fox jumps over the lazy dog. " * 10000
                   + "\nSummarize.", 64)
        tc = raw_send_close(big, 2.0)
    [x.join() for x in th]
    return [g[1].get("tensorfold", {}).get("sha256") if g and g[0] == 200 else g for g in got], tc


a, _ = four()
b, tc = four(True)
time.sleep(1.0)
h = health()
res["queued"] = {"alone": a, "with_5th": b, "same": a == b, "health_after": h}
print(f"2 queued 5th gone: 4 replies unchanged {a == b}; after: requests_running {h.get('requests_running')} "
      f"inflight {h.get('inflight')}", flush=True)
save()

# 3 errors + USR1 ---------------------------------------------------------------------------------------------------
e1 = post(dict(chat("hi", 8), temperature=True, stream=True))
e2 = post(dict(chat("hi", 8), chat_template_kwargs="x"))
e3 = post(None, raw=b'{"model": "x", "messages": [{"role": "user", "content": "\xff\xfe"}]}')
res["errors"] = {"temperature_true_stream": e1, "template_kwargs_str": e2, "non_utf8": e3}
print(f"3 errors: temperature true streamed {e1[0]}, chat_template_kwargs 'x' {e2[0]}, non-UTF-8 {e3[0]}", flush=True)
since = str(int(time.time()) - 1)   # epoch seconds (the container clock is UTC)
subprocess.run(["docker", "exec", "glm53-tf-r0", "sh", "-c", "kill -USR1 1"], timeout=20)   # no /bin/kill in the image
subprocess.run(["ssh", "-o", "BatchMode=yes", "$WORKER_SSH", "docker exec glm53-tf-r1 sh -c 'kill -USR1 1'"], timeout=20)
time.sleep(3)
l0 = subprocess.run(["docker", "logs", "--since", since, "glm53-tf-r0"], capture_output=True, text=True).stderr + \
    subprocess.run(["docker", "logs", "--since", since, "glm53-tf-r0"], capture_output=True, text=True).stdout
l1 = subprocess.run(["ssh", "-o", "BatchMode=yes", "$WORKER_SSH", f"docker logs --since {since} glm53-tf-r1 2>&1"],
                    capture_output=True, text=True).stdout
st, ok = post(chat("What is 17*23? Answer with the number only.", 32))
ans = ok.get("choices", [{}])[0].get("message", {}).get("content") if st == 200 else ok
sig = subprocess.run(["docker", "exec", "glm53-tf-r0", "grep", "SigCgt", "/proc/1/status"], capture_output=True, text=True).stdout.split()
caught = bool(sig) and bool(int(sig[-1], 16) >> 9 & 1)
res["usr1"] = {"r0_thread_lines": l0.count("most recent call first"), "r1_thread_lines": l1.count("most recent call first"),
               "usr1_caught_r0": caught, "after": [st, ans]}
print(f"3 USR1: stack lines r0 {res['usr1']['r0_thread_lines']} r1 {res['usr1']['r1_thread_lines']}; request after: "
      f"{st} {ans!r}", flush=True)
save()

# 4 image URL -------------------------------------------------------------------------------------------------------
URL = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
try:
    img = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "w19-check/1.0"}), timeout=30).read()
    dataurl = "data:image/png;base64," + base64.b64encode(img).decode()

    def imgreq(u):
        return post({"model": M, "max_tokens": 64, "temperature": 0, "top_p": 1,
                     "chat_template_kwargs": {"enable_thinking": False},
                     "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe this image in one sentence."},
                                                               {"type": "image_url", "image_url": {"url": u}}]}]})
    su, du = imgreq(URL), imgreq(dataurl)
    key = lambda r: (r[0], r[1].get("usage", {}).get("prompt_tokens") if r[0] == 200 else None,
                     r[1].get("tensorfold", {}).get("sha256") if r[0] == 200 else str(r[1])[:200])
    bad = {u: imgreq(u) for u in ("http://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
                                  "https://127.0.0.1/x.png", "https://169.254.169.254/latest/meta-data/")}
    badk = {u: (r[0], ("URL IN MESSAGE" if u.split("//")[1][:12] in json.dumps(r[1]) else "no url")) for u, r in bad.items()}
    res["image"] = {"https": key(su), "data": key(du), "equal": key(su) == key(du), "refusals": badk,
                    "bytes": len(img)}
    print(f"4 image: https {key(su)} data {key(du)} equal {key(su) == key(du)}; refusals {badk}", flush=True)
except Exception as e:  # noqa: BLE001
    res["image"] = {"skipped": f"{type(e).__name__}: {e}"}
    print(f"4 image: skipped ({type(e).__name__}: {e})", flush=True)
save()

# 5 /health totals --------------------------------------------------------------------------------------------------
h0 = health()
done = {}
th = threading.Thread(target=lambda: done.update(r=post(chat("Write a detailed plain-prose explanation of the causes "
                                                              "of ocean tides.", 512))))
th.start()
time.sleep(3)
h1 = health()
time.sleep(2)
h2 = health()
th.join()
h3 = health()
keys = ("completion_tokens_total", "rounds_total", "drafted_total", "accepted_total", "requests_total")
res["health"] = {"before": {k: h0.get(k) for k in keys}, "during1": {k: h1.get(k) for k in keys},
                 "during2": {k: h2.get(k) for k in keys}, "after": {k: h3.get(k) for k in keys},
                 "streams": h2.get("streams"), "has_0150": all(k in h3 for k in ("ok", "inflight", "uptime_s"))}
grow = (h2.get("completion_tokens_total") or 0) > (h1.get("completion_tokens_total") or 0)
print(f"5 /health: completion_tokens_total during {h1.get('completion_tokens_total')} -> {h2.get('completion_tokens_total')} "
      f"(grows {grow}); after {h3.get('completion_tokens_total')}; rounds {h0.get('rounds_total')} -> {h3.get('rounds_total')}; "
      f"drafted {h3.get('drafted_total')} accepted {h3.get('accepted_total')}; 0150 fields {res['health']['has_0150']}",
      flush=True)
save()
