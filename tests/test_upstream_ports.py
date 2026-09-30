"""patches/0600: host-only ports from upstream TensorFold 0.3.6.2 / 0.5.0 (docs/UPSTREAM-050-AUDIT.md ranks 1-4, 6, 9).

Host only (fake engines, a one-id-per-character tokenizer, local sockets; no GPU, no network):

- **Client disconnect** (upstream 24afe5e, adapted to our batcher): the socket check (a socketpair: open, pipelined
  data, closed); ``App.run`` refuses a request whose client left before it started (no engine call), stops a running
  one at its next round and raises ``RequestCancelled`` (nothing more written), and never lets a callback failure
  into the engine (raised after ``generate``). Over HTTP: a **non-streamed** request whose client closes the socket
  stops within a few rounds (not at ``max_tokens``); a request whose client leaves while it is queued / prefilling
  (no token yet: the real ``Batcher._collect`` polling ``on_tokens.cancelled``) is cancelled before its first token,
  streamed and not. ``Batcher._collect`` alone: the poll while nothing arrives, the interval check while tokens flow,
  a callback exception cancelling the job and raised after its end marker. GLM53_TF_DISCONNECT=0 and its checking.
  (Both ranks in lockstep: tests/cuda/test_disconnect_patches.py.)
- **Request hardening** (2bc35c4): malformed sampling / length fields and ``chat_template_kwargs`` are 400s before a
  stream's headers (streamed and not), a template refusal / any other ``check`` failure is a 400 with its message (the
  log line redacted), a non-UTF-8 body is "not JSON", an engine failure is a 500 / a stream error event; valid fields
  give the same ``Sampling`` as before 0600.
- **kill -USR1** (a586f3d): ``cmd_serve`` registers the stack dump before it hands over to ``_serve_cuda``.
- **Image URLs** (391713e): ``check_url`` (HTTPS on 443 only, credentials / fragments / control characters / local
  host names refused; the local-testing knobs), ``public_ip`` over private / loopback / link-local / CGNAT / metadata /
  IPv6 equivalents (mapped, 6to4, Teredo, NAT64, ULA), a host resolving to any private address refused, every
  redirect re-checked (a DNS-rebinding-style redirect to a private address never connects; the connection is pinned
  to the checked address), redirect limits, a local server (with the knobs) for types, encodings, sizes, redirects
  and a stalled server's deadline; the bound on concurrent preparations (503); messages without the URL.
- **Small bundle**: ``closed_json`` and the GLM tool-argument repair (#87, typed arrays / objects only);
  ``completion_tokens`` at a stop string exact however the tokens arrive (88407d1); ``return_token_ids`` (e6be5ff).
- **/health live totals** (51b098d / c4bf25f): upstream's keys beside 0150's, completion tokens moving while a reply
  runs, the engine's stats folded at the end, failed requests counted, a batch engine's ``streams``.

Run against the patched tree: PYTHONPATH=<tree>/src pytest -q tests/test_upstream_ports.py
"""

from __future__ import annotations

import http.client
import io
import json
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tensorfold.cuda import server

if not hasattr(server, "parse_numbers"):
    pytest.skip("patches/0600 not applied", allow_module_level=True)

from tensorfold.cuda.health import Health  # noqa: E402
from tensorfold.server.cancellation import RequestCancelled, socket_cancellation  # noqa: E402

app_mod = pytest.importorskip("tensorfold.families.glm5_next.cuda.app")
vp = pytest.importorskip("tensorfold.families.glm5_next.cuda.vision_prep")
from tensorfold.families.glm5_next.cuda import image_fetch as fetch  # noqa: E402

EOS = 0
WAIT = 10.0


# -- fakes ------------------------------------------------------------------------------------------------------------
class Encoded:
    def __init__(self, ids):
        self.ids = ids


class Tok:
    """One id per character; id 0 is the end of sequence."""

    def __init__(self):
        self.vocab = ["<eos>", "^"]

    def id(self, piece):
        if piece not in self.vocab:
            self.vocab.append(piece)
        return self.vocab.index(piece)

    def ids(self, text):
        return [self.id(c) for c in text]

    def encode(self, text, add_special_tokens=False):
        return Encoded(([1] if add_special_tokens else []) + self.ids(text))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[i] for i in ids)

    def get_vocab_size(self, with_added_tokens=True):
        return 1000


class Template:
    def __init__(self, refuse=None):
        self.refuse = refuse

    def render(self, messages, *, tools, enable_thinking, extra=None):
        if self.refuse is not None:
            raise self.refuse
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


class PromptTokens:
    def __init__(self, tok):
        self.tok = tok

    def encode(self, text):
        return self.tok.encode(text).ids


class Paced:
    """Decodes ``text`` round by round (``chunks``: tokens a round, cycled), ``pause`` seconds a round, until
    ``on_tokens`` returns True or ``max_tokens``; records every call and the sampling it got."""

    eos = (EOS,)

    def __init__(self, text="ok", chunks=(1,), pause=0.0, stats=None, hold_at=None):
        self.text, self.chunks, self.pause, self.stats = text, chunks, pause, stats or {}
        self.request = threading.local()
        self.calls = 0
        self.rounds = 0
        self.stopped = None
        self.samplings = []
        self.hold_at = hold_at
        self.held = threading.Event()
        self.release = threading.Event()

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.calls += 1
        self.samplings.append(sampling)
        ids = self.tok.ids(self.text) if self.text else []
        pos, i = 0, 0
        while pos < max_tokens:
            if self.hold_at is not None and pos >= self.hold_at and not self.held.is_set():
                self.held.set()
                assert self.release.wait(WAIT)
            n = min(self.chunks[i % len(self.chunks)], max_tokens - pos)
            new = [ids[(pos + j) % len(ids)] if ids else self.tok.id("a") for j in range(n)]
            pos += n
            i += 1
            self.rounds += 1
            if self.pause:
                time.sleep(self.pause)
            if on_tokens(new):                   # an exception here would reach the engine: the test fails
                self.stopped = pos
                break
        return dict(self.stats)


def make_app(engine, *, template=None, glm=True):
    tok = Tok()
    engine.tok = tok
    app = object.__new__(app_mod.GlmApp if glm else server.App)
    app.served = "GLM-5.3-Flash-EXL3"
    app.tok = tok
    app.template = template or Template()
    app.default_thinking = False
    app.sampling = {"temperature": 0.0, "top_k": 20, "top_p": 0.95}
    app.max_tokens = 100
    app.lock = threading.Lock()
    app.created = 1700000000
    app.disconnect = True
    if glm:
        app.effort_field, app.default_effort = False, None
        app.prompt_tokens = PromptTokens(tok)
        app.prompt_memo = app_mod.Memo()
        app.reqlog = None
        app._rl = threading.local()
    app.health = Health(mode="basic", stall_s=0)
    app.engine = app.health.track(engine)
    return app


@pytest.fixture
def serving():
    servers = []

    def start(app):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return srv.server_address[1]

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def post(port, body, path="/v1/chat/completions", raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    try:
        data = raw if raw is not None else json.dumps(body).encode()
        conn.request("POST", path, data, {"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, r.getheader("Content-Type"), r.read().decode()
    finally:
        conn.close()


def get(port, path="/health"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, json.loads(r.read())
    finally:
        conn.close()


def raw_request(port, body, path="/v1/chat/completions") -> socket.socket:
    data = json.dumps(body).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=WAIT)
    s.sendall(f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(data)}"
              f"\r\n\r\n".encode() + data)
    return s


def wait_for(cond, timeout=WAIT):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


MESSAGES = [{"role": "user", "content": "hi"}]


# -- 1. client disconnect ---------------------------------------------------------------------------------------------
def test_socket_check_open_pipelined_closed():
    a, b = socket.socketpair()
    try:
        gone = socket_cancellation(a)
        assert gone.cancelled is False
        b.sendall(b"GET / HTTP/1.1\r\n")                 # a pipelined next request is data, not a close
        assert gone.cancelled is False and a.recv(4, socket.MSG_PEEK) == b"GET "
        b.close()
        a.recv(64)                                       # the pipelined bytes consumed: the close is next
        assert gone.cancelled is True and gone.cancelled is True     # sticky
    finally:
        a.close()
    closed, other = socket.socketpair()
    other.close()
    gone = socket_cancellation(closed)
    closed.close()
    assert gone.cancelled is True                        # a closed descriptor reads as gone


def test_run_refuses_a_request_whose_client_left_while_waiting():
    eng = Paced()
    app = make_app(eng)
    with pytest.raises(RequestCancelled):
        app.run({"messages": MESSAGES}, True, lambda d: True, cancelled=lambda: True)
    assert eng.calls == 0                                # no prefill, no engine call


def test_run_stops_at_the_next_round_and_writes_nothing_more():
    eng = Paced(text="abcdefgh")
    app = make_app(eng)
    rounds = {"n": 0}
    sent = []

    def emit(delta):
        sent.append(delta)
        rounds["n"] += 1
        return True

    with pytest.raises(RequestCancelled) as info:
        app.run({"messages": MESSAGES, "max_tokens": 1000}, True, emit, cancelled=lambda: rounds["n"] >= 5)
    assert eng.stopped == 5 and eng.rounds == 5          # stopped at the round the check turned true
    assert info.value.result["completion_tokens"] == 5 and info.value.result["stats"]["cancelled"] is True
    assert len(sent) == 5


def test_a_callback_failure_never_reaches_the_engine():
    eng = Paced(text="abcdefgh")
    app = make_app(eng)
    calls = []

    def emit(delta):
        calls.append(delta)
        if len(calls) == 3:
            raise TimeoutError("write timed out")
        return True

    with pytest.raises(TimeoutError, match="write timed out"):
        app.run({"messages": MESSAGES, "max_tokens": 1000}, True, emit)
    assert eng.stopped == 3 and len(calls) == 3          # the engine got True, not the exception


def test_tracked_engine_forwards_the_check():
    seen = {}

    class E:
        eos = (EOS,)

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            seen["cancelled"] = getattr(on_tokens, "cancelled", None)
            return {}

    eng = Health(mode="basic", stall_s=0).track(E())

    def cb(new):
        return False

    check = lambda: False                                # noqa: E731
    cb.cancelled = check
    eng.generate([1], 1, None, cb)
    assert seen["cancelled"] is check
    eng.generate([1], 1, None, lambda new: False)
    assert seen["cancelled"] is None


def test_non_streamed_request_stops_when_its_client_closes(serving):
    """The prod case: a non-streamed request whose client timed out ran to max_tokens (32,768) before 0600."""

    eng = Paced(text="abc", pause=0.002)
    app = make_app(eng)
    port = serving(app)
    s = raw_request(port, {"messages": MESSAGES, "max_tokens": 1_000_000})
    assert wait_for(lambda: eng.rounds >= 20)
    s.close()
    t0 = time.monotonic()
    assert wait_for(lambda: eng.stopped is not None), "the engine was never asked to stop"
    assert time.monotonic() - t0 < 2.0 and eng.stopped < 1_000_000
    assert wait_for(lambda: app.health.status()[1]["inflight"] == 0)
    body = app.health.status(app)[1]
    assert body["requests"] == 1 and body["requests_running"] == 0 and body["errors"] == 0


def test_knob_off_keeps_the_old_behaviour(serving, monkeypatch):
    monkeypatch.setenv("GLM53_TF_DISCONNECT", "0")
    assert server.disconnect_on() is False
    monkeypatch.setenv("GLM53_TF_DISCONNECT", "yes")
    with pytest.raises(ValueError, match="GLM53_TF_DISCONNECT"):
        server.disconnect_on()
    monkeypatch.delenv("GLM53_TF_DISCONNECT")
    assert server.disconnect_on() is True
    eng = Paced(text="abc", pause=0.001)
    app = make_app(eng)
    app.disconnect = False
    port = serving(app)
    s = raw_request(port, {"messages": MESSAGES, "max_tokens": 400})
    assert wait_for(lambda: eng.rounds >= 5)
    s.close()
    assert wait_for(lambda: eng.rounds >= 400)           # ran to max_tokens, as before 0600
    assert eng.stopped is None


# -- the batcher's collect loop ---------------------------------------------------------------------------------------
batch_mod = None


def _batch():
    global batch_mod
    if batch_mod is None:
        pytest.importorskip("torch")
        from tensorfold.families.glm5_next.cuda import batch as b

        batch_mod = b
    return batch_mod


def _collector(poll=0.01):
    b = _batch()
    bat = object.__new__(b.Batcher)
    bat.cv = threading.Condition()
    bat.poll_s = poll
    return bat


def test_poll_s_knob(monkeypatch):
    b = _batch()
    assert b._poll_s() == 0.25
    monkeypatch.setenv("GLM53_TF_DISCONNECT_POLL_MS", "40")
    assert b._poll_s() == 0.04
    monkeypatch.setenv("GLM53_TF_DISCONNECT_POLL_MS", "5")
    with pytest.raises(ValueError, match="10..10000"):
        b._poll_s()


def test_collect_polls_while_nothing_arrives():
    bat = _collector()
    job = SimpleNamespace(out=queue.SimpleQueue(), cancel=False)
    flag = {"gone": False}
    got = []

    def on_tokens(new):
        got.append(new)
        return False

    on_tokens.cancelled = lambda: flag["gone"]
    result = {}
    t = threading.Thread(target=lambda: result.update(r=bat._collect(job, on_tokens)))
    t.start()
    time.sleep(0.05)
    assert not job.cancel                                # queued / prefilling: nothing arrives, client still there
    flag["gone"] = True
    assert wait_for(lambda: job.cancel, 2.0)             # detected within the poll
    job.out.put(None)                                    # the plan dropped it
    t.join(WAIT)
    assert result["r"] == [] and got == []


def test_collect_checks_between_tokens_that_keep_coming():
    bat = _collector(poll=0.02)
    job = SimpleNamespace(out=queue.SimpleQueue(), cancel=False)
    flag = {"gone": False, "checks": 0}

    def check():
        flag["checks"] += 1
        return flag["gone"]

    def on_tokens(new):
        return False                                     # a silent non-streamed reply: no write ever fails

    on_tokens.cancelled = check
    stop = threading.Event()

    def feeder():
        while not stop.is_set():
            job.out.put([7])
            time.sleep(0.002)                            # rounds far faster than the poll
        job.out.put(None)

    t = threading.Thread(target=feeder)
    t.start()
    got = {}
    c = threading.Thread(target=lambda: got.update(r=bat._collect(job, on_tokens)))
    c.start()
    time.sleep(0.1)
    flag["gone"] = True
    assert wait_for(lambda: job.cancel, 2.0)
    stop.set()
    t.join(WAIT)
    c.join(WAIT)
    assert 2 <= flag["checks"] < 200                     # rate-limited, not once a token


def test_collect_callback_exception_cancels_then_raises():
    bat = _collector()
    job = SimpleNamespace(out=queue.SimpleQueue(), cancel=False)
    seen = []

    def on_tokens(new):
        seen.append(new)
        if len(seen) == 2:
            raise KeyError("tool parser")
        return False

    for item in ([1], [2], [3], None):                   # the slot's rounds until its cancel lands, then the end
        job.out.put(item)
    with pytest.raises(KeyError, match="tool parser"):
        bat._collect(job, on_tokens)
    assert job.cancel is True and seen == [[1], [2]]     # nothing handed out after the failure
    job2 = SimpleNamespace(out=queue.SimpleQueue(), cancel=False)
    job2.out.put(ValueError("bad"))
    with pytest.raises(ValueError):
        bat._collect(job2, lambda new: False)


class BatchLike:
    """GlmEngine.generate in batch mode, reduced: a job queued to a fake round loop that 'prefills' (no tokens) for
    ``prefill_s`` unless cancelled, then decodes; the HTTP thread waits in the real ``Batcher._collect``."""

    eos = (EOS,)

    def __init__(self, prefill_s=5.0):
        self.prefill_s = prefill_s
        self.request = threading.local()
        self.bat = _collector(poll=0.02)
        self.log = []
        self.batch = SimpleNamespace(seqs=[None, None], n=2)

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        job = SimpleNamespace(out=queue.SimpleQueue(), cancel=False, stats={})

        def loop():
            end = time.monotonic() + self.prefill_s
            while time.monotonic() < end:
                if job.cancel:                           # the plan cancels it: no token was ever decoded
                    self.log.append("cancelled while prefilling")
                    job.stats["cancelled"] = True
                    job.out.put(None)
                    return
                time.sleep(0.005)
            self.log.append("decoded")
            job.out.put(self.tok.ids("ok"))
            job.out.put(None)

        threading.Thread(target=loop, daemon=True).start()
        self.bat._collect(job, on_tokens)
        return job.stats


@pytest.mark.parametrize("stream", [False, True])
def test_client_gone_while_prefilling_is_cancelled_before_a_token(serving, stream):
    pytest.importorskip("torch")
    eng = BatchLike(prefill_s=5.0)
    app = make_app(eng)
    port = serving(app)
    s = raw_request(port, {"messages": MESSAGES, "max_tokens": 100, "stream": stream})
    time.sleep(0.1)
    s.close()
    assert wait_for(lambda: eng.log == ["cancelled while prefilling"], 3.0), eng.log
    assert wait_for(lambda: app.health.status()[1]["inflight"] == 0)


def test_streamed_client_gone_gets_no_final_chunk(serving):
    """A stream whose client left (seen at a round without text) ends without a final chunk or [DONE]."""

    eng = Paced(text="abc", pause=0.003)
    app = make_app(eng)
    port = serving(app)
    s = raw_request(port, {"messages": MESSAGES, "max_tokens": 1_000_000, "stream": True})
    s.settimeout(WAIT)
    head = s.recv(4096)
    assert b"200" in head.split(b"\r\n", 1)[0]
    assert wait_for(lambda: eng.rounds >= 5)             # decoding (a half-close before would stop it unstarted)
    s.shutdown(socket.SHUT_WR)                           # half-closed: upstream counts that as gone
    data = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()
    assert wait_for(lambda: eng.stopped is not None), data[-300:]
    assert b"[DONE]" not in data and b'"finish_reason": "length"' not in data


# -- 2. request hardening ---------------------------------------------------------------------------------------------
BAD_FIELDS = [
    ({"temperature": True}, "temperature must be a finite number"),
    ({"temperature": "hot"}, "temperature must be a finite number"),
    ({"temperature": float("nan")}, "temperature must be a finite number"),
    ({"temperature": float("inf")}, "temperature must be a finite number"),
    ({"top_p": "x"}, "top_p must be a finite number"),
    ({"top_p": [0.9]}, "top_p must be a finite number"),
    ({"top_k": 2.5}, "top_k must be an integer"),
    ({"top_k": False}, "top_k must be an integer"),
    ({"seed": True}, "seed must be an integer"),
    ({"seed": "7.5"}, "seed must be an integer"),
    ({"max_tokens": 1.5}, "max_tokens must be an integer"),
    ({"max_completion_tokens": {}}, "max_completion_tokens must be an integer"),
    ({"chat_template_kwargs": "x"}, "chat_template_kwargs must be a JSON object"),
    ({"chat_template_kwargs": []}, "chat_template_kwargs must be a JSON object"),
    ({"chat_template_kwargs": 0}, "chat_template_kwargs must be a JSON object"),
    ({"chat_template_kwargs": False}, "chat_template_kwargs must be a JSON object"),
    ({"chat_template_kwargs": [["enable_thinking", True]]}, "chat_template_kwargs must be a JSON object"),
]


@pytest.mark.parametrize("glm", [True, False], ids=["GlmApp", "App"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fields,message", BAD_FIELDS, ids=[str(f) for f, _ in BAD_FIELDS])
def test_malformed_fields_are_400_before_headers(serving, fields, message, stream, glm):
    eng = Paced()
    app = make_app(eng, glm=glm)
    port = serving(app)
    status, kind, text = post(port, dict({"messages": MESSAGES, "stream": stream}, **fields))
    assert status == 400 and kind == "application/json", text
    err = json.loads(text)["error"]
    assert message in err["message"] and err["type"] == "invalid_request_error"
    assert eng.calls == 0


@pytest.mark.parametrize("fields", [{}, {"temperature": 0.7, "top_k": 20.0, "top_p": "0.9", "seed": 7},
                                    {"temperature": "0.5", "top_k": "40", "seed": 123456789012},
                                    {"temperature": 1, "top_p": 1, "top_k": -1}, {"temperature": 0, "top_p": None},
                                    {"chat_template_kwargs": None}, {"chat_template_kwargs": {}},
                                    {"max_tokens": 5.0}])
def test_valid_fields_give_the_same_sampling_as_before(fields):
    from tensorfold.engine.exact_sampling import Sampling, seed_for

    eng = Paced()
    app = make_app(eng)
    body = dict({"messages": MESSAGES}, **fields)
    assert app.check(body) is None
    prompt = [5, 6, 7]
    # the pre-0600 formula (App.sampling_for as 0.3.4 had it)
    d = app.sampling
    temp = float(body["temperature"] if body.get("temperature") is not None else d["temperature"])
    if temp <= 0:
        before = None
    else:
        seed = body.get("seed")
        top_k = body["top_k"] if body.get("top_k") is not None else d["top_k"]
        top_p = body["top_p"] if body.get("top_p") is not None else d["top_p"]
        before = Sampling(int(seed) if seed is not None else seed_for(prompt), temp, int(top_k), float(top_p))
    assert app.sampling_for(body, prompt) == before


def test_template_refusal_is_a_400(serving):
    from jinja2.exceptions import TemplateError

    eng = Paced()
    eng.limit = 100000                                   # GlmApp.check renders the prompt to count it
    app = make_app(eng, template=Template(refuse=TemplateError("System message must be first")))
    port = serving(app)
    for stream in (False, True):
        status, kind, text = post(port, {"messages": MESSAGES, "stream": stream})
        assert status == 400 and kind == "application/json"
        assert json.loads(text)["error"]["message"] == \
            "the chat template rejected the request: System message must be first"
    status, _, text = post(port, {"messages": MESSAGES}, path="/tokenize")
    assert status == 400 and "rejected" in text
    assert eng.calls == 0


def test_template_refusal_found_only_by_run_is_still_the_clients_error(serving):
    """An app whose ``check`` does not render (no context limit): ``run``'s render refusal is a 400 too."""

    from jinja2.exceptions import TemplateError

    eng = Paced()
    port = serving(make_app(eng, template=Template(refuse=TemplateError("no tools here"))))
    status, _, text = post(port, {"messages": MESSAGES})
    assert status == 400 and "the chat template rejected the request: no tools here" in text
    status, _, text = post(port, {"messages": MESSAGES, "stream": True})
    assert status == 200 and "rejected the request" in text and "invalid_request_error" in text
    assert eng.calls == 0


def test_any_check_failure_is_a_400_logged_redacted(serving, capsys):
    eng = Paced()
    eng.limit = 100000
    app = make_app(eng, template=Template(refuse=KeyError("https://user:hunter2@img.example/a.png?sig=abc")))
    port = serving(app)
    status, _, text = post(port, {"messages": MESSAGES})
    assert status == 400 and "KeyError" not in json.loads(text)["error"].get("type", "")
    log = capsys.readouterr().out
    assert "request refused (400): KeyError" in log and "Traceback" in log
    assert "hunter2" not in log and "sig=abc" not in log and "<redacted>" in log


def test_non_utf8_and_bad_bodies(serving):
    port = serving(make_app(Paced()))
    status, _, text = post(port, None, raw=b'{"messages": "\xff\xfe"}')
    assert status == 400 and "not JSON" in text
    status, _, text = post(port, None, raw=b"[1, 2]")
    assert status == 400 and "JSON object" in text
    status, _, text = post(port, {"messages": "hi"})
    assert status == 400 and "messages must be a list" in text


def test_engine_failure_is_500_or_an_error_event(serving, capsys):
    class Boom(Paced):
        def generate(self, *a, **k):
            raise RuntimeError("CUDA error: an illegal memory access")

    port = serving(make_app(Boom()))
    status, _, text = post(port, {"messages": MESSAGES})
    assert status == 500 and json.loads(text)["error"]["type"] == "server_error"
    status, kind, text = post(port, {"messages": MESSAGES, "stream": True})
    assert status == 200 and kind == "text/event-stream"
    assert '"type": "server_error"' in text and text.rstrip().endswith("data: [DONE]")
    assert "request failed" in capsys.readouterr().out


def test_redact():
    r = server.redact
    assert r("fetch https://bob:pw@h.example/x.png?token=s3cret#frag done") == \
        "fetch https://<redacted>@h.example/x.png?<redacted> done"
    assert r("url data:image/png;base64,AAAA== end") == "url data:<redacted> end"
    assert r("plain text, no url") == "plain text, no url"


def test_problem_status_503(serving):
    class Busy(Paced):
        pass

    app = make_app(Busy())
    app.check = lambda body: server.Problem("image processing capacity is busy; retry shortly", param="messages",
                                            status=503)
    port = serving(app)
    status, _, text = post(port, {"messages": MESSAGES})
    assert status == 503 and json.loads(text)["error"]["type"] == "server_error"


# -- 3. kill -USR1 ----------------------------------------------------------------------------------------------------
def test_usr1_registered_before_the_cuda_server(monkeypatch, tmp_path):
    import faulthandler
    import signal

    from tensorfold import cli, families, hub

    order = []
    monkeypatch.setattr(faulthandler, "register", lambda sig, all_threads=False: order.append(("usr1", sig)))
    monkeypatch.setattr(cli, "_config_dir", lambda model: tmp_path)
    monkeypatch.setattr(families, "detect", lambda d: SimpleNamespace(package=SimpleNamespace(), title="t"))
    monkeypatch.setattr(cli, "_backend", lambda choice, family: "cuda")
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(families, "read_config", lambda d: {})
    monkeypatch.setattr(cli, "_note_untested", lambda *a: None)
    monkeypatch.setattr(cli, "_model_context", lambda d: 0)
    monkeypatch.setattr(hub, "is_repo_id", lambda m: False)
    monkeypatch.setattr(hub, "resolve", lambda m, required_files=(): tmp_path)
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, family, model_dir: order.append("cuda") or 0)
    args = SimpleNamespace(no_update_check=True, model=str(tmp_path), backend="cuda", context=None)
    assert cli.cmd_serve(args) == 0
    assert order == [("usr1", signal.SIGUSR1), "cuda"]


# -- 4. image URLs ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("url,why", [
    ("http://img.example/a.png", "HTTPS on port 443"),
    ("https://img.example:8443/a.png", "HTTPS on port 443"),
    ("https://user:pw@img.example/a.png", "HTTPS on port 443"),
    ("https://img.example/a.png#x", "HTTPS on port 443"),
    ("https://img.example\\@evil/a.png", "HTTPS on port 443"),
    ("https://img.example/a b.png", "whitespace/control"),
    ("https://img.example/a\x00.png", "whitespace/control"),
    ("https://" + "a" * 5000 + ".example/", "too long"),
    ("https:///a.png", "HTTPS on port 443"),
    ("https://localhost/a.png", "public internet hosts"),
    ("https://metadata.google.internal/computeMetadata/v1/", "public internet hosts"),
    ("ftp://img.example/a.png", "HTTPS on port 443"),
])
def test_check_url_refuses(url, why):
    with pytest.raises(fetch.ImageFetchError, match=why):
        fetch.check_url(url)


def test_check_url_accepts_and_the_local_knobs():
    assert fetch.check_url("https://img.example/p/a.png?x=1") == ("https", "img.example", 443, "/p/a.png?x=1")
    assert fetch.check_url("https://IMG.example:443") == ("https", "img.example", 443, "/")
    local = fetch.Policy(http=True, private=True)
    assert fetch.check_url("http://127.0.0.1:8080/a.png", local) == ("http", "127.0.0.1", 8080, "/a.png")
    assert fetch.check_url("https://localhost/a.png", local)[1] == "localhost"
    with pytest.raises(fetch.ImageFetchError):
        fetch.check_url("http://user:pw@127.0.0.1/a.png", local)          # credentials never


BLOCKED = ["10.0.0.1", "172.16.5.4", "192.168.1.1", "127.0.0.1", "0.0.0.0", "169.254.169.254", "100.64.0.1",
           "100.100.100.200", "224.0.0.1", "255.255.255.255", "192.0.0.192", "168.63.129.16", "198.18.0.1",
           "::1", "::", "fe80::1", "fc00::1", "fd00:ec2::254", "::ffff:10.0.0.1", "::ffff:8.8.8.8",
           "2002:a00:1::1", "2001:0:4136:e378:8000:63bf:3fff:fdd2", "64:ff9b::a00:1", "64:ff9b:1::a00:1", "::a00:1",
           "ff02::1", "not-an-ip", "fe80::1%eth0"]
PUBLIC = ["8.8.8.8", "93.184.216.34", "1.1.1.1", "2606:4700:4700::1111", "2001:4860:4860::8888"]


@pytest.mark.parametrize("ip", BLOCKED)
def test_non_public_addresses(ip):
    assert fetch.public_ip(ip) is False


@pytest.mark.parametrize("ip", PUBLIC)
def test_public_addresses(ip):
    assert fetch.public_ip(ip) is True


def _dns(monkeypatch, table):
    """getaddrinfo from a table: host -> list of addresses (or a list of lists, one a lookup, for rebinding)."""

    calls = []

    def getaddrinfo(host, port, type=0, *a, **k):
        calls.append(host)
        found = table[host]
        if found and isinstance(found[0], list):
            found = found[min(sum(1 for c in calls if c == host) - 1, len(found) - 1)]
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
                for ip in found]

    monkeypatch.setattr(fetch.socket, "getaddrinfo", getaddrinfo)
    return calls


def test_resolve_needs_every_address_public(monkeypatch):
    _dns(monkeypatch, {"good.example": ["93.184.216.34", "2606:4700:4700::1111"],
                       "mixed.example": ["93.184.216.34", "10.1.2.3"]})
    deadline = time.monotonic() + 5
    assert len(fetch._resolve("good.example", 443, deadline, fetch.Policy())) == 2
    with pytest.raises(fetch.ImageFetchError, match="public internet addresses"):
        fetch._resolve("mixed.example", 443, deadline, fetch.Policy())
    assert len(fetch._resolve("mixed.example", 443, deadline, fetch.Policy(private=True))) == 2


def _fake_requests(monkeypatch, replies):
    """_request replaced: each call records (host, the address it would connect to) and returns the next reply."""

    seen = []

    def request(scheme, host, port, target, addresses, max_bytes, deadline):
        seen.append((host, addresses[0][4][0], target))
        return replies.pop(0)

    monkeypatch.setattr(fetch, "_request", request)
    return seen


def test_redirect_to_a_private_address_never_connects(monkeypatch):
    """DNS-rebinding style: a public image host redirects to a name that resolves to an internal address."""

    _dns(monkeypatch, {"img.example": ["93.184.216.34"], "internal.example": ["10.0.0.7"]})
    seen = _fake_requests(monkeypatch, [(None, "https://internal.example/latest/meta-data/", None)])
    with pytest.raises(fetch.ImageFetchError, match="public internet addresses"):
        fetch.fetch_image("https://img.example/cat.png", max_bytes=1000, deadline=time.monotonic() + 5)
    assert seen == [("img.example", "93.184.216.34", "/cat.png")]         # the private address was never connected


def test_rebinding_on_the_same_host_is_caught_at_the_redirect(monkeypatch):
    """The same name answers public first and private at the next lookup: every hop is resolved and checked, and
    each connection goes to the address checked for it (no second lookup inside the connect)."""

    calls = _dns(monkeypatch, {"flip.example": [["93.184.216.34"], ["127.0.0.1"]]})
    seen = _fake_requests(monkeypatch, [(None, "/again.png", None)])
    with pytest.raises(fetch.ImageFetchError, match="public internet addresses"):
        fetch.fetch_image("https://flip.example/a.png", max_bytes=1000, deadline=time.monotonic() + 5)
    assert calls == ["flip.example", "flip.example"] and seen == [("flip.example", "93.184.216.34", "/a.png")]


@pytest.mark.parametrize("location,why", [
    ("http://img2.example/a.png", "HTTPS on port 443"),
    ("https://127.0.0.1/a.png", "public internet addresses"),
    ("https://[::1]/a.png", "public internet addresses"),
    ("https://169.254.169.254/latest/", "public internet addresses"),
    ("https://img2.example:8080/a.png", "HTTPS on port 443"),
    ("file:///etc/passwd", "HTTPS on port 443"),
])
def test_every_redirect_is_rechecked(monkeypatch, location, why):
    _dns(monkeypatch, {"img.example": ["93.184.216.34"], "img2.example": ["1.1.1.1"], "127.0.0.1": ["127.0.0.1"],
                       "::1": ["::1"], "169.254.169.254": ["169.254.169.254"]})
    seen = _fake_requests(monkeypatch, [(None, location, None)])
    with pytest.raises(fetch.ImageFetchError, match=why):
        fetch.fetch_image("https://img.example/a.png", max_bytes=1000, deadline=time.monotonic() + 5)
    assert len(seen) == 1


def test_redirects_followed_to_a_limit(monkeypatch):
    _dns(monkeypatch, {"img.example": ["93.184.216.34"], "cdn.example": ["1.1.1.1"]})
    seen = _fake_requests(monkeypatch, [(None, "https://cdn.example/b.png", None), (b"PNG", None, "image/png")])
    assert fetch.fetch_image("https://img.example/a.png", max_bytes=1000,
                             deadline=time.monotonic() + 5) == (b"PNG", "image/png")
    assert [h for h, _, _ in seen] == ["img.example", "cdn.example"]
    _fake_requests(monkeypatch, [(None, "/r", None)] * 10)
    with pytest.raises(fetch.ImageFetchError, match="too many redirects"):
        fetch.fetch_image("https://img.example/a.png", max_bytes=1000, deadline=time.monotonic() + 5)


def _png(w=40, h=30) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 30)).save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def image_host():
    """A local image server: /ok.png, /html, /gzip, /big (no length), /bigdecl, /redir -> /ok.png, /loop, /stall."""

    png = _png()
    stall = threading.Event()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, headers, body=b""):
            self.send_response(code)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            p = self.path
            if p.startswith("/ok.png"):
                self._send(200, {"Content-Type": "image/png", "Content-Length": str(len(png))}, png)
            elif p == "/html":
                self._send(200, {"Content-Type": "text/html", "Content-Length": "2"}, b"hi")
            elif p == "/gzip":
                self._send(200, {"Content-Type": "image/png", "Content-Encoding": "gzip", "Content-Length": "2"},
                           b"hi")
            elif p == "/big":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"\0" * 5000)
                self.close_connection = True
            elif p == "/bigdecl":
                self._send(200, {"Content-Type": "image/png", "Content-Length": "999999"})
            elif p == "/redir":
                self._send(302, {"Location": "/ok.png?from=redir", "Content-Length": "0"})
            elif p == "/loop":
                self._send(302, {"Location": "/loop", "Content-Length": "0"})
            elif p == "/stall":
                stall.wait(5)
            else:
                self._send(404, {"Content-Length": "0"})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    stall.set()
    srv.shutdown()
    srv.server_close()


def test_local_fetch_with_the_testing_knobs(image_host):
    local = vp.Settings(fetch_http=True, fetch_private=True, max_bytes=1000)
    raw = vp.load_bytes(image_host + "/ok.png", local)
    assert raw == _png() and vp.decode(raw, local).size == (40, 30)
    assert vp.load_bytes(image_host + "/redir", local) == _png()
    for path, why in (("/html", "JPEG, PNG, WebP or GIF"), ("/gzip", "compressed"), ("/big", "MAX_BYTES"),
                      ("/bigdecl", "MAX_BYTES"), ("/loop", "too many redirects"), ("/nothing", "HTTP 404")):
        with pytest.raises(vp.VisionError, match=why):
            vp.load_bytes(image_host + path, local)
    with pytest.raises(vp.VisionError, match="HTTPS on port 443"):
        vp.load_bytes(image_host + "/ok.png", vp.Settings())               # the defaults refuse it
    with pytest.raises(vp.VisionError, match="public internet addresses"):
        vp.load_bytes(image_host + "/ok.png", vp.Settings(fetch_http=True))


def test_a_stalled_server_hits_the_deadline(image_host):
    local = vp.Settings(fetch_http=True, fetch_private=True, fetch_timeout=0.5)
    t0 = time.monotonic()
    with pytest.raises(vp.VisionError, match="could not fetch"):
        vp.load_bytes(image_host + "/stall", local)
    assert time.monotonic() - t0 < 3.0


def test_fetch_messages_never_carry_the_url(image_host):
    local = vp.Settings(fetch_http=True, fetch_private=True)
    secret = image_host.replace("http://", "http://user:hunter2@") + "/nothing?token=s3cret"
    for url in (secret, image_host + "/nothing?token=s3cret", "https://user:hunter2@img.example/?token=s3cret"):
        with pytest.raises(vp.VisionError) as info:
            vp.load_bytes(url, local)
        assert "hunter2" not in str(info.value) and "s3cret" not in str(info.value)


def test_env_knobs(monkeypatch):
    for k, v in {"GLM53_TF_VISION_FETCH_HTTP": "1", "GLM53_TF_VISION_FETCH_PRIVATE": "1",
                 "GLM53_TF_VISION_FETCH_TOTAL_S": "12", "GLM53_TF_VISION_PREP_SLOTS": "3",
                 "GLM53_TF_VISION_PREP_WAITERS": "5", "GLM53_TF_VISION_PREP_WAIT_S": "2"}.items():
        monkeypatch.setenv(k, v)
    s = vp.Settings.read(None)
    assert (s.fetch_http, s.fetch_private, s.fetch_total, s.prep_slots, s.prep_waiters, s.prep_wait) == \
        (True, True, 12.0, 3, 5, 2.0)
    monkeypatch.setenv("GLM53_TF_VISION_FETCH_HTTP", "yes")
    with pytest.raises(ValueError, match="GLM53_TF_VISION_FETCH_HTTP"):
        vp.Settings.read(None)
    monkeypatch.setenv("GLM53_TF_VISION_FETCH_HTTP", "0")
    monkeypatch.setenv("GLM53_TF_VISION_PREP_SLOTS", "0")
    with pytest.raises(ValueError, match="PREP_SLOTS"):
        vp.Settings.read(None)
    for k in ("GLM53_TF_VISION_FETCH_HTTP", "GLM53_TF_VISION_FETCH_PRIVATE", "GLM53_TF_VISION_PREP_SLOTS"):
        monkeypatch.delenv(k)
    d = vp.Settings.read(None)
    assert (d.fetch_http, d.fetch_private, d.prep_slots) == (False, False, 16)


def test_preparations_are_bounded(monkeypatch):
    monkeypatch.setenv("GLM53_TF_VISION_PREP_SLOTS", "1")
    monkeypatch.setenv("GLM53_TF_VISION_PREP_WAITERS", "1")
    monkeypatch.setenv("GLM53_TF_VISION_PREP_WAIT_S", "0.2")
    host = vp.Host(None)
    inside, go = threading.Event(), threading.Event()

    def slow(ref, deadline=None):
        inside.set()
        go.wait(WAIT)
        return SimpleNamespace(tokens=1)

    monkeypatch.setattr(host, "prepare_ref", slow)
    t = threading.Thread(target=lambda: host.request([object()]))
    t.start()
    assert inside.wait(WAIT)
    with pytest.raises(vp.VisionBusy, match="capacity is busy"):                   # waited 0.2 s for the slot
        host.request([object()])
    host.waiters.acquire()                                                          # the queue is full now
    try:
        with pytest.raises(vp.VisionBusy, match="queue is full"):
            host.request([object()])
    finally:
        host.waiters.release()
    go.set()
    t.join(WAIT)
    assert len(host.request([object()]).images) == 1                               # free again
    assert host.request([]).images == []


# -- 6. the small bundle ----------------------------------------------------------------------------------------------
def test_closed_json():
    c = server.closed_json
    assert c('{"a": [1, 2') == '{"a": [1, 2]}'
    assert c('[{"x": "]"}') == '[{"x": "]"}]'                                      # brackets inside strings ignored
    assert c('{"a": "open') is None                                                 # a string left open
    assert c('{"a": 1}}') is None and c("[1]") is None and c("") is None           # over-closed / complete / empty
    assert c('{"a": [1}') is None                                                   # the wrong closer


def test_glm_tool_arguments_one_bracket_short():
    tools = [{"type": "function", "function": {"name": "edit", "parameters": {"properties": {
        "edits": {"type": "array"}, "opts": {"type": "object"}, "path": {"type": "string"}, "n": {"type": "integer"}}}}}]
    text = ('<tool_call>edit<arg_key>edits</arg_key><arg_value>[{"old": "a", "new": "b"}</arg_value>'
            '<arg_key>opts</arg_key><arg_value>{"dry": true, "tags": ["x"</arg_value>'
            '<arg_key>path</arg_key><arg_value>[{"not json</arg_value>'
            '<arg_key>n</arg_key><arg_value>[1, 2</arg_value></tool_call>')
    _, calls = server.parse_tool_calls(text, tools)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["edits"] == [{"old": "a", "new": "b"}]
    assert args["opts"] == {"dry": True, "tags": ["x"]}
    assert args["path"] == '[{"not json'                                            # typed string: text as before
    assert args["n"] == "[1, 2"                                                     # not array / object: unchanged
    ok = '<tool_call>edit<arg_key>edits</arg_key><arg_value>[1]</arg_value></tool_call>'
    assert json.loads(server.parse_tool_calls(ok, tools)[1][0]["function"]["arguments"]) == {"edits": [1]}
    wrong = '<tool_call>edit<arg_key>opts</arg_key><arg_value>[1, 2</arg_value></tool_call>'
    assert json.loads(server.parse_tool_calls(wrong, tools)[1][0]["function"]["arguments"]) == {"opts": "[1, 2"}


@pytest.mark.parametrize("chunks", [(1,), (3,), (7,), (2, 5, 1, 9), (16,)])
def test_stop_string_count_is_token_exact(chunks):
    """Drafted rounds bring several tokens at once: completion_tokens is where the stop completes, as serial."""

    text = "hello world STOP and more text after it"
    eng = Paced(text=text, chunks=chunks)
    app = make_app(eng)
    body = {"messages": MESSAGES, "max_tokens": 200, "stop": ["STOP"], "return_token_ids": True}
    result = app.run(body, True, lambda d: True)
    at = text.index("STOP") + len("STOP")
    assert result["content"] == "hello world " and result["finish"] == "stop"
    assert result["completion_tokens"] == at
    assert result["stats"]["token_ids"] == eng.tok.ids(text[:at])


def test_return_token_ids(serving):
    eng = Paced(text="abc")
    app = make_app(eng)
    port = serving(app)
    ids = eng.tok.ids("abcab")
    status, _, text = post(port, {"messages": MESSAGES, "max_tokens": 5, "return_token_ids": True})
    assert status == 200 and json.loads(text)["tensorfold"]["token_ids"] == ids
    status, _, text = post(port, {"messages": MESSAGES, "max_tokens": 5})
    assert "token_ids" not in json.loads(text)["tensorfold"]
    status, _, text = post(port, {"messages": MESSAGES, "max_tokens": 5, "return_token_ids": True, "stream": True})
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
    assert events[-1]["tensorfold"]["token_ids"] == ids


# -- 9. /health live totals -------------------------------------------------------------------------------------------
def test_health_live_totals(serving):
    eng = Paced(text="abcd", stats={"prefill_s": 0.25, "decode_s": 0.5, "rounds": 4, "drafted": 12, "accepted": 4,
                                    "cached": 2}, hold_at=2)
    app = make_app(eng)
    eng.limit = 4096
    port = serving(app)
    _, before = get(port)
    assert before["ok"] is True and before["backend"] == "tensorfold" and before["busy"] is False
    assert before["requests_running"] == 0 and before["requests_total"] == 0 and before["context_length"] == 4096
    assert before["mode"] == "basic" and before["inflight"] == 0 and "streams" not in before       # 0150's fields
    reply = {}
    worker = threading.Thread(target=lambda: reply.update(r=post(port, {"messages": MESSAGES, "max_tokens": 4})))
    worker.start()
    assert eng.held.wait(WAIT)
    _, during = get(port)
    assert during["busy"] is True and during["requests_running"] == 1 and during["inflight"] == 1
    assert during["completion_tokens_total"] == 2 and during["rounds_total"] == 0       # live tokens, stats at end
    eng.release.set()
    worker.join(WAIT)
    assert reply["r"][0] == 200
    _, after = get(port)
    assert after["busy"] is False and after["requests_total"] == 1 and after["requests"] == 1
    assert after["completion_tokens_total"] == 4 and after["prompt_tokens_total"] > 0
    assert (after["prefill_seconds_total"], after["decode_seconds_total"]) == (0.25, 0.5)
    assert (after["rounds_total"], after["drafted_total"], after["accepted_total"], after["cached_tokens_total"]) == \
        (4, 12, 4, 2)


def test_health_counts_a_failed_request_and_batch_streams():
    h = Health(mode="basic", stall_s=0)

    class Failing:
        eos = (EOS,)

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            on_tokens([5, 6])
            raise RuntimeError("the engine failed")

    eng = h.track(Failing())
    with pytest.raises(RuntimeError):
        eng.generate([1, 2, 3], 4, None, lambda new: False)
    body = h.status()[1]
    assert body["requests_total"] == 1 and body["completion_tokens_total"] == 2 and body["prompt_tokens_total"] == 3
    assert body["rounds_total"] == 0 and body["errors"] == 1
    seqs = [SimpleNamespace(stepper=None), None, SimpleNamespace(stepper=object()), SimpleNamespace(stepper=object())]
    app = SimpleNamespace(engine=SimpleNamespace(batch=SimpleNamespace(seqs=seqs, n=4), limit=1048576))
    body = h.status(app)[1]
    assert body["streams"] == {"decoding": 2, "prefilling": 1, "max": 4} and body["context_length"] == 1048576
    assert "streams" not in h.status(SimpleNamespace(engine=SimpleNamespace(batch=None)))[1]
