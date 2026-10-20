"""patches/0160: OpenAI compatibility for replacing a vLLM server: ``stop`` strings, the ``reasoning`` alias of
``reasoning_content`` (GLM53_TF_REASONING_FIELDS), ``n`` > 1 refused, ``/v1/models``.

Host only: a fake engine replays fixed token chunks through ``on_tokens`` and a fake tokenizer maps ids to text
pieces. Run against the patched tree: PYTHONPATH=<tree>/src pytest tests/test_openai_compat.py
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from tensorfold.cuda import server
from tensorfold.cuda.server import App

EOS = 0
TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
    "command": {"type": "string"}}}}}]


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class Tok:
    """Token id i is the text piece ``vocab[i]``; id 0 is the end of sequence."""

    def __init__(self):
        self.vocab = ["<eos>"]

    def id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab.append(piece)
        return self.vocab.index(piece)

    def encode(self, text, add_special_tokens=False):
        return Encoded([self.id(c) for c in text])

    def encode_batch_fast(self, texts, add_special_tokens=False):
        return [self.encode(t, add_special_tokens) for t in texts]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[i] for i in ids)


class Engine:
    """Replays ``chunks`` (lists of token ids) through ``on_tokens``. ``batch``: stop at the first True, as the batch
    engine's cancel does (a round later: one more chunk); otherwise decode on regardless, as the single engine."""

    eos = (EOS,)

    def __init__(self, chunks, batch=False):
        self.chunks, self.batch = chunks, batch
        self.stopped_after = None

    def generate(self, prompt, max_tokens, sampling, on_tokens):
        for i, chunk in enumerate(self.chunks):
            if on_tokens(chunk) and self.stopped_after is None:
                self.stopped_after = i
                if self.batch:
                    if i + 1 < len(self.chunks):
                        on_tokens(self.chunks[i + 1])
                    break
        return {}


class Template:
    def render(self, messages, *, tools, enable_thinking, extra=None):
        return "prompt"


def make_app(pieces, *, batch=False, served="glm-5.3-flash"):
    """``pieces``: a list of chunks, each a list of text pieces (one token each); "<eos>" ends the reply."""

    tok = Tok()
    chunks = [[tok.id(p) for p in chunk] for chunk in pieces]
    app = object.__new__(App)
    app.engine = Engine(chunks, batch=batch)
    app.served = served
    app.tok = tok
    app.template = Template()
    app.default_thinking = False
    app.sampling = {"temperature": 0.0, "top_k": 1, "top_p": 1.0}
    app.max_tokens = 100
    app.lock = threading.Lock()
    app.sampling_for = lambda body, prompt: None
    try:                                  # patches/0150: the handler reads app.health (absent without 0150)
        from tensorfold.cuda.health import Health
        app.health = Health()
        app.engine = app.health.track(app.engine)
    except ImportError:
        pass
    return app


def run(app, body, chat=True):
    deltas = []
    result = app.run(dict({"messages": [{"role": "user", "content": "hi"}]}, **body), chat,
                     lambda d: deltas.append(d) or True)
    return result, deltas


def streamed(deltas, result, key="content"):
    return "".join(d.get(key, "") for d in deltas) + result["final"].get(key, "")


THINK = {"chat_template_kwargs": {"enable_thinking": True}}


@pytest.fixture(autouse=True)
def _fields(monkeypatch):
    monkeypatch.delenv("GLM53_TF_REASONING_FIELDS", raising=False)


# -- stop ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("batch", [False, True])
def test_stop_cuts_reply_and_finishes_stop(batch):
    app = make_app([["Hello"], [" wor"], ["ld"], ["END"], [" more"], ["<eos>"]], batch=batch)
    result, deltas = run(app, {"stop": "END"})
    assert result["content"] == "Hello world"
    assert streamed(deltas, result) == "Hello world"
    assert result["finish"] == "stop"
    assert result["completion_tokens"] == 4        # up to the chunk the stop string arrived in
    assert "more" not in json.dumps(deltas)


def test_stop_split_across_chunks_never_streams_a_prefix():
    app = make_app([["abc S"], ["T"], ["O"], ["P tail"], ["<eos>"]])
    result, deltas = run(app, {"stop": ["STOP"]})
    assert result["content"] == "abc "
    for d in deltas:                               # nothing of "STOP" ever went out
        assert "S" not in d.get("content", "")
    assert streamed(deltas, result) == "abc "
    assert result["finish"] == "stop"


def test_held_back_prefix_is_released_when_it_is_not_a_stop():
    app = make_app([["x ST"], ["ay"], ["<eos>"]])
    result, deltas = run(app, {"stop": ["STOP"]})
    assert deltas[0].get("content") == "x "       # "ST" held back while it could start "STOP"
    assert streamed(deltas, result) == "x STay"
    assert result["content"] == "x STay"
    assert result["finish"] == "stop"             # the EOS


def test_earliest_of_several_stops_wins():
    app = make_app([["one "], ["two "], ["three"], ["<eos>"]])
    result, deltas = run(app, {"stop": ["three", "two", ""]})
    assert result["content"] == "one "
    assert streamed(deltas, result) == "one "


def test_no_stop_hit_is_unchanged():
    app = make_app([["plain"], [" text"]])
    result, deltas = run(app, {"stop": ["zzz"]})
    assert result["content"] == streamed(deltas, result) == "plain text"
    assert result["finish"] == "length"


def test_stop_inside_reasoning_is_ignored():
    app = make_app([["think STOP"], [" more"], ["</think>"], ["answer "], ["STOP"], [" after"], ["<eos>"]])
    result, deltas = run(app, dict(THINK, stop="STOP"))
    assert result["reasoning"] == "think STOP more"
    assert streamed(deltas, result, "reasoning_content") == "think STOP more"
    assert result["content"] == streamed(deltas, result) == "answer "
    assert result["finish"] == "stop"


def test_stop_inside_tool_call_is_ignored():
    call = ["<tool_call>", "bash", "<arg_key>", "command", "</arg_key>", "<arg_value>", "echo STOP",
            "</arg_value>", "</tool_call>"]
    app = make_app([["Running."], call, ["<eos>"]])
    result, deltas = run(app, {"stop": "STOP", "tools": TOOLS})
    assert result["finish"] == "tool_calls"
    assert json.loads(result["calls"][0]["function"]["arguments"]) == {"command": "echo STOP"}
    assert result["content"] == "Running."


def test_stop_before_tool_call_drops_the_call():
    call = ["<tool_call>", "bash", "<arg_key>", "command", "</arg_key>", "<arg_value>", "ls", "</arg_value>",
            "</tool_call>"]
    app = make_app([["Done.", "STOP"], call, ["<eos>"]])
    result, deltas = run(app, {"stop": "STOP", "tools": TOOLS})
    assert result["calls"] is None
    assert result["content"] == "Done."
    assert result["finish"] == "stop"


def test_stop_after_tool_call_keeps_the_call():
    call = ["<tool_call>", "bash", "<arg_key>", "command", "</arg_key>", "<arg_value>", "ls", "</arg_value>",
            "</tool_call>"]
    app = make_app([["a"], call, ["b STOP c"], ["<eos>"]])
    result, deltas = run(app, {"stop": "STOP", "tools": TOOLS})
    assert result["calls"] and result["finish"] == "tool_calls"
    assert result["content"] == "ab"


def test_stop_in_text_completion():
    app = make_app([["1, 2, "], ["3\n\n4"], ["<eos>"]])
    result = app.run({"prompt": "count", "stop": "\n\n"}, False, lambda d: True)
    assert result["content"] == "1, 2, 3"
    assert result["finish"] == "stop"


# -- reasoning fields ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode,keys", [(None, {"reasoning_content", "reasoning"}), ("both", {"reasoning_content", "reasoning"}),
                                       ("reasoning_content", {"reasoning_content"}), ("reasoning", {"reasoning"})])
def test_reasoning_fields_in_stream(monkeypatch, mode, keys):
    if mode:
        monkeypatch.setenv("GLM53_TF_REASONING_FIELDS", mode)
    app = make_app([["let me "], ["think"], ["</think>"], ["hi"], ["<eos>"]])
    result, deltas = run(app, THINK)
    for key in {"reasoning_content", "reasoning"}:
        text = streamed(deltas, result, key)
        assert text == ("let me think" if key in keys else "")
    for d in deltas:
        if "reasoning_content" in d and "reasoning" in d:
            assert d["reasoning_content"] == d["reasoning"]


# -- over HTTP ---------------------------------------------------------------------------------------------------------
@pytest.fixture
def http():
    from http.server import ThreadingHTTPServer

    servers = []

    def start(app):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_http_message_reasoning_alias_and_stop(http, monkeypatch):
    base = http(make_app([["hmm"], ["</think>"], ["yes"], ["###"], ["no"], ["<eos>"]]))
    code, text = post(base + "/v1/chat/completions", dict(THINK, model="anything-at-all", stop=["###"],
                                                          messages=[{"role": "user", "content": "q"}]))
    assert code == 200
    body = json.loads(text)
    msg = body["choices"][0]["message"]
    assert msg["content"] == "yes"
    assert msg["reasoning_content"] == msg["reasoning"] == "hmm"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["model"] == "glm-5.3-flash"

    monkeypatch.setenv("GLM53_TF_REASONING_FIELDS", "reasoning")
    base = http(make_app([["hmm"], ["</think>"], ["yes"], ["<eos>"]]))
    msg = json.loads(post(base + "/v1/chat/completions", dict(THINK, messages=[]))[1])["choices"][0]["message"]
    assert msg["reasoning"] == "hmm" and "reasoning_content" not in msg


def test_http_stream_reasoning_alias_and_stop(http):
    base = http(make_app([["hmm"], ["</think>"], ["ye"], ["s#"], ["##"], ["no"], ["<eos>"]]))
    code, text = post(base + "/v1/chat/completions", dict(THINK, stream=True, stop="###", messages=[]))
    assert code == 200
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
    deltas = [e["choices"][0]["delta"] for e in events]
    assert "".join(d.get("content", "") for d in deltas) == "yes"
    assert "".join(d.get("reasoning", "") for d in deltas) == "hmm"
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "hmm"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert text.rstrip().endswith("data: [DONE]")


def test_http_models_lists_served_name(http):
    base = http(make_app([["x"]], served="my-prod-name"))
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
        assert [m["id"] for m in json.loads(r.read())["data"]] == ["my-prod-name"]


@pytest.mark.parametrize("extra,words", [({"n": 2}, "n must be 1"), ({"stop": 5}, "stop must be"),
                                         ({"stop": ["a", 1]}, "stop must be")])
def test_http_rejects_n_and_bad_stop(http, extra, words):
    base = http(make_app([["x"]]))
    code, text = post(base + "/v1/chat/completions", dict({"messages": []}, **extra))
    assert code == 400 and words in json.loads(text)["error"]["message"]


def test_http_accepts_n_1_and_unknown_fields(http):
    base = http(make_app([["ok"], ["<eos>"]]))
    code, text = post(base + "/v1/chat/completions", {"messages": [], "n": 1, "logprobs": True,
                                                      "frequency_penalty": 0.5, "user": "u", "whatever": {"x": 1}})
    assert code == 200 and json.loads(text)["choices"][0]["message"]["content"] == "ok"
