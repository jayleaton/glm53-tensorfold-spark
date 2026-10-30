"""patches/0490: the context limit where clients can see it, and vLLM-shaped extras RigMark needs.

- ``/v1/models`` reports the engine's limit as ``max_model_len`` and ``context_length`` (vLLM / OpenRouter keys that
  clients such as Hermes Agent read); without a limit the entry is as before.
- A request past the limit gets HTTP 400 with OpenAI's ``context_length_exceeded`` code and wording, which opencode's
  overflow patterns and Hermes Agent's limit / split parsers recognise (checked against copies of their regexes).
- ``POST /tokenize`` (vLLM's shape) and one list of token ids as ``prompt`` on ``/v1/completions``.
- The ``[tensorfold] context:`` boot line.

Host only (fake engine and tokenizer). Run against the patched tree: PYTHONPATH=<tree>/src pytest tests/test_api_context.py
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request

import pytest

from tensorfold.cuda import server

if not hasattr(server, "context_problem"):
    pytest.skip("patches/0490 not applied", allow_module_level=True)

app_mod = pytest.importorskip("tensorfold.families.glm5_next.cuda.app")
from tensorfold.cuda.server import App, context_problem  # noqa: E402

EOS = 0


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class Tok:
    """One id per character; id 0 is the end of sequence. ``add_special_tokens`` prepends id 1 ("^")."""

    def __init__(self):
        self.vocab = ["<eos>", "^"]

    def id(self, piece):
        if piece not in self.vocab:
            self.vocab.append(piece)
        return self.vocab.index(piece)

    def encode(self, text, add_special_tokens=False):
        return Encoded(([1] if add_special_tokens else []) + [self.id(c) for c in text])

    def encode_batch_fast(self, texts, add_special_tokens=False):
        return [self.encode(t, add_special_tokens) for t in texts]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[i] for i in ids)

    def get_vocab_size(self, with_added_tokens=True):
        return 1000


class Engine:
    eos = (EOS,)

    def __init__(self, limit=None, reply=("o", "k")):
        if limit is not None:
            self.limit = limit
        self.reply = reply
        self.prompts = []
        self.max_tokens = []
        self.request = threading.local()

    def generate(self, prompt, max_tokens, sampling, on_tokens):
        self.prompts.append(list(prompt))
        self.max_tokens.append(max_tokens)
        tok = self.tok
        on_tokens([tok.id(c) for c in self.reply] + [EOS])
        return {}


class Template:
    def render(self, messages, *, tools, enable_thinking, extra=None):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


class PromptTokens:
    def __init__(self, tok):
        self.tok = tok

    def encode(self, text):
        return self.tok.encode(text).ids


def make_app(limit=None, *, glm=True, served="GLM-5.3-Flash-EXL3"):
    tok = Tok()
    engine = Engine(limit)
    engine.tok = tok
    app = object.__new__(app_mod.GlmApp if glm else App)
    app.engine = engine
    app.served = served
    app.tok = tok
    app.template = Template()
    app.default_thinking = False
    app.sampling = {"temperature": 0.0, "top_k": 1, "top_p": 1.0}
    app.max_tokens = 100
    app.lock = threading.Lock()
    app.created = 1700000000
    app.sampling_for = lambda body, prompt: None
    if glm:
        app.effort_field, app.default_effort = False, None
        app.prompt_tokens = PromptTokens(tok)
        app.prompt_memo = app_mod.Memo()
        app.reqlog = None
        app._rl = threading.local()
    try:
        from tensorfold.cuda.health import Health
        app.health = Health()
        app.engine = app.health.track(app.engine)
    except ImportError:
        pass
    return app


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


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


# -- the wording clients parse ---------------------------------------------------------------------------------------
# opencode (packages/llm/src/provider-error.ts, 2026-09-28): a matching message (or code context_length_exceeded)
# makes a turn a context overflow, which compacts the session instead of failing it
OPENCODE = [r"maximum context length is \d+ tokens", r"reduce the length of the messages", r"context[_ ]length[_ ]exceeded"]
# Hermes Agent (agent/model_metadata.py): the limit, and "(N in the messages, M in the completion)"
HERMES_LIMIT = r"(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})"
HERMES_SPLIT = (r"\((\d+)\s+(?:tokens\s+)?in (?:the messages|your prompt|the prompt)\s*[;,]\s*"
                r"(\d+)\s+(?:tokens\s+)?(?:in|for) the completion\)")


def test_context_problem_fits():
    assert context_problem(100, 28, 128) is None
    assert context_problem(127, None, 128) is None


def test_context_problem_wording_and_code():
    p = context_problem(60000, 32000, 65536)
    assert p.code == "context_length_exceeded" and p.param == "messages"
    assert p.startswith("This model's maximum context length is 65536 tokens. However, you requested 92000 tokens "
                        "(60000 in the messages, 32000 in the completion).")
    assert "CONTEXT=92000" in p
    for pat in OPENCODE[:2]:
        assert re.search(pat, p, re.I), pat
    assert re.search(HERMES_LIMIT, p.lower()).group(1) == "65536"
    assert re.search(HERMES_SPLIT, p).groups() == ("60000", "32000")
    q = context_problem(70000, None, 65536, chat=False)
    assert q.param == "prompt" and "your prompt resulted in 70000 tokens" in q


# -- /v1/models --------------------------------------------------------------------------------------------------------
def test_models_reports_the_limit(http):
    body = get(http(make_app(1048576)) + "/v1/models")
    m = body["data"][0]
    assert m["id"] == "GLM-5.3-Flash-EXL3" and m["object"] == "model" and m["owned_by"] == "tensorfold"
    assert m["max_model_len"] == m["context_length"] == 1048576
    assert m["root"] == "GLM-5.3-Flash-EXL3" and m["created"] == 1700000000


def test_models_without_a_limit_is_unchanged(http):
    body = get(http(make_app(None, glm=False)) + "/v1/models")
    assert body["data"] == [{"id": "GLM-5.3-Flash-EXL3", "object": "model", "owned_by": "tensorfold"}]


# -- a request past the limit ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("stream", [False, True])
def test_over_limit_is_a_400_with_the_code(http, stream):
    app = make_app(64)
    base = http(app)
    code, text = post(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "x" * 40}],
                                                      "max_tokens": 32, "stream": stream})
    assert code == 400
    err = json.loads(text)["error"]
    assert err["type"] == "invalid_request_error" and err["code"] == "context_length_exceeded"
    assert err["param"] == "messages"
    assert err["message"].startswith("This model's maximum context length is 64 tokens. However, you requested 89 "
                                     "tokens (57 in the messages, 32 in the completion).")
    assert app.engine.prompts == []
    # within the limit it runs
    code, text = post(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "x" * 40}],
                                                      "max_tokens": 4})
    assert code == 200 and json.loads(text)["choices"][0]["message"]["content"] == "ok"


def test_plain_errors_keep_their_shape(http):
    base = http(make_app(64))
    code, text = post(base + "/v1/chat/completions", {"messages": [], "n": 3})
    assert code == 400 and json.loads(text)["error"] == {
        "message": "n must be 1 on this server (got 3): it returns one choice a request", "type": "invalid_request_error"}


def test_stream_engine_refusal_is_a_client_error(http):
    app = make_app(None, glm=False)

    def refuse(*a, **k):
        raise ValueError("prompt (9) + max_tokens (9) needs 2 KV pages of 256 tokens; the pool has 1")

    inner = getattr(app.engine, "_engine", app.engine)      # 0150's wrapper binds generate when it is made
    inner.generate = refuse
    if hasattr(app, "health"):
        app.engine = app.health.track(inner)
    code, text = post(http(app) + "/v1/chat/completions", {"messages": [], "stream": True})
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
    assert events[-1]["error"]["type"] == "invalid_request_error" and "KV pages" in events[-1]["error"]["message"]


# -- /tokenize and token-ID prompts ----------------------------------------------------------------------------------
def test_tokenize_string_and_messages(http):
    app = make_app(1048576)
    base = http(app)
    code, text = post(base + "/tokenize", {"model": "GLM-5.3-Flash-EXL3", "prompt": "hello",
                                           "add_special_tokens": False})
    body = json.loads(text)
    assert code == 200 and body["tokens"] == app.tok.encode("hello").ids and body["count"] == 5
    assert body["max_model_len"] == 1048576 and body["token_strs"] is None
    code, text = post(base + "/v1/tokenize", {"prompt": "hello"})          # vLLM's default: special tokens on
    assert json.loads(text)["tokens"] == [1] + app.tok.encode("hello").ids
    msgs = [{"role": "user", "content": "hi"}]
    code, text = post(base + "/tokenize", {"messages": msgs})
    assert json.loads(text)["tokens"] == app.tok.encode(Template().render(msgs, tools=None, enable_thinking=False)).ids
    code, text = post(base + "/tokenize", {"prompt": [1, 2]})
    assert code == 400 and "string prompt" in json.loads(text)["error"]["message"]
    assert app.engine.prompts == []


def test_token_id_prompt_runs_as_given(http):
    app = make_app(1048576)
    base = http(app)
    ids = [5, 6, 7, 999, 3]
    code, text = post(base + "/v1/completions", {"prompt": ids, "max_tokens": 8, "stream": True,
                                                 "stream_options": {"include_usage": True}, "ignore_eos": True})
    assert code == 200
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
    assert events[-1]["usage"]["prompt_tokens"] == len(ids)
    assert app.engine.prompts[-1] == ids and app.engine.max_tokens[-1] == 8
    code, text = post(base + "/v1/completions", {"prompt": ids, "max_tokens": 4})
    assert code == 200 and json.loads(text)["usage"]["prompt_tokens"] == len(ids)


@pytest.mark.parametrize("prompt,words", [([], "must not be empty"), ([[1, 2], [3]], "one prompt a request"),
                                         (["a", "b"], "one prompt a request"), ([1, True], "list of integers"),
                                         ([1, 2.0], "list of integers"), ([1, -1], "outside the vocabulary"),
                                         ([1, 1000], "outside the vocabulary")])
def test_bad_token_ids_are_400(http, prompt, words):
    app = make_app(1048576)
    code, text = post(http(app) + "/v1/completions", {"prompt": prompt, "max_tokens": 4})
    assert code == 400 and words in json.loads(text)["error"]["message"]
    assert json.loads(text)["error"]["param"] == "prompt" and app.engine.prompts == []


def test_64k_token_prompt_and_the_limit(http):
    ids = [7] * 65536
    base = http(make_app(1048576))
    code, text = post(base + "/v1/completions", {"prompt": ids, "max_tokens": 8})
    assert code == 200 and json.loads(text)["usage"]["prompt_tokens"] == 65536
    code, text = post(http(make_app(65536)) + "/v1/completions", {"prompt": ids, "max_tokens": 8})
    err = json.loads(text)["error"]
    assert code == 400 and err["code"] == "context_length_exceeded" and err["param"] == "prompt"
    assert "(65536 in the prompt, 8 in the completion)" in err["message"]


# -- the boot line ---------------------------------------------------------------------------------------------------
class Pool:
    npages, page = 4096, 256


class Batch:
    n, kvp = 4, Pool()


def test_context_summary(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LATENT_KV", "1")
    monkeypatch.setenv("GLM53_TF_KV_DTYPE", "fp8")
    e = Engine(1048576)
    e.batch = Batch()
    line = app_mod.context_summary(e)
    assert line.startswith("context: 1048576 tokens a request")
    assert "KV cache: latent fp8" in line and "4 request slot(s)" in line
    assert "KV pool: 1048576 tokens shared by the slots" in line and "HTTP 400" not in line
    monkeypatch.delenv("GLM53_TF_LATENT_KV")
    small = app_mod.context_summary(Engine(32768))
    assert "per-head K/V" in small and "1 request slot(s)" in small
    assert "Requests past 32768 tokens" in small and "config/prod.env.example" in small


# -- the pool admits every request the context check lets through (0290 + 0490) ------------------------------------
def test_pool_holds_every_request_past_the_check():
    kvpool = pytest.importorskip("tensorfold.families.glm5_next.cuda.kvpool")
    rows, page, slack = 16, 256, kvpool.DEFAULT_SLACK
    for context in (1048576, 262144, 1000000, 65536):
        capacity, limit = context + rows, context
        set_tokens = -(-context // page) * page                  # settings() rounds to a page
        pool = kvpool.pool_tokens(set_tokens, page, capacity, limit)
        assert set_tokens <= pool <= set_tokens + page
        for total in range(limit - 200, limit + 1):             # prompt + max_tokens that pass the check
            for prompt in (1, total // 2, total - 1):
                need = kvpool.pages_for(kvpool.need_tokens(prompt, total - prompt, capacity, slack), page)
                assert need <= pool // page, (context, prompt, total)
    # the control: before 0490 the pool was CONTEXT, and a request at the limit needed one page more
    need = kvpool.pages_for(kvpool.need_tokens(1048000, 576, 1048576 + rows, slack), page)
    assert need == 4097 > 1048576 // page
    # a pool set smaller than one request stays as set
    assert kvpool.pool_tokens(524288, page, 1048576 + rows, 1048576) == 524288
    assert kvpool.pool_tokens(0, page, 1048576 + rows, 1048576) == 0
