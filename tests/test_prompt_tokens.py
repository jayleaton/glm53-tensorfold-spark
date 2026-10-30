"""patches/0210: a request's prompt encoded once (``check`` -> ``run``), and incrementally across turns
(GLM53_TF_TOKCACHE): the ids must be exactly ``tok.encode(text, add_special_tokens=False).ids``.

Host only; needs the checkpoint's tokenizer.json, tokenizer_config.json and chat_template.jinja in
$GLM53_TF_TOKENIZER_DIR (not in the repo; skipped without them). Run against the patched tree:
GLM53_TF_TOKENIZER_DIR=<dir> PYTHONPATH=<tree>/src pytest -q -s tests/test_prompt_tokens.py
"""

from __future__ import annotations

import copy
import os
import random
import threading
import time
from pathlib import Path

import pytest

pt = pytest.importorskip("tensorfold.families.glm5_next.cuda.prompt_tokens")
app_mod = pytest.importorskip("tensorfold.families.glm5_next.cuda.app")
tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

TOKDIR = Path(os.environ.get("GLM53_TF_TOKENIZER_DIR", "/nonexistent"))
if not all((TOKDIR / f).exists() for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")):
    pytest.skip("GLM53_TF_TOKENIZER_DIR has no GLM tokenizer.json / tokenizer_config.json / chat_template.jinja",
                allow_module_level=True)

from tensorfold.cuda.server import ChatTemplate  # noqa: E402

TOK = tokenizers.Tokenizer.from_file(str(TOKDIR / "tokenizer.json"))
TEMPLATE = app_mod.ThinkingOffTemplate(ChatTemplate(TOKDIR))


def full(text):
    return TOK.encode(text, add_special_tokens=False).ids


def render(messages, tools=None, thinking=True, extra=None):
    return TEMPLATE.render(copy.deepcopy(messages), tools=tools or [], enable_thinking=thinking, extra=extra or {})


def cache(mode="on", entries=32, logs=None):
    return pt.PromptTokens(TOK, TOKDIR / "tokenizer.json", mode=mode, entries=entries,
                           log=(logs.append if logs is not None else None))


# -- random conversations -----------------------------------------------------------------------------------------

PIECES = ["hello", " world", "\n", "\n\n", "  ", "\t", " ", "é", "naïve", "日本語のテキスト", "中文", "🙂", "👩‍💻", "🇻🇳",
          "def f(x):\n    return x ** 2\n", "```python\nprint('hi')\n```", "{\"a\": [1, 2, 3]}", "<", ">", "|", "<|",
          "|>", "<|user", "user|>", "<|user|>", "<|assistant|>", "<think>", "</think>", "<tool_call>", "</arg_value>",
          "<|observation|>", "<|endoftext|>", "[gMASK]", "123456789", "3.14159", "'s", "'ll", "\r\n", "​",
          "   \n  ", "Ω≈ç√∫", "ﬁ", "Å", "ｱｲｳ", "\\n", "\\\"", "---", "***", "...", "$", "%"]


def rand_text(rng, lo=0, hi=12):
    return "".join(rng.choice(PIECES) for _ in range(rng.randint(lo, hi)))


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file 📄",
                                             "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                            "required": ["path"]}}},
         {"type": "function", "function": {"name": "bash", "description": "Run a command",
                                             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}]


def next_messages(rng, messages, call_ids):
    """Append one turn: a user message, or an assistant reply (thinking / tool calls) with its tool results."""

    last = messages[-1]["role"] if messages else None
    if last in (None, "system", "assistant", "tool") and (last != "assistant" or rng.random() < 0.7) \
            and not (last == "assistant" and messages[-1].get("tool_calls")):
        if last == "tool" and rng.random() < 0.5:
            pass                                    # an assistant reply after tool results instead
        else:
            content = rand_text(rng, 1) if rng.random() < 0.8 else [{"type": "text", "text": rand_text(rng, 1)}]
            messages.append({"role": "user", "content": content})
            return
    if last == "assistant" and messages[-1].get("tool_calls"):
        for call in messages[-1]["tool_calls"]:
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": rand_text(rng, 0, 20)})
        return
    m = {"role": "assistant", "content": rand_text(rng, 0)}
    r = rng.random()
    if r < 0.4:
        m["reasoning_content"] = rand_text(rng, 0)
    elif r < 0.55:
        m["content"] = "<think>" + rand_text(rng, 0) + "</think>" + m["content"]
    if rng.random() < 0.35:
        calls = []
        for _ in range(rng.randint(1, 3)):
            call_ids[0] += 1
            args = {"path": rand_text(rng, 1, 4)} if rng.random() < 0.5 else {"cmd": rand_text(rng, 1, 4), "n": 3}
            calls.append({"id": f"call_{call_ids[0]}", "type": "function",
                          "function": {"name": rng.choice(["read_file", "bash"]),
                                       "arguments": args if rng.random() < 0.5 else __import__("json").dumps(args)}})
        m["tool_calls"] = calls
    messages.append(m)


@pytest.mark.parametrize("seed", range(40))
def test_incremental_equals_full_random_conversations(seed):
    rng = random.Random(seed)
    logs: list[str] = []
    pc = cache("verify", logs=logs)
    messages = []
    if rng.random() < 0.7:
        messages.append({"role": "system", "content": rand_text(rng, 1)})
    tools = TOOLS if rng.random() < 0.5 else None
    call_ids = [0]
    for turn in range(25):
        next_messages(rng, messages, call_ids)
        thinking = rng.random() < 0.7
        extra = {"reasoning_effort": rng.choice(["low", "high"])} if thinking and rng.random() < 0.3 else {}
        if rng.random() < 0.1:
            extra["clear_thinking"] = False
        if rng.random() < 0.08 and messages and messages[0]["role"] == "system":
            messages[0]["content"] = rand_text(rng, 1)      # the system prompt changes mid-conversation
        text = render(messages, tools, thinking, extra)
        ids = pc.encode(text)
        assert ids == full(text), (seed, turn)
    assert pc.stats["mismatch"] == 0, logs
    assert pc.stats["hit"] > 0


def test_interleaved_sessions_and_edits():
    """Several conversations through one cache, a regenerate (same prompt again), an edited earlier message."""

    rng = random.Random(1234)
    pc = cache("on", entries=8)
    convs = [[{"role": "system", "content": "You are helpful. 🙂"}] for _ in range(5)]
    ids_counter = [0]
    for step in range(200):
        messages = rng.choice(convs)
        r = rng.random()
        if r < 0.1 and len(messages) > 3:
            i = rng.randrange(1, len(messages))
            if isinstance(messages[i].get("content"), str):
                messages[i]["content"] += rand_text(rng, 1, 3)       # an edited earlier message
        elif r < 0.2:
            pass                                                     # the same prompt again
        else:
            next_messages(rng, messages, ids_counter)
        text = render(messages, TOOLS, thinking=rng.random() < 0.5)
        assert pc.encode(text) == full(text), step
    assert pc.stats["hit"] > 100


def test_raw_prompts_and_edge_texts():
    pc = cache("verify")
    base = "[gMASK]<sop><|system|>sys<|user|>hi<|assistant|><think>"
    texts = [base, base + "x", base + "</think>ok<|user|>", base + "</think>ok<|user|> again 🙂<|assistant|>",
             base[:-1], base + "<|user|><|user|><|user|>", "", "no special tokens at all", "<|user|>",
             "<|user|>" + "a" * 5000 + "<|assistant|>", "<|user|>" + "a" * 5000 + "<|assistant|>" + "b",
             "<|user|>x<|assistant|>\n\n  <|observation|>   <tool_response>é</tool_response>"]
    for t in texts * 2:
        assert pc.encode(t) == full(t), t[:80]
    assert pc.stats["mismatch"] == 0


def test_tokenizer_preconditions():
    import json

    spec = json.loads((TOKDIR / "tokenizer.json").read_text())
    assert pt._unsafe(spec) is None
    cuts = pt.cut_tokens(spec)
    names = set(cuts.values())
    assert {"<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"} <= names
    # the checker refuses what would break the argument
    assert pt._unsafe({**spec, "normalizer": {"type": "NFC"}})
    assert pt._unsafe({**spec, "pre_tokenizer": {"type": "Metaspace", "prepend_scheme": "first"}})
    bad = copy.deepcopy(spec)
    bad["added_tokens"][0]["rstrip"] = True
    assert pt._unsafe(bad)
    over = copy.deepcopy(spec)
    over["added_tokens"].append({"id": 999999, "content": "|>x", "special": False})
    assert "<|user|>" not in pt.cut_tokens(over).values()
    over = copy.deepcopy(spec)
    over["added_tokens"].append({"id": 999999, "content": "<|user|>!", "special": False})
    assert "<|user|>" not in pt.cut_tokens(over).values()


def test_mode_env(monkeypatch):
    monkeypatch.delenv("GLM53_TF_TOKCACHE", raising=False)
    assert pt.tokcache_env() == ("on", 32)
    monkeypatch.setenv("GLM53_TF_TOKCACHE", "0")
    assert pt.tokcache_env()[0] == "off"
    monkeypatch.setenv("GLM53_TF_TOKCACHE", "verify")
    assert pt.tokcache_env()[0] == "verify"
    monkeypatch.setenv("GLM53_TF_TOKCACHE", "maybe")
    with pytest.raises(ValueError):
        pt.tokcache_env()
    pc = pt.PromptTokens(TOK, TOKDIR / "tokenizer.json", mode="off", entries=4)
    t = render([{"role": "user", "content": "hi"}])
    assert pc.encode(t) == full(t) and not pc.lru


def test_verify_mode_catches_a_bad_entry():
    logs: list[str] = []
    pc = cache("verify", logs=logs)
    t1 = render([{"role": "user", "content": "one"}])
    pc.encode(t1)
    e = next(iter(pc.lru.values()))
    e.ids[2] = 7                                   # corrupt the cached ids
    t2 = render([{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
                 {"role": "user", "content": "three"}])
    assert pc.encode(t2) == full(t2)
    assert pc.stats["mismatch"] == 1 and "MISMATCH" in logs[0]


# -- the app: one encode a request, same limits and errors -----------------------------------------------------------

class FakeEngine:
    def __init__(self, limit):
        self.limit = limit
        self.eos = (154820,)
        self.request = threading.local()
        self.prompts = []

    def generate(self, prompt, max_tokens, sampling, on_tokens):
        self.prompts.append(list(prompt))
        on_tokens([self.eos[0]])
        return {}


def make_app(monkeypatch, limit, mode):
    monkeypatch.setenv("GLM53_TF_TOKCACHE", mode)
    app = app_mod.GlmApp(FakeEngine(limit), TOKDIR, "glm", default_thinking=True)
    calls = {"n": 0}
    real = app.tok

    class Counting:
        def __getattr__(self, name):
            return getattr(real, name)

        def encode(self, *a, **k):
            calls["n"] += 1
            return real.encode(*a, **k)

        def encode_batch_fast(self, *a, **k):  # 0700 tokenizes the prompt through this
            calls["n"] += 1
            return real.encode_batch_fast(*a, **k)

    app.tok = app.prompt_tokens.tok = Counting()
    return app, calls


def old_check_message(prompt, asked, limit):
    """The context check's answer (patches/0490: OpenAI's context_length_exceeded wording and code)."""

    from tensorfold.cuda.server import context_problem

    return context_problem(prompt, int(asked) if asked else None, limit)


@pytest.mark.parametrize("mode", ["0", "1", "verify"])
def test_check_then_run_encodes_once(monkeypatch, mode):
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello 🙂 " * 50}]
    text = render(messages)
    n = len(full(text))
    for limit, asked in ((n + 10, 5), (n + 10, 50), (n, None), (n + 1, None), (4096, None)):
        app, calls = make_app(monkeypatch, limit, mode)
        body = {"messages": copy.deepcopy(messages), "max_tokens": asked}
        got, want = app.check(body), old_check_message(n, asked, limit)
        assert got == want and getattr(got, "code", None) == getattr(want, "code", None)
        assert calls["n"] == 1
        if app.check(body) is None:
            calls["n"] = 0
            app.check(body)
            app.run(body, True, lambda d: True)
            assert app.engine.prompts[-1] == full(text)
            # once; the cache (modes 1, verify) already holds this exact prompt from the first check
            assert calls["n"] == (1 if mode == "0" else 0), "check + run must encode once"


def test_run_without_check_and_completions(monkeypatch):
    app, calls = make_app(monkeypatch, 100000, "1")
    body = {"messages": [{"role": "user", "content": "hi"}]}
    app.run(body, True, lambda d: True)
    assert app.engine.prompts[-1] == full(render(body["messages"]))
    body = {"prompt": "<|user|>raw prompt<|assistant|>"}
    assert app.check(body) is None
    app.run(body, False, lambda d: True)
    assert app.engine.prompts[-1] == full(body["prompt"])


def test_multiturn_through_the_app(monkeypatch):
    app, calls = make_app(monkeypatch, 10 ** 6, "verify")
    rng = random.Random(7)
    messages = [{"role": "system", "content": "You are GLM."}]
    ids_counter = [0]
    for _ in range(15):
        next_messages(rng, messages, ids_counter)
        body = {"messages": copy.deepcopy(messages), "tools": TOOLS}
        assert app.check(body) is None
        app.run(body, True, lambda d: True)
        assert app.engine.prompts[-1] == full(render(messages, TOOLS))
    assert app.prompt_tokens.stats["mismatch"] == 0


# -- timing ---------------------------------------------------------------------------------------------------------

def long_conversation(target_tokens, rng):
    code = (TOKDIR / "chat_template.jinja").read_text()
    messages = [{"role": "system", "content": "You are a coding agent. " * 40}]
    ids_counter = [0]
    while True:
        messages.append({"role": "user", "content": rand_text(rng, 5, 30)})
        call_ids = ids_counter[0] = ids_counter[0] + 1
        messages.append({"role": "assistant", "content": "", "reasoning_content": rand_text(rng, 5, 30),
                         "tool_calls": [{"id": f"c{call_ids}", "type": "function",
                                         "function": {"name": "read_file", "arguments": {"path": "a.py"}}}]})
        start = rng.randrange(0, len(code) - 3000)
        messages.append({"role": "tool", "tool_call_id": f"c{call_ids}", "content": code[start:start + 3000]})
        messages.append({"role": "assistant", "content": rand_text(rng, 10, 40)})
        text = render(messages, TOOLS)
        if len(full(text)) >= target_tokens:
            return messages


@pytest.mark.parametrize("target", [40_000, 128_000])
def test_timing(target):
    rng = random.Random(target)
    messages = long_conversation(target, rng)
    pc = cache("on")
    text0 = render(messages[:-1], TOOLS)
    pc.encode(text0)                                   # the previous turn, cached
    text = render(messages, TOOLS)
    n = len(full(text))
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        ref = full(text)
    t_full = (time.perf_counter() - t0) / reps * 1e3
    t_inc = []
    for _ in range(reps):
        pc2 = cache("on")
        pc2.encode(text0)
        t1 = time.perf_counter()
        ids = pc2.encode(text)
        t_inc.append((time.perf_counter() - t1) * 1e3)
        assert ids == ref
    t_cold = []
    for _ in range(3):
        pc3 = cache("on")
        t1 = time.perf_counter()
        pc3.encode(text)
        t_cold.append((time.perf_counter() - t1) * 1e3)
    t_render = []
    for _ in range(3):
        t1 = time.perf_counter()
        render(messages, TOOLS)
        t_render.append((time.perf_counter() - t1) * 1e3)
    inc = sorted(t_inc)[len(t_inc) // 2]
    print(f"\n[tokcache] {n} tokens ({len(text)} chars): full encode {t_full:.1f} ms, incremental next turn "
          f"{inc:.2f} ms, cold (full + cache fill) {min(t_cold):.1f} ms, render {min(t_render):.2f} ms; "
          f"per turn before (2 full encodes) {2 * t_full:.1f} ms -> after {inc:.2f} ms")
    assert inc < t_full / 3
