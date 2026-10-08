"""patches/0620: tool-calling fixes for multi-step chains (GLM53_TF_TOOL_FIXES; ``glm5_next/cuda/toolfix.py``,
``cuda/server.py``'s ``typed_value`` / ``reply_hook``, 0610's ``grammar`` opt-in; docs/TOOL-CALLING.md).

Host only. A fake engine replays token chunks through the real ``App.run`` (as tests/test_openai_compat.py); the
checkpoint's chat template is used where $GLM53_TF_TOKENIZER_DIR holds it (tokenizer.json, tokenizer_config.json,
chat_template.jinja, config.json; those tests skip otherwise), and the 0610 grammar test needs xgrammar and torch.
Run against the patched tree: GLM53_TF_TOKENIZER_DIR=<dir> PYTHONPATH=<tree>/src pytest -q tests/test_tool_fixes.py
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

server = pytest.importorskip("tensorfold.cuda.server")
if not hasattr(server, "tool_fixes"):
    pytest.skip("patches/0620 not applied", allow_module_level=True)
toolfix = pytest.importorskip("tensorfold.families.glm5_next.cuda.toolfix")
app_mod = pytest.importorskip("tensorfold.families.glm5_next.cuda.app")

from tensorfold.cuda.server import App, parse_tool_calls, schema_types, tool_fixes, typed_value  # noqa: E402

TOKDIR = Path(os.environ.get("GLM53_TF_TOKENIZER_DIR", "/nonexistent"))
real_template = pytest.mark.skipif(
    not all((TOKDIR / f).exists() for f in ("tokenizer_config.json", "chat_template.jinja")),
    reason="GLM53_TF_TOKENIZER_DIR without the checkpoint's chat_template.jinja / tokenizer_config.json")

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "set_reminder", "parameters": {"type": "object", "properties": {
        "message": {"type": "string"}, "datetime": {"type": "string"}}, "required": ["message", "datetime"]}}},
]
EOS, OBS = 0, 1


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("GLM53_TF_TOOL_FIXES", "GLM53_TF_KEEP_REASONING_ENTRIES", "GLM53_TF_KEEP_REASONING_MB",
              "GLM53_TF_REASONING_FIELDS"):
        monkeypatch.delenv(k, raising=False)


def args(call):
    return json.loads(call["function"]["arguments"])


# -- the knob --------------------------------------------------------------------------------------------------------

def test_knob_parsing():
    assert tool_fixes("") == frozenset() and tool_fixes("0") == frozenset() and tool_fixes("off") == frozenset()
    assert tool_fixes("all") == frozenset(server.TOOL_FIXES_ALL)
    assert "grammar" not in tool_fixes("all")
    assert tool_fixes("all,grammar") == frozenset(server.TOOL_FIXES)
    assert tool_fixes(" History , ARGS ") == {"history", "args"}
    with pytest.raises(ValueError, match="unknown fix"):
        tool_fixes("history,bogus")


def test_off_changes_nothing():
    fixes = toolfix.ToolFixes(OBS, raw="")
    assert not fixes
    body = {"messages": [{"role": "user", "content": "x"},
                         {"role": "assistant", "content": None, "reasoning": "r",
                          "tool_calls": [{"id": "call_" + "a" * 24, "function": {"name": "f", "arguments": ""}}]}],
            "tools": TOOLS, "tool_choice": "none", "parallel_tool_calls": False}
    before = copy.deepcopy(body)
    assert fixes.prepare(body) is None and body == before
    calls = [{"id": "c1"}, {"id": "c2"}]
    assert fixes.reply(body, content="", calls=calls, reasoning="r", last=OBS, thinking=True, stopped=False) == \
        ("", calls, "r")


# -- history ---------------------------------------------------------------------------------------------------------

def test_history_normalises_messages():
    fixes = toolfix.ToolFixes(OBS, raw="history")
    body = {"messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "reasoning": "think",
         "tool_calls": [{"id": "a", "function": {"name": "get_weather", "arguments": ""}},
                        {"id": "b", "function": {"name": "get_weather", "arguments": None}},
                        {"id": "c", "function": {"name": "get_weather", "arguments": '{"city":"Rome"}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": None},
        {"role": "assistant", "content": "done", "reasoning_content": "kept", "reasoning": "other"}]}
    assert fixes.prepare(body) is None
    m = body["messages"]
    assert m[1]["content"] == "" and m[1]["reasoning_content"] == "think"
    assert [c["function"]["arguments"] for c in m[1]["tool_calls"]] == [{}, {}, '{"city":"Rome"}']
    assert m[2]["content"] == ""
    assert m[3]["reasoning_content"] == "kept"                  # a client's own reasoning_content wins
    again = copy.deepcopy(body)
    fixes.prepare(body)
    assert body == again                                        # idempotent (check, then run)


def _render(messages, thinking, tools=TOOLS):
    tpl = app_mod.ThinkingOffTemplate(server.ChatTemplate(TOKDIR))
    return tpl.render(copy.deepcopy(messages), tools=tools, enable_thinking=thinking, extra={})


CHAIN = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Weather in Paris? If rain, remind me at 8."},
    {"role": "assistant", "content": None, "reasoning": "Check the weather first.",
     "tool_calls": [{"id": "call_" + "1" * 24, "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]},
    {"role": "tool", "tool_call_id": "call_" + "1" * 24, "content": '{"condition":"rain"}'},
]


@real_template
@pytest.mark.parametrize("thinking", [False, True])
def test_history_renders_without_none(thinking):
    before = _render(CHAIN, thinking)
    assert "<think></think>None<tool_call>" in before           # the bug: content null rendered as "None"
    body = {"messages": copy.deepcopy(CHAIN), "tools": TOOLS}
    toolfix.ToolFixes(OBS, raw="history").prepare(body)
    after = _render(body["messages"], thinking)
    assert "None" not in after
    # the reasoning (sent back as ``reasoning``, the field this server answers with) is the step's think block
    assert "<|assistant|><think>Check the weather first.</think><tool_call>get_weather" in after
    assert after.endswith("<|assistant|><think>" + ("" if thinking else "</think>"))


@real_template
def test_empty_arguments_render_instead_of_a_400():
    msgs = copy.deepcopy(CHAIN)
    msgs[2]["tool_calls"][0]["function"]["arguments"] = ""
    with pytest.raises(Exception):
        _render(msgs, False)
    body = {"messages": msgs, "tools": TOOLS}
    toolfix.ToolFixes(OBS, raw="history").prepare(body)
    assert "<tool_call>get_weather</tool_call>" in _render(body["messages"], False)


TWO_TURNS = [
    {"role": "user", "content": "Pick a number."},
    {"role": "assistant", "content": "7", "reasoning_content": "Seven is a fine pick."},
    {"role": "user", "content": "Why?"},
]


@real_template
@pytest.mark.skipif("clear_thinking" not in app_mod.ThinkingOffTemplate.__init__.__code__.co_varnames,
                    reason="patches/0650 not applied")
def test_keep_thinking_across_user_turns():
    # patches/0650: the turn before the last user message keeps its reasoning only with GLM53_TF_CLEAR_THINKING=0
    def render(clear, extra):
        tpl = app_mod.ThinkingOffTemplate(server.ChatTemplate(TOKDIR), clear_thinking=clear)
        return tpl.render(copy.deepcopy(TWO_TURNS), tools=None, enable_thinking=True, extra=extra)

    kept = "<|assistant|><think>Seven is a fine pick.</think>7"
    plain = app_mod.ThinkingOffTemplate(server.ChatTemplate(TOKDIR)).render(
        copy.deepcopy(TWO_TURNS), tools=None, enable_thinking=True, extra={})
    assert kept not in plain                                    # the default: dropped, as before 0650
    assert render(True, {}) == plain
    assert kept in render(False, {})
    assert kept not in render(False, {"clear_thinking": True})  # the request's own key wins
    assert kept in render(True, {"clear_thinking": False})


# -- a fake engine through the real App.run ---------------------------------------------------------------------------

class Encoded:
    def __init__(self, ids):
        self.ids = ids


class Tok:
    def __init__(self):
        self.vocab = ["<eos>", "<|observation|>"]

    def id(self, piece):
        if piece not in self.vocab:
            self.vocab.append(piece)
        return self.vocab.index(piece)

    def encode(self, text, add_special_tokens=False):
        return Encoded([self.id(c) for c in text])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[i] for i in ids)


class Engine:
    eos = (EOS, OBS)

    def __init__(self, chunks):
        self.chunks = chunks
        self.prompts = []

    def generate(self, prompt, max_tokens, sampling, on_tokens):
        self.prompts.append(prompt)
        for chunk in self.chunks:
            if on_tokens(chunk):
                break
        return {}


class Template:
    def __init__(self):
        self.seen = []

    def render(self, messages, *, tools, enable_thinking, extra=None):
        self.seen.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        return "prompt"


def make_app(pieces, raw):
    """``pieces``: text pieces, one token each ("<eos>" / "<|observation|>" end the reply). The app is ``App`` with
    GlmApp's 0620 hooks (``_toolfix``, ``reply_hook``) bound to it."""

    tok = Tok()
    app = object.__new__(App)
    app.engine = Engine([[tok.id(p)] for p in pieces])
    app.served, app.tok, app.template = "glm", tok, Template()
    app.default_thinking, app.max_tokens, app.lock = False, 100, threading.Lock()
    app.sampling = {"temperature": 0.0, "top_k": 1, "top_p": 1.0}
    app.sampling_for = lambda body, prompt: None
    app.toolfix = toolfix.ToolFixes(OBS, raw=raw)
    app.reply_hook = lambda body, **kw: app_mod.GlmApp.reply_hook(app, body, **kw)
    return app


def run(app, body):
    problem = app_mod.GlmApp._toolfix(app, body)                  # as GlmApp.check / run
    assert problem is None
    deltas = []
    result = app.run(body, True, lambda d: deltas.append(d) or True)
    return result, deltas


CALL = ["<tool_call>", "get_weather", "<arg_key>", "city", "</arg_key>", "<arg_value>", "Paris", "</arg_value>",
        "</tool_call>"]
THINK = {"chat_template_kwargs": {"enable_thinking": True}}


def test_reasoning_memory_by_server_ids():
    app = make_app(["Check the weather.", "</think>"] + CALL + ["<|observation|>"], "reasoning")
    body = dict(THINK, messages=[{"role": "user", "content": "Weather in Paris?"}], tools=TOOLS)
    result, _ = run(app, body)
    assert result["finish"] == "tool_calls" and result["reasoning"] == "Check the weather."
    call = result["calls"][0]
    # the client sends the turn back without its reasoning (as most OpenAI clients do)
    body2 = dict(THINK, tools=TOOLS, messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": "rain"}])
    app.engine.chunks = [[app.tok.id("It rains.")], [EOS]]
    run(app, body2)
    assert app.template.seen[-1][0][1]["reasoning_content"] == "Check the weather."


def test_reasoning_memory_by_signature_when_ids_are_renumbered():
    """spark-bench renumbers the ids (``call_{turn}_{i}``) and re-serialises the arguments."""

    app = make_app(["Check the weather.", "</think>"] + CALL + ["<|observation|>"], "reasoning")
    user = {"role": "user", "content": "Weather in Paris?"}
    run(app, dict(THINK, messages=[user], tools=TOOLS))
    body2 = dict(THINK, tools=TOOLS, messages=[
        {"role": "system", "content": "agent"}, user,
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0_0", "type": "function", "function": {"name": "get_weather",
                                                                "arguments": json.dumps({"city": "Paris"})}}]},
        {"role": "tool", "tool_call_id": "call_0_0", "content": "rain"}])
    app.engine.chunks = [[app.tok.id("ok")], [EOS]]
    run(app, body2)
    assert app.template.seen[-1][0][2]["reasoning_content"] == "Check the weather."
    # other arguments, or another user message: not the same step
    for other in ({"city": "Rome"}, None):
        msgs = copy.deepcopy(body2["messages"])
        del msgs[2]["reasoning_content"]                    # put there by the request above (in place)
        if other is None:
            msgs[1]["content"] = "Weather in Paris, please?"
        else:
            msgs[2]["tool_calls"][0]["function"]["arguments"] = json.dumps(other)
        run(app, dict(THINK, tools=TOOLS, messages=msgs))
        assert "reasoning_content" not in app.template.seen[-1][0][2]


def test_reasoning_memory_keeps_a_clients_own_and_skips_thinking_off():
    app = make_app(["</think>"] + CALL + ["<|observation|>"], "reasoning")       # empty reasoning: nothing kept
    run(app, dict(THINK, messages=[{"role": "user", "content": "q"}], tools=TOOLS))
    assert len(app.toolfix.memory) == 0
    app = make_app(CALL + ["<|observation|>"], "reasoning")                       # thinking off
    run(app, {"messages": [{"role": "user", "content": "q"}], "tools": TOOLS})
    assert len(app.toolfix.memory) == 0
    fixes = toolfix.ToolFixes(OBS, raw="reasoning")
    fixes.memory.put(["call_" + "2" * 24], "remembered")
    body = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "reasoning_content":
            "mine", "tool_calls": [{"id": "call_" + "2" * 24, "function": {"name": "f", "arguments": "{}"}}]}]}
    fixes.prepare(body)
    assert body["messages"][1]["reasoning_content"] == "mine"


def test_reasoning_fix_alone_reads_the_reasoning_field():
    fixes = toolfix.ToolFixes(OBS, raw="reasoning")
    body = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "reasoning": "own",
            "tool_calls": [{"id": "call_0_0", "function": {"name": "f", "arguments": "{}"}}]}]}
    fixes.prepare(body)
    assert body["messages"][1]["reasoning_content"] == "own"


def test_reasoning_memory_bounds(monkeypatch):
    mem = toolfix.ReasoningMemory(entries=2, chars=10)
    mem.put(["a"], "1234")
    mem.put(["b"], "5678")
    mem.put(["c"], "90")                        # 3 entries > 2: "a" goes
    assert mem.get(["a"]) is None and mem.get(["b"]) == "5678" and mem.get(["c"]) == "90"
    mem.put(["d"], "123456")                    # 14 chars > 10: the oldest ("c" after b's use) goes first
    assert mem.get(["d"]) == "123456" and len(mem) <= 2 and mem.size <= 10
    mem.put(["b"], "new")                       # a key put again points at the newer reply
    assert mem.get(["b"]) == "new"
    assert toolfix.ReasoningMemory(entries=0).get(["x"]) is None
    monkeypatch.setenv("GLM53_TF_KEEP_REASONING_ENTRIES", "7")
    assert toolfix.ToolFixes(OBS, raw="reasoning").memory.entries == 7
    monkeypatch.setenv("GLM53_TF_KEEP_REASONING_MB", "-1")
    with pytest.raises(ValueError):
        toolfix.ToolFixes(OBS, raw="reasoning")


# -- tool_choice / parallel_tool_calls -------------------------------------------------------------------------------

def test_tool_choice_none_offers_no_tools_and_parses_no_call():
    app = make_app(CALL + ["<|observation|>"], "choice")
    result, _ = run(app, {"messages": [{"role": "user", "content": "q"}], "tools": TOOLS, "tool_choice": "none"})
    assert app.template.seen[-1][1] == []
    assert result["calls"] is None and result["finish"] == "stop"
    app = make_app(CALL + ["<|observation|>"], "")                               # off: as before
    result, _ = run(app, {"messages": [{"role": "user", "content": "q"}], "tools": TOOLS, "tool_choice": "none"})
    assert result["calls"] and app.template.seen[-1][1] == TOOLS


def test_named_tool_choice_offers_that_tool_only():
    fixes = toolfix.ToolFixes(OBS, raw="choice")
    body = {"tools": copy.deepcopy(TOOLS), "tool_choice": {"type": "function", "function": {"name": "set_reminder"}}}
    assert fixes.prepare(body) is None and [t["function"]["name"] for t in body["tools"]] == ["set_reminder"]
    assert "does not offer" in fixes.prepare({"tools": copy.deepcopy(TOOLS), "tool_choice": {
        "type": "function", "function": {"name": "nope"}}})
    assert "needs" in fixes.prepare({"tools": copy.deepcopy(TOOLS), "tool_choice": {"type": "function"}})
    body = {"tools": copy.deepcopy(TOOLS), "tool_choice": "required"}
    assert fixes.prepare(body) is None and body["tools"] == TOOLS               # 0610's grammar enforces it


def test_parallel_tool_calls_false_keeps_the_first():
    two = CALL + ["<tool_call>", "set_reminder", "<arg_key>", "message", "</arg_key>", "<arg_value>", "x",
                  "</arg_value>", "</tool_call>", "<|observation|>"]
    app = make_app(two, "choice")
    result, deltas = run(app, {"messages": [{"role": "user", "content": "q"}], "tools": TOOLS,
                               "parallel_tool_calls": False})
    assert [c["function"]["name"] for c in result["calls"]] == ["get_weather"]
    result, _ = run(app, {"messages": [{"role": "user", "content": "q"}], "tools": TOOLS})
    assert len(result["calls"]) == 2


# -- a call left inside the think block ---------------------------------------------------------------------------------

def test_calls_inside_an_unclosed_think_block():
    pieces = ["I will check.\n"] + CALL + ["<|observation|>"]
    app = make_app(pieces, "")
    result, _ = run(app, dict(THINK, messages=[{"role": "user", "content": "q"}], tools=TOOLS))
    assert result["calls"] is None and result["content"] == ""                 # before: an empty answer
    app = make_app(pieces, "thinkcalls")
    result, _ = run(app, dict(THINK, messages=[{"role": "user", "content": "q"}], tools=TOOLS))
    assert result["finish"] == "tool_calls" and args(result["calls"][0]) == {"city": "Paris"}
    assert result["reasoning"] == "I will check."


def test_thinkcalls_only_when_the_model_waits_for_a_result():
    # ends on <eos> (<|user|> / <|endoftext|>), not <|observation|>: left as it was
    app = make_app(["I will check.\n"] + CALL + ["<eos>"], "thinkcalls")
    result, _ = run(app, dict(THINK, messages=[{"role": "user", "content": "q"}], tools=TOOLS))
    assert result["calls"] is None
    # a call written as an example in the middle of the thinking is not a trailing call
    app = make_app(["e.g. "] + CALL + [" then more thought"] + ["<|observation|>"], "thinkcalls")
    result, _ = run(app, dict(THINK, messages=[{"role": "user", "content": "q"}], tools=TOOLS))
    assert result["calls"] is None
    # an answer with text keeps its own (absent) calls
    app = make_app(["hm", "</think>", "Sure."] + ["<|observation|>"], "thinkcalls")
    result, _ = run(app, dict(THINK, messages=[{"role": "user", "content": "q"}], tools=TOOLS))
    assert result["calls"] is None and result["content"] == "Sure."


def test_trailing_calls():
    c = "<tool_call>a</tool_call>"
    assert toolfix.trailing_calls("think " + c + "\n" + c + " \n") == ("think", c + "\n" + c)
    assert toolfix.trailing_calls("x " + c + "</think>") == ("x", c)
    assert toolfix.trailing_calls(c + " tail") == (c + " tail", "")
    assert toolfix.trailing_calls("none") == ("none", "")


# -- arguments by schema ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("prop,raw,want", [
    ({"type": "string"}, "42", "42"),
    ({"type": "string"}, "null", "null"),
    ({"type": ["string", "null"]}, "null", None),
    ({"type": ["string", "null"]}, "42", "42"),
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "007", "007"),
    ({"type": "integer"}, "5", 5),
    ({"type": "integer"}, " 5\n", 5),
    ({"type": "integer"}, '"5"', 5),
    ({"type": "integer"}, "5.0", 5),
    ({"type": "integer"}, "5.5", "5.5"),
    ({"type": "number"}, "5.5", 5.5),
    ({"type": "number"}, "5", 5),
    ({"type": "boolean"}, "true", True),
    ({"type": "boolean"}, "True", True),
    ({"type": "boolean"}, '"false"', False),
    ({"type": ["integer", "null"]}, "None", None),
    ({"type": "array"}, '["a", "b"', ["a", "b"]),
    ({"type": "object"}, '{"a": {"b": 1}', {"a": {"b": 1}}),
    ({"type": "array"}, "not json", "not json"),
    ({"enum": ["low", "high"]}, "high", "high"),
    ({"enum": [1, 2, 3]}, "2", 2),
    ({}, "3", 3),                                   # untyped: as before (JSON, else text)
    ({}, "hello", "hello"),
    ({"$ref": "#/defs/x"}, '{"a":1}', {"a": 1}),
    (None, "[1]", [1]),
])
def test_typed_value(prop, raw, want):
    got = typed_value(raw, prop)
    assert got == want and type(got) is type(want)


def test_schema_types():
    assert schema_types({"type": "string"}) == {"string"}
    assert schema_types({"oneOf": [{"type": "integer"}, {"type": "string"}]}) == {"integer", "string"}
    assert schema_types({"anyOf": [{"type": "integer"}, {"$ref": "#/x"}]}) is None
    assert schema_types({"const": True}) == {"boolean"}
    assert schema_types({"description": "x"}) is None


def test_args_knob_in_the_parser(monkeypatch):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "n": {"type": ["integer", "null"]}, "s": {"type": ["string", "null"]}, "flag": {"type": "boolean"}}}}}]
    text = ("<tool_call>f<arg_key>n</arg_key><arg_value>\"7\"</arg_value><arg_key>s</arg_key><arg_value>123"
            "</arg_value><arg_key>flag</arg_key><arg_value>True</arg_value></tool_call>")
    _, before = parse_tool_calls(text, tools)
    assert args(before[0]) == {"n": "7", "s": 123, "flag": "True"}           # 0002's reading
    monkeypatch.setenv("GLM53_TF_TOOL_FIXES", "args")
    _, after = parse_tool_calls(text, tools)
    assert args(after[0]) == {"n": 7, "s": "123", "flag": True}


# -- 0610's grammar for auto tool requests ----------------------------------------------------------------------------

def test_grammar_opt_in_for_auto_tools(monkeypatch):
    grammar = pytest.importorskip("tensorfold.families.glm5_next.cuda.grammar")
    body = {"tools": TOOLS}
    assert grammar.request_spec(body) is None
    monkeypatch.setenv("GLM53_TF_TOOL_FIXES", "grammar")
    spec = grammar.request_spec(body)
    assert spec.kind == "tools" and json.loads(spec.text)["tool_choice"] == "auto"
    assert grammar.request_spec(dict(body, tool_choice="none")) is None
    assert grammar.request_spec(dict(body, response_format={"type": "json_object"})).kind == "json"


@pytest.mark.skipif(not (TOKDIR / "tokenizer.json").exists() or not (TOKDIR / "config.json").exists(),
                    reason="GLM53_TF_TOKENIZER_DIR without tokenizer.json / config.json")
def test_grammar_auto_tools_real_tokenizer(monkeypatch):
    pytest.importorskip("xgrammar")
    pytest.importorskip("torch")
    grammar = pytest.importorskip("tensorfold.families.glm5_next.cuda.grammar")
    from tokenizers import Tokenizer

    monkeypatch.setenv("GLM53_TF_TOOL_FIXES", "grammar")
    g = grammar.Grammars.from_model(TOKDIR, grammar.vocab_size(TOKDIR), (154820, 154827, 154829))
    tok = Tokenizer.from_file(str(TOKDIR / "tokenizer.json"))
    spec = grammar.request_spec({"tools": TOOLS})
    compiled = g.compile(spec)
    marks = {s: tok.token_to_id(s) for s in grammar.TOOL_TOKENS}

    def ids(parts):
        out = []
        for p in parts:
            out += [marks[p]] if p in marks else tok.encode(p, add_special_tokens=False).ids
        return out

    def con():
        return g.constraint(g.bind(spec, compiled, [1, 154842]))           # thinking off: grammar from row 0

    c = con()
    c.advance(ids(["It is sunny in Paris."]))                              # free text stays free
    c = con()
    c.advance(ids(["Let me check. "] + CALL[:-1] + ["<arg_key>", "days", "</arg_key>", "<arg_value>", "3",
                                                     "</arg_value>", "</tool_call>"]))
    bad = CALL[:-1] + ["<arg_key>", "days", "</arg_key>", "<arg_value>", "three", "</arg_value>", "</tool_call>"]
    with pytest.raises(grammar.GrammarError):
        con().advance(ids(bad))
    with pytest.raises(grammar.GrammarError):                              # an unknown tool name
        con().advance(ids(["<tool_call>", "nope", "<arg_key>"]))
