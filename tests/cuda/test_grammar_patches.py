"""patches/0610 (GLM53_TF_GRAMMAR=1, ``glm5_next/cuda/grammar.py``): structured output (OpenAI ``response_format``
json_object / json_schema, vLLM ``guided_*`` / ``structured_outputs``, strict / required tool calls) enforced with
xgrammar masks, exact under drafting and batching.

Checked (CPU only; xgrammar 0.2.8 and torch on the CPU; the real GLM tokenizer where $GLM53_TF_TOKENIZER_DIR holds
``tokenizer.json`` + ``config.json``):

- request fields: every accepted shape and every 400 (malformed, beside a required call), tools opt-in only (strict
  or required / named), the knob's parsing, ``pack`` / ``split`` round trips (text of any length and alphabet);
- masks against the grammar library: for random committed paths and random draft chains (grammar-valid, invalid,
  stop tokens mid-chain), ``Constraint.cut`` keeps exactly the drafts a fresh matcher accepts (up to the first
  rejection or the stop token) and ``fill``'s bits equal a fresh matcher's bitmask at every kept row's path; the
  matcher ends where it started; ``advance`` == accepting; the think gate (rows before ``</think>`` unconstrained,
  the grammar from the row after it, drafts included); worker-thread fills == inline fills;
- ``apply`` on this rank's vocabulary half == the whole row's mask sliced (offsets on and off the 32-token words,
  padded columns never allowed); the keyed draw of a masked row == the same rule over the allowed tokens only
  (greedy and sampled, top-k / top-p, rows with fewer allowed tokens than top_k); two ranks (threads exchanging
  their candidates as the all-gather does) == one rank with the whole vocabulary, for ``batch.sample_multi`` and
  ``decode.sample_rows``;
- the real ``Batcher`` (``_plan`` / ``_execute`` / ``_admit`` / ``_piece`` / ``_verify`` / ``_finish`` / ``follow``,
  the real ``Stepper``, ``sample_multi`` and ``decode.prefill``) on a hostile fake model whose float logits over a
  160-token JSON vocabulary read a hash of every token before them, with fake MTP / DFlash2 drafters that draft the
  constrained continuation with corruptions (grammar-invalid tokens, wrong valid tokens, stop tokens, ``</think>``)
  and lookup drafts: constrained drafted replies == the same request with ``"draft": false`` == an independent
  serial reference, greedy and sampled, every policy; 4 slots mixing constrained and plain requests == each alone;
  plain requests with the knob on == knob off (replies, keeps, drafters); random JSON schemas: every reply that ends
  parses and validates (jsonschema); thinking on (grammar after ``</think>``, the reasoning == the unconstrained
  reply's) and off; stop / EOS interplay (no EOS inside the JSON, the stop token right where the grammar completes,
  ``ignore_eos`` decodes on unconstrained after it, ``max_tokens`` cuts); drafts cut before the forward; a grammar
  failure ends only its own slot; a follower batcher replaying rank 0's plans compiles the same grammars and decides
  the same windows; worker-thread fills == inline fills; the lone engine's loops (serial / MTP / DFlash2 / auto)
  likewise;
- the HTTP app: knob off ignores every field (as before), on it answers 400s before any header and hands the
  compiled grammar to the engine;
- the real GLM tokenizer (optional): the 154,880-column masks never allow an added token inside JSON (``</think>``,
  ``<|user|>``, image markers) and allow the stop tokens only where the JSON may end; the tool grammar takes GLM's
  markup tokens and its calls parse (``server.parse_tool_calls``) into arguments that validate against the schema.

Run: PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda pytest -q tests/cuda/test_grammar_patches.py
(the microbenchmark: bench/grammar_bench.py).
"""

from __future__ import annotations

import collections
import copy
import json
import os
import queue
import random
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
xgr = pytest.importorskip("xgrammar")

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows  # noqa: E402
from tensorfold.families.glm5_next.cuda import batchplan, grammar  # noqa: E402

TOKDIR = Path(os.environ.get("GLM53_TF_TOKENIZER_DIR", "/nonexistent"))
real_tok = pytest.mark.skipif(not (TOKDIR / "tokenizer.json").is_file() or not (TOKDIR / "config.json").is_file(),
                              reason="GLM53_TF_TOKENIZER_DIR without the checkpoint's tokenizer.json / config.json")

# -- a JSON-friendly vocabulary (RAW pieces) ------------------------------------------------------------------------
EOS, THINK, THINK_END = 0, 1, 2
PIECES = (["{", "}", "[", "]", ":", ",", '"', " ", "\n", "-", ".", "e", "E", "+", "\\", '\\"', "true", "false", "null",
           '{"', '":', '": ', '", "', '"}', ', "', '"]', "[]", "{}", '":"', "0.", "10", "42", "-1", "1e3"]
          + [str(d) for d in range(10)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
          + ["A", "B", "Z", "name", "id", "ok", "tag", "items", "value", "city", "zip", "score", "é", "ß", "<", ">",
             "ab", "cd", "the", " the", "on", "in", "x_", "_", "/", "#", "?", "!", "@", "%", "*", "(", ")", "'", "=",
             "tr", "ue", "fa", "lse", "nu", "ll", "12", "345", "6.5", "-0", "00", "a b", " z", "  ", "\t", "\r\n"])
V = 160
VOCAB = ["", "<think>", "</think>"] + PIECES
assert len(VOCAB) <= V - 8
VOCAB += [""] * (V - len(VOCAB))                     # padded rows: never allowed inside a grammar
SPECIALS = {EOS, THINK, THINK_END} | {i for i, p in enumerate(VOCAB) if i > 2 and p == ""}
QUOTE = VOCAB.index('"')
CLOSE = VOCAB.index("}")


def text(tokens) -> str:
    return "".join(VOCAB[t] for t in tokens if t not in SPECIALS)


_GRAMMARS = None


def grammars() -> grammar.Grammars:
    """A ``Grammars`` over the fake vocabulary (stop token 0, ``<think>`` 1, ``</think>`` 2)."""

    global _GRAMMARS
    if _GRAMMARS is None:
        enc = ["" if i in SPECIALS else p for i, p in enumerate(VOCAB)]
        info = xgr.TokenizerInfo(enc, xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[EOS], add_prefix_space=False)
        _GRAMMARS = grammar.Grammars(xgr, info, info, V, [EOS], THINK, THINK_END)
    return _GRAMMARS


def other_grammars() -> grammar.Grammars:
    """Rank 1's own compiler (a second instance built the same way)."""

    enc = ["" if i in SPECIALS else p for i, p in enumerate(VOCAB)]
    info = xgr.TokenizerInfo(enc, xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[EOS], add_prefix_space=False)
    return grammar.Grammars(xgr, info, info, V, [EOS], THINK, THINK_END)


SCHEMA = {"type": "object", "properties": {"name": {"type": "string", "maxLength": 12}, "id": {"type": "integer"},
                                            "ok": {"type": "boolean"}},
          "required": ["name", "id", "ok"], "additionalProperties": False}


def bound(spec: grammar.Spec, prompt=(5, 6), g=None) -> grammar.Bound:
    g = g or grammars()
    return g.bind(spec, g.compile(spec), list(prompt))


def spec_schema(schema=SCHEMA) -> grammar.Spec:
    return grammar.request_spec({"response_format": {"type": "json_schema", "json_schema": {"schema": schema}}})


def spec_object() -> grammar.Spec:
    return grammar.request_spec({"response_format": {"type": "json_object"}})


def _bits_row(m) -> np.ndarray:
    bits = np.zeros((1, (V + 31) // 32), dtype=np.int32)
    m.fill_next_token_bitmask(bits, 0)
    return bits[0]


def _allowed_ids(bits_row) -> list[int]:
    b = bits_row.view(np.uint32)
    return [t for t in range(V) if (int(b[t >> 5]) >> (t & 31)) & 1]


def _random_path(compiled, rng, n: int) -> list[int]:
    """n tokens a fresh matcher accepts (random allowed tokens, never the stop token)."""

    m = xgr.GrammarMatcher(compiled)
    out = []
    for _ in range(n):
        ids = [t for t in _allowed_ids(_bits_row(m)) if t != EOS]
        if not ids:
            break
        t = int(rng.choice(ids))
        assert m.accept_token(t)
        out.append(t)
    return out


# -- request fields ---------------------------------------------------------------------------------------------------
def test_knob(monkeypatch):
    monkeypatch.delenv("GLM53_TF_GRAMMAR", raising=False)
    assert not grammar.enabled()
    monkeypatch.setenv("GLM53_TF_GRAMMAR", "1")
    assert grammar.enabled()
    monkeypatch.setenv("GLM53_TF_GRAMMAR", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        grammar.enabled()
    monkeypatch.setenv("GLM53_TF_GRAMMAR_THREADS", "0")
    assert grammar.threads() == 0
    monkeypatch.setenv("GLM53_TF_GRAMMAR_THREADS", "99")
    with pytest.raises(ValueError):
        grammar.threads()


TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city", "days"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def test_request_spec_shapes():
    rs = grammar.request_spec
    assert rs({}) is None and rs({"response_format": {"type": "text"}}) is None
    assert rs({"response_format": {"type": "json_object"}}) == grammar.Spec("json")
    s = rs({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}}})
    assert s.kind == "json_schema" and json.loads(s.text) == SCHEMA
    assert rs({"guided_json": json.dumps(SCHEMA)}).kind == "json_schema"
    assert rs({"guided_regex": "[a-z]+"}) == grammar.Spec("regex", "[a-z]+", "guided_regex")
    assert rs({"guided_choice": ["yes", "no"]}).kind == "choice"
    assert rs({"structured_outputs": {"json": SCHEMA}}).kind == "json_schema"
    assert rs({"structured_outputs": {"json_object": True}}).kind == "json"
    # tools: opt-in only
    assert rs({"tools": TOOLS}) is None
    assert rs({"tools": TOOLS, "tool_choice": "auto"}) is None
    assert rs({"tools": TOOLS, "tool_choice": "none"}) is None
    strict = [dict(TOOLS[0], function=dict(TOOLS[0]["function"], strict=True)), TOOLS[1]]
    assert rs({"tools": strict}).kind == "tools"
    t = rs({"tools": TOOLS, "tool_choice": "required", "parallel_tool_calls": False})
    assert t.kind == "tools" and json.loads(t.text)["tool_choice"] == "required"
    assert json.loads(t.text)["parallel_tool_calls"] is False
    t = rs({"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "bash"}}})
    assert json.loads(t.text)["tool_choice"]["function"]["name"] == "bash"
    # an output format wins over strict auto tools; beside a required call it is refused
    assert rs({"tools": strict, "response_format": {"type": "json_object"}}).kind == "json"
    with pytest.raises(ValueError, match="cannot be combined"):
        rs({"tools": TOOLS, "tool_choice": "required", "response_format": {"type": "json_object"}})


@pytest.mark.parametrize("body, match", [
    ({"response_format": "json"}, "must be an object"),
    ({"response_format": {"type": "xml"}}, "type must be"),
    ({"response_format": {"type": "json_schema"}}, "json_schema.schema"),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": "{not json"}}}, "not valid JSON"),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": [1]}}}, "JSON schema object"),
    ({"guided_regex": ""}, "non-empty string"),
    ({"guided_choice": []}, "non-empty list"),
    ({"structured_outputs": {"yaml": "x"}}, "not supported"),
    ({"tools": "bash", "tool_choice": "required"}, "must be a list"),
    ({"tools": TOOLS, "tool_choice": "sometimes"}, "tool_choice must be"),
    ({"tools": [{"type": "function", "function": {"parameters": {}}}], "tool_choice": "required"}, "needs a name"),
])
def test_request_spec_refusals(body, match):
    with pytest.raises(ValueError, match=match):
        grammar.request_spec(body)


def test_compile_refusals():
    g = grammars()
    with pytest.raises(ValueError, match="cannot be enforced"):
        g.compile(grammar.Spec("grammar", "root ::= undefined_rule", "guided_grammar"))
    with pytest.raises(ValueError, match="cannot be enforced"):
        g.compile(grammar.Spec("regex", "(unclosed", "guided_regex"))
    with pytest.raises(ValueError, match="cannot be enforced"):
        g.compile(grammar.Spec("tools", json.dumps({"tools": TOOLS, "tool_choice": {"type": "function", "function": {
            "name": "nope"}}, "parallel_tool_calls": True}), "tools"))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 100, 1001])
def test_pack_split_round_trip(n):
    rng = random.Random(n)
    text_ = "".join(rng.choice("{}\"aé中\U0001f600\n\\ ") for _ in range(n))
    b = grammar.Bound(grammar.Spec("json_schema", text_), None, 154842, bool(n % 2))
    header = [7, 1, 1, 5, 6]
    packed = grammar.pack(b) + header
    assert all(-1 <= v < 1 << 24 for v in packed)
    got, rest = grammar.split(packed)
    assert rest == header
    g = SimpleNamespace(compile=lambda spec: ("compiled", spec))
    f = grammar.Grammars.follow(g, got)
    assert f.spec.kind == "json_schema" and f.spec.text == text_ and f.think_end == 154842 and f.active == bool(n % 2)
    assert grammar.split(header) == (None, header)


def test_think_state():
    assert grammar.think_active([5, 6], THINK, THINK_END)
    assert not grammar.think_active([5, THINK], THINK, THINK_END)
    assert grammar.think_active([5, THINK, THINK_END], THINK, THINK_END)
    assert not grammar.think_active([THINK, THINK_END, 9, THINK, 4], THINK, THINK_END)


# -- masks against the grammar library ----------------------------------------------------------------------------------
def _reference_window(compiled, committed, tokens, think_end, active):
    """A fresh matcher per row: (kept count, [(row, bits)])."""

    def matcher_at(path):
        m = xgr.GrammarMatcher(compiled)
        for t in path:
            assert m.accept_token(t)
        return m

    grammar_path = list(committed)
    keep, rows = 1, []
    on = active
    if on:
        rows.append((0, _bits_row(matcher_at(grammar_path))))
    for r in range(1, len(tokens)):
        t = tokens[r]
        if on:
            m = matcher_at(grammar_path)
            if not m.accept_token(t) or m.is_terminated():
                break
            grammar_path.append(t)
        else:
            on = t == think_end
        keep += 1
        if on:
            rows.append((r, _bits_row(matcher_at(grammar_path))))
    return keep, rows


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("which", ["schema", "object", "choice"])
def test_cut_and_fill_equal_a_fresh_matcher(seed, which):
    g = grammars()
    rng = np.random.default_rng(seed)
    spec = {"schema": spec_schema(), "object": spec_object(),
            "choice": grammar.Spec("choice", json.dumps(["alpha", "beta", "gamma delta"]))}[which]
    compiled = g.compile(spec)
    for trial in range(6):
        committed = _random_path(compiled, rng, int(rng.integers(0, 25)))
        con = g.constraint(grammar.Bound(spec, compiled, THINK_END, True))
        con.advance(committed[:-1] if committed else [])
        pending = committed[-1] if committed else 3
        if committed:
            con.advance([pending])
        # drafts: a valid continuation with a corruption somewhere (invalid token, stop token, random)
        cont = _random_path_from(compiled, committed, rng, int(rng.integers(0, 12)))
        drafts = list(cont)
        if drafts and rng.random() < 0.7:
            j = int(rng.integers(0, len(drafts)))
            drafts[j] = int(rng.choice([EOS, int(rng.integers(0, V)), THINK_END]))
        tokens = [pending] + drafts
        before = _bits_row(con.m)
        win = con.fill(con.cut(tokens))
        keep, rows = _reference_window(compiled, committed, tokens, THINK_END, True)
        assert len(win.tokens) == keep and win.tokens == tokens[:keep] and win.cut == len(tokens) - keep
        assert win.rows == [r for r, _ in rows]
        for j, (_, bits) in enumerate(rows):
            assert np.array_equal(win.bits[j], bits), (seed, trial, j)
        assert np.array_equal(_bits_row(con.m), before)            # the matcher ends where it started
        # advance == accepting the kept tokens (the next window starts there)
        kept = win.tokens[1:]
        con.advance(kept)
        if not con.finished:
            ref = xgr.GrammarMatcher(compiled)
            for t in committed + kept:
                ref.accept_token(t)
            assert np.array_equal(_bits_row(con.m), _bits_row(ref))


def _random_path_from(compiled, committed, rng, n):
    m = xgr.GrammarMatcher(compiled)
    for t in committed:
        assert m.accept_token(t)
    out = []
    for _ in range(n):
        ids = _allowed_ids(_bits_row(m))
        if not ids:
            break
        t = int(rng.choice(ids))
        out.append(t)
        if not m.accept_token(t) or m.is_terminated():
            break
    return out


def test_stop_token_draft_ends_the_window():
    g = grammars()
    spec = grammar.Spec("choice", json.dumps(["ab"]))
    con = g.constraint(bound(spec))
    a, b = VOCAB.index("a"), VOCAB.index("b")
    con.advance([a])                                # the pending token (row 0) is followed already
    win = con.fill(con.cut([a, b, EOS, 5, 6]))     # after "ab" the grammar ends: its stop token is not verified
    assert win.tokens == [a, b] and win.cut == 3 and win.rows == [0, 1]
    assert _allowed_ids(win.bits[1]) == [EOS]


@pytest.mark.parametrize("threads", [0, 3])
def test_think_gate(threads):
    """Rows before </think> are unconstrained (no bits), the grammar starts at the row after it (drafts too); the
    worker-thread fill equals the inline one."""

    g = grammars()
    spec = spec_schema()
    b = bound(spec, prompt=[5, THINK])
    assert not b.active
    con = g.constraint(b)
    filler = grammar.Filler(threads)
    reasoning = [VOCAB.index("a"), VOCAB.index("ab"), 3, THINK_END]
    tokens = [7] + reasoning + [VOCAB.index('{"'), VOCAB.index("name")]
    win = con.cut(tokens)
    assert win.cut == 0 and win.rows == [4, 5, 6]
    assert filler.start([(con, win)]).wait() == [None]
    fresh = xgr.GrammarMatcher(g.compile(spec))
    assert np.array_equal(win.bits[0], _bits_row(fresh))           # the row after </think>: the grammar's start
    # an invalid token after </think> is cut; before it nothing is
    win2 = con.cut([7, VOCAB.index("zip"), THINK_END, VOCAB.index("zip")])
    assert win2.tokens == [7, VOCAB.index("zip"), THINK_END] and win2.rows == [2]
    con.advance(reasoning)
    assert con.active
    win3 = con.fill(con.cut([THINK_END]))
    assert win3.rows == [0] and np.array_equal(win3.bits[0], _bits_row(fresh))


def test_filler_threads_equal_inline():
    g = grammars()
    spec = spec_schema()
    rng = np.random.default_rng(3)
    items_a, items_b = [], []
    for k in range(6):
        compiled = g.compile(spec)
        path = _random_path(compiled, rng, int(rng.integers(0, 20)))
        cont = _random_path_from(compiled, path, rng, 10)
        for items in (items_a, items_b):
            con = g.constraint(grammar.Bound(spec, compiled, THINK_END, True))
            con.advance(path)
            items.append((con, con.cut([path[-1] if path else 3] + cont)))
    grammar.Filler(0).start(items_a).wait()
    four = grammar.Filler(4)
    assert len(four.pool._threads) == 4                 # made at load by the loading thread (0370's cpus), not later
    assert four.start(items_b).wait() == [None] * 6
    for (_, wa), (_, wb) in zip(items_a, items_b):
        assert wa.rows == wb.rows and (wa.bits is None and wb.bits is None or np.array_equal(wa.bits, wb.bits))


def test_fill_errors_are_grammar_errors():
    g = grammars()
    con = g.constraint(bound(spec_schema()))
    win = con.cut([3])
    win.rows = [0, 1]                              # tampered: the rows changed between cut and fill
    assert isinstance(grammar.Filler(2).start([(con, win)]).wait()[0], grammar.GrammarError)
    with pytest.raises(grammar.GrammarError, match="rejected chosen token"):
        con.advance([VOCAB.index("zip")])


# -- masks on the logits, the keyed draw, two ranks ---------------------------------------------------------------------
def _window_with_bits(rng, rows_total: int, which_rows: list[int], density: float = 0.3) -> grammar.Window:
    bits = np.zeros((len(which_rows), (V + 31) // 32), dtype=np.uint32)
    for j in range(len(which_rows)):
        allowed = rng.random(V) < density
        allowed[rng.integers(0, V)] = True
        for t in np.nonzero(allowed)[0]:
            bits[j, t >> 5] |= np.uint32(1 << (int(t) & 31))
    return grammar.Window(list(range(rows_total)), which_rows, bits.view(np.int32))


def _allowed_mask(win, j) -> np.ndarray:
    b = win.bits[j].view(np.uint32)
    return np.array([(int(b[t >> 5]) >> (t & 31)) & 1 for t in range(V)], dtype=bool)


@pytest.mark.parametrize("offset", [0, 32, 80, 77, 150])
def test_apply_on_a_vocabulary_half(offset):
    rng = np.random.default_rng(offset)
    R = 5
    win = _window_with_bits(rng, R, [0, 2, 3])
    full = torch.tensor(rng.standard_normal((R, V)), dtype=torch.float32)
    whole = grammar.apply(full, win, 0)
    for j, r in enumerate(win.rows):
        ok = _allowed_mask(win, j)
        assert torch.equal(whole[r][torch.tensor(ok)], full[r][torch.tensor(ok)])
        assert bool(torch.isinf(whole[r][torch.tensor(~ok)]).all())
    for r in (1, 4):
        assert torch.equal(whole[r], full[r])
    width = V - offset + 16                          # past the grammar's vocabulary: never allowed
    part = torch.cat([full[:, offset:], torch.zeros((R, 16))], dim=1)
    got = grammar.apply(part, win, offset)
    assert torch.equal(got[:, :V - offset], whole[:, offset:])
    assert bool(torch.isinf(got[win.rows, V - offset:]).all()) and got.shape[1] == width
    assert torch.equal(got[[1, 4], V - offset:], torch.zeros((2, 16)))
    grammar.stage(win, width, part.device, offset)          # staged right after the forward: the same mask
    assert win.ok is not None and torch.equal(grammar.apply(part, win, offset), got)
    assert torch.equal(grammar.apply(full, win, 0), whole)   # another width / offset: not the staged one
    assert grammar.apply(full, grammar.Window([0], []), 0) is full


def _fake_w(offset: int = 0, comm=None, world: int = 1):
    return SimpleNamespace(vocab_offset=offset, comm=comm, world=world, meta={}, mtp=object())


def _reference_draw(row: np.ndarray, allowed: np.ndarray, position: int, sampling) -> int:
    """The keyed rule over the allowed tokens only (no top-k gather at all)."""

    ids = np.nonzero(allowed)[0].astype(np.int64)
    vals = row[ids].astype(np.float32)
    if sampling is None:
        order = np.lexsort((ids, -vals))
        return int(ids[order[0]])
    return choose_rows(vals[None, :], ids[None, :], [position], sampling)[0]


SAMPLINGS = [None, Sampling(11, 1.0, 20, 0.95), Sampling(12, 0.7, 5, 1.0), Sampling(13, 1.3, 50, 0.5)]


@pytest.mark.parametrize("si", range(len(SAMPLINGS)))
@pytest.mark.parametrize("density", [0.02, 0.3, 1.0])
def test_masked_draw_is_the_rule_over_allowed_tokens(si, density):
    from tensorfold.families.glm5_next.cuda import batch

    sampling = SAMPLINGS[si]
    rng = np.random.default_rng(int(density * 100) + si)
    for trial in range(8):
        R = int(rng.integers(1, 9))
        rows = sorted(rng.choice(R, size=int(rng.integers(1, R + 1)), replace=False).tolist())
        win = _window_with_bits(rng, R, rows, density)
        logits = torch.tensor(rng.standard_normal((R, V)) * 3, dtype=torch.float32)
        pos = [int(rng.integers(0, 10000)) + r for r in range(R)]
        got = batch.sample_multi(_fake_w(), logits, [(0, R, pos, sampling)], masks=[win])[0]
        plain = batch.sample_multi(_fake_w(), logits, [(0, R, pos, sampling)])[0]
        for r in range(R):
            if r in rows:
                ok = _allowed_mask(win, rows.index(r))
                assert got[r] == _reference_draw(logits[r].numpy(), ok, pos[r], sampling), (trial, r)
                assert ok[got[r]]
            else:
                assert got[r] == plain[r]


class _Wire:
    """Two ranks in threads: ``fast_gather`` concatenates both ranks' packed candidates, as the all-gather does."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2)
        self.parts: dict[int, torch.Tensor] = {}
        self.local = threading.local()

    def fast_gather(self, comm, flat, buf) -> None:
        self.parts[self.local.rank] = flat.clone()
        self.barrier.wait()
        buf.copy_(torch.cat([self.parts[0], self.parts[1]]))
        self.barrier.wait()


def _two_ranks(monkeypatch, fn):
    from tensorfold.families.glm5_next.cuda import comm as comm_mod

    wire = _Wire()
    monkeypatch.setattr(comm_mod, "fast_gather", wire.fast_gather)
    monkeypatch.setattr(comm_mod, "check", lambda comm: None)
    out, errs = [None, None], []

    def run(rank):
        wire.local.rank = rank
        try:
            out[rank] = fn(rank)
        except BaseException as exc:  # noqa: BLE001
            errs.append(exc)
            wire.barrier.abort()

    ts = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0]
    return out


@pytest.mark.parametrize("si", range(len(SAMPLINGS)))
def test_two_ranks_equal_one(monkeypatch, si):
    """Each rank masks its vocabulary half and exchanges its candidates (unchanged exchange): both ranks draw the
    token one rank draws from the whole row, for ``batch.sample_multi`` (several sequences, some unconstrained) and
    ``decode.sample_rows`` (host draw)."""

    from tensorfold.families.glm5_next.cuda import batch, decode

    sampling = SAMPLINGS[si]
    rng = np.random.default_rng(40 + si)
    half = V // 2
    for trial in range(5):
        Rs = [int(rng.integers(1, 7)) for _ in range(3)]
        offs = [sum(Rs[:i]) for i in range(3)]
        T = sum(Rs)
        logits = torch.tensor(rng.standard_normal((T, V)) * 3, dtype=torch.float32)
        wins = [_window_with_bits(rng, Rs[0], list(range(Rs[0])), 0.1), None,
                _window_with_bits(rng, Rs[2], [0, Rs[2] - 1] if Rs[2] > 1 else [0], 0.05)]
        specs = [(o, R, [100 * i + r for r in range(R)], sampling) for i, (o, R) in enumerate(zip(offs, Rs))]
        want = batch.sample_multi(_fake_w(), logits, specs, masks=wins)
        got = _two_ranks(monkeypatch, lambda rank: batch.sample_multi(
            _fake_w(rank * half, comm=object(), world=2), logits[:, rank * half:(rank + 1) * half].contiguous(),
            specs, masks=wins))
        assert got[0] == got[1] == want, trial
        # the lone engine's sampler, host draw
        win = wins[0]
        lone_want = decode.sample_rows(_fake_w(), grammar.apply(logits[:Rs[0]], win, 0), specs[0][2], sampling,
                                       host=True)
        lone = _two_ranks(monkeypatch, lambda rank: decode.sample_rows(
            _fake_w(rank * half, comm=object(), world=2),
            grammar.apply(logits[:Rs[0], rank * half:(rank + 1) * half].contiguous(), win, rank * half),
            specs[0][2], sampling, host=True))
        assert lone[0] == lone[1] == lone_want == want[0]


# -- the real Batcher on a hostile fake model -------------------------------------------------------------------------
def _h():
    import test_batch_sessions_patches as h

    return h


COSTS = {"verify": [31.0, 39.0, 45.0, 51.0, 56.0, 62.0, 68.0, 74.0], "mtp": 2.04, "mtp_step": 1.68, "mtp_row": 0.1,
         "block": 3.88, "taps_row": 0.05}
BIAS = np.zeros(V)
for _piece, _b in (('"', 2.0), ("}", 2.5), ("]", 2.0), ('"}', 1.5), ('"]', 1.0), (",", 0.5), ("1", 0.5), ("2", 0.5)):
    BIAS[VOCAB.index(_piece)] = _b
BIAS[EOS] = 2.5
BIAS[THINK_END] = 4.0
WORDS = (V + 31) // 32


def _model():
    """The sessions harness's hostile model (a row's state reads every KV entry before it, its token and position),
    with float logits over the fake vocabulary: a row's logits are a function of its state hash only."""

    h = _h()

    class JsonModel(h._Model):
        def compute(self, w, st, b, R, **kw):
            hid = super().compute(w, st, b, R, **kw)
            rows = [np.random.default_rng(int(x) & ((1 << 63) - 1)).standard_normal(V) * 2.0 + BIAS
                    for x in hid[:, 0].tolist()]
            return torch.tensor(np.stack(rows), dtype=torch.float32)

    return JsonModel()


JM = None


def _jm():
    global JM
    if JM is None:
        JM = _model()
    return JM


def _w():
    return SimpleNamespace(mtp=object(), vocab_offset=0, comm=None, world=1, meta={},
                           cfg=SimpleNamespace(hidden=1, eos=(EOS,)), device="cpu")


def _draw(w, logits, positions, sampling, win=None) -> list[int]:
    from tensorfold.families.glm5_next.cuda import batch

    return batch.sample_multi(w, logits, [(0, logits.shape[0], list(positions), sampling)],
                              masks=[win] if win is not None and win.rows else None)[0]


def _truth(w, st, con, pending: int, k: int, sampling) -> list[int]:
    """The next k tokens serial decoding commits after ``pending`` from a copy of the slot's state, under a fork of
    the reply's grammar (the constrained continuation a perfect drafter would propose)."""

    jm = _jm()
    st = copy.deepcopy(st)
    m = con.m.fork() if con is not None else None
    active = con.active if con is not None else True
    out, t = [], pending
    for _ in range(k):
        pos = st.pos
        jm.stage(None, st, None, [t])
        lg = jm.compute(None, st, None, 1)
        jm.commit(None, st, None, 1, 1)
        win = None
        if m is not None and active and not m.is_terminated():
            bits = np.empty((1, WORDS), dtype=np.int32)
            m.fill_next_token_bitmask(bits, 0)
            win = grammar.Window([t], [0], bits)
        t = _draw(w, lg, [pos + 1], sampling, win)[0]
        out.append(t)
        if m is not None:
            if not active:
                active = t == THINK_END
            elif not m.is_terminated():
                m.accept_token(t)
    return out


def _corrupt(pos: int, drafts: list[int], salt: int) -> list[int]:
    """Deterministic corruptions (a function of the committed position: both ranks alike): a grammar-invalid token,
    the stop token, </think>, or a wrong token, now and then."""

    mix = _h().MODEL.mix
    out = []
    for j, d in enumerate(drafts):
        r = mix(pos, j, salt) % 100
        out.append(d if r < 72 else VOCAB.index("zip") if r < 80 else EOS if r < 86 else THINK_END if r < 90
                   else (d + 1 + r) % V)
    return out


class _FakeDFlash:
    """A DFlash2 stand-in: the constrained continuation with corruptions, a probability telling them apart."""

    block = 8

    def __init__(self, bat, slot: int) -> None:
        self.bat, self.slot = bat, slot
        self.context_end = 0
        self.pos_dev = torch.zeros(1, dtype=torch.int64)

    def reset(self) -> None:
        self.context_end = 0

    def add_taps(self, taps) -> None:
        self.context_end += int(taps.shape[0])

    def propose(self, last, k, sampling, confidence, probs=None):
        bat = self.bat
        q = bat.seqs[self.slot]
        st = bat.states[self.slot]
        k = min(k, self.block - 1)
        true = _truth(bat.g.w, st, q.con, int(last), k, sampling)
        out = _corrupt(st.pos, true, 11)
        if probs is not None:
            probs.extend(0.97 if a == b else 0.2 for a, b in zip(out, true))
        return out


_BATCHERS: list = []                              # every fake batcher: the MTP stand-in finds a state's slot


def _fake_mtp_draft(bat):
    if bat not in _BATCHERS:
        _BATCHERS.append(bat)

    def draft(e, hidden, next_tokens, position, count, sampling, confidence=0.0, opt=None):
        st = e.st
        bat, slot = next((b, i) for b in reversed(_BATCHERS) for i, s in enumerate(b.states) if s is st)
        true = _truth(bat.g.w, st, bat.seqs[slot].con, int(next_tokens[-1]), count, sampling)
        bad = _corrupt(st.pos, true, 29)
        drafts, chained = [], 0
        if opt is not None:
            opt.mtp_begin()
        for j in range(count):
            ok = bad[j] == true[j]
            if opt is not None:
                take, more = opt.mtp_next(j, 0.98 if ok else 0.3, count)
                if not take:
                    break
                drafts.append(bad[j])
                if not more:
                    break
            else:
                drafts.append(bad[j])
            if j + 1 < count:
                chained += 1
        st.mtp_drafted = chained
        return drafts

    return draft


def _batcher(monkeypatch, *, n: int = 4, threads: int = 2, on: bool = True, rank: int = 0, store: bool = False,
             gr=None):
    """A real ``Batcher`` with the real Stepper / sample_multi / prefill on the fake model (sessions off unless
    ``store``)."""

    from tensorfold.families.glm5_next.cuda import batch, decode

    h = _h()
    jm = _jm()
    real_stepper, real_sample = batch.Stepper, batch.sample_multi
    bat = h._fake_batcher(monkeypatch, n=n, rows=64, fast=False, piece=64, budget_pages=4000.0, rank=rank)
    for name in ("stage", "compute", "commit"):
        monkeypatch.setattr(decode, name, getattr(jm, name))
    monkeypatch.setattr(batch, "Stepper", real_stepper)
    monkeypatch.setattr(batch, "sample_multi", real_sample)
    monkeypatch.setattr(batch, "commit", jm.commit)
    monkeypatch.setattr(decode, "draft", _fake_mtp_draft(bat))
    if not store:
        bat.store = None
        bat.g.store = None
        bat._remember = lambda slot, snap: None
    bat.eos = (EOS,)
    g = bat.g
    g.w = g.e.w = _w()
    g.drafter = object()
    g.f_most = 7
    g.grammar_on = on
    g.grammars = gr or grammars()
    g.filler = grammar.Filler(threads)
    bat.drafters = [_FakeDFlash(bat, s) for s in range(n)]
    bat.m_rows = [torch.zeros((batch.BACKLOG, 1), dtype=torch.int64) for _ in range(n)]
    bat.f_taps = [torch.zeros((batch.BACKLOG, 1), dtype=torch.int64) for _ in range(n)]
    e = g.e
    e.sample = lambda logits, positions, sampling, draft=False, probs=None: _draw(g.w, logits, positions, sampling)
    e.tap_rows = lambda R: torch.zeros((R, 1), dtype=torch.int64)
    e.main_hidden = lambda rows: e.st.hid[0:rows.stop - rows.start]
    e.buf = SimpleNamespace(taps=[torch.zeros((n * 16 + 16, 1), dtype=torch.int64)])
    bat.costs = COSTS
    bat.round_costs = batchplan.RoundCosts(COSTS, 6.5)
    windows = []

    def spy(active, wins):                         # the round's forward: each slot's rows on its own state
        windows.append([len(x) for x in wins])
        out = []
        for s, win in zip(active, wins):
            with bat._on(s) as ee:
                jm.stage(None, ee.st, None, win)
                out.append(jm.compute(None, ee.st, None, len(win)))
        return torch.cat(out)

    bat._forward = spy
    bat.windows = windows
    return bat


def _codes():
    from tensorfold.families.glm5_next.cuda import depth, lookup

    return {"serial": [0, 0, 0, 0], "mtp": [1, 3, 0, 0], "dflash": [11, 5, 0, 0], "auto+lookup": [4, 2, 8, 30000, 1, 3],
            "of7": depth.parse_policy("of7"), "om7": depth.parse_policy("om7"), "l7:3": lookup.parse_policy("l7:3")}


def _job(prompt, tokens, code, sampling, gram=None, *, stop_eos=False, draft=True):
    from tensorfold.families.glm5_next.cuda.batch import Job

    h = _h()
    values = dict(h._values(False, 64), auto_fdrafts=7)
    cost = int(code[0] in (4, 5) or code[0] == 6)
    job = Job(list(prompt), tokens, sampling, stop_eos, draft, list(code), "t", cost, values, out=queue.SimpleQueue())
    job.gram = gram
    return job


def _serve(bat, jobs, arrive=None):
    """Rank 0's loop, round by round, the jobs queued at their arrival rounds. -> [(reply, stats, error)]."""

    arrive = arrive or [0] * len(jobs)
    queued = [False] * len(jobs)
    rnd = 0
    while True:
        for i, j in enumerate(jobs):
            if not queued[i] and arrive[i] <= rnd:
                bat.queue.append(j)
                queued[i] = True
        if bat.queue or any(s is not None for s in bat.seqs):
            cancels, admits, pieces = bat._plan()
            bat._execute(cancels, admits, pieces)
        elif all(queued):
            break
        rnd += 1
        assert rnd < 20000
    out = []
    for j in jobs:
        got, err, done = [], None, False
        while not j.out.empty():
            item = j.out.get()
            if item is None:
                done = True
            elif isinstance(item, BaseException):
                err = item
                break
            else:
                got.extend(item)
        assert done or err is not None
        out.append((got, j.stats, err))
    return out


def _reference(prompt, tokens, sampling, gram=None, *, stop_eos=False):
    """A fresh prefill and serial decoding with the reply's grammar, outside the batcher."""

    from tensorfold.families.glm5_next.cuda import decode

    h = _h()
    jm = _jm()
    st = h._St()
    e = h._E(st, 64, False)
    w = e.w = _w()
    e.sample = lambda logits, positions, sampling_, draft=False, probs=None: _draw(w, logits, positions, sampling_)
    con = grammars().constraint(gram) if gram is not None else None
    with pytest.MonkeyPatch.context() as m:
        for name in ("stage", "compute", "commit"):
            m.setattr(decode, name, getattr(jm, name))
        with grammar.first_token(e, con):
            first = decode.prefill(e, list(prompt), sampling, mtp=True, drafter=None)
    if con is not None:
        con.advance([first])
    out = [first]
    while len(out) < tokens and not (stop_eos and out[-1] == EOS):
        pos = st.pos
        jm.stage(None, st, None, [out[-1]])
        lg = jm.compute(None, st, None, 1)
        jm.commit(None, st, None, 1, 1)
        win = con.fill(con.cut([out[-1]])) if con is not None else None
        t = _draw(w, lg, [pos + 1], sampling, win)[0]
        if con is not None:
            con.advance([t])
        out.append(t)
    return out


def _prompt(seed: int, think: bool | None = None) -> list[int]:
    rng = random.Random(seed)
    p = [rng.randrange(3, V - 20) for _ in range(rng.randint(20, 90))]
    if think is True:
        p += [THINK]
    elif think is False:
        p += [THINK, THINK_END]
    return p


def _json_ok(reply: list[int], schema=None) -> bool:
    """The reply's visible text after </think> (when the prompt thinks) parses; with ``schema`` it validates."""

    body = reply[:reply.index(EOS)] if EOS in reply else reply
    value = json.loads(text(body))
    if schema is not None:
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.validate(value, schema)
    return True


GIDS = ["greedy", "sampled"]


@pytest.mark.parametrize("greedy", [True, False], ids=GIDS)
@pytest.mark.parametrize("policy", ["mtp", "dflash", "auto+lookup", "of7", "om7", "l7:3"])
def test_drafted_equals_serial_and_reference(monkeypatch, policy, greedy):
    """A constrained drafted reply == the same request with "draft": false == the independent serial reference;
    windows were cut before the forward, replies are valid JSON."""

    bat = _batcher(monkeypatch, n=2)
    sampling = None if greedy else Sampling(1234, 1.0, 20, 0.95)
    code = _codes()[policy]
    specs = [spec_schema(), spec_object()]
    for seed in range(3):
        prompt = _prompt(seed, think=seed == 2)
        gram = bound(specs[seed % 2], prompt)
        (drafted, s1, e1), = _serve(bat, [_job(prompt, 120, code, sampling, gram, stop_eos=True)])
        (serial, s2, e2), = _serve(bat, [_job(prompt, 120, _codes()["serial"], sampling, gram, stop_eos=True,
                                               draft=False)])
        assert e1 is None and e2 is None
        want = _reference(prompt, 120, sampling, gram, stop_eos=True)
        assert drafted == serial == want, (policy, seed, s1.get("drafters"))
        assert s1["grammar"]["rows"] > 0 and s2["grammar"]["cut"] == 0
        if EOS in drafted:
            vis = drafted[drafted.index(THINK_END) + 1:] if seed == 2 else drafted
            assert _json_ok(vis, SCHEMA if seed % 2 == 0 else None)
    cuts = sum(d.get("grammar", {}).get("cut", 0) for d in [s1])
    assert any(max(r) > 1 for r in bat.windows), "no drafted window"
    assert cuts >= 0


@pytest.mark.parametrize("greedy", [True, False], ids=GIDS)
@pytest.mark.parametrize("threads", [0, 4])
def test_four_slots_mixed_equal_alone(monkeypatch, greedy, threads):
    """4 slots mixing constrained (schema / object / thinking) and plain requests, arriving at different rounds ==
    each request alone; plain replies == the knob off."""

    bat = _batcher(monkeypatch, n=4, threads=threads)
    sampling = None if greedy else Sampling(77, 1.0, 20, 0.95)
    codes = _codes()
    pols = ["auto+lookup", "mtp", "dflash", "om7", "of7", "l7:3"]
    reqs = []
    for i in range(6):
        prompt = _prompt(100 + i, think=True if i == 3 else None)
        gram = None if i in (1, 4) else bound([spec_schema(), spec_object()][i % 2], prompt)
        reqs.append((prompt, codes[pols[i]], gram))
    jobs = [_job(p, 90, c, sampling, g_, stop_eos=True) for p, c, g_ in reqs]
    together = _serve(bat, jobs, arrive=[0, 0, 1, 3, 5, 9])
    assert all(e is None for _, _, e in together)
    assert any(s.get("batched_rounds", 0) > 0 for _, s, _ in together)
    for (p, c, g_), (got, stats, _) in zip(reqs, together):
        (alone, _, _), = _serve(bat, [_job(p, 90, c, sampling, g_, stop_eos=True)])
        assert got == alone == _reference(p, 90, sampling, g_, stop_eos=True), stats.get("drafters")
    off = _batcher(monkeypatch, n=4, on=False)
    plain = [(p, c) for p, c, g_ in reqs if g_ is None]
    got_off = _serve(off, [_job(p, 90, c, sampling, stop_eos=True) for p, c in plain])
    got_on = _serve(bat, [_job(p, 90, c, sampling, stop_eos=True) for p, c in plain])
    for (a, sa, _), (b, sb, _) in zip(got_off, got_on):
        assert a == b and sa["keeps"] == sb["keeps"] and sa["drafters"] == sb["drafters"]
        assert "grammar" not in sb


def _random_schema(rng: random.Random, depth: int = 0) -> dict:
    kind = rng.choice(["string", "integer", "number", "boolean", "enum", "array"] + (["object"] * 2 if depth < 2
                                                                                     else []))
    if kind == "string":
        return {"type": "string", "maxLength": rng.randint(1, 10)}
    if kind == "integer":
        return {"type": "integer", "minimum": -50, "maximum": 5000} if rng.random() < 0.5 else {"type": "integer"}
    if kind == "number":
        return {"type": "number"}
    if kind == "boolean":
        return {"type": "boolean"}
    if kind == "enum":
        return {"type": "string", "enum": rng.sample(["ab", "cd", "the", "x_", "zip"], rng.randint(1, 4))}
    if kind == "array":
        return {"type": "array", "items": _random_schema(rng, depth + 1), "maxItems": rng.randint(1, 3)}
    names = rng.sample(["name", "id", "ok", "tag", "items", "value", "city", "zip", "score"], rng.randint(1, 4))
    props = {k: _random_schema(rng, depth + 1) for k in names}
    req = [k for k in names if rng.random() < 0.7]
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


@pytest.mark.parametrize("seed", range(4))
def test_random_schemas_give_valid_json(monkeypatch, seed):
    """Random schemas (nested objects, arrays, enums, bounds; the root always an object), greedy and sampled, drafted:
    every reply that ends (its stop token) parses and validates; no stop token inside the JSON; drafted == serial."""

    jsonschema = pytest.importorskip("jsonschema")
    bat = _batcher(monkeypatch, n=4)
    rng = random.Random(seed)
    jobs, meta = [], []
    for i in range(8):
        schema = {"type": "object", "properties": {"a": _random_schema(rng, 1), "b": _random_schema(rng, 1)},
                  "required": ["a"] if i % 2 else ["a", "b"], "additionalProperties": False}
        prompt = _prompt(1000 * seed + i, think=bool(i % 3 == 0))
        sampling = None if i % 2 else Sampling(9 + i, 1.0, 20, 0.95)
        gram = bound(spec_schema(schema), prompt)
        jobs.append(_job(prompt, 200, _codes()[["auto+lookup", "mtp", "dflash", "om7"][i % 4]], sampling, gram,
                         stop_eos=True))
        meta.append((schema, prompt, sampling, gram))
    ended = 0
    for (got, stats, err), (schema, prompt, sampling, gram) in zip(_serve(bat, jobs), meta):
        assert err is None
        if not gram.active and THINK_END not in got:
            vis = []                                    # it ended (or was cut) while thinking: no answer yet
        else:
            vis = got[got.index(THINK_END) + 1:] if not gram.active else got
        if EOS in vis:
            ended += 1
            assert vis.index(EOS) == len(vis) - 1          # the stop token ends it: nothing inside the JSON
            jsonschema.validate(json.loads(text(vis[:-1])), schema)
            assert stats["grammar"]["finished"]
        assert got == _reference(prompt, 200, sampling, gram, stop_eos=True)
    assert ended >= 4, ended


@pytest.mark.parametrize("greedy", [True, False], ids=GIDS)
def test_thinking_on_and_off(monkeypatch, greedy):
    """Thinking on: the reasoning is the unconstrained reply's, token for token, up to and including </think>; the
    grammar starts after it. Thinking off (<think></think> in the prompt): constrained from the first token."""

    bat = _batcher(monkeypatch, n=2)
    sampling = None if greedy else Sampling(5, 1.0, 20, 0.95)
    code = _codes()["auto+lookup"]
    seen = 0
    for seed in range(6):
        prompt = _prompt(300 + seed, think=True)
        gram = bound(spec_schema(), prompt)
        assert not gram.active
        (con_reply, stats, _), (plain, _, _) = _serve(bat, [_job(prompt, 150, code, sampling, gram, stop_eos=True),
                                                            _job(prompt, 150, code, sampling, stop_eos=True)])
        if THINK_END in plain:
            k = plain.index(THINK_END)
            if EOS not in plain[:k]:
                seen += 1
                assert con_reply[:k + 1] == plain[:k + 1]
                after = con_reply[k + 1:]
                if EOS in after:
                    assert _json_ok(after, SCHEMA)
        off_prompt = _prompt(300 + seed, think=False)
        g_off = bound(spec_schema(), off_prompt)
        assert g_off.active
        (reply, _, _), = _serve(bat, [_job(off_prompt, 150, code, sampling, g_off, stop_eos=True)])
        assert text(reply[:1]) in ("{", '{"')
        if EOS in reply:
            assert _json_ok(reply, SCHEMA)
    assert seen >= 2


def test_stop_and_eos_interplay(monkeypatch):
    """stop_eos: the stop token exactly where the grammar completes, never inside; ignore_eos: the same tokens up to
    it, then unconstrained decoding on; max_tokens cuts a reply mid-JSON without an error."""

    bat = _batcher(monkeypatch, n=2)
    code = _codes()["auto+lookup"]
    for seed in range(5):
        prompt = _prompt(500 + seed)
        gram = bound(spec_object(), prompt)
        (stop, s1, _), (ign, s2, _) = _serve(bat, [_job(prompt, 160, code, None, gram, stop_eos=True),
                                                   _job(prompt, 160, code, None, gram, stop_eos=False)])
        if EOS in stop:
            k = stop.index(EOS)
            assert k == len(stop) - 1 and _json_ok(stop)
            assert ign[:k + 1] == stop and len(ign) == 160 and s2["grammar"]["finished"]
            assert ign == _reference(prompt, 160, None, gram, stop_eos=False)
        cut = max(1, (len(stop) - 1) // 2)             # mid-JSON
        (short, s3, err), = _serve(bat, [_job(prompt, cut, code, None, gram, stop_eos=True)])
        assert err is None and short == stop[:cut] and not s3["grammar"]["finished"] or cut >= len(stop) - 1


def test_drafts_are_cut_before_the_forward(monkeypatch):
    """Drafts the grammar rejects never reach the forward: the verified windows are the cut ones (stats count the
    drafts cut), and the reply is unchanged."""

    bat = _batcher(monkeypatch, n=1)
    code = _codes()["dflash"]
    prompt = _prompt(700)
    gram = bound(spec_schema(), prompt)
    (got, stats, _), = _serve(bat, [_job(prompt, 100, code, None, gram, stop_eos=True)])
    assert stats["grammar"]["cut"] > 0
    assert got == _reference(prompt, 100, None, gram, stop_eos=True)
    rows = sum(len(k) for k in bat.windows)
    assert sum(stats["depths"]) + stats["rounds"] == sum(sum(k) for k in bat.windows) and rows


def test_grammar_failure_ends_only_its_slot(monkeypatch):
    bat = _batcher(monkeypatch, n=3)
    code = _codes()["mtp"]
    prompts = [_prompt(800 + i) for i in range(3)]
    grams = [bound(spec_object(), p) for p in prompts]
    jobs = [_job(p, 60, code, None, g_, stop_eos=False) for p, g_ in zip(prompts, grams)]
    admit = bat._admit
    calls = collections.Counter()

    def admit_spy(slot, cached, job):
        admit(slot, cached, job)
        q = bat.seqs[slot]
        if job is jobs[1]:
            fill = q.con.fill

            def failing(win):
                calls["fill"] += 1
                if calls["fill"] == 6:
                    raise grammar.GrammarError("injected")
                return fill(win)

            q.con.fill = failing

    bat._admit = admit_spy
    out = _serve(bat, jobs)
    assert isinstance(out[1][2], grammar.GrammarError) and "injected" in str(out[1][2])
    for i in (0, 2):
        assert out[i][2] is None and out[i][0] == _reference(prompts[i], 60, None, grams[i])
    assert all(s is None for s in bat.seqs)


def test_follower_compiles_and_decides_the_same(monkeypatch):
    """Rank 1 (a second batcher with its own compiler) replaying rank 0's plans: the grammar travels in front of the
    header, rank 1 compiles it, and both decide the same windows, keeps and replies."""

    r0 = _batcher(monkeypatch, n=3)
    r1 = _batcher(monkeypatch, n=3, rank=1, gr=other_grammars())
    sent: list[list[int]] = []

    def record(values):
        sent.append([int(v) for v in values])
        return list(values)

    class Done(Exception):
        pass

    def replay(values):
        if not sent:
            raise Done
        return sent.pop(0)

    r0.g._share, r1.g._share = record, replay
    sampling = Sampling(3, 1.0, 20, 0.95)
    codes = _codes()
    jobs = []
    for i, pol in enumerate(["auto+lookup", "mtp", "dflash", "om7"]):
        p = _prompt(900 + i, think=i == 2)
        jobs.append(_job(p, 80, codes[pol], sampling, None if i == 3 else bound(spec_schema(), p), stop_eos=True))
    _serve(r0, jobs, arrive=[0, 0, 2, 4])
    assert sum(plan.count(grammar.MARK) for plan in sent) == 3      # three headers carried a grammar
    with pytest.raises(Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["arms"], d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log]
    assert r1.windows == r0.windows


@pytest.mark.parametrize("policy", ["serial", "mtp", "dflash", "auto+lookup"])
@pytest.mark.parametrize("threads", [0, 2])
def test_lone_loops_equal_serial(monkeypatch, policy, threads):
    """The lone engine's loops (GLM53_TF_BATCH=1) with ``e.constraint``: serial / MTP / DFlash2 / auto == the serial
    reference; the prefill's first token masked (``first_token``)."""

    from tensorfold.families.glm5_next.cuda import decode, depth, lookup
    from tensorfold.families.glm5_next.cuda.decode import (DepthPolicy, DrafterChoice, auto_decode, dflash_decode,
                                                           mtp_decode, prefill, serial_decode)

    h = _h()
    jm = _jm()
    for name in ("stage", "compute", "commit"):
        monkeypatch.setattr(decode, name, getattr(jm, name))
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    cut = drafted = 0
    for seed in range(4):
        sampling = None if seed % 2 else Sampling(40 + seed, 1.0, 20, 0.95)
        prompt = _prompt(1200 + seed, think=seed == 2)
        gram = bound(spec_schema(), prompt)
        want = _reference(prompt, 100, sampling, gram, stop_eos=True)
        st = h._St()
        e = h._E(st, 64, False)
        e.w = _w()
        e.sample = lambda logits, positions, s_, draft=False, probs=None, e=e: _draw(e.w, logits, positions, s_)
        e.buf = SimpleNamespace(taps=[torch.zeros((64, 1), dtype=torch.int64)])
        e.tap_rows = lambda R: torch.zeros((R, 1), dtype=torch.int64)
        e.main_hidden = lambda rows, e=e: e.st.hid[rows]
        e.rows = 16

        def forward(win, e=e):
            jm.stage(None, e.st, None, win)
            return jm.compute(None, e.st, None, len(win))

        e.forward = forward
        e.constraint = grammars().constraint(gram)
        e.grammar_filler = grammar.Filler(threads)
        bat = SimpleNamespace(states=[st], seqs=[SimpleNamespace(con=e.constraint)], g=SimpleNamespace(w=e.w))
        monkeypatch.setattr(decode, "draft", _fake_mtp_draft(bat))
        dr = _FakeDFlash(bat, 0) if policy in ("dflash", "auto+lookup") else None
        with grammar.first_token(e, e.constraint):
            first = prefill(e, list(prompt), sampling, mtp=True, drafter=dr)
        e.constraint.advance([first])
        e.lookup = lookup.lookup_for(_codes()["auto+lookup"], prompt, COSTS, stop_eos=True) \
            if policy == "auto+lookup" else None
        e.calib = None
        e.depth = None
        if policy == "serial":
            res = serial_decode(e, first, 100, sampling, stop_eos=True)
        elif policy == "mtp":
            res = mtp_decode(e, first, 100, sampling, policy=DepthPolicy(3, fixed=True), stop_eos=True)
        elif policy == "dflash":
            res = dflash_decode(e, dr, first, 100, sampling, policy=DepthPolicy(5, fixed=True), stop_eos=True)
        else:
            res = auto_decode(e, dr, first, 100, sampling, choice=DrafterChoice(COSTS, first="f"),
                              m_policy=DepthPolicy(3, fixed=True), f_policy=DepthPolicy(7, fixed=True),
                              stop_eos=True)
        assert res.tokens == want, (policy, seed)
        cut += e.constraint.stats["cut"]
        drafted += max(res.keeps, default=1) > 1
    if policy != "serial":
        assert cut > 0 and drafted > 0


def test_sessions_resume_under_a_grammar(monkeypatch):
    """With the session store behind the slots: the same constrained request again resumes from the store (the first
    token still from a masked fresh row) and gives the same reply; a follow-up turn over the reply too."""

    bat = _batcher(monkeypatch, n=2, store=True)
    code = _codes()["auto+lookup"]
    prompt = _prompt(1500) + list(range(3, 150))
    gram = bound(spec_object(), prompt)
    (a, sa, _), = _serve(bat, [_job(prompt, 80, code, None, gram, stop_eos=True)])
    (b, sb, _), = _serve(bat, [_job(prompt, 80, code, None, gram, stop_eos=True)])
    assert a == b == _reference(prompt, 80, None, gram, stop_eos=True)
    assert sb["cached"] > 0
    nxt = prompt + a + [5, 6, 7]
    g2 = bound(spec_schema(), nxt)
    (c, sc, _), = _serve(bat, [_job(nxt, 80, code, None, g2, stop_eos=True)])
    assert c == _reference(nxt, 80, None, g2, stop_eos=True)


# -- the engine's wiring (lone path, two ranks), load-time setup, the HTTP app ---------------------------------------------
class _Done(Exception):
    pass


def _lone_engine(rank: int, share, gr):
    import contextlib

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    h = _h()
    g = object.__new__(GlmEngine)
    runs = []
    g.__dict__.update(batch=None, request=threading.local(), limit=1 << 20, serial_only=False, policy="auto",
                      w=SimpleNamespace(mtp=object()), longctx_buffers=True, online=None, store=None, rank=rank,
                      grammar_on=True, grammars=gr, vision=None, cache=[], runs=runs, e=SimpleNamespace())
    g.parse_knobs = lambda raw: {}
    g._effective = lambda code: code
    g._knob_state = lambda: dict(h._values(False, 64))
    g._request_grid = lambda values: 0
    g._resume = lambda prompt, code, grid=None: None
    g._adopt = lambda table: None
    g._knobs = lambda values: contextlib.nullcontext()
    g._share = share

    def run(prompt, max_tokens, sampling, stop_eos, on_tokens, code, hit, draft, sess=None, vis=None):
        con = getattr(g.e, "constraint", None)       # the reply's grammar on the engine while it runs
        runs.append((list(prompt), max_tokens, getattr(con, "bound", None)))
        return {}

    g._run = run
    return g


def test_lone_engine_header_and_follower(monkeypatch):
    """GLM53_TF_BATCH=1: rank 0 puts the grammar in front of the request header, rank 1 splits it off, compiles it
    with its own compiler and runs the same reply under it; a plain request's header is unchanged."""

    sent = []

    def record(values):
        sent.append([int(v) for v in values])
        return list(values)

    def replay(values):
        if not sent:
            raise _Done
        return sent.pop(0)

    r0 = _lone_engine(0, record, grammars())
    r1 = _lone_engine(1, replay, other_grammars())
    prompt = _prompt(1700, think=True)
    spec = spec_schema()
    r0.request.grammar = (spec, grammars().compile(spec))
    r0.generate(prompt, 30, None, lambda new: False)
    constrained = list(sent)
    r0.request.grammar = None
    r0.generate(prompt, 30, None, lambda new: False)
    assert constrained[0][0] == grammar.MARK and sent[2][0] == 30             # a plain header starts at max_tokens
    assert grammar.split(constrained[0])[1] == sent[2]                        # the rest of the header unchanged
    with pytest.raises(_Done):
        r1.follow()
    (p0, m0, g0), (p1, m1, g1) = r0.runs
    (q0, n0, h0), (q1, n1, h1) = r1.runs
    assert p0 == q0 and m0 == n0 and g1 is None and h1 is None
    assert (h0.spec.kind, h0.spec.text, h0.think_end, h0.active) == (g0.spec.kind, g0.spec.text, g0.think_end,
                                                                      g0.active) and not g0.active


def test_setup(monkeypatch):
    def fake_g(on_other: int, rank: int = 0, vocab: int = V):
        return SimpleNamespace(_gather_ints=lambda v: [list(v), [on_other] + list(v[1:])], model_dir=Path("/x"),
                               eos=(EOS,), rank=rank, w=SimpleNamespace(head=SimpleNamespace(n=vocab // 2),
                                                                         comm=object(), world=2))

    monkeypatch.setenv("GLM53_TF_GRAMMAR", "0")
    g = fake_g(0)
    grammar.setup(g)
    assert g.grammar_on is False and g.grammars is None and g.filler is None
    with pytest.raises(RuntimeError, match="different GLM53_TF_GRAMMAR"):
        grammar.setup(fake_g(1))
    monkeypatch.setenv("GLM53_TF_GRAMMAR", "1")
    monkeypatch.setattr(grammar, "vocab_size", lambda d: V)
    monkeypatch.setattr(grammar.Grammars, "from_model", classmethod(lambda cls, d, v, s: grammars()))
    g = fake_g(1)
    grammar.setup(g)
    assert g.grammar_on and g.grammars is grammars() and g.filler is not None
    g = fake_g(1, vocab=V + 32)                          # config.json's vocab_size is not the logits' width
    grammar.setup(g)
    assert g.grammars is None and "not the logits" in g.grammar_why
    assert "not the logits" in grammar.check(g, {"response_format": {"type": "json_object"}})

    def no_xgrammar(cls, d, v, s):
        raise ImportError("xgrammar")

    monkeypatch.setattr(grammar.Grammars, "from_model", classmethod(no_xgrammar))
    g = fake_g(1)
    grammar.setup(g)
    assert g.grammars is None and "needs xgrammar" in grammar.check(g, {"response_format": {"type": "json_object"}})
    assert grammar.check(g, {}) is None


def _app(on: bool):
    from tensorfold.families.glm5_next.cuda.app import GlmApp

    app = object.__new__(GlmApp)

    class Engine:
        eos = (EOS,)
        request = threading.local()
        grammar_on = on
        grammars = globals()["grammars"]()

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            return {}

    app.__dict__.update(engine=Engine(), effort_field=False, default_effort=None, default_thinking=False,
                        vision=None, reqlog=None, tok=None)
    return app


def test_app_knob_off_ignores_the_fields():
    app = _app(False)
    for body in ({"response_format": "json"}, {"response_format": {"type": "json_schema"}},
                 {"guided_regex": ""}, {"tools": TOOLS, "tool_choice": "required"}):
        assert app.check(dict(body, messages=[])) is None


def test_app_knob_on(monkeypatch):
    from tensorfold.cuda import server

    app = _app(True)
    assert "must be an object" in app.check({"messages": [], "response_format": "json"})
    assert "cannot be enforced" in app.check({"messages": [], "guided_grammar": "root ::= nope"})
    assert "cannot be combined" in app.check({"messages": [], "tools": TOOLS, "tool_choice": "required",
                                              "response_format": {"type": "json_object"}})
    ok = {"messages": [], "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}}
    assert app.check(ok) is None
    monkeypatch.setattr(server.App, "run", lambda self, body, chat, emit: {"grammar": self.engine.request.grammar})
    got = app.run(dict(ok), True, lambda d: True)["grammar"]
    assert got[0].kind == "json_schema" and got[1] is not None
    assert app.run({"messages": []}, True, lambda d: True)["grammar"] is None


# -- the real GLM tokenizer (optional) -----------------------------------------------------------------------------------
_REAL = None


def _real() -> grammar.Grammars:
    global _REAL
    if _REAL is None:
        _REAL = grammar.Grammars.from_model(TOKDIR, grammar.vocab_size(TOKDIR), (154820, 154827, 154829))
    return _REAL


def _real_tok():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKDIR / "tokenizer.json"))


@real_tok
def test_real_tokenizer_masks():
    g = _real()
    tok = _real_tok()
    assert g.vocab_size == 154880 and g.think_open == tok.token_to_id("<think>") == 154841
    assert g.think_end == tok.token_to_id("</think>") == 154842
    added = json.loads((TOKDIR / "tokenizer.json").read_text())["added_tokens"]
    con = g.constraint(bound(spec_object(), prompt=[1, 2], g=g))
    start = con.fill(con.cut([0]))
    ids = tok.encode('{"a": "x', add_special_tokens=False).ids
    con.advance(ids)
    inside = con.fill(con.cut([ids[-1]]))                  # inside a JSON string: nearly everything is allowed
    for win in (start, inside):
        allowed = set(np.nonzero(np.unpackbits(win.bits[0].view(np.uint8), bitorder="little"))[0].tolist())
        assert not any(a["id"] in allowed for a in added)   # no added token (</think>, <|user|>, <tool_call>, ...)
        assert not any(t in allowed for t in range(154856, 154880))
    assert len(np.nonzero(np.unpackbits(inside.bits[0].view(np.uint8), bitorder="little"))[0]) > 100000
    con.advance(tok.encode('"}', add_special_tokens=False).ids)
    end = con.fill(con.cut([0]))
    assert set(np.nonzero(np.unpackbits(end.bits[0].view(np.uint8), bitorder="little"))[0].tolist()) == {
        154820, 154827, 154829}
    # the think gate reads the prompt: the template's thinking-on prompt ends with <think>, thinking off <think></think>
    assert not grammar.think_active([1, 154828, 154841], g.think_open, g.think_end)
    assert grammar.think_active([1, 154828, 154841, 154842], g.think_open, g.think_end)


@real_tok
@pytest.mark.parametrize("choice", ["required", "strict", "named"])
def test_real_tokenizer_tool_calls(choice):
    """The tool grammar takes GLM's markup tokens (single tokens, as the model emits them), holds the arguments to
    the schema, and its calls parse into OpenAI tool_calls whose arguments validate."""

    jsonschema = pytest.importorskip("jsonschema")
    from tensorfold.cuda.server import parse_tool_calls

    g = _real()
    tok = _real_tok()
    tools = [dict(TOOLS[0], function=dict(TOOLS[0]["function"], strict=True)), TOOLS[1]]
    body = {"tools": tools if choice == "strict" else TOOLS,
            "tool_choice": {"required": "required", "strict": "auto",
                            "named": {"type": "function", "function": {"name": "get_weather"}}}[choice]}
    spec = grammar.request_spec(body)
    assert spec.kind == "tools"
    b = bound(spec, prompt=[1, 154842], g=g)
    marks = {s: tok.token_to_id(s) for s in grammar.TOOL_TOKENS}

    def ids(parts):
        out = []
        for p in parts:
            out += [marks[p]] if p in marks else tok.encode(p, add_special_tokens=False).ids
        return out

    good = ["<tool_call>", "get_weather", "<arg_key>", "city", "</arg_key>", "<arg_value>", "Paris", "</arg_value>",
            "<arg_key>", "days", "</arg_key>", "<arg_value>", "3", "</arg_value>", "</tool_call>"]
    con = g.constraint(b)
    seq = ids(good)
    con.advance(seq)
    win = con.fill(con.cut([seq[-1]]))
    allowed = set(np.nonzero(np.unpackbits(win.bits[0].view(np.uint8), bitorder="little"))[0].tolist())
    assert 154829 in allowed                                  # <|observation|> (a stop token) may end the reply
    content, calls = parse_tool_calls(tok.decode(seq, skip_special_tokens=False), TOOLS)
    assert calls and calls[0]["function"]["name"] == "get_weather"
    jsonschema.validate(json.loads(calls[0]["function"]["arguments"]), TOOLS[0]["function"]["parameters"])
    bad = list(good)
    bad[12] = "three"
    con = g.constraint(b)
    with pytest.raises(grammar.GrammarError):
        con.advance(ids(bad))
    # a JSON grammar never takes the markup tokens
    j = g.constraint(bound(spec_object(), prompt=[1], g=g))
    assert j.cut([0, marks["<tool_call>"]]).cut == 1


def test_batcher_generate_reads_the_threads_grammar(monkeypatch):
    """``Batcher.generate`` (an HTTP thread) takes the grammar ``GlmEngine.generate`` bound on that thread
    (``request.grammar_bound``), and a grammar failure reaches the caller as the GrammarError itself."""

    bat = _batcher(monkeypatch, n=2)
    g = bat.g
    g.request = threading.local()
    g.parse_knobs = lambda raw: {}
    g.limit, g.policy, g.longctx_buffers = 1 << 20, "auto", True
    g._effective = lambda code: code
    bat.defaults = dict(_h()._values(False, 64))
    prompt = _prompt(1900)
    gram = bound(spec_schema(), prompt)
    results = {}

    def caller(name, bnd, fail=False):
        g.request.policy, g.request.stop_eos, g.request.knobs, g.request.background = "3", True, None, False
        g.request.grammar_bound = bnd
        got = []
        try:
            results[name] = (bat.generate(prompt, 60, None, lambda new: got.extend(new) or False), got)
        except BaseException as exc:  # noqa: BLE001
            results[name] = (exc, got)

    t = threading.Thread(target=caller, args=("ok", gram))
    t.start()
    while t.is_alive():
        if bat.queue or any(s is not None for s in bat.seqs):
            bat._execute(*bat._plan())
    t.join()
    stats, got = results["ok"]
    assert got == _reference(prompt, 60, None, gram, stop_eos=True) and stats["grammar"]["rows"] > 0
    # a failing grammar: the caller gets the GrammarError (a 500 / stream error event), not "the batch loop failed"
    admit = bat._admit

    def admit_spy(slot, cached, job):
        admit(slot, cached, job)
        con = bat.seqs[slot].con

        def failing(win):
            raise grammar.GrammarError("injected")

        con.fill = failing

    bat._admit = admit_spy
    t = threading.Thread(target=caller, args=("bad", gram))
    t.start()
    while t.is_alive():
        if bat.queue or any(s is not None for s in bat.seqs):
            bat._execute(*bat._plan())
    t.join()
    assert isinstance(results["bad"][0], grammar.GrammarError)
