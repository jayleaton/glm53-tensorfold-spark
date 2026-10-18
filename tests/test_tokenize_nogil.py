"""patches/0700: the prompt is tokenized through ``encode_batch_fast`` (GIL released), with the ids ``encode`` returns.

Host only; builds a small tokenizer in process, so it needs no checkpoint. Run against the patched tree:
PYTHONPATH=<tree>/src pytest -q tests/test_tokenize_nogil.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

server = pytest.importorskip("tensorfold.cuda.server")
if not hasattr(server.App, "_encode_ids"):
    pytest.skip("patches/0700 not applied", allow_module_level=True)
pt = pytest.importorskip("tensorfold.families.glm5_next.cuda.prompt_tokens")
tokenizers = pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402

WORDS = ["[UNK]", "[BOS]", "the", "cat", "sat", "on", "mat", "tool", "call"]


def small_tokenizer() -> Tokenizer:
    """Word-level tokenizer whose post-processor prepends [BOS], so ``add_special_tokens`` changes the ids."""

    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    return tok


class NoEncode:
    """A tokenizer whose ``encode`` fails: a call site still on ``encode`` holds the GIL for the whole text."""

    def __init__(self, tok: Tokenizer):
        self._tok = tok

    def encode(self, *_a, **_k):
        raise AssertionError("tok.encode called: the GIL is held while the prompt is tokenized")

    def encode_batch_fast(self, *a, **k):
        return self._tok.encode_batch_fast(*a, **k)


TEXTS = ["", "the cat sat on the mat", "the " * 5000 + "tool call", "unknown words fall back to unk"]


@pytest.mark.parametrize("special", [False, True])
@pytest.mark.parametrize("text", TEXTS)
def test_same_ids_as_encode(text, special):
    tok = small_tokenizer()
    app = server.App.__new__(server.App)
    app.tok = tok
    assert app._encode_ids(text, special) == tok.encode(text, add_special_tokens=special).ids


def test_special_flag_is_honoured():
    tok = small_tokenizer()
    app = server.App.__new__(server.App)
    app.tok = tok
    assert app._encode_ids("cat", True) == [1, 3]
    assert app._encode_ids("cat", False) == [3]


def test_app_call_sites_avoid_encode():
    app = server.App.__new__(server.App)
    app.tok = NoEncode(small_tokenizer())
    assert app.prompt_ids("the cat") == [2, 3]
    assert app.tokenize({"prompt": "the cat"}) == [1, 2, 3]
    assert app.tokenize({"prompt": "the cat", "add_special_tokens": False}) == [2, 3]


def test_tokenize_messages_path_avoids_encode():
    app = server.App.__new__(server.App)
    app.tok = NoEncode(small_tokenizer())
    app.default_thinking = True
    app.template = SimpleNamespace(render=lambda messages, **_k: "the cat sat")
    assert app.tokenize({"messages": [{"role": "user", "content": "x"}]}) == [2, 3, 4]


def test_prompt_tokens_full_avoids_encode():
    p = pt.PromptTokens.__new__(pt.PromptTokens)
    p.tok = NoEncode(small_tokenizer())
    assert p._full("the cat sat") == [2, 3, 4]
