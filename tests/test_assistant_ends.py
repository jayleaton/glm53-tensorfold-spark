"""patches/0680: GLM's ``<|assistant|>`` role token ends a reply (GLM53_TF_ASSISTANT_ENDS; ``glm5_next/cuda/ends.py``,
which ``weights.Config.read`` calls for ``Config.eos``).

Host only. The checkpoint tests read $GLM53_TF_TOKENIZER_DIR (config.json, tokenizer.json; they skip without it).
``test_config_read_eos`` runs the real ``Config.read`` in a child process, with mocks standing in for torch / triton
where they are not installed (``Config.read`` only reads json).
Run against the patched tree: GLM53_TF_TOKENIZER_DIR=<dir> PYTHONPATH=<tree>/src pytest -q tests/test_assistant_ends.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    from tensorfold.families.glm5_next.cuda import ends
except ImportError:                     # patches/0680 not applied: ``test_config_read_eos`` still runs (and fails)
    ends = None
needs_ends = pytest.mark.skipif(ends is None, reason="patches/0680 not applied (no glm5_next/cuda/ends.py)")
ENV = "GLM53_TF_ASSISTANT_ENDS"

TOKDIR = Path(os.environ.get("GLM53_TF_TOKENIZER_DIR", "/nonexistent"))
checkpoint = pytest.mark.skipif(
    not all((TOKDIR / f).exists() for f in ("config.json", "tokenizer.json")),
    reason="GLM53_TF_TOKENIZER_DIR without the checkpoint's config.json / tokenizer.json")
LISTED = [154820, 154827, 154829]       # the checkpoint's eos_token_id: <|endoftext|>, <|user|>, <|observation|>
ASSISTANT = 154828


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


_READ = """
import importlib.util, sys
from unittest import mock
for names in (("torch",), ("triton", "triton.language")):
    if importlib.util.find_spec(names[0]) is None:
        sys.modules.update((m, mock.MagicMock(name=m)) for m in names)
from tensorfold.families.glm5_next.cuda.weights import Config
print(list(Config.read(sys.argv[1]).eos))
"""


def _config_read_eos(model_dir: Path, knob: str | None) -> list[int]:
    env = {k: v for k, v in os.environ.items() if k != ENV}
    if knob is not None:
        env[ENV] = knob
    done = subprocess.run([sys.executable, "-c", _READ, str(model_dir)], env=env, capture_output=True, text=True)
    if done.returncode:
        lines = done.stderr.strip().splitlines()
        raise RuntimeError(lines[-1] if lines else "Config.read failed")
    return json.loads(done.stdout.strip().splitlines()[-1])


@checkpoint
@pytest.mark.skipif(importlib.util.find_spec("tensorfold.families.glm5_next") is None,
                    reason="no tensorfold glm5_next family on PYTHONPATH")
def test_config_read_eos():
    """The real ``weights.Config.read`` on the served checkpoint's config.json + tokenizer.json."""

    assert _config_read_eos(TOKDIR, None) == LISTED + [ASSISTANT]          # default: on, after the checkpoint's own
    assert _config_read_eos(TOKDIR, "1") == LISTED + [ASSISTANT]
    assert _config_read_eos(TOKDIR, "0") == LISTED
    with pytest.raises(RuntimeError, match="ValueError.*GLM53_TF_ASSISTANT_ENDS"):
        _config_read_eos(TOKDIR, "yes")


def _eos(model_dir: Path) -> tuple[int, ...]:
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return ends.config_eos(dict(raw.get("text_config") or raw), raw, model_dir)


@needs_ends
def test_knob():
    assert ends.assistant_ends("") is True and ends.assistant_ends("1") is True and ends.assistant_ends(" 1 ")
    assert ends.assistant_ends("0") is False
    for bad in ("2", "on", "true", "yes", "-1"):
        with pytest.raises(ValueError, match=ENV):
            ends.assistant_ends(bad)


@needs_ends
@checkpoint
def test_checkpoint_assistant_joins_eos(monkeypatch):
    raw = json.loads((TOKDIR / "config.json").read_text(encoding="utf-8"))
    assert list(raw.get("eos_token_id") or raw["text_config"]["eos_token_id"]) == LISTED
    assert ends.assistant_id(TOKDIR) == ASSISTANT
    assert _eos(TOKDIR) == tuple(LISTED) + (ASSISTANT,)
    monkeypatch.setenv(ENV, "0")
    assert _eos(TOKDIR) == tuple(LISTED)
    monkeypatch.setenv(ENV, "maybe")
    with pytest.raises(ValueError):
        _eos(TOKDIR)


def _model(path: Path, added: list | None, eos=(7, 8)) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": list(eos)}}), encoding="utf-8")
    if added is not None:
        (path / "tokenizer.json").write_text(json.dumps({"added_tokens": added}), encoding="utf-8")
    return path


@needs_ends
def test_read_by_content_not_by_number(tmp_path):
    d = _model(tmp_path, [{"id": 3, "content": "<|user|>", "special": True},
                          {"id": 42, "content": "<|assistant|>", "special": True}])
    assert _eos(d) == (7, 8, 42)


@needs_ends
def test_not_special_absent_unreadable_or_already_listed(tmp_path):
    assert _eos(_model(tmp_path / "a", [{"id": 42, "content": "<|assistant|>", "special": False}])) == (7, 8)
    assert _eos(_model(tmp_path / "b", None)) == (7, 8)                          # no tokenizer.json
    assert _eos(_model(tmp_path / "c", [{"id": 8, "content": "<|assistant|>", "special": True}])) == (7, 8)
    bad = _model(tmp_path / "d", None)
    (bad / "tokenizer.json").write_text("{not json", encoding="utf-8")
    assert _eos(bad) == (7, 8)


@needs_ends
def test_single_eos_int(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 5}), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": 9, "content": "<|assistant|>", "special": True}]}), encoding="utf-8")
    assert _eos(tmp_path) == (5, 9)
