"""patches/0670: the NVMe session tier's in-RAM index keeps (length, page chain, SUB digests, tail), not every
entry's ids as a list[int] (the DeepSeek kit's G18 drift, fixed the same way).

- ``DiskIds`` equals a token list exactly (length, chain, tail), also as ``list != entry.ids``; ``SessionIndex._prefix``
  (and so ``DiskIndex.find``) gives the same answers on it as on the list.
- ``DiskIndex.lcp``: exact past the last full page, else the exact prefix rounded down to SUB tokens, so the fork
  marks (rounded to the snapshot grid / PAGE, multiples of SUB) are unchanged.
- Iterating, or reading an id inside a full page, raises (nothing silently uses ids that are not in RAM).
- Retained index memory: well under 2 B a token (a list[int] is ~36 B).

Run: PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_sessdisk_index.py
"""

from __future__ import annotations

import random
import tracemalloc
from types import SimpleNamespace

import pytest

sd = pytest.importorskip("tensorfold.families.glm5_next.cuda.sessdisk")
from tensorfold.families.glm5_next.cuda.sessions import PAGE, SessionIndex, chain, common_prefix  # noqa: E402

V = 154880
LENGTHS = [0, 1, 63, 64, 65, PAGE - 1, PAGE, PAGE + 1, 2 * PAGE + 64, 3 * PAGE + 200, 5 * PAGE + 17]


def _toks(rng, n):
    return [rng.randrange(V) for _ in range(n)]


def _cases(seed, count=300):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.choice(LENGTHS)
        base = _toks(rng, n)
        prompt = base + _toks(rng, rng.choice([0, 1, 50, PAGE + 9]))
        if prompt and rng.random() < 0.5:                       # a mismatch somewhere (often inside the entry)
            k = rng.randrange(len(prompt))
            prompt[k] = (prompt[k] + 1) % V
        yield base, prompt


def test_equality_is_exact():
    for base, prompt in _cases(1):
        d = sd.DiskIds(base, chain(base))
        assert len(d) == len(base) and d.n == len(base) and len(d.tail) == len(base) % PAGE
        assert d == base and base == d and not (base != d) and d == tuple(base)
        other = prompt[:len(base)]
        assert (other == d) == (other == base) and (other != d) == (other != base)
        assert (prompt != d) == (prompt != base)
    d = sd.DiskIds(list(range(3 * PAGE + 5)), chain(list(range(3 * PAGE + 5))))
    assert d == sd.DiskIds(list(range(3 * PAGE + 5)), chain(list(range(3 * PAGE + 5))))
    assert d != sd.DiskIds(list(range(3 * PAGE + 4)), chain(list(range(3 * PAGE + 4))))
    assert (d == 5) is False and d != "x"


def test_prefix_and_find_match_the_list():
    for base, prompt in _cases(2):
        blocks = chain(prompt)
        as_list = SimpleNamespace(ids=base, chain=chain(base))
        as_disk = SimpleNamespace(ids=sd.DiskIds(base, chain(base)), chain=chain(base))
        assert SessionIndex._prefix(None, as_disk, prompt, blocks) == SessionIndex._prefix(None, as_list, prompt, blocks)


def test_index_find_and_lcp():
    rng = random.Random(3)
    ix = sd.DiskIndex(1 << 40, lambda mtp: 1000)
    stored = []
    system = _toks(rng, 700)
    for i in range(30):
        ids = (system if i % 2 else []) + _toks(rng, rng.choice(LENGTHS[1:]))
        if ids:
            ix.add(tag=0, ids=ids, mtp_len=-1, drafter=False, entry_bytes=10)
            stored.append(ids)
    for e in ix.entries.values():
        assert isinstance(e.ids, sd.DiskIds) and e.ids.chain is e.chain
    for _ in range(300):
        src = rng.choice(stored)
        prompt = src[:rng.randrange(len(src) + 1)] + _toks(rng, rng.choice([0, 1, 40, 300]))
        if prompt and rng.random() < 0.5:
            k = rng.randrange(len(prompt))
            prompt[k] = (prompt[k] + 1) % V
        best = max((s for s in stored if len(s) < len(prompt) and prompt[:len(s)] == s), key=len, default=None)
        got = ix.find(prompt, 0, False, False)
        assert (got is None) == (best is None) and (got is None or len(got.ids) == len(best))
        exact = max(common_prefix(prompt, s) for s in stored)
        v = ix.lcp(prompt)
        assert v <= exact and v // sd.SUB == exact // sd.SUB
        for s in stored:                                        # per entry: exact in the tail
            x, y = common_prefix(prompt, s), sd.DiskIds(s, chain(s)).lcp(prompt, chain(prompt))
            assert y <= x and y // sd.SUB == x // sd.SUB and (x < len(s) // PAGE * PAGE or x == y)
        for step in (64, 128, 192, 256, 512):                   # what the fork marks see
            for limit in (len(prompt), len(prompt) // step * step):
                assert min(v, limit) // step == min(exact, limit) // step


def test_only_the_tail_is_readable():
    ids = list(range(1000, 1000 + 2 * PAGE + 31))
    d = sd.DiskIds(ids, chain(ids))
    assert d[2 * PAGE:] == ids[2 * PAGE:] and d[2 * PAGE + 3] == ids[2 * PAGE + 3] and d[-1] == ids[-1]
    assert d[5:5] == [] and d[len(ids):] == []
    for bad in (lambda: list(d), lambda: d[0], lambda: d[PAGE:], lambda: d[:10], lambda: d[2 * PAGE::2]):
        with pytest.raises((TypeError, LookupError)):
            bad()


def test_index_memory_per_token():
    rng = random.Random(4)
    entries = [_toks(rng, 16 * PAGE + 77) for _ in range(40)]
    ix = sd.DiskIndex(1 << 40, lambda mtp: 1000)
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for ids in entries:
        ix.add(tag=0, ids=ids, mtp_len=-1, drafter=False, entry_bytes=10)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    kept = sum(s.size_diff for s in after.compare_to(before, "filename"))
    tokens = sum(len(i) for i in entries)
    assert kept / tokens < 2.0, f"{kept / tokens:.2f} B a token"
