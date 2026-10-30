"""patches/0250 (GLM53_TF_SESSION_DISK=<dir>, ``glm5_next/cuda/sessdisk.py``): the session store's NVMe tier.

Host only (no torch; runnable anywhere):

- settings, defaults and validation; the compat hash moves with every knob that decides a stored bit (KV format,
  weights format, snapshot grid, image, model, cache layout) and not with the ones that cannot (session / batch /
  path knobs);
- the disk index: pages shared by key with reference counts, LRU eviction within the budget (shared pages pinned),
  oversize entries skipped, lookups; replaying the same saves gives the same index (digest), a divergence shows;
- the entry header (its size does not depend on the checksums), the chunk plan covers every byte once;
- the plan message carries the disk entry and the disk index's digest; rank 1 refuses a diverged index.

With torch on the CPU (the real store, tier, writer and reader on fake caches):

- entries round trip bit for bit into zeroed caches (rows and snapshot tensors), shared pages are one file and are
  not read again when the live caches hold them, the write-through happens off the serving thread;
- a corrupted page, a truncated entry, a missing page, a failed write: the read fails, the entry is dropped (files
  removed), the store falls back; both ranks fall back when only one fails (the all-gathered verdict);
- a restart (a new store and tier over the same directory) indexes the entries both ranks hold, removes temporary,
  invalid and orphaned files, trims to the budget oldest first; another compat hash sees nothing;
- GLM53_TF_SESSION_DISK_WRITE=evict writes an entry when the RAM store evicts it (from the slabs);
- the lone engine's real ``_run`` on patches/0080's hostile fake model (exact and fast, serial and MTP-drafted), a
  RAM budget far too small: every reply and the whole state equal a fresh prefill; many resumes come from disk; a
  restart resumes the old conversations from disk; corrupted files fall back to a cold prefill, still exact;
- patches/0180's real ``Batcher`` on its hostile fake model with the tier: every reply and slot state == fresh
  prefill + serial decode, restores from disk into any slot, after a restart too; a follower replaying rank 0's
  messages makes the same decisions (same RAM and disk indexes, same files on its own rank directory).

On the GPU (TensorFold's synthetic checkpoint, one GPU playing rank 0 of two):

- sessions interleaved past a tiny RAM budget are evicted to disk and resume from there: == fresh prefill + serial
  (sampled and greedy); a new engine over the same directory (a restart) resumes them; a corrupted file falls back
  to a cold prefill, == fresh; a second engine replaying rank 0's messages (rank 1) reads its own files and makes
  the same decisions; 0180 batch slots restore from disk; the latent FP8 cache with fast prefill past the dense
  limit; ``sessdisk.bench`` on the real per-rank shapes (prints write / read GB/s; set
  GLM53_TF_SESSION_DISK_BENCH=<dir on the NVMe> to time the drive instead of the test's temporary folder).

Host / CPU: PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda pytest -q tests/cuda/test_session_disk_patches.py
In the image: PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_session_disk_patches.py
"""

from __future__ import annotations

import os
import random
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:          # the host-only tests still run
    torch = None
    CUDA = False

from tensorfold.families.glm5_next.cuda import sessdisk, sessions
from tensorfold.families.glm5_next.cuda.sessdisk import ADDED, PRESENT, SKIPPED, DiskIndex
from tensorfold.families.glm5_next.cuda.sessions import PAGE

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")


def _toks(rng, n):
    return [rng.randrange(1000) for _ in range(n)]


def _clear_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("GLM53_TF_"):
            monkeypatch.delenv(k)


# -- host only ------------------------------------------------------------------------------------------------------
def test_settings(monkeypatch):
    _clear_env(monkeypatch)
    assert sessdisk.root() is None and sessdisk.settings() == [0, 0, 0, 0]          # off unless asked for
    for off in ("", "0", "off", "false"):
        monkeypatch.setenv("GLM53_TF_SESSION_DISK", off)
        assert sessdisk.root() is None
    monkeypatch.setenv("GLM53_TF_SESSION_DISK", "/sessions")
    assert sessdisk.root() == Path("/sessions")
    assert sessdisk.settings() == [1, 64 * 1024, 0, 1024]
    assert sessdisk.policy() == "save" and sessdisk.threads() == 8 and sessdisk.gain_tokens() == 256
    assert sessdisk.queue_bytes() == 2 ** 30 and sessdisk.verify_reads() and sessdisk.direct_io()
    monkeypatch.setenv("GLM53_TF_SESSION_DISK_GIB", "0.5")
    monkeypatch.setenv("GLM53_TF_SESSION_DISK_WRITE", "evict")
    monkeypatch.setenv("GLM53_TF_SESSION_DISK_MIN", "4096")
    assert sessdisk.settings() == [1, 512, 1, 4096]
    for k, v, call in (("GLM53_TF_SESSION_DISK_GIB", "0", sessdisk.settings),
                       ("GLM53_TF_SESSION_DISK_WRITE", "always", sessdisk.settings),
                       ("GLM53_TF_SESSION_DISK_MIN", "0", sessdisk.settings),
                       ("GLM53_TF_SESSION_DISK_VERIFY", "sample", sessdisk.verify_reads),
                       ("GLM53_TF_SESSION_DISK_THREADS", "0", sessdisk.threads)):
        with monkeypatch.context() as m:
            m.setenv(k, v)
            with pytest.raises(ValueError):
                call()
    # the RAM store's own settings (and their test) are unchanged
    monkeypatch.setenv("GLM53_TF_SESSION_GIB", "12")
    monkeypatch.setenv("GLM53_TF_SESSION_EVERY", "0")
    assert sessions.settings() == [12 * 1024, 0, 512]


def test_compat_covers_what_decides_the_bits(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("GLM53_TF_IMAGE_ID", "sha256:img1")
    lay = [[256, 0, "torch.uint8", [528]], [64, 0, "torch.bfloat16", [64]]]
    h = lambda **kw: sessdisk.compat_hash(sessdisk.compat_ident(layout=kw.pop("layout", lay), **kw))  # noqa: E731
    h0 = h()
    changes = [("GLM53_TF_KV_DTYPE", "fp8"), ("GLM53_TF_NONEXPERT", "q4mse"), ("GLM53_TF_LATENT_TC", "1"),
               ("GLM53_TF_SNAPSHOT_GRID", "128"), ("GLM53_TF_KDA_PROJ_BF16", "1"), ("GLM53_TF_IMAGE_ID", "sha256:img2"),
               ("GLM53_TF_INDEX_RING", "0"), ("GLM53_TF_A_FUTURE_KNOB", "1")]
    for k, v in changes:
        with monkeypatch.context() as m:
            m.setenv(k, v)
            assert h() != h0, k
    same = [("GLM53_TF_SESSION_GIB", "4"), ("GLM53_TF_SESSION_DISK", "/x"), ("GLM53_TF_SESSION_DISK_GIB", "9"),
            ("GLM53_TF_BATCH", "4"), ("GLM53_TF_BATCH_SESSIONS", "1"), ("GLM53_TF_PREFILL_ROWS", "auto"),
            ("GLM53_TF_TOKCACHE", "0"), ("GLM53_TF_LAUNCH_T0", "123.4"), ("GLM53_TF_CALIB", "cached"),
            ("GLM53_TF_PREPARED", "/prepared"), ("GLM53_TF_HEALTH", "strict"), ("GLM53_TF_STALL_S", "30")]
    for k, v in same:
        with monkeypatch.context() as m:
            m.setenv(k, v)
            assert h() == h0, k
    assert h(layout=lay[:1]) != h0
    with monkeypatch.context() as m:
        m.setattr(sessions, "KV_TAG", b"kv:fp8")
        assert h() != h0
    model = tmp_path / "models--org--m" / "snapshots" / "abcdef0123456789"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    hm = h(model_dir=model)
    assert hm != h0
    (model / "config.json").write_text('{"x": 1}')
    assert h(model_dir=model) != hm
    assert sessdisk.ints_key(sessdisk.key_ints(b"0123456789abcdef")) == b"0123456789abcdef"


PB = 1000


def _ix(budget_pages: float = 100) -> DiskIndex:
    return DiskIndex(int(budget_pages * PB), lambda m: PB if m else PB // 2)


def _add(ix, ids, *, eb=500, tag=0, mtp=True, drafter=False):
    return ix.add(tag=tag, ids=ids, mtp_len=len(ids) - 1 if mtp else -1, drafter=drafter, entry_bytes=eb)


def test_disk_index_shares_pages_and_evicts_lru():
    rng = random.Random(1)
    system = _toks(rng, 700)
    a, b, c = system + _toks(rng, 600), system + _toks(rng, 700), _toks(rng, 1300)
    ix = _ix()
    ra = _add(ix, a)
    assert ra.status == ADDED and ra.new == list(range(5)) and len(ra.entry.pages) == 5
    rb = _add(ix, b)
    assert rb.status == ADDED and rb.new == [2, 3, 4]                # pages 0, 1 (tokens < 513) are the system's
    assert rb.entry.pages[:2] == ra.entry.pages[:2] and ix.pages[ra.entry.pages[0]][0] == 2
    assert ix.used == 2 * 500 + 8 * PB and len(ix.pages) == 8
    used_b = rb.entry.used
    again = _add(ix, a)
    assert again.status == PRESENT and again.entry is ra.entry and ra.entry.used > used_b
    assert ix.find(a + [1, 2], 0, True, False) is ra.entry and ix.find(a, 0, True, False) is None
    assert ix.find(a + [1], 0, True, True) is None and ix.find(a + [1], 64, True, False) is None
    assert ix.find(b[:900] + [1], 0, True, False) is None           # only whole entries resume
    # patches/0670: inside a stored full page the disk index's common prefix is exact to SUB (64) tokens
    assert ix.has(0, b, True, False) is rb.entry and ix.lcp(system + [5] * 400) == 700 // 64 * 64
    # room for one more 5-page entry only by evicting: the least recently used (b; a was touched after it)
    ix.budget = ix.used + 3 * PB
    rc = _add(ix, c)
    assert rc.status == ADDED and [e.id for e in rc.evicted] == [rb.entry.id]
    assert sorted(rc.freed) == sorted(rb.entry.pages[2:])            # b's own pages; the shared ones stay (a)
    assert ix.pages[ra.entry.pages[0]][0] == 1 and ix.used <= ix.budget
    # larger than the whole budget: skipped, nothing evicted
    big = _add(ix, _toks(rng, 200), eb=ix.budget + 1)
    assert big.status == SKIPPED and not big.evicted and len(ix.entries) == 2
    # a new entry sharing a's pages pins them: evicting a (the LRU) keeps them
    ix.budget = ix.used + PB
    rd = _add(ix, a + _toks(rng, 300))
    assert rd.status == ADDED and [e.id for e in rd.evicted] == [ra.entry.id] and rd.new == [5] and not rd.freed
    assert all(k in ix.pages for k in rd.entry.pages) and ix.used <= ix.budget
    # MTP-less entries use the smaller page files, and their pages are other keys
    re_ = _add(ix, a + [9], mtp=False, eb=10)
    assert re_.status == ADDED and not set(re_.entry.pages) & set(rd.entry.pages)


def test_disk_index_replay_is_deterministic():
    rng = random.Random(4)
    systems = [_toks(rng, n) for n in (300, 800, 1500)]
    one, two = _ix(60), _ix(60)
    for step in range(300):
        if one.entries and rng.random() < 0.1:
            victim = rng.choice(sorted(one.entries))
            one.evict(victim, "dropped")
            two.evict(victim, "dropped")
            continue
        ids = rng.choice(systems) + _toks(rng, rng.randrange(1, 900))
        kw = dict(eb=rng.randrange(100, 3000), mtp=rng.random() < 0.8, tag=rng.choice([0, 64]))
        x, y = _add(one, ids, **kw), _add(two, ids, **kw)
        assert (x.status, x.new, [e.id for e in x.evicted]) == (y.status, y.new, [e.id for e in y.evicted])
        assert one.digest() == two.digest() and one.used <= one.budget
    assert one.counters == two.counters and one.counters["evicted"] > 20
    two.evict(next(iter(two.entries)))
    assert one.digest() != two.digest()


def test_header_and_chunk_plan():
    ids = list(range(3, 5000))
    meta = {"format": 1, "key": "ab" * 16, "ids": sessdisk.ids_b64(ids), "sums": [sessdisk._ZERO_SHA] * 3,
            "segments": [{"name": "rec", "off": 0, "nbytes": 10}]}
    h = sessdisk.place_header(meta)
    assert h % sessdisk.ALIGN == 0 and meta["data_off"] == h and len(sessdisk.encode_header(meta)) <= h
    filled = dict(meta, sums=["f" * 64] * 3)
    assert sessdisk.place_header(filled) == h                     # checksums filled in later: same header size
    assert sessdisk.b64_ids(meta["ids"]) == ids
    rng = random.Random(2)
    for _ in range(200):
        segs, off = [], 0
        for _ in range(rng.randrange(1, 6)):
            n = rng.randrange(0, 30000)
            segs.append((off, n))
            off += -(-n // 4096) * 4096
        size = off
        chunk = rng.choice([4096, 8192, 65536])
        plan = sessdisk.chunk_plan(segs, size, chunk)
        cover = [None] * size
        for k, pieces in enumerate(plan):
            want = min(chunk, size - k * chunk)
            assert sum(n for _, _, _, n in pieces) == want
            for seg, a, s, n in pieces:
                for j in range(n):
                    pos = k * chunk + s + j
                    assert cover[pos] is None
                    cover[pos] = (seg, a + j)
        for i, (o, n) in enumerate(segs):
            assert cover[o:o + n] == [(i, j) for j in range(n)]
        assert all(c is not None for c in cover)


def test_plan_message_carries_the_disk_entry():
    rng = random.Random(8)
    store = object.__new__(sessions.SessionStore)
    store.index = sessions.SessionIndex(10 ** 9, (1024, 64, 512, 128), extent=4)
    dix = _ix()
    d = _add(dix, _toks(rng, 900)).entry
    store.disk = SimpleNamespace(index=dix, digest=dix.digest)
    msg = sessions.Plan(None, [256], d, dix.digest()).encode(store.index.digest())
    plan = store.follow(msg)
    assert plan.entry is None and plan.disk is d and plan.marks == [256] and plan.disk_digest == dix.digest()
    plan = store.follow(sessions.Plan(None, [], None, dix.digest()).encode(store.index.digest()))
    assert plan.disk is None
    with pytest.raises(RuntimeError, match="disk indexes diverged"):
        store.follow(sessions.Plan(None, [], d, dix.digest() ^ 1).encode(store.index.digest()))
    ghost = SimpleNamespace(id=99, ids=[1] * 900)
    with pytest.raises(RuntimeError, match="no session disk entry"):
        store.follow(sessions.Plan(None, [], ghost, dix.digest()).encode(store.index.digest()))


# -- torch on the CPU: the tier on fake caches --------------------------------------------------------------------------
def _fake_store(tmp, *, budget_pages=1000.0, write="save", min_tok=64, rank=0, ident=None, disk_budget=1 << 40,
                agree=None, queue=1 << 30):
    from test_session_patches import _FakeState

    st = _FakeState(torch)
    e = SimpleNamespace(st=st, w=SimpleNamespace(mtp=object()))
    store = sessions.SessionStore(e, None, 0, every_tokens=0, fork_tokens=64)
    store.index.budget = int(budget_pages * store.index.page_bytes)
    tier = sessdisk.DiskTier(store, tmp, rank, ident or {"test": "fake"}, budget=disk_budget, write=write,
                             min_tok=min_tok, gain=0, n_threads=3, queue=queue, quiet=True, agree=agree)
    tier.reconcile()
    store.attach_disk(tier)
    return store, st


def _snap(ids):
    n = float(len(ids))
    return SimpleNamespace(ids=list(ids), rec=torch.full((5,), n), conv=torch.arange(3.0) + n,
                           pending=torch.full((1, 2), -n), mtp_len=len(ids) - 1, drafter_end=-1, grid=0, window=None,
                           kv=0)


def _prefill_and_save(store, st, ids):
    store.begin(None, 0)
    st.write(ids, 0, len(ids), 0, len(ids) - 1, torch, junk=5)
    snap = _snap(ids)
    store.save(snap)
    return snap


def _restore(store, entry):
    for x in [*store.e.st.kc, store.e.st.mtp_kc, *[t for trip in store.e.st.index for t in trip]]:
        x.fill_(-1.0)
    store.live = {}
    return store.restore_disk(sessions.Plan(None, [], entry, store.disk.digest()))


def _check_snap(snap, ids):
    n = float(len(ids))
    assert snap.ids == list(ids) and snap.mtp_len == len(ids) - 1 and snap.grid == 0
    assert torch.equal(snap.rec, torch.full((5,), n)) and torch.equal(snap.conv, torch.arange(3.0) + n)
    assert torch.equal(snap.pending, torch.full((1, 2), -n))


def _damage(tier) -> int:
    """Flip a byte in every page file and in every entry's data region."""

    n = 0
    for p in (tier.dir / "pages").rglob("*.tfp"):
        raw = bytearray(p.read_bytes())
        raw[len(raw) // 3] ^= 0x5A
        p.write_bytes(bytes(raw))
        n += 1
    for p in (tier.dir / "entries").glob("*.tfs"):
        off = sessdisk.read_header(p)["data_off"]
        raw = bytearray(p.read_bytes())
        raw[off + 3] ^= 0x5A
        p.write_bytes(bytes(raw))
        n += 1
    return n


def _files(tier):
    return (sorted(p.name for p in (tier.dir / "entries").glob("*.tfs")),
            sorted(p.name for p in (tier.dir / "pages").rglob("*.tfp")))


@needs_torch
def test_round_trip_shared_pages_and_skips(tmp_path):
    store, st = _fake_store(tmp_path)
    tier = store.disk
    rng = random.Random(3)
    system = _toks(rng, 700)
    a, b = system + _toks(rng, 600), system + _toks(rng, 700)
    for ids in (a, b, _toks(rng, 40)):                      # the last is below GLM53_TF_SESSION_DISK_MIN
        _prefill_and_save(store, st, ids)
    tier.writer.drain()
    ix = tier.index
    assert len(ix.entries) == 2 and tier.stats["writes"] == 2
    entries, pages = _files(tier)
    assert len(entries) == 2 and len(pages) == len(ix.pages) == 8 < sum(len(e.pages) for e in ix.entries.values())
    ea, eb = ix.has(0, a, True, False), ix.has(0, b, True, False)
    snap = _restore(store, ea)
    _check_snap(snap, a)
    st.check(a, len(a), len(a) - 1)
    assert tier.last["pages_read"] == 5 and tier.last["pages_skipped"] == 0 and tier.last["ok"]
    assert store.live == dict(enumerate(ea.pages))
    # b while the live caches hold a: the system prompt's two pages are not read again
    snap = store.restore_disk(sessions.Plan(None, [], eb, tier.digest()))
    _check_snap(snap, b)
    st.check(b, len(b), len(b) - 1)
    assert tier.last["pages_read"] == 3 and tier.last["pages_skipped"] == 2
    assert store.stats["disk_restores"] == 2


@needs_torch
@pytest.mark.parametrize("damage", ["page", "entry_chunk", "truncated", "missing_page", "write_failed"])
def test_a_bad_file_fails_the_read_and_drops_the_entry(tmp_path, damage):
    store, st = _fake_store(tmp_path)
    tier = store.disk
    rng = random.Random(5)
    system = _toks(rng, 700)
    a, b = system + _toks(rng, 600), system + _toks(rng, 700)
    for ids in (a, b):
        _prefill_and_save(store, st, ids)
    tier.writer.drain()
    ix = tier.index
    eb = ix.has(0, b, True, False)
    own = tier._page_path(eb.pages[3])
    path = tier._entry_path(eb.key)
    if damage == "page":
        raw = bytearray(own.read_bytes())
        raw[1234] ^= 1
        own.write_bytes(bytes(raw))
    elif damage == "entry_chunk":
        with open(path, "r+b") as f:
            f.seek(sessdisk.read_header(path)["data_off"] + 5)
            x = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([x[0] ^ 0x40]))
    elif damage == "truncated":
        os.truncate(path, path.stat().st_size - 4096)
    elif damage == "missing_page":
        own.unlink()
    else:
        tier.bad.add(eb.pages[4])
    assert _restore(store, eb) is None
    assert eb.id not in ix.entries and store.live == {} and store.stats["disk_fallbacks"] == 1
    assert tier.stats["failed_loads"] == 1 and "error" in tier.last
    tier.writer.drain()
    assert not path.exists() and not tier._page_path(eb.pages[3]).exists()
    assert tier._page_path(eb.pages[0]).exists()                # a's pages stay
    ea = ix.has(0, a, True, False)
    _check_snap(_restore(store, ea), a)
    st.check(a, len(a), len(a) - 1)
    # the session can be written again (new files) and read back
    _prefill_and_save(store, st, b)
    tier.writer.drain()
    _check_snap(_restore(store, ix.has(0, b, True, False)), b)
    st.check(b, len(b), len(b) - 1)


@needs_torch
def test_both_ranks_fall_back_when_one_fails(tmp_path):
    flags: list = [None, None]
    barrier = threading.Barrier(2)

    def agree_for(r):
        def agree(ok):
            flags[r] = bool(ok)
            barrier.wait()
            both = all(flags)
            barrier.wait()
            return both
        return agree

    stores = [_fake_store(tmp_path, rank=r, agree=agree_for(r)) for r in (0, 1)]
    rng = random.Random(6)
    prompts = [_toks(rng, 900), _toks(rng, 1100)]
    for store, st in stores:
        for ids in prompts:
            _prefill_and_save(store, st, ids)
        store.disk.writer.drain()
    assert stores[0][0].disk.digest() == stores[1][0].disk.digest()
    assert stores[0][0].disk.dir != stores[1][0].disk.dir                     # each rank its own files
    results: dict = {}

    def run(r, ids):
        store = stores[r][0]
        results[r] = _restore(store, store.disk.index.has(0, ids, True, False))

    for i, ids in enumerate(prompts):
        if i == 1:                                                              # rank 1's copy of the 2nd is bad
            t1 = stores[1][0].disk
            p = t1._page_path(t1.index.has(0, ids, True, False).pages[2])
            p.write_bytes(bytes(p.stat().st_size))
        th = [threading.Thread(target=run, args=(r, ids)) for r in (0, 1)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        if i == 0:
            assert results[0] is not None and results[1] is not None
            _check_snap(results[0], ids)
        else:
            assert results[0] is None and results[1] is None                    # rank 0's read was fine
    assert stores[0][0].disk.digest() == stores[1][0].disk.digest() and len(stores[0][0].disk.index.entries) == 1


@needs_torch
def test_restart_indexes_the_directory(tmp_path):
    store, st = _fake_store(tmp_path)
    rng = random.Random(7)
    system = _toks(rng, 600)
    prompts = [system + _toks(rng, n) for n in (500, 700, 900)] + [_toks(rng, 800)]
    for ids in prompts:
        _prefill_and_save(store, st, ids)
    tier = store.disk
    tier.writer.drain()
    keys = sorted(e.key for e in tier.index.entries.values())
    # debris a crash leaves: a temporary file, a garbage entry, an orphaned page
    (tier.dir / "entries" / f".{'0' * 32}.tfs.tmp-1-2").write_bytes(b"x" * 100)
    (tier.dir / "entries" / f"{'1' * 32}.tfs").write_bytes(b"not an entry")
    orphan = tier._page_path(b"\x07" * 16)
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(bytes(tier.lay[True][1]))
    store2, st2 = _fake_store(tmp_path)
    t2 = store2.disk
    t2.writer.drain()
    assert sorted(e.key for e in t2.index.entries.values()) == keys and t2.dir == tier.dir
    entries, pages = _files(t2)
    assert len(entries) == 4 and len(pages) == len(t2.index.pages) and not orphan.exists()
    assert not list((t2.dir / "entries").glob(".*"))
    for ids in prompts:
        _check_snap(_restore(store2, t2.index.has(0, ids, True, False)), ids)
        st2.check(ids, len(ids), len(ids) - 1)
    # another engine (compat) sees none of it
    store3, _ = _fake_store(tmp_path, ident={"test": "other"})
    assert not store3.disk.index.entries and store3.disk.dir != tier.dir
    # a smaller budget at the next start: the oldest go first
    newest = t2.index.has(0, prompts[-1], True, False)
    store4, _ = _fake_store(tmp_path, disk_budget=newest.nbytes + len(newest.pages) * t2.lay[True][1])
    t4 = store4.disk
    t4.writer.drain()
    assert [e.key for e in t4.index.entries.values()] == [newest.key]
    assert len(_files(t4)[0]) == 1 and len(_files(t4)[1]) == len(newest.pages)


@needs_torch
def test_restart_keeps_only_what_both_ranks_hold(tmp_path):
    rng = random.Random(8)
    prompts = [_toks(rng, 700), _toks(rng, 900), _toks(rng, 1100)]
    for r in (0, 1):
        store, st = _fake_store(tmp_path, rank=r)
        for ids in prompts[:3 if r == 0 else 2]:                        # rank 1 lost the last write
            _prefill_and_save(store, st, ids)
        store.disk.writer.drain()
    sent = []
    gathered = {}

    def share0(values):
        sent.append(list(values))
        return list(values)

    def share1(values):
        assert values is None
        return sent[0]

    tiers = []
    for r, share in ((0, share0), (1, share1)):
        store, _ = _fake_store(tmp_path / "x", rank=r)                  # built empty, then pointed at the real dir
        t = sessdisk.DiskTier(store, tmp_path, r, {"test": "fake"}, budget=1 << 40, min_tok=64, gain=0, quiet=True)
        tiers.append(t)
    have = {}
    for r, t in enumerate(tiers):                                        # each rank's answer, then both together
        cands = t.scan()
        have[r] = {k for _, k, _ in cands}
    order = [k for _, k, _ in tiers[0].scan()]
    both = [[1] * len(order), [int(k in have[1]) for k in order]]
    for r, (t, share) in enumerate(zip(tiers, (share0, share1))):
        gathered[r] = t.reconcile(share, lambda v: both)
    assert gathered == {0: 2, 1: 2} and tiers[0].digest() == tiers[1].digest()
    for t in tiers:
        t.writer.drain()
    assert len(_files(tiers[0])[0]) == 2                                  # rank 0's third entry is gone too


@needs_torch
def test_evict_policy_writes_what_the_ram_store_evicts(tmp_path):
    store, st = _fake_store(tmp_path, budget_pages=17.0, write="evict")     # one slab (16 pages) and the snapshots
    tier = store.disk
    rng = random.Random(9)
    prompts = [_toks(rng, 1900), _toks(rng, 1850), _toks(rng, 1800)]         # 7 pages each: the third evicts
    _prefill_and_save(store, st, prompts[0])
    tier.writer.drain()
    assert not tier.index.entries                                        # nothing written while it is in RAM
    for ids in prompts[1:]:
        _prefill_and_save(store, st, ids)
    tier.writer.drain()
    evicted = store.index.counters["evicted"]
    assert evicted == 1 and len(tier.index.entries) == evicted
    assert store.index.has(0, prompts[0], True, False) is None
    e0 = tier.index.has(0, prompts[0], True, False)
    _check_snap(_restore(store, e0), prompts[0])
    st.check(prompts[0], len(prompts[0]), len(prompts[0]) - 1)


@needs_torch
def test_an_entry_larger_than_the_queue_is_written_in_place(tmp_path):
    store, st = _fake_store(tmp_path, queue=1)
    rng = random.Random(10)
    ids = _toks(rng, 1500)
    _prefill_and_save(store, st, ids)
    assert store.disk.stats["writes"] == 1                              # before save() returned
    _check_snap(_restore(store, store.disk.index.has(0, ids, True, False)), ids)
    st.check(ids, len(ids), len(ids) - 1)


# -- torch on the CPU: the lone engine's real request path on the hostile fake model ---------------------------------
def _lone(monkeypatch, tmp, rows: int, fast: bool, budget_pages: float):
    from test_session_patches import _fake_session_engine

    g = _fake_session_engine(monkeypatch, rows, fast, budget_pages)
    tier = sessdisk.DiskTier(g.store, tmp, 0, {"lone": rows, "fast": fast}, budget=1 << 40, min_tok=64, gain=0,
                             n_threads=3, quiet=True)
    tier.reconcile()
    g.store.attach_disk(tier)
    return g


def _lone_turn(g, prompt, policy: str, tokens: int = 30):
    """What ``GlmEngine.generate`` does on rank 0 (plan: RAM, then disk, the read starting at once), then ``_run``."""

    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    code = encode_policy(policy)
    hit = g._resume(list(prompt), code)
    _, need_mtp, _ = g._drafters(code)
    sess = g.store.plan(list(prompt), g._grid(), need_mtp, False, len(hit.ids) if hit else 0, True)
    if sess.entry is not None:
        hit = sess.entry.payload[0]
    elif sess.disk is not None:
        hit = None
        g.cache = []
        g.store.prefetch(sess)
    out: list[int] = []
    stats = g._run(list(prompt), tokens, None, False, out.extend, code, hit, True, sess)
    st, f = g.e.st, g.fake
    state = (st.pos, int(st.rec[st.cur[0], 0]), st.conv.tolist(), f.kv[:st.pos].tolist(), st.mtp_len,
             f.mkv[:st.mtp_len].tolist())
    return out, stats, state


@needs_torch
@pytest.mark.parametrize("policy", ["0", "2"])
@pytest.mark.parametrize("rows", [64, 200])
def test_lone_engine_disk_sessions_on_a_hostile_fake_model(monkeypatch, tmp_path, rows, policy):
    """Conversations over three system prompts, interleaved at random, with a RAM store of 20 pages: exact and fast,
    every reply and the whole state == a fresh prefill; resumes come from disk. Then a restart (a new engine over
    the same directory) resumes the old conversations from disk; then corrupted files fall back to cold prefills."""

    np = pytest.importorskip("numpy")
    pytest.importorskip("triton")
    from tensorfold.families.glm5_next.cuda import decode, fastpf
    from test_fastpf_patches import _fake_engine
    from test_session_patches import _fake_request_ref

    def use(eng):
        for name in ("stage", "compute", "commit"):
            monkeypatch.setattr(decode, name, getattr(eng.fake, name))

    rng = np.random.default_rng(rows + len(policy))
    more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]     # noqa: E731
    C = fastpf.grid(rows)
    for fast in (False, True):
        d = tmp_path / ("fast" if fast else "exact")
        g = _lone(monkeypatch, d, rows, fast, 20.0)
        ref = _fake_engine(monkeypatch, rows, fast)
        systems = [more(int(n)) for n in (300, 700, 1100)]
        convs: list[tuple[list[int], list[int]]] = []

        def turn(eng, prompt):
            use(eng)
            got, stats, state = _lone_turn(eng, prompt, policy)
            use(ref)
            ref.cache = []
            want, c0, want_state = _fake_request_ref(ref, prompt, policy)
            assert c0 == 0 and got == want and state == want_state, (fast, len(prompt), stats)
            if fast:
                assert stats["cached"] % C == 0
            return got, stats

        for _ in range(30):
            k = int(rng.integers(0, 6))
            if convs and k < 3:
                i = int(rng.integers(0, len(convs)))
                last, reply = convs[i]
                prompt = (last + reply if k < 2 else last) + more(int(rng.choice([1, 5, 40, C + 3])))
            else:
                prompt = systems[int(rng.integers(0, 3))] + more(int(rng.integers(1, 300)))
                convs.append(([], []))
                i = len(convs) - 1
            prompt = prompt[:2600]
            got, _ = turn(g, prompt)
            convs[i] = (prompt, got)
        restored = g.store.stats.get("disk_restores", 0)
        assert restored >= 4, restored
        assert g.store.index.used <= g.store.index.budget and g.store.index.counters["evicted"] > 0
        g.store.disk.writer.drain()

        # a restart: a new engine (fresh caches, empty RAM store) over the same directory
        g2 = _lone(monkeypatch, d, rows, fast, 20.0)
        assert len(g2.store.disk.index.entries) == len(g.store.disk.index.entries) > 0
        for i, (last, reply) in enumerate(convs[:4]):
            prompt = last + reply + more(3)
            got, stats = turn(g2, prompt)
            if i == 0:
                assert stats["cached"] >= len(last) - C and "disk" in stats and stats["disk"]["ok"], stats
            convs[i] = (prompt, got)
        assert g2.store.stats.get("disk_restores", 0) >= 1
        g2.store.disk.writer.drain()

        # another restart, with every page file damaged: cold prefills, still exact; the entries go
        g3 = _lone(monkeypatch, d, rows, fast, 20.0)
        _damage(g3.store.disk)
        last, reply = convs[0]
        got, stats = turn(g3, last + reply + more(2))
        assert stats["cached"] == 0 and not stats["disk"]["ok"] and g3.store.stats["disk_fallbacks"] == 1
        assert stats["disk"]["entry"] not in g3.store.disk.index.entries             # dropped (then re-saved)


# -- torch on the CPU: patches/0180's real Batcher with the tier ------------------------------------------------------
def _batcher(monkeypatch, tmp, *, rank: int = 0, fast: bool = False, budget_pages: float = 20.0):
    from test_batch_sessions_patches import _fake_batcher

    bat = _fake_batcher(monkeypatch, n=3, rows=64, fast=fast, piece=256, budget_pages=budget_pages, rank=rank)
    tier = sessdisk.DiskTier(bat.store, tmp, rank, {"batch": 64, "fast": fast}, budget=1 << 40, min_tok=64, gain=0,
                             n_threads=3, quiet=True)
    tier.reconcile()
    bat.store.attach_disk(tier)
    return bat


@needs_torch
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_batched_disk_sessions_on_a_hostile_fake_model(monkeypatch, tmp_path, fast):
    """4 sessions over a shared system prompt on 3 slots with a 20-page RAM store: every reply and slot state ==
    fresh prefill + serial decode; admissions restore from disk into whichever slot is free; after a restart the
    same sessions resume from disk."""

    import numpy as np
    from test_batch_sessions_patches import _conversations

    bat = _batcher(monkeypatch, tmp_path, fast=fast, budget_pages=20.0)
    stats = _conversations(bat, rng=np.random.default_rng(11 + int(fast)), fast=fast, rows=64, turns=10)
    from_disk = [s for s in stats if s.get("restored_disk")]
    assert len(from_disk) >= 4, [(s.get("cached"), s.get("restored"), s.get("restored_disk")) for s in stats]
    assert len({s["slot"] for s in from_disk}) >= 2 and all(s["disk"]["ok"] for s in from_disk)
    assert bat.store.index.counters["evicted"] > 0
    bat.store.disk.writer.drain()
    again = _batcher(monkeypatch, tmp_path, fast=fast, budget_pages=20.0)
    assert again.store.disk.index.entries
    stats = _conversations(again, rng=np.random.default_rng(11 + int(fast)), fast=fast, rows=64, turns=3)
    assert any(s.get("restored_disk") for s in stats[:4]), [(s.get("cached"), s.get("restored_disk"))
                                                            for s in stats]


@needs_torch
def test_batched_follower_replays_disk_plans(monkeypatch, tmp_path):
    """Rank 0's messages replayed through ``follow`` on a second batcher (rank 1, its own directory): the same
    requests end the same way, both RAM and disk indexes end identical, each rank wrote the same entries."""

    import numpy as np
    from test_batch_sessions_patches import _conversations

    r0 = _batcher(monkeypatch, tmp_path, rank=0, budget_pages=20.0)
    r1 = _batcher(monkeypatch, tmp_path, rank=1, budget_pages=20.0)
    sent: list[list[int]] = []

    def record(values):
        sent.append([int(v) for v in values])
        return list(values)

    class Done(Exception):
        pass

    def replay(values):
        assert values is None
        if not sent:
            raise Done
        return sent.pop(0)

    r0.g._share, r1.g._share = record, replay
    _conversations(r0, rng=np.random.default_rng(3), fast=False, rows=64, turns=8, check=False)
    with pytest.raises(Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log] and len(r0.log) == 32
    a, b = r0.store, r1.store
    assert a.index.digest() == b.index.digest() and a.disk.digest() == b.disk.digest()
    assert a.stats == b.stats and a.stats.get("disk_restores", 0) > 0
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]
    a.disk.writer.drain()
    b.disk.writer.drain()
    assert _files(a.disk)[0] == _files(b.disk)[0] and a.disk.dir != b.disk.dir


# -- GPU ------------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")


def _disk_env(m, root: Path, *, gib: float = 4.0, write: str = "save") -> None:
    m.setenv("GLM53_TF_SESSION_DISK", str(root))
    m.setenv("GLM53_TF_SESSION_DISK_GIB", str(gib))
    m.setenv("GLM53_TF_SESSION_DISK_WRITE", write)
    m.setenv("GLM53_TF_SESSION_DISK_MIN", "64")
    m.setenv("GLM53_TF_SESSION_DISK_GAIN", "0")
    m.setenv("GLM53_TF_SESSION_DISK_THREADS", "4")


def _engine(path, root, *, gib: float = 1.0, fast: bool = False, rows: int = 64, context: int = 0,
            latent_kv: bool = False, kv: str = "bf16", fork: int = 256, every: int = 0, nonexpert: str | None = None,
            write: str = "save", disk: bool = True):
    """test_session_patches' lone engine, with the disk tier at ``root``."""

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from test_glm_engine import _TwoCopies

    with pytest.MonkeyPatch.context() as m:
        if nonexpert:
            m.setattr(weights, "NONEXPERT", nonexpert)
            m.setenv("GLM53_TF_NONEXPERT", nonexpert)
        m.setenv("GLM53_TF_SESSION_GIB", str(gib))
        m.setenv("GLM53_TF_SESSION_EVERY", str(every))
        m.setenv("GLM53_TF_SESSION_FORK_MIN", str(fork))
        m.setenv("GLM53_TF_SESSION_RESERVE_GIB", "0")
        m.setenv("GLM53_TF_FAST_PREFILL", "1" if fast else "0")
        m.setenv("GLM53_TF_PREFILL_ROWS", str(rows))
        m.setenv("GLM53_TF_PREFILL_ROWS_MAX", str(max(rows, 256)))
        m.setenv("GLM53_TF_SNAPSHOT_GRID", str(max(64, rows // 64 * 64)))
        m.setenv("GLM53_TF_LATENT_KV", "1" if latent_kv else "0")
        m.setenv("GLM53_TF_KV_DTYPE", kv)
        m.setenv("GLM53_TF_LOOKUP", "0")
        for k in ("GLM53_TF_BATCH", "GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH",
                  "GLM53_TF_FAST_GATHER", "GLM53_TF_FP8_PREFILL", "GLM53_TF_LEAN_PREFILL", "GLM53_TF_SESSION_DISK"):
            m.delenv(k, raising=False)
        if disk:
            _disk_env(m, root, write=write)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", context=context,
                         comm=_TwoCopies())


def _batch_engine(path, root, *, batch: int = 3, gib: float = 1.0):
    """test_batch_sessions_patches' batch engine (0180 on), with the disk tier at ``root``."""

    from tensorfold.families.glm5_next.cuda import weights
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from test_glm_engine import _TwoCopies

    with pytest.MonkeyPatch.context() as m:
        m.setattr(weights, "NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_NONEXPERT", "q4mse")
        m.setenv("GLM53_TF_BATCH", str(batch))
        m.setenv("GLM53_TF_BATCH_SESSIONS", "1")
        m.setenv("GLM53_TF_SESSION_GIB", str(gib))
        m.setenv("GLM53_TF_SESSION_EVERY", "0")
        m.setenv("GLM53_TF_SESSION_FORK_MIN", "256")
        m.setenv("GLM53_TF_SESSION_RESERVE_GIB", "0")
        m.setenv("GLM53_TF_PREFILL_ROWS", "64")
        m.setenv("GLM53_TF_PREFILL_ROWS_MAX", "64")
        m.setenv("GLM53_TF_FAST_PREFILL", "0")
        m.setenv("GLM53_TF_BATCH_PREFILL_SHARE", "1.0")
        m.setenv("GLM53_TF_BATCH_RESERVE_GB", "0.25")
        m.setenv("GLM53_TF_BATCH_ADMIT_GB", "0")
        for k in ("GLM53_TF_CALIB_ONLINE", "GLM53_TF_PROFILE", "GLM53_TF_DEPTH", "GLM53_TF_FAST_GATHER",
                  "GLM53_TF_BATCH_GRAPH_ROWS", "GLM53_TF_BATCH_MAX_GRAPHS", "GLM53_TF_LATENT_KV", "GLM53_TF_BATCH_PIECE",
                  "GLM53_TF_LEAN_PREFILL", "GLM53_TF_FP8_PREFILL", "GLM53_TF_SNAPSHOT_GRID"):
            m.delenv(k, raising=False)
        _disk_env(m, root)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_session_disk")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def ref(ckpt, tmp_path_factory):
    return _engine(ckpt, tmp_path_factory.mktemp("unused"), gib=0, disk=False)


def _tiny_ram(e) -> None:
    """A RAM budget of about one entry's snapshots and one slab of pages: most entries are evicted."""

    ix = e.store.index
    ix.budget = max((x.private for x in ix.entries.values()), default=ix.extent_bytes) + ix.extent_bytes




@gpu
@pytest.mark.parametrize("sampling", ["sampled", "greedy"])
def test_gpu_evicted_sessions_resume_from_disk(ckpt, ref, tmp_path, sampling):
    from test_session_patches import _gen, _interleave, _sampling, _sessions

    s = _sampling(sampling)
    e = _engine(ckpt, tmp_path)
    _gen(e, _sessions(71)[0], s)
    _tiny_ram(e)
    runs = _interleave(e, ref, s, _sessions(72 if sampling == "sampled" else 73))
    assert e.store.index.counters["evicted"] > 0
    assert e.store.stats.get("disk_restores", 0) >= 1, [x[3].get("disk") for x in runs]
    from_disk = [st for *_, st in runs if "disk" in st]
    assert all(st["disk"]["ok"] and st["cached"] > 0 for st in from_disk)
    print("disk reads:", [st["disk"] for st in from_disk])


@gpu
def test_gpu_restart_resumes_from_disk_and_damage_falls_back(ckpt, ref, tmp_path):
    from test_session_patches import _fresh, _gen, _sampling, _sessions

    s = _sampling("sampled")
    a, b, _ = _sessions(81)
    first = _engine(ckpt, tmp_path)
    ra, _ = _gen(first, a, s)
    rb, _ = _gen(first, b, s)
    first.store.disk.writer.drain()
    del first
    torch.cuda.empty_cache()
    again = _engine(ckpt, tmp_path)                           # a restart: a new engine over the same directory
    assert len(again.store.disk.index.entries) >= 2 and not again.store.index.entries
    after = a + ra + [4, 5]
    reply, stats = _gen(again, after, s)
    assert stats["cached"] >= len(a) + len(ra) - 1 and stats["disk"]["ok"], stats
    assert reply == _fresh(ref, after, s)
    again.store.disk.writer.drain()
    del again
    torch.cuda.empty_cache()
    third = _engine(ckpt, tmp_path)
    assert _damage(third.store.disk) > 0
    after_b = b + rb + [6]
    reply, stats = _gen(third, after_b, s)
    assert stats["cached"] == 0 and not stats["disk"]["ok"] and third.store.stats["disk_fallbacks"] == 1, stats
    assert reply == _fresh(ref, after_b, s)


class _Done(Exception):
    pass


@gpu
def test_gpu_rank1_follows_disk_plans(ckpt, tmp_path):
    """Rank 0's messages replayed to a second engine (rank 1, its own directory): the same disk restores, the same
    replies, the same RAM and disk indexes."""

    from test_session_patches import _gen, _sampling, _sessions

    s = _sampling("sampled")
    lead = _engine(ckpt, tmp_path / "r0")
    follower = _engine(ckpt, tmp_path / "r1")
    sent: list[list[int]] = []
    share = lead._share

    def record(values):
        got = share(values)
        sent.append(list(got))
        return got

    def replay(values):
        assert values is None
        if not sent:
            raise _Done
        return sent.pop(0)

    shas, got = [], []
    run = follower._run

    def capture(*args, **kw):
        st = run(*args, **kw)
        got.append(st.get("sha256"))
        return st

    lead._share = record
    follower.rank = 1
    follower._share = replay
    follower._run = capture
    turns = {n: list(p) for n, p in zip("ABC", _sessions(91))}
    for i, name in enumerate("ABACBACB"):
        if i == 1:
            _tiny_ram(lead)
            _tiny_ram(follower)
        reply, stats = _gen(lead, turns[name], s, policy="auto:1:1:0")
        shas.append(stats.get("sha256"))
        turns[name] = turns[name] + reply + [i]
        with pytest.raises(_Done):
            follower.follow()
    assert got == shas and lead.store.stats.get("disk_restores", 0) > 0
    assert follower.store.stats == lead.store.stats
    assert follower.store.index.digest() == lead.store.index.digest()
    assert follower.store.disk.digest() == lead.store.disk.digest()


@gpu
def test_gpu_batch_slots_restore_from_disk(ckpt, tmp_path):
    from test_batch_sessions_patches import _gpu_sessions, _sampling, _serial, _waves

    ref_b = _engine(ckpt, tmp_path / "unused", gib=0, disk=False, nonexpert="q4mse")
    eng = _batch_engine(ckpt, tmp_path)
    sampling = _sampling(False)
    prompts = _gpu_sessions(12)
    _waves(eng, ref_b, sampling, prompts, waves=1)
    _tiny_ram(eng)
    waves = _waves(eng, ref_b, sampling, _gpu_sessions(13), waves=3,
                   orders=[list("ABCD"), list("CADB"), list("DBCA")])
    disk = [st for w in waves for _, st in w.values() if st.get("restored_disk")]
    assert disk and all(st["disk"]["ok"] for st in disk)
    eng.store.disk.writer.drain()
    eng.batch.stop()
    del eng
    torch.cuda.empty_cache()
    again = _batch_engine(ckpt, tmp_path)                   # a restart: the sessions come back from disk
    turn = _gpu_sessions(13)["A"]
    (reply, stats), = again.batch.generate_batch([dict(prompt=turn + [1] * 40, max_tokens=12, sampling=sampling)])
    assert stats.get("restored_disk") and stats["cached"] >= 512, stats
    assert reply == _serial(ref_b, turn + [1] * 40, sampling, 12)
    again.batch.stop()


@gpu
@pytest.mark.parametrize("fast", [False, True], ids=["exact", "fast"])
def test_gpu_fp8_latent_sessions_from_disk(tmp_path_factory, tmp_path, fast):
    """The production cache format: latent rows in FP8 (0220), a 2,300-token system prompt past the dense limit,
    exact and fast prefill; with a tiny RAM store, resumes from disk == the fresh reference."""

    from test_glm_engine import _checkpoint, _drafter
    from test_patches import _index_heads_32
    from test_session_patches import _interleave, _sampling, _sessions

    path = tmp_path_factory.mktemp("glm_session_disk_long")
    _checkpoint(path / "model", exl3=True)
    _index_heads_32(path / "model")
    _drafter(path / "dflash2")
    kw = {"fast": fast, "rows": 256, "context": 4096, "latent_kv": True, "kv": "fp8", "nonexpert": "q4mse"}
    e = _engine(path, tmp_path, **kw)
    r = _engine(path, tmp_path / "unused", gib=0, disk=False, **kw)
    s = _sampling("greedy")
    prompts = _sessions(101, system=2300, own=(200, 220, 180))
    from test_session_patches import _gen

    _gen(e, prompts[0], s, tokens=8)
    _tiny_ram(e)
    _interleave(e, r, s, prompts, policy="2" if fast else "auto:1:1:0", tokens=16)
    assert e.store.stats.get("disk_restores", 0) >= 1
    if fast:
        assert all(x.tag == 256 for x in e.store.disk.index.entries.values())


@gpu
def test_gpu_bench_real_shapes(tmp_path):
    where = Path(os.environ.get("GLM53_TF_SESSION_DISK_BENCH", "") or tmp_path)
    for tokens in (40960, 102400):
        out = sessdisk.bench(where, tokens, "fp8", "cuda")
        print("session disk bench:", out)
        assert out["exact"]
