"""patches/0440 on the CPU, no GPU: the streaming decode kernels give the bits of the kernels they replace, for every
row count, in the emulation of ``tests/decode_kernels_emu.py`` (numpy; see its docstring for what is modelled).

- E1 (``exl3_stream.cu``) against exl3.cu's routed path (rot_in, grouped_kernel, gateup_epilogue, grouped_kernel,
  down_epilogue), Xd and Y bit for bit, on a 512 / 128-wide layer of 16 experts, top 4: windows of 1-16 rows (serial
  decode, verify windows), 4-slot mixes of 11-13 rows from four independently routed sequences, and skewed windows of
  20-40 rows whose hot experts have 17-40 members (the extra member tiles); every allowed (column tiles, stages) of
  gate/up and down, grids of 1-50 CTAs, CTAs interleaved at random, copies landing at random; every row of a window
  equals the same row run alone; tickets left at zero; one epilogue per (routed pair, Hadamard block).
- E2 (``q4_stream.cu``) against ``_qmm`` + ``_reduce`` as compiled (four chained m16n8k16 from +0.0, then
  fma(xs, b, fma(p, s, acc)); slices added in order): 1-64 rows, K slices 1 / 2 / 4, N not a multiple of 64,
  bf16 and fp32 outputs, every allowed (groups a stage, stages), grids of 1-11 CTAs; rows alone == rows in the window.
- Negative controls: each mutation of a port (one cp.async group too few waited, the A swizzle dropped, warps added in
  reverse, a live row not copied; a k half moved, the fmas swapped, k chunks reversed, slices added in reverse)
  changes the bits, so the checks can fail.
- The walk at the real per-rank shapes (4,096 / 1,024, up to 289 experts and 64-row windows), index only: every
  (member tile, expert, matrix, split, column block) exactly once, each warp's k tiles exl3.cu's, the grid bound.
- The host side (needs the patched tree on PYTHONPATH): the knobs, and which windows / matmuls the hooks take.

Run: PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_decode_kernels_emulator.py      (~2-4 minutes)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import decode_kernels_emu as E  # noqa: E402

D, NI, NE, TOP = 512, 128, 16, 4
SLOTS = TOP + 1
SKG, SKD, LIMIT = 4, 1, 10.0           # exl3_mm.GATEUP_CFG[2], DOWN_CFG[2]
EXPERT_CFGS = [(2, 4), (2, 6), (2, 8), (4, 4), (4, 6), (8, 3)]      # decode_stream.EXPERTS_STAGES
QMM_CFGS = [(1, 4), (1, 6), (1, 8), (2, 4)]


# -- E1 fixtures -------------------------------------------------------------------------------------------------------
def _layer(rng):
    tr = lambda k, n: rng.integers(0, 2**32, size=(NE, k // 16, n // 16, 32), dtype=np.uint64).astype(np.uint32)  # noqa: E731
    hs = lambda n, sc: E.f32_to_f16((rng.standard_normal((NE, n)) * sc).astype(np.float32))                      # noqa: E731
    return dict(gt=tr(D, NI), ut=tr(D, NI), dt=tr(NI, D), suh_g=hs(D, 1.0), suh_u=hs(D, 1.0), svh_g=hs(NI, 0.5),
                svh_u=hs(NI, 0.5), suh_d=hs(NI, 0.05), svh_d=hs(D, 0.2))


def _group(pick):
    """glue.select's grouping: distinct experts ascending (the shared expert, id NE, last), members in row order."""

    R = pick.shape[0]
    used = sorted({int(e) for e in pick[:, :TOP].flatten()})
    maxu = min(R * TOP, NE) + 1
    ids = np.zeros(maxu, np.int64)
    mem = np.full((maxu, R), -1, np.int64)
    for u, e in enumerate(used + [NE]):
        ids[u] = e
        j = 0
        for r in range(R):
            for s in range(SLOTS):
                if pick[r, s] == e:
                    mem[u, j] = r * 32 + s
                    j += 1
    return ids, len(used) + 1, mem


def _picks(rng, R, weights=None):
    pick = np.full((R, SLOTS), NE, np.int64)
    w = np.ones(NE) if weights is None else weights
    for r in range(R):
        pick[r, :TOP] = rng.choice(NE, TOP, replace=False, p=w / w.sum())
    return pick


def _mix(rng, rows_per_slot):
    """A 4-slot batched window: each sequence routes from its own preference (rows of one sequence share experts,
    rows of different sequences share few)."""

    picks = []
    for n in rows_per_slot:
        pref = rng.permutation(NE)
        w = np.ones(NE) * 0.2
        w[pref[:6]] = 3.0
        picks.append(_picks(rng, n, w))
    return np.concatenate(picks)


def _stream(ex, x, pick, cfg_g, cfg_d, grid, rng, mutate=""):
    """exl3_mm.routed through the E1 port: exl3.cu's rot_in, then the gate/up and down stream kernels."""

    R = x.shape[0]
    P = R * SLOTS
    ids, cnt, mem = _group(pick)
    xg, xu = E.rot_in(x, pick, ex["suh_g"], ex["suh_u"], D, SLOTS, P)
    xd = np.zeros((P, NI), np.uint16)
    y = np.full((P, D), np.nan, np.float32)
    calls = [0, 0]

    def epi_g(Z, e, p, hb):
        calls[0] += 1
        E.gateup_epilogue(Z, e, ex["svh_g"], ex["svh_u"], ex["suh_d"], xd, p, hb, P, NI, SKG, LIMIT)

    def epi_d(Z, e, p, hb):
        calls[1] += 1
        E.down_epilogue(Z, e, ex["svh_d"], y, p, hb, P, D, SKD)

    maxu = ids.shape[0]
    tick_g = np.zeros(4 * maxu * (NI // 128), np.int64)
    tick_d = np.zeros(4 * maxu * (D // 128), np.int64)
    Lg = E.StreamLaunch(xg, xu, ex["gt"], ex["ut"], ids, cnt, mem, D, NI, P, SKG, SLOTS, NE, 2, epi_g)
    E.exl3_stream_port(Lg, cfg_g[0], cfg_g[1], grid, tick_g, rng, mutate=mutate)
    Ld = E.StreamLaunch(xd, xd, ex["dt"], ex["dt"], ids, cnt, mem, NI, D, P, SKD, SLOTS, NE, 1, epi_d)
    E.exl3_stream_port(Ld, cfg_d[0], cfg_d[1], grid, tick_d, rng, mutate=mutate)
    assert not tick_g.any() and not tick_d.any()                    # the tickets are left at zero
    return xd, y, calls


def _reference(ex, x, pick):
    ids, cnt, mem = _group(pick)
    return E.exl3_routed_reference(x, pick, ids, cnt, mem, ex, D, NI, SLOTS, NE, SKG, SKD, LIMIT)


def _routed_rows(R):
    return np.array([r * SLOTS + s for r in range(R) for s in range(TOP)])


def _same_f32(a, b):
    return np.array_equal(np.asarray(a, np.float32).view(np.uint32), np.asarray(b, np.float32).view(np.uint32))


E1_CASES = [
    # (name, rows or mix, weights skew, gate/up cfg, down cfg, grid)
    ("serial 1 row", 1, None, (4, 4), (4, 4), 48),
    ("verify 2", 2, None, (2, 6), (8, 3), 5),
    ("verify 3", 3, None, (4, 6), (2, 4), 1),
    ("verify 5", 5, None, (8, 3), (4, 6), 13),
    ("verify 8", 8, None, (2, 8), (2, 8), 7),
    ("verify 16", 16, None, (4, 4), (8, 3), 50),
    ("4 slots 3+3+3+2", (3, 3, 3, 2), None, (4, 4), (4, 4), 9),
    ("4 slots 5+1+4+3", (5, 1, 4, 3), None, (2, 4), (8, 3), 31),
    ("skewed 20 (17-20 members)", 20, "zipf", (4, 4), (2, 6), 11),
    ("skewed 40 (3 member tiles)", 40, "zipf", (8, 3), (4, 4), 23),
]


@pytest.mark.parametrize("case", E1_CASES, ids=[c[0] for c in E1_CASES])
def test_e1_equals_exl3_cu(case):
    name, rows, skew, cfg_g, cfg_d, grid = case
    rng = np.random.default_rng(sum(name.encode()) * 7919)
    ex = _layer(rng)
    if isinstance(rows, tuple):
        pick = _mix(rng, rows)
    else:
        pick = _picks(rng, rows, 1.0 / np.arange(1, NE + 1) ** 2 if skew else None)
    R = pick.shape[0]
    x = E.f32_to_bf16(rng.standard_normal((R, D)).astype(np.float32))
    ids, cnt, mem = _group(pick)
    if skew:
        assert (mem[:, 16] >= 0).any()                          # an expert with a second member tile
    xd_ref, y_ref = _reference(ex, x, pick)
    xd, y, calls = _stream(ex, x, pick, cfg_g, cfg_d, grid, rng)
    rr = _routed_rows(R)
    assert np.isfinite(y_ref[rr]).all()
    assert np.array_equal(xd[rr], xd_ref[rr]), name
    assert _same_f32(y[rr], y_ref[rr]), name
    assert calls == [len(rr) * (NI // 128), len(rr) * (D // 128)]     # one epilogue a (pair, Hadamard block)
    assert np.isnan(y[np.arange(R) * SLOTS + TOP]).all()            # the shared slot is never written


def test_e1_rows_alone_equal_rows_in_window():
    """A verify window's rows equal the same rows run alone (serial decode) through the stream kernels."""

    rng = np.random.default_rng(11)
    ex = _layer(rng)
    pick = _picks(rng, 6)
    x = E.f32_to_bf16(rng.standard_normal((6, D)).astype(np.float32))
    _, y, _ = _stream(ex, x, pick, (4, 4), (4, 4), 17, rng)
    for r in range(6):
        _, y1, _ = _stream(ex, x[r:r + 1], pick[r:r + 1], (2, 6), (8, 3), 48, rng)
        assert _same_f32(y1[:TOP], y[r * SLOTS:r * SLOTS + TOP]), r


@pytest.mark.parametrize("mutate", ["wait", "swizzle", "warps", "rows"])
def test_e1_negative_controls(mutate):
    rng = np.random.default_rng(5)
    ex = _layer(rng)
    pick = _picks(rng, 12, 1.0 / np.arange(1, NE + 1))
    x = E.f32_to_bf16(rng.standard_normal((12, D)).astype(np.float32))
    _, y_ref = _reference(ex, x, pick)
    _, y, _ = _stream(ex, x, pick, (4, 4), (4, 4), 7, rng, mutate=mutate)
    rr = _routed_rows(12)
    assert not _same_f32(y[rr], y_ref[rr]), mutate


@pytest.mark.parametrize("K,SK,mats", [(4096, SKG, 2), (1024, SKD, 1)], ids=["gate/up K 4096", "down K 1024"])
def test_e1_real_k_ranges(K, SK, mats):
    """The per-rank K of gate/up (4,096, 4 splits) and down (1,024, 1 split), 16 k tiles a warp, on a 256-wide N and
    2 experts: Z of the stream kernel == exl3.cu's for every live row (the address and k-range arithmetic at the real
    K; the full layer is too slow to emulate)."""

    rng = np.random.default_rng(K + SK)
    Ex, N, R, slots = 3, 256, 3, 3
    T = [rng.integers(0, 2**32, size=(Ex, K // 16, N // 16, 32), dtype=np.uint64).astype(np.uint32) for _ in range(mats)]
    P = R * slots
    X = [E.f32_to_f16((rng.standard_normal((P, K)) * 0.1).astype(np.float32)) for _ in range(mats)]
    pick = np.array([[0, 1, Ex], [1, 2, Ex], [0, 2, Ex]])
    ids = np.array([0, 1, 2, Ex])
    mem = np.full((4, R), -1)
    for u in range(3):
        j = 0
        for r in range(R):
            for sl in range(slots):
                if pick[r, sl] == u:
                    mem[u, j] = r * 32 + sl
                    j += 1
    ref = E.exl3_reference(X[0], X[-1], T[0], T[-1], ids, 4, mem, K, N, P, SK, slots, Ex, mats)
    L = E.StreamLaunch(X[0], X[-1], T[0], T[-1], ids, 4, mem, K, N, P, SK, slots, Ex, mats, lambda *a: None)
    tick = np.zeros(4 * 4 * (N // 128), np.int64)
    Z, _ = E.exl3_stream_port(L, 4, 4, 5, tick, rng)
    Z = Z.reshape(mats, SK, P, N)
    live = [r * slots + sl for r in range(R) for sl in range(slots - 1)]
    assert np.isfinite(ref[:, :, live]).all()
    assert _same_f32(Z[:, :, live], ref[:, :, live])


# -- E1 walk at the real shapes (index only) ---------------------------------------------------------------------------
def _walk(ids, cnt, mem, E_, mats, SK, N, NT):
    """The (mt, u, mat, split, nb) of every item, as exl3_stream.cu's walk numbers them."""

    nexp = cnt - (1 if cnt > 0 and ids[cnt - 1] >= E_ else 0)
    MT = (mem.shape[1] + 15) // 16
    extra = [(mt, u) for mt in range(1, MT) for u in range(nexp) if mem[u, mt * 16] >= 0]
    NB = N // (16 * NT)
    ipe = mats * SK * NB
    out = []
    for it in range((nexp + len(extra)) * ipe):
        pi, r = divmod(it, ipe)
        mt, u = (0, pi) if pi < nexp else extra[pi - nexp]
        ms, nb = divmod(r, NB)
        out.append((mt, u, ms // SK, ms % SK, nb))
    return out, nexp, len(extra)


@pytest.mark.parametrize("R", [1, 3, 8, 16, 44, 64])
@pytest.mark.parametrize("NT", [2, 4, 8])
def test_e1_walk_real_shapes(R, NT):
    rng = np.random.default_rng(R * 10 + NT)
    Dr, NIr, Er, top = 4096, 1024, 288, 8
    pick = np.full((R, top + 1), Er, np.int64)
    w = 1.0 / np.arange(1, Er + 1) ** 0.8
    for r in range(R):
        pick[r, :top] = rng.choice(Er, top, replace=False, p=w / w.sum())
    used = sorted({int(e) for e in pick[:, :top].flatten()})
    maxu = min(R * top, Er) + 1
    ids = np.zeros(maxu, np.int64)
    mem = np.full((maxu, R), -1, np.int64)
    for u, e in enumerate(used + [Er]):
        ids[u] = e
        j = 0
        for r in range(R):
            for s in range(top + 1):
                if pick[r, s] == e:
                    mem[u, j] = r * 32 + s
                    j += 1
    MT = (R + 15) // 16
    assert maxu <= 1024 and (MT - 1) * maxu <= E.MAX_EXTRA              # decode_stream.experts_ok's bound
    for mats, SK, K, N in ((2, SKG, Dr, NIr), (1, SKD, NIr, Dr)):
        items, nexp, n_extra = _walk(ids, len(used) + 1, mem, Er, mats, SK, N, NT)
        assert len(items) == len(set(items))
        want = {(mt, u, mat, sp, nb) for u in range(nexp) for mt in range(MT) if mem[u, mt * 16] >= 0
                for mat in range(mats) for sp in range(SK) for nb in range(N // (16 * NT))}
        assert set(items) == want                                     # every live work item exactly once
        maxi = MT * maxu * mats * SK * (N // (16 * NT))
        assert len(items) <= maxi                                     # the host's grid bound
        # each warp's k tiles: exl3.cu's kt0 = split * KT / SK + w * PW, PW = KT / (SK * 4)
        KT = K // 16
        PW = KT // (SK * 4)
        cover = np.zeros(KT, np.int64)
        for sp in range(SK):
            for w_ in range(4):
                cover[sp * (KT // SK) + w_ * PW:sp * (KT // SK) + w_ * PW + PW] += 1
        assert (cover == 1).all() and PW == 16


# -- E2 ----------------------------------------------------------------------------------------------------------------
def _q4_inputs(rng, M, n, k):
    x = E.f32_to_bf16(rng.standard_normal((M, k)).astype(np.float32))
    xf = E.bf16_to_f32(x)
    xs = np.stack([xf[:, g * 64:(g + 1) * 64].sum(1, dtype=np.float32) for g in range(k // 64)], 1).astype(np.float32)
    words = rng.integers(-2**31, 2**31 - 1, size=(n, k // 8), dtype=np.int64).astype(np.int32)
    sc = E.f32_to_bf16((rng.random((n, k // 64)) * 0.01 + 0.005).astype(np.float32))
    bi = E.f32_to_bf16(-7.5 * E.bf16_to_f32(sc))
    return x, xs, words, sc, bi


def _rt(M):
    rt = (M + 15) // 16
    return 1 if rt == 1 else (2 if rt == 2 else 4)


E2_CASES = [
    # (M, n, k, sk, f32, (gps, stages), grid, serial)
    (1, 128, 256, 2, False, (1, 4), 3, False),
    (1, 200, 512, 4, False, (1, 8), 11, False),
    (1, 200, 512, 4, False, (1, 6), 3, True),
    (2, 72, 256, 1, True, (2, 4), 2, False),
    (3, 320, 512, 2, False, (1, 6), 5, False),
    (3, 320, 512, 2, True, (2, 4), 2, True),
    (8, 128, 512, 4, True, (1, 4), 1, False),
    (8, 136, 512, 8, False, (1, 4), 2, True),
    (16, 200, 256, 1, False, (2, 4), 4, False),
    (17, 136, 256, 2, False, (2, 4), 3, False),
    (24, 136, 256, 4, False, (1, 4), 1, True),
    (33, 128, 256, 4, True, (1, 4), 2, False),
    (64, 72, 256, 2, False, (1, 4), 3, False),
    (64, 72, 256, 2, True, (1, 4), 2, True),
]


@pytest.mark.parametrize("case", E2_CASES,
                         ids=[f"M{c[0]}-n{c[1]}-k{c[2]}-sk{c[3]}{'-serial' if c[7] else ''}" for c in E2_CASES])
def test_e2_equals_qmm(case):
    M, n, k, sk, f32, (gps, st), grid, serial = case
    rng = np.random.default_rng(M * 1000 + n + k + sk)
    x, xs, words, sc, bi = _q4_inputs(rng, M, n, k)
    ref = E.qmm_reference(x, xs, words, sc, bi, n, k, sk, f32)
    rt = _rt(M)
    if M > 16:                                  # decode_stream.run_qmm's choice for 17-64 rows
        gps, st = 1, 4
    got = E.q4_stream_port(x, xs, words, sc, bi, n, k, sk, f32, rt, gps, st, grid, rng, serial=serial)
    view = np.uint32 if f32 else np.uint16
    assert np.array_equal(np.asarray(got).view(view), np.asarray(ref).view(view))


def test_e2_rows_alone_equal_rows_in_window():
    rng = np.random.default_rng(3)
    M, n, k, sk = 12, 136, 256, 2
    x, xs, words, sc, bi = _q4_inputs(rng, M, n, k)
    win = E.q4_stream_port(x, xs, words, sc, bi, n, k, sk, False, 1, 1, 4, 5, rng)
    for r in (0, 5, 11):
        one = E.q4_stream_port(x[r:r + 1], xs[r:r + 1], words, sc, bi, n, k, sk, False, 1, 2, 4, 7, rng)
        assert np.array_equal(one[0], win[r])
        one = E.q4_stream_port(x[r:r + 1], xs[r:r + 1], words, sc, bi, n, k, sk, False, 1, 1, 6, 1, rng, serial=True)
        assert np.array_equal(one[0], win[r])


@pytest.mark.parametrize("mutate,serial", [("wait", False), ("khalf", False), ("fma", False), ("chunks", False),
                                           ("slices", False), ("wait", True), ("khalf", True), ("fma", True)])
def test_e2_negative_controls(mutate, serial):
    rng = np.random.default_rng(9)
    M, n, k, sk = 5, 128, 256, 4
    x, xs, words, sc, bi = _q4_inputs(rng, M, n, k)
    ref = E.qmm_reference(x, xs, words, sc, bi, n, k, sk, True)
    got = E.q4_stream_port(x, xs, words, sc, bi, n, k, sk, True, 1, 1, 4, 3, rng, mutate=mutate, serial=serial)
    assert not np.array_equal(np.asarray(got).view(np.uint32), np.asarray(ref).view(np.uint32)), mutate


def test_mma_model_is_position_sensitive():
    """The model separates a product moved to another k slot (what the fragment checks rely on)."""

    rng = np.random.default_rng(0)
    A = E.bf16_to_f32(E.f32_to_bf16(rng.standard_normal((16, 16)).astype(np.float32)))
    B = E.bf16_to_f32(E.f32_to_bf16(rng.standard_normal((16, 8)).astype(np.float32) * 1e3))
    C = np.zeros((16, 8), np.float32)
    perm = np.arange(16)
    perm[[2, 10]] = perm[[10, 2]]
    assert not np.array_equal(E.mma_model(C, A, B), E.mma_model(C, A[:, perm], B[perm]))


# -- host side (the patched tree) ---------------------------------------------------------------------------------------
def _ds():
    torch = pytest.importorskip("torch")
    try:
        from tensorfold.families.glm5_next.cuda import decode_stream as ds
    except ImportError:
        pytest.skip("needs the patched tree (patches/0440) on PYTHONPATH")
    return torch, ds


def test_knobs(monkeypatch):
    torch, ds = _ds()
    for k in ("GLM53_TF_DEC_EXPERTS", "GLM53_TF_DEC_QMM", "GLM53_TF_DEC_PDL", "GLM53_TF_DEC_EXPERTS_CFG",
              "GLM53_TF_DEC_QMM_CFG", "GLM53_TF_DEC_CTAS"):
        monkeypatch.delenv(k, raising=False)
    cfg = ds.parse()
    extra = {"qmm_max_bytes": 0, "qmm_table": True, "qmm_exclude": frozenset()} if "qmm_max_bytes" in cfg else {}
    assert cfg == {"experts": False, "qmm": False, "pdl": False, "experts_cfg": (4, 4, 4, 4), "qmm_cfg": (1, 4),
                   "ctas": 0, "qmm_serial": "auto", **extra}          # extra: patches/0570's size switch, off
    assert ds.serial_for(12576, 4) and ds.serial_for(160, 1) and not ds.serial_for(160, 8)
    assert not ds.serial_for(2048, 4) and ds.serial_for(2048, 4, "1") and not ds.serial_for(77440, 2, "0")
    monkeypatch.setenv("GLM53_TF_DEC_EXPERTS", "1")
    monkeypatch.setenv("GLM53_TF_DEC_EXPERTS_CFG", "2,8,8,3")
    monkeypatch.setenv("GLM53_TF_DEC_QMM_CFG", "2,4")
    cfg = ds.parse()
    assert cfg["experts"] and cfg["experts_cfg"] == (2, 8, 8, 3) and cfg["qmm_cfg"] == (2, 4)
    for name, bad in (("GLM53_TF_DEC_EXPERTS_CFG", "8,4,4,4"), ("GLM53_TF_DEC_EXPERTS_CFG", "4,4"),
                      ("GLM53_TF_DEC_QMM_CFG", "2,8"), ("GLM53_TF_DEC_CTAS", "-1"), ("GLM53_TF_DEC_QMM_SERIAL", "2")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError):
            ds.parse()
        monkeypatch.delenv(name)
    from tensorfold.families.glm5_next.cuda import exl3_mm, qmm, sessdisk
    assert exl3_mm.STREAM is None and qmm.STREAM is None                    # off by default
    # same bits: the knobs stay out of the NVMe compat hash (0250)
    env = {"GLM53_TF_DEC_EXPERTS": "1", "GLM53_TF_DEC_EXPERTS_CFG": "2,8,8,3", "GLM53_TF_DEC_QMM": "1",
           "GLM53_TF_DEC_QMM_CFG": "1,6", "GLM53_TF_DEC_QMM_SERIAL": "0", "GLM53_TF_DEC_PDL": "1",
           "GLM53_TF_DEC_CTAS": "40", "GLM53_TF_NONEXPERT": "q4mse"}
    assert sessdisk.knobs(env) == {"GLM53_TF_NONEXPERT": "q4mse"}
    # the dense kernel only under a Triton that fuses _qmm's epilogue (3.7.x); force overrides
    import triton
    monkeypatch.setenv("GLM53_TF_DEC_QMM", "1")
    monkeypatch.setattr(triton, "__version__", "3.8.0")
    assert not ds.parse()["qmm"]
    monkeypatch.setenv("GLM53_TF_DEC_QMM", "force")
    assert ds.parse()["qmm"]
    monkeypatch.setattr(triton, "__version__", "3.7.1")
    monkeypatch.setenv("GLM53_TF_DEC_QMM", "1")
    assert ds.parse()["qmm"]
    monkeypatch.delenv("GLM53_TF_DEC_QMM")
    with ds.using(experts=True, qmm_on=True):
        assert exl3_mm.STREAM is ds and qmm.STREAM is ds
    assert exl3_mm.STREAM is None and qmm.STREAM is None


def test_dispatch_rules():
    torch, ds = _ds()
    from types import SimpleNamespace

    from tensorfold.families.glm5_next.cuda import exl3_mm, qmm

    def grp(maxu, maxm):
        return SimpleNamespace(ids=torch.zeros(maxu, dtype=torch.int32), count=torch.zeros(1, dtype=torch.int32),
                               members=torch.full((maxu, maxm), -1, dtype=torch.int32))

    ex = SimpleNamespace(dims=4096, width=1024, count=288, gt=torch.zeros(4, dtype=torch.int32),
                         ut=torch.zeros(4, dtype=torch.int32), dt=torch.zeros(4, dtype=torch.int32))
    s = SimpleNamespace(z=torch.zeros(4), xg=torch.zeros(4), xu=torch.zeros(4), xd=torch.zeros(4))
    assert ds.experts_ok(grp(9, 1), ex, s)                  # serial decode
    assert ds.experts_ok(grp(129, 16), ex, s)               # a 16-row verify window
    assert ds.experts_ok(grp(289, 44), ex, s)               # a 4-slot batched window
    assert ds.experts_ok(grp(289, 64), ex, s)
    assert not ds.experts_ok(grp(289, 65), ex, s)           # prefill chunks: exl3.cu's loop kernel
    assert not ds.experts_ok(grp(9, 1), ex, SimpleNamespace(z=None))    # lean prefill scratch
    assert exl3_mm.GATEUP_CFG[1] == exl3_mm.DOWN_CFG[1] == 4 and (exl3_mm.GATEUP_CFG[2], exl3_mm.DOWN_CFG[2]) == (SKG, SKD)
    q = qmm.Q4(torch.zeros(((12576 + 63) // 64) * 64 * 512, dtype=torch.int32), torch.zeros(64 * 12576, dtype=torch.bfloat16),
               torch.zeros(64 * 12576, dtype=torch.bfloat16), 12576, 4096)
    x = torch.zeros((4, 4096), dtype=torch.bfloat16)
    part = torch.zeros(4 * 64 * 12576)
    assert ds.qmm_ok(x, q, None, None, part)
    assert ds.qmm_ok(x, q, None, None, None)                # 197 tiles: whole-tile items, no partial buffer needed
    q2 = qmm.Q4(torch.zeros(32 * 64 * 512, dtype=torch.int32), torch.zeros(64 * 2048, dtype=torch.bfloat16),
                torch.zeros(64 * 2048, dtype=torch.bfloat16), 2048, 4096)
    assert ds.qmm_ok(x, q2, None, None, part)
    assert not ds.qmm_ok(x, q2, None, None, None)           # 32 tiles: split items need the forward's partial buffer
    assert not ds.qmm_ok(torch.zeros((65, 4096), dtype=torch.bfloat16), q, None, None, part)
    assert not ds.qmm_ok(x[:, 8:], q, None, None, part)     # K mismatch
    b16 = qmm.B16(torch.zeros((8, 8), dtype=torch.bfloat16), 8, 8)
    assert not ds.qmm_ok(x, b16, None, None, part)          # BF16 weights: qmm's kernels
