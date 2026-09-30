"""patches/0580 on the CPU, no GPU: ``exl3_ld.cu``'s ld_kernel gives exl3.cu's grouped_kernel bits for every row count
and every load setting, in the emulation of ``tests/decode_loads_emu.py`` (numpy; see its docstring for what is
modelled), plus the host side of 0580 (knobs, the hook in ``exl3_mm.routed``).

- Z of every (ld, nt, pd) the host can launch == ``decode_kernels_emu.exl3_reference`` (exl3.cu's grouped kernel at
  matrix level), bit for bit including the elements never stored (NaN), on gate/up-like (K 512, 2 matrices, 1-4 K
  splits) and down-like (K 256, 1 matrix) layers of 16 experts, top 4: windows of 1-16 rows, 4-slot mixes, skewed
  windows of 20-40 rows (2-3 member tiles: what GLM53_TF_EXPERT_LOOP=0 would send here), stale distinct-expert slots
  past ucount, the shared expert; cp.async copies landing at random, NaN-initialised shared memory.
- exl3_mm.routed's non-fast path with the two grouped launches through the port == the reference Xd and Y.
- The real per-rank K ranges (gate/up K 4,096 in 4 splits, down K 1,024: 16 k tiles a warp, every ring depth).
- Rows alone == rows in the window.
- Negative controls: each mutation (staging read lane-major, a slot refilled before it is read, the cp.async ring slot
  or wait off by one, A fragments of the wrong k step, warps added in reverse, a warp's k order rotated) changes the
  bits, so the checks can fail.

Run: PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_decode_loads_emulator.py      (~2-4 minutes; the host
tests need the patched tree and torch, CPU is enough)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import decode_kernels_emu as E  # noqa: E402
import decode_loads_emu as L  # noqa: E402

NE, TOP = 16, 4
SLOTS = TOP + 1
CFGS = [("nc", 8, 2), ("nc", 8, 1), ("nc", 8, 4), ("nc", 4, 2), ("nc", 4, 4), ("w32", 8, 2), ("w32", 4, 4),
        ("cpa", 8, 2), ("cpa", 4, 4)]                      # expert_loads.CFGS


def _group(pick, rng=None):
    """glue.select's grouping (distinct experts ascending, the shared expert NE last, members in row order); slots past
    the count hold garbage (stale values of an earlier window) when ``rng`` is given."""

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
    cnt = len(used) + 1
    if rng is not None and cnt < maxu:
        ids[cnt:] = rng.integers(0, NE + 3, maxu - cnt)
        mem[cnt:] = rng.integers(-1, R * 32, (maxu - cnt, R))
    return ids, cnt, mem


def _picks(rng, R, weights=None):
    pick = np.full((R, SLOTS), NE, np.int64)
    w = np.ones(NE) if weights is None else weights
    for r in range(R):
        pick[r, :TOP] = rng.choice(NE, TOP, replace=False, p=w / w.sum())
    return pick


def _mix(rng, rows_per_slot):
    picks = []
    for n in rows_per_slot:
        pref = rng.permutation(NE)
        w = np.ones(NE) * 0.2
        w[pref[:6]] = 3.0
        picks.append(_picks(rng, n, w))
    return np.concatenate(picks)


def _trellis(rng, k, n):
    return rng.integers(0, 2**32, size=(NE, k // 16, n // 16, 32), dtype=np.uint64).astype(np.uint32)


def _x(rng, P, K):
    return E.f32_to_f16((rng.standard_normal((P, K)) * 0.5).astype(np.float32))


def _same(a, b):
    return np.array_equal(np.asarray(a, np.float32).view(np.uint32), np.asarray(b, np.float32).view(np.uint32))


def _pick_for(rng, rows, skew):
    if isinstance(rows, tuple):
        return _mix(rng, rows)
    return _picks(rng, rows, 1.0 / np.arange(1, NE + 1) ** 2 if skew else None)


# (name, rows or mix, skew, K, N, SK, mats)
LAYERS = {"gate/up SK 1": (512, 128, 1, 2), "gate/up SK 2": (512, 128, 2, 2), "gate/up SK 4": (512, 128, 4, 2),
          "down": (256, 256, 1, 1)}
WINDOWS = [("1 row", 1, False), ("verify 3", 3, False), ("verify 8", 8, False), ("verify 16", 16, False),
           ("4 slots 3+3+3+2", (3, 3, 3, 2), False), ("skewed 20 (2 member tiles)", 20, True),
           ("skewed 40 (3 member tiles)", 40, True)]


def _cases():
    out = []
    i = 0
    for cfg in CFGS:
        for lname, (K, N, SK, mats) in LAYERS.items():
            PW = K // 16 // SK // 4
            if PW % cfg[2] or N % (16 * cfg[1]):
                continue
            w = WINDOWS[i % len(WINDOWS)]
            i += 1
            out.append((cfg, lname, w))
    return out


CASES = _cases()


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0][0]},{c[0][1]},{c[0][2]} {c[1]} {c[2][0]}" for c in CASES])
def test_z_equals_grouped_kernel(case):
    (ld, nt, pd), lname, (wname, rows, skew) = case
    K, N, SK, mats = LAYERS[lname]
    rng = np.random.default_rng(sum(f"{case}".encode()) * 131)
    pick = _pick_for(rng, rows, skew)
    R = pick.shape[0]
    P = R * SLOTS
    ids, cnt, mem = _group(pick, rng)
    if skew:
        assert (mem[:cnt, 16] >= 0).any()
    T0, T1 = _trellis(rng, K, N), _trellis(rng, K, N)
    X0, X1 = _x(rng, P, K), _x(rng, P, K)
    ref = E.exl3_reference(X0, X1, T0, T1, ids, cnt, mem, K, N, P, SK, SLOTS, NE, mats)
    got = L.exl3_ld_port(X0, X1, T0, T1, ids, cnt, mem, K, N, P, SK, SLOTS, NE, mats, nt, pd, ld, rng)
    live = np.isfinite(ref)
    assert live.any()
    assert _same(got, ref), (ld, nt, pd, lname, wname)          # stored elements equal, never-stored stay NaN


def _routed(ex, x, pick, D, NI, SKG, SKD, cfg_g, cfg_d, rng, limit=10.0):
    """exl3_mm.routed's non-fast path with both grouped launches through the port (rot_in / epilogues: exl3.cu's)."""

    R = x.shape[0]
    P = R * SLOTS
    ids, cnt, mem = _group(pick, rng)
    xg, xu = E.rot_in(x, pick, ex["suh_g"], ex["suh_u"], D, SLOTS, P)
    Zg = L.exl3_ld_port(xg, xu, ex["gt"], ex["ut"], ids, cnt, mem, D, NI, P, SKG, SLOTS, NE, 2, cfg_g[1], cfg_g[2],
                        cfg_g[0], rng)
    xd = np.zeros((P, NI), np.uint16)
    for row in range(R):
        for slot in range(SLOTS - 1):
            for blk in range(NI // 128):
                E.gateup_epilogue(Zg.reshape(-1), pick[row, slot], ex["svh_g"], ex["svh_u"], ex["suh_d"], xd,
                                  row * SLOTS + slot, blk, P, NI, SKG, limit)
    Zd = L.exl3_ld_port(xd, xd, ex["dt"], ex["dt"], ids, cnt, mem, NI, D, P, SKD, SLOTS, NE, 1, cfg_d[1], cfg_d[2],
                        cfg_d[0], rng)
    y = np.full((P, D), np.nan, np.float32)
    for row in range(R):
        for slot in range(SLOTS - 1):
            for blk in range(D // 128):
                E.down_epilogue(Zd.reshape(-1), pick[row, slot], ex["svh_d"], y, row * SLOTS + slot, blk, P, D, SKD)
    return xd, y


def _layer(rng, D, NI):
    hs = lambda n, sc: E.f32_to_f16((rng.standard_normal((NE, n)) * sc).astype(np.float32))       # noqa: E731
    return dict(gt=_trellis(rng, D, NI), ut=_trellis(rng, D, NI), dt=_trellis(rng, NI, D), suh_g=hs(D, 1.0),
                suh_u=hs(D, 1.0), svh_g=hs(NI, 0.5), svh_u=hs(NI, 0.5), suh_d=hs(NI, 0.05), svh_d=hs(D, 0.2))


@pytest.mark.parametrize("cfg_g,cfg_d,rows", [(("nc", 8, 2), ("nc", 8, 2), 3), (("cpa", 8, 2), ("w32", 8, 2), 5),
                                              (("nc", 4, 2), ("cpa", 4, 4), (2, 1, 2, 1))])
def test_routed_equals_exl3_routed(cfg_g, cfg_d, rows):
    """Xd and Y of exl3_mm.routed (exl3_mm's K splits: gate/up 4 on K 512 = 2 k tiles a warp... 1 on the down's K 256)."""

    D, NI, SKG, SKD = 512, 256, 2, 1                       # PW: gate/up 512/16/2/4 = 4, down 256/16/1/4 = 4
    rng = np.random.default_rng(7 + (rows if isinstance(rows, int) else sum(rows)))
    ex = _layer(rng, D, NI)
    pick = _pick_for(rng, rows, False)
    R = pick.shape[0]
    x = E.f32_to_bf16(rng.standard_normal((R, D)).astype(np.float32))
    ids, cnt, mem = _group(pick)
    xd_ref, y_ref = E.exl3_routed_reference(x, pick, ids, cnt, mem, ex, D, NI, SLOTS, NE, SKG, SKD, 10.0)
    xd, y = _routed(ex, x, pick, D, NI, SKG, SKD, cfg_g, cfg_d, rng)
    rr = np.array([r * SLOTS + s for r in range(R) for s in range(TOP)])
    assert np.isfinite(y_ref[rr]).all()
    assert np.array_equal(xd[rr], xd_ref[rr])
    assert _same(y[rr], y_ref[rr])
    assert np.isnan(y[np.arange(R) * SLOTS + TOP]).all()         # the shared slot is never written


@pytest.mark.parametrize("K,SK,mats", [(4096, 4, 2), (1024, 1, 1)], ids=["gate/up K 4096", "down K 1024"])
@pytest.mark.parametrize("cfg", [("nc", 8, 2), ("nc", 8, 4), ("cpa", 8, 2), ("w32", 8, 2)], ids=lambda c: ",".join(map(str, c)))
def test_real_k_ranges(K, SK, mats, cfg):
    """The per-rank K of gate/up (4,096, 4 splits) and down (1,024, 1 split): 16 k tiles a warp, every ring depth, on
    a 128-wide N slice and 2 experts (the columns are independent; the full width is too slow to emulate)."""

    rng = np.random.default_rng(K + SK + sum(map(ord, cfg[0])) + cfg[2])
    N = 128
    pick = np.full((2, SLOTS), NE, np.int64)
    pick[0, :TOP] = [3, 9, 0, 1]
    pick[1, :TOP] = [9, 2, 0, 1]
    ids, cnt, mem = _group(pick)
    keep = np.isin(ids, [3, 9, NE])                 # 2 routed experts (+ the shared one) to keep it fast
    ids2 = ids[keep]
    mem2 = mem[keep]
    cnt2 = int(keep[:cnt].sum())
    P = 2 * SLOTS
    T0, T1 = _trellis(rng, K, N), _trellis(rng, K, N)
    X0, X1 = _x(rng, P, K), _x(rng, P, K)
    ref = E.exl3_reference(X0, X1, T0, T1, ids2, cnt2, mem2, K, N, P, SK, SLOTS, NE, mats)
    got = L.exl3_ld_port(X0, X1, T0, T1, ids2, cnt2, mem2, K, N, P, SK, SLOTS, NE, mats, cfg[1], cfg[2], cfg[0], rng)
    assert np.isfinite(ref).any() and _same(got, ref)


def test_rows_alone_equal_rows_in_window():
    """A verify window's Z rows equal the same routed rows run alone (serial decode)."""

    rng = np.random.default_rng(11)
    K, N, SK = 512, 128, 2
    pick = _picks(rng, 5)
    P = 5 * SLOTS
    T0, T1 = _trellis(rng, K, N), _trellis(rng, K, N)
    X0, X1 = _x(rng, P, K), _x(rng, P, K)
    ids, cnt, mem = _group(pick)
    Zw = L.exl3_ld_port(X0, X1, T0, T1, ids, cnt, mem, K, N, P, SK, SLOTS, NE, 2, 8, 2, "nc", rng)
    for r in range(5):
        ids1, cnt1, mem1 = _group(pick[r:r + 1])
        rows = slice(r * SLOTS, r * SLOTS + SLOTS)
        Z1 = L.exl3_ld_port(X0[rows], X1[rows], T0, T1, ids1, cnt1, mem1, K, N, SLOTS, SK, SLOTS, NE, 2, 4, 4, "cpa", rng)
        assert _same(Z1[:, :, :TOP], Zw[:, :, r * SLOTS:r * SLOTS + TOP]), r


MUTATIONS = [("stage", "nc"), ("early", "nc"), ("early", "w32"), ("ring", "cpa"), ("wait", "cpa"), ("aoff", "nc"),
             ("warps", "nc"), ("kfirst", "nc")]


@pytest.mark.parametrize("mutate,ld", MUTATIONS, ids=[f"{m}-{ld}" for m, ld in MUTATIONS])
def test_negative_controls(mutate, ld):
    rng = np.random.default_rng(5)
    K, N, SK = 512, 128, 1                                 # 8 k tiles a warp: every mutation has room to act
    pick = _picks(rng, 4)
    P = 4 * SLOTS
    T0, T1 = _trellis(rng, K, N), _trellis(rng, K, N)
    X0, X1 = _x(rng, P, K), _x(rng, P, K)
    ids, cnt, mem = _group(pick)
    ref = E.exl3_reference(X0, X1, T0, T1, ids, cnt, mem, K, N, P, SK, SLOTS, NE, 2)
    ok = L.exl3_ld_port(X0, X1, T0, T1, ids, cnt, mem, K, N, P, SK, SLOTS, NE, 2, 8, 2, ld, rng)
    assert _same(ok, ref)
    bad = L.exl3_ld_port(X0, X1, T0, T1, ids, cnt, mem, K, N, P, SK, SLOTS, NE, 2, 8, 2, ld, rng, mutate=mutate)
    assert not _same(bad, ref), mutate


def test_bytes_in_flight():
    """The design point (docs/DECODE-KERNELS-2.md §2): >= 4 KB an SM in flight even with one CTA an SM left (the
    tail), 6-8 KB+ with the SM full; W16 littles: nc reaches 232.7 GB/s at 4 KB an SM."""

    assert L.ring_bytes_in_flight(8, 2, 1) == 8192
    assert L.ring_bytes_in_flight(8, 2, 3) == 24576
    assert L.ring_bytes_in_flight(8, 1, 1) == 4096
    assert L.ring_bytes_in_flight(4, 2, 4) == 16384


# -- host side (the patched tree + torch; CPU is enough) ------------------------------------------------------------------
def _mods():
    pytest.importorskip("torch")
    try:
        from tensorfold.families.glm5_next.cuda import exl3_mm, expert_loads
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"needs the patched tree on PYTHONPATH ({e})")
    return exl3_mm, expert_loads


def test_knobs(monkeypatch):
    _, el = _mods()
    monkeypatch.delenv("GLM53_TF_DEC_EXPERT_LOADS", raising=False)
    monkeypatch.delenv("GLM53_TF_DEC_EXPERT_LOADS_CFG", raising=False)
    monkeypatch.delenv("GLM53_TF_DEC_EXPERT_LOADS_PDL", raising=False)
    assert el.parse() == {"on": False, "gu": ("nc", 8, 2), "dn": ("nc", 8, 2), "pdl": False}
    monkeypatch.setenv("GLM53_TF_DEC_EXPERT_LOADS", "1")
    monkeypatch.setenv("GLM53_TF_DEC_EXPERT_LOADS_CFG", "cpa,8,2/nc,4,4")
    monkeypatch.setenv("GLM53_TF_DEC_EXPERT_LOADS_PDL", "1")
    assert el.parse() == {"on": True, "gu": ("cpa", 8, 2), "dn": ("nc", 4, 4), "pdl": True}
    monkeypatch.setenv("GLM53_TF_DEC_EXPERT_LOADS_CFG", "w32, 4, 4")
    assert el.parse()["dn"] == ("w32", 4, 4)
    for bad in ("nc,8,3", "tma,8,2", "nc,8", "nc,8,2/nc,8,2/nc,8,2", "nc,16,2", "cpa,8,4"):
        monkeypatch.setenv("GLM53_TF_DEC_EXPERT_LOADS_CFG", bad)
        with pytest.raises(ValueError):
            el.parse()


def test_fits_real_shapes():
    exl3_mm, el = _mods()
    D, NI = 4096, 1024
    for cfg in el.CFGS:
        gnt, gw, gsk = exl3_mm.GATEUP_CFG
        dnt, dw, dsk = exl3_mm.DOWN_CFG
        assert el.fits(D, NI, gsk, gw, cfg)                 # gate/up: 16 k tiles a warp
        assert el.fits(NI, D, dsk, dw, cfg)                 # down: 16 k tiles a warp
    assert not el.fits(D, NI, 4, 8, ("nc", 8, 2))           # 8 warps: another sum order, never taken
    assert not el.fits(512, 128, 4, 4, ("nc", 8, 4))        # 2 k tiles a warp < ring depth
    assert not el.fits(512, 120, 1, 4, ("nc", 8, 2))        # columns not a multiple of the block


class _Rec:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append(name)
        return f


def test_hook_dispatch(monkeypatch):
    """exl3_mm.routed (not fast): windows of <= 16 rows run ld_kernel for both grouped launches when the knob is on,
    exl3.cu's grouped otherwise; longer windows keep grouped_loop; rot_in / epilogues are exl3.cu's either way."""

    import torch

    exl3_mm, el = _mods()
    rec = _Rec()
    monkeypatch.setattr(exl3_mm, "_ext", lambda: rec)
    ran = []
    monkeypatch.setattr(el, "run", lambda *a, **k: ran.append((a[6], a[-1] if len(a) > 12 else None)))
    D, NI = 512, 128

    class _G:
        def __init__(self, R):
            self.ids = torch.zeros(R * 4 + 1, dtype=torch.int32)
            self.count = torch.zeros(1, dtype=torch.int32)
            self.members = torch.full((R * 4 + 1, R), -1, dtype=torch.int32)

    tr = lambda k, n: torch.zeros((4, k // 16, n // 16, 32), dtype=torch.int32)       # noqa: E731
    h = lambda n: torch.zeros((4, n), dtype=torch.float16)                            # noqa: E731
    ex = exl3_mm.Exl3Experts(tr(D, NI), tr(D, NI), tr(NI, D), h(D), h(D), h(NI), h(NI), h(NI), h(D), 4, NI, D)
    s = exl3_mm.Scratch(32, SLOTS, D, NI, "cpu")
    y = torch.zeros((32 * SLOTS, D))
    for on in (False, True):
        with el.using(on=on, gu=("nc", 8, 2), dn=("nc", 8, 2)):
            for R in (1, 8, 16, 17):
                rec.calls.clear()
                ran.clear()
                x = torch.zeros((R, D), dtype=torch.bfloat16)
                exl3_mm.routed(x, torch.zeros((R, SLOTS), dtype=torch.int32), _G(R), ex, s, y, R, 10.0)
                loop = exl3_mm.LOOP and R > 16
                if loop:
                    assert rec.calls.count("grouped_loop") == 2 and not ran
                elif on:
                    assert rec.calls.count("grouped") == 0 and [m for m, _ in ran] == [2, 1]
                else:
                    assert rec.calls.count("grouped") == 2 and not ran
                assert rec.calls[0] == "rot_in" and "gateup_epilogue" in rec.calls and "down_epilogue" in rec.calls
    assert exl3_mm.LOADS is (el if el.CFG["on"] else None)


def test_session_compat_ignores_the_knob(monkeypatch):
    pytest.importorskip("torch")
    try:
        from tensorfold.families.glm5_next.cuda import sessdisk
    except ImportError:
        pytest.skip("needs the patched tree")
    monkeypatch.setenv("GLM53_TF_DEC_EXPERT_LOADS", "1")
    monkeypatch.setenv("GLM53_TF_DEC_EXPERT_LOADS_CFG", "cpa,8,2")
    assert not any(k.startswith("GLM53_TF_DEC_EXPERT_LOADS") for k in sessdisk.knobs())


def test_bench_gate_lines():
    """bench_decode_kernels.gate_0580 on synthetic rows: one config for every gate window, flush >= 1.05x and rotate
    >= 1.00x on each, only same-bits configs, the probe line at 220 GB/s; windows outside U 8-22 ignored."""

    pytest.importorskip("torch")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cuda"))
    try:
        import bench_decode_kernels as bdk
    except ImportError as e:
        pytest.skip(f"needs the patched tree ({e})")

    def row(U, old, new, bits=True, probe=230.0):
        r = {"window": f"U {U}", "U": U, "old_flush_us": old, "old_rotate_us": old}
        for k, us in new.items():
            r[k] = {"flush_us": us, "rotate_us": us * 0.99, "same_bits": bits}
        r["probe 3 nc"] = {"flush_GBs": probe}
        return r

    rows = [row(8, 100, {"new nc,8,2": 90, "new cpa,8,2": 97}), row(13, 150, {"new nc,8,2": 140, "new cpa,8,2": 140}),
            row(22, 250, {"new nc,8,2": 230, "new cpa,8,2": 240}), row(51, 500, {"new nc,8,2": 600, "new cpa,8,2": 600})]
    g = bdk.gate_0580(rows)
    assert g["best"] == "new nc,8,2" and g["pass"] and g["probe_pass"] and "PASS" in g["line"]
    rows[1]["new nc,8,2"]["flush_us"] = 145                  # 150 / 145 = 1.034 < 1.05 on one window
    g = bdk.gate_0580(rows)
    assert not g["pass"] and "FAIL" in g["line"]
    rows[0]["new nc,8,2"]["same_bits"] = False               # a config that changes bits is never eligible
    assert bdk.gate_0580(rows)["best"] == "new cpa,8,2"
    rows[2]["probe 3 nc"]["flush_GBs"] = 210.0
    assert not bdk.gate_0580(rows)["probe_pass"]
