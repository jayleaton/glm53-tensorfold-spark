"""patches/0590 on the CPU, no GPU: ``exl3_fat2.cu`` (GLM53_TF_FAST_EXPERTS=fat2) gives ``exl3_fast.cu``'s fat bits, in
the emulation of ``tests/fat2_emu.py`` (numpy; see its docstring for what is modelled).

- fat's lane-level port == the matrix-level reference (so the reference is fat's arithmetic, not an idea of it): gate/up
  and down, 3 / 4 stages, ticket and static stride, uniform and skewed routing, 1-300-row windows, the >= 4,096-row
  down configuration;
- fat2's lane-level port == the reference, bit for bit, every output written exactly once and nothing else written:
  the three configurations (128 members / 4 stages, 64 / 4, 128 / 3), 1-7 CTAs, ticket and static stride, 1-300-row
  windows (1-3 passes an expert, a last pass of any size), odd column-block counts (down's missing half), stale grouping
  entries past the count, the shared expert's id, the real per-rank shapes (4,096 / 1,024, 288-expert ids);
- under a position-sensitive mma model (an element's summation order depends on its place in the 16 x 8 tile) fat2 ==
  fat still (the reference no longer applies): fat2 keeps every element at fat's tile position;
- row independence: a window's rows alone (any sub-window) == the same rows in the window;
- negative controls: every mutation in ``fat2_emu.MUTATIONS`` (ring slot, wait depth, swizzle, k order, the next item's
  rows / stage / claim, accumulators kept, a member group skipped, epilogue rows swapped, down's column block, a lost
  ticket) changes the bits or the write counts; the neutral one (no zero-fill past the last member) does not.

Run: pytest -q tests/test_fat2_emulator.py        (numpy + pytest; ~9 minutes; FAT2_EMU_REAL=0 skips the real
shapes, which take ~5 minutes of it)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import fat2_emu as F  # noqa: E402

D, NI, E, TOP = 256, 256, 6, 3
SLOTS = TOP + 1
LIMIT = 10.0
CFGS = [(16, 4), (8, 4), (16, 3)]                           # exl3_fat2.cu FAT2_CFG0 / 1 / 2


def _case(rows, kind="uniform", seed=0, d=D, ni=NI, e=E, top=TOP, stale=False):
    rng = np.random.default_rng(seed)
    L = F.Layer(d, ni, e, rng)
    picks = F.make_picks(rows, e, top, rng, kind)
    grp = F.Group(picks, e, spare=2 if stale else 1, stale=rng if stale else None)
    X = F.rand_x(rows * (top + 1), d, rng)
    return L, grp, X, rng


def _refs(L, grp, X):
    xd = F.reference_gateup(X, L.gt, L.ut, grp, L.svh_g, L.svh_u, L.suh_d, LIMIT)
    y = F.reference_down(xd, L.dt, grp, L.svh_d)
    return xd, y


def _gu(port, L, grp, X, rng, **kw):
    return port(0, X, L.gt, L.ut, grp, L.NI, L.svh_g, L.svh_u, L.suh_d, LIMIT, rng, **kw)


def _dn(port, L, grp, XD, rng, **kw):
    return port(1, XD, L.dt, L.dt, grp, L.D, L.svh_d, L.svh_d, L.svh_d, 0.0, rng, **kw)


def _written_once(out, grp):
    live = F.pair_rows(grp)
    n = out.n
    dead = np.ones(n.shape[0], bool)
    dead[live] = False
    return bool((n[live] == 1).all() and (n[dead] == 0).all())


@pytest.mark.parametrize("rows,kind", [(1, "uniform"), (8, "skewed"), (40, "uniform"), (130, "skewed")])
@pytest.mark.parametrize("nsa,ticket", [(3, True), (4, False)])
def test_fat_port_is_the_reference(rows, kind, nsa, ticket):
    L, grp, X, rng = _case(rows, kind, seed=rows)
    xd, y = _refs(L, grp, X)
    a = _gu(F.fat_port, L, grp, X, rng, NSA=nsa, ticket=ticket, grid=3)
    b = _dn(F.fat_port, L, grp, xd, rng, NSA=nsa, ticket=ticket, grid=2)
    assert F.same(a, xd) and _written_once(a, grp)
    assert F.same(b, y) and _written_once(b, grp)


@pytest.mark.parametrize("cfg", CFGS)
@pytest.mark.parametrize("rows,kind", [(1, "uniform"), (8, "skewed"), (40, "uniform"), (130, "skewed"),
                                       (300, "uniform")])
def test_fat2_is_the_reference(cfg, rows, kind):
    ng, nsa = cfg
    L, grp, X, rng = _case(rows, kind, seed=100 + rows)
    xd, y = _refs(L, grp, X)
    a = _gu(F.fat2_port, L, grp, X, rng, NG=ng, NSA=nsa, grid=3)
    b = _dn(F.fat2_port, L, grp, xd, rng, NG=ng, NSA=nsa, grid=4)
    assert F.same(a, xd) and _written_once(a, grp)
    assert F.same(b, y) and _written_once(b, grp)


@pytest.mark.parametrize("grid,ticket", [(1, True), (2, False), (5, True), (7, False)])
def test_fat2_any_grid_and_walk(grid, ticket):
    L, grp, X, rng = _case(90, "skewed", seed=7 + grid)
    xd, y = _refs(L, grp, X)
    for cfg in CFGS[:2]:
        a = _gu(F.fat2_port, L, grp, X, rng, NG=cfg[0], NSA=cfg[1], grid=grid, ticket=ticket)
        b = _dn(F.fat2_port, L, grp, xd, rng, NG=cfg[0], NSA=cfg[1], grid=grid, ticket=ticket)
        assert F.same(a, xd) and _written_once(a, grp)
        assert F.same(b, y) and _written_once(b, grp)


def test_fat2_odd_column_blocks_and_stale_entries():
    """3 column blocks (gate/up: 3 items a pass; down: a pair and a lone block whose second warp half idles) and
    grouping entries past the count holding garbage."""

    L, grp, X, rng = _case(70, "skewed", seed=3, d=384, ni=384, stale=True)
    xd, y = _refs(L, grp, X)
    for ng, nsa in CFGS:
        a = _gu(F.fat2_port, L, grp, X, rng, NG=ng, NSA=nsa, grid=3)
        b = _dn(F.fat2_port, L, grp, xd, rng, NG=ng, NSA=nsa, grid=3)
        assert F.same(a, xd) and _written_once(a, grp)
        assert F.same(b, y) and _written_once(b, grp)
    assert F.same(_dn(F.fat_port, L, grp, xd, rng), y)


def test_fat_large_down_and_fat2_many_passes():
    """A >= 4,096-row window: fat's down runs FAT_DN_LARGE (128 members, 2 k tiles a stage); experts of ~2,700 members
    are 22 / 43 passes of fat2's 128 / 64 members."""

    rows = 4096
    L, grp, X, rng = _case(rows, "skewed", seed=11, d=256, ni=256, e=3, top=2)
    xd = F.reference_gateup(X, L.gt, L.ut, grp, L.svh_g, L.svh_u, L.suh_d, LIMIT)
    y = F.reference_down(xd, L.dt, grp, L.svh_d)
    assert grp.members.shape[1] >= 4096
    assert F.same(_dn(F.fat_port, L, grp, xd, rng, grid=4), y)
    for ng in (16, 8):
        b = _dn(F.fat2_port, L, grp, xd, rng, NG=ng, NSA=4, grid=5)
        assert F.same(b, y) and _written_once(b, grp)
    a = _gu(F.fat2_port, L, grp, X, rng, NG=16, NSA=4, grid=5)
    assert F.same(a, xd) and _written_once(a, grp)


def test_fat2_equals_fat_under_a_position_sensitive_mma():
    L, grp, X, rng = _case(150, "skewed", seed=21)
    F.MMA_MODEL = "position"
    try:
        fa = _gu(F.fat_port, L, grp, X, rng)
        for ng, nsa in CFGS:
            assert F.same(_gu(F.fat2_port, L, grp, X, rng, NG=ng, NSA=nsa), fa)
        fd = _dn(F.fat_port, L, grp, fa.a, rng)
        for ng, nsa in CFGS:
            assert F.same(_dn(F.fat2_port, L, grp, fa.a, rng, NG=ng, NSA=nsa), fd)
        xd, _ = _refs(L, grp, X)
        assert not F.same(fa, xd)                        # the model does change the bits: the check has teeth
    finally:
        F.MMA_MODEL = "order"


@pytest.mark.parametrize("lo,hi", [(0, 1), (3, 17), (40, 99), (98, 99)])
def test_fat2_rows_alone_equal_rows_in_the_window(lo, hi):
    """patches/0085's row independence: the rows [lo, hi) of a 99-row window as their own window (a chunk boundary,
    a multi-prefill member) give the bits they have in the window."""

    rows = 99
    L, grp, X, rng = _case(rows, "skewed", seed=31)
    full = _gu(F.fat2_port, L, grp, X, rng)
    fy = _dn(F.fat2_port, L, grp, full.a, rng)
    sub_picks = np.full((hi - lo, SLOTS), E, np.int32)       # the case's picks of rows [lo, hi), from its grouping
    for u in range(int(grp.count[0])):
        for c in grp.members[u][grp.members[u] >= 0]:
            r, s = int(c) >> 5, int(c) & 31
            if lo <= r < hi:
                sub_picks[r - lo, s] = grp.ids[u]
    sg = F.Group(sub_picks, E)
    Xs = X[lo * SLOTS:hi * SLOTS]
    a = _gu(F.fat2_port, L, sg, Xs, rng)
    b = _dn(F.fat2_port, L, sg, a.a, rng)
    assert np.array_equal(a.a, full.a[lo * SLOTS:hi * SLOTS])
    assert np.array_equal(b.a.view(np.uint32), fy.a[lo * SLOTS:hi * SLOTS].view(np.uint32))


@pytest.mark.skipif(os.environ.get("FAT2_EMU_REAL", "1") == "0", reason="FAT2_EMU_REAL=0")
def test_fat2_real_shapes():
    """The per-rank shapes: gate/up K 4,096 (128 stages an item), N 1,024 (8 blocks); down K 1,024, N 4,096 (16 block
    pairs); 288 experts (a few picked); fat2 == fat == the reference."""

    rng = np.random.default_rng(41)
    d, ni, e, top, rows = 4096, 1024, 288, 8, 3
    L = F.Layer(d, ni, e, rng)
    picks = np.full((rows, top + 1), e, np.int32)
    few = np.array([5, 77, 200, 287])
    for r in range(rows):
        picks[r, :top] = rng.permutation(np.concatenate([few, rng.choice(np.setdiff1d(np.arange(e), few), 4,
                                                                         replace=False)]))[:top]
    grp = F.Group(picks, e)
    X = F.rand_x(rows * (top + 1), d, rng)
    xd, y = _refs(L, grp, X)
    a = _gu(F.fat2_port, L, grp, X, rng, grid=6)
    assert F.same(a, xd) and _written_once(a, grp)
    b = _dn(F.fat2_port, L, grp, xd, rng, grid=6)
    assert F.same(b, y) and _written_once(b, grp)
    assert F.same(_dn(F.fat_port, L, grp, xd, rng, grid=6), y)


@pytest.mark.parametrize("mutation", sorted(F.MUTATIONS))
def test_mutations_are_caught(mutation):
    L, grp, X, rng = _case(40, "skewed", seed=51, d=384, ni=256)
    xd, y = _refs(L, grp, X)
    ok = []
    for ng in (16, 8):
        a = _gu(F.fat2_port, L, grp, X, rng, NG=ng, NSA=4, grid=3, mutate=mutation)
        b = _dn(F.fat2_port, L, grp, xd, rng, NG=ng, NSA=4, grid=3, mutate=mutation)
        ok.append(F.same(a, xd) and _written_once(a, grp) and F.same(b, y) and _written_once(b, grp))
    assert not any(ok), f"{mutation}: {F.MUTATIONS[mutation]} was not caught"


@pytest.mark.parametrize("mutation", sorted(F.NEUTRAL))
def test_neutral_mutation_keeps_the_bits(mutation):
    L, grp, X, rng = _case(40, "skewed", seed=52, d=384, ni=256)
    xd, y = _refs(L, grp, X)
    a = _gu(F.fat2_port, L, grp, X, rng, mutate=mutation)
    b = _dn(F.fat2_port, L, grp, xd, rng, mutate=mutation)
    assert F.same(a, xd) and F.same(b, y)


def test_plan_model_items():
    """fast2's plan (the kernel's binary search for the first -1, the prefix sum): passes of BM members an expert,
    none for ids >= E or entries past the count."""

    L, grp, X, rng = _case(300, "skewed", seed=61, stale=True)
    for bm in (64, 128):
        plan = F.plan_model(grp.ids, grp.count, grp.members, E, bm)
        n = [int((grp.members[u] >= 0).sum()) if u < grp.count[0] and grp.ids[u] < E else 0
             for u in range(grp.members.shape[0])]
        items = [-(-c // bm) for c in n]
        assert plan[0] == sum(items)
        assert list(plan[2:]) == list(np.cumsum(items))
