"""patches/0580 (GLM53_TF_DEC_EXPERT_LOADS) and 0570 (GLM53_TF_DEC_QMM_MAXMB) on one GPU. Nothing may change a bit.

- 0580 (``exl3_ld.cu``): ``exl3_mm.routed`` with the knob on == off (exl3.cu's grouped_kernel), Xd and Y bit for bit,
  at the real per-rank shapes (4096 / 1024, 288 experts, top 8) and a synthetic one: windows of 1-16 rows, 4-slot mixes,
  and (with GLM53_TF_EXPERT_LOOP=0, so they reach grouped) skewed windows of 24 / 64 rows with 2-4 member tiles; every
  (ld, nt, pd) of ``expert_loads.CFGS`` for gate/up and down; eagerly and in CUDA graphs; every row of a window == the
  row alone; the shared slot untouched; the raw Z of one grouped launch == exl3.cu's element for element (the
  never-written elements too); repeatable run to run; the probes launch. PDL (GLM53_TF_DEC_EXPERT_LOADS_PDL): a late
  writer releases the launch at once and writes X / Xd ~0.3 ms later (NaN until then, DRAM kept busy by a side stream):
  Z == the plain launch's; routed with the PDL knob == off, eager and in a graph.
- 0570: ``qmm.matmul`` with the size switch (3.5 MiB) == off, bit for bit, for every per-rank decode shape (incl. the
  1024 x 4096 grid W16 did not measure), 1-16 / 24 / 44 / 64 rows, bf16 and fp32, PDL off / on; the shapes <= 3.5 MiB
  (and <= their row cap) run q4_stream.cu, the others _qmm. Skipped where the Triton is not 3.7.x.

Run inside the image (prod stopped, one GPU):
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_decode_loads_patches.py
The engine suites with both knobs on (read at import):
    GLM53_TF_DEC_EXPERT_LOADS=1 GLM53_TF_DEC_QMM_MAXMB=3.5 PYTHONPATH=... pytest -q tests/cuda/test_decode_patches.py \\
        tests/cuda/test_batch_parallel_patches.py tests/cuda/test_decode_stream_patches.py -k "windows or replies or resume"
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import decode_stream as ds, exl3_mm, expert_loads as el, qmm  # noqa: E402
from test_decode_stream_patches import _busy_side, _exl3, _group, _late, _mix, _picks, _poison, _q4, _windows  # noqa: E402

DEV = "cuda"
TOP_SHAPES = [(12576, 4096), (4096, 4096), (4096, 128), (2048, 4096), (8192, 1536), (8192, 512), (4096, 8192),
              (12288, 4096), (4096, 6144), (4096, 1024), (77440, 4096), (4096, 1536), (160, 4096), (1024, 4096),
              (3072, 4096)]
ROWS = list(range(1, 17)) + [24, 44, 64]
PDLS = [False] + ([True] if ds.pdl_supported() else [])


@pytest.fixture(autouse=True)
def _no_loop(monkeypatch):
    monkeypatch.setattr(exl3_mm, "LOOP", False)            # windows past 16 rows reach grouped (and so ld_kernel)
    monkeypatch.setattr(exl3_mm, "STREAM", None)            # 0440's whole-path hook would take routed first
    monkeypatch.setattr(exl3_mm, "V2", None)


def _routed(ex, x, pick, grp, s, R, top_k, *, on, cfg=None, graph=False):
    slots = top_k + 1
    y = torch.full((s.rows * slots, ex.dims), 3.0, dtype=torch.float32, device=DEV)

    def call():
        with el.using(on=on, gu=cfg, dn=cfg):
            exl3_mm.routed(x, pick, grp, ex, s, y, R, 10.0)

    s.xd.zero_()
    call()
    if graph:
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with el.using(on=on, gu=cfg, dn=cfg), torch.cuda.graph(cg):
            exl3_mm.routed(x, pick, grp, ex, s, y, R, 10.0)
        y.fill_(3.0)
        s.xd.zero_()
        cg.replay()
    torch.cuda.synchronize()
    return y.view(s.rows, slots, ex.dims)[:R].clone(), s.xd.view(s.rows, slots, ex.width)[:R].clone()


@pytest.mark.parametrize("dims", [(4096, 1024, 288, 8), (512, 256, 16, 4)], ids=["real", "synthetic"])
def test_loads_equal_exl3(dims):
    D, NI, E, top_k = dims
    ex = _exl3(E, D, NI, 3)
    s = exl3_mm.Scratch(64, top_k + 1, D, NI, DEV)
    for name, pick in _windows(E, top_k):
        R = pick.shape[0]
        grp = _group(pick, E, top_k)
        x = (torch.randn((R, D), generator=torch.Generator().manual_seed(R)) * 0.5).to(torch.bfloat16).to(DEV)
        pk = pick.to(DEV)
        ref_y, ref_xd = _routed(ex, x, pk, grp, s, R, top_k, on=False)
        assert torch.isfinite(ref_y[:, :top_k]).all()
        for cfg in el.CFGS:
            gsk, dsk = exl3_mm.GATEUP_CFG[2], exl3_mm.DOWN_CFG[2]
            if not (el.fits(D, NI, gsk, 4, cfg) and el.fits(NI, D, dsk, 4, cfg)):
                continue
            for graph in (False, True):
                y, xd = _routed(ex, x, pk, grp, s, R, top_k, on=True, cfg=cfg, graph=graph)
                assert torch.equal(y[:, :top_k], ref_y[:, :top_k]), (name, cfg, graph)
                assert torch.equal(xd[:, :top_k], ref_xd[:, :top_k]), (name, cfg, graph)
                assert (y[:, top_k] == 3.0).all()                      # the shared slot is left alone
        if R <= 8:
            for r in range(R):
                g1 = _group(pick[r:r + 1], E, top_k)
                alone, _ = _routed(ex, x[r:r + 1], pk[r:r + 1], g1, s, 1, top_k, on=True, cfg=el.DEFAULT)
                assert torch.equal(alone[0, :top_k], ref_y[r, :top_k]), (name, r)


def test_raw_z_equal_and_probes():
    """One gate/up and one down launch: Z of ld_kernel == exl3.cu's grouped over the whole buffer (NaN-filled first,
    so an element written by one and not the other shows up); repeatable; the probes launch."""

    D, NI, E, top_k = 4096, 1024, 288, 8
    ex = _exl3(E, D, NI, 4)
    pick = _mix((4, 4, 4, 4), E, top_k, 11)
    R, slots = pick.shape[0], top_k + 1
    P = R * slots
    grp = _group(pick, E, top_k)
    xg = (torch.randn((P, D), device=DEV) * 0.5).half()
    xu = (torch.randn((P, D), device=DEV) * 0.5).half()
    xd = (torch.randn((P, NI), device=DEV) * 0.5).half()
    ext = exl3_mm._ext()
    for mats, (x0, x1, t0, t1, K, N, sk) in ((2, (xg, xu, ex.gt, ex.ut, D, NI, exl3_mm.GATEUP_CFG[2])),
                                             (1, (xd, xd, ex.dt, ex.dt, NI, D, exl3_mm.DOWN_CFG[2]))):
        zr = torch.full((mats * sk * P * N,), float("nan"), device=DEV)
        ext.grouped(x0, x1, t0, t1, grp.ids, grp.count, grp.members, zr, mats, K, N, P, sk, slots, 8, 4)
        for cfg in el.CFGS:
            if not el.fits(K, N, sk, 4, cfg):
                continue
            for _ in range(2):
                z = torch.full_like(zr, float("nan"))
                el.run(x0, x1, t0, t1, grp, z, mats, K, N, P, sk, slots, cfg)
                torch.cuda.synchronize()
                assert torch.equal(z.view(torch.int32), zr.view(torch.int32)), (mats, cfg)
        for ld, probe in sorted(el.PROBES):
            el.run(x0, x1, t0, t1, grp, torch.empty_like(zr), mats, K, N, P, sk, slots, (ld, 8, 2), probe=probe)
    torch.cuda.synchronize()


@pytest.mark.skipif(not ds.pdl_supported(), reason="programmatic dependent launch needs sm_90+")
def test_pdl_waits_for_x():
    """GLM53_TF_DEC_EXPERT_LOADS_PDL: each grouped launch right after a "late writer" that releases its dependents at
    once, sleeps ~0.3 ms and only then writes the launch's X (Xg / Xu, then Xd), poisoned (NaN) until then, with a side
    stream keeping DRAM busy: Z must equal the plain launch's. A load of X before griddepcontrol.wait would see NaN."""

    D, NI, E, top_k = 4096, 1024, 288, 8
    ex = _exl3(E, D, NI, 21)
    slots = top_k + 1
    start, side = _busy_side()
    ext = exl3_mm._ext()
    gsk, dsk = exl3_mm.GATEUP_CFG[2], exl3_mm.DOWN_CFG[2]
    for R in (1, 4, 16):
        pick = _picks(R, E, top_k, 60 + R)
        grp = _group(pick, E, top_k)
        P = R * slots
        good_g = (torch.randn((P, D), device=DEV) * 0.5).half()
        good_u = (torch.randn((P, D), device=DEV) * 0.5).half()
        good_d = (torch.randn((P, NI), device=DEV) * 0.5).half()
        zg_ref = torch.full((2 * gsk * P * NI,), float("nan"), device=DEV)
        zd_ref = torch.full((dsk * P * D,), float("nan"), device=DEV)
        ext.grouped(good_g, good_u, ex.gt, ex.ut, grp.ids, grp.count, grp.members, zg_ref, 2, D, NI, P, gsk, slots, 8, 4)
        ext.grouped(good_d, good_d, ex.dt, ex.dt, grp.ids, grp.count, grp.members, zd_ref, 1, NI, D, P, dsk, slots, 8, 4)
        for cfg in (("nc", 8, 2), ("cpa", 8, 2), ("w32", 4, 4), ("nc", 8, 4)):
            for _ in range(3):
                xg, xu, xd = torch.empty_like(good_g), torch.empty_like(good_u), torch.empty_like(good_d)
                _poison(xg), _poison(xu), _poison(xd)
                zg = torch.full_like(zg_ref, float("nan"))
                zd = torch.full_like(zd_ref, float("nan"))
                start()
                _late().late(good_g, xg, good_u, xu, 300_000)
                el.run(xg, xu, ex.gt, ex.ut, grp, zg, 2, D, NI, P, gsk, slots, cfg, pdl=True)
                _late().late(good_d, xd, good_d[:1], torch.empty_like(good_d[:1]), 300_000)
                el.run(xd, xd, ex.dt, ex.dt, grp, zd, 1, NI, D, P, dsk, slots, cfg, pdl=True)
                torch.cuda.synchronize()
                side.synchronize()
                assert torch.equal(zg.view(torch.int32), zg_ref.view(torch.int32)), (R, cfg)
                assert torch.equal(zd.view(torch.int32), zd_ref.view(torch.int32)), (R, cfg)


def test_pdl_knob_equal_in_routed():
    """The PDL knob through exl3_mm.routed (eager and in a CUDA graph) == off."""

    if not ds.pdl_supported():
        pytest.skip("programmatic dependent launch needs sm_90+")
    D, NI, E, top_k = 4096, 1024, 288, 8
    ex = _exl3(E, D, NI, 5)
    s = exl3_mm.Scratch(16, top_k + 1, D, NI, DEV)
    for R in (1, 3, 16):
        pick = _picks(R, E, top_k, 70 + R)
        grp = _group(pick, E, top_k)
        x = torch.randn((R, D), device=DEV).to(torch.bfloat16)
        ref_y, ref_xd = _routed(ex, x, pick.to(DEV), grp, s, R, top_k, on=False)
        with el.using(pdl=True):
            for graph in (False, True):
                y, xd = _routed(ex, x, pick.to(DEV), grp, s, R, top_k, on=True, cfg=el.DEFAULT, graph=graph)
                assert torch.equal(y[:, :top_k], ref_y[:, :top_k]) and torch.equal(xd[:, :top_k], ref_xd[:, :top_k])


# -- 0570 ------------------------------------------------------------------------------------------------------------------
def _need_fused():
    if not ds.qmm_reference_fused():
        pytest.skip("this Triton leaves some of _qmm's epilogues unfused: the streaming dense kernel stays off")


@pytest.mark.parametrize("n,k", TOP_SHAPES)
def test_size_switch_equals_qmm(n, k, monkeypatch):
    _need_fused()
    q = _q4(n, k, n + k)
    part = torch.empty((8 * 64 * max(n, 16384) * 2,), dtype=torch.float32, device=DEV)
    ran = []
    real = ds.run_qmm
    monkeypatch.setattr(ds, "run_qmm", lambda *a, **kw: (ran.append(1), real(*a, **kw))[1])
    small = q.nbytes() <= 3.5 * (1 << 20)
    for m in ROWS:
        x = torch.randn((m, k), device=DEV, generator=torch.Generator(device=DEV).manual_seed(m)).to(torch.bfloat16)
        xs = qmm.group_sums(x)
        for f32 in (False, True):
            with ds.using(qmm_on=False, qmm_max_mb=0):
                ref = qmm.matmul(x, q, xs, f32=f32, part=part).clone()
            for pdl in PDLS:
                ran.clear()
                with ds.using(qmm_on=True, qmm_max_mb=3.5, pdl=pdl):
                    got = qmm.matmul(x, q, xs, f32=f32, part=part)
                    torch.cuda.synchronize()
                    taken = ds.small_shape(q, m) is not None
                assert torch.equal(got, ref), (n, k, m, f32, pdl)
                assert bool(ran) == (taken and small), (n, k, m, taken)
    assert not ds.counters(part).any()
