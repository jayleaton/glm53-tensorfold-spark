"""patches/0590: the ``fat2`` routed-expert kernels (``GLM53_TF_FAST_EXPERTS=fat2``, ``exl3_fat2.cu``): fat's arithmetic
in one persistent software-pipelined kernel a projection (256-thread CTAs, 128-member items from a device plan claimed
by ticket one item ahead, a 4-stage cp.async ring that runs across items, the epilogue in its own buffer).

It must equal fat / fast2 BIT FOR BIT, be row-independent (patches/0085) and deterministic. The offline proof is
``tests/test_fat2_emulator.py`` (lane-level ports, order- and position-sensitive mma models, planted mutations) and
``tests/test_fat2_compile.py`` (sm_121 resources, PTX dataflow of every stored value and mma operand == fat's, SASS
census); this file is the hardware side:

- host only: the env switch (``fat2`` is a fat-family mode; FAT2_CFG / FAT2_TICKET / FAT2_CTAS parsed and refused) and
  ``exl3_mm.routed``'s dispatch with fake extensions (fat2 on the shared input, fat when gate / up differ, fast2 inside
  auto's window or when the per-request knob says 0);
- GPU kernels: fat2 == fat == fast2 bit for bit (Xd and Y; configurations 0-2 each side; ticket on / off; CTA caps
  1 / 7 / 40; 64-8,192 rows, uniform and skewed; real per-rank shapes and a small shape with 3 column blocks and many
  passes); row independence (row subsets and permutations); repeatable (the ticket order varies run to run); the timing
  probes launch;
- GPU engine (synthetic EXL3 checkpoint, gate / up sign vectors equal as in the real one): committed state fat2 == fat
  (lean and not, 256-8,192-row chunks); drafted == serial and resumed == fresh with fat2; fat2 and fat resume each
  other's snapshots.

Run inside the image, under a timeout (a kernel bug could hang):
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout 900 pytest -q -x -s tests/cuda/test_fat2_patches.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    CUDA = False

gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")
needs_torch = pytest.mark.skipif(torch is None, reason="torch")

D, NI, E, TOP, LIMIT = 4096, 1024, 288, 8, 10.0
_ENVS = ("GLM53_TF_FAST_EXPERTS", "GLM53_TF_FAT2_CFG", "GLM53_TF_FAT2_TICKET", "GLM53_TF_FAT2_CTAS",
         "GLM53_TF_TC_CFG", "GLM53_TF_TC_CTAS")


# -- host only ---------------------------------------------------------------------------------------------------------
@needs_torch
def test_env_modes(monkeypatch):
    from tensorfold.families.glm5_next.cuda import exl3_mm

    try:
        for mode, fat, fat2 in (("fat2", True, True), ("FAT2", True, True), ("fat", True, False),
                                ("tc", True, False), ("fast2", False, False), (None, False, False)):
            for k in _ENVS:
                monkeypatch.delenv(k, raising=False)
            if mode is not None:
                monkeypatch.setenv("GLM53_TF_FAST_EXPERTS", mode)
            importlib.reload(exl3_mm)
            assert (exl3_mm.FAT, exl3_mm.FAT2, exl3_mm.FAT2_CFG, exl3_mm.FAT2_TICKET, exl3_mm.FAT2_CTAS) == \
                (fat, fat2, (0, 0), True, 0), mode
            assert exl3_mm.family() == int(fat)
        monkeypatch.setenv("GLM53_TF_FAST_EXPERTS", "fat2")
        monkeypatch.setenv("GLM53_TF_FAT2_CFG", "1,2")
        monkeypatch.setenv("GLM53_TF_FAT2_TICKET", "0")
        monkeypatch.setenv("GLM53_TF_FAT2_CTAS", "40")
        importlib.reload(exl3_mm)
        assert (exl3_mm.FAT2_CFG, exl3_mm.FAT2_TICKET, exl3_mm.FAT2_CTAS) == ((1, 2), False, 40)
        for bad in (("GLM53_TF_FAT2_CFG", "3,0"), ("GLM53_TF_FAT2_CFG", "1"), ("GLM53_TF_FAT2_CTAS", "-1")):
            monkeypatch.setenv(*bad)
            with pytest.raises(ValueError):
                importlib.reload(exl3_mm)
            monkeypatch.delenv(bad[0])
        # the multi-slot prefill (0560) keeps grouping pieces: fat2 picks nothing by the call's rows
        from tensorfold.families.glm5_next.cuda import mpf

        importlib.reload(exl3_mm)
        assert mpf.kernels_ok()
    finally:
        for k in _ENVS:
            monkeypatch.delenv(k, raising=False)
        importlib.reload(exl3_mm)


class _FakeExt:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def f(*a):
            self.calls.append((name, a))
        return f


@needs_torch
@pytest.mark.parametrize("fat,fat2,auto,shared", [(True, True, False, True), (True, True, False, False),
                                                  (False, True, False, True), (True, True, True, True),
                                                  (True, False, False, True)])
def test_routed_dispatch(monkeypatch, fat, fat2, auto, shared):
    """fat2 runs inside the fat family (tf_knobs.fat_experts = 1) on the one rotated input; fat when gate and up have
    different sign vectors; fast2 when the knob says 0 or auto's window holds the chunk."""

    from tensorfold.families.glm5_next.cuda import exl3_mm

    ext, fx, f2 = _FakeExt(), _FakeExt(), _FakeExt()
    monkeypatch.setattr(exl3_mm, "_ext", lambda: ext)
    monkeypatch.setattr(exl3_mm, "_fast_ext", lambda: fx)
    monkeypatch.setattr(exl3_mm, "_fat2_ext", lambda: f2)
    monkeypatch.setattr(exl3_mm, "V2", None)
    monkeypatch.setattr(exl3_mm, "STREAM", None)
    monkeypatch.setattr(exl3_mm, "FAT", fat)
    monkeypatch.setattr(exl3_mm, "FAT2", fat2)
    monkeypatch.setattr(exl3_mm, "TC", False)
    monkeypatch.setattr(exl3_mm, "ONCE", False)
    monkeypatch.setattr(exl3_mm, "AUTO", auto)
    monkeypatch.setattr(exl3_mm, "FAST2_ROWS", (0, 4096))
    monkeypatch.setattr(exl3_mm, "FAT_SHARED_X", True)
    monkeypatch.setattr(exl3_mm, "FAT2_CFG", (1, 2))
    monkeypatch.setattr(exl3_mm, "FAT2_TICKET", True)
    monkeypatch.setattr(exl3_mm, "FAT2_CTAS", 7)
    suh = torch.ones((2, 256), dtype=torch.float16)
    z = lambda *s: torch.zeros(s, dtype=torch.float16)            # noqa: E731
    ex = exl3_mm.Exl3Experts(torch.zeros(1), torch.zeros(1), torch.zeros(1), suh, suh.clone() if shared else -suh,
                             z(2, 128), z(2, 128), z(2, 128), z(2, 256), 2, 128, 256)
    s = NS(slots=3, rows=4, xg=z(12, 256), xu=z(12, 256), xd=z(12, 128))
    grp = NS(ids=torch.zeros(3, dtype=torch.int32), count=torch.ones(1, dtype=torch.int32),
             members=torch.full((3, 4), -1, dtype=torch.int32))
    x = torch.zeros((4, 256), dtype=torch.bfloat16)
    exl3_mm.routed(x, torch.zeros((4, 3), dtype=torch.int32), grp, ex, s, torch.zeros((12, 256)), 4, LIMIT, fast=True)
    names = [c[0] for c in ext.calls + fx.calls + f2.calls]
    if not fat:
        assert names == ["rot_in", "gateup", "down"]
    elif auto:
        assert names == ["rot_in1", "gateup", "down"]
    elif not fat2:
        assert names == ["rot_in1", "gateup_fat", "down_fat"]
    elif not shared:
        assert names == ["rot_in", "gateup_fat", "down_fat"]
    else:
        assert names == ["rot_in1", "gateup_fat2", "down_fat2"]
        gu, dn = f2.calls[0][1], f2.calls[1][1]
        assert gu[0] is s.xg and gu[-3:] == (1, True, 7)
        assert dn[0] is s.xd and dn[-3:] == (2, True, 7)


# -- GPU: kernels ----------------------------------------------------------------------------------------------------
def _layer(shared: bool = True, d=D, ni=NI, e=E, seed=5):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_patches import _exl3_layer

    ex = _exl3_layer(d, ni, e, seed=seed)
    if shared:
        ex.suh_u = ex.suh_g.clone()
    ex.__dict__.pop("_shared_suh", None)
    return ex


def _picks(rows: int, kind: str, d=D, e=E, top=TOP):
    x = (torch.randn((rows, d), generator=torch.Generator().manual_seed(6)) * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(7)
    w = torch.ones(e) if kind == "uniform" else 1.0 / torch.arange(1, e + 1).float() ** 0.8
    picks = torch.full((rows, top + 1), e, dtype=torch.int32)
    picks[:, :top] = torch.multinomial(w.expand(rows, e), top, replacement=False, generator=g).int()
    return x, picks


def _run(ex, x, picks, mode, *, cfg=(0, 0), ticket=True, ctas=0):
    """One MoE layer's routed experts (fast=True) with ``mode`` in fast2 / fat / fat2; (Y, Xd) of the routed slots."""

    sys.path.insert(0, str(Path(__file__).parent))
    from test_fastpf_patches import _fast_group

    from tensorfold.families.glm5_next.cuda import exl3_mm

    names = ("FAT", "ONCE", "AUTO", "TC", "FAT2", "FAT2_CFG", "FAT2_TICKET", "FAT2_CTAS", "FAT_SHARED_X", "FAT_STAGES")
    saved = {k: getattr(exl3_mm, k) for k in names}
    exl3_mm.FAT, exl3_mm.ONCE, exl3_mm.AUTO, exl3_mm.TC = mode in ("fat", "fat2"), False, False, False
    exl3_mm.FAT2, exl3_mm.FAT2_CFG, exl3_mm.FAT2_TICKET, exl3_mm.FAT2_CTAS = mode == "fat2", cfg, ticket, ctas
    exl3_mm.FAT_SHARED_X, exl3_mm.FAT_STAGES = True, 3
    try:
        n, slots = x.shape[0], picks.shape[1]
        scratch = exl3_mm.Scratch(n, slots, ex.dims, ex.width, "cuda")
        y = torch.zeros((n * slots, ex.dims), dtype=torch.float32, device="cuda")
        exl3_mm.routed(x.cuda(), picks.cuda(), _fast_group(picks, ex.count), ex, scratch, y, n, LIMIT, fast=True)
        torch.cuda.synchronize()
        top = slots - 1
        return (y.view(n, slots, ex.dims)[:, :top].cpu(), scratch.xd.view(n, slots, ex.width)[:, :top].cpu())
    finally:
        for k, v in saved.items():
            setattr(exl3_mm, k, v)


@gpu
@pytest.mark.parametrize("rows,kind", [(64, "uniform"), (300, "skewed"), (1024, "uniform"), (2048, "skewed"),
                                       (2048, "uniform"), (4096, "uniform"), (4160, "skewed"), (8192, "uniform"),
                                       (8192, "skewed")])
def test_fat2_equals_fat_bitwise(rows, kind):
    ex = _layer()
    x, picks = _picks(rows, kind)
    y2, xd2 = _run(ex, x, picks, "fast2")
    assert torch.isfinite(y2).all() and y2.abs().max() > 0
    yf, xdf = _run(ex, x, picks, "fat")
    assert torch.equal(yf, y2) and torch.equal(xdf, xd2), "fat != fast2 (0170 broken?)"
    for cfg in ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0)):
        for ticket, ctas in ((True, 0), (False, 0), (True, 7), (True, 40), (True, 1)):
            if ctas == 1 and rows > 2048:
                continue                                             # one CTA: minutes at 8,192 rows
            yt, xdt = _run(ex, x, picks, "fat2", cfg=cfg, ticket=ticket, ctas=ctas)
            assert torch.equal(xdt, xdf), ("Xd", cfg, ticket, ctas)
            assert torch.equal(yt, yf), ("Y", cfg, ticket, ctas)


@gpu
@pytest.mark.parametrize("rows", [5, 200, 700])
def test_fat2_odd_shapes(rows):
    """d = ni = 384: 3 column blocks (down: a block pair and a lone block); 20 experts, top 4: many passes."""

    ex = _layer(d=384, ni=384, e=20, seed=9)
    x, picks = _picks(rows, "skewed", d=384, e=20, top=4)
    yf, xdf = _run(ex, x, picks, "fat")
    for cfg in ((0, 0), (1, 1), (2, 2)):
        yt, xdt = _run(ex, x, picks, "fat2", cfg=cfg)
        assert torch.equal(xdt, xdf) and torch.equal(yt, yf), cfg


@gpu
@pytest.mark.parametrize("rows", [700, 4160])
def test_fat2_row_independent_and_repeatable(rows):
    ex = _layer()
    x, picks = _picks(rows, "skewed")
    y, xd = _run(ex, x, picks, "fat2")
    for _ in range(3):
        y2, xd2 = _run(ex, x, picks, "fat2")
        assert torch.equal(y, y2) and torch.equal(xd, xd2)                   # the ticket order varies between runs
    rng = np.random.default_rng(2)
    for sub in ([5], list(range(64, 200)), list(range(rows - 1, -1, -3)), list(rng.permutation(rows)),
                list(range(0, rows, 2))):
        s = torch.tensor(sub)
        ys, xds = _run(ex, x[s], picks[s], "fat2")
        assert torch.equal(ys, y[s]) and torch.equal(xds, xd[s]), sub[:3]


@gpu
def test_fat2_probes_run():
    """The timing probes (no decode / no mma) launch; their outputs are not checked."""

    sys.path.insert(0, str(Path(__file__).parent))
    from test_fastpf_patches import _fast_group

    from tensorfold.families.glm5_next.cuda import exl3_mm

    ex = _layer()
    f2 = exl3_mm._fat2_ext()
    for rows in (1024, 4160):
        x, picks = _picks(rows, "uniform")
        grp = _fast_group(picks, E)
        s = exl3_mm.Scratch(rows, TOP + 1, D, NI, "cuda")
        y = torch.zeros((rows * (TOP + 1), D), dtype=torch.float32, device="cuda")
        for probe in (1, 2):
            f2.gateup_fat2(s.xg, ex.gt, ex.ut, grp.ids, grp.count, grp.members, ex.svh_g, ex.svh_u, ex.suh_d, s.xd,
                           D, NI, TOP + 1, LIMIT, 0, True, 0, probe)
            f2.down_fat2(s.xd, ex.dt, grp.ids, grp.count, grp.members, ex.svh_d, y, NI, D, TOP + 1, 0, True, 0, probe)
        torch.cuda.synchronize()


# -- GPU: engine -----------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_TF_LOOKUP", "0")
    monkeypatch.setenv("GLM53_TF_NONEXPERT", "q4mse")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    if not CUDA:
        pytest.skip("CUDA only")
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_fat2590")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


def _engine(path, **kw):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_mia_prefill_patches import _engine as mia_engine

    return mia_engine(path, **kw)


@pytest.fixture(scope="module")
def ea(ckpt):
    return _engine(ckpt)


@pytest.fixture(scope="module")
def eb(ckpt):
    return _engine(ckpt, lean_block=256, rows_max=8192)


class _Mode:
    """Switch the process-wide expert kernels (what GLM53_TF_FAST_EXPERTS sets at import) for a block."""

    def __init__(self, mode: str):
        self.mode = mode

    def __enter__(self):
        from tensorfold.families.glm5_next.cuda import exl3_mm

        self.saved = (exl3_mm.FAT, exl3_mm.FAT2, exl3_mm.TC, exl3_mm.ONCE, exl3_mm.AUTO)
        exl3_mm.FAT, exl3_mm.FAT2 = self.mode in ("fat", "fat2"), self.mode == "fat2"
        exl3_mm.TC = exl3_mm.ONCE = exl3_mm.AUTO = False

    def __exit__(self, *exc):
        from tensorfold.families.glm5_next.cuda import exl3_mm

        exl3_mm.FAT, exl3_mm.FAT2, exl3_mm.TC, exl3_mm.ONCE, exl3_mm.AUTO = self.saved


@gpu
@pytest.mark.parametrize("n", [3, 65, 300, 1000])
def test_engine_state_fat2_equals_fat(ea, eb, n):
    from test_cindep_patches import _Variant, _same, _state

    prompt = list(np.random.default_rng(590 + n).integers(0, 1000, size=n))
    for eng, rows in ((ea, 1024), (eb, 8192), (eb, 256)):
        with _Variant(eng, rows):
            with _Mode("fat"):
                ref = _state(eng, prompt)
            with _Mode("fat2"):
                got = _state(eng, prompt)
            assert _same(ref, got), (rows, [i for i, (a, b) in enumerate(zip(ref, got)) if not torch.equal(a, b)])


@gpu
def test_engine_drafted_serial_and_resumed_fresh(eb):
    """With fat2 on (the knob fat_experts = 1 selects it while FAT2 is set): drafted == serial, resumed == fresh with
    other chunk sizes."""

    from test_cindep_patches import _cold, _gen, _sampling

    with _Mode("fat2"):
        knobs = {"fat_experts": 1}
        rng = np.random.default_rng(590)
        more = lambda n: [int(t) for t in rng.integers(0, 1000, size=n)]       # noqa: E731
        for sampling in ("sampled", "greedy"):
            s = _sampling(sampling)
            p1 = more(900)
            eb.cache = []
            r1, _ = _gen(eb, p1, s, policy="auto:1:1:0", knobs=dict(knobs, prefill_rows=8192))
            serial = _cold(eb, p1, s, knobs=dict(knobs, prefill_rows=1024))
            assert r1 == serial[:len(r1)], sampling                               # drafted == serial
            _gen(eb, p1, s, knobs=dict(knobs, prefill_rows=8192))
            p2 = p1 + r1 + more(80)
            warm, stats = _gen(eb, p2, s, knobs=dict(knobs, prefill_rows=64))
            assert stats["cached"] > 0
            assert warm == _cold(eb, p2, s, knobs=dict(knobs, prefill_rows=4096)), sampling   # resumed == fresh


@gpu
def test_fat2_shares_snapshots_with_fat(eb):
    """A snapshot written with fat2 resumes under fat (and back): the same bits, one snapshot tag."""

    from test_cindep_patches import _cold, _gen, _sampling

    s = _sampling("greedy")
    for first, second in (("fat2", "fat"), ("fat", "fat2")):
        rng = np.random.default_rng(591 if first == "fat2" else 592)
        p1 = [int(t) for t in rng.integers(0, 1000, size=700)]
        eb.cache = []
        with _Mode(first):
            r1, _ = _gen(eb, p1, s, knobs={"prefill_rows": 1024})
        p2 = p1 + r1 + [int(t) for t in rng.integers(0, 1000, size=50)]
        with _Mode(second):
            warm, stats = _gen(eb, p2, s, knobs={"prefill_rows": 1024})
        assert stats["cached"] > 0
        with _Mode(first):
            assert warm == _cold(eb, p2, s, knobs={"prefill_rows": 4096})
