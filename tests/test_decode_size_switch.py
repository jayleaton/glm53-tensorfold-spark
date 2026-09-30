"""patches/0570 on the CPU: the dense size switch (GLM53_TF_DEC_QMM_MAXMB) in ``decode_stream``.

- The knob: off by default (0 = today's _qmm everywhere); X > 0 turns 0440's dense hook on by itself, in MiB;
  GLM53_TF_DEC_QMM_MAX_MB is the same knob; bad values refuse to start; the Triton guard still applies (a Triton
  that does not fuse ``_qmm``'s epilogue keeps it off); GLM53_TF_DEC_QMM_EXCLUDE / _TABLE parse.
- Which per-rank decode shapes go to q4_stream.cu at 3.5 MiB (THEORY-2 item 7 / W16): KDA f_b / g_b, index k, DSA kv_b,
  shared down, index q_b and the 1024 x 4096 grid; never shared gate/up (4.72 MB) or anything larger. The table's row
  caps (index k <= 4 rows; DSA kv_b only with PDL), the W16 placements (with and without PDL), the table off, the
  exclude list.
- ``qmm.matmul``'s dispatch through the hook: a switched shape reaches run_qmm with its placement, a larger one returns
  to _qmm (None), and the size check is cached on the matrix.
- The session store's compat hash ignores the size knobs (same bits).

Bits: no new kernel -- q4_stream.cu has _qmm's bits for every shape and row count (0440: CPU emulator + W12 GPU
bitwise), so choosing it per (shape, rows) cannot change a row. Run:
    PYTHONPATH=<patched TensorFold>/src pytest -q tests/test_decode_size_switch.py
"""

from __future__ import annotations

import pytest

# imported in a fixture, not at collection: suites collected after this one (e.g. tests/test_memory_safety.py) set
# TRITON_INTERPRET at import, and Triton kernels defined before that would not run in their interpreter
torch = ds = qmm = None


@pytest.fixture(autouse=True, scope="module")
def _mods():
    global torch, ds, qmm
    torch = pytest.importorskip("torch")
    try:
        from tensorfold.families.glm5_next.cuda import decode_stream as _ds, qmm as _qmm
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"needs the patched tree on PYTHONPATH ({e})")
    if not hasattr(_ds, "small_shape"):
        pytest.skip("needs patches/0570")
    ds, qmm = _ds, _qmm

MIB = 1 << 20
# every per-rank decode q4 shape (W11 w11dec's manifest list + the two grids it could not name: (1, 16, 8) and
# (1, 48, 4)) and whether the 3.5 MiB switch takes it
SHAPES = {(77440, 4096): False, (12576, 4096): False, (12288, 4096): False, (4096, 8192): False, (4096, 6144): False,
          (4096, 4096): False, (8192, 1536): False, (3072, 4096): False, (2048, 4096): False, (4096, 1536): True,
          (4096, 1024): True, (8192, 512): True, (1024, 4096): True, (160, 4096): True, (4096, 128): True}
KEYS = ("GLM53_TF_DEC_QMM", "GLM53_TF_DEC_QMM_MAXMB", "GLM53_TF_DEC_QMM_MAX_MB", "GLM53_TF_DEC_QMM_TABLE",
        "GLM53_TF_DEC_QMM_EXCLUDE", "GLM53_TF_DEC_PDL")


def _q4(n, k):
    return qmm.make_q4(torch.zeros((n, k // 8), dtype=torch.int32), torch.zeros((n, k // 64), dtype=torch.bfloat16),
                       torch.zeros((n, k // 64), dtype=torch.bfloat16))


@pytest.fixture
def env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ds, "qmm_reference_fused", lambda: True)
    saved = dict(ds.CFG)
    yield monkeypatch
    ds.CFG.clear()
    ds.CFG.update(saved)
    ds.configure()


def test_knob_default_off(env):
    c = ds.parse()
    assert c["qmm"] is False and c["qmm_max_bytes"] == 0 and c["qmm_table"] is True and c["qmm_exclude"] == frozenset()


@pytest.mark.parametrize("name", ["GLM53_TF_DEC_QMM_MAXMB", "GLM53_TF_DEC_QMM_MAX_MB"])
def test_knob_turns_the_hook_on(env, name):
    env.setenv(name, "3.5")
    c = ds.parse()
    assert c["qmm"] is True and c["qmm_max_bytes"] == int(3.5 * MIB)


def test_knob_keeps_the_triton_guard(env, capsys):
    env.setattr(ds, "qmm_reference_fused", lambda: False)
    env.setenv("GLM53_TF_DEC_QMM_MAXMB", "3.5")
    assert ds.parse()["qmm"] is False
    assert "stays off" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["-1", "x", "nan", "inf"])
def test_bad_values_refuse(env, bad):
    env.setenv("GLM53_TF_DEC_QMM_MAXMB", bad)
    with pytest.raises(ValueError):
        ds.parse()


def test_exclude_and_table_parse(env):
    env.setenv("GLM53_TF_DEC_QMM_EXCLUDE", "1024x4096, 160x4096")
    env.setenv("GLM53_TF_DEC_QMM_TABLE", "0")
    c = ds.parse()
    assert c["qmm_exclude"] == frozenset({"1024x4096", "160x4096"}) and c["qmm_table"] is False
    env.setenv("GLM53_TF_DEC_QMM_EXCLUDE", "1024 x 4096")
    with pytest.raises(ValueError):
        ds.parse()


def test_which_shapes_at_3_5(env):
    ds.configure(qmm_on=True, qmm_max_mb=3.5, qmm_table=False)          # the size alone (the table's row caps below)
    taken = {}
    for (n, k), want in SHAPES.items():
        q = _q4(n, k)
        got = ds.small_shape(q, 1) is not None
        taken[f"{n}x{k}"] = (round(q.nbytes() / 1e6, 2), got)
        assert got == want, (n, k, q.nbytes())
    print("3.5 MiB switch:", {s: v for s, v in taken.items()})
    # the boundary: index q_b (3.54 MB = 3.375 MiB) in, shared gate/up (4.72 MB = 4.5 MiB) out
    assert _q4(4096, 1536).nbytes() <= 3.5 * MIB < _q4(2048, 4096).nbytes()


def test_off_means_every_shape(env):
    ds.configure(qmm_on=True, qmm_max_mb=0)
    for n, k in SHAPES:
        assert ds.small_shape(_q4(n, k), 16) == (f"{n}x{k}", None)          # 0440's behaviour: no size, no table


def test_row_cap_table_and_exclude(env):
    ds.configure(qmm_on=True, qmm_max_mb=3.5, qmm_table=True, pdl=False)
    q = _q4(160, 4096)
    assert ds.small_shape(q, 4)[1] == ds.SMALL_PLACE["160x4096"][False]
    assert ds.small_shape(q, 5) is None                      # index k loses from 8 rows cold: _qmm there
    kv = _q4(8192, 512)
    assert ds.small_shape(kv, 1) is None                     # DSA kv_b without PDL: 0.93-0.98x at 1-2 rows
    env.setattr(ds, "pdl_supported", lambda: True)
    ds.configure(pdl=True)
    assert ds.small_shape(kv, 16)[1] == ds.SMALL_PLACE["8192x512"][True]
    ds.configure(pdl=False)
    assert ds.small_shape(_q4(1024, 4096), 16) == ("1024x4096", None)     # unmeasured: 0440's CFG / auto
    ds.configure(qmm_table=False)
    assert ds.small_shape(q, 16) == ("160x4096", None)
    ds.configure(qmm_table=True)
    ds.CFG["qmm_exclude"] = frozenset({"4096x1024"})
    assert ds.small_shape(_q4(4096, 1024), 1) is None
    ds.CFG["qmm_exclude"] = frozenset()
    q2 = _q4(4096, 1024)
    assert ds.small_shape(q2, 1) is not None and q2.__dict__["_dec_small"][2] is True     # cached on the matrix


@pytest.mark.parametrize("pdl", [False, True])
def test_matmul_dispatch(env, pdl):
    ds.configure(qmm_on=True, qmm_max_mb=3.5, qmm_table=True, pdl=pdl)
    env.setattr(ds, "pdl_supported", lambda: True)
    env.setattr(ds, "qmm_ok", lambda *a, **k: True)
    calls = []

    def fake(x, q, xs=None, **kw):
        calls.append((f"{q.n}x{q.k}", kw.get("cfg"), kw.get("serial")))
        return "q4_stream"

    env.setattr(ds, "run_qmm", fake)
    for (n, k), want in SHAPES.items():
        calls.clear()
        x = torch.zeros((4, k), dtype=torch.bfloat16)
        got = ds.matmul(x, _q4(n, k))
        place = ds.SMALL_PLACE.get(f"{n}x{k}")
        if want and place is not None and place[pdl][3] < 4:
            assert got is None and not calls
            continue
        if not want:
            assert got is None and not calls, (n, k)          # qmm.matmul runs _qmm + _reduce
            continue
        assert got == "q4_stream"
        place = ds.SMALL_PLACE.get(f"{n}x{k}")
        if place is None:
            assert calls == [(f"{n}x{k}", None, None)]
        elif place[pdl][3] < 4:
            assert got is None and not calls                  # past the shape's row cap: _qmm
        else:
            gps, st, serial, _ = place[pdl]
            assert calls == [(f"{n}x{k}", (gps, st), serial)]


def test_placements_are_valid():
    for key, cols in ds.SMALL_PLACE.items():
        assert set(cols) == {False, True}
        for gps, st, serial, rows in cols.values():
            assert (gps, st) in ds.QMM_CFGS and isinstance(serial, bool) and 0 <= rows <= ds.QMM_ROWS


def test_session_compat_ignores_the_size_knobs(monkeypatch):
    from tensorfold.families.glm5_next.cuda import sessdisk

    monkeypatch.setenv("GLM53_TF_DEC_QMM_MAXMB", "3.5")
    monkeypatch.setenv("GLM53_TF_DEC_QMM_EXCLUDE", "1024x4096")
    assert not any(k.startswith("GLM53_TF_DEC_QMM") for k in sessdisk.knobs())
