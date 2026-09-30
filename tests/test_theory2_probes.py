"""THEORY-2 session probes, CPU side: the pure helpers of results/THEORY2-SESSION/probes/glprobe.py (item 2, graph
launch delay) and tests/cuda/bench_decode_cold.py (item 7, cold dense re-bench), plus their Triton kernels compiled
for sm_121 without a GPU (skipped when ptxas cannot target it or under TRITON_INTERPRET).

    PYTHONPATH=<patched TensorFold>/src:tests/cuda pytest -q tests/test_theory2_probes.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


glp = _load("glprobe", ROOT / "results" / "THEORY2-SESSION" / "probes" / "glprobe.py")
cold = _load("bench_decode_cold", ROOT / "tests" / "cuda" / "bench_decode_cold.py")


# -- glprobe ----------------------------------------------------------------------------------------------------------------
def test_stats():
    s = glp.stats(list(range(1, 101)))
    assert s["n"] == 100 and s["median"] == 50.5
    assert math.isclose(s["p10"], 10.9) and math.isclose(s["p90"], 90.1)
    assert glp.stats([None, float("nan")])["median"] is None
    assert glp.stats([7])["p10"] == 7
    c = cold.stats([3, 1, 2])
    assert c["median"] == 2 and c["n"] == 3


def test_first_node_delay():
    # eager stamp at GPU 1,000,000 ns; node 0 at 1,500,000; host: eager returned at 10,000, replay called at 60,000
    assert glp.first_node_delay_us(1_000_000, 1_500_000, 10_000, 60_000) == pytest.approx(450.0)
    assert glp.first_node_delay_us(0, 5_000, 0, 0) == pytest.approx(5.0)


def test_chain_analysis():
    se, tb = 1_000_000, 0
    starts = [1_100_000, 1_400_000]
    ends = [1_300_000, 1_900_000]
    calls, rets = [20_000, 90_000], [80_000, 120_000]
    r = glp.chain_analysis(se, starts, ends, tb, calls, rets)
    assert r["first_delay"] == pytest.approx(80.0)            # 100 us on the GPU - 20 us of host
    assert r["piece_delay"][1] == pytest.approx(310.0)        # 400 - 90
    assert r["host"] == pytest.approx([60.0, 30.0])
    assert r["gap"] == pytest.approx([0.0, 100.0])
    assert r["span"] == pytest.approx(800.0)
    assert r["end_to_end"] == pytest.approx(880.0)
    assert r["host_total"] == pytest.approx(100.0)
    single = glp.chain_analysis(se, starts[:1], ends[:1], tb, calls[:1], rets[:1])
    assert single["gap"] == [0.0]
    with pytest.raises(ValueError):
        glp.chain_analysis(se, starts, ends[:1], tb, calls, rets)


def test_plan_nodes():
    for n in glp.SIZES:
        for v in ("mix", "tiny"):
            p = glp.plan_nodes(n, v)
            assert len(p) == n and p[0] == "stamp" and p[-1] == "stamp" and p.count("stamp") == 2
    mix = glp.plan_nodes(1650, "mix")
    streams = mix.count("stream")
    assert 180 <= streams <= 185                                   # ~1/9: ~50 ms of 64 MB reads at 235 GB/s
    assert 40 <= streams * glp.STREAM_MB * 1.048576e6 / 235e9 * 1e3 <= 60
    assert "stream" not in glp.plan_nodes(1650, "tiny")
    assert glp.plan_nodes(0, "calib") == ["stamp", "stamp"]
    with pytest.raises(ValueError):
        glp.plan_nodes(1, "mix")


def test_split():
    assert glp.split_sizes(10, 3) == [4, 3, 3]
    assert sum(glp.split_sizes(1648, 8)) == 1648
    with pytest.raises(ValueError):
        glp.split_sizes(2, 3)
    for k in glp.SPLITS:
        pieces = glp.split_plan(1650, "mix", k)
        assert len(pieces) == k
        assert all(p[0] == "stamp" and p[-1] == "stamp" for p in pieces)
        body = [x for p in pieces for x in p[1:-1]]
        assert body == glp.plan_nodes(1650, "mix")[1:-1]          # the same work, cut in order
        assert sum(len(p) for p in pieces) == 1650 - 2 + 2 * k


def test_gate_item2():
    ok, line = glp.gate_item2(1105.4)
    assert ok and line.startswith("GATE item2: uncaptured first-node delay at ~1650 nodes = 1105 us") \
        and line.endswith("PASS")
    ok, line = glp.gate_item2(250.0)
    assert not ok and line.endswith("FAIL")
    assert glp.gate_item2(400.0)[0]
    ok, line = glp.gate_item2(None)
    assert not ok and "n/a" in line
    assert "informational" in glp.gate_item2(900, informational=True)[1]


def test_split_saving():
    single = {"first_delay": {"median": 900.0}, "end_to_end": {"median": 50_900.0}}
    piece = {"first_delay": {"median": 80.0}, "end_to_end": {"median": 50_200.0}}
    s = glp.split_saving(single, piece)
    assert s == {"expected_us": 820.0, "end_to_end_us": 700.0}
    assert glp.split_saving(None, piece) is None
    assert glp.split_saving({"first_delay": {}}, piece) is None


def test_under_profiler(monkeypatch):
    for k in list(os.environ):
        if k.startswith("NSYS_") or k in ("CUDA_INJECTION64_PATH", "LD_PRELOAD"):
            monkeypatch.delenv(k, raising=False)
    assert not glp.under_profiler()
    monkeypatch.setenv("CUDA_INJECTION64_PATH", "/opt/nvidia/nsight-systems/lib/libToolsInjection64.so")
    assert glp.under_profiler()


def test_as_ptr_and_fmt():
    assert glp._as_ptr(12345) == 12345
    assert glp._as_ptr(None) is None
    assert "n/a" in glp.fmt_stat(None)
    assert glp.fmt_stat(glp.stats([1, 2, 3])).strip().startswith("2")


# -- bench_decode_cold --------------------------------------------------------------------------------------------------------
SHAPES = [(12576, 4096), (4096, 4096), (4096, 128), (2048, 4096), (8192, 1536), (8192, 512), (4096, 8192),
          (12288, 4096), (4096, 6144), (4096, 1024), (77440, 4096)] + cold.EXTRA_SHAPES


def test_copies_for_rotation():
    assert cold.ROT_MIN_BYTES == 96 * 1024 * 1024
    for n, k in SHAPES:
        nb = cold.q4_nbytes(n, k)
        c = cold.copies_for_rotation(nb)
        assert c >= cold.HOT_COPIES and c * nb >= cold.ROT_MIN_BYTES
        assert (c - 1) * nb < cold.ROT_MIN_BYTES or c == cold.HOT_COPIES   # no more than needed
    assert cold.copies_for_rotation(cold.q4_nbytes(4096, 128)) == 342    # "hundreds of copies"
    assert cold.copies_for_rotation(cold.q4_nbytes(77440, 4096)) == 3
    assert cold.copies_for_rotation(cold.q4_nbytes(12576, 4096)) == 4
    assert cold.copies_for_rotation(1, max_copies=100) == 100
    with pytest.raises(ValueError):
        cold.copies_for_rotation(0)


def test_q4_nbytes_matches_qmm():
    torch = pytest.importorskip("torch")
    try:
        from tensorfold.families.glm5_next.cuda import qmm
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tensorfold not importable: {e}")
    for n, k in ((4096, 128), (160, 4096), (256, 512)):
        words = torch.zeros((n, k // 8), dtype=torch.int32)
        scales = torch.ones((n, k // 64), dtype=torch.bfloat16)
        try:
            q = qmm.make_q4(words, scales, scales)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"make_q4 needs a GPU here: {e}")
        assert q.nbytes() == cold.q4_nbytes(n, k)
    assert cold.q4_nbytes(4096, 4096) / 1e6 == pytest.approx(9.44, abs=0.01)   # KDA o 9.4 MB (DECODE-KERNELS)


def test_flush_and_roof():
    assert cold.flush_call_us([130.0, 128.0, 131.0], [100.0, 101.0, 100.0], 10) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        cold.flush_call_us([1.0], [], 1)
    assert cold.roof_valid(235.0) and cold.roof_valid(240.0)
    assert not cold.roof_valid(311.0) and not cold.roof_valid(None)
    assert cold.gbs(235_000, 1.0) == pytest.approx(235.0)
    assert cold.gbs(1, 0) is None
    assert cold.geomean([1.21, 1.0]) == pytest.approx(1.1)
    assert cold.geomean([None]) is None


def _row(shape, rows, old, news, roof=230.0, mode="rotate"):
    r = {"mode": mode, "shape": shape, "rows": rows, "MB": cold.q4_nbytes(*map(int, shape.split("x"))) / 1e6,
         "old_us": old, "roof_GBs": roof, "roof_valid": cold.roof_valid(roof), "copies": 11, "rotation_MB": 100.0}
    for i, (us, same) in enumerate(news):
        r[f"new (1, {4 + i}) split"] = {"us": us, "GBs": 1.0, "same_bits": same}
    return cold.finish_row(r)


def test_best_eligible():
    r = _row("4096x4096", 1, 50.0, [(30.0, False), (40.0, True), (45.0, True)])
    assert r["best_new"] == "new (1, 5) split" and r["best_us"] == 40.0 and r["speedup"] == pytest.approx(1.25)
    r = _row("4096x4096", 1, 50.0, [(30.0, False)])
    assert r["best_new"] is None and r["speedup"] is None
    r["new x"] = {"us": None, "same_bits": None, "error": "boom"}
    assert cold.best_eligible(r) is None
    assert "no eligible" in cold.fmt_row(r)
    assert "1.250x" in cold.fmt_row(_row("4096x4096", 1, 50.0, [(40.0, True)]))
    assert "INVALID" in cold.fmt_row(_row("4096x4096", 1, 50.0, [(40.0, True)], roof=311.0))
    assert "ERROR" in cold.fmt_row({"mode": "rotate", "shape": "1x1", "rows": 1, "MB": 0.1, "old_us": None,
                                    "error": "x"})


def test_gate_item7():
    res = {"rotate": [], "flush": [], "hot": []}
    for s in cold.GATE_SHAPES:
        for m in (1, 8):
            fast = s in ("2048x4096", "4096x1024")
            res["rotate"].append(_row(s, m, 12.0, [(10.0 if fast else 11.5, True)]))
            res["flush"].append(_row(s, m, 12.0, [(10.5 if fast else 11.9, True)], mode="flush"))
            res["hot"].append(_row(s, m, 12.0, [(5.0, True)], roof=900.0, mode="hot"))   # never counts
    g = cold.gate_item7(res)
    assert g["modes"] == ["rotate", "flush"]
    assert g["passing"] == ["2048x4096", "4096x1024"]
    assert "4096x4096" in g["failing"] and not g["unmeasured"]
    assert g["line"].startswith("GATE item7: new >= 1.10x old cold on shapes < 8 MB (KDA o, shared gate/up/down, "
                                "DSA q_b/kv, index)")
    assert "2048x4096 (shared gate/up)" in g["line"]
    # a shape that passes rotate but fails flush fails
    res["flush"] = [r for r in res["flush"] if r["shape"] != "4096x1024"] + \
        [_row("4096x1024", m, 12.0, [(11.8, True)], mode="flush") for m in (1, 8)]
    assert cold.gate_item7(res)["passing"] == ["2048x4096"]
    # L2-resident (invalid roof) rows are ignored: all invalid -> not measured
    res2 = {"rotate": [_row("8192x512", 1, 12.0, [(6.0, True)], roof=728.0)]}
    g2 = cold.gate_item7(res2)
    assert "8192x512" in g2["unmeasured"] and not g2["passing"]
    # rotate only
    g3 = cold.gate_item7({"rotate": [_row("8192x512", 1, 12.0, [(10.0, True)])]})
    assert g3["modes"] == ["rotate"] and g3["passing"] == ["8192x512"]


def test_suggest_max_mb():
    rows = [_row("160x4096", 1, 3.0, [(2.0, True)]), _row("4096x1024", 1, 12.0, [(10.0, True)]),
            _row("2048x4096", 1, 22.0, [(18.0, True)]), _row("4096x4096", 1, 45.0, [(44.0, True)]),
            _row("12576x4096", 1, 140.0, [(120.0, True)])]
    mb = cold.suggest_max_mb({"rotate": rows})
    assert mb == pytest.approx(cold.q4_nbytes(2048, 4096) / 1e6)       # stops at the first failing size (KDA o)
    assert cold.suggest_max_mb({"rotate": [_row("160x4096", 1, 3.0, [(2.9, True)])]}) is None


# -- Triton kernels compile for sm_121 (no GPU) ------------------------------------------------------------------------------
def _compile(fn, sig: dict, cst: dict, warps: int = 4):
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("compiles kernels: run without TRITON_INTERPRET")
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    sig = dict(sig)
    for k in cst:
        sig[k] = "constexpr"
    try:
        return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cst), target=GPUTarget("cuda", 121, 32),
                              options={"num_warps": warps})
    except Exception as exc:  # noqa: BLE001
        if "ptxas" in str(exc).lower() or "not found" in str(exc).lower():
            pytest.skip(f"cannot compile for sm_121 here: {exc}")
        raise


def test_glprobe_kernels_compile():
    pytest.importorskip("triton")
    tk = glp.triton_kernels()
    k = _compile(tk["stamp"], {"OUT": "*i64", "slot": "i32"}, {}, 1)
    assert "%globaltimer" in k.asm["ptx"]
    _compile(tk["stream"], {"P": "*i32", "OUT": "*i32", "n": "i32"}, {"BLOCK": 2048, "ITERS": 8}, 8)
    _compile(tk["glue"], {"X": "*bf16", "W": "*bf16", "Y": "*bf16"}, {"N": 4096, "EPS": 1e-6})
    _compile(tk["wide"], {"X": "*bf16", "Y": "*bf16", "n": "i32"}, {"BLOCK": 1024})


def test_cold_pred_kernel_compiles():
    pytest.importorskip("triton")
    k = _compile(cold.pred_kernel(), {"P": "*i32", "OUT": "*i32", "n": "i32"}, {"BLOCK": 2048, "ITERS": 8}, 8)
    ptx = k.asm["ptx"]
    assert "ld.global" in ptx and ".cs" not in ptx.split("ld.global", 1)[1][:8]   # plain loads: they allocate in L2


# -- the scripts' main() end to end on fake GPU layers (report / summary / JSON code paths) --------------------------------
def test_glprobe_main_mocked(monkeypatch, tmp_path, capsys):
    import types

    class FakeK:
        pass

    def fake_case(K, name, plans, reps, warm, first_reps):
        k = len(plans)
        n = sum(len(p) for p in plans)
        delay = 5.0 + 0.6 * len(plans[0])                         # first piece's launch cost ~ its node count
        reps_ = []
        for i in range(reps):
            starts, ends, calls, rets, t = [], [], [], [], 1_000
            for j, p in enumerate(plans):
                calls.append(t)
                t += int(0.6e3 * len(p))
                rets.append(t)
            dev = 1_000_000 + int(delay * 1e3) + 1_000
            for j, p in enumerate(plans):
                starts.append(dev)
                dev += 30_000 * len(p) // 9 + 1
                ends.append(dev)
            reps_.append(glp.chain_analysis(1_000_000, starts, ends, 0, calls, rets))
        r = {"case": name, "pieces": k, "planned_nodes": [len(p) for p in plans], "nodes": [None] * k,
             "keep_graph": False, "capture_s": 0.0, "steady": glp._summ(reps_)}
        if first_reps:
            r["first_replay"] = {"torch_default": glp._summ(reps_[:2]), "instantiate_upload": {"error": "x"}}
        assert n >= 2
        return r

    monkeypatch.setattr(glp, "Kernels", FakeK)
    monkeypatch.setattr(glp, "run_case", fake_case)
    fake_torch = types.SimpleNamespace(__version__="fake", cuda=types.SimpleNamespace(
        get_device_name=lambda i: "fake", clock_rate=lambda: 2250))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    out = tmp_path / "g.json"
    monkeypatch.setattr(sys, "argv", ["glprobe.py", "--reps", "50", "--json", str(out)])
    assert glp.main() == 0
    text = capsys.readouterr().out
    assert "GATE item2: uncaptured first-node delay at ~1650 nodes = 995 us" in text and "PASS" in text
    assert "split saving mix 1650 / 4" in text and "piece 3:" in text
    import json
    d = json.loads(out.read_text())
    assert d["gate"]["pass"] and not d["errors"] and len(d["cases"]) == 1 + 2 * (3 + 3)


def test_cold_main_mocked(monkeypatch, tmp_path, capsys):
    import types

    torch = pytest.importorskip("torch")
    last = {}

    class FakeQ4:
        def __init__(self, n, k):
            self.n, self.k = n, k

        def nbytes(self):
            return cold.q4_nbytes(self.n, self.k)

    def old_mm(x, q, xs=None, part=None):
        last["v"] = ("old", q.n)
        return torch.ones((x.shape[0], 8))

    def run_qmm(x, q, xs=None, part=None, cfg=None, pdl=None, serial=None):
        last["v"] = ("new", q.n, cfg, pdl, serial)
        return torch.ones((x.shape[0], 8)) * (2 if cfg == (2, 4) else 1)   # (2, 4) changes bits: not eligible

    class Using:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    ds = types.SimpleNamespace(QMM_CFGS=((1, 4), (1, 6), (1, 8), (2, 4)), pdl_supported=lambda: True,
                               qmm_reference_fused=lambda: True, run_qmm=run_qmm, using=Using,
                               serial_for=lambda n, sk, mode=None: n >= 6144)
    qmm = types.SimpleNamespace(split_k=lambda n, k: 1 if k <= 512 else 4, group_sums=lambda x: x, matmul=old_mm)

    class FakeG:
        def __init__(self):
            self.torch = types.SimpleNamespace(
                randn=torch.randn, bfloat16=torch.bfloat16, equal=torch.equal, __version__="fake",
                cuda=types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None,
                                           get_device_name=lambda i: "fake"))
            self.ds, self.qmm, self.dev, self.part = ds, qmm, "cpu", None
            self.bdk = types.SimpleNamespace(SHAPES=SHAPES[:11])

        def q4_copies(self, n, k, copies):
            return [FakeQ4(n, k) for _ in range(copies)]

        def graph(self, fns, ctx=None):
            return None

        def pred(self):
            pass

        def roof(self, mode, nbytes, copies, reps, flush_calls, gpred):
            return 900.0 if (mode == "rotate" and nbytes < 1e6) else 230.0   # 4096x128 / 160x4096 "L2-resident"

        def call_us(self, mode, fns, reps, flush_calls, gpred=None, ctx=None):
            fns[0]()
            v = last["v"]
            mb = cold.q4_nbytes(v[1], 4096) / 1e6
            base = mb / 0.2                                            # 200 GB/s old
            if v[0] == "old":
                return base
            return base / (1.3 if mb < 5 else 0.9)                   # new: 1.3x on < 5 MB, 0.9x above

    monkeypatch.setattr(cold, "Gpu", FakeG)
    out = tmp_path / "c.json"
    monkeypatch.setattr(sys, "argv", ["bench_decode_cold.py", "--mode", "both", "--json", str(out)])
    assert cold.main() == 0
    text = capsys.readouterr().out
    assert "GATE item7:" in text and "GLM53_TF_DEC_QMM_MAX_MB candidate" in text
    import json
    d = json.loads(out.read_text())
    assert not d["errors"], d["errors"]
    assert len(d["rotate"]) == len(d["flush"]) == len(d["hot"]) == len(SHAPES) * 5     # 13 shapes + 0570's 1024x4096
    r = d["rotate"][0]
    assert {"shape", "rows", "MB", "sk", "old_us", "old_GBs", "roof_GBs", "auto serial", "best_new",
            "speedup"} <= set(r)
    assert all(not r["best_new"].startswith("new (2, 4)") for r in d["rotate"])       # bits differ: never best
    assert "2048x4096" in d["gate"]["passing"] and "4096x4096" in d["gate"]["failing"]
    assert "160x4096" in d["gate"]["unmeasured"]                                      # roof 900: INVALID
