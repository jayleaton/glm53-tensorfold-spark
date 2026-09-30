"""patches/0440 micro-benchmark for the GPU window: the decode-window routed experts (exl3.cu vs exl3_stream.cu) and
4-bit dense matmuls (qmm._qmm + _reduce vs q4_stream.cu) at the real per-rank shapes, us a call and GB/s, next to a
plain streaming-read roof probe over the same bytes. One GPU, ~5-8 minutes.

    PYTHONPATH=/src/TensorFold/tests/cuda python tests/cuda/bench_decode_kernels.py [--quick] [--json out.json]
        [--experts-only | --qmm-only] [--no-pdl]

Method (every number a median of 7 replays of a CUDA graph of back-to-back calls):

- weights come from DRAM, not L2: each timed graph cycles through 3 copies of the layer (routed experts: 3 x 1.81 GB
  a copy is too much, so 3 copies of the 64 experts the windows use, far past the 24 MB L2; dense: 3 weight copies);
- expert windows reproduce the measured distinct-expert counts U (ROOFLINE 2.1): single stream R = 1 / 2 / 4 / 8 / 16
  with U = 8 / 13 / 22 / 35 / 64, and 4-slot mixes 3+3+3+2 (U 51), 4+4+4+4 (U 60), 8+8+8+8 (U 110);
  bytes = U x 6.29 MB (gate + up + down trellis) + the scales, the reads of X / Xd and the writes of Z / Y;
- old = what production runs today (exl3_mm.routed with the knobs off: rot_in + grouped + gateup_epilogue + grouped +
  down_epilogue; qmm.matmul: _qmm + _reduce), new = decode_stream's kernels in every configuration (dense: (tile,
  slice) "split" items and whole-tile "serial" items; "auto serial" names what the default picks), PDL off / on,
  plus exl3_stream's probes (1: no trellis decode, 2: no mma) to split data movement from ALU;
- the roof probe streams the same byte count with 16-byte loads from a persistent grid (the W11-style probe inside
  this process, so the numbers share the clock state); GB/s = bytes / time.

patches/0580 (``--loads``): exl3_ld.cu's ld_kernel (GLM53_TF_DEC_EXPERT_LOADS) vs exl3.cu's grouped_kernel -- the two
grouped launches of a routed layer (gate/up + down), every (ld, nt, pd) of expert_loads.CFGS, COLD two ways: "rotate"
(3 layer copies of 128 experts back to back, as above) and "flush" (a 64 MiB streaming read before every call; graph A
= N x (flush, call), graph B = N x flush, us = median (A - B) / N: cold L2 plus a DRAM-busy predecessor tail, the
in-situ-like number); the probes 1 (no decode), 2 (no mma), 3 (load path alone) at (8, 2); the whole routed layer
(rot_in + epilogues too) old vs the default; Z compared bit for bit (NaN-filled buffers) before timing. Two gate lines:

    GATE 0580 load path: probe 3 (nc,8,2, no decode / mma) >= 220 GB/s [flush] on every U 8-22 window: PASS|FAIL
    GATE 0580: <cfg> >= 1.05x grouped_kernel [flush] (and >= 1.00x [rotate]) on every U 8-22 window: PASS|FAIL

    PYTHONPATH=/src/TensorFold/tests/cuda python tests/cuda/bench_decode_kernels.py --loads [--quick] [--json out.json]

Reading it: new / probe is the kernel's share of the attainable bandwidth; old -> new is the E1 / E2 gain; if probe 1 is
not faster than new, the decode ALU is hidden and only data movement matters. Every new variant's output is compared
with old's bit for bit before it is timed (a False there means the variant must not be used).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from tensorfold.families.glm5_next.cuda import decode_stream as ds, exl3_mm, qmm  # noqa: E402

try:                                                         # patches/0580
    from tensorfold.families.glm5_next.cuda import expert_loads as el  # noqa: E402
except ImportError:                                          # pragma: no cover
    el = None

DEV = "cuda"
D, NI, E, TOP, LIMIT = 4096, 1024, 288, 8, 10.0
SLOTS = TOP + 1
EXPERT_BYTES = (2 * D * NI + NI * D) // 2          # 6.29 MB: gate + up + down trellis words a rank
SINGLE = [(1, 8), (2, 13), (4, 22), (8, 35), (16, 64)]
MIXES = [((3, 3, 3, 2), 51), ((4, 4, 4, 4), 60), ((8, 8, 8, 8), 110)]
SHAPES = [(12576, 4096), (4096, 4096), (4096, 128), (2048, 4096), (8192, 1536), (8192, 512), (4096, 8192),
          (12288, 4096), (4096, 6144), (4096, 1024), (77440, 4096)]
EXPERT_CFGS = [(4, 4, 4, 4), (4, 6, 4, 6), (2, 6, 2, 6), (2, 8, 2, 8), (8, 3, 8, 3), (4, 4, 8, 3), (2, 8, 4, 4)]
COPIES = 3


# -- timing -------------------------------------------------------------------------------------------------------------
def _graph_us(fns, reps: int = 7) -> float:
    """us per call: a CUDA graph of every fn in ``fns`` once (a cycle over weight copies), replayed ``reps`` times."""

    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    g.replay()
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        t.append(a.elapsed_time(b) * 1e3 / len(fns))
    return statistics.median(t)


# -- the roof probe -------------------------------------------------------------------------------------------------------
_PROBE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void __launch_bounds__(256) stream_read(const uint4* __restrict__ p, size_t n, unsigned* __restrict__ out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;               // never true in practice: keeps the loads
}
void probe(torch::Tensor buf, int64_t bytes, int64_t ctas, torch::Tensor out) {
    stream_read<<<(unsigned)ctas, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(buf.data_ptr()), (size_t)bytes / 16, reinterpret_cast<unsigned*>(out.data_ptr()));
}
"""


_probe_mod = None


def _probe():
    global _probe_mod
    if _probe_mod is None:
        from torch.utils.cpp_extension import load_inline

        _probe_mod = load_inline("tf_decode_roof_probe", cpp_sources=(  # W12: the binding needs the declaration
            "#include <torch/extension.h>\nvoid probe(torch::Tensor buf, int64_t bytes, int64_t ctas, torch::Tensor out);\n"),
                                 cuda_sources=_PROBE_SRC, functions=["probe"],
                                 extra_cuda_cflags=["-O3"], verbose=False)
    return _probe_mod


def roof_gbs(nbytes: int) -> float:
    """GB/s of a plain 16-byte streaming read of ``nbytes`` (3 buffers cycled, so nothing stays in L2)."""

    nbytes = max(16, nbytes // 16 * 16)
    bufs = [torch.empty(nbytes, dtype=torch.uint8, device=DEV) for _ in range(COPIES)]
    out = torch.zeros(1, dtype=torch.int32, device=DEV)
    ctas = torch.cuda.get_device_properties(0).multi_processor_count * 8
    us = _graph_us([lambda b=b: _probe().probe(b, nbytes, ctas, out) for b in bufs])
    return nbytes / us / 1e3


# -- routed experts ---------------------------------------------------------------------------------------------------------
def _layer(seed: int, n_exp: int):
    """``n_exp`` experts of the real per-rank shape (random trellis words: the decode does the same work)."""

    g = torch.Generator(device=DEV).manual_seed(seed)

    def trellis(k, n):
        return torch.randint(-2**15, 2**15, (n_exp, k // 16, n // 16, 64), dtype=torch.int16, device=DEV,
                             generator=g).view(torch.int32)

    def hs(n, sc):
        return (torch.randn((n_exp, n), device=DEV, generator=g) * sc).to(torch.float16)

    return exl3_mm.Exl3Experts(trellis(D, NI), trellis(D, NI), trellis(NI, D), hs(D, 0.02), hs(D, 0.02), hs(NI, 0.5),
                               hs(NI, 0.5), hs(NI, 0.05), hs(D, 0.2), n_exp, NI, D)


def _window(sizes, U: int, n_exp: int, seed: int):
    """Picks of len(sizes) sequences (rows each), U distinct experts in all: each sequence routes over its own share
    of the pool (rows of one sequence share experts), every pool expert used at least once."""

    g = torch.Generator().manual_seed(seed)
    pool = torch.randperm(n_exp, generator=g)[:U].tolist()
    shares, i = [], 0
    rows_total = sum(sizes)
    for n in sizes:
        k = max(TOP, round(U * n / rows_total))
        share = pool[i:i + k] if i + k <= U else pool[i:] + pool[:k - (U - i)]
        shares.append(share)
        i = min(i + k, U)
    picks = []
    for n, share in zip(sizes, shares):
        order = list(share)
        for r in range(n):
            row = []
            j = r * TOP
            while len(row) < TOP:
                e = order[j % len(order)]
                if e not in row:
                    row.append(e)
                j += 1
            picks.append(row + [n_exp])
    pick = torch.tensor(picks, dtype=torch.int32)
    R = pick.shape[0]
    used = sorted({int(e) for e in pick[:, :TOP].flatten()})
    maxu = min(R * TOP, n_exp) + 1
    ids = torch.zeros((maxu,), dtype=torch.int32)
    members = torch.full((maxu, R), -1, dtype=torch.int32)
    for u, e in enumerate(used + [n_exp]):
        ids[u] = e
        j = 0
        for r in range(R):
            for s in range(SLOTS):
                if int(pick[r, s]) == e:
                    members[u, j] = r * 32 + s
                    j += 1
    grp = qmm.Group(ids.to(DEV), torch.tensor([len(used) + 1], dtype=torch.int32, device=DEV), members.to(DEV))
    return pick.to(DEV), grp, len(used)


def bench_experts(pdls, quick: bool) -> list[dict]:
    n_exp = 128                                        # enough distinct experts for every window, x3 copies
    layers = [_layer(10 + c, n_exp) for c in range(COPIES)]
    s = exl3_mm.Scratch(64, SLOTS, D, NI, DEV)
    y = torch.zeros((64 * SLOTS, D), dtype=torch.float32, device=DEV)
    rows = []
    cases = [((r,), u) for r, u in SINGLE] + list(MIXES)
    if quick:
        cases = [((1,), 8), ((8,), 35), ((3, 3, 3, 2), 51)]
    for sizes, U in cases:
        pick, grp, used = _window(sizes, U, n_exp, sum(sizes) * 7 + U)
        R = pick.shape[0]
        x = torch.randn((R, D), device=DEV).to(torch.bfloat16)
        nbytes = used * EXPERT_BYTES
        label = f"{'+'.join(map(str, sizes))} rows, U {used}"

        def old(ex):
            with ds.using(experts=False):
                exl3_mm.routed(x, pick, grp, ex, s, y, R, LIMIT)

        ref = None
        old(layers[0])
        torch.cuda.synchronize()
        ref = y[:R * SLOTS].clone()
        t_old = _graph_us([lambda ex=ex: old(ex) for ex in layers])
        row = {"window": label, "rows": R, "U": used, "MB": nbytes / 1e6, "old_us": t_old,
               "old_GBs": nbytes / t_old / 1e3, "roof_GBs": roof_gbs(nbytes)}
        cfgs = EXPERT_CFGS[:2] if quick else EXPERT_CFGS
        for cfg in cfgs:
            for pdl in pdls:
                y.zero_()
                ds.run_experts(x, pick, grp, layers[0], s, y, R, LIMIT, cfg=cfg, pdl=pdl)
                torch.cuda.synchronize()
                same = torch.equal(y[:R * SLOTS].view(R, SLOTS, D)[:, :TOP], ref.view(R, SLOTS, D)[:, :TOP])
                t = _graph_us([lambda ex=ex: ds.run_experts(x, pick, grp, ex, s, y, R, LIMIT, cfg=cfg, pdl=pdl)
                               for ex in layers])
                row[f"new {cfg}{' pdl' if pdl else ''}"] = {"us": t, "GBs": nbytes / t / 1e3, "same_bits": same}
        for probe in (1, 2):
            t = _graph_us([lambda ex=ex: ds.run_experts(x, pick, grp, ex, s, y, R, LIMIT, cfg=(4, 4, 4, 4), probe=probe)
                           for ex in layers])
            row[f"probe {probe} ({'no decode' if probe == 1 else 'no mma'})"] = {"us": t, "GBs": nbytes / t / 1e3}
        rows.append(row)
    return rows


# -- dense ------------------------------------------------------------------------------------------------------------------
def _q4(n: int, k: int, seed: int) -> qmm.Q4:
    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int64, device=DEV, generator=g).to(torch.int32)
    scales = (torch.rand((n, k // 64), device=DEV, generator=g) * 0.01 + 0.005).to(torch.bfloat16)
    return qmm.make_q4(words, scales, (-7.5 * scales.float()).to(torch.bfloat16))


def bench_qmm(pdls, quick: bool) -> list[dict]:
    rows = []
    part = torch.empty((8 * 64 * 77440,), dtype=torch.float32, device=DEV)
    shapes = SHAPES if not quick else [(12576, 4096), (77440, 4096), (8192, 512)]
    ms = (1, 2, 4, 8, 16, 11, 44) if not quick else (1, 8)
    for n, k in shapes:
        qs = [_q4(n, k, c) for c in range(COPIES)]
        nbytes = qs[0].nbytes()
        roof = roof_gbs(nbytes)
        for m in ms:
            x = torch.randn((m, k), device=DEV).to(torch.bfloat16)
            xs = qmm.group_sums(x)
            with ds.using(qmm_on=False):
                ref = qmm.matmul(x, qs[0], xs, part=part).clone()
                t_old = _graph_us([lambda q=q: qmm.matmul(x, q, xs, part=part) for q in qs])
            row = {"shape": f"{n}x{k}", "rows": m, "MB": nbytes / 1e6, "sk": qmm.split_k(n, k), "old_us": t_old,
                   "old_GBs": nbytes / t_old / 1e3, "roof_GBs": roof}
            if not ds.qmm_reference_fused():
                row["note"] = "Triton is not 3.7.x: the stream kernel would not keep _qmm's bits (timed anyway)"
            cfgs = ds.QMM_CFGS if m <= 16 else ((1, 4),)
            serials = (False, True) if qmm.split_k(n, k) > 1 else (True,)
            for cfg in cfgs:
                for pdl in pdls:
                    for serial in serials:
                        got = ds.run_qmm(x, qs[0], xs, part=part, cfg=cfg, pdl=pdl, serial=serial)
                        torch.cuda.synchronize()
                        same = torch.equal(got, ref)
                        t = _graph_us([lambda q=q: ds.run_qmm(x, q, xs, part=part, cfg=cfg, pdl=pdl, serial=serial)
                                       for q in qs])
                        tag = f"new {cfg}{' serial' if serial else ' split'}{' pdl' if pdl else ''}"
                        row[tag] = {"us": t, "GBs": nbytes / t / 1e3, "same_bits": same}
            row["auto serial"] = ds.serial_for(n, qmm.split_k(n, k), "auto")
            rows.append(row)
    return rows


# -- patches/0580: the expert load path, cold ------------------------------------------------------------------------------
LOAD_WINDOWS = [((1,), 8), ((2,), 13), ((3,), 17), ((4,), 22), ((8,), 35), ((16,), 64), ((3, 3, 3, 2), 51),
                ((4, 4, 4, 4), 60)]
GATE_U = (8, 22)                         # the gate's windows: 1-stream decode / verify (1-4 rows)
LOAD_GATE_X = 1.05
LOAD_PROBE_GBS = 220.0
FLUSH_BYTES = 64 << 20


def _flush_us(fns, pred, reps: int = 9) -> float:
    """us a call with a cold-L2 predecessor: median over reps of (graph[pred, fn, pred, fn ...] - graph[pred ...]) / N,
    the two graphs replayed interleaved."""

    for f in fns:
        pred()
        f()
    torch.cuda.synchronize()
    ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(ga):
        for f in fns:
            pred()
            f()
    with torch.cuda.graph(gb):
        for _ in fns:
            pred()
    ga.replay()
    gb.replay()
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        ga.replay()
        ev[1].record()
        ev[2].record()
        gb.replay()
        ev[3].record()
        torch.cuda.synchronize()
        t.append((ev[0].elapsed_time(ev[1]) - ev[2].elapsed_time(ev[3])) * 1e3 / len(fns))
    return statistics.median(t)


def gate_0580(rows: list[dict], lo_hi=GATE_U, x: float = LOAD_GATE_X, probe_gbs: float = LOAD_PROBE_GBS) -> dict:
    """The two gate lines from ``bench_loads`` rows (pure: CPU-testable). The config is one knob setting for every
    window, so it is chosen once: the eligible (same-bits) config with the best geometric mean flush speed-up over
    the gate windows; it passes if it is >= x on EVERY gate window in flush mode and >= 1.00 in rotate mode."""

    import math

    gate = [r for r in rows if lo_hi[0] <= r["U"] <= lo_hi[1]]
    out = {"windows": [r["window"] for r in gate]}
    pr = [r.get("probe 3 nc", {}).get("flush_GBs") for r in gate]
    ok_p = bool(gate) and all(v is not None and v >= probe_gbs for v in pr)
    out["probe_pass"] = ok_p
    out["probe_line"] = (f"GATE 0580 load path: probe 3 (nc,8,2, no decode / mma) >= {probe_gbs:.0f} GB/s [flush] on "
                         f"every U {lo_hi[0]}-{lo_hi[1]} window: {'PASS' if ok_p else 'FAIL'} "
                         f"({', '.join('n/a' if v is None else f'{v:.0f}' for v in pr)})")
    cfgs = sorted({k for r in gate for k, v in r.items() if k.startswith("new ") and isinstance(v, dict)})
    best, best_g = None, 0.0
    for c in cfgs:
        sp = []
        for r in gate:
            v = r.get(c)
            if not v or not v.get("same_bits") or not v.get("flush_us"):
                sp = None
                break
            sp.append(r["old_flush_us"] / v["flush_us"])
        if sp:
            g = math.exp(sum(math.log(a) for a in sp) / len(sp))
            if g > best_g:
                best, best_g = c, g
    out["best"] = best
    if best is None:
        out["pass"] = False
        out["line"] = f"GATE 0580: no eligible config measured on the U {lo_hi[0]}-{lo_hi[1]} windows: FAIL"
        return out
    fl = [r["old_flush_us"] / r[best]["flush_us"] for r in gate]
    ro = [r["old_rotate_us"] / r[best]["rotate_us"] for r in gate if r[best].get("rotate_us")]
    ok = all(v >= x for v in fl) and all(v >= 1.0 for v in ro)
    out.update(flush=fl, rotate=ro, geomean=best_g)
    out["pass"] = ok
    out["line"] = (f"GATE 0580: {best[4:]} >= {x:.2f}x grouped_kernel [flush] (and >= 1.00x [rotate]) on every U "
                   f"{lo_hi[0]}-{lo_hi[1]} window: {'PASS' if ok else 'FAIL'} (flush "
                   f"{' '.join(f'{v:.3f}' for v in fl)}; rotate {' '.join(f'{v:.3f}' for v in ro)}; geomean {best_g:.3f})")
    return out


def bench_loads(quick: bool) -> list[dict]:
    """patches/0580: grouped_kernel (gate/up + down) vs ld_kernel, cold (rotate and flush), probes, whole routed."""

    if el is None:
        raise RuntimeError("patches/0580 (expert_loads) is not in this tree")
    n_exp = 128
    layers = [_layer(10 + c, n_exp) for c in range(COPIES)]
    s = exl3_mm.Scratch(64, SLOTS, D, NI, DEV)
    y = torch.zeros((64 * SLOTS, D), dtype=torch.float32, device=DEV)
    ext = exl3_mm._ext()
    gsk, dsk = exl3_mm.GATEUP_CFG[2], exl3_mm.DOWN_CFG[2]
    flush_buf = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=DEV)
    fout = torch.zeros(1, dtype=torch.int32, device=DEV)
    fctas = torch.cuda.get_device_properties(0).multi_processor_count * 8

    def pred():
        _probe().probe(flush_buf, FLUSH_BYTES, fctas, fout)

    cases = LOAD_WINDOWS if not quick else [((1,), 8), ((2,), 13), ((4,), 22), ((3, 3, 3, 2), 51)]
    cfgs = el.CFGS if not quick else (("nc", 8, 2), ("nc", 8, 1), ("cpa", 8, 2), ("w32", 8, 2), ("nc", 4, 2))
    rows = []
    for sizes, U in cases:
        pick, grp, used = _window(sizes, U, n_exp, sum(sizes) * 7 + U)
        R = pick.shape[0]
        P = R * SLOTS
        x = torch.randn((R, D), device=DEV).to(torch.bfloat16)
        ext.rot_in(x, x.stride(0), pick, layers[0].suh_g, layers[0].suh_u, s.xg, s.xu, R, D, SLOTS)
        s.xd[:P].copy_((torch.randn((P, NI), device=DEV) * 0.5).half())
        nbytes = used * EXPERT_BYTES
        zg = torch.empty((2 * gsk * P * NI,), dtype=torch.float32, device=DEV)
        zd = torch.empty((dsk * P * D,), dtype=torch.float32, device=DEV)

        def old(ex):
            ext.grouped(s.xg, s.xu, ex.gt, ex.ut, grp.ids, grp.count, grp.members, zg, 2, D, NI, P, gsk, SLOTS, 8, 4)
            ext.grouped(s.xd, s.xd, ex.dt, ex.dt, grp.ids, grp.count, grp.members, zd, 1, NI, D, P, dsk, SLOTS, 8, 4)

        def new(ex, cfg, probe=0, pdl=False):
            el.run(s.xg, s.xu, ex.gt, ex.ut, grp, zg, 2, D, NI, P, gsk, SLOTS, cfg, probe, pdl)
            el.run(s.xd, s.xd, ex.dt, ex.dt, grp, zd, 1, NI, D, P, dsk, SLOTS, cfg, probe, pdl)

        zg.fill_(float("nan"))
        zd.fill_(float("nan"))
        old(layers[0])
        torch.cuda.synchronize()
        ref_g, ref_d = zg.view(torch.int32).clone(), zd.view(torch.int32).clone()
        t_rot = _graph_us([lambda ex=ex: old(ex) for ex in layers])
        t_fl = _flush_us([lambda ex=ex: old(ex) for ex in layers], pred)
        row = {"window": f"{'+'.join(map(str, sizes))} rows, U {used}", "rows": R, "U": used, "MB": nbytes / 1e6,
               "old_rotate_us": t_rot, "old_flush_us": t_fl, "old_rotate_GBs": nbytes / t_rot / 1e3,
               "old_flush_GBs": nbytes / t_fl / 1e3, "roof_GBs": roof_gbs(nbytes)}
        variants = [(c, False) for c in cfgs] + ([(el.DEFAULT, True)] if ds.pdl_supported() else [])
        for cfg, pdl in variants:                             # (the default again with GLM53_TF_DEC_EXPERT_LOADS_PDL)
            if not (el.fits(D, NI, gsk, 4, cfg) and el.fits(NI, D, dsk, 4, cfg)):
                continue
            zg.fill_(float("nan"))
            zd.fill_(float("nan"))
            new(layers[0], cfg, 0, pdl)
            torch.cuda.synchronize()
            same = torch.equal(zg.view(torch.int32), ref_g) and torch.equal(zd.view(torch.int32), ref_d)
            tr = _graph_us([lambda ex=ex: new(ex, cfg, 0, pdl) for ex in layers])
            tf = _flush_us([lambda ex=ex: new(ex, cfg, 0, pdl) for ex in layers], pred)
            row[f"new {','.join(map(str, cfg))}{' pdl' if pdl else ''}"] = {
                "rotate_us": tr, "flush_us": tf, "rotate_GBs": nbytes / tr / 1e3, "flush_GBs": nbytes / tf / 1e3,
                "same_bits": same}
        for ld, probe in sorted(el.PROBES):
            cfg = (ld, 8, 2)
            tr = _graph_us([lambda ex=ex: new(ex, cfg, probe) for ex in layers])
            tf = _flush_us([lambda ex=ex: new(ex, cfg, probe) for ex in layers], pred)
            row[f"probe {probe} {ld}"] = {"rotate_us": tr, "flush_us": tf, "rotate_GBs": nbytes / tr / 1e3,
                                          "flush_GBs": nbytes / tf / 1e3}

        def routed(ex, on):
            with el.using(on=on, gu=el.DEFAULT, dn=el.DEFAULT), ds.using(experts=False):
                exl3_mm.routed(x, pick, grp, ex, s, y, R, LIMIT)

        row["routed_old_rotate_us"] = _graph_us([lambda ex=ex: routed(ex, False) for ex in layers])
        row["routed_new_rotate_us"] = _graph_us([lambda ex=ex: routed(ex, True) for ex in layers])
        rows.append(row)
    return rows


def _fmt_loads(rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        best = min(((v["flush_us"], k) for k, v in r.items() if k.startswith("new ") and v.get("same_bits")),
                   default=(float("nan"), "none"))
        out.append(f"loads {r['window']}: {r['MB']:.0f} MB, roof {r['roof_GBs']:.0f} GB/s | grouped_kernel rotate "
                   f"{r['old_rotate_us']:.1f} us {r['old_rotate_GBs']:.0f} GB/s, flush {r['old_flush_us']:.1f} us "
                   f"{r['old_flush_GBs']:.0f} GB/s | best new [flush] {best[1][4:]} {best[0]:.1f} us "
                   f"({r['old_flush_us'] / best[0]:.3f}x) | routed layer {r['routed_old_rotate_us']:.1f} -> "
                   f"{r['routed_new_rotate_us']:.1f} us")
        for k, v in r.items():
            if isinstance(v, dict):
                bits = "" if "same_bits" not in v else ("  same bits" if v["same_bits"] else "  BITS DIFFER")
                out.append(f"    {k:18s} rotate {v['rotate_us']:8.1f} us {v['rotate_GBs']:5.0f} GB/s ({r['old_rotate_us'] / v['rotate_us']:.3f}x)"
                           f" | flush {v['flush_us']:8.1f} us {v['flush_GBs']:5.0f} GB/s ({r['old_flush_us'] / v['flush_us']:.3f}x){bits}")
    g = gate_0580(rows)
    out += [g["probe_line"], g["line"]]
    return out


def _fmt(rows: list[dict], key: str) -> list[str]:
    out = []
    for r in rows:
        head = r.get("window") or f"{r['shape']} x {r['rows']} rows (sk {r['sk']})"
        best = min((v["us"], k) for k, v in r.items() if isinstance(v, dict) and k.startswith("new"))
        out.append(f"{key} {head}: {r['MB']:.1f} MB, roof {r['roof_GBs']:.0f} GB/s | old {r['old_us']:.1f} us "
                   f"{r['old_GBs']:.0f} GB/s | best new {best[1]} {best[0]:.1f} us {r['MB'] * 1e3 / best[0]:.0f} GB/s "
                   f"({r['old_us'] / best[0]:.3f}x)")
        for k, v in r.items():
            if isinstance(v, dict):
                bits = "" if "same_bits" not in v else ("  same bits" if v["same_bits"] else "  BITS DIFFER")
                out.append(f"    {k:26s} {v['us']:9.1f} us {v['GBs']:6.0f} GB/s{bits}")
    return out


def summary() -> list[str]:
    pdls = [False] + ([True] if ds.pdl_supported() else [])
    return _fmt(bench_experts(pdls, True), "experts") + _fmt(bench_qmm(pdls, True), "dense")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--experts-only", action="store_true")
    ap.add_argument("--qmm-only", action="store_true")
    ap.add_argument("--no-pdl", action="store_true")
    ap.add_argument("--loads", action="store_true", help="patches/0580 only: grouped_kernel vs exl3_ld.cu, cold")
    a = ap.parse_args()
    if a.loads:
        res = {"device": torch.cuda.get_device_name(0), "loads": bench_loads(a.quick)}
        lines = _fmt_loads(res["loads"])
        res["gate"] = {k: v for k, v in gate_0580(res["loads"]).items() if k != "windows"}
        print("\n".join(lines), flush=True)
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1, default=str))
        return
    pdls = [False] + ([True] if ds.pdl_supported() and not a.no_pdl else [])
    res = {"device": torch.cuda.get_device_name(0), "triton_fused_qmm": ds.qmm_reference_fused()}
    if not a.qmm_only:
        res["experts"] = bench_experts(pdls, a.quick)
        print("\n".join(_fmt(res["experts"], "experts")), flush=True)
    if not a.experts_only:
        res["qmm"] = bench_qmm(pdls, a.quick)
        print("\n".join(_fmt(res["qmm"], "dense")), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
