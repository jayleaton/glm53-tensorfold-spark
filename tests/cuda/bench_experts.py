"""patches/0260 micro-benchmark: the fast-prefill routed-expert kernels of one MoE layer on the real per-rank shapes
(GLM-5.3-Flash at TP 2: 288 experts, top 8, hidden 4096, 1024 of each expert's 2048 width), one GPU, ~2-3 minutes.

    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python tests/cuda/bench_experts.py [rows ...] [--no-v1]

Per row count (default 1024 2048 4096 8192) and routing (uniform; skewed = Zipf 0.8 over the experts, as the tests):

- gate/up and down separately, ms (median of 20) and TFLOP/s, for v1 (0080, run in a subprocess because the C++
  switch reads GLM53_TF_FAST_EXPERTS once), fast2, fat (3 / 4 stages), once (pairs on; GLM53_TF_ONCE_PAIR=0's split
  units only), and once's timing probes: no trellis decode (the decode's share) and no mma (the decode + data
  movement alone). The input rotation (rot_in / rot_in1) is timed once on its own; it does not change with the kernel;
- how many times a weight tile is decoded in the chunk (fat vs once, from the actual routing);
- whether fat and once gave fast2's bits (Xd and Y). A False there means the kernel must not be used.

Reading it: once's gain over fat ~= (fat - once) and should grow with rows (the decode pairs only exist for experts with
2+ passes: ~none at 1024 rows). "no decode" bounds what any decode-sharing can win; if fat - probe1 is small, the
kernel is mma / data-movement bound and pairing cannot help.

patches/0330 modes (docs/EXPERT-TC.md):

    bench_experts.py 2048 4096 8192 --tc [--contend] [--variants fast2,fat,tc] [--no-v1]

- ``--tc`` adds the ``tc`` kernels (exl3_tc.cu): configurations 0 (by rows) / 1 / 2 / 3, the static stride (ticket
  off), CTA caps (40 / 44 of 48 SMs), and tc's probes (no decode, no mma); bits against fast2 as above. It also prints
  the DRAM floor of the layer (weights once + Xg read + Xd written and read + Y written, at 220 GB/s) and each
  variant's share of it;
- ``--contend`` times every variant a second time while a side stream runs what patches/0084's overlap puts beside
  the experts in production (a DRAM-bound copy of 2 x 64 MB and a small bf16 GEMM that holds a few SMs, enqueued so
  they outlast the kernel): the W5 question, why fast2's isolated win (-2 ms a layer at 2,048 rows) became -2% end to
  end. A static-stride kernel (fast2, tc with ticket off) waits for its slowest CTA; a ticket kernel rebalances;
- ``--variants a,b``: only the variants whose names start with one of these (e.g. ``fast2,fat s3,tc``).

patches/0590 mode (docs/EXPERT-PREFILL-V2.md):

    bench_experts.py 2048 4096 8192 --fat2 --contend --no-v1 --variants fast2,fat,fat2

- ``--fat2`` adds the ``fat2`` kernels (exl3_fat2.cu): configurations 0 (128 members, 4 stages) / 1 (64, 4) /
  2 (128, 3), the static stride (ticket off), CTA caps (44 / 40 of 48 SMs), and the probes (no decode, no mma); bits
  against fast2 as above, and the DRAM floor as with ``--tc``;
- with ``--contend`` every variant also gets its MAKESPAN: from the side stream's start until both the kernel and the
  side work are done (the side work alone is printed too). The contended kernel time alone rewards a kernel that
  starves the side stream; the makespan is what the overlapped prefill pays;
- the GATE line (with ``--fat2 --contend``, 2,048 and 4,096 rows, uniform routing): fat2's bits "same" and its
  contended time <= 0.75 x fat's (``fat s3``, production). It is printed as PASS / FAIL with the ratios, the makespan
  ratios and the DRAM floor of the layer (the bound any kernel that keeps fat's dataflow -- Xg in, fp32 Y out --
  cannot beat: at 2,048 rows it sits at ~0.74x fat's isolated time).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

D, NI, E, TOP, LIMIT = 4096, 1024, 288, 8, 10.0
SLOTS = TOP + 1


def _ms(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        t.append(a.elapsed_time(b))
    return float(np.median(t))


def _setup(rows, kind):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_fastpf_patches import _fast_group
    from test_patches import _exl3_layer

    from tensorfold.families.glm5_next.cuda import exl3_mm

    ex = _exl3_layer(D, NI, E, seed=5)
    ex.suh_u = ex.suh_g                                   # the real checkpoint: gate and up share their sign vector
    x = (torch.randn((rows, D), generator=torch.Generator().manual_seed(6)) * 0.5).to(torch.bfloat16)
    g = torch.Generator().manual_seed(7)
    w = torch.ones(E) if kind == "uniform" else 1.0 / torch.arange(1, E + 1).float() ** 0.8
    picks = torch.full((rows, SLOTS), E, dtype=torch.int32)
    picks[:, :TOP] = torch.multinomial(w.expand(rows, E), TOP, replacement=False, generator=g).int()
    grp = _fast_group(picks, E)
    s = exl3_mm.Scratch(rows, SLOTS, D, NI, "cuda")
    x, picks = x.cuda(), picks.cuda()
    n = np.bincount(picks[:, :TOP].flatten().cpu().numpy(), minlength=E)
    return exl3_mm, ex, x, picks, grp, s, n[n > 0]


def _decodes(n, rows):
    gu = np.ceil(n / 64)
    dn = np.ceil(n / (128 if rows >= 4096 else 64))
    fat = (2 * gu + dn).sum() / (3 * n.size)
    once = (2 * np.ceil(gu / 2) + np.ceil(dn / 2)).sum() / (3 * n.size)
    return fat, once


def run_v1(rows, kind):
    """In a GLM53_TF_FAST_EXPERTS=v1 process: v1's gate/up and down (``gateup`` / ``down`` run v1 there)."""
    exl3_mm, ex, x, picks, grp, s, _ = _setup(rows, kind)
    fx, ext = exl3_mm._fast_ext(), exl3_mm._ext()
    y = torch.zeros((rows * SLOTS, D), dtype=torch.float32, device="cuda")
    ext.rot_in(x, x.stride(0), picks, ex.suh_g, ex.suh_u, s.xg, s.xu, rows, D, SLOTS)
    gu = _ms(lambda: fx.gateup(s.xg, s.xu, ex.gt, ex.ut, grp.ids, grp.count, grp.members, ex.svh_g, ex.svh_u, ex.suh_d,
                               s.xd, D, NI, SLOTS, LIMIT))
    dn = _ms(lambda: fx.down(s.xd, ex.dt, grp.ids, grp.count, grp.members, ex.svh_d, y, NI, D, SLOTS))
    return gu, dn


def _contender():
    """patches/0330 --contend: a side stream's work that outlasts one expert kernel: DRAM copies (the bf16 all-gather
    / hc slab traffic of patches/0084) and a small GEMM whose few CTAs sit on some SMs."""
    side = torch.cuda.Stream()
    a = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    m = torch.randn((512, 4096), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((4096, 512), dtype=torch.bfloat16, device="cuda")

    def start(n=24):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for i in range(n):
                (b if i % 2 else a).copy_(a if i % 2 else b)
                torch.mm(m, w)
    return start, side


def _ms_under(fn, start, side, reps=20, warm=3):
    """The kernel's time on the current stream while the side stream is busy (the side work is started just before
    each rep and drained after it)."""
    for _ in range(warm):
        start()
        fn()
        torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        start()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        t.append(a.elapsed_time(b))
    return float(np.median(t))


def _makespan_under(fn, start, side, reps=20, warm=3):
    """patches/0590: from just before the side stream's work is enqueued until the kernel AND the side work are done
    (the side stream joined back), ms; and the side work alone (no kernel), ms."""
    def once(with_kernel):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        start()
        if with_kernel:
            fn()
        torch.cuda.current_stream().wait_stream(side)
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b)
    for _ in range(warm):
        once(True)
    both = float(np.median([once(True) for _ in range(reps)]))
    alone = float(np.median([once(False) for _ in range(reps)]))
    return both, alone


def _floor_ms(rows, bw=220e9):
    """The layer's DRAM floor at ``bw``: every weight once (1.81 GB a rank), Xg read, Xd written and read, Y written."""
    P = rows * TOP
    weights = E * (2 * D * NI + NI * D) / 2
    return 1e3 * (weights + P * D * 2 + 2 * P * NI * 2 + P * D * 4) / bw


def run(rows, kind, v1, tc=False, contend=False, only=(), fat2=False):
    exl3_mm, ex, x, picks, grp, s, n = _setup(rows, kind)
    fx, ext = exl3_mm._fast_ext(), exl3_mm._ext()
    y = torch.zeros((rows * SLOTS, D), dtype=torch.float32, device="cuda")
    G = (grp.ids, grp.count, grp.members)
    flops_gu = 2.0 * rows * TOP * 2 * D * NI
    flops_dn = 2.0 * rows * TOP * NI * D

    rot = _ms(lambda: ext.rot_in(x, x.stride(0), picks, ex.suh_g, ex.suh_u, s.xg, s.xu, rows, D, SLOTS))
    rot1 = _ms(lambda: fx.rot_in1(x, x.stride(0), picks, ex.suh_g, s.xg, rows, D, SLOTS))

    variants = {
        "fast2": (lambda: fx.gateup(s.xg, s.xg, ex.gt, ex.ut, *G, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, D, NI, SLOTS, LIMIT),
                  lambda: fx.down(s.xd, ex.dt, *G, ex.svh_d, y, NI, D, SLOTS)),
    }
    for st in (3, 4):
        variants[f"fat s{st}"] = (
            lambda st=st: fx.gateup_fat(s.xg, s.xg, ex.gt, ex.ut, *G, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, D, NI, SLOTS,
                                        LIMIT, True, st, True),
            lambda st=st: fx.down_fat(s.xd, ex.dt, *G, ex.svh_d, y, NI, D, SLOTS, st, True))
    for name, pair, probe in (("once", True, 0), ("once split-only", False, 0), ("once probe: no decode", True, 1),
                              ("once probe: no mma", True, 2)):
        variants[name] = (
            lambda pair=pair, probe=probe: fx.gateup_once(s.xg, s.xg, ex.gt, ex.ut, *G, ex.svh_g, ex.svh_u, ex.suh_d,
                                                          s.xd, D, NI, SLOTS, LIMIT, True, True, pair, probe),
            lambda pair=pair, probe=probe: fx.down_once(s.xd, ex.dt, *G, ex.svh_d, y, NI, D, SLOTS, True, pair, probe))
    if tc:                                                    # patches/0330
        tx = exl3_mm._tc_ext()
        for name, cfg, ticket, ctas, probe in (("tc", 0, True, 0, 0), ("tc cfg1", 1, True, 0, 0),
                                               ("tc cfg2", 2, True, 0, 0), ("tc cfg3", 3, True, 0, 0),
                                               ("tc static", 0, False, 0, 0), ("tc ctas 44", 0, True, 44, 0),
                                               ("tc ctas 40", 0, True, 40, 0), ("tc probe: no decode", 0, True, 0, 1),
                                               ("tc probe: no mma", 0, True, 0, 2)):
            variants[name] = (
                lambda cfg=cfg, ticket=ticket, ctas=ctas, probe=probe: tx.gateup_tc(
                    s.xg, ex.gt, ex.ut, *G, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, D, NI, SLOTS, LIMIT, cfg, ticket, ctas,
                    probe),
                lambda cfg=cfg, ticket=ticket, ctas=ctas, probe=probe: tx.down_tc(
                    s.xd, ex.dt, *G, ex.svh_d, y, NI, D, SLOTS, cfg, ticket, ctas, probe))
    if fat2:                                                  # patches/0590
        f2 = exl3_mm._fat2_ext()
        for name, cfg, ticket, ctas, probe in (("fat2", 0, True, 0, 0), ("fat2 cfg1", 1, True, 0, 0),
                                               ("fat2 cfg2", 2, True, 0, 0), ("fat2 static", 0, False, 0, 0),
                                               ("fat2 ctas 44", 0, True, 44, 0), ("fat2 ctas 40", 0, True, 40, 0),
                                               ("fat2 probe: no decode", 0, True, 0, 1),
                                               ("fat2 probe: no mma", 0, True, 0, 2)):
            variants[name] = (
                lambda cfg=cfg, ticket=ticket, ctas=ctas, probe=probe: f2.gateup_fat2(
                    s.xg, ex.gt, ex.ut, *G, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, D, NI, SLOTS, LIMIT, cfg, ticket, ctas,
                    probe),
                lambda cfg=cfg, ticket=ticket, ctas=ctas, probe=probe: f2.down_fat2(
                    s.xd, ex.dt, *G, ex.svh_d, y, NI, D, SLOTS, cfg, ticket, ctas, probe))
    if only:
        variants = {k: v for k, v in variants.items() if k == "fast2" or k == "fat s3" or k.startswith(tuple(only))}

    # bits: every non-probe variant's Xd and Y against fast2 (fast2 reads XU = XG here: the shared input)
    ref = None
    same = {}
    for name, (gu, dn) in variants.items():
        if "probe" in name:
            continue
        s.xd.zero_()
        y.zero_()
        gu()
        dn()
        torch.cuda.synchronize()
        got = (s.xd.clone(), y.clone())
        if ref is None:
            ref = got
        same[name] = bool(torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]))

    res = {}
    if v1:
        env = dict(os.environ, GLM53_TF_FAST_EXPERTS="v1")
        out = subprocess.run([sys.executable, __file__, "--v1-child", str(rows), kind], env=env, capture_output=True,
                             text=True)
        if out.returncode == 0:
            res["v1"] = tuple(json.loads(out.stdout.strip().splitlines()[-1]))
        else:
            print(f"  v1 subprocess failed: {out.stderr.strip().splitlines()[-1:]}")
    for name, (gu, dn) in variants.items():
        res[name] = (_ms(gu), _ms(dn))
    under, span = {}, {}
    side_alone = None
    if contend:                                               # patches/0330: the same kernels beside a busy stream
        start, side = _contender()
        for name, (gu, dn) in variants.items():
            under[name] = _ms_under(lambda gu=gu, dn=dn: (gu(), dn()), start, side)
            if fat2 and "probe" not in name:                  # patches/0590: and what both streams pay
                span[name], side_alone = _makespan_under(lambda gu=gu, dn=dn: (gu(), dn()), start, side)

    fat_dec, once_dec = _decodes(n, rows)
    print(f"\nrows {rows}, {kind}: {n.mean():.1f} members an expert (max {n.max()}); a weight tile is decoded "
          f"{fat_dec:.2f}x a chunk by fat, {once_dec:.2f}x by once; rot_in {rot:.2f} ms, rot_in1 {rot1:.2f} ms")
    base = res["fat s3"][0] + res["fat s3"][1]
    floor = _floor_ms(rows)
    print(f"  DRAM floor of the layer (weights once, Xg, Xd, Y at 220 GB/s): {floor:.2f} ms"
          + (f" (at 235 GB/s: {_floor_ms(rows, 235e9):.2f} ms = {_floor_ms(rows, 235e9) / base:.2f}x fat)" if fat2 else ""))
    if span:
        print(f"  side stream's work alone: {side_alone:.2f} ms; makespan = kernel + side work from the side's start")
    print(f"  {'kernels':24s} {'gate/up ms':>10s} {'TF/s':>6s} {'down ms':>8s} {'TF/s':>6s} {'sum ms':>7s} {'vs fat':>7s} "
          f"{'floor':>6s}" + (f" {'contended':>9s} {'x':>5s}" if under else "") + " bits")
    for name, (gu, dn) in res.items():
        bits = "" if "probe" in name or name == "v1" else ("same" if same[name] else "DIFFERENT")
        extra = f" {under[name]:9.2f} {under[name] / (gu + dn):5.2f}" if name in under else ""
        if name in span:
            extra += f"  makespan {span[name]:7.2f}"
        print(f"  {name:24s} {gu:10.2f} {flops_gu / gu / 1e9:6.1f} {dn:8.2f} {flops_dn / dn / 1e9:6.1f} {gu + dn:7.2f} "
              f"{base / (gu + dn):6.2f}x {floor / (gu + dn):6.2f}{extra} {bits}")
    return res, same, under, span


def main(argv):
    if argv[:1] == ["--v1-child"]:
        print(json.dumps(run_v1(int(argv[1]), argv[2])))
        return
    v1 = "--no-v1" not in argv
    tc, contend, fat2 = "--tc" in argv, "--contend" in argv, "--fat2" in argv
    only = ()
    if "--variants" in argv:
        only = tuple(v.strip() for v in argv[argv.index("--variants") + 1].split(",") if v.strip())
    rows = [int(a) for a in argv if a.isdigit()] or [1024, 2048, 4096, 8192]
    print(torch.cuda.get_device_name(), torch.version.cuda)
    bad = []
    gate = {}
    for r in rows:
        for kind in ("uniform", "skewed"):
            res, same, under, span = run(r, kind, v1, tc, contend, only, fat2)
            bad += [(r, kind, k) for k, v in same.items() if not v]
            if fat2 and kind == "uniform" and "fat2" in res:
                gate[r] = (res, same, under, span)
    print("\nALL BITS SAME" if not bad else f"\nBITS DIFFER: {bad}")
    if fat2:
        print(gate_line(gate, contend))


def gate_line(gate, contend):
    """patches/0590's GATE: fat2's bits same, and <= 0.75x fat's (fat s3) time at 2,048 and 4,096 rows beside a busy
    side stream (uniform routing)."""
    if not contend:
        return "GATE 0590: needs --contend (the gate is measured beside a busy side stream)"
    need = [r for r in (2048, 4096) if r not in gate]
    if need:
        return f"GATE 0590: needs rows {need}"
    ok, parts = True, []
    for r in (2048, 4096):
        res, same, under, span = gate[r]
        x = under["fat2"] / under["fat s3"]
        m = span["fat2"] / span["fat s3"] if "fat s3" in span else float("nan")
        iso = sum(res["fat2"]) / sum(res["fat s3"])
        fl = _floor_ms(r, 235e9) / sum(res["fat s3"])
        ok &= bool(same.get("fat2")) and x <= 0.75
        parts.append(f"{r}: contended {x:.3f}x fat (makespan {m:.3f}x, isolated {iso:.3f}x; DRAM floor at 235 GB/s "
                     f"{fl:.3f}x){'' if same.get('fat2') else ' BITS DIFFER'}")
    return f"GATE 0590: {'PASS' if ok else 'FAIL'} (<= 0.75x fat, same bits) -- " + "; ".join(parts)


if __name__ == "__main__":
    main(sys.argv[1:])
