"""THEORY-2 item 7 / session step 1c: 0440's dense kernel (q4_stream.cu via decode_stream.run_qmm) vs the old
qmm.matmul (_qmm + _reduce), COLD, per shape of bench_decode_kernels.SHAPES (+ the indexer's two q4 shapes) and rows.

W12's kbench cycled 3 weight copies: every shape < 8 MB stayed in the 24 MB L2 (roofs 311-1,041 GB/s where DRAM gives
235), so its 1.2-2.1x on small shapes was an L2 number. This bench re-times them in the server's state
(THEORY-2 §7 rule 1):

- ``rotate`` (default): per shape, enough weight copies that one cycle is >= 96 MiB (4x L2; e.g. 4096 x 128 needs
  ~330 copies), all captured in ONE CUDA graph (one call per copy), us a call = median graph time / copies. The roof
  probe (bench_decode_kernels' 16-byte streaming read) is rotated the same way; a roof > 240 GB/s means the bytes
  came from L2: the row is marked INVALID and the gate ignores it.
- ``flush``: a 64 MiB streaming Triton kernel (plain loads, so it also evicts L2) before every call inside the
  graph: graph A = N x (pred, call over 3 rotating copies), graph B = N x pred, replayed interleaved; us a call =
  median over reps of (A - B) / N. Cold L2 and a realistic predecessor tail (PDL prefetch overlaps it) in one number.
  The roof is measured the same way (pred + probe).
- ``hot``: 3 copies in a graph, what W12 measured (to show the L2 effect next to the cold rows).

Per shape x rows: old us, every new config (ds.QMM_CFGS x split / serial x PDL off / on when ds.pdl_supported(),
like bench_decode_kernels.bench_qmm), same_bits (torch.equal vs old on copy 0: False = that config is NOT eligible),
GB/s, the best ELIGIBLE config and the speed-up old / new. Rows are bench_decode_kernels-style dicts (keys "shape",
"rows", "MB", "sk", "old_us", "old_GBs", "roof_GBs", "new (g, s) split|serial[ pdl]": {"us", "GBs", "same_bits"},
"auto serial") plus "mode", "name", "copies", "rotation_MB", "roof_valid", "best_new", "best_us", "speedup", so the
JSON compares with results/W12/kbench.json.

Reading the gate (cold rows only; rotate and flush each, when run):

    GATE item7: new >= 1.10x old cold on shapes < 8 MB (KDA o, shared gate/up/down, DSA q_b/kv, index): <list>

A gate shape passes a mode when the geometric mean over the timed rows of old / best-eligible-new is >= 1.10 (valid
rows only); it passes the gate when it passes every cold mode that ran. Passing shapes -> add
GLM53_TF_DEC_QMM_MAX_MB (new kernel only for weights <= that size) and A/B; the line after the gate suggests the
largest size such that every shape at or below it passes.

    PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda \\
        python tests/cuda/bench_decode_cold.py --mode both --json cold.json          # ~4-7 min
    ... --quick                                                                    # gate shapes, rows 1 / 8, ~1-2 min

Needs the image's Triton 3.7.x (the new kernel keeps _qmm's bits only there: ds.qmm_reference_fused()); lock the SM
clock at 2,250 MHz first (production cap). One GPU, nothing else on it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

try:                                     # the predecessor kernel's module globals; importable without Triton / a GPU
    import triton
    import triton.language as tl
except ImportError:                      # pragma: no cover
    triton = tl = None

sys.path.insert(0, str(Path(__file__).parent))

L2_BYTES = 24 * 1024 * 1024
ROT_MIN_BYTES = 4 * L2_BYTES             # 96 MiB a rotation
ROOF_MAX_GBS = 240.0                     # DRAM ~235: above this the bytes came from L2
GATE_X = 1.10
HOT_COPIES = 3
FLUSH_MB = 64
FLUSH_COPIES = 3
ROWS = (1, 2, 4, 8, 16)
EXTRA_SHAPES = [(4096, 1536), (160, 4096)]          # indexer q_b / [wk | weights_proj]: q4, not in bdk.SHAPES
# patches/0570: W11's (1, 16, 8) grid (n 1,024; 8.5 calls a 1-stream round, 16.2 us in situ; likely DFlash2's context
# k|v projection), 2.36 MB: under the 3.5 MiB size switch but not measured cold in W16
EXTRA_SHAPES = EXTRA_SHAPES + [(1024, 4096)]
NAMES = {"12576x4096": "KDA in_proj", "4096x4096": "KDA o", "4096x128": "KDA f_b/g_b", "2048x4096": "shared gate/up",
         "8192x1536": "DSA q_b", "8192x512": "DSA kv_b", "4096x8192": "DSA o", "12288x4096": "MLP gate/up",
         "4096x6144": "MLP down", "4096x1024": "shared down", "77440x4096": "LM head", "4096x1536": "index q_b",
         "160x4096": "index k", "1024x4096": "(1,16,8) grid"}
GATE_SHAPES = ("4096x4096", "2048x4096", "4096x1024", "8192x1536", "8192x512", "4096x1536", "160x4096")


# -- pure helpers (CPU-testable) ---------------------------------------------------------------------------------------
def stats(xs) -> dict:
    v = sorted(float(x) for x in xs if x is not None)
    if not v:
        return {"n": 0, "median": None, "p10": None, "p90": None}

    def pct(p):
        pos = p * (len(v) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)

    return {"n": len(v), "median": statistics.median(v), "p10": pct(0.1), "p90": pct(0.9)}


def copies_for_rotation(nbytes: int, min_bytes: int = ROT_MIN_BYTES, min_copies: int = HOT_COPIES,
                        max_copies: int = 4096) -> int:
    """Weight copies so one cycle through them reads >= min_bytes (at least min_copies)."""

    if nbytes <= 0:
        raise ValueError("copies_for_rotation: nbytes must be > 0")
    return int(min(max_copies, max(min_copies, -(-min_bytes // nbytes))))


def q4_nbytes(n: int, k: int) -> int:
    """Bytes of a qmm.Q4 [n, k]: 4-bit words in 64-column tiles (n rounded up to 64, x k / 2) + bf16 scales and
    biases per 64 inputs (2 x n * k / 64 x 2)."""

    return -(-n // 64) * 64 * k // 2 + 2 * (k // 64) * n * 2


def roof_valid(roof_gbs, limit: float = ROOF_MAX_GBS) -> bool:
    """A cold row's roof must be DRAM-bound (<= limit GB/s); None (not measured) is not valid."""

    return roof_gbs is not None and roof_gbs <= limit


def per_call_us(graph_us: float, calls: int) -> float:
    return graph_us / calls


def flush_call_us(with_us: list, pred_us: list, calls: int) -> float:
    """Interleaved reps of graph(N x (pred, call)) and graph(N x pred): median of the paired differences / N."""

    if not with_us or len(with_us) != len(pred_us):
        raise ValueError("flush_call_us: need equal, non-empty rep lists")
    return statistics.median(a - b for a, b in zip(with_us, pred_us)) / calls


def gbs(nbytes: int, us) -> float | None:
    return None if not us or us <= 0 else nbytes / us / 1e3


def geomean(xs) -> float | None:
    v = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def best_eligible(row: dict):
    """(key, us) of the fastest new config whose bits equal old's; None if there is none."""

    c = [(v["us"], k) for k, v in row.items()
         if k.startswith("new") and isinstance(v, dict) and v.get("same_bits") is True and v.get("us")]
    if not c:
        return None
    us, k = min(c)
    return k, us


def finish_row(row: dict) -> dict:
    """Fill best_new / best_us / speedup (eligible configs only)."""

    b = best_eligible(row)
    row["best_new"], row["best_us"] = (b if b else (None, None))
    row["speedup"] = row["old_us"] / b[1] if b and row.get("old_us") else None
    return row


def gate_item7(results: dict, gate_shapes=GATE_SHAPES, threshold: float = GATE_X) -> dict:
    """results: mode -> list of rows (only the cold modes 'rotate' / 'flush' count). Per gate shape and mode: the
    geometric mean of the valid rows' speed-ups, min, and pass; a shape passes when it passes every cold mode run."""

    modes = [m for m in ("rotate", "flush") if results.get(m)]
    per = {}
    for s in gate_shapes:
        per[s] = {"name": NAMES.get(s, s), "modes": {}}
        for m in modes:
            rows = [r for r in results[m] if r.get("shape") == s]
            valid = [r for r in rows if r.get("roof_valid") and r.get("speedup")]
            sp = [r["speedup"] for r in valid]
            g = geomean(sp)
            per[s]["modes"][m] = {"rows": len(rows), "valid": len(valid), "geomean": g, "min": min(sp) if sp else None,
                                  "pass": g is not None and g >= threshold,
                                  "mb": rows[0].get("MB") if rows else None}
        ms = per[s]["modes"]
        per[s]["measured"] = bool(ms) and all(v["valid"] > 0 for v in ms.values())
        per[s]["pass"] = per[s]["measured"] and all(v["pass"] for v in ms.values())
    passing = [s for s in gate_shapes if per[s]["pass"]]
    unmeasured = [s for s in gate_shapes if not per[s]["measured"]]
    failing = [s for s in gate_shapes if per[s]["measured"] and not per[s]["pass"]]

    def lab(s):
        return f"{s} ({NAMES.get(s, s)})"

    line = (f"GATE item7: new >= {threshold:.2f}x old cold on shapes < 8 MB (KDA o, shared gate/up/down, DSA q_b/kv, "
            f"index) [{'+'.join(modes) or 'no cold mode'}]: passing {[lab(s) for s in passing] or 'none'}; "
            f"failing {[lab(s) for s in failing] or 'none'}"
            + (f"; not measured / invalid {[lab(s) for s in unmeasured]}" if unmeasured else ""))
    return {"modes": modes, "per_shape": per, "passing": passing, "failing": failing, "unmeasured": unmeasured,
            "line": line}


def suggest_max_mb(results: dict, threshold: float = GATE_X) -> float | None:
    """Largest weight size (MB) such that every measured shape at or below it passes in every cold mode: the
    GLM53_TF_DEC_QMM_MAX_MB candidate (None: the smallest shape already fails)."""

    modes = [m for m in ("rotate", "flush") if results.get(m)]
    shapes = {}
    for m in modes:
        for r in results[m]:
            shapes.setdefault(r["shape"], r.get("MB"))
    ok_at = None
    for s, mb in sorted(shapes.items(), key=lambda t: t[1] or 0):
        g = gate_item7(results, gate_shapes=(s,), threshold=threshold)
        if not g["per_shape"][s]["measured"]:
            continue
        if not g["per_shape"][s]["pass"]:
            break
        ok_at = mb
    return ok_at


def fmt_row(r: dict) -> str:
    rv = "" if r.get("roof_GBs") is None else (f"roof {r['roof_GBs']:.0f} GB/s"
                                               + ("" if r.get("roof_valid", True) else " INVALID (L2-resident)"))
    head = (f"{r['mode']:6s} {r['shape']:>10s} {NAMES.get(r['shape'], ''):14s} x{r['rows']:>2} rows: {r['MB']:.2f} MB, "
            f"{r.get('copies', '?')} copies ({r.get('rotation_MB', 0):.0f} MB) {rv}")
    if r.get("old_us") is None:
        return head + f" | ERROR {r.get('error', '?')}"
    s = head + f" | old {r['old_us']:.2f} us {r.get('old_GBs') or 0:.0f} GB/s"
    if r.get("best_new"):
        s += (f" | best new {r['best_new'][4:]} {r['best_us']:.2f} us {gbs(int(r['MB'] * 1e6), r['best_us']):.0f} GB/s "
              f"({r['speedup']:.3f}x)")
    else:
        s += " | no eligible new config (bits differ or failed)"
    return s


# -- GPU part (lazy) --------------------------------------------------------------------------------------------------------
_TK = None


def pred_kernel():
    """The flush / predecessor kernel: a plain-load (L2-allocating) streaming read, Triton, built lazily."""

    global _TK
    if _TK is None:
        if triton is None:
            raise RuntimeError("triton is not installed")

        @triton.jit
        def stream_k(P, OUT, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
            pid = tl.program_id(0)
            acc = tl.zeros((BLOCK,), dtype=tl.int32)
            for i in range(ITERS):
                off = (pid * ITERS + i) * BLOCK + tl.arange(0, BLOCK)
                acc += tl.load(P + off, mask=off < n, other=0)
            tl.store(OUT + pid, tl.sum(acc, axis=0))

        _TK = stream_k
    return _TK


class Gpu:
    def __init__(self):
        import torch

        import bench_decode_kernels as bdk

        self.torch, self.bdk, self.ds, self.qmm = torch, bdk, bdk.ds, bdk.qmm
        self.dev = bdk.DEV
        self.ctas = torch.cuda.get_device_properties(0).multi_processor_count * 8
        self.out = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.part = torch.empty((8 * 64 * 77440,), dtype=torch.float32, device=self.dev)
        n = FLUSH_MB * 1024 * 1024 // 4
        self.pbuf = torch.randint(0, 1 << 20, (n,), dtype=torch.int32, device=self.dev)
        self.pn, self.PB, self.PI = n, 2048, 8
        self.pgrid = -(-n // (self.PB * self.PI))
        self.pout = torch.zeros(self.pgrid, dtype=torch.int32, device=self.dev)

    def pred(self):
        pred_kernel()[(self.pgrid,)](self.pbuf, self.pout, self.pn, BLOCK=self.PB, ITERS=self.PI, num_warps=8)

    def graph(self, fns, ctx=None):
        """Warm eagerly, capture ``fns`` into one graph (inside ``ctx`` if given: the old path's knobs)."""

        torch = self.torch
        import contextlib

        with (ctx or contextlib.nullcontext()):
            for f in fns:
                f()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for f in fns:
                    f()
        g.replay()
        torch.cuda.synchronize()
        return g

    def time_graph(self, g) -> float:
        torch = self.torch
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1e3

    def call_us(self, mode: str, call_fns: list, reps: int, flush_calls: int, gpred=None, ctx=None) -> float:
        """us a call. rotate / hot: one graph over every fn (one per copy). flush: N x (pred, fn) vs N x pred."""

        if mode in ("rotate", "hot"):
            g = self.graph(call_fns, ctx)
            t = [self.time_graph(g) for _ in range(reps)]
            del g
            return per_call_us(statistics.median(t), len(call_fns))
        seq = []
        for i in range(flush_calls):
            seq += [self.pred, call_fns[i % len(call_fns)]]
        ga = self.graph(seq, ctx)
        wa, wb = [], []
        for _ in range(reps):
            wa.append(self.time_graph(ga))
            wb.append(self.time_graph(gpred))
        del ga
        return flush_call_us(wa, wb, flush_calls)

    def q4_copies(self, n: int, k: int, copies: int):
        bdk, qmm = self.bdk, self.qmm
        base = [bdk._q4(n, k, c) for c in range(min(copies, HOT_COPIES))]
        qs = list(base)
        while len(qs) < copies:
            q = base[len(qs) % len(base)]
            qs.append(qmm.Q4(q.weight.clone(), q.scales.clone(), q.biases.clone(), q.n, q.k))
        return qs

    def roof(self, mode: str, nbytes: int, copies: int, reps: int, flush_calls: int, gpred) -> float:
        torch, bdk = self.torch, self.bdk
        nbytes = max(16, nbytes // 16 * 16)
        bufs = [torch.empty(nbytes, dtype=torch.uint8, device=self.dev) for _ in range(copies)]
        fns = [lambda b=b: bdk._probe().probe(b, nbytes, self.ctas, self.out) for b in bufs]
        us = self.call_us(mode, fns, reps, flush_calls, gpred)
        del bufs, fns
        return gbs(nbytes, us)


def bench_shape(G: Gpu, mode: str, n: int, k: int, rows, pdls, reps: int, flush_calls: int, log) -> list:
    torch, ds, qmm = G.torch, G.ds, G.qmm
    shape = f"{n}x{k}"
    nbytes = q4_nbytes(n, k)
    copies = {"rotate": copies_for_rotation(nbytes), "hot": HOT_COPIES, "flush": FLUSH_COPIES}[mode]
    qs = G.q4_copies(n, k, copies)
    if qs[0].nbytes() != nbytes:                      # a layout change: recompute with the real size
        nbytes = qs[0].nbytes()
        if mode == "rotate" and copies_for_rotation(nbytes) > copies:
            qs = G.q4_copies(n, k, copies_for_rotation(nbytes))
        copies = len(qs)
    gpred = None
    if mode == "flush":
        gpred = G.graph([G.pred] * flush_calls)
    base = {"mode": mode, "shape": shape, "name": NAMES.get(shape, ""), "MB": nbytes / 1e6, "sk": qmm.split_k(n, k),
            "copies": copies, "rotation_MB": copies * nbytes / 1e6}
    try:
        roof = G.roof(mode, nbytes, copies, reps, flush_calls, gpred)
    except Exception as e:  # noqa: BLE001
        roof = None
        base["roof_error"] = f"{type(e).__name__}: {e}"
    base["roof_GBs"] = roof
    base["roof_valid"] = True if mode == "hot" else roof_valid(roof)
    out = []
    for m in rows:
        row = dict(base, rows=m)
        try:
            x = torch.randn((m, k), device=G.dev).to(torch.bfloat16)
            xs = qmm.group_sums(x)
            part = G.part
            with ds.using(qmm_on=False):
                ref = qmm.matmul(x, qs[0], xs, part=part).clone()
            t_old = G.call_us(mode, [lambda q=q: qmm.matmul(x, q, xs, part=part) for q in qs], reps, flush_calls,
                              gpred, ctx=ds.using(qmm_on=False))
            row.update(old_us=t_old, old_GBs=gbs(nbytes, t_old))
            if not ds.qmm_reference_fused():
                row["note"] = "Triton is not 3.7.x: the stream kernel would not keep _qmm's bits (timed anyway)"
            sk = qmm.split_k(n, k)
            serials = (False, True) if sk > 1 else (True,)
            for cfg in ds.QMM_CFGS:
                for pdl in pdls:
                    for serial in serials:
                        tag = f"new {cfg}{' serial' if serial else ' split'}{' pdl' if pdl else ''}"
                        try:
                            got = ds.run_qmm(x, qs[0], xs, part=part, cfg=cfg, pdl=pdl, serial=serial)
                            torch.cuda.synchronize()
                            same = bool(torch.equal(got, ref))
                            t = G.call_us(mode, [lambda q=q: ds.run_qmm(x, q, xs, part=part, cfg=cfg, pdl=pdl,
                                                                        serial=serial) for q in qs],
                                          reps, flush_calls, gpred)
                            row[tag] = {"us": t, "GBs": gbs(nbytes, t), "same_bits": same}
                        except Exception as e:  # noqa: BLE001
                            row[tag] = {"us": None, "GBs": None, "same_bits": None,
                                        "error": f"{type(e).__name__}: {e}"}
            row["auto serial"] = ds.serial_for(n, sk, "auto")
            finish_row(row)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            row["error"] = f"{type(e).__name__}: {e}"
            row.setdefault("old_us", None)
        out.append(row)
        log(fmt_row(row))
    del qs, gpred
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="cold re-bench of 0440 E2 (q4_stream) vs qmm.matmul (THEORY-2 item 7)")
    ap.add_argument("--mode", choices=["rotate", "flush", "both"], default="rotate")
    ap.add_argument("--no-hot", action="store_true", help="skip the 3-copy (L2-hot, W12-style) rows")
    ap.add_argument("--quick", action="store_true", help="gate shapes + KDA in_proj, rows 1 / 8")
    ap.add_argument("--json")
    ap.add_argument("--rows", type=int, nargs="+", default=None, help=f"row counts (default {ROWS})")
    ap.add_argument("--shapes", nargs="*", default=None, help="NxK subset (default: bdk.SHAPES + the indexer's)")
    ap.add_argument("--reps", type=int, default=9, help="graph replays (rotate / hot) a timing")
    ap.add_argument("--flush-reps", type=int, default=15, help="interleaved (pred+call, pred) pairs a timing")
    ap.add_argument("--flush-calls", type=int, default=12, help="(pred, call) pairs in the flush graph")
    ap.add_argument("--no-pdl", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print every config's line")
    a = ap.parse_args()

    def log(s):
        print(s, flush=True)

    res = {"argv": sys.argv, "errors": []}
    try:
        G = Gpu()
        torch, ds = G.torch, G.ds
        res.update(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                   triton=getattr(triton, "__version__", None), triton_fused_qmm=ds.qmm_reference_fused(),
                   pdl_supported=ds.pdl_supported())
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        res["errors"].append(f"setup: {type(e).__name__}: {e}")
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1, default=str))
        return 1
    pdls = [False] + ([True] if ds.pdl_supported() and not a.no_pdl else [])
    shapes = list(G.bdk.SHAPES) + [s for s in EXTRA_SHAPES if s not in G.bdk.SHAPES]
    rows = tuple(a.rows) if a.rows else ROWS
    if a.quick:
        shapes = [s for s in shapes if f"{s[0]}x{s[1]}" in GATE_SHAPES] + [(12576, 4096)]
        rows = tuple(a.rows) if a.rows else (1, 8)
    if a.shapes:
        want = set(a.shapes)
        shapes = [s for s in shapes if f"{s[0]}x{s[1]}" in want]
    modes = {"rotate": ["rotate"], "flush": ["flush"], "both": ["rotate", "flush"]}[a.mode]
    if not a.no_hot:
        modes.append("hot")
    log(f"[cold] device={res['device']} triton={res['triton']} fused_ref={res['triton_fused_qmm']} pdl={pdls} "
        f"modes={modes} rows={rows} shapes={len(shapes)}")
    if not res["triton_fused_qmm"]:
        log("[cold] WARNING: Triton is not 3.7.x: same_bits will be False on some shapes (not the image?)")
    t0 = time.perf_counter()
    for mode in modes:
        res[mode] = []
        reps = a.flush_reps if mode == "flush" else a.reps
        for n, k in shapes:
            try:
                res[mode] += bench_shape(G, mode, n, k, rows, pdls, reps, a.flush_calls, log)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                res["errors"].append(f"{mode} {n}x{k}: {type(e).__name__}: {e}")
                log(f"[cold] {mode} {n}x{k}: ERROR {type(e).__name__}: {e}")
            log(f"[cold] {mode} {n}x{k} done, {time.perf_counter() - t0:.0f} s")
            if a.verbose and res[mode]:
                for r in res[mode][-len(rows):]:
                    for kk, v in r.items():
                        if isinstance(v, dict) and kk.startswith("new"):
                            b = ("  same bits" if v.get("same_bits") else "  BITS DIFFER") if v.get("us") else \
                                f"  ERROR {v.get('error')}"
                            log(f"    {r['shape']} x{r['rows']} {kk:26s} {v.get('us') or 0:9.2f} us "
                                f"{v.get('GBs') or 0:6.0f} GB/s{b}")

    # -- summary -----------------------------------------------------------------------------------------------------
    log("\n== summary: speed-up old / best eligible new (geomean over rows; * = a roof > 240 GB/s: INVALID, ignored "
        "by the gate; rNNN = roof GB/s)")
    by_shape = {}
    for mode in modes:
        for r in res.get(mode, []):
            by_shape.setdefault(r["shape"], {}).setdefault(mode, []).append(r)
    log(f"{'shape':>11s} {'name':14s} {'MB':>6s} " + " ".join(f"{m:>28s}" for m in modes))
    for s, d in sorted(by_shape.items(), key=lambda t: t[1][next(iter(t[1]))][0]["MB"]):
        first = d[next(iter(d))][0]
        cells = []
        for m in modes:
            rs = d.get(m, [])
            sp = [r["speedup"] for r in rs if r.get("speedup")]
            inval = any(not r.get("roof_valid") for r in rs)
            gm = geomean(sp)
            cell = "n/a" if gm is None else f"{gm:.3f}x (min {min(sp):.2f})"
            cell += "*" if inval else " "
            roof = rs[0].get("roof_GBs") if rs else None
            if roof is not None:
                cell += f" r{roof:.0f}"
            cells.append(cell)
        log(f"{s:>11s} {NAMES.get(s, ''):14s} {first['MB']:6.2f} " + " ".join(f"{c:>28s}" for c in cells))
    g = gate_item7(res)
    res["gate"] = {k: v for k, v in g.items() if k != "line"}
    res["gate"]["line"] = g["line"]
    mx = suggest_max_mb(res)
    res["gate"]["suggest_max_mb"] = mx
    log(g["line"])
    log(f"GLM53_TF_DEC_QMM_MAX_MB candidate (every measured shape <= it passes cold): "
        f"{'none' if mx is None else f'{mx:.1f}'}")
    if res["errors"]:
        log(f"[cold] {len(res['errors'])} error(s): " + " | ".join(res["errors"]))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
        log(f"[cold] wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
