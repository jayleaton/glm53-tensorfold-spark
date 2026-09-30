#!/usr/bin/env python3
"""W19: patches/0590 (fat2) kernel gate, FLOOR-based (the revised rule; the 0.75x letter gate is printed too but is
physically out of reach: EXPERT-PREFILL-V2.md section 6).

    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python fat2gate.py OUT.json [rows ...]

Runs tests/cuda/bench_experts.py's run() for each row count (default 2048 4096 8192), uniform and skewed, with
--fat2 --contend and the variants fast2, fat, fat2 (fat2 cfg1/cfg2/static/ctas/probes come with fat2), then:

    GATE 0590 floor: bits same (every row count, both routings) AND at 4,096 rows fat2 contended <= 1.12 x the layer's
    DRAM floor at 235 GB/s AND fat2's makespan beside the busy side stream <= fat's (fat s3), at 2,048 and 4,096 rows.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/work/tests/cuda")
import bench_experts as be  # noqa: E402

out = sys.argv[1]
rows = [int(a) for a in sys.argv[2:]] or [2048, 4096, 8192]
res_all, bad, gate = {}, [], {}
for r in rows:
    for kind in ("uniform", "skewed"):
        res, same, under, span = be.run(r, kind, False, False, True, ("fast2", "fat", "fat2"), True)
        res_all[f"{r}-{kind}"] = {"ms": {k: list(v) for k, v in res.items()}, "same": same, "contended": under,
                                  "makespan": span, "floor235": be._floor_ms(r, 235e9), "floor220": be._floor_ms(r)}
        bad += [(r, kind, k) for k, v in same.items() if not v]
        if kind == "uniform":
            gate[r] = (res, same, under, span)
        json.dump(res_all, open(out, "w"), indent=1)

print("\nALL BITS SAME" if not bad else f"\nBITS DIFFER: {bad}")
print(be.gate_line(gate, True))
lines, ok = [], not bad
for r in (2048, 4096):
    if r not in gate:
        continue
    res, same, under, span = gate[r]
    fl = be._floor_ms(r, 235e9)
    xf = under["fat2"] / fl
    ms = span["fat2"] / span["fat s3"]
    iso = sum(res["fat2"]) / sum(res["fat s3"])
    ok &= ms <= 1.0
    if r == 4096:
        ok &= xf <= 1.12
    lines.append(f"{r}: fat2 contended {under['fat2']:.2f} ms = {xf:.3f}x floor(235) {fl:.2f} ms"
                 f"{' (bar 1.12)' if r == 4096 else ''}; makespan fat2/fat {ms:.3f} (bar 1.00); isolated fat2/fat {iso:.3f};"
                 f" contended fat2/fat {under['fat2'] / under['fat s3']:.3f}")
line = f"GATE 0590 floor: {'PASS' if ok else 'FAIL'} -- " + "; ".join(lines)
print(line)
res_all["gate_floor"] = {"pass": bool(ok), "line": line, "bits_bad": bad}
json.dump(res_all, open(out, "w"), indent=1)
