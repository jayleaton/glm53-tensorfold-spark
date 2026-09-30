#!/usr/bin/env python3
"""W19: control vs combined from the two captures (w19att.py outputs). attcmp.py CONTROL COMBINED [rank]

Per segment (prefill; 1 stream prose; 1 stream code; 4 streams): per family the exclusive kernel time (families + idle
= wall) normalised -- prefill: us a prompt token; decode: ms a generated token (the segment's own tokens) -- control,
combined and the delta, plus the summed (non-exclusive) kernel time of the families the fixes touch."""
import json
import os
import sys

R = os.path.dirname(os.path.abspath(__file__))
a, b = sys.argv[1], sys.argv[2]
rk = sys.argv[3] if len(sys.argv) > 3 else "r0"
A = {s["name"]: s for s in json.load(open(f"{R}/att-{a}-{rk}.json"))["segments"]}
B = {s["name"]: s for s in json.load(open(f"{R}/att-{b}-{rk}.json"))["segments"]}


def norm(s):
    if s["name"] == "prefill":
        return 1e3 / max(s.get("prompt_tokens") or 1, 1), "us/prompt token"
    return 1.0 / max(s.get("tokens") or 1, 1), "ms/token"


for seg in ("prefill", "prose", "code", "conc4"):
    if seg not in A or seg not in B:
        print(f"== {seg}: missing ({seg in A} / {seg in B})")
        continue
    sa, sb = A[seg], B[seg]
    fa, unit = norm(sa)
    fb, _ = norm(sb)
    print(f"== {seg} ({rk}): wall {sa['wall_ms']:.0f} -> {sb['wall_ms']:.0f} ms; tokens {sa.get('tokens', sa.get('prompt_tokens'))}"
          f" / {sb.get('tokens', sb.get('prompt_tokens'))}; [{unit}]")
    fams = sorted(set(sa["excl_ms"]) | set(sb["excl_ms"]), key=lambda k: -sa["excl_ms"].get(k, 0))
    rows = [(k, sa["excl_ms"].get(k, 0) * fa, sb["excl_ms"].get(k, 0) * fb, sa["sum_ms"].get(k, 0) * fa,
             sb["sum_ms"].get(k, 0) * fb) for k in fams]
    rows.append(("GPU idle", sa["idle_ms"] * fa, sb["idle_ms"] * fb, sa["idle_ms"] * fa, sb["idle_ms"] * fb))
    rows.append(("WALL", sa["wall_ms"] * fa, sb["wall_ms"] * fb, sa["wall_ms"] * fa, sb["wall_ms"] * fb))
    print(f"  {'family':32s} {'ctl excl':>9s} {'comb excl':>9s} {'delta':>8s} {'%':>7s} | {'ctl sum':>8s} {'comb sum':>8s}")
    for k, x, y, xs, ys in rows:
        print(f"  {k:32s} {x:9.3f} {y:9.3f} {y - x:+8.3f} {((y / x - 1) * 100 if x else 0):+6.1f}% | {xs:8.3f} {ys:8.3f}")

print("\n== kernel names of the touched families (summed kernel ms, normalised as above)")
for seg in ("prefill", "prose", "code", "conc4"):
    if seg not in A or seg not in B:
        continue
    fa, unit = norm(A[seg])
    fb, _ = norm(B[seg])
    for f in ("routed experts", "dense q4 / GEMM", "NCCL", "RoCE all-gather", "hc"):
        na = dict((n, v) for n, v in A[seg]["top_names"].get(f, []))
        nb = dict((n, v) for n, v in B[seg]["top_names"].get(f, []))
        names = sorted(set(na) | set(nb), key=lambda n: -(na.get(n, 0) + nb.get(n, 0)))[:6]
        print(f"  {seg:7s} {f:18s} " + "; ".join(f"{n} {na.get(n, 0) * fa:.3f} -> {nb.get(n, 0) * fb:.3f}" for n in names))
