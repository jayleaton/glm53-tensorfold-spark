#!/usr/bin/env python3
"""W17: glmbench cells of A vs B: median decode tok/s, delta, reply hashes equal (greedy cells), geomean. cmpglm.py A.json B.json"""
import json, math, statistics, sys
def cells(f):
    d = json.load(open(f))["suites"]; out = {}
    for s, rows in d.items():
        for r in rows:
            k = f"{s} {r['prompt']} T={r['temperature']} {r['tokens']}"
            out[k] = (statistics.median(x["decode_tps"] for x in r["runs"]), sorted({x["sha256"] for x in r["runs"]}))
    return out
a, b = cells(sys.argv[1]), cells(sys.argv[2]); logs = []; same = n = 0
for k in a:
    if k not in b: continue
    d = a[k][0] / b[k][0] - 1; logs.append(math.log(a[k][0] / b[k][0]))
    eq = a[k][1] == b[k][1]; n += 1; same += eq
    print(f"{k:34s} {a[k][0]:7.2f} {b[k][0]:7.2f} {d:+6.1%} hashes {'==' if eq else '!='}")
print(f"geomean {math.exp(sum(logs)/len(logs))-1:+.2%}; hashes equal {same}/{n}")
