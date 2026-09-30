#!/usr/bin/env python3
"""W19: per-load checks summ.py does not print. extra.py NAME [REF_GLMBENCH]
- replay: every ab.py warm resend has cached == (prompt - 1) // 64 * 64 (0540's rule) and the same sha as its cold send;
- glmbench cell hashes vs a reference load's glmbench JSON (default results/W18/glmbench-A.json = b9 prod): the
  knobs-off control == b9 check (greedy cells must match; sampled cells too when the seed path is unchanged);
- the needle resend's cached vs the same rule;
- structured.py (0610) and ports.py (0600) summary lines."""
import json
import os
import sys

R = os.path.dirname(os.path.abspath(__file__))
N = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "..", "W18", "glmbench-A.json")


def J(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


ok, tot = 0, 0
for suf in ("", "-2", "-3", "-4"):
    for row in J(f"{R}/ab-{N}{suf}.json") or []:
        c, w = row["cold"], row["warm"]
        want = (c["prompt_tokens"] - 1) // 64 * 64
        tot += 1
        ok += w.get("cached") == want and w.get("sha") == c.get("sha")
print(f"replay (ab.py warm resends): cached == (n-1)//64*64 and sha == cold: {ok}/{tot}")
nd = J(f"{R}/needle-{N}.json")
if nd:
    c, w = nd.get("cold", {}), nd.get("warm", {})
    pt = c.get("prompt_tokens")
    if pt:
        print(f"needle resend: cached {w.get('cached')} (rule {(pt - 1) // 64 * 64}), found {w.get('found')}")


def cells(d):
    out = {}
    for s, rows in (d or {}).get("suites", {}).items():
        for r in rows:
            out[f"{s} {r['prompt']} T={r['temperature']} {r['tokens']}"] = (sorted({x["sha256"] for x in r["runs"]}),
                                                                             r["temperature"])
    return out


a, b = cells(J(f"{R}/glmbench-{N}.json")), cells(J(REF))
same = [k for k in a if k in b and a[k][0] == b[k][0]]
diff = [k for k in a if k in b and a[k][0] != b[k][0]]
print(f"glmbench hashes vs {os.path.relpath(REF, R)}: {len(same)}/{len([k for k in a if k in b])} equal"
      + (f"; differ: {diff}" if diff else ""))
st = J(f"{R}/structured-{N}.json")
if st:
    print("structured:", json.dumps({k: (v if not isinstance(v, (dict, list)) else "...") for k, v in st.items()})[:300])
p = J(f"{R}/ports-{N}.json")
if p:
    print("ports: disconnect freed", p.get("disconnect_one", {}).get("freed_s"), "s; 4 freed",
          p.get("disconnect_four", {}).get("freed_s"), "s; queued same", p.get("queued", {}).get("same"),
          "; errors", {k: v[0] for k, v in p.get("errors", {}).items()}, "; usr1", p.get("usr1"),
          "; image equal", p.get("image", {}).get("equal"))
