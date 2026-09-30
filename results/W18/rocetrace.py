#!/usr/bin/env python3
"""W16: the RoCE per-exchange trace of a load (patches/0530 GLM53_TF_ROCE_TRACE_DUMP; 0460's ring) -> transport vs
skew, THEORY-2 §1.4's table measured in-server without nsys (idea 6's gate). Standard library only.

  rocetrace.py PREFIX [--vs PREFIX2] [--json OUT]

PREFIX-r0.jsonl / PREFIX-r1.jsonl: one line a dump ({"t", "rank", "completed", "rows": [{seq, start, bell, flag, end,
seen, posted}]}). Rows of a rank are merged by seq (dumps overlap), then paired across ranks by seq. Per op, a rank's
``wait`` = its doorbell -> the peer's flags seen (the peer's lateness + transport); ``transport`` = the smaller of the
two ranks' waits (neither was waiting for a late peer); ``skew`` = |wait r0 - wait r1|. ``beyond`` = a rank's mean wait
- mean transport: THEORY-2's "waiting beyond transport" (1.26 / 1.49 ms a round at one stream in W11, ~100 exchanges a
round). Gate (THEORY-2 §4 item 6): K's mean skew wait (beyond, both ranks) >= 25% lower than CT's.
"""
import argparse
import json
import statistics
import sys


def rows_of(path):
    by = {}
    try:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                for r in d["rows"]:
                    if r.get("start") and r.get("bell") and r.get("flag") and r.get("end"):
                        by[r["seq"]] = r
    except FileNotFoundError:
        return {}
    return by


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def stats(v):
    return {"n": len(v), "mean_us": round(statistics.fmean(v), 2) if v else None, "p50_us": round(pct(v, 0.5), 2),
            "p90_us": round(pct(v, 0.9), 2)}


def summarize(prefix):
    a, b = rows_of(prefix + "-r0.jsonl"), rows_of(prefix + "-r1.jsonl")
    out = {"prefix": prefix, "ops_r0": len(a), "ops_r1": len(b)}
    for name, rows in (("r0", a), ("r1", b)):
        comp = {k: [] for k in ("stage", "wait", "copy", "total", "notice", "post")}
        for r in rows.values():
            comp["stage"].append((r["bell"] - r["start"]) / 1e3)
            comp["wait"].append((r["flag"] - r["bell"]) / 1e3)
            comp["copy"].append((r["end"] - r["flag"]) / 1e3)
            comp["total"].append((r["end"] - r["start"]) / 1e3)
            if r.get("seen") and r.get("posted"):
                comp["notice"].append((r["seen"] - r["bell"]) / 1e3)
                comp["post"].append((r["posted"] - r["seen"]) / 1e3)
        out[name] = {k: stats(v) for k, v in comp.items() if v}
    both = sorted(set(a) & set(b))
    wa = [(a[s]["flag"] - a[s]["bell"]) / 1e3 for s in both]
    wb = [(b[s]["flag"] - b[s]["bell"]) / 1e3 for s in both]
    tr = [min(x, y) for x, y in zip(wa, wb)]
    sk = [abs(x - y) for x, y in zip(wa, wb)]
    out["paired"] = len(both)
    if both:
        out["transport"] = stats(tr)
        out["skew"] = stats(sk)
        mt = statistics.fmean(tr)
        out["beyond_us_per_op"] = {"r0": round(statistics.fmean(wa) - mt, 2), "r1": round(statistics.fmean(wb) - mt, 2)}
        out["r0_late_share"] = round(sum(1 for x, y in zip(wa, wb) if y > x) / len(both), 3)   # r1 waited longer
    return out


def show(s):
    print(f"{s['prefix']}: ops r0 {s['ops_r0']} r1 {s['ops_r1']}, paired {s['paired']}")
    for k in ("transport", "skew"):
        if k in s:
            v = s[k]
            print(f"  {k:9} mean {v['mean_us']:7.2f} us  p50 {v['p50_us']:7.2f}  p90 {v['p90_us']:7.2f}  (n {v['n']})")
    if "beyond_us_per_op" in s:
        print(f"  waiting beyond transport, us an exchange: r0 {s['beyond_us_per_op']['r0']}, r1 "
              f"{s['beyond_us_per_op']['r1']}; rank 0 late on {100 * s['r0_late_share']:.0f}% of exchanges")
    for r in ("r0", "r1"):
        if s.get(r):
            print(f"  {r}: " + ", ".join(f"{k} p50 {v['p50_us']}" for k, v in s[r].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--vs")
    ap.add_argument("--json")
    a = ap.parse_args()
    s = summarize(a.prefix)
    show(s)
    res = {"load": s}
    if a.vs:
        c = summarize(a.vs)
        show(c)
        res["control"] = c
        if s.get("beyond_us_per_op") and c.get("beyond_us_per_op"):
            k = sum(s["beyond_us_per_op"].values())
            k0 = sum(c["beyond_us_per_op"].values())
            drop = 1 - k / k0 if k0 > 0 else float("nan")
            res["gate"] = {"beyond_sum_us": round(k, 2), "control_beyond_sum_us": round(k0, 2),
                           "drop": round(drop, 3), "pass": bool(drop >= 0.25)}
            print(f"GATE item6 (skew wait -25%): load {k:.2f} vs control {k0:.2f} us an exchange (r0 + r1): "
                  f"{100 * drop:+.0f}% lower -> {'PASS' if drop >= 0.25 else 'FAIL'} (or 1s / 4s >= +0.7%: summary.py)")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
