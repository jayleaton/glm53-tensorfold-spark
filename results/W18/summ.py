#!/usr/bin/env python3
"""W18: one table row set per load. summ.py CONTROL [LOAD ...] (names as in results/W18: ab-N*.json, glmbench-N.json, ...)
Prefill = every ab.py cold prefill (24.5k / 98k) in run order; glmbench geomean vs CONTROL (+ greedy / all hashes equal);
4 streams = mean of the 6 conc reps; 1 stream = conc1 if present. Stdlib only."""
import glob, json, math, os, re, statistics, sys

R = os.environ.get("SUMM_DIR") or os.path.dirname(os.path.abspath(__file__))
SHA = "8794a3463259cc2f"


def J(p):
    try:
        return json.load(open(os.path.join(R, p)))
    except (OSError, ValueError):
        return None


def T(p):
    try:
        return open(os.path.join(R, p)).read()
    except OSError:
        return ""


def cells(n):
    d = J(f"glmbench-{n}.json")
    if not d:
        return {}
    out = {}
    for s, rows in d["suites"].items():
        for r in rows:
            k = f"{s} {r['prompt']} T={r['temperature']} {r['tokens']}"
            out[k] = (statistics.median(x["decode_tps"] for x in r["runs"]), sorted({x["sha256"] for x in r["runs"]}),
                      r["temperature"])
    return out


def conc(n):
    reps = []
    for p in ("a", "b"):
        for line in T(f"conc-{n}-{p}.log").splitlines():
            m = re.search(r"4 streams rep \d+: aggregate ([\d.]+)", line)
            if m:
                reps.append(float(m.group(1)))
    return reps


def ab(n):
    out, shas = [], set()
    for suf in ("", "-2", "-3", "-4"):
        d = J(f"ab-{n}{suf}.json") or []
        for row in d:
            out.append((row["ctx"], row["cold"]["prefill_tps"], row["cold"]["t"]))
            shas |= {row["cold"]["sha"], row["warm"]["sha"]}
    return out, shas


ctl = sys.argv[1]
cc = cells(ctl)
for n in sys.argv[1:]:
    print(f"== {n}")
    ex = [T(f"exact-{n}{s}.log").count("identical=True") for s in ("", "2")]
    bx = [re.findall(r"batched == alone: (\d+/\d+)", T(f"batchexact-{n}{s}.log")) for s in ("", "2")]
    tr = [l for l in T(f"transcripts-{n}.log").splitlines() if "together" in l]
    print(f"  exact {ex[0]}/10, {ex[1]}/10 | batchexact {bx} | transcripts {tr[-1][:90] if tr else '-'}")
    rows, shas = ab(n)
    print(f"  reply sha {sorted(shas)} {'OK' if shas == {SHA} else 'MISMATCH' if shas else '-'}")
    for ctx in (24500, 98000):
        v = [r[1] for r in rows if r[0] == ctx]
        print(f"  prefill {ctx}: " + ", ".join(f"{x:,.0f}" for x in v) + (f"  (min {min(v):,.0f}, mean {statistics.mean(v):,.0f})" if v else ""))
    c = cells(n)
    if c and cc:
        logs, eqg, eqa, k = [], 0, 0, 0
        for key in c:
            if key in cc:
                logs.append(math.log(c[key][0] / cc[key][0])); k += 1
                eqa += c[key][1] == cc[key][1]
                eqg += c[key][2] == 0 and c[key][1] == cc[key][1]
        ng = sum(1 for key in c if key in cc and c[key][2] == 0)
        print(f"  glmbench 1 stream geomean vs {ctl} {math.exp(sum(logs) / len(logs)) - 1:+.2%}; greedy hashes {eqg}/{ng}, all {eqa}/{k}")
    r4 = conc(n)
    if r4:
        print(f"  4 streams mean of {len(r4)} {statistics.mean(r4):.2f} (sd {statistics.pstdev(r4):.2f}) [{', '.join(f'{x:.1f}' for x in r4)}]")
    q = T(f"quality-{n}.log").strip().replace("\n", "; ")
    if q:
        print(f"  {q}")
    n1 = [l for l in T(f"n1-{n}.log").splitlines() if l.startswith("N1")]
    if n1:
        print("  " + " | ".join(n1))
    nd = J(f"needle-{n}.json")
    if nd:
        c, w = nd.get("cold", {}), nd.get("warm", {})
        print(f"  needle {c.get('prompt_tokens')} tokens: cold found {c.get('found')} prefill {c.get('prefill_s')} s "
              f"({c.get('new_prefill_tps')} tok/s); resend found {w.get('found')} cached {w.get('cached')}")
    ms = T(f"memsum-{n}.txt")
    if ms:
        print("  " + ms.strip().replace("\n", "\n  "))
    o = T(f"oom-{n}.txt").strip()
    if o:
        print("  " + o)
