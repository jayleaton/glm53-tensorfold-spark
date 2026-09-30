#!/usr/bin/env python3
"""W20 final glmbench: per-cell table over rounds. table.py TAG [REF_TAG] (in this dir: glmbench-TAG-rN.json).
Per round a cell's value = median decode tok/s of its reps; table = mean / min / max over rounds, greedy (T=0) vs sampled
(T=1) labelled, vs the W20 image b10 control load (../glmbench-B10.json, one round), yesterday's image and the vLLM kit (fixed numbers).
Hash check: every rep of every round of a greedy cell returns the same sha; sampled cells are seeded (same per rep)."""
import glob, json, os, statistics, sys

here = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "final"
YEST = {("tf", "chat", 0.0): 44.6, ("tf", "code", 0.0): 77.6, ("kit", "structured", 0.0): 100.6}
VLLM = {("tf", "chat", 0.0): 22.8, ("tf", "code", 0.0): 41.9, ("kit", "structured", 0.0): 72.7,
        ("kit", "hashmap", 0.0): 30.0, ("kit", "essay", 0.0): 26.1}


def cells(path):
    d = json.load(open(path))
    out = {}
    for s, rows in d["suites"].items():
        for r in rows:
            out[(s, r["prompt"], float(r["temperature"]), r["tokens"])] = (
                statistics.median(x["decode_tps"] for x in r["runs"]), [x["sha256"] for x in r["runs"]])
    return out


rounds = [cells(p) for p in sorted(glob.glob(os.path.join(here, f"glmbench-{tag}-r[0-9].json")))]
ref = cells(os.path.join(here, "..", "glmbench-B10.json"))
print(f"rounds: {len(rounds)} ({tag})\n")
print("| suite | cell | mode | tokens | mean | min | max | W20 b10 | vs B10 | yesterday | vLLM kit | vs vLLM | hashes |")
print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
for k in rounds[0]:
    s, p, t, n = k
    v = [r[k][0] for r in rounds if k in r]
    shas = [r[k][1] for r in rounds if k in r]
    same = "same" if len({h for x in shas for h in x}) == 1 else ("rounds agree" if all(x == shas[0] for x in shas) else "DIFFER")
    m = statistics.mean(v)
    o = ref.get(k, (None,))[0]
    y = YEST.get((s, p, t)); vl = VLLM.get((s, p, t))
    mode = "greedy (T=0)" if t == 0 else "sampled (T=1)"
    f = lambda x: f"{x:.1f}" if x is not None else ""
    print(f"| {s} | {p} | {mode} | {n} | **{m:.1f}** | {min(v):.1f} | {max(v):.1f} | {f(o)} | "
          f"{(100 * (m / o - 1)):+.1f}% | {f(y)} | {f(vl)} | {(f'{m / vl:.2f}x' if vl else '')} | {same} |"
          if o else f"| {s} | {p} | {mode} | {n} | **{m:.1f}** | {min(v):.1f} | {max(v):.1f} | | | {f(y)} | {f(vl)} | | {same} |")
g = [statistics.mean([r[k][0] for r in rounds]) / ref[k][0] for k in rounds[0] if k in ref]
print(f"\ngeomean vs B10 over {len(g)} cells: {100 * (statistics.geometric_mean(g) - 1):+.2f}%")
