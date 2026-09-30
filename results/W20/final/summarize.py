#!/usr/bin/env python3
"""FINAL-20260930: aggregate 3 rounds of benchmarks on the final production config into summary.md.

    summarize.py [DIR]     (DIR defaults to this script's directory; writes DIR/summary.md)

Expected in DIR (any file or round may be missing; each table reports n = rounds found):
  rigmark-r{1,2,3}/        copy of a `scripts/rigmark/run.sh tensorfold` result dir (the receipt *.json inside;
                           metadata/models/preflight json are skipped)
  glmbench-r{1,2,3}.json   bench/glmbench.py --suites tf,kit,edit --reps 3 --long-tokens 512
  conc1-r{1,2,3}.json      bench/multiturn.py --modes concurrent --streams 1 --reps 3 --long-tokens 512
  conc4-r{1,2,3}.json      same with --streams 4
  ab-r{1,2,3}.json         results/W5/ab.py OUT 24500,98000 '{"prod":{}}'
  quality.json             bench/quality.py (MMLU-200 + refusal)
  exact.json               bench/glmbench.py --suites exact
  batchexact.json          bench/multiturn.py --modes batchexact
Stdlib only.
"""
import json, statistics, sys
from pathlib import Path

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
ROUNDS = (1, 2, 3)
AB_SHA = "8794a3463259cc2f"
SKIP = {"metadata.json", "models.json", "preflight.json"}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def rounds(pattern):
    """[(round, data)] for every round file that exists and parses."""
    return [(r, d) for r in ROUNDS if (d := load(D / pattern.format(r=r))) is not None]


def receipt(r):
    rd = D / f"rigmark-r{r}"
    if not rd.is_dir():
        return None
    for p in sorted(rd.glob("*.json")):
        if p.name not in SKIP and (d := load(p)) and "decode" in d and "prefill" in d:
            return d
    return None


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def stats(vals):
    v = [x for x in vals if x is not None]
    return (statistics.mean(v), min(v), max(v), len(v)) if v else (None, None, None, 0)


def f(x, nd=2):
    if x is None:
        return "-"
    return f"{x:,.{nd}f}" if abs(x) < 10 else f"{x:,.1f}"


def pct(new, ref):
    return f"{(new / ref - 1) * 100:+.1f}%" if new is not None and ref else "-"


def ratio(new, ref, lower=False):
    if new is None or not ref:
        return "-"
    return f"{(ref / new if lower else new / ref):.2f}x"


def safe(fn, d):
    try:
        return fn(d)
    except (KeyError, TypeError, IndexError, ZeroDivisionError):
        return None


out, M = [], {}          # M: metric key -> (mean, min, max, n)


def table(title, metrics, per_round, note=""):
    """metrics: [(key, label, fn(data))]; per_round: [(round, data)]."""
    out.append(f"### {title}\n")
    if note:
        out.append(note + "\n")
    out.append(f"rounds found: {len(per_round)} ({', '.join(f'r{r}' for r, _ in per_round) or 'none'})\n")
    out.append("| metric | mean | min | max | n | " + " | ".join(f"r{r}" for r, _ in per_round) + " |")
    out.append("|---|---:|---:|---:|---:|" + "---:|" * len(per_round))
    for key, label, fn in metrics:
        vals = [safe(fn, d) for _, d in per_round]
        M[key] = s = stats(vals)
        out.append(f"| {label} | {f(s[0])} | {f(s[1])} | {f(s[2])} | {s[3]} | " + " | ".join(f(v) for v in vals) + " |")
    out.append("")


# ---------------------------------------------------------------- RigMark
rig = [(r, d) for r in ROUNDS if (d := receipt(r)) is not None]
rm = []
for w in ("code", "prose", "structured"):
    rm.append((f"{w}_dec", f"{w} decode tok/s", lambda d, w=w: med(x["decode_tokens_per_second"] for x in d["decode"][w]["runs"])))
for w in ("code", "prose", "structured"):
    rm.append((f"{w}_ttft", f"{w} TTFT s", lambda d, w=w: med(x["ttft_seconds"] for x in d["decode"][w]["runs"])))
for dep in ("8192", "32768", "65536"):
    rm.append((f"cold{dep}", f"{int(dep) // 1024}K cold prefill tok/s",
               lambda d, dep=dep: d["prefill"][dep]["cold"]["effective_prefill_tokens_per_second"]["median"]))
for dep in ("8192", "32768", "65536"):
    rm.append((f"rep{dep}", f"{int(dep) // 1024}K immediate replay tok/s",
               lambda d, dep=dep: d["prefill"][dep]["warm_replay"]["effective_prefill_tokens_per_second"]["median"]))
for dep in ("8192", "32768", "65536"):
    rm.append((f"rept{dep}", f"{int(dep) // 1024}K replay TTFT s",
               lambda d, dep=dep: d["prefill"][dep]["warm_replay"]["ttft_seconds"]["median"]))
for c in ("1", "2", "4"):
    rm.append((f"c{c}agg", f"C{c} aggregate tok/s", lambda d, c=c: d["concurrency"][c]["aggregate_end_to_end_tokens_per_second"]["median"]))
for c in ("1", "2", "4"):
    rm.append((f"c{c}ttft", f"C{c} per-stream TTFT s", lambda d, c=c: d["concurrency"][c]["per_stream_ttft_seconds"]["median"]))
rm.append(("gates", "basic output gates passed (of 15)",
           lambda d: sum(d["decode"][w]["completion_gate"]["passed"] for w in ("code", "prose", "structured"))))
table("RigMark (per-round value = RigMark median)", rm, rig)
ids = [f"r{r}: {safe(lambda d: d['run']['comparison_id'], d)} / "
       f"rev {(safe(lambda d: d['protocol']['repository_revision'], d) or '-')[:12]}" for r, d in rig]
if ids:
    out.append("Receipts: " + "; ".join(ids) + "\n")

# ---------------------------------------------------------------- glmbench
gb = rounds("glmbench-r{r}.json")
cells = {}                                # (suite, prompt, temp) -> {round: cell}
for r, d in gb:
    for suite, cl in (d.get("suites") or {}).items():
        for c in cl:
            cells.setdefault((suite, c["prompt"], c["temperature"]), {})[r] = c
out.append("### glmbench (per-round value = median decode tok/s of the reps)\n")
out.append(f"rounds found: {len(gb)} ({', '.join(f'r{r}' for r, _ in gb) or 'none'}). "
           "hashes: `same` = every rep of every round returned the same reply; `rounds agree` = each round's "
           "per-rep hash list is identical (sampled cells are seeded).\n")
if cells:
    out.append("| cell | mean | min | max | n | " + " | ".join(f"r{r}" for r, _ in gb) + " | reply hashes |")
    out.append("|---|---:|---:|---:|---:|" + "---:|" * len(gb) + "---|")
for key, byr in cells.items():
    vals = [byr[r]["median_tps"] if r in byr else None for r, _ in gb]
    s = stats(vals)
    M["gb:" + "/".join(map(str, key))] = s
    lists = [tuple(x.get("sha256") for x in byr[r].get("runs", [])) for r in byr]
    flat = {h for L in lists for h in L}
    h = ("same " + next(iter(flat)) if len(flat) == 1 else
         "rounds agree" if len(set(lists)) == 1 else f"**differ** ({len(flat)} distinct)")
    name = f"{key[0]} {key[1]} T={key[2]:g}"
    if (key[0], key[1], key[2]) in (("tf", "chat", 0.0), ("tf", "code", 0.0), ("kit", "structured", 0.0)):
        name = f"**{name}**"
    out.append(f"| {name} | {f(s[0])} | {f(s[1])} | {f(s[2])} | {s[3]} | " + " | ".join(f(v) for v in vals) + f" | {h} |")
out.append("")

# ---------------------------------------------------------------- multiturn concurrent
def agg(d):
    xs = [x["aggregate_tps"] for x in d.get("concurrent", [])]
    return statistics.mean(xs) if xs else None


def ttft(d):
    return med(t for x in d.get("concurrent", []) for t in x.get("ttft_s", []))


for n in (1, 4):
    rr = rounds(f"conc{n}-r{{r}}.json")
    table(f"multiturn concurrent, {n} stream{'s' * (n > 1)} (per-round value = mean of reps)",
          [(f"conc{n}", "aggregate tok/s", agg), (f"conc{n}ttft", "per-stream TTFT s (median)", ttft)], rr)

# ---------------------------------------------------------------- ab prefill
ab = rounds("ab-r{r}.json")


def abrow(ctx, k="cold"):
    return lambda d: next(x[k]["prefill_tps"] for x in d if x["ctx"] == ctx)


table("ab.py prefill (cold, unique prompt)", [
    ("ab24", "prefill tok/s @ 24.5k", abrow(24500)), ("ab98", "prefill tok/s @ 98k", abrow(98000)),
    ("ab24d", "cold decode tok/s @ 24.5k", lambda d: next(x["cold"]["decode_tps"] for x in d if x["ctx"] == 24500)),
    ("ab98d", "cold decode tok/s @ 98k", lambda d: next(x["cold"]["decode_tps"] for x in d if x["ctx"] == 98000))], ab)
shas = [(r, x["ctx"], k, x[k].get("sha")) for r, d in ab for x in d for k in ("cold", "warm")]
bad = [s for s in shas if s[3] != AB_SHA]
out.append(f"Reply sha: {len(shas) - len(bad)}/{len(shas)} replies == `{AB_SHA}`"
           + ("" if not bad else " -- **mismatch:** " + ", ".join(f"r{r} {c} {k} {h}" for r, c, k, h in bad)) + "\n")

# ---------------------------------------------------------------- single-run gates
out.append("### Single-run correctness\n\n| check | result |\n|---|---|")
q = load(D / "quality.json")
out.append("| quality.json MMLU-200 | " + (f"{q['mmlu']['correct']}/{q['mmlu']['n']} ({q['mmlu']['accuracy'] * 100:.1f}%)"
           if safe(lambda d: d["mmlu"]["n"], q) else "missing") + " |")
out.append("| quality.json refusals | " + (f"{q['refusal']['refused']}/{q['refusal']['n']} refused"
           if safe(lambda d: d["refusal"]["n"], q) else "missing") + " |")
e = safe(lambda d: d["suites"]["exact"], load(D / "exact.json"))
out.append("| exact.json drafted == serial | " + (f"{sum(bool(c.get('identical')) for c in e)}/{len(e)} identical" +
           ("" if all(c.get("identical") for c in e) else " -- **differ:** " + ", ".join(
               f"{c['prompt']} T={c['temperature']:g}" for c in e if not c.get("identical")))
           if e else "missing") + " |")
b = safe(lambda d: d["batchexact"]["same"], load(D / "batchexact.json"))
out.append("| batchexact.json batched == alone | " + (f"{sum(map(bool, b))}/{len(b)} same" if b else "missing") + " |\n")

# ---------------------------------------------------------------- comparisons
ALEX = [("code_dec", "code decode tok/s", 44.0, 0), ("prose_dec", "prose decode tok/s", 18.9, 0),
        ("structured_dec", "structured decode tok/s", 64.9, 0), ("code_ttft", "code TTFT s", 0.60, 1),
        ("prose_ttft", "prose TTFT s", 0.49, 1), ("structured_ttft", "structured TTFT s", 0.47, 1),
        ("cold8192", "8K cold prefill tok/s", 1813, 0), ("cold32768", "32K cold prefill tok/s", 1908, 0),
        ("cold65536", "64K cold prefill tok/s", 1922, 0), ("rep8192", "8K immediate replay tok/s", 1812, 0),
        ("rep32768", "32K immediate replay tok/s", 11046, 0), ("rep65536", "64K immediate replay tok/s", 11364, 0),
        ("rept8192", "8K replay TTFT s", 4.52, 1), ("rept32768", "32K replay TTFT s", 2.97, 1),
        ("rept65536", "64K replay TTFT s", 5.77, 1), ("c1agg", "C1 aggregate tok/s", 31.6, 0),
        ("c2agg", "C2 aggregate tok/s", 42.0, 0), ("c4agg", "C4 aggregate tok/s", 66.1, 0),
        ("c1ttft", "C1 per-stream TTFT s", 0.60, 1), ("c2ttft", "C2 per-stream TTFT s", 0.68, 1),
        ("c4ttft", "C4 per-stream TTFT s", 0.81, 1)]
cmp = ["## vs Alex Ellis's vLLM TP2 k=7 RigMark receipt\n",
       "Reference: docs/RESULTS.md W15 §5 table, column \"vLLM TP2 k=7 (Alex)\". Ratio = how many times better "
       "TensorFold is (tok/s: ours / theirs; seconds: theirs / ours), for mean, worst and best round.\n",
       "| metric | TensorFold mean (min-max, n) | vLLM TP2 k=7 | ratio mean | ratio worst | ratio best |",
       "|---|---:|---:|---:|---:|---:|"]
for key, label, ref, lower in ALEX:
    m, lo, hi, n = M.get(key, (None,) * 3 + (0,))
    worst, best = (hi, lo) if lower else (lo, hi)
    cmp.append(f"| {label}{' (lower is better)' if lower else ''} | {f(m)} ({f(lo)}-{f(hi)}, {n}) | {f(ref)} | "
               f"{ratio(m, ref, lower)} | {ratio(worst, ref, lower)} | {ratio(best, ref, lower)} |")
MORNING = [("gb:tf/chat/0.0", "chat (glmbench tf chat T=0)", 44.6), ("gb:tf/code/0.0", "code (glmbench tf code T=0)", 77.6),
           ("gb:kit/structured/0.0", "structured (glmbench kit structured T=0)", 100.6),
           ("conc4", "4 users aggregate (conc4)", 78.0), ("ab24", "prefill @ 24.5k (ab)", 1607),
           ("ab98", "prefill @ 98k (ab)", 1607)]
cmp += ["\n## vs this morning's image\n",
        "Reference numbers: chat 44.6, code 77.6, structured 100.6, 4 users ~78, prefill 1,607 tok/s.\n",
        "| metric | this morning | now mean | now min | now max | n | delta mean | delta min | delta max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for key, label, ref in MORNING:
    m, lo, hi, n = M.get(key, (None,) * 3 + (0,))
    cmp.append(f"| {label} | {f(ref)} | {f(m)} | {f(lo)} | {f(hi)} | {n} | {pct(m, ref)} | {pct(lo, ref)} | {pct(hi, ref)} |")

md = ["# FINAL-20260930: final production config, 3-round summary\n",
      f"Generated by `summarize.py` from `{D}`. Per metric: mean, min and max across rounds; n = rounds found.\n",
      *cmp, "\n## Averaged numbers\n", *out, "## Config and notes\n", "TODO\n"]
(D / "summary.md").write_text("\n".join(md))
print(f"wrote {D / 'summary.md'}")
