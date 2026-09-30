#!/usr/bin/env python3
"""W19: per-stage kernel time of one rank's capture (cap.sh; nsys sqlite export), for the control vs combined
attribution.

    w19att.py SQLITE CAP.jsonl OUT.json [--gap 1.5]

Segments: runs of GPU activity separated by > GAP s of no kernel (cap.sh sends its four requests 4 s apart), named in
order prefill (a cold ~21.5k prompt, 16 tokens), prose (chat 256), code (LRU 384), conc4 (4 x prose 384). Per segment:
wall (first kernel start .. last kernel end), GPU-busy union, idle = wall - busy, and per kernel family the summed
kernel time and the EXCLUSIVE time (every instant split evenly among the kernels running then, W7's method, so the
families + idle add up to the wall). Normalised: prefill per prompt token (us), decode per generated token (ms) using
the capture's own request stats (CAP.jsonl: w19req.py / cap.sh lines). Also per family the kernel names seen.
"""
import collections
import json
import re
import sqlite3
import sys

db, capj, out = sys.argv[1:4]
GAP = float(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 1.5

FAM = [
    ("L2 prefetch (0460, side stream)", r"^segments_kernel$"),
    ("routed experts", r"(?i)^(grouped_kernel|grouped_loop_kernel|grouped_epi_kernel|expert_kernel|ld_kernel.*|.*gateup_epilogue.*|.*down_epilogue.*|rot_in1?_kernel|plan_kernel|.*fat2?.*kernel.*|gateup.*|down_.*|.*_fat.*|.*once.*|.*exl3.*)$"),
    ("dense q4 / GEMM", r"(?i)^(_qmm|_fq4.*|_reduce|_group_sums|_swiglu|_fb16|chain_mm.*|q4s?_.*|.*q4.*|.*stream_kernel.*|.*gemm.*|.*gemv.*|nvjet.*|.*cutlass.*|.*xmma.*|sm\d+_.*|ampere_.*|.*_mm_.*|_fused_q.*)$"),
    ("router / grouping / combine", r"^(_router.*|_topk|_group.*|_combine.*|DeviceRadixSort.*|searchsorted.*|fill_reverse_indices_kernel|DeviceScanKernel|_select.*|_moe_.*|.*route.*)$"),
    ("KDA", r"(?i)^(chain_kernel|_kda.*|_conv.*|_dconv.*|replay_layers_kernel|_dattn_kernel|.*kda.*|.*chunk.*)$"),
    ("attention + indexer", r"(?i)^(_lchunks|_lmerge|_lsparse.*|_expand.*|_absorb.*|_lwrite.*|_index.*|_pool_keys|_scores.*|gatherTopK|compute(BlockDigitCounts|DigitCumSum|BlockwiseWithinKCounts)|DeviceScanByKeyKernel|_gather_sel|_sparse.*|_latent.*|_mla.*|_rope.*|_q_.*|_kv_.*|.*b12x.*|.*attn.*|.*flash.*)$"),
    ("hc", r"(?i)^(_hc.*|_stream_mean|.*sinkhorn.*|.*hc_.*)$"),
    ("RoCE all-gather", r"^gather_kernel$"),
    ("NCCL", r"^nccl"),
    ("sampling / argmax / softmax", r"(?i).*(argmax|softmax|topk|top_k|sort|sample|gumbel|multinomial|cumsum|max_kernel).*"),
    ("norms / elementwise / other", r".*"),
]
FAMC = [(n, re.compile(p)) for n, p in FAM]
cache = {}


def fam(nm):
    f = cache.get(nm)
    if f is None:
        f = cache[nm] = next(n for n, p in FAMC if p.match(nm))
    return f


c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
K = c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
K = [(s, e, S.get(n, str(n))) for s, e, n in K]
segs = []
# gaps are looked for among the non-NCCL kernels: rank 1 waits for rank 0's next plan inside an NCCL broadcast that
# spins on the GPU through the idle time between requests; the segment bounds are its first / last non-NCCL kernel
K = [k for k in K if fam(k[2]) != "NCCL" or True]
KN = [k for k in K if fam(k[2]) != "NCCL"]
for k in KN:
    if segs and k[0] - segs[-1][1] < GAP * 1e9:
        segs[-1][2].append(k)
        segs[-1][1] = max(segs[-1][1], k[1])
    else:
        segs.append([k[0], k[1], [k]])
# drop tiny segments (stray single kernels, < 20 ms of kernels) that are not requests
segs = [s for s in segs if sum(e - b for b, e, _ in s[2]) > 20e6]
# every kernel (NCCL included) that overlaps a segment's bounds belongs to it (clipped in excl)
import bisect
ks0 = [k[0] for k in K]
for sg in segs:
    i = max(0, bisect.bisect_left(ks0, sg[0]) - 2000)
    sg[2] = [k for k in K[i:bisect.bisect_right(ks0, sg[1])] if k[1] > sg[0]]
names = ["prefill", "prose", "code", "conc4"]
reqs = [json.loads(l) for l in open(capj)] if capj != "-" else []
pf = [r for r in reqs if r.get("kind") == "prefill"]
dec = [r for r in reqs if r.get("kind") != "prefill"]


def excl(ks, t0, t1):
    """W7's partition: every instant split evenly among the kernels running then; the rest is idle."""
    ev = []
    for i, (s, e, n) in enumerate(ks):
        ev.append((max(s, t0), 1, i))
        ev.append((min(e, t1), -1, i))
    ev.sort()
    run, last, out, idle = set(), t0, collections.Counter(), 0
    for t, d, i in ev:
        if t > last:
            if run:
                share = (t - last) / len(run)
                for j in run:
                    out[fam(ks[j][2])] += share
            else:
                idle += t - last
        last = max(last, t)
        (run.add if d == 1 else run.discard)(i)
    return out, idle


res = {"db": db, "segments": []}
for idx, (t0, t1, ks) in enumerate(segs):
    nm = names[idx] if idx < len(names) else f"seg{idx}"
    wall = (t1 - t0) / 1e6
    tot = collections.Counter()
    kn = collections.defaultdict(collections.Counter)
    for s, e, n in ks:
        d = (min(e, t1) - max(s, t0)) / 1e6
        tot[fam(n)] += d
        kn[fam(n)][n] += d
    ex, idle = excl(ks, t0, t1)
    row = {"name": nm, "wall_ms": round(wall, 1), "kernels": len(ks), "idle_ms": round(idle / 1e6, 1),
           "sum_ms": {k: round(v, 2) for k, v in tot.most_common()},
           "excl_ms": {k: round(v / 1e6, 2) for k, v in ex.most_common()},
           "top_names": {k: [(n, round(v, 1)) for n, v in kn[k].most_common(6)] for k in kn}}
    if nm == "prefill" and pf:
        row["prompt_tokens"] = pf[-1]["prompt"]
    elif nm in ("prose", "code", "conc4"):          # cap.sh order: chat 256, code 384, then 4 concurrent
        r = dec[0:1] if nm == "prose" else dec[1:2] if nm == "code" else dec[2:6]
        row["tokens"] = sum(d.get("tokens", 0) for d in r)
        row["rounds"] = sum((d.get("tf") or {}).get("rounds", 0) for d in r) if nm != "conc4" else None
        row["decode_s"] = max((d.get("decode_s") or 0) for d in r) if r else None
    res["segments"].append(row)
json.dump(res, open(out, "w"), indent=1, default=str)
for row in res["segments"]:
    print(f"{row['name']:8s} wall {row['wall_ms']:9.1f} ms  kernels {row['kernels']:7d}  idle {row['idle_ms']:7.1f} ms | "
          + ", ".join(f"{k} {v:.1f}" for k, v in row["excl_ms"].items()))
