#!/usr/bin/env python3
"""W18: memory summary of one load: memsum.py NAME (in results/W18).
MemAvailable minimum per phase (mem-NAME.log, 2 s, both nodes) between the marks (mem-marks-NAME.txt), the minimum over
stress + MMLU + needle (stress-start .. needle-end, the W17 goal: >= 8 GiB), and MemFree steps up of >= 1 GiB within
<= 1 s (memfast-NAME-r0/r1.log, 0.5 s: an allocator empty_cache: a 0550 trim, a scratch growth or a graph capture)
per phase."""
import datetime as dt, os, sys

N = sys.argv[1]
R = os.path.dirname(os.path.abspath(__file__))
marks = []
for line in open(f"{R}/mem-marks-{N}.txt"):
    k, t = line.split()
    marks.append((k, t))
day = dt.date.today()


def sec(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def phase(t):  # t = seconds of day
    cur = "load"
    for k, m in marks:
        if t >= sec(m):
            cur = k.replace("-start", "")
    return cur


rows = []
for line in open(f"{R}/mem-{N}.log"):
    p = line.split()
    if len(p) != 3:
        continue
    try:
        rows.append((sec(p[0]), float(p[1]), float(p[2])))
    except ValueError:
        pass
ph = {}
for t, a, b in rows:
    k = phase(t)
    m = ph.setdefault(k, [99.0, 99.0])
    m[0], m[1] = min(m[0], a), min(m[1], b)
print(f"{N}: MemAvailable min by phase (head / worker GiB)")
order = [k.replace("-start", "") for k, _ in marks]
for k in ["load"] + order:
    if k in ph:
        print(f"  {k:8s} {ph[k][0]:6.2f} / {ph[k][1]:6.2f}")
md = dict(marks)
if "stress-start" in md and "needle-end" in md:
    s0, s1 = sec(md["stress-start"]), sec(md["needle-end"])
    sel = [(a, b) for t, a, b in rows if s0 <= t <= s1]
    if sel:
        a, b = min(x[0] for x in sel), min(x[1] for x in sel)
        print(f"  stress+mmlu+needle min {a:.2f} / {b:.2f}  -> {'PASS' if min(a, b) >= 8 else 'FAIL'} (>= 8 GiB)")
    for lo, hi, name in (("stress-start", "mmlu-start", "stress"), ("needle-start", "needle-end", "needle")):
        sel = [(a, b) for t, a, b in rows if sec(md[lo]) <= t <= sec(md[hi])]
        if sel:
            print(f"  {name} min {min(x[0] for x in sel):.2f} / {min(x[1] for x in sel):.2f}")
allmin = (min(r[1] for r in rows), min(r[2] for r in rows)) if rows else (0, 0)
print(f"  whole load min {allmin[0]:.2f} / {allmin[1]:.2f}")
for node in ("r0", "r1"):
    f = f"{R}/memfast-{N}-{node}.log"
    if not os.path.exists(f):
        continue
    pts = []
    for line in open(f):
        p = line.split()
        try:
            pts.append((float(p[0]), float(p[1])))
        except (ValueError, IndexError):
            pass
    steps = {}
    lst = []
    for i in range(1, len(pts)):
        # a step: MemFree up >= 1 GiB against the lowest of the last 1 s
        t, f1 = pts[i]
        lo = min(v for tt, v in pts[max(0, i - 3):i] if t - tt <= 1.1)
        if f1 - lo >= 1.0 and (not lst or t - lst[-1][0] > 2):
            loc = dt.datetime.fromtimestamp(t)
            k = phase(loc.hour * 3600 + loc.minute * 60 + loc.second)
            steps[k] = steps.get(k, 0) + 1
            lst.append((t, f1 - lo, k, loc.strftime("%H:%M:%S.%f")[:-4]))
    print(f"  MemFree steps >= 1 GiB ({node}): " + ", ".join(f"{k} {v}" for k, v in steps.items()))
    with open(f"{R}/memsteps-{N}-{node}.txt", "w") as o:
        for t, d, k, s in lst:
            o.write(f"{s} +{d:.2f} GiB {k}\n")
