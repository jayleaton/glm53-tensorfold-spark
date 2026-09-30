#!/usr/bin/env python3
"""W18: MemAvailable at the start of the stress and the needle, the minimum inside each and the dip (start - min), per
node, from mem-NAME.log (2 s) and mem-marks-NAME.txt: dips.py NAME ... (the start level depends on the boot path:
a re-measured calibration leaves ~1 GiB more free than a cached one, so the dips compare loads better than the minima)."""
import os, sys
R = os.environ.get("SUMM_DIR") or os.path.dirname(os.path.abspath(__file__))
sec = lambda s: sum(int(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))
for n in sys.argv[1:]:
    marks = dict(l.split() for l in open(f"{R}/mem-marks-{n}.txt"))
    rows = [(sec(p[0]), float(p[1]), float(p[2])) for p in (l.split() for l in open(f"{R}/mem-{n}.log")) if len(p) == 3 and ":" in p[0]]
    out = [n]
    for a, b in (("stress-start", "mmlu-start"), ("needle-start", "needle-end")):
        s0, s1 = sec(marks[a]), sec(marks[b])
        sel = [r for r in rows if s0 <= r[0] <= s1]
        st = sel[0]
        m0, m1 = min(r[1] for r in sel), min(r[2] for r in sel)
        out.append(f"{a.split('-')[0]}: start {st[1]:.2f} / {st[2]:.2f}, min {m0:.2f} / {m1:.2f}, dip {st[1] - m0:.2f} / {st[2] - m1:.2f}")
    print(" | ".join(out))
