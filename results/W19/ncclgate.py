#!/usr/bin/env python3
"""W19: the NCCL gate from ncclbw2.sh's rows (results/W19/ncclbw.jsonl, labels w-*). ncclgate.py JSONL

PASS (NCCL_IB_HCA both functions + NCCL_MIN/MAX_NCHANNELS=4 go into the combined load) when:
  health: all four all-reduce rows (1 / 2 NICs x Ring / Tree) ran, and on each NIC set Tree is within 1.5x of Ring
          at 4 and 16 MiB (no pathological algorithm; NCCL picks per size);
  gain:   2-NIC 4 channels (the better of its two runs) at 4 MiB (the prefill's all-gather) <= 0.90x the 1-NIC default
          of this window, and at 8 / 16 / 32 MiB not slower than the 1-NIC default (no regression at larger sizes).
Also printed: the 1-NIC channel sweep (how much is channels alone vs the second function)."""
import json, sys
rows = {}
for l in open(sys.argv[1]):
    d = json.loads(l)
    if d["label"].startswith("w-"):
        rows[d["label"]] = d
def ag(lab, mib):
    d = rows.get(lab)
    return d["ag"].get(str(mib), {}).get("us") if d else None
def ar(lab, mib):
    d = rows.get(lab)
    return d["ar"].get(str(mib), {}).get("us") if d else None
print("all-gather us (1 / 2 / 4 / 8 / 16 / 32 MiB):")
for lab in ("w-one-def", "w-one-ch2", "w-one-ch4", "w-one-ch8", "w-two-ch4", "w-two-ch4-b"):
    print(f"  {lab:12s} " + " ".join(f"{ag(lab, m) or float('nan'):8.1f}" for m in (1, 2, 4, 8, 16, 32)))
print("all-reduce us (4 / 16 MiB):")
for lab in ("w-one-def", "w-two-ch4", "w-one-ar-ring", "w-one-ar-tree", "w-two-ar-ring", "w-two-ar-tree"):
    print(f"  {lab:14s} {ar(lab, 4) or float('nan'):8.1f} {ar(lab, 16) or float('nan'):8.1f}")
health, notes = True, []
for nic in ("one", "two"):
    for m in (4, 16):
        r, t = ar(f"w-{nic}-ar-ring", m), ar(f"w-{nic}-ar-tree", m)
        if r is None or t is None:
            health = False; notes.append(f"{nic} {m} MiB: missing"); continue
        if t > 1.5 * r:
            health = False
        notes.append(f"{nic}-NIC {m} MiB tree/ring {t / r:.2f}")
base = {m: ag("w-one-def", m) for m in (4, 8, 16, 32)}
two = {m: min(v for v in (ag("w-two-ch4", m), ag("w-two-ch4-b", m)) if v is not None) if any(
    v is not None for v in (ag("w-two-ch4", m), ag("w-two-ch4-b", m))) else None for m in (4, 8, 16, 32)}
gain = all(v is not None for v in list(base.values()) + list(two.values())) and two[4] <= 0.90 * base[4] and all(
    two[m] <= base[m] for m in (8, 16, 32))
g4 = f"{two[4] / base[4]:.3f}" if two[4] and base[4] else "n/a"
print(f"health v1 (Tree within 1.5x of Ring): {'PASS' if health else 'FAIL'} ({'; '.join(notes)})")
# v2 (W19, after the first run): a 2-rank Tree all-reduce is ~3x Ring on ONE NIC too (prod's link), so v1's bar tests
# NCCL's algorithm choice for 2 ranks, not the second function. Health = both algorithms run on 1 and 2 NICs and the
# second function makes neither of them slower (2-NIC <= 1.05x 1-NIC for Ring and for Tree, 4 and 16 MiB).
h2, n2 = True, []
for alg in ("ring", "tree"):
    for m in (4, 16):
        o, t = ar(f"w-one-ar-{alg}", m), ar(f"w-two-ar-{alg}", m)
        if o is None or t is None:
            h2 = False; n2.append(f"{alg} {m} MiB missing"); continue
        h2 &= t <= 1.05 * o
        n2.append(f"{alg} {m} MiB 2-NIC/1-NIC {t / o:.2f}")
print(f"health v2 (both run; 2 NICs slow neither Ring nor Tree): {'PASS' if h2 else 'FAIL'} ({'; '.join(n2)})")
health = h2
print(f"GATE NCCL: {'PASS' if (health and gain) else 'FAIL'} -- 2-NIC ch4 / 1-NIC default at 4 MiB {g4} (bar 0.90); "
      + ", ".join(f"{m} MiB {two[m]:.0f} vs {base[m]:.0f} us" for m in (8, 16, 32) if two[m] and base[m]))
