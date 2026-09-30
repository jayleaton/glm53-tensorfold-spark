"""patches/0580: a lane-level CPU port of ``exl3_ld.cu``'s ``ld_kernel`` (exl3.cu's grouped kernel with a deeper load
path), for ``tests/test_decode_loads_emulator.py``. The reference it must equal is ``decode_kernels_emu.exl3_reference``
(exl3.cu's grouped_kernel at MATRIX level: member tile x K split x column block; warp w's k tiles in order from +0.0;
warps added 0..3; Z rows of live members only).

Ported line for line from the CUDA, per program (u, n block, z) and per warp, lanes as numpy vectors:

- the prologue's exits (u >= ucount, uids[u] >= E, an empty member tile) and the member rows;
- the PD-deep ring: LD "nc" = the NT / 4 16-byte loads a lane (lane l: words 128 v + 4 l .. + 3 of the step's
  NT x 32-word image) held in "registers", stored to the warp's staging area (an alias of the warp's slice of red,
  NaN-initialised shared memory) and read back as word i * 32 + lane; "w32" = word i * 32 + lane loaded straight into
  the ring; "cpa" = the same 16-byte chunks as cp.async copies into ring slot s % (PD + 1) (``AsyncCopies``: a copy
  lands at a random time after its issue and no later than the ``cp.async.wait_group`` that covers it), commit groups
  as the kernel commits them (PD in the prologue, one a step, empty past the end);
- the A fragments (load_pair's values) PD steps ahead; the refill order of the kernel (take slot d, refill it with step
  s + PD, then compute);
- decode_tile and the mma through the ORDER-SENSITIVE model of decode_kernels_emu (``mma_frag``);
- the warps' partial sums parked in red and added in warp order by the reduction loop, Z stored for live rows only.

``mutate`` (negative controls; each must change the bits): "stage" reads the staging area lane-major, "early" refills a
ring slot before its words are taken, "ring" puts step s + PD in slot s % PD (the slot being read), "wait" waits for
one cp.async group too few, "aoff" loads the A fragments of the wrong k step, "warps" adds the warps in reverse,
"kfirst" starts each warp one k tile late (wrapping inside its range).
"""

from __future__ import annotations

import numpy as np

from decode_kernels_emu import LANE, G, T, AsyncCopies, decode_tile, mma_frag

LDS = ("nc", "w32", "cpa")


def _pairs(X: np.ndarray, rows: np.ndarray, col: np.ndarray) -> np.ndarray:
    """load_pair for 32 lanes: X[row][col], X[row][col + 1] (fp16 bits) as one 32-bit word; 0 for a dead row."""

    out = np.zeros(32, np.uint32)
    ok = rows >= 0
    r = np.where(ok, rows, 0)
    lo = X[r, col].astype(np.uint32)
    hi = X[r, col + 1].astype(np.uint32)
    out[ok] = (lo | (hi << 16))[ok]
    return out


def exl3_ld_port(X0, X1, T0, T1, uids, ucount, members, K, N, P, SK, slots, E, mats, NT, PD, LD, rng,
                 W: int = 4, mutate: str = "") -> np.ndarray:
    """Z [mats, SK, P, N] fp32 as ld_kernel<NT, PD, LD, 0> leaves it (never-stored elements: NaN). Programs run in a
    random order (each is independent: disjoint Z rows / columns / splits)."""

    assert LD in LDS and NT % 4 == 0 and 1 <= PD <= 4
    KT, NTILES = K >> 4, N >> 4
    maxu, maxm = members.shape
    MT = (maxm + 15) // 16
    per_split = KT // SK
    PW = per_split // W
    assert K % (16 * SK * W) == 0 and N % (16 * NT) == 0 and PW % PD == 0 and PW >= PD
    NV = NT // 4
    S = PD + 1                                              # cpa ring slots
    Z = np.full((mats, SK, P, N), np.nan, np.float32)
    progs = [(u, nb, z) for u in range(maxu) for nb in range(N // (16 * NT)) for z in range(mats * SK * MT)]
    rng.shuffle(progs)
    for u, nb, z in progs:
        mtile = z % MT
        split = (z // MT) % SK
        mat = z // MT // SK
        cnt, e = int(ucount), int(uids[u])
        first = int(members[u, mtile * 16])
        if u >= cnt or e >= E or first < 0:
            continue
        rows_sh = np.full(16, -1, np.int64)
        for i in range(16):
            m = mtile * 16 + i
            code = int(members[u, m]) if m < maxm else -1
            rows_sh[i] = (code >> 5) * slots + (code & 31) if code >= 0 else -1
        X, Tw = (X1, T1) if mat else (X0, T0)
        red = np.full((W, 16, NT * 16), np.nan, np.float32)             # shared memory (garbage at start)
        r0, r1 = rows_sh[G], rows_sh[G + 8]
        warp_acc = []
        for w in range(W):
            kt0 = split * per_split + w * PW
            stage = red[w].reshape(-1).view(np.uint32)                 # the warp's staging area / ring aliases red[w]
            cp = AsyncCopies(stage, rng)
            ring_v = [None] * PD
            ring_w = [None] * PD
            ring_a = [None] * PD

            def kt_of(s):
                return kt0 + ((s + 1) % PW if mutate == "kfirst" else s)

            def image(s):
                return Tw[e, kt_of(s), nb * NT:nb * NT + NT].reshape(-1)          # NT x 32 words, memory order

            def issue_w(d, s):
                img = image(s)
                if LD == "nc":
                    ring_v[d] = np.stack([np.stack([img[128 * v + 4 * LANE + j] for j in range(4)], 1)
                                          for v in range(NV)])                      # [NV, lane, 4]
                elif LD == "w32":
                    ring_w[d] = np.stack([img[i * 32 + LANE] for i in range(NT)])  # [NT, lane]
                else:
                    slot = s % PD if mutate == "ring" else s % S
                    for v in range(NV):
                        for ln in range(32):
                            off = 128 * v + 4 * ln
                            cp.copy(slot * NT * 32 + off, img[off:off + 4])

            def issue_a(d, s):
                k = kt_of(s - 1 if mutate == "aoff" and s > 0 else s) * 16
                col = k + 2 * T
                ring_a[d] = np.stack([_pairs(X, r0, col), _pairs(X, r1, col), _pairs(X, r0, col + 8),
                                      _pairs(X, r1, col + 8)], 1)

            for d in range(PD):
                issue_w(d, d)
                if LD == "cpa":
                    cp.commit()
                    cp.jitter()
            for d in range(PD):
                issue_a(d, d)

            acc = np.zeros((NT, 2, 32, 4), np.float32)
            for s in range(PW):
                d = s % PD
                more = s + PD < PW
                a = ring_a[d]
                if LD == "nc":
                    if mutate == "early" and more:
                        issue_w(d, s + PD)
                    for v in range(NV):
                        for j in range(4):
                            stage[128 * v + 4 * LANE + j] = ring_v[d][v][:, j]
                    if more and mutate != "early":
                        issue_w(d, s + PD)
                    if mutate == "stage":
                        words = np.stack([stage[LANE * NT + i] for i in range(NT)])
                    else:
                        words = np.stack([stage[i * 32 + LANE] for i in range(NT)])
                elif LD == "w32":
                    if mutate == "early" and more:
                        issue_w(d, s + PD)
                    words = ring_w[d].copy()
                    if more and mutate != "early":
                        issue_w(d, s + PD)
                else:
                    cp.wait(PD if mutate == "wait" else PD - 1)
                    if more:
                        issue_w(d, s + PD)
                    cp.commit()
                    cp.jitter()
                    base = (s % S) * NT * 32
                    words = np.stack([stage[base + i * 32 + LANE] for i in range(NT)])
                if more:
                    issue_a(d, s + PD)
                for i in range(NT):
                    (b00, b01), (b10, b11) = decode_tile(words[i])
                    acc[i, 0] = mma_frag(acc[i, 0], a, b00, b01, "f16")
                    acc[i, 1] = mma_frag(acc[i, 1], a, b10, b11, "f16")
            if LD == "cpa":
                cp.wait(0)
            warp_acc.append(acc)
        # member_tiles<NT, W, 1>: park, then add in warp order
        for w in range(W):
            acc = warp_acc[w]
            for i in range(NT):
                for h in range(2):
                    col = i * 16 + h * 8 + 2 * T
                    red[w, G, col] = acc[i, h][:, 0]
                    red[w, G, col + 1] = acc[i, h][:, 1]
                    red[w, G + 8, col] = acc[i, h][:, 2]
                    red[w, G + 8, col + 1] = acc[i, h][:, 3]
        order = list(range(W))[::-1] if mutate == "warps" else list(range(W))
        for row in range(16):
            r = int(rows_sh[row])
            if r < 0:
                continue
            s_ = red[order[0], row].copy()
            for w in order[1:]:
                s_ = (s_ + red[w, row]).astype(np.float32)
            Z[mat, split, r, nb * NT * 16:nb * NT * 16 + NT * 16] = s_
    return Z


def ring_bytes_in_flight(NT: int, PD: int, ctas_per_sm: int, W: int = 4) -> int:
    """Nominal trellis bytes in flight an SM while every warp computes a step: PD steps of NT x 128 B a warp."""

    return ctas_per_sm * W * PD * NT * 128
