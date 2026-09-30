"""patches/0590: lane-level CPU ports of exl3_fast.cu's ``fat`` kernel and exl3_fat2.cu's ``fat2`` kernel, and the
matrix-level reference both must equal, for ``tests/test_fat2_emulator.py``.

The claim under test: fat2 gives fat's bits, element for element (Xd of gate/up, Y of down), for every routing, member
count, configuration and CTA interleaving. Three levels, on purpose:

- ``reference_gateup`` / ``reference_down`` (MATRIX level, written from exl3_fast.cu's documented arithmetic, not from
  either kernel's code): per (pair p of distinct expert e, output column c) the fp32 chain over k = 0 .. K-1 of the
  products W[k, c] * X[p, k], W the decoded trellis (``decode_tile`` as the m16n8k16 A fragment: A = the tile's (k, n)
  matrix transposed), then fwht_row, the scales, GLM's limited SwiGLU, fwht_row, fp16 (gate/up) or the scaled fp32 row
  (down). Pairs that no expert of the window holds are never written;
- ``fat_port``: fat line for line (fast2's plan kernel, the static stride or ticket item walk, the NSA-stage cp.async
  ring of member rows and trellis words with fat's XOR swizzle, ldmatrix.x4, decode_tile into the A fragment, mma,
  the ER-row epilogue rounds through shared memory, one warp a row), gate/up (8 warps, 64 members) and down (4 warps;
  64 members, or 128 at >= 4,096-row windows);
- ``fat2_port``: fat2 line for line: 256-thread CTAs, 128 (or 64) member items, the device plan, ticket claims NSA
  stages ahead, the 4-stage ring with 2 k tiles a stage running ACROSS items (the next item's first stages load under
  the current item's tail and epilogue), zero-fill to the 16-row group, the skipped member groups, the separate
  epilogue buffer, down's two column blocks an item (an odd last block: its warps idle).

The tensor-core mma is replaced by an ORDER-SENSITIVE model (``mma_batch``: the 16 exact products added to the
accumulator one k at a time with an fp32 rounding each step), through the canonical m16n8k16 fragment layouts. Hardware
adds differently, but identically for two kernels whose every mma gets the same operands in the same positions and the
same accumulator -- the property under test. With this model a whole chain over K is the sequential fp32 sum over k, so
the matrix reference is that sum: a k tile out of order, a k position moved inside a fragment, a wrong member row, a
split chain or a stale stage changes the bits. Shared memory starts as NaN garbage; cp.async copies land at random
times between their issue and the ``cp.async.wait_group`` that covers them (``decode_kernels_emu.AsyncCopies``), with
random landings between the warps of a stage; CTAs interleave at every ticket claim in a random order.

``mutate`` (negative controls, each must change the bits or leave an output unwritten): see ``MUTATIONS``.
"""

from __future__ import annotations

import numpy as np

from decode_kernels_emu import LANE, AsyncCopies, bf16r, f16_to_f32, f32_to_f16, mcg2

G = LANE >> 2
T = LANE & 3
HAD = np.float32(0.08838834764831845)
NB = 8
NAN16 = np.uint16(0x7E00)
M32 = np.uint64(0xFFFFFFFF)

MUTATIONS = {
    "slot": "a stage is loaded into the ring slot being computed (gs % NSA instead of (gs + NSA - 1) % NSA)",
    "wait": "cp.async.wait_group NSA - 1 (one group too few waited for)",
    "swz": "the row copies ignore the XOR swizzle (ldmatrix still applies it)",
    "kk": "the two k tiles of a stage multiplied in reverse order",
    "next_rows": "the next item's first stages gather the CURRENT item's member rows",
    "next_stage": "the next item's stages are loaded one k step late (stage ls - S + 1)",
    "claim_late": "the next item is claimed one stage later than its first load needs (its info is stale)",
    "acc": "the accumulators are not cleared between items",
    "skip": "the last live member group is skipped (16 np + 16 < cnt instead of 16 np < cnt)",
    "epi_row": "the epilogue writes accumulator c into ep rows 2t / 2t + 1 swapped",
    "down_pair": "down's second warp half multiplies column block 2 q instead of 2 q + 1",
    "ticket": "two CTAs can take the same ticket (a lost update)",
}
# Mutations that must NOT change the stored bits (the model's columns are independent; rows past the last member feed
# only columns that are never stored)
NEUTRAL = {
    "fill_cnt": "rows past the last member are not zero-filled at all (NaN garbage in the unused B columns)",
}


# -- helpers ------------------------------------------------------------------------------------------------------------
def f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def decode_tiles(w: np.ndarray):
    """exl3_fast.cu's decode_tile for the 32 lanes of any number of tiles (w [..., 32] uint32, lane on the last axis):
    b0[0], b0[1], b1[0], b1[1] per lane."""

    w64 = np.asarray(w, dtype=np.uint32).astype(np.uint64)
    p = w64[..., (LANE + 31) & 31]                                                     # __shfl_sync(w, lane - 1)
    s = ((w64 >> np.uint64(20)) | (p << np.uint64(12))) & M32                          # __funnelshift_r(w, p, 20)
    F = np.uint64(0xFFFF)
    b00 = mcg2((s >> np.uint64(8)) & F, (s >> np.uint64(4)) & F)
    b01 = mcg2(s & F, w64 >> np.uint64(16))
    b10 = mcg2((w64 >> np.uint64(12)) & F, (w64 >> np.uint64(8)) & F)
    b11 = mcg2((w64 >> np.uint64(4)) & F, w64 & F)
    return b00, b01, b10, b11


def a_fragment(words: np.ndarray) -> np.ndarray:
    """fast2 / fat / fat2: af[0] = b0[0], af[1] = b1[0], af[2] = b0[1], af[3] = b1[1] -> [..., 32, 4] uint32."""

    b00, b01, b10, b11 = decode_tiles(words)
    return np.stack([b00, b10, b01, b11], axis=-1).astype(np.uint32)


def _halves(u32: np.ndarray):
    u32 = np.asarray(u32, dtype=np.uint32)
    return f16_to_f32((u32 & 0xFFFF).astype(np.uint16)), f16_to_f32((u32 >> 16).astype(np.uint16))


# "order": every element adds its 16 products in k order. "position": element (i, j) of the 16 x 8 tile adds them in k
# order rotated by (8 i + j) % 16 -- a (hypothetical) tensor core whose summation order depends on where an element
# sits in the mma tile. fat2 keeps every element at fat's tile position (same column tile row, member % 8), so fat2 ==
# fat must hold under this model too (not == the matrix reference, which assumes k order).
MMA_MODEL = "order"
_I = np.arange(16)[:, None]
_J = np.arange(8)[None, :]
_ROT = (_I * 8 + _J) % 16


def mma_batch(acc: np.ndarray, a: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """M independent mma.m16n8k16.row.col.f32.f16.f16.f32 on lane registers: acc [M, 32, 4] fp32 (C = D), a [M, 32, 4],
    b0 / b1 [M, 32] uint32. A [16, 16]: a0 row g k 2t.., a1 row g + 8, a2 row g k 2t + 8.., a3 row g + 8 k 2t + 8..;
    B [16, 8] (k, n): b0 k 2t.. of column g, b1 k 2t + 8..; D: d0 (g, 2t), d1 (g, 2t + 1), d2 (g + 8, 2t), d3 (g + 8,
    2t + 1). The products (exact in fp32) are added to C one k at a time, in k order, rounding to fp32 each step."""

    M = acc.shape[0]
    A = np.full((M, 16, 16), np.nan, np.float32)
    for r, (dr, dk) in enumerate(((0, 0), (8, 0), (0, 8), (8, 8))):
        lo, hi = _halves(a[:, :, r])
        A[:, G + dr, dk + 2 * T] = lo
        A[:, G + dr, dk + 2 * T + 1] = hi
    B = np.full((M, 16, 8), np.nan, np.float32)
    for reg, dk in ((b0, 0), (b1, 8)):
        lo, hi = _halves(reg)
        B[:, dk + 2 * T, G] = lo
        B[:, dk + 2 * T + 1, G] = hi
    C = np.full((M, 16, 8), np.nan, np.float32)
    C[:, G, 2 * T], C[:, G, 2 * T + 1] = acc[:, :, 0], acc[:, :, 1]
    C[:, G + 8, 2 * T], C[:, G + 8, 2 * T + 1] = acc[:, :, 2], acc[:, :, 3]
    with np.errstate(invalid="ignore", over="ignore"):
        if MMA_MODEL == "position":
            for step in range(16):
                kk = (step + _ROT) % 16
                C = (C + (A[:, _I, kk] * B[:, kk, _J]).astype(np.float32)).astype(np.float32)
        else:
            for k in range(16):
                C = (C + (A[:, :, k:k + 1] * B[:, k:k + 1, :]).astype(np.float32)).astype(np.float32)
    return np.stack([C[:, G, 2 * T], C[:, G, 2 * T + 1], C[:, G + 8, 2 * T], C[:, G + 8, 2 * T + 1]], axis=2)


def ldsm4(mem: np.ndarray, addr: np.ndarray) -> np.ndarray:
    """ldmatrix.sync.aligned.m8n8.x4.shared.b16: lane i gives the start (in halves) of row i % 8 of matrix i // 8;
    register j of lane L = matrix j's row L // 4, halves 2 (L % 4), + 1 (low = first). -> [32, 4] uint32."""

    out = np.zeros((32, 4), np.uint32)
    for j in range(4):
        src = addr[8 * j + LANE // 4] + 2 * (LANE % 4)
        out[:, j] = mem[src].astype(np.uint32) | (mem[src + 1].astype(np.uint32) << 16)
    return out


def fwht_row(v: np.ndarray) -> np.ndarray:
    """fast2::fwht_row on [..., 32 lanes, 4] fp32 (column 4 * lane + j): the in-lane butterflies, then the shuffles
    (lane bit m: upper lanes take o - v, lower v + o)."""

    v = f32(v).copy()
    with np.errstate(invalid="ignore", over="ignore"):
        a, b = v[..., 0].copy(), v[..., 1].copy()
        v[..., 0], v[..., 1] = a + b, a - b
        a, b = v[..., 2].copy(), v[..., 3].copy()
        v[..., 2], v[..., 3] = a + b, a - b
        a, b = v[..., 0].copy(), v[..., 2].copy()
        v[..., 0], v[..., 2] = a + b, a - b
        a, b = v[..., 1].copy(), v[..., 3].copy()
        v[..., 1], v[..., 3] = a + b, a - b
        m = 1
        while m < 32:
            o = v[..., LANE ^ m, :]
            up = ((LANE & m) != 0)[:, None]
            v = np.where(up, o - v, v + o).astype(np.float32)
            m <<= 1
    return v


def gateup_formula(v, w, sg, su, sdd, limit) -> np.ndarray:
    """fat's per-element gate/up arithmetic between the two transforms (fp32, left to right as written)."""

    lim = np.float32(limit)
    with np.errstate(invalid="ignore", over="ignore"):
        gg = np.fmin(bf16r(f32(f32(v * HAD) * sg)), lim)
        uu = np.fmin(np.fmax(bf16r(f32(f32(w * HAD) * su)), -lim), lim)
        a = bf16r(f32(bf16r(f32(gg / f32(np.float32(1) + np.exp(-gg)))) * uu))
        return f32(a * sdd)


def down_formula(v, sv) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        return f32(f32(v * HAD) * sv)


def to_f16(v) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        return f32_to_f16(f32(v * HAD))


# -- the problem ----------------------------------------------------------------------------------------------------------
class Group:
    """glue's grouping as tests/cuda/test_fastpf_patches._fast_group builds it: distinct picked ids ascending (the shared
    expert's id E included: the kernels skip it), each one's row * 32 + slot codes in row order, one spare entry;
    ``maxm`` = the window's rows."""

    def __init__(self, picks: np.ndarray, E: int, spare: int = 1, stale: np.random.Generator | None = None) -> None:
        n, slots = picks.shape
        flat = picks.reshape(-1).astype(np.int64)
        code = (np.repeat(np.arange(n), slots) * 32 + np.tile(np.arange(slots), n)).astype(np.int32)
        used = np.unique(flat)
        self.members = np.full((used.size + spare, n), -1, np.int32)
        for u, e in enumerate(used):
            m = code[flat == e]
            self.members[u, :m.size] = m
        self.ids = np.zeros(used.size + spare, np.int32)
        self.ids[:used.size] = used
        self.count = np.array([used.size], np.int32)
        self.slots, self.rows, self.E = slots, n, E
        if stale is not None and spare:                       # entries past the count: an earlier window's garbage
            self.ids[used.size:] = stale.integers(0, E + 3, spare)
            self.members[used.size:] = stale.integers(-1, n * 32, (spare, n))


def make_picks(rows: int, E: int, top: int, rng: np.random.Generator, kind: str = "uniform") -> np.ndarray:
    """[rows, top + 1] int32: top distinct experts a row (uniform, or Zipf 0.8 skewed), the last slot = E (shared)."""

    w = np.ones(E) if kind == "uniform" else 1.0 / np.arange(1, E + 1) ** 0.8
    picks = np.full((rows, top + 1), E, np.int32)
    for r in range(rows):
        picks[r, :top] = rng.choice(E, size=top, replace=False, p=w / w.sum())
    return picks


def pair_rows(grp: Group) -> np.ndarray:
    """Pair indices (row * slots + slot) of every member of a real expert (id < E), sorted."""

    out = []
    for u in range(int(grp.count[0])):
        if grp.ids[u] >= grp.E:
            continue
        c = grp.members[u][grp.members[u] >= 0]
        out.append((c >> 5) * grp.slots + (c & 31))
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)


def dense_weight(words: np.ndarray) -> np.ndarray:
    """One expert's trellis [KT, NT, 32] -> W [K, N] fp32 as the kernels multiply it: W[kt 16 + k, nt 16 + n] = the
    value the A fragment puts at row n, column k of tile (kt, nt)."""

    KT, NTL = words.shape[:2]
    af = a_fragment(words)                                   # [KT, NT, 32, 4]
    Wt = np.full((KT, NTL, 16, 16), np.nan, np.float32)      # [kt, nt, n, k]
    for r, (dr, dk) in enumerate(((0, 0), (8, 0), (0, 8), (8, 8))):
        lo, hi = _halves(af[..., r])
        Wt[:, :, G + dr, dk + 2 * T] = lo
        Wt[:, :, G + dr, dk + 2 * T + 1] = hi
    return Wt.transpose(0, 3, 1, 2).reshape(KT * 16, NTL * 16)


def _chain(xrows: np.ndarray, W: np.ndarray) -> np.ndarray:
    """[n, K] x [K, N] -> [n, N]: the fp32 sum over k in order from +0.0, rounding each step (the model's mma chain)."""

    acc = np.zeros((xrows.shape[0], W.shape[1]), np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        for k in range(W.shape[0]):
            acc = (acc + (xrows[:, k:k + 1] * W[k:k + 1, :]).astype(np.float32)).astype(np.float32)
    return acc


def _pairs_of(grp: Group, u: int) -> np.ndarray:
    c = grp.members[u][grp.members[u] >= 0]
    return (c >> 5) * grp.slots + (c & 31)


def reference_gateup(X, TG, TU, grp: Group, svh_g, svh_u, suh_d, limit) -> np.ndarray:
    """Xd [P, N] fp16 bits (NaN pattern where never written)."""

    P, K = X.shape
    N = TG.shape[2] * 16
    xd = np.full((P, N), NAN16, np.uint16)
    for u in range(int(grp.count[0])):
        e = int(grp.ids[u])
        if e >= grp.E:
            continue
        ps = _pairs_of(grp, u)
        if ps.size == 0:
            continue
        xr = f16_to_f32(X[ps])
        ag = _chain(xr, dense_weight(TG[e]))
        au = _chain(xr, dense_weight(TU[e]))
        for nb in range(N // 128):
            cs = slice(nb * 128, nb * 128 + 128)
            v = fwht_row(ag[:, cs].reshape(-1, 32, 4))
            w = fwht_row(au[:, cs].reshape(-1, 32, 4))
            sc = [f16_to_f32(s[e, cs]).reshape(32, 4) for s in (svh_g, svh_u, suh_d)]
            v = fwht_row(gateup_formula(v, w, *sc, limit))
            xd[ps, cs] = to_f16(v).reshape(-1, 128)
    return xd


def reference_down(XD, TD, grp: Group, svh_d) -> np.ndarray:
    """Y [P, N] fp32 (NaN where never written)."""

    P, K = XD.shape
    N = TD.shape[2] * 16
    y = np.full((P, N), np.nan, np.float32)
    for u in range(int(grp.count[0])):
        e = int(grp.ids[u])
        if e >= grp.E:
            continue
        ps = _pairs_of(grp, u)
        if ps.size == 0:
            continue
        acc = _chain(f16_to_f32(XD[ps]), dense_weight(TD[e]))
        for nb in range(N // 128):
            cs = slice(nb * 128, nb * 128 + 128)
            v = fwht_row(acc[:, cs].reshape(-1, 32, 4))
            y[ps, cs] = down_formula(v, f16_to_f32(svh_d[e, cs]).reshape(32, 4)).reshape(-1, 128)
    return y


# -- the shared device pieces -------------------------------------------------------------------------------------------
def plan_model(uids, ucount, members, E: int, BM: int) -> np.ndarray:
    """fast2::plan_kernel: plan[0] = passes in all, plan[1 + u] = the first pass of distinct expert u."""

    maxu, maxm = members.shape
    items = np.zeros(max(maxu, 1), np.int64)
    for u in range(maxu):
        if u < int(ucount[0]) and uids[u] < E:
            m = members[u]
            lo, hi = 0, maxm                                  # the kernel's binary search for the first -1
            while lo < hi:
                mid = (lo + hi) >> 1
                if m[mid] >= 0:
                    lo = mid + 1
                else:
                    hi = mid
            items[u] = (lo + BM - 1) // BM
    sc = np.cumsum(items)
    plan = np.zeros(maxu + 2, np.int64)
    plan[2:2 + maxu] = sc[:maxu]
    plan[1] = 0
    plan[0] = sc[maxu - 1]
    return plan


def swz(CPR: int, r, c):
    return (c ^ (r & 7)) if CPR == 8 else (c ^ ((r >> 1) & 3))


class Ticket:
    def __init__(self, rng: np.random.Generator, lossy: bool = False) -> None:
        self.v, self.rng, self.lossy, self.last = 0, rng, lossy, None

    def take(self) -> int:
        if self.lossy and self.last is not None and self.rng.random() < 0.3:
            return self.last                                  # a lost update: the previous value again
        self.last = self.v
        self.v += 1
        return self.last


class Out:
    """An output array with a write count per element (every stored element must be written exactly once)."""

    def __init__(self, shape, kind: str) -> None:
        self.kind = kind
        self.a = np.full(shape, NAN16, np.uint16) if kind == "f16" else np.full(shape, np.nan, np.float32)
        self.n = np.zeros(shape, np.int32)

    def put(self, row: int, cols: np.ndarray, vals: np.ndarray) -> None:
        self.a[row, cols] = vals
        self.n[row, cols] += 1


def run_ctas(gens: list, rng: np.random.Generator) -> None:
    """Advance the CTAs' generators in a random interleaving (each yields at its ticket claims)."""

    live = list(gens)
    while live:
        i = int(rng.integers(len(live)))
        try:
            next(live[i])
        except StopIteration:
            live.pop(i)


def _lane_rows(v: np.ndarray) -> np.ndarray:
    return v.reshape(32, 4)


# -- fat (exl3_fast.cu, namespace fat) ------------------------------------------------------------------------------------
def fat_port(kind: int, X, T0, T1, grp: Group, N: int, sv0, sv1, sd, limit, rng: np.random.Generator, *,
             NSA: int = 3, grid: int = 3, ticket: bool = True) -> Out:
    """fat's expert_kernel. kind 0: gate/up (MATS 2, FAT_GU = MTL 2, NG 8, KS 4, NGR 4, the shared input); kind 1:
    down (MATS 1, NGR 8; FAT_DN = MTL 2, NG 8, KS 4, or FAT_DN_LARGE = MTL 2, NG 16, KS 2 when maxm >= 4096)."""

    uids, ucount, members, slots = grp.ids, grp.count, grp.members, grp.slots
    maxu, maxm = members.shape
    P, K = X.shape
    E = T0.shape[0]
    MATS = 2 if kind == 0 else 1
    if kind == 0:
        MTL, NG, KS, NGR = 2, 8, 4, 4
    elif maxm >= 4096:
        MTL, NG, KS, NGR = 2, 16, 2, 8
    else:
        MTL, NG, KS, NGR = 2, 8, 4, 8
    BM, WPM = NG * 8, NB // MTL
    W = MATS * WPM
    XM, CPR, LDA, LDE, ER, ROUNDS = 1, KS * 2, KS * 16, 132, NGR * 8, NG // NGR
    A_U16 = XM * BM * LDA
    STAGE = A_U16 + MATS * KS * NB * 32 * 2
    KT, NTILES, NBLK = K >> 4, N >> 4, N // 128
    S = KT // KS
    plan = plan_model(uids, ucount, members, E, BM)
    total = int(plan[0]) * NBLK
    out = Out((P, N), "f16" if kind == 0 else "f32")
    tk = Ticket(rng)
    T0h, T1h = T0.view(np.uint16), T1.view(np.uint16)
    lr = (LANE & 7) + ((LANE >> 4) << 3)
    lc = (LANE >> 3) & 1

    def cta(bid: int):
        smem = np.full(max(NSA * STAGE, MATS * ER * LDE * 2), NAN16, np.uint16)
        cp = AsyncCopies(smem, rng)
        u, nxt_static = 0, bid
        while True:
            yield
            it = tk.take() if ticket else nxt_static
            nxt_static += grid
            if it >= total:
                return
            tt, nb = it // NBLK, it % NBLK
            while u + 1 < maxu and plan[2 + u] <= tt:
                u += 1
            pas, e = tt - int(plan[1 + u]), int(uids[u])
            rows = np.full(BM, -1, np.int64)
            cnt = 0
            for i in range(BM):
                m = pas * BM + i
                code = int(members[u, m]) if m < maxm else -1
                rows[i] = (code >> 5) * slots + (code & 31) if code >= 0 else -1
                if code >= 0:
                    cnt = max(cnt, i + 1)

            def load(s):
                base = (s % NSA) * STAGE
                k0 = s * KS * 16
                for i in range(XM * BM * CPR):
                    r, c = (i // CPR) % BM, i % CPR
                    row = rows[r]
                    data = X[row, k0 + c * 8:k0 + c * 8 + 8] if row >= 0 else np.zeros(8, np.uint16)
                    cp.copy(base + r * LDA + swz(CPR, r, c) * 8, data)
                for i in range(MATS * KS * NB * 8):
                    q, n, kk, m = i & 7, (i >> 3) % NB, (i // (8 * NB)) % KS, i // (8 * NB * KS)
                    src = (T1h if m else T0h)[e, s * KS + kk, nb * NB + n]
                    cp.copy(base + A_U16 + (((m * KS + kk) * NB + n) * 32 + q * 4) * 2, src[q * 8:q * 8 + 8])

            acc = np.zeros((W, MTL, NG, 32, 4), np.float32)
            for s in range(NSA - 1):
                if s < S:
                    load(s)
                cp.commit()
            cp.wait(NSA - 2)
            for s in range(S):
                if s + NSA - 1 < S:
                    load(s + NSA - 1)
                cp.commit()
                base = (s % NSA) * STAGE
                for warp in rng.permutation(W):
                    cp.jitter()
                    mat, sl = warp // WPM, warp % WPM
                    for kk in range(KS):
                        widx = base + A_U16 + 2 * (mat * KS * NB * 32 + sl * MTL * 32 + LANE + (kk * NB +
                                                                                              np.arange(MTL)[:, None]) * 32)
                        words = smem[widx].astype(np.uint32) | (smem[widx + 1].astype(np.uint32) << 16)
                        af = a_fragment(words)                             # [MTL, 32, 4]
                        chunk = swz(CPR, lr, kk * 2 + lc)
                        idx, aa, bb0, bb1 = [], [], [], []
                        for np_ in range(NG // 2):
                            if 16 * np_ < cnt:
                                b = ldsm4(smem, base + (16 * np_ + lr) * LDA + chunk * 8)
                                for lo in range(MTL):
                                    idx += [(lo, 2 * np_), (lo, 2 * np_ + 1)]
                                    aa += [af[lo], af[lo]]
                                    bb0 += [b[:, 0], b[:, 2]]
                                    bb1 += [b[:, 1], b[:, 3]]
                        if idx:
                            ii = tuple(np.array(idx).T)
                            acc[warp][ii] = mma_batch(acc[warp][ii], np.stack(aa), np.stack(bb0), np.stack(bb1))
                cp.wait(NSA - 2)
            # epilogue (shared memory reused as fp32 ep[mat][row][col])
            ep = np.full(MATS * ER * LDE, np.nan, np.float32)
            for rd in range(ROUNDS):
                if ER * rd < cnt:
                    for warp in range(W):
                        mat, sl = warp // WPM, warp % WPM
                        for n in range(NGR):
                            ng = rd * NGR + n
                            Eb = (mat * ER + 8 * n + 2 * T) * LDE + sl * 16 * MTL + G
                            for lo in range(MTL):
                                ep[Eb + lo * 16] = acc[warp, lo, ng, :, 0]
                                ep[Eb + LDE + lo * 16] = acc[warp, lo, ng, :, 1]
                                ep[Eb + lo * 16 + 8] = acc[warp, lo, ng, :, 2]
                                ep[Eb + LDE + lo * 16 + 8] = acc[warp, lo, ng, :, 3]
                for r in range(ER):
                    if ER * rd + r >= cnt:
                        break
                    row = int(rows[ER * rd + r])
                    c0 = 4 * LANE
                    cols = (nb * 128 + c0[:, None] + np.arange(4)).reshape(-1)
                    v = fwht_row(ep[r * LDE + c0[:, None] + np.arange(4)])
                    if kind == 0:
                        w = fwht_row(ep[(ER + r) * LDE + c0[:, None] + np.arange(4)])
                        sc = [f16_to_f32(a[e, cols]).reshape(32, 4) for a in (sv0, sv1, sd)]
                        v = fwht_row(gateup_formula(v, w, *sc, limit))
                        out.put(row, cols, to_f16(v).reshape(-1))
                    else:
                        out.put(row, cols, down_formula(v, f16_to_f32(sv0[e, cols]).reshape(32, 4)).reshape(-1))

    run_ctas([cta(b) for b in range(grid)], rng)
    return out


# -- fat2 (exl3_fat2.cu) ----------------------------------------------------------------------------------------------
def fat2_port(kind: int, X, T0, T1, grp: Group, N: int, sv0, sv1, sd, limit, rng: np.random.Generator, *,
              NG: int = 16, NSA: int = 4, grid: int = 3, ticket: bool = True, mutate: str = "") -> Out:
    """exl3_fat2.cu's expert_kernel<KIND, NG, NSA, 0> line for line (see the module docstring)."""

    uids, ucount, members, slots = grp.ids, grp.count, grp.members, grp.slots
    maxu, maxm = members.shape
    P, K = X.shape
    E = T0.shape[0]
    KS, MTL, WPM, W = 2, 2, 4, 8
    BM, CPR, LDA = NG * 8, KS * 2, KS * 16
    NGR, LDE = 4, 132
    ER, ROUNDS = NGR * 8, NG // NGR
    A_U16 = BM * LDA
    STAGE = A_U16 + 2 * KS * NB * 32 * 2
    KT, NTILES, NBLK = K >> 4, N >> 4, N // 128
    S = KT // KS
    assert K % 32 == 0 and S >= NSA and N % 128 == 0
    NBI = NBLK if kind == 0 else (NBLK + 1) // 2
    plan = plan_model(uids, ucount, members, E, BM)
    total = int(plan[0]) * NBI
    out = Out((P, N), "f16" if kind == 0 else "f32")
    tk = Ticket(rng, lossy=mutate == "ticket")
    T0h, T1h = T0.view(np.uint16), T1.view(np.uint16)
    lr = (LANE & 7) + ((LANE >> 4) << 3)
    lc = (LANE >> 3) & 1
    wait_n = NSA - 1 if mutate == "wait" else NSA - 2

    def cta(bid: int):
        smem = np.full(NSA * STAGE, NAN16, np.uint16)            # the ring (dynamic shared memory, garbage)
        ep = np.full(2 * ER * LDE, np.nan, np.float32)           # the epilogue buffer (after the ring)
        rows_sh = np.full((2, BM), -777, np.int64)
        it_sh, u_sh, e_sh, cnt_sh = [0, 0], [0, 0], [0, 0], [0, 0]
        cp = AsyncCopies(smem, rng)
        st = {"u": 0, "next": bid}

        def claim(b):                                            # thread 0
            yield
            it = tk.take() if ticket else st["next"]
            st["next"] += grid
            it_sh[b] = it
            if it < total:
                tt = it // NBI
                while st["u"] + 1 < maxu and plan[2 + st["u"]] <= tt:
                    st["u"] += 1
                u_sh[b] = st["u"]
            cnt_sh[b] = 0

        def fill(b):                                             # every thread
            it = it_sh[b]
            if it >= total:
                return
            uu = u_sh[b]
            pas = it // NBI - int(plan[1 + uu])
            e_sh[b] = int(uids[uu])
            for i in range(BM):
                m = pas * BM + i
                code = int(members[uu, m]) if m < maxm else -1
                rows_sh[b][i] = (code >> 5) * slots + (code & 31) if code >= 0 else -1
                if code >= 0:
                    cnt_sh[b] = max(cnt_sh[b], i + 1)

        def load_stage(b, s, slot, rows_b=None):
            q, e, cnt = it_sh[b] % NBI, e_sh[b], cnt_sh[b]
            rows = rows_sh[b] if rows_b is None else rows_sh[rows_b]
            base = slot * STAGE
            k0 = s * KS * 16
            rows16 = BM if mutate == "fill_cnt" else min(BM, (cnt + 15) & ~15)
            for i in range(rows16 * CPR):
                r, c = i // CPR, i % CPR
                row = int(rows[r])
                if mutate == "fill_cnt" and row < 0:
                    continue
                data = X[row, k0 + c * 8:k0 + c * 8 + 8] if row >= 0 else np.zeros(8, np.uint16)
                cc = c if mutate == "swz" else swz(CPR, r, c)
                cp.copy(base + r * LDA + cc * 8, data)
            for i in range(2 * KS * NB * 8):
                qq, n, kk, hh = i & 7, (i >> 3) % NB, (i // (8 * NB)) % KS, i // (8 * NB * KS)
                nb = q if kind == 0 else 2 * q + hh
                if nb < NBLK:
                    src = (T1h if (kind == 0 and hh) else T0h)[e, s * KS + kk, nb * NB + n]
                    cp.copy(base + A_U16 + (((hh * KS + kk) * NB + n) * 32 + qq * 4) * 2, src[qq * 8:qq * 8 + 8])

        yield from claim(0)
        if it_sh[0] >= total:
            return
        fill(0)
        for p in range(NSA - 1):
            load_stage(0, p, p)
            cp.commit()
            cp.jitter()
        cp.wait(wait_n)

        cur, gs = 0, 0
        acc = np.zeros((W, MTL, NG, 32, 4), np.float32)
        while True:
            nxt = cur ^ 1
            q, cnt, e = it_sh[cur] % NBI, cnt_sh[cur], e_sh[cur]
            if mutate != "acc":
                acc = np.zeros((W, MTL, NG, 32, 4), np.float32)
            for s in range(S):
                ls = s + NSA - 1
                slot = gs % NSA if mutate == "slot" else (gs + NSA - 1) % NSA
                if ls < S:
                    load_stage(cur, ls, slot)
                elif it_sh[nxt] < total:
                    if ls == S:
                        fill(nxt)
                    if mutate == "next_rows":
                        load_stage(nxt, ls - S, slot, rows_b=cur)
                    elif mutate == "next_stage":
                        load_stage(nxt, min(ls - S + 1, S - 1), slot)
                    else:
                        load_stage(nxt, ls - S, slot)
                cp.commit()
                base = (gs % NSA) * STAGE
                for warp in rng.permutation(W):
                    cp.jitter()
                    h, sl = warp // WPM, warp % WPM
                    nbw = q if kind == 0 else 2 * q + h
                    if nbw >= NBLK:
                        continue                                     # live == false (warp-uniform)
                    hw = 0 if (mutate == "down_pair" and kind == 1) else h
                    for kk in (range(KS - 1, -1, -1) if mutate == "kk" else range(KS)):
                        widx = base + A_U16 + 2 * (hw * KS * NB * 32 + sl * MTL * 32 + LANE +
                                                   (kk * NB + np.arange(MTL)[:, None]) * 32)
                        words = smem[widx].astype(np.uint32) | (smem[widx + 1].astype(np.uint32) << 16)
                        af = a_fragment(words)
                        chunk = swz(CPR, lr, kk * 2 + lc)
                        idx, aa, bb0, bb1 = [], [], [], []
                        for np_ in range(NG // 2):
                            live_grp = (16 * np_ + 16 < cnt) if mutate == "skip" else (16 * np_ < cnt)
                            if live_grp:
                                b = ldsm4(smem, base + (16 * np_ + lr) * LDA + chunk * 8)
                                for lo in range(MTL):
                                    idx += [(lo, 2 * np_), (lo, 2 * np_ + 1)]
                                    aa += [af[lo], af[lo]]
                                    bb0 += [b[:, 0], b[:, 2]]
                                    bb1 += [b[:, 1], b[:, 3]]
                        if idx:
                            ii = tuple(np.array(idx).T)
                            acc[warp][ii] = mma_batch(acc[warp][ii], np.stack(aa), np.stack(bb0), np.stack(bb1))
                claim_at = S - NSA + (1 if mutate == "claim_late" else 0)
                if s == claim_at:
                    yield from claim(nxt)
                cp.wait(wait_n)
                gs += 1

            # epilogue (its own buffer; the ring keeps landing the next item's stages)
            rows = rows_sh[cur]
            for rd in range(ROUNDS):
                if ER * rd < cnt:
                    for warp in range(W):
                        h, sl = warp // WPM, warp % WPM
                        if (q if kind == 0 else 2 * q + h) >= NBLK:
                            continue
                        for n in range(NGR):
                            ng = rd * NGR + n
                            Eb = (h * ER + 8 * n + 2 * T) * LDE + sl * 16 * MTL + G
                            r0, r1 = (LDE, 0) if mutate == "epi_row" else (0, LDE)
                            for lo in range(MTL):
                                ep[Eb + r0 + lo * 16] = acc[warp, lo, ng, :, 0]
                                ep[Eb + r1 + lo * 16] = acc[warp, lo, ng, :, 1]
                                ep[Eb + r0 + lo * 16 + 8] = acc[warp, lo, ng, :, 2]
                                ep[Eb + r1 + lo * 16 + 8] = acc[warp, lo, ng, :, 3]
                c0 = 4 * LANE
                if kind == 0:
                    for r in range(ER):
                        if ER * rd + r >= cnt:
                            break
                        row = int(rows[ER * rd + r])
                        cols = (q * 128 + c0[:, None] + np.arange(4)).reshape(-1)
                        v = fwht_row(ep[r * LDE + c0[:, None] + np.arange(4)])
                        w = fwht_row(ep[(ER + r) * LDE + c0[:, None] + np.arange(4)])
                        sc = [f16_to_f32(a[e, cols]).reshape(32, 4) for a in (sv0, sv1, sd)]
                        v = fwht_row(gateup_formula(v, w, *sc, limit))
                        out.put(row, cols, to_f16(v).reshape(-1))
                else:
                    for j2 in range(2 * ER):
                        hh, r = j2 // ER, j2 % ER
                        nbh = 2 * q + hh
                        if ER * rd + r >= cnt or nbh >= NBLK:
                            continue
                        row = int(rows[ER * rd + r])
                        cols = (nbh * 128 + c0[:, None] + np.arange(4)).reshape(-1)
                        v = fwht_row(ep[(hh * ER + r) * LDE + c0[:, None] + np.arange(4)])
                        out.put(row, cols, down_formula(v, f16_to_f32(sv0[e, cols]).reshape(32, 4)).reshape(-1))
            cur = nxt
            if it_sh[cur] >= total:
                break

    run_ctas([cta(b) for b in range(grid)], rng)
    return out


# -- a random layer ------------------------------------------------------------------------------------------------------
class Layer:
    """Random EXL3-shaped expert weights on small or real shapes: gate / up trellis [E, K/16, N/16, 32] (K = D),
    down [E, N/16, D/16, 32]; the fp16 scales. The words are random 32-bit values (every trellis word is valid)."""

    def __init__(self, D: int, NI: int, E: int, rng: np.random.Generator) -> None:
        self.D, self.NI, self.E = D, NI, E
        w = lambda a, b: rng.integers(0, 2 ** 32, size=(E, a // 16, b // 16, 32), dtype=np.uint64).astype(np.uint32)
        self.gt, self.ut, self.dt = w(D, NI), w(D, NI), w(NI, D)
        sc = lambda n: f32_to_f16(f32(rng.choice([-1.0, 1.0], size=(E, n)) * rng.uniform(0.6, 1.4, size=(E, n))))
        self.svh_g, self.svh_u, self.suh_d, self.svh_d = sc(NI), sc(NI), sc(NI), sc(D)


def rand_x(P: int, K: int, rng: np.random.Generator) -> np.ndarray:
    return f32_to_f16(f32(rng.standard_normal((P, K)) * 0.5))


def same(a: Out | np.ndarray, b: Out | np.ndarray) -> bool:
    """Bitwise equal (NaN patterns included)."""

    x = a.a if isinstance(a, Out) else a
    y = b.a if isinstance(b, Out) else b
    return x.shape == y.shape and bool(np.array_equal(x.view(np.uint16 if x.dtype == np.uint16 else np.uint32),
                                                      y.view(np.uint16 if y.dtype == np.uint16 else np.uint32)))
