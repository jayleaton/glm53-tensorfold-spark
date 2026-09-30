# Decode kernels 2: the dense size switch (0570) and the expert load path (0580), same bits (2026-09-30)

> **Update 2026-09-30 (W19, docs/RESULTS.md):** measured on the GPU in image b10. **0580 adopted**
> (`GLM53_TF_DEC_EXPERT_LOADS=1`, `_CFG=nc,8,1`): same bits, load probe 226-231 GB/s, routed-expert decode -4.1..-4.6%
> in situ at 1 stream, -2.9..-3.1% at 4 streams (the strict 1.05x kernel bar missed on some windows; the trace decided).
> **0570 not adopted**: the cold bench passed, but in situ the switched shapes ran 1.7x slower than `_qmm` + `_reduce`.
> The estimates below are the offline ones.


> Offline work: no GPU was used and the Sparks were not touched. Inputs: RESULTS W11 (§2-§4: the measured round and
> kernels), W12 (§1: 0440's microbench), W16 (§1: THEORY-2 items 7 and 8, `results/W16/probes-head/littles.log`,
> `cold.log` / `cold.json`), THEORY-2 §2-§3.3, DECODE-KERNELS.md (0440), the production `exl3.cu` / `qmm.py` and
> 0440's `exl3_stream.cu` / `q4_stream.cu`. Every speed number for the new code is an **estimate**; §7 is the GPU plan
> that turns them into measurements. Stack: image b9's list (0001-0490, 0500, 0540, 0550, 0560) + 0570 + 0580.

## 0. Bottom line

| patch | knob (default off) | what | bits | estimate, 1 stream prose (W11 round 53.3 ms) | 4 streams (121.3 ms) |
| --- | --- | --- | --- | --- | --- |
| 0570 | `GLM53_TF_DEC_QMM_MAXMB=3.5` | 0440's `q4_stream.cu` only for 4-bit matrices <= 3.5 MiB, with W16's cold placement per shape; `_qmm` for the rest | `q4_stream.cu` == `_qmm` + `_reduce` for every shape and row count (0440, W12 GPU-verified); the switch picks by (shape, rows) only | **-0.15 to -0.23 ms (+0.3-0.4%)**; with `GLM53_TF_DEC_PDL=1` up to -0.37 ms | -0.17 to -0.30 ms (+0.15-0.25%) |
| 0580 | `GLM53_TF_DEC_EXPERT_LOADS=1` (+ `_CFG`, `_PDL`) | exl3.cu's `grouped_kernel` (the routed experts' gate/up and down GEMVs of every <= 16-row window) with a new load path: one-round-trip prologue, a 2-deep register ring of 128-bit `ld.global.nc` loads a warp (24 KB an SM in flight), optional PDL prologue | same grid, same per-warp K ranges, same mma chain per accumulator, same warp sum order, same Z rows: exl3.cu's bits | 203 -> 220-230 GB/s: **-2.0 to -3.1 ms (+4.0-6.2%)**; the gate's floor (1.05x) is -1.3 ms (+2.5%) | 219 -> 225-230 GB/s: -2.1 to -3.6 ms (+1.7-3.1%) |

- **Which load technique reached 232.7 GB/s in `littles.cu`: plain 128-bit non-coherent loads into registers**
  (`ld.global.nc.L1::no_allocate.v4.u32`, the `nc` rows, "tile" pattern = the EXL3 gate/up layout): D1 C2 = one
  16-byte load a lane, 8 warps an SM = **4 KB an SM in flight -> 232.7 GB/s, 54 registers**. The cp.async ring needs
  6 KB an SM (S2 G1 C3: 234.3) and TMA bulk copies 8-16 KB (bulk tile CH4K S2 C2: 234.2 at 16 KB; flat 8 KB). 0580 uses
  it (`nc`, default) and keeps the other two as settings (`w32`: grouped_kernel's own 32-bit loads made deeper; `cpa`:
  a cp.async ring) for the GPU A/B.
- **A finding that changes THEORY-2's reading of E1.** littles ran E1's exact ring shape (warp-private cp.async,
  S4 G1, 3 CTAs x 4 warps, 18 KB an SM) at **234.1 GB/s**. So the mechanism was never the cap; 0440's E1 lost
  (~158 GB/s with decode and mma off) in the rest of its design: a persistent walk where every CTA stops at every item
  boundary (a `__threadfence`, a ticket atomic, 3-4 `__syncthreads`, a fused epilogue that reads Z / scales back
  through L2), all CTAs at the same moment because the items are equal. And the "~5 µs effective latency" was
  inferred from E1 (18 KB / 158 GB/s); the probe's latency under load is 0.47 µs (0.25 streaming CTAs an SM) to
  1.15 µs (1 CTA an SM at 238 GB/s), rising only once the SM is oversubscribed (3.75 µs at 4 CTAs). 0580 therefore
  keeps grouped_kernel's grid and CTAs and changes only the loads.
- **Why grouped_kernel is at 203-214 GB/s in the 1-stream windows (and 219-221 at U 50-60).** A warp issues its 8
  loads (1 KB) at the top of a k step and waits; during the decode + 16 mma (≈20% of issue time: ncu SM throughput
  18-25%) nothing is in flight; each CTA starts with three dependent round trips (ucount -> uids -> members ->
  `__syncthreads`) before its first weight load, and a small-U launch is only 3.6 waves of 144 CTA slots (U 8), so the
  ramp and the tail (1-2 CTAs an SM left, ~2-4 KB in flight) are a large share. The in-flight time average falls
  under littles' ~4 KB knee exactly where the kernel is short: U 8-22 (an inference from W11's kernel table and ncu;
  §7 step 3's probes and settings test it).
- **0580's design against that** (§2): the prologue in one round trip with the first 2 k steps of weights issued
  before the member rows are in shared memory; while step s is decoded and multiplied, steps s+1 and s+2 are in
  flight (2 KB a warp, 8 KB a CTA: a lone CTA on an SM in the tail still sits at littles' 229-237 GB/s point); the
  A fragments ride the same ring. Registers 163 (grouped_kernel 109), shared memory unchanged (the staging area lives
  in the warp's slice of the reduction buffer), 3 CTAs an SM as before, 0 spills.
- **Gate (§7, `tests/cuda/bench_decode_kernels.py --loads`, cold L2):** the load path alone (probe 3: no decode, no
  mma) >= 220 GB/s on every U 8-22 window before anything else; then one config >= 1.05x grouped_kernel (gate/up +
  down) on every U 8-22 window with a 64 MiB streaming predecessor ("flush", in-situ-like) and >= 1.00x back to back.
- **0570 is small and says so:** the shapes under 3.5 MiB are 1.3-1.5 ms of the 16.1 ms dense time a round; W16's
  cold 1.1-1.3x on them is worth ~0.2 ms (0.3-0.4%), below one A/B's noise (W16: ±1%). It is exact and free, so it can
  ride with 0580; adopt it on a per-kernel in-situ trace (§7 step 5), not on tok/s.

## 1. What W16 measured (`results/W16/probes-head/littles.log`, clocks locked at 2,250 MHz)

| mechanism, pattern | smallest KB an SM in flight for >= 230 GB/s | at that point | best |
| --- | --- | --- | --- |
| **`ld.global.nc.v4`, tile (EXL3 gate/up layout)** | **4.0** | **D1 C2: 232.7 GB/s, 8 warps, 54 regs** (D2 C1: 229.1; D1 C1 2 KB: 180.7) | 237.4 (D2 C2, 8 KB) |
| `ld.global.nc.v4`, flat | 4.0 | D1 C2: 231.4 (D1 C1 2 KB: 139.6) | 232.9 |
| cp.async ring, tile | 6.0 | S2 G1 C3: 234.3 (E1's S4 G1 C3, 18 KB: 234.1) | 235.6 |
| TMA bulk, tile / flat | 16.0 / 8.0 | CH4K S2 C2: 234.2 / CH4K S2 C1: 236.5 | 235.5 / 237.8 |
| dependent chase (512-B nodes) | none (best 229.6 at 64 KB) | | |

Latency: pointer chase 416 ns (64-128 B granules over 1.8 GB), 1.88 µs at a new 2 MB page each step, L2 hit 152 ns;
a random 512-B chase beside N streaming CTAs an SM: 0.47 µs (0.25, 159 GB/s streamed), 0.65 (0.5, 224), 1.15 (1, 238),
2.09 (2), 3.75 (4), 7.04 µs (8). Little's law read from the table: at 4 KB an SM the device holds 196 KB in flight,
196 KB / 232.7 GB/s = 0.84 µs effective latency -- the memory system is not a 5 µs pipe.

Item 7 (`cold.log`, 0440's dense kernel vs `_qmm` + `_reduce`, cold): every shape up to index q_b (3.54 MB) wins in
both cold modes; shared gate/up (4.72 MB) is 1.21x rotate but 1.03x flush with an invalid (L2) flush roof; from 14 MB
up 0.82-1.05x. `GLM53_TF_DEC_QMM_MAX_MB candidate: 3.5` is 3.54 MB rounded: **0570 reads the knob in MiB**, so 3.5
MiB = 3.67 MB takes index q_b and leaves shared gate/up (4.5 MiB) out.

## 2. 0580: `exl3_ld.cu` (`ld_kernel<NT, PD, LD, PROBE>`), `expert_loads.py`

**Where it runs.** `exl3_mm.routed` (not fast), the two `ext.grouped` calls (gate/up: 2 matrices, K 4,096 in 4
splits; down: K 1,024, 1 split) of every window whose member table has <= 16 columns (decode, verify, MTP, DFlash2
and the 4-slot windows: W11's 4-stream in-situ kernels are all `grouped`, R <= 16). Windows past 16 rows keep
`grouped_loop`; rot_in and both epilogues stay exl3.cu's. Not reached while 0440's `GLM53_TF_DEC_EXPERTS` or 0130's
decode_v2 take the whole routed path.

**Kept (the bits).**
- The grid and work item: program (u, n block, (mat x SK + split) x MT + member tile), 16 member rows of distinct
  expert u times W_q over one K split for NT column tiles; W = 4 warps.
- Warp w runs k tiles [split x KT/SK + w x PW, + PW) ascending, one `mma.m16n8k16` f16 -> f32 per (k tile, column
  tile, n8 half) into accumulators from +0.0, on `decode_tile`'s B fragments (verbatim) and `load_pair`'s A values.
- The warps' partials through shared memory added `s = red[0]; s += red[1]; ...`, Z stored for live rows only
  (member_tiles' code for MG = 1); the three early exits (u >= ucount, the shared expert, an empty member tile).
- NT places work only (each accumulator is its own chain), as in exl3.cu. PD and LD change only when bytes arrive.

**Changed (data movement).**

| | grouped_kernel<8, 4> (today) | ld_kernel<8, 2, nc> (0580 default) |
| --- | --- | --- |
| prologue | `ucount[0]`, then `uids[u]`, then `members[...]`, `__syncthreads`, then the first loads (3 dependent round trips) | ucount, uids[u], the member codes and the tile's first code in ONE round trip (`ld.global.cg`); the first 2 k steps of trellis words issued before the member rows reach shared memory |
| trellis loads | 8 x `ld.global.nc.u32` a lane at the top of a k step (1 KB a warp), consumed at once | 2 x `ld.global.nc.L1::no_allocate.v4` a lane a k step, issued 2 steps ahead: while step s is decoded / multiplied, s+1 and s+2 are in flight (2 KB a warp) |
| to the decode layout | the loaded word is `tile[i x 32 + lane]` | the v4 words (lane l: words 4 (l % 8) .. + 3 of tile 4 v + l / 8) stored to a 1-KB warp staging area (16-B stores of a contiguous 512 B), read back as `stage[i x 32 + lane]`; two `__syncwarp` a step; conflict-free |
| A fragments (X / Xd rows) | 4 x 4 B a lane at each step, waited for with the weights | the same 4 loads, 2 steps ahead |
| shared memory | `red[4][16][128]` (32 KB) + rows | the same; the staging area / ring is the warp's own slice of `red` (written with partials only after its last step) |
| registers / CTAs an SM | 109 / 3 | 163 / 3 (launch bounds 128 x 3) |
| nominal in flight an SM | 12 KB only at the top of a step, 0 during compute | 24 KB with every warp resident; 8 KB from a lone CTA |

Settings (`GLM53_TF_DEC_EXPERT_LOADS_CFG="ld,nt,pd[/ld,nt,pd]"`, gate/up / down; default `nc,8,2`):

| ld, nt, pd | registers | CTAs an SM | nominal KB an SM in flight | why it is there |
| --- | ---: | ---: | ---: | --- |
| nc, 8, 1 / **2** / 4 | 124 / **163** / 168 | 3 | 12 / **24** / 48 | littles' winner; depth sweep |
| nc, 4, 2 / 4 | 103 / 123 | 4 | 16 / 32 | twice the programs, a finer tail (U 8: 1,024 CTAs instead of 512) |
| w32, 8, 2 / 4, 4, 4 | 161 / 126 | 3 / 4 | 24 / 32 | grouped_kernel's own instruction, deeper: separates "depth" from "vector width" |
| cpa, 8, 2 / 4, 4, 4 | 150 / 110 | 3 / 4 | 24 / 32 | cp.async ring (littles: 234 at 6 KB); no data registers held |
| probes at (8, 2): nc 1 / 2 / 3, w32 3, cpa 3 | 140 / 137 / 77, 56, 64 | 3 | 24 | 1 = no decode, 2 = no mma, **3 = the load path alone** |

`GLM53_TF_DEC_EXPERT_LOADS_PDL=1` (sm_90+, default off): launch as a programmatic dependent. The prologue (grouping
reads + first 2 k steps of weights) runs before `griddepcontrol.wait` and overlaps the previous kernel's tail; X / Xd
are read and Z written after it. Safe because the grouping is written before `routed` starts and the launch right
before each grouped launch (rot_in, gateup_epilogue) is a plain launch, so every earlier kernel has completed (PTX
check: before the wait only `ld.global.cg` and the no_allocate trellis loads / cp.async; no store). W11 measured PDL
on sm_121 (0.73 -> 0.42 µs a boundary, early starts in graphs). 84 grouped launches a 1-stream round: ~0.1-0.3 ms if
the ramp moves under rot_in's / the epilogue's tail (estimate).

## 3. Exactness: why the bits are exl3.cu's, and how that is checked offline

- For every output element, the operands of every mma (the A fragment of the same 16 k values of the same member row,
  the B fragment decoded from the same trellis word), their order along the chain (k ascending within the warp's
  fixed range, starting at +0.0), the warp sum order and the store are grouped_kernel's. Loads only move in time; a
  value loaded earlier is the same value (weights are constant; X / Xd are written by earlier kernels, read after the
  PDL wait). mma rows are independent, so a dead row (A = 0) never leaks into a live one. Hence Z, Xd and Y are the bits
  of today for every row count, NT / PD / LD / PDL setting, and window: drafted == serial, batched == alone, resumed ==
  fresh hold, and the two ranks may run different settings.
- `tests/test_decode_loads_emulator.py` (58 tests, ~4.5 min, numpy): a lane-level port of ld_kernel
  (`tests/decode_loads_emu.py`: the v4 lane mapping, the staging store / read-back, the register ring and its refill
  order, the cp.async ring slots with copies landing at random before the covering `wait_group`, NaN-initialised shared
  memory aliased with `red`, the A fragments PD steps ahead, the prologue exits, the warp-order reduction) against the
  matrix-level model of grouped_kernel (`decode_kernels_emu.exl3_reference`, order-sensitive mma model):
  - Z bit for bit, including the never-stored elements, for all 9 settings on gate/up-like (1 / 2 / 4 K splits) and
    down-like layers: 1-16-row windows, 4-slot mixes, skewed 20 / 40-row windows with 2-3 member tiles (LOOP=0),
    stale distinct-expert slots past ucount, the shared expert;
  - `exl3_mm.routed`'s Xd and Y through the port == the reference routed path (3 setting pairs);
  - the real per-rank K ranges (gate/up 4,096 in 4 splits, down 1,024: 16 k tiles a warp) at PD 2 and 4, nc / cpa /
    w32; rows alone == rows in the window;
  - **8 negative controls, each caught**: staging read lane-major, a slot refilled before it is read (nc and w32), the
    cp.async ring slot off by one, one group too few waited, A fragments of the wrong k step, warps added in reverse,
    a warp's k order rotated;
  - host: the knobs (bad values refuse), `fits` at the real shapes (every setting), the hook dispatch (<= 16 rows ->
    ld_kernel for both launches; 17+ -> grouped_loop; rot_in / epilogues unchanged), the NVMe compat hash ignores the
    knobs, the bench gate logic.
- `tests/test_decode_loads_compile.py` (5 tests, nvcc 13.4, `-arch=sm_121 -O3`): 14 instantiations, 0 spills,
  registers within the launch bounds, static shared memory == grouped_kernel's, CTAs an SM >= grouped_kernel's at
  NT 8; the PTX has the v4 / u32 no_allocate loads, `cp.async.cg` 16 B + `wait_group`, the f16 mma, no atomics; the PDL
  prologue check above; `mcg2` / `decode_tile` / `mma16816` character for character exl3.cu's; **SASS** of the default
  kernel: inside the unrolled k loop the `LDG.E.NA.128` refills sit between the HMMAs of the step being computed (8
  LDG.128, 32 HMMA a 2-step loop body) -- the loads are in flight during the math.
- The full `exl3_ld.cu` (host + device) compiles against torch's headers (C++20, header stubs for the CUDA libraries),
  and `exl3_ld.cpp` passes a syntax check.

## 4. 0570: the dense size switch (`decode_stream.py`)

`GLM53_TF_DEC_QMM_MAXMB=X` (MiB; `GLM53_TF_DEC_QMM_MAX_MB` is the same knob): the dense hook (0440's
`decode_stream.matmul`, turned on by the size knob alone) sends a 4-bit matmul of <= 64 rows to `q4_stream.cu` only if
`Q4.nbytes()` (words + scales + biases) <= X MiB, and only within the shape's row cap; otherwise `qmm.matmul` runs
`_qmm` + `_reduce` as today. `GLM53_TF_DEC_QMM_TABLE=1` (default) gives the switched shapes their W16 placement
(groups a stage, stages, whole-tile or split items; a column for `GLM53_TF_DEC_PDL` on and off), picked by the best
geometric mean over 1-16 rows in both cold modes with no window below ~0.95x. `GLM53_TF_DEC_QMM_EXCLUDE="NxK,..."`
keeps shapes on `_qmm`. The Triton guard stays (3.7.x only). Size checks are cached on the matrix.

Per-rank decode shapes at 3.5 MiB (W11 grids; 1-stream prose calls a round and in-situ µs a call from
`results/W11/insitu-prose-r0.txt`; speed-ups = W16 cold old / new at the table's placement, 2-4 rows, flush / rotate):

| shape (per rank) | MB (MiB) | W11 grid | calls a round, 1s (4s) | in situ µs a call | switched at 3.5 | placement (no PDL / PDL) | W16 flush / rotate, 2-4 rows | ms saved a 1s round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KDA f_b / g_b 4096 x 128 | 0.29 (0.28) | (1,64,1) | 68 (68) | 4.4 | yes | (1,8) whole / (1,6) whole | 1.20 / 1.10 (PDL 1.16 / 1.94) | 0.026-0.050 (PDL 0.04-0.14) |
| index k 160 x 4096 | 0.43 (0.41) | (1,3,8) | 12.1 (13.3) | 8.2 + `_reduce` 1.1 | yes, <= 4 rows | (1,8) split / same | 1.21 / 1.25 (16 rows: 0.76 / 0.72) | 0.019-0.022 |
| shared down 4096 x 1024 | 2.36 (2.25) | (1,64,2) | 43 (43) | ~16.5 (grid median 18.8) | yes | (1,4) whole / same | 1.17 / 1.11 | 0.07-0.10 |
| index q_b 4096 x 1536 | 3.54 (3.375) | (1,64,2) | 12 (12) | ~25 | yes | (1,4) whole / same | 1.15 / 1.08 | 0.02-0.04 |
| (1,16,8) grid: 1024 x 4096 (likely DFlash2's context k/v) | 2.36 (2.25) | (1,16,8) | 8.5 (24.0) | 16.2 + 1.1 | yes (0440's default placement) | not measured cold | (same bytes as shared down: ~1.1-1.17, a guess) | 0.013-0.02 |
| DSA kv_b 8192 x 512 | 2.36 (2.25) | (1,128,4) | 0 in decode (0390's latent kernels absorb it) | - | PDL column only (no PDL: 0.93-0.98x at 1-2 rows) | never / (1,6) whole | 1.04 / 0.94 (PDL 1.05 / 1.26) | 0 |
| shared gate/up, DSA proj 2048 x 4096 | 4.72 (4.5) | (1,32,4) | 55 | 28.5 | **no** | | 1.03 flush (invalid roof) | |
| everything larger (KDA o 9.4, DSA q_b 7.1, 3072 x 4096 7.1, MLP down 14.2, DSA o 18.9, MLP gate/up 28.3, KDA in_proj 29.0, head 178 MB) | | | | | **no** | | 0.82-1.1 | |

Sum: **1 stream -0.15 to -0.23 ms a round (+0.3-0.4%)** without PDL, up to -0.37 ms with `GLM53_TF_DEC_PDL=1` (the
rotate mode's back-to-back PDL wins are the optimistic end); 4 streams (windows of 8-16 rows: f_b/g_b 1.19 / 1.03,
shared down 1.22 / 1.14, index q_b 1.14 / 1.08, index k off past 4 rows, the (1,16,8) grid 24 calls) **-0.17 to
-0.30 ms of 121.3 (+0.15-0.25%)**. It removes ~75 `_reduce` launches a 1-stream round (0.08 ms, inside the numbers).

## 5. End-to-end estimate (W11 / W12 round; 0580)

1 stream prose: the round is 53.3 ms, routed experts 27.8 ms of which rot_in + epilogues 0.90 ms, so the two grouped
launches a layer are **26.9 ms for 5.47 GB = 203 GB/s**. Code: 35.9 ms for 7.46 GB = 208. 4 streams: 73.6 ms for
16.1 GB = 219.

| grouped_kernel -> ld_kernel, in situ | 1 stream prose | code (63.6 ms) | 4 streams (121.3 ms) |
| --- | --- | --- | --- |
| 215 GB/s (the 1.05x gate floor ~ -1.28 ms) | -1.46 ms, **+2.8%** | -1.2 ms, +1.9% | 0 |
| 220 GB/s | -2.04 ms, **+4.0%** | -2.0 ms, +3.2% | -0.4 ms, +0.4% |
| 225 GB/s | -2.59 ms, **+5.1%** | -2.8 ms, +4.5% | -2.1 ms, +1.7% |
| 230 GB/s (the plain-read ceiling is 233-237) | -3.12 ms, **+6.2%** | -3.5 ms, +5.8% | -3.6 ms, +3.1% |

Mid case (225): 1 stream 53.3 -> 50.7 ms, 45 -> 47.7 tok/s; with 0570 (-0.2 ms) and the PDL prologue (-0.1 to
-0.3 ms, unmeasured) +5.6-6.2%. Where it comes from, by the analysis in §0: the ramp (1 round trip instead of 3 before
the first load, then 2 KB a warp at once), the steady state (2 k steps in flight during the math instead of 0), and the
tail (a lone CTA holds 8 KB). The 4-stream windows (U 50-60, 5-7 waves) already stream at 219, so they gain less.
Risks: the extra 54 registers change nothing at 3 CTAs an SM (smem-limited before and after), but the v4 + staging
path adds ~12 instructions a step (2 STS.128, 8 LDS, 2 syncs) on a kernel at 20% issue use; if probe 3 passes and the
kernel does not, the w32 / cpa settings and PD 1 separate "depth" from "instruction mix" in the same bench run.

## 6. What was not done, and why

- No persistent grid, no fused epilogues, no tickets: that is what cost E1 its bandwidth (§0); 0580 keeps five
  launches a layer. Fusing the epilogues is a separate step once the loads are proven.
- No TMA bulk copies: littles needs 2-4x the bytes in flight for the same GB/s, and the EXL3 decode needs word `lane`
  of each tile in a register, which a v4 load + staging (or a cp.async ring) gives more cheaply than an mbarrier ring.
- `grouped_loop` (prefill chunks, windows past 16 rows) is untouched.

## 7. GPU test plan (one window, ~75 min, prod stopped; one session owns the hardware; everything under `timeout`)

Image **b10** = b9's list (`results/W17/build-patches.txt`) + 0570 + 0580 (both apply in order after 0560; they touch
no weight-preparing module, so the prepared folders are reused). Lock clocks at 2,250 MHz for steps 1-3 (as W16), and
unlock before the loads. Drop the page cache before each server start (W16's C lesson).

1. **CPU suites in the image** (both nodes, 6 min): `tests/test_decode_loads_emulator.py`,
   `tests/test_decode_loads_compile.py` (NVCC=/usr/local/cuda/bin/nvcc; prints the register / CTA / in-flight table and
   the SASS line), `tests/test_decode_size_switch.py`, `tests/test_decode_kernels_emulator.py`,
   `tests/test_theory2_probes.py`. Gate: all pass.
2. **GPU bitwise** (head, ~12 min): `PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s
   tests/cuda/test_decode_loads_patches.py` (0580 on == off at real / synthetic shapes, every setting, graphs, rows
   alone, raw Z, probes, the PDL late-writer race, 0570 == `_qmm` for 15 shapes x 19 row counts x bf16 / fp32 x PDL).
   Then the engine suites with both knobs: `GLM53_TF_DEC_EXPERT_LOADS=1 GLM53_TF_DEC_QMM_MAXMB=3.5 pytest -q
   tests/cuda/test_decode_patches.py tests/cuda/test_batch_parallel_patches.py tests/cuda/test_decode_stream_patches.py
   -k "windows or replies or resume"` (0440's engine checks: windows == off, drafted == serial, resumed == fresh), and
   once more with `GLM53_TF_DEC_EXPERT_LOADS_PDL=1`. Gate: all pass; any bit difference stops the window.
3. **Microbench** (worker in parallel with step 2, ~15 min):
   - `python tests/cuda/bench_decode_kernels.py --loads --json loads.json`: windows R 1 / 2 / 3 / 4 (U 8 / 13 / 17 /
     22), 8, 16, 4-slot 3+3+3+2 and 4x4; every setting + the default with PDL; probes; roof; rotate and flush.
     - `GATE 0580 load path: probe 3 (nc,8,2, no decode / mma) >= 220 GB/s [flush] on every U 8-22 window`. **If it
       fails, stop**: the load path itself does not reach the bandwidth (compare the w32 / cpa probe 3 and the roof
       column; that is a finding about the mechanism in this kernel's CTA shape, not about the ALU).
     - `GATE 0580: <cfg> >= 1.05x grouped_kernel [flush] (and >= 1.00x [rotate]) on every U 8-22 window`. Pass: that
       cfg is the one the loads use (`GLM53_TF_DEC_EXPERT_LOADS_CFG`, PDL if the `pdl` row won). Fail with the probe
       passing: the per-step overhead eats it; record probe 1 / 2 (decode vs mma) and stop.
   - `python tests/cuda/bench_decode_cold.py --mode both --no-hot --shapes 1024x4096 4096x128 160x4096 4096x1024
     4096x1536 8192x512 --json cold570.json` (~3 min): confirms SMALL_PLACE on this image and measures the (1,16,8)
     grid; if 1024x4096 is < 1.0x cold in either mode, add `GLM53_TF_DEC_QMM_EXCLUDE=1024x4096` to the loads.
4. **Server loads** (W16's "full set" each, from `config/prod.env` on b10; ~10 min a load): **C** (knobs off: must
   equal prod b9's bits and speed), **E** (`GLM53_TF_DEC_EXPERT_LOADS=1` + the step-3 cfg), **EQ** (E +
   `GLM53_TF_DEC_QMM_MAXMB=3.5`), **EQP** (EQ + `GLM53_TF_DEC_EXPERT_LOADS_PDL=1` + `GLM53_TF_DEC_PDL=1`, only if the
   PDL rows won in step 3). Each: exact 10/10, batchexact 4/4, W9 transcripts, reply sha 8794a3463259cc2f, glmbench
   tf / kit / edit x3 hashes == C, 4 streams x6, lone slots. Gates: bits equal everywhere; **E: 1-stream geomean >=
   +2.0% vs C and 4 streams >= -0.5%** (W16's noise band is ±1%, so +2% is the smallest claim this A/B can make);
   EQ >= E.
5. **One short nsys capture of E** (THEORY-2 §7 rule 4; W11's `load.sh nsys` + `cap.sh`, prose only, ~5 min): the
   in-situ grouped GB/s by window size (w11dec.py) must show the ld_kernel rows at >= 215 GB/s for U 8-22; the 0570
   shapes' per-call µs vs W11's table decide Q on its own (adopt if every switched shape is faster in situ).
6. **Adoption gates** for whatever passed: MMLU-200 88.0%, 4 x 250k stress (no new buffers: 0580 allocates nothing,
   0570 only 0440's 32 KB of tickets a partial buffer), needle 314k. Update `config/prod.env` with the header block and
   the revert line (drop the knobs = b9's behaviour on b10).

Stop rules: any bit difference; probe gate fail; a server load that serves fewer than 4 slots (fix, reload, do not
compare). Revert: the knobs off (b10 with no knob == b9 + two unused modules).

## 8. Files

| file | what |
| --- | --- |
| `patches/0570-glm-dense-size-switch.patch` | `decode_stream.py`: `GLM53_TF_DEC_QMM_MAXMB` / `_TABLE` / `_EXCLUDE`, `SMALL_PLACE`, `small_shape`, the dispatch in `matmul`, `configure(qmm_max_mb=, qmm_table=)` |
| `patches/0580-glm-expert-loads.patch` | `exl3_ld.cu` / `exl3_ld.cpp` (ld_kernel), `expert_loads.py` (knobs, `fits`, `run`, the hook), `exl3_mm.py` (`LOADS` hook in the grouped call), `forward.py` (import), `sessdisk.py` (compat hash skips the knob) |
| `tests/decode_loads_emu.py`, `tests/test_decode_loads_emulator.py` | 0580 CPU emulator + host tests (§3) |
| `tests/test_decode_loads_compile.py` | 0580 sm_121 compile, PTX, PDL prologue, verbatim helpers, SASS |
| `tests/test_decode_size_switch.py` | 0570 CPU tests |
| `tests/cuda/test_decode_loads_patches.py` | GPU bitwise + PDL race for 0580, 0570 == `_qmm` |
| `tests/cuda/bench_decode_kernels.py` (`--loads`) | the 0580 microbench and its two GATE lines (`gate_0580`) |
| `tests/cuda/bench_decode_cold.py` | + the 1024 x 4096 shape (0570's unmeasured grid) |
| `tests/test_decode_kernels_emulator.py`, `tests/test_theory2_probes.py` | updated for 0570's new CFG keys / the extra cold shape |
