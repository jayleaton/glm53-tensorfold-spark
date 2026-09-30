# Routed experts in prefill, v2: patch 0590 (`GLM53_TF_FAST_EXPERTS=fat2`)

> **Update 2026-09-30 (W19, docs/RESULTS.md): not adopted.** Same bits on the GPU, but fat2 is slower than `fat`:
> 1.16x / 1.19x isolated, makespan 1.07x / 1.09x at 2,048 / 4,096 rows (1.76x / 1.88x the DRAM floor). Production
> keeps `fat`. The estimates below are the offline ones.


Written offline (2026-09-30). No GPU was used, the Sparks were not touched and production was not changed. The kernel
compiles for sm_121 and its bits are proven equal to production's `fat` kernel on the CPU (emulator, PTX and SASS
checks). **Nothing here has been timed.** The GPU plan is in section 7.

## Short version

- **What.** `exl3_fat2.cu` is a new routed-expert kernel for fast-prefill chunks. It is built on the structure of the
  fused prefill MoE kernel in docs/KINDLING-AUDIT.md (adoption #2). Ideas were re-implemented; no code was taken (the
  kindling repository has no licence).
  - One persistent 256-thread CTA an SM.
  - 128-member items of the expert-sorted member lists, from a device-side plan, claimed by ticket one item ahead.
  - A 4-stage `cp.async` ring that runs **across items**, with the member rows gathered by pair index inside the loads.
  - The epilogue in its own shared buffer.
- **Bits.** It is bit-identical to `fat` by construction and by proof:
  - the same m16n8k16 chain per output, the same K order and the same epilogue;
  - checked by a lane-level emulator with an order-sensitive and a position-sensitive mma model (12 planted mutations
    caught);
  - checked by a PTX dataflow comparison of every stored value and every mma operand (3 planted source changes
    caught);
  - checked by a SASS census of the epilogue arithmetic.
  - It stays row-independent (C-independent prefill) and shares snapshots with `fat`.
- **Resources (sm_121, nvcc 13.4).** Every configuration:
  - 168 registers (`__maxnreg__`), 0 spills;
  - 82.0 KB of shared memory a CTA (65.5 / 70.0 KB for the other two configurations), one CTA an SM;
  - 22.5 K registers and 16 KB left on each SM for a CTA of the 0084 overlap stream. fat's gate/up fills the SM's
    register file (2 CTAs x 256 x 121).
- **Expected speed.**
  - **2,048 and 4,096 rows: ~0.77-0.86x fat.** At these sizes the kernel is DRAM-bound.
  - **8,192 rows: ~0.65-0.75x fat.** There it is MMA-bound.
- **The gate as set (<= 0.75x fat at 2,048 and 4,096, contended) sits on the DRAM floor.** Any kernel with fat's
  dataflow (Xg in, fp32 Y out) is bounded by that floor. At 235 GB/s the floor is 0.737x / 0.734x of W8's isolated fat
  (9.71 / 11.71 ms against 13.18 / 15.96). Passing therefore needs about 98% of the measured DRAM peak. **Expect FAIL
  at 2,048, likely FAIL at 4,096, and a pass at 8,192.** Section 6 proposes a floor-based gate for the user to decide.
- **Combine fusion (adoption #6): not done in 0590.** It cannot be made bit-identical inside the expert kernel without
  cross-CTA ordering, and it would save little (section 5). The audit's form of it is a change to `_combine_s` and the
  send buffer, independent of this kernel.
- **End to end (estimate).** Prefill runs 4,096-row lone chunks and 2,048-row batch pieces today.
  - Kernel: -23 to -39 us a token of 621, which would be **+4 to +7%**.
  - After the W5 / W8 transfer discount (isolated kernel wins have transferred at a half or less): **+2 to +4%**, i.e.
    ~1,610 -> ~1,640-1,680 tok/s.
  - With 8,192-row lone chunks (audit item 3) the kernel gain roughly doubles.

## 1. Why fat is slow, and what the audit changes

**Where fat stands (ROOFLINE gap 1; EXPERT-TC.md section 1; RESULTS W2 / W8).** Per rank and MoE layer (288 experts,
top 8, hidden 4,096, 1,024 of each expert's 2,048 width):

| rows | DRAM floor, 235 GB/s | MMA floor, 110 TF/s | fat, W8 isolated / contended (ms) | fat / DRAM floor |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 9.71 ms (2.28 GB) | 3.7 ms | 13.18 / 15.24 | 1.36x |
| 4,096 | 11.71 ms (2.75 GB) | 7.5 ms | 15.96 / 18.10 | 1.36x |
| 8,192 | 15.71 ms (3.69 GB) | 15.0 ms | 27.70 / 28.79 | 1.76x |

The floor counts:

- the weights once (1.81 GB);
- Xg read (P x 8 KB);
- Xd written and read (P x 2 KB each way);
- Y written (P x 16 KB fp32);

with P = 8 x rows routed pairs (`bench_experts._floor_ms`).

**What goes wrong in fat** (from its code, `exl3_fast.cu` namespace `fat`):

1. **The pipeline drains at every item.**
   - An item is (expert, 64-member pass, 128-column block).
   - Each item starts with a 2-stage prologue from nothing and ends with an epilogue while no loads are in flight.
   - At 2,048 rows a down item is only 16 stages (K = 1,024, 4 k tiles a stage). The drain-and-refill is a large part
     of its time.
   - This is ROOFLINE's "sum of the roofs, not the max": MMA and memory run one after the other.
2. **64-member items.**
   - Each trellis tile is decoded and each weight tile fetched once per 64 members.
   - At 4,096 rows (~114 members an expert) that is twice; at 8,192 rows it is 4x (gate/up).
3. **Small register tiles.** Each decoded A fragment feeds 8 mma, and each `ldmatrix.x4` feeds 4.
4. **Two CTAs of 8 warps at 121 registers fill the SM's register file** (gate/up). Nothing from the overlap stream can
   co-reside until a persistent CTA exits.

**What the audited kernel does and what 0590 takes from it:**

| kindling's prefill MoE (NVFP4, idea only) | fat2 (EXL3 x fp16, fat's bits) |
| --- | --- |
| 256 threads, 8 warps | the same: 8 warps, one CTA an SM, `__maxnreg__(168)` (a 17-warp CTA does not fit GB10: 0330 / W8) |
| 128 x 128 tiles | 128 members x (128 columns x 2 matrices) for gate/up; 128 members x 2 column blocks for down. The column block must stay one 128-wide Hadamard block, because the epilogue transforms it |
| 4-stage `cp.async` ring, XOR swizzle, `ldmatrix` | 4 stages of 2 k tiles (12 KB: 8 KB of member rows, 4 KB of trellis words), fat's swizzle (4 chunks a row), `ldmatrix.x4` |
| device list of (expert, m-tile) over expert-sorted rows; padded entries exit | fast2's device plan (per-expert pass counts, prefix sum) over glue's expert-sorted member lists; items by ticket, no empty programs |
| token gather inside the A loads | member rows gathered by pair index inside the `cp.async` loads (as fat). The input rotation (`rot_in1`: x -> Xg) stays a separate kernel, because the Hadamard is 128-k wide and per expert: see below |
| SwiGLU + requant in the fc1 epilogue | fat's gate/up epilogue already does SwiGLU (+ the down input's rotation) in registers |
| (not in theirs) | **the ring continues across items**: the next item's first 3 stages load under the current item's last stages and its epilogue; the epilogue has its own buffer |
| optional L2 prefetch of the next tile | not needed: the cross-item ring already has the next item's bytes in flight |

**Why the input rotation stays in `rot_in1`.**

- Folding it into the A loads (x rows gathered by token, then sign vector, 128-point Hadamard, fp16 in shared memory)
  would save the Xg round trip: -0.12 GB at 2,048 rows, -0.24 GB at 4,096, about 0.5 / 1.0 ms a layer. Bits could be
  kept by reusing `rot_in1`'s code; its fma contractions would need the same PTX checks.
- But the transform needs the whole 128-k block of a row. With 128 members that is a 32 KB bf16 staging buffer a stage.
  Three stages of that plus the words and the epilogue do not fit in 99 KB.
- The rotation would also be recomputed for every column block: 8x for gate/up.
- It is a candidate for a v3 with 64-member items (section 8), not for 0590.

## 2. The kernel (`exl3_fat2.cu`)

- **Configurations** (`GLM53_TF_FAT2_CFG="gu,dn"`; all give the same bits):

  | cfg | members an item | stages | shared memory a CTA | registers | spills |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 0 (default) | 128 | 4 | 82.0 KB | 168 | 0 |
  | 1 | 64 | 4 | 65.5 KB | 168 | 0 |
  | 2 | 128 | 3 | 70.0 KB | 168 | 0 |

  Probes: 1 = no decode, 168 registers; 2 = no mma, 96-102 registers. One CTA an SM in every configuration. The SM
  has 22,528 registers and 16-32 KB left beside it.

- **Warps.**
  - gate/up: warp = (matrix h in gate / up, 32-column slice) of the item's 128-column block.
  - down: warp = (column block 2q + h, slice). An odd last block leaves its h = 1 warps idle.
  - A warp holds 2 column tiles x 128 members (128 fp32 accumulators). A decoded tile feeds 16 mma and an
    `ldmatrix.x4` feeds 4.
- **A stage** (2 k tiles):
  - the members' rows, only up to the last member's 16-row group (zero-filled past the last member inside it);
  - the item's trellis words of both halves;
  - one `cp.async` group a stage and one `__syncthreads` a stage, as fat.
  - Member groups past the count are skipped (warp-uniform).
- **Across items.**
  - Thread 0 claims the next item (ticket) NSA stages before the current item ends, into the second of two item
    buffers (rows, count, expert).
  - The last NSA - 1 iterations of the current item load the next item's stages 0 .. NSA - 2 into the ring slots
    freed behind them.
  - After the epilogue, the next item starts computing at once.
  - Every barrier is block-uniform. There are no mbarriers or spin-waits, so there is none of 0330's hang risk.
- **Epilogue.** fat's code, in rounds of 32 rows through a separate 33 KB buffer:
  - accumulators -> `ep[half][row][col]`;
  - then one warp a row: `fwht_row`, the scales, the limited SwiGLU, `fwht_row` and fp16 (gate/up), or the scaled
    fp32 row (down).
- **Knobs.** `GLM53_TF_FAST_EXPERTS=fat2` (default off), `GLM53_TF_FAT2_CFG`, `GLM53_TF_FAT2_TICKET` (0 = static
  stride), `GLM53_TF_FAT2_CTAS` (a CTA cap for the overlap stream).
- **Scope and fallbacks.**
  - It runs inside the fat family: the per-request `fat_experts = 1` selects it, 0 gives fast2, 2 gives auto.
  - It needs gate / up sharing their sign vector (true for the checkpoint) and K >= 128; otherwise fat runs.
  - Own extension (`tensorfold_glm_exl3_fat2_v1`), built at first use for the device's architecture only.
  - It picks nothing by the call's rows, so 0560's multi-slot groups stay allowed (`mpf.kernels_ok`).

## 3. Exactness and how it is proven offline

**Claim.** Every stored element (Xd of gate/up, Y of down) is computed by the same sequence of operations as in fat:

- the same m16n8k16 mma with the same A fragment (`decode_tile` of the same word, verbatim) and the same B values
  (the same member row and k positions through `ldmatrix`);
- the same accumulator from +0.0, over ascending k tiles, all of K in one warp;
- the same tile position: column tile row c % 16, member column m % 8, because 128 and 64 are multiples of 8;
- then fat's epilogue.

What changes only moves data: the CTA, warp and item that hold an element, and when bytes arrive. So:

- fat2 == fat == fast2 == v1;
- row-independent (patches/0085);
- deterministic (the ticket only picks the CTA).

**Proof, offline** (all green; stack = b9's list + 0590, also applied on b9 + 0570 + 0580):

| test | what | result |
| --- | --- | --- |
| `tests/test_fat2_emulator.py` (+ `tests/fat2_emu.py`) | Lane-level CPU ports of **fat** and **fat2**, line for line: the plan, ticket and static walks under random CTA interleavings; the NaN-initialised ring with `cp.async` copies landing at random between issue and their `wait_group`, and random landings between warps; swizzle, `ldmatrix.x4`, `decode_tile`, fragments; fat2's claim-ahead and cross-item stages; the epilogue rounds. The mma is an **order-sensitive** model: the products are added in k order with fp32 rounding each step, so a whole chain is the sequential sum over K. Checked against a matrix-level reference written from fat's documented arithmetic. The runs cover: fat == reference (3 / 4 stages, ticket / stride, >= 4,096-row down); fat2 == reference with every output written exactly once (3 configurations, 1-7 CTAs, ticket / stride, 1-4,096-row windows with 1-43 passes an expert, odd column-block counts, stale grouping entries, the real per-rank shapes); fat2 == fat under a **position-sensitive** mma model (an element's summation order depends on its place in the tile); rows alone == rows in the window; **12 planted mutations caught** (ring slot, wait depth, swizzle, k order within a stage, the next item's rows / stage / claim timing, accumulators kept, a member group skipped, epilogue rows swapped, down's column block, a lost ticket); one neutral mutation (no zero-fill past the last member) keeps the bits, as the model says it must | 49 passed, 8.5 min |
| `tests/test_fat2_compile.py` (+ `tests/fat2_ptx.py`) | nvcc 13.4 for sm_121: the resources above. The helpers are exl3_fast.cu's text (11 bodies), and the epilogue's arithmetic lines are fat's. **PTX dataflow:** every value fat2 stores has the expression tree of a value fat stores (gate/up vs fat's gate/up; down vs fat's down and down-large). The trees are hashed with opcodes, rounding / approximation modifiers, constants and operand order, and selects are normalised for `setp.ne` / `setp.eq`. They reach `ex2.approx.ftz`, `div.rn`, `fma.rm/rn`, `cvt.rn.bf16`, `cvt.rn.f16` and `shfl.bfly`, and never an undefined register. **PTX mma:** one form, C == D registers, A = the decode tree of one shared word, B = `ldmatrix` registers (0, 1) / (2, 3), the same set as fat's. **Atomics:** only the ticket (`atom.global.add`) and the member count (`atom.shared.max`). **SASS census** of the epilogue FP instructions (FADD / FMUL / FFMA.* / MUFU.EX2 / F2F / FMNMX / FSEL / FCHK / HADD2.F32): fat2 with 2 epilogue rounds == fat's gate/up and down-large exactly; 4 rounds == 2 rounds + 2 of fat's per-round delta, so ptxas made the same contractions at fat2's register cap. No local memory. **3 planted source changes caught** (an extra rounding in the gate scale, an extra multiply before down's output, the two B registers of an mma swapped) | 24 passed (with `test_fat2_bench.py`), 17 s |
| `exl3_fat2.cu` (host + device) and `exl3_fat2.cpp` | compile against torch 2.14's headers (C++20, CUDA stub headers); the exported entry points match the bindings | ok |
| `tests/cuda/test_fat2_patches.py`, host part | the env switch and refusals; `routed`'s dispatch with fake extensions (fat2 on the shared input, fat when gate / up differ, fast2 for knob 0 or auto's window); `mpf.kernels_ok()` stays true | 6 passed (21 GPU tests skipped) |
| regressions (host) | `test_expert_tc_patches`, `test_fast_experts_auto_patches`, `test_expert_once_patches`, `test_multi_prefill_patches`, `test_lean_patches`, `test_solo_piece_patches` on the patched tree | all pass except 3 in `test_solo_piece_patches`, which fail identically on b9 without 0590 (stale expectations) |

**Two limits of the offline proof, both checked on the GPU by the plan:**

- The emulator assumes the tensor core computes an element from its A row, B column and accumulator only. The
  position-sensitive run removes the "same order everywhere" assumption, but not that one.
- The PTX / SASS checks cover the arithmetic, not addressing. Addressing is the emulator's job.

## 4. Compile report (sm_121, `ptxas -v`)

| kernel | registers | spills | static smem | dynamic smem | CTAs an SM | left on the SM |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fat2 gate/up / down, cfg 0 | 168 | 0 | 1,056 B | 82,944 B | 1 | 22,528 registers, 16 KB |
| fat2 cfg 1 (64 members) | 168 | 0 | 544 B | 66,560 B | 1 | 22,528 registers, 32.5 KB |
| fat2 cfg 2 (3 stages) | 168 | 0 | 1,056 B | 70,656 B | 1 | 22,528 registers, 28 KB |
| fat gate/up (production) | 121 | 0 | 272 B | 49,152 B | 2 | ~3.5 K registers |
| fat down / down-large (4 warps) | 171 / 252 | 0 | 272 / 528 B | 36,864 / 33,792 B | 2 / 1 | - |

- **Caps.** Without the cap, ptxas uses 240-255 registers and spills 4-12 bytes (down, and gate/up with 3 stages). 168
  is the largest cap tried with no spill in any configuration; 200 spills.
- **Main loop (SASS, cfg 0 gate/up).** 64 HMMA a stage in ~290 instructions, against fat's 128 HMMA in ~1,190: about
  half the non-MMA instructions an mma.

## 5. The combine (adoption #6), and why it is not in 0590

`_combine_s` adds each row's 8 routed slots (weighted, in slot order) and the shared expert in fp32, then rounds once.
An item of the expert kernel holds one expert's members, so a row's other 7 slots are computed by other CTAs at other
times. The options inside the expert kernel:

1. **fp32 atomics into the row.** The order depends on scheduling: not bit-identical, not even deterministic.
   Rejected.
2. **"Last writer combines."**
   - A per-(row, column block) arrival counter. The CTA that brings it to 8 reads the other 7 slots' Y and sums in
     slot order.
   - This can be bit-identical, but the 7 other slots must still be written in fp32 and read back (from L2 at best:
     in the expert-sorted order a row's slots finish milliseconds apart, across the whole kernel).
   - It saves at most 1/8 of Y's write (~0.03-0.07 GB a layer) and one launch. It costs fences and counters in a
     kernel whose value is its streaming.
   - Not worth it.

The audit's item #6 is a different and exact change: `_combine_s` writes its bf16 partial rows, with the shared-expert
add, straight into 0084's all-gather send buffer, and starts the exchange per slab. It lives in the glue / overlap
code, needs nothing from the expert kernel, and is estimated at +1-2%. It should be its own patch.

## 6. Performance model, expectations, and the gate

**Per SM and k tile** (gate/up, 128 members):

- MMA: 32 mma a warp x 8 warps x 16 clocks / 4 SMSPs = 1,024 tensor clocks.
- DRAM: 2 KB of trellis words, plus 4 KB of member rows (mostly L2 hits: the 8 column blocks of a pass run together).
  Each SM's share of 235 GB/s is ~2 B a clock, so that is ~1,000-1,250 clocks.

So from 4,096 rows up the two roofs are about equal, and only overlap helps. 0590 overlaps them:

- 3 stages (36 KB) in flight an SM, 1.7 MB across the GPU. Little's law needs ~0.5 MB for 235 GB/s at ~2 us of loaded
  latency.
- The flow continues through item boundaries and epilogues.

**Expected times** (gate/up + down, uniform routing, isolated):

| rows | floor (max of DRAM @235, MMA @110) | fat (W8) | fat2 expected | fat2 / fat | 0.75x gate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 9.71 (DRAM) | 13.18 | 10.2-11.4 (85-95% of DRAM peak) | 0.77-0.87 | 9.89: 98% of the DRAM peak |
| 4,096 | 11.71 (DRAM) | 15.96 | 12.3-13.8 | 0.77-0.86 | 11.97: 98% |
| 8,192 | 15.71 (DRAM; MMA 15.0) | 27.70 | 18-21 (MMA at 75-85%, overlapped) | 0.65-0.76 | 20.8 |

- **Contended.** The side stream's copies take DRAM bandwidth from both kernels. fat lost 14-16% there in W8. fat2
  leaves room on each SM for the side stream's CTAs, so its own contended time may lose more while the makespan (both
  streams) gains. `bench_experts.py --fat2 --contend` now prints both.
- **The gate on the letter.** "Same bits and <= 0.75x fat at 2,048 and 4,096, contended" needs ~98% of the DRAM peak
  with fat's dataflow. **Expect FAIL at 2,048, likely FAIL at 4,096.**
- **Proposal (decision before the GPU run).**
  - Keep the letter gate printed.
  - Also judge fat2 by the floor: contended at <= 1.12x the DRAM floor (>= 89% of peak) at 4,096 rows, makespan not
    worse than fat's, and bits same.
  - Then let the end-to-end A/B decide: >= +2% prefill, same reply sha, decode not lower.
  - The only ways under the floor change the dataflow: fold the rotation in (v3, -4-8% bytes) or bf16 Y (new bits,
    ROOFLINE gap 1b).

## 7. GPU plan (one window, production down ~60-75 min)

Build image `b10` = `b9`'s patch list + 0590 (plus 0570 / 0580 only if they were adopted by then; 0590 applies on
either). Keep `b9` for the revert. Run everything on the head node under a lease with production stopped. Every command runs
under `timeout`.

1. **Correctness (~10 min).**
   `PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda timeout 900 pytest -q -x -s tests/cuda/test_fat2_patches.py -k "not engine"`
   - Covers: bits against fat and fast2 at 64-8,192 rows, uniform / skewed, cfgs 0-2, ticket on / off, CTA caps 1 / 7
     / 40; odd shapes; row subsets and permutations; repeatability; the probes launch.
   - Gate: all green. A timeout means a pipeline bug: stop. (The design has no spin-waits, so a hang is not expected.)
2. **Kernel bench (~15 min).**
   `timeout 2400 python tests/cuda/bench_experts.py 2048 4096 8192 --fat2 --contend --no-v1 --variants fast2,fat,fat2`
   - Read: `fat2` vs `fat s3`, isolated and contended, with the DRAM floor column and the makespans. Read
     `fat2 static` vs `fat2` under contention (the ticket's value). Read `fat2 ctas 44 / 40` (room for the overlap
     stream) and cfg 1 / 2.
   - Read the probes. "no mma" near the full time means DRAM-bound (expected at <= 4,096); "no decode" near the full
     time means the decode is hidden.
   - The **GATE** line prints the letter gate.
   - Continue if bits are "same" and either the letter gate passes or the floor gate of section 6 passes. Otherwise
     stop, record, and keep the patch off.
3. **Engine tests (~15 min).**
   - `timeout 1800 pytest -q tests/cuda/test_fat2_patches.py -k engine`: state fat2 == fat, drafted == serial,
     resumed == fresh, snapshots shared both ways.
   - Regressions: `test_mia_prefill_patches.py`, `test_multi_prefill_patches.py -k "gpu or engine"`,
     `test_solo_piece_patches.py`.
4. **Load A: production (b10, knob off) as the control.**
   - `results/W5/ab.py` 24.5k / 98k x2: reply sha `8794a3463259cc2f`, cold prefill tok/s.
   - `--suites exact` 10/10 and batchexact 4/4.
   - `multiturn.py --modes concurrent --streams 4 --reps 3`: batch pieces at 2,048 rows, decode aggregate.
5. **Load B: `GLM53_TF_FAST_EXPERTS=fat2`**, everything else production. The same runs as load A, plus:
   - `GLM53_TF_PROFILE=1` for one 24.5k run: `moe.routed` per chunk against load A;
   - one nsys capture of a 24.5k prefill (method: PROFILE.md) to see the expert kernel beside the overlap stream's
     all-gathers and hc slabs, and whether side-stream CTAs co-reside.
   - If the bench preferred it, a load B' with `GLM53_TF_FAT2_CTAS=44` or `GLM53_TF_FAT2_CFG=1,1`.
6. **Adopt** if all of these hold:
   - same reply sha and exact / batchexact green;
   - prefill >= +2% at 24.5k and at 98k (two runs each, beyond W17's run-to-run spread);
   - 4-stream decode not lower.
   - Adoption is `GLM53_TF_FAST_EXPERTS=fat2` in `config/prod.env`, with the same snapshots (no tag change).
7. **Revert** to `config/prod.env` (b9 or b10 with the knob off), run the canary, check https, re-arm the watchdog,
   delete the lease.

## 8. Risks and next steps

- **The W5 effect.** Two earlier kernels won in isolation and lost in the engine (fast2 -2%, tc cfg 3 -4%). The ticket,
  the room left beside each CTA and the makespan metric address the likely causes. Only load B decides.
- **168 registers / 8 warps an SM** is thin for latency hiding. If the "no decode" probe shows the decode exposed, or
  the MMA rate is low at 8,192 rows, try cfg 1 (64 members), or a variant with KS = 1 and 6 stages. Both are template
  constants; the emulator and compile tests take new configurations with one line each.
- **16 KB of shared memory left an SM** may not fit an overlap-stream CTA that wants more. cfg 1 / 2 leave 28-32 KB.
- **v3 (not written):** 64-member items with the input rotation folded into the A loads (rows gathered by token, the
  128-k transform in shared memory, `rot_in1`'s arithmetic under the same PTX checks). It removes the Xg round trip
  (-0.12 / -0.24 GB a layer) and `rot_in1`'s ~15 us a token. It is the only same-bits way below today's DRAM floor.

## Files

- `patches/0590-glm-expert-prefill-v2.patch`: `exl3_fat2.cu`, `exl3_fat2.cpp`, `exl3_mm.py`.
- Offline proof: `tests/fat2_emu.py`, `tests/test_fat2_emulator.py`, `tests/fat2_ptx.py`, `tests/test_fat2_compile.py`,
  `tests/test_fat2_bench.py`.
- GPU: `tests/cuda/test_fat2_patches.py`, `tests/cuda/bench_experts.py` (`--fat2`, the makespan, the GATE line).
