# What the kindlingai GX10 vLLM recipe does for prefill that we could use

This audit covers [kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10) (formerly
`mmastrac/glm-5.3-flash-4x-gx10`; GitHub redirects the old name). We read main @ `45b438b` (2026-09-29):

- README, NOTES, KNOBS, `model.yaml`;
- `image/entrypoint.sh` and the image patches;
- every `experimental/` overlay and its README;
- the TP=3 / TP=6 notes;
- the benchmark scripts (`gate/`, `dev/repro/`);
- the prefill-relevant source: `megamoe/moe_prefill.cu`, `fixes/model.py` (SP), `fixes/gb10_sparse_mla.py`,
  `fixes/kda.py`, `snapshot/dense_fp8.py`;
- the git history for when the numbers were taken.

Credit: the ideas this project took from it (0590's pipelined routed-expert kernel, both CX7 functions in NCCL, see
section 4 and docs/EXPERT-PREFILL-V2.md) are kindlingai's. **Their repository has no licence, and we copied no code
from it:** everything adopted here is our own re-implementation from the ideas (section 8).

Offline only: no GPU was used, the Sparks were not touched, and nothing was posted. "Receipt" means a number quoted
from their repo; we did not re-measure any of them. "Gain for us" is our estimate against today's production (W17:
~621 us a token a rank = 1,606-1,610 tok/s at 24.5k / 98k).

## Short version

- **Their TP=2 prefill is 1.8x ours: 2,929 tok/s cold at 32k against our 1,634 (RigMark 32K).** That is ~341 us a
  token a rank against our ~621. It is also 1.54x Alex Ellis's stock-ish vLLM NVFP4 TP=2 receipt (1,908).
- **Setup.** Same GB10 chip (ASUS Ascent GX10). vLLM nightly `ddd6fbca` plus nine overlays of their own kernels and
  patches. Model: `nvidia/GLM-5.3-Flash-NVFP4`.
- **About 60% of their per-token advantage comes from number formats our constraints rule out:**
  - routed experts run **W4A4**: NVFP4 weights and NVFP4 activations, on block-scaled FP4 `mma.sync`;
  - the dense projections (KDA in_proj, shared expert, DSA, o_proj) run **FP8 W8A8** in prefill;
  - the SP gathers move FP8 / NVFP4 activations;
  - the MoE intermediate is stored as FP8 (`Y8`).
- **The largest adoptable lesson is about the MoE kernel, not the format.** Their fused prefill MoE reaches ~90
  TFLOP/s effective. That is only ~21% of the FP4 peak, but ~84% of the bf16 `mma.sync` roof we can use. So our
  routed-expert gap (~193 us a token today) is mostly kernel structure, not the format. Their structure:
  - a 4-stage `cp.async` ring with XOR-swizzled shared memory and `ldmatrix`;
  - 8 warps on 128x128 tiles;
  - a device-side (expert, m-tile) list over expert-sorted rows;
  - the token gather folded into the A loads;
  - SwiGLU and requantization in the fc1 epilogue.
  All of this is portable in idea to an EXL3 x bf16 kernel with our bits.
- **Cheapest exact item: NCCL on both CX7 PCIe functions.** Their receipt: all-reduce 110 -> 190 Gb/s, TP=4
  prefill +11%. Our 4 MiB prefill all-gathers still run over one function (`NCCL_IB_HCA=rocep1s0f1`) at ~9 GB/s
  `RING_LL`. The second function (`roceP2p1s0f1`) is already addressed for 0350. An all-gather is a copy, so this
  is exact by construction. Expected +1-3% for us. It is config only and needs about an hour of GPU time.
- **Adoptable at our exactness bar** (same bits or deterministic, row- and C-independent):
  - the expert-kernel rebuild: +7-13%;
  - 8,192-row lone chunks under a memory rule: +4%, measured in W10;
  - dual-root NCCL: +1-3%;
  - hc boundary fusion in prefill: +1.5-2.5%;
  - combine written straight into the send buffer: +1-2%;
  - a better q4 GEMM for the projections (ROOFLINE gap 3; their answer is FP8): +7-9%.
  Stacked: **~2,100-2,200 tok/s**.
- **Precision changes, listed for the record** (all behind the quality gate, and each a user decision):
  - FlashKDA (MIT) for the KDA recurrence: bf16 operands instead of our tf32;
  - a bf16 MoE intermediate `Y`;
  - `LATENT_TC`.
  These add another ~60-80 us a token, to ~2,450-2,550 tok/s. The rest of their lead is FP4 / FP8 arithmetic.
- **Licence: the repo has no licence** (GitHub reports `license: null`; there is no LICENSE file). Their own
  kernels (`megamoe/`, `arx/`, `gb10_sparse_mla.py`, `dense_fp8.py`, ...) are all-rights-reserved by default. Copy
  no code: re-implement from the ideas. Their vLLM / FlashInfer-derived files keep Apache-2.0 headers, the MiaAI
  patch is MIT, and FlashKDA (`vllm-project/FlashKDA`) is MIT.

## 1. Their setup and numbers, against ours

### 1.1 What they run

| | kindling GX10 recipe | ours (TensorFold prod, W17) |
| --- | --- | --- |
| Hardware | ASUS Ascent GX10 (GB10, sm_121a, 128 GB) x 2, 3, 4 or 6; TP=4 is their default | DGX Spark (GB10) x 2 |
| Fabric | MikroTik CRS812 200G switch. **Both ConnectX-7 PCIe roots** in NCCL (`FABRIC_SUBNETS`, `NCCL_MAX_NCHANNELS=8`). Own RDMA collectives: `arx` for ≤256 KB all-reduce, `arxbig` for the prefill reduce-scatter | Direct QSFP cable. NCCL on **one** function (`rocep1s0f1`) for the prefill all-gathers. 0350 RoCE on both functions for ≤256 KiB |
| Parallelism | TP, plus sequence parallelism for forwards ≥1,024 tokens | TP=2, plus the 0320 row split for hc in prefill |
| Checkpoint | `nvidia/GLM-5.3-Flash-NVFP4` (ModelOpt). Routed experts NVFP4, **activations NVFP4 too** (FlashInfer CUTLASS W4A4). The checkpoint's bf16 dense layers are converted after load to **FP8 W8A8** (per-channel / per-token), with NVFP4 W4A16 copies for decode (in_proj, o_proj, shared experts, drafter) | `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` (abliterated). EXL3 4.0 bpw experts, q4mse non-experts, **bf16 activations**, fp32 accumulation |
| Engine | vLLM nightly `ddd6fbca`, mentat instead of Ray, 9 compose overlays (`experimental/`), FlashKDA `17a037d`, torch.compile, CUDA graphs ≤ decode sizes only | TensorFold `2f8e514` + our patches (image b9) |
| KV / context | fp8_e4m3, block 2304. TP=2: 8 GiB pin, 875k-1.10M tokens, **160k per request** | FP8 latent, **1,048,576 per request**, 4 slots, shared pool |
| Scheduler | `max_num_batched_tokens` 8,192 at TP=2 (16,384 at TP=4), `long_prefill_token_threshold` 2,304 | 4,096-row lone chunks (`SOLO_PIECE` / `PREFILL_ROWS_MAX`), 2,048 in batches |
| Memory posture | ~91.9 GiB GPU per rank. At TP=2 the head is **1.3 GiB free at its lowest** | ≥ ~7-8 GiB MemAvailable floor, gated |
| Exactness | none claimed. Greedy at 42k gives 3 distinct completions in 16 runs (`dev/repro/greedy_nondet.py`) | drafted == serial, resume == fresh, batched == alone, C-independent prefill, all byte-exact |

The per-step budget matters to them at TP=2. Going from 16,384 to 8,192 took 32k from 3,201 to 2,929 tok/s
(commit `6db84d4`: "~9% slower"), and at TP=3 8,192 cost 4% at 22k. So a lone prompt's step is the budget, not the
2,304 threshold. We did not verify how the nightly scheduler applies the threshold to a single request.

### 1.2 Their numbers and how they were taken

| | TP=2 | TP=3 | TP=4 | TP=6 |
| --- | ---: | ---: | ---: | ---: |
| prefill @32k, cold (tok/s) | **2,929** | 3,847 | 4,981 | 4,907 |
| prefill @128k, cold | **2,864** | 3,648 | 4,822 | 4,731 |
| RigMark cold 8k / 32k / 64k | not published | 3,295-3,325 / 3,602-3,610 / 3,565-3,583 (older build) | - | 4,776 / 4,902 / 4,903 |
| decode code / prose / structured (RigMark, reasoning low) | 60.5 / 36.4 / 89.5 | 79.8 / 45.2 / 119.8 | 114.6 / 59.5 / 161.6 | 120.3 / 66.5 / 176.4 |
| code streams 1 / 2 / 4 / 8, aggregate | 74 / 84 / 117 / 130 | 89 / 107 / 146 / 173 | 129 / 150 / 201 / 240 | 145 / 154 / 219 / 282 |

TP=4 stock vs all overlays (their A/B): 32k 2,730 -> 4,946, 128k 2,679 -> 4,750 (+81%).

**Methodology** (`gate/prefill.py`, `gate/run.sh`):

- **What is measured.** prefill = `usage.prompt_tokens` / client-side TTFT. TTFT is the first streamed token of
  either reasoning or content, so it includes HTTP, templating and tokenization.
- **Prompts.** Random consonant-vowel gibberish from a per-run seed, seeded by the word count too, so nothing hits
  vLLM's prefix cache. "32k" and "128k" are targets of 32,000 and 128,000 tokens, calibrated from a 2,000-word probe.
- **Request.** One request at a time on a quiet stack; `max_tokens` 128, temperature 0; no reasoning field, so the
  template's default (thinking on). The gate runs 4 seeds (2 in the quick gate); the README shows one value a cell.
- **Boots.** Measured on boots that restored weight snapshots. The first boot is 7-15% slower at 128k TP=2, and the
  first long prompt after a boot is dropped (it pays for JIT compiles).
- **TP=2 date.** The README's TP=2 row dates from 2026-09-28 (`6db84d4`, 8,192 budget). Later commits changed
  decode paths only.
- **Consistency check.** Where both exist, their gate and RigMark agree within ~1% (TP=3: 32k 3,574 vs RigMark 32K
  3,602; TP=6: 4,900 vs 4,902). So their TP=2 RigMark 32K should be ~2,900-2,950.
- **Clocks.** GPU clocks were locked at 1,989 MHz only for the TP=6 run; not locked at TP=2.
- **Decode.** RigMark single stream, reasoning low. Streams: `gate/conc_workload.py`, 512 tokens of code each.

### 1.3 Like for like

| metric | kindling TP=2 | ours (W17 prod) | vLLM TP2 (Alex Ellis, RigMark) | kindling / ours |
| --- | ---: | ---: | ---: | ---: |
| cold prefill ~32k (tok/s) | 2,929 (gate, 32,000) | 1,634 (RigMark 32K) | 1,908 | **1.79x** |
| cold prefill long (tok/s) | 2,864 (128,000) | 1,610 (ab.py 85,781); RigMark 64K 1,620 | 1,922 (64K) | **1.78x** |
| us a token a rank (both ranks work in parallel) | ~341-349 | ~621 | ~521 | 0.55x |
| decode code / prose / structured (RigMark low) | 60.5 / 36.4 / 89.5 | 67.9 / 43.0 / 88.8 | 44.0 / 18.9 / 64.9 | 0.89 / 0.85 / 1.01x |
| replay TTFT | vLLM APC (not reported for TP=2) | 0.22-0.27 s (8K-64K) | 2.97-5.77 s | - |
| context a request | 160k | 1,048,576 | 262k | - |

Differences that the ratios do not correct for:

1. **Quant and precision.** NVFP4 W4A4 experts and FP8 W8A8 dense GEMMs in prefill, against our EXL3 / q4 weights
   with bf16 activations. This is the dominant difference (section 2).
2. **Engine.** vLLM with torch.compile and FlashInfer / FlashKDA / DeepGEMM kernels, against TensorFold.
3. **Memory envelope.** They run 8,192-token prefill steps with 1.3 GiB free; our W10 8,192-row chunks gave +4%
   and were rejected at a 7.2 GiB floor.
4. **Measurement.** Their gate uses gibberish words and 32,000 / 128,000-token targets. RigMark uses its own corpus
   at 32,768 / 65,536. Both are cold and single-request. The ~1% gate-vs-RigMark agreement they show makes this
   small.
5. **Node count.** The same (2), but through a switch with both roots, against our direct cable with one root for
   NCCL.
6. **Exactness.** They make no exactness guarantee. We keep byte-exact resume, batching and drafting.

## 2. Where their prefill speed comes from

### 2.1 Techniques, with their receipts (TP=4 unless noted)

"Precision" marks what changes the numeric format of activations or weights relative to a bf16-activation path.

| # | technique (file) | what it does | their receipt | precision |
| ---: | --- | --- | --- | --- |
| K1 | **NVFP4 W4A4 MoE, CUTLASS** (stock, `MOE_BACKEND=flashinfer_cutlass`) | activations quantized to e2m1 with e4m3 per-16 scales and the checkpoint's input scales; FP4 tensor cores | stock baseline; `marlin` (W4A16) is the 16-bit alternative, untested on this image | **FP4 activations** |
| K2 | **Fused prefill MoE** (`megamoe/moe_prefill.cu`, `VLLM_MOE_PREFILL`, ≥1,024 tokens) | hand-written `mma.sync.kind::mxf4nvf4.block_scale` m16n8k64, BM = BN = 128, 256 threads, 4-stage `cp.async` ring, 64 B of K a stage, XOR swizzle + `ldmatrix`; tiles from a device list of (expert, m-tile) over expert-sorted rows, padded entries exit; **gather by row index inside the A loads**; fc1 epilogue does SwiGLU (with clamp) and NVFP4 requant; optional bulk L2 prefetch of the next tile's rows; `finalize` = weighted top-8 sum per token in fp32 | ~18 ms a layer at 16k tokens vs CUTLASS 22-26; **+7-8% prefill**; "equally close to an fp32 reference" | same W4A4 math as K1 |
| K3 | **FP8 MoE intermediate** (`VLLM_MOE_PREFILL_Y8`) | fc2 writes per-expert rows as e4m3 with a scale per 128 columns instead of bf16; halves the 1.07 GB a layer written and read back | **+5%**; GSM8K / HumanEval / NLL / tool-call unchanged | **FP8 storage** (stock is bf16; ours is fp32) |
| K4 | **Sequence parallelism** (`fixes/model.py`, `VLLM_GLM_SP_TP`) | residual and mHC state split by rows; mHC, norms and residuals on 1/TP of the tokens; all-gather before and reduce-scatter after each attention / MLP (same bytes as the all-reduce); decode stays plain TP (SP there cost 10-15%) | **+17%** (TP=3: -16% without; TP=6: -25% without) | none |
| K5 | **Inter-layer mHC fusion** (`hc_fused_post_pre`, vLLM's copy of SGLang's TileLang kernel) | hc_post of one block fused with the next hc_pre + RMSNorm | part of stock | none |
| K6 | **FP8 KDA-input gather** (`VLLM_GLM_SP_FP8_GATHER`) | the SP all-gather ahead of KDA in_proj moves per-token FP8 (in_proj would quantize the same rows anyway) | +2% | **FP8 activations** |
| K7 | **Quantized MoE gather** (`VLLM_GLM_SP_MOE_QUANT_GATHER`) | each rank routes its own rows, gathers NVFP4 (routed) + FP8 (shared expert) instead of bf16; a quarter of the fp32 router GEMM a rank | +3-4%; layer output within 1e-4 relative of the bf16 gather | **FP4 / FP8** |
| K8 | **MoE finalize into the RDMA reduce-scatter** (`arx/arxbig.cu`, `VLLM_GLM_SP_MOE_FUSED`) | one kernel writes shared + weighted routed sum (fp32, rounded once) straight into pinned send buffers and publishes rows as it goes, so the wire runs under the finalize; replaces the scale / add passes and NCCL's RS | ~12.5 -> ~8.4 ms a MoE layer at 16k; **+2.5-3%** | none |
| K9 | **Both CX7 PCIe roots in NCCL** (`FABRIC_SUBNETS`) | NCCL_IB_HCA lists both functions; one root tops out near 110 Gb/s | all-reduce 95-111 -> 175-191 Gb/s; **126k prefill 2,412 -> 2,680 (+11%)**; decode unchanged | none |
| K10 | **FP8 dense GEMMs in prefill** (`snapshot/dense_fp8.py`) | bf16 dense layers -> FP8 (per-channel W, per-token A, CUTLASS scaled_mm); weights > 16 MiB are run in 2,048-row pieces so the weight stays in L2 (16k x 6,416 x 4,096: 68 -> 169 TFLOP/s) | decode +14-20% (the prefill gain is not reported separately); ~+0.5% NLL on prose | **FP8 weights + activations** |
| K11 | **Triton sparse MLA** (`fixes/gb10_sparse_mla.py`) | one program a query token, all heads' Q in registers, 32 gathered FP8 latent rows a block, fp16 dots, online softmax; split-K + combine for < 96 tokens | 3.3x FlashInfer's ~10 TFLOP/s | fp16 dot operands (FP8 KV -> fp16 is exact) |
| K12 | **KDA conv writes q / k / v as separate dense tensors** (`causal_conv1d.py` `out_group`) | FlashKDA needs dense q, k, v; saves three 56 MB copies a KDA layer at 16k | +2-3%, bit-identical | none |
| K13 | **FlashKDA** (stock, rebuilt at `17a037d`) | fused CUTLASS chunked KDA prefill, "~2-4x faster" than the Triton chunk_kda; the nightly's `b59532f` rounded the recurrent state to bf16 every 16 tokens and long prefills then corrupted tool calls, so `17a037d` keeps it fp32 | stock | bf16 operands, fp32 state |
| K14 | Small vLLM fixes (`fixes/`) | FlashInfer's MLA planner cloned a 136 MB buffer every step; the indexer head gate ran an fp32 GEMM at 69 us a layer; DeepGEMM FP8 MQA logits for the indexer | not broken out | indexer FP8 (stock) |

The multiplicative sum of the separately reported gains (K2, K3, K4, K6, K7, K8, K12) is ~1.47x, against their
measured 1.81x over stock. The rest is K10, K11, K14 and a newer stock image. K9 was already in the stock baseline.

### 2.2 Estimated per-token budget, theirs vs ours (TP=2, us a token a rank)

**Ours now.** ROOFLINE's `now` column (W8: 680 us at 8,192-row chunks), corrected for three later changes:

- W9 4,096-row chunks: +25 us on the routed experts (W10's M8 vs M);
- W9 b12x bit 4: -41 us of sparse attention;
- W10 0390 MLA expand v2: -36 us expand, -7 us absorb.

That gives 621 us, matching the measured 1,606-1,610 tok/s.

**Theirs.** Only the MoE line is grounded: 18 ms a layer at 16k tokens at TP=4, doubled for TP=2's twice-as-wide
expert slice. The rest is FLOPs at their formats' plausible rates, and the total is constrained to their measured
341 us. Treat it as an attribution, not a measurement.

| stage | ours now | theirs (est.) | gap | why theirs is faster | closable at our precision? |
| --- | ---: | ---: | ---: | --- | --- |
| Routed experts + router / combine | 233 (193 + 40) | 105-130 | **~110** | K2 fused kernel at ~90 TF/s effective, K3 FP8 Y, bf16 finalize | **mostly**: ~90 TF/s is 84% of the bf16 roof. The format buys fewer A-operand bytes and no trellis decode |
| KDA projections | 91 | 30-40 | ~55 | FP8 W8A8 at up to 169 TF/s | partly: a better q4 GEMM at 80-90 TF/s gives ~55 us (ROOFLINE gap 3) |
| KDA recurrence | 64 | 20-40 | ~35 | FlashKDA, fused and bf16 tensor cores | only with a precision change (bf16 vs our tf32), or a fused tf32 design (0400 fused measured slower) |
| Sparse attention + MLA expand / o_proj + DSA proj + indexer | 121 (55 + 33 + 21 + 13) | 70-90 | ~40 | bf16 tensor-core absorb / expand (we keep fp32 FMA by decision), FP8 projections | expand: only with `LATENT_TC`; projections: gap 3 |
| Shared expert + dense MLP | 38 | 10-15 | ~25 | FP8 | partly (gap 3) |
| Hyper-connections | 46 | 20-30 | ~20 | SP on half the rows + fused post / pre | partly: 0320 already halves the rows; fuse the boundary |
| Comm exposed + host + other + MTP rows / taps | 29 | 25-40 | ~0 | - | - |
| **total** | **621** | **~341 (measured)** | **~280** | | |

## 3. Each technique against our engine and constraints

Our constraints:

- resume == fresh prefill; row-invariant, C-independent prefill; drafted == serial;
- FP8 prefill activations rejected;
- EXL3 4-bit experts stay unless quality is proven equal.

Gains are against 621 us (Δ = us a token saved; +% = 621 / (621 - Δ) - 1).

| # | technique | applies to us? | expected gain | effort | exactness risk | precision change? |
| ---: | --- | --- | --- | --- | --- | --- |
| K1 | W4A4 NVFP4 MoE | **no**: FP4 activations are a stronger version of the rejected FP8 prefill, and the weights would have to leave EXL3 | - | - | - | yes: FP4 A and W |
| K2 | fused prefill MoE **structure** | **yes, in idea.** Our fat kernel costs the *sum* of its MMA (77) and DRAM (73) roofs (ROOFLINE 1.2). Their design is the working GB10 answer that 0330 lacked: 256 threads (0330's cfg 1/2 could not launch), a 4-stage `cp.async` ring with swizzled `ldmatrix`, expert-sorted 128-row tiles from a device list (no host sync, no empty programs), the gather inside the A loads, SwiGLU in the fc1 epilogue, optional L2 bulk prefetch of the next tile's gathered rows. For EXL3 the B operand is trellis-decoded into bf16 fragments, not loaded as e2m1, so reuse each decoded tile over 128 rows. Also check whether `rot_in` (~15 us a token at 2,048 rows) can move into the A-load stage | Δ 40-70: **+7-13%** | 1-3 weeks; 0330's tests and bench exist | **low**: same bits if each output's k16 mma chain and order are fat's (the existing bitwise tests decide) | no |
| K3 | FP8 MoE intermediate | no as FP8. **The bf16 version is ROOFLINE gap 1b** (Y is 128 KB a token and layer in fp32 today) | bf16 Y: Δ 15-25: **+2.5-4%** | 2-3 days | new fast-prefill bits (deterministic, row / C-independent), needs a snapshot tag and the quality gate | **yes** (storage: fp32 -> bf16). Their evidence: even FP8 Y left GSM8K / HumanEval / NLL / tool calls unchanged |
| K4 | sequence parallelism | **already ours** in the form that matters at TP=2: 0320 row split (hc on each rank's half, +8.8% in W8). Norms and residual adds beyond hc are small | ≤ Δ 5 | - | - | no |
| K5 | inter-layer hc fusion | **yes.** Fuse hc_post + next hc_pre (+ norm) on the row-split half in prefill. 0520 did it for decode windows ≤16 rows with the Triton bits reproduced; extend it to prefill slabs, or enable `hc_fused` bit 1 | Δ 8-15: **+1.3-2.5%** | 3-5 days (0520's spec / interpreter harness exists) | low: same bits by 0520's method | no |
| K6 | FP8 KDA-input gather | **no** (FP8 activations) | - | - | - | yes |
| K7 | NVFP4 / FP8 MoE gather | **no** (FP4 / FP8 activations); our gathers are bf16 partials | - | - | - | yes |
| K8 | finalize into the send buffer | **yes, adapted.** Our `_combine_s` (at its DRAM roof) writes the MoE partial, then 0084 all-gathers it. Write the bf16 partial rows, with the shared-expert add folded in, straight into the NCCL send buffer in the same pass, and start the exchange per slab. Keep the buffers device-resident: they found GEMMs reading pinned memory 3.7x slower on GB10 (that is why `VLLM_ARXBIG_AG=0`) | Δ 5-12: **+1-2%** | 3-5 days | low: same bits if the summation order and single rounding are kept | no |
| K9 | both CX7 functions in NCCL | **yes, config only.** Prefill all-gathers (4 MiB, `RING_LL`, ~9 GB/s, ~456 us each) run on `rocep1s0f1` alone. `roceP2p1s0f1` (<second-subnet>.x, GID 3) is already up for 0350. Set `NCCL_PASSTHROUGH=1` + `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`. Also A/B `NCCL_MAX_NCHANNELS` (they pin 8; fewer NCCL CTAs beside `_hc_post`, which ran 3x slower next to the all-gathers in W7) and `NCCL_PROTO=Simple` for the 4 MiB transfers | Δ 5-20: **+1-3%** (their +11% was at TP=4, where collectives are a larger share) | 1-2 h GPU window | **none**: an all-gather is a copy, and we sum partials ourselves in fixed order; 0350's ≤256 KiB path is untouched | no |
| K10 | FP8 dense GEMMs | **no** (FP8 activations; weights stay q4mse). Their lesson for our own q4 GEMM (gap 3): tile for weight reuse in L2. Their 2,048-row pieces took a large GEMM from 68 to 169 TF/s | via gap 3: Δ 40-50: +7-9% | 1-2 weeks | low: same bits if `matmul_fast`'s per-output k-chain is kept | no (gap 3 itself) |
| K11 | Triton sparse MLA | **nothing new.** Our b12x bit 4 is the same shape (a row's heads in one tile, FP8 latent gathered, one pass) at ~27 TF/s against their ~33. fp16 operands would be new bits for ~Δ 5-10 | ≤ +1.5% | - | new bits | fp16 vs bf16 |
| K12 | conv `out_group` | **not needed**: `_kda_prep` reads the conv taps straight from the projection rows (`_conv_act`); there is no q / k / v copy | 0 | - | - | - |
| K13 | FlashKDA | **possible, as a precision change.** MIT, builds for SM12x. Replace `_kda_prep` / `_state` / `_norm` (64 us a token; 0400 split +1%, fused slower) with its fused kernel. It uses bf16 operands in the chunk products where ours use tf32 (`DOT_PRECISION="tf32"`), so it is new fast-prefill bits. Its state must be fp32 at `17a037d` or later: their bf16-every-16-tokens bug corrupted long-prompt tool calls, which shows the KDA path is precision-sensitive. C-independence needs its internal chunk boundaries on our 64-token grid (to verify) | Δ 25-40: **+4-7%** | 3-5 days + the quality gate | medium: new bits; row / C-independence must be re-proven | **yes**: tf32 -> bf16 operands |
| K14 | vLLM fixes | not applicable (vLLM's planner and indexer code). Our indexer is 13 us a token | - | - | - | - |
| - | 8,192-token steps (their budget; +9% for 16,384 at TP=2) | **ours already** (W10 M8: +4% over 4,096, same sha), rejected on the memory floor (7.23 GiB on the worker node). They run with 1.3 GiB free; we should not. Use 8,192 only when one request is active and admission memory allows (PROFILE §6); 0550's bounded scratch makes the peak predictable | Δ ~25: **+4%** | 2-3 days + the stress run | none (measured same sha) | no |
| - | `LATENT_TC` (their stack does absorb / expand on bf16 tensor cores) | exists (`GLM53_TF_LATENT_TC=1`), **off by user decision** | Δ ~15-25: +2.5-4% | 0 | new bits | yes (fp32 -> bf16 operands) |

## 4. Ranked adoption list

Mapped to our stages. Same-bits / exact items come first; precision items only behind the quality gate and a user
decision.

| rank | item | stage (ours now, us a token) | gain | effort | why here |
| ---: | --- | --- | --- | --- | --- |
| 1 | **K9: NCCL on both CX7 functions + channel / protocol A/B** | comm + overlap tax on hc / shared expert (~29 + part of hc 46) | +1-3% | 1-2 h | exact by construction, config only, and it tells us how much of the hc lap is still comm tax |
| 2 | **K2 structure: rebuild the routed-expert prefill kernel** | routed experts 193 | +7-13% | 1-3 weeks | the largest single gap; their kernel shows ~90 TF/s effective is reachable on GB10 with a 256-thread, 4-stage design, i.e. near our bf16 roof |
| 3 | **8,192-row lone chunks under a memory rule** | routed experts (weight re-reads) | +4% (measured) | 2-3 days + stress | measured and exact; blocked only by memory policy |
| 4 | **K10 via ROOFLINE gap 3: a better q4 GEMM** (L2-aware weight reuse, persistent 48-SM grid, pre-shuffled fragments) | KDA proj 91, shared / dense 38, DSA proj 21 | +7-9% | 1-2 weeks | their FP8 route is closed to us; this is the same-precision way to take half that gap |
| 5 | **K5: fused hc boundary in prefill** | hc 46 | +1.3-2.5% | 3-5 days | 0520's bit-reproduction method carries over |
| 6 | **K8: combine + shared add into the send buffer** | router / combine 40 | +1-2% | 3-5 days | one DRAM pass fewer; keep the buffers device-resident |
| 7 | K3 as bf16 Y (precision: storage) | routed experts / combine | +2.5-4% | 2-3 days | behind the gate: new bits |
| 8 | K13 FlashKDA (precision: bf16 operands) | KDA recurrence 64 | +4-7% | 3-5 days | behind the gate; KDA precision is known to matter |
| - | `LATENT_TC` (precision) | MLA expand 33 | +2.5-4% | 0 | user decision; listed only because it is part of their lead |

**Stacked.**

- Ranks 1-6 (exact / same bits): Δ ≈ 10 + 55 + 25 + 45 + 10 + 8 ≈ 150 us -> ~470 us a token, **~2,100-2,200 tok/s
  (+30-37%)**.
- Adding 7, 8 and `LATENT_TC`: ~400 us, **~2,450-2,550 tok/s**.
- Their remaining ~60-100 us lead is FP4 / FP8 arithmetic (K1, K6, K7, K10), which our constraints exclude. That
  still matches ROOFLINE's "~3,000 tok/s realistic ceiling" for our dataflow: they sit right at it with cheaper
  formats.

### First prototype to test

1. **GPU window, ~1 h (rank 1): dual-function NCCL for the prefill all-gathers.**
   - **Before loading.** Run a 4 MiB all-gather / all-reduce bus-bandwidth check over one and over both functions.
     Healthy is Ring ≫ Tree (their 180-191 vs 55-93 Gb/s at TP=4). If Tree beats Ring, the CX7 has latched the slow
     state after a DAC re-plug: power-drain the node (a reboot does not clear it).
   - **Loads** on the prod config (`NCCL_PASSTHROUGH=1`), one variable at a time:
     `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, then `NCCL_MAX_NCHANNELS` 2 / 4 / 8, then `NCCL_PROTO=Simple`.
   - **Per load:** ab.py 24.5k / 98k x2 (reply sha `8794a3463259cc2f`), exact 10/10, batchexact 4/4, 1 / 4-stream
     decode (control exchanges ride NCCL; decode must not drop).
   - **Adopt at ≥ +1.5% prefill with decode not lower.**
2. **Offline code prototype (rank 2): an EXL3 x bf16 grouped GEMM with K2's skeleton.**
   - Build it as a standalone microbench next to `expert_kernel` fat, at 4,096 and 8,192 rows with real routing
     (W5 / 0260 bench).
   - **Skeleton:** 256 threads, 128x128 tiles over expert-sorted rows from a device (expert, m-tile) list, a
     multi-stage `cp.async` ring for the gathered bf16 A rows with an XOR swizzle and `ldmatrix`, the trellis tile
     decoded once into shared memory per K step and reused by all 128 rows, SwiGLU in the fc1 epilogue, and bf16 H
     between fc1 and fc2 as fat does today.
   - **Gate:** bitwise == fat on the existing tests, and ≤ 0.75x fat's time a layer. Only then wire it in.

## 5. Not applicable, or already covered

| item | why |
| --- | --- |
| DFlash2 adaptive-k scheduler, RecoverSSM, drafter KV pool | decode / vLLM KV geometry. We have cost-derived depth (0071), 16-row verify (0380) and our own KDA replay |
| megamoe decode kernel (W4A16 NVFP4, ≤8 tokens), megadense4 W4A16 | decode; NVFP4 weights. Our EXL3 decode kernel runs at ~205 GB/s |
| arx small all-reduce (13-27 us) and its L2 prefetch during the wait | decode. Ours: 0350 RoCE (11-27 us) and 0460 L2 prefetch, both in prod |
| Weight snapshots (restart ~3 min) | ours: 0140 prepared weights (starts in 25-40 s) |
| vLLM prefix caching | our session store resumes 8K-64K replays in 0.22-0.27 s (their APC path is vLLM's) |
| CUDA graphs / torch.compile for prefill | they capture decode sizes only and run SP prefill eagerly; not a prefill lever either side |
| spin_wait (vLLM's shm queue busy-loop; -20 °C SoC) | vLLM EngineCore. The general lesson (CPU spin steals GB10's shared power budget) is worth one check: watch clocks and power during a 98k prefill (`nvidia-smi dmon`) for throttling |
| TP=3 / TP=4 / TP=6 / TP=RING4, mentat, fabric ring relay | two nodes, direct cable |
| `GPU_MEM_UTIL`, KV pin, `busy_loop_s`, head election, GID derivation | vLLM / Ray-replacement operations. We don't pin `NCCL_IB_GID_INDEX` (the two functions share GID 3 today; re-read it if a node's addresses change) |
| Their preflight (PCIe link width, ports < 200 Gb/s, headless, clock-limit events) | mostly already in `serve.sh preflight`. PCIe link degradation and clock-event checks are cheap additions if wanted (MIA-AUDIT item 8 covers clocks) |

## 6. Decode, briefly

At TP=2 we lead on RigMark code and prose, and match structured: 67.9 / 43.0 / 88.8 against 60.5 / 36.4 / 89.5
(+12% / +18% / -1%). Their 1-8-stream code aggregate (74-130) uses a different workload from RigMark's C1 / C2 / C4,
so it is not compared here. Their decode levers are either ours already (L2 prefetch in the all-reduce wait, RDMA
small collectives, adaptive draft length) or vLLM-specific (RecoverSSM, the drafter pool, spin-wait).

## 7. Quality numbers they report, for the precision items

- **Overall, all overlays vs stock:** GSM8K-250 97.2% vs 97.2%; HumanEval 156/164 both; count to 200 (thinking off)
  0 corrupt; tool call at 42k (40 greedy runs) 10 diverge on stock, 0 with overlays.
- **NLL vs bf16 dense layers:** prose +0.008 to +0.025, code within ±0.007 (the FP8 / NVFP4 dense layers).
- **Per-change NLL:** "about 1%" on prose in total; FP8 dense alone ~+0.5%; NVFP4 on every dense layer 2-3x more
  (rejected).
- **Y8 (FP8 MoE rows):** GSM8K / HumanEval / NLL / tool-call results unchanged.
- **Quantized MoE gather:** layer output within 1e-4 relative of the bf16 gather.
- **Not reported:** the W4A4 MoE itself against a 16-bit-activation MoE (marlin is untested on their current image),
  and FlashKDA against a higher-precision KDA.
- **Their determinism:** 42k greedy gives 3 distinct completions in 16 runs, from vLLM's batch-shape rounding. Our
  bar (byte-exact across drafting, batching and resume) is stricter than anything their numbers are held to.

## 8. Licence and reuse

- **Repository:** no licence (GitHub API `license: null`, no LICENSE file). By default all rights are reserved for
  the files they list as "ours":
  - `arx/*`;
  - `adaptive-k/`, `megamoe/*`, `quality/`;
  - `fixes/gb10_sparse_mla.py`, `fixes/recoverssm.py`;
  - `snapshot/weight_snapshot.py`, `snapshot/dense_fp8.py`;
  - the image patches marked "ours".
  **We copied none of their code or kernels into this repo.** Everything in section 4 is written as ideas to
  re-implement.
- **vLLM / FlashInfer-derived files** (`fixes/*.py` other than the two above, `snapshot/base_loader.py`,
  `modelopt.py`, `flashinfer_cutlass_moe.py`, `arx/cuda_communicator.py`): Apache-2.0 per their headers. Reusable
  with notice, but they are vLLM internals and do not map onto TensorFold.
- **`image/patches/glm53-flash_SM121.py`:** MiaAI-Lab, MIT (licence file included).
- **FlashKDA** (`vllm-project/FlashKDA`): MIT. Usable as a dependency with its notice. CUTLASS is BSD-3.

## 9. What needs a GPU before relying on it

- Rank 1's A/B (above), and a bandwidth check of both functions with the model stopped.
- The routed-expert microbench (rank 2) against fat at 4,096 / 8,192 rows before any engine wiring.
- For rank 3: the 4 x 250k stress plus the needle at 8,192-row lone chunks with the admission rule, bar ≥ 8 GiB.
- For FlashKDA: whether it builds and runs on sm_121 in our image, its chunk size against our 64-token grid, and
  row / C-independence tests before any quality gate.
- One nsys capture of today's config (ROOFLINE's standing request) to replace the reconstructed "ours now" column.
