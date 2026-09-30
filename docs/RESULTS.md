# Results

> **Work in progress.** Measured on one pair of DGX Sparks, 2026-09-27 to 2026-09-30, with the **abliterated** checkpoint
> below. The sections are in the order the work happened; the newest ones are the current state: **W20** (the end of
> this file) for 0620's tool-calling fixes and the current 3-run RigMark numbers (image b11), **W19** for the production config (`config/prod.env.example`: 4 requests sharing a 1M-token KV pool, image input,
> replay snapshots, multi-slot prefill, the 0580 expert load path, two-function NCCL, 0550's scratch, structured
> output), W18 for the 0550 attribution, **W17** for the earlier 3-run RigMark numbers (image b9), W15 for image input and replay, W10-W12 for the text-only stack it grew from, "Stacked run and production config" for the older
> single-stream config. The public repo carries the benchmark JSON and the window scripts of the runs in `results/`
> (E*, F*, L*, M*, P*, Q*, A1, B1, B2, S1, X1, Y1, Z1, Z2, W1-W12, W14-W20, FINAL-20260930, tooleval, rigmark (W13, W15's two
> RigMark runs, W17's three and W20's three), roofline, sim0340, sim0380, theory2, THEORY2-SESSION, upstream-0362, draftvocab); logs (`*.log`, `*.out`, `*.err`,
> RigMark's `run.log`), nsys traces, test output of the early runs and some runs (B3, K0, K1, P1, T*) are summarized
> here only, so some file names below point at logs that are not in the repo. Hosts in the JSON were normalized to
> `127.0.0.1` / `<worker-ssh>`, node names to head / worker (also in file names: `tests-head/`, `probe-head.log`, `kg-head/`),
> and paths in the window scripts to `$HOME/glm53-tensorfold-spark`. `results/W12/n1-corpus.txt` (= W15's, the long
> prompts of N1, cut from an earlier copy of this repo's docs) had its four node-name mentions replaced the same
> way, so its prompts, and N1's reply hashes, differ slightly from the ones measured. Config names refer to the
> `config/*.env.example` files; `results/W*/prod.env.before-W*` are the production config before each window, with the
> same placeholders.

GLM-5.3-Flash abliterated EXL3 (`neko-legends/GLM-5.3-Flash-Uncensored-EXL3` @ `07135ec0`) on two DGX Sparks
(GB10, TP=2 over the 200 Gb/s CX7 link). Three stacks on the same pair, the same weights and the same client:

| Stack | What it is |
| --- | --- |
| vLLM prod kit | [Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) @ `8e443d6` (2026-09-20; a fork of [MiaAI-Lab's kit](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)) as we ran it in production on this pair: vLLM, 1M context, FP8 KV, DFlash2 drafter k=7 with adaptive k (EMA), fused EXL3 MoE kernels, the same abliterated weights |
| TensorFold upstream | `vendor/TensorFold` @ `2f8e514` (0.3.4) with `GLM53_TF_NONEXPERT=bf16`, i.e. upstream behaviour; drafter policy `auto` (upstream `EXL3_AUTO`) |
| TensorFold + patches | the same engine with `patches/0001`-`0004`, `GLM53_TF_NONEXPERT=q4mse`, drafter policy `auto` |

Both TensorFold stacks load the same DFlash2 drafter revision as the vLLM kit (`incoai/GLM-5.3-Flash-DFlash2` @
`7d74cdd`) next to the checkpoint's own MTP layer.

## Decode (tok/s, median of 5, single stream, thinking off)

| Cell | vLLM prod kit | TF upstream (bf16) | TF + patches (q4mse) | vs vLLM | vs upstream |
| --- | ---: | ---: | ---: | ---: | ---: |
| tf code, sampled, 64 tok | 35.2 | 32.3 | **44.9** | 1.28x | 1.39x |
| tf chat, sampled, 64 tok | 27.1 | 30.4 | **41.7** | 1.54x | 1.37x |
| tf code, greedy, 64 tok | 41.9 | 46.4 | **61.0** | 1.46x | 1.31x |
| tf chat, greedy, 64 tok | 22.8 | 28.7 | **44.3** | 1.94x | 1.54x |
| sequence, 512 tok | 67.5 | 61.8 | **80.6** | 1.19x | 1.30x |
| code, 512 tok | 42.4 | 44.3 | **58.4** | 1.38x | 1.32x |
| json, 512 tok | 50.4 | 55.9 | **68.5** | 1.36x | 1.23x |
| kit hashmap, 200 tok | 30.0 | 36.4 | **49.1** | 1.64x | 1.35x |
| kit structured, 200 tok | 72.7 | 65.1 | **85.9** | 1.18x | 1.32x |
| kit essay, 200 tok | 26.1 | 32.7 | **43.9** | 1.68x | 1.34x |

Serial decode (no drafts, `"draft": false`): 17 tok/s with BF16 non-expert weights, 33 tok/s with q4mse. The
one-row verify step went from 57 ms to 30 ms; that is where most of the drafted gain comes from.

## Prefill and long context

| Stack | Prefill tok/s |
| --- | --- |
| vLLM prod kit | 960 @ 1.8k, 1340 @ 7k, 1448 @ 28k prompt tokens |
| TF upstream (bf16, 64-row chunks) | 256-266; prompts past 2,051 tokens get HTTP 400 unless `--context` is set |
| TF + patches (q4mse) | 404-420 over the measured prompt sizes |

Prefill tok/s = prompt tokens / cold time to first token (unique prompt prefix, so no cache hit). Prefill is
where vLLM stays well ahead (2.3-3.6x); see `docs/PREFILL-ANALYSIS.md` for the chunk-size work.

Decode behind a ~28k-token prompt: vLLM 65.4 tok/s, TF + patches 70.8 tok/s.

## Exactness and quality

| Check | TF upstream (bf16) | TF + patches (q4mse) | vLLM prod kit |
| --- | --- | --- | --- |
| drafted == serial, byte-identical (10 cases) | 10/10 | 10/10 | n/a |
| MMLU-200, greedy, thinking off | 87.0% | 88.0% (13/200 answers differ from bf16) | pending |
| refusals (10 prompts) | 0/10 | 0/10 | 0/10 |

"drafted == serial" means that for every request the drafted reply is the same bytes as the one-token-a-round
reply of the same engine and weights. q4mse is a different set of weights from bf16 (the non-expert matrices are
re-quantized), so bf16 and q4mse replies differ from each other; the MMLU and refusal rows are the check that
the re-quantization does not cost quality.

## Methodology

- Client: `bench/glmbench.py` (standard library only), the same script against every stack, from the head node.
  Every request streams through the OpenAI API; decode tok/s = `(completion_tokens - 1) / (last content chunk -
  first content chunk)`, so prefill and time to first token are excluded.
- Each cell: one short warm-up request, then 5 measured requests; the table reports the median.
- `tf` suite (TensorFold's published cells): 64-token replies with `ignore_eos`; `code` is a raw completion,
  `chat` is chat with thinking off. Sampled = temperature 1, top-k 20, top-p 0.95, seeds 1234-1238; greedy =
  temperature 0.
- `tweet` suite (sequence / code / json): chat, thinking off, greedy, 512-token replies.
- `kit` suite (hashmap / structured / essay): the vLLM kit's own decode prompts verbatim, chat, thinking off,
  greedy, 200 tokens.
- `ctx` suite: a unique tag plus filler text sized for ~2k / 8k / 32k tokens (1.8k / 7k / 28k actual prompt
  tokens), then a short question and a 256-token greedy reply; cold and warm runs.
- `exact` suite: 5 prompts x (greedy, sampled seed 1234), 128 tokens each, drafted vs `"draft": false`,
  compared by SHA-256 of the reply text.
- Quality: `bench/quality.py`, MMLU 200 questions stratified over the 57 subjects with a fixed seed
  (`bench/data/mmlu200.jsonl`), greedy, thinking off, first A-D letter of the reply; plus 10 prompts a
  safety-tuned model tends to decline, counting replies that open with a refusal.
- One request at a time, nothing else on the GPUs. Only one stack runs at a time (`scripts/serve.sh start`
  refuses to start while another CUDA process is up on either node).

Commands:

```bash
python3 bench/glmbench.py --base http://127.0.0.1:8080 --model GLM-5.3-Flash-Uncensored \
    --suites tf,tweet,kit,ctx,exact --label tf-q4mse --out results/tf-q4mse.json
python3 bench/quality.py --base http://127.0.0.1:8080 --model GLM-5.3-Flash-Uncensored \
    --label tf-q4mse --out results/Q-tf-q4mse.json
```

## Drafter policy sweep

_Placeholder._ Per-policy decode (MTP-only, DFlash2-only, `auto`, draft lengths) with q4mse non-expert weights,
same cells as above. To be filled in.

## Stacked run and production config, 2026-09-27/28

One image with every committed patch (0001-0180; 0080 fast2 experts, 0083 FP8 tile 128,64,4,2, 0085 chunk-size-independent
prefill, 0110 sessions, 0120 batching, 0130 decode kernels, 0140 fast boot, 0150 health/metrics, 0160 OpenAI compat,
0170 fat experts / BF16 KDA copy, 0180 batch sessions). Load A (`results/A1/`): `CONTEXT=262144`, q4mse, latent KV,
fast + lean prefill (block 1024, rows `auto`, max 8192), overlap, bf16 gathers, expert loop, real calibration, cost
depths, lookup, `SESSION_GIB=12`. Production (`config/prod-single.env.example`, `results/S1/`): the same with `CONTEXT=524288`,
`GLM53_TF_FAST_EXPERTS=fat`, cached calibration, decode kernels off, FP8 off, on :8000 as `GLM-5.3-Flash-EXL3`.

| | earlier best (F9/F10) | Load A, FP8 off | Load A, FP8 on | **production** | vLLM prod kit |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill 7k (tok/s, cold prompt, warm kernels) | 1,024 (FP8 on) | 1,084-1,122 | - | **1,162** | 1,340 |
| prefill 28k | 1,034 (FP8 on) | 1,062 | 1,177 | **1,209** | 1,448 |
| prefill 112k | 1,013 (FP8 on) | 1,105 | 1,186 | **1,162** | - |
| routed experts in a 28k / 112k prompt (s, rank 0) | 10.0 / 39.3 | 5.9 / 23.1 | 6.1 / 25.8 | | |
| follow-up: 34.8k conversation + reply + 2.2k new tokens (TTFT) | 3.6-6.8 s | 2.9 s | | 3.3 s | |
| session revisit, ~37-39k tokens (A,B,A,C,B,A) | ~36 s (re-prefill) | 0.45-0.54 s | | 0.38-1.01 s | |
| decode tf code greedy / chat greedy / kit structured / tweet sequence | 65.7 / 41.6 / 95.8 / - | 64.3 / 47.6 / 96.7 / 90.1 | | 65.8 / 44.7 / 98.1 / 93.3 | 41.9 / 22.8 / 72.7 / 67.5 |
| drafted == serial (`exact`) | 10/10 | 10/10 | | 10/10 | |
| MMLU-200 / refusals | 89.0% / 0 | 89.5% / 0 | 88.5% (Q5-1) | 89.5% / 0 | |
| tool calls, opencode toolset, T 0 (clean / corrupt) | | 200/210 / 0 (thinking off) | 200/210 / 0 | 95/105 / 0 (thinking on) | |
| load time | ~6-8 min | 490 s | | **34-37 s** (prepared folders) | |

Prefill tok/s = prompt tokens / cold TTFT on a unique prompt, after the kernels compiled (production: the canary's
4k / 16k warm-up at start does that). At 28k vLLM is still ~20% ahead on prefill; decode is 1.3-2x vLLM.

**Final production (2026-09-28 01:55, image `glm53-tensorfold:z` = patches through 0210, `config/prod-single.env.example`)**: the
column above plus 0190's `GLM53_TF_MOE_GLUE=5` (parallel MoE grouping + in-place combine; same bits) and 0210's
prompt-token cache (on by default; `verify` mode showed no difference). Measured on that load (`results/Z1/`):

| | production (final) | vLLM prod kit |
| --- | ---: | ---: |
| prefill 28k / 112k (tok/s) | **1,266 / 1,238** (`moe_glue` 0: 1,197 / 1,177) | 1,448 / - |
| follow-up: 34.8k + reply + 2.2k new tokens | **2.35 s** | |
| session revisit ~37k tokens | 0.41-0.51 s | |
| exact (drafted == serial) | 10/10 | |
| load time | 36 s | |
| verified through the HTTPS reverse proxy in front of the API | models, thinking (`reasoning` == `reasoning_content`), `stop` streamed / not, stream == non-stream, tool calls 38/42 clean, 0 corrupt | |

0190 knobs A/B'd per request on one load (`results/X1/`, `results/Z1/`, `results/bench_glue_*.txt`):

| knob | 28k / 112k tok/s | GPU tests | production |
| --- | ---: | --- | --- |
| none | 1,195-1,197 / 1,176-1,177 | | |
| `moe_glue` 1 (grouping) | 1,266 / 1,230 | pass | on (in 5) |
| `moe_glue` 5 (+ in-place combine) | 1,266 / 1,238 | pass | **on** |
| `moe_glue` 7 (+ one-kernel router: 2.2 vs 1.3 ms at 8192 rows) | 1,218 / 1,229 | pass | off |
| `attn_bm32` | 1,247 / 1,211 | engine test fails (tile needs 128 KB of shared memory on the test shapes) | off |
| `mtp_window` 8192 | 1,194 / 1,216 | main-model state / resume tests fail | off |
| grouping + bm32 + mtp_window | 1,276 / 1,289 | | off |
| `hc_fused` | - | out of shared memory (131 KB > 101 KB) | off |

Batching with 0200's on-set (`results/Y1/`), BATCH=4 at 262k after dropping page caches (4 slots fit):
- batched == alone: 4/4;
- aggregate 76-79 tok/s at 4 streams (single 47-69);
- per-slot resume: 24.5k cached, ~2 s; a 5th session evicts a slot;
- a 35k prefill beside 3 decoders: 62 s TTFT, 2.4 s decode gaps;
- single-request prefill 5% below single-stream;
- MemAvailable fell to 2-4 GiB under load.

Not used in production (memory).

**Last test window (01:35-02:00, image `z2` = 0190 fixes, `results/Z2/`, `results/T8/`).**

- `test_glue_patches`: 79/80 pass; the one failure is `latent_tc`, which is off.
- `hc_fused` is bitwise but 7-8x slower (num_stages=1): off.
- Per request at 28k / 112k (tok/s):

  | knobs | 28k | 112k |
  | --- | ---: | ---: |
  | production knobs | 1,261 | 1,235 |
  | + `attn_bm32` | 1,286 | 1,284 |
  | + `mtp_window` 4096 | 1,338 | 1,306 |
  | both | 1,389 | 1,357 |

  `exact` 10/10 with both. Decode right after the 112k prompt with both: 56 tok/s (81 without).
- Rank 0 then aborted during a repeat `attn_bm32` run (`terminate called without an active exception`, exit 133). The
  kernel log shows `NVRM ... Out of memory` at the same minute: unified memory ran out at 524k context + 12 GiB store.
- Production went back to image `z` with the committed single-stream config and was re-verified through the API (02:00).
  `attn_bm32` / `mtp_window` stay off until they are run at a smaller CONTEXT or SESSION_GIB and checked for memory.

What was tried and left out of production, and why:

- **Decode kernels (0130)**. On the real model they cost time instead of saving it. 1-row verify: off 31.8 ms,
  `v2` 32.7 ms, `v2,pdl` 32.9 ms (the L1 load: 31.3 ms). Decode, off / v2 / v2,pdl (tok/s):

  | cell | off | v2 | v2,pdl |
  | --- | ---: | ---: | ---: |
  | tf code greedy | 64.3 | 62.4 | 56.4 |
  | kit structured | 96.7 | 96.4 | 93.6 |
  | tweet sequence | 90.1 | 88.5 | 87.7 |

  This was the decode regression of Load A.
- **FP8 prefill (0083)**: +11% at 28k, +7% at 112k with the new tile. Greedy replies diverge from FP8-off within the
  first ~20 tokens on 25 of 30 prompts. Needle retrieval at 28k: 10/10 at 10 / 50 / 90% depth both ways. Tool calls
  equal. Decision: off (a real behaviour change for a single-digit gain).
- **Batching (0120)**, `GLM53_TF_BATCH=4`. At 262k context only 1 sequence fits (3.57 GB of cache slots each, 4 GB
  kept free); at 131k, 2 fit. With 2 slots:
  - batched == alone 4/4;
  - aggregate 56-74 tok/s against 46-72 single-stream (per stream ~30);
  - a 35k prefill stalls a decoding stream for up to 3.2 s.

  0180 (sessions in batch slots) failed 4 of its GPU tests (no cache reuse, follower replay). Production is single
  request + session store.
- **BF16 KDA projection copy (0170)**: 1.45x on that matmul (~7% of a 28k prefill), for +3.26 GiB a rank. At 524k
  context with the session store that would take MemAvailable below 8 GiB during a 128k request (measured minimum
  without it: 11 / 10 GiB). Off.
- **Fat experts (0170)**: bitwise equal to fast2 (tests) and +3-5% end to end (fast2 1,129 / 1,154 / 1,122 tok/s at
  7k / 28k / 112k). On.
- **Cold first requests**: right after a load the first fast prefill compiles Triton kernels (7k: 727 tok/s).
  Production's canary warm-up (`WARMUP_LENGTHS="4096 16384"`) pays that at start.
- **Sessions (0110)**: a revisit resumes (36.8k of 36.9k tokens cached). A shared ~2.4k-token system prompt was
  reused by the third session (1,984-2,048 tokens) but not the second (its first visit came before a fork mark
  existed). Eviction is exact: 46 evictions under a one-entry budget, every reply == fresh (sampled and greedy).
- **Known in production**: with one request at a time, a `stop` match ends the reply but the engine keeps decoding
  silently to EOS / `max_tokens` before the next queued request (0160).

## 4 x 256k batch production, 2026-09-28 (09:15-11:45)

**Incident first.** The 524k single-stream production (`config/prod-single.env.example` with `SESSION_GIB=12`) died at
~07:00-07:04: both kernel logs show `NVRM ... Out of memory [NV_ERR_NO_MEMORY]` (07:00:38-07:01:28 on the head node), rank 1
exited 137 and rank 0 exited; nothing restarted it until this window (~2 h down). Most likely cause: the 12 GiB
session store filling under the morning automations on top of 524k of caches. No watchdog was installed.

**What runs now** (`config/prod.env.example`, an image with every patch through 0220, built on
both nodes): 4 concurrent requests x 262,144 tokens each (`GLM53_TF_BATCH=4`), FP8 latent KV (0220), sessions inside
the batch slots (0180, `BATCH_SESSIONS=1`) with a **2 GiB** store, `PREFILL_ROWS_MAX=2048`, `LEAN_BLOCK=512`, the 0200
on-set. It is `config/prod-batch.env.example` with `SESSION_GIB` 4 -> 2 (see the stress row). The watchdog user timer is
installed on the head node (`glm53-tf-watchdog.timer`, `CONFIG=config/prod.env`, `WATCH_HEAL=1`).

### Why only 1 slot fit before, and what the memory goes to

- Every slot is allocated at full capacity at load (3.59 GiB a slot at 262k bf16, 2.1 GiB FP8). The load-time rule adds a
  slot while `cudaMemGetInfo free - slot - store budget >= BATCH_RESERVE_GB` on both ranks. On GB10 that free figure is
  MemFree: page cache counts as used. The "1 slot" runs had `BATCH_SESSIONS=1` with the 12 GiB store budget counted
  (3.6 + 12 + 4 GiB needed before the first extra slot). With the store off and 0140's O_DIRECT loads (no page cache
  from the weights), 4 bf16 slots fit without dropping caches (B1).
- The rest, per rank: weights 78 GiB, window buffers 8.5 GiB at `LEAN_BLOCK` 1024 (4.2 at 512), the lean set 3.1 / 1.55 /
  0.78 GiB at `PREFILL_ROWS_MAX` 8192 / 4096 / 2048 (batching prefills in 2048-token pieces, so more is never used),
  latent KV 3.31 GiB a slot bf16 / 1.86 FP8. Breakdown: `docs/MEMORY-4x256k.md`.
- The worker node (rank 1) is the binding node: 2 GiB less MemTotal, ~1.5 GiB lower in every measurement below.

### Phase 1: bf16 KV, image `z`, BATCH=4, CONTEXT=262144, store off (`results/B1-B3/`)

| | B1: rows 8192 | B2: rows 4096 | B3: rows 2048 |
| --- | ---: | ---: | ---: |
| slots | 4 | 4 | 4 |
| MemAvailable after load, r0 | 8.0 | 9.6 | 10.4 (r1 8.2) |
| minimum under the quick bench, r0 / r1 (GiB) | 5.7 / 4.4 | 6.7 / 5.0 | 7.5 / 5.9 |
| prefill 28.7k alone (tok/s) | 1,197 | 1,202 | 1,198 |
| 1 / 2 / 4 streams aggregate (tok/s) | - / - / 74-82 | 44-64 / 58-62 / 74-79 | 44-64 / 57-60 / 75-81 |
| batched == alone | 4/4 | 4/4 | 4/4 |
| stall: longest decode gap during a ~35k prefill | 2.0 s (2 decoders) | 2.1 s (3) | 2.1 s (3), TTFT 38.9 s |

B1 decode (tok/s): tf code greedy 61.5, chat greedy 47.1, kit structured 97.1, hashmap 49.4, essay 44.2; decode
after the 28.7k prompt 88. B3 stress, 4 x 112k prompts at once: minimum MemAvailable 6.3 / 4.8 GiB. bf16 at 4 x 262k
cannot reach 8 GiB of headroom with these knobs.

### Phase 2: FP8 latent KV, `config/prod-batch.env` (`results/K1/`, `results/K1-tests*`)

GPU tests on image `fp8kv`:

| file | result | note |
| --- | --- | --- |
| test_fp8_kv_patches | 4 pass, 6 fail, 16 errors | **the engine-level FP8 tests do not run**: the toy model has `kv_lora` 128 and 0220 only lays out 512 (`ValueError`). Kernel tests: FP8 attention != bf16 attention on the dequantized rows bit for bit (2 fails); FP8 rel. error vs the fp32 expanded reference 5.7e-2 against a 4e-2 bound (bf16: 3.4e-3) (2 fails). To fix in 0220 / its tests. |
| test_latent / test_1m | 20/20, 45/45 | |
| test_glue | 79/80 | `latent_tc` (off), as before |
| test_batch_sessions (0180 fixed) | 28/28 | was 4 failures |
| test_batch_parallel / test_session | 26/26, 32/32 | |
| test_batch2 | 40/44 | the same 4 as in T-runs before (fast-prefill admissions, per-sequence knobs) |

So FP8 was checked on the real model instead:

| | K1: FP8, store 4 GiB | production before (524k, single) | vLLM prod kit |
| --- | ---: | ---: | ---: |
| load | 150 s first (calibration re-measured), 35 s after | 36 s | |
| MemAvailable after load, r0 / r1 | 19.4 / 17.8 GiB | | |
| exact (drafted == serial) / batched == alone | 10/10 / 4/4 | 10/10 / - | |
| prefill alone 24.5k / 98k (tok/s) | 1,154 / 1,127 | 1,266 / 1,238 (28k / 112k) | 1,448 (28k) |
| decode tf code greedy / chat greedy / kit structured / hashmap / essay | 75.3 / 41.2 / 95.7 / 49.5 / 42.0 | 65.8 / 44.7 / 98.1 / - / - | 41.9 / 22.8 / 72.7 / 30.0 / 26.1 |
| concurrent streams 1 / 2 / 4, aggregate tok/s | 43-70 / 56-62 / 72-77 | one at a time | |
| stall: 3 decoders + a 39.8k prefill | gap 2.1 s, TTFT 45 s | queued behind | |
| slot resume (~40k sessions, 4 slots + a 5th) | revisits 2.5 s (39.8k cached); session 1 resumed after the 5th | | |
| MMLU-200 / refusals | 88.0% / 0/10 | 89.5% / 0 | |
| greedy replies vs bf16 KV (fp8ab `replies`, 20 prompts) | 5/20 identical; the rest diverge in the first 0-78 tokens, stay on topic (see below) | | |
| needle, fast prefill: 28k 3 depths x 3, 112k 3 depths x 1 | 9/9, 3/3 | 10/10 at 28k | |

**Memory stress** (`multiturn.py --modes stress`: 4 conversations grown together by ~60k-token turns, resumed in their
slots, store filling; then 3 decode 512 tokens while the 4th adds a 32k turn):

| run | sizes | fill | final 32k turn | MemAvailable min r0 / r1 | gate (>= 8) |
| --- | --- | ---: | --- | ---: | --- |
| K1 stress2, store 4 GiB | 251.6k, 251.9k, 251.9k, 250.2k | 1,049 s | TTFT 50.7 s, decode gap 2.9 s | 8.85 / **7.32** | fail |
| P1 production, store 2 GiB | 162.3k, 162.2k, 162.3k, 158.5k | 638 s | TTFT 49.2 s, decode gap 2.8 s | **16.06 / 14.44** | pass |

No NVRM OOM, no request errors, `/health` ok after both. Resumed turns show `cached` = the previous prompt (e.g. 187,264
of 251,562). K1's floor came after ~50 minutes of every other
benchmark on the same load (MemAvailable r1 went 17.8 -> 11.3 GiB during exact / concurrency / slots / 112k prefill /
decode, before the stress began), then sank ~0.3 GiB a stress round while the 4 GiB store filled. P1 ran on a fresh
production load (heal restart), so it does not show that drift: with the drift and a full 2 GiB store, the expected
long-uptime floor on the worker node is ~9.3 GiB (K1's 7.3 + the 2 GiB of store). Worth watching `MemAvailable` on the worker node over
the first days; the knobs if it goes under 8: `SESSION_GIB=1`, `BATCH_MAX_GRAPHS` below 256. After P1, a 98k prefill alone (1,099 tok/s,
decode after it 80 tok/s) and the `slots` run kept the floor at 15.9 / 14.3 GiB. Cost of the smaller store: in `slots`,
session 1 coming back after a 5th session was a cold 42 s prefill with 2 GiB (it resumed from the store with 4 GiB in K1);
revisits while it still holds its slot resume in 2.5 s either way.

Watchdog heal test: `docker kill glm53-tf-r0` at 10:53:40. The first heal (10:57) did nothing: the unit is a oneshot and
systemd killed the detached `serve.sh restart` with the tick's cgroup (empty `heal.log`). Fixed with `KillMode=process` in
`scripts/systemd/glm53-tf-watchdog.service`; the next heal (11:04:50) restarted both ranks from `config/prod.env`, ready
after 35 s, canary ok.

Not run in this window (time went to the stress runs): the 0190 per-request knobs `attn_bm32` and `mtp_window` on the
FP8 load, the `LEAN_BLOCK` 512 vs 1024 A/B (the ~4% lower prefill than B3's 1,198 at 28k is block 512 plus FP8 rows;
unseparated), and `followup,sessions` (the `slots` mode covered resume).

**What still needs real-use testing**: FP8 KV changes greedy replies from the first tokens on (expected). Reviewing
the bf16 replies against the FP8 replies (`bench/fp8ab.py --modes replies`; not in `results/`), the long-prompt FP8 replies read as
correct and specific as the bf16 ones, but that is a spot check. Please use it on real long agent sessions (past 100k:
recall of early details, tool-call formatting). Fallback without a rebuild: `GLM53_TF_KV_DTYPE=bf16` with
`SESSION_GIB=0`/`BATCH_SESSIONS=0` (worst case ~5-6 GiB on the worker node at 4 x 262k: under target), or 3 slots, or the
single-stream config (`config/prod-single.env.example`, `SESSION_GIB` <= 6).

Verified through the HTTPS reverse proxy after the switch (11:18): `/v1/models` lists
`GLM-5.3-Flash-EXL3`; a thinking reply returns `reasoning_content` and the answer (391, finish `stop`); a tool call returns
`get_weather({"city":"Paris"})` with finish `tool_calls`; streaming delivers the reply in chunks.

## W1: 0190 prefill knobs on the 4 x 256k production, 2026-09-28 (12:18-12:55)

Load: `config/prod.env` (image `fp8kv`, then `w1` = the same with 0190's new MTP prefill cache rows), `GLM53_TF_PROFILE=1`,
per-request `tf_knobs`, unique filler prompts, non-streaming (engine `prefill_s`, `decode_s`, `tokens_per_round`),
256-token greedy replies. JSON / logs: `results/W1/` on the head node (`ab*.json`, `memtest.log`, `conc.json`, `tests.log`).

| prefill tok/s | 24.5k | 98k |
| --- | ---: | ---: |
| production (`fp8kv`, knobs off) | 1,146-1,154 (4 runs) | 1,133 |
| + `attn_bm32` | 1,202-1,207 (3 runs; the first run, 1,141, compiled the tile) | 1,190 |
| + `mtp_window` 4096 | 1,156 | 1,130 |
| + both | 1,209 | 1,185 |
| `w1`, `GLM53_TF_MTP_PREFILL_CACHE=1` | 1,214 / 1,214 | 1,193 |
| `w1` cache rows + `attn_bm32` (**production now**) | **1,283 / 1,280** | **1,258** |

- Decode after the prompt: 83-91 tok/s cold and warm in every cell, tokens a round 7.29-7.5, the same drafter mix
  (5 MTP / 29-30 DFlash2 rounds) and the same reply hash everywhere: no knob changes decode.
- `mtp_window` does nothing here: batch mode prefills 2,048-token pieces, each a prefill of `prompt[:end]`, so its
  start `end - W` is below the piece for any W >= 2,048. Why it cost decode in Z2 (single stream): the zeroed head
  rows score exactly 0 in the head's indexer, above real pools with negative (signed-weight) scores, so the head
  attends to zero rows and its drafts get worse; larger windows only push that back. Not measured in single-stream
  mode this window (production is batch). Replaced by the cache-rows switch (docs/PATCHES.md, 0190 update): prefill
  never reads the head's outputs, so only its cache writes run; same bits, drafts unchanged. GPU tests 21/21
  (`-k "mtp_prefill_cache or mtp_window or mtp_absorb"`).
- `attn_bm32` + cache rows together: +11% at 24.5k and 98k (vs 1,448 for the vLLM kit at 28k).
- Memory (`memtest.py`, `attn_bm32` on, cache rows on): 3 x 98k prompts, then those 3 decoding 1,500 tokens each
  (resumed from their slots, 97,984 cached) while a 4th 98k prompt prefilled (TTFT 151 s): minimum MemAvailable
  **14.8 / 13.4 GiB** (r0 / r1); a single 98k prefill 16.6 / 15.0. Neither knob allocates memory.

**Batched decode, 4 streams** (`multiturn.py --modes concurrent --streams 1,4 --long-tokens 512`): aggregate 79 / 71
tok/s (1 stream 64 / 44). Per round: verify (forward, sampling, accept, commit) 95-139 ms, batched MTP drafting
4-5 ms (`mtp_batched` in 20-45% of rounds), so the verify forward is ~95% of a round. 70-90% of the rounds are
`eager` (keys met fewer than `CAPTURE_AFTER=3` times in a 25 s run; captures 9-22 a stream): the next cost to attack
is the eager rounds (+10-25 ms each per 0200's model), then the rows themselves (~6-7 ms a verify row: 4 x 2-7
rows). During a 98k prefill beside 3 decoders (memtest phase B), a decoder's 145 s went ~70 s to waiting through
prefill pieces, ~65 s to verify, ~3 s to drafting (2.0 tokens a round, 10 tok/s each).

## W2-W4: 0260 expert bench, 0230 RoCE, 0240 b12x, 0250 NVMe sessions, 2026-09-28 (13:00-13:44)

Images built from committed trees only: `w2` (through 0260 at 57d4631), `w3` (+ 0240), `w4` = `sessdisk` (+ 0250 at
74191b1). Logs: `results/W2`, `W3`, `W4`.

**0260 (`bench_experts.py 1024 2048 4096 8192`, one GPU, prod stopped; all kernels "same" bits, 48/48 tests).**
Sum of gate/up + down, ms (uniform routing; skewed in brackets):

| rows | fast2 | fat | once | once, no decode | once, no mma |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 9.4 [10.0] | 11.6 [11.9] | 11.6 [12.1] | 11.5 [12.4] | 11.5 [11.7] |
| 2048 (prod piece) | 10.9 [12.5] | 13.0 [14.2] | 13.0 [14.6] | 12.9 [14.7] | 12.9 [13.3] |
| 4096 | 16.9 [19.0] | 16.0 [19.0] | 16.7 [20.9] | 16.4 [19.7] | 15.8 [16.9] |
| 8192 | 31.1 [32.9] | 28.6 [30.2] | 32.1 [33.4] | 30.1 [31.2] | 22.5 [24.4] |

- `once` never beats `fat` (0.89-1.01x): do not use.
- At <= 2,048 rows (what batch production runs), removing the trellis decode or the MMA changes nothing (within 1%):
  the kernel is bound by data movement / scheduling (weight reads + member-row gathers), not by either. Also there
  **fast2 is 1.13-1.24x faster than fat** (the reverse of 8,192 rows): with `PREFILL_ROWS_MAX=2048` production
  should A/B `GLM53_TF_FAST_EXPERTS=fast2` end to end (bitwise equal, so a per-load swap is safe).
- At 8,192 rows: no-mma saves 21-27% (MMA-bound share), no-decode 3-5%: decode-sharing cannot pay; better MMA
  (tiling / tensor-core use) is where the kernel time is.

**0230 RoCE: stopped at step 2.** Unit tests 38/38 (one GPU, no RDMA). NIC loopback on the head node (both CX7 functions,
16k first size) timed out at sequence 312 on roceP2p1s0f1: "the flag HAS reached this host's memory (the GPU did not
observe it: sparkring #278's signature)". That is the failure mode the design must not have (GPU-side visibility of
RDMA-written host memory); the 2-node bench / fault / soak and the engine A/B were not run. To investigate before
the next attempt: the kernel's acquire load path on pinned host memory (volatile / `ld.acquire.sys` vs caching),
single-HCA loopback (`GLM53_TF_ROCE_HCAS=1`). A `/cache/roce-failed` marker already existed (06:17, earlier run).

**0240 b12x: not adopted.** `bench_b12x.py --sweep`: KDA 0.94x of fast_kda (slower; no BV / warps / stages setting
beats the default tf32 BV 64 w4 s1 at 1.37-2.05 ms, stages 2-3 exceed shared memory), fused mhc 0.69-0.74x (slower;
fused == unfused bitwise True), one-pass sparse attention 1.47x (bf16 and FP8). Tests 39 pass / 14 fail: resumes never
hit with b12x bits on (`cached` 0: drafted == serial / resumed == fresh, snapshot control), state differs across
C / overlap for bits 3, FP8 one-pass attention row subsets not bitwise, mhc vs float64 3.8e-4 (bound 1e-5), mhc vs
today 3e-2, a KDA dispatch test error. End to end on the production load (`w3`, per request): 24.5k: b0 1,275 / 1,278,
b1 1,163, b2 1,112, b4 1,290 / 1,322, b7 1,145; 98k: b0 1,248, b4 1,303 (+4.4%). Decode unchanged. Only bit 4 helps
(+1-4%), below the 5% bar, and its FP8 row-independence test fails (production is FP8 KV).

**0250 NVMe session tier: adopted.** GPU tests 30/32 (the 2 failures: the FP8 engine tests cannot run on the toy
model's 128-wide latent, as in 0220). `sessdisk bench` on the head node's NVMe: 40k write 0.30 s / read 0.14 s, 100k 0.60 /
0.28 s, exact. Load with the tier on (prod config): 6 sessions of 39,226 tokens A..F (31 s cold each) on 4 slots +
2 GiB RAM, then A, B: 38,912 cached, prefill 0.42 s, same reply hash; server restarted, then C, F: 20 entries indexed
in 0.2 s, 38,912 cached, 0.43-0.45 s, same hashes. `exact` 10/10. MemAvailable after it 18 / 16 GiB. Production
now runs `glm53-tensorfold:sessdisk` with `GLM53_TF_SESSION_DISK=/sessions`, 64 GiB; verified through https
(13:43: models, thinking reply 391, tool call `get_weather({"city":"Paris"})`, streaming), watchdog re-armed.

## W5: fast2 vs fat end to end (0270), round buckets for batched graphs (0280), 2026-09-28 (13:57-14:24)

Image `glm53-tensorfold:w5` (every patch through 0280, built on both nodes), loads from `config/prod.env` with
overrides (`results/W5/load.sh`). Prod down 27 min. Logs and JSON: `results/W5/`. **Nothing adopted: production
stays `glm53-tensorfold:sessdisk` / `config/prod.env` unchanged**, restored 14:23, checked through https (models,
`17*23` -> `391`, canary), watchdog timer re-armed, lease deleted.

**Why fat was picked (history).** 0170 made `fat` the production kernel on the 8,192-row lean chunks of that time:
+3-5% end to end (RESULTS "Fat experts (0170)"). At 8,192 rows fat is still the faster kernel (W2: 28.6 vs 31.1 ms).
`PREFILL_ROWS_MAX` went to 2,048 with the 4 x 256k batch production, where W2 found fast2 1.13-1.24x faster in
isolation. Decode never runs either kernel: fast2 / fat / once serve fast prefill chunks only (`exl3_mm.routed(...,
fast=True)`); decode and verify windows use the row-invariant grouped kernels. So there is no decode-side reason for
either; the rows that matter are fast-chunk sizes (multiples of 64 up to 2,048: batch pieces, prompt tails, short
prompts).

**Kernel sweep** (`bench_experts.py`, one GPU, gate/up + down ms, uniform [skewed]; 1-40 rows listed for the decode
question only; 1024-8192 from W2; fast2 here reads the shared rotated input, as 0270's auto does):

| rows | 1 | 8 | 40 | 64 | 128 | 256 | 512 | 1024 (W2) | 2048 (W2) | 4096 (W2) | 8192 (W2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast2 | 0.29 | 1.66 [1.50] | 5.62 [4.40] | 6.61 [5.28] | 8.03 [6.92] | 8.37 [8.17] | 8.69 [8.96] | 9.4 [10.0] | 10.9 [12.5] | 16.9 [19.0] | 31.1 [32.9] |
| fat | 0.28 | 2.01 [1.90] | 7.23 [5.63] | 8.28 [7.02] | 10.10 [8.72] | 11.15 [10.19] | 10.91 [11.04] | 11.6 [11.9] | 13.0 [14.2] | 16.0 [19.0] | 28.6 [30.2] |

fast2 wins from 8 to 2,048 rows (1.19-1.33x), ties at 1 row, loses from ~4,096. All bits "same". So 0270's
`GLM53_TF_FAST_EXPERTS=auto` (fast2 below `GLM53_TF_FAST2_ROWS` hi = 4096, fat at and above) was the candidate.

**End to end, the kernel win does not transfer** (one load, `FAST_EXPERTS=auto`, per request `tf_knobs.fat_experts`
1 = fat, 2 = auto, 0 = plain fast2; `results/W5/ab1.log`; unique prompts, cold prefill):

| prefill tok/s | 24.5k (2 runs) | 98k |
| --- | ---: | ---: |
| fat (production) | 1,284 / 1,278 | 1,259 |
| auto (fast2 on rot_in1 below 4,096 rows) | 1,251 / 1,253 (-2.2%) | 1,232 (-2.1%) |
| fast2 (0170's plain path, two rotations) | 1,224 / 1,222 (-4.6%) | 1,199 (-4.7%) |

Same reply hash everywhere; decode after the prompt 80-90 tok/s in every cell. With production's pipelined lean
chunks (`PREFILL_OVERLAP=1`: all-gathers and hc slabs on another stream while the experts run) fat is faster in the
engine even though fast2 wins the isolated kernel by ~2 ms a layer. Likely cause (not profiled): fast2's gate/up is
register-bound at one CTA an SM (224 registers) and competes worse with the overlapped kernels than fat's 2 CTAs an SM
at 128 registers. auto on the same load: `exact` 10/10, tf suite code/chat sampled 40.3 / 38.7, greedy 72.6 / 40.5
(unchanged kernels for decode). **0270 not adopted** (-2%, gate was +3%); it stays in the tree, off, as a tool
(bitwise-tested, per-request `fat_experts=2`). MMLU was not run (nothing to adopt; fat == auto bit for bit anyway).

**Batched decode graphs (0280).** 4 streams, `multiturn.py --modes batchexact,concurrent --streams 4 --reps 3
--long-tokens 512`, all on image w5 with fat:

| load | aggregate tok/s (3 reps) | mean | rounds graph / eager / capture | verify ms a round | batchexact / exact |
| --- | --- | ---: | --- | ---: | --- |
| production knobs (buckets off) | 78.6 / 70.3 / 73.7 | 74.2 | 8-33% / 49-82% / 8-10% | 105-115 | 4/4 / 10/10 |
| `GLM53_TF_BATCH_BUCKETS=4,8` (tied routing) | 73.6 / 67.5 / 70.8 | 70.6 (-5%) | 80-91% / 6-12% / 3-5% | 114-121 | 4/4 / 10/10 |
| `GLM53_TF_BATCH_GRAPHS=0` (every round eager) | 80.7 / 74.3 / 75.5 | 76.8 (+3.5%) | 0 / 100% / 0 | 103-110 | - |

- Buckets did what they were built for: graph replays went from 8-33% to 80-91% of rounds, byte-identical output
  (batchexact 4/4, exact 10/10, GPU tests 9/9). But each round got ~6-9 ms slower: ~1,750-1,980 padded rows a run
  (~2.7 a slot and round). Even routed to their window's last row (no new expert reads) a padded row costs ~0.8-1 ms
  (KDA chain steps, attention, dense rows). Not adopted (-5%, gate was +10%).
- The decisive number is the third row: with no batched graphs at all the aggregate is the same or slightly higher.
  So W1's premise ("eager rounds cost +10-25 ms each") does not hold at these shapes: a 4-slot verify of ~10-20 rows
  is GPU-bound at ~100-120 ms, the host enqueues the eager kernels faster than the GPU runs them, and a replay saves
  ~nothing, while captures cost a full extra forward each (8-10% of rounds). No bucketing / padding / capture policy
  can reach +10%: the ceiling of "every round a graph for free" is ~0 here. `GRAPHS=0`'s +3.5% is within the
  rep-to-rep spread (70-81) and was not A/B'd for exactness or memory; not adopted. A cheaper knob to try in a later
  window: `GLM53_TF_BATCH_CAPTURE_AFTER` higher (fewer captures) or `GLM53_TF_BATCH_GRAPHS=0`, both with batchexact.
- What would move 4-stream throughput instead: the verify rows themselves (~6-7 ms a real row, mostly routed-expert
  reads that rows of different sequences rarely share) and fewer rejected draft rows (per-stream tokens a round
  2.0-2.5 for the MTP-heavy streams). 0200's "one launch per layer for per-slot KDA / attention" would only remove
  launches, which this window shows are not the bottleneck.
- Memory: not measured for 0280 (not adopted). Graphs are few with buckets (fewer than without), so it would not
  have been the constraint.

Tests (the worker node, image w5): `test_fast_experts_auto_patches` 29/29, `test_batch_buckets_patches` 9/9,
`test_batch_parallel_patches` 26/26; host-only parts of `test_mia_prefill`, `test_expert_once`, `test_knob`,
`test_batch2`, `test_batch_sessions` 112 passed (GPU parts skipped there).

## Prefill work

Fast prefill (patches 0080-0084, knobs 0091-0093) on the latent-KV load (`CONTEXT=262144`, q4mse, expert loop,
real calibration, cost depths, lookup, bf16 gathers). Prefill tok/s = prompt tokens / cold TTFT (unique prompt, no
cache hit), 1.8k / 7k / 28k / 112k-token prompts. Every fast run follows a warm-up pass (the first fast request of a
row bucket compiles Triton kernels: F4's 2k cell took 49 s). JSON in `results/F*-ctx*.json`, per-request profiles
in `results/profiles/`.

| Config | 1.8k | 7k | 28k | 112k | warm TTFT 28k |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline: exact, 1024-row chunks (L1) | 498 | 625 | 660 | 582 | 42 s |
| exact + 0065 (blocked index selection), 1024 (F5) | 649 | 662 | 659 | 638 | 43 s |
| fast, 1024 (F5: chunked KDA, exact-order qmm tiles, fused experts) | 754 | 782 | 767 | 772 | 0.9 s |
| fast, 1024, + one-accumulator matmuls, hc_pre row tiles (F9) | | 837 | 863 | 852 | 0.7 s |
| fast, lean 2048 (F9) | | 866 | 892 | 871 | 1.8 s |
| fast, lean 4096 (F9) | | 892 | 908 | 890 | 4.0 s |
| fast, lean 8192 (F9) | | 913 | 920 | 898 | 4.0 s |
| lean 8192 + `fp8_prefill` (F9) | | 962 | 966 | 944 | 3.8 s |
| lean 8192 + `prefill_overlap` (F9) | | 974 | 982 | 961 | 3.8 s |
| **lean 8192 + fp8 + overlap (F9; the defaults left running, F10: 1,032 at 28k)** | | **1,024** | **1,034** | **1,013** | 3.6 s |
| 1M context load, fast 1024 (F6; `CONTEXT=1000000`, `PREFILL_ROWS_MAX=1024`) | | | 863 | 848 | 0.7 s |
| 1M context load, exact 1024 (F6) | | | 673 | 656 | 42 s |
| vLLM prod kit | 960 | 1,340 | 1,448 | | |

Warm TTFT: a follow-up that extends the prompt resumes from the last grid point, so it re-prefills up to one chunk
(C - 1 tokens): ~1 s at C = 1024, 4-7 s at C = 8192 (7k prompt: 6.8 s). `tf_knobs.prefill_rows` picks C per request.

Profile of a 28k prompt (s, rank 0; F9 and L1):

| Component | exact 1024 (L1/F5) | fast 1024 (F9) | fast lean 8192 | 8192 + fp8 + overlap |
| --- | ---: | ---: | ---: | ---: |
| routed experts | 14.6-15.0 | 12.3 | 9.9 | 10.0 |
| all-gathers | 4.6-5.7 | 3.0 | 3.0 | 0.0 (overlapped) |
| DSA sparse attention | 3.1 | 3.0 | 3.1 | 2.3 |
| KDA chain | 3.5 | 1.8 | 1.8 | 1.8 |
| KDA projections | 3.1 | 1.9 | 1.9 | 2.2 |
| hyper-connections | 2.2 | 1.9 | 2.0 | 2.9 (hc slabs, incl. gather waits) |
| DSA o-proj | 2.2 | 1.9 | 1.9 | 0.7 |
| DSA indexer | 1.5 (0.5 with 0065) | 0.4 | 0.4 | 0.4 |
| MTP head absorb | 1.3-1.4 | 1.3 | 1.3 | 1.3 |
| total GPU | 42.5 | 32.4 | 30.4 | 27.0 |

At 112k (8192 + fp8 + overlap, 110 s): routed experts 39.3, hc 11.5, sparse attention 9.7, KDA proj 8.8, KDA chain
7.2, MTP 5.4, router 5.0, shared expert 4.9, indexer 4.8 (26 s before 0065).

Exactness and quality with fast prefill: `--suites exact` 10/10 identical with fast on (F5, lean 2048) and with
lean 8192 + fp8 + overlap (F9). MMLU-200: exact latent (Q3) 88.5%, fast 2048 (Q4) 88.5% (5 answers differ from
Q3), lean 8192 fp8 off (Q5-0) 89.0%, fp8 on (Q5-1) 88.5% (5 answers differ from fp8 off); refusals 0/10 everywhere.
Decode is unchanged (F5 fast vs L1: tf/kit/edit cells within run-to-run spread).

## W6: patch 0290 shared KV pool (2026-09-28 15:27-16:35) — adopted

Image `glm53-tensorfold:kvpool` (0290 on the prod set), FP8 KV, BATCH=4. Load A: pool 1,048,576, CONTEXT 262,144.
Load B/C: pool 1,048,576, CONTEXT 1,048,576, `GLM53_TF_KV_POOL_CHECK=1`. Files: `results/W6/`.

| Check | Production (sessdisk, 4 x 262k) | Pool, CONTEXT 262k | Pool, CONTEXT 1M |
| --- | ---: | ---: | ---: |
| Prefill 24.5k (tok/s) | 1,278-1,284 | 1,258 / 1,259 | 1,252 |
| Prefill 98k (tok/s) | 1,259 | 1,252 / 1,253 | 1,250 |
| Reply sha (ab.py) | 8794a3463259cc2f | same | same |
| exact / batchexact | 10/10 / 4/4 | 10/10 / 4/4 | 10/10 / - |
| 4-stream aggregate (tok/s, 3 reps) | 70-81 (W5 spread) | 76.9 / 70.5 / 73.8 | - |
| MMLU-200 / refusals | 88.0% / 0 | - | 88.0% / 0 |
| Max context a request | 262,144 | 262,144 | 1,048,576 |

- Needle at 357,820 prompt tokens in one slot: found; prefill 1,034.6 tok/s (345.8 s), decode 54-59 tok/s; the
  follow-up resumed 357,760 cached tokens (prefill 0.40 s).
- Stress 4 x ~300k (1.22M tokens against a 1.05M pool, so spills/waits happen): completed in 1,162 s, longest decode
  gap 6.8 s, MemAvailable minimum 15.11 / 14.05 GiB (head / worker) -> PASS (>= 8). health: 0 errors.
- GPU tests: test_kv_pool_patches 22/23 first run; the failure (`test_gpu_disk_sessions_into_pooled_slots`) was a
  test bug (an identical prompt never resumes by design); fixed to extend the prompt on round 2, rerun 23/23.
  test_latent_patches 20/20, test_1m_patches 45/45.
- Cost: prefill -1.5 to -2% at 24.5k, -0.5 to -0.7% at 98k (over the 1% bar at 24.5k); accepted for 4x the per-request
  context at the same memory.
- Adopted: prod.env IMAGE=kvpool, CONTEXT=1048576, GLM53_TF_KV_POOL_TOKENS=1048576 (pool check off). Restarted 16:3x,
  ready in 40 s, canary ok (decode 76.3 tok/s), https chat ok, MemAvailable 18 / 17 GiB idle. The benchmark driver died of an
  API error during load B; the window was finished by hand.

## W7: profile (2026-09-28 16:53-17:13, prod down 20 min) — measurement only, nothing adopted

Per-stage breakdown of prefill (one 2,048-row piece, whole 21,464- and 85,781-token prompts = the "24.5k" / "98k"
cells) and decode rounds (1 stream, 4 streams) on the production config, both ranks, with nsys (CUDA + NVTX marks from
the `GLM53_TF_PROFILE` probe sites); piece / chunk size sweep; the 0230 RoCE loopback rerun. Full tables, method and
conclusions: **`docs/PROFILE.md`**. Files: `results/W7/` (scripts, logs, `tables.md`, `analysis/*.json`); the four
nsys reports (`pf24-r{0,1}`, `mix-r{0,1}` = 98k + decodes) are on the head node in `~/w7-traces/` (not in git, 15-106 MB).

**Prefill, % of wall (both ranks within 0.5 point of each other; 24.5k / 98k):** routed experts 26.8 / 24.8, DSA/MLA
attention 25.9 / 28.4 (sparse attention 12.1-12.3, latent expand + o_proj 8.7, q/kv 3.5, indexer 1.6 / 3.9),
KDA 19.5 (projections 11.5, recurrence 8.0), hyper-connections 13.0 / 12.9, shared expert + dense MLP 5.5-5.8,
router + combine 5.0, DFlash2 taps 1.0, MTP cache rows 0.4, **NCCL exposed 0.8, GPU idle 1.2-1.3**, memcpy 0.4.
The all-gathers (4,038 / 15,541 of 4 MiB, RING_LL, ~456 us each) run 94% beside compute, but the kernels beside them
pay an **overlap tax of 8.2-8.7% of the wall** (`_hc_post` 139 us alone vs 413 us beside an all-gather): communication
costs ~9% of prefill in total. `prefill_overlap: 0` costs -6.4% / -5.8%. Ranks are balanced (each waits 2-2.3% of the
wall inside all-gathers, jitter; per-piece GPU time equal to 0.01 s).

**Decode round, kernel ms (1 stream 59.5 ms / 4 streams 125.5 ms under capture; 54-60 / 113-122 ms without):** routed
experts 26.4 / 72.3 (44% / 58%), dense q4 GEMMs 16.3 / 23.0, NCCL 4.8 / 7.7-8.3 (fully exposed; 100 / 120 all-gathers,
half of it waiting for the peer), attention 2.6 / 5.2, KDA 1.6 / 5.5, hc 1.5 / 1.8, idle 4.5-7.3 ms; verify forward
82-83% of a round, drafting 8% / 11%.

**Piece / chunk sweep, one active request (cold prefill tok/s, 24.5k / 98k):**

| load | chunk rows | 24.5k | 98k | vs 2,048 |
| --- | ---: | ---: | ---: | ---: |
| B: `BATCH_PIECE=8192`, `PREFILL_ROWS_MAX=8192` | 2,048 | 1,266 | 1,252 | base |
| B | 4,096 | 1,310 | 1,294 | +3.5% / +3.4% |
| B | **8,192** | **1,339** | **1,319** | **+5.8% / +5.4%** |
| B | 2,048, `prefill_overlap: 0` | 1,184 | 1,178 | -6.4% / -5.8% |
| C: `BATCH_PIECE=4096`, `PREFILL_ROWS_MAX=4096` | 4,096 | 1,307 | 1,290 | +3.2% / +3.0% |
| prod (A1, nsys attached, idle) | 2,048 | 1,264 | - | |

Same reply sha in every cell (`8794a3463259cc2f`), `exact` 10/10 on load B (8,192-row chunks). The gain is the chunk
size, not the piece (B at 4,096 rows == C within 0.3%). Cost: lean chunk buffers 0.78 -> 1.55 -> 3.10 GiB a rank;
MemAvailable minimum 14.97 / 13.60 (prod config, 98k + decodes), 15.91 / 14.71 (C, one 98k), 13.38 / 12.17 (B sweep),
12.38 / 11.18 (B incl. 4-stream decode + exact); worst case (4 full slots) at 8,192 not measured. A larger piece only
while one request is active is exact but needs a scheduler change (`GLM53_TF_BATCH_PIECE` is fixed at load) and the
8,192-row buffers loaded regardless (docs/PROFILE.md section 6).

**0230 RoCE loopback rerun (image kvpool, W2's harness): same failure, same place** — timeout at sequence 312 on
roceP2p1s0f1 with the same "flag HAS reached this host's memory" diagnosis (`results/W7/roce-loopback.log`). So it is
deterministic, not a transient. 312 = 1 + 10 + 300 + 1 is the first collective of the harness's single-rank graph
warm-up, which matches 0350's explanation (f45b7ef: the harness, not GPU visibility). With 0350's harness
(`tests/cuda/bench_roce.py` at f45b7ef, md5 86ab4277, run from a copy against the image's 0230 runtime; RoCE not
enabled in the server) the same loopback completes 21,044 exchanges with no timeout, bit-exact at 16k / 64k / 128k
(graph 11.3 / 17.7 / 26.9 us), but **`bits_equal: false` at 1 MiB** (97 us), above the engine's 256 KiB default
`GLM53_TF_ROCE_MAX_KB`: a separate data problem to look at before any size above 256 KiB goes over RoCE
(`results/W7/roce-loopback-0350harness.log`).

**Conclusions (docs/PROFILE.md section 7):** the largest lever is expert GEMM efficiency (routed experts 25-27% of
prefill and 44-58% of decode rounds, dense q4 GEMMs next); attention is second and grows with context; communication
is ~9% of prefill but nearly all overlap tax, so pipeline parallelism (which would roughly halve single-stream decode)
is not worth it; the cheapest win is the chunk size (+3.4% at 4,096, +5.4-5.8% at 8,192, same bits, +0.8 / +2.3 GiB a
rank); host gaps are ~1%.

Ops notes: nsys needs `--cap-add SYS_ADMIN` here (`RmProfilingAdminOnly: 1`) and `--trace=cuda-sw` for more than one
capture; a second `nsys start` in the same session lost the agent and took both ranks down (load A1, 16:55; the
captured trace was intact). Prod restored 17:13 from `config/prod.env` (image kvpool, standard entrypoint), ready in
25 s, canary ok (decode 74.1 tok/s), https `/v1/models` ok, `17*23` -> `391`, watchdog timer re-armed, lease
refresher and memory sampler stopped, lease deleted.

## W8: batch 2 (0300, 0310, 0320, 0330, 0335 + chunk size), 2026-09-28 (17:30-18:25, 18:26-19:15) — adopted

Image `glm53-tensorfold:b2` = every patch through 0360, built on the head node and loaded on the worker node (after the 0330 loader fix
below). Off == production: `tests/kvpool_ptx.py` against `kvpool` finds 33 of 33 Triton kernels' PTX identical, and load
A (b2 with only the request log on) gives kvpool's reply sha, exact 10/10 and batchexact 4/4. Every load is
`config/prod.env` plus overrides (`results/W8/load.sh`); per load `results/W8/run.sh` = exact, batchexact, `ab.py`
24.5k / 98k twice (cold prefill tok/s, reply sha, decode after the prompt), `multiturn.py --modes concurrent --streams
1,4 --reps 3`. Two windows, prod restored in between (18:25, ~1 min before the second window). Files: `results/W8/`.

**GPU unit tests** (`results/W8/tests-head*`, `tests-worker*`):

| test | result |
| --- | --- |
| 0320 `test_prefill_pp_patches` (incl. the one-GPU row-split kernel identity test on real shapes, fused hc 0 / 3) | 15 passed; **the kernels are bitwise on half sub-blocks, so the row split is usable** |
| 0320 two engines on one GPU (`PP_TWO_PROC=1`) | failed first: the test's host-staged gloo communicator synchronizes inside the engine's decode-graph capture (`cudaErrorStreamCaptureUnsupported`), a test bug; fixed (no decode graphs in that test), then passed |
| 0310 `test_prefix_share_patches` | 31 passed on the rerun; the first full run hit an illegal memory access while building a reference engine (prefix share off, in the Graphs warm-up) after 28 passes; not reproduced in two more runs (one with `CUDA_LAUNCH_BLOCKING=1`). Unexplained, recorded |
| 0335 `test_solo_piece_patches` | 12 of 14. `test_second_request_turns_pieces_normal` needed a 4,096-token test context (fixed); its greedy case still ends with every piece solo (the newcomer was admitted after the toy prompt finished: timing, the replies are equal). `test_lazy_xu_engine_same_reply` fails (Xu allocated anyway under the test's default fast2): `LEAN_LAZY_XU` not used |
| 0300 `test_request_log` | 12 passed |
| regressions: overlap 149, kv_pool 23 passed; batch2 `fast_prefill_admissions` 2 failed | the batch2 failure (grid stat 64 != 128) is the same on image kvpool: a stale test expectation, not a regression |
| 0330 tc kernels (under `timeout`) | **the extension did not build**: the base image's `TORCH_CUDA_ARCH_LIST` includes 8.0, and ptxas refuses mbarrier / cp.async.bulk below sm_90. Fixed in patches/0330 (`_tc_ext` builds for the device's arch only). Then: **cfg 1 / 2 (544-thread CTAs) cannot launch on GB10** ("does not fit an SM": 17 warps x 120 registers need 5 warps x 3,840 = 19,200 registers on one SM sub-partition, which has 16,384). cfg 3 (fat's tile, 288 threads): bit-identical to fast2 / fat at 64-8,192 rows, ticket on / off, CTA cap; no hang |

**Expert bench** (`bench_experts.py 2048 4096 8192 --tc --contend`, gate/up + down ms, isolated / contended, uniform
routing; only cfg 3 runs):

| rows | fast2 | fat (prod) | tc cfg 3 | tc vs fat isolated / contended |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 11.03 / 12.99 | 13.18 / 15.24 | 11.11 / 13.62 | 1.19x / 1.12x (skewed 1.07x / 1.06x) |
| 4,096 | 40.41 / 19.76 | 15.96 / 18.10 | 18.68 / 21.30 | 0.85x / 0.85x |
| 8,192 | 30.92 / 35.05 | 27.70 / 28.79 | 35.68 / 38.85 | 0.78x / 0.74x |

All bits "same". Like fast2 in W5, tc cfg 3's 2,048-row kernel win did not survive end to end (load C below).

**Per-feature loads** (prefill tok/s, cold, two runs; every cell reply sha `8794a3463259cc2f`, exact 10/10 and
batchexact 4/4; decode = `multiturn.py` aggregate tok/s, 3 reps):

| load | 24.5k | 98k | vs A | 1 stream | 4 streams |
| --- | --- | --- | ---: | --- | --- |
| prod (kvpool, live, before the window) | 1,265.5 / 1,269.4 | 1,254.1 / 1,254.1 | +0.5% / +0.5% | 60.9 / 43.4 / 47.7 | 78.5 / 71.4 / 53.2 (a stray request) |
| A: b2 + request log (control) | 1,259.7 / 1,262.7 | 1,247.9 / 1,248.0 | base | 62.8 / 45.1 / 47.7 | 79.0 / 71.9 / 74.1 |
| B: A + `PREFILL_PP=1` + `PREFIX_SHARE=1` | 1,348.4 / 1,358.9 | 1,348.3 / 1,349.6 | **+7.3% / +8.1%** | 64.0 / 44.7 / 47.6 | 78.1 / 71.6 / 74.0 |
| C: A + `FAST_EXPERTS=tc`, `TC_CFG=3,3` | 1,210.3 / 1,210.8 | 1,198.5 / 1,192.2 | -4.0% / -4.2% | 63.9 / 45.0 / 47.4 | 79.0 / 71.3 / 73.2 |
| D: A + `PREFILL_ROWS_MAX=8192`, `SOLO_PIECE=8192`; per request 8,192 rows | 1,343.1 / 1,341.9 | 1,322.4 / 1,319.4 | **+6.4% / +5.8%** | 63.5 / 45.2 / 47.2 | 77.0 / 70.7 / 72.0 |
| D, per request `prefill_rows: 4096` | 1,310.4 / 1,309.0 | 1,295.4 / 1,275.1 | +3.8% / +3.0% | | |
| D, per request `prefill_rows: 2048` | 1,265.7 / 1,257.9 | 1,253.5 / 1,254.2 | 0 / +0.5% | | |
| **E: A + PP + PREFIX_SHARE + ROWS_MAX / SOLO_PIECE 8192 (combined)** | **1,468.4 / 1,473.4** | **1,446.9 / 1,451.1** | **+16.6% / +16.1%** | 63.1 / 45.0 / 47.0 | 77.7 / 70.7 / 72.7 |

- 0300 request log: A vs live prod -0.5% at both lengths, inside the boot-to-boot spread (the log's work runs on the
  HTTP thread after the reply and cannot touch the engine's `prefill_s`); one line a request (177 lines by load C's
  start), no text. Adopted.
- 0320 row split: +7.3-8.1%, twice the estimate (+3-5%); at 8,192-row chunks (E) it compounds with the chunk gain
  (+6.1% x +7.7% would be +14%; measured +16%). Adopted.
- 0330 tc: -4%, not adopted (and cfg 1 / 2 cannot run on GB10; a 544-thread design needs <= 96 registers a thread or
  <= 16 warps a CTA). The patch stays, off.
- Chunk size: +6.4% / +5.8% at 8,192 (W7: +5.8 / +5.4), +3.8 / +3.0% at 4,096. The per-request 2,048 cell inside load D
  equals A, so the gain is the chunk. With `SOLO_PIECE` the 8,192 pieces run only for a lone request; 4-stream decode
  keeps 2,048 pieces (aggregate within the 70-81 spread). Adopted at 8,192 via 0335 (0335's solo rule is what keeps
  pieces at 2,048 beside decoders; the piece size itself does not matter, W7).
- Decode after the prompt (`ab.py`, "count to 100") 80-89 tok/s in every load; single-stream and 4-stream aggregates
  within rep-to-rep spread everywhere.

**0310 shared prefixes** (`bench/prefixshare.py`; "12k" / "20k" arguments give ~19.4-20.3k and ~31.7k-token prompts:
the system prompt is ~18k / ~29.5k tokens; A = today, B = PP + share, E = combined):

| case | A (off) | B (on) | E (on, combined) |
| --- | --- | --- | --- |
| 12k: session 2 cached / TTFT | 16,384 / 3.25 s | **17,920 / 2.24 s** | 17,920 / 2.13 s |
| 12k: session 3 cached / TTFT | 17,152 / 2.27 s | 17,920 / 2.22 s | 17,920 / 2.11 s |
| 12k: burst of 4, wall / cached | 72.4 s / 0-896 | **28.2 s** / 3 of 4 at 17,920 (waits 8-9 rounds) | 24.8 s / 3 of 4 at 17,920 |
| 20k: session 2 cached / TTFT | 16,384 / 13.88 s | **29,376 / 2.21 s** | - |
| 20k: burst of 4, wall | 114.1 s | **35.7 s** (3 of 4 at 29,376) | - |
| resumed == fresh (draft off, fresh prefill) | 7/7, 7/7 | 7/7, 7/7 | 7/7 |

The expected cached tokens (system prompt rounded down to 64 and the template tokens) appear, and replies are identical.
Adopted.

**Combined config E: worst case and quality** (`results/W8/runE2.sh`, after E's gates and prefix bench, so the session
store and allocator caches were already warm):

- 4 x ~250k stress (`multiturn.py --modes stress --stress-target 250000`; decoders at 252.8k, the 4th prefilling 32k):
  filled in 845 s, final TTFT 49.0 s, longest decode gap 4.2 s; **MemAvailable minimum 9.23 / 8.14 GiB** (head /
  worker; final phase 9.45 / 8.42) -> PASS (>= 8), no OOM in dmesg, health 0 errors. The margin on the worker node is thin:
  W6's 2,048-row stress had 14.05. Of the ~6 GiB, 2.3 is the 8,192-row lean buffers; the rest is the drift of a load
  that has served (session store full, allocator caches after 98k-300k prompts: every load in this window drifts from
  ~18 / 17 after boot to ~13 / 12 at 2,048 rows and ~11 / 10 at 8,192). If memory ever runs short, the first step back
  is `PREFILL_ROWS_MAX` / `SOLO_PIECE` 4,096 (+0.8 GiB instead of +2.3, keeps +3-4%).
- Needle, one prompt of 314,262 tokens (the "350k" argument of `needle.py`) alone after the stress (pool holding the
  stress sessions, solo 8,192-row chunks): **found**, prefill 1,287 tok/s (W6 at 358k: 1,035), decode 58.9 tok/s; the
  follow-up resumed 314,240 cached (0.25 s). MemAvailable minimum during it 9.28 / 8.35.
- MMLU-200 **88.0%** (176/200, same as W6), refusals 0/10; exact 10/10 and batchexact 4/4 again after the stress.

**Adopted:** `config/prod.env` -> `IMAGE=glm53-tensorfold:b2`, `GLM53_TF_PREFILL_PP=1`, `GLM53_TF_PREFILL_ROWS_MAX=8192`,
`GLM53_TF_SOLO_PIECE=8192`, `GLM53_TF_PREFIX_SHARE=1`, `GLM53_TF_REQUEST_LOG=/sessions/requests.jsonl` (fat experts
stay). Prod restarted 19:14 on it: ready in ~1 min, canary ok (decode 76.3 tok/s), warm-up 16,384 prefill 10.98 s
(kvpool: 12.57 s), https `/v1/models` ok, `17*23` -> `391`, watchdog timer re-armed, lease refresher and memory
sampler stopped, lease deleted. MemAvailable idle after start 15 / 13 GiB. The NVMe session store starts cold (the
image id is in 0250's compat hash).

Ops notes: `pkill -f results/W8/lease.sh` inside an `ssh '...'` command also matches (and killed) that ssh's own shell;
kill by pid instead. The canary's tokens-a-round can differ between loads (5.00 vs 5.71 on load E: the drafter cost
calibration at load differs by ~0.2 ms); its replies are exact.

## W9: batch 3 (0310 IMA, memory margin, 0350 RoCE, 0360 b12x bit 4, per-slot verify cost), 2026-09-28 (19:23-20:26, 20:36-21:20, 21:30-22:25, 22:35-23:26) — RoCE, b12x bit 4 and 4,096-row chunks adopted

Image `glm53-tensorfold:b2` throughout (no rebuild: only test harnesses changed). Loads from `config/prod.env` plus
overrides (`results/W9/load.sh`). Four windows, prod restored and verified in between (https, `17*23` -> `391`,
canary, watchdog timer active, lease deleted). Files: `results/W9/` (scripts, logs, JSONs; `SUMMARY` files per step).

### 1. The 0310 illegal memory access (W8): not a production bug; PREFIX_SHARE stays on

`tests/cuda/test_prefix_share_patches.py`, whole file, 22 runs in W9:

| condition | runs | IMA |
| --- | ---: | ---: |
| warm caches (the worker node) | 6 | 0 |
| cold Triton cache (`TRITON_CACHE_DIR` in the container, 43 s runs) | 3 | 0 |
| fresh empty `/cache` volume (extensions, Triton, calibration built in the run), alone, `CUDA_LAUNCH_BLOCKING=1` | 1 | 0 |
| fresh volume after W8's crashed two-engine test (0320 `test_two_engines_one_gpu` at a08d7d6, two processes on one GPU), CLB | 1 | 0 |
| fresh volume after that test, no CLB | 4 | 0 |
| copy of the warm prod cache minus the kda / exl3 extension builds, after that test | 4 | 0 |
| **W8's order on the head node's own cache: the crashed two-engine test (which rebuilt the kda / exl3 extensions), then the file** | 3 | **1 (the first)** |
| memcheck (`compute-sanitizer`, GPU tests, warm) | stopped after 7 min to free the GPU | - |

The one reproduction (`results/W9/tests-head-seq/seq1-prefix.log`) is W8's failure exactly: the same three tests,
the same place (test 29, the first engine built after the four `test_gpu_new_session_resumes_at_the_system_end`
cases: a reference engine with prefix share **off**, IMA surfacing at the synchronize after `Graphs`' eager warm-up
in `GlmEngine.__init__`), 28 passes before it. Both failing runs (W8, W9 seq1) were the first run of the file after a
crashed two-process test and both spent ~50 s in the container before pytest started (63 s wall vs 10.6 s in pytest;
W8: 124 vs 71); every passing run spent 3-4 s. A hypothesis that a warm-up kernel reads an uninitialized device
buffer as an index (recycled memory from earlier engines in the process) was tested directly: device memory poisoned
with 0xFF / 0x7F (60 GiB, freed back) before building the test's engines, 3 rounds each, under CLB -> no fault
(`results/W9/poison_engine.py`, `ima/p1.log`, `ima-worker/p2.log`).

Verdict: an artifact of the test process after a crashed two-process GPU run, not of the production path. The fault
is in an engine with 0310 disabled, in engine construction, in a process that had already built and freed ~20
engines (production builds one engine per process and restarts clean: every start in W6-W9 passed its canary), and
it appeared only right after a crashed two-process run: 2 of the 13 runs that followed one (W8 + W9), 0 of the 12
without one (W9's 10, W8's 2 reruns). The faulting kernel could not be named (the two
reproductions were without CLB; the 11 other post-crash runs, 1 with CLB, did not), so a latent engine-construction bug
cannot be excluded completely, but nothing points at the live 0310 path. `GLM53_TF_PREFIX_SHARE=1` stays.

### 2. Memory margin: under 8 GiB today at 8,192-row chunks -> 4,096 (W8's documented step back)

| when | head MemAvailable | worker MemAvailable |
| --- | ---: | ---: |
| prod idle, 9 min after W8's 19:14 start (before W9) | 16.05 GiB | 14.62 GiB |
| prod idle, 10 min after the W9 window-1 restore / at the window-4 start | 15.52 / 15.62 GiB | 14.08 / 14.32 GiB |
| minimum in window 2 (loads A, C, B incl. 98k prompts, 60k sessions) | 10.32 GiB | 9.05 GiB |
| 4 x 250k stress, F = prod + RoCE + b12x (8,192 rows) | 8.89 | **7.82** |
| 4 x 250k stress, G = prod + RoCE (8,192 rows) | 8.91 | **7.83** |
| 4 x 250k stress, **H = prod unchanged** (W8 config, 8,192 rows) | 8.44 | **7.21** |
| 4 x 250k stress, **I = prod + RoCE + b12x + 4,096 rows (adopted)** | **9.72** | **8.67** |

Each stress ran after the same ~10 min of gates (exact, batchexact, 24.5k / 98k twice, 10 concurrent reps), as W8's E
(9.23 / 8.14). The production config itself now bottoms out under 8 GiB on the worker node (H: 7.21), so W8's 8.14 was a
thin pass, not a stable margin; RoCE (a ~1.5 MB pinned region) and b12x do not account for it (G == F; both above H).
W8's documented first step, `PREFILL_ROWS_MAX` / `SOLO_PIECE` 4,096 (lean chunk buffers 1.55 instead of 3.10 GiB a
rank), gives +1.45 GiB on the worker node over H and passes. `scripts/gpuwatch.py` has no memory threshold (clocks, power,
the slow state); the memory guards are the engine's (`BATCH_RESERVE_GB=11`, `SESSION_RESERVE_GIB=6`, `ADMIT_GB=2`)
and `MEM_GATE_GIB=108` (MemFree at start); unchanged, they still fit (idle after start ~15.5 / ~14.3 GiB).

### 3. 0350 RoCE: every stage clean after four harness fixes; +4-11% decode, adopted

Details in docs/ROCE-FIX.md "W9". Summary:

- **W7's 1 MiB mismatch was a harness race**, not a chunk / slot boundary: `loopback` built its inputs on the default
  stream and gathered them on non-blocking side streams without a wait. With the race, 512 KiB-4 MiB first gathers
  differ in 19-20 of 20 trials, **including the rank's own shard** (copied from its input, never on the wire); 16k-256k
  in 0 of 20. With a synchronize: no difference in 20 trials at 16k / 128k / 4 MiB and 320 trials each at 256 KiB-2
  MiB. The engine gathers on the producing stream, so it cannot hit this. Fixed loopback: `bits_equal: true` at 1
  MiB. One unexplained difference: a second gather of fully written 2 MiB inputs, once, not again in 360 more (above
  the engine's 256 KiB).
- Three more harness bugs found and fixed in `tests/cuda/bench_roce.py` (`stress --loop` host deadlock on a
  first-time allocation; `fault` launching rank 1's "next" op at once; `soak` graphs whose inputs were freed, and ranks
  stopping on their own clocks).
- Preflight: `PCI_WR_ORDERING = per_mkey(0)` on all four functions. GPU unit tests 25/25.
- Stage 1-2 (one node): loopback both functions, `HCAS=1`, each function alone: bits equal 16k-1m. `stress --loop`
  100k ops per size and mode (16k, 128k; eager, graph), striped and per function, plus 20k at 256k / 1m: 0 mismatches.
- Stage 3 (two nodes): `bench` bits equal at 16k / 64k / 128k / 1m, graph latency RoCE 11.7 / 16.9 / 20.3 / 77.7 us vs
  NCCL 45.3 / 76.1 / 66.1 / 204.3 us (a 90-exchange step saves 3.0 ms at 16k); `fault`: rank 0 raises after 3.00 s,
  `never`, rank 1's next exchange completes; `stress` 100k a size and mode: 0 mismatches; `soak` 20 min (~69.5M ops a
  rank, every replay compared) + 3 min after the end-condition fix (113,664 replays): clean.
- **Engine A/B** (load A = prod, NCCL; load C = A + `GLM53_TF_COMM_BACKEND=roce`, 256 KiB max over both functions):

| check | A (NCCL) | C (RoCE) |
| --- | --- | --- |
| exact / batchexact | 10/10, 4/4 | 10/10, 4/4 |
| transcripts (6 prompts x greedy / sampled seed 1234, alone; 4 together) | reference | **12/12 and 4/4 byte-identical** |
| decode 1 stream, 5 reps (tok/s; each rep its own prompt) | 63.8 / 45.2 / 47.2 / 91.8 / 39.3, median 47.2 | 66.4 / 48.1 / 52.3 / 96.7 / 41.5, median **52.3 (+10.8%)**; per rep +4.0 to +10.7% |
| decode 4 streams, 5 reps (aggregate) | 78.2 / 71.5 / 73.2 / 72.9 / 69.1, median 72.9 | 81.5 / 74.8 / 76.1 / 76.0 / 72.3, median **76.0 (+4.3%)**; per rep +3.9 to +4.6% |
| round time at equal tokens / round | 1 stream 55-77 ms, 4 streams 98-148 ms | 3-5 ms shorter a round (1 stream), ~3 ms (4 streams) |
| load-time calibration: verify 1 row / DFlash2 block | 31.8 / 3.74 ms | 30.4 / 3.11 ms |
| RoCE failure / fallback lines | - | none |

Adopted (rule: every stage clean, identical transcripts, decode >= +3%): `GLM53_TF_COMM_BACKEND=roce`,
`GLM53_TF_ROCE_MARK=/cache/roce-failed` (a run-time failure pins the next start to NCCL). The stale W2 marker
(`/cache/roce-failed` on the head node, 09:59) and the markers W9's own harness failures wrote were saved
(`results/W9/roce/roce-failed-marker-head.txt`, `roce-marks-before-*.txt`) and deleted.

### 4. 0360 b12x bit 4 under FP8: +6.3% / +6.7% prefill, adopted

- GPU tests (`test_b12x_attn_patches.py`): 23 passed, including FP8 == dequantized at 2,051 tokens (W3's failure),
  rows / subsets / permutations, engine resumed == fresh with cached > 0, bits never cross. Kernel: FP8 one pass 5.01-
  5.40 ms vs chunked 9.08-9.24 ms (1.71-1.81x), bf16 1.40-1.50x. The 0240 subset: 5 of 6; the failure is
  `test_engine_snapshots_never_cross_bits`' control for **bits 3** (0240's KDA / hc kernels, not adopted; the
  bit-4 path has its own passing test), recorded, not pursued.
- Load B = prod + `GLM53_TF_B12X=4`: exact 10/10, batchexact 4/4; sessions (3 conversations x 3 turns, ~60k-token
  prompts, `tf_knobs.b12x: 4`): every follow-up cached 60,864 and the same reply sha as the same request cold
  (`draft: false`), 6/6; 4 concurrent == alone 4/4.
- Prefill A/B in the same load, per request `b12x` 4 vs 0, 3 runs (cold tok/s):

| prompt | b12x 0 | b12x 4 | gain | reply sha (all cells, cold and warm) |
| --- | --- | --- | ---: | --- |
| 24.5k (21,464 tokens) | 1,475.1 / 1,469.4 / 1,471.3 | 1,566.4 / 1,564.1 / 1,561.6 | **+6.3%** | 8794a3463259cc2f |
| 98k (85,782 tokens) | 1,452.1 / 1,450.3 / 1,373.6 | 1,547.2 / 1,546.1 / 1,548.3 | **+6.7%** | 8794a3463259cc2f |

Decode unchanged (after the prompt 72.7-89.2 tok/s both ways; concurrent 1 / 4 streams 63.5 / 45.1 / 47.5 and 78.4 /
71.3 / 73.1 vs load A's 63.8 / 45.2 / 47.2 and 78.2 / 71.5 / 73.2). Adopted: `GLM53_TF_B12X=4`.

### 5. Per-slot fixed verify cost (ADAPTIVE-DRAFT.md §6 step 2): measured; the lever exists

Load A, `bench/multiturn.py --modes concurrent --streams 1,2,4 --reps 3 --long-tokens 512 --extra '{"draft": false}'`
(serial: 1 row a slot a round), and the same with drafting (5 reps):

| slots x rows a round | ms / round (3 reps) | aggregate tok/s |
| --- | --- | --- |
| 1 x 1 (serial) | 33.0 / 33.1 / 33.2 | 29.2-29.6 |
| 2 x 1 (serial) | 45.5 / 45.5 / 45.5 | 42.5-42.6 |
| 4 x 1 (serial) | 69.3 / 69.0 / 69.5 | 55.5-55.9 |
| 1 x ~8 (drafting, mostly DFlash2 blocks: rep 3, 7.2 tokens a round) | 71.6 verify + 5.8 drafting | 91.8 |
| 1 x mixed (drafting, 2.2-2.9 tokens a round) | 50.4-54.1 verify + 5.1-5.9 drafting | 39.3-47.2 |

So a round is ~21 ms plus **12.1 ms per slot-row**; going from 1 to ~8 rows in one slot adds ~38 ms (~5.5 ms a row,
mostly expert reads), which leaves **~6.6 ms fixed per slot** (the fit's 7 + 6 holds: predicted ~31 / ~44 / ~70 ms,
measured 33 / 45.5 / 69.3; the "cheap slots" case, ~31 / ~38 / ~50, is ruled out). In a 4-stream round (98-148 ms,
~115 typical) the fixed per-slot part is ~26 ms (~23%); drafting is 4-5.5 ms a round. The dominant term is still
per-row work (routed experts), but the per-slot fixed cost is real and large enough that halving it (fused per-slot
KDA / attention / commit across slots) is worth the simulator's **~+6%** at 4 streams. Measurement only.

### Combined gates (windows 3 and 4)

Each load: `results/W9/gates.sh` (`w3.sh` / `w3g.sh` for F / G) = exact, batchexact, `ab.py` 24.5k / 98k twice,
concurrent 1 / 4 streams x 5 reps, 4 x ~250k stress (`multiturn.py --modes stress`), MMLU-200 + refusals, exact /
batchexact again, dmesg OOM count. Decode vs load A (window 2, same prompts per rep; H is the same-window control):

| load | exact / batchexact (before, after stress) | reply sha (8 cells) | prefill 24.5k / 98k tok/s | decode 1 / 4 streams, median of 5 (vs A) | stress MemAvailable min head / worker | MMLU-200, refusals | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F: RoCE + b12x 4 (8,192 rows) | 10/10, 4/4; 10/10, 4/4 | 8794a3463259cc2f | 1,575 / 1,555 | 50.7 (+7.4%) / 74.5 (+2.2%) | 8.89 / **7.82** | 88.0%, 0/10 | **fail (memory)** |
| G: RoCE (8,192 rows) | 10/10, 4/4; 10/10, 4/4 | 8794a3463259cc2f | 1,465 / 1,445 | 50.1 (+6.1%) / 75.0 (+2.9%) | 8.91 / **7.83** | 88.0%, 0/10 | fail (memory) |
| H: prod unchanged | 10/10, 4/4 | 8794a3463259cc2f | 1,473 / 1,450 | 46.1 (-2.3%) / 73.4 (+0.7%) | 8.44 / **7.21** | - | fail (memory) |
| **I: RoCE + b12x 4 + 4,096 rows** | **10/10, 4/4; 10/10, 4/4** | **8794a3463259cc2f** | **1,499 / 1,496** | **50.5 (+7.0%) / 76.1 (+4.4%)** | **9.72 / 8.67** | **88.0%, 0/10** | **pass: adopted** |

I against H (same window, same prompts): prefill +1.7% / +3.2%, decode per rep +4.3 to +9.5% (mean +6.6%) at 1
stream and +3.4 to +5.1% (mean +4.0%) at 4 streams. OOM kills 0 / 0 in every load; the stress filled in 806-847 s,
longest decode gap 3.8-4.0 s. (b12x's +6.5% at 8,192 rows mostly pays for the chunk step back: 8,192 -> 4,096 costs
~2.6% in W8.)

**Adopted:** `config/prod.env` -> `GLM53_TF_COMM_BACKEND=roce`, `GLM53_TF_ROCE_MARK=/cache/roce-failed`,
`GLM53_TF_B12X=4`, `GLM53_TF_PREFILL_ROWS_MAX=4096`, `GLM53_TF_SOLO_PIECE=4096`; image `glm53-tensorfold:b2` (tagged on
both nodes, unchanged). Prod restarted on it 23:26: ready in 22 s, canary ok (decode 77.7 tok/s; warm-up 16,384 prefill
10.54 s, was 11.0), RoCE connected on both ranks (256 KiB, both functions), lean set 4,096 rows (1.55 GiB), https
`/v1/models` ok, `17*23` -> `391`, watchdog timer active, lease refresher stopped, lease deleted. First hour on RoCE
watched every 5 min (`results/W9/prod-watch.log`): 23:26-00:26, 13 samples: 0 RoCE failure / fallback lines on either rank, no marker, health errors 0 over 101 requests; light
traffic every 5 min (`results/W9/prod-traffic.log`: 1 stream 64.2-69.1 tok/s, 4 streams 78.4-79.4 tok/s, 256 tokens) and
`exact` 10/10 on the live server at the start and at the end; MemAvailable 17.39 / 16.07 GiB after start, 14.87 / 13.59
after the hour (allocator and session-store warm-up, as in W8).

Ops notes: `pkill -f <script>` inside an `ssh '...'` command killed that ssh's own shell again (W8's note); kill by
pid. A local tmp quota error lost one launching command's output (window 3 had started; the windows.log line was added
afterwards). The b12x bits-3 test and the 2 MiB single mismatch are the open ends.

## W10: batch 4 (0390 MLA expand v2, 0400 KDA v2, 0410 sparse v2, 0370 decode overlap / CPU pin, 0380 deep verify, 8,192-row chunks), 2026-09-29 (00:36-01:30, 01:41-02:27, 02:38-03:22, 03:33-04:25) — 0390, 0370 overlap and 0380 16 rows adopted

Image `glm53-tensorfold:b4` = every patch through 0410, built on the head node and loaded on the worker node (tagged on both). Two
patch fixes made during window 1 went into the final b4 (built three times; the first two never served):

- **0410 had a syntax error** in the `engine.py` hunk: the load-time message split an f-string expression across two
  string literals (`... else 'in "` / `f"shared memory'}`). `engine.py` did not import, so every engine-level test
  of the first b4 failed (`SyntaxError: unterminated string literal`, line 303) and **no load could have started**.
  The kernel-level tests import `sparse_v2` / `latent` / `kda_v2` directly and passed. Fixed in
  `patches/0410-glm-sparse-v2.patch` (same hunk sizes); `compileall` over the installed package is clean.
- **0390's tile table** (`latent.V2_TILES`) retuned from the GPU sweep (`bench_mla_expand.py --sweep`): 4 warps
  instead of 8 everywhere, absorb 64 rows x k 32 (was 128 x 16), expand 64 x 16 with BN 64 unchanged. Speed only;
  the bitwise test covers every forced tile and was re-run on the final image (50 passed).

**Off == production** on the final b4: `tests/kvpool_ptx.py` against b2 **33 of 33** Triton kernels' PTX identical;
the patches' own compile checks in the image (0390 21, 0410 36, 0400 17 passed: v1 / one-pass / fast_kda PTX
unchanged); load P0 (b4, no new knob) gives b2's reply sha `8794a3463259cc2f`, exact 10/10, batchexact 4/4, session
follow-up resumed == cold, prefill 1,500 / 1,503 tok/s (W9 load I: 1,499 / 1,496). Files: `results/W10/`
(`tests-head*/SUMMARY`, `tests-worker*/SUMMARY`, `summ.py` prints a line a load).

### 1. Kernel bitwise tests and microbenches (GPU, both nodes, everything under `timeout`)

| test | result |
| --- | --- |
| 0390 `test_mla_expand_patches.py` | **50 passed** (first b4 and final tiles): v2 == v1 bit for bit, q4 / q4mse / bf16 kv_b, 1-8,192 rows, windows, every forced tile, in a CUDA graph; control differs |
| 0390 regressions with `GLM53_TF_MLA_EXPAND=v2` | `test_latent_patches.py` 20 passed; `test_kv_pool_patches.py` / `test_fp8_kv_patches.py -k latent`: 2 failed, **the same 2 with v1** (`test_fp8_latent_close_to_expanded_reference`: 0.0566 > 0.04 tolerance, identical numbers both ways: a stale tolerance, not 0390) |
| 0400 `test_kda_v2_patches.py` | **345 passed** (every mode / setting == fast_kda, 50 repeats, busy stream, resume == fresh, engine entry); `test_lean_patches.py` with `KDA_V2=1` 53 passed |
| 0410 `test_sparse_v2_patches.py` | first run 27 passed, 6 failed: **all 6 `OutOfResources` (133,120 B of shared memory > 101,376) in bf16 cases**, none a bit difference: the test's `_v2` forced FP8's 3 stages on bf16 caches, which `sparse_v2.config` itself refuses. Test fixed (format default; unfit settings skipped): **30 passed, 3 skipped**; every FP8 (production) case passed both times. `test_b12x_attn_patches.py -k "one_pass or fp8"` 3 passed |
| 0370 `test_decode_overlap_patches.py` | 29 passed (incl. the affinity test on the Spark's cores); `test_batch_parallel_patches.py` with `DECODE_OVERLAP=1` 26 passed; `test_batch_sessions_patches.py` with it: 26 passed, 2 failed (`test_follower_replays_rank0_sessions`: the test records rank 0's `_share` calls only and replays them to a one-GPU follower; with `plan` the plan rides on the sampler exchange instead, so the replay is out of step: `parse_plan([1, 0])`. A harness limit; 0370's own lockstep test replays both kinds and passes, and the two-rank engine loads below are exact, with cancels) |
| 0380 `test_deep_verify_patches.py` | 4 passed + the 16-row child: 17 passed, 2 failed, both `assert deepest >= 9` in `test_gpu_drafted_replies_equal_serial` (the synthetic checkpoint never drafted deeper than 2: a coverage assertion; the replies equalled serial before it). Dense 1-16 == serial and long-context / MTP 1-16 == eager passed. The engine evidence is below (identical reply hashes with 16-row windows in use) |

Microbenches (one GPU each, prod stopped):

| kernel | today | new (best bit-identical setting) | a 512-row sub-block | projected prefill | plan gate |
| --- | --- | --- | --- | --- | --- |
| 0390 absorb + expand, 512 rows | 677 + 2,560 us | 346 + 886 us (final tiles; offline tiles: 399 + 1,543) | **0.38x** | +6.3% (43 us a token) | <= 0.5x: **pass** |
| 0390 at 1 / 8 rows (decode) | absorb 28 / 26, expand 114 / 105 us | 25 / 25, 68 / 68 us | 1.1x / 1.5-1.7x | decode +~1% | - |
| 0400 split (bv 32, 4 warps, maxnreg 168) | fast_kda 0.913 ms | 0.845 ms | 0.93x | +0.7% (4.5 us a token) | <= 0.85x: **fail** |
| 0400 fused (every setting) | 0.913 ms | 1.25-4.0 ms | 0.23-0.87x (slower) | negative | fail |
| 0410 (FP8, overlapping lists, 10.7k / 85.8k ctx) | one pass 2.41 / 9.23 / 36.3 ms (512 / 2,048 / 8,192 rows) | 2.03 / 7.54 / 29.7 ms (cfg 2,1,1; default 3,1,1 2-3% slower) | 0.82x | +1.3-1.6% (random lists at 85.8k: +2-3%) | <= 0.7x: **fail** |

The KDA and sparse kernels are far from their offline estimates (0400: 0.93x, estimated 0.85x split and 0.5-0.7x
fused; 0410: 0.82x, estimated 0.32-0.55x). Every timed setting was bitwise equal.

### 2. Engine A/B on the production config (per knob, then combined)

Every load: `config/prod.env` + overrides on b4 (`results/W10/load.sh`), then `run.sh` = exact, batchexact, a session
follow-up (1 conversation x 3 turns over ~61k tokens: turns 2-3 cached 60,864 with the cold request's reply sha, plus 4
batched == alone), `ab.py` 24.5k / 98k twice (cold prefill, reply sha), concurrent 1 / 4 streams x5 (aggregate tok/s,
the same 5 prompts every load).

| load | prefill 24.5k (2 runs) | 98k (2 runs) | vs P0 | reply sha | exact / batchexact / session | 1 stream median / 4 streams median | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0: b4, no new knob (control, window 1) | 1,501 / 1,500 | 1,504 / 1,502 | base | 8794a3463259cc2f | 10/10, 4/4, OK | 52.1 / 75.8 | == prod (W9 I 1,499 / 1,496) |
| **M: `MLA_EXPAND=v2`** | 1,611 / 1,604 | 1,608 / 1,599 | **+7.1% / +6.7%** | same | 10/10, 4/4, OK | 51.3 / 76.4 (per rep +1.7, +1.0, -1.5, +3.3, +1.0 / +1.0, +0.1, +1.6, +0.8, +1.2%) | **adopt** |
| K1: `KDA_V2=1` (split) | 1,518 / 1,518 | 1,517 / 1,518 | +1.2% / +1.0% | same | 10/10, 4/4, OK | 50.4 / 75.5 | < +2%: off |
| S: `SPARSE_V2=1` (window 2) | 1,498 / 1,503 | 1,502 / 1,504 | 0.0% / 0.0% | same | 10/10, 4/4, OK | 50.4 / 75.5 | < +3%: off |
| MSK: M + S + K1 | 1,631 / 1,632 | 1,626 / 1,635 | +8.7% / +8.5% (+1.5% over M) | same | 10/10, 4/4, OK | 52.6 / 76.8 | K and S each below their bar |
| M8: M + 8,192-row lone chunks (gates without MMLU) | 1,677 / 1,676 | 1,659 / 1,665 | +11.7% / +10.5% | same | 10/10, 4/4 | 51.8 / 77.0 | **stress 8.32 / 7.23 GiB: fail** |

KDA v2 fused (mode 2) was not loaded: every fused setting was slower than fast_kda in the bench.

**8,192-row chunks (item 3).** The prefill winner does not change memory, so the W9 finding stands: M8's 4 x ~250k
stress bottomed at **8.32 / 7.23 GiB** (head / worker; W9's H at 8,192: 8.44 / 7.21), under the 8 GiB floor on
the worker node, no OOM (0 / 0), longest decode gap 3.9 s. 4,096 stays. (8,192 would add another ~+4% on top of M.)

### 3. Decode (window 3, all on top of M; MA = M again as the same-window control)

Decode set a load (`dec.sh` / `deep.sh`): exact, batchexact, the W9 transcripts (6 prompts x greedy / sampled alone,
4 together; all == W9 load A's hashes in every load: 0390 keeps the bits), concurrent 1 / 4 streams x5; the 0380
loads also `glmbench --suites tf,tweet,kit,edit --reps 3 --long-tokens 512`.

| load | 1 stream per rep vs control (median) | 4 streams per rep vs control (median) | other | result |
| --- | --- | --- | --- | --- |
| D: `DECODE_OVERLAP=1` vs MA | +1.9 +1.2 +1.3 -0.5 +2.1% (53.0 -> 53.7, +1.3%) | +0.1 +0.7 0.0 +1.7 +2.6% (76.0 -> 77.3) | transcripts identical, exact 10/10, batchexact 4/4 | |
| D2 (repeat) vs MA2 (control repeat) | +1.8 +2.5 +1.9 +1.8 +1.7% (52.9 -> 53.9, +1.9%) | 0.0 +4.1 +1.6 -0.4 +1.2% | same; **cancel check** (3 decoding + 1 streamed client gone after 50 chunks) x2: the 3 == alone, request log `finish: cancelled`, next request fine | **adopt** |
| C: `CPU_PIN=auto` vs MA | +1.2 +0.4 +0.4 -1.0 +0.9% (+0.4%) | -0.4 +0.7 -0.4 +2.1 +0.3% | pinning as planned on both ranks (`tf-serve` cpu 19, RoCE proxy 18, `NCCL Progress` 16-17, rest 0-15; 40 samples at 0.5 s) | < +1%: off |
| MA2 vs MA (control noise) | +0.9 -0.6 -0.2 -1.1 +0.2% | +0.4 -0.7 -0.9 +1.3 +0.7% | | |
| R16: `MAX_DRAFT_ROWS=16` vs MA | +3.5 +0.2 -0.9 -0.3 +0.9% (53.0 -> 52.5) | 0.0 +0.5 -2.2 +1.2 +2.1% (76.0 -> 75.8) | calibration lists 16 windows (9-16: 75.2-101.6 ms, ~3.8 ms a row) | see below |

0380 glmbench cells (tok/s, MA -> R16): **edit-rename 89.8 -> 111.2 (+23.8%), edit-comments 81.8 -> 94.7 (+15.8%),
edit-print-to-log 91.1 -> 115.5 (+26.9%)**; code / chat T=1 +1.2 / +0.9%, code / chat T=0 +1.4 / -0.4%, sequence -0.2%,
code 512 -0.3%, json +0.3%, hashmap +1.9%, structured +0.9%, essay +1.6% (all within +-2%). **Every reply hash of
all 13 cells x reps is identical between MA and R16**, including the edit cells, where the 9-16-row windows ran, and the
sampled T=1 cells: the deeper windows keep the bits. Adopted (rule: edit >= +10%, everything else in noise).

0370 nsys (plan §4) was not captured: no window time left after the gates; the decode A/B carries the adoption.

### 4. Combined candidate FIN = prod + `MLA_EXPAND=v2` + `DECODE_OVERLAP=1` + `MAX_DRAFT_ROWS=16` (b4, 4,096 rows): full gates

`results/W10/gates.sh FIN` (W9's gate set) + `w4.sh` (needle, edit cells, cancel):

| gate | FIN | bar |
| --- | --- | --- |
| exact / batchexact, before and after the stress | 10/10, 4/4; 10/10, 4/4 | 10/10, 4/4 |
| reply sha, 24.5k / 98k x2 | 8794a3463259cc2f (all cells) | same |
| prefill 24.5k / 98k (tok/s) | 1,614 / 1,602; 1,577 / 1,606 (**+7.1% / +5.9%** vs P0; the 1,577 run is the low one) | - |
| decode 1 stream, per rep vs P0 / MA / MA2 (mean) | +3.7% / +1.1% / +1.3% (median 52.1; reps 71.1 / 49.8 / 52.1 / 97.5 / 43.2) | not lower |
| decode 4 streams, per rep vs P0 / MA / MA2 (mean) | +3.1% / +2.5% / +2.4% (median 78.1) | not lower |
| 4 x ~250k stress MemAvailable min, head / worker | **10.49 / 9.36 GiB** (W9 I, today's prod: 9.72 / 8.67); filled in 770 s, longest decode gap 3.9 s | >= 8 |
| OOM kills (dmesg, both nodes; after the stress and after the needle) | 0 / 0; 0 / 0 | 0 |
| MMLU-200, refusals | **88.0%** (176/200), 0/10 | >= 87% |
| needle, 314,249 tokens alone after the stress + MMLU | **found** cold and resumed (cold prefill 1,376 tok/s, W8 1,287; resume 314,240 cached, 0.19 s) | found |
| edit cells (glmbench `edit`, 3 reps) | 115.3 / 97.1 / 116.5 tok/s | - |
| cancel (3 decoding + 1 client gone after 50 chunks) | 3 == alone, `cancelled`, next request fine | - |
| /health | 340 requests, 0 errors | - |

**All gates pass.** One observation outside the gates: **MemAvailable during the needle** (sampled every 2 s,
after the stress and MMLU, with the pool holding the stress sessions) **bottomed at 8.55 / 7.39 GiB**, under 8 on
the worker node, with no OOM. A same-window control on today's production config (H2 = b2, W9 knobs, `w4b.sh`: stress then
the needle on a fresh start, without the gates before) gave 12.85 / 11.68 (stress) and 11.22 / 10.36 (needle), but
it is not a like-for-like baseline: W9 measured ~3 GiB of drift from the gates that precede FIN's stress (FIN's own
stress minimum, after them, was 10.49 / 9.36 and the highest yet), and the extra FIN-only memory is small (0380: ~115
MB a rank; 0390 / 0370: none). W8's needle after its stress (no MMLU between) bottomed at 9.28 / 8.35 with 8,192-row
buffers. Recorded as an open end: the needle-after-stress+MMLU minimum on the W9 config was never measured.

**Adopted:** `config/prod.env` -> `IMAGE=glm53-tensorfold:b4`, `GLM53_TF_MLA_EXPAND=v2`, `GLM53_TF_DECODE_OVERLAP=1`,
`GLM53_TF_MAX_DRAFT_ROWS=16` (4,096-row chunks, everything else as W9; the previous file is
`results/W10/prod.env.before-W10`). Not adopted, knobs off: `GLM53_TF_KDA_V2` (+1.0-1.2%), `GLM53_TF_SPARSE_V2` (+0%),
`GLM53_TF_CPU_PIN` (+0.4%), 8,192-row chunks (memory). Prod restarted on it at 04:25: ready in 23 s, canary ok (decode
80.1 tok/s; warm-up 16,384 prefill 9.90 s, was 10.6), both ranks print the 0390 / 0370 lines, drafter costs list 16
windows, RoCE connected (no marker), https `/v1/models` ok, `17*23` -> `391`, live `exact` 10/10, watchdog timer
active (its 04:26 run exited 0), lease refresher stopped, lease deleted. MemAvailable idle after start 16.84 / 15.49 GiB.
The NVMe session store starts cold (new image id).

Ops notes: window 2's restore ran twice (a stray `restore.sh` without a name before the named one: two prod starts,
02:26:50 and 02:27:32, both verified; `windows.log` line "w restored"). The 0370 nsys capture and the plan's arrival-latency
check were not run. `pgrep -f` inside the ssh commands matched the ssh's own shell again (harmless here: used for
listing only).

## W11: decode measurement (2026-09-29 09:34-10:27, prod down 53 min) — measurement only, nothing adopted

Step 1 of `docs/DECODE-PLAN.md` on today's production (`config/prod.env`: image `glm53-tensorfold:b4`, RoCE all-gathers,
b12x 4, `MLA_EXPAND=v2`, `DECODE_OVERLAP=1`, `MAX_DRAFT_ROWS=16`): nsys of single-stream prose / code and 4 streams, a
bandwidth and launch probe on one GB10, ncu of the decode kernels, the 4-stream CUDA-graph policy, and per-position draft
acceptance on prose / code / agent-like content. Files: `results/W11/` (scripts, logs, `tables.md`, `dec-*.json`,
`insitu-*.txt`, `accept*.json`); the four nsys reports are on the head node in `~/w11-traces/` (not in git, 40-67 MB).
The updated decomposition is in `docs/DECODE-PLAN.md` (sections 0, 1, 3 and 4 now use these numbers).

### 1. Method

- **Window 1 (prod stopped, 09:34-09:38):** `probe.cu` (plain streaming read / copy kernels over one decode round's byte
  volumes; launch overhead; PDL) on both GPUs at once, `kbench.py` (the production decode kernels: `exl3_mm`'s grouped
  decode path and `qmm.matmul`, random weights of the real per-rank shapes, one GPU), ncu on `kbench.py --ncu`, and a
  check whether nsys GPU-metrics sampling exposes DRAM bandwidth on GB10 (`bench.sh`).
- **Capture loads (W7's method):** `load.sh NAME nsys` runs `scripts/serve.sh` with the nsys shim (idle session, `nsys
  launch --trace=cuda-sw,nvtx --cuda-graph-trace=node`), `SYS_ADMIN`, and `results/W11/profile.py` (W7's NVTX copy) on
  both ranks; `cap.sh` / `cap4.sh` send the requests uncaptured first (`ctl*.jsonl`), then open **one** nsys window
  (`cap*.jsonl`); requests are 4 s apart so the analysis can split them. Greedy, thinking off.
  - Load `nsys` (09:39): chat 256, essay 384, code (LRU module) 384, 4 x prose 384. **Under nsys the slot rule admitted
    one slot** (MemFree 14.2 GiB after the engine's buffers, prod 18.1: `GLM53_TF_BATCH=4: only 1 sequence(s) fit`), so
    the four prose requests ran one after another; they count as more single-stream prose.
  - Load `nsys4` (09:44): the same with `GLM53_TF_BATCH_RESERVE_GB=8` (admission reserve only; 4 slots): 4 x prose 384
    concurrent, then chat 256.
- **Analysis:** `w11dec.py` (W7's `dec.py` rewritten: the RoCE `gather_kernel` is the exchange family; phases from the
  NVTX ranges; distinct experts per verify layer U from kbench's single-GPU `grouped_kernel` fit; dense bytes by launch
  grid from the prepared manifest's q4 shapes; host gaps between rounds). Checked against the W7 `mix-r0` trace: it
  reproduces W7's 60.0 / 125.5 ms rounds and families. `tables.py`, `insitu.py`, `accept.py`, `graphs_summary.py`.
- **Capture overhead:** the same requests uncaptured vs captured (engine `verify_ms + draft_ms` a round): prose 53.3 ->
  54.3 ms (+2.0%), code 63.6 -> 64.7 (+1.8%), 4 streams 121.3 ms uncaptured (`decode_s` / rounds) vs 122.0 in the
  trace. Same reply hashes captured and uncaptured. GPU clock 2,223-2,242 MHz, 58-65 °C during the captures.

### 2. The decode round, measured (rank 0; rank 1 within 0.02 ms)

| per round | 1 stream, prose (361 rounds) | 1 stream, code (91) | 4 streams, prose (162) |
| --- | ---: | ---: | ---: |
| **round, ms (captured / uncaptured)** | **54.6 / 53.3** | **64.7 / 63.6** | **122.0 / 121.3** |
| tokens a round | 2.42 | 4.16 | 9.1 (4 x 2.28) |
| largest verify window, rows (mean) | 3.5 | 5.1 | 11.8 |
| routed experts (`grouped_kernel` + rot_in / epilogues) | 27.8 | 36.8 | 75.5 |
| distinct experts a verify layer U (kbench fit; R = 1 proportional) | 20.1 (21.3) | 27.8 (28.3) | 59.9 (57.5) |
| trellis GB a round and rank; `grouped_kernel` GB/s | 5.47; 205 | 7.46; 211 | 16.1; 220 (15.2; 208) |
| dense q4 GEMV (`_qmm` + `_reduce` + `_swiglu`) | 16.1 | 16.5 | 21.7 |
| q4 GB a round; GB/s | 2.75; **175** | 2.78; 173 | 3.51; 169 |
| RoCE all-gathers: count, median (mean) us | 100, 14.5 (23.1) | 101, 17.3 (27.2) | 115, 20.5 (35.9) |
| exchanges exposed (nothing else running) | **2.3** | 2.8 | 4.2 |
| NCCL (2 control all-gathers a round) | 0.02 | 0.04 | 0.08 |
| DSA attention + indexer | 2.1 | 2.1 | 4.8 |
| KDA | 1.6 | 1.7 | 6.0 |
| hc | 1.45 | 1.5 | 1.8 |
| router / grouping / combine | 1.2 | 1.2 | 1.6 |
| norms / elementwise / memcpy / sampling | 0.3 | 0.3 | 1.2 |
| **GPU idle inside the round** (of it inside the verify forward) | **1.8** (1.1) | 1.7 (0.9) | 5.1 (2.5) |
| host gap between rounds | 0.00 | 0.01 | 0.01 |
| kernels a round | 1,853 | 1,879 | 2,621 |
| by launch phase: verify forward / drafting / sampling / other | 48.0 / 4.3 / 0.08 / 0.4 | 57.8 / 4.4 / 0.08 / 0.7 | 102.9 / 11.4 / 0.6 / 1.4 |
| host time in the drafting ranges (propose, MTP chains) | 5.0 | 5.3 | 16.7 |
| rounds mostly graph-replayed | 348 / 361 | 62 / 91 | 8 / 162 (fresh server) |
| floor (235 GB/s experts, 230.5 dense, 0.3 exchanges, small kernels) | 37.2 = 68% | 45.8 = 71% | 89.0 = 73% |

Against W7 (pre-RoCE, pre-0370): the 1-stream round is 59.5 -> 54.6 ms captured. **Exchanges 4.8 -> 2.3 ms** (RoCE:
100 a round at 14.5 us median; the mean is 23 us because of peer-wait skew). **GPU idle 4.65 -> 1.8 ms and the gap
between rounds ~0.9 -> 0.00 ms** (0370: the next round's plan rides on the sampler exchange; rounds are back to back).
Attention 2.6 -> 2.1 (0390). Experts and dense are unchanged in kind: they are **81% of the 1-stream round** (44 of
54.6 ms) and 80% at 4 streams.

Per window size (1 stream, all three single-stream segments, captured): R = 2 / 3 / 4 / 5 / 6 / 7 / 8 rows cost 44.5 /
51.2 / 57.8 / 64.6 / 68.0 / 76.5 / 82.7 ms (**~6.4 ms a row**, U grows ~5 experts a row: 12 / 18 / 23 / 28 / 31 / 38 /
42). The 7- and 8-row windows (DFlash2 blocks) run eager in the lone-slot path, with **no idle penalty** (1.25-1.36 ms
idle vs 1.8 for graph rounds): the W7 finding "eager / capture rounds hold 55% of the idle" no longer holds after 0370.

### 3. Decode kernels in situ vs alone vs a plain read of the same bytes

| kernel (launch grid) | bytes | in situ, 1 stream (median) | alone (`kbench`, M = 4) | plain read (`probe`) |
| --- | ---: | ---: | ---: | ---: |
| head 77,440 x 4,096 (1, 1210, 1), 2.6 calls a round | 178.4 MB | 834 us = 214 GB/s | 828 us = 216 | 750 us = 238 |
| KDA proj 12,576 x 4,096 (1, 197, 4), 34 | 29.0 MB | 147 us = 197 | 155 = 187 | 123 = 235 |
| dense MLP gate/up 12,288 x 4,096 (1, 192, 1) | 28.3 MB | 149 us = 190 | 150 = 189 | 120 = 235 |
| DSA o / MTP eh 4,096 x 8,192 (1, 64, 8), 13 | 18.9 MB | 100 us = 189 | 110 = 171 | 80 = 235 |
| dense MLP down 4,096 x 6,144 (1, 64, 4) | 14.2 MB | 73.5 us = 193 | 89 = 158 | 62 = 229 |
| shared gate/up, DSA proj 2,048 x 4,096 (1, 32, 4), 55 | 4.7 MB | 28.5 us = 166 | 34 = 138 (launch-bound) | 22 = 215 |
| KDA fb / gb 4,096 x 128 (1, 64, 1), 68 | 0.29 MB | 4.4 us = 67 | - | 2.7 = 111 |
| `_reduce` (split-K sums), 224 calls a round | - | 1.1 us each, 0.29 ms a round | | |
| `grouped_kernel` gate/up, R = 1 (MTP step, U = 8 exactly) | 33.6 MB | 157 us = **214** | 162 = 207 | 144 = 233 |
| `grouped_kernel` down, R = 1 | 16.8 MB | 83 us = **202** | 82 = 204 | 74 = 227 |
| `grouped_kernel` gate/up, R = 3 / R = 12 | U 17-20 / 54-60 | 378 / 1,108 us | 333 (U 17) / 1,023 (U 54): 214 / 221 | 303 (U 17) / 905 (U 51): 235 / 236 |

- The in-situ kernels run at their isolated speed or faster (no contention penalty; the head is within 1%). **The gap is in the kernels, not the system:** a plain read of the same bytes is 10-15% faster for the
  experts and 15-25% for the mid-size dense shapes; small dense shapes run at a third to a half of the plain read.
- **ncu** (`ncu-kb-raw.csv`; ncu locks clocks, so its times are not used): `grouped_kernel<8,4>` 108 registers, 128-thread
  CTAs, 3 CTAs an SM (shared memory limit) = 25% theoretical occupancy, ~25% warps active, SM throughput 18-25%;
  `_qmm` 127 registers, 4 CTAs an SM (register limit) = 33% occupancy, 29-50% of peak memory-pipe throughput. Both are
  latency-bound streamers with few bytes in flight an SM, not ALU-bound: the E1 / E2 lever is more loads in flight
  (deeper `cp.async` pipelines, more CTAs), not arithmetic. **GB10 exposes no `dram__*` counters to ncu 2026.2 and no
  DRAM metric to nsys GPU-metrics sampling** (`gm.log`), so in-situ bytes stay inferred (U from kernel time).

### 4. Attainable bandwidth on one GB10 (`probe.cu`, both GPUs agree within 1%)

| access | size | read GB/s | copy GB/s (read + write) |
| --- | --- | ---: | ---: |
| one MoE layer's experts, gathered (U random of 288) or contiguous, gate+up / down | U 8-65: 34-273 MB / 17-136 MB | **233-238 / 227-237**, gathered = contiguous | - |
| a round's expert reads: 42 layers x (gate/up, down), 84 launches back to back | U 8-51: 2.1-13.5 GB | **235-237** | - |
| the verify's whole dense q4 set, 304 launches in layer order | 2.36 GB | **230.5** | - |
| one launch, cold | 0.25 / 1 / 2 / 4 / 8 / 16 / 32 MB | 95 / 138 / 179 / 201 / 219 / 229 / 232 | 188 / 177 / 200 / 208 / 215 / 219 / 219 |
| one launch, cold | 64 MB-1 GB | 233-236 | 216-220 |

- **Ceiling: 235 GB/s** for streaming reads of the expert volumes (86% of the 273 GB/s nominal), whether the experts
  are gathered or contiguous; 230.5 for the dense set with its small matrices; a launch needs ~8 MB to reach 219 and
  ~16 MB to reach 229. ROOFLINE's 230 is confirmed (slightly conservative). The probe ran at 2,418 MHz SM clock (prod is
  capped at 2,250; reads are DRAM-bound).
- **Launch overhead** (`probe.cu`): an empty kernel costs 1.62 us eager, 0.47 us inside a CUDA graph. Between dependent
  kernels in a graph the last block of one and the first work of the next are **0.73 us** apart; with **PDL**
  (`cudaLaunchAttributeProgrammaticStreamSerialization` + `griddepcontrol.wait` / `launch_dependents`) **0.42 us**, and
  a 1 MB-read kernel chain goes 6.20 -> 5.76 us a kernel. **PDL works on sm_121** (GB10): with it every dependent
  kernel that reads 1-4 MB started its prologue before the previous one ended (399 of 399; 4.8 us early on 5.3 us
  kernels), in eager streams and in captured graphs. At ~1,850 kernels a 1-stream round, -0.31 us a boundary is ~0.6 ms
  a round.

### 5. CUDA graphs at 4 streams (`graphs.sh`: batchexact, then `multiturn.py --modes concurrent --streams 4 --reps 3
--long-tokens 512` twice: pass a right after load, pass b on the warm graph set)

| load | aggregate tok/s, pass a / pass b (3 reps each) | mean of 6 | request-rounds graph / eager / capture % (a; b) | batchexact |
| --- | --- | ---: | --- | --- |
| DEF (prod: `CAPTURE_AFTER=3`) | 83.3 75.7 76.7 / 83.0 74.6 77.3 | 78.4 | 11/80/7; 22/66/11 | 4/4 |
| **G0: `GLM53_TF_BATCH_GRAPHS=0`** | 84.4 77.2 79.2 / 84.0 76.9 79.4 | **80.2 (+2.6%)** | 0/97/0; 0/97/0 | 4/4 |
| CA8: `GLM53_TF_BATCH_CAPTURE_AFTER=8` | 84.2 76.4 78.2 / 83.9 77.0 79.4 | 79.9 (+2.2%) | 2/97/1; 7/92/1 | 4/4 |
| DEF2 (prod again, control) | 82.1 74.6 77.2 / 82.9 74.3 76.4 | 77.9 | 13/80/7; 28/60/12 | 4/4 |

Percentages vs the mean of DEF and DEF2 (which agree within 0.6%). Every G0 rep beats the same rep and pass of both
controls (+1.2 to +3.9%). The graph keys of 4-slot rounds (slots x rows x context buckets) rarely repeat in 6 x 512 tokens: 60-80%
of rounds run eager and 7-12% pay a capture (a warm-up forward + capture + instantiate); the graph rounds do not win the
captures back. Engine ms a round (verify + draft): 113.4-115.2 (DEF) vs 109.8-111.4 (G0 / CA8). **F0 is worth +2.2-2.6%
at 4 streams**, exact (batchexact 4/4 on every load). Not measured: lone requests in slots 1-3, which use the same
multi-sequence graph keys (a single stream on slot 2 ran 22% eager, 8% capture rounds in load `nsys4` and was 2% slower
than on slot 0); the adoption A/B should include them. Not adopted here (measurement window).

### 6. Draft acceptance on real content (baseline for drafter retraining; `w11req.py`, `accept.py`)

14 prompts: prose (glmbench chat, essay, hashmap; tides; a letter), code (LRU module, fib, a Go queue, a React table),
agent-like (glmbench's edit source renamed / print-to-log as whole-file rewrites, a unified diff, 25 JSON records, a
plan with shell commands), each greedy and sampled (T 1, top-k 20, top-p 0.95), 512 tokens (1,024 for the rewrites and
JSON); plus 2 prose prompts with thinking on (1,024). Positions: a_j = P(draft j kept | drafts 1..j-1 kept) and
p_j = P(draft j kept) (cumulative, what vLLM reports).

**Untruncated** (`tf_policy` `4` = MTP 4 drafts every round, `f7` = DFlash2 7-draft blocks every round; load DEF2, 384
tokens, 1,024 / 768 for agent):

| drafter, class | rounds | a_1 .. a_7 (conditional) | p_1 .. p_4 (cumulative) | tokens a round |
| --- | ---: | --- | --- | ---: |
| MTP x4, prose | 1,549 | 0.72 0.60 0.51 0.44 | **0.72 0.43 0.22 0.10** | 2.48 |
| MTP x4, code | 887 | 0.90 0.79 0.75 0.60 | 0.90 0.71 0.53 0.32 | 3.46 |
| MTP x4, agent | 1,580 | 0.94 0.90 0.84 0.73 | 0.94 0.84 0.71 0.51 | 3.99 |
| DFlash2 x7, prose | 1,427 | 0.68 0.63 0.59 0.62 0.60 0.48 0.62 | 0.68 0.43 0.26 0.16 | 2.69 |
| DFlash2 x7, code | 700 | 0.87 0.81 0.79 0.78 0.80 0.76 0.80 | 0.87 0.71 0.56 0.44 | 4.39 |
| DFlash2 x7, agent | 1,095 | 0.90 0.90 0.91 0.93 0.90 0.90 0.89 | 0.90 0.81 0.74 0.68 | 5.76 |

Greedy and sampled agree within 0.03 at every position. The stock MTP head's prose profile **0.72 / 0.43 / 0.22** equals
DECODE-PLAN's 0.74 / 0.45 / 0.22 (cumulative). DFlash2 is not better than MTP on prose at any position (0.68 vs 0.72 at
position 1, the same 0.43 at position 2); it wins only through depth on code / agent content.

**Production policy** (auto: DFlash2 greedy rounds, MTP sampled, lookup, cost-derived depths; load DEF):

| class | greedy: tok/s, tokens a round (rounds m / f / l) | sampled: tok/s, tokens a round |
| --- | --- | --- |
| prose (5) | 46.2, 2.39 (587 / 483 / 0) | 46.2, 2.28 (MTP only) |
| prose, thinking on (2) | 47.3, 2.48 (369 / 455 / 0) | - |
| code (4) | 64.6, 4.10 (183 / 315 / 0) | 55.9, 3.32 |
| agent-like (5) | 83.6, 6.34 (258 / 264 / 110) | 77.1, 5.30 |

With the policy's depth cut, conditional acceptance of the drafts it does send is 0.71-0.73 at position 1 for prose
and flat (0.6-0.7) behind it; mean depth 1.9-2.0 (MTP) and 2.8 (DFlash2) on prose. Lookup rounds on the rewrites keep
15.5 tokens (of 16 rows). Full tables: `accept-DEF.json`, `acceptfix-4.json`, `acceptfix-f7.json`.

### 7. What changed in the +40% decomposition (details: DECODE-PLAN.md)

- Today, measured: 1 stream prose **53.3 ms, 2.42 tokens = 45 tok/s**; code 63.6 ms, 4.16 tokens; 4 streams **121.3 ms,
  9.1 tokens = 75 tok/s** (77.9-78.4 on the multiturn prompts). The plan's reconstruction (54.4 ms) was within 2%, but
  its split was not: idle 1.8 ms (est. 3.8), exchanges 2.3 (est. 1.7), dense 175 GB/s (assumed 190).
- Larger than estimated: **E2 dense q4** (-1.9 -> **-3.0 ms**: 175 GB/s against a 230.5 ceiling for the same set),
  **E8 RoCE** (-0.4 -> -0.8), **E7 vocab trim** (the head runs 2.6 times a 1-stream round and 4.9 at 4 streams:
  -0.8 -> -1.0 / -1.2 -> -2.5 ms), **F0** (now measured +2.2-2.6%).
- Smaller: **E5 graph coverage for a lone stream** (-1.0 -> -0.2: eager rounds have no idle penalty after 0370),
  **E6 device sampling** (-1.0 -> -0.6: 1.8 ms idle left in total).
- Confirmed: E1 experts (205-214 GB/s in situ vs a 235 ceiling: -2.7 ms mid), E4 (PDL available on sm_121, worth ~0.6
  ms of the 1.5 mid).
- Engineering mid: -10.5 ms 1 stream (**+25%**), -25 ms 4 streams (**+26%**); with the middle drafter case **+38% / +42%**.
  The 4-stream mid is lower than the plan's +28% / +44%: in situ the 4-stream expert reads already run at 208-220 GB/s,
  so F1 has less room than the 193 GB/s W7 figure suggested.

Ops notes: the first `load.sh nsys` wrote an empty `serve-w11.sh` (a sed `#` delimiter inside the replacement) and
"started" nothing in 2 s; fixed (python edit), started at 09:39. Under nsys the 4th slot did not fit (above); the second
capture load used `GLM53_TF_BATCH_RESERVE_GB=8`. One nsys window per server start was kept (two starts, two windows).
`pkill -f` of a waiting loop from inside `ssh` killed that ssh's own shell again (W8's note); nothing else affected.
Prod restored 10:27 from `config/prod.env`: ready in 25 s, canary ok (decode 80.5 tok/s), 4 slots, RoCE connected,
https `/v1/models` ok, `17*23` -> `391`, watchdog timer active (its 10:27 tick exited 0), lease refresher stopped,
lease deleted, `/var/tmp/w11` removed on both nodes (reports kept in `~/w11-traces/`).

## W12: decode A/B (0440 / 0450 / 0460 / 0490, the 4-stream graph policy), 2026-09-29 (12:49-17:13, one window, prod down 4 h 24 min) — 0460 L2 prefetch (8 MiB) and `CAPTURE_AFTER=8` adopted on image b5

One window with prod stopped: the GPU plans of DECODE-KERNELS §8 (0440), GPU-ROUND §6 (0450) and PREFETCH-COMM §6
(0460), the 0490 checks, SFXNZ-AUDIT's N1 (exactness past the indexer's top-2,048) and the 4-stream graph policy from
W11, then a combined candidate through the full gate set. Files: `results/W12/` (`tests-head/SUMMARY`, `tests-worker/SUMMARY`, `roce/`,
`cmp.py` / `lone.py` print the tables below from the per-load files, `loads.log`, `windows.log`).

**Image `glm53-tensorfold:b5`** = every patch through 0490 (`build-patches.txt`: 0420 / 0430 / 0470 present but off;
**0500 not included**), built on the head node, loaded on the worker node, tagged on both. Two **0460 bugs** found by the unit tests
and fixed in `patches/0460-glm-prefetch-comm.patch` before any engine load (the first b5 never served; it is
`glm53-tensorfold:b5-pre460fix` on the head node only):

- `roce.py` `_ext()`: the new comment on the `load(name="tensorfold_glm_roce_v3", ...)` line swallowed the
  `sources=[...]` argument (`# patches/0460: ... sources=[...]` on one line). The extension could not build, so **any
  image with 0460 would have fallen back from RoCE to NCCL**, knobs off included (`test_roce_patches.py`: 6 failed, 19
  errors; after the fix 25 passed).
- `l2pf.cu`: `cp.async.bulk.prefetch.L2` without an `__CUDA_ARCH__ >= 900` guard. The image's `TORCH_CUDA_ARCH_LIST`
  also builds compute_80, where ptxas refuses it, so `GLM53_TF_L2PF=1` could not load (offline, the compile test built
  sm_121 only). Guarded, with a line-prefetch loop for the pre-sm_90 targets (never run on GB10).

Test-harness fixes (tests only): the PDL race test's and the bench's roof-probe `load_inline` helpers had no C++
declaration of their function (`'late' / 'probe' was not declared in this scope`).

### 0. Control: b5 with every new knob off == production b4 (this also gates 0490)

Three control loads spread over the window (C at the start, C2 after the 0440 / 0460 loads, C3 at the end):

| check | C / C2 / C3 | prod b4 |
| --- | --- | --- |
| reply sha (ab.py 24.5k / 98k) | 8794a3463259cc2f in every load | 8794a3463259cc2f |
| exact / batchexact / W9 transcripts (alone and together) | 10/10, 4/4, identical to W9 load A, every load | same |
| glmbench tf / tweet / kit / edit (13 cells x 3) reply hashes | equal in all three controls and in every other load | - |
| N1 (§2) reply shas | 12 / 12 equal to prod b4 | - |
| 1 stream (geomean of 13 cells), each vs the mean of the three | +0.1 / +0.0 / -0.1% | - |
| 4 streams (mean of 6 reps) | 78.6 / 79.2 / 79.3 tok/s | W11: 77.9-78.4 |
| lone-request probe (8 x 384 tokens, prompt by prompt) | within +-0.3% of each other | - |
| prefill 24.5k / 98k (tok/s) | 1,611 / 1,605; 1,609 / 1,609; 1,622 / 1,618 | W10 FIN 1,614 / 1,602 |

**0490** (on in every load): `tests/test_api_context.py` + `test_prompt_tokens.py` in the image 20 passed, 1 skipped
(no tokenizer dir); `/v1/models` gives `max_model_len` = `context_length` = 1,048,576; a request of 13 + 1,048,576
tokens gets HTTP 400 `context_length_exceeded` with OpenAI's wording; the pool edge, 13 + 1,048,531 = 1,048,544
tokens (32 under CONTEXT, reservation 64 over it), is **admitted** and answers `OK` (b4 refuses it as more than the
whole pool); boot line `KV pool: 1048832 tokens` (4,097 pages). `api-C*.json`.

### 1. Kernel GPU tests and microbenches (prod stopped, everything under `timeout`)

| test | result |
| --- | --- |
| offline suites in the image | 0440 emulator 61 passed; compile 6 passed, **Triton 3.7.1: every `_qmm` epilogue fused** (E2 allowed); 0460 host / compile / RoCE protocol model 190 passed; 0450 compile / interpreter 2 passed (the rest need the interpreter without a GPU) |
| 0440 E1 / E2 bitwise + **PDL race** (`test_decode_stream_patches.py -k "experts or qmm or pdl"`) | **16 passed** (after the harness fix; before it the race test could not build its helper) |
| 0440 engines (`-k "windows or replies or resume or timing"`) | 9 passed; `test_decode_patches` / `test_batch_parallel_patches` with E1 + E2 on: 7 / 26 passed; `test_deep_verify_patches` 4 passed + the known W10 coverage assert (`deepest >= 9`, replies equal serial before it) |
| 0440 `bench_decode_kernels.py` (`kbench.json`, every row `same bits`, 1,062 timed variants) | **gate failed**: see below |
| 0450 KDA fold bitwise (`replay_slots` == `replay_layers`, `chain_slots` == replay + `chain`, rows 1-16, prologues 0-16) | **8 passed** |
| 0450 sampler at scale (`sampler.py`: `self_check`'s two halves at size) | **104,857,600 keyed draws** (100 seeds x 2^20 rows, 2 ranks x 28 candidates with ties, T 0.7, top-k 20, top-p 0.9) and **2.5 x 10^9 libm values** (log, log of -log, exp) against the node's numpy / `choose_rows`: **0 differences** |
| 0450 synthetic-engine tests (`test_gpu_round_patches.py`, `TRITON_INTERPRET=0`) | 7 failed (4 requests == serial for sample / resident / 1, the lone request + resume): **a harness artifact**, see below. The real engine is exact (§3) |
| 0450 regressions with `GPU_ROUND=1` | `test_decode_overlap_patches` 29 passed; `test_batch_parallel_patches` 5 failed / `test_batch_sessions_patches` 5 failed, all sampled cases or `sample_drafts` on CPU tensors (the same artifact; one test passes CPU logits to the device path) |
| 0460 `test_prefetch_comm_patches.py` (L2PF bulk / lines / touch, sites a f o e, the Batcher, RoCE loop runtime with `STRIPE_KB` 0 / 32 x `LEAN` 0 / 1 eager and in graphs) | **25 passed** (after the two 0460 fixes); `test_roce_patches` 25, `test_comm_patches` 15, `test_batch_parallel_patches` with L2PF 26, `test_decode_stream_patches -k pdl` with L2PF 1 passed |

**0440 microbench** (one GPU, median of 7 graph replays over 3 weight copies; best configuration of 7 E1 / 4 E2
settings, PDL on or off):

| window / shape | old (production) | best new | new / old | gate |
| --- | --- | --- | ---: | --- |
| experts R 1 / 2 / 4 (U 8 / 13 / 22) | 241 / 381 / 637 us (209-217 GB/s) | 309 / 501 / 847 us (163 GB/s; cfg 8,3,8,3 + PDL) | **0.78 / 0.76 / 0.75** | >= 1.05: fail |
| experts R 8 / 16, 4-slot mixes (U 35-110) | 201-218 GB/s | 160-163 GB/s | 0.74-0.80 | - |
| E1 probe 1 (no trellis decode) / probe 2 (no mma), U 8 / 22 | - | 319 / 323 us; 864 / 873 us | = new | data movement is the limit, not the ALU |
| dense 77,440 x 4,096 (head), M 1-16 | 816-852 us (209-218 GB/s) | 994-1,016 us | **0.82-0.86** | fail |
| dense 12,576 x 4,096 (KDA proj) | 137-149 us | 162-171 us | 0.85-0.89 | fail |
| dense 12,288 x 4,096 (MLP gate/up) | 135-140 us | 151-153 us | 0.89-0.91 | fail |
| dense 4,096 x 8,192 (DSA o / MTP eh) | 91-97 us | 105-107 us | 0.87-0.91 | fail |
| dense 4,096 x 6,144 (MLP down) | 77 us | 76-81 us | 0.96-1.01 | fail |
| dense small (2,048 x 4,096; 8,192 x 1,536 / 512; 4,096 x 1,024 / 4,096; 4,096 x 128) | 3-58 us | | 1.1-2.1x | (L2-resident in the bench: 3 copies < 24 MB) |

E1's persistent ring streams at ~160 GB/s whatever the configuration, and removing the decode or the mma changes
nothing: the loads in flight per warp are too few for GB10's latency. E2 wins only the small shapes, which the bench
keeps in L2; every shape that carries the bytes of a round loses 10-18%. Per the plan: stop E1. One engine load of
E2 anyway (below) measured the sign: -3.9%. No PDL / E1 engine loads.

### 2. N1: exactness past the indexer's top-2,048 (`bench/longexact.py`, new)

Three prompts of repo text (8,743 / 16,889 / 32,098 tokens, a different slice each, `n1-corpus.txt` frozen from
`docs/PATCHES.md` at HEAD), 512-token replies, thinking off: drafted vs serial (`"draft": false`) greedy (drafted
first, cold) and sampled (T 1, top-k 20, top-p 0.95, seed 1234; serial first, cold), then all six requests again
batched (4 + 2 at once) against their drafted replies alone.

| config | drafted == serial | batched == alone | reply shas vs prod b4 |
| --- | --- | --- | --- |
| prod b4 (before the window, live) | **6/6** | **6/6** | - |
| C (b5, knobs off) | 6/6 | 6/6 | 12/12 equal |
| FIN (the candidate, §4) | 6/6 | 6/6 | 12/12 equal |

No sign of the vLLM kpool class of bug (rejected drafts corrupting indexer pool keys): every decode token at these
lengths selects 2,048 of 8.7k-32k rows. Drafted decode at 8k / 16k / 32k: 55.7 / 49.9 / 45.4 tok/s greedy on prod,
59.1 / 51.8 / 46.1 on FIN; serial 31.0-31.5 -> 32.6-32.8 (+4-5%: the 1-row window gains most from the L2 prefetch).

### 3. Engine A/B (one load a setting, `config/prod.env` + the knob on b5; `ab.sh`)

Per load: exact, batchexact, the W9 transcripts, ab.py 24.5k / 98k once (reply sha, prefill), `glmbench --suites
tf,tweet,kit,edit --reps 3` (13 cells: **1 stream**, geometric mean of the per-cell medians; prose = tf chat T 0 / 1,
kit hashmap / essay; code = tf code T 0 / 1, tweet code; edit = the three rewrites), `multiturn --modes concurrent
--streams 4 --reps 3` twice (**4 streams**, mean of 6), and a **lone-request probe** (`slots.py`: 8 fresh 384-token
prose prompts one after another; the batcher gives a fresh lone request the least recently used slot, so they cover
slots 0-3 twice; compared prompt by prompt). Everything vs the mean of C / C2 / C3.

| load | knob | 1 stream: all / prose / code / edit | 4 streams | lone probe: slot 0 / slots 1-3 | exact, batchexact, transcripts, 13/13 cell hashes, sha | result |
| --- | --- | --- | --- | --- | --- | --- |
| C / C2 / C3 | none | +0.1 / +0.0 / -0.1% | 78.6 / 79.2 / 79.3 | +-0.3% | all pass | control |
| E2 | `DEC_QMM=1` (0440 E2) | **-3.9** / -5.3 / -4.5 / -2.1% | 76.6 (-3.1%) | -4.0 / -3.8% | all pass | off |
| G0 | `BATCH_GRAPHS=0` | +0.1 / -0.4 / +0.4 / +1.0% | 80.5 (+1.9%) | -1.4 / **-3.1%** (slots 1-3 run eager: -1.6 to -5.2%) | all pass | off (< +2%, lone slots slower) |
| CA8 | `BATCH_CAPTURE_AFTER=8` | -0.2 / -0.4 / -0.7 / +0.6% | 80.6 (**+2.0%**) | +1.6 / -0.0% | all pass | **adopt** |
| PF | `L2PF=1` (bulk, sites a,f,o, 4 MiB) | +0.8 / +0.8 / +0.3 / +1.0% | 78.9 (-0.1%) | +2.7 / +2.3% | all pass | below +1.5% |
| PFE | `L2PF=1`, sites a,f,o,e | +0.4 / +1.2 / -0.1 / -0.4% | 78.9 (-0.1%) | +3.1 / +1.5% | all pass | off |
| PF8 | `L2PF=1`, `L2PF_MB=8` | **+2.4** / +2.9 / +2.5 / +1.7% | 80.0 (+1.2%) | +3.5 / +2.6% | all pass | **adopt** |
| GRS | `GPU_ROUND=sample` | -0.1 / -0.4 / -0.2 / +0.5% | 78.7 (-0.4%) | -0.6 / +1.2% | all pass | neutral: off |
| GRR | `GPU_ROUND=resident` | -2.3 / -2.6 / -5.9 / +0.0% (tf code greedy -18%, hashmap -8%) | 79.4 (+0.5%) | -4.2 / +0.4% | all pass | off |
| GR1 | `GPU_ROUND=1` | -2.3 / -1.7 / -5.2 / -0.9% (tf code greedy -16%) | 80.0 (+1.2%) | -0.9 / -0.9% | all pass | off |
| GR1L | `GPU_ROUND=1`, `PEEK=0` | -4.3 / -5.8 / -8.5 / -1.2% | 75.6 (-4.3%) | -5.2 / -4.9% | all pass | off |

Bars (the patch docs): 0440 1 stream >= +3%; 0460 >= +1.5% 1 stream, 4 streams not lower; 0450 >= +3% at 4 streams
(median) and 1 stream not lower; graph policy >= +2% at 4 streams with lone requests in slots 1-3 not slower.

- **Every load is exact**: exact 10/10, batchexact 4/4, transcripts identical to W9 A, the reply sha, and all 13
  glmbench cells' reply hashes equal to the controls in every load (0440 E2, 0450 in all four modes, 0460 at every
  setting). The 0450 synthetic-test failures (§1) do not reproduce on the model.
- **0450's failing GPU tests are the harness**, not the sampler: the one-GPU tests play rank 1 with a copy of rank 0's
  candidates (`_TwoCopies`, `_Gather`), so every id appears twice with the same value. The host's lexsort takes one;
  the device kernel's greedy pick adds the tied winners (instrumented: local id 88 -> 176, 491 -> 982). Real ranks
  hold disjoint vocabulary halves, so the duplicate cannot occur (and the engine loads are exact), but the device pick
  and the host rule disagree on exact duplicates: a latent difference to fix in the kernel or the tests.
- **W11's open question on `BATCH_GRAPHS=0`**: lone requests in slots 1-3 do run graphs today (same speed as slot 0);
  with graphs off they run eager and lose 1.6-5.2%, and since the batcher rotates fresh lone requests over the least
  recently used slot, three of four land there. `CAPTURE_AFTER=8` keeps them (+0.0%) and gives the same 4-stream gain.
- **0460**: 8 MiB a site (plans 338-368 MiB a rank) clears the bar; 4 MiB does not; the device-indexed expert site
  `e` adds nothing. The 1-row serial windows gain most (N1: +4-5%).

**RoCE knobs (0460 E8, `roce.sh`, bench `--trace 4096` both nodes, bits equal everywhere):** per-op graph latency
(us) at 16 / 32 / 64 / 128 / 256 KiB: base 11.9 / 13.4 / 15.5 / 21.3 / 27.8 and base again 12.5 / 13.3 / 15.6 / 19.9 /
27.9 (the noise is +-0.5-1.4 us); `LEAN=1` 11.1 / 13.1 / 15.3 / 19.8 / 27.5; `STRIPE_KB=32` 14.2 / 13.4 / 16.4; `=64`
12.7 / 14.4 / 15.6; `INLINE=256` 12.0 / 13.4 / 15.5; `LAZY_CQ=1` 12.1 / 14.0 / 15.5; all together 11.5 / 14.8 / 15.4.
Only `LEAN` comes near the 0.3 us bar at 16-64 KiB (-0.2 to -1.4 us, within the base-to-base spread); at ~100
exchanges a round that is <= 0.1 ms (~0.2%), below any engine A/B, so **no RoCE knob went to stress or an engine load**.
The trace's `notice` component prints garbage (~1.8 x 10^15 us: a clock-domain mix-up in the stamp), the others look
sane: wait 5-18 us p50, of it skew 8-30 us across the two ranks, transport 0.9-2.2 us.

### 4. Combined candidate FIN = prod + `L2PF=1` + `L2PF_MB=8` + `BATCH_CAPTURE_AFTER=8` (b5): full gates (`gates.sh FIN`)

| gate | FIN | bar |
| --- | --- | --- |
| exact / batchexact, before and after the stress + MMLU | 10/10, 4/4; 10/10, 4/4 | 10/10, 4/4 |
| W9 transcripts (6 prompts greedy / sampled alone, 4 together) | identical to W9 load A | identical |
| glmbench reply hashes (13 cells x 3) | 13/13 equal to the controls | equal |
| reply sha, 24.5k / 98k x 2 | 8794a3463259cc2f in every cell | same |
| N1 (§2) | drafted == serial 6/6, batched == alone 6/6, 12/12 shas == prod b4 | all |
| decode vs the b5 controls (= prod b4) | **1 stream +2.5%** (prose +3.3, code +2.8, edit +1.0; every cell but print-to-log up); **4 streams 81.2 tok/s, +2.8%** (84.6 78.4 80.3 / 86.4 77.1 80.4); lone probe **+3.2%** (slot 0 +2.8, slots 1-3 +3.3) | not lower |
| prefill 24.5k / 98k (tok/s) | 1,602 / 1,579; 1,604 / 1,612 (controls 1,601-1,622; the 1,579 run is the low one, as W10 FIN's 1,577) | not lower |
| 4 x ~250k stress MemAvailable min, head / worker | **9.74 / 8.32 GiB**, PASS; filled in 785 s, longest decode gap 3.9 s | >= 8 |
| OOM kills (dmesg, both nodes) | 0 / 0 (the 2 `out of memory` lines on the worker node are NVRM allocation messages at 13:51-13:52, during b5's first start, below) | 0 |
| MMLU-200, refusals | **88.0%** (176/200), 0/10 | >= 87% |
| needle, 314,326 tokens alone after the stress + MMLU | **found** cold and resumed (cold prefill 1,347 tok/s, decode 61.8; resume 314,304 cached in 0.25 s, decode 71.1) | found |
| /health | 410 requests, 0 errors | - |

**All gates pass.** Two notes:

- **Memory during the needle.** MemAvailable sampled every 2 s over the whole gate run bottomed at **8.27 / 6.58 GiB**,
  in the needle's last 30 s on the worker node (14 samples under 8), after the stress and MMLU left the pool and session store
  full; no OOM. W10 FIN saw the same pattern at 8.55 / 7.39 and recorded it as an open end. The stress minimum on
  worker is also ~1 GiB under W10 FIN's (8.32 vs 9.36) while the idle footprint after load differs by 0.2-0.4 GiB
  (MemAvailable at `engine ready` on the worker node: FIN 16.9, C3 17.1, W10 FIN 17.3 GiB), so most of the gap is the longer
  gate sequence before the stress (ab.sh, a second ab.py, N1's six long sessions), W9's ~3 GiB drift effect. No
  like-for-like needle-after-stress control on b4 exists; the stress gate (the documented worst case) passes.
- **b5's first start** loaded from the checkpoint (0470 keys the prepared folders by precision, so b4's did not match),
  wrote new prepared folders (~84 GB a node) and, with the page cache full, **fit only 1 slot** (canary 32 tok/s); that
  load was aborted before any measurement (`first-start/`) and every later start read the prepared folders (ready in
  ~75 s, 4 slots). The NVRM `Out of memory` lines on the worker node are from that start.

**Adopted:** `config/prod.env` -> `IMAGE=glm53-tensorfold:b5`, `GLM53_TF_L2PF=1`, `GLM53_TF_L2PF_MB=8`,
`GLM53_TF_BATCH_CAPTURE_AFTER=8` (was 3); everything else as W10 (the previous file is
`results/W12/prod.env.before-W12`). 0490 comes with the image (on by default). Not adopted, knobs off: 0440
(`DEC_QMM` -3.9%, `DEC_EXPERTS` 0.75x in the bench), 0450 (`GPU_ROUND` every mode), `BATCH_GRAPHS=0`, L2PF at 4 MiB
and site `e`, 0460's RoCE knobs. Prod restarted on it at 17:13: ready in 26 s, canary ok (decode 73.7 tok/s), 4 slots,
RoCE connected, the `l2pf: bulk, 8 MiB a site` line on both ranks, https `/v1/models` ok (now with `max_model_len`
1,048,576), `17*23` -> `391`, live `exact` 10/10, watchdog timer active (its 17:13 tick exited 0), lease refresher
stopped, lease deleted. The NVMe session store starts cold (new image id).

**What W12 says about the decode plan** (DECODE-PLAN updated): of the engineering items built so far only E3 (L2
prefetch, +2.4% 1 stream) and F0's gentle form (+2.0% at 4 streams) paid. E1 / E2 as built are slower than the
kernels they replace (E1's ring is capped at ~160 GB/s by loads in flight; E2 loses every large shape), and 0450's
resident rounds cost more than they save at 1 stream and recover +0.5-1.2% at 4. Measured today vs W11: 1 stream +2.5%,
4 streams +2.8%.

Ops notes: `pkill -f` from an ssh command killed that ssh's own shell once more (worker, 13:07; nothing else hit),
after which processes were stopped by PID. An `rsync` of three source paths without `-R` put `results/W12` at the
repo root on the head node once (removed). Editing `tests-node.sh` while a copy ran on the worker node was harmless (bash had parsed
the step list). The first `g-unit` run was killed at 18 min: the module forces `TRITON_INTERPRET=1`, so on the GPU
node its kernels ran in the CPU interpreter; `g-unit0` reran it with `TRITON_INTERPRET=0` (its own 3-second check).
One lease refresher restart at 15:59 (a new 240-minute cap).

## W13: RigMark on TensorFold production (2026-09-29 17:19-17:29, prod b5 serving, no restart) — standard suite, all runs valid

RigMark's standard suite against production as it serves (image `glm53-tensorfold:b5`, `config/prod.env`), no vLLM run:
the comparison is against Alex Ellis's two published GLM-5.3 TP2 vLLM receipts. Files: `results/rigmark/tensorfold-20260929/`
(receipt `glm53-flash-exl3-tensorfold-tp2-low.json` + `.sha256`, card, `run.log`, `command.txt`, `metadata.json`,
`preflight.json`, `models.json`, `requests.jsonl` = the server's 0300 request log for the run, no text),
`results/rigmark/compare-published-20260929.md` (`compare_published.py`: the table below with every row),
`results/rigmark/rigmark-compare-*-vs-tensorfold.txt` (RigMark's own `compare --allow-mismatch` against each receipt).

- **How.** `scripts/rigmark/install.sh` cloned RigMark to `~/rigmark` on the head node at the pin `c5a0db01b054` (clean),
  then `scripts/rigmark/run.sh tensorfold`: standard settings (no count / length / depth / concurrency flags),
  `--extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}'`, comparison ID
  `2026-09-glm53-exl3-2xspark-vllm-vs-tensorfold-v1`, loopback client on the head node, 643 s.
- **Preflight** (`preflight.json`): chat, `/tokenize` and token-ID `/v1/completions` all ok (0490), so the prefill phase
  ran (no `--skip-prefill`); thinking on with the low-effort body. One gap: reasoning is sent as both
  `delta.reasoning` and `delta.reasoning_content`, so the receipt's `reasoning_characters` are doubled (timings and
  tokens unaffected). `GLM53_TF_REASONING_FIELDS` needs a restart, so production was left as it is and the column is
  halved below.
- **Quiet endpoint.** `/health` inflight 0 at the start; the request log shows requests n = 27-82 during the run, all
  RigMark's (15 decode, 18 prefill, 21 concurrency, 2 preflight): no other traffic. The watchdog timer stayed on
  (read-only ticks). The receipt's `competing_traffic` text says "test window held" (a fixed string in `run.sh`);
  no window was held, the endpoint was idle.

**Card** (`rigmark report` regenerates it byte-identical from the receipt):

```
│  ●  15/15 BASIC OUTPUT GATES PASSED                                                      │
│  MODEL      GLM-5.3-Flash-EXL3                                                           │
│  APPLIANCE  2x NVIDIA DGX Spark (GB10, 128 GB unified memory each)                       │
│  RUN        reasoning=low  •  protocol=1.1.0                                             │
│  SOURCE     git:c5a0db01b054  •  clean                                                   │
│  CODE              68.6 tok/s      26.3s last     66.5–69.3     ✓ 5/5                    │
│  PROSE             43.2 tok/s      24.1s last     43.0–43.5     ✓ 5/5                    │
│  STRUCTURED*       88.2 tok/s       8.2s last     87.3–89.1     ✓ 5/5                    │
│  64K PREFILL   cold 1,621 tok/s  •  immediate replay 6,304 tok/s                         │
│  AGGREGATE   C1 54.3  •  C2 65.5  •  C4 82.2 tok/s                                       │
│  C4 OUTPUT STATE   normal stop 0/12  •  visible 12/12  •  reasoning may be included      │
│  JSON       sha256:4af01c23364e22b6…                                                     │
```

**Validity** (every check RigMark makes, plus the receipt itself):

| check | result |
| --- | --- |
| basic output gates (visible answer, `finish_reason` stop, `[DONE]`) | 15/15 (code 5/5, prose 5/5, structured 5/5) |
| structured output: exact 50-object JSON array | 5/5 valid; all 5 outputs identical (one sha) |
| prefill: server `prompt_tokens` == requested depth | 18/18 rows (8,192 / 32,768 / 65,536, cold and replay) |
| concurrency streams | 21/21 visible; all stop at the 256-token cap (`length`), as in both published receipts (a capacity test) |
| receipt | `sha256sum -c` ok; RigMark tree clean at the pin; protocol 1.1.0, prompts 1.0.0 (`0c3ac401…`) |
| strict `rigmark compare` vs either published receipt | refused, as expected: protocol 1.0 vs 1.1, revision, comparison ID, source sha differ; `--allow-mismatch` used for the files above |

Nothing was flagged invalid.

**Side by side with the published vLLM TP2 receipts** (medians; ratios = ours / theirs; full table with ranges, TTFTs,
token and reasoning counts in `compare-published-20260929.md`):

| metric | TensorFold (ours) | vLLM TP2 k=7 (`glm53-libert-nvfp4-tp2-low`) | vLLM TP2 adaptive (`glm53-libert-nvfp4-tp2-adaptive-low`) | ours / k=7 | ours / adaptive |
|---|---:|---:|---:|---:|---:|
| code decode tok/s (5-run range) | **68.6** (66.5-69.3) | 44.0 (31.6-47.7) | 42.6 (39.0-44.7) | 1.56x | 1.61x |
| code time to last output s | **26.3** | 46.6 | 50.2 | 0.57x | 0.52x |
| prose decode tok/s | **43.2** (43.0-43.5) | 18.9 (17.8-19.3) | 22.2 (21.8-22.3) | 2.29x | 1.95x |
| prose time to last output s | **24.1** | 55.7 | 45.4 | 0.43x | 0.53x |
| structured ceiling tok/s | **88.2** (87.3-89.1) | 64.9 (63.9-66.8) | 54.6 (53.0-55.7) | 1.36x | 1.61x |
| structured time to last output s | 8.2 | **7.3** | 8.7 | 1.13x | 0.95x |
| structured completion tokens | 687 | 441 | 441 | 1.56x | 1.56x |
| TTFT s, code / prose / structured | 0.51 / 0.40 / 0.46 | 0.60 / 0.49 / 0.47 | 0.59 / 0.47 / 0.47 | 0.84-0.98x | 0.86-0.97x |
| cold prefill 8K / 32K / 64K tok/s | 1,598 / 1,641 / 1,621 | **1,813 / 1,908 / 1,922** | 1,835 / 1,898 / 1,905 | 0.88 / 0.86 / 0.84x | 0.87 / 0.86 / 0.85x |
| immediate replay 8K / 32K / 64K tok/s | 1,597 / 3,319 / 6,304 | 1,812 / **11,046 / 11,364** | 1,831 / 11,339 / 11,464 | 0.88 / 0.30 / 0.55x | 0.87 / 0.29 / 0.55x |
| C1 / C2 / C4 aggregate tok/s | **54.3 / 65.5 / 82.2** | 31.6 / 42.0 / 66.1 | 31.2 / 42.8 / 61.1 | 1.72 / 1.56 / 1.24x | 1.74 / 1.53 / 1.34x |
| C1 / C2 / C4 per-stream decode tok/s | **60.5 / 37.5 / 24.6** | 33.9 / 23.9 / 18.7 | 33.1 / 24.9 / 16.4 | 1.78 / 1.57 / 1.32x | 1.83 / 1.51 / 1.50x |
| C1 / C2 / C4 per-stream TTFT s | 0.48 / 0.93 / 1.91 | 0.60 / 0.68 / **0.81** | 0.54 / 1.07 / 0.78 | 0.80 / 1.37 / 2.34x | 0.89 / 0.86 / 2.46x |
| reasoning chars, prose (ours halved) | 60 | 60 | 60 | | |

**Not a strict comparison.** The three receipts differ in things RigMark does not control, so this is an appliance
comparison in RigMark's sense (same corpus, settings, request body, counts, lengths, depths, concurrency), not an
engine-only or matched sweep:

- **Weights.** Ours: `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` (abliterated, EXL3 4-bit experts, non-expert
  weights re-quantized to 4-bit at load). Alex's: `LibertAIDAI/GLM-5.3-Flash-NVFP4` (ModelOpt NVFP4). The outputs
  differ (e.g. our structured answer is pretty-printed JSON, 687 tokens / 1,600 chars, against their compact 441 /
  1,299), so the structured ceiling and time-to-last-output rows compare different token streams.
- **Drafting.** Ours: DFlash2 `7d74cdd` + the MTP head, cost-derived depth up to 7, verify windows up to 16 rows,
  suffix lookup. Theirs: DFlash2 `bf582e4` at k=7 (static) or adaptive k=3/5.
- **Engine and memory.** TensorFold with our patches vs vLLM `0.1.dev20051`; FP8 latent KV, 1,048,576 context,
  4 slots vs FP8 E4M3 KV, 262,144 context, 6 sequences.
- **Hardware and runs.** A different pair of Sparks (same model, both direct-link 200 Gb/s RoCE), a different day,
  protocol 1.1.0 vs 1.0.0, x86 vs aarch64 client. RigMark refuses the pair strictly; no rerun of theirs on our pair.

**Readings.**

- Decode (the drafter-driven phases) is well ahead: code +56-61%, prose +95-129%, C1 +72-74%. This matches the
  internal gap between TensorFold and our own vLLM kit (W1-W12); the prose gap is larger than code's because GLM's
  prose accepts few DFlash2 drafts and TensorFold's MTP + cost-derived depth recovers more of it.
- Cold prefill is 12-16% below vLLM NVFP4 (1,598-1,641 vs 1,813-1,922 tok/s): the known prefill gap (PREFILL-ANALYSIS).
- **Immediate replay is where TensorFold loses clearly**, and the request log says why: the replays resumed
  **0 / 16,384 / 49,152** tokens at 8K / 32K / 64K (`cache_src` ram). RigMark's depths are exact multiples of the
  prefill grid, so the prompt's last grid snapshot is the whole prompt, and a resume needs a strict prefix
  (`decode._prefill`); the next snapshot down is the 16,384-token mark (`GLM53_TF_SESSION_EVERY`). A replay re-prefills
  the last 8-16K tokens (9.8-10.3 s). Chat prompts are rarely on the grid, so agents resume within 64 tokens of the
  end (W14's image conversations: 448 of 481); the RigMark case is an edge. Fix idea (not done): when a prompt ends on
  the grid, also keep the snapshot one grid step earlier; the 32K / 64K replays would then prefill 64 tokens.
  **Done offline as patches/0540 (the snapshot moves to n - 64; docs/REPLAY-TTFT.md; GPU run pending).**
- C4: aggregate +24-34%, per-stream decode +32-50%, but per-stream TTFT 1.9 s vs 0.8 s: the batcher admits the
  concurrent prompts one prefill piece at a time (`queue_s` 0.47-0.91 s, request log), where vLLM prefills them
  together. patches/0540 (offline) emits each first token when its piece ends (0370 held it to the round's end);
  multi-slot prefill analysed in docs/REPLAY-TTFT.md.

## W14: vision (patch 0500) on the GPU, image b6 (2026-09-29, three windows 17:32-17:39, 17:45-18:25, 18:26-18:32; prod down 53 min) — works, one 0500 bug fixed, not adopted

docs/VISION.md §7 on the Sparks. Files: `results/W14/` (`run-window.sh`, `run-window2.sh` the whole sequence;
`tests/`, `tests-worker/` GPU tests; `*-V0` / `*-V1` text gates; `correct-V1.*`, `cache-V1.*`, `ttft-V1.*`, `stress-V1.*`,
`needle-V1.*`, `mem-V1.log` + `mem-marks.txt`, `replay-V1R.*`, `fresh-V1F.*`, `pair2-*.json`, `disk1/2.json`,
`pixels.json`; `img/` the generated test images with `truth.json` (`mkimg.py`, all synthetic); `vis.py` the client;
`hf_tower.py` / `tower_ours.py` the tower against transformers).

**Image `glm53-tensorfold:b6`** = b5's patch list (0001-0490, `build-patches.txt`) + 0500, built on the head node, loaded on
worker, tagged `b6` on both. 0500 applies on the stack through 0490 (file by file with `patch`, no rejects; the whole
stack through 0530 also applies with the fixed 0500). The first b6 (`b6-premtpfix`, head only) failed the GPU test
and never served.

**0500 bug found and fixed (`patches/0500-glm-vision.patch`, a `decode.py` hunk):** the MTP head's prefill absorb
(`decode.absorb` -> `Engine.mtp`) replays the head's **CUDA graphs**, which cannot do `glue.embed`'s row substitution,
so an image's virtual ids (>= 2^24) were read as vocabulary rows: an illegal memory access
(`test_image_rows_equal_their_tokens`, pinned with `CUDA_LAUNCH_BLOCKING=1` to `decode.mtp`'s `g.replay()`; every
later test then failed on the poisoned context). VISION.md's "graphs only ever hold decode windows" missed the head's
graphs. Fix: `Engine.mtp` runs eager (no graph, no long-context graph) when `vision.ACTIVE` is set and the rows carry
virtual ids. Production's config was not exposed (latent KV + `GLM53_TF_MTP_PREFILL_CACHE=1` absorbs through
`pfglue.cache_absorb`, eager), but `MTP_PREFILL_CACHE=0` or the per-head KV cache would have crashed rank 0 on the
first image.

### 1. GPU tests (prod stopped)

| test | result |
| --- | --- |
| `tests/cuda/test_vision_patches.py` (synthetic engine; real tower via `GLM53_TF_MODEL`), after the fix | **7 passed**, 1 failed: the real-tower test's bf16-vs-fp32 bound (below) |
| the same on the production MTP path (`GLM53_TF_LATENT_KV=1`, `GLM53_TF_MTP_PREFILL_CACHE=1`, bf16 latent: the synthetic checkpoint's 128-wide latent cannot take fp8) | **7 passed** |
| TensorFold's `test_glm_engine.py` in b6 (worker) | 12 passed |
| host vision tests in the image (`test_vision_prep`, `test_vision`, `test_vision_server`) | 58 passed, 26 skipped (the transformers-parity checks: no transformers in the image; parity was verified offline) |
| real tower, 1024 x 768 | 1,036 rows, **deterministic** (two runs bit-identical), 196-199 ms (27 TFLOP/s), peak +1.43 GB with the 1.13 GB tower |
| real tower, 3840 x 2160 | 7,973 rows, 2.75 s, peak +2.35 GB |
| bf16 vs fp32, same tower | 0.059 relative: **fails the test's 2% bound**, see next row |
| against transformers' `Glm5NextVisionModel` (head venv, transformers 5.17, same inputs) | transformers' own bf16 vs fp32: **0.060** (random image) / 0.168 (screenshot); ours 0.059 / 0.180. Our fp32 vs theirs fp32: 0.004 (row cosine min 0.996) / 0.021 (mean cosine 0.99955; their Conv3d patch embed runs on cuDNN with TF32). The tower matches the reference; the 2% bound is below what bf16 gives on this model and should be ~8% (test not changed) |

Latency is 2.2-2.6x VISION.md's estimate (75-90 ms for 1024 x 768), still 14% of that image's TTFT (§4 below).

### 2. Text unchanged: vision on vs off (b6), and vs production

| check | V0 (`GLM53_TF_VISION` unset) | V1 (`GLM53_TF_VISION=1`) |
| --- | --- | --- |
| exact (glmbench) | 10/10 | 10/10 |
| batchexact | 4/4 | 4/4 |
| W9 transcripts (alone, together == alone) | identical to W9 load A | identical to W9 load A |
| reply sha, ab.py 24.5k / 98k (cold and warm) | 8794a3463259cc2f x4 | 8794a3463259cc2f x4 |
| glmbench tf / tweet / kit / edit reply hashes (13 cells) | = V1 13/13, = W12 FIN 13/13 | = V0 13/13 |
| 1 stream geomean (13 cells, 1 rep) / prefill 24.5k, 98k | 73.1 tok/s / 1,608, 1,617 | 73.6 / 1,615, 1,612 (W12 FIN 73.4 / 1,602-1,612) |
| MemAvailable at engine ready, head / worker | 17.8 / 16.5 GiB (first start: kernel compile, ready 209 s) | 17.3 / 16.8 GiB (ready 60 s; boot line `vision tower: 1.13 GB on the GPU (7.3s)`) |
| /health errors, r0 error lines | 0, 0 | 0, 0 |

### 3. Real images (V1, greedy, `reasoning_effort` low; max where noted)

| check | result |
| --- | --- |
| (a) describe a 1920 x 1080 desktop screenshot (2 windows, calculator display, taskbar clock) | both titles exact (`notes.txt - Text Editor`, `Calculator`), display `391`, clock `14:32`, the note's 6 lines quoted |
| (a) describe a 1024 x 768 drawn scene (JPEG) | house, door, window, tree, sun, sky, grass |
| (b) read a terminal screenshot (3 commands + output) | low: CER 0.118, only because it also copied the window title `user@demo: ~` (every command / output line exact); **max: CER 0.000** |
| (b) a 12 px paragraph (3 lines) / a receipt | **CER 0.000** / TOTAL 22.50 and all 4 items |
| (c) count 3 / 7 / 12 circles, a 5 x 4 star grid | 3, 7, 12, 20 at low and at max (8/8) |
| (d) two charts in one message, "which is larger", both chart orders x images-first / text-first | 4/4 right, both values quoted |
| (e) RGBA PNG, black text on transparent | `TRANSPARENT OK 42` (reads as text on white) |
| (f) `http://` URL (local `http.server` on the head) vs the same image as `data:` | same `prompt_tokens`, **byte-identical reply**; bare-string `image_url` shape ok |
| refusals (400, `param: messages`, nothing streamed) | 30 MB image (`MAX_BYTES`), 10k x 10k (`MAX_PIXELS`, from the header), 20k x 20k (Pillow's own decompression-bomb check fires first), dead URL, 9 images, `file:` scheme; the next text request is served (`391`), `/health` ok, 0 errors |
| `usage.prompt_tokens` | text + rows (e.g. 1,036 rows for 1024 x 768), as VISION.md; no vLLM-kit comparison run (skipped: prod down for the kit start; counts already equal transformers' processor offline) |

### 4. Caches

| check | result |
| --- | --- |
| same image conversation twice | 2nd `cached` 448 of 481 (the 64-row grid point), tower not rerun (`encoded` 0), **reply identical** |
| 3 turns, image in turn 1 | turns 2 and 3 `cached` 320 (past the image: 285 rows in a 324-token prompt, ending before position 320), `encoded` 0 |
| image B after image A, same text (28 text tokens before the image) | B `cached` 0 (<= 28), B's reply == **B alone on a fresh session dir** (V1F) |
| the same with 825 text tokens before the image (D2 after D1) | B `cached` 0: no snapshot inside the shared text yet (the store puts a fork mark there for the next request, SESSION_FORK_MIN 512); B's reply == B fresh (e68ffbc90c288686). The image rows never resumed across images |
| NVMe tier: restart, same conversation | the 390-token 3-turn conversation is under `GLM53_TF_SESSION_DISK_MIN` (1,024) and was not written: served cold, reply identical. A 2,735-token screenshot conversation (D1), restart (D3): **`restored_disk`**, 2,688 cached from NVMe (115 MB in 57 ms), **reply identical**. The tower reran after the restart (0.74 s; the row cache is per process) although no image row was prefilled: a small follow-up (skip encoding when the resume covers every image row) |

### 5. Memory under 4 x 250k with vision on (V1, after the image and cache checks, tower + row caches warm)

| phase | MemAvailable min head / worker | bar / reference |
| --- | --- | --- |
| stress fill (4 x ~250k, 773 s; longest decode gap 3.9 s) | **9.43 / 9.82 GiB**, PASS | >= 8; W12 FIN 9.74 / 8.32 |
| an image request right after the stress (pool and store full) | served (`7`), no dip below the stress minimum | |
| needle (298,352 tokens, alone after the stress) | **8.34 / 9.07 GiB**; found cold (1,383 tok/s) and resumed (298,304 cached) | W12 FIN 8.27 / **6.58** (worker) |
| OOM kills | 0 / 0 (the worker node's 2 `out of memory` lines are W12's NVRM messages) | 0 |

With the tower rank 0 is now the lower node (it was ~1 GiB above rank 1 without it). The needle dip stayed above
8 GiB here, but this run had no MMLU / N1 / second ab.py before it, which W12's 6.58 GiB followed; it is not a
like-for-like answer to that open end.

### 6. TTFT per image size (V1, thinking off, 8-token reply, 3 unique images a size, medians)

| image | prompt tokens (rows) | TTFT | tower (`vision_s`) | LM prefill (`prefill_s`) | tower share |
| --- | --- | --- | --- | --- | --- |
| 512 x 512 | 397 (361) | 0.76 s | 0.058 s | 0.69 s | 8% |
| 1024 x 768 | 1,071 (1,036) | 1.44 s | 0.196 s | 1.20 s | 14% |
| 1920 x 1080 | 2,724 (2,691) | 2.82 s | 0.657 s | 2.07 s | 23% |
| 3840 x 2160 | 8,009 (7,973) | 8.10 s | 2.66 s | 5.25 s | 33% |

The tower is slower than VISION.md §4 (27 TFLOP/s against the 55-65% of peak assumed) and its share grows with size;
the LM prefill of the rows is as estimated (~0.9-1.5k tok/s at these lengths).

**Not adopted** (as asked): `config/prod.env` unchanged; production restored to b5 after each window (17:39, 18:25,
18:32: `/v1/models` local and https, `17*23` -> `391`, canary ok 76.8 tok/s, watchdog timer active, refresher killed,
lease deleted; the prod container has no `GLM53_TF_VISION`). The b5 NVMe session store restarted cold (the test loads'
session compat dirs pushed it out, as in W12). To adopt later: IMAGE b6 + `GLM53_TF_VISION=1`; rank 0 becomes the
low-memory node; relax the tower test's bound; consider `GLM53_TF_VISION_FETCH=0` if the API is ever exposed.

## W15: vision + replay/TTFT in prod (2026-09-29, one window 19:43-21:18, prod down 1 h 35 min) — image b7 with 0500 + 0540, `GLM53_TF_VISION=1` and `GLM53_TF_REASONING_FIELDS=reasoning` adopted

Files: `results/W15/` (`run-window.sh` the whole window; `tests.sh` every test run, logs in `tests-cpu/`, `tests-gpu/`,
`tests-gpu2/`, `tests-gpu3/`, `tests-worker/`; `gates.sh` = W12's gate sequence with phase marks and a PRE hook; `B7` the
candidate's gate files, `B7b` its restart for the replay / C4 / cache checks, `E0` the EMIT_FIRST=0 control, `b5ctl*`
live production b5 before the window; `replay.py`, `c4.py`, `vis.py` (W14's with the sha over `reasoning`),
`meminfo.sh`, `cmp.py` (W12's, reads `W12/...` tags), `rigcmp.py`; `prod.env.before-W15`).

**Image `glm53-tensorfold:b7`** = b5's patch list (0001-0490, as `results/W12/build-patches.txt`) + 0500 + 0540
(`build-patches.txt`), built on the head node (0510-0530 skipped; 0540 applies with offsets only, it was written on the stack
through 0530), shipped with `docker save | docker load` to the worker node, tagged `b7` on both. The whole stack through 0540 also
still applies.

### 1. Tests

| run | result |
| --- | --- |
| CPU, in b7, prod still serving (`tests-cpu/`) | `test_replay_ttft_patches` 32 passed; `test_session_patches` 21 passed (11 GPU-only skipped); `test_decode_overlap_patches` 30; host vision (`test_vision_prep`, `test_vision`, `test_vision_server`) 58 passed, 26 skipped (transformers parity); `test_api_context` + `test_prompt_tokens` + `test_openai_compat` 42 passed |
| GPU, first pass (`tests-gpu/`, head) | replay 32, batch sessions 28, cindep 54, overlap 30 passed; **vision 7 + 1 failed, vision latent path 6 + 1, fastpf 25 + 4, session 31 + 1**; worker: TensorFold's `test_glm_engine.py` **10 + 2 failed**, replay 32 |
| cause of all 9 failures | expectations of the old snapshot rule, not 0540's code: each asserted a resume at the prompt's end (`cached == len(first)`, a 256-token on-grid snapshot, a next turn resuming >= 112 of 112 tokens) or a store eviction that 0540's smaller / duplicate snapshots no longer force. With `GLM53_TF_SNAPSHOT_BEFORE_END=1` the snapshot is at `(n - 1) // 64 * 64` (64 of 70, 64 of 112, 128 of 256 on the tests' 128 grid), exactly 0540's documented rule; every later exactness assert in those tests (resume == fresh, rank 1 == rank 0) passed once the expectation was fixed |
| test updates (tests only) | `test_fastpf_patches.py`: the on-grid case expects the snapshot at 128 with the knob on, 256 off; `test_vision_patches.py`: tail 30 tokens (was 9) so the last grid point (128) lies past the image, assert `cached >= text + image`; `test_session_patches.py::test_rank1_follows_rank0`: budget 4 entries + 1 extent (was + 2: with 0540, 6 of 19 saves are duplicates, the entries share one extent and nothing was evicted), asserts split; TensorFold's `test_glm_engine.py` (`test_resumed_prompts_equal_fresh_prefills`): a hunk added to `patches/0540` expecting `(len(first) - 1) // 64 * 64` with the knob on (b7's image predates the hunk; the runs mount the patched file, the source files of b7 are byte-identical to the updated patch's) |
| GPU, after the updates (`tests-gpu2/`, `tests-gpu3/`, `tests-worker/`) | vision **8 passed** incl. the real tower (bf16 vs fp32 **0.0591**, under the 8% bound; W14 0.059), vision latent path 7, fastpf 29 with the knob on and 29 off, session **32 on and 32 off** (evicted 14 / restores 5 on, 14 / 6 off), vision with the knob off 7, `test_glm_engine.py` **12 on and 12 off** |

### 2. Candidate B7 = `config/prod.env` + IMAGE b7 + `GLM53_TF_VISION=1`, `GLM53_TF_VISION_CACHE_MB=64`, `GLM53_TF_VISION_PREP_MB=64`, `GLM53_TF_REASONING_FIELDS=reasoning` (0540's knobs default on)

First start of the new image: ready in 206 s (kernel JIT for the new image id), then 30-40 s; boot lines `vision tower:
1.13 GB on the GPU`, `image_url parts on`, MemAvailable at engine ready 16.7 / 16.1 GiB. Order: the image checks first
(so the tower and its caches are warm for the stress), then W12's `gates.sh` sequence unchanged (ab.sh, ab.py again, N1,
4 x 250k stress, MMLU-200, exact / batchexact again, needle), MemAvailable every 2 s with phase marks; then a restart (B7b)
for the replay, regenerate, C2 / C4 and image-cache checks on a fresh server.

| gate | B7 / B7b | bar / reference | |
| --- | --- | --- | --- |
| exact / batchexact, before and after the stress + MMLU | 10/10, 4/4; 10/10, 4/4 | 10/10, 4/4 | pass |
| W9 transcripts (alone, together == alone) | identical to W9 load A | identical | pass |
| reply sha, ab.py 24.5k / 98k x 2 | 8794a3463259cc2f in every cell | same | pass |
| glmbench tf / tweet / kit / edit (13 cells x 3) reply hashes | **13/13 equal to W12** (controls and FIN) | equal | pass |
| N1 (`bench/longexact.py` 8.7k / 16.9k / 32.1k) | drafted == serial 6/6, batched == alone 6/6, **12/12 shas == W12 FIN and prod b4** | all | pass |
| prefill 24.5k / 98k (tok/s) | 1,610 / 1,607; 1,605 / 1,606 | W12 FIN 1,602 / 1,579; 1,604 / 1,612 (98k within -1%) | pass |
| cold prefill of grid-aligned prompts (token ids, RigMark's shape, median of 3) | 8,192: **1,565** (5.24 s); 32,768: 1,636; 65,536: 1,619; off-grid 32,700: 1,640 | b5 live the same day: 1,593 / 1,631 / 1,618 / 1,634; W13 1,598 / 1,641 / 1,621 | 8K **-1.8%** (bar -4%): 0540's one extra 64-row chunk, ~0.09 s; 32K / 64K +0.3% / +0.1% |
| decode 1 stream (13-cell geomean) / 4 streams (mean of 6) | **-0.4%** vs W12 FIN (every cell within -2.3..+1.2%); **80.6 tok/s** | W12 FIN 81.2 (its passes 81.0 / 81.4; W12 controls 78.6-79.3) | within noise |
| lone-slot probe | 53.0 / 53.9 / 52.0 / 54.7 tok/s (slots 0-3) | W12 FIN 53.9 / 52.0 / 54.8 / 53.0 | same |
| replay: identical token-id prompt resent (3 pairs a depth) | 8K / 32K / 64K: **cached 8,128 / 32,704 / 65,472 (= n - 64)**, 1 piece, **TTFT 0.23 / 0.26 / 0.29 s** (max 0.296); off-grid 32,700: 32,640, 0.24 s; the 8 tokens equal cold's every time | cached n - 64, TTFT <= 0.5 s; b5 live: cached 0 / 16,384 / 49,152, TTFT 5.13 / 10.2 / 10.9 s | pass |
| chat regenerate (14,035-token prompt, greedy and seeded T 1) | 2nd cached 14,016, TTFT 8.78 -> 0.18 s, replies identical | within 64, identical | pass (off-grid: as b5) |
| C4 first tokens (4 x ~100-token prompts released together, 256 tokens, 3 rounds; `c4.py`) | thinking off: first tokens staggered per piece, **0.34 / 0.68 / 1.10 / 1.44 s**, median **0.93 s**; with `GLM53_TF_EMIT_FIRST=0` (load E0) 0.38 / 1.49 / 1.49 / 1.49, median 1.46 s. Thinking on (RigMark's `reasoning_effort` low): median 1.68 s, the 2nd-4th still together | 0540 §2 | mechanism works (-36% median C4 TTFT without thinking); **no effect with thinking on**, see below |
| images (`vis.py correct` / `cache`, before the gates and again on B7b) | screenshot: both window titles, calculator `391`, clock `14:32`; scene objects; receipt TOTAL 22.50 + 4 items; 12 px paragraph CER 0.000; terminal CER 0.000 at max (0.118 at low: copies the title, as W14); circles 3 / 7 / 12 and 20 stars at low and max; two-chart comparison 4/4; RGBA; URL == data: (same prompt, same reply); 30 MB / 20k x 20k / dead URL / 9 images / `file:` are 400s and the next request is served; same image conversation twice: 2nd cached 448 of 482, tower not rerun, reply identical; turns 2-3 of a receipt conversation cached 320 | W14 | pass (every answer as W14) |
| MMLU-200, refusals | **88.0%** (176/200), 0/10 | >= 87% | pass |
| 4 x ~250k stress, MemAvailable min head / worker | **8.27 / 8.28 GiB**, filled in 775 s, longest decode gap 3.85 s | >= 8 (W12 FIN 9.74 / 8.32; W14 V1 9.43 / 9.82) | pass, 0.27 GiB of margin |
| needle 314,305 tokens alone after the stress + MMLU | found cold (1,368 tok/s, decode 68.5) and resumed (314,304 cached in 0.13 s, decode 77.3); an image right after it served (`7`) | found | pass |
| OOM kills / engine errors | 0 / 0 (dmesg counts unchanged: 0 / 1, the worker node's line is W12's NVRM message); /health 440 requests, 0 errors | 0 | pass |

**C4 with thinking on.** `GLM53_TF_EMIT_FIRST` hands a piece's first token to its caller at once (thinking off shows it:
each first token leaves as its own piece ends). With thinking on, the first generated token streams no text (a
`max_tokens: 1` request at effort low or high returns neither content nor reasoning: a template / think-tag token), so
the first visible delta is the first decode round's, and that round still waits for the round's other pieces. RigMark's C4 (reasoning low) therefore sees no change from 0540 §2; getting it there needs
multi-slot prefill (REPLAY-TTFT §2) or the first decode round of a finished piece run before the next piece.

### 3. Memory

- **Stress.** 8.27 / 8.28 GiB: head is ~1.5 GiB lower than W12 FIN (the tower's 1.13 GB, its transients and the 64 +
  64 MB caches, warm from the image checks); worker as W12 (8.32). Both nodes now sit at the same floor.
- **The needle dip, W12's sequence re-run** (ab.sh, ab.py, N1, stress, MMLU, exact, needle; the image checks before
  them): MemAvailable minimum over the gates **6.39 / 6.42 GiB** (W12 FIN 8.27 / 6.58; W14 V1, without MMLU / N1 before,
  8.34 / 9.07), 17 / 16 two-second samples under 8 GiB, all inside the needle's cold prefill, lowest 4 s before it
  ended; no OOM; back to 11.6 / 11.6 when it ended. Phase minima: image checks 13.4, ab.sh 11.4, N1 10.8, stress 8.27,
  MMLU 9.3, needle 6.4 (`mem-B7.log`, `mem-marks-B7.txt`).
- **What it is** (`meminfo-needle-B7b.log`: the needle alone on the fresh B7b, 298,388 tokens, with Cached / Dirty /
  AnonPages / Shmem sampled): MemAvailable 12.3 -> **9.3 GiB** on both nodes, all of it in **MemFree** (page cache,
  dirty pages, anon and shmem flat), flat for the first ~60% of the prefill and falling over its last ~90 s, then
  14.1 GiB when the request ends. So it is device memory the engine takes at the end of a lone ~300k prefill (context-
  length-sized working buffers of the last chunks) and releases after; it is not the session store, the NVMe tier's
  page cache or a leak. After the stress + MMLU the needle starts ~1.4 GiB lower (10.9) and the drop is larger at 314k
  (4.5 GiB), which gives the 6.4 floor. With the tower, head now dips as far as worker did in W12. The stress gate
  (the documented worst case for four slots) passes; a lone prompt near 1M tokens would dip further than 314k's and was
  not measured.
- **Page cache can serialize concurrent requests (found on b5 before the window).** While `docker save` of b7 was
  streaming 36 GB from the head node, the C4 control on production ran its rounds 2-3 one request at a time (queue 6.3 / 12.5 /
  18.6 s: `c4-b5ctl.log`): head had MemFree 2.1 GiB (16.8 GiB of page cache), and the batcher admits a request beside
  others only while free device memory (`torch.cuda.mem_get_info`) less the store's headroom is >= `GLM53_TF_BATCH_ADMIT_GB`
  (2); on GB10 that free figure evidently tracks MemFree, not MemAvailable (reclaimable page cache does not count). After `drop_caches` the same run was normal (`c4-b5ctl2.log`). Large
  file operations on the head node (image saves, copies, possibly a long-filled NVMe session tier) silently cost concurrency;
  not changed here.

### 4. Adopted

`config/prod.env`: `IMAGE=glm53-tensorfold:b7`, `GLM53_TF_VISION=1`, `GLM53_TF_VISION_CACHE_MB=64`,
`GLM53_TF_VISION_PREP_MB=64`, `GLM53_TF_REASONING_FIELDS=reasoning` (0540's `GLM53_TF_SNAPSHOT_BEFORE_END` /
`GLM53_TF_EMIT_FIRST` default on), header with these gates and the revert (previous file `results/W15/prod.env.before-W15`).
Production restarted on it at 21:17: ready in 32 s, canary ok (decode 81.8 tok/s), local and https `/v1/models` ok,
`17*23` -> `391`, **an image request over https** (`data:` receipt PNG -> `22.50`, 285 image rows), a thinking reply
carries `reasoning` only (no `reasoning_content`), live exact 10/10, watchdog timer active (its first tick exited 0),
lease refresher gone, lease deleted. Note for clients: anything that reads only `reasoning_content` no longer sees the
thinking (the vLLM kit never sent it either). `GLM53_TF_VISION_FETCH` stays 1 (http(s) image URLs are fetched from
the head node's network; the endpoint is private-network only).

### 5. RigMark on the new production (two runs, prod serving, no restart; not published)

`scripts/rigmark/run.sh tensorfold` twice against b7 production, RigMark pinned at `c5a0db01b054` (clean), standard
suite, `reasoning_effort` low, new comparison IDs `2026-09-glm53-exl3-2xspark-tensorfold-w15-b7-v1` (run 1, 21:19-21:29)
and `-v2` (run 2, 21:37-21:47), so no earlier sweep's prompt could be in the NVMe session tier. Files:
`results/rigmark/tensorfold-20260929-w15/` and `tensorfold-20260929-w15-run2/` (receipt + sha256, card, run.log,
metadata, preflight, `requests.jsonl` = the request log of the run; the directories were renamed after the run, so
`command.txt` still names `tensorfold-20260929-141934` / `-143739`), `results/W15/rigmark-w13-w15.md` (table below).
Both runs valid: 15/15 basic output gates, preflight gaps none (reasoning is no longer sent twice), every prefill row's
`prompt_tokens` == depth. Run 1 overlapped three 1-token probe requests of ours (21:21:12, < 0.4 s, during code run 4).

| metric (median) | W13 b5 | W15 b7 run 1 | W15 b7 run 2 | vLLM TP2 k=7 (Alex) |
|---|---:|---:|---:|---:|
| code / prose / structured decode tok/s | 68.6 / 43.2 / 88.2 | 67.5 / 44.0 / 89.0 | 67.2 / 42.7 / 88.7 | 44.0 / 18.9 / 64.9 |
| code / prose / structured TTFT s | 0.50 / 0.40 / 0.46 | 0.51 / 0.40 / 0.46 | 0.49 / 0.44 / 0.48 | 0.60 / 0.49 / 0.47 |
| prose reasoning characters | 120 raw (60 sent twice) | **60** | 0 (empty thinking at low effort) | 60 |
| cold prefill 8K / 32K / 64K tok/s | 1,598 / 1,641 / 1,621 | 1,559 / 1,635 / 1,619 | 1,564 / 1,638 / 1,619 | 1,813 / 1,908 / 1,922 |
| immediate replay 8K / 32K / 64K tok/s | 1,597 / 3,319 / 6,304 | **38,881 / 146,067 / 257,263** | **37,117 / 142,316 / 250,472** | 1,812 / 11,046 / 11,364 |
| replay TTFT 8K / 32K / 64K s | 5.13 / 9.87 / 10.4 | 0.21 / 0.22 / 0.25 | 0.22 / 0.23 / 0.26 | 4.52 / 2.97 / 5.77 |
| C1 / C2 / C4 aggregate tok/s | 54.3 / 65.5 / 82.2 | 54.8 / 67.0 / 81.8 | 54.3 / 67.3 / 82.8 | 31.6 / 42.0 / 66.1 |
| C1 / C2 / C4 per-stream TTFT s | 0.48 / 0.93 / 1.91 | 0.49 / 0.95 / 1.63 | 0.48 / 0.76 / 1.91 | 0.60 / 0.68 / 0.81 |

Against W13: replay 24x / 44x / 41x faster (0.2-0.26 s at every depth), reproduced within 5% by run 2; cold 8K -2%
(0540's extra chunk on a grid-aligned prompt; 32K / 64K equal); decode and aggregates within run-to-run noise (code
-2% in both runs, prose -1..+2%, structured +1%); C4 per-stream TTFT 1.63 / 1.91 s (W13 1.91): unchanged within noise,
as §2 explains (thinking on). Reasoning characters are now single-counted (60 = vLLM's 60).

### 6. Replay verification (is ~146k tok/s at 32K real?)

**What the number is.** RigMark's "immediate replay tok/s" is `prompt_tokens / TTFT` of an identical resend. The
request log of both runs (`requests.jsonl`, 9 replays each) has every replay at **`cached` = n - 64** (8,128 / 32,704 /
65,472), `cache_src` `slot` (the slot's own prompt snapshot, still on the GPU), **1 piece, `prefill_s` 0.17-0.20 s**,
server `first_s` 0.18-0.23 s; RigMark's client TTFT adds 0.02-0.04 s (0.21-0.27 s). So 32,768 / 0.224 s = 146k tok/s is
an effective rate: 32,704 tokens are not recomputed, the last 64 are (a real 64-row prefill chunk through all layers,
~0.17 s, about what a cold 54-token prompt costs: 0.285 s in W13), then the first token is sampled. The cold runs of the
same prompts took 5.2 / 20.0 / 40.4 s with `cached` 0 and 2 / 8 / 16 pieces.

**Same output as computing it.** RigMark's receipts: each replay's 8-token output sha == its cold run's, 9/9 in both
runs. `verify.py` (`verify-prod.log` / `.json`, production, API only): prompts of exactly 8,192 / 32,768 / 65,536 token
ids this server had never seen (the log: `cached` 0, `cache_src` none: computed from scratch), 64 greedy tokens with
`ignore_eos`; the replay and a second replay (`cached` n - 64, TTFT 0.31-0.34 s) returned the **byte-identical 64
tokens** at all three depths. (A restart to compare against another server process was not done: it would cost prod
downtime and add nothing a never-seen prompt does not; resume == fresh is also covered by the GPU tests of §1, knob on
and off, and by N1 / the W9 transcripts.)

**Negative controls** (each variant sent once after its base; `cached` must never exceed the prefix the variant shares
with anything stored):

| variant (one token changed at) | 8,192 | 32,768 | 65,536 | expected |
| --- | --- | --- | --- | --- |
| start (position 3) | cached 0, 5.43 s | 0, 20.9 s | 0, 42.2 s | 0 (cold) |
| middle (0.52 n: 4,259 / 17,039 / 34,078) | 0, 5.71 s | 16,384 (ram), 11.0 s | 32,768 (ram), 21.9 s | the last session mark before the change (every 16,384), rest re-prefilled |
| near the end (n - 10) | 8,128 (slot), 0.31 s | 32,704 (ram), 0.33 s | 65,472 (ram), 0.36 s | n - 64: the snapshot lies before the change |
| end - 100 (n - 100) | 4,224 (ram), 2.81 s | 17,024 (ram), 10.4 s | 49,152 (disk), 11.1 s | below the change: n - 64 contains it |
| a different prompt, same length | 0, 5.44 s | 0, 21.0 s | 0, 43.3 s | 0 (cold) |

Every `cached` is at or below the changed position. The 4,224 and 17,024 are the fork marks the store placed where the
"middle" variants diverged from the base (4,259 / 17,039 rounded down to the 64-grid; `GLM53_TF_SESSION_FORK_MIN`),
valid prefixes of the n - 100 variants too; 49,152 is a session mark read back from the NVMe tier. Each variant's output
differs from the base's except the 64K "middle" one (sha equal): one word changed 31k tokens before the end of a
numbered-line document does not change the next 64 greedy tokens; its log shows 32,768 cached and 32,768 re-prefilled
(21.8 s), so it was recomputed, not served from the base.

**Why vLLM replays at ~11k tok/s** (Alex's receipts, `vLLM 0.1.dev20051`, and our vLLM kit's copy of the same machinery, read
only):

- The receipts' replay TTFTs (TP2 k=7): 4.52 / 2.97 / 5.77 s at 8K / 32K / 64K against cold 4.52 / 17.2 / 34.1 s. At the
  cold rate that is a recomputed tail of ~8,200 / ~5,700 / ~11,100 tokens: the whole prompt at 8K, and a tail that
  doubles from 32K to 64K. The adaptive receipt is the same (8,208 / 5,485 / 10,892); a TP4 receipt recomputes ~3-3.7k
  at every depth (40.9k tok/s at 64K).
- vLLM's prefix cache for this hybrid model can only restore the KDA (linear-attention) recurrent state at page-aligned
  checkpoints: with prefix caching on it forces `mamba_cache_mode = "align"` (`model_executor/models/config.py` ~621-643),
  the state is saved only when a scheduler step ends on a page boundary (our kit sets `MAX_NUM_BATCHED_TOKENS=3584`
  for exactly that, `.env`: "prefix caching NEEDS chunk ends on page boundaries"), a hit is capped at `num_tokens - 1`
  and then rounded down to a whole block (`v1/core/kv_cache_manager.py` ~258), the DSA indexer's scratch cache never
  prefix-caches (so no sub-page hits), and with a DFlash drafter every cache group drops its last matched page. The
  replay therefore restarts from an aligned checkpoint several thousand tokens before the end and re-prefills that
  tail; at 8K the only checkpoint falls in the dropped page, so nothing is reused (replay == cold). Consistent with
  those rules, not checked against Alex's exact source (his image is not on our Sparks): a 3,584-token page and
  7,168-token checkpoints give exactly 8,192 / 4,096 / 8,192 recomputed tokens plus ~0.17 s, which fits all six TP2
  numbers.
- TensorFold keeps a snapshot of the whole state (latent KV + KDA state + MTP / DFlash2 context) at n - 64 for every
  prompt (0540) and resumes there, so an identical resend recomputes 64 tokens at any depth. Our own (unpublished)
  build of the vLLM kit patched the same limits (64-token hits, drafter-group page drop, an n - 1 tail floor) and
  recorded 0.26 s exact replays, so the gap is vLLM's stock hybrid caching, not a TensorFold shortcut.

## W16: THEORY-2 prototype session on the prod stack (image b8) + drafter comparison (2026-09-29, one window 22:10-23:55, prod down 1 h 45 min) — nothing adopted, prod stays on b7

Files: `results/W16/` (`session.log`, not in the public repo, first; `window.sh` = `results/THEORY2-SESSION/run.sh all` with Part B in its
`AFTER_LOADS` hook; `summary.txt` = run.sh's own summary against load C, **`summary-C2ctl.txt` the valid one, against
C2**; `probes-head/`, `probes-worker/`; `rocetrace-K.txt`, `lines-CT-r0.txt`, `lines-VS-r0.txt`; `DS/` Part B with
`DS/dscmp.txt`; `drafters.sh`, `dscmp.py`, `w16req.py` = W11's driver with a `POLICIES` env). The THEORY2-SESSION
scripts (prepared as W13 on b5, never run as such: the W13 / W14 labels went to RigMark and vision) were relabelled
W13 -> W16 and b6 -> b8, the control switched to prod as it runs (`config/prod.env`, image b7, vision on), GR made
skippable (`W16_SKIP_GR`, default on) and `probes/littles.cu` fixed (below).

**Image `glm53-tensorfold:b8`** = b7's list (`results/W15/build-patches.txt`) + 0510 / 0520 / 0530
(`results/W16/build-patches.txt`); all five of 0500-0540 apply in order with offsets only. Built on the head node, shipped with
`docker save | docker load`, tagged `b8` on both nodes. The prepared weight folders are reused (0510-0530 touch no
`WEIGHT_CODE` module). Every b8 load served the **same bits** as b7: exact 10/10, batchexact 4/4, W9 transcripts
identical, reply sha 8794a3463259cc2f, and all 10 glmbench cell hashes == the control.

Schedule: at 22:50 the wrap-up was moved to ~00:30 (0550 / 0560 and the averaged suite need the GPUs).
GR (nsys) was skipped, and the stock-base (abliteration-gap) arm was dropped.

### 1. Probes (no server; clocks locked at 2,250 MHz; head and worker in parallel, 22:10-22:14)

| item | probe | result | gate |
| --- | --- | --- | --- |
| CPU suites (0510 / 0530 / hc_fused, in b8) | pytest | 169 passed | |
| 0510 GPU (split / probe / lone: same bits) | `test_batch_graphs_lone_patches.py` | 19 passed | |
| 0520 GPU bits vs Triton | `test_hc_fused_patches.py` | 76 passed | |
| 2 verify graph launch | `glprobe.py`, 1,650-node graph | first-node delay **4 us** uncaptured (host 4 us); under nsys `graph` 193 us, `node` 188 us; splitting into 2 / 4 / 8 pieces costs 26 / 90 / 200 us end-to-end | **FAIL** (>= 400 us to build the split). THEORY-2's 1.1 ms exposure was nsys inflation |
| 3 fused hc boundary | `bench_hc_fused.py`, cold, 50 MB predecessor | fused / Triton 0.58x at R = 1 (pdl), 0.92-0.98x at R = 2-4, 1.1-1.4x at R = 6-16; geomean 1.02x; same bits | **FAIL** (<= 50% at rows 1-16), so load H was skipped |
| 7 E2 small dense shapes cold | `bench_decode_cold.py --mode both` | rotate / flush speed-up on < 8 MB shapes: KDA f_b/g_b 1.70 / 1.38x, shared down 1.47 / 1.18x, index q_b 1.35 / 1.26x, DSA kv_b 1.28 / 1.21x, index k 1.25 / 1.14x; >= 14 MB shapes 0.82-1.05x | **PASS**: `GLM53_TF_DEC_QMM_MAX_MB` candidate **3.5** (every shape at or below it passes cold) |
| 8 memory pipeline | `littles.cu` | first run: `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, optin)` **invalid argument** (`bulk_kernel` has 128 B static smem, so opt-in + static > the limit). Fixed (the limit minus `sharedSizeBytes`, and the bulk sweep's fit check + 128 B) and re-run `--quick` at 23:55: `ld.global.nc` tile **232.7 GB/s at 4 KB/SM in flight** (D1 C2, 8 warps/SM, 54 regs); cp.async 6 KB/SM (234); bulk 8-16 KB/SM (234-237); idle random chase 1,158 ns/step, x2.7 beside 1 streaming CTA/SM | **PASS** (>= 230 GB/s, <= 4 KB/SM, <= 112 regs) |

### 2. Server loads (THEORY2 "full set" each: exact, batchexact, transcripts, ab.py 24.5k / 98k, glmbench tf,kit,edit x3, 4 streams x6, lone slots)

**Control C was not a control.** It started at 22:14, right after b8's 36 GB `docker load` on the worker node; rank 1 had
~2.4 GB less free memory after its engine step (MemFree 13.1 vs 15.5 GiB in L), the batcher's fit check
(`free - per - store_budget >= BATCH_RESERVE_GB`, both ranks) failed, and C served with **1 batch slot** (4 streams
60.4 tok/s, every round `alone`). C's bits and 1-stream numbers are valid; its 4-stream / slot numbers are not, and
run.sh's own `summary.txt` (+34% at 4 streams for every load) is wrong for that reason. **C2** (23:04: the same
prod config, page cache dropped on both nodes first, 4 slots) ran as the first drafter arm (inco-7d7) with the same
full set; `summary-C2ctl.txt` uses C2 alone.

| load (vs C2) | knobs | exact / bexact / sha / hashes | 1 stream geo | code-like | prose | sampled | 4 streams (mean of 6) | lone slots | gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C2 | prod (b7) | 10/10, 4/4, ok, 10/10 | 0 | 0 | 0 | 0 | 80.55 tok/s | 0 | |
| L | `BATCH_GRAPHS=lone` | same | +0.2% | +0.6% | +0.4% | -0.9% | 80.87 (**+0.4%**) | +0.7% | item 1: 4s >= +0.7% **FAIL** |
| CT | trace + `GRAPH_PROBE=100` (short set) | hashes 7/7 == C | (x1) | | | | 81.47 (x3) | | skew baseline |
| K | `-lgc 2250,2250` + `CPU_PIN=http` + trace | same | -0.2% | -0.2% | +0.1% | -0.7% | 81.12 (+0.7%) | +0.7% | item 6: 4s +0.7% at the bar; skew 42.6 -> 29.1 us an exchange (-32%) **PASS on paper**, see below |
| VS | `VERIFY_SPLIT=4` + `GRAPH_PROBE=100` | same | -0.5% | -0.3% | -0.1% | -1.4% | 81.02 (+0.6%) | -0.4% | item 2: 1s >= +0.4% **FAIL** |
| H | `HC_CUDA=1` | skipped: item 3's microbench gate failed | | | | | | | |
| GR | `GPU_ROUND=resident` + nsys | skipped (schedule) | | | | | | | next window |

- **Every b8 load sits within +0.4 to +1.1% of C2 at 4 streams, including CT, which has no speed knob at all** (b8 with
  only the trace on, 81.47 over 3 reps). That is the noise band; nothing here is a measured gain.
- **K is not adoptable as tested.** Its 4-stream +0.7% equals CT's noise; its skew improvement compares a
  clock-locked load (45,056 paired exchanges) against an unlocked control with only 8,192 (two dumps from the short
  CT set), so clocks and pinning are confounded. `threads-K-r0.txt`, taken at idle, shows every `tf-serve` thread
  still allowed on 0-19 (the HTTP threads are created per request). A clean A/B needs `CPU_PIN=http` alone on unlocked
  clocks against a traced control of the same length.
- **The real verify graph launch is cheap** (`lines-CT-r0.txt`, `('main', R, 1)`): replay host p50 11-14 us, first node
  +12-15 us (p90 <= 24). In 4 pieces (`lines-VS-r0.txt`): host 26-45 us, first piece +7-18 us. The split only adds
  host time, as the synthetic probe said.

**Adoption:** none. L and VS fail their bars, H did not reach a load, and K's pass is noise plus a confound. None of
them had time for the full gates (MMLU-200, 4 x 250k stress, needle) after the schedule change, so the rule (bar AND
full gates) could not be met anyway. **Next build:** item 7's size knob (`GLM53_TF_DEC_QMM_MAX_MB=3.5`: 0440's dense
kernel only for shapes <= 3.5 MB) and item 8's kernel design (`ld.global.nc` at 4 KB/SM in flight is enough for
232 GB/s), both on their probe gates.

### 3. Drafter comparison (DRAFTER-SEARCH §6; prod config b7, one `DRAFTER=` start each, 23:03-23:55)

The arms: incoai `7d74cdd` (current prod, baseline; this start also ran C2's full set), modal-labs
`GLM-5.3-Flash-DFlash@dae6d31`, and incoai `bf582e4e`. They were downloaded to both nodes' HF caches before the window.
The modal-labs `LICENSE` is **MIT** ("Copyright (c) 2026 Z.AI Co., Ltd"); its card says `license: mit`.

Hygiene: every arm's reply sha equals the baseline's on every request (acceptfix 28/28 f7 rows, accept 30/30, glmbench
10/10 cells) and equals **W11's `acceptfix-f7` / `-4` rows (56/56)**. The MTP control rows reproduce W11's table
exactly (prose a_1..4 0.725 / 0.600 / 0.507 / 0.437, tok/rnd 2.475; code 3.463; agent 3.992).

Untruncated DFlash2 (`tf_policy "f7"`, 7-draft block every round, thinking off, greedy + sampled):

| class | arm | rounds | p_cum pos 1..4 | tokens a round | vs 7d74cdd | §5 bar |
| --- | --- | ---: | --- | ---: | --- | --- |
| prose | incoai 7d74cdd | 1,427 | 0.681 / 0.430 / 0.255 / 0.158 | 2.689 | | |
| prose | modal-labs | 1,554 | 0.653 / 0.387 / 0.213 / 0.114 | 2.468 | -0.028 / -0.043 / -0.042 / -0.044, **-8.2%** | FAIL (>= -0.01 each) |
| prose | incoai bf582e4e | 1,419 | 0.689 / 0.438 / 0.250 / 0.153 | 2.705 | +0.008 / +0.008 / -0.005 / -0.005, +0.6% | pass |
| code | incoai 7d74cdd | 700 | 0.874 / 0.707 / 0.560 / 0.437 | 4.387 | | |
| code | modal-labs | 765 | 0.855 / 0.681 / 0.502 / 0.379 | 4.014 | **-8.5%** | FAIL (>= -3%) |
| code | incoai bf582e4e | 681 | 0.888 / 0.708 / 0.578 / 0.462 | 4.508 | +2.8% | pass |
| agent | incoai 7d74cdd | 1,095 | 0.897 / 0.810 / 0.737 / 0.684 | 5.762 | | |
| agent | modal-labs | 1,327 | 0.888 / 0.770 / 0.656 / 0.540 | 4.754 | **-17.5%** | FAIL |
| agent | incoai bf582e4e | 1,101 | 0.899 / 0.810 / 0.741 / 0.672 | 5.730 | -0.6% | pass |

| | incoai 7d74cdd | modal-labs | incoai bf582e4e |
| --- | ---: | ---: | ---: |
| draft ms a round (f7 rows) | 4.09 | 4.61 (+12.7%) | 4.04 (-1.3%) |
| production policy, greedy decode tok/s: prose / code / agent | 48.97 / 68.11 / 86.30 | 47.62 / 61.69 / 84.49 (-2.8 / **-9.4** / -2.1%) | 47.71 / 66.54 / 88.76 (-2.6 / -2.3 / +2.8%) |
| sampled: prose / code / agent | 48.03 / 57.91 / 80.39 | 47.99 / 56.76 / 80.49 | 48.13 / 57.96 / 80.41 |
| glmbench tf,kit,edit x3, geo-mean (hashes == base) | 0 (10/10) | **-2.5%** (10/10); kit hashmap -6.9%, structured -6.6%, tf chat -5.3% | +0.9% (10/10); one outlier cell tf chat greedy +14.7%, hashmap -6.3% |
| exact / batchexact | (C2: 10/10, 4/4) | 10/10, 4/4 | not run |

- **modal-labs is not adopted.** It fails every §5 bar. Its acceptance falls with depth (agent position 4: -0.14), it
  drafts 13% slower (6 layers against 5), and it is 2.5% slower on glmbench. It loads and runs exact (the canary's
  TPR stayed healthy), so the tap convention is right; it simply drafts our abliterated target worse. It is not
  worth using as the public kit's default drafter or as the T2 warm start either. DFlash2-G stays the T2 candidate
  (DRAFTER-SEARCH §5).
- **incoai bf582e4e is parity, not a win.** Acceptance is within ±0.01 at positions 1-4 on prose, +2.8% tokens a round
  on code and -0.6% on agent. Its production-policy greedy tok/s is lower on prose and code by 2-3%, so the "not lower
  on any class" bar fails at this sample size, and it had no exact / batchexact run. Prod stays on 7d74cdd.
- **The abliteration-gap arm** (`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, stock base, same EXL3 format as ours) was not
  run: cut by the schedule. It would also need `GLM53_TF_PREPARED_WRITE=0` (a third prepared key a rank would evict
  one of the two kept) and a cold checkpoint load (~164 GB read on each node). `drafters.sh` has the arm (`stock`).

### 4. Production

Restore at 23:55:04 (`run.sh` EXIT path), after dropping the page cache on both nodes. Prod was started from
`config/prod.env` unchanged (image b7, vision on, DRAFTER incoai 7d74cdd), attempt 1: canary ok (3 probes,
tokens/round 5.71, decode 73.0 tok/s), local and https `/v1/models` ok, 17*23 -> 391, **batching 4 requests** (4
slots). Clocks were restored to `-lgc 300,2250`, the watchdog timer is active, the lease refresher was killed and the
lease deleted, and no `w16-*` containers are left.

**Operational finding (prod risk): the 4-slot fit has ~1-2 GB of margin on rank 1.** A start right after a large file
read (here b8's `docker load`) comes up with **1 slot and no error**: 4 concurrent requests are then served one after
another. `MEM_GATE_DROP_CACHES=1` does not help, because the gate compares `MemAvailable` (page cache counts as
available) and drops caches only below 108 GiB. Suggested fix for the next build: drop caches unconditionally before
a start, or have serve.sh check the `batching N requests` line against `GLM53_TF_BATCH` and restart / alert. The
watchdog's health check does not look at slot count.

## W17: 0550 (memory safety) + 0560 (multi-slot prefill) on the GPU, image b9, and final averaged numbers (2026-09-30, one window 00:06-02:59, prod down 2 h 53 min) — 0560 adopted (`GLM53_TF_MULTI_PREFILL=1`), 0550 built in but off; prod on b9

Files: `results/W17/` (`window-start.sh`, `run-window.sh` B9 / C7, `run-M6.sh`, `gates.sh` / `ab.sh` = W15's, `load.sh` and
`restore.sh` now drop the page cache on both nodes before every start and check the slot count (`dropc.sh`,
`slotcheck.sh`), `tests.sh`, `mpf.py` (grouped == alone on the real model), `s900.py` (prepared, not run), `cmpglm.py`,
`diag_multi.py`; per load `*-B9*`, `*-C7*`, `*-M6*`, `meminfo-*.log` (MemAvailable / MemFree / Cached / Dirty / Anon every
2 s), `mem-marks-*.txt`; `tests-head/`, `tests-worker/`, `tests-probe01/`, `tests-probe02/`; `prod.env.before-W17`) and
`results/FINAL-20260930/` (phase 2).

**Image `glm53-tensorfold:b9`** = b7's list + 0550 + 0560 (`results/W17/build-patches.txt`; both apply cleanly with
`git apply`), built on the head node with `--build-arg PATCHES` (serve.sh build passes none = every patch), shipped with
`docker save | docker load`, tagged `b9` on both nodes. Prepared weights reused.

**serve.sh fix (`scripts/serve.sh` in this update).** W16's load C came up with 1 batch slot after a `docker load`.
`serve.sh start` now drops the page cache (`echo 3`, sudo -n) on a node whose MemAvailable - MemFree >= 4 GiB (with
`MEM_GATE_DROP_CACHES=1`, as in prod.env), and after `/v1/models` reads rank 0's `N request slot(s)`: fewer than
`GLM53_TF_BATCH` -> stop, drop the caches on both nodes, restart (at most `SLOT_RETRIES=2`), then exit 3. Every W17
start (B9, C7, M6, the adopt restart) also dropped the caches explicitly first and came up with **4 slots**.

### 1. Tests (prod stopped, both nodes in parallel, every run under `timeout`)

| run | head | worker |
| --- | --- | --- |
| page-cache probe (0550 step P, `bench/pagecache_probe.py`, 20 GiB of a prepared `data.bin` read, then 1 GiB device steps) | **`cache_reclaimed` true**, no `failed_at`: 101 GiB allocated from MemFree 94.5, Cached 20.9 -> 5.7 GiB, 0.04 s a GiB step past MemFree (as below it) | same: 99 GiB, Cached 20.9 -> 6.3 |
| `test_memory_safety_patches.py` (0550) | **25 passed** | **25 passed** |
| `test_multi_prefill_patches.py` (0560) | 83 passed, **4 failed** | 83 passed, 4 failed (same) |
| regressions | lean 53, overlap 149, prefill_pp 15 (+1 skip), replay 32 | batch_sessions 28, fastpf 29, cindep 54, TensorFold `test_glm_engine.py` 12; `test_1m` 42 + 3 failed, `test_session_disk` 30 + 2 failed |

- The 4 `test_gpu_group_equals_alone` failures are in its second-wave assert `cached > 0`: every reply equals the lone
  engine's in both waves; a 149-token next turn is not resumed. `diag_multi.py` (3 waves, per-member stats, knob on and
  off) shows the **same `cached` with the knob off** (0 for the < 256-token members, 256 for the 300-token one): the
  test's expectation, not 0560.
- `test_1m` (3) and `test_session_disk` (2, `GLM53_TF_KV_DTYPE=fp8 is laid out for a 512-wide latent, not 128`) fail
  **identically on b7**; `test_1m` passes 45/45 with `GLM53_TF_SNAPSHOT_BEFORE_END=0` (0540's rule: resume at
  `(n - 1) // 64 * 64`, the W15 class of expectation). Not new.
- Since the probe passed, `GLM53_TF_ADMIT_MEM=available` stays 0550's default.

### 2. Loads (each: page cache dropped, 4 slots checked; the same sequence for all three)

Sequence: PRE = `mpf.py group` (below) + `c4.py` thinking low and off; then W15's `gates.sh` unchanged (ab.sh set, ab.py
again, N1, 4 x ~250k stress, MMLU-200, exact / batchexact again, needle ~314k alone), meminfo every 2 s. This PRE is
~200 short requests, so the whole history is heavier than W15's (which started the gates with a 1-minute image check).

- **B9** = `config/prod.env` + IMAGE b9 + `GLM53_TF_MULTI_PREFILL=1` (0550's defaults on)
- **C7** = production as it was (b7, `config/prod.env` unchanged): the control, added when B9 missed the stress bar
- **M6** = B9 with 0550 off (`GLM53_TF_ADMIT_MEM=free GLM53_TF_SELECT_SCRATCH=off GLM53_TF_ALLOC_TRIM_GB=0`) = 0560 alone

| gate | B9 (0550 + 0560) | M6 (0560 alone) | C7 (b7 control) | bar |
| --- | --- | --- | --- | --- |
| exact / batchexact, before and after the stress + MMLU | 10/10, 4/4; 10/10, 4/4 | same | same | pass |
| W9 transcripts (alone == W9 A, together == alone) | True, 4/4 | True, 4/4 | True, 4/4 | pass |
| reply sha (ab.py, every cell) | 8794a3463259cc2f | same | same | pass |
| glmbench 13 cells: hashes / 1-stream geomean | **13/13 == b7 (W15)**, -0.36% vs W15 B7, +0.03% vs C7 | 13/13 == C7, +0.26% | 13/13 | pass |
| N1 | 6/6 + 6/6, **18/18 shas == W15** | 18/18 | 18/18 | pass |
| **grouped == alone on the real model** (`mpf.py`, 92 requests) | **92/92**, 89 cold sends in a group forward (`multi` 2-4) | **92/92**, 90 grouped | 92/92 (never grouped) | pass |
| C4 per-stream TTFT, RigMark shape, reasoning low (3 rounds, median) | **0.78 s** (max 0.80) | **0.82 s** | 1.64 s (W15 B7b 1.68) | <= 0.8 target; -52% |
| C4 thinking off / C2 low / C2 off | 0.62 / 0.59 / 0.46 s | 0.63 / 0.57 / 0.45 | 0.92 / 0.69 / 0.55 | improved |
| prefill 24.5k / 98k (ab.py) | **1,454 / 1,550** (first, after PRE), 1,608 / 1,597; after the needle 1,577 / 1,594 and **1,512 / 1,541** | 1,603 / 1,612, 1,610 / 1,610 | 1,613 / 1,576, 1,605 / 1,614 | **B9 fails** "not lower" |
| decode 4 streams (mean of 6) / lone slots | 82.1 / 51.6-57.1 | 82.5 / 52.0-57.4 | 81.2 / 51.6-57.0 | within noise |
| MMLU-200, refusals | 88.0%, 0/10 | 88.0%, 0/10 | 88.0%, 0/10 | pass |
| 4 x 250k stress, MemAvailable min head / worker | 7.37 / 6.89 | 7.50 / 6.92 | 7.47 / 7.41 | >= 8: **all three fail** after this history |
| needle ~314k after stress + MMLU: found / min | found (resend cached n - 18 in 0.24 s) / **7.92 / 7.48** (dip 1.7 GiB) | found / 6.04 / 5.46 | found / 6.21 / 5.81 (dip 1.9 / 4.0) | |
| **minimum over all gates** | **7.37 / 6.89** | 6.04 / 5.46 | 6.21 / 5.81 | |
| NVRM `NV_ERR_NO_MEMORY` lines on the worker node (non-fatal, 0 engine errors) | 4 (during the stress minimum) | 0 | 1 (needle minimum) | |
| OOM kills / engine errors / request errors | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | pass |

**Memory attribution.** After the same PRE, MemAvailable was B9 11.9 / 11.5, C7 12.4 / 12.0 (0.5 GiB less with b9,
+0.3 GiB host AnonPages); at the stress start B9 9.4 / 8.9, C7 10.1 / 9.9. During the stress B9 dips 2.0 GiB, C7
2.5-2.7 (0550's scratch). M6 = 0560 alone reaches B9's stress floor (7.50 / 6.92), so the worker node's -0.5 GiB there is
**0560's** (the group forwards' larger transients); the needle floor is 0550's: with it the lone 314k dip is 1.7 GiB
(W15: 4.5), without it 4.0 on the worker node. The absolute 8 GiB stress bar was set on W15's lighter sequence; the control
misses it in this one too.

**0550's prefill.** Only B9 had slow long prefills: the first ab.py after the PRE (1,454 / 1,550: -9.8% / -3.5%, same
piece count) and the second pair after the needle (1,512 / 1,541); C7 and M6 never (1,576-1,614). The difference
between B9 and M6 is 0550's three knobs. Not isolated further (no time): the trim (`empty_cache` at most every 10 s
when > 2 GiB unused) or the scratch's growth before a prefill are the candidates. 0550's memory results are real and
large for long lone prompts (the 1M bound in MEMORY-SAFETY.md), so it stays in the image, off.

### 3. Adopted

- **0560: adopted** (`GLM53_TF_MULTI_PREFILL=1`): same bits everywhere (92/92 on the real model, batchexact,
  transcripts, N1, 13/13 hashes), C4 first tokens -52% with thinking on (the case 0540 could not reach), C2 -18%,
  prefill and decode unchanged. Cost: -0.2 / -0.35 GiB of the needle floor and -0.5 GiB on the worker node in the 4 x 250k stress.
- **0550: not adopted** (fails "prefill not lower" in 2 of 4 B9 prefill pairs), its knobs set off in prod.env so b9
  behaves as b7 + 0560 (0550's unconditional part, dropping its own session-tier writes from the page cache, stays).
  Next: B9 with `GLM53_TF_ALLOC_TRIM_GB=0` (scratch + admission only) through the same sequence, prefill x4.
- Not run (time): the ~900k prompt beside 3 busy slots (`s900.py` ready), 0550's C4-under-a-copy test.

`config/prod.env`: `IMAGE=glm53-tensorfold:b9`, `GLM53_TF_MULTI_PREFILL=1`, `GLM53_TF_ADMIT_MEM=free`,
`GLM53_TF_SELECT_SCRATCH=off`, `GLM53_TF_ALLOC_TRIM_GB=0`, header with the gates and the revert (previous file
`results/W17/prod.env.before-W17`). Production restarted on it at 02:59 (page cache dropped, ready in 34 s, **4 request
slots**, canary ok 74.8 tok/s, local and https `/v1/models`, `17*23` -> `391`), watchdog timer active, lease refresher
gone, lease deleted.

### 4. Final numbers on the adopted config (phase 2: prod serving, 3 rounds, 03:00-03:45)

`results/FINAL-20260930/` (`final.sh`, `summarize.py`, `summary.md` with every round, the vLLM and "this morning"
tables). Each round = RigMark standard suite (new COMPARISON_ID `...-tensorfold-w17-final-rN`), glmbench tf / kit / edit
x3, multiturn concurrent 1 / 4 streams x3, ab.py 24.5k / 98k; then MMLU-200 and exact / batchexact once. All rounds valid
(15/15 RigMark gates each, reply sha 12/12, greedy glmbench hashes identical across rounds).

| metric (mean of 3, min-max) | W17 prod (b9 + 0560) | W15 prod (b7) runs 1 / 2 | vLLM TP2 k=7 (Alex) |
| --- | ---: | ---: | ---: |
| RigMark code / prose / structured decode tok/s | 67.9 / 43.0 / 88.8 | 67.5 / 44.0 / 89.0; 67.2 / 42.7 / 88.7 | 44.0 / 18.9 / 64.9 |
| RigMark cold prefill 8K / 32K / 64K tok/s | 1,560 / 1,634 / 1,620 | 1,559 / 1,635 / 1,619 | 1,813 / 1,908 / 1,922 |
| RigMark replay TTFT 8K / 32K / 64K s | 0.22 / 0.25 / 0.27 | 0.21 / 0.22 / 0.25 | 4.52 / 2.97 / 5.77 |
| RigMark C1 / C2 / C4 aggregate tok/s | 53.1 / **70.1 / 91.0** (C4 89.5-92.1) | 54.8 / 67.0 / 81.8; 54.3 / 67.3 / 82.8 | 31.6 / 42.0 / 66.1 |
| RigMark C1 / C2 / C4 per-stream TTFT s | 0.49 / **0.65 / 0.89** | 0.49 / 0.95 / 1.63; 0.48 / 0.76 / 1.91 | 0.60 / 0.68 / 0.81 |
| glmbench tf chat / tf code / kit structured (T=0) | 48.0 / 83.3 / 105.5 | 47.9 / 83.5 / 105.7 (W15 B7) | |
| concurrent 1 / 4 streams aggregate | 60.3 / 82.8 (82.6-83.0) | - / 80.6 | |
| ab.py prefill 24.5k / 98k | 1,606 (1,600-1,611) / 1,610 (1,607-1,612) | 1,610 / 1,607 | |
| MMLU-200 / refusals; exact; batchexact | 88.0% / 0 of 10; 10/10; 4/4 | 88.0%; 10/10; 4/4 | |

Against "this morning's image" (chat 44.6, code 77.6, structured 100.6, 4 users ~78, prefill 1,607): +7.7%, +7.4%,
+4.9%, +6.2%, -0.1% / +0.2% (most of that is W12-W15's; W17 adds the concurrency: C4 aggregate +10% and C4 first
tokens -45..-53% vs W15 on RigMark).

The three RigMark receipts of this run are in `results/rigmark/tensorfold-20260930-w17-final-r1` / `-r2` / `-r3`
(comparison IDs `2026-09-glm53-exl3-2xspark-tensorfold-w17-final-r1..r3`, RigMark `c5a0db01b054` clean, standard suite,
reasoning low; receipt sha256 `fbaf073b...`, `bbc1af37...`, `4dca115c...`), with the server's request log of each run
(`requests.jsonl`: every replay row `cached` = n - 64 from the slot, 9/9 a run). Mean of the three runs' medians, with
min-max, against Alex Ellis's vLLM TP2 k=7 receipt (from `results/FINAL-20260930/summary.md`):

| RigMark (mean of 3, min-max) | TensorFold (b9, W17 prod) | vLLM TP2 k=7 (Alex) | ratio (mean) |
| --- | ---: | ---: | ---: |
| code decode tok/s | 67.9 (67.4-68.6) | 44.0 | 1.54x |
| prose decode tok/s | 43.0 (42.2-43.5) | 18.9 | 2.28x |
| structured decode tok/s | 88.8 (88.6-89.0) | 64.9 | 1.37x |
| cold prefill 8K / 32K / 64K tok/s | 1,560 (1,556-1,562) / 1,634 (1,631-1,637) / 1,620 (1,618-1,621) | 1,813 / 1,908 / 1,922 | 0.86x / 0.86x / 0.84x |
| immediate replay 8K / 32K / 64K tok/s | 36,474 / 132,814 / 243,980 | 1,812 / 11,046 / 11,364 | 20x / 12x / 21x |
| replay TTFT 8K / 32K / 64K s | 0.22 (0.22-0.23) / 0.25 (0.24-0.25) / 0.27 (0.27-0.27) | 4.52 / 2.97 / 5.77 | 20x / 12x / 21x |
| C1 / C2 / C4 aggregate tok/s | 53.1 (52.1-53.9) / 70.1 (68.8-70.9) / 91.0 (89.5-92.1) | 31.6 / 42.0 / 66.1 | 1.68x / 1.67x / 1.38x |
| C1 / C2 / C4 per-stream TTFT s | 0.49 / 0.65 / 0.89 (0.88-0.89) | 0.60 / 0.68 / 0.81 | 1.22x / 1.04x / 0.91x |

Not a strict RigMark comparison (different weights: abliterated EXL3 4-bit here vs LibertAIDAI NVFP4; drafter, context
limit, protocol, day and machines differ; `results/rigmark/README.md`). Where we are behind: cold prefill (0.84-0.86x)
and C4 per-stream TTFT (0.89 vs 0.81 s). Replay is identical-prompt caching (n - 64 resumed), not a faster prefill.

### 5. Notes

- Harness slip: `run-window.sh` was edited while bash was executing it (to add the B9b steps), so B9's run continued
  into the new B9b lines after its gates (an extra ab.py pair: the post-needle 1,577 / 1,594 and 1,512 / 1,541) and then
  stopped on a syntax error. B9's gates had all completed; C7 and M6 ran from frozen copies (`run-window-exec.sh`,
  `run-M6.sh`). B9b (restart + mpf alone + replay) was not needed: B9's own lone sends were already cold (`cached` 0)
  and the needle / ab.py resends showed the n - 64 rule (`cached` = (n - 1) // 64 * 64).
- final.sh's RigMark rename missed (the UTC stamp sorts before `-w15`); the three dirs were renamed by stamp afterwards.
- Memory risk that remains in prod (0550 off): a lone prompt far beyond 314k still grows the allocator's key blocks
  quadratically (MEMORY-SAFETY.md: ~16 GiB bound at 1M); M6's 314k needle floor was 6.04 / 5.46 GiB. The ~900k test
  (`s900.py`) was not run on prod for that reason. 0550 fixes it; its prefill regression is the next thing to isolate.

## W18: 0550 slowdown attribution on b9 (no rebuild) + the KINDLING K9 NCCL bandwidth sweep (2026-09-30, daytime; windows 06:44-07:11 (aborted), 07:32-08:22, 08:43-09:35, 09:56-10:46) — nothing adopted (the fixes go into one combined W19 window); recommended 0550 setting below; prod on b9, 4 slots

Files: `results/W18/` (`windows.log`, `loads.log`; harness from W17: `run-window.sh` = window-start + deadman + optional
`prewindow-NAME.sh` + `gates.sh` + `restore.sh`, `chain.sh` = windows with >= 21 min of prod between them; per load
`run-N.out`, `summ.py` table, `memsum-N.txt` (MemAvailable minimum per phase), `dips.py` (dip at the stress and the
needle), `memfast-N-r0/r1.log` (MemFree / MemAvailable every 0.5 s on each node, sampled locally) and
`memsteps-N-*.txt` (MemFree steps >= 1 GiB: allocator `empty_cache`), `meminfo-N.log`, `reqlog-N.jsonl` (the request
log lines of the load: slot, pieces, prefill_s); `ncclbw.py` / `ncclbw.sh` / `ncclbw.jsonl` / `ncclbw-*-r0.log`;
prepared but not run: `run-t2.sh` (b10 + CPU_PIN), `run-multi.sh` + `prewindow-N.sh` + `ncclbw2.sh` +
`prod-2hca.env` / `prod-1hca-pt.env` (the K9 loads)).

**Test port.** The first window (load A, 06:44) served on :8000 like every earlier window, and live client traffic
(three long tool-using sessions, 50-93k-token prompts, 06:52-07:05) used it: glmbench fell to 20-57 tok/s on
some cells, two 4-stream reps to 47 / 55 tok/s, one C4 request queued 19 s. It was aborted at 07:10 (prod restored
07:11, 4 slots), its files are in `results/W18/A-aborted-live-traffic/`, and every later load served on **:8001**
(`PORT=8001` in `load.sh`; `ab-p.py`, `req-p.py`, `transcripts-p.py`, `needle-p.py` = W5 / W7 / W10 / W6's scripts on
:8001; `BASE` for mpf.py / c4.py; `--base` for longexact.py). In a window the https endpoint now refuses instead of
being served by a test config; that is the approved downtime. Earlier night windows had no foreign traffic
(checked for W17 in the request log: only the stress's own requests).

**Page cache / slots.** Every start dropped the page cache on both nodes first (`dropc.sh`, then serve.sh's own
drop), and serve.sh's slot check printed `request slots: 4 (GLM53_TF_BATCH=4)` on every test and prod start; the boot
lines show `4 request slot(s)`.

### 1. 0550 attribution (task 1)

Loads (b9, `config/prod.env` + overrides; each the W17 sequence: PRE = `mpf.py group` + `c4.py` thinking low and off
(the heavy warm-up), ab.sh set (exact, batchexact, W9 transcripts, ab.py, glmbench 13 cells x3, 4 streams x3 twice,
lone slots), ab.py again, N1, 4 x ~250k stress, MMLU-200, exact / batchexact again, needle ~314k alone, then two more
ab.py pairs (W17 B9's post-needle slow pair came there)):

- **A** = prod control (0550 off: `ADMIT_MEM=free SELECT_SCRATCH=off ALLOC_TRIM_GB=0`)
- **T** = 0550 without the trim (`ADMIT_MEM=available SELECT_SCRATCH=grow ALLOC_TRIM_GB=0`)
- **S** = scratch only (`ADMIT_MEM=free SELECT_SCRATCH=grow ALLOC_TRIM_GB=0`)
- **F** = 0550 full on: **not run** (plan change at ~09:05: the remaining items moved into W19, and this window compared
  "trim off vs scratch only"); W17's B9 is the full-on reference.

| gate | A (control) | T (trim off) | S (scratch only) | W17 B9 (full on) | bar |
| --- | --- | --- | --- | --- | --- |
| exact / batchexact, before and after stress + MMLU | 10/10, 4/4; 10/10, 4/4 | same | 10/10, 4/4; 10/10, 4/4 | same | pass |
| W9 transcripts, N1 | True, 4/4; 6/6 + 6/6 | same | True, 4/4; 6/6 + 6/6 | same | pass |
| reply sha (all 8 ab.py cells, cold and warm) | 8794a3463259cc2f | same | 8794a3463259cc2f | same | pass |
| mpf.py grouped == alone (92 requests) | 92/92 | 92/92 | 92/92 | 92/92 | pass |
| **prefill 24.5k** (4 runs, tok/s) | 1,600 / 1,613 / 1,601 / 1,603 | 1,602 / 1,607 / 1,610 / 1,606 | 1,603 / 1,603 / 1,605 / 1,604 | **1,454** / 1,608 / 1,577 / **1,512** | not lower |
| **prefill 98k** | 1,612 / 1,605 / 1,602 / 1,602 | 1,601 / 1,611 / 1,604 / 1,610 | 1,606 / 1,610 / 1,610 / 1,611 | **1,550** / 1,597 / 1,594 / **1,541** | not lower |
| glmbench 1 stream geomean vs A, hashes | 0, 13/13 | -0.18%, 13/13 | -0.17%, 13/13 | (+0.03% vs C7) | noise |
| 4 streams, mean of 6 (paired per rep vs A) | 82.40 | 82.72 (+0.4%) | 82.42 (+0.0%) | 82.1 | noise |
| C4 TTFT median, thinking low / off | 0.81 / 0.62 s | 0.79 / 0.62 s | 0.80 / 0.61 s | 0.78 / 0.62 | |
| MMLU-200, refusals | 88.0%, 0/10 | 88.0%, 0/10 | 88.0%, 0/10 | 88.0% | >= 87% |
| needle ~314k: found, lone prefill | yes, 243.6 s (1,290 tok/s) | yes, 229.3 s (**1,371**) | yes, 298,345 tokens (the needle script built a shorter prompt this time), 216.2 s (**1,380**) | yes, 243.8 s | |
| stress: MemAvailable at start -> min (dip), head / worker | 8.14 / 7.71 -> **6.51 / 6.14** (1.6 / 1.6) | 8.95 / 8.70 -> **7.70 / 7.58** (1.25 / 1.1) | 8.99 / 8.70 -> **7.84 / 7.56** (1.15 / 1.1) | 9.39 / 8.92 -> 7.37 / 6.89 (2.0 / 2.0) | >= 8 |
| needle: start -> min (dip) | 8.54 / 8.35 -> **5.10 / 4.75** (3.4 / 3.6) | 9.23 / 8.66 -> **8.10 / 7.86** (1.1 / 0.8) | 8.95 / 8.76 -> **8.39 / 8.01** (0.6 / 0.75; 298k) | 9.58 / 9.17 -> 7.92 / 7.48 (1.7 / 1.7) | >= 8 |
| minimum over stress + MMLU + needle | 5.10 / 4.75 | **7.70 / 7.58** | **7.84 / 7.56** | 7.37 / 6.89 | >= 8: all fail |
| OOM / NVRM `NV_ERR_NO_MEMORY` lines / engine errors | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 4 / 0 | |

- **The prefill slowdown needs the trim.** A, T and S ran 8 prefills each at 1,600-1,613 tok/s after the same heavy
  history, including the two post-needle pairs; every slow prefill in W17 / W18 (5 of B9's 8) had
  `GLM53_TF_ALLOC_TRIM_GB=2`. With F not run, this is attribution by elimination: the trim is the only 0550 knob B9 had
  that T lacks. The mechanism (MEMORY-SAFETY.md §6): the trim runs before prefill pieces, at most every 10 s, but
  during a long prefill almost every piece start qualifies; `empty_cache` synchronizes the device, returns the pages,
  the next piece cudaMallocs them again, and it is rank-local, so each rank's trim also stalls the other one.
- **The scratch is the memory fix.** The 314k needle's dip falls from 3.4-3.6 GiB (A; W17 M6 3.6 at 298k) to
  0.8-1.1 GiB (T; S 0.6-0.75 at 298k), and the stress dip from 1.6 to 1.1-1.25 GiB (T, S). The lone needle prefill
  is also 6-7% faster with it (T 1,371, S 1,380 against A 1,290 tok/s: without the scratch every new key-block size is
  a fresh cudaMalloc).
- **No 0550 setting reaches ">= 8 GiB through stress + needle" after this history.** T's minimum is 7.70 / 7.58, S's
  7.84 / 7.56 (both in the stress), and part of their margin over A is the boot path, not the knobs: T and S (new knob
  sets) re-measured the calibration
  (`calibration: real (33.9 s)`, engine ready MemAvailable 16.8 / 16.6 GiB) while A took the cached path
  (`calibration: cached`, engine ready **15.6 / 15.0**: the "engine ready" step ran 7.6 s instead of 0.4 s and kept
  ~1.2 GiB more; S: 16.7 / 16.3). Prod restarts take the cached path, so on prod these knobs would give about A's
  start minus T's / S's dips: stress ~6.9 / 6.6, needle ~7.4 / 7.6 GiB. The stress floor is set by four slots at 250k (0560's group transients,
  W17), not by 0550. b7 (W17 C7, cached) did not show the 1.2 GiB: a boot-time `empty_cache` after the warm-up is
  the cheapest next gain (MEMORY-SAFETY.md §6 item 4).
- **Admission (`available` vs `free`):** no measurable difference between T (`available`) and S (`free`) in these loads: prefill, decode, C4 TTFT and the memory dips are the same within noise (stress 1.25 / 1.1 vs 1.15 / 1.1 GiB). The rule only changes when a request is admitted while page cache is large (the W15 copy case, MEMORY-SAFETY.md §5 step 6), which no W18 load had. So the memory and speed result belongs to the scratch; `available` is a separate decision that needs the C4-under-a-copy test.

**Recommended 0550 setting (not adopted: plan change; for W19):** `GLM53_TF_SELECT_SCRATCH=grow`,
`GLM53_TF_ALLOC_TRIM_GB=0`, `GLM53_TF_ADMIT_MEM=free` (as in S; `available` after the C4-under-a-copy test passes). Follow-up patch (0550 v2, MEMORY-SAFETY.md §6): trims only
as planned rounds on both ranks, at idle (no slot busy for 2 s) or under memory pressure while busy (at most every
60 s), never inside a healthy lone prefill; plus one `empty_cache` after the boot warm-up.

### 2. CPU_PIN=http retest (task 2): not run

0530 is **not** in b9 (b9 = b7's list + 0550 + 0560; `results/W17/build-patches.txt`), so it needs b10 = b9's list + 0530.
Checked offline: 0530 applies cleanly on b9's tree (`git apply` of every b9 patch + 0530 in order, `results/W18`
harness), and its CPU test passes on that tree (`tests/test_http_pin.py`: 3 passed, 1 skipped). `run-t2.sh` is ready
(build b10 + ship, then CT = b10 + RoCE trace + dump, P = CT + `CPU_PIN=http` with thread placement taken during a
4-stream burst, CT2 = a second control; each the ab.sh set, 4 streams x6, glmbench 1 stream; `rocetrace.py --vs`). Moved
into W19 by the plan change. Note for its bar: the 4-stream mean of 6 has sd ~3 tok/s (3.5%) a load; the reps are the
same prompts in the same order, so compare them paired (A vs T per rep: -1.6..+2.0%, mean +0.4%).

### 3. KINDLING K9: NCCL over both CX7 functions (extra item; part (a) done, loads moved to W19)

Part (a) ran in window T before its load (model stopped, 08:43-08:44; `ncclbw.sh`: one container per node from the
b9 image, torch `all_gather_into_tensor` of bf16, 10 warm-up + 50 timed ops a size, CUDA events; sizes are the
all-gather output). `NCCL_DEBUG=INFO` confirms `NET/IB : Using [0]rocep1s0f1:1/RoCE [1]roceP2p1s0f1:1/RoCE` for the
two-function runs.

| setting | channels | 1 MiB | 2 MiB | **4 MiB** (GB/s out) | 8 MiB | 16 MiB (GB/s) | 32 MiB | all-reduce 4 / 16 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one function (prod), default | 64 | 144 us | 262 | **252 (16.7)** | 392 | 799 (21.0) | 1,532 | 425 / 1,448 |
| one, repeated at the end | 64 | 130 | 216 | 248 (16.9) | 395 | 824 (20.4) | 1,567 | 413 / 1,481 |
| one, `NCCL_ALGO=Ring` | 64 | 163 | 294 | 275 (15.3) | 421 | 817 (20.5) | 1,770 | 517 / 1,443 |
| both functions, default | 64 | 173 | 344 | 267 (15.7) | 321 | **459 (36.5)** | 911 | 464 / 831 |
| both, Ring | 64 | 170 | 364 | 273 (15.3) | 338 | 475 (35.3) | 907 | 494 / 828 |
| both, `NCCL_PROTO=Simple` | 64 | 168 | 348 | 252 (16.7) | 330 | 480 (35.0) | 909 | 479 / 829 |
| both, `NCCL_MIN/MAX_NCHANNELS=2` | 2 | 63 | 87 | 156 (26.8) | 270 | 534 (31.4) | 1,085 | 257 / 872 |
| **both, 4 channels** | 4 | 68 | 125 | **149 (28.1)** | 263 | 507 (33.1) | 1,002 | 241 / 847 |
| both, 8 channels | 8 | 60 | 82 | 157 (26.7) | 266 | 479 (35.0) | 1,026 | 222 / 832 |
| both, Simple + 4 channels | 4 | 60 | 89 | **144 (29.1)** | 261 | 517 (32.5) | 995 | 242 / 842 |

- NCCL defaults to **64 channels** for this 2-rank ring. At the prefill's ~4 MiB, the second function alone gains
  nothing (267 vs 252 us); the gain at 4 MiB comes with few channels: both functions + 2-8 channels take 144-157 us,
  **-40%** against prod's 252 us (and -75% at 1-2 MiB). From 16 MiB up the second function gives +75% (36.5 vs 21 GB/s).
- `NCCL_ALGO=Tree` has no all-gather in NCCL ("No algorithm/protocol available for function AllGather ... NCCL_ALGO
  was set to Tree"), so the audit's Ring-vs-Tree health check needs all-reduce; `ncclbw2.sh` (prepared) has it
  (`AR_ONLY=1`), plus 2 / 4 / 8 channels on **one** function, to tell the channel count from the second function.
- Candidate for W19's load: `prod-2hca.env` (`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, `NCCL_PASSTHROUGH=1`) +
  `NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4`; control of the channel effect `prod-1hca-pt.env` + the same channels.
  Fewer NCCL CTAs beside the prefill kernels may matter as much as the wire (KINDLING K9 notes `_hc_post` ran 3x slower
  beside the all-gathers in W7). KINDLING-AUDIT's estimate for the prefill is +1-3%; the adoption bar is +1.5%.

### 4. Production

Prod = `config/prod.env` unchanged (image b9, `GLM53_TF_MULTI_PREFILL=1`, 0550's knobs off). Restored after every
window by `restore.sh` (page cache dropped, `request slots: 4`, canary ok, local and https `/v1/models`, `17*23` ->
`391`, watchdog timer active, lease refresher gone, lease deleted): 07:11:32, 08:22:23, 09:34:52, 10:46:27 (after S: `request slots: 4`, canary ok, https `/v1/models` ok, `17*23` -> `391`, watchdog active, no lease, no helper processes left).
Windows: 27 min (aborted), 49, 51, 50 min; prod up 21 min between them.

## W19: combined traced window (image b10 = b9 + 0530 / 0570 / 0580 / 0590 / 0600 / 0610), 2026-09-30 (windows 11:13-12:23 and 12:35-13:49, prod down 69 + 74 min, up 12 min between) — adopted: 0580, NCCL on both CX7 functions with 4 channels, 0550's scratch, 0530 CPU_PIN=http, 0610 grammar (+ 0600, on by default); not adopted: 0570, 0590; prod on b10

Files: `results/W19/` (`windows.log`; harness from W18: `win.sh` / `win2.sh` = window-start + deadman + kernel gates or
NCCL sweep + one TRACED start (`load.sh NAME-T TRACE=1`, `cap.sh`) + one PLAIN start (`gates.sh`: W17/W18's full
sequence with a `POST_AB` hook) + restore; `kgates.sh`, `fat2gate.py`, `ncclbw2.sh` / `ncclgate.py`, `ports.py` (0600),
`threads.sh`, `cap.sh` / `nsysctl.sh` / `bin/tensorfold` / `entry.sh` (W11's nsys method), `export.sh`, `w19att.py` /
`attcmp.py` (attribution), `summ.py` / `extra.py`, `run-c2.sh`; outputs per load `*-CONTROL*`, `*-COMBINED*`,
`*-COMBINED2*`, `kg-head/`, `kg-worker/`, `tests-cpu/`, `ncclbw*.log`, `att-*-r0/r1.json|txt`, `attcmp-r0/r1.txt`,
`prod.env.before-W19`). The nsys reports and sqlite exports are on the head node in `/var/tmp/w19/out/` (not in git, 60-170 MB).

**Image `glm53-tensorfold:b10`** = `results/W17/build-patches.txt` + 0530 0570 0580 0590 0600 0610
(`results/W19/build-patches.txt`); every patch applies (0610 with `patch` fuzz 1 at the import next to 0510's, which b9
does not carry), built on the head node while prod served, shipped with `docker save | docker load` (147 s; page cache dropped
after), identical layer digests on both nodes. **Dockerfile fix:** the base's torch is a pre-release
(`2.13.0a0+9186a08b2c.nv26.07`), so pip refused `xgrammar==0.2.8` with a `torch==<that>` constraint ("conflicting
dependencies"); now `pip install --no-deps xgrammar==0.2.8` + `transformers==5.17.0` (does not require torch) and the
build fails if `torch.__version__` changed. `pip freeze` b9 -> b10: only xgrammar, transformers, typer, shellingham,
annotated-doc added. Prepared weights reused.

### 1. Tests and kernel gates (prod stopped for the GPU part; both nodes in parallel; clocks unlocked)

| item | result | gate |
| --- | --- | --- |
| host suites in b10 (worker, prod serving, nice 19, 4 little cores) | 0600 `test_upstream_ports` 169, `test_disconnect_patches` 8; 0610 `test_grammar_patches` 129 (real tokenizer); 0530 `test_http_pin` 4; 0570 `test_decode_size_switch` 16; 0580 compile 5, emulator 58; 0590 compile 21, bench logic 3, emulator 49: **all passed** | pass |
| 0580 bitwise (`test_decode_loads_patches.py`, incl. 0570 == `_qmm` per shape) | 20 passed | pass |
| 0580 probe 3 (load path alone, flush) U 8 / 13 / 17 / 22 | 231 / 231 / 230 / 226 GB/s | **PASS** (>= 220) |
| 0580 kernel, best cfg `nc,8,1` vs grouped_kernel, U 8-22 | flush 1.113 / 1.045 / 1.187 / 1.032, rotate 0.981 / 1.012 / 1.086 / 1.083, geomean 1.093; every setting same bits; grouped_kernel itself already at 206-224 GB/s flush here | **FAIL** (1.05x flush on every window, 1.00x rotate) |
| 0570 cold per shape, geomean old / best new, rotate / flush | KDA f_b/g_b 1.70 / 1.04, index k 1.23 / 1.12, DSA kv_b 1.34 / 1.30, shared down 1.50 / 1.32, (1,16,8) 1024x4096 1.33 / 1.12, index q_b 1.37 / 1.18 (flush roof L2-invalid); single windows down to 0.76 (1 row flush, KDA) | PASS (every switched shape >= 1.0 in both modes; no EXCLUDE) |
| 0590 bitwise (`test_fat2_patches -k "not engine"`) + bench bits (2,048 / 4,096 / 8,192, uniform / skewed, every cfg) | 22 passed; ALL BITS SAME | pass |
| 0590 floor gate (fat2 contended <= 1.12x DRAM floor at 4,096, makespan <= fat) | fat2 is **slower than fat**: isolated 1.16x / 1.19x, contended 1.12x / 1.20x, makespan 1.07x / 1.09x fat at 2,048 / 4,096; 1.76x / 1.88x the floor (letter gate 0.75x: FAIL too) | **FAIL** -> off |
| NCCL (`ncclbw2.sh`, one container a node): all-gather 4 MiB 1-NIC default / ch2 / ch4 / ch8 / **2-NIC ch4** | 322 / 213 / 206 / 188 / **142** us (8 / 16 / 32 MiB: 432 / 791 / 1,682 -> 259 / 500 / 1,034) | gain PASS (0.44x, bar 0.90) |
| NCCL Ring vs Tree all-reduce 4 / 16 MiB | 1-NIC ring 642 / 1,454, tree 1,892 / 4,494; 2-NIC ring 429 / 814, tree 1,433 / 4,403 | see below |

- **0580 went into the combined load anyway** (decision at 12:35): bit-identical, probe passed, the strict
  bar missed on two flush windows (1.045, 1.032) and one rotate window (0.981); the in-situ trace decides (adopt only if
  the traced routed-expert decode time drops and 1-stream decode is not lower).
- **NCCL health criterion revised after its first evaluation.** v1 ("Tree within 1.5x of Ring") failed on the 1-NIC
  prod link as much as on 2 NICs (tree / ring 2.95-3.09 on one NIC): it tested NCCL's algorithm for 2 ranks, not the
  second function. v2: both algorithms run on 1 and 2 NICs and the second function slows neither (2-NIC / 1-NIC ring
  0.67 / 0.56, tree 0.76 / 0.98): PASS. The first COMBINED start (12:36, NCCL knobs off by v1) was aborted after 1 min
  and restarted with them on (12:37); `ncclgate-v1.txt` / `ncclgate.txt`, `run-COMBINED-aborted-1236.out`.
- Knobs of the combined load: `GLM53_TF_SELECT_SCRATCH=grow`, `ALLOC_TRIM_GB=0`, `ADMIT_MEM=free`, `DEC_QMM_MAXMB=3.5`,
  `DEC_EXPERT_LOADS=1` + `_CFG=nc,8,1`, `CPU_PIN=http`, `GRAMMAR=1`, `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` +
  `NCCL_PASSTHROUGH=1` + `NCCL_MIN/MAX_NCHANNELS=4` (`results/W19/combined.env`). 0590 off.

### 2. Loads (each: a TRACED start with one nsys capture, then a PLAIN start with the full sequence; :8001; page cache dropped, 4 slots every start)

**Why two starts a load.** `nsys launch` costs ~4 GiB of MemFree (W11): 4 slots only fit with
`GLM53_TF_BATCH_RESERVE_GB=6`, and the 4 x 250k stress / 314k needle under it would run 4 GiB closer to OOM with
minima that mean nothing for prod. So each load's capture ran in its own start (one capture per server start, W7's
rule), and every gate below ran on a plain start of the same config. Capture (`cap.sh`): a cold 21.5k prompt, then
prose 256 and code 384 alone, then 4 x prose 384 at once, 4 s apart (the same requests uncaptured first).

- **CONTROL** = `config/prod.env` + IMAGE b10 (every new knob off; 0600 on by default)
- **COMBINED** = the knobs above; **COMBINED2** = COMBINED without `DEC_QMM_MAXMB` (0570 off: the culprit, section 3),
  re-run for the gates 0570 can move only (bits, decode, one prefill pair)

| gate | CONTROL | COMBINED | COMBINED2 | bar |
| --- | --- | --- | --- | --- |
| exact / batchexact (before and after stress + MMLU) | 10/10, 4/4; 10/10, 4/4 | same | 10/10, 4/4 | pass |
| W9 transcripts, reply sha (all ab.py cells) | True, 4/4; 8794a3463259cc2f | same | same | pass |
| glmbench 13 cells: hashes | **13/13 == W18 A (b9)** | 13/13 == CONTROL | 13/13 | pass (knobs off == b9) |
| N1 long exactness | 6/6 + 6/6 | 6/6 + 6/6 | - | pass |
| grouped == alone (`mpf.py`, 92 requests) | 92/92 (90 grouped) | 92/92 | - | pass |
| replay n-64 (ab.py resends: cached == (n-1)//64*64, same sha) | 8/8; needle 298,304 of 298,325 | 8/8; needle 298,368 of 298,381 | 2/2 | pass |
| prefill 24.5k (4 runs) | 1,611 / 1,601 / 1,606 / 1,610 | **1,630 / 1,632 / 1,630 / 1,632** | 1,631 | not lower: +1.5% |
| prefill 98k (4 runs) | 1,605 / 1,605 / 1,599 / 1,600 | **1,627 / 1,629 / 1,628 / 1,624** | 1,630 | +1.6% |
| glmbench 1 stream geomean vs CONTROL | 0 | +2.16% | **+3.34%** | not lower |
| 4 streams, mean of 6 (paired per rep vs CONTROL) | 82.17 | 84.08 (+1.7..+2.9%) | **84.63 (+1.9..+3.6%, every rep)** | not lower |
| lone slots 0-3 (tok/s) | 52.8-57.3 | 53.0-58.3 | 53.4-59.0 | |
| C4 / C2 per-stream TTFT (RigMark shape, reasoning low) | 0.805 / 0.595 s | 0.763 / 0.547 s | - | |
| MMLU-200, refusals | 88.0%, 0/10 | 88.0%, 0/10 | - | >= 87% |
| 4 x 250k stress MemAvailable min, head / worker | 7.75 / 7.61 | **8.34 / 8.09** | - | >= 8: CONTROL fails, COMBINED passes |
| ~314k needle: found, lone prefill, min | yes, 1,374 tok/s, 6.58 / 6.31 | yes, **1,396**, **8.68 / 8.51** | - | |
| minimum over the whole load | 6.58 / 6.31 | **8.29 / 8.09** | - | |
| engine-ready MemFree (both first starts of their knob set) | 15.5 / 14.6 GiB | 16.9 / 16.1 | 16.2 / 15.7 | |
| OOM / NVRM / engine errors | 0 / 0 / 0 | 0 / 0 / 0 | 0 / - / 0 | |

**0600 checks** (`ports.py`, both loads): a non-streamed 32k-token request whose client closed after 5 s freed its
slot **0.18 s** after the close (request log `finish: cancelled`, 431-439 decode tokens); 4 at once: all free 0.26-0.31 s
after the last close, then 4 normal requests admitted at once (queue 4-7 ms); a queued ~100k-token 5th whose client
left after 2 s never ran and the 4 replies' hashes were unchanged; `"temperature": true` streamed, `chat_template_kwargs:
"x"` and a non-UTF-8 body: 400 each; `kill -USR1` on both ranks: stacks in both logs (7 / 6 thread blocks, COMBINED;
CONTROL's check read the logs with a local-time `--since` against the UTC container clock and saw nothing: harness bug,
fixed), serving after (17*23 -> 391); https image URL == the same image as a `data:` URL (653 prompt tokens, same
hash); `http://`, `https://127.0.0.1/`, `https://169.254.169.254/` refused with 400s and no URL in the message;
`/health` `completion_tokens_total` grows during a reply, `rounds / drafted / accepted_total` fold in after it.

**0610 checks** (`bench/structured.py`, COMBINED): plain (unconstrained) hashes == CONTROL's knob-off reference;
schemas 32/32 (8 schemas x greedy / sampled x thinking on / off: drafted == `"draft": false` == 4 concurrent, all
valid); tools 6/6 (required, named, strict auto; valid arguments, no markup); RigMark-like 50-object task with
`response_format`: valid both reps, same hash, 71.9 tok/s vs 72.2 unconstrained (-0.4%), tokens / round 4.32 vs 4.44;
exposed mask wait 350-406 ms over 546 windows (**~0.7 ms a round, above STRUCTURED-OUTPUT §8's 0.3 ms**: follow-up
`GLM53_TF_GRAMMAR_THREADS=8`). The unconstrained run is not valid JSON (the doc expected both valid; the gate is on the
schema run).

**0530 check:** boot line `http: HTTP threads on 0-4,10-14, the engine's threads unpinned`; during a streamed reply the
request's `tf-http` thread is allowed `0-4,10-14` and runs on cpu 1 (`threads-COMBINED2-stream.txt`, `ps -L`); the round
loop (`tf-serve`) and the rest stay on 0-19.

### 3. Attribution from the two traces (rank 0; rank 1 shows the same deltas within ~2 us / 0.04 ms a token; `attcmp-r0.txt`, `attcmp-r1.txt`)

Exclusive kernel time (each instant split among the kernels running then; families + GPU idle = wall), per prompt
token (prefill, 21,454 tokens) or per generated token (decode). Captured under nsys (the same requests uncaptured
first), CONTROL vs COMBINED (0570 still on).

| stage | CONTROL | COMBINED | delta | fix |
| --- | ---: | ---: | ---: | --- |
| **prefill wall**, us / token | 634.8 | 621.3 | **-13.5 (-2.1%)** | |
| prefill NCCL exposed (SendRecv / all-gathers; summed 161.1 -> 132.3) | 99.3 | 78.2 | -21.1 | NCCL 2 functions + 4 channels |
| prefill GPU idle | 9.7 | 6.6 | -3.2 | most likely NCCL (not isolated) |
| prefill dense / KDA / router / hc (exclusive; summed +1.6 / +0.5 / +0.1 / +0.5) | 245.5 | 256.0 | +10.4 | overlap shift: time NCCL no longer shares |
| prefill routed experts (fat, rot_in1) | 189.3 | 189.1 | -0.2 | 0590 off: unchanged |
| **1 stream prose wall**, ms / token | 21.45 | 20.95 | **-0.50 (-2.3%)** | |
| routed experts decode (grouped_kernel 10.22 -> ld_kernel 9.77) | 11.19 | 10.71 | **-0.48 (-4.3%)** | **0580** |
| dense decode (_qmm + _reduce 5.25 -> 4.99, + q4_kernel 0.44) | 4.88 | 5.03 | **+0.15 (+3.1%)** | **0570: slower in situ** |
| RoCE all-gathers (incl. waiting for the peer) | 0.74 | 0.66 | -0.08 | less waiting (not attributed to one knob) |
| GPU idle | 1.15 | 1.09 | -0.05 | CPU_PIN / noise |
| **1 stream code wall** | 16.35 | 16.14 | -0.21 (-1.3%) | routed -0.38 (-4.1%), dense +0.12, idle +0.05 |
| **4 streams wall**, ms / token | 13.65 | 13.55 | -0.10 (-0.7%) | routed -0.24 (-2.9%: ld_kernel 7.57 vs 7.97), dense +0.10, RoCE -0.05, idle +0.08 |

Per fix:

| fix | measured contribution | decision |
| --- | --- | --- |
| 0580 expert load path (`nc,8,1`) | in situ routed decode -4.1..-4.6% (1 stream), -2.9..-3.1% (4 streams); the bulk of the +3.3% 1-stream / +3.0% 4-stream gain | **adopt** (trace dropped, decode higher) |
| 0570 dense size switch (3.5 MiB) | in situ the switched shapes take 0.44 ms / token where `_qmm` + `_reduce` took 0.26 (1.7x slower; cold bench said 1.0-1.7x faster): dense +3-5%; removing it (COMBINED2) +1.2% 1 stream, +0.7% 4 streams | **off** (culprit) |
| NCCL both CX7 functions, 4 channels | prefill exposed NCCL -21%, prefill +1.5-1.6% untraced; engine-ready MemFree +0.7..+1.5 GiB over CONTROL (likely 4 instead of 64 channels' buffers; not isolated); decode unchanged | **adopt** |
| 0550 scratch (`SELECT_SCRATCH=grow`) | 314k needle minimum +2.1 / +2.2 GiB and its lone prefill +1.6% (W18 S: the needle dip 3.4 -> <1 GiB); stress minimum +0.6 / +0.5 with NCCL | **adopt** |
| 0530 `CPU_PIN=http` | no measurable GPU-idle change (-0.05 / +0.05 / +0.08 ms / token over prose / code / 4 streams, r1 -0.02..-0.08): within noise | adopt (no cost; gates passed with it) |
| 0610 grammar | unconstrained unchanged (hashes, trace); schema run -0.4% tok/s | **adopt** |
| 0600 upstream ports | host only; every check above | on (default) |
| 0590 fat2 | kernel slower than fat | **off** |

### 4. Adopted and production

`config/prod.env` (header with the W19 gates and per-knob reverts; previous file `results/W19/prod.env.before-W19`):
`IMAGE=glm53-tensorfold:b10`, `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, `NCCL_PASSTHROUGH=1`, `NCCL_MIN_NCHANNELS=4`,
`NCCL_MAX_NCHANNELS=4`, `GLM53_TF_SELECT_SCRATCH=grow`, `GLM53_TF_DEC_EXPERT_LOADS=1`, `GLM53_TF_DEC_EXPERT_LOADS_CFG=nc,8,1`,
`GLM53_TF_CPU_PIN=http`, `GLM53_TF_GRAMMAR=1` (= COMBINED2). Before / after (CONTROL = b9's behaviour -> adopted):
prefill 24.5k 1,607 -> 1,631, 98k 1,602 -> 1,628 tok/s (+1.5%); 1-stream glmbench +3.3% (e.g. chat 47.7 -> 49.5, code 512
70.4 -> 72.3, structured 104.3 -> 108.1 tok/s); 4 streams 82.2 -> 84.6 tok/s (+3.0%); memory minima after the heavy
sequence: stress 7.75 / 7.61 -> 8.34 / 8.09 GiB, needle 6.58 / 6.31 -> 8.68 / 8.51, whole load 6.58 / 6.31 -> 8.29 / 8.09.

Production restarted on it at 13:49 (`restore.sh W19-adopt`: page cache dropped, ready in 37 s, calibration cached
(COMBINED2's table), **4 request slots**, engine-ready MemFree 17.0 GiB (b9 prod W17: 14.6), canary ok 74.9 tok/s,
local and https `/v1/models`, `17*23` -> `391`), both ranks on b10 with the knobs above, watchdog timer active, lease
refresher and deadman gone, lease deleted.

Notes: the 20-min / 90-min window rules were lifted at 12:35 (results over uptime): prod was up 12 min between the
windows and the second window ran the combined load, the 0570-off re-run and the attribution back to back. The kernel
gates ran with unlocked clocks (each compares old vs new in one process). The 0570 cold bench times its best eligible
placement, not 0570's fixed `SMALL_PLACE`; in situ it lost anyway. Not re-run with 0570 off: the memory sequence (0570
allocates 32 KB) and the capture.

## W20: recipe batch (b11 = b10 + 0620) + final RigMark x3 and tool-calling benches (2026-09-30, one campaign 13:54-19:20 + tool benches after) — adopted: image b11 with `GLM53_TF_TOOL_FIXES=all`; not adopted: container cpusets on the X925s, `GLM53_TF_GRAMMAR_THREADS=8`; prod on b11

Files: `results/W20/` (text / JSON only in this repo: per load `*-B10*` (the image b10 control load) and `*-RECIPE*`
in W18's harness naming; `summary-B10-RECIPE.md`; `structured-B10|RECIPE.json`; `build-patches.txt` (b11); `final/`
(the 3 RigMark rounds' `summary.md`; receipts in `results/rigmark/tensorfold-20260930-w20-final-r*`);
`final-glmbench/`; `*-PROD*` = checks on the restored prod). Logs, per-probe CPU / RoCE traces and the request logs
are not published. Harness: W18's gates; every load on :8001, page cache dropped, 4 slots, RoCE trace on (same in
every load).

Host-side tuning outside this repo, done in the same campaign on image b10 before the recipe batch: +4.8% single-stream
decode, +4.3% at 4 streams, more memory headroom (stress-test memory minimum now ~10.7 GiB). Bits unchanged (glmbench
reply hashes 13/13 equal). B10 below is image b10 after that tuning.

### 1. Recipe batch (one load; `summary-B10-RECIPE.md`)

**Image `glm53-tensorfold:b11`** = `results/W19/build-patches.txt` + 0620 (`results/W20/build-patches.txt`); every patch
applies, built on the head (cached base layers, 8 s), shipped with `docker save | docker load` (152 s), identical layer
list on both nodes. **RECIPE** = `config/prod.env` + `IMAGE=b11 CPUSET=5-9,15-19 GLM53_TF_GRAMMAR_THREADS=8
GLM53_TF_TOOL_FIXES=all` (boot lines: `tool calling (patches/0620): GLM53_TF_TOOL_FIXES=args,choice,history,reasoning,
thinkcalls`, grammar `mask fills on 8 thread(s)`, both containers `--cpuset-cpus 5-9,15-19`, and **`http: HTTP threads
on 5-9`**: with no A725 in the set, 0530 puts the HTTP threads on half the X925s). First start: 1 slot, serve.sh's
automatic restart gave 4.

| gate | B10 (b10) | RECIPE | bar |
| --- | --- | --- | --- |
| exact / batchexact (before and after stress + MMLU), transcripts, reply sha | 10/10, 4/4 x2, True, 8794a3463259cc2f | same | pass |
| glmbench 13 cells: hashes / geomean | 13/13 | 13/13 == B10; **+0.21%** | not lower: pass |
| 4 streams, mean of 6 | 88.58 [92.5, 84.6, 87.8, 93.5, 85.0, 88.1] | **87.22** [90.9, 83.7, 86.3, 91.5, 84.4, 86.5]: **-1.5%, lower in every paired rep** (-0.7..-2.1%) | **FAIL** |
| 4 streams RoCE skew mean / p90 (us) | 39.8 / 88.1 | 46.0 / 121.4 | |
| prefill 24.5k / 98k (mean of 4) | 1,659 / 1,654 | 1,661 / 1,652 | pass |
| N1 12/12, grouped == alone 92/92, MMLU-200 88.0%, refusals 0/10, needle 314k | pass | pass | pass |
| stress / whole-load min r0 / r1 | 10.72 / 10.77 | 10.57 / 10.43 | >= 8: pass |
| structured (`bench/structured.py` schemas / rigmark / tools / plain) | PASS (plain written as the reference) | PASS, plain hashes == B10 | pass |
| rigmark-shape schema run: tok/s, exposed mask wait a window | 76.5 / 76.8; 0.87 / 0.42 ms (mean 0.65) | 77.7 / 77.3; 0.75 / 0.59 ms (mean 0.67) | 8 threads: **no gain** |
| C4 per-stream TTFT (reasoning low / off) | 0.721 / 0.565 s | 0.722 / 0.568 s | |

Decisions:

- **0620 / `GLM53_TF_TOOL_FIXES=all`: adopted** (every gate passes; host only, tool requests only; its benefit is in §4).
- **`CPUSET=5-9,15-19`: not adopted.** The only knob in the load that touches unconstrained decode, and 4 streams fell
  in all 6 paired reps; the HTTP threads (tokenizing / streaming 4 replies) now share the X925s with the engine. A
  cpuset that keeps an A725 or two for HTTP (`CPUSET=0,5-9,15-19` + `CPU_PIN=http=0`) was not tried.
- **`GLM53_TF_GRAMMAR_THREADS=8`: not adopted** (exposed wait unchanged at ~0.65 ms a window; the unconstrained run moved
  as much as the schema run, +2%).

### 2. Production

`config/prod.env` (`config/prod.env.example` here): `IMAGE=glm53-tensorfold:b11` + `GLM53_TF_TOOL_FIXES=all` (header
with the gates and the revert). Prod restarted (page cache dropped, ready in 64 s, **4 request slots**, RoCE on both
functions, the `tool calling (patches/0620)` boot line, grammar 4 threads, HTTP threads on 0-4,10-14, canary 79.7
tok/s). Checks on prod (`*-PROD*`): exact 10/10, batchexact 4/4, reply sha 8794a3463259cc2f, prefill 24.5k 1,658 tok/s.

### 3. RigMark on the final prod (3 runs, 18:49-19:16, prod serving, no restart, nothing else on the endpoint)

W17's final RigMark procedure (`scripts/rigmark/run.sh tensorfold`, RigMark pinned at
`c5a0db01b054` clean, body `{"chat_template_kwargs":{"reasoning_effort":"low"}}`, a new COMPARISON_ID a
run `...-tensorfold-w20-final-rN`, run.sh's metadata); receipts `results/rigmark/tensorfold-20260930-w20-final-r1..r3`
(sha256 ok, **15/15 basic output gates each**), summary `results/W20/final/summary.md` (W17's `summarize.py`).

| metric (mean of 3, min-max) | **W20 prod (b11)** | W17 prod (b9), 3 runs | change | vLLM TP2 k=7 (Alex) | W20 / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| code decode tok/s | **72.4** (72.2-72.6) | 67.9 | +6.6% | 44.0 | 1.65x |
| prose decode tok/s | **45.7** (45.3-45.8) | 43.0 | +6.3% | 18.9 | 2.42x |
| structured decode tok/s | **95.6** (95.6-95.7) | 88.8 | +7.7% | 64.9 | 1.47x |
| code / prose / structured TTFT s | 0.48 / 0.38 / 0.45 | 0.50 / 0.41 / 0.47 | | 0.60 / 0.49 / 0.47 | |
| cold prefill 8K / 32K / 64K tok/s | **1,610 / 1,684 / 1,667** | 1,560 / 1,634 / 1,620 | +3.2 / +3.0 / +2.9% | 1,813 / 1,908 / 1,922 | 0.89 / 0.88 / 0.87x |
| replay TTFT 8K / 32K / 64K s | 0.21 / 0.23 / 0.26 | 0.22 / 0.25 / 0.27 | | 4.52 / 2.97 / 5.77 | 21 / 13 / 22x |
| C1 / C2 / C4 aggregate tok/s | **57.5 / 76.0 / 95.1** (C4 93.0-98.4) | 53.1 / 70.1 / 91.0 | +8.3 / +8.4 / +4.5% | 31.6 / 42.0 / 66.1 | 1.82 / 1.81 / 1.44x |
| C1 / C2 / C4 per-stream TTFT s | 0.47 / 0.61 / 0.84 | 0.49 / 0.65 / 0.89 | | 0.60 / 0.68 / 0.81 | C4 0.96x |

W17 -> W20 includes W19 (b10: 0580 expert loads, NCCL 2 functions, scratch, CPU_PIN, grammar: +3.3% 1 stream / +3.0%
4 streams / +1.5% prefill in W19's own A/B) and the host-side tuning above (+4.8% / +4.3%). C4 first tokens (0.84 s)
are still the one row behind vLLM's (0.81 s).

### 4. glmbench on the final prod (3 rounds, 19:35-19:42, prod serving on :8000, nothing else on the endpoint)

`results/W20/final-glmbench/` (suites tf,tweet,kit,edit, `--reps 3 --long-tokens 512` a round = the W20 loads'
ab.sh settings; `table.py` -> `table-final.md`). A round's cell value = the median of its 3 reps; mean / min / max over
the 3 rounds. "W20 b10" = the same suite once in the B10 load (b10, :8001, RoCE trace on). Yesterday's image and the
vLLM kit: the fixed reference numbers (greedy chat / code / structured; vLLM also hashmap / essay).

| suite | cell | mode | tokens | mean | min | max | W20 b10 | vs B10 | yesterday | vLLM kit | vs vLLM | hashes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tf | code | sampled (T=1) | 64 | **51.1** | 51.0 | 51.1 | 51.8 | -1.4% |  |  |  | rounds agree |
| tf | chat | sampled (T=1) | 64 | **48.6** | 48.1 | 48.9 | 49.5 | -1.7% |  |  |  | rounds agree |
| tf | code | greedy (T=0) | 64 | **89.6** | 89.3 | 90.0 | 88.9 | +0.8% | 77.6 | 41.9 | 2.14x | same |
| tf | chat | greedy (T=0) | 64 | **51.6** | 51.5 | 51.6 | 49.8 | +3.6% | 44.6 | 22.8 | 2.26x | same |
| tweet | sequence | greedy (T=0) | 512 | **105.1** | 105.0 | 105.1 | 105.0 | +0.1% |  |  |  | same |
| tweet | code | greedy (T=0) | 512 | **75.9** | 75.8 | 76.0 | 70.8 | +7.3% |  |  |  | same |
| tweet | json | greedy (T=0) | 512 | **84.0** | 84.0 | 84.0 | 84.3 | -0.4% |  |  |  | same |
| kit | hashmap | greedy (T=0) | 200 | **59.6** | 59.5 | 59.7 | 60.0 | -0.8% |  | 30.0 | 1.99x | same |
| kit | structured | greedy (T=0) | 200 | **112.3** | 111.9 | 113.1 | 113.9 | -1.4% | 100.6 | 72.7 | 1.54x | same |
| kit | essay | greedy (T=0) | 200 | **50.5** | 50.4 | 50.6 | 50.6 | -0.3% |  | 26.1 | 1.93x | same |
| edit | edit-rename | greedy (T=0) | 1024 | **124.1** | 123.1 | 124.7 | 125.1 | -0.8% |  |  |  | same |
| edit | edit-comments | greedy (T=0) | 1024 | **108.0** | 105.9 | 109.2 | 109.9 | -1.7% |  |  |  | same |
| edit | edit-print-to-log | greedy (T=0) | 1024 | **126.5** | 125.7 | 127.1 | 128.7 | -1.7% |  |  |  | same |

geomean vs B10 over 13 cells: +0.09%

Hashes: every round's 13/13 cells (11/11 greedy) == B10's (b10), i.e. b11 + `TOOL_FIXES=all` is bit-identical on this
suite; greedy cells return one sha across all 9 reps; sampled (seeded) cells agree round to round. Geomean vs B10
+0.09% (the prod server has no RoCE trace on; the cells move -1.7..+7.3%, tweet code 512 being B10's one low cell).
Against yesterday's image: chat greedy 44.6 -> 51.6 (+15.7%), code greedy 77.6 -> 89.6 (+15.5%), structured 100.6 ->
112.3 (+11.6%). Against the vLLM kit: 2.14x code, 2.26x chat, 1.54x structured, 1.99x hashmap, 1.93x essay.

### 5. Tool-calling benchmarks (docs/TOOL-CALLING.md §5) — partial

Run from a separate machine (`scripts/tooleval/run.sh`; tool-eval-bench c7b5b95, spark-bench 125ba16) against prod
over an SSH port forward to the head's 127.0.0.1:8000. spark-bench was stopped mid-run (the rig was needed for §4).
Results: `results/tooleval/20260930-teb-off-fixes-PARTIAL/`. **One run on our checkpoint, not an average.**

| run | bench | result |
| --- | --- | --- |
| F1: prod b11, `TOOL_FIXES=all`, thinking off | tool-eval-bench, 69 scenarios, temperature 0 | **complete: score 90** (124 / 138 points; deployability 86, responsiveness 78); **C multi-step chains 8/8 (100%)**; A 6/6, B 6/6, D 5/6, E 6/6, F 6/6, G 4/6, H 10/10, I 18/20, J 6/6, K 23/26, L 7/8, M 3/6, N 6/6, O 10/12; not passed: TC-21, 43, 51, 62, 68 (fail), TC-11, 39, 52, 57 (partial). The tester's run (their checkpoint, issue #6): 90 / C 75% |
| F1 | spark-bench TrueScore | stopped mid-run: no score |
| B1 (fixes off, thinking off), F2 (fixes on, thinking high) | both | not run yet |

### 6. Notes

- The 1-stream RoCE trace is empty in B10 and RECIPE: the dump fires every 65,536 exchanges and the 1-stream probe
  sits right at that count, so it got no dump; the 4-stream trace (the plan's gate) is in every load.
- A test window that ends with prod stopped (`serve.sh stop` removes the containers) is not healed by the watchdog
  (it stands down on "both ranks absent"): start prod again yourself.
