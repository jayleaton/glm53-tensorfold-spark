Follow me on X for more updates: https://x.com/jayleaton

# GLM-5.3-Flash on TensorFold, 2x NVIDIA DGX Spark

Serve GLM-5.3-Flash (the abliterated EXL3 4-bit checkpoint
[`neko-legends/GLM-5.3-Flash-Uncensored-EXL3`](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3))
across two NVIDIA DGX Sparks, tensor-parallel over the 200 Gb/s CX7 link, behind an OpenAI-compatible API. The
engine is [TensorFold](https://github.com/ashhart/TensorFold) (pinned, unmodified submodule) plus 77 patches applied
at image build: 4-bit non-expert weights, image input (the checkpoint's own vision tower), structured output, tool-calling fixes, a latent (absorbed MLA) KV cache, fast chunked prefill, a multi-session
state cache with an NVMe tier, batching of up to 4 requests over a shared 1M-token KV pool, FP8 KV storage,
shared system-prompt reuse, a RoCE all-gather, fast restarts, deeper drafting and verify windows, and ops tooling.
Every patch is off by default; the configs in `config/` turn on the measured set, and
[`docs/PATCHES.md`](docs/PATCHES.md) records the ones that were measured and not adopted.

> **Work in progress.** This is an experimental setup, measured on one pair of Sparks. Knobs, defaults, APIs and
> numbers may change between commits. Read [Limits](#limits-and-negatives) before relying on it.

Weights attribution (the checkpoint's license requires it): the weights are by **Local Inference Lab, Inc.**
(<https://local-inference-lab.ai/>), upstream source
<https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3>, under the ShapleyMCG License 1.0. They are not
included here. See [Licensing](#licensing).

SPDX-License-Identifier: Apache-2.0 (this project's own code, scripts, benchmarks and docs; see [Licensing](#licensing)).

## What's new (W20)

Patch 0620 (tool-calling fixes, 77 patches in total), test window W20 and a new 3-run RigMark. Production config
(`config/prod.env.example`): image b11 = image b10's list + 0620, with `GLM53_TF_TOOL_FIXES=all`. Numbers from
[`docs/RESULTS.md`](docs/RESULTS.md) W20.

Host-side tuning outside this repo: +4.8% single-stream decode, +4.3% at 4 streams, more memory headroom (stress-test
memory minimum now ~10.7 GiB). Same bits (reply hashes unchanged).

**Not adopted:** pinning the server containers onto the X925 cores (`CPUSET=5-9,15-19`: 4 streams -1.5% in every
paired rep, because the HTTP threads then share the engine's cores) and 8 grammar threads
(`GLM53_TF_GRAMMAR_THREADS=8`: no gain).

**Tool calling (patch 0620, adopted, `GLM53_TF_TOOL_FIXES=all`).** Fixes found while chasing an independent
tester's multi-step results (issue #6): a `content: null` assistant turn rendered as the text `None` before its tool
calls; earlier steps lost their reasoning when the client did not send it back (GLM-5.3 interleaves thinking with
tool calls); `tool_choice: none` / a named function and `parallel_tool_calls: false` were ignored; argument values
were typed by the first schema type only; complete calls at the end of an unclosed think block were dropped. Host
only, same bits: requests without tools are untouched (every W20 hash equal). One tool-eval-bench run on production
(thinking off, our abliterated checkpoint): **score 90, multi-step chains 8/8** (the tester's run on their checkpoint:
90, chains 6/8). One partial run, not an average; spark-bench was stopped before it finished.
([`docs/TOOL-CALLING.md`](docs/TOOL-CALLING.md), [`results/tooleval/`](results/tooleval/20260930-teb-off-fixes-PARTIAL/README.md))

**RigMark, W20 production, 3 runs averaged** (Alex Ellis's [RigMark](https://github.com/alexellis/rigmark) standard
suite, unmodified; receipts in [`results/rigmark/`](results/rigmark/README.md)):

| RigMark (mean of 3 runs' medians) | **TensorFold W20** | TensorFold W17 (previous) | vLLM TP2 k=7 (Alex Ellis, published) | W20 vs vLLM |
|---|---:|---:|---:|---:|
| Code / prose / structured decode tok/s | **72.4 / 45.7 / 95.6** | 67.9 / 43.0 / 88.8 | 44.0 / 18.9 / 64.9 | 1.65x / 2.42x / 1.47x |
| C1 / C2 / C4 aggregate tok/s | **57.5 / 76.0 / 95.1** | 53.1 / 70.1 / 91.0 | 31.6 / 42.0 / 66.1 | 1.82x / 1.81x / 1.44x |
| Cold prefill 8K / 32K / 64K tok/s | 1,610 / 1,684 / 1,667 | 1,560 / 1,634 / 1,620 | **1,813 / 1,908 / 1,922** | **0.89x / 0.88x / 0.87x** |
| Resend of an identical 64K prompt, TTFT s (cached) | **0.26** | 0.27 | 5.77 | 22x |
| C4 per-stream TTFT s | 0.84 | 0.89 | **0.81** | **0.96x** |

**Our own suite** (`bench/glmbench.py`, same client for both stacks; not RigMark), W20 production, 3 rounds, each the
median of 3 reps; the vLLM column is the production kit on the same abliterated weights (table (a) below):

| glmbench cell (decode tok/s) | **TensorFold W20** | vLLM kit | vs vLLM |
| --- | ---: | ---: | ---: |
| tf chat, greedy (T=0), 64 tok | **51.6** | 22.8 | 2.26x |
| tf code, greedy (T=0), 64 tok | **89.6** | 41.9 | 2.14x |
| kit structured, greedy, 200 tok | **112.3** | 72.7 | 1.54x |
| kit hashmap (prose), greedy, 200 tok | **59.6** | 30.0 | 1.99x |
| kit essay, greedy, 200 tok | **50.5** | 26.1 | 1.93x |
| tf chat / code, sampled (T=1), 64 tok | 48.6 / 51.1 | 27.1 / 35.2 | 1.79x / 1.45x |

Still behind vLLM: **cold prefill** (0.87-0.89x on RigMark's 8K-64K prompts) and **C4 first token** (0.84 vs 0.81 s).
Still off: 0570 (dense size switch) and 0590 (fat2 prefill experts), both slower in the server (W19). Different
weights than Alex's runs (abliterated EXL3 4-bit here, NVFP4 there): not a strict comparison.

More lever A/Bs are running on the rig now; results will follow in a later update.

## What's new (W19)

Patches 0570-0610 (5 new, 76 in total), test windows W18 and W19, and a new production config
(`config/prod.env.example`: image b10 = image b9's list + 0530, 0570, 0580, 0590, 0600, 0610). Numbers from
[`docs/RESULTS.md`](docs/RESULTS.md) W18 / W19, measured against a control that is the same image with every new knob
off (it served image b9's bits: 13/13 reply hashes equal). RigMark was not re-run for this update; the RigMark table
below is still the image b9 release.

| | Before (control) | **W19 production** | Change |
| --- | ---: | ---: | --- |
| Memory: 4 x 250k stress, MemAvailable minimum (head / worker) | 7.75 / 7.61 GiB | **8.34 / 8.09 GiB** | the 8 GiB gate passes again |
| Memory: ~314k needle after the stress, minimum | 6.58 / 6.31 GiB | **8.68 / 8.51 GiB** | +2.1 / +2.2 GiB |
| Decode, 1 stream (glmbench geomean, 13 cells) | | | **+3.3%** (chat 47.7 -> 49.5, code 70.4 -> 72.3, structured 104.3 -> 108.1 tok/s) |
| Decode, 4 streams aggregate (mean of 6) | 82.2 tok/s | **84.6 tok/s** | **+3.0%** (every paired rep +1.9..+3.6%) |
| Prefill 24.5k / 98k | 1,607 / 1,602 tok/s | **1,631 / 1,628 tok/s** | +1.5% |
| C4 per-stream first token (RigMark shape, reasoning low; our client) | 0.81 s | **0.76 s** | -0.05 s |
| Exactness (drafted == serial, batched == alone, reply hashes, grouped == alone 92/92), MMLU-200, refusals | pass, 88.0%, 0/10 | pass, 88.0%, 0/10 | unchanged |

- **Memory headroom solved for the heavy stress case.** W17 left every config under the 8 GiB MemAvailable target
  after a heavy warm-up. W18 traced the slow prefills that kept 0550 off to its allocator trim alone, and W19 adopted
  0550's pre-grown selection scratch (`GLM53_TF_SELECT_SCRATCH=grow`; trim and page-cache admission stay off) together
  with NCCL on 4 channels (less NCCL buffer memory): the stress minimum is back over 8 GiB on both nodes and the 314k
  needle no longer dips (it had dropped ~3.5 GiB for ~35 s). This headroom is room for future gains that were
  rejected on memory before, such as 8,192-row lone prefill chunks (+~4% prefill in W8, rejected at a 7.2-7.8 GiB
  minimum).
- **Decode +3.3% (1 stream) / +3.0% (4 streams)** from patch 0580 (`GLM53_TF_DEC_EXPERT_LOADS=1`, `_CFG=nc,8,1`): a
  new load path for decode's routed experts (16-byte non-coherent vector loads issued a step ahead of the math, a
  one-round-trip prologue), same bits.
  The nsys traces put the routed experts' decode time 4.1-4.6% lower at 1 stream and 2.9-3.1% lower at 4 streams.
  ([`docs/DECODE-KERNELS-2.md`](docs/DECODE-KERNELS-2.md))
- **Prefill +1.5%** from NCCL on both functions of the CX7 port with 4 channels (`NCCL_IB_HCA` lists both,
  `NCCL_PASSTHROUGH=1`, `NCCL_MIN/MAX_NCHANNELS=4`): a 4 MiB all-gather 322 -> 142 us, prefill's exposed NCCL time -21%.
  The idea comes from the [kindlingai GX10 recipe](https://github.com/kindlingai/glm-5.3-flash-gx10)
  ([`docs/KINDLING-AUDIT.md`](docs/KINDLING-AUDIT.md)).
- **Structured output** (patch 0610, `GLM53_TF_GRAMMAR=1`, using [xgrammar](https://github.com/mlc-ai/xgrammar)):
  OpenAI `response_format` (`json_object`, `json_schema`), vLLM's `guided_*` / `structured_outputs`, and tool calls
  with `tool_choice: "required"`, a named function or `"strict": true`, enforced token by token. It stays exact with
  speculative decoding and batching: drafts are checked against the grammar, and a constrained reply is the same
  bytes drafted, undrafted and next to 3 other requests (8 schemas x greedy / sampled x thinking on / off: 32/32;
  tools 6/6). Unconstrained replies are unchanged; a 50-object JSON task ran -0.4% tok/s.
  ([`docs/STRUCTURED-OUTPUT.md`](docs/STRUCTURED-OUTPUT.md))
- **Upstream TensorFold fixes, ported** (patch 0600, on by default; MIT code from TensorFold 0.3.6.2 / 0.5.0): a
  client that disconnects frees its slot (0.18 s after the close, also while queued or prefilling); malformed requests
  (wrong-typed sampling fields, bad `chat_template_kwargs`, non-UTF-8 bodies) get 400s instead of engine errors; image
  URLs are fetched over https only, from public addresses only (no SSRF to loopback, LAN or cloud metadata), with the
  connection pinned to the checked address; `return_token_ids`; `/health` reports token totals; `kill -USR1` dumps
  every thread's stack. ([`docs/UPSTREAM-PORTS.md`](docs/UPSTREAM-PORTS.md), [`docs/UPSTREAM-050-AUDIT.md`](docs/UPSTREAM-050-AUDIT.md))
- **Also adopted:** 0530's `GLM53_TF_CPU_PIN=http` (rank 0's HTTP threads off the engine's cores; no measurable
  change, no cost). The image build installs xgrammar with `--no-deps` plus `transformers` and fails if the base
  image's torch changed (`docker/Dockerfile`).

**What didn't work in W19** (both stay in the series, off by default):

- **0570 dense size switch** (0440's dense kernel for small 4-bit matrices only): 1.0-1.7x faster per shape in the
  cold microbenchmark, but in the server the switched shapes took 1.7x the time of today's kernels (dense decode
  +3-5%); with it on, 1-stream decode gained only +2.2% instead of +3.3%. Off.
- **0590 fat2 prefill experts** (one persistent pipelined routed-expert kernel for prefill, ideas from the kindlingai
  recipe re-implemented; no code copied): same bits, but 1.16x / 1.19x slower than today's `fat` kernel at 2,048 /
  4,096 rows. Off. ([`docs/EXPERT-PREFILL-V2.md`](docs/EXPERT-PREFILL-V2.md))

## What's new (2026-09-30)

Patches 0420-0560 (14 new, 71 in total), test windows W11-W17, and the RigMark receipts. The production config
(`config/prod.env.example`) is W17's: image b9 = patches 0001-0490 + 0500 + 0540 + 0550 + 0560, with 0560's
multi-slot prefill on (`GLM53_TF_MULTI_PREFILL=1`) and 0550 built in but off. Full list: [`docs/CHANGES-SUMMARY.md`](docs/CHANGES-SUMMARY.md#update-2026-09-30-patches-0420-0560-test-windows-w11-w17);
measurements: [`docs/RESULTS.md`](docs/RESULTS.md) W11-W17.

**RigMark, release config, 3 runs averaged** (2026-09-30, image b9; receipts in [`results/rigmark/`](results/rigmark/README.md)):

| RigMark (mean of 3 runs' medians) | TensorFold (this release) | vLLM TP2 k=7 (Alex Ellis, published) |
|---|---:|---:|
| Code / prose / structured decode tok/s | **67.9 / 43.0 / 88.8** | 44.0 / 18.9 / 64.9 |
| C1 / C2 / C4 aggregate tok/s | **53.1 / 70.1 / 91.0** | 31.6 / 42.0 / 66.1 |
| Cold prefill 8K / 32K / 64K tok/s | 1,560 / 1,634 / 1,620 | **1,813 / 1,908 / 1,922** |
| Warm replay of an identical prompt, TTFT 8K / 32K / 64K s | **0.22 / 0.25 / 0.27** | 4.52 / 2.97 / 5.77 |
| C4 per-stream TTFT s | 0.89 | **0.81** |

Run-to-run spread (min-max of the 3 runs): code 67.4-68.6, prose 42.2-43.5, structured 88.6-89.0, C4 89.5-92.1,
cold 64K 1,618-1,621, replay TTFT 64K 0.27-0.27, C4 TTFT 0.88-0.89. Where we are behind: **cold prefill** (0.84-0.86x
of vLLM) and **C4 time to first token** (0.89 vs 0.81 s). Different weights than Alex's runs (abliterated EXL3 4-bit
here, NVFP4 there), so this is not a strict RigMark comparison; see [`results/rigmark/`](results/rigmark/README.md).

- **Image input, with GLM-5.3-Flash's own vision tower** (patch 0500, on in production since W15). The checkpoint
  ships a BF16 vision tower (24 blocks, 1.13 GB) that the text-only stack ignored; rank 0 now runs it and the prompt
  carries the image rows. Send OpenAI `image_url` parts (`data:` or `http(s)` URLs) on `/v1/chat/completions`
  ([example below](#image-input)). The preprocessing matches transformers' `Glm5NextImageProcessor` bit for bit on
  CPU; text-only requests are bit-identical with vision on or off. Checked on synthetic images (`results/W14/img/`,
  generated by `mkimg.py`): window titles, a calculator display and a clock read from a 1920 x 1080 screenshot, a
  receipt total, a 12 px paragraph and a terminal at 0 character errors (max effort), counts of 3 / 7 / 12 circles
  and 20 stars, two-chart comparisons 4/4; bad images are HTTP 400s. Time to first token: 0.76 s (512 x 512) to 2.8 s
  (1920 x 1080) and 8.1 s (4K). Images are content-hashed into the caches, so a conversation's earlier images are
  not re-encoded and a resend of the same image conversation resumes from the cache.
- **Verified warm replay** (patch 0540). When the same prompt comes back (a regenerate, a retry, an agent
  re-sending a conversation, RigMark's "immediate replay"), the server resumes from a snapshot of the whole model
  state taken 64 tokens before the prompt's end and computes only those 64 tokens: **0.21-0.26 s to the first token
  at 8K, 32K and 64K** (W13 before the fix: 5.1 / 9.9 / 10.4 s). This is prefix / session caching of an identical
  prompt, not a faster prefill (cold prefill is the row above), and the outputs are byte-identical to computing the
  prompt from scratch. Verification (docs/RESULTS.md W15 §6): the request log shows `cached` = n - 64 for every
  replay; RigMark's receipts show each replay's output hash equal to its cold run's (9/9, both runs); never-seen
  8K / 32K / 64K token-id prompts gave byte-identical 64-token greedy outputs cold, replayed and replayed again; and
  negative controls (one token changed at the start, middle, n - 10 or n - 100, or a different prompt of the same
  length) never reused more than the prefix they share with what was stored. vLLM replays slower on this model
  because GLM-5.3-Flash is a hybrid (KDA linear attention + MLA): vLLM can restore the recurrent KDA state only at
  page-aligned checkpoints and rounds a hit down to a whole block, so a replay recomputes several thousand tokens
  (~8.2k / 5.7k / 11.1k at 8K / 32K / 64K from Alex's receipts; nothing is reused at 8K).
- **RigMark receipts.** The release run above (3 runs on image b9), the W13 baseline (image b5) and two W15 runs
  (image b7), all 15/15 gates, with receipts, cards, request logs and the comparison with Alex Ellis's
  published vLLM runs: [`results/rigmark/`](results/rigmark/README.md). W15 against W13: replay 24-44x faster, decode,
  concurrency and cold prefill within run-to-run noise (cold 8K -2%). Turnkey runs: `scripts/rigmark/`,
  [`docs/RIGMARK.md`](docs/RIGMARK.md).
- **1M-context fixes and OpenAI-style errors** (patch 0490, on). `/v1/models` reports `max_model_len` /
  `context_length`; a prompt over the context gets an OpenAI `context_length_exceeded` 400 (agents such as opencode
  then compact instead of failing); `POST /tokenize` and token-id prompts on `/v1/completions` (RigMark's prefill
  phase needs them); a request within 64 tokens of the 1M limit was refused by the KV pool (4,097 pages needed, 4,096
  held) and now fits; `serve.sh` checks and logs the served context.
- **Memory safety.** W15 found two memory effects and bounded them: the long-prompt dip (MemAvailable fell to
  6.4 GiB for ~35 s during a 314k prompt: torch's caching allocator keeping a new segment every ~4k tokens for a
  growing prefill buffer) and page cache silently serializing concurrent requests during a large file copy. Patch
  0550 fixes both (one pre-grown scratch buffer, an allocator trim, admission that counts reclaimable page cache;
  same bits). **Measured in W17: built into the production image but not adopted (off).** Its memory results held
  (the 314k needle's dip 1.7 instead of 4.0 GiB), but with it on the first long prefill after a burst of short
  requests ran up to 9.8% slower in 2 of 4 prefill pairs, never seen with it off; next step: the same run with only
  its allocator trim off.
  ([`docs/MEMORY-SAFETY.md`](docs/MEMORY-SAFETY.md)).
- **Multi-prompt prefill** (patch 0560, `GLM53_TF_MULTI_PREFILL`): several waiting prompts prefilled in one forward
  (the routed experts read once for all of them), each with the bits it gets alone; estimated C4 first tokens
  ~0.6-0.8 s (then 1.6-1.9 s) and 1.7-2.8x prefill throughput on bursts of short prompts. **Measured in W17 and
  adopted (`GLM53_TF_MULTI_PREFILL=1`)**: same bits everywhere (92/92 grouped replies == the same request alone on the
  real model, batchexact, transcripts, N1, 13/13 glmbench hashes), C4 first tokens 1.64 -> 0.82 s with thinking on,
  C2 -18%, prefill and decode unchanged; on RigMark C4 aggregate 81.8-82.8 -> 91.0 tok/s and C4 per-stream TTFT
  1.63-1.91 -> 0.89 s. Cost: -0.5 GiB on the worker in the 4 x 250k stress.
  ([`docs/MULTI-PREFILL.md`](docs/MULTI-PREFILL.md)).
- **Decode +2.5-3%** (W12): L2 prefetch of the next kernels' weights in every decode path (patch 0460,
  `GLM53_TF_L2PF=1`, 8 MiB a site; 1 stream +2.5%, lone requests +3.2%) and CUDA graphs captured after 8 sightings
  of a 4-slot round (`GLM53_TF_BATCH_CAPTURE_AFTER=8`; 4 streams +2.8%, 81.2 tok/s aggregate); every reply hash
  unchanged.
- **Also:** reasoning is sent in the `reasoning` field only (`GLM53_TF_REASONING_FIELDS=reasoning`, as the vLLM kit
  does; `both` restores `reasoning_content`); the first token of a prompt leaves as soon as its prefill piece ends
  (0540, `GLM53_TF_EMIT_FIRST`); drafter-training tools (0430 records, `train/`, `docs/DRAFTER-TRAINING.md`), full
  MMLU and KL-divergence harnesses (`bench/mmlu_full.py`, `bench/divergence.py`), a long-prompt exactness check
  (`bench/longexact.py`), and analyses: `docs/THEORY-2.md` (the decode round's critical path),
  `docs/UPSTREAM-0362-AUDIT.md`, `docs/SFXNZ-AUDIT.md`, `docs/DRAFTER-SEARCH.md`, `docs/QUALITY-PLAN.md`.

**What didn't work** (every patch stays in the series, off by default; numbers in `docs/PATCHES.md` / `docs/RESULTS.md`):

- **0440 streaming decode kernels** (persistent routed-expert and 4-bit GEMV kernels, same bits): bitwise and race
  tests passed, but the expert kernel reached only 0.74-0.80x of today's in the microbenchmark (capped at ~160 GB/s
  by loads in flight) and the dense kernel cost -3.9% end to end. Off.
- **0450 GPU-resident decode rounds** (the exact sampler on the GPU, glibc's `log` / `exp` ported bit for bit):
  exact in every mode, but 0% (sampler), -2.3% at 1 stream and only +0.5-1.2% at 4 streams (resident rounds). Off.
- **0400 KDA recurrence v2** (+1.0-1.2%, under its +2% bar; the fused variant slower) and **0410 sparse attention v2**
  (+0% end to end although its kernel is 0.82x): off (W10, listed again because they were the last kernel rewrites
  before this round).
- **8-bit non-expert weights (0470)**: built and CPU-tested, not adopted. Estimated -6.5% (output side) to -16%
  (everywhere) decode and +0.8-2.0 GiB a node for about a point of MMLU at most; the routed experts (EXL3 4-bit) and
  the abliteration set the model's quality ([`docs/QUALITY-PLAN.md`](docs/QUALITY-PLAN.md)).
- **0420 trimmed draft vocabulary**: estimated +0.5-0.8% at 1 stream with a list built from real agent traffic;
  the list shipped here is built from public text (the original came from private sessions) and covers fewer reply
  tokens (79% at 16k ids against 90%), so it is off ([`docs/DRAFT-VOCAB.md`](docs/DRAFT-VOCAB.md)).
- **C4 time to first token with thinking on** was 1.6-1.9 s against vLLM's 0.8 s after 0540 alone: its early first
  token helps only without thinking (median 1.46 -> 0.93 s), because the first thinking token carries no visible text.
  0560 (above) brought it to 0.89 s on RigMark, still 0.08 s behind.
- **Rebase onto TensorFold 0.3.6.2: not done.** Its new EXL3 kernels do not reach the GLM path, their two ideas are
  already inside 0440, and the rebase is a multi-day port with no expected speed gain; it is deferred to after
  RigMark ([`docs/UPSTREAM-0362-AUDIT.md`](docs/UPSTREAM-0362-AUDIT.md)). We stay on 0.3.4 + patches.
- **W16, nothing adopted.** The THEORY-2 prototypes on the production stack: 0510 lone-slot graphs (+0.4% at 4
  streams, under its bar) and split verify graphs (1 stream -0.5%; the real verify graph launch is only ~11-14 us), 0520's
  fused hyper-connection kernel (failed its microbenchmark gate: 0.58x at 1 row, so no server load), 0530's HTTP-thread
  pinning with locked clocks (+0.7%, inside the noise band that a trace-only load also showed, and confounded with the
  clock lock). All within +0.4 to +1.1% of the control, i.e. noise. Two probes passed and feed the next build (a
  3.5 MB size limit for 0440's dense kernel; a memory-pipeline design at 232 GB/s).
- **Drafter comparison (W16): no change.** The modal-labs GLM-5.3-Flash DFlash drafter drafts our abliterated target
  worse (tokens a round -8% prose, -9% code, -18% agent; glmbench -2.5%) and was not adopted; incoai's newer
  `bf582e4e` is at parity with the production `7d74cdd` (acceptance within +-0.01, greedy decode -2 to -3% on prose /
  code, +2.8% on agent), not a win, so production keeps `7d74cdd`. Every arm returned the same replies.

## Quickstart

This runs exactly the production config the numbers below come from (`config/prod.env.example`: 4 concurrent
requests, a 1,048,576-token context). AI coding agents: follow [`AGENTS.md`](AGENTS.md), which has the same steps
with checks and fixes.

**Prerequisites** (details in [Requirements](#requirements)): two DGX Sparks cabled CX7 to CX7 with an IP address on
the link on each; Docker with the NVIDIA runtime on both; passwordless `ssh` from the head node to the worker (and
passwordless `sudo -n` on both for the memory gate); the weights in each node's Hugging Face cache (gated repo:
request access first):

```bash
hf download neko-legends/GLM-5.3-Flash-Uncensored-EXL3 --revision 07135ec082f8f11f7a71e4244a4e5167a0f96277   # both nodes
hf download incoai/GLM-5.3-Flash-DFlash2 --revision 7d74cdd881ed7e32c31175984a67823127b66cfe   # optional drafter, CC BY-NC-ND 4.0
```

**On the head node:**

```bash
git clone --recurse-submodules https://github.com/jayleaton/glm53-tensorfold-spark
cd glm53-tensorfold-spark
cp config/prod.env.example config/prod.env
$EDITOR config/prod.env
```

Fill in these four fields and change nothing else:

| Field | Value |
| --- | --- |
| `WORKER_SSH` | ssh target of the worker, e.g. `user@<worker CX7 address>` |
| `HEAD_IP` | the head's IPv4 address on the CX7 link (`ip -br addr show <netdev>`) |
| `HEAD_HF` / `WORKER_HF` | absolute path of each node's `~/.cache/huggingface` |

Check `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` against `ibdev2netdev` (the usual Spark names are preset), and set
`DRAFTER=` empty if you did not download the drafter.

```bash
scripts/serve.sh build       # build the image here, copy it to the worker
scripts/serve.sh preflight   # checks both nodes; fix what it reports
scripts/serve.sh start       # first start: 8+ min (compiles kernels, loads, writes prepared weights); later ~25-40 s
```

`serve.sh` reads `config/prod.env` by default; no `CONFIG=` needed. To run under Podman (rootful) instead of Docker,
set `CONTAINER_RT=podman` either in `config/prod.env` or on the command line: `CONTAINER_RT=podman scripts/serve.sh build`.
The serving pod always needs host networking (NCCL/RoCE own the CX7 NIC), so this is never the rootless/pasta path.
Podman exposes the GPU via CDI (`--device nvidia.com/gpu=all`): generate the spec once with the NVIDIA Container
Toolkit, `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` (see AGENTS.md Docker + NVIDIA runtime row).
Regenerate it after a driver or toolkit upgrade (the spec references the installed driver and can go stale).

Podman differs from Docker in three small ways the rest of the flow already handles:
- the container image ID has no `sha256:` prefix (`podman image inspect --format '{{.Id}}'`), so switching runtimes
  re-runs calibration once (the cache is keyed on the image ID);
- `--log-opt max-size` / `max-file` are honored but write `k8s-file` logs, not Docker's `json-file` (the `LOG_MAX_*`
  knobs still apply; the log driver differs);
- `docker compose` is `podman compose` plus the override file `-f docker/compose.yaml -f docker/compose.podman.yaml`,
  which adds the CDI device path (the base compose file keeps the Docker `deploy.resources` GPU form only).

**Verify:**

```bash
scripts/serve.sh status                 # both containers Up, /v1/models and /health answer
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3", "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}]}'
```

### Image input

On in the production config (`GLM53_TF_VISION=1`). Send OpenAI `image_url` parts, as a `data:` URL or an `http(s)` URL
the head node can fetch (set `GLM53_TF_VISION_FETCH=0` to accept `data:` only; the server downloads URLs from its own
network):

```bash
IMG=$(base64 -w0 screenshot.png)
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3", "max_tokens": 1024,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What does this screenshot show? Quote any numbers."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}}]}]}'
```

Up to 8 images a request (`GLM53_TF_VISION_MAX_IMAGES`), 20 MiB and 64 M pixels each; an image takes up to 8,000
prompt tokens (1,036 for 1024 x 768) and `usage.prompt_tokens` counts them. A bad image is an HTTP 400 before
anything streams. Video is not supported. Details and every knob: [`docs/VISION.md`](docs/VISION.md).

**Clients:** base URL `http://127.0.0.1:8000/v1`, model `GLM-5.3-Flash-EXL3`, any API key. Set the context window
to **1,048,576** and the maximum output to **32,768** (Continue, Cline, Open WebUI and others:
[client settings](docs/TRYING.md#client-settings)). opencode (`~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "glm-tf": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "GLM-5.3-Flash (TensorFold)",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": {
        "GLM-5.3-Flash-EXL3": {
          "name": "GLM-5.3-Flash",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 1048576, "output": 32768 }
        }
      }
    }
  }
}
```

Without `limit.context` opencode never compacts a long session. Thinking arrives in the `reasoning` field
(`GLM53_TF_REASONING_FIELDS=reasoning`, as the vLLM kit sends it); a client that reads only `reasoning_content` needs
`GLM53_TF_REASONING_FIELDS=both`. The API has no authentication and binds to
`127.0.0.1`; put a reverse proxy with auth in front of it before exposing it. Stop with `scripts/serve.sh stop`.
Long prompts refused or cut short: [Context smaller than expected](docs/TRYING.md#10-context-smaller-than-expected).

## Contents

- [What's new (W20)](#whats-new-w20) (tool calling, RigMark 3-run)
- [What's new (W19)](#whats-new-w19)
- [What's new (2026-09-30)](#whats-new-2026-09-30)
- [Quickstart](#quickstart) ([image input](#image-input))
- [RigMark](#rigmark)
- [Benchmarks](#benchmarks)
- [Real-agent use](#real-agent-use)
- [Requirements](#requirements)
- [Ways to run it](#ways-to-run-it)
- [Knobs](#knobs)
- [Limits and negatives](#limits-and-negatives)
- [Tests](#tests)
- [Layout](#layout)
- [Licensing](#licensing)
- [Credits](#credits)

## RigMark

[RigMark](https://github.com/alexellis/rigmark) standard suite, unmodified settings, `reasoning_effort` low
(receipts and details in [`results/rigmark/`](results/rigmark/README.md)). These are RigMark numbers only; our own
suite is under [Benchmarks](#benchmarks). Current: W20 production (image b11), mean of 3 runs' medians:

| RigMark (mean of 3 runs' medians) | **TensorFold W20** (3 runs) | TensorFold W17 (3 runs) | vLLM TP2 k=7 (Alex Ellis, published) |
|---|---:|---:|---:|
| Code decode tok/s | **72.4** (72.2-72.6) | 67.9 | 44.0 |
| Prose decode tok/s | **45.7** (45.3-45.8) | 43.0 | 18.9 |
| Structured decode tok/s | **95.6** (95.6-95.7) | 88.8 | 64.9 |
| C1 / C2 / C4 aggregate tok/s | **57.5 / 76.0 / 95.1** | 53.1 / 70.1 / 91.0 | 31.6 / 42.0 / 66.1 |
| Cold prefill 8K / 32K / 64K tok/s | 1,610 / 1,684 / 1,667 | 1,560 / 1,634 / 1,620 | **1,813 / 1,908 / 1,922** |
| Immediate replay 64K tok/s (identical prompt resent) | **248,359** | 243,980 | 11,364 |
| Replay TTFT 8K / 32K / 64K s | **0.21 / 0.23 / 0.26** | 0.22 / 0.25 / 0.27 | 4.52 / 2.97 / 5.77 |
| C1 / C2 / C4 per-stream TTFT s | **0.47 / 0.61** / 0.84 | 0.49 / 0.65 / 0.89 | 0.60 / 0.68 / **0.81** |

Earlier runs (W13 baseline on image b5, two W15 runs on image b7) are in `results/rigmark/README.md`. The replay rows
are prefix caching of an identical prompt (byte-identical output), not prefill speed; how it was verified:
[What's new (2026-09-30)](#whats-new-2026-09-30). Different weights and drafter policy than Alex's runs (abliterated
EXL3 4-bit here, NVFP4 there); see the notes in [`results/rigmark/`](results/rigmark/README.md).

## Benchmarks

Our own suite (`bench/glmbench.py`), not RigMark. All our numbers are on the **abliterated** checkpoint
`neko-legends/GLM-5.3-Flash-Uncensored-EXL3` @ `07135ec0`, on one pair of DGX Sparks (GB10, TP=2), 2026-09-27 to
2026-09-30. The decode table is W20 production (3 rounds); the other tables are from test window W10
(2026-09-29 04:20); since then W12 added +2.5% decode at 1 stream and +2.8% at 4 streams (81.2 tok/s aggregate), W15
image input and warm replay with every text gate unchanged (reply hashes equal, prefill 24.5k / 98k ~1,605-1,610
tok/s), W17 multi-slot prefill (4 streams 82.8 tok/s aggregate, mean of 3 runs; same reply hashes), and W19 decode +3.3% at 1
stream / +3.0% at 4 streams (84.6 tok/s aggregate, mean of 6), prefill +1.5% (~1,630 tok/s at 24.5k and 98k) with every
reply hash unchanged ([What's new (W19)](#whats-new-w19)), and W20 decode +4.8% / +4.3% from host-side tuning outside this repo (the decode
table below has W20's numbers). Full tables and methodology: [`docs/RESULTS.md`](docs/RESULTS.md) (sections W6-W20 for
the current production config); raw JSON and the window scripts in [`results/`](results/).

### (a) Ours vs the vLLM production kit, same weights, same client

The vLLM column is [Reederey87's kit](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) @ `8e443d6` (a fork
of [MiaAI-Lab's kit](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)) as we ran it in production on
this pair: 1M context, FP8 KV, DFlash2 k=7 with adaptive k, fused EXL3 MoE kernels, **the same abliterated weights**.
Both stacks were measured with the same client (`bench/glmbench.py`), one stack at a time, nothing else on the GPUs.

Two TensorFold configurations:

| Config | File | What |
| --- | --- | --- |
| **Production** (4 requests, shared 1M-token pool) | [`config/prod.env.example`](config/prod.env.example) | 4 concurrent requests sharing one **1,048,576-token FP8 latent KV pool** (each request up to 1M tokens while the pool has room), sessions inside the batch slots plus an NVMe session tier, shared system-prompt reuse, row-split prefill in 4,096-row chunks for a lone request, the RoCE all-gather, 16-row verify windows |
| Single-stream (earlier, 2026-09-28) | [`config/prod-single.env.example`](config/prod-single.env.example) | one request at a time, 524,288-token context, bf16 latent KV, fast/lean prefill in 8192-row chunks, session store |

**Decode**, tok/s, single stream, thinking off (decode excludes prefill and time to first token). "Greedy" = T=0,
"sampled" = T=1 (seeded). TF production = W20 (image b11, `config/prod.env.example`): 3 rounds, each the
median of 3 reps, mean of the rounds (`results/W20/final-glmbench/`); every greedy cell returned one reply hash across
all 9 reps. TF W10 = the earlier production column (load R16, median of 3); vLLM kit and single-stream: median of 5.

| Cell | vLLM kit | **TF production (W20)** | vs vLLM | TF W10 | TF single-stream (09-28) |
| --- | ---: | ---: | ---: | ---: | ---: |
| tf code, sampled (T=1), 64 tok | 35.2 | **51.1** | 1.45x | 42.3 | 40.5 |
| tf chat, sampled (T=1), 64 tok | 27.1 | **48.6** | 1.79x | 41.0 | 37.5 |
| tf code, greedy (T=0), 64 tok | 41.9 | **89.6** | 2.14x | 77.6 | 65.8 |
| tf chat, greedy (T=0), 64 tok | 22.8 | **51.6** | 2.26x | 44.6 | 44.7 |
| kit hashmap (prose), greedy, 200 tok | 30.0 | **59.6** | 1.99x | 53.2 | 50.4 |
| kit structured, greedy, 200 tok | 72.7 | **112.3** | 1.54x | 100.6 | 98.1 |
| kit essay, greedy, 200 tok | 26.1 | **50.5** | 1.93x | 44.9 | 44.2 |
| tweet sequence, greedy, 512 tok | 67.5 | **105.1** | 1.56x | 93.0 | 93.3 |
| tweet code, greedy, 512 tok | 42.4 | **75.9** | 1.79x | 66.7 | 62.3 |
| tweet json, greedy, 512 tok | 50.4 | **84.0** | 1.67x | 74.8 | 71.7 |
| edit (rename / comments / print-to-log), greedy, 1024 tok | not run | **124.1 / 108.0 / 126.5** | - | 111.2 / 94.7 / 115.5 | 81.8 / 78.2 / 82.6 |

The edit cells (the model rewrites a file it was given) benefit from prompt-lookup drafts (patch 0020) and, since
W10, from verify windows of up to 16 rows (patch 0380: +16-27% on these cells, every reply hash unchanged).

**Prefill and time to first token** (cold, unique prompt, no cache hit; kernels already compiled):

| Prompt | vLLM kit | **TF production** (alone) | TF single-stream (09-28) |
| --- | ---: | ---: | ---: |
| ~7k tokens | 1,340 tok/s | - | 1,162 tok/s |
| ~24.5k-28k tokens | 1,448 tok/s at 28k (TTFT 19.4 s) | **~1,607 tok/s** at 24.5k (1,614 / 1,602; TTFT ~13.4 s); W19: **1,630-1,632** | 1,266 tok/s at 28k (TTFT 22.2 s) |
| ~98k-112k tokens | - | **~1,600 tok/s** at 98k (1,577-1,606); W19: **1,624-1,630** | 1,238 tok/s at 112k (TTFT 90.5 s) |
| 314k tokens (needle, after the stress run) | - | 1,376 tok/s, found (W19: 1,396) | - |
| TF vs vLLM at ~28k | | **~1.11x** (24.5k vs vLLM's 28k; W19 ~1.13x) | 0.87x |

**Multi-turn, concurrency, boot, quality**:

| | vLLM kit | **TF production** | TF single-stream (09-28) |
| --- | --- | --- | --- |
| context | 1M in one context | **4 concurrent requests sharing a 1,048,576-token KV pool, each request up to 1M** | 1 x 524k |
| concurrent streams 1 / 4, aggregate tok/s (median of 5) | batches up to 4; Reederey87 publishes 63.4-66.3 warm at 4 in flight | 52.1 / **78.1** (5 mixed prompts; 4-stream reps 70-80 across windows) | one at a time (queued) |
| switch back to a stored ~39k-token session | - | **0.42-0.45 s** from the NVMe session tier instead of a 31 s cold prefill, also after a server restart | 0.4-1.0 s (RAM store) |
| resume a 314k-token conversation | - | 0.19 s (314,240 tokens cached) | - |
| 4 new sessions at once over one ~18k-token system prompt (subagent burst) | - | **28.2 s** wall instead of 72.4 s (shared-prefix reuse, patch 0310) | - |
| decoders during a long prefill | - | longest decode gap 3.9 s (4 x ~250k stress) | queued behind it |
| restart to ready (prepared weight folders, patch 0140) | - | **22-23 s** | 34-37 s (490 s from the raw checkpoint) |
| drafted == serial, byte-identical (10 cases) | n/a | 10/10; batched == alone 4/4 | 10/10 |
| MMLU-200 (greedy, thinking off) / refusals (10 prompts) | - / 0/10 | **88.0%** / 0/10 | 89.5% / 0/10 |
| needle retrieval | - | found at 314k (cold and resumed); earlier at 358k | 10/10 at 28k |
| memory stress: 4 conversations grown to ~250k each, then a 32k turn beside 3 decoders | - | no OOM, no request errors; worst MemAvailable 10.49 / 9.36 GiB (head / worker) | - |
| RoCE all-gather instead of NCCL (patch 0230/0350, W9 A/B) | - | decode +10.8% (1 stream) / +4.3% (4 streams) median, transcripts byte-identical | - |

"Drafted == serial" means every drafted reply is the same bytes as the one-token-a-round reply of the same engine
and weights. "Batched == alone": a request served next to 3 others returns the same bytes as served alone. The
TF production column is W10's final config (`FIN`) unless the row says otherwise; the session-tier row is W4 and the
system-prompt row W8 (both features unchanged since).

### (b) MiaAI-Lab's and Reederey87's published numbers (different weights and settings)

These are the kits' own published figures, copied from their READMEs on 2026-09-28. They use the **base** (not
abliterated) weights `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (MiaAI-Lab serves the byte-identical mirror
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`), their own clients (their dashboard, `tests/bench_decode.py`) and their own
prompts, so they are **not** directly comparable to table (a).

[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks):

| Date | Settings | Result |
| --- | --- | --- |
| 2026-09-07 | cold prefill, E3 grouped MoE; 900k context, util 0.86, MNBT 7168, DFlash2 k=7, 4 seqs, thinking off | 1,492 / 1,554 / 1,428 / 1,587 / 1,562 / 1,517 tok/s at ~8k / 16k / 32k / 64k / 128k / 256k (TTFT 32k: 23.0 s, 128k: 84.0 s) |
| 2026-08-28 | decode, structured + code prompts, DFlash2 k=7, temp 0, thinking off, 400 tok, 1M context | x1: 62.9 tok/s (TTFT 719 ms); x2: 51.7 a stream, 103.3 aggregate; x4: 37.1 a stream, 146.5 aggregate |
| 2026-09-17 | decode, prose, adaptive k (EMA) + dense FP8 + cooperative MoE, 850k context | x1: 36.1 a stream; x4: 19.4 a stream, 75.3 aggregate |
| 2026-09-21 | custom qualification run (their "OFF" arm, 850k) | structured 78.6, code 53.5, prose 33.2 tok/s; cold 32k TTFT 27.7 s, cold 100.7k TTFT 85.6 s |

[Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) (production
stack of 2026-09-20, 1M context):

| Metric | Published |
| --- | --- |
| prose decode (hashmap) / hard essay | ~33 tok/s (median 33.03) / ~26 tok/s (median 25.92) |
| structured decode | ~74 tok/s (median 74.19, acceptance 1.0) |
| cold prefill (2026-09-09 stack) | ~1,454 tok/s at 60k, ~1,408 tok/s at 240k |
| 4 in-flight, warm aggregate | 63.4-66.3 tok/s |

How to read the two tables together:

- **Our numbers are with an abliterated model.** The abliterated checkpoint keeps attention, the shared expert, the
  dense layers and the head in BF16; we re-quantize those to 4 bits at load (`q4mse`, patch 0001), which costs a
  little accuracy (13 of 200 MMLU answers change) and still reads more than a natively 4-bit layout. The MTP head
  and the DFlash2 drafter were trained against the base model, so draft acceptance on abliterated weights is likely
  lower. We expect base GLM-5.3-Flash weights (for example the MLX 4-bit checkpoint TensorFold's own recipe uses) to
  net further improvements; that is an expectation, not a measurement.
- On our pair, the vLLM kit on the abliterated weights measured hashmap 30.0 / structured 72.7 / essay 26.1 tok/s,
  close to Reederey87's published 33 / 74 / 26 on base weights.
- Cross-kit comparisons also differ in context length, KV format, drafter settings, prompts, clients and dates.

## Real-agent use

This has been used in [opencode](https://opencode.ai) for real agent workflows (multi-file edits, tool calls, long
sessions) and performed well with thinking at **high** reasoning effort, the default in the production configs
(`GLM53_TF_DEFAULT_EFFORT=high`).

| Check | Result |
| --- | --- |
| opencode tool-call harness (`bench/toolcall_harness.py`: opencode's tool set, 21 cases x 10 runs, streamed, T 0, thinking off) | **200/210 passed, 0 corrupted** calls (same with FP8 prefill on or off) |
| same harness, thinking on, single-stream production load (21 x 5) | 95/105 passed, 0 corrupted |
| same harness through the HTTPS reverse proxy in front of the API (21 x 2) | 38/42 passed, 0 corrupted |
| real-model API checks after each switch | `/v1/models`; a thinking reply returns `reasoning_content` + the answer (finish `stop`); a tool call returns `get_weather({"city":"Paris"})` with finish `tool_calls`; streamed == non-streamed; `stop` strings streamed and not |

"Corrupted" means a tool call with leaked GLM markup (`<arg_key>`, `<tool_call>`, `</think>`) or unparseable
arguments. Every failure is a case where the model made a different, reasonable call than the one the case
expects: `edit_file` reads the file before editing it (a `read` call where the case expects `edit`), and with
thinking on `multi_turn_chain` takes another step first.

The opencode provider entry is in the [Quickstart](#quickstart). Set `limit.context` to the server's `CONTEXT`
(1048576 for the production config, 524288 for single-stream): without it opencode never compacts a long session,
and it sends `max_tokens` = `limit.output` (capped at 32,000), which counts against the context. In the
production config the four requests share one 1,048,576-token pool: one request can use all of it, but four long
ones together wait for pages or spill idle sessions to the store.

## Requirements

- Two DGX Sparks (GB10, 128 GB unified memory each) connected by a QSFP cable between their ConnectX-7 ports, with
  the link configured (an IP address on one CX7 netdev per node; the RDMA device visible in `ibv_devices`).
- Docker with the NVIDIA Container Toolkit on both nodes (stock DGX OS has both). Or Podman (rootful) with the
  NVIDIA CDI / container-toolkit wrapper: set `CONTAINER_RT=podman` in the config (see below). Not rootless: the
  serving pod needs `--network host` so NCCL / the RoCE proxy own the CX7 NIC.
- Passwordless `ssh` from the head node to the worker, as a user that can run `docker` there. The production
  config's memory gate also drops page caches with `sudo -n` on both nodes.
- The weights in each node's Hugging Face cache, same revision on both (the repo is gated: request access on its
  model card first):

  ```bash
  hf download neko-legends/GLM-5.3-Flash-Uncensored-EXL3 --revision 07135ec082f8f11f7a71e4244a4e5167a0f96277
  ```

- Optional: the DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2` (revision `7d74cdd`), in both caches. It is
  **CC BY-NC-ND 4.0 (non-commercial only)**; this repo never ships it. Without it, drafting uses the checkpoint's own
  MTP layer.
- Disk: the checkpoint, plus ~83 GB a node for the prepared weight folder (fast restarts) and up to 64 GiB a node
  for the NVMe session tier (`GLM53_TF_SESSION_DISK_GIB`).
- Network access at build time to pull `nvcr.io/nvidia/pytorch:26.07-py3`. At run time the container is offline.

## Ways to run it

The production config is the default and the one to run. [`docs/TRYING.md`](docs/TRYING.md) also covers:
single-stream long context (524k), 4 x 256k batch (FP8 KV), the 32k debugging baseline (`config/minimal.env.example`),
per-request knob A/B (`tf_knobs`), draft policies (`model@policy`), reasoning effort, sessions, fast boot, how to
benchmark (`glmbench`, `multiturn`, `quality`, `toolcall_harness`), how to check exactness, how to roll back, what to
check when the context is smaller than expected, and client settings (opencode, Continue, Cline, Open WebUI).

## Knobs

Every engine change is a patch in [`patches/`](patches/) with its own `GLM53_TF_*` knob, off by default (upstream
behaviour) unless stated. [`docs/PATCHES.md`](docs/PATCHES.md) documents each knob and why each patch keeps the
output exact; [`docs/CHANGES-SUMMARY.md`](docs/CHANGES-SUMMARY.md) lists every patch with its measured gain and
status (on / opt-in / rejected), ordered by impact.

Main groups:

| Area | Patches | Main knobs |
| --- | --- | --- |
| Weights | 0001 | `GLM53_TF_NONEXPERT=q4mse` |
| Drafting | 0010, 0020, 0070, 0071 | `GLM53_TF_AUTO_FDRAFTS=7`, `GLM53_TF_LOOKUP=1`, `GLM53_TF_CALIB`, `GLM53_TF_DEPTH=cost` |
| Drafting / verify | 0380 | `GLM53_TF_MAX_DRAFT_ROWS=16` (verify windows of up to 16 rows) |
| Long context / KV | 0050, 0060, 0065, 0220, 0290 | `GLM53_TF_LATENT_KV=1`, `GLM53_TF_KV_DTYPE=bf16\|fp8`, `GLM53_TF_KV_POOL_TOKENS=1048576` |
| Prefill | 0003-0006, 0080-0085, 0170, 0190, 0320, 0335, 0360, 0390 | `GLM53_TF_FAST_PREFILL=1`, `GLM53_TF_LEAN_PREFILL=1`, `GLM53_TF_PREFILL_ROWS=auto`, `GLM53_TF_FAST_EXPERTS=fat`, `GLM53_TF_MOE_GLUE=5`, `GLM53_TF_ATTN_BM32=1`, `GLM53_TF_MTP_PREFILL_CACHE=1`, `GLM53_TF_PREFILL_PP=1`, `GLM53_TF_SOLO_PIECE=4096`, `GLM53_TF_B12X=4`, `GLM53_TF_MLA_EXPAND=v2` |
| Sessions / batching | 0110, 0120, 0180, 0200, 0250, 0310 | `GLM53_TF_SESSION_GIB`, `GLM53_TF_BATCH=4`, `GLM53_TF_BATCH_SESSIONS=1`, `GLM53_TF_SESSION_DISK=/sessions`, `GLM53_TF_PREFIX_SHARE=1` |
| Communication / decode host work | 0230, 0350, 0370, 0460 | `GLM53_TF_COMM_BACKEND=roce` (NCCL fallback + failure marker), `GLM53_TF_DECODE_OVERLAP=1`, `GLM53_TF_L2PF=1` + `GLM53_TF_L2PF_MB=8` (L2 weight prefetch in decode), `GLM53_TF_BATCH_CAPTURE_AFTER=8` |
| Image input | 0500 | `GLM53_TF_VISION=1`, `GLM53_TF_VISION_CACHE_MB=64`, `GLM53_TF_VISION_PREP_MB=64` (`_FETCH`, `_MAX_IMAGES`, `_MAX_BYTES`, `_MAX_PIXELS`, `_MAX_TOKENS`: docs/VISION.md) |
| Replay / first token | 0540 | `GLM53_TF_SNAPSHOT_BEFORE_END=1`, `GLM53_TF_EMIT_FIRST=1` (both default on) |
| API / context | 0490, 0160 | on by default: `max_model_len` in `/v1/models`, `context_length_exceeded` 400s, `POST /tokenize`, token-id prompts; `GLM53_TF_REASONING_FIELDS=reasoning\|reasoning_content\|both` |
| Per request | 0090-0093 | `"tf_knobs": {...}` in the request body |
| Serving / ops | 0002, 0140, 0150, 0160, 0210, 0300 | prepared folders, `/health`, `/metrics`, `reasoning_effort`, `stop`, prompt-token cache, request log (`GLM53_TF_REQUEST_LOG`, no text) |
| Multi-slot prefill | 0560 | `GLM53_TF_MULTI_PREFILL=1` (a round's prefill pieces in one forward; W17) |
| Decode expert loads / NCCL / memory (W19) | 0580, 0550, 0530 | `GLM53_TF_DEC_EXPERT_LOADS=1` + `_CFG=nc,8,1`, `GLM53_TF_SELECT_SCRATCH=grow`, `GLM53_TF_CPU_PIN=http`; NCCL on both CX7 functions with 4 channels (`NCCL_IB_HCA`, `NCCL_PASSTHROUGH=1`, `NCCL_MIN/MAX_NCHANNELS=4`) |
| Structured output | 0610 | `GLM53_TF_GRAMMAR=1` (`response_format`, strict / required tool calls; exact under drafting and batching) |
| Upstream ports | 0600 | on by default: `GLM53_TF_DISCONNECT=1` (a departed client frees its slot), request 400 hardening, image URL hardening (`GLM53_TF_VISION_FETCH_*`), `/health` token totals, USR1 stacks |
| Measured, not adopted (off) | 0240 (bits 1-2), 0260, 0270, 0280, 0330, 0340, 0400, 0410, 0440, 0450, 0370's `GLM53_TF_CPU_PIN`, 0460's RoCE latency knobs, 0510 / 0520 (W16), 0550's trim and page-cache admission (`GLM53_TF_ALLOC_TRIM_GB=0`, `GLM53_TF_ADMIT_MEM=free`), 0570 and 0590 (W19) | see [Limits](#limits-and-negatives) |
| Offline / tools, off | 0420 (draft vocabulary), 0430 (drafter-training records), 0470 (8-bit non-experts) | `docs/PATCHES.md` |

## Limits and negatives

| | TensorFold + patches (production) | vLLM kit |
| --- | --- | --- |
| Single-stream decode (our suite) | 1.45-2.26x faster on every measured cell (W20; sampled cells 1.45 / 1.79x, greedy 1.54-2.26x) | baseline |
| Prefill | ~1,659 tok/s at 24.5k and ~1,654 at 98k (W20; W19 ~1,631 / ~1,628), against 1,448 measured for the kit at 28k (~1.15x; not the same prompt length). On RigMark's cold 8K-64K prompts we are **0.87-0.89x** of Alex's vLLM receipt. MiaAI-Lab publishes 1,492-1,587 on base weights (table b) | measured 1,340-1,448 here |
| 4 concurrent streams | ~78 tok/s aggregate in W10 (median; 70-80 across runs); 84.6 in W19, 88.6 in W20 (mean of 6, 5 mixed prompts); RigMark C4 95.1 vs vLLM 66.1, but C4 first token 0.84 vs 0.81 s | Reederey87 publishes 63-66 warm; **MiaAI-Lab publishes 146.5 aggregate on 4-stream structured output** (base weights, their client), higher than anything we measured at 4 streams (`docs/RESEARCH-NIGHT.md` §5) |
| Context | 4 requests share one 1,048,576-token pool: a request can grow to 1M, but not four at once (admission waits or spills idle sessions to the store) | 850k-1M in one context |
| KV precision | **FP8** latent KV: greedy replies diverge from bf16 KV within the first 0-78 tokens on 15 of 20 prompts (quality checks above held; long-session recall checked by needle at 314k-358k only) | FP8 KV too |
| Memory margin | **W20: stress-test memory minimum now ~10.7 GiB** (10.6-10.8 GiB). W19: after W17's heavy warm-up (~200 short requests before the gates) the 4 x 250k stress bottomed at **8.34 / 8.09 GiB** MemAvailable (head / worker; W17's config in the same sequence 7.75 / 7.61) and a 314k needle right after the stress and MMLU at **8.68 / 8.51 GiB** (6.58 / 6.31), no OOM and no engine error. The margin is thin on the worker in the stress (8.09 GiB, ~0.1 GiB over the gate). A lone prompt far beyond 314k no longer grows the allocator's prefill key blocks (0550's scratch, docs/MEMORY-SAFETY.md), but a ~900k prompt beside 3 busy slots has not been tested. A large file copy on the head node can serialize concurrent requests (page cache; `serve.sh start` drops the page cache and checks the slot count; 0550's page-cache admission is built in, off, untested on the GPU). 8,192-row prefill chunks (+~4% prefill) were rejected for memory before W19 and have not been re-tested with the new headroom | - |
| API | no `logprobs`, `n > 1` rejected, images yes (0500) but no video, structured output yes (0610); in single-stream mode a `stop` match ends the reply but the engine keeps decoding silently to EOS / `max_tokens` before the next queued request starts | full OpenAI surface of vLLM |
| Maturity | **work in progress**: one pair of Sparks, one checkpoint, four days of measurements (W1-W20) | production kits with many contributors |

Other negatives and trade-offs, measured:

- **FP8 prefill (0083) was rejected**: +7-11% prefill, but greedy replies diverged from bf16 prefill on 25 of 30
  prompts; off.
- **Decode-step kernels (0130) regressed** on the real model (tf code greedy 64.3 -> 62.4 / 56.4 tok/s); off.
- **`hc_fused` (0190)** is bit-exact but 7-8x slower on GB10 (shared-memory limits); off. `mtp_window` (0190) cost
  decode after long prompts; off. (`attn_bm32` is on in production since W1: +5% prefill, same bits.)
- **Patches measured and not adopted** (they stay in the tree, off; `docs/PATCHES.md` and `docs/RESULTS.md` have the
  numbers): 0240 b12x bits 1-2 (slower than today's kernels; only bit 4 is used, via 0360), 0260 `once` expert
  kernel (never beats `fat`), 0270 `FAST_EXPERTS=auto` (-2% end to end despite faster isolated kernels), 0280
  batch round buckets (-5% at 4 streams), 0330 warp-specialized `tc` expert kernels (cfg 1/2 cannot launch on GB10,
  cfg 3 -4%), 0340 per-slot drafter choice (simulated -0.4%), 0400 KDA recurrence v2 (+1.0-1.2%, under its bar),
  0410 sparse attention v2 (+0%), 0370's CPU pinning (+0.4%), and 8,192-row lone chunks (memory, above).
- **The single-stream config died of unified-memory OOM** once, with a 12 GiB session store filling under agent
  traffic at 524k context. Its example config uses 6 GiB; the watchdog (`scripts/systemd/`) restarts a dead pair.
- `q4mse` is a re-quantization of the BF16 non-expert weights: not bit-identical to BF16 (13 of 200 MMLU answers
  differ; accuracy 87.0% -> 88.0%); exactness (drafted == serial) holds within each mode.
- The RoCE all-gather (0230/0350) is limited to 256 KiB a message: one unexplained 2 MiB mismatch was seen once in a
  harness run (above that limit). A run-time RoCE failure writes a marker and the next start uses NCCL.
- Only this checkpoint and this two-node topology have been tested.

## Tests

The patch tests run in the image on one GPU with TensorFold's synthetic checkpoint (no real weights):

```bash
docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $PWD:/work --entrypoint bash glm53-tensorfold:dev \
    -c "bash /work/scripts/run_tests_in_image.sh /work/results/tests -- tests/cuda/test_patches.py tests/test_glm_tool_calls.py"
```

Host-only (no GPU, no Docker): `python -m pytest -q tests/test_serve_ops.py tests/test_gpuwatch.py` (launcher, canary,
Xid parser, GPU clock watch); the image-input front end (`tests/test_vision_prep.py`, `tests/test_vision_server.py`)
and the API context checks (`tests/test_api_context.py`) against a patched tree; 0620's tool-calling fixes (`tests/test_tool_fixes.py`, `PYTHONPATH=<tree>/src`); the other `tests/test_*.py` run against a patched tree (`PYTHONPATH=<tree>/src`), several
of them in Triton's CPU interpreter. Against a
running server: `python3 bench/glmbench.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --suites exact`
checks drafted == serial on the real model. Known GPU-test failures are listed in `docs/RESULTS.md` (for example the
engine-level FP8 KV tests do not run on the synthetic model).

Before publishing a fork: `scripts/check-public.sh` scans the tree for private IPs, hostnames, keys and tokens.

## Layout

| Path | What |
| --- | --- |
| `vendor/TensorFold` | TensorFold, pinned submodule (`2f8e514`, 0.3.4), unmodified |
| `patches/` | engine patches, applied in order at image build |
| `docker/` | Dockerfile, entrypoint, compose files (`compose.yaml` + `compose.podman.yaml` override) |
| `scripts/` | `serve.sh` (build / start / stop / status / logs / canary / watchdog / gpucheck; `PATCHES="..." serve.sh build` for a subset), `prepare.sh`, `gpuwatch.py` (GB10 clock / slow-state watch), `traffic-report.py` (request-log summary), `rigmark/` (turnkey RigMark runs, docs/RIGMARK.md), `tooleval/` (tool-calling benchmark runner, docs/TOOL-CALLING.md), systemd units, `check-public.sh` |
| `config/` | `prod.env.example` (production, the default), earlier configs, `minimal.env.example` (32k debugging baseline) |
| `AGENTS.md` | step-by-step setup for AI coding agents: checks, commands, expected logs, failures and fixes |
| `bench/` | benchmark clients, MMLU-200 subset and full MMLU (`mmlu_full.py`), KL divergence (`divergence.py`), long-prompt exactness (`longexact.py`), tool-call harness, shared-prefix bench, draft-policy and lookup simulators, draft-vocabulary study and public ranking (`draftvocab.py`, `draftvocab_public.py`), page-cache probe |
| `train/` | drafter training (MTP head and DFlash2 distillation from 0430 records, a PyTorch reference of both, offline acceptance evaluation; docs/DRAFTER-TRAINING.md) |
| `tests/` | patch tests (GPU) and launcher tests (host) |
| `results/` | raw benchmark JSON and the test windows' scripts (W1-W20), the release run (`FINAL-20260930/`), RigMark receipts (`rigmark/`), tool-calling runs (`tooleval/`), synthetic test images (`W14/img/`); logs omitted |
| `docs/` | results, patch notes, design and analysis notes |

## Licensing

| Part | License |
| --- | --- |
| This project's code, patches, scripts, benchmarks and docs | **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Redistributions, modified or not, must keep the copyright line and the NOTICE attributions and state their changes. |
| TensorFold (`vendor/TensorFold`) | MIT, Copyright (c) 2026 TensorFold contributors; unmodified submodule, the patches are applied at build time. The TensorFold code the patches modify stays under its MIT License. Its third-party notices: `vendor/TensorFold/THIRD_PARTY_NOTICES.md`. |
| RoCE all-gather in `patches/0230`, fast-prefill kernels in `patches/0240` | adapted from / re-implementing [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0, Luke Alonso and the b12x contributors); details in [`NOTICE`](NOTICE). |
| Fat-expert MoE kernel structure in `patches/0170` | adapted from the Apache-2.0 [Reederey87 kit](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) (code MiaAI-Lab contributed under MIT before 2026-09-07); its NOTICE is reproduced in [`NOTICE`](NOTICE). The BF16 KDA copy in the same patch re-implements an idea from MiaAI-Lab PR #233 without its code. |
| Ported upstream code in `patches/0600`, `patches/0610` | from later TensorFold releases (0.3.6.2, 0.5.0), MIT, Copyright (c) 2026 TensorFold contributors; each ported piece names its source commit ([`NOTICE`](NOTICE), [`docs/UPSTREAM-PORTS.md`](docs/UPSTREAM-PORTS.md)). |
| xgrammar (structured output, `patches/0610`) | Apache-2.0 ([mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)), installed into the image by pip, not vendored. |
| Docker base image | NVIDIA Deep Learning Container License (`nvcr.io/nvidia/pytorch:26.07-py3`) |
| Model weights (not included) | `neko-legends/GLM-5.3-Flash-Uncensored-EXL3`: ShapleyMCG License 1.0 per its model card (the Local Inference Lab Attribution License 1.0: MIT-like with a required attribution, given at the top of this README and in NOTICE). Its sources `orcarouter/GLM-5.3-Flash-Uncensored-FP8` and `zai-org/GLM-5.3-Flash` are MIT per their model cards. The weights are abliterated (refusals removed); you are responsible for how you use them. |
| DFlash2 drafter (not included) | `incoai/GLM-5.3-Flash-DFlash2`: **CC BY-NC-ND 4.0, non-commercial only**. Never bundled; download it yourself, or run without it (MTP drafts only). |

## Credits

- [Ash Hart / TensorFold](https://github.com/ashhart/TensorFold): the engine, kernels, drafting and server this
  project patches.
- [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) and its contributors: the GLM-5.3-Flash
  2x DGX Spark vLLM kit, the fat-expert MoE design, and the ops ideas listed in `docs/MIA-AUDIT.md`.
- [Reederey87](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark): the production vLLM kit we measured
  against and the Apache-2.0 kernel code `patches/0170` adapts.
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (Luke Alonso and contributors): the RoCE
  one-shot all-gather `patches/0230` ports and the kernel designs `patches/0240` / `0360` re-implement.
- [0xSero](https://huggingface.co/0xSero): GLM-5.3-Flash EXL3 builds and DGX Spark recipes.
- [neko-legends](https://huggingface.co/neko-legends) (abliterated EXL3 weights, under Local Inference Lab's
  ShapleyMCG license) and [orcarouter](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) (the
  uncensored FP8 source).
- [brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw): the TR3 4-bit EXL3 weights the other
  kits publish their numbers on.
- [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3): the EXL3 format.
- [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2): the DFlash2 drafter.
- [kindlingai](https://github.com/kindlingai/glm-5.3-flash-gx10): the GX10 vLLM recipe behind the two-function NCCL
  setting (W19) and the pipelined prefill-expert kernel idea (0590); ideas only, no code copied (their repository has
  no licence; [`docs/KINDLING-AUDIT.md`](docs/KINDLING-AUDIT.md)).
- [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) (Apache-2.0): the grammar engine behind structured output
  (`patches/0610`).
- [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash): GLM-5.3-Flash.
- [Vontra](https://huggingface.co/Vontra): the MLX checkpoints TensorFold's GLM recipe uses.
- NVIDIA: the PyTorch container (`nvcr.io/nvidia/pytorch`).
- The [vLLM project](https://github.com/vllm-project/vllm).
- [Alex Ellis / RigMark](https://github.com/alexellis/rigmark): the agent-workload benchmark and the published vLLM
  GLM-5.3 receipts we compare against.
- [Hugging Face transformers](https://github.com/huggingface/transformers): the `Glm5Next` image processor and vision
  model that patch 0500's preprocessing and tower are checked against.
- The public text behind the shipped draft-vocabulary ranking (`patches/0420`): OpenAssistant oasst1, SWE-Gym, CPython,
  denoland/std and ripgrep (licenses in [`NOTICE`](NOTICE)).
