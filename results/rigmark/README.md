# RigMark: TensorFold on 2x DGX Spark

Four sets of receipts, newest first:

0. **W20 run: 3 rounds on the W20 production config** (2026-09-30, image b11 = b10's patches + 0620 with
   `GLM53_TF_TOOL_FIXES=all`): [`tensorfold-20260930-w20-final-r1/`](tensorfold-20260930-w20-final-r1/),
   [`-r2/`](tensorfold-20260930-w20-final-r2/), [`-r3/`](tensorfold-20260930-w20-final-r3/). See [below](#w20-run-3-rounds-averaged).
1. **W17 release run: the 3-round averaged run on the release config** (2026-09-30, image b9 = patches 0001-0490 +
   0500 + 0540 + 0550 + 0560, `config/prod.env.example` of this release): [`tensorfold-20260930-w17-final-r1/`](tensorfold-20260930-w17-final-r1/),
   [`-r2/`](tensorfold-20260930-w17-final-r2/), [`-r3/`](tensorfold-20260930-w17-final-r3/). See [below](#release-run-3-rounds-averaged).
2. **W15 validation (2026-09-29, image b7 = patches 0001-0490 + 0500 + 0540, the previous production config):** two full standard-suite runs, [`tensorfold-20260929-w15/`](tensorfold-20260929-w15/) and
   [`tensorfold-20260929-w15-run2/`](tensorfold-20260929-w15-run2/). See [below](#w15-validation-runs-2026-09-29).
3. **The W13 baseline (2026-09-29, image b5 = patches through 0490):** [`tensorfold-20260929/`](tensorfold-20260929/),
   the section after that.

- Alex Ellis's published vLLM receipts (unmodified copies, MIT): [`reference-alexellis/`](reference-alexellis/README.md)

## W20 run (3 rounds, averaged)

Production after W20 (image b11, `config/prod.env.example` of this update),
serving, not restarted between runs. Three back-to-back RigMark standard-suite runs (2026-09-30 11:49-12:16 UTC,
538 / 541 / 543 s), RigMark pinned at `c5a0db01b054` (clean), `reasoning_effort` low, a new comparison ID a run
(`2026-09-glm53-exl3-2xspark-tensorfold-w20-final-r1` / `-r2` / `-r3`). All three valid: 15/15 basic output gates each.
All numbers: [`results/W20/final/summary.md`](../W20/final/summary.md).

Each directory: the receipt JSON + `.sha256` (r1 `bc08bfddfe9705d3...`, r2 `8135ad83b5e7fba0...`, r3
`bfbd879edcf68631...`), the card, `command.txt` (paths replaced by `<repo>`; the directories were renamed after the
runs), `metadata.json`, `preflight.json` and `models.json`. No `requests.jsonl` this time; RigMark's `run.log` is not
included.

| RigMark (mean of 3 runs' medians, min-max) | TensorFold W20 (b11) | TensorFold W17 (b9) | vLLM TP2 k=7 (Alex) |
|---|---:|---:|---:|
| Code / prose / structured decode tok/s | **72.4** (72.2-72.6) / **45.7** (45.3-45.8) / **95.6** (95.6-95.7) | 67.9 / 43.0 / 88.8 | 44.0 / 18.9 / 64.9 |
| Cold prefill 8K / 32K / 64K tok/s | 1,610 (1,609-1,611) / 1,684 (1,679-1,689) / 1,667 (1,663-1,671) | 1,560 / 1,634 / 1,620 | **1,813 / 1,908 / 1,922** |
| Immediate replay 8K / 32K / 64K tok/s | **38,183** / **139,893** / **248,359** | 36,474 / 132,814 / 243,980 | 1,812 / 11,046 / 11,364 |
| Replay TTFT 8K / 32K / 64K s | **0.21** / **0.23** / **0.26** | 0.22 / 0.25 / 0.27 | 4.52 / 2.97 / 5.77 |
| C1 / C2 / C4 aggregate tok/s | **57.5** (54.8-59.1) / **76.0** (73.1-79.2) / **95.1** (93.0-98.4) | 53.1 / 70.1 / 91.0 | 31.6 / 42.0 / 66.1 |
| C1 / C2 / C4 per-stream TTFT s | **0.47** / **0.61** / 0.84 | 0.49 / 0.65 / 0.89 | 0.60 / 0.68 / **0.81** |
| Code / prose / structured TTFT s | **0.48** / **0.38** / **0.45** | 0.50 / 0.41 / 0.47 | 0.60 / 0.49 / 0.47 |

Against W17: decode +5-8%, cold prefill +3%, C4 aggregate +4.5%; from W19's adopted changes (0580 expert loads, NCCL on
both CX7 functions) and host-side tuning outside this repo. Still behind Alex's vLLM receipt: cold prefill (0.87-0.89x) and C4 per-stream
TTFT (0.84 vs 0.81 s). The "Read this before comparing" notes under the W13 baseline apply: different weights
(abliterated EXL3 4-bit here, NVFP4 there), drafter policy, context limit, day and machines.

## Release run (3 rounds, averaged)

Production after W17 (image b9: W15's config plus 0560's multi-slot prefill on, `GLM53_TF_MULTI_PREFILL=1`; 0550 in
the image but off), serving, not restarted after the adopt. Three back-to-back RigMark standard-suite runs
(2026-09-30 02:59-03:37 local, 572 / 571 / 577 s), RigMark pinned at `c5a0db01b054` (clean), `reasoning_effort` low, a new
comparison ID a run (`2026-09-glm53-exl3-2xspark-tensorfold-w17-final-r1` / `-r2` / `-r3`) so no run could replay
another's prompts from the NVMe session tier. Each run was followed by glmbench, the multiturn concurrency bench and
ab.py before the next (`results/FINAL-20260930/final.sh`; all numbers: `results/FINAL-20260930/summary.md`). All three
valid: 15/15 basic output gates each.

Each directory: the receipt JSON + `.sha256` (r1 `fbaf073b7eb5075e...`, r2 `bbc1af3785d6da9f...`, r3
`4dca115c5ac4a5a7...`), the card, `command.txt` (the directories were renamed after the runs, so it names
`tensorfold-20260929-195957` / `-201355` / `-202752`), `metadata.json`, `preflight.json`, `models.json` and
`requests.jsonl` (the server's request log of the run, cut by the run's start / finish time: token counts, cache source,
timings; no text). Every replay row in each log is `cached` = n - 64 (8,128 / 32,704 / 65,472) from the slot, 9/9 a
run. RigMark's `run.log` is not included.

The table: mean of the three runs' medians, with min-max in brackets.

| RigMark (mean of 3 runs' medians, min-max) | TensorFold release (b9) | vLLM TP2 k=7 (Alex) |
|---|---:|---:|
| Code / prose / structured decode tok/s | **67.9** (67.4-68.6) / **43.0** (42.2-43.5) / **88.8** (88.6-89.0) | 44.0 / 18.9 / 64.9 |
| Cold prefill 8K / 32K / 64K tok/s | 1,560 (1,556-1,562) / 1,634 (1,631-1,637) / 1,620 (1,618-1,621) | **1,813 / 1,908 / 1,922** |
| Immediate replay 8K / 32K / 64K tok/s | **36,474** (35,856-37,185) / **132,814** (129,084-136,610) / **243,980** (242,167-246,603) | 1,812 / 11,046 / 11,364 |
| Replay TTFT 8K / 32K / 64K s | **0.22** (0.22-0.23) / **0.25** (0.24-0.25) / **0.27** (0.27-0.27) | 4.52 / 2.97 / 5.77 |
| C1 / C2 / C4 aggregate tok/s | **53.1** (52.1-53.9) / **70.1** (68.8-70.9) / **91.0** (89.5-92.1) | 31.6 / 42.0 / 66.1 |
| C1 / C2 / C4 per-stream TTFT s | **0.49** (0.49-0.50) / **0.65** (0.64-0.66) / 0.89 (0.88-0.89) | 0.60 / 0.68 / **0.81** |
| Code / prose / structured TTFT s | 0.50 / 0.41 / 0.47 | 0.60 / 0.49 / 0.47 |

Against W15 (b7, same RigMark, table below): C4 aggregate 91.0 vs 81.8 / 82.8 (+10%), C2 70.1 vs 67.0 / 67.3 (+4%), C2
per-stream TTFT 0.65 vs 0.95 / 0.76 s and C4 0.89 vs 1.63 / 1.91 s, all from 0560's grouped prefill (several waiting
prompts' prefill pieces in one forward, each with the bits it gets alone). Decode, cold prefill and replay are
unchanged within run-to-run noise. Where we are still behind Alex's vLLM receipt: cold prefill (0.84-0.86x; RigMark's
grid-aligned token-id prompts, our own chat-prompt bench gives ~1,606-1,610 tok/s) and C4 per-stream TTFT (0.89 vs
0.81 s). The "Read this before comparing" notes under the W13 baseline apply here too: different weights (abliterated
EXL3 4-bit here, LibertAIDAI NVFP4 there), drafter policy, context limit (1M vs 262k), protocol, day and machines;
reasoning is single-counted since W15.

## W15 validation runs (2026-09-29)

Production after W15 (image b7: image input on, 0540's replay snapshot and early first token, reasoning in the
`reasoning` field only), RigMark pinned at `c5a0db01b054` (clean), standard suite, `reasoning_effort` low, two new
comparison IDs (`2026-09-glm53-exl3-2xspark-tensorfold-w15-b7-v1` / `-v2`) so no earlier sweep's prompt could sit in
the NVMe session tier. Both runs valid: 15/15 basic output gates, no preflight gaps (reasoning is no longer sent twice),
every prefill row's `prompt_tokens` == its depth. Each directory: the receipt JSON + `.sha256` (run 1
`94533a0f...`, run 2 `aaeb3751...`), the card, `command.txt` (the directories were renamed after the runs, so it names
`tensorfold-20260929-141934` / `-143739`), `metadata.json`, `preflight.json`, `models.json` and `requests.jsonl`
(the server's request log of the run: token counts, cache source, timings; no text). RigMark's `run.log` is not
included. Side by side with W13 and Alex Ellis's k=7 receipt: [`../W15/rigmark-w13-w15.md`](../W15/rigmark-w13-w15.md).

| RigMark (median) | W13 b5 | W15 b7 run 1 | W15 b7 run 2 | vLLM TP2 k=7 (Alex) |
|---|---:|---:|---:|---:|
| Code / prose / structured decode tok/s | 68.6 / 43.2 / 88.2 | 67.5 / 44.0 / 89.0 | 67.2 / 42.7 / 88.7 | 44.0 / 18.9 / 64.9 |
| Cold prefill 8K / 32K / 64K tok/s | 1,598 / 1,641 / 1,621 | 1,559 / 1,635 / 1,619 | 1,564 / 1,638 / 1,619 | 1,813 / 1,908 / 1,922 |
| Immediate replay 8K / 32K / 64K tok/s | 1,597 / 3,319 / 6,304 | **38,881 / 146,067 / 257,263** | **37,117 / 142,316 / 250,472** | 1,812 / 11,046 / 11,364 |
| Replay TTFT 8K / 32K / 64K s | 5.13 / 9.87 / 10.4 | 0.21 / 0.22 / 0.25 | 0.22 / 0.23 / 0.26 | 4.52 / 2.97 / 5.77 |
| C1 / C2 / C4 aggregate tok/s | 54.3 / 65.5 / 82.2 | 54.8 / 67.0 / 81.8 | 54.3 / 67.3 / 82.8 | 31.6 / 42.0 / 66.1 |
| C4 per-stream TTFT s | 1.91 | 1.63 | 1.91 | 0.81 |

**What "immediate replay" is here, and how it was checked.** RigMark's replay rate is `prompt_tokens / TTFT` for an
identical prompt sent again right after its cold run. TensorFold keeps a snapshot of the whole model state (latent KV,
KDA recurrent state, MTP / DFlash2 context) at the last 64-token grid point strictly before a prompt's end (patch 0540),
so an identical resend resumes at n - 64 and computes only the last 64 tokens through every layer (~0.17-0.20 s), then
samples. That is prefix / session caching of an identical prompt, not a faster prefill: the cold rows above are the
prefill speed. Checks (docs/RESULTS.md W15 §6, `results/W15/verify.py`, `verify-prod.json`):

- **The request log** (`requests.jsonl`, 9 replays a run): every replay `cached` = n - 64 (8,128 / 32,704 / 65,472),
  `cache_src` `slot`, one 64-row piece; the cold runs `cached` 0.
- **Same output as computing it:** each replay's 8-token output sha equals its cold run's in RigMark's own receipts
  (9/9 in both runs). Separately, prompts of exactly 8,192 / 32,768 / 65,536 token ids the server had never seen
  (`cached` 0), 64 greedy tokens with `ignore_eos`: the replay and a second replay returned **byte-identical 64
  tokens** at all three depths.
- **Negative controls:** variants with one token changed at the start, the middle, n - 10 and n - 100, and a
  different prompt of the same length. `cached` never exceeded the prefix the variant shares with anything stored
  (start / different prompt: 0 and a full cold prefill; middle: the last 16,384 session mark before the change;
  n - 100: below the change). Every variant's output differs from the base's except one (64K, middle: a word
  changed 31k tokens before the end did not change the next 64 greedy tokens; its log shows 32,768 tokens recomputed).
- **Why vLLM replays at ~11k tok/s on this model:** GLM-5.3-Flash is a hybrid (KDA linear attention + MLA). vLLM's
  prefix cache can only restore the recurrent KDA state at page-aligned checkpoints (`mamba_cache_mode = "align"`),
  caps a hit at n - 1 rounded down to a whole block, never prefix-caches the DSA indexer's scratch, and drops the last
  matched page per cache group with a drafter. A replay therefore restarts several thousand tokens before the end
  (Alex's receipts imply ~8,200 / ~5,700 / ~11,100 recomputed tokens at 8K / 32K / 64K) and at 8K reuses nothing.
  This is vLLM's stock hybrid caching, not a TensorFold shortcut.

Against W13: replay 24x / 44x / 41x faster, reproduced within 5% by run 2; cold 8K -2% (0540 adds one 64-row chunk to a
grid-aligned cold prompt; 32K / 64K equal); decode and aggregates within run-to-run noise. C4 per-stream TTFT is
unchanged within noise: 0540 hands a piece's first token over when the piece ends, but with thinking on (RigMark's
low effort) the first token carries no visible text, so the first visible delta still waits for the round
(docs/RESULTS.md W15 §2; patch 0560, multi-slot prefill, is the fix in progress). Reasoning characters are now
single-counted (60, as vLLM's 60).

## W13 baseline (2026-09-29)

[RigMark](https://github.com/alexellis/rigmark) by Alex Ellis, pinned revision `c5a0db0` (protocol 1.1.0), **standard
suite, unmodified settings**, with the same `{"chat_template_kwargs":{"reasoning_effort":"low"}}` request body Alex
uses for his published GLM-5.3 runs. Run from the head node against `127.0.0.1:8000` (loopback, no proxy, no other
traffic during the run).

**This is a baseline.** We think we can do better with more optimizations over the coming days, and will rerun RigMark
and publish the new receipt here when we do.

- Receipt (the content-hashed result JSON): [`tensorfold-20260929/glm53-flash-exl3-tensorfold-tp2-low.json`](tensorfold-20260929/glm53-flash-exl3-tensorfold-tp2-low.json)
  (`sha256sum -c` against the `.sha256` file; `./rigmark report --save <json>` regenerates the card byte-identically)
- Card: [`tensorfold-20260929/glm53-flash-exl3-tensorfold-tp2-low.card.txt`](tensorfold-20260929/glm53-flash-exl3-tensorfold-tp2-low.card.txt)
- Command: [`tensorfold-20260929/command.txt`](tensorfold-20260929/command.txt)
- Side-by-side with Alex Ellis's two published vLLM GLM-5.3 TP2 receipts: [`compare-published-20260929.md`](compare-published-20260929.md)

```
15/15 BASIC OUTPUT GATES PASSED | GLM-5.3-Flash-EXL3 | 2x DGX Spark | reasoning=low | protocol 1.1.0 | git:c5a0db01b054 clean
CODE 68.6 tok/s 26.3s last (66.5–69.3) 5/5 | PROSE 43.2 tok/s 24.1s last (43.0–43.5) 5/5 | STRUCTURED* 88.2 tok/s 8.2s last (87.3–89.1) 5/5
64K PREFILL cold 1,621 • replay 6,304 tok/s | C1 54.3 • C2 65.5 • C4 82.2 tok/s | C4 normal stop 0/12, visible 12/12 | sha256:4af01c23364e22b6…
```

| RigMark (median) | TensorFold (this repo) | vLLM TP2 k=7 (Alex) | vLLM TP2 adaptive (Alex) |
|---|---:|---:|---:|
| Code decode tok/s | **68.6** | 44.0 | 42.6 |
| Prose decode tok/s | **43.2** | 18.9 | 22.2 |
| Structured decode tok/s | **88.2** | 64.9 | 54.6 |
| Code / prose time to last output (s) | **26.3 / 24.1** | 46.6 / 55.7 | 50.2 / 45.4 |
| Cold prefill 8K / 32K / 64K tok/s | 1,598 / 1,641 / 1,621 | **1,813 / 1,908 / 1,922** | 1,835 / 1,898 / 1,905 |
| Immediate replay 32K / 64K tok/s | 3,319 / 6,304 | **11,046 / 11,364** | 11,339 / 11,464 |
| C1 / C2 / C4 aggregate tok/s | **54.3 / 65.5 / 82.2** | 31.6 / 42.0 / 66.1 | 31.2 / 42.8 / 61.1 |
| C4 per-stream TTFT (s) | 1.9 | **0.8** | 0.8 |

### Read this before comparing

- Not a strict RigMark comparison (RigMark refuses one across different receipts/protocols; the table uses
  `--allow-mismatch`): different weights (abliterated EXL3 4-bit here vs LibertAIDAI NVFP4), drafter policy, context
  limit (1M vs 262k), protocol (1.1.0 vs 1.0.0), day and machines.
- Structured output: our model emits pretty-printed JSON (687 tokens) vs 441 compact tokens in Alex's runs, so that
  row compares different outputs (all 5/5 valid in every run).
- Our reasoning text is emitted in both `reasoning` and `reasoning_content`, which RigMark sums; reasoning character
  counts in our receipt are doubled (timings and token counts are unaffected). Fixed in a later config.
- Where we lose: cold prefill (12-16% behind), immediate replay (our snapshot grid made identical-prompt replays resume
  from the last 16k mark, not the end) and C4 time to first token (concurrent prompts admitted one prefill piece at a
  time). Fixes for replay and C4 TTFT are in progress.
- The receipt's `competing_traffic` string says "test window held": that text came from our run script; the endpoint
  was idle and nothing else ran, but no maintenance window was held.

### Recipe

The server was this repo's production config (`config/prod.env.example`) on image `b5`: TensorFold v0.3.4 (`2f8e514`)
plus this repo's patches through 0490. Patches 0420-0490 (L2 weight prefetch, OpenAI-style context errors,
`/tokenize` for RigMark's prefill phase, and others) are in the repo since the 2026-09-30 update. To reproduce:
`docs/RIGMARK.md` and `scripts/rigmark/` (`install.sh`, then `run.sh tensorfold` on the head node). The directory
also holds the run's `metadata.json`, `preflight.json`, `models.json` and `requests.jsonl` (request log, no text).
