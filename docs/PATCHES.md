# Patches to TensorFold

TensorFold is vendored as the git submodule `vendor/TensorFold`, pinned at `2f8e514` (0.3.4). The image build
(`docker/Dockerfile`) applies `patches/*.patch` in name order with `git apply`, then installs the patched tree.
The submodule stays at the pinned commit; the patches are the whole diff. Patches 0001, 0003 and 0004 touch
only the GLM (`glm5_next`) CUDA engine; 0002 touches the shared CUDA server's tool-call parsing.

Every knob defaults to upstream behaviour, so an image with the patches and no environment set serves exactly
what upstream serves (plus the GLM tool-call parser, which only adds a format upstream did not read).

| Patch | Knob | Default | Effect |
| --- | --- | --- | --- |
| 0001 `glm-exl3-nonexpert-q4` | `GLM53_TF_NONEXPERT=bf16\|q4\|q4mse` | `bf16` | EXL3 non-expert weights stored in 4 bits; decode 1.2-1.9x |
| 0002 `glm-tool-call-parser` | none | on | GLM `<arg_key>/<arg_value>` tool calls become OpenAI `tool_calls` |
| 0003 `glm-prefill-rows` | `GLM53_TF_PREFILL_ROWS=N` | `64` | prefill chunk size; kernels tile past 128 rows |
| 0004 `glm-sparse-long-prefill` | none | on | sparse attention (DSA) scratch sized per window; vectorized token selection |
| 0050 `glm-longctx-decode` | `GLM53_TF_LONGCTX_GRAPHS=0\|1` | `1` | past 2,051 tokens: indexer scores only existing pools (power-of-2 bucket); decode/verify/MTP steps of 1-8 rows replay CUDA graphs per (rows, parity, bucket), captured lazily. `0` = upstream path |
| 0060 `glm-latent-kv` | `GLM53_TF_LATENT_KV=0\|1` | `0` | DSA/MLA layers cache the 512-wide latent instead of per-head K/V (absorbed MLA): 18.75 KB a token a rank instead of 390.75 KB (20.8x), 1M context in ~19 GB; attention reads ~16-30x fewer bytes. New arithmetic (replies differ from `0` in the last bits), every exactness guarantee kept within `1` |
| 0065 `glm-1m-memory-indexer` | `GLM53_TF_SELECT=blocked\|sort`; `GLM53_TF_SELECT_MB=N`; `GLM53_TF_INDEX_RING=0\|1`; `GLM53_TF_DRAFTER_RING=0\|1` | `blocked`; `256`; `1`; `1` | 1M context: DFlash2 context and the model layers' index keys/gates in rings (29.5 -> 13.6 KB reserved a token a rank); prefill selection in row blocks with tile-shared scoring and a top-k threshold instead of a full sort (bounded scratch, ~4x faster indexer, estimated). Same bits: replies equal `sort`/`0`/`0` |
| 0080 `glm-fast-prefill` | `GLM53_TF_FAST_PREFILL=0\|1`; `GLM53_TF_FAST_GATHER=bf16\|fp32` | `0`; `bf16` | prefill chunks through kernels that need not be row-invariant (fused EXL3 expert GEMMs, bf16 rank partials, head on the last row; 0081's chunked KDA and large-M matmuls when present) at absolute multiples of a chunk grid C; only the state at the prompt's last grid point is kept. New arithmetic (replies differ from `0`), drafted == serial and resumed == fresh kept within `1` |
| 0082 `glm-lean-prefill` | `GLM53_TF_LEAN_PREFILL=0\|1`; `GLM53_TF_LEAN_BLOCK=N` | `0`; `1024` | fast chunks of up to `PREFILL_ROWS_MAX` rows (e.g. 8192) in sub-blocks of N rows on window buffers of N rows; the routed experts run once over the whole chunk. Same bits as 0080 at the same grid. 8192-row chunks need 11.6 GiB of row buffers a rank, against 17.5 for today's 2048-row config |
| 0083 `glm-fp8-prefill` | `GLM53_TF_FP8_PREFILL=0\|1`; `GLM53_TF_FP8_TILE=bm,bn,warps,stages` | `0`; `128,64,4,2` | fast prefill chunks only: the large non-expert matmuls on FP8 (e4m3) tensor cores with per-row power-of-two activation scales and the exact 4-bit weights; on the latent cache, absorb / expand on bf16 tensor cores and 32-query attention tiles. New arithmetic (fast replies differ from `0`); snapshots tagged C + 1; drafted == serial and resumed == fresh kept. Estimated -115 to -165 us a token |
| 0084 `glm-prefill-overlap` | `GLM53_TF_PREFILL_OVERLAP=0\|1\|gather,slab,direct`; `GLM53_TF_PREFILL_HC_SLAB=N` | `0`; auto (L2 / 2) | lean chunks pipelined: each sub-block's all-gather on a comm stream while the next sub-block computes; partials exchanged in the dtype they are written in (no conversion copies); hc_post + the next hc_pre per L2-sized row slab. Same kernels, same inputs, same collectives in the same order: same bits as 0082. Estimated -90 to -120 us a token (of ~1,200-1,300 at 32k) |
| 0085 `glm-c-independent-prefill` | `GLM53_TF_PREFILL_ROWS=auto`; `GLM53_TF_SNAPSHOT_GRID=N`; `GLM53_TF_SNAPSHOT_TAIL=N` | `64` (`config/` sets `auto`); `64`; `256` | a fast prefill's state no longer depends on its chunk size C (every fast kernel row-independent, the KDA scan on absolute 64-row blocks; the < 64-row qmm fallback removed from fast chunks): fast snapshots at the prompt's last multiple of 64, tagged by mode only, resumable by any C; `prefill_rows` "auto" picks C per prefill. A next turn re-prefills < 64 old tokens + the reply instead of up to C - 1. Fast replies change only for prompts whose last chunk had < 64 rows |
| 0090 `glm-request-knobs` | `"tf_knobs": {...}` per request; `GLM53_TF_PREFILL_ROWS_MAX=N` | env defaults; max = `PREFILL_ROWS` | the speed knobs of 0003/0005/0006/0010/0020/0050/0070/0071 switch per request on a loaded server (both ranks, via the request header); the env sets their defaults |
| 0091 `glm-fast-prefill-knob` | `"tf_knobs": {"fast_prefill": 0\|1}` | `GLM53_TF_FAST_PREFILL` | 0080's fast prefill switched per request; fast and exact snapshots never mix |
| 0092 `glm-fp8-prefill-knob` | `"tf_knobs": {"fp8_prefill": 0\|1}` | `GLM53_TF_FP8_PREFILL` | 0083 switched per request (in the header, so both ranks agree); FP8 and bf16 fast snapshots never mix; refused where FP8 dots are unavailable |
| 0093 `glm-prefill-overlap-knob` | `"tf_knobs": {"prefill_overlap": 0\|1}` | `GLM53_TF_PREFILL_OVERLAP` | 0084's pipeline switched per request (same bits either way; 1 = the env's variant, or `gather,slab`) |
| 0110 `glm-session-cache` | `GLM53_TF_SESSION_GIB=N`; `GLM53_TF_SESSION_EVERY=N`; `GLM53_TF_SESSION_FORK_MIN=N`; `GLM53_TF_SESSION_RESERVE_GIB=N` | `0` (off; `config/` sets 12); `16384`; `512`; `2` | many sessions resident per rank (snapshots + their attention rows in 256-token pages, shared prefixes stored once, LRU within the budget): a request resumes from the longest stored prefix by copying it into the live caches (~5 ms at 30k) instead of re-prefilling; marks at fork points and every N tokens; `usage.prompt_tokens_details.cached_tokens`. Same bits: resumed == fresh, drafted == serial |
| 0120 `glm-batch-v2` | `GLM53_TF_BATCH=N`; `GLM53_TF_BATCH_PIECE=N`; `GLM53_TF_BATCH_PREFILL_SHARE=F`; `GLM53_TF_BATCH_GRAPH_ROWS` / `_MAX_GRAPHS`; `GLM53_TF_BATCH_RESERVE_GB` / `_ADMIT_GB`; `"priority": "background"` | `1` (off); `2048`; `0.5`; `8` / `256`; `4` / `1` | 0030's batching on the current engine: 2-4 requests a round, each with its own latent KV / rings, KDA state, MTP, DFlash2 context, lookup, cost-derived depths and `tf_knobs`; prompts prefill in pieces through `decode.prefill` (exact / fast / lean / FP8) between the others' rounds; graphs per (slots, rows, dense / pool bucket); background requests step aside. Same bits: batched == alone == serial |
| 0140 `glm-fast-boot` | `GLM53_TF_PREPARED=DIR`; `GLM53_TF_PREPARED_WRITE=0\|1`; `GLM53_TF_PREPARED_VERIFY=full\|sample\|off`; `GLM53_TF_PREPARED_THREADS=N`; `GLM53_TF_CALIB=cached`; `GLM53_TF_CALIB_DIR`; `GLM53_TF_CLOCK_CAP`; `GLM53_TF_BOOT_WARMUP=0\|1` | unset (off; `serve.sh` sets `/prepared`, write 1, `cached`); `0`; `sample`; `8`; `real`; `/cache/calib`; none; `1` | restarts: each rank's weights (and the DFlash2 drafter's) read back from a prepared folder exactly as the load built them (split, q4mse, tiled, EXL3 words), with a parallel O_DIRECT reader into pinned buffers; the calibration table reused per (image, knobs, engine shape, GPUs, clock caps), rank 0 deciding for both; `[boot]` timeline lines. Same bits: prepared == load-time (tested), replies unchanged |
| 0150 `glm-mia-wins` | `GLM53_TF_HEALTH=basic\|strict`; `GLM53_TF_STALL_S=N`; `GLM53_TF_STALL_PREFILL_TPS=N`; `GLM53_TF_EFFORT_FIELD=0\|1`; `GLM53_TF_DEFAULT_EFFORT=low\|high\|max` | `basic`; `0` (off); `200`; `0`; unset | `/health` reports a fatal engine error, requests in flight and stalls (`strict`: 503, and new completions refused after a fatal error); `GET /metrics` (Prometheus counters); an engine error answers a JSON 500 / an SSE error event instead of dropping the connection; OpenAI `reasoning_effort` mapped onto the GLM template's thinking / effort, and a server default effort. Serving path unchanged; replies change only for requests the effort knobs rewrite |
| 0170 `glm-mia-prefill` | `GLM53_TF_FAST_EXPERTS=fat`; `"tf_knobs": {"fat_experts": 0\|1}`; `GLM53_TF_FAT_STAGES=3\|4`; `GLM53_TF_FAT_TICKET=0\|1`; `GLM53_TF_FAT_SHARED_X=auto\|0`; `GLM53_TF_KDA_PROJ_BF16=0\|1`; `GLM53_TF_KDA_BF16_TILE` | `fast2`; env; `3`; `1`; `auto`; `0`; table | fast chunks' routed experts through the `fat` kernels (MiaAI-Lab / Reederey87 E2/E3 data movement: trellis words in the cp.async ring, one rotated input for gate and up, swizzled 48 KB stages at 2 CTAs an SM, ticket scheduling) with fast2's arithmetic: **same bits** as fast2, row-independent. KDA input projection as a bf16 copy of the 4-bit weights in fast chunks (+98.25 MiB a KDA layer and rank; new arithmetic, row-independent, load-time only) |
| 0180 `glm-batch-sessions` | `GLM53_TF_BATCH_SESSIONS=0\|1` | `0` | with `GLM53_TF_BATCH` > 1 and `GLM53_TF_SESSION_GIB` > 0: 0110's store behind 0120's slots. An admission restores the longest stored prefix into a free slot (the one already holding most of its pages) and prefills only the suffix in pieces; marks, prompt and reply snapshots are saved from every slot; one index / budget / page pool, a live-page map per slot; the budget is set aside at load and at admission. Same bits: batched == alone == serial, resumed == fresh |
| 0190 `glm-prefill-glue` | `GLM53_TF_MOE_GLUE=0..7`; `GLM53_TF_MTP_PREFILL_WINDOW=N`; `GLM53_TF_HC_FUSED=0..3`; `GLM53_TF_ATTN_BM32=0\|1` (all per request: `tf_knobs.moe_glue` / `mtp_window` / `hc_fused` / `attn_bm32`); `GLM53_TF_LATENT_TC=0\|1` (load-time) | all `0` | prefill glue: MoE grouping by a parallel sort, the router in one kernel, the combine reading the shared expert in place (same bits); the MTP head run only on a prompt's last ~N positions (drafts may change, replies not); hc_post fused with the next hc_pre's dots + an unrolled finish (same bits, GPU-checked); 32-query latent attention tiles in bf16 fast chunks (same bits, GPU-checked); opt-in: latent absorb / expand on bf16 tensor cores (new arithmetic, own snapshot tag G + 2). Microbenchmark: `tests/cuda/bench_glue.py` |
| 0200 `glm-batch-parallel` | `GLM53_TF_BATCH_CAPTURE_AFTER=N`; `GLM53_TF_BATCH_PARITY_KEY=0\|1`; `GLM53_TF_BATCH_MTP=0\|1`; `GLM53_TF_BATCH_ROW_MS=F`; `GLM53_TF_BATCH_SHORT=N`; `GLM53_TF_BATCH_PAD=sizes` | `1`; `0`; `0`; `0`; `0`; off | with `GLM53_TF_BATCH` > 1: batched-round graph keys captured on their N-th sighting; KDA parities in the key instead of 71 MB state copies a slot and round; every slot's MTP chain drafted in one head pass a step; batch-aware cost depths price rows past the verify table at >= F ms; prompts of <= N tokens prefill in their admission round outside the fair share; windows padded to fewer graph keys. Per-request `round_kinds` stats. Same bits: replies == served alone |
| 0340 `glm-batch-adapt` | `GLM53_TF_BATCH_ADAPT=0\|1`; `GLM53_TF_BATCH_ADAPT_SERIAL=0\|1`; `GLM53_TF_BATCH_ADAPT_EVERY=N`; `GLM53_TF_BATCH_ADAPT_WINDOW=N` | `0`; `0`; `8`; `8` | with `GLM53_TF_BATCH` > 1 and cost-derived depths: in SHARED rounds each slot picks its drafter (MTP / DFlash2, or with SERIAL a one-row serial round) by the drafts' surplus at the shared rate and the round's marginal row costs (`adapt.SlotChoice`) instead of `DrafterChoice`'s tokens per lone-model ms; depths still from `BatchDepth`; alone unchanged. Drafts only: replies unchanged. Offline simulator (`bench/draftsim.py`, W1 / W5 streams): ~0 (-0.4%) aggregate, serial -4 to -7%: **not recommended**; `docs/ADAPTIVE-DRAFT.md` |
| 0210 `glm-prompt-tokens` | `GLM53_TF_TOKCACHE=0\|1\|verify`; `GLM53_TF_TOKCACHE_ENTRIES=N` | `1`; `32` | host only (rank 0's HTTP app). A request's prompt is encoded once (the context check hands its ids to `run`; it was encoded twice), and the ids of the last N prompts are kept: a new prompt reuses the longest prefix it shares with one up to the end of a special token (`<\|user\|>`, `<\|assistant\|>`, `<\|observation\|>`, ...) and encodes only the rest. Exact: the tokenizer splits on added tokens before BPE (no normalizer, every added token plain; checked at start, off otherwise); `verify` also runs the full encode and logs + uses it on any difference. ~34 ms -> ~1 ms a turn at 40k tokens, ~113 ms -> ~3 ms at 128k, plus the second encode saved. Ids unchanged |
| 0220 `glm-fp8-latent-kv` | `GLM53_TF_KV_DTYPE=bf16\|fp8` (load-time, both ranks equal) | `bf16` | with `GLM53_TF_LATENT_KV=1`: the latent caches (11 DSA layers + the MTP head) store a token as 512 e4m3 values + one power-of-two fp32 scale (528 B rows, 16-byte aligned) instead of 1,024 B of bf16: **13,616 -> 7,664 B a token a rank** (4 x 262k slots: 13.26 -> 7.45 GiB a node). Rows quantized one at a time in `latent_write`; the dense / sparse latent kernels dequantize exactly into bf16 and run the bf16 arithmetic. KV storage only (prefill activations stay bf16). New arithmetic vs bf16 KV; within fp8, same bits: drafted == serial, resumed == fresh, batched == alone. Snapshots and session keys carry the format |
| 0240 `glm-b12x-prefill` | `GLM53_TF_B12X=0..7` (per request: `tf_knobs.b12x`); `GLM53_TF_B12X_KDA_PREC=tf32\|bf16`, `_BV`, `_WARPS` (load-time) | `0`; `tf32`, `64`, `4` | fast chunks only, NEW arithmetic, own snapshot tag (G + 4 x bits): bit 1 hc mixing dots over column blocks of all four streams fused with hc_post (`b12x_mhc`), bit 2 KDA in 16-row tiles with prep + recurrence in one program and no workspace (`b12x_kda`), bit 4 one-pass sparse latent attention, no chunk partials (`b12x_attn`). Algorithms from b12x (Apache-2.0), re-implemented in Triton. Deterministic, row-independent, drafted == serial and resumed == fresh within each setting. Estimated 28k: 1,266 -> ~1,440-1,560 tok/s with 7 |
| 0250 `glm-session-nvme` | `GLM53_TF_SESSION_DISK=DIR`; `GLM53_TF_SESSION_DISK_GIB=N`; `GLM53_TF_SESSION_DISK_WRITE=save\|evict`; `GLM53_TF_SESSION_DISK_MIN=N`; `_GAIN`, `_THREADS`, `_QUEUE_GIB`, `_VERIFY`, `_DIRECT` | unset (off; `serve.sh` mounts `/sessions`); `64`; `save`; `1024`; `256`, `8`, `1`, `full`, `1` | the session store's entries also on local NVMe, per rank (pages by the store's page keys, shared prefixes once; an entry file with the snapshot + a checksummed manifest; write-through or on eviction), looked up after RAM and read straight into the live slot (a 40k FP8 session ~0.4 GB a rank, ~0.15-0.3 s instead of a 35-45 s cold prefill); indexed again at load, so sessions survive restarts; stale config never read (compat hash), any failed read on either rank = both prefill cold. Same bits: resumed == fresh |
| 0230 `glm-roce-gather` | `GLM53_TF_COMM_BACKEND=nccl\|roce` (both ranks equal, checked at load); `GLM53_TF_ROCE_MAX_KB`; `GLM53_TF_ROCE_TIMEOUT_S`; `GLM53_TF_ROCE_HCA` / `_HCAS`; `GLM53_TF_ROCE_FALLBACK`; `GLM53_TF_ROCE_MARK`; `GLM53_TF_ROCE_CPU` / `_TC` / `_BLOCKS` / `_THREADS` | `nccl`; `256`; `120`; auto / `2`; `nccl`; `/cache/roce-failed`; none / `NCCL_IB_TC` / `8` / `512` | the model's exchanges of up to 256 KiB a rank (decode / verify / MTP windows, samplers, drafter, small prefill chunks) over b12x's one-shot RoCE all-gather ("RoCEnante", Apache-2.0): pinned host slots the CX7 RDMA-writes and the GB10 reads in place, a busy-spinning libibverbs proxy striping over both CX7 functions, device epoch (CUDA-graph safe); control exchanges and large prefill gathers stay on NCCL. A data move: **same bits** as NCCL. RoCE v2 GIDs detected per port, ports paired by subnet; setup + probe vs NCCL collective, any failure -> both ranks on NCCL; a run-time timeout poisons, raises a diagnosis and leaves a marker so the restart uses NCCL. Estimated -1.2 to -1.5 ms a 1-row step |
| 0260 `glm-expert-decode-once` | `GLM53_TF_FAST_EXPERTS=once`; `GLM53_TF_ONCE_PAIR=0\|1`; `GLM53_TF_ONCE_MIN_ROWS=N` | off (`fast2`); `1`; `0` | fast chunks' routed experts: fat with two work items a CTA; two passes of an expert share each trellis-tile decode (a tile decoded ceil(P / 2) instead of P times a chunk: 8192 rows 3.3 -> 1.7x, 2048 1.15 -> 1.0x, 1024 already 1.0x). Same bits as fat / fast2 (shares snapshots; `tf_knobs.fat_experts` switches once vs fast2). Offline, untimed: expected -1 to -3 ms a layer at 8192 rows, ~0 at 2048 |
| 0270 `glm-fast-experts-auto` | `GLM53_TF_FAST_EXPERTS=auto`; `GLM53_TF_FAST2_ROWS=lo,hi`; `"tf_knobs": {"fat_experts": 0\|1\|2}` | off (`fast2`); `0,4096`; env | fast chunks' routed experts picked per chunk by its rows: fast2 (on fat's one rotated input when gate / up share their sign vector) for lo <= R < hi, the fat family (fat / once) otherwise. Same bits every way (shares snapshots); per request 0 fast2, 1 fat family, 2 auto. W5: fast2 wins the kernel 1.19-1.33x at 8-2,048 rows but auto is -2% end to end on production: **not used** |
| 0280 `glm-batch-buckets` | `GLM53_TF_BATCH_BUCKETS=sizes`; `GLM53_TF_BATCH_PAD_TIE=auto\|0\|1` | off; `auto` (on with buckets) | with `GLM53_TF_BATCH` > 1: every window of a multi-slot graph round padded with its last token to the smallest listed size >= the round's longest window (keys: slots, bucket, modes, parities); a padded row's router logits replaced by its window's last real row's, so it reads no new experts. Padded rows never kept: same bits (batched == alone). W5: graph rounds 8-33% -> 80-91%, but -5% aggregate (padded rows ~1 ms each; eager rounds turned out to cost ~nothing): **not used** |
| 0290 `glm-kv-pool` | `GLM53_TF_KV_POOL_TOKENS=N`; `GLM53_TF_KV_POOL_PAGE`; `GLM53_TF_KV_POOL_SLACK`; `GLM53_TF_KV_POOL_CHECK` | `0` (off); `256`; `64`; `0` | with `GLM53_TF_LATENT_KV=1`: every slot's capacity-sized caches (11 latent + MTP latent, MTP index keys / gates, 12 pool keys: 7,616 B a token a rank with FP8 rows) in ONE pool of N tokens in 256-token pages, a page table per slot. Any slot grows to `CONTEXT` (up to ~1M) while the sum fits: a 1M pool costs what 4 x 262k slots cost today (7.44 GiB a node). Admission reserves prompt + max_tokens + slack in pages; when short, idle slots' pages are spilled (their sessions are in the store / NVMe already), else the request waits; larger than the pool: refused. Every reader / writer maps rows through the table (same values, other addresses): **same bits**; off, the kernels compile to the same PTX as before. `docs/KV-POOL.md`. Offline only so far |
| 0300 `glm-request-log` | `GLM53_TF_REQUEST_LOG=FILE`; `GLM53_TF_REQUEST_LOG_MB`, `_KEEP`, `_PROMPTS`, `_SALT` | unset (off); `64`, `3`, `64`, none | rank 0 appends one JSON line a request: prompt / cached tokens and where the cache came from (slot / RAM / NVMe / none), prefill s and tok/s, queue wait, first delta, decode tokens / s / tok/s, tokens a round, KV pool pages reserved / free, marks, effort / thinking, max_tokens, finish reason, hashes of the first 4k token ids and of the system part, a conversation key, and the longest common prefix with the last 64 prompts (any / same / other conversation). No text. Host work on the request's thread after the reply, no GPU sync. `scripts/traffic-report.py` summarizes it (reuse rate by source, prefill avoidable by shared prefixes, size / speed distributions, top shared prefixes). `docs/PREFIX-SHARE.md` |
| 0310 `glm-prefix-share` | `GLM53_TF_PREFIX_SHARE=0\|1`; `_MIN`, `_MAX`, `_POINTS`, `_KEEP`, `_WAIT`, `_TOKENS` | `0` (off); `2048`, `131072`, `1`, `4`, `1`, from the tokenizer | shared-prefix reuse across sessions: rank 0 also marks the end of the system prompt (the first user / assistant / observation role token, on the snapshot grid), so the FIRST session over a system prompt leaves the snapshot a new session resumes at (today the 2nd session over a < 16k system prompt prefills it all, and a burst of N sessions prefills it N times); in batch mode an admission waits while an in-flight partner prefills toward such a mark in their shared prefix, then resumes at it. At most `KEEP` system-prompt entries in RAM (~94 MB a rank each). Marks and restores are 0110 / 0180's: **same bits**. `docs/PREFIX-SHARE.md`. Offline only so far |
| 0320 `glm-prefill-row-split` | `GLM53_TF_PREFILL_PP=0\|1` (load-time; both ranks equal, with `GLM53_TF_PREFILL_OVERLAP`'s default, checked at load) | `0` | pipelined lean chunks (0084) of 2+ sub-blocks: each rank runs the hyper-connections, taps and final norm on its half of every sub-block's rows (64-aligned) instead of all of them; the partials are traded as row halves (NCCL send/recv in one group: half the bytes on the critical path) and the normed rows / final rows / taps shared back in place before their readers. Same bytes a rank in total, half the replicated hc work. **Same bits** (row-independent kernels, copies, rank-0-first sum unchanged). Offline: expected +3 to +5% prefill; true PP (needs a second ~75 GiB weight copy) and expert parallelism analysed and rejected in `docs/PREFILL-PP.md` |
| 0330 `glm-expert-tc` | `GLM53_TF_FAST_EXPERTS=tc`; `GLM53_TF_TC_CFG=gu,dn`; `GLM53_TF_TC_TICKET=0\|1`; `GLM53_TF_TC_CTAS=N` | off (`fast2`); `0,0` (by rows); `1`; `0` (every SM) | fast chunks' routed experts through `exl3_tc.cu` (own extension, built on first use): a warp-specialized persistent kernel. One producer warp a CTA claims items by ticket, gathers member rows with cp.async and copies trellis words with cp.async.bulk into an mbarrier-guarded 3-stage ring; 16 consumer warps run fat's inner loop with no CTA-wide barrier in the K loop; items of 2-4 column blocks (< 4,096 rows) or 128 members (4,096+); the epilogue overlaps the next item's loads. A fat-family mode (`tf_knobs.fat_experts` 1); needs gate / up to share their sign vector (fat otherwise). **Same bits** as fat / fast2 (the same mma chain per element; shares snapshots). Offline: compiles for sm_121 (120 registers, 89 KB shared); untimed. `docs/EXPERT-TC.md` |
| 0335 `glm-solo-piece` | `GLM53_TF_SOLO_PIECE=N`; `GLM53_TF_LEAN_LAZY_XU=0\|1` | `0` (off); `0` | with `GLM53_TF_BATCH` > 1: a fast prefill piece of a request **alone** in the batch is N tokens instead of `GLM53_TF_BATCH_PIECE`; the next piece is a normal one once another request is admitted (decided from the slots, alike on both ranks; checked equal at load). Bigger chunks need `GLM53_TF_PREFILL_ROWS_MAX` >= N (lean set 397 KiB a row: +0.78 GiB a rank at 4,096, +2.33 at 8,192). Lazy Xu: the lean set's unused up-matrix rows (72 KiB a row) allocated at first use (the fat family never reads them on the real checkpoint). **Same bits** (pieces end on the snapshot grid; resumed == fresh). Offline; expected +4-8% single-stream prefill, a newcomer waits up to one solo piece. `docs/EXPERT-TC.md` |
| 0350 `glm-roce-loopback-fix` | none new (0230's `GLM53_TF_COMM_BACKEND=roce` stays opt-in) | - | fixes on 0230: the W2 loopback failure was the single-process harness (one rank's graph warm-ups alone; `bench_roce.py` now interleaves them), not GPU visibility; a failed wait's record says whether the flag was in host memory AT the failure (`not_seen`), arrived after it (`late`) or never (`roce_watch.h`, polled by the proxy); `bench_roce.py stress` (payload moves every op). docs/ROCE-FIX.md |
| 0360 `glm-b12x-attn-fp8` | none new (0240's `tf_knobs.b12x` bit 4) | - | bit 4 fit for production: requests with b12x bits resume their sessions (the lone engine's lookup lacked the bits: `cached` was always 0); FP8 one-pass attention == bf16 on the dequantized rows by construction (the dots' operand layout of the bf16 kernel); row-independent (fixed per-row order, no R-dependent split); the NVMe compat hash no longer holds the default bits. Same replies with bit 4 off |
| 0370 `glm-decode-overlap-cpu-pin` | `GLM53_TF_DECODE_OVERLAP=0\|1\|sync,emit,plan,gil` (`plan` equal on both ranks, checked at load); `GLM53_TF_SWITCH_US=N`; `GLM53_TF_CPU_PIN=0\|fast\|auto\|serve=..;roce=..;comm=..;rest=..;nice=N`; `GLM53_TF_CPU_PIN_EVERY=S`; serve.sh `CPUSET` / `HEAD_CPUSET` / `WORKER_CPUSET` | `0`; `500`; `0`; `30`; empty | host scheduling only. N1b: no timing-only device sync after verify forwards (online timer from CUDA events); rank 0 hands a round's tokens to the HTTP threads after the next forward's launch (batch) / an emitter thread (no batch); in decode-only rounds the next round's plan (cancels) rides on the sampler's all-gather instead of 4 NCCL control all-gathers + 4 syncs at the round's top; 500 us GIL switch interval. N1a: the round loop, 0230's RoCE proxy and NCCL's threads on dedicated X925 cores (GB10 auto: serve 19, roce 18, comm 16-17, rest 0-15), or every thread on the X925s (`fast` = `CPUSET=5-9,15-19`). **Same bits** (same kernels, same collectives in the same order). Offline: +1-2% 1-stream / +0.6-1.5% 4-stream decode (the between-round gap is ~0.9 ms in W7, not 3.9); pinning +0-2%. `docs/DECODE-OVERLAP.md` **W10: `DECODE_OVERLAP=1` adopted (+1.3-1.9% 1 stream, +1.0-1.3% 4 streams); `CPU_PIN` off (+0.4%).** |
| 0380 `glm-deep-verify` | `GLM53_TF_MAX_DRAFT_ROWS=8..16`; `GLM53_TF_DFLASH_BLOCK=N` (both load-time, both ranks equal) | `8`; the checkpoint's `block_size` (8) | verify windows of up to N rows (a pending token and N - 1 drafts): 0071's cost depths and 0020's lookup gate keep deciding, with their caps raised (`engine.MAX_ROWS`, `depth` / `lookup` `MAX_DRAFTS`, `auto_fdrafts`, `cN:P` / `lN` / `om[N]` / `of[N]`); sized to match: 0050's long-context scratch and graph rows, `latent.SPARSE_ROWS`, the batch slots' KDA rows, batch graph / pad rows (~+115 MB a rank at 1M); the calibration times 1..N windows (the 1..8 table as before, 9..N on its own line); stats `depths`. `GLM53_TF_DFLASH_BLOCK=16`: 16-row DFlash2 block passes (an experiment: the drafter was trained on 8). **Same bits** (row-independent kernels, no kernel changed; PTX identical). Offline: copy / edit cells +16-30%, agent turns +0.2-0.6%, prose / code / 4 streams ±0.2% (DFlash2 proposes 7 positions at most); cross-request lookup (N3) +0.0%: not built. `docs/DEEP-VERIFY.md` **W10: `MAX_DRAFT_ROWS=16` adopted (edit cells +16-27%, identical reply hashes).** |
| 0390 `glm-mla-expand-v2` | `GLM53_TF_MLA_EXPAND=v1\|v2` (load-time) | `v1` | the latent MLA absorb / expand (`latent.absorb` / `expand`: every DSA layer and the MTP head, in prefill, decode, verify and MTP steps) through `_absorb2` / `_expand2`: the same fp32 FMA chain per output element (k in order from +0.0, exactly dequantized kv_b, one bf16 rounding), retiled: 16-128 rows a program share one dequantization of the kv_b tile (v1: 16), 16-wide dot steps (v1's 64-deep dot spills 452 B in `_expand`), a head's programs adjacent. **Same bits** as v1 for every row at any row count (offline: Triton source + compiled IR + interpreter model; GPU bitwise test pending). Offline estimate: `_expand` 2,415 -> 250-450 us, `_absorb` 676 -> 200-300 us a 512-row sub-block: +7.5-8.5% prefill. `docs/MLA-EXPAND.md` **W10: GPU bitwise passed, tiles retuned (4 warps), adopted: prefill +7.1% / +6.7%.** |
| 0400 `glm-kda-v2` | `GLM53_TF_KDA_V2=0\|1\|2` (`split` / `fused`; load-time, checked at load); `_BV`, `_WARPS`, `_MAXNREG`, `_KSPLIT`; `_FUSED_BV`, `_CTAS`, `_LAG`, `_RING`, `_FUSED_MAXNREG` | `0`; `32`, `4`, `168`, `1`; `64`, `0` (one an SM), `1`, `3`, `0` | fast chunks' KDA recurrence (`fastpf.kda_chain`, every KDA layer) through `kda_v2.kda_prefill_v2` instead of `fast_kda.kda_prefill_chunked`, **same bits** (every output row, every state / snapshot, any call size; shares snapshots, left out of the NVMe compat hash). `1` split: `fast_kda`'s prep, then a scan in 32-value-row blocks (128 programs, not 64; a head's blocks adjacent so DRAM serves each chunk's operands once; dots over two 64-key halves: 24 KB shared memory, 3 programs an SM). `2` fused: one persistent kernel takes prep and scan-step items by ticket, waits on release / acquire flags (every wait on an earlier ticket: no deadlock), the workspace a 3-chunk ring kept in L2 (not 46 MB a 512-row call through DRAM). Value columns of the state are independent; each dot is the same mma chain; reductions only in the unchanged prep. Offline: compiled IR == `fast_kda`'s op for op (3.7.1 and 3.8), interpreter bitwise at 1-8,192 rows and call splits; estimate -7..-11 (split) / -20..-35 (fused) us a token = +1-1.6% / +3-5% prefill. `docs/KDA-V2.md` **W10: bitwise passed; split +1.0-1.2% end to end, fused slower: off.** |
| 0410 `glm-sparse-v2` | `GLM53_TF_SPARSE_V2=0\|1` (load-time); `GLM53_TF_SPARSE_V2_CFG=stages,qkl[,qreg]` | `0`; `3,1,1` (FP8), `2,0,0` (bf16) | b12x bit 4's one-pass sparse latent attention (`b12x_attn.sparse_latent_one`, production's fast-prefill sparse attention since W9) through `sparse_v2._lsparse_v2`, a Gluon kernel with **the one-pass kernel's bits** (same 32-token tiles in list order, same mma chains and kWidth 2, both reductions on the reference's own `#mma` [1, 8], same element-wise ops; row-invariant; shares bit 4's snapshots, left out of the NVMe compat hash): the selected FP8 rows gathered with `cp.async` into a 3-slot ring through the 0290 page table (the reference loads each tile synchronously, one CTA an SM), dequantized once into shared memory, the scores on [2, 4] without the reference's duplicated warps (512 mma a tile, not 768), half of q in registers (~286 KB of shared-memory traffic a tile, not ~430). Offline: compiled IR == the reference's arithmetic op for op + equal PTX float counts (3.7.1 and 3.8), CPU emulator of v2's own source (cp.async ring, barriers modelled) == the reference in the interpreter bitwise, reference PTX unchanged. Estimate 1.8-3.1x the kernel, +3.5-6% prefill. `docs/SPARSE-V2.md` **W10: engine.py syntax error fixed; FP8 bitwise passed; kernel 1.2x, end to end +0%: off.** |
| 0420 `glm-draft-vocab` | `GLM53_TF_DRAFT_VOCAB=0\|N\|path[:N]` (load-time); `GLM53_TF_DRAFT_VOCAB_ARMS=mtp\|dflash\|all`; `GLM53_TF_DRAFT_VOCAB_FALLBACK=RATE[,WINDOW]\|off` | `0` (off); `mtp`; `0.03,256` | DECODE-PLAN E7 (FR-Spec style): the drafters' head over a ranked token list, each rank its first N / 2 ids of its vocabulary half (the shipped ranking `glm5_next/cuda/draft_vocab.txt`: in this public release 118,040 ids built from public, permissively licensed text by `bench/draftvocab_public.py`; the study below used a private list from 826 GLM-5.3-Flash agent steps, `bench/draftvocab.py` writes one from your own sessions), the same 4-bit rows gathered at load (24,576 rows a rank: 56.6 MB, a head read 0.89 -> 0.28 ms); draft columns map to token ids before the exchange, so draws use the target's keyed noise; the target's sampling is untouched (drafted == serial by construction; CPU tests through the real MTP loop, PTX 33 / 33). A request drafts over the whole vocabulary while > RATE of its last WINDOW absorbed tokens are unlisted (CJK: 63% coverage). Held-out coverage 98.1-98.7% at 24,576 a rank; estimate MTP rounds +1.2-1.7%, ~+0.5-0.8% at 1 stream overall, DFlash2 arm net negative. `docs/DRAFT-VOCAB.md` |
| 0430 `glm-draft-dump` | `GLM53_TF_DRAFT_DUMP=<dir>` (+ `_TOPK`, `_TAPS=bf16\|fp8\|0`, `_WHAT`, `_GIB`, `_TAP_LAYERS`); `GLM53_TF_MTP_WEIGHTS=<dir>` (load-time, both ranks) | off; off | Training records for the drafters (docs/DRAFTER-TRAINING.md): every committed row (prefill chunks in every mode, kind p; kept decode rows of batch slots, kind d) as the token, the final-normed row the MTP head reads, the DFlash2 taps and the target's top-32 log-probabilities over the whole vocabulary (each rank's half, one control all-gather, merged); rank 0 writes sharded files linked by prefix hashes. Reads only: replies byte-identical. 8.4 / 28.9 / 49.3 KB a token (no / fp8 / bf16 taps); estimate prefill +4-6%, decode < 1%. `_TAP_LAYERS` dumps another tap set (DFlash2-G's 9) with the DFlash2 drafter off. `GLM53_TF_MTP_WEIGHTS` (`mtpw.py`, outside `weights.py` so prepared folders stay valid when unset): a distilled MTP head's BF16 tensors replace the checkpoint's, sliced and quantized as the checkpoint's own. Offline only so far (CPU tests); `docs/DRAFTER-TRAINING.md` |
| 0440 `glm-decode-stream` | `GLM53_TF_DEC_EXPERTS=0\|1` (+ `_CFG=nt_gu,stages_gu,nt_dn,stages_dn`); `GLM53_TF_DEC_QMM=0\|1\|force` (+ `_CFG=gps,stages`, `_SERIAL=auto\|0\|1`); `GLM53_TF_DEC_PDL=0\|1`; `GLM53_TF_DEC_CTAS=N` | `0`; `4,4,4,4`; `0`; `1,4`, `auto`; `0`; `0` (every slot) | DECODE-PLAN E1 / E2 (+ E4's PDL): the decode-window routed experts (member tables of <= 64 columns: decode, verify, MTP, DFlash2, 4-slot windows) through two persistent kernels (`exl3_stream.cu`: warp-private cp.async rings 4-8 k steps deep across items, the gate/up and down epilogues run by the last item of each (expert, Hadamard block): 3 launches a layer, not 5) and the 4-bit dense matmuls of 1-64 rows through one (`q4_stream.cu`: persistent CTA ring, nibbles straight into the bf16 mma fragments, the K slices reduced in the launch, in order). **Same bits** as exl3.cu's grouped kernels + epilogues and as `_qmm` + `_reduce` for every row count (the same per-element instruction sequence; the dense kernel only under Triton 3.7.x, where `_qmm`'s epilogue is fully fused -- Triton 3.8 leaves some of it unfused, bucket-dependently). PDL: L2 prefetch of the first item, nothing else before `griddepcontrol.wait`. Offline: 67 tests (CPU emulator of both kernels vs the references with an order-sensitive mma model and 12 negative controls; sm_121 compile: 0 spills, 2-5 CTAs an SM). Estimate +9.7% 1 stream, +12.8% 4 streams (W11 ceilings). `docs/DECODE-KERNELS.md` **W12: bitwise and PDL race passed; microbench gate failed (E1 0.74-0.80x at every window, capped at ~160 GB/s by loads in flight; E2 1.1-2.1x on small L2-resident shapes, 0.82-0.91x on the five largest); E2 engine load -3.9% 1 stream, exact: off.** |
| 0450 `glm-gpu-round` | `GLM53_TF_GPU_ROUND=0\|1\|sample,resident,kda` (load-time, both ranks, checked); `GLM53_TF_GPU_ROUND_PEEK=1\|0` | `0`; `1` | DECODE-PLAN E6 + the KDA part of F3. `sample`: the keyed sampler, greedy pick and drafts' probabilities on the GPU (`gpusample.py`) with **the host's bits**: glibc 2.39's aarch64 `__log` / `__exp` ported instruction for instruction (its fmas, its tables), numpy's pairwise sum, float64 everywhere, no contraction; the host reads tokens only; a load check against the node's numpy turns it off on both ranks if anything differs. `resident` (needs 0370's `plan`): steady batched decode as GPU-resident rounds (`resident.py`): stage, sample, `Stepper.accept`'s rule, commits (every slot's replay in one launch), backlogs, MTP chains (`DepthOptimizer.mtp_next` on the device) and DFlash2 chains / `f_depth` on the device; the host processes round k - 1 while round k's forward runs and waits only on small pinned records at the draft decisions (PEEK=1; 0: a whole round ahead, stale bounds, padded rows); anything unsteady drains to the normal path. `kda`: commits folded into the next window's chain, every slot in one launch a layer (`kda.cu` `chain_slots`: same `update` routine and row order). Offline: sampler == host on 1,900+ random row sets (Triton 3.7.1 / 3.8), the port == Ubuntu's aarch64 libm on 15.9M inputs (unicorn), fake-model resident rounds == serial with two ranks in threads, sm_121 PTX checks. Estimate +1-2% 1 stream, +3-6% 4 streams (W11). `docs/GPU-ROUND.md` **W12: KDA fold bitwise 8/8; 1.05 x 10^8 keyed draws + 2.5 x 10^9 libm values == the node's numpy; engine exact in every mode (the synthetic tests fail on a harness that duplicates rank 0's candidates); sample 0%, resident -2.3% 1 stream / +0.5% 4 streams, 1 -2.3% / +1.2%, PEEK=0 -4%: off.** |
| 0460 `glm-prefetch-comm` | `GLM53_TF_L2PF=0\|1\|bulk\|lines\|touch` (+ `_MB`, `_SITES=a,f,o,e`, `_EXPERT_KB`, `_ROWS`, `_CHUNK_KB`); `GLM53_TF_ROCE_STRIPE_KB`, `_INLINE`, `_LAZY_CQ`, `_LEAN`, `_TRACE` | `0`; `4`, `a,f,o`, `64`, `64`, `32`; `0` each (0350's path) | **E3:** the next kernels' weights into L2 during decode's latency-bound windows, on EVERY forward path: 0040's all-gather sites now also in `batch.compute_multi` (production's multi-slot rounds never prefetched), plus `o` (the attention output projection, fired before the KDA chains / attention core) and `e` (the router's picked experts' first tiles, device-indexed); `cp.async.bulk.prefetch.L2` from a tiny side-stream kernel (no SM held), one-wave 4-bit launches prefetched as the head of every (tile, K slice) chunk; loads only, same bits. **E8:** small shards on one HCA (agreed at load), release / acquire ordering in the gather kernel, inline posts, lazy CQ reaping, a timestamp ring (`bench_roce.py --trace`); same bytes, same order. Estimate +2.4% 1 stream, +1.2% 4 streams (mid). `docs/PREFETCH-COMM.md` **W12: two bugs fixed (roce.py's `load()` lost `sources=` to a comment: every 0460 image fell back to NCCL; l2pf.cu's bulk prefetch unguarded for compute_80); L2PF 8 MiB +2.4% 1 stream, +1.2% 4 streams, exact: adopted (`GLM53_TF_L2PF=1`, `_MB=8`); 4 MiB +0.8%, site e 0%; RoCE knobs within bench noise: off.** |
| 0470 `glm-nonexpert-q8` | `GLM53_TF_NONEXPERT=q8`; `GLM53_TF_NONEXPERT_MAP=key=mode,...` (load-time, both ranks); `GLM53_TF_Q8_DECODE_CFG`, `GLM53_TF_Q8_TILE` (speed only) | unset (0001's modes); unset | non-expert weights in 8 bits (int8 symmetric, a bf16 scale per 128 inputs: 8.125 bpw, relative error 0.65-0.73% on the model's own weights against q4mse's 8.6-9.4%) with row-invariant decode (`qmm._q8mm`) and prefill (`fast_qmm._fq8`) kernels and 8-bit kv_b in the latent MLA kernels; the format per module class (`o_proj=q8,down=q8,lm_head=bf16,...`, `mtp.` for the MTP layer); prepared folders keyed by the resolved precision. Estimated: output side at q8 -6.5% decode / +0.82 GiB, all q8 -16% / +1.99 GiB (docs/QUALITY-PLAN.md) |
| 0490 `glm-api-context` | none (on: host-side API only) | on | The context limit where clients see it, and vLLM-shaped extras. `GET /v1/models` adds `max_model_len` / `context_length` (= `--context`, the engine's `limit`), `created`, `root`. A request past the limit gets HTTP 400 with OpenAI's `code: context_length_exceeded`, `param` and wording ("This model's maximum context length is N tokens. However, you requested M tokens (P in the messages, R in the completion)...", plus the `CONTEXT` to restart with), which opencode and Hermes Agent recognise and answer by compacting (the old message matched none of their patterns). An engine refusal (`ValueError`) in a stream is an `invalid_request_error` event. One `[tensorfold] context:` boot line (limit, KV cache, slots, pool). RigMark compatibility (docs/RIGMARK.md gap 1): `POST /tokenize` / `/v1/tokenize` (vLLM's `{count, max_model_len, tokens, token_strs}`; a string `prompt` with `add_special_tokens`, or `messages` rendered as a chat renders them; host only, no engine, nothing to rank 1) and one flat list of token ids as `prompt` on `/v1/completions` (validated: ints in the vocabulary, one prompt a request; used as given, so `usage.prompt_tokens` is exactly its length). Pool fix: a 0290 pool set to hold a whole request (`GLM53_TF_KV_POOL_TOKENS` >= CONTEXT) is sized to the slot capacity rounded up to a page (+1 page, ~2 MB a rank), so a request within 64 tokens of CONTEXT is no longer refused as "more than the whole pool" after passing the context check; no kernel change, replies are byte-identical. **W12: in-image tests pass; /v1/models 1,048,576, over-limit 400 `context_length_exceeded`, the pool edge (32 under CONTEXT) admitted; knobs-off b5 == b4 bit for bit: in production with image b5.** |
| 0500 `glm-vision` | `GLM53_TF_VISION=0\|1` (rank 0; rank 1 follows the data); `GLM53_TF_VISION_MAX_IMAGES`, `_MAX_TOKENS`, `_LOW_TOKENS`, `_FETCH`, `_FETCH_TIMEOUT`, `_MAX_BYTES`, `_MAX_PIXELS`, `_CACHE_MB`, `_PREP_MB` | `0`; `8`, `0` (processor's 8,000), `0` (detail ignored), `1`, `10` s, 20 MiB, 64 M px, `256`, `256` | Image input: OpenAI `image_url` parts (`data:` and http(s) URLs) on chat completions and `/tokenize`. The LM is NoPE throughout (no M-RoPE: `qk_rope_head_dim` 0, transformers passes plain `arange` positions no layer reads), so an image is only its rows: rank 0 preprocesses exactly as transformers' `Glm5NextImageProcessor` (bit for bit on CPU), runs the BF16 tower (`vision.py`: 24 blocks, 2D axial RoPE, full attention, 2x2 merge + merger; 1.13 GB on rank 0; ~5.4 TFLOP / est. 75-90 ms for a 1024x768 screenshot = 1,036 rows), and ships the rows to rank 1 after the prompt; `glue.embed` substitutes them inside the prefill. Each image row is a **virtual token id** >= 2^24 derived from the preprocessed pixels, so every token-keyed cache (snapshots, session store, prefix share, NVMe tier) tells images apart unchanged, and shares state past an image only for the same pixels. The checkpoint's template renders images as a "cannot see" reminder; parts are rendered as the vision template's `<\|begin_of_image\|><\|image\|><\|end_of_image\|>` (prompt ids == `Glm5NextProcessor` with the vLLM kit's template). Text requests: same bits, same protocol. `docs/VISION.md` **W14: works on the GPU, one bug fixed (MTP head graphs in the prefill absorb); W15: adopted (`GLM53_TF_VISION=1`, 64 / 64 MB caches, image b7).** |
| 0510 `glm-batch-graphs-lone` | `GLM53_TF_BATCH_GRAPHS=1\|0\|lone`; `GLM53_TF_VERIFY_SPLIT=0\|2..8` (+ `_FIRST=L`); `GLM53_TF_GRAPH_PROBE=N`; `GLM53_TF_RESIDENT_REPORT=N` | `1`; `0`; `0`; `0` | THEORY-2 ideas 1 and 2. `lone`: the batcher's CUDA graphs (0120 / 0200 rounds, 0450's resident rounds and batched MTP head passes) only for rounds with ONE active slot; rounds of 2+ slots run eagerly, exactly `0`'s path (W11 / W12: `0` is +1.9-2.6% at 4 streams but -2.7% for lone requests in slots 1-3); policy checked equal on both ranks. `VERIFY_SPLIT=N`: the engine's main graphs (the lone verify / serial step; 0050's long-context main graphs too) captured as N graphs over layer ranges and replayed back to back (`vsplit.py`), so the GPU starts on the first ~1/N of the ~1,650 nodes while the host launches the rest (W11: 1.1 ms a round inside `cudaGraphLaunch`, nsys-inflated). `GRAPH_PROBE=N`: every main-graph replay timed without per-node tracing (host `replay()` us per piece; first-node delay from a `%globaltimer` stamp captured as node 0 vs an eager stamp), one stderr line per key every N replays. `RESIDENT_REPORT=N`: 0450's resident rounds / entries / drains by reason every N drains. **Same bits** (same kernels in the same order; stamps write only the probe's buffer). CPU tests (`tests/test_batch_graphs_lone.py`, 44) + GPU test `tests/cuda/test_batch_graphs_lone_patches.py`; W13 (results/THEORY2-SESSION). **W16 (image b8): not adopted: `BATCH_GRAPHS=lone` 4 streams +0.4% (bar +0.7%), `VERIFY_SPLIT=4` 1 stream -0.5% (the real verify graph launch is ~11-14 us host; splitting only adds host time); same bits (RESULTS W16).** |
| 0520 `glm-hc-fused` | `GLM53_TF_HC_CUDA=0\|1` (load-time, both ranks); `GLM53_TF_HC_CUDA_ROWS=1..64`; `GLM53_TF_HC_CUDA_RG=1\|2\|4\|8`; `GLM53_TF_HC_CUDA_PDL=0\|1` | `0`; `16`; `4`; `0` | THEORY-2 idea 3: a hyper-connection boundary (`glue.hc_post` + the next `glue.hc_pre` = Triton `_hc_post`, `_hc_partial`, `_hc_finish`) in one CUDA launch (`hc_fused.cu`) with the Triton kernels' bits on every output (x, normed, xs, post, comb, partials), in `layer_forward` / `compute_multi` (both boundaries of a layer; the next layer's first hc_pre skipped), windows of 1-16 rows, not fast chunks; used only after a load-time bitwise self-check passed on both ranks (Triton 3.7.1 arithmetic; under 3.8 it stays off). Microbenchmark: `tests/cuda/bench_hc_fused.py` (gate: <= 50% of the three kernels, cold). docs/HC-FUSED.md. **W16: not adopted: the microbenchmark gate failed (fused / Triton 0.58x at 1 row, 0.92-0.98x at 2-4, 1.1-1.4x at 6-16, same bits), so no server load (RESULTS W16).** |
| 0530 `glm-http-pin` | `GLM53_TF_CPU_PIN=http[=cpus][;engine=cpus\|fast]`; `GLM53_TF_ROCE_TRACE_DUMP=<prefix>` (+ `_EVERY=N`, needs 0460's `GLM53_TF_ROCE_TRACE`) | off; off (`16 x` the ring) | THEORY-2 idea 6. `http`: 0370's pinning reduced to rank 0's HTTP request threads (`tf-http`: tokenize, stream, detokenize) on the non-fast cores (GB10: the A725s 0-4, 10-14); nothing else moves (`engine=` also keeps every other thread on the given / fast cores); without batching the request thread decodes on the engine's cpus and goes back. `ROCE_TRACE_DUMP`: each rank appends 0460's exchange trace ring (`Runtime.trace_rows`) to `<prefix>-r<rank>.jsonl` every N exchanges, checked once a round (`cpupin.tick`), so transport vs skew is measured in a running server without nsys (`results/THEORY2-SESSION/rocetrace.py`). Host scheduling / reads only: **same bits**. CPU tests `tests/test_http_pin.py`. **W16: not adopted: `CPU_PIN=http` with locked clocks 4 streams +0.7%, inside the noise band (a trace-only load +1.1%) and confounded with the clock lock; RoCE skew 42.6 -> 29.1 us an exchange, not a clean A/B (RESULTS W16).** **W19 (image b10): adopted (`GLM53_TF_CPU_PIN=http`): no measurable GPU-idle change (within noise), no cost, every gate passed with it on (RESULTS W19).** |
| 0540 `glm-replay-ttft` | `GLM53_TF_SNAPSHOT_BEFORE_END=1\|0` (load-time, both ranks, checked); `GLM53_TF_EMIT_FIRST=1\|0` (rank 0) | `1`; `1` | RigMark W13 fixes (docs/REPLAY-TTFT.md). A whole prompt's snapshot sits at its last grid point **strictly before** its end ((n - 1) // G * G; was n for a prompt ending on the grid, and n for every exact prompt), replacing the one at n: an identical prompt sent again (regenerate, retry, RigMark's immediate replay) resumes all but <= 64 tokens (W13: 0 / 16,384 / 49,152 of 8K / 32K / 64K). Off-grid prompts unchanged; one more chunk on a cold on-grid prompt (~0.2 s). A prompt's first token leaves when its batch piece ends instead of after the round's other pieces and verify launch (0370 `emit` held it): C4 TTFT est. 1.91 -> ~1.2 s. Same bits (a cut / snapshot on the 64-grid; scheduling). Multi-slot prefill analysed, not implemented (now 0560). **W15: adopted (image b7, both knobs default on): an identical 8K / 32K / 64K prompt resent resumes at n - 64, TTFT 0.21-0.26 s, byte-identical output (RESULTS W15 §6); C4 first tokens median 1.46 -> 0.93 s without thinking, unchanged with thinking; cold grid-aligned 8K -1.8%.** |
| 0550 `glm-memory-safety` | `GLM53_TF_SELECT_SCRATCH=grow\|max\|off`; `GLM53_TF_ALLOC_TRIM_GB=N` (+ `_S`); `GLM53_TF_ADMIT_MEM=available\|free` (+ `GLM53_TF_ADMIT_CACHE_KEEP_GB`, `GLM53_TF_ADMIT_FREE_FLOOR_GB`); `GLM53_TF_DROP_OWN_CACHE=1\|0` | `grow`; `2` (10 s); `available` (2, 1); `1` | W15 memory findings (docs/MEMORY-SAFETY.md). **The needle dip**: 0065's prefill key blocks (~512 B a prompt position, <= `SELECT_MB`) grow every 512-row sub-block, so torch's caching allocator took a new segment every ~4k tokens and kept the old ones: reserved memory grew quadratically (bound 5.8 GiB at 314k, measured 4.5; 16 GiB at 1M) until the next CUDA graph capture emptied the cache (the MTP head's new pool bucket right after the prompt). Now one scratch per device / stream, grown before a prefill to its largest block (<= 256 MiB), plus a trim of the unused cache before a batch piece over 2 GiB (at most every 10 s). **Page cache**: admission and the session store count MemAvailable - MemFree less Dirty / Writeback / max(Mapped, 2 GiB), keep a 1 GiB free-now floor, reserve the prompt's scratch growth and log waits (W15: C4 serialized during a 36 GB copy). Same bits (views with the same shape / strides / alignment, every element written before read; trims and admission change addresses and scheduling only). CPU tests `tests/test_memory_safety.py`; **W17 (image b9): built into production, not adopted (knobs set off in `config/prod.env.example`: `GLM53_TF_ADMIT_MEM=free`, `GLM53_TF_SELECT_SCRATCH=off`, `GLM53_TF_ALLOC_TRIM_GB=0`; `DROP_OWN_CACHE` stays on).** GPU tests 25/25 on both nodes, the page-cache probe passed, the 314k needle's dip 1.7 instead of 4.0 GiB, same bits; but the first long prefill after a burst of short requests ran 1,454 / 1,550 tok/s (-9.8% / -3.5%) and 1,512 / 1,541 after the needle, in 2 of 4 prefill pairs, never with it off. Next: the same sequence with `GLM53_TF_ALLOC_TRIM_GB=0` only (RESULTS W17) **W18: the slow prefills came from the trim (`ALLOC_TRIM_GB=2`); scratch only was as fast as off. W19 (image b10): scratch adopted (`GLM53_TF_SELECT_SCRATCH=grow`, trim 0, admission `free`): 314k needle minimum 6.58 / 6.31 -> 8.68 / 8.51 GiB, stress minimum 7.75 / 7.61 -> 8.34 / 8.09 GiB with the NCCL change; 0550 v2 (planned trims) proposed in MEMORY-SAFETY.md §6.** |
| 0560 `glm-multi-prefill` | `GLM53_TF_MULTI_PREFILL=0\|1` (load-time, both ranks, checked); `GLM53_TF_MULTI_PREFILL_ROWS=N` (checked); `GLM53_TF_MULTI_PREFILL_WAIT_MS=ms` (rank 0) | `0`; the lean rows; `10` | RigMark C4 / agent bursts (docs/MULTI-PREFILL.md). A round's prefill pieces (fast lean, same knobs, <= ROWS piece rows) go through ONE forward: each member's rows contiguous and cut into its own lone sub-blocks (KDA / DSA / carries / commit / snapshots / marks / MTP absorb / taps / head / sample per member, in the lone order), the router and routed experts once over every row (the ~74 GB expert read paid once), in 0082's, 0084's and 0320's loops; each piece then ends in `Batcher._piece` as alone. Rank 0 takes more pieces a round when the fair share allows one (fewest left first) and, idle, waits <= 10 ms for more arrivals. **Same bits** a member as alone (0085's row-independent fast kernels; groups not formed under FAST_EXPERTS auto / once+min rows). Est. C4 first tokens ~0.6-0.8 s (thinking off / on; b7 0.93 / 1.9 s), bursts of short prompts 1.7-2.8x prefill tok/s. **W17 (image b9): adopted (`GLM53_TF_MULTI_PREFILL=1`).** 92/92 grouped replies == the same request alone on the real model, batchexact, transcripts, N1, 13/13 glmbench hashes; C4 per-stream first tokens (RigMark shape, thinking low) 1.64 -> 0.82 s, thinking off 0.92 -> 0.63, C2 0.69 -> 0.57; prefill and decode unchanged; -0.5 GiB on the worker in the 4 x 250k stress. RigMark (3 runs): C4 aggregate 91.0 tok/s, C4 TTFT 0.89 s. The GPU test's 4 `cached > 0` failures are its own expectation (same `cached` with the knob off; RESULTS W17) |
| 0570 `glm-dense-size-switch` | `GLM53_TF_DEC_QMM_MAXMB=X` (MiB; `GLM53_TF_DEC_QMM_MAX_MB` is the same knob); `GLM53_TF_DEC_QMM_TABLE=1\|0`; `GLM53_TF_DEC_QMM_EXCLUDE=NxK,...` | `0` (off); `1`; none | THEORY-2 item 7 / W16 (0440's dense kernel wins cold on every shape up to 3.54 MB, loses 10-18% from 14 MB up): 0440's `q4_stream.cu` only for 4-bit matrices of <= X MiB (`Q4.nbytes`), `_qmm` + `_reduce` for every larger one; the size knob turns 0440's dense hook on by itself and keeps its Triton 3.7.x guard. At 3.5: KDA f_b / g_b, index k (<= 4 rows), shared down, index q_b, the (1,16,8) 1024 x 4096 grid, DSA kv_b (PDL column only); shared gate/up (4.72 MB) and everything larger stay on `_qmm`. `TABLE`: W16's cold placement per shape (groups a stage, stages, whole-tile / split, row cap; with and without `GLM53_TF_DEC_PDL`). **Same bits** (q4_stream == `_qmm` for every shape and row count; the switch picks by (shape, rows) only). Est. -0.15 to -0.23 ms a 1-stream round (+0.3-0.4%), 4 streams +0.15-0.25%: below one A/B's noise, adopt on an in-situ trace. CPU tests `tests/test_decode_size_switch.py` (16); `docs/DECODE-KERNELS-2.md` §4. **W19 (image b10): not adopted (off): the cold bench passed (every switched shape 1.0-1.7x faster), but in situ the switched shapes took 1.7x `_qmm` + `_reduce`'s time (dense decode +3-5%); removing it gave +1.2% 1 stream (RESULTS W19 §3).** |
| 0580 `glm-expert-loads` | `GLM53_TF_DEC_EXPERT_LOADS=0\|1`; `GLM53_TF_DEC_EXPERT_LOADS_CFG=ld,nt,pd[/ld,nt,pd]` (ld `nc`\|`w32`\|`cpa`, nt 8\|4, pd 1\|2\|4); `GLM53_TF_DEC_EXPERT_LOADS_PDL=0\|1` | `0`; `nc,8,2`; `0` | THEORY-2 item 8 / W16 (`littles.cu`: `ld.global.nc.v4` into registers reaches 232.7 GB/s at 4 KB an SM in flight; cp.async needs 6 KB, TMA bulk 8-16 KB; E1's own ring shape alone reaches 234, so 0440's E1 lost in its persistent walk, not its loads). The two grouped launches of every <= 16-row routed window (decode, verify, MTP, DFlash2, 4-slot) through `exl3_ld.cu`'s `ld_kernel`: exl3.cu's grid, work items, per-warp K ranges, mma chain, warp sum order and Z rows, with a new load path -- the prologue in one round trip (grouping via `ld.global.cg`) with the first k steps of trellis words issued before the member rows land, a PD-deep register ring of 128-bit no_allocate loads a warp (lane-major v4 -> a warp staging area in the warp's own slice of `red` -> word `lane` of each tile), the A fragments PD steps ahead; `w32` / `cpa` are grouped_kernel's 32-bit loads made deeper / a cp.async ring. Default: 163 registers, 3 CTAs an SM (as today), 24 KB an SM in flight, 0 spills. `PDL`: the prologue before `griddepcontrol.wait` overlaps rot_in's / gateup_epilogue's tail (plain launches; X / Xd read, Z written after it). **Same bits** as exl3.cu for every row count and setting. rot_in / epilogues / grouped_loop unchanged. Est. 203 -> 220-230 GB/s in the 1-stream windows: -2.0 to -3.1 ms a round (+4.0-6.2%), 4 streams +0.4-3.1%; gate: probe 3 (load path alone) >= 220 GB/s and one config >= 1.05x grouped_kernel cold on U 8-22. CPU emulator + host tests (58), sm_121 compile / PTX / SASS (5); `docs/DECODE-KERNELS-2.md`. **W19 (image b10): adopted (`GLM53_TF_DEC_EXPERT_LOADS=1`, `_CFG=nc,8,1`): same bits; load probe 226-231 GB/s; the strict 1.05x kernel bar missed on some windows, but in situ the routed experts' decode time fell 4.1-4.6% (1 stream) / 2.9-3.1% (4 streams), the bulk of W19's +3.3% / +3.0% (RESULTS W19).** |
| 0590 `glm-expert-prefill-v2` | `GLM53_TF_FAST_EXPERTS=fat2`; `GLM53_TF_FAT2_CFG=gu,dn` (0 = 128 members / 4 stages, 1 = 64 / 4, 2 = 128 / 3); `GLM53_TF_FAT2_TICKET=1\|0`; `GLM53_TF_FAT2_CTAS=N` | off (`fat`); `0,0`; `1`; `0` (every SM) | KINDLING-AUDIT adoption #2 / ROOFLINE gap 1: the fast-prefill routed experts as one persistent software-pipelined kernel a projection (`exl3_fat2.cu`): 256-thread CTAs (one an SM, 168 registers, 82 KB), 128-member items of the expert-sorted member lists from a device plan, claimed by ticket one item ahead, a 4-stage `cp.async` ring (member rows gathered by pair index, trellis words) that runs ACROSS items, the epilogue in its own buffer. **Same bits as fat** (fat's mma chain, K order and epilogue; emulator + PTX + SASS proofs), row-independent, shares snapshots. Expected 0.77-0.86x fat at 2,048 / 4,096 rows (DRAM-bound; the 0.75x gate sits on the DRAM floor), 0.65-0.76x at 8,192. **W19 (image b10): not adopted (off): same bits, but fat2 is 1.16x / 1.19x slower than fat at 2,048 / 4,096 rows (contended makespan 1.07x / 1.09x) (RESULTS W19 §1).** |
| 0600 `glm-upstream-ports` | `GLM53_TF_DISCONNECT=0\|1`; `GLM53_TF_DISCONNECT_POLL_MS=N`; `GLM53_TF_VISION_FETCH_HTTP=0\|1`, `_FETCH_PRIVATE=0\|1` (local testing only), `_FETCH_TOTAL_S`, `_PREP_SLOTS`, `_PREP_WAITERS`, `_PREP_WAIT_S` (rank 0) | `1`; `250`; `0`, `0`, `30`, `16`, `128`, `60` | Host-only ports from upstream TensorFold (MIT; docs/UPSTREAM-050-AUDIT.md ranks 1-4, 6, 9; docs/UPSTREAM-PORTS.md). **Client disconnect** (24afe5e on our batcher): a socket check on every completion, streamed or not; the HTTP thread waiting in `Batcher._collect` polls it while queued / prefilling / decoding silently, so a departed client's request is dropped from the queue or cancelled through the round plan rank 1 receives (a non-streamed request no longer runs to `max_tokens`); a callback exception cancels the slot and is raised after; nothing more is written to a departed client. **Request hardening** (2bc35c4): `check` / `tokenize` exceptions are 400s (template refusals say so), strict sampling / length fields and `chat_template_kwargs` before a stream's headers, non-UTF-8 bodies, redacted tracebacks for 400 / 500. **`kill -USR1`** stacks on both CUDA ranks (a586f3d). **Image URLs** (391713e) for 0500: HTTPS on 443 to public addresses only (private / loopback / link-local / CGNAT / metadata / IPv6 equivalents refused), each redirect re-checked, the connection pinned to the checked address, declared JPEG / PNG / WebP / GIF, bounded size and time, <= 16 preparations at once then 503. **Small bundle**: `return_token_ids` (e6be5ff), `completion_tokens` exact at a stop string (88407d1, our 0160 scope), GLM tool arguments one bracket short closed (c30a1a9, #87). **`/health`** gains upstream's live totals (51b098d / c4bf25f) beside 0150's fields. **Same bits**; malformed requests that were accepted by accident are now 400s, image URLs that are not public HTTPS are 400s (UPSTREAM-PORTS.md 1.2, 1.4). Host tests (177). **W19 (image b10): on by default in production; every check passed (a departed client's slot freed 0.18 s after the close, 400s, USR1, image URL hardening, /health totals; RESULTS W19 §2).** |
| 0610 `glm-structured-output` | `GLM53_TF_GRAMMAR=0\|1` (load-time, both ranks, checked); `GLM53_TF_GRAMMAR_THREADS=N` (rank-local, 0-64) | `0`; `4` | Structured output (docs/STRUCTURED-OUTPUT.md): OpenAI `response_format` (`json_object`, `json_schema`), vLLM `guided_json` / `_regex` / `_choice` / `_grammar` / `structured_outputs`, and tool calls (`tool_choice` required / a named function, or `strict` tools: GLM's `<tool_call>` markup, arguments held to each schema) enforced with xgrammar 0.2.8 token masks (Apache-2.0; the Dockerfile installs it, torch pinned) on the batcher (prod) and the lone loops. Drafts the grammar rejects are cut before the forward; each kept row is masked by its own path (-inf on this rank's vocabulary half before the keyed top-k; the RoCE candidate gather unchanged); the masks are filled on worker threads while the forward runs (only the ~1-2 us-a-draft cut precedes it). Thinking: the grammar starts after `</think>` (the reasoning is the unconstrained reply's). 400s before any header; a mid-reply grammar failure ends only its request (500). **Exact**: constrained drafted == `"draft": false`, a slot among 4 == alone, 2 ranks == 1; unconstrained requests take today's path (knob on == off; no header byte changes). Est. ~0.1-0.3% of a constrained round, 0 otherwise. CPU tests (129) + microbench; **W19 (image b10): adopted (`GLM53_TF_GRAMMAR=1`): unconstrained replies unchanged (hashes), schemas 32/32 and tools 6/6 valid, drafted == undrafted == 4 concurrent; a 50-object `response_format` task -0.4% tok/s (RESULTS W19 §2).** **W20: `GLM53_TF_GRAMMAR_THREADS=8` not adopted (exposed mask wait 0.67 vs 0.65 ms a window: no gain; stays 4).** |
| 0620 `glm-tool-calling` | `GLM53_TF_TOOL_FIXES=` comma list of `history`, `reasoning`, `choice`, `args`, `thinkcalls`, `grammar` (`all` = every one but `grammar`); `GLM53_TF_KEEP_REASONING_ENTRIES=N`, `GLM53_TF_KEEP_REASONING_MB=N` (rank 0) | empty (off); `1024`, `32` | Tool-calling fixes for multi-step chains (docs/TOOL-CALLING.md). `history`: `content: null` no longer renders as the text `None` before an assistant turn's calls (or in a null tool result), `reasoning` is read as `reasoning_content`, empty `arguments` render as `{}` instead of a 400. `reasoning`: the reasoning of each reply that called tools is kept (by our call ids, and by a signature of the calls + last user message for clients that renumber ids) and put back into a later request's assistant turn sent without it, so GLM's interleaved thinking sees its own earlier steps. `choice`: `tool_choice` `none` offers no tools, a named function only that tool (unknown: 400), `parallel_tool_calls: false` keeps the first call (upstream 0.5.0's rules). `args`: argument values typed by the whole schema (type lists, anyOf / oneOf, enum, quoted numbers). `thinkcalls`: complete calls at the end of an unclosed think block, in a reply that ends on `<\|observation\|>`, are returned as calls. `grammar` (with 0610 on): every auto tools request held to its schemas. **Host only, same bits**: no engine, kernel or token change; requests without tools, or with the knob off, render and parse exactly as before. Host tests: `tests/test_tool_fixes.py` (46). **W20 (image b11): adopted (`GLM53_TF_TOOL_FIXES=all`): every W20 gate passed with it on (tool-free replies and hashes unchanged); tool-eval-bench on production, thinking off, one run: 90, multi-step chains 8/8 (results/tooleval/20260930-teb-off-fixes-PARTIAL; spark-bench not finished).** |
| 0670 `glm-sessdisk-index-tail` | none | on | Issue #11 follow-up. The NVMe session tier's in-RAM index (`sessdisk.DiskIndex`, the same on both ranks) kept every entry's token ids as a Python `list[int]` (8-36 B a token, one list per entry even when entries share a prefix), so host RSS crept up as the tier filled (`GLM53_TF_SESSION_DISK_GIB`, 64 in prod): the DeepSeek kit's G18 drift. An entry now keeps `DiskIds`: its length, the page chain it already had, an 8-byte digest per 64 tokens of each full page and the < 256-token int32 tail (~1.2 B a token in total with the page keys, measured). `find` matches exactly as before (length, chain, tail); `lcp` (fork marks only) is exact to 64 tokens inside a full page, and the marks round it to the snapshot grid or the page, both multiples of 64, so they are unchanged. The full ids live in the entry file's header, written from the snapshot and read back by the load; the boot scan holds them as int32 arrays. `tests/test_sessdisk_index.py` |

## 0001 — EXL3 non-expert weights in 4 bits

**Problem.** The EXL3 checkpoint stores the routed experts in 4-bit EXL3 (mcg codebook) but every other weight
in BF16: the KDA and DSA attention projections, the DSA indexer projections, the dense MLPs, the shared
experts and `lm_head`. Per rank that is 10.7 GB
read on every decode step, against 5.0 GB for the all-4-bit MLX checkpoint TensorFold was tuned on. One-row
verify took 57 ms.

**Change.** `GLM53_TF_NONEXPERT` controls how those BF16 matrices are stored at load time
(`families/glm5_next/cuda/weights.py`):

- `bf16`: upstream (`make_b16`).
- `q4`: the engine's existing MLX-style affine 4-bit format, groups of 64 along K, `q = round((w - min) / scale)`
  (`qmm.quantize4`, already used upstream for draft-only copies).
- `q4mse`: same format and same kernels, but each group's `[min, max]` range is shrunk about its centre by the
  factor from `{1.0, 0.97, ..., 0.76}` with the smallest squared reconstruction error (computed with the
  bf16-rounded scale and bias the kernel will actually use). Factor 1.0 is in the grid, so the error is never
  worse than `q4`. Load-time cost only.

The routed experts are untouched, and so are the router, norms, convolutions and the other small tensors. With a 4-bit `lm_head`, the separate 4-bit draft copy of the head is no longer
made (the head already is that copy).

`engine.py`: upstream maps the `auto` drafter policy on an EXL3 checkpoint to `EXL3_AUTO`, which was measured
with BF16 non-expert weights (where an MTP step is expensive). With 4-bit non-expert weights the MTP head costs
what it costs on the MLX checkpoint, so `auto` again chooses between MTP and DFlash2 the way it does there.

**Result.** One-row verify 57 ms -> 30 ms; serial decode 17 -> 33 tok/s; drafted decode 1.2-1.5x over upstream
on every cell (`docs/RESULTS.md`). MMLU-200: bf16 87.0%, q4mse 88.0%.

**Exactness.** Drafting and verification read the same stored weights, and the verify step decides every
emitted token, so a drafted reply is still byte-identical to the serial reply of the same configuration. q4mse
is not bit-identical to bf16: it is a different (re-quantized) model, which is why its quality is checked
separately (MMLU, refusals).

## 0002 — GLM tool-call parser

**Problem.** GLM-4.5 through 5.3 write tool calls as

```
<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la</arg_value></tool_call>
```

Upstream's CUDA server only parsed the Qwen form (`<function=name><parameter=k>v</parameter></function>`), so
GLM tool calls came back as plain text and agents (opencode) could not use tools.

**Change.** `cuda/server.py` gains `_glm_call`, following vLLM's `glm47` parser: the name is the text before the
first `<arg_key>`; a value stays text when the tool's JSON schema types that parameter as `string`, and is
parsed as JSON otherwise (falling back to the text). A block whose name is empty or contains markup is left as
text. The Qwen format still parses.

**Exactness.** Output parsing only; the tokens are unchanged.

## 0003 — prefill chunk rows

**Problem.** Upstream prefills in fixed 64-row chunks. Every chunk reads every weight once, so on a
bandwidth-bound GPU larger chunks prefill faster. The attention and quantized matmul paths refused more than
128 rows.

**Change.** `GLM53_TF_PREFILL_ROWS=N` sets the chunk size (`engine.py`). `attention.py` accepts any row count
(the grid already tiles rows by the block size, and rows are independent) as long as the scratch holds the
window; `qmm.split_k` picks the 128-row tile for longer windows instead of raising. Costs about 5 MB of window
buffers per row on the EXL3 checkpoint.

**Exactness.** A row's result never depends on which chunk it is in, so the committed KV state has the same
bits for any chunk size; replies and resumed conversations match 64-row prefill bit for bit (tested).

## 0004 — sparse long-context prefill

**Problem.** Past 2,051 tokens GLM's DSA attention runs sparse (top-k pools). Upstream sized the sparse
kernels' partial buffers for 128 rows (the verify windows), so a prefill chunk longer than 128 rows past that
point wrote out of bounds: an illegal memory access at 8k context with big chunks. Token selection also
looped over rows in Python.

**Change.** `sparse.py`: the chunk partials (`po`, `pm`, `pl`) are sized for the window's `R` rows and the row
stride is passed to `_sparse_chunks` / `_sparse_merge`. `select_tokens` builds every row's list at once with
tensor ops (the selected pools' tokens ascending, then the visible tokens of the incomplete last pool; a dense
row gets count 0), producing the same lists as the loop.

**Exactness.** The row stride only places partials in memory; the arithmetic and its order are unchanged. The
vectorized selection is compared against upstream's loop, kept in the test as the reference.

## 0050 — long-context decode (graphs past 2,051 tokens, bounded indexer)

**Problem.** Graphs replayed only while `pos + R <= dense_limit`; every later step ran eagerly (~1,500 launches
from Python), and `select_tokens` scored and sorted `capacity / 4` pools in every DSA layer whatever the length.

**Change.** (a) `sparse.pool_bucket(pos + R, cap)`: the pools to score, `(pos + R) // 4` rounded up to a power of
two (>= 1024, <= capacity / 4); used by every eager selection (prefill chunks too). (b) `sparse.select_tokens_dev`:
the same `_scores` kernel over the bucket, then unique int64 keys (`_keys`: score order, ties to the lower pool,
-0.0 == +0.0, NaN highest), `torch.topk(sorted=False)` + `torch.sort` of the 512 indices, and `_expand` (tokens and
counts from the device position), all into `LongScratch` buffers sized at init (8 rows); `sparse_attention` takes
preallocated partials. `Engine.forward`/`.mtp` replay `Graphs.long[("main", R, parity, bucket)]` /
`[("mtp", n, bucket)]` when every row is past 2,050 and R <= 8; the first step of a key runs eagerly through the
same code (warm-up and result) and is then captured. Prefill chunks and mixed windows stay eager. A failed capture
logs once and falls back to eager. `GLM53_TF_LONGCTX_MAX_GRAPHS` (default 256) caps the lazily captured graphs.
Both ranks must agree on the knob (checked at start).

**Exactness.** Pools past `(pos + R) // 4` score -inf and sit after every kept pool in the stable descending
order, so the top 512 are unchanged; the keys are unique and order pools exactly as that sort does, so any top-k
over them returns the same set, listed ascending as before. Scores come from the same kernel. Everything is per
row, so verify rows keep the bits of serial steps. Tests: `tests/cuda/test_longctx_patches.py`.

## 0060 — latent (absorbed) MLA KV cache

**Problem.** The 11 DSA layers and the MTP layer cache decompressed per-head keys and values: 32 local heads x
(256 + 256) x 2 B = 32 KB a token a layer, **384 KB a token a rank** for 12 layers (plus 6.75 KB of indexer
caches). The ~35-40 GB free per rank hold ~100k tokens; 1M would need ~400 GB. Every decode row past 2,051 tokens
gathers 2,051 selected tokens x 32 KB = 67 MB per layer (0.74 GB over 11 layers, ~3 ms).

**Change.** `GLM53_TF_LATENT_KV=1` (new module `families/glm5_next/cuda/latent.py`). GLM-5.3 has no rotary part
in its heads (`qk_rope_head_dim = 0`), so `k_h = W_k,h c` and `v_h = W_v,h c` with `c = kv_a_layernorm(kv_a(x))`
(512 wide), and attention runs in latent space: `score = (W_k,h^T q_h) . c`, `o_h = W_v,h (sum p c / sum p)`.

- Cache: one bf16 latent row a token a layer (`State.kc[i]` is `[capacity, 512]`, `State.vc is State.kc`: keys =
  values; the MTP head's cache likewise). 12 x 1 KB + 6.75 KB indexer = **18.75 KB a token a rank (20.8x)**:
  100k tokens = 1.9 GB, 1M tokens = 19.2 GB a rank.
- `absorb`: `q'_h = bf16(W_k,h^T q_h)` from kv_b's key rows **as stored** (4-bit groups of 64, or BF16 on an EXL3
  checkpoint with `GLM53_TF_NONEXPERT=bf16`): each program dequantizes 64 x 64 tiles exactly in fp32 and multiplies
  in fp32 (IEEE, FMA). No absorbed copy is kept, so weight memory and the weight bytes a step reads are unchanged
  (the kernel replaces the key projection).
- Dense attention (`attention_latent`): 16 (row, head) queries a tensor-core tile against 32-row tiles of the
  latent, which serve as keys and values at once (MQA-style: a tile is read once for 16 heads), online softmax,
  512-key chunks fixed by absolute position, merged in chunk order into the fp32 normalized latent `u`.
- Sparse attention past 2,051 tokens (`sparse_latent`): the same tile over each row's own top-k tokens
  (`sparse.select_tokens`, and patches/0050's device-side `select_tokens_dev` in captured long-context steps),
  16 heads of the row a tile; preallocated partials for windows up to 8 rows (captured steps allocate nothing).
- `expand`: `o_h = bf16(W_v,h u_h)` from kv_b's value rows as stored (same fp32 dequant and FMA), then `o_proj` as
  before.
- Scratch: chunk partials are sized by the dense limit (dense attention never runs past it), not by the capacity:
  ~0.25 GB a buffer set at 512 prefill rows whatever the context (the expanded scratch grows with capacity:
  3.3 GB at 100k).
- Dispatch: one branch at the top of `forward.dsa_block` (serial, verify, prefill, MTP, graphs, 0050's long
  steps) and one at the top of `batch._dsa` (patches/0030 rounds). `decode.Engine` reads the knob before any
  `State`/`Buffers` and prints the latent KV bytes a token (rank 0); `GlmEngine` checks both ranks agree.

**Exactness.** The arithmetic differs from the expanded path (k and v are never rounded per head; q' is), so
replies differ from `GLM53_TF_LATENT_KV=0` in the last bits: a new engine configuration. Within it every
guarantee holds for the same reasons as before: each (row, head) is one row of a fixed 16-row tile whatever its
tile-mates; absorb/expand reduce in a fixed order per row; chunks are fixed by absolute key position and merged in
order; empty chunks and rows past a row's causal limit are exact no-ops; no atomics. So a verify row gets the bits
of the serial step (drafted == serial), prefill chunk size changes no bit, and a resumed prefill equals a fresh
one (snapshots still leave the attention caches in place). Accuracy against an fp32 reference on the real shapes
(CPU emulation, round-to-nearest bf16): latent 2.7e-3 / 4.3e-3 relative error vs expanded 2.8e-3 / 4.8e-3.

**Expected speed (arithmetic, unmeasured).** Past 2,051 tokens (8k and 32k alike: the selection is 2,051 tokens
either way) a verify row reads ~46 MB of latent (23 MB if the second head tile hits L2) instead of 0.74 GB:
**about -3 ms per window row** (R = 1: -3 ms of ~30 ms, +10%; R = 4: up to -12 ms) and -0.29 ms per MTP draft step.
Below 2,051 the saving grows with the context: -3.1 ms a window at 2,000 tokens. Prefill: attention FLOPs double
per head (512-wide dot products), and absorb/expand add ~0.1 TFLOP a 512-row chunk on FMA; estimated +1-2% of a
chunk, not measured.

**Memory at CONTEXT=32768, 1024-row prefill buffers (per node, arithmetic).** KV 12.21 -> 0.59 GiB; attention
scratch (two buffer sets: the model's and the MTP head's) 4.26 -> 1.10 GiB, because the expanded scratch grows 128 KB
per capacity token at 1024 rows and the latent one is capped by the dense limit. **Saving 14.8 GiB**: today's
9-11 GiB headroom becomes ~24-26 GiB. At 18.75 KB a token that would hold ~1.3-1.4M tokens; after the eager
selection's transient scores and sorts at 1024 rows (~6 KB per capacity token) and the DFlash2 drafter's own
capacity-sized cache (check its config), expect ~0.8-1M.

**Prefill (estimate, from the measured 28k profile: sparse attention 30% of a 1024-row chunk).** The expanded
sparse kernel puts one query in a 16-row tile (1/16 of each MMA used) and gathers 2,051 tokens x 32 KB per row per
layer (67 MB). The latent kernel is heads-as-rows: 16 heads of one row fill the tile, and one 32-row latent tile
load serves them all, so each row gathers 2 x 2,051 x 1 KB = 4.2 MB (16x fewer bytes) and issues ~8x less MMA
work. Taking a 5-10x faster sparse phase and +1-2% for absorb/expand: a 28k chunk costs 0.73-0.78 of today's,
so **prefill throughput +25-35% at 28k**. At 8k about the same per sparse row: every row past 2,051 selects
2,051 tokens whatever the context, and the dense rows below it also read 32x fewer bytes. So also +25-35% at 8k,
slightly less if the first 2k rows are a larger share of the chunk.

**Verified on the head node (GB10).** `tests/cuda/test_latent_patches.py` passes, together with `test_patches.py`,
`test_longctx_patches.py` and `test_knob_patches.py` (99 passed) on the full set 0001-0090. Kernel times per layer
(one rank's 32 heads; latent 512; sparse rows select 2,051 tokens of a 32k context; dense rows end at 2,051):

| Rows | Path | Expanded ms | Latent ms (BM = 16, default) | Latent ms (BM = 32) |
| ---: | --- | ---: | ---: | ---: |
| 1 | dense / sparse | 0.305 / 0.311 | 0.051 / 0.057 | 0.068 / 0.072 |
| 2 | dense / sparse | 0.307 / 0.577 | 0.054 / 0.058 | 0.068 / 0.072 |
| 4 | dense / sparse | 0.309 / 1.077 | 0.056 / 0.058 | 0.068 / 0.073 |
| 8 | dense / sparse | 0.313 / 1.931 | 0.091 / 0.103 | 0.070 / 0.076 |
| 1024 | dense / sparse | 4.29 / 58.7 | 9.36 / 11.2 | 7.53 / 8.49 |

Decode: -0.25 ms (1 row) to -1.8 ms (8 rows) of attention per layer past 2,051 tokens, about -3 ms to -20 ms a
window over 11 layers. Prefill: a 1024-row sparse chunk's attention drops 705 -> 135 ms over 12 layers. At the
measured 30% share that is ~0.76x the chunk time, **~+30% prefill throughput at 28k** (and similar at 8k). Dense
1024-row chunks (the first 2,051 tokens only) are slower, +5 ms a layer: twice the dot-product width, compute
bound. BM = 32 (32 heads a tile) gave the same bits on these inputs. It is faster from 8 rows up (-2.7 ms a layer
per 1024-row sparse chunk, ~2% of a chunk) but slower for 1-4-row decode windows (+0.015 ms a layer, +0.18 ms a
step), and it must be one fixed value because decode windows and prefill chunks compute the same positions. So the
default stays 16. The knob stays `0` by default until the `exact` suite passes on the real model.

## 0065 — 1M-token memory and the prefill indexer

**Problem.** CONTEXT=1000000 was OOM-killed at load: besides 0060's 18.75 KB a token, the DFlash2 drafter reserved
its context K/V for the whole capacity (10 KB a token a rank), so 29.5 KB a token was reserved. Past 2,051 tokens
every prefill chunk scored all its rows against a power-of-two bucket of pools, one program per (row, 64 pools)
each loading its own copy of the pool tile, and stable-sorted the [rows, bucket] scores (~9 GB of transient
scratch at the end of a 1M prompt, then held by the allocator); the indexer was 14% of a 112k prefill.

**Change.** (a) `dflash2.py`: every drafter layer is a sliding-window layer, so its context lives in a ring of
4,096 slots (`GLM53_TF_DRAFTER_RING`); the attention loop starts at the first 64-key tile any row sees. (b)
`forward.State`: the 11 model layers' index keys and gates in a ring of `index_ring_rows(rows)` rows
(`GLM53_TF_INDEX_RING`; the kernels address rows at position mod ring); pool keys and the MTP layer's caches stay
full. `commit` keeps the last 3 committed rows in a trailing slot of `conv` (which snapshots copy) and `set_pos`
writes them back on a restore. (c) `sparse.select_tokens`: windows of more than 8 rows go through
`select_pools_blocked` (`GLM53_TF_SELECT=blocked`, `GLM53_TF_SELECT_MB`): row blocks within the budget, each scoring
only up to its last row's pools; `_scores_rows` runs `_scores`' per-row code for 16 rows against one pool tile and
writes 0050's key (upper half, int32); the 512th largest key per row by `torch.topk`, then `_gather_sel` lists the
pools above it and the lowest-index ties, in pool order. Details and the memory table: `docs/MEMORY-1M.md`.

**Exactness.** Ring slots hold every position a kernel reads (window + 3 rows; window + block + a tile), and the
drafter's skipped tiles are fully masked no-ops of the online softmax, so every bit is unchanged; a snapshot
restore brings back the incomplete pool's rows. The blocked scores are `_scores`' bits (same per-row code and
shapes; checked on the device at start, `blocked_ok`, which falls back to the sort); pools past a block's last row
are -inf with higher indices than every kept pool; the threshold + lowest-index ties is exactly the first 512 of
the stable descending sort. Decode/verify/MTP windows and 0050's graphs are unchanged. One difference: a snapshot
resumed after the drafter wrote a whole ring past it masks the positions it lost, so drafts (never replies) differ
from a fresh prefill's. Tests: `tests/cuda/test_1m_patches.py`.

## 0080 — fast prefill (the chunk-grid rule, fused EXL3 experts)

**Problem.** Every prefill chunk ran the decode kernels, which are row-invariant (a row's bits never depend on its
chunk-mates). That is what drafted == serial and resumed == fresh rest on, and it rules out large-M GEMM tiles,
split-K choices by M, a chunked KDA scan and reduced-precision gathers. Measured at 1024-row chunks (patches/0005,
`results/M-P1c.profile.log`): 671 / 542 / 493 tok/s at 1.8k / 7k / 28k tokens; per chunk the routed experts take
520-580 ms (26-44%), sparse attention 450-610 ms past 2k, all-gathers 140-190 ms, the KDA chain 110-125 ms, the
4-bit projections and shared expert ~250-280 ms.

**The rule** (`fastpf.py` has it with the proof). Prefill bits need not equal decode bits:

- drafted == serial: both decodings of a request start from the same prefilled state, so any *deterministic*
  prefill keeps it (decode and verify windows stay on the row-invariant kernels);
- resumed == fresh: with C = `prefill_rows` rounded down to a multiple of 64, fast chunks sit at absolute multiples
  of C, and a fast prefill of n tokens keeps only the state at B = floor(n / C) * C (taken before the partial last
  chunk; the prompt's end state when n is on the grid; nothing when B = 0), tagged with C. The reply's rows are never
  kept (decode wrote them). A fast request resumes only from a snapshot of its own grid, and re-prefills from B: the
  old prompt's tail (< C tokens), the old reply and the new tokens. By induction over the full chunks before B, the
  state at B is what a fresh prefill of any prompt extending `ids[:B]` holds there (KDA state, conv window,
  attention / indexer / MTP / DFlash2 caches below B, the MTP head's pending row B-1 with `mtp_len = B - 1`); from B
  on both run the same chunks. The MTP head and DFlash2 stay on the row-invariant kernels.

**Change.**

- `fastpf.py`: the switch, the grid, the dispatch (`chunk(b)` marks a main-model chunk and installs
  `fast_qmm.matmul_prefill` behind `qmm.matmul`; `kda_chain` calls `fast_kda.kda_prefill_chunked`), with 0081's modules
  imported when present (a module that is present but broken is reported at load, not silently skipped).
- `exl3_fast.cu` (own extension, `tensorfold_glm_exl3_fast_v1`): two fused grouped GEMMs per MoE layer. A program is
  (128-column block = one Hadamard block, distinct expert); its 4 warps decode each 16x16 trellis tile of a K slab
  once into shared memory (as the lanes' mma B fragments), the next slab's words in flight, and every warp multiplies
  its own member tiles by the whole slab (64 members a pass for gate/up, 128 for down). One warp sums the full K of
  its rows (no split-K, no Z partials: ~0.8 GB a layer of Z traffic gone at 1024 rows), and the Hadamard output
  rotation, scales, GLM's limited SwiGLU and the down input rotation run on the mma accumulators (butterflies in
  `fwht128`'s order, checked bit-identical in a numpy emulation of the fragment layout); Xd and Y are stored once.
  It is row-independent and deterministic. `exl3.cu`'s `rot_in` still makes the fp16 rotated inputs.
- `forward.py`: `compute(..., fast=True, head=...)`: blocks read `b.fast`; a fast chunk's partials are all-gathered
  as bf16 (`GLM53_TF_FAST_GATHER=bf16`, both ranks add the same rounded partials rank 0 first); the final-normed rows
  are computed for every row (the MTP head reads them) but the head only for the last row of the last chunk.
- `decode.py`: `_prefill` cuts chunks on the grid, takes the grid snapshot (`Snapshot.grid`), and refuses a resume
  off the grid or from the other mode; `engine.py`: `_resume` matches the request's grid, `_run` keeps the grid
  snapshot and no reply snapshot in fast mode; both ranks check the fast settings equal at load; ignored with
  `GLM53_TF_BATCH` > 1 (the batch scheduler keeps exact reply snapshots).

**Cost of the rule.** A conversation's next turn re-prefills the old tail and reply through the fast path (e.g.
~500 + 300 tokens at C = 1024, ~0.8 s at 1,000 tok/s) where the exact mode resumes after the reply. That is why the
switch is also per request (0091). Replies depend on C: a fast request's tokens change with `prefill_rows`.

**Expected** (arithmetic from the measured 1024-row profile; nothing timed): experts -130 to -170 ms a chunk (Z
traffic and epilogue grids gone, weights read once at ~8.4 ms a layer), gathers -60 to -90 ms (bf16), head -11 ms;
with 0081, the KDA chain -90 to -105 ms and the projections -60 to -90 ms.

| Prefill tok/s | 1.8k | 7k | 28k |
| --- | ---: | ---: | ---: |
| measured, exact, 1024-row chunks | 671 | 542 | 493 |
| fast (0080 + 0081), 1024-row chunks, expanded KV | 850-1,000 | 650-750 | 580-660 |
| + latent KV (0060, sparse attention ~5x cheaper) | 850-1,000 | 850-1,000 | 850-1,000 |
| + 2048-row chunks (`PREFILL_ROWS_MAX=2048`, ~10 GB of buffers a rank; the experts' 76 GB read once per 2,048 tokens) | 1,050-1,200 | 1,000-1,150 | 1,000-1,100 |
| vLLM, same weights | 960 | 1,340 | 1,448 |

Next steps, in order of the remaining profile: fuse `rot_in` into the gate/up kernel (~590k one-warp programs and
134 MB a layer at 1024 rows); pipeline each chunk's all-gathers (two half-chunks, A's gather on a side stream while
B computes, layer by layer; needs per-half views of the buffers and the KDA/conv state handed from A to B); tune the
fused kernels' occupancy (237-242 registers: 2 blocks of 4 warps an SM; `__launch_bounds__(128, 3)` spills).

**Verified on GB10** (`tests/cuda/test_fastpf_patches.py`, all green; F3/F5 runs). Fused expert kernels timed on
the real per-rank shapes (288 experts, top 8, hidden 4096, 1024 of 2048 wide; random routing; one MoE layer,
`test_fast_expert_timing` and a profiler split):

| rows, routing | row-invariant (grouped loop + epilogues) | fused (rot_in + gate/up + down) |
| --- | ---: | ---: |
| 1024, uniform | 15.7-15.9 ms (rot_in 0.76, grouped 12.5, epilogues 2.5) | 14.5 ms (0.74 + 7.7 + 6.3); 13.9 with the down prefetch |
| 1024, skewed | 16.6-16.8 ms | 15.9-16.0 ms |
| 2048, uniform | 24.9-25.3 ms | 19.7 ms |
| 2048, skewed | 26.3-26.4 ms | 21.7 ms |

In the model (28k prompt, 1024-row chunks) `moe.routed` went 14.6 -> 13.7 s, and with 2048-row chunks 13.9 -> 12.4 s.
Nsight Compute on the gate/up kernel: 222 registers (2 blocks of 4 warps an SM, 17% occupancy), 25% of DRAM
bandwidth, 43% of the stalls at the slab barrier (warps without member tiles decode and wait). `__launch_bounds__(128,
3)` spills and is slower (8.4 / 8.0 ms). The down kernel loads its A fragments one k tile ahead (6.4 -> 5.8 ms, same
bits); the same in gate/up is slower (the registers). Next: spread the n tiles of a block over the warps (every warp
busy whatever the member count) with the Hadamard epilogue through shared memory.

**fast2 (the default since 2026-09-27; `GLM53_TF_FAST_EXPERTS=v1` keeps the kernels above).** Spreading the work over
every warp (a cp.async ring for the member rows, every warp decoding a share of each slab) gave only 1.2-1.8x: the
kernels became bound by shared memory. Each warp re-read every decoded 16 x 16 fragment (512 B) for its 16 member
rows, ~300 KB a 32-k stage and SM, as long as the stage's mma; removing the mma or the trellis decode changed
nothing. fast2 therefore computes Y^T = W^T X^T:

- A warp owns two 16-column tiles of the item's 128 columns (one matrix) for all of the item's members. It decodes
  its own trellis tiles straight into registers: a decoded tile, as `decode_tile` lays it out, is exactly an
  m16n8k16 A fragment. Decoded weights never touch shared memory, and each fragment feeds (members / 8) mma.
- Only the member rows are shared: a 3-deep cp.async ring of 4-k-tile stages (zero-filled past the last member),
  read with ldmatrix as B fragments that each feed 2 mma.
- Work items (expert, pass of 64 members, 128-column block) come from a one-block plan kernel (per-expert pass
  counts, prefix sum). A persistent grid walks them pass-major, with the column blocks of a pass together. No
  program is launched for an absent expert, and a skewed expert's passes spread over the SMs.
- Epilogue through shared memory: fp32 rows, then one warp a row runs the 128-point transforms (butterflies bit 0 to
  bit 6, fwht128's order) and v1's formulas element by element. Down on chunks of 4096+ rows uses 128-member items
  with 2-k-tile stages (same bits; measured faster there).

Same bits as v1: each element is the same chain of m16n8k16 products over ascending k tiles (swapping A and B gives
the same element bits), the same transforms and the same epilogue. Checked bit for bit on GB10 (1024-8192 rows,
uniform and skewed routing: `tests/cuda/test_fast_experts_patches.py` and the micro-benchmark). Row-independent and
deterministic: the tiling is fixed by the shapes.

| one MoE layer, per rank (ms) | v1 gate/up + down | fast2 gate/up + down | x | + rot_in (unchanged) |
| --- | ---: | ---: | ---: | ---: |
| 1024 rows, uniform / skewed | 13.5 / 14.2 | 9.7 / 10.5 | 1.39 / 1.35 | 0.8 |
| 2048, skewed | 18.8 | 13.6 | 1.38 | 1.5 |
| 4096, uniform / skewed | 28.5 / 29.6 | 17.7 / 19.8 | 1.61 / 1.50 | 2.9 |
| 8192, uniform / skewed | 54.7 / 53.8 | 32.1 / 33.6 | 1.70 / 1.60 | 6.4 |

At 8192 rows fast2 runs at ~50 TFLOP/s against the ~110 TFLOP/s measured mma.sync peak (f16 in, f32 accumulate) on GB10. `rot_in`
(6.4 ms, writing 1.2 GB of Xg/Xu) and down's fp32 Y (1.2 GB) are now a third of the layer. Next steps: fuse the
input rotation into gate/up's A loads (the transform runs on the 128-k blocks of a stage), and store Y narrower (new
bits).

## 0081 — fast prefill kernels (`fast_qmm`, `fast_kda`)

New files only (`families/glm5_next/cuda/fast_qmm.py`, `fast_kda.py`); nothing calls them until 0080's fast-prefill
path (`GLM53_TF_FAST_PREFILL`) try-imports them, so without 0080 the patch changes nothing.

- `fast_qmm.matmul_prefill(x, q, xs=None, *, out=None, f32=False, part=None, exact=True)`: `qmm.matmul`'s contract
  for Q4 and B16, for 64+ rows (fewer go to `qmm.matmul`). One program per BM x BN tile over all of K, M tiles
  consecutive so a column block's weights are read from DRAM about once; no split-K partial buffer and no reduce
  kernel. qmm's per-group arithmetic, and qmm's K slices kept as a summation order: **qmm's bits** (checked on GB10,
  `test_fast_qmm_bitwise`, every shape and every tile config of the sweep). Because of that the choice between the two
  is free, and `TUNED` holds the measured fastest per (N x K) and row bucket (512-1535 / 1536+ rows): a tile config,
  or `qmm.matmul` itself where its split-K kernels win.
- `fast_kda.kda_prefill_chunked(<kda.chain's arguments>, *, pos, save_replay=False)`: the KDA recurrence in 64-row
  chunks at absolute multiples of 64 (WY / UT transform, per-channel decay handled in 16-row blocks with a
  block-local reference so no exponential overflows), fp32 state, tf32 tensor-core products (`precision="tf32x3"` /
  `"ieee"` for more). Three kernels (FLA's layout): (1) one program per (64-row chunk, head), all parallel: prologue
  (conv, norms, gates), A, P, the UT transform T (the four 16 x 16 diagonal blocks solved together, the off-diagonal
  blocks by doubling), W = T(beta K~), U = T(beta V), Q~, K^ -- everything that does not need the state; (2) one
  program per (head, 64 value rows of the state) walking the chunks: E = U - W S^T, O = Q~ S^T + P E, S <- S
  diag(e^{G_C}) + E^T K^; (3) the gated RMSNorm per (32 rows, head). Scratch ~177 KB per (chunk, head), 94 MB at
  1024 rows (grown on demand). Same outputs and final state layout as `kda.chain`; not its bits.

**Measured on GB10** (`tests/cuda/test_fastk_patches.py -s`, 1024 rows):

| kernel | before (F2) | now |
| --- | ---: | ---: |
| KDA, 32 heads (`kda.chain` 3.70 ms) | 13.76 ms (one program a head, serial chunks) | 1.82-1.89 ms (prep 1.06, state 0.65, norm 0.08) |
| `fast_qmm` 12576x4096 / 4096x4096 / 2048x4096 | 3.02 / 1.02 / 0.46 | 2.54-2.71 / 0.80-0.82 / 0.42-0.48 |
| 8192x1536 / 8192x512 / 4096x8192 | 0.76 / 0.27 / 2.49 | 0.66 / 0.23 / 2.04 (qmm) |
| 12288x4096 / 4096x6144 / 4096x1024 | 2.23 / 1.47 / 0.24 | 1.82 (qmm) / 1.30-1.39 / 0.21 |
| 4096x128 / 160x4096 / 4096x1536 | 0.038 / 0.073 / 0.34 | 0.027 (qmm) / 0.045 (qmm) / 0.30 |

`fast_kda.STORE_BF16 = "wuqk"` hands W, U, Q~, K^ to kernel 2 as bf16 (1.50 ms) but doubles the state error against
the serial chain (1.1e-3 -> 2.3e-3; 3.9e-3 with slow decays), so it is off. Kernel 2 is bound by reading those
operands (128 KB a chunk and head); software-pipelining them does not fit the 101 KB of shared memory.

**Exactness.** `fast_qmm`: qmm's bits. `fast_kda`: new arithmetic, deterministic, and a prompt prefilled in calls cut
at multiples of 64 gets the same bits whatever the call sizes; against the serial chain on GB10 (1024 rows, tf32):
state 1.1e-3 relative (1.9e-3 with slow decays), outputs 5.7e-4 mean absolute, at most one bf16 step off at 2^-5.
The ieee variant matches the fp32 reference to 1e-4. `save_replay` writes the chain's replay inputs; a few conv
activations round to the neighbouring bf16 value (the fp32 conv sum's order differs), which the test allows.

`tests/test_fastk_interpreter.py` (the Triton CPU interpreter) skips itself where a GPU is visible: in the image the
interpreter misread qmm's inputs (errors of 1e7), and the GPU tests cover the same checks.

## 0082 — lean prefill (fast chunks of 4,096-8,192 rows)

**Problem.** Bigger fast chunks amortize the routed experts: once a chunk touches every expert, the fused kernels
read each one's weights once a chunk. But the window buffers cost ~3 MB a row a set (and grow faster than the rows
past 1024), there are two sets, and `State`'s KDA replay scratch and projection rows add more:

| rows | row-sized buffers a rank |
| ---: | ---: |
| 1024 | 8.5 GiB |
| 2048 | 17.5 GiB |
| 4096 | 37 GiB |
| 8192 | 82 GiB |

**Change.** `GLM53_TF_LEAN_PREFILL=1` (both ranks; checked at load with `GLM53_TF_LEAN_BLOCK`) keeps the window buffers
(both sets, `State`'s rows) at `GLM53_TF_LEAN_BLOCK` rows (default 1024, a multiple of 64). A fast chunk of up to
`GLM53_TF_PREFILL_ROWS_MAX` rows runs through `lean.py`:

- **Sub-blocks.** Every block runs in sub-blocks of the block's rows on those buffers: hc_pre/hc_post, KDA and DSA
  with their projections and o_proj all-gathers, the dense MLPs, the shared expert, the combine, the MoE all-gather
  and the final norm.
  - KDA: the first sub-block starts from the committed state and conv window. Each later one continues from the
    state its predecessor wrote (the layer's output buffer, through a copy) and the last 3 projection rows it saw
    (`LeanBuffers.tail`).
  - DSA: each sub-block runs as a window at `pos + a` (its keys and index keys written before it attends).
- **Whole chunk, once.** The MoE routing (router, top-k, grouping) and the routed experts (patches/0080's fused
  kernels) run once over all rows.
- **The lean set** (`LeanBuffers`, ~397 KiB a row on the real model) holds what spans the chunk: the streams, the FFN
  half's normed rows and mixes, the routing, Xg/Xu/Xd and `ey`, the final-normed rows (MTP absorb) and the DFlash2
  taps (`Engine.main_hidden` / `tap_rows` read them after a lean chunk).
- **Commit.** `lean.commit` installs the carried conv windows and flips the KDA buffers.
- **Engine.** With lean on, `Engine` splits its rows: buffers of the block, `prefill_max` = PREFILL_ROWS_MAX.
  `tf_knobs.prefill_rows` goes up to `prefill_max` (patches/0090 was regenerated for that: two lines read
  `e.prefill_max`). Exact chunks are capped at the buffers' rows (their bits do not depend on the chunk). With
  `GLM53_TF_BATCH` > 1 the lean set is dropped (fast prefill is off there).

**Exactness.** Bit-identical to 0080's fast chunk of the same rows, so replies do not depend on the switch or the
block. Everything run in sub-blocks is row-independent:

- `fast_qmm` (`matmul_fast`, FP8 too): one accumulator over K, row-independent, and the tile does not change bits.
  Its one row-count switch is `MIN_ROWS`: calls of fewer than 64 rows go to qmm (other bits). So `lean.compute` runs
  the chunk in `lean.chunk_rows(w, R)`: when the chunk has 64+ rows, a partial last sub-block of 1-63 rows still runs
  the fast kernels, as the whole-chunk path does. The head, one row in both paths, keeps qmm;
- glue: per row;
- attention and selection: per row and position;
- the bf16 gather: a copy;
- `fast_kda`: calls cut at multiples of 64 give the same bits given the entering state and conv window, and sub-blocks
  start at `pos + k x block`, multiples of 64.

The experts see exactly 0080's full-chunk call. 0080's grid rule is unchanged. Both ranks must agree on the lean
settings: they set the number of all-gathers.

**Memory and expected speed** (`docs/PREFILL-ANALYSIS.md`, "patches/0082").

| setting | memory a rank |
| --- | --- |
| block 1024 | 8.50 GiB of window buffers + the lean set: 0.78 / 1.55 / 3.10 GiB at 2048 / 4096 / 8192 rows |
| block 1024, 8192-row chunks | 11.6 GiB in total, 5.9 GiB less than the 2048-row config in use today |
| block 512 | 4.19 GiB of window buffers + the same lean set |

Estimated fast-prefill rates with 8192-row chunks (not timed):

| tok/s | 32k | 128k |
| --- | ---: | ---: |
| measured today, 2048-row chunks | 770 | 667 |
| estimated, 8192-row chunks | 830-880 | 710-750 |

The per-row costs (sparse attention, all-gathers, KDA projections, hyper-connections) are now most of a chunk. A
larger grid C also costs ~C / 2 re-prefilled tokens a conversation turn.

## 0083 — FP8 fast prefill (`fp8pf.py`, `fast_qmm.matmul_fp8`, latent tensor-core tiles)

Written offline (no GPU; numerics checked in Triton's CPU interpreter). Nothing here is timed.

**Problem.** In fast chunks the per-row costs now dominate (900-row profile at 8k, us a token a rank: sparse attention
117, all-gathers 106, `kda.proj` 94, `hc` 77, `dsa.o_proj` 76, `kda.chain` 63). The projections are compute-bound bf16
GEMMs (`kda.proj`'s 12576 x 4096 at 37-58 TFLOP/s of a ~60 bf16 peak), and GB10 (sm_121) runs e4m3 `mma` at about
twice the bf16 rate. On the latent cache, `absorb` / `expand` (in `dsa.proj` / `dsa.o_proj`) run as fp32 FMA-pipe dots
over 16-row tiles, dequantizing kv_b in fp32 for every 16 rows: ~8.6 GFLOP each a layer at 1024 rows with no tensor
cores. Sparse latent attention uses 16 (row, head) queries a tile, so each row gathers its 2,051 latent rows twice.

**Change.** `GLM53_TF_FP8_PREFILL=1` (default 0; per request: 0092) switches three things in the MAIN model's FAST
chunks only (`fp8pf.ON`, set by `decode._prefill`; decode, verify, the MTP head, DFlash2 and exact prefills never
see it):

- `fast_qmm.matmul_fp8` behind `matmul_fast` for matrices of at least `FP8_MIN_NK` = 4M weights and calls of 64+
  rows (every KDA / DSA / indexer-q_b / dense-MLP / shared-expert projection; f_b / g_b and the indexer's k|w stay
  bf16, and the head's single row stays on qmm):
  - `_fp8_rows`: each row of x gets its own power-of-two scale s_x, the smallest with amax / s_x <= 448, is rounded to
    e4m3 (RTNE), and its 64-input group sums of the ROUNDED values are kept (fp32). A power of two keeps x / s_x exact
    (no division; the host model reproduces the kernel's e4m3 values bit for bit) at no cost in relative precision.
  - `_f8q4`: a 4-bit value q (0..15) is an exact e4m3 number (`FP8_WENC = "cvt"`; or `"bits"`: the nibble read as an
    e4m3 bit pattern is exactly q x 2^-9, subnormals for q < 8, no conversion instruction; the test checks both give
    the same bits on the hardware). Per group g: P = x8 . q8 on the fp8 tensor cores from zero (at most 64 exact
    products, so the tensor cores' accumulator width does not matter), then `acc += P s[n,g] + X8[m,g] b[n,g]` in
    fp32 in group order, and `y = s_x acc`. That is exactly "x rounded to e4m3 per row, times the exact 4-bit
    weights"; the bias term on the rounded sums makes the error sum((x8 - x) w), not sum((x8 - x)(w - b)).
  - BF16 weights (`GLM53_TF_NONEXPERT=bf16`): per-output-channel e4m3 scales (amax / 448, computed once, kept on the
    matrix), fp32 accumulation in 64-input steps.
  - One tile for every shape and row count (`FP8_TILE` = 128 x 64, 4 warps, 2 stages since the GB10 sweep; `GLM53_TF_FP8_TILE` for the
    hardware sweep): 128 rows share each unpacked weight group, 128 columns share each e4m3 row tile.
- Latent cache (0060): `absorb_tc` / `expand_tc`, the same products on bf16 tensor cores over 64-row tiles (kv_b's
  tile dequantized exactly in fp32, then rounded to bf16; fp32 sums in the same K order).
- Latent attention: `FAST_BM` = 32 (row, head) queries a tile (all 32 local heads of a row: one gather of its selected
  latent rows instead of two). 0060 measured this at 11.2 -> 8.5 ms a layer (sparse) and 9.4 -> 7.5 (dense) for 1024
  rows, and it was kept at 16 only because decode windows must share the tile with prefill; fast chunks need not.
  `FAST_STAGES` (1) is the key loop's pipelining depth, for the hardware (no bit changes).

**Why not FP8 attention.** The latent sparse kernel runs at ~12-16 TFLOP/s on a 1024-row chunk (the MMA pipe is
~25% busy): it waits on the per-tile gathers (32 selected rows of 1 KB each, not pipelined) and the online softmax on
small tiles. FP8 Q'K and PV would halve the part that is not the bound, add e4m3 rounding to q', the latent and P, and
save no bytes (the cache stays bf16). The 32-query tile attacks the gathers instead. **Why not the routed experts
(yet).** `exl3_fast.cu` runs at ~15 TFLOP/s at 1024 rows: bound by trellis decode ALU and the slab barrier (43% of
stalls), not the `mma` rate, and a decoded EXL3 weight is an fp16 value with a full mantissa, so e4m3 would add a
second quantization comparable to the 3-4-bit one. Plan, once the kernel is MMA-bound (after 0080's "spread the n
tiles over the warps" and at 8192-row chunks, ~230 members an expert): `rot_in` writes Xg/Xu as e4m3 with a power-of-two
scale per pair (halving the 1.3 GB of Xg/Xu/Xd at 8192 rows), each decoded 16 x 16 trellis tile is converted once to
e4m3 with a fixed power-of-two scale per expert block (the trellis values are bounded), `mma.m16n8k32.e4m3`, and the
existing scale / Hadamard epilogue with the product of the two scales. Worth measuring only then.

**Exactness.** New arithmetic, like 0080: a fast prefill's tokens change with the switch. Every kernel above is
deterministic (no atomics, fixed order, one tile per shape) and row-independent (a row's scale is its own), so
`fastpf`'s argument holds as is: drafted == serial, resumed == fresh. Snapshots carry the tag C + 1 (`fp8pf.tag`;
C is a multiple of 64), so FP8 fast requests resume only from FP8 snapshots of their grid, and bf16 fast and exact
requests never from FP8 ones (`decode._prefill` refuses a mismatched resume, `engine._grid` picks the tag). Lean
chunks (0082) keep "same bits as the fast chunk" within FP8. Both ranks exchange `fp8pf.settings()` at load. The
response's stats show `fast_prefill: C` and `fp8_prefill: 1`.

**Quality.** e4m3 has 3 mantissa bits: ~2.5% rms relative error per rounded activation, so each projection output
carries ~2-3% relative noise (measured on random inputs with x30 outlier channels: 2.7% against fp32; the 4-bit
weights' own error is larger). Residual streams, norms, the KDA recurrence, softmax and the experts stay as before.
This is the arithmetic of W8A8-FP8 with dynamic per-token scales, which is typically within noise on MMLU-style
evaluations; check with MMLU (`fp8_prefill` 0 vs 1 on the same server) before making it a default. The synthetic-model
test prints the FP8 vs bf16 fast prefill distance of hidden rows, logits and KDA states.

**Expected** (per rank, us a token; arithmetic from the 900-row profile and 0081 / 0060's kernel timings):

| component | before | after | how |
| --- | ---: | ---: | --- |
| `kda.proj` | 94 | 55-65 | 12576 x 4096 at 1.5-1.8x, plus ~2 of row quantization |
| `kda.o_proj` (in the rest) | ~20 | 12-14 | 4096 x 4096 in fp8 |
| `dsa.o_proj` | 76 | 25-45 | expand on tensor cores (the FMA-pipe expand is most of it: the 4096 x 8192 matmul alone is ~13), o_proj in fp8 |
| `dsa.proj` (absorb, q_a / kv_a, q_b) | ? | -15 to -30 | absorb on tensor cores, the projections in fp8 |
| sparse attention | 117 | ~89 | 32-query tiles (-24% per kernel, measured in 0060) |
| shared expert, dense MLP, indexer q_b | ~30 | ~20 | fp8 |
| **total** | | **-115 to -165** | |

| prefill tok/s | 8k | 32k | 128k |
| --- | ---: | ---: | ---: |
| 2048-row chunks, measured / modelled today | 889-994 | 770 | 667 |
| + 0083 | 990-1,190 | 845-885 | 725-750 |
| lean 8192-row chunks (0082 estimate) | 966-1,183 | 827-879 | 710-747 |
| lean 8192 + 0083 | 1,090-1,470 | 915-1,030 | 775-850 |
| vLLM, same weights | 1,340 (7k) | 1,448 (28k) | - |

0084's -90 to -120 us a token (all-gathers, hc) is independent of these and adds on top. At 128k the context-growing
costs (indexer scores, sparse selection) dominate what is left.

**Verify on the Spark.** `tests/cuda/test_fp8_patches.py -s` (prints the fp8 vs bf16 timings per shape and M, a tile
sweep, absorb / expand, and the quality distances), then serve with `GLM53_TF_FAST_PREFILL=1 GLM53_TF_PROFILE=1` and
compare `"tf_knobs": {"fp8_prefill": 0}` against `1` on 8k / 32k / 128k prompts, and MMLU with each. If
`test_fp8_weight_encodings_and_tiles_same_bits` fails only for "bits", the tensor cores flush e4m3 subnormals: keep
"cvt" (the default). Set `FP8_TILE` to the fastest config of the sweep. Done (2026-09-27): 128,64,4,2 at every shape, M = 2048 and 8192 (1.0-1.27x the bf16 fast kernel; 128,128,8,3 was 0.8-0.9x, so before the switch the FP8 matmuls were a net loss and the FP8 gain came from absorb / expand). `test_fp8_patches.py` host-model tolerances: 6e-5 (Q4, measured 2.6-3.2e-5) and 5e-4 (BF16 weights, measured 1.9e-4): fp32 summation order only.

## 0084 — pipelined lean prefill (all-gathers hidden, hyper-connections in L2)

**Problem.** In the fast-prefill profile (per token and rank, 2048-row chunks) the all-gathers took 106 us and the
hyper-connections 77 us. Every exchange ran on the compute stream, and the next kernel waited for it. About a third
of the `allgather` time was not the link: `forward.gather` copied the fp32 partial to bf16 before the exchange and the
gathered bf16 back to fp32 after it (88 KB a row a site, 90 sites a token). The hyper-connection kernels are
memory-bound over the [rows, 4 x 4096] bf16 streams. hc_post reads and writes them, then the next hc_pre reads them
twice (its mixing dots, then the collapse). Each read came from DRAM, because a 1024-row sub-block of streams (32 MB)
does not fit in L2.

**Change** (`pfoverlap.py`; hooks in `forward.out_proj` / `gather`, `lean.py`, `glue.hc_post` / `combine`). With
`GLM53_TF_PREFILL_OVERLAP=1`, a lean chunk (0082) runs as a sequence of pieces (layer, half, sub-block). Each piece's
pre work ends in a rank partial, and its post work consumes the gathered partials.

- `gather`: piece k's all-gather goes on a comm stream. It is high priority and has two partial/gather slots with
  events. The compute stream runs piece k+1's pre work (hc-mixed inputs, projections, KDA chain or attention, o_proj;
  or shared expert + combine), then waits for exchange k and runs post(k). Rules:
  - A piece whose inputs come from the post just before it drains first: one-sub-block chunks, and the MoE routing,
    which reads every row.
  - A slot is rewritten only after its exchange was waited for. `Pipe` checks this.
  - Both ranks issue the same collectives in the same order from one stream.
- `direct` (always on in the pipeline): o_proj / down store the partial in the exchanged dtype (bf16 by default), the
  MoE combine stores bf16, and hc_post reads the gathered bf16 and widens it itself. That is 96 KB a row a site less
  traffic and 2 fewer kernels a site.
- `slab`: a piece's post runs hc_post and then the next hc_pre of the same rows, in slabs of
  `GLM53_TF_PREFILL_HC_SLAB` rows (default: half the L2 over 32 KB a row, 384 rows at 24 MB). The next hc_pre is the
  FFN's after attention, the next layer's attention's after the FFN, or the final norm and taps after the last layer.
  hc_pre's two reads then hit L2. Its outputs wait in the lean set's rows (`lb.normed` / `xs` / `post` / `comb`), which
  0082 already holds. DSA sub-blocks copy their 8 KB a row into the window buffers (11 layers); KDA reads the lean rows
  in place.

**Exactness.** The same kernels run on the same values; only the order and three store/load dtypes change:
- The o_proj / down matmuls and the combine now store bf16. That is the fp32 sum rounded to nearest even in the
  store, which is what `copy_` did.
- hc_post widens bf16 to fp32 itself. That is exact, so it sums the same fp32 values in the same order.
- Slabs are cut at multiples of 64 rows, and every glue / hc kernel is row-independent.

So the committed state and replies equal 0082's lean chunk bit for bit. The ranks need not even agree on the knob:
the collectives and their order are the same. Tests: `tests/cuda/test_overlap_patches.py`.

**Not done, and why.**
- Merging hc_post with the next hc_partial into one kernel would add little over the slabs: a few L2 re-reads and one
  launch a slab. It would also need fast_qmm's row-tiled dots and sum of squares reproduced bit for bit, which only
  running the same kernel guarantees (a Triton reduction's order follows the layout the compiler picks).
- Re-tiling hc_finish (per-row Sinkhorn over 4 x 4) would change its reduction order: new bits.
- Coalescing exchanges: a layer's MoE partials (or its attention partials) could go in one exchange of the whole
  chunk, exact since a gather is a copy. That saves only (sub-blocks - 1) x alpha a site, about 1-2 us a token, and
  leaves nothing to overlap the exchange with. Pipelining beats it.
- GLM53_TF_FAST_GATHER=fp32 works (fp32 slots, twice the wire bytes: overlap matters more there).
- 0040's L2 prefetch is skipped for pipelined exchanges: its plans are decode-shaped.

**Expected** (arithmetic, per token and rank, against the profile above; nothing timed):

| part | saves |
| --- | ---: |
| `direct`: 2 conversion passes + narrower partial writes / hc_post reads, 96 KB a row x 90 sites | 35-40 us |
| `gather`: the ~70 us of NCCL + wait left, minus the MoE boundary (the last attention exchange overlaps only the previous sub-block's post) and NCCL's SMs taken from the compute kernels | 40-60 us |
| `slab`: hc_pre's 2 x 32 KB a row from L2, 90 sites, minus slab tails | 15-22 us |
| total | ~90-120 us |

At 32k: 2048-row chunks 770 -> ~830-850 tok/s. 8192-row lean chunks (est. 830-880) -> ~910-970 tok/s.

**On the Sparks.**
- `GLM53_TF_PROFILE=1`: `allgather` is now the time the compute stream still waited for.
- Compare tok/s at 8k / 32k / 128k with `prefill_overlap` 0 vs 1 (per request, no restart), and each variant alone
  (`gather`, `slab`, `direct`).
- Try `NCCL_MAX_NCHANNELS=1` or `2` (fewer SMs taken by NCCL beside the compute kernels) and `GLM53_TF_PREFILL_HC_SLAB`
  256 / 384 / 512.
- Take an nsys trace of one chunk, to see `ncclDevKernel_AllGather` running beside the compute kernels.

## 0085 — chunk-size-independent fast prefill (`pfgrid.py`, 64-token snapshots, `prefill_rows=auto`)

Written offline (no GPU); the host tests ran, the GPU tests are for the Sparks.

**Problem.** 0080's grid rule put fast chunks and snapshots on absolute multiples of C and let a request resume
only from a snapshot of its own C, because the fast kernels were only required to be deterministic. At C = 8192 (the
fastest cold prefill, 1,034 tok/s at 28k) a follow-up turn re-prefilled up to C - 1 old tokens plus the reply:
3.6-6.8 s warm TTFT against ~0.7-1 s at C = 1024 (`docs/RESULTS.md`).

**Audit** (is a row's output independent of which rows share the call, and of how many?):

| kernel / op on the fast path | row-independent? | why |
| --- | --- | --- |
| `fast_qmm.matmul_fast` 4-bit (`_fq4`, one accumulator) | yes, but **not below 64 rows** (fixed) | tile by (N, K) only (`LOOSE` / `LOOSE_TUNED`, never by M), one fp32 accumulator walking K in group order, no split-K; `GROUP_M`'s grouped program order only reorders programs. `MIN_ROWS`: a call of < 64 rows went to qmm's split-K kernels (other bits), so a prompt's last chunk of 1-63 rows got other bits than the same rows inside a longer chunk |
| `fast_qmm` BF16 weights (`_fb16`) | yes (tile now fixed) | BM was picked by M (>= 512 rows); a tile never changes a row's operations (mma rows are independent, K order fixed by BK), but the choice is now by shape only |
| `matmul_fp8` (`_fp8_rows`, `_f8q4`, `_f8b16`) | yes, but not below 64 rows (fixed) | per-row power-of-two scale and group sums; `FP8_TILE` fixed for every shape and M; the per-channel weight scales are per weight. < 64 rows fell back to the bf16 kernel / qmm |
| `hc_partial` | yes | 64-row tiles, fixed K order, per-row sum of squares |
| latent `absorb_tc` / `expand_tc` | yes | `TC_ROWS` = 64 fixed, fixed K order |
| `exl3_fast.cu` gate/up and down | yes | a member tile's warp runs one ascending mma chain over all of K; the expert's member count only decides which pass / warp holds a pair (no K split by members, no reduction across members); epilogue per element; `rot_in` per pair. Keep it so in the "spread n tiles over warps" rework |
| router, top-k, grouping, combine, glue, hc_pre / hc_post, norms | yes | the row-invariant kernels, or per-row arithmetic |
| attention (dense / sparse, latent / expanded, `FAST_BM` 32 = one row's heads) | yes | the row-invariant kernels of every exact prefill; a row sees keys <= its position whatever the chunk (`nch`, `pool_bucket` only add masked work) |
| indexer blocked selection (0065, `GLM53_TF_SELECT_MB` row blocks) | yes | exact selection (== sort), scores per row |
| bf16 gathers, 0084 overlap, slabs | yes | copies; 0084 is bit-identical by construction |
| lean sub-blocks (0082) | yes | sub-blocks at `pos + k x block` (multiples of 64); `lean.chunk_rows` already forced the fast kernels for a < 64-row tail sub-block |
| `fast_kda` (3 kernels) | per 64-row block at ABSOLUTE multiples of 64 | `off = pos % 64`: a call starting mid-block would split the block into two partial ones (other bits); a block's bits depend on its rows and the fp32 state entering it, and a state handed between calls is the fp32 value the one-call scan keeps in registers. Every fast call now starts on a multiple of 64 (asserted in `fastpf.kda_chain`); the only partial block is the prompt's last one, the same in every schedule |
| MTP head absorb, DFlash2 taps, the head | yes | row-invariant kernels; the head runs on the one last row in every schedule (kept on qmm) |

**Change.**

- `fastpf.chunk(b, head)` / `fast_qmm.FAST_HEAD`: in a fast chunk every matmul but the head's runs the fast kernels
  whatever its row count (`min_rows` on `matmul_prefill` / `matmul_fp8`; direct calls keep `MIN_ROWS`). The one-accumulator
  BF16 tile is chosen by shape only. `fastpf.kda_chain` refuses a position off the 64 grid.
- `pfgrid.py` (new, no torch; the one place the grid / tag / chunk rules live, for 0110 and 0120 to call):
  `tag(fast, fp8)` = 0 / G / G + 1 (G = `GLM53_TF_SNAPSHOT_GRID`, default 64; never C), `grid_of`, `resumable`
  (same tag, position a multiple of 64), `parse_rows` ("auto" = 0), `chunk_rows` (auto: the rows to prefill rounded
  up to 64, at most the buffers), `plan(begin, n, C, marks)`: chunks of C from the resume point, cut at marks and at
  the snapshot point. The module docstring has the proof.
- Snapshot rule: one fast snapshot per prefill, at S = the prompt's last multiple of G (after the prefill when S = n,
  else before a chunk starting at S; the prefill cuts one there). Tail rule (`GLM53_TF_SNAPSHOT_TAIL`, default
  256): when the schedule's last chunk already starts within that many rows before S, snapshot there instead (saves
  an extra < 64-row chunk now; the next turn re-prefills at most tail + 63 more rows). The reply is still never kept
  in fast mode: its rows were written by the row-invariant decode kernels, which a fast prefill of the next prompt
  does not reproduce, so it is re-prefilled.
- `decode._prefill`: `pfgrid.plan` replaces the C grid; `e.checkpoints` marks (0110's) cut chunks and snapshot
  there (`e.mark_snaps`); `e.fast_rows` = the C used (`stats.fast_prefill`). `engine._grid` = `pfgrid.tag`, so
  `_resume` offers every fast snapshot of the request's mode whatever its C. `GLM53_TF_PREFILL_ROWS=auto` at load;
  both ranks exchange G and the tail with `fastpf.settings()`.
- 0090 (regenerated): `tf_knobs.prefill_rows` accepts `"auto"` (0 in the header; rank 1 applies it; the chunk size
  is then a pure function of the header's prompt and resume lengths and the load-checked buffers, so both ranks cut
  the same chunks); the echo says "auto"; `PREFILL_ROWS_MAX` defaults to 64 with `auto`. 0091-0093 apply unchanged.
- 0110: its `_prefill` loop hunk is gone (0085 does the marks); `sessions.snapshot_grid(tag)` already reads G from
  the tag, so fast pages are keyed by their 64-token block.
- Tests that check 0080's C-grid rule (hostile C-dependent fakes, snapshot positions) run with
  `GLM53_TF_SNAPSHOT_GRID` = C, which is exactly 0080's rule (`test_fastpf` / `test_lean` / `test_session` engines,
  `_FakeEngine.snap_grid`).

**Exactness.** Within fast mode: drafted == serial (deterministic prefill), resumed == fresh for ANY pair of chunk
sizes (and auto). A fast reply now depends only on the tokens, not on `prefill_rows`. Against 0084's bits: identical
except prompts whose last fast chunk had 1-63 rows (they now run the fast kernels there instead of qmm).

**Follow-up turn cost** (auto C, 8192 max; per-row ~1 ms and a tiny chunk ~0.2-0.3 s estimated from the 28k
profiles, nothing timed): a turn re-prefills the old prompt's last < 64 tokens (< 320 under the tail rule) + the reply
+ the new tokens in one chunk, plus a < 64-row tail chunk to put its own snapshot on the grid: ~0.8-1.1 s for a
300-token reply and a short new message, against 3.6-6.8 s at C = 8192 before; a cold 28k prefill gains one tail chunk
(+0.2-0.3 s, ~1%).

## 0090 — per-request knobs (`tf_knobs`)

**Problem.** A load takes ~6 minutes on the two Sparks, and every speed knob above was read from the environment
at load, so each A/B variant cost a restart.

**Change.** A request may carry `"tf_knobs": {...}` (any subset; `families/glm5_next/cuda/knobs.py`). The
GLM53_TF_* environment now only sets each knob's default; after the request the defaults apply again (also when it
fails). The response's `tensorfold.tf_knobs` (non-streamed body, and the last streamed chunk) echoes every knob's
value for that request.

| key | values | default from | per request? |
| --- | --- | --- | --- |
| `lookup` | 0, 1 | `GLM53_TF_LOOKUP` (1) | yes (auto/o's lookup gate) |
| `lookup_min` | 1-64 | `GLM53_TF_LOOKUP_MIN` (4) | yes |
| `auto_fdrafts` | 1-7 | `GLM53_TF_AUTO_FDRAFTS` (7) | yes |
| `expert_loop` | 0, 1 | `GLM53_TF_EXPERT_LOOP` (1) | yes (only windows > 16 rows use it: never graph-captured) |
| `prefill_rows` | 1 to `GLM53_TF_PREFILL_ROWS_MAX` | `GLM53_TF_PREFILL_ROWS` (64) | yes; the buffers are sized for the max (default = PREFILL_ROWS) |
| `calib_online` | 0, 1 | `GLM53_TF_CALIB_ONLINE` (0) | yes (rank 0's table travels in the header as before) |
| `longctx_graphs` | 0, 1 | `GLM53_TF_LONGCTX_GRAPHS` (1) | yes when loaded with 1 (the buffers exist); `1` refused on an engine loaded with 0 |
| `profile` | 0, 1 | `GLM53_TF_PROFILE` (0) | yes |
| `depth` | `"cost"`, `"threshold"` | `GLM53_TF_DEPTH` (threshold) | yes (plain `auto`; `o`/`om`/`of` policies are cost-derived anyway) |
| `nonexpert`, `latent_kv`, `comm`, `batch`, `prefill_rows_max`, `calib` | | | no: HTTP 400 naming the reason (weight format, cache layout, graph-captured comm / NCCL env, scheduler, buffer size, load-time calibration) |

Rank 0 validates the knobs in `App.check` (HTTP 400 before anything streams: unknown key, load-only key, range,
type, `GLM53_TF_BATCH` > 1). The header gains, after 0071's cost flag and before the policy code, a block
`[6, expert_loop, prefill_rows, longctx_graphs, profile, auto_fdrafts, calib_online]` holding rank 0's value of
every knob for the request (its defaults filled in); rank 1 sets exactly these, runs the request, and restores its
own. `lookup`/`lookup_min` travel in the policy code (0020's slots), `depth` as 0071's flag, `calib_online`'s
effect as 0070's table. Both ranks check `GLM53_TF_PREFILL_ROWS` and `GLM53_TF_PREFILL_ROWS_MAX` equal at start.

**Exactness.** Each knob only selects between paths already shown to give the same bits: chunk size (0003: rows
never depend on chunk-mates; resumes across chunk sizes match), `grouped_loop` vs grid (0006: same work item per
member tile), bounded/graph vs upstream selection (0050), timing-only probes (0005), and draft depth / lookup /
cost tables (0010/0020/0070/0071: they choose which drafts are verified; the verify window, keyed sampler and commit
decide every token, so drafted == serial). Rank 1 never reads its own environment for these: the values come from
rank 0's header, so both ranks run the same windows and collectives. Tests: `tests/cuda/test_knob_patches.py`.

## 0091 — per-request `fast_prefill` knob

`"tf_knobs": {"fast_prefill": 0|1}` (default `GLM53_TF_FAST_PREFILL`), in 0090's header block (now 7 knobs; both
ranks must run the same patch set). The request's grid is `fastpf.grid(prefill_rows)` of its own `prefill_rows`, and
`_resume` only offers snapshots of that grid (exact requests: grid 0), so the two modes never resume from each
other's states. Refused when the buffers hold fewer than 64 rows. Unlike every 0090 knob, this one changes a
request's tokens (the prefill's arithmetic); drafted == serial and resumed == fresh hold within each setting.

## 0092 — per-request `fp8_prefill` knob

`"tf_knobs": {"fp8_prefill": 0|1}` (default `GLM53_TF_FP8_PREFILL`), in 0090's header block after `fast_prefill` (8
knobs; both ranks must run the same patch set). No effect on a request with `fast_prefill: 0`. The request's snapshot
tag is `fp8pf.tag(grid(prefill_rows), fp8_prefill)`, so `_resume` offers FP8 requests only FP8 snapshots of their grid.
Refused (HTTP 400) where FP8 tensor-core dots are unavailable (`fp8pf.available()`). Like `fast_prefill`, it changes a
request's tokens; drafted == serial and resumed == fresh hold within each setting.

## 0093 — per-request `prefill_overlap` knob

`"tf_knobs": {"prefill_overlap": 0|1}` (default: `GLM53_TF_PREFILL_OVERLAP` on or off), in 0090's header block after
0092's `fp8_prefill` (9 knobs; both ranks must run the same patch set). `1` runs the environment's variant, or
`gather,slab` when the environment has none. Same bits either way (0084), so snapshots are shared across the setting.

## 0110 — multi-session state cache (`sessions.py`)

**Problem.** The engine kept the committed state of the last request only (the snapshots after its prompt and its
reply, `GlmEngine.cache`), so an agent switching between sessions (opencode's subagents, parallel tool loops)
re-prefilled the whole context on every switch: ~34 s at 30k tokens, ~115 s at 100k. The vLLM kit keeps ~14
conversations cached (97%+ prefix hits).

**Change.** `GLM53_TF_SESSION_GIB=N` (default 0: off, upstream behaviour; `config/minimal.env.example` and
`docker/compose.yaml` set 12) keeps a store of session entries per rank (`families/glm5_next/cuda/sessions.py`,
design in `docs/SESSIONS-DESIGN.md`):

- *Entry* = a `decode.Snapshot` (ids; KDA recurrent states + conv windows, which also hold 0065's index-ring tail;
  pending MTP rows; `mtp_len`, `drafter_end`, grid tag) + every per-position cache row below its length: DSA KV rows
  (latent or expanded), full-size index keys/gates (the MTP layer's; the model layers' only with
  `GLM53_TF_INDEX_RING=0`), pool keys, and the MTP head's rows below `mtp_len`. With DFlash2, the snapshot also keeps
  the drafter's context rows in its sliding window (`Snapshot.window`, ~20 MB), so a restored session drafts as
  before (0065's ring alone would mask what other sessions overwrote: drafts, not replies, would change).
- *Pages*: rows live in 256-token pages (16-page slabs per cache tensor), keyed by the tokens that determine their
  bits: exact rows by their prefix plus the next token (an MTP row reads it); fast rows (0080, grid C) by every token
  up to the end of their chunk; plus the tag and MTP validity. Equal keys share one page (reference counted), so a
  system prompt common to many subagents is stored once. The rest of an entry (last partial page, a reply's MTP
  backlog) is a private tail.
- *Restore* (copy-in): the entry's pages and tail are copied into the live caches at their positions, skipping
  pages the live caches already hold (tracked per page), plus the drafter window; `prefill(resume=snapshot)` then
  restores the KDA state as for any resume. A request uses the store only when it resumes more of the prompt than
  the live snapshots; `usage.prompt_tokens_details.cached_tokens` (and `tensorfold.cached`) is the resumed length.
- *Saves*: after the prefill, the prompt snapshot (exact: at the prompt end; fast: at its last grid point) and the
  marks; after the reply, the reply snapshot (exact only), exactly the snapshots the engine already keeps.
- *Marks*: the KDA state cannot be rebuilt from attention rows, so a prompt that forks from a stored one can only
  resume at a snapshot. Rank 0 asks the prefill for extra snapshots at the page (exact) / grid point (fast) at or
  before the longest common prefix with any entry (at least `GLM53_TF_SESSION_FORK_MIN` = 512 tokens past the resume
  point) and every `GLM53_TF_SESSION_EVERY` = 16,384 tokens; an exact chunk is cut there (no bit changes).
- *Eviction*: least recently used entry (restored, resumed from, saved again = used) until the new one fits; pages go
  with their last reference, empty slabs are released; an entry larger than the budget is not stored. Rank 0 also
  skips a save that would leave less than `GLM53_TF_SESSION_RESERVE_GIB` (2) of device memory.
- *Two ranks*: rank 0 plans (entry to restore, marks) and sends the plan after the prompt with a digest of its store;
  rank 1 checks the digest and follows. For each save rank 0 sends its decision (stored / skipped / duplicate, and the
  entries it evicted); rank 1 applies it and fails loudly if its store disagrees. Settings are checked equal at load.
  Off with `GLM53_TF_BATCH` > 1 (0030's slots keep their own states) unless `GLM53_TF_BATCH_SESSIONS=1` (0180).

**Memory a rank (real model, latent KV, index ring on).** 13.25 KB a token of pages (12 x 1 KB latent, the MTP
layer's index keys/gates 0.5 KB, 12 layers' pool keys 0.75 KB; 18.75 KB with `GLM53_TF_INDEX_RING=0`); ~94 MB a
snapshot (KDA 71.3 MB, conv 2.6 MB, DFlash2 window ~20 MB). A session with its prompt and reply snapshots:

| Tokens | Pages | + 2 snapshots | + marks (every 16k) | Sessions in 12 GiB (with marks, no sharing) |
| ---: | ---: | ---: | ---: | ---: |
| 8k | 0.11 GB | 0.30 GB | 0.30 GB | ~43 |
| 30k | 0.41 GB | 0.60 GB | 0.69 GB | ~19 |
| 100k | 1.36 GB | 1.55 GB | 2.11 GB | ~6 |

Expanded KV (0060 off) is 390 KB a token: the store works but holds ~30x fewer tokens.

**Switch latency (estimate).** A restore copies the entry's bytes once: 30k tokens ~0.5 GB, ~4-5 ms at ~110 GB/s of
device copy (100k: ~13 ms; 8k: ~2 ms), plus ~1 ms of host hashing; then the new turn's tokens prefill as usual. A
save after the prefill copies only new pages (a new 30k prompt: ~4 ms) and the snapshot.

**Exactness.** No new arithmetic. A page's key fixes every input its rows were computed from (exact rows are
row-invariant functions of their prefix; fast rows of their chunk and everything before it; 0080's lemma), so a
shared page has the bits the entry's own prefill wrote; tails, KDA states and drafter windows are copies of the
live state when the snapshot was taken. Snapshots are taken only where the engine already takes them (any position
exact, the grid fast; marks included), so restoring an entry gives the fresh prefill's state: resumed == fresh, and
drafted == serial as before. Tests: `tests/cuda/test_session_patches.py`.

## 0120 — batching on the current engine (`batch.py`, `batchplan.py`)

**Problem.** 0030 (`GLM53_TF_BATCH=N`) was written against 0001-0020. In batch mode DFlash2 ran as MTP drafts, the
lookup drafter and cost-derived depths were not used, `tf_knobs` were refused, fast / lean prefill were switched off,
windows over 4 rows and every round past 2,051 tokens ran eager, and an admitted prompt prefilled whole while the
others waited (a 100k prompt: minutes).

**Change** (design: `docs/BATCHING-DESIGN.md`, "v2 on the current engine").

- *Per sequence*: a `State` per slot (latent KV, index rings sized for prefill chunks), MTP graphs per slot (0050's
  long-context ones too), a DFlash2 context per slot (`_drafter_view`: shared weights, own context ring, positions
  and graphs), and `Stepper`, `decode.auto_decode`'s round cut at the forward, so every policy (MTP, DFlash2, `auto`,
  `lN` and auto's lookup gate, thresholds, `o` / `om` / `of`) drafts per request exactly as alone. Slots other than 0
  keep 8-row KDA window buffers and borrow slot 0's for a prefill.
- *Knobs*: a request's `tf_knobs` travel in its admission header (`batchplan.encode_header`); its prefill runs in
  `GlmEngine._knobs(values)`, its drafter with its `auto_fdrafts` / lookup / depth. Only `calib_online=1` is refused
  in batch mode (`knobs.parse`).
- *Prefill in pieces*: `GLM53_TF_BATCH_PIECE` tokens (fast: multiples of the chunk grid) a piece, each a
  `decode.prefill` resumed from the previous piece's snapshot, one piece a round, shortest remaining prompt first; while
  others decode, pieces take at most `GLM53_TF_BATCH_PREFILL_SHARE` of the time. Fast requests keep grid snapshots
  only, exact ones prompt and reply snapshots, per slot.
- *Rounds*: one forward over every decoding window; patches/0050's device-side selection per slot past 2,051 tokens
  (latent and expanded); one all-gather for every request's sampling candidates; CUDA graphs captured lazily per
  (slots, rows per slot, dense / pool bucket) up to 8-row windows and 256 graphs, KDA states normalized to buffer 0
  before a graphed round. Cost-derived depths price rows past the other slots' rows at the rounds' aggregate rate.
- *Server*: `"priority": "background"` or a session-title request waits behind foreground requests and steps aside
  (runs again later; its caller gets each token once) when one waits and no slot is free.
- *Admission control*: at load, slots are added while `GLM53_TF_BATCH_RESERVE_GB` stays free (both ranks agree on
  the count); at admission a request waits while less than `GLM53_TF_BATCH_ADMIT_GB` is free and others run.
- *Two ranks*: rank 0 plans each round (cancels, admissions with headers, the piece) as one int list
  (`batchplan.encode_plan`); everything else is computed alike from shared inputs. Settings checked equal at load.
- `engine.py`: the batcher is built after the knob defaults and before 0110's store (which stays off in batch mode);
  `generate` no longer refuses knobs. `app.py`: the background flag.

**Exactness.** A request's rows are its lone forward's bits (row-local kernels; per-slot kernels see only the slot's
state and rows), its prefill is `decode.prefill` (in pieces: resumed == fresh in every mode), its commits are
`forward.commit` on its own state, its sampling the keyed rule on the same candidates; drafts only propose. So each
batched reply equals the same request served alone and serial decoding, whatever shares its rounds; per-request knobs
change a request's tokens exactly as they do alone (fast / FP8 prefill), and never another request's. KDA parity
normalization moves the same values. Tests: `tests/cuda/test_batch2_patches.py` (replaces `test_batch_patches.py`).

**Expected** (arithmetic, BATCHING-DESIGN section 6 model): ~60 tok/s total at 2 sequences (1.3-1.35x one MTP
stream), ~80-90 at 4 (1.25-1.4x vLLM's 63-66). ~0.85 GB a rank per extra sequence at 32k, ~2.1 GB at 128k (latent KV).
Not run on a GPU yet.

## 0140 — fast restarts (`fastboot.py`)

**Problem.** A restart took 274-490 s (launcher to ready; 303-313 s in the current configuration), and almost all of
it rebuilt the same thing: each rank sliced its half out of the full 164 GB checkpoint through a file-backed mmap
(single thread, 1.4 ms and ~0.4 GB/s of share a tensor, over ~150k tensors, touching ~1/3 more bytes than the half),
re-quantized the BF16 non-experts (q4mse clip search), and 4-bit-quantized the drafter; then timed every verify
window and drafter again (patches/0070). The Spark's NVMe streams ~10 GB/s with O_DIRECT. `docs/BOOT.md` has the
timeline.

**Change.**

- `weights.load` / `Drafter.read_weights` (new, the drafter's old reading code) go through `fastboot.cached_tree`:
  with `GLM53_TF_PREPARED=DIR`, a valid folder `DIR/<model>-<rev>/<key>/rank<R>` is read back instead of building;
  otherwise the checkpoint path runs (and, with `GLM53_TF_PREPARED_WRITE=1`, the result is written for the next
  start). A folder holds the built objects as they are: every tensor's storage bytes (4 KiB aligned in one
  `data.bin`) with dtype, shape, stride and storage offset, and the dataclasses around them (`manifest.json`, which
  also has a SHA-256 per 64 MiB chunk). The key covers the checkpoint (revision, file names and sizes,
  `config.json`), the rank, `GLM53_TF_NONEXPERT`, torch's version, the device type, and the source of every
  function and class that builds the weights (`weights.py`, `split.py`, the quantizers, the EXL3 word layout, the
  drafter's reader), so a patch that changes how weights are built misses old folders. `scripts/prepare.sh` writes
  both ranks' folders once (`python -m tensorfold.families.glm5_next.cuda.fastboot prepare|verify|status`).
- Reader: 8 threads read 64 MiB chunks with O_DIRECT (buffered + `POSIX_FADV_DONTNEED` where O_DIRECT is refused)
  into pinned buffers and copy each chunk's pieces to the device on a stream per thread; no page cache (on GB10 it
  is GPU memory) and no file-backed mmap handed to CUDA. `GLM53_TF_PREPARED_VERIFY`: `sample` (default, every 16th
  chunk and the last), `full`, `off`; a mismatch falls back to the checkpoint.
- `GLM53_TF_CALIB=cached`: both ranks all-gather a digest of their identity (`GLM53_TF_IMAGE_ID`, every
  `GLM53_TF_*` knob but launch-only ones, context / capacity / drafter / policy / prefill rows, torch, GPU name,
  nvidia-smi's clocks and power limit, `GLM53_TF_CLOCK_CAP`); rank 0 looks up `calib-<both digests>.json` in
  `GLM53_TF_CALIB_DIR` (`/cache/calib`) and shares its bytes, so both ranks hold the same floats; a hit prefills the
  calibration prompt once (warm-up, `GLM53_TF_BOOT_WARMUP=0` skips it), a miss measures as `real` and stores the
  table. `real` always measures and refreshes the stored table (the forced re-measure).
- `[boot] rR +T s phase (own s) | MemFree, MemAvailable` lines from the engine (start, NCCL, weights, barrier,
  drafter, engine + graphs, drafter graphs, calibration, ready), counted from `serve.sh`'s launch
  (`GLM53_TF_LAUNCH_T0`) or the entrypoint (`GLM53_TF_T0`).
- Image / launcher: `CUDA_CACHE_PATH=/cache/nv/ComputeCache` (4 GiB) joins the torch-extension and Triton caches in
  the `/cache` volume; `serve.sh` starts both ranks at once, polls readiness every second (the worker over ssh
  every 10th), mounts `HEAD_PREPARED` / `WORKER_PREPARED` at `/prepared`.

**Exactness.** A prepared tensor is the built tensor's bytes and layout, so replies cannot change: tested bit for
bit against `load_checkpoint` for both ranks, every NONEXPERT mode and the drafter (`tests/test_fastboot_prepared.py`
on the CPU, `tests/cuda/test_boot_patches.py` on the GPU, where the load-time path is also checked deterministic),
and an engine on prepared weights replies as one built from the checkpoint. The cached calibration table is the
table a real calibration produced for the same key; costs only choose draft depths, so drafted == serial holds on it
(tested).

## 0150 — liveness, metrics and reasoning effort (ideas from the MiaAI-Lab kit, `health.py`)

See `docs/MIA-AUDIT.md` for the audit these come from.

**Liveness.** Upstream's `/health` answers `{"ok": true}` whatever happens. With two ranks, a CUDA error or an
out-of-memory in the middle of a request leaves rank 1 in a collective forever, and the next request hangs behind it
while `/health` still says ok (the vLLM kits saw the same: vLLM's `/health` stays 200 through a stuck NCCL collective
or a UVM livelock). `tensorfold/cuda/health.py` wraps the engine's `generate` (`Health.track`: every other attribute
reads and writes through, the signature is kept for `App.check`'s `draft` probe) and keeps:

- `fatal`: the first exception out of `generate` that is not a `ValueError` (a bad request);
- the requests in flight: prompt length, start, last token time; a request is `stalled` when its last token (or its
  start, before the first token) is older than `GLM53_TF_STALL_S` (0: never) plus, before the first token, its prompt
  at `GLM53_TF_STALL_PREFILL_TPS` tokens a second (200: a 1M prompt gets ~83 min);
- counters: requests, errors, refusals, prompt / cached / completion tokens, decode rounds and seconds, prefill
  seconds.

`GET /health` returns them (`ok`, `mode`, `uptime_s`, `inflight`, `oldest_s`, `idle_s`, `requests`, `errors`, `fatal`,
`stalled`). `GLM53_TF_HEALTH=basic` (default) always answers 200; `strict` answers 503 when `fatal` is set or a request
is stalled, and a new completion gets 503 at once after a fatal error. `GET /metrics` is Prometheus text
(`tensorfold_*_total` counters and gauges; (completion tokens - requests) / rounds over an interval is the drafter's
health). An exception in a request now answers a JSON 500 (400 for a `ValueError`) or, streaming, an SSE
`{"error": ...}` event and `[DONE]`, instead of a dropped connection. `scripts/serve.sh watch` restarts both ranks on
`strict`'s 503 and alerts on a low tokens-a-round rate.

**Reasoning effort.** GLM-5.3's template renders `Reasoning Effort: Low|High|Max` (Max unless
`chat_template_kwargs.reasoning_effort` says `low` or `high`). The MiaAI-Lab kit measured structured output (a long
numeric table): thinking off garbles 5-6 of 6 replies; thinking on at low effort garbles none (791/792 numbers
right). `GLM53_TF_EFFORT_FIELD=1` maps OpenAI's top-level `reasoning_effort` (`none`/`minimal`: thinking off; `low`:
Low; `medium`/`high`: High; `max`/`xhigh`: Max) onto `chat_template_kwargs`, only for keys the request did not set;
`GLM53_TF_DEFAULT_EFFORT` is the effort of a thinking request that names none. An unknown effort is a 400. Both run in
`GlmApp.check` (so the context check counts the rendered prompt) and `run` (idempotent).

**Exactness.** Nothing on the engine side changes: the wrapper only observes `generate`'s calls and callbacks. With
the effort knobs on, requests they rewrite render a different prompt (the point); off, nothing changes.

Tests (host only, no GPU): `tests/test_health.py` (modes, fatal vs `ValueError`, stall allowance before and after the
first token, counters without float rounding, the real handler: `/health`, `/metrics`, 400 / 500 / 503, the SSE error
event, the wrapper's transparency), `tests/test_effort.py` (mapping, request keys win, idempotent, default effort
only for thinking requests, env validation, the rendered effort line).

## 0170 — Mia's prefill wins, ported (`fat` expert kernels, KDA projection in bf16)

Written offline (no GPU): the host tests ran and the CUDA compiled for sm_120 (clang + ptxas, register counts
below); the GPU tests are for the Sparks.

**Sources and licenses.** MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks has been AGPL-3.0 since 2026-09-07; before
that date it was MIT. Its Apache-2.0 fork Reederey87/glm53-flash-exl3-2x-dgx-spark (`6c337d2`) carries the E2/E3
grouped fat-expert kernels (`overlay/exl3_fat_moe.cu`, `exl3_fat_gemm.cu`, ticket scheduler, `cp.async` pipeline).
Those files come from Mia's MIT-era kit, with the fork's own changes under Apache-2.0. The `fat` kernels take their
structure from that fork and keep our arithmetic (NOTICE). PR #233 (the KDA projection's bf16 copy) was merged
into the AGPL repository, so only the idea is used here, re-implemented without its code.

**(1) `GLM53_TF_FAST_EXPERTS=fat`** (default stays `fast2`; per request `"tf_knobs": {"fat_experts": 0|1}`, header
block after 0093's `prefill_overlap`, 10 knobs). `exl3_fast.cu` namespace `fat`: fast2's arithmetic with Mia's data
movement.

- **Trellis words through the `cp.async` ring.** fast2 loaded each warp's words with `__ldg` one stage ahead into
  registers. Now NSA - 1 stages of weights and member rows are in flight (Mia S2b: +39-41% kernel throughput for
  them), and no registers hold words.
- **One rotated input for gate and up** (Mia's "gate/up Hadamard reuse"). The checkpoint's gate and up input sign
  vectors are identical: checked on the head node, 18/18 sampled experts of `07135ec0` have equal `suh` bytes, and
  `exl3_mm.shared_suh` checks every layer on the device at first use. `rot_in1` writes Xg only (0.6 GB of rot_in's
  1.2 GB at 8192 rows), and a stage holds one matrix of rows: half the gather traffic and shared memory.
  `GLM53_TF_FAT_SHARED_X=0` turns it off.
- **Swizzled row stages, 2 CTAs an SM.** The row stages are XOR-swizzled (Mia's `fm_swz`) instead of padded, so a
  gate/up stage is 16 KB and 3 stages are 48 KB. Two CTAs of 8 warps then fit an SM, with
  `__launch_bounds__(256, 2)`. clang/ptxas sm_120: 128 registers, no spills (fast2's gate/up: 224 registers, one
  CTA of 8 warps an SM).
- **Ticket scheduling** (Mia S2a). Each CTA claims items with one `atomicAdd` instead of a static stride.
  `GLM53_TF_FAT_TICKET=0` restores the stride. The expert of an item is walked forward from the CTA's last one, so
  there is no 4 KB offsets table in shared memory.
- `GLM53_TF_FAT_STAGES=3|4` (default 3; 4 stages leave one CTA an SM).

**Not taken:** Mia's orientation (rows as the mma A operand) and its epilogue. The sorted fat-row buffer is not
needed: we gather member rows by index into the ring.

**Exactness.** fat == fast2 == v1 bit for bit by construction. Every output element is the same chain of m16n8k16
mma: the same decoded A fragment, the same B values, ascending k tiles. It also gets the same transforms and
epilogue code. Only the data movement changed, and stages, ticket and CTA count cannot change a bit. With the
shared input, this also needs `rot_in1` == `rot_in`'s first output (the same code; tested).
Row-independent (0085: the member count only decides which B columns are zero-filled) and deterministic (the ticket
only decides which CTA runs an item). Because the bits are the same, the per-request knob shares snapshots with
fast2.

**(2) `GLM53_TF_KDA_PROJ_BF16=1`** (default 0; load-time, in `fastpf.settings()`, so both ranks must agree). Each
KDA layer keeps a bf16 copy of its fused input projection [q|k|v|f_a|g_a|b], 12576 x 4096 a rank. The copy holds the
4-bit weights' own values (s * q + b in fp32, rounded once), not the checkpoint's original bf16. Fast chunks multiply
it with `fast_qmm`'s one-accumulator bf16 kernel (`_fb16`, tile by shape; `GLM53_TF_KDA_BF16_TILE=bm,bn,warps,stages`
to tune, same bits). Decode, verify, exact prefill, the MTP head and FP8 prefill keep the 4-bit weights.

- **Memory:** 98.25 MiB a KDA layer and rank. With GLM-5.3-Flash's 34 KDA layers (`config.json`: 34 `linear_attention` of 45) that is +3.26 GiB
  a rank, made at load after the prepared-folder read (not stored in it).
- **Row independence.** Mia switches at M > 512 (FP8-Marlin below). We do not: a row-count switch would give a
  prompt's short last chunk or a lean tail other bits than the same rows in a long chunk, which breaks 0085. Every
  fast-chunk call uses the copy, and there is no small-M penalty, because our 4-bit path is a Triton kernel and not
  Marlin.
- **Exactness.** New arithmetic: the weights are rounded to bf16 (relative 2^-9), so fast replies differ from `0`
  in the last bits. Deterministic and row-independent: drafted == serial and resumed == fresh hold within the
  setting. It is not a per-request knob, because snapshots are not tagged by it.

**Expected** (arithmetic, nothing timed):

- **fat.** fast2's gate/up is register-limited: 8 warps an SM, 25% of DRAM bandwidth, barrier stalls. Doubling the
  resident warps and moving the weight loads onto the async ring should give gate/up +15-35% and down +0-15%.
  rot_in1 halves the input rotation (-3 ms a layer at 8192 rows). Experts are ~37% of a 32k prefill, so the
  estimate is **-6 to -12% prefill time** (1,060 -> ~1,130-1,200 tok/s at 32k).
- **KDA bf16.** Mia's -11-12% came from FP8-Marlin being 3.1-3.7x slower than bf16 at large M. Our 4-bit path already
  runs at ~58 TF/s (1.82 ms at 1024 rows), so a bf16 kernel at 70-85 TF/s saves ~0.3-0.55 ms a 1024-row sub-block
  and KDA layer: **~1-2% prefill time** for 3.3 GiB. Keep it only if `test_kda_copy_timing` shows >= 1.3x.

## 0180 — sessions in batch mode (`GLM53_TF_BATCH_SESSIONS=1`)

Written offline (no GPU); the host tests ran (fake model through the real batcher and store), the GPU tests are for
the Sparks.

**Problem.** 0120 built the batcher before 0110's store and left the store off in batch mode, so production had to
choose between 4 concurrent requests and resuming cached sessions.

**Change** (opt-in: `GLM53_TF_BATCH_SESSIONS=1`, both ranks checked equal; default 0 keeps 0120's behaviour).

- `sessions.SessionStore.bind_slots(states)`: one store for every slot. The index, budget, page slabs and LRU are
  shared; each slot has its own live-page map (`on(slot)` switches the tensors and the map every save / restore /
  `begin` acts on). Slots of another cache layout are refused at load.
- *Admission* (rank 0, `Batcher._plan` → `_session_plan`): `SessionStore.plan` (the longest stored strict prefix of
  the request's prefill mode whose draft caches fit, and the marks). When it resumes more than the free slots' own
  snapshots, the request goes to the free slot already holding most of the entry's pages (`batchplan.place_entry`,
  then least recently admitted), and `_admit` copies the entry into that slot (restore = copy-in) and resumes the
  first piece from the entry's snapshot; the slot's own snapshots are dropped (its caches changed).
- *Pieces* (`_piece`): the store's marks are the prefill's checkpoints in every piece; mark snapshots (and a piece
  end that is a mark) are saved right after the piece, the prompt snapshot (exact: the prompt end; fast: its last
  grid point) after the last piece; the reply snapshot when a drafted exact request ends (`_finish`). The same
  snapshots the lone engine saves.
- *Fast pieces* (`batchplan.piece_end` / `piece_grid`): a stored fast snapshot or mark sits on the 64-token grid,
  possibly between two bounds of the request's chunk grid C (e.g. `tf_knobs.prefill_rows=1024`); its first piece
  runs to the next bound, the rest stay on the grid. Pieces end on a multiple of C and of the snapshot grid.
- *Eviction vs live slots*: a restore copies pages into the slot's own caches, and a snapshot's KDA state / conv /
  pending rows / drafter window are copies, so an eviction (entry or slab) never touches a running slot; the slot's
  live map only avoids re-copying pages whose key (hence bits) it already holds.
- *Two ranks*: each admission's session plan (entry id and length, marks, rank 0's store digest) is shared right
  after its prompt inside the round plan's messages; rank 1 checks the digest and restores the same entry. Every
  save's decision (stored / skipped / duplicate, evictions) is shared where the save happens, inside the round both
  ranks run in the same order (0110's mechanism); rank 1 applies it and fails loudly on a divergence.
- *Memory*: at load the batcher adds slots only while `GLM53_TF_BATCH_RESERVE_GB` AND the store's budget stay free
  (it prints when the budget costs a slot); at admission a request counts free memory less the store's unused budget
  (`batchplan.admit_free`); the store keeps at least `GLM53_TF_BATCH_ADMIT_GB` free when it grows.
- Stats: `cached` (as before), `restored` (entry id) and `sessions` per request.

**Exactness.** Nothing new: a restored slot holds the bits the lone engine's restore gives (the same copy, into
caches of the same layout), pieces are `decode.prefill` resumed from snapshots (resumed == fresh), and the saves are
the lone engine's. So batched == alone == serial, with or without the store.

**Deferred.** Restores copy (no paged kernels, as 0110); stored sessions are not preferred over a slot's own
snapshot of the same length (no copy either way); spill to NVMe; marks for pieces' own boundaries beyond the store's
marks (a piece of an exact prompt is not a store checkpoint unless it is a mark).

**Fixes (2026-09-28)** (the 4 GPU failures of `results/T4-worker/test_batch_sessions_patches.log`; written offline,
the CPU fake-model tests reproduce both and pass now, the GPU tests are still for the Sparks).

- *No cache reuse* (`test_concurrent_interleaved_sessions_equal_alone[4-*]`: the fork request got `cached` 0, not
  512). Rank 0 plans an admission's marks when it admits it, and `SessionIndex.marks` placed a fork mark only at the
  common prefix with a *stored* entry (`sessions.py` `d = self.lcp(prompt, blocks)`). With as many slots as sessions,
  all 4 sessions over the shared system prompt are admitted in one round, before anything is stored: no fork mark.
  Their later turns resume from their own reply snapshots, so the fork point stays behind `resume + fork_min` and is
  never marked. On 3 slots the 4th session waited for a slot, saw the stored prompts and took the mark, which is why
  only `[4-*]` failed. Fix: the prompts in flight (running slots and earlier admissions of the same round, not
  cancelled) count as fork partners too (`Batcher._plan` passes them to `_session_plan` → `SessionStore.plan(others=)`
  → `SessionIndex.marks(others=)`, `sessions.common_prefix`). The marks still travel in the admission's session plan,
  so rank 1 is unchanged. Partners admitted together may each take the same mark; the second save is a duplicate,
  and only its snapshot copy is wasted.
- *Follower replay* (`test_follower_replays_rank0_sessions[*]`: `assert [1, 0] is None`). `Batcher._save` went
  through `GlmEngine._store_save`, which picks the side of the save decision by `self.rank == 0`. Every other message
  of a batch round picks its side by role (`_plan` sends, `follow` receives). The test's second engine (one GPU
  playing both ranks) is rank 0, so its `follow` *sent* save decisions instead of replaying rank 0's. On real rank 1
  both rules agree, but the save was the only rank-keyed message in the round protocol. Fix: `Batcher.following`
  (set by `follow`); `_save` sends (`_share(save_all(...))`) or applies (`save_all(forced=_share(None))`) by that. The
  lone engine's `_store_save` is unchanged.
- Tests: `test_fake_follower_role_is_follow_not_rank` (a rank-0 follower batcher on the fake model) and
  `test_fake_concurrent_forks_leave_a_fork_mark[3|4-exact|fast]` (4 sessions submitted at once; on 3 slots the 4th
  now resumes at the mark too) fail before the fix and pass after it. The GPU tests were right and are unchanged.
- *Memory (unchanged, for the record)*: the store's whole `GLM53_TF_SESSION_GIB` is set aside at load (a slot is
  added only while free − per-slot − budget ≥ `GLM53_TF_BATCH_RESERVE_GB`) and at admission
  (`admit_free` = free − (budget − used) ≥ `GLM53_TF_BATCH_ADMIT_GB`). The store also keeps
  max(`GLM53_TF_SESSION_RESERVE_GIB`, `GLM53_TF_BATCH_ADMIT_GB`) free when it grows. Not in the budget: each slot's
  own ≤ 2 snapshots (0120's, which can outlive an evicted entry) and transient duplicate / skipped snapshots; the
  admission minimum covers them.

## 0190 — prefill glue (`pfglue.py`; `glue.py`, `latent.py`, `pfoverlap.py`, `lean.py`, `decode.py`)

Written offline (no GPU): host tests and Triton CPU-interpreter checks ran; the GPU tests and `tests/cuda/bench_glue.py`
are for the Sparks. Nothing here is timed.

**Problem.** The 28,045-token cold prompt (fast + lean 8192-row chunks, fast2, overlap, latent KV) takes 23.09 s
(1,154-1,209 tok/s; vLLM 1,448). Past the routed experts (4.7 s) the time is glue: `hc` 2.9 s, `moe.router` 1.3 s,
`moe.shared` + `moe.combine` 1.8 s, the MTP head ~1.3 s, sparse attention 3.0 s, `dsa.o_proj` 1.9 s.

**Change.** Five switches, each its own knob, default off. 1-4 travel per request in 0090's header (4 knobs appended
after 0170's `fat_experts`; both ranks must run the same patch set).

| knob | what | bits |
| --- | --- | --- |
| `moe_glue` bit 1 | windows of 64+ rows group their (row, slot) picks by a stable sort + scatter (`glue.group_sorted`) instead of `_group`, one program walking R x 9 picks twice and R member columns twice (O(R) serial: ~1 us a row a MoE layer, most of `moe.router`) | the same integers (ids, count, every member cell), element for element |
| `moe_glue` bit 2 | the router's 8 K-slice partials summed in registers (`_router_fused`), no [8, R, 288] fp32 buffer (150 MB a layer at 8192 rows) | each slice the same dot chain, slices added in order: same bits (GPU test) |
| `moe_glue` bit 4 | lean chunks' combine reads the shared expert's fp32 rows where its matmul wrote them (`_combine_s`), no copy into `ey` | the same fp32 adds in the same order |
| `mtp_window` = N | a prefill of n tokens runs the MTP head only from lo = (n - N) rounded down to 64; below lo its caches (latent / K,V, index keys and gates, the pools they complete) are zeroed and `mtp_len` advances | main model untouched; drafts may change, replies never (verify + keyed sampler decide every token); deterministic (zeros whatever ran before); lo depends on n only, not on C |
| `hc_fused` bit 1 | in 0084's pipelined slabs, hc_post and the next hc_pre's 24 mixing dots + square sums in one kernel (`_hc_post_part`): the new bf16 tiles go from registers into the dots instead of being re-read, one launch fewer | `_hc_post`'s expressions operand for operand, `_hc_partial_mm`'s tiles / dot chain / square sums; the GPU test decides (a Triton reduction's order follows the compiler's layout) |
| `hc_fused` bit 2 | `_hc_finish_u`: hc_pre's finish with its 16 partial loads unrolled (all in flight; the finish was 16 dependent L2 round trips a row) | same adds in the same order; fast chunks only (decode keeps `_hc_finish`) |
| `attn_bm32` | bf16 fast chunks use the 32-query latent attention tile (FP8 chunks already do; 0060: -24% a layer at 1024 sparse rows, same bits as 16 on the real shapes) | same bits (GPU test); prefill need not match decode anyway (fastpf), only row independence, and a tile is one row's heads |
| `GLM53_TF_LATENT_TC=1` (load-time) | bf16 fast chunks run 0083's tensor-core `absorb_tc` / `expand_tc` instead of the fp32 FMA-pipe absorb / expand (most of `dsa.o_proj`, part of `dsa.proj`) | NEW arithmetic (q, u and kv_b's tile rounded to bf16 at the dot; bf16-matmul precision, not e4m3): fast replies change; deterministic, row-independent; snapshots tagged G + 2 (`pfgrid.TC`); refused per request |

**Exactness.** Items with "same bits" leave every committed state and reply unchanged, so snapshots and the session store
are shared across the switch. The ones that depend on Triton's lowering (router fusion, hc fusion, 32-query tiles) are
checked bit for bit by `tests/cuda/test_glue_patches.py` and `bench_glue.py` prints "bitwise True/False" per kernel: a
knob whose line says False must stay off (its per-request switch would otherwise mix snapshots of two arithmetics).
The MTP window changes only the MTP head's rows: replies are the full head's (tested with MTP, DFlash2 and auto drafts,
greedy and sampled, fresh and resumed); a resumed prefill keeps the head rows the earlier request wrote, so its drafts
(not replies) can differ from a fresh one's, as 0065 allows for the DFlash2 ring. Batch mode prefills in pieces (each a
prefix), so each piece keeps its own last N positions: a smaller saving, no correctness issue.

**Expected** (arithmetic from the measured profile; ranges; not timed):

| item | 28k (23.09 s) | 112k (95.8 s) |
| --- | ---: | ---: |
| `moe_glue` = 7 (grouping ~-1.0 s, router partials ~-0.1 s, shared copy ~-0.2 s) | -1.0 to -1.3 s | -4.0 to -5.4 s |
| `mtp_window` = 4096 (MTP head ~85% / 96% skipped) | -1.0 to -1.1 s | -4.5 to -4.8 s |
| `hc_fused` = 3 (the hc lap is ~70 us a token without gather waits, ~25-30 of it hc_post's DRAM floor) | -0.2 to -0.6 s | -0.8 to -2.4 s |
| `attn_bm32` (sparse -24%, dense -20%) | -0.6 to -0.8 s | -2.5 to -3.2 s |
| same-bits + drafts-only total | -2.8 to -3.8 s: ~1,380-1,450 tok/s | -11.8 to -15.8 s: ~1,330-1,400 tok/s |
| opt-in `GLM53_TF_LATENT_TC` (`dsa.o_proj` 1.9 -> ~0.8 s, `dsa.proj` -0.3 s) | -1.1 to -1.4 s more: ~1,460-1,570 tok/s | -4.5 to -5.5 s more: ~1,410-1,500 tok/s |

Watch the MTP window's effect on decode (tokens a round with MTP / auto drafts on long prompts): the head's own attention
no longer sees the prompt's early part. Remaining headroom, not attacked: `kda.proj` runs at ~52 TFLOP/s (12576 x 4096
at 8192 rows; ~110 measured mma.sync peak): a better-tiled kernel could save ~0.5 s at 28k; `kda.chain`'s state kernel
is bound by reading W / U / Q / K in fp32 (bf16 operands were rejected for accuracy in 0081); `moe.combine` already
reads `ey` at ~215 GB/s (only a bf16 `ey` or a combine fused into the down epilogue would cut it: new bits / a fixed-
order cross-expert reduction); the hc lap's gather waits (~30 us a token at 28k) are exposed NCCL time, not hc.

**GPU round 1 (T7, image z) and fixes.** `moe_glue` bitwise everywhere; `moe_glue=5` (grouping + shared in place) is
+5-6% prefill and in production. The one-kernel router (bit 2) is bitwise but slower at 8192 rows (2.2 vs 1.3 ms: 576
long programs against 4,608 short ones), so leave bit 2 off. `hc_fused` bit 1 could not launch: with `num_stages=3`
Triton 3.7.1 staged the 4 stream tiles, both partial tiles and 4 fn tiles a step in shared memory (128 KB > the 99 KB
a block on sm_121; reproduced offline by compiling for sm_121 with 16-byte-aligned pointers, 8 KB with
`num_stages=1`, now the default `glue.HC_FUSED_STAGES`); a kernel that still cannot launch now falls back to the
separate kernels (same bits) and says so once. The offline TTGIR shows the fused kernel's square sums reduce in the
same blocked layout ([1, 8] x [4, 8] x [4, 1]) as `_hc_partial_mm`'s. `hc_fused` bit 2 (`_hc_finish_u`) passed
bitwise. `attn_bm32`: the kernel test passed bit for bit; its engine test failed only because it also turned on the
fused hc (now a separate step). `mtp_window`: replies equal with and without the window (passed); two test bugs
failed it: the follow-up needed a drafted snapshot (the cold serial run leaves snapshots without MTP rows, so nothing
resumed), and on the latent engine the "main state" compared included the MTP layer's own indexer caches, which the
window zeroes by design. `GLM53_TF_LATENT_TC` stays off (it changes replies). `bench_glue.py` now reports a failing section
and keeps going.

**On the Sparks.** `python tests/cuda/bench_glue.py` first (a minute: old vs new kernels at 1024 / 4096 / 8192 rows,
with a bitwise check each), then `pytest -q -s tests/cuda/test_glue_patches.py`, then per-request A/B on one load:
`"tf_knobs": {"moe_glue": 7}`, `{"mtp_window": 4096}`, `{"hc_fused": 3}` (with `prefill_overlap` on), `{"attn_bm32": 1}`,
each against 0, with `GLM53_TF_PROFILE=1`. `GLM53_TF_LATENT_TC=1` needs a restart and the `exact` suite / MMLU.

**Update 2026-09-28 (W1): MTP prefill cache rows, `GLM53_TF_MTP_PREFILL_CACHE=1` (load-time; `pfglue.cache_absorb`).**
Why `mtp_window` did not work out: (1) in batch mode (production) it is a no-op: `batch._piece` prefills
`prompt[:end]` for each 2,048-token piece, so `lo = end - W` is below the piece's start for any W >= 2,048 (W1:
1,156 / 1,130 tok/s at 24.5k / 98k, the same as off); (2) where it acts (single stream, Z2) it zeroes the head's rows
below lo, and the head's indexer scores a zero pool exactly 0 (`sum_h w_h relu(q . 0)`) while real pools score
below 0 whenever the signed head weights w_h outweigh their positive dots: zero pools win those top-512 places, the
head then attends to zero latent rows (logit 0, value 0), its attention output is diluted and MTP acceptance falls
(Z2: decode 81 -> 56 tok/s after a 112k prompt). A larger W only moves the problem further back. The saving it was
after comes without it: a prefill absorbs the prompt into the head only for the head's caches (latent row, index
key / gate, pools: functions of the row's own input through row-local kernels), and never reads the head's logits
or output rows (the pending last row is absorbed with its kept token in decode). So with the switch the prefill runs
exactly the calls that produce those entries (`a.proj`, kv norm, `latent_write`, index `kw` / gate,
`index_update`) and skips the head's queries, selection, sparse attention, o_proj, MoE, their all-gathers and the
final norm / lm_head: the cache entries, `mtp_len`, drafts and replies are the full head's, bit for bit. Both ranks
must agree (other collectives): `load_settings` now carries it next to `GLM53_TF_LATENT_TC`. Latent KV only; with
`mtp_window` set its rule still applies. Tests (`test_glue_patches.py`): `test_engine_mtp_prefill_cache_state` (the
whole committed state on == off, 65 / 1,000 / 3,000 tokens past the dense limit, C = 256 / 1,024, window 0 / 256),
`test_engine_mtp_prefill_cache_replies` (MTP, DFlash2, auto: replies and keeps equal; resumed == fresh): 21/21 with
the window tests on image `w1`. Measured on the 4 x 256k load: +5.5% / +5.3% prefill at 24.5k / 98k, +11% with
`attn_bm32` (docs/RESULTS.md, W1).

## 0200 — batched parallelism (`batch.py`, `batchplan.py`)

**Measured (0120, `GLM53_TF_BATCH=4`, `results/B2`) and what it means.** 4 streams: 61-74 tok/s aggregate, per
stream 27-64; 2 streams 56-60. Per stream the batched rate is 0.6-0.7x the same prompt's lone rate (lone, `B1`:
46 / 47 / 66 / 98 tok/s for the four bench prompts; batched 27-33 / 34 / 42 / 62-64): the spread is the prompts'
own draft acceptance (the 98 tok/s prompt commits ~2x the tokens a round), not the scheduler; every active slot
already gets one window every round. The aggregate is below the sum of the per-stream rates because the streams
barely overlap: the four prompts' first tokens are staggered (one piece a round, then a fair-share wait between
pieces) and the fastest stream ends early. `results/B1` is the unbatched server: its stall run's 105.8 s TTFT is
three 1,024-token replies served first (~75 s) plus the 31 s prefill; batched (`B2`, one decoder) the same prompt
took 57.7 s (the 0.5 share), longest decode gap 3.2 s (a fast piece rounds up to the chunk grid).

Model of a batched round (two Sparks: verify 31/39/45/51/56/62/68/74 ms at 1-8 rows, past 8 ~6-7 ms a row; MTP
draft 2.04 ms + 1.68 a chained draft per slot; sampling + commit ~0.5 ms a slot; parity copy 0.6 ms a slot): 4 x
~3.5 rows = 14 rows: verify ~113 ms, MTP drafting 4 x 5.4 = 21.6 ms, parity copies 2.4 ms, host ~2 ms: ~139 ms
for ~8.8 tokens (63 tok/s; 70-80 with one high-acceptance stream), when every round replays a graph. A round that
meets a new (slots, rows, modes) key runs eagerly (+10-25 ms) and then captures (+capture, instantiation); with
2-4 slots of 1-8 rows most keys are met once, and the 256-graph cap fills with them.

**Knobs** (each off by default = 0120's behaviour; both ranks must agree, checked at load):

- `GLM53_TF_BATCH_CAPTURE_AFTER=N` (1): a key is captured on its N-th sighting (`batchplan.Sightings`); before that
  its rounds run eagerly through the same capturable code (same bits). Keys met once no longer pay a capture and
  the graph cap is kept for keys that recur. Suggested 3.
- `GLM53_TF_BATCH_PARITY_KEY=1`: each slot's KDA parity goes into the key instead of `_parity0`'s copy of a state
  left in buffer 1 (71 MB a slot, every round: every commit flips the parity). The graph bakes in `rec[cur]` /
  `rec[1 - cur]` as the engine's own (rows, parity) graphs do. Keys at most double. -0.6 ms a slot and round.
- `GLM53_TF_BATCH_MTP=1`: `MtpChains` drafts every slot's MTP chain together: one head pass absorbs every slot's
  backlog (`mtp_multi`: row-local kernels over all rows, the DSA cache write / attention per slot on its own head
  cache through `_dsa(..., caches=)`), then one pass per chained step over the slots still drafting. Each chain
  follows `decode.draft` step for step (own sampling, confidence or cost-depth stop, positions, head cache), and
  each row has the bits of the slot's own head pass, so the drafts are the ones it drafts alone. An MTP step reads
  ~270 MB a rank, ~165 MB of it the vocabulary head: 4 slots read it once instead of 4 times. The drafts of a pass
  are sampled with one all-gather (`sample_drafts`: `sample_rows`' candidates, draw and probability per row), so a
  4-slot chain step has one hard sync instead of four. Eager passes (the per-slot head graphs are kept for a single
  MTP slot). Expected -8 to -12 ms a 4-slot round (21.6 -> ~10 ms), -4 ms at 2 slots.
- `GLM53_TF_BATCH_ROW_MS=F` (0): batch-aware cost-derived depths (`BatchDepth`, 0120) extend the verify table past
  8 rows at no less than F ms a row. 0120 extends it along the table's upper-half slope (5.75 ms on the table
  above), below what rows of other sequences cost (6-10 ms): 3-4 sequences drafted too deep. Suggested 6.5.
  Changes only drafts (never bits).
- `GLM53_TF_BATCH_SHORT=N` (0): every prompt with at most N tokens left prefills in the round it is admitted,
  several in one round, outside the fair share (`batchplan.pick_pieces`); long prompts keep
  `GLM53_TF_BATCH_PREFILL_SHARE`. Four simultaneous short prompts get their first tokens in one round instead of
  one piece + one share-wait apart. Suggested 1024 (a piece of that size is ~0.4-1 s).
- `GLM53_TF_BATCH_PAD=2,4,8` (off): each slot's window padded to the next listed size with its last token (fewer
  keys). Padded rows are verified like drafts that can never be accepted (the accept rule reads only the real
  drafts) and `commit` gets the padded row count (KDA replay of the kept prefix), so every kept bit is the
  unpadded window's. A padded row costs ~6 ms: only worth it when `round_kinds` shows mostly eager rounds.
- Stats: each request's `round_kinds`: rounds by kind (`alone`, `graph`, `eager`, `capture`), `pad_rows`,
  `mtp_batched`, and the wall ms of its verify rounds (`verify_ms`), of its drafting (`draft_ms`) and of other
  requests' prefill pieces it waited through (`piece_ms`). `bench/multiturn.py --modes concurrent` prints TTFTs,
  tokens a round, ms a round (+ piece seconds) and the round kinds per stream.

**Exactness.** CAPTURE_AFTER, PARITY_KEY: which of two identical code paths (captured or not, parity baked in or
normalized) runs. PAD: row independence, and the padded rows are never kept (the same argument as rejected drafts).
MTP, ROW_MS: drafts only. SHORT: which pieces run when (resumed == fresh). Each batched reply stays byte-identical to
the same request served alone.

**Expected** (the model above; steady state, all slots decoding, graphs hit): 4 streams ~72-80 tok/s (0120's
model 63-70 steady state; +10-17%: MTP -10 ms, parity -2.4 ms, ROW_MS 6.5 ms moves 4-slot windows from ~3.5 to
~2.8 rows); a typical stream ~18-20 tok/s, a high-acceptance one ~35-40. 2 streams ~58-62 (+8%). The measured
aggregate gains more where 0120 lost rounds to captures and stagger (the concurrent bench's `round_kinds` shows
how many). The ceiling is structural: a verify row costs ~6-7 ms (its routed experts; rows of different sequences
share few of 288), so 4 x 3 rows cost ~2.3x one window.

**Prefill under load.** TTFT ~= alone / share (time slicing; `B2`: 57.7 s = 1.86 x 31 s at 0.5). share 0.7: ~44 s
with the decoders at ~30% speed during it; 0.8: ~39 s. The longest decode gap is one piece: a fast piece rounds up
to the chunk grid (auto rows up to 8192: ~3 s); `"tf_knobs": {"prefill_rows": 2048}` on the long request or a
smaller `GLM53_TF_BATCH_PIECE` with a smaller grid trades ~5-10% prefill speed for ~1 s gaps.

**Not done, and why.**

- *Mixed rounds (prefill piece + decode rows in one forward, vLLM style).* The gain in vLLM comes from decode rows
  riding on the prefill's weight reads. Here a decode row's cost is its routed experts (~6 ms of expert reads a
  row), and the fast prefill kernels (fast2 / fat experts, bf16 gathers, chunked KDA) are not the row-invariant
  decode kernels: decode rows through them would change their bits (replies != served alone). Kept exact, decode
  rows need their own grouped expert launch (their expert reads are not shared), their own fp32 gathers and their
  per-slot KDA / attention: what is shared is the dense 4-bit matmuls (fast_qmm has qmm's bits), ~10-12 ms once
  per piece of 0.4-7 s: <1-3%. Meanwhile each decoder advances one window per piece (a round as long as the piece)
  unless pieces shrink, and pieces under ~1k rows lose prefill speed (every piece reads all experts once, ~0.3 s).
  The exact alternative (prefill pieces on the row-invariant kernels, expert reads shared with the decode rows)
  prefills ~2x slower (500-670 vs ~1,100 tok/s) and differs from the request's fast prefill alone. Time slicing
  with fast pieces dominates both.
- *One launch per layer for per-slot KDA / attention / indexer* (kernels taking per-sequence row offsets and state
  pointers, graphs keyed by total rows): the KDA chain is a CUDA extension (`kda.cu`), untestable offline here.
  With CAPTURE_AFTER / PARITY_KEY the key churn it would remove is mostly handled; what remains is ~(N-1) x 45
  layers x 3-5 small nodes and the chain's 32-block grids (~2-5 ms a 4-slot round, 2-4%). Design:
  `chain_kernel` grid (heads, sequences) with a device table (row offset, rows, rec in/out, conv, scratch) over
  one `[slots, 2, layers, H, 128, 128]` state tensor; `kv_write` / attention / indexer with a per-row sequence id
  -> (cache base, pos); graphs keyed by (T, modes). 3-4 days with a GPU.

**GPU window: quick check (in this order).**

1. Tests: `PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q tests/cuda/test_batch_parallel_patches.py`
   then `test_batch2_patches.py` and `test_batch_sessions_patches.py` (knobs off: 0120 / 0180 unchanged).
2. Worth it? Same load twice (`GLM53_TF_BATCH=4`), knobs off, then on
   (`GLM53_TF_BATCH_CAPTURE_AFTER=3 GLM53_TF_BATCH_PARITY_KEY=1 GLM53_TF_BATCH_MTP=1 GLM53_TF_BATCH_ROW_MS=6.5
   GLM53_TF_BATCH_SHORT=1024`, passed through `scripts/serve.sh`'s environment):
   `python3 bench/multiturn.py --base http://127.0.0.1:8080 --model GLM-5.3-Flash-Uncensored --modes
   batchexact,concurrent --streams 1,2,4 --reps 2 --long-tokens 256 --out results/B3/<off|on>.json` (~3 min).
   Look at: `batchexact` all true; aggregate at 2 / 4 streams; per stream `ms/round` (verify + own drafting) and
   `rounds {...}` (with knobs off, a large `capture` + `eager` share confirms the key churn; on, mostly `graph`);
   `ttft` spread at 4 streams (SHORT). Worth keeping if 4 streams gain >= ~8% with batchexact true.
3. Prefill under load: `--modes stall --streams 4 --long-tokens 1024 --doc 28000` with
   `GLM53_TF_BATCH_PREFILL_SHARE=0.5` and `0.7` (TTFT vs the decoders' tok/s).
4. Optional A/B: `GLM53_TF_BATCH_PAD=4,8` (only if step 2 still shows many eager rounds).

Tests: `tests/cuda/test_batch_parallel_patches.py`.

## 0220 — FP8 latent KV cache (`GLM53_TF_KV_DTYPE=fp8`)

**Problem.** 4 concurrent threads x 256k context need 4 slots of capacity-sized caches: 13,616 B a token a rank in
bf16 (docs/MEMORY-1M.md), 13.26 GiB a node for 4 x 262,152 slots, and with the session store and prefill buffers that
left MemAvailable at 2-4 GiB under load (results/Y1). The latent rows are 90% of it (12 x 1,024 B).

**Change.** `GLM53_TF_KV_DTYPE=bf16|fp8` (`latent.kv_dtype`, load-time; fp8 needs `GLM53_TF_LATENT_KV=1`;
`GlmEngine` checks both ranks agree; `decode.Engine` sets `w.meta["kv_fp8"]` before any `State`).

- Layout: `latent.caches` allocates each latent cache (the 11 DSA layers' and the MTP head's; keys = values as before)
  as uint8 `[capacity, ROW8 = 528]`: bytes 0-511 the e4m3 values, 512-515 the fp32 scale, 516-527 padding so rows stay
  16-byte aligned (128-bit loads in the sparse gather; unpadded 516 B would save 144 B a token). Every consumer that
  copies rows generically (sessions' pages, batch slots, snapshots, pfglue's MTP-window zeroing, `State.clone`)
  moves the scale with its row; an all-zero row reads as zeros. The indexer's keys, gates and pool keys stay bf16:
  they decide the token selection.
- Write (`_lwrite8`, from `latent_write`): each row on its own. s = 2^e, the smallest power of two with amax / s <= 448
  (from amax's exponent and mantissa bits: no log2, exact); y = x / s (exact); y rounded to the e4m3 grid to nearest
  even in fp32 bit arithmetic (3 mantissa bits; below 2^-6 the fixed 2^-9 step via +- 1.5 x 2^14), sign kept (-0.0
  too), then converted (an exact relabelling, whatever the backend's cvt rounding). Equal to torch's
  `float8_e4m3fn` conversion byte for byte (`quantize_rows_reference`; checked on every finite bf16 value <= 448).
  Triton's CPU interpreter drops the carry when its own conversion rounds across a power of two (7.84 -> 4.0),
  another reason not to rely on it.
- Read (`_lrows`, in `_lchunks` and `_lsparse_chunks`, `FP8` constexpr): e4m3 -> fp32 x s -> bf16. With a
  power-of-two scale this is exact, so the tile fed to the dots is exactly the bf16 tile of the dequantized row and
  the rest of 0060's arithmetic (16- and 32-query tiles, online softmax, chunk merge) is unchanged: an FP8 cache
  gives the bits of a bf16 cache holding `dequantize_rows` of it (tested).
- Tags: `decode.Snapshot.kv` (0 / 1) set by `take_snapshot`; `restore` refuses the other format. `sessions.KV_TAG`
  (b"" for bf16, so earlier keys are unchanged; b"kv:fp8") is hashed into every entry and page key. The fastboot
  calibration key includes every `GLM53_TF_*`, so an fp8 load calibrates its own table. `knobs.LOAD_ONLY` lists
  `kv_dtype`. batch.py needs no change: its rounds call `latent._attend` per slot and size slots from the tensors.

**Exactness.** Quantization is per row and depends only on the row (the same latent gives the same bytes whatever
the window, chunk, slot or position), reads are exact, and the attention arithmetic is 0060's. So every 0060 / 0085
guarantee holds within the fp8 configuration: verify rows == serial rows (drafted == serial), chunk size changes no
bit, resumed == fresh, batched == alone. Against bf16 KV the stored latent loses precision (e4m3: relative step 2^-3
to 2^-4, rms error ~2.6% a value; half a step at most), so replies differ: a new configuration with a quality gate.

**Memory.** 12 x 528 + 512 (MTP index keys + gates) + 768 (pool keys) + 48 (0050 scratch) = **7,664 B a token a rank**
(bf16: 13,616; 0.56x). At CONTEXT=262144: 1.86 GiB a slot instead of 3.31; 4 slots 7.45 GiB instead of 13.26
(docs/MEMORY-4x256k.md). Decode past 2,051 tokens reads half the latent bytes per selected token.

**Tests.** `tests/cuda/test_fp8_kv_patches.py` (GPU) and `tests/test_fp8kv_interpreter.py` (host, Triton's CPU
interpreter; run it in its own pytest process: another module importing Triton first disables the interpreter).

## 0230 — one-shot RoCE all-gather (`GLM53_TF_COMM_BACKEND=roce`; `roce.py`, `roce.cpp`, `roce.cu`)

**Problem.** A decode step makes 90 all-gathers of 16 KiB a row (`docs/COMM-ANALYSIS.md`); NCCL's captured
all-gather costs ~27 us whatever the size (its kernel hands off to NCCL's proxy thread), ~2.4 ms of a ~30 ms
one-row step, all latency.

**Change.** b12x's "RoCEnante" (local-inference-lab/b12x `b12x/comm/roce`, PRs #295/#315 + #438's fixes; Apache-2.0,
attributed in `NOTICE` and TensorFold's `THIRD_PARTY_NOTICES.md`), ported: each rank owns a pinned host region
(`recv[src][slot]`, `flag[src][slot][hca]`, `send[slot]`, a control record); one kernel stages the shard into
`send[seq & 1]`, rings a doorbell, a busy-spinning C++ proxy thread (libibverbs, in the torch extension) RDMA-writes
the slot and then a 4-byte sequence flag on the same RC queue pair, striped over both CX7 PCIe functions
(rocep1s0f1 / roceP2p1s0f1); the kernel spins on the peer's flags with system-scope acquire loads and copies every
shard out in rank order (NCCL's layout). The epoch is on the device, so captured graphs replay it. Changes from b12x:
CUDA C++ instead of CuTe DSL, dim-0 gathers of any size and alignment, a wall-clock timeout, one GID per port, HCA
pairing by subnet, a probe against NCCL at load, the engine integration.

- What goes over RoCE: `comm.fast_gather` call sites only — `forward.gather` (every block's partials: decode,
  verify, MTP head, batched rounds, prefill chunks that fit), the samplers (`decode.sample_rows`,
  `batch.sample_multi` / `sample_drafts`), the DFlash2 drafter's rows and candidates, 0084's pipelined exchanges —
  and only when one rank's shard is <= `GLM53_TF_ROCE_MAX_KB` (a 16-row fp32 window at the default; prefill's
  64+ row chunks stay on NCCL). Control exchanges (`GlmEngine._share`, settings checks, calibration, the
  follower's idle wait for a request) stay on NCCL, which may wait forever.
- Exactness: an all-gather moves bytes; the consumers and the rank-0-first sums are unchanged. Probe at load:
  gathers of 4 B to the slot size, aligned and not, compared byte for byte with NCCL's result on the same inputs.
- Robustness: GIDs detected per port at start (lowest RoCE v2 IPv4-mapped index, like `detect-gids.sh`;
  `GLM53_TF_ROCE_HCA=name:gid` forces); both ports if both ranks have them on common subnets, else one. Setup is
  collective: any failure on either rank -> both serve on NCCL with a log line (`GLM53_TF_ROCE_FALLBACK=error`
  refuses instead). At run time a wait gives up after `GLM53_TF_ROCE_TIMEOUT_S` (120 s: generous, since a peer
  compiling a Triton kernel the other has cached can be late by tens of seconds), records peer / HCA / sequence
  in host memory and poisons the runtime (later launches are no-ops); the samplers check after every step and
  raise a diagnosis (was the peer's flag in this host's memory — sparkring #278's "delivered but not seen"
  signature — or never sent; what this rank's proxy posted); the engine error is fatal (0150), and the marker
  `GLM53_TF_ROCE_MARK` makes the restarted pair use NCCL until it is deleted.
- Cost: 6 slots of pinned memory (1.5 MiB at 256 KiB), one CPU core busy-spinning while exchanges flow (naps after
  ~1 s idle; `GLM53_TF_ROCE_CPU` pins it), the extension built at first use (~1 min, cached in /cache).

**Expected** (b12x measured 10.5-12.5 us all-gathers of 128 B-8 KB on 4 Sparks, graph replay): 1-row step
90 x (27 - ~12) us = ~1.3 ms (~4-5 %); 8-row window ~1.2-1.6 ms; MTP step ~0.03 ms; prefill ~0 at the default
threshold. To be measured with `tests/cuda/bench_roce.py` (both nodes: bits, eager / graph / step latency, fault,
soak; `loopback` on one node).

**Tests.** `tests/test_roce_logic.py` (host only: GID detection on a fake sysfs incl. a moved index, pairing,
knobs, the collective `select` with two fake ranks: settings mismatch refused on both, marker and failed setup
fall back on both, `error` refuses on both; `RoceComm` dispatch). `tests/cuda/test_roce_patches.py` (one GPU, no
RDMA: a host thread plays the peer through the real kernel — sizes 1 B-256 KiB, unaligned, dtypes, both rank
positions, graphs replayed, two streams, timeout -> poison + diagnosis + marker, poisoned replays are no-ops; the
engine with every model exchange through the kernel == the `_TwoCopies` engine bit for bit: windows 1-8, replies
drafted == serial, resumes). Needs the two Sparks: the RDMA proxy, two-rank replies vs NCCL, latency, soak.

## 0260 — routed experts that share a trellis decode between two passes (`GLM53_TF_FAST_EXPERTS=once`)

Written offline (no GPU): the host tests ran; the CUDA compiled with clang + ptxas for sm_121 (register counts below).
The GPU tests and `tests/cuda/bench_experts.py` are for the Sparks.

**How often fat decodes a weight tile (the premise, checked in the code).** PARADIGMS row 4 assumed that a tile is
re-decoded for every 16-32 member rows (0083's "~15 TF/s, MG = 2"). That holds for the exact path (`grouped_loop`)
and v1, not for fast2 / fat. There an item is (expert, pass of 64 members, 128-column block). Each warp decodes its
own column tiles straight into mma A fragments once per item, and every fragment feeds all 8 n8 groups (64 members).
So a tile is decoded ceil(n / 64) times a chunk (down: ceil(n / 128) from 4096 rows). With n ~ 8 C / 288 members an
expert (`test_decode_count_table`, top 8 of 288, skewed = Zipf 0.8):

| chunk rows | members an expert | fat: decodes a tile (gate/up, down), uniform | skewed | once: uniform / skewed |
| ---: | ---: | --- | --- | --- |
| 1024 | 28 | **1.00** (1.00, 1.00) | 1.16 | 1.00 / 1.05 |
| 2048 (production) | 57 | 1.15 (1.15, 1.15) | 1.44 | 1.00 / 1.15 |
| 4096 | 114 | 1.75 (2.08, 1.08) | 1.96 | 1.06 / 1.35 |
| 8192 | 228 | 3.34 (4.01, 2.02) | 3.44 | 1.68 / 1.94 |

At 1024 rows fat already decodes every tile once a chunk. The redundancy is real only from ~4096 rows.

**Why not "all members of an expert in one CTA".** fat's bits need one ascending K chain per output element in
registers. A CTA's accumulators are 2 x 128 fp32 columns a member for gate/up: 64 members = 64 KB, which is what fat's
two CTAs an SM hold (128 regs x 512 threads). 256 members would need the whole register file. Splitting K or staging
accumulators in shared memory either changes the bits or costs more than the decode. Decoding into a bf16/fp16
global scratch reads 4x the weight bytes a pass, which is worse at every chunk size.

**Change** (`exl3_fast.cu` namespace `once`, `exl3_fast.cpp` `gateup_once` / `down_once`, `exl3_mm.py`):

- A CTA is two fat CTAs ("halves": 2 x 8 warps for gate/up, 2 x 4 for down) that claim one unit by ticket.
- **Pair unit:** passes 2q and 2q + 1 of one (expert, column block). Warp (h, mat, slice) decodes the column tiles
  2i + h of its MTL = 2 and swaps them with its partner (1 - h, mat, slice) through a 2 x 512 B shared slot (one
  64-thread named barrier a k tile, slots alternating by k-tile parity). The words are loaded once for both halves.
- **Split unit:** one pass, two adjacent column blocks (an expert's odd last pass; every pass with
  `GLM53_TF_ONCE_PAIR=0`). The halves decode their own tiles as in fat and share the member rows of the stage (loaded
  once, not twice). An odd column-block count leaves the second half empty on its last split unit.
- A plan kernel gives the units: per expert, floor(P / 2) x NBLK pair units (nb fastest), then ceil(NBLK / 2) split
  units for an odd P. A tile is then decoded ceil(P / 2) times.
- Shared memory (<= 99 KB, one CTA an SM): gate/up with the shared input 3 x 24 KB stages + 16 KB exchange = 88 KB;
  own input (2-k-tile stages) 76 KB; down 68 / 67.6 KB. ptxas sm_121: gate/up (shared input) **126 registers, no
  spills** at 512 threads; down 165 / 236 registers at 256 threads, no spills; gate/up with its own up input (not
  the real checkpoint) 128 with 8 B of spills.
- `GLM53_TF_FAST_EXPERTS=once` sets `FAT` and `ONCE`. The per-request knob `fat_experts` still switches between the
  configured fat family (once here) and fast2. `GLM53_TF_ONCE_MIN_ROWS=N` runs once only for windows of >= N rows
  and fat below. It is the same bits either way, so the switch keeps 0085. `GLM53_TF_FAT_STAGES` stays fat's: once
  always uses 3 stages.
- Timing probes (`probe=1`: no trellis decode, `probe=2`: no mma; outputs are garbage) exist only in the bindings,
  for `bench_experts.py`.

**Exactness: bitwise == fat == fast2 == v1.** Every output element is the same chain of m16n8k16 mma over ascending
k tiles, with the same A fragment. `decode_tile` of the same word is a pure function, so the partner's decode is the
value this warp would have computed. The B values are the same fp16 rows, then fat's transforms and epilogue code
run. The unit kind, the half, the CTA and the ticket order only move data. It is row-independent: a pair's outputs
depend only on its own row, and the member counts decide only how items group into units and which B columns are
zero-filled. It is deterministic (no atomics on data). It shares snapshots with fast2 / fat (the per-request knob
may mix them).

**Expected** (arithmetic, nothing timed). The decode is ~42 integer instructions a lane and tile. At 8192 rows fat
decodes ~47 M tiles a layer and rank, ~2.0 G warp instructions: ~4 ms a layer if it were fully serialized with the
mma at 1 instruction a clock and SMSP (8 ms at half rate), against ~30 ms for the layer's kernels. That leaves ~15
ms of mma at 110 TF/s and ~8 ms of weight reads at 1024-2048 rows. once halves the decodes at 8192 rows. The best
case is -2 to -4 ms a layer (-7-13%); with the decode partly hidden behind the mma, **-1 to -3 ms a layer at 8192 rows,
~0 at 2048, possibly +0.1-0.3 ms at 1024** (16-warp barriers, no pairs to share). End to end at 28k with 8192-row
chunks (experts 5.9 s): **+0.5-2.5% prefill**. At production's 2048-row chunks: ~0-0.5%. PARADIGMS' +20-27%
(1,266 -> 1,520-1,600) is not reachable this way: its premise (a tile re-decoded per 16-32 rows) describes v1 / the
exact path, not fast2 / fat. Keep once only if `bench_experts.py` shows a win at the chunk sizes in use. Otherwise the
probe lines say where fat's time goes: if "no decode" is < 10% faster than fat, the kernel is mma / data-movement
bound and no decode sharing (not even a 2-SM cluster sharing 4 passes over DSMEM, the next step) can pay.

## 0270 — routed-expert kernels picked by chunk rows (`GLM53_TF_FAST_EXPERTS=auto`)

**Problem.** W2's `bench_experts.py`: at 1,024-2,048 rows (production's `PREFILL_ROWS_MAX=2048` batch pieces) fast2
is 1.13-1.24x faster than fat, at 8,192 rows fat is faster. fat (0170) was chosen on 8,192-row chunks. Decode never
runs either (fast chunks only), so the choice is a function of chunk rows alone.

**Change** (`exl3_mm.py`, `engine.py`, `knobs.py`). `GLM53_TF_FAST_EXPERTS=auto` is a fat-family mode: a fast chunk of
R rows runs fast2 when `lo <= R < hi` (`GLM53_TF_FAST2_ROWS`, default `0,4096`), else fat (or once, when `ONCE` is
configured). The fast2 branch reuses the fat branch's input rotation, i.e. `rot_in1` and one rotated input for gate and
up when their sign vectors are equal (the real checkpoint; half of `rot_in`'s 1.5 ms a layer at 2,048 rows). The
per-request knob `fat_experts` takes 0 (fast2), 1 (the fat family) or 2 (auto): `exl3_mm.family()` /
`set_family()`; the header carries the int as before.

**Exactness.** fast2 == fat == once bit for bit (0170 / 0260), and fast2 reading XG for XU is the same input when the
sign vectors are equal (fat already relies on it). So auto has fast2's bits for any threshold, keeps 0085's row
independence and shares snapshots with fast2 / fat.

**Measured (W5).** Kernel: fast2 1.19-1.33x faster than fat at 8-2,048 rows, a tie at 1 row. End to end on
production (lean 2,048-row chunks, pipelined): fat 1,278-1,284 / 1,259 tok/s at 24.5k / 98k, auto 1,251-1,253 / 1,232
(-2%), plain fast2 1,222-1,224 / 1,199 (-5%). The isolated kernel win does not survive the overlapped pipeline.
Production keeps `fat`.

Tests: `tests/cuda/test_fast_experts_auto_patches.py`.

## 0280 — batched rounds padded to buckets (`GLM53_TF_BATCH_BUCKETS`)

**Problem.** W1: 70-90% of 4-stream batched rounds ran eager. A graph key is (slots, rows per slot, modes, parities);
with 1-8 rows a slot few keys repeat and 0200's `CAPTURE_AFTER` keeps most rounds eager. 0200's per-slot
`GLM53_TF_BATCH_PAD` pads with rows that route to their own experts (~6 ms a row).

**Change** (`batch.py`, `batchplan.py`, `forward.py`).

- `GLM53_TF_BATCH_BUCKETS=4,8`: in a multi-slot round on the graph path, every window is padded with its last token to
  `batchplan.bucket_rows` = the smallest listed size >= the round's longest window (after `BATCH_PAD`, if set), where
  0200's padding rule (`Batcher._can_pad`: cache room, window rows, graph rows, same context mode) allows. The key
  becomes (slots, bucket x slots, modes, parities); since every slot flips its KDA parity each round, 4 decoding slots
  meet ~2 keys a bucket.
- `GLM53_TF_BATCH_PAD_TIE` (`auto`: on with buckets; `1` also for `BATCH_PAD`): `Buffers.route_src`, an int64 row table
  on the device (`batchplan.route_sources`: identity for real rows, the window's last real row for padded ones),
  written before every round and read inside the captured graph. `forward.moe_block` replaces each row's router
  logits by its source row's before `glue.select` (and skips 0130's fused route), so a padded row picks the same
  experts as a real row of its window: no extra expert reads.
- The batch-aware depth model prices real rows only when tied. Stats: `bucket_rows` (padded by buckets) next to
  `pad_rows`. Both knobs are in the ranks' settings check.

**Exactness.** Padded rows are never kept (0200's argument: verified like drafts that cannot be accepted, `commit`
gets the padded count). Real rows keep their own router logits (identity copy) and every kernel is row-independent,
so each real row has its lone bits: batched == alone == serial.

**Measured (W5, 4 streams x 512 tokens, 3 reps).** Graph rounds 8-33% -> 80-91%; batchexact 4/4, exact 10/10;
aggregate 74.2 -> 70.6 tok/s (-5%): ~2.7 padded rows a slot and round at ~0.8-1 ms each (KDA steps, attention, dense
rows). Control: `GLM53_TF_BATCH_GRAPHS=0` (all eager) 76.8 tok/s, i.e. at 4-slot verify sizes eager rounds cost about
what replays cost (the round is GPU-bound, ~100-120 ms) and captures are the only graph overhead left. Not used; a
graph-coverage approach cannot reach +10% here (docs/RESULTS.md, W5).

Tests: `tests/cuda/test_batch_buckets_patches.py`.

## 0340 — per-slot drafter choice in shared batched rounds (`GLM53_TF_BATCH_ADAPT`; `adapt.py`, `batch.py`)

**Question.** 4-stream batched decode runs 70-81 tok/s aggregate. A round is ~100-150 ms of GPU work, ~6-7 ms a
verify row, and per-slot tokens a round vary from 2.0 to 7.0. Would per-slot adaptive draft lengths, drafter choice
or skipping drafts for low-acceptance slots raise it? Full write-up: `docs/ADAPTIVE-DRAFT.md`.

**Finding.** Draft length is already per slot and adaptive: 0071's per-draft cost depths at 0120 / 0200's shared
rate and `ROW_MS`. `bench/draftsim.py` drives the engine's own `depth` / `batchplan` / `adapt` code on streams
fitted to the 11 recorded 4-stream runs (keeps and verify ms per round reproduced). Against today, it gives:
- drafter choice at the margin: -0.4% (steady -0.3%);
- serial rounds for weak slots: -6.5% (steady -4.3%). A slot costs ~7 ms + a row even serial, and the slow chat
  streams' first draft is kept ~80-85% of the time;
- `ROW_MS` unset: +0.5 to +1.5%;
- an oracle that knows how many drafts will be kept: +15-21%. That is the value of better drafts, not of scheduling.

The same model puts halving the per-slot fixed cost at +6% and batched DFlash2 blocks at +2%.

**Change** (off by default). `adapt.SlotChoice` per slot of a request with cost-derived depths:
- In shared rounds (`RoundCosts.rate()` not None), each drafting arm's surplus over a serial round is
  (keep − 1) − λ × (rows 2..R on the slot's `RoundCosts.table` + the drafter's model ms). It is kept over the arm's
  last `WINDOW` shared rounds and re-priced at the current λ.
- The best arm runs; serial runs only with `GLM53_TF_BATCH_ADAPT_SERIAL=1` and when no arm pays. The stalest arm is
  probed every `EVERY` rounds, and every arm runs once first.
- Alone, the request's `DrafterChoice` / fixed arm decides exactly as before, and it still records every drafted
  round.
- A serial round is the pending token's window alone. `BatchDepth` records it as a one-row window; lookup and the
  base choice skip it.
- Stats: `adapt: {serial, probes}`, and "s" in `drafters`.
- The knobs join the batch settings both ranks compare at start. Every input is committed keeps, the load-time
  costs and the round's windows (no clock).

**Exactness.** Tokens are keyed samples of each row's own logits at its own position (`sample_multi`). Rows do not
depend on the window's length or its mates. The accept rule and `commit(R, keep)` keep only matching rows. So any
per-slot window length, including none, gives the same bytes as serial decoding and as the request alone. Tested
on the real `Stepper` / `Batcher` with a hostile fake model, including random forced arms per slot and round (CPU),
and on the synthetic checkpoint (GPU, not yet run).

**GPU plan** (`docs/ADAPTIVE-DRAFT.md` §6):
1. Measure the per-slot fixed cost with `"draft": false` at 1 / 2 / 4 streams.
2. A/B production vs `ADAPT=1` vs `ROW_MS` unset: concurrent 1 / 4 streams × 5 reps, `batchexact`, `exact`.
3. Adopt only at ≥ +5% aggregate with single stream unchanged. Expected: not adopted.

## 0240 — b12x-derived fast-prefill kernels (`b12xpf.py`, `b12x_kda.py`, `b12x_mhc.py`, `b12x_attn.py`)

Written offline (no GPU): host tests and Triton CPU-interpreter checks ran (`tests/test_b12x_interpreter.py`, 19 pass
standalone), and the kernels were compiled for sm_121 offline with Triton 3.8 (image: 3.7.1): `b12x_kda` (BV 64, 4
warps) 58 KB shared, 255 registers + a 328 B stack; `b12x_mhc` fused 8 KB / 151 registers, unfused 8 KB / 144, no spills
(a first version that fed the dots from registers took 255 + 8.8 KB of stack, so the fused kernel re-reads its own
stores after a barrier); `b12x_attn` 80 KB (bf16) / 64 KB (FP8) shared, 163 / 251 registers, no spills (2 stages of the
FP8 variant: 96 KB, at the limit: 1 stage by default). The GPU tests and `tests/cuda/bench_b12x.py` are for the Sparks.
Nothing is timed.

**Source and license.** [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) at `d44247b6`
(2026-09-27), Apache License 2.0 (Copyright Luke Alonso and the b12x contributors; no NOTICE file). b12x's kernels are
CuTe DSL (`nvidia-cutlass-dsl` 4.6.2, its own compile / preparation runtime); nothing of it is copied or imported. What
is taken are the algorithms and kernel structures, re-implemented in Triton on this engine's arithmetic, with
attribution in the file headers, `NOTICE` and TensorFold's `THIRD_PARTY_NOTICES.md` (under 0230's "Code adapted from
b12x"). Also checked: MoonshotAI/FlashKDA (MIT) and vLLM's `mhc_fused_post_pre` (Apache-2.0, TileLang); neither is used
(see "Why not" below).

**Problem.** The 28k cold prompt (production knobs, `results/A1`, 23.7 s): `kda.chain` 1.87 s, `hc` 2.86 s (its
kernels alone ~1.3 s in `bench_glue`: 0.51 us a row and site), `dsa.sparse_attn` 2.29 s. `fast_kda` (0081) runs at
~1.8 ms per 1,024 rows x 32 heads because it writes and re-reads ~94 MB of fp32 scratch a call; the chunked sparse
latent attention writes [5, R, 32, 512] fp32 partials (~335 MB a 1,024-row call) and merges them; hc_pre re-reads the
streams hc_post just wrote.

**Change.** One per-request knob, `tf_knobs.b12x` (bits; default GLM53_TF_B12X, 0), in 0090's header after 0190's
knobs. Every bit is new arithmetic in fast chunks only (decode, verify, the MTP head, DFlash2 and exact prefill never
run them) and keeps 0085's rules (deterministic, row-independent, no atomics, shapes fixed by the weights):

| bit | kernel | what it is | numerics vs today's fast path |
| --- | --- | --- | --- |
| 1 | `b12x_mhc` | hc_pre's 24 mixing dots + square sums split by column blocks of ALL FOUR streams (16 blocks of 256 columns x 4 streams), square sums as the diagonal of X' X'^T on tensor cores; in 0084's slabs fused with hc_post (`glue.hc_post_pre`), one launch and one read of the new streams less a site | new streams X' bit-identical (hc_post unchanged); only the mixes' fp32 summation order moves (post / comb ~1e-7, normed rows within a bf16 step). The fused and unfused kernels give the same bits by construction (every number is an mma chain in column order), GPU-tested; b12x's own variant (Gram-matrix square sum, one rounding less in the collapse) is NOT taken |
| 2 | `b12x_kda` | KDA in 16-row tiles at absolute multiples of 16 (b12x's chunk, operands q~, k~, k_inv, k_r, lambda_C, Neumann-product (I + L)^-1, causal q~ k_inv^T), the tile preparation and the recurrence in ONE program per (head, 64 value rows), fp32 state on chip, no workspace; then `fast_kda`'s gated RMSNorm | tf32 products like `fast_kda` (b12x rounds operands and a shadow of the state to bf16: `GLM53_TF_B12X_KDA_PREC=bf16`, ~8x the error, not the default); the 16 x 16 solves and the two K = 16 products in fp32 FMA. Interpreter, fp32: 1e-7 of the serial chain; tf32-rounded: state 3-4e-4 relative vs fast_kda's 4-6e-4 |
| 4 | `b12x_attn` | sparse latent attention in one pass a row (all 32 local heads, one online softmax over the row's whole ~2,051-token list), u written directly: no 512-token chunk partials, no merge kernel (latent KV only) | the same tile steps (`latent._ltile`) in the same list order; only the merge's rescaling is gone (interpreter: equally close to a float64 softmax as the chunked kernel; ~2e-4 apart) |

Snapshots: a fast prefill's tag is G + (FP8 1 / TC 2) + 4 x the effective bits (`pfgrid.tag`, `b12xpf.tag_bits`; bit 4
counts only on a latent-KV engine), so a request resumes only snapshots made by the same kernels; `pfgrid.is_fp8` and
the `fp8_prefill` / new `b12x` stats read the offsets correctly. Load-time and checked equal on both ranks
(`b12xpf.load_settings`): `GLM53_TF_B12X_KDA_PREC` (tf32), `_BV` (64) and `_WARPS` (4), which decide the KDA
kernel's bits. Batch mode (0120) passes each sequence's `b12x` to its tag. Nothing changes with the knob at 0.

**Row independence / C-independence.** `b12x_mhc`: a row is one row of every mma tile; masked rows are zeros; the
square sum is a diagonal (one term a row). `b12x_attn`: one program per (row, head tile). `b12x_kda`: a row's bits
depend on its 16-row tile's rows <= itself and the fp32 state entering the tile (strictly lower L, lower Mqk and Minv;
padded rows are identity steps and add exact zeros); fast calls start at multiples of 64, so the only partial tile is
the prompt's last one, the same in every schedule, and a state handed between calls is the program's fp32 value
(interpreter: split at 64 == one call, first k rows == a k-row call, bitwise).

**Expected** (arithmetic from the profiles; not timed):

| item | 28k (22.2 s at 1,266 tok/s) | 112k (90.5 s at 1,238) |
| --- | ---: | ---: |
| bit 2, `kda.chain` 1.87 s / 7.2 s, 3-6x (inputs read once, ~0.3-0.6 ms a 1,024-row call instead of ~1.8) | -1.2 to -1.5 s | -4.8 to -6.0 s |
| bit 4, `dsa.sparse_attn` 2.3 s / 9.7 s, -25 to -40% (the partial traffic) | -0.6 to -0.9 s | -2.4 to -3.9 s |
| bit 1, hc kernels ~0.51 -> ~0.40-0.45 us a row and site (one launch and one L2 read less; the lap also carries NCCL contention, which 0230 attacks) | -0.15 to -0.3 s | -0.6 to -1.2 s |
| all bits | -2.0 to -2.7 s: **~1,440-1,560 tok/s** | -7.8 to -11 s: **~1,390-1,450 tok/s** |

The hc target of <= 1 s is not reachable by kernel work: a site must read the streams (32 KB) and the gathered partials
(16 KB) and write the new streams (32 KB) and the normed row (8 KB): 88 KB a row and site, 28,045 x 90 x 88 KB / 225
GB/s = 0.99 s at 28k, against ~1.3 s of hc kernels today. The rest of the 2.86 s lap is interference from the
concurrent NCCL all-gather kernels of 0084's pipeline (the lap was 2.0 s without the pipeline and its slabs); 0230's
host-proxied RoCE gather removes those kernels.

**Why not the others.**
- b12x's CuTe kernels as a dependency: `nvidia-cutlass-dsl` 4.6.2 + b12x's preparation runtime in the image, bf16 /
  FP8-quantized operands everywhere (its sparse MLA quantizes q to e4m3 and P to an e4m3 hi/lo pair; its mHC takes the
  collapse's square sum from a Gram matrix), capacity-planned autotuning (a tile choice by token count breaks 0085's
  row independence), and GLM_NEXT sparse MLA only for FP8 / NVFP4 caches with 4 scales a row (ours: one).
- `dsa_indexer`: FP8 index keys and a radix top-k whose ties follow shared-memory atomic arrival order: not
  deterministic, and not our stable-sort selection. At most 0.1-0.2 s at 28k (a radix threshold + our exact gather).
- FlashKDA (MIT): chunk 16, bf16 operands, and a bf16 state between tiles in the builds the Reederey87 kit tried (fixed
  2026-09-26 in vllm-project/FlashKDA 17a037d); the kit's "~150x off, uncorrelated" is most likely that build's
  missing TMA store wait (a3e42bb, fixed in 17a037d), not a convention mismatch (its parity script matches the API:
  raw gate, beta logits, [V, K] state, 128^-0.5). Its sequence-relative chunks would also break 0085's absolute
  alignment. `b12x_kda` keeps an fp32 state and tf32 products instead.
- vLLM `mhc_fused_post_pre`: TileLang; its fused path feeds the mixes the unrounded fp32 streams (other math than
  GLM's reference), and for > 32 tokens it is post + a GEMM, which is what we already run.

**On the Sparks.** `python tests/cuda/bench_b12x.py --sweep` first (a minute: old vs new at 1024 / 2048 / 8192 rows,
max diff and bitwise flags; a `[hc]` line saying "fused == unfused bitwise False" means bit 1 must stay off), then
`pytest -q -s tests/cuda/test_b12x_patches.py`, the regressions (`test_cindep`, `test_glue`, `test_knob`,
`test_fastk`, `test_fp8`), then per request on one load with `GLM53_TF_PROFILE=1`: `"tf_knobs": {"b12x": 1}`, `2`, `4`,
`7` against `0` at 28k and 112k; `exact` 10/10 and MMLU-200 with `GLM53_TF_B12X=7` before production.

## 0250 — the session store on NVMe (`GLM53_TF_SESSION_DISK`; `sessdisk.py`)

Written offline (no GPU); the host / CPU tests ran (the real store, tier, writer, reader, lone engine and batcher on
fake models), the GPU tests are for the Sparks. Idea from MiaAI-Lab PR #232 (NVMe prefix-cache offload); its code is
AGPL and was not read: this is a re-implementation from the description.

**Problem.** Production keeps 2 GiB of sessions a rank in device memory (4 x 256k slots leave no more). A session
evicted by newer ones comes back as a cold prefill (35-45 s at 40k), and a restart (watchdog heal, upgrade) loses
all of them.

**Change** (off unless `GLM53_TF_SESSION_DISK` is set; needs the RAM store, `GLM53_TF_SESSION_GIB` > 0; lone engine
and 0180 batch slots).

- *On disk*, one directory a rank (each rank stores its own shard): `<dir>/<compat hash>/rank<R>/pages/<hh>/<page
  key>.tfp` (one 256-token page: every cache tensor's rows of the page, back to back, 4 KiB-padded) and
  `entries/<entry key>.tfs` (a JSON header — token ids, page keys + SHA-256 each, segment table, a SHA-256 per 8 MiB
  chunk — then the rows past the full pages and the snapshot tensors: KDA state, conv, pending MTP rows, drafter
  window). Keys are 0110's (`page_keys`, `ids_key`: tokens, tag, KV format, MTP / drafter validity), so a shared
  system prompt is one set of page files and a new turn writes only its new pages + its snapshot. Files are written
  under a temporary name and renamed (complete or absent); O_DIRECT where the filesystem takes it.
- *Compat hash*: image id (else the engine source), checkpoint + drafter (revision, files, config.json), every
  `GLM53_TF_*` knob except ones that cannot change a stored bit (session / batch / calibration / paths / health /
  `PREFILL_ROWS`), KV format, cache layout, snapshot fields, torch. Another hash = another directory: stale state is
  never read (the 2 newest other directories of a rank are kept, older ones removed).
- *Writes*: `save` (default, write-through: every entry of >= `_MIN` tokens the RAM store saves, skips for size, or
  finds again; a crash loses only writes in flight) or `evict` (when the RAM store evicts it, from the slabs). Rows are
  packed page-major on the serving stream (one strided copy per run of pages and tensor), then one background thread
  copies them to a pinned buffer, hashes and writes them. At most `_QUEUE_GIB` (1) of packed rows wait; a larger
  entry is written in place, a group of pages at a time.
- *Lookup*: RAM first; a disk entry when it resumes >= `_GAIN` (256) more tokens than RAM / the live snapshots. Fork
  points against disk entries count for marks too.
- *Restore = read into the live slot*: pages the slot already holds (its live map) are skipped; `_THREADS` (8) threads
  read groups of consecutive pages / 8 MiB entry chunks into pinned buffers, check SHA-256 (`_VERIFY=full`), copy to
  the device and scatter per cache tensor with one strided copy. Rank 0 starts the read when it plans (before the
  header travels), rank 1 when the plan arrives (batch: when the chosen slot is idle); both finish it where the RAM
  restore would copy. Stats: `disk` (bytes, s, GB/s, pages read / skipped), `restored_disk` in batch mode.
- *Two ranks*: the disk index (entries, page references, LRU, budget) changes only on the save / eviction decisions
  rank 0 already ships (0110's forced saves), so both ranks hold the same index; its digest rides in every plan and
  rank 1 fails loudly on a difference. The restore choice is in the plan; each rank reads its own files; then both
  all-gather an ok flag: if either failed (missing / short / corrupt file, failed write), both drop the entry and
  prefill the prompt from scratch.
- *Restart*: each rank scans its directory (headers, compat, page files present; temporary / invalid / orphaned
  files removed), rank 0 shares its list oldest first, rank 1 answers which it holds, both index the intersection in
  rank 0's order (trimmed to the budget, oldest first). Settings that decide the index (`DISK` on, `_GIB`, `_WRITE`,
  `_MIN`) and the compat hash are checked equal at load.
- *Launcher*: `serve.sh` / compose mount `HEAD_SESSIONS` / `WORKER_SESSIONS` (default `<HF cache>/../glm53-tf/sessions`)
  at `/sessions`; set `GLM53_TF_SESSION_DISK=/sessions` to use it. `python -m tensorfold.families.glm5_next.cuda.sessdisk
  status|verify|bench DIR` (bench: one synthetic entry with the real per-rank shapes through the tier's own writer and
  reader, exactness checked).

**Exactness.** A restore puts back the bytes that were saved (checked by SHA-256 on every read), which are the bytes
the RAM store would have restored; so resumed == fresh, drafted == serial and batched == alone hold as for 0110 /
0180. A failed read never leaves partial state in use: the live map and the live snapshots are cleared, and the
prefill starts from 0.

**Sizes and latency (estimates; per rank, FP8 KV).** Pages 7,616 B a token (0220's 7,664 less the 0050 scratch) +
a ~94 MB snapshot: 40k tokens 0.41 GB, 100k 0.88 GB, 256k ~2.1 GB (measured by `bench`: 406,978,560 and 875,257,856
bytes). Each later turn writes its new pages + one snapshot (~0.1 GB); marks add a snapshot each. On the dev box's
consumer NVMe (CPU path, SHA-256 on) `bench` read 40k in 0.15-0.20 s and 100k in 0.31-0.38 s, wrote at ~0.8 GB/s in
the background. On the Spark (NVMe ~5-10 GB/s with O_DIRECT, 0140) expect ~0.1-0.2 s at 40k and ~0.2-0.35 s at 100k,
plus the new turn's prefill, instead of 35-45 s / ~90 s. Memory: 17 pinned 8 MiB buffers (~136 MiB) after the first
read, 8 MiB of device staging a read thread during a read, and up to `_QUEUE_GIB` of packed rows waiting to be written.
Disk: `_GIB` (64) a rank at most; both nodes need that much free on the mounted NVMe.

**Not done.** Admission does not wait for the read in a later round (the read overlaps the plan exchange, not other
slots' decode rounds); writes are one thread (~0.8 GB/s measured on the dev box, background); NVMe wear with
`save` is ~0.1 GB a turn a rank (use `evict` to write only what RAM drops).

## 0290 — shared latent KV pool (`GLM53_TF_KV_POOL_TOKENS`; `kvpool.py`)

Written offline (no GPU). The host and CPU tests ran: Triton's CPU interpreter for every touched kernel, the real
`Batcher` / `SessionStore` / `DiskTier` on 0180's hostile fake model, and sm_121 compiles of every kernel without a
GPU. The GPU tests and a window plan are in `docs/KV-POOL.md`.

**Problem.** Every batch slot reserves its whole capacity of latent KV at load (`State.kc`, `mtp_kc`, the MTP head's
index keys / gates, the pool keys): 1.86 GiB a slot at 262k (FP8), 7.44 GiB for 4 slots. It is paid whether a slot holds
1k tokens or 262k, and no request can pass 262k.

**Change** (off unless `GLM53_TF_KV_POOL_TOKENS` > 0; needs latent KV; both ranks must agree, checked at load).

- *Pool* (`kvpool.KvPool`, built in `decode.Engine` before any `State`):
  - One physical tensor per cache family, (N / page + 1) x page rows (zeros, committed at load). The extra page is the
    **null page**: unmapped table entries point at it.
  - Pages are 256 tokens (`GLM53_TF_KV_POOL_PAGE`, a power of two >= 256, so a session-store page never straddles two
    pool pages); pool keys are 64 rows a page.
  - A per-slot `SlotPages`: host list + device int32 table (16 KiB at 1M) + the admission's quota. The lowest free page
    goes first, so both ranks' pools stay identical.
- *State* (`forward.State._paged`):
  - `kc` / `mtp_kc` and the capacity-sized index tensors become `kvpool.Paged` views: logical shape, the slot's table,
    and in-page slices as real views (a slice across pages raises).
  - 0065's rings are decided first, so ring layers never allocate full keys / gates (today's 1M load would allocate
    5.5 GiB of them transiently).
  - Model-layer index rings, KDA state, conv and DFlash2 stay per slot.
- *Kernels*: a logical row is mapped as `table[r >> shift] << shift | r & (rows a page - 1)` (`_prow`, under the
  access's own mask; `PSH` constexpr, 0 = pruned) in:
  - `latent._lwrite`, `_lwrite8`;
  - `_lrows` (dense `_lchunks`, sparse `_lsparse_chunks`, 0240's `_lsparse_one`);
  - `sparse._index_write`, `_pool_keys`;
  - `_scores` and `_scores_rows`: one entry per 64-pool tile, a tile being one page.
- *Mapping points*: `forward.check_room` (every forward, verify, MTP step, lean chunk, batched round) calls
  `ensure(pos + R)`. Also `Graphs.__init__` (warm-up rows without `stage`), the store's `restore` / `prefetch`, and 0190's
  `zero_mtp_rows`.
- *Sessions (0110 / 0180 / 0250)*:
  - Store runs are gathered / scattered with one `index_copy_` / `index_select` per cache, or a copy per store page
    when pool pages are larger.
  - Tails (up to a fast grid past the last full page, so they may span pages) go through `Paged.gather` / `write`.
  - The NVMe tier packs and reads a pool page per copy and stages tails, writing them only after the read is checked.
  - `GLM53_TF_KV_POOL*` is left out of the tier's compat hash (where rows live, never their bits).
  - Snapshots never held attention rows: unchanged. `State.clone` refuses a pooled state (tests only).
- *Admission* (`Batcher._plan`, rank 0; spills travel in one extra message after the round plan):
  - A request reserves ceil((prompt + max_tokens + `SLACK` 64) / page) pages, at most the slot's capacity.
  - If they are not free and unreserved, idle slots are spilled, least recently admitted first and the slot it would
    resume in last (`batchplan.pool_spills`). A spill returns all of an idle slot's pages; its sessions are already in
    the store / on NVMe, so later turns resume from there.
  - If even that is short, the request waits (FIFO). If it needs more than the whole pool, it is refused.
  - Spills run first in a round (`_spill`), then admissions: set the quota, return pages past the resume point.
  - A request that ends keeps its committed rows' pages (its snapshots and live-page map) and gives back the rest
    (`_release`). An admitted request's pages are reserved, so it always finishes. Decode never waits.
  - `GLM53_TF_KV_POOL_CHECK=1` verifies the null page after every round.

**Memory** (per node, FP8): a 1,048,576-token pool is 7.439 GiB, against 7.438 GiB for today's 4 x 262,152 slots
(+36 MiB of 0050 scratch with `CONTEXT=1048576`). 1 GiB = 141k tokens. Full table in `docs/KV-POOL.md`.

**Expected cost.** Sparse gathers get one dependent 4-byte table load per row, from an L1/L2-resident table: estimated
0-3% of that kernel, <= 0.3% of prefill and decode. Dense attention, scoring and writers: ~0. A page is mapped every 256
tokens (a ~10-20 us H2D copy). Unmeasured.

**Exactness.** Same row values, same order, other addresses:

- **Interpreter:** paged == contiguous bit for bit for every kernel. Scrambled tables, NaN-poisoned unmapped pages,
  page-straddling windows; bf16 / FP8, BM 16 / 32. The control (mapping planted wrong) fails 11 / 11.
- **Compiled:** with the pool off, the 16 touched kernel variants compile for sm_121 to PTX identical to the tree
  without 0290 (`tests/kvpool_ptx.py`).
- **Fake model:** replies and slot states == fresh + serial through waits, spills, fragmented tables, 256 / 512 pages,
  RAM and NVMe restores, and a follower replay. A 64-run multi-seed stress was exact every time.

## 0300 — per-request log (`GLM53_TF_REQUEST_LOG`; `reqlog.py`, `app.py`; `scripts/traffic-report.py`)

Written offline (host tests; no GPU needed: nothing on the engine side changes but three stats).

**Problem.** Nothing recorded what the traffic looks like: prompt sizes, how much of each prompt was resumed and
from where, which prefixes sessions share. So nobody could tell whether shared-prefix reuse (0310), a larger store
or other session work would pay.

**Change** (off unless `GLM53_TF_REQUEST_LOG` names a file; production would use `/sessions/requests.jsonl`, the NVMe
mount).

- `GlmApp` (rank 0 only: rank 1 serves no HTTP) opens a ticket when the request's prompt ids are known (`prompt_ids`,
  O(1)). After the reply, or a failure, which still raises, it writes one JSON line. The fields are listed in
  `reqlog.py`'s docstring and in `docs/PREFIX-SHARE.md`.
- Everything runs on the request's HTTP thread after `generate` returns: an int32 copy of the ids, blake2b-64 hashes,
  and C-speed prefix compares against the last `_PROMPTS` (64) prompts and the ones that arrived before it and are
  still running. That is ~1-20 ms at 100k tokens, with no GPU sync and nothing on the engine loop.
- A failure to write is printed once and never fails a request. The file is rotated past `_MB` (64) MiB, keeping
  `_KEEP` (3) old files. `_SALT` keys the hashes.
- Engine stats added for it:
  - `restored` / `restored_disk` in the lone engine (the batcher already had them);
  - `marks` and `kv_free` (pool pages left after the reservation) in batch admissions.
- `GLM53_TF_REQUEST_LOG*` is left out of 0250's compat hash.
- `scripts/traffic-report.py <file>` reads the file and its rotated siblings (standard library only). It prints:
  - volume, sizes and speed;
  - resumes by source;
  - *avoidable prefill*: per request, max(0, floor64(min(lcp, prompt - 1)) - cached) tokens at its own prefill rate,
    for another conversation's prefix, the same conversation's, or either;
  - the new conversations that share a prefix, and the top system-prompt / 4k-head hashes.

  `--json` gives the same data as JSON.

**Exactness.** Not involved: no engine arithmetic or scheduling changes. Tests: `tests/test_request_log.py`.

**W8 (GPU, adopted):** request log on in production (`/sessions/requests.jsonl`); boot-to-boot prefill spread only (docs/RESULTS.md W8).

## 0310 — shared-prefix reuse across sessions (`GLM53_TF_PREFIX_SHARE`; `sessions.py`, `batch.py`, `engine.py`)

Written offline (host and CPU torch tests; the GPU tests are in the test file and the plan in
`docs/PREFIX-SHARE.md`).

**Problem** (measured on the fake model, `test_second_session_resumes_at_the_system_end[today-*]`). A new session
resumes only at a snapshot some earlier prefill took inside the prefix it shares. There are two kinds:

- `GLM53_TF_SESSION_EVERY` marks every 16,384 tokens;
- fork marks, which the session that discovers the fork takes.

So the 2nd session over a 12k agent system prompt prefills all of it, and a burst of N subagents prefills it N times.
Only later sessions hit. The full hit / miss table is in `docs/PREFIX-SHARE.md`.

**Change** (off unless `GLM53_TF_PREFIX_SHARE=1`).

- *System-prompt marks* (`sessions.PrefixShare`, `SessionStore.plan`, `SessionIndex.marks(extra=)`). Rank 0 adds a
  mark at the prompt's first `<|user|>` / `<|assistant|>` / `<|observation|>` token, with these rules:
  - role ids come from `tokenizer_config.json` / `tokenizer.json`, or `_TOKENS`;
  - it is rounded down to the page (exact) or the 64-grid (fast);
  - it must lie in [`_MIN`, `_MAX`], be `fork_min` past the resume point, and not be stored in RAM or on disk;
  - up to `_POINTS` role boundaries a prompt.

  The first session over a system prompt leaves the snapshot the second resumes at.
- *Waiting for a partner* (`Batcher._share_wait`, `_WAIT`). An admission is held back for a round while a request
  in flight has a planned mark it has not reached, inside their common prefix, and at least `fork_min` past what
  this request can resume now. That request must be prefilling or admitted this round, with the same prefill mode
  and drafters that cover this one's. The wait ends when the partner passes the mark, finishes or leaves. It is
  rank 0's scheduling only: rank 1's protocol is unchanged.
- *Bounds*. The entries of these marks are `kind == "prefix"` (rank 0 remembers the marks it planned). At most
  `_KEEP` (4) stay in RAM: rank 0 evicts the least recently used one first, and rank 1 replays the eviction. One
  snapshot (~94 MB a rank) per distinct system prompt; the pages are shared with the prompt entries.
- `GLM53_TF_PREFIX_SHARE*` is left out of 0250's compat hash.

**Exactness.** A system-prompt mark is a mark (0110 / 0180: an exact chunk cut, or a fast snapshot on the 64-grid);
the restore is 0110's copy-in (0290: into the slot's pages); the wait is scheduling. So resumed == fresh and batched
== alone as before.

**Tests** (`tests/cuda/test_prefix_share_patches.py`, 24 host / CPU tests ran offline):

- settings and the role-token lookup, boundary points and `extra` marks (exact / fast grid), the prefix-entry cap and
  its replay on rank 1;
- the real `Batcher` + `SessionStore` on 0180's hostile fake model:
  - today vs the switch (the 2nd session: 0 -> 512 / 576 cached);
  - bursts with and without the wait;
  - a cancelled partner never strands a waiter;
  - random agent traffic over 2 system prompts (exact / fast, roomy / tiny store, keep 4 / 1, and on 0290's pool
    roomy / tight): every reply and every slot's whole end state == fresh prefill + serial decode;
  - a replaying follower ends with the same store and states.

**Alternatives** (finer early marks, copy-free refcounted pool pages): `docs/PREFIX-SHARE.md`. Neither saves more
prefill for its memory or complexity.

**W8 (GPU, adopted):** 31 GPU tests pass (one unexplained illegal memory access in a first full run, not reproduced); a 2nd session over an ~18k / ~29.5k system prompt resumes at 17,920 / 29,376, a 4-session burst takes 28 / 36 s instead of 72 / 114 s, resumed == fresh. `GLM53_TF_PREFIX_SHARE=1` in production.

## 0320 — row-split prefill hyper-connections (`GLM53_TF_PREFILL_PP`; `pfpp.py`, `comm.py`, `pfoverlap.py`, `engine.py`)

Written offline (no GPU); host tests ran (two processes over gloo); the GPU tests and the two-Spark plan are in
`docs/PREFILL-PP.md`, which also has the analysis this patch comes from.

**Problem.** The task was pipeline-parallel prefill. Its premise, communication on the critical path, no longer
holds: 0084 already overlaps every exchange with the next sub-block (the measured `allgather` lap is 0.05% of a 28k
prefill). The bytes cannot shrink either: with two ranks, the all-gather of partials is all-reduce's minimum. True PP
needs each rank to hold whole layers: a second ~75 GiB weight copy per node (impossible on 121 GiB), or giving up TP
(-30% single-stream decode). Exact expert parallelism ships 8x the MoE bytes. What TP still wastes is replicated
row-wise work: both ranks run the hyper-connections (hc_post, hc_pre with 20 Sinkhorn steps, norms), the taps and the
final norm on every row. That is ~8% of a prefill as work (the hc lap is ~12% with NCCL contention), done twice.

**Change.** With `GLM53_TF_PREFILL_PP=1`, a pipelined lean chunk with at least 2 sub-blocks (a 2,048-row batch piece
has 4 of 512) runs `pfpp._SplitChunk`, 0084's piece order with the post work split:

- Rows: each sub-block's rows are split at half rounded up to 64 (`pfpp.split`); rank 0 owns the first part.
- `SplitPipe.issue`:
  - replaces the all-gather: the producer writes its whole partial into `slot[rank]` (a [2, rows, D] slot);
  - one NCCL group sends the other rank's rows and receives the other rank's partial of the own rows into
    `slot[other]`;
  - so `slot[:, own rows]` is hc_post's usual [rank 0, rank 1] operand.
- The post: hc_post and the next hc_pre (0190's fused form where on; the taps; after the last layer the final norm)
  run on the own rows only.
- `share` trades the own rows of the normed input + group sums (after the last layer: final rows, their sums, the
  DFlash2 taps) in place, on the comm stream.
- `need` makes the next reader wait for the share: the pre work of the same sub-block 3-4 pieces later, or the MoE
  routing after the drain.
- `comm.NCCL.swap` (ncclGroupStart / ncclSend / ncclRecv / ncclGroupEnd through the ctypes handle; the RoCE
  communicator forwards it) plus `comm.swap` (a padded byte all-gather for communicators without it).
- Smaller chunks, one rank, or the knob off: 0084's path, unchanged.

**Exactness.** The same kernels on the same operands, on row subsets: hc / stream_mean / rmsnorm are row-independent
(0084 slabs them on 64-row cuts, 0085 made fast-chunk kernels independent of call size). The exchanges are copies, and
hc_post still adds rank 0's partial first. Every row's streams, normed rows, sums, taps and final rows equal 0084's,
so the committed state, replies and snapshots are 0084's / 0082's / 0080's: they are shared across the switch. The
collectives differ, so both ranks must agree: `pfpp.settings()` = [knob, overlap default] is checked at load. The
other rank's rows of the chunk's residual streams go stale; nothing reads them after the chunk.

**Expected** (arithmetic, `docs/PREFILL-PP.md` section 5): half the ~130 ms of hc work a 2,048-row piece, minus the
share exposed at the 42 MoE boundaries: -45 to -60 ms of 1,632, **+3 to +5% prefill** (1,255 -> ~1,295-1,320 tok/s
at 24.5k-98k). Decode and prompts under 513 tokens: unchanged. Memory: +16 MiB of slots + 0.5 MiB + NCCL p2p
buffers.

**Tests.** `tests/cuda/test_prefill_pp_patches.py`.
- Host, two real processes over gloo, each running 0082's hash model as its rank with rank-specific partials:
  - split == 0084 bit for bit on both ranks: 4 variants, 6 lengths, 2 positions, EXL3 / MLX;
  - the swap's rank layout;
  - the all-gather fallback;
  - controls caught: routing without the wait, the wrong half sent, no split for one sub-block;
  - the piece order, knob, settings, split.
- GPU, one GPU: each row-wise kernel on half a sub-block == the whole call's rows (real shapes, fused hc 0 / 3).
- GPU, `PP_TWO_PROC=1`: two engines (rank 0 / 1) on one GPU through a host-staged gloo communicator, split == not.
- Two Sparks: `docs/PREFILL-PP.md` section 7 (ab.py hash, exact 10/10, batchexact 4/4, 24.5k / 98k A/B with
  `GLM53_TF_PROFILE=1`, nsys).

**W8 (GPU, adopted):** the half-sub-block kernel identity test is bitwise on real shapes; the two-engine test needed a test fix (no decode graphs with the host-staged communicator). End to end +7.3-8.1% prefill (twice the estimate), same sha / exact / batchexact, decode unchanged. `GLM53_TF_PREFILL_PP=1` in production.

## 0330 — warp-specialized fast-prefill experts (`GLM53_TF_FAST_EXPERTS=tc`; `exl3_tc.cu`)

Written offline (no GPU). The kernels compile for sm_121 (nvcc 13.4) and the host tests pass; nothing is timed.
Design, analysis and the GPU plan are in `docs/EXPERT-TC.md`.

**Problem.** Per MoE layer and rank, the fast-prefill experts have a DRAM floor (weights once, Xg, Xd, Y at 220 GB/s)
and an MMA floor (~110 TFLOP/s).

- **At 2,048 rows:** floor 10.4 ms; fast2 takes 10.9 ms and fat 13.0 ms. The kernel is DRAM-bound, which is why the
  no-decode / no-mma probes changed nothing.
- **At 8,192 rows:** the DRAM and MMA floors are 16.8 and 15.0 ms; fat takes 28.6 ms (58 TFLOP/s) and fast2 31.1. The
  no-mma probe saved 21-27%: memory traffic and MMA do not overlap.
- **W5:** fast2's isolated 2 ms a layer did not survive end to end. The leading explanation from the code: fast2
  assigns items to CTAs statically, fat by ticket, and with patches/0084's overlap a second stream shares SMs, so a
  static kernel waits for its slowest CTA. `bench_experts.py --contend` tests this.

**Change** (`exl3_tc.cu`, `exl3_tc.cpp`: extension `tensorfold_glm_exl3_tc_v1`; `exl3_mm.py`: mode and dispatch).

- `GLM53_TF_FAST_EXPERTS=tc` is a fat-family mode (`FAT` and `TC`). `tf_knobs.fat_experts`: 1 = tc, 0 = fast2, 2 = auto.
- **CTA structure:** 1 producer warp + 16 consumer warps (544 threads, `__maxnreg__(120)`), one CTA an SM, persistent.
  - The producer claims items by ticket (`GLM53_TF_TC_TICKET=0`: static stride, for the A/B).
  - It writes each item's rows, expert and count into a 2-deep mbarrier queue.
  - For each K stage it waits on the slot's *empty* barrier, gathers the member rows with cp.async (fat's swizzle,
    zero-filled past the count), and copies the words with 1 KB `cp.async.bulk` copies (`expect_tx`). Each lane's
    `cp.async.mbarrier.arrive.noinc` completes the *full* barrier (count 33).
- **Consumers** wait on *full*, run fat's inner loop verbatim (decode into the A fragment, ldmatrix the rows as B,
  m16n8k16 over ascending k tiles), then arrive on *empty*.
  - Their epilogue is fat's, in its own shared buffer, so the next item's loads overlap it. It uses a consumer-only
    named barrier.
- **Configurations** (`GLM53_TF_TC_CFG=gu,dn`, 0 = by rows; the same bits):
  - 1: 64 members x 2 (gate/up) or 4 (down) column blocks; the default below 4,096 rows.
  - 2: 128 members x 1 or 2 blocks; the default at 4,096+.
  - 3: fat's tile with 8 consumer warps, 2 CTAs an SM.
  - A column-block group may be partial (shapes with an odd block count): absent blocks get no words and idle warps.
- `GLM53_TF_TC_CTAS=N` caps the grid, leaving SMs to the overlap stream.
- Timing probes: `probe` 1 = no decode, 2 = no mma.
- sm_121 has no wgmma, tcgen05 or setmaxnreg. cp.async, mbarrier and cp.async.bulk are available.
  `docs/EXPERT-TC.md` has the ptxas probe table.

**Exactness.** tc == fat == fast2 == v1 bit for bit, by 0170 / 0260's argument. Each element gets the same m16n8k16
chain with the same decoded A fragment and the same fp16 B values, ascending k tiles, all of K in one warp, then the
same `fwht_row` and epilogue formulas. The helpers are verbatim copies; the test compares the text. Only the data
movement changed. It is row-independent and deterministic, and shares snapshots with fat / fast2 / once / auto.

**Offline:**

- nvcc 13.4 for sm_121: gate/up 120 registers, 0 spills; down 120 registers, 8-12 B spilled; cfg 3 111-112 registers.
  Shared memory: 88.9 KB + 0.6-1.2 KB static (cfg 1/2), 40.4 KB (cfg 3).
- Host tests: 111 passed (the model of the index arithmetic, the mbarrier protocol under random interleavings,
  mutations caught).
- Expected, arithmetic only:
  - 2,048 rows: 13.0 -> 10.5-11.5 ms a layer, **+4-7% prefill**, if it holds with the overlap stream beside it.
  - 8,192 rows: 28.6 -> 20-24 ms.
  - Risk: 0260's `once` (also 16-warp CTAs over 128 members) did not beat fat.

**W8 (GPU, not adopted):** (1) the extension did not build in the image: its `TORCH_CUDA_ARCH_LIST` (8.0 ... 12.0+PTX) compiled compute_80, where ptxas refuses mbarrier / cp.async.bulk; `_tc_ext` now builds for the device's arch only (this patch). (2) cfg 1 / 2 cannot launch on GB10 (occupancy 0: 17 warps x 120 registers need 19,200 registers on one SM sub-partition of 16,384). (3) cfg 3 is bit-identical, 1.12x fat contended at 2,048 rows but 0.74-0.85x at 4,096-8,192, and **-4% end to end** at 2,048-row chunks. Stays off.

## 0335 — solo prefill pieces (`GLM53_TF_SOLO_PIECE`; lazy Xu in the lean set)

Written offline (no GPU); the host tests pass. Details, memory and the GPU plan are in `docs/EXPERT-TC.md`, section 4.

**Problem.** In batch mode every fast prompt prefills in `GLM53_TF_BATCH_PIECE` = 2,048-token pieces, and
`PREFILL_ROWS_MAX` = 2,048. So even a request alone on the server runs 2,048-row chunks. The routed experts cost 6.35 us
a token and layer there (fat), against 3.91 at 4,096 rows and 3.49 at 8,192.

**Change** (`batch.py`, `batchplan.py`, `lean.py`).

- `GLM53_TF_SOLO_PIECE=N` (0 = off; 1-63 refused). In `_piece`, `batchplan.solo_piece` cuts a **fast** piece at N
  tokens when no other slot holds a request, and at the batch piece otherwise.
  - The rule is re-evaluated at every piece boundary, so a request admitted in a round's plan makes the next piece
    normal.
  - `self.seqs` is identical on both ranks (cancels, admissions and finishes come from the shared plans), and the
    value is checked equal at load in its own `_gather_ints`.
- `job.stats["solo_pieces"]` counts the solo pieces; a boot line prints the setting and the chunk cap.
- Chunks are as big as the piece only when `GLM53_TF_PREFILL_ROWS_MAX` >= N (the lean set's rows). Otherwise a solo
  piece runs several chunks in one `decode.prefill`, which saves only per-piece overhead.
- `GLM53_TF_LEAN_LAZY_XU=1`: the lean set's Xu (72 of its 397 KiB a row) is allocated at first use. The fat family never
  reads it when gate and up share their sign vector, as in the real checkpoint.

**Exactness** (checked in the code):

- Pieces are `decode.prefill(prompt[:end], resume=previous snapshot)`, and fast pieces end on the 64-token snapshot
  grid (0180's `piece_grid`).
- 0085 makes every fast kernel row-independent, the KDA scan run on absolute 64-row blocks, and lean sub-blocks start
  at `pos + k x block`. The expert kernels' row-count switches are bit-neutral.
- So the state at every cut is a fresh prefill's, and replies are the same bytes with the knob on or off.

**Memory a rank:** lean set 0.78 / 1.55 / 3.10 GiB at 2,048 / 4,096 / 8,192 rows (lazy Xu: 0.63 / 1.27 / 2.54); chunk
transients add +18 / +54 MiB.

- Worst-case MemAvailable (W6 stress minimum 14.05 GiB on the worker node) becomes ~13.2 (4,096) or ~11.6 GiB (8,192): >= 8
  either way.
- The 4th slot's load-time check loses the same amount of margin.

**Expected:**

- +4-6% single-stream prefill at 4,096, +5-8% at 8,192 (kernel savings 105 / 123 us a token, discounted to F9's ~1/3
  in-engine transfer), plus ~1% from fewer piece resumes.
- A newcomer waits up to one solo piece (~3.2 / ~6.5 s instead of ~1.6 s).
- Recommendation: start with 4,096, then compare with W7's static 8,192-piece sweep.

**W8 (GPU, adopted at 8,192):** `SOLO_PIECE=8192` + `PREFILL_ROWS_MAX=8192`: +6.4% / +5.8% at 24.5k / 98k (4,096: +3.8 / +3.0), same sha, 4-stream decode unchanged (pieces stay 2,048 beside decoders); worst-case 4 x 250k stress MemAvailable 9.23 / 8.14 GiB. `LEAN_LAZY_XU` not used (its engine test fails: Xu still allocated). Test fix: the newcomer test needed a 4,096-token context.

## 0350 — RoCE loopback failure explained, a diagnosis that can tell, a stress test (`roce_watch.h`)

Written offline (no GPU, no RDMA); the details, the audit table and the GPU plan are in **docs/ROCE-FIX.md**.

**Finding.** W2's single-node loopback timed out at sequence 312 on `roceP2p1s0f1`, and 0230's diagnosis said "the
flag HAS reached this host's memory (the GPU did not observe it)". The cause is the harness. `bench_roce.py loopback`
drives both ranks from one process. After `1 + (10 + iters)` = 311 paired collectives, `_graph` warmed rank 0's CUDA
graph with rank 0's collectives **alone** and synchronized. Rank 0's sequence 312 waited for a rank 1 the host had not
launched yet, and timed out. Rank 1's warm-up then sent 312 about 10 s later, and `check` found it in host memory.
`tests/test_roce_protocol_model.py` (an executable model of 0230's protocol and of the harness) reproduces every number
in the log line and predicts `iters + 12` for any `--iters` and either HCA count. The memory-model code is correct:
- `ld.acquire.sys` polls, inside the loop;
- system-scope payload loads;
- mapped, portable, not write-combined pinned memory;
- the flag on the payload's RC queue pair, no relaxed-ordering MR;
- fences before the doorbell;
- two-slot reuse, checked in the model under random schedules.

**Change** (the protocol and wire format are unchanged; the extension becomes `tensorfold_glm_roce_v2`).
- **Kernel.** The first timed-out wait of a runtime claims the failure record with a CAS on the device poison word, so
  the record is never torn across blocks x HCAs waiters. It stores its last read of the flag (`CTRL_ERR_SEEN`), then
  the failure word after a system fence. The epoch is published only when the device poison is clear.
- **`roce_watch.h`** (no CUDA / torch / verbs). `FailWatch::poll` runs on the proxy thread's loop (and on the tests'
  `Loop`): one acquire load of the failure word. When the word turns 1, it records the flag word as host memory holds
  it then, and later how long after the failure the flag arrived, if it does.
- **`roce.classify`.** It returns `not_seen` (in host memory at the failure while the GPU read a stale value:
  sparkring #278), `late` (arrived N ms after), `never`, or `unknown` (no watch record). `diagnose` and `snapshot`
  (`gpu_seen`, `watch`) use it.
- **`tests/cuda/bench_roce.py`**:
  - `loopback` warms both ranks' graphs together (`_graph_pair`);
  - new `stress` mode: `--iters` (100k) ops a size, eager and as graph replays, with every rank's payload moving each
    op (a stale slot or a flag overtaking its payload cannot compare equal; 0230's `soak` replays constant bytes);
    `--loop` for one node, two nodes otherwise.

**Tests.** Host:
- `test_roce_protocol_model.py` (90);
- `test_roce_watch.py` (compiles `roce_watch.h`, 7);
- `test_bench_roce_stress.py` (9);
- `test_roce_logic.py` unchanged (15).

GPU (`test_roce_patches.py`): a stalled peer is `never` with the watch record; a peer that delivers after the failure
is `late`, not "not seen". Offline compile for sm_121: 64 registers, no spills.

**GPU plan** (docs/ROCE-FIX.md):
1. preflight, including `PCI_WR_ORDERING`;
2. `loopback` on both functions and on each alone;
3. `stress --loop` at 100k ops, both functions and each alone;
4. 0230's two-node stages: `bench`, `fault`, `stress`, `soak`, then the engine A/B (`exact`, identical transcripts,
   decode at 1 and 4 streams).

Adopt only if every stage is clean and decode improves by >= 3%.

**W9 (2026-09-28): GPU plan run, adopted** (`GLM53_TF_COMM_BACKEND=roce`). W7's 1 MiB mismatch was the loopback
harness (inputs made on the default stream, gathered on side streams without a wait); `bench_roce.py` also had three
more harness bugs (`stress --loop`, `fault`, `soak`), fixed. Every stage clean after that; engine decode +10.8% (1
stream) / +4.3% (4 streams), transcripts byte-identical. docs/ROCE-FIX.md "W9", docs/RESULTS.md "W9".

## 0360 — b12x bit 4 on FP8 latent KV, with sessions (`b12x_attn.py`, `engine.py`, `sessdisk.py`)

Written offline (no GPU). Scope: `tf_knobs.b12x` **bit 4** (one-pass sparse latent attention). Bits 1 and 2 stay as
0240 left them (slower, not adopted). With bit 4 off nothing changes. The chunked kernels and the decode / verify /
MTP / exact paths are untouched.

**What W3 actually failed** (`results/W3/b12x-fail.log`, read again). There were three failures:
- **Session reuse, the real bug.** `GlmEngine.generate` looked snapshots up with
  `self._grid(fast, rows, fp8)`, without `b12x`, so the lookup used the rank's default bits (0 between requests).
  `decode._prefill` tagged the snapshots with the request's bits (G + 4 x bits). A request with b12x bits never
  resumed (`cached` 0), even after a request with the same bits. The batcher already passed the bits.
- **"FP8 one-pass not row-independent".** This is not what failed. The row-subset and permutation asserts passed for
  FP8; the failing assert was "FP8 rows == the kernel on `dequantize_rows`". Compiling for sm_121 offline (Triton
  3.7.1 and 3.8) shows the cause. Triton gives a dot operand upcast from 8-bit data another register layout
  (`kWidth` 4 instead of 2), which places K elements differently inside each m16n8k16 MMA.
- **"bits-3 chunk-size dependence".** Also not what failed. The C- and overlap-independence asserts passed; the
  failing one was the bit-1 control ("state differs from bits 0") on a 3-token prompt. That is bit 1, outside 0360.

**Change.**
- `engine.GlmEngine._request_grid(values)`: the lookup tag from the request's knobs, including `b12x`. `generate` uses
  it for `_resume` and for the session store's plan.
- `b12x_attn`: the dequantized FP8 tile passes `_opaque`, an impure inline-asm identity (one `mov.b32` per two
  values; constexpr `OPQ`, compiled kernels only, since the interpreter has no inline asm). Triton's upcast analysis
  does not look through it, so both dots get exactly the bf16 kernel's operand and accumulator encodings. The loop
  runs the same compute ops in the same order; only data movement differs (how the transposed tile reaches the
  first dot). So the FP8 kernel is the bf16 kernel on the dequantized rows, bit for bit, by construction. Registers
  (ptxas 13.4, sm_121a):
  - bf16 160;
  - FP8 in 0240: 254;
  - FP8 in 0360: 202.

  None spill. Shared memory: 67.5 KB (0240's FP8 kernel: 64 KB).
- `sessdisk.KNOB_SKIP += GLM53_TF_B12X`. The default bits are in every entry's tag, so changing the default no longer
  hides the NVMe store. `GLM53_TF_B12X_KDA_*` decide bits and stay in the hash.

**Row independence.** The grid is (R, ceil(H / BMQ)), with BMQ a function of H only. A program reads only its row's
query, count and token list, and walks the list in order in KT = 32-token tiles. That is one fixed chain of MMA and
online-softmax steps, whatever R, the other rows, the chunk, or the lean / pipelined schedule. Nothing is split by R.

**Must it equal the chunked kernel's bits? No, and where it may run.** One-pass is not bit-identical to
`latent.sparse_latent`, because the chunk merge's rescaling is gone. A kernel with other bits may only serve rows that
no other kernel recomputes in the same configuration. Fast chunks are such rows (0080 / 0085):
- a fast-prefilled row is reused only through a snapshot of its own tag (G + 4 x bits) at a multiple of 64;
- a resumed request re-prefills every later row with fast chunks, so with this kernel;
- reply rows (decoded by the chunked kernel) are never kept in fast snapshots;
- exact requests never resume fast snapshots.

With bit 4 on, therefore:
- resumed == fresh: both prefill every row past the snapshot with fast chunks;
- drafted == serial: the same deterministic prefill, then unchanged decode / verify;
- batched == alone: batch pieces run `decode.prefill` per slot, and decode rounds never set `b.fast`.

It is not used for decode / verify, which would also be slow (1-2 programs for a 1-row step).

**Offline results.**
- `tests/test_b12x_onepass_interpreter.py`, 10 passed. bf16 and FP8, 16- and 32-head tiles, counts 0 / 1 / 31 / 32 /
  33 / 64 / 65 / 137 / 300:
  - every row == the row alone, in subsets, orders and duplicates, bitwise;
  - FP8 == dequantized;
  - paged FP8 == contiguous;
  - as close to float64 as the chunked kernel.
- `tests/test_b12x_onepass_compile.py`, 6 passed under Triton 3.8 and under 3.7.1 (the image's):
  - the FP8 dot signatures equal bf16's (contiguous, paged, BM 16 / 32);
  - the loop's compute ops are equal;
  - control: 0240's FP8 kernel has `kWidth` 4;
  - the wrapper passes `OPQ` for FP8 only, with grid (R, 1) for every R.
- `tests/cuda/test_b12x_attn_patches.py` host part, 11 passed:
  - `_request_grid` equals the prefill's tag for bits 0-7;
  - the batcher's tag equals the engine's;
  - on the hostile fake model with the real `_run` / `_resume` / `_prefill`, 6-turn conversations with bits 4 / 7 / 0
    resume every follow-up (cached > 0, on the grid) and equal a fresh prefill (reply and whole state);
  - the 0240 lookup never resumes (control);
  - 4 / 0 / 7 never cross;
  - the compat hash leaves out only `GLM53_TF_B12X`.
- PTX unchanged where it must be:
  - `tests/kvpool_ptx.py` (0290's tool, which now passes `OPQ` = False) compiles 33 of 33 kernels to the PTX of the
    tree before 0360, including every chunked kernel;
  - the bf16 one-pass kernel is byte-identical before and after, contiguous and paged, under Triton 3.8 and 3.7.1.
- Regression: the host parts of `test_session_disk`, `test_b12x`, `test_cindep`, `test_fastpf`, `test_batch_sessions`,
  `test_kv_pool`, `test_knob`, `test_session` and `test_roce_logic` on the 0350 + 0360 tree: 143 passed, 0 failed.

**GPU plan** (one window, production stopped):
1. `pytest -q -s tests/cuda/test_b12x_attn_patches.py`:
   - real-shape kernel checks: FP8 == dequantized at 2,051 tokens (the W3 failure), rows / subsets / permutations,
     timing print;
   - engine on the latent cache past the dense limit: C / pipeline independence, drafted == serial, resumed == fresh
     with cached > 0 through `generate`, bits never cross.
2. The 0240 subset for bit 4: `test_b12x_patches.py -k "attn or latent_sparse or snapshots or knob"`. The old
   drafted/resumed tests use bits 3 and should now pass their `cached > 0`. Then `bench_b12x.py`'s `[attn]` lines:
   FP8 speed with the new layout vs 0240's 6.29 ms.
3. On the production config (FP8 KV, batch 4, pool), per request `"tf_knobs": {"b12x": 4}` vs `0`:
   - `exact` 10/10 with `GLM53_TF_B12X=4`;
   - session follow-ups: 3 conversations x 3 turns at ~40k, where turn 2+ must report `cached` > 0 and the same reply
     hash as a cold request with the same knobs (`b12x: 4`, `draft: false`);
   - batch: 4 concurrent sessions == alone.
4. Prefill A/B at 24.5k and 98k: 3 runs each, b4 vs b0, with `GLM53_TF_PROFILE=1`.

Adopt (`GLM53_TF_B12X=4` in `config/prod.env`) only if all of these hold: every check above passes, prefill improves by
>= 3% at both sizes (W3 measured +1-4%, and the new FP8 layout may move it), decode is unchanged, and MMLU-200 is
within noise.

**Risks.**
- The opaque identity pins Triton's layout choice through an inline-asm side effect. A future Triton could treat it
  differently; `test_b12x_onepass_compile.py` fails if the layouts diverge again.
- 0360 does not measure speed: the FP8 kernel now stores bf16 tiles to shared memory (2 B a value, like the bf16
  kernel) instead of e4m3 bytes. W3 had bf16 one-pass at 5.85 ms vs FP8 at 6.29 ms (1,024 rows).
- The bf16 toy checkpoint has a 128-wide latent, so FP8 engine behaviour is only covered by the kernel equality and
  the real-model steps.

## 0370 — decode overlap and CPU pinning (`GLM53_TF_DECODE_OVERLAP`, `GLM53_TF_CPU_PIN`; `decode_overlap.py`, `cpupin.py`)

RESEARCH-NIGHT N1a + N1b, written offline (no GPU). Both knobs are off by default. Scheduling only: the same kernels on
the same inputs, and the same collectives in the same order on both ranks. Details, measurements and the GPU plan:
`docs/DECODE-OVERLAP.md`.

**What is on the critical path today** (tree through 0360):

- The plan share at the top of every batched round (`batch.py:1338`, `1460`, `1463`): `engine._share` is two NCCL
  all-gathers, each followed by `.item()` / `.tolist()` (`engine.py:759-771`). With the KV pool that is four control
  all-gathers and four host syncs a round with the GPU idle.
- A timing-only `torch.cuda.synchronize()` after the verify forward (`batch.py:1888`; `decode.py:847`, `625`, `693`,
  `565`).
- The HTTP threads' detokenizing / SSE work (`server.py:397-402`), which holds the GIL while the round loop drafts
  through its hard syncs. Without batching it runs inside the decode loop (`decode.py:888`).

**How big.** W7's per-round records (`results/W7/analysis/mix-r*.json`):

- the gap between verify rounds is **0.83-0.92 ms** median (1 and 4 streams), not the ~3.9 ms quoted in PROFILE §5;
- the median in-round idle is 1.8-3.8 ms, mostly the drafting and sampling readbacks, which reordering cannot move;
- the mean idle is dominated by eager / capture rounds (8 of 106 single-stream rounds hold 55% of it).

**Parts of `GLM53_TF_DECODE_OVERLAP`** (`1` = all):

- `sync`: `Batcher._verify` and the lone engine's decode loops skip the timing-only sync. 0070's online timer reads
  two CUDA events (`ForwardTimer`) after the sampler's readback.
- `emit`: `_emit` holds rank 0's tokens and `_flush` hands them out right after the next round's verify forward is
  launched. They also go out at a request's end (before its end marker), at the top of `_plan`, and on a loop failure.
  A streamed token arrives up to one round later. Without batching, `Emitter` runs the HTTP callback on its own thread
  and `generate` joins it.
- `plan`: in a verify round where every slot in flight decodes and nothing waits, rank 0 decides the next round's
  plan (`_plan_ahead`: cancels only) while the forward runs. The plan's words ride at the end of `sample_multi`'s
  all-gather (`rider`, n + 8 int32, zeros from rank 1). Both ranks read rank 0's words, and the next round starts
  from them (`_take_ahead`) with no `_plan` / `_share`. Every other round plans at the top as before. `plan` is
  compared at load. An arrival during decode may wait one more round for admission.
- `gil`: `sys.setswitchinterval(GLM53_TF_SWITCH_US / 1e6)`, 500 us.

**`GLM53_TF_CPU_PIN`** (GB10: X925 = cpus 5-9, 15-19; the 16 MB-L3 cluster is 10-19; read on both Sparks):

- `early()`, before NCCL creates its threads: sets `NCCL_SET_THREAD_NAME=1`, sets `GLM53_TF_ROCE_CPU` when RoCE is on
  and it is unset, and puts the main thread on `rest`.
- The round loop pins itself (`serving()`, named `tf-serve`) and re-sorts all threads at start and every 30 s:
  `NCCL*` to `comm`, the RoCE proxy left on its cpu, the rest to `rest`.
- Without batching, the request thread pins itself while it decodes.
- `fast` puts every thread on the X925s. `scripts/serve.sh` `CPUSET` is the docker `--cpuset-cpus` form of it.
- HTTP threads are named `tf-http`. `nice` is optional and 0 by default (GIL priority inversion).

**Offline results:**

- `tests/cuda/test_decode_overlap_patches.py`: 29 passed. It covers:
  - every part, alone and together, on 0180's hostile fake model (every reply and slot state == fresh prefill +
    serial);
  - rank 0's shares and sampler exchanges replayed in one ordered stream through `follow` (same kinds in the same
    order, zero riders from rank 1, same logs / traces / stores / states, cancels riding);
  - the `emit` ordering;
  - `_plan_ahead`'s refusals;
  - `auto_decode` with `sync` (same tokens, keeps and timer rows, one sync less a round);
  - cpupin's plans on a fake GB10 sysfs, and real affinity calls on this host.
- `tests/test_serve_ops.py` 36/36 (with `test_cpuset`).
- Existing `tests/cuda/test_*_patches.py` with the knob unset and with `GLM53_TF_DECODE_OVERLAP=1`: 494 passed, the
  same 9 environment failures as without 0370.

**Estimate:** `plan` -0.5 to -0.7 ms a round, `sync` -0.05 to -0.1, `emit` 0 to -0.8 (more at 4 streams):

- 1 stream: **+1.1-2.0%** decode;
- 4 streams: **+0.6-1.5%**;
- pinning: +0 to +2%, mostly as less rank skew; the scheduler already kept the busy threads on X925s in 6 s of
  samples.

**GPU plan** (`docs/DECODE-OVERLAP.md` §6):

- loads A (prod), B (`OVERLAP=1`), C (`CPU_PIN=auto`), D (B + C);
- exact 10/10, batchexact 4/4, and the W9 transcripts byte-identical;
- `multiturn.py --streams 1,4 --reps 5`, cancel and arrival checks, a `ps -L` pinning check;
- one nsys capture (the gap between verify ranges should fall from ~0.85 to ≤ 0.3 ms).

Adopt the overlap at ≥ +1% single-stream with 4 streams not worse; pinning on its own merit.

**W10 (2026-09-29): GPU plan run, adopted** (`GLM53_TF_DECODE_OVERLAP=1`; `GLM53_TF_CPU_PIN` not adopted). Unit tests
29 passed; the batch tests with the knob 26 passed, and 26 of 28 in `test_batch_sessions_patches.py` (the follower
replay test records `_share` only, so the plan rider puts it out of step: a harness limit). On the production config +
0390: 1 stream +1.3% / +1.9% (medians; per rep -0.5 to +2.5%), 4 streams +1.0% / +1.3% in two load pairs, W9
transcripts byte-identical, exact 10/10, batchexact 4/4, a disconnected streamed client beside 3 decoders ends
`cancelled` with the 3 replies == alone. `CPU_PIN=auto` placed every role as planned but gave +0.4% / +0.5%: off.
No nsys capture (no window time). docs/RESULTS.md "W10".

## 0380 — deeper verify windows (`GLM53_TF_MAX_DRAFT_ROWS`, `GLM53_TF_DFLASH_BLOCK`; `deep.py`)

**Question.** RESEARCH-NIGHT §1: 34% of code-like and 82% of repetitive DFlash2 rounds keep all 8 rows. Would
windows of 12-16 rows pay, and would cross-request suffix drafts (N3)? Full write-up, estimates and GPU plan:
`docs/DEEP-VERIFY.md`.

**Change** (default 8 = before).
- `GLM53_TF_MAX_DRAFT_ROWS=N` (8..16) raises every window cap to N rows. The depth rules are unchanged: a row past
  the 8th is verified only when its draft's expected token beats the request's rate times the row's marginal ms.
- Sizing to match:
  - `sparse.LongScratch` and the long-context graph rows 1..N (lazy);
  - `latent.SPARSE_ROWS`;
  - the batch slots' KDA projection / replay rows;
  - batched KDA rows, batch graph and pad rows.
- The calibration times windows of 1..N rows. Spans still start every 8 tokens, so the 1..8 table is what an 8-row
  calibration fits; rows 9..N sit on a line of their own.
- The ranks compare N and the drafter block at load. Stats gain `depths` (drafts verified a round).
- `GLM53_TF_DFLASH_BLOCK=N` runs the DFlash2 block pass over N rows. It is an experiment: the published drafter's
  `block_size` is 8, so without it DFlash2 never proposes more than 7 drafts.

**Exactness.** Tokens are keyed samples of each row's own logits. Every decode / verify / MTP kernel is
row-independent from 1 to 512 rows: the exact prefill path runs the same code, and `qmm.bucket` puts 1-16 rows in
one tile. `commit(R, keep)` replays the kept prefix. No Triton kernel changed: `tests/kvpool_ptx.py` gives 33 / 33
identical at N = 8 and 16. CPU tests: the real `Stepper` / `Batcher` and the lone `auto_decode` on the hostile fake
model with windows of 9-16 rows == serial.

**Offline estimate** (`bench/draftsim.py` deep variants on the recorded streams; `bench/lookupsim.py` on token-exact
replays):
- copy / edit cells: +16-30%;
- agent turns (826 real GLM-5.3-Flash steps): +0.2-0.6%;
- prose, code generation, 4 streams: within ±0.2%;
- repetitive: +1.5%;
- with a 16-row DFlash2 block, depending on how sure the drafter stays past position 7: repetitive -1% .. +26%,
  code -2% .. +3%. DBloom says naive widening fails.

N3 (a global index of every earlier reply) adds +0.0% on the agent traffic, so it is not built.

**GPU plan.** A production / B `MAX_DRAFT_ROWS=16` / C B + `DFLASH_BLOCK=16`:
- glmbench tf / tweet / kit / edit;
- exact 10/10, batchexact 4/4;
- concurrent 1 / 4 streams;
- per-position acceptance from `depths` / `keeps`;
- memory (+~0.1 GiB a rank expected; 4 x 250k stress ≥ 8 GiB).

Adopt B at ≥ +10% on edit cells with everything else within noise.

**W9 (2026-09-28): GPU plan run, adopted** (`GLM53_TF_B12X=4`): `test_b12x_attn_patches.py` 23 passed (FP8 one
pass 1.71-1.81x the chunked kernel); on the production config exact 10/10, batchexact 4/4, 60k-token session
follow-ups cached > 0 with the cold reply sha 6/6, batch == alone 4/4, prefill +6.3% / +6.7% at 24.5k / 98k with the
same reply sha. The 0240 subset's `test_engine_snapshots_never_cross_bits` still fails its bits-3 control (not bit 4).
docs/RESULTS.md "W9".

**W10 (2026-09-29): GPU plan run, adopted** (`GLM53_TF_MAX_DRAFT_ROWS=16`; `GLM53_TF_DFLASH_BLOCK` not run). GPU tests: 4
passed, the 16-row child 17 of 19 (both failures `deepest >= 9` in `test_gpu_drafted_replies_equal_serial`: the
synthetic checkpoint never drafts that deep; replies == serial). On the production config + 0390: glmbench edit cells
+23.8% / +15.8% / +26.9%, every other cell within +-2%, **every reply hash of the 13 cells identical to the 8-row
load**, exact 10/10, batchexact 4/4, decode 1 / 4 streams in noise; memory in the combined gates. docs/RESULTS.md "W10".

## 0390 — latent absorb / expand retiled, same bits (`GLM53_TF_MLA_EXPAND=v2`; `latent.py`)

ROOFLINE gap 4: `_expand` runs at 1.8 TFLOP/s (2,415 us a 512-row sub-block) and `_absorb` at 6.4 (676 us), ~66 us a
token of prefill, on the FMA pipe because `LATENT_TC` changes replies (`GLM53_TF_LATENT_TC` stays off). 0390 keeps the arithmetic and
changes the tiling. Details, evidence and the GPU plan: `docs/MLA-EXPAND.md`.

**What v1 computes, per output element** (Triton 3.7.1 / 3.8, sm_121, read from its source and our compiled IR):
- one fp32 FMA chain over k = 0 .. K - 1 in order (K = 256 absorb, 512 expand), from +0.0:
  - Triton's Combine pass folds `acc + tl.dot(x, w)` into `tl.dot(x, w, acc)` (the TTIR dot takes the loop's
    iter_arg);
  - an `ieee` fp32 dot is lowered by FMADotUtility.cpp to `acc = fmuladd(a_k, b_k, acc)` per element, k ascending,
    whatever the tile or layout;
  - the PTX has only `fma.rn.f32` / `fma.rn.f32x2` and conversions;
- weights dequantized exactly (q * s + b; q * s is exact);
- one `cvt.rn.bf16.f32`.

**v2** (`_absorb2` / `_expand2`, dispatched inside `absorb` / `expand` when `EXPAND_V2`):
- the same chain, cut into 16-wide dots that each continue the accumulator;
- 16-128 rows a program by launch rows (`V2_TILES`: 16 for decode / verify / MTP windows, 64 for expand and 128 for
  absorb in prefill), so a kv_b slab is dequantized once for all of them;
- expand's 4 column tiles of a head and absorb's 8 latent groups are adjacent programs (u / q rows from L2);
- no spills, 128 registers x 8 warps at prefill tiles (2 programs an SM; v1's `_expand`: 255 registers, 452 B spill,
  4 warps);
- FMAs are 64-77% of the loop's issue slots (v1: 18-20%).

**Knob:** `GLM53_TF_MLA_EXPAND` = `v1` (default; empty / 0 / off too) or `v2` (or 1), read at import; each rank prints
a line with v2. No snapshot tag, no rank check, not per request (nothing to choose if the bits are equal). The NVMe
session compat hash covers it (not added to `KNOB_SKIP`: conservative). With v1, the PTX of `_absorb` / `_expand` is
byte-identical to the tree without 0390.

**Offline results:**
- `tests/test_mla_expand_interpreter.py`: 24 passed (~12 min, Triton 3.8 CPU):
  - v2 == v1 (as folded) == a numpy chain at 1-8,192 rows, q4 and bf16;
  - rows == alone;
  - every tile and BN gives the same bits;
  - the unfolded control differs.
- `tests/test_mla_expand_compile.py`: 21 passed under Triton 3.7.1 (the image's) and under 3.8.
  - v1's dot takes the loop-carried accumulator and v2's dots chain;
  - FMA path only;
  - no unfused fp32 add / mul in the PTX;
  - FMA counts match;
  - no spills (3.7.1).

**Expected** (arithmetic, not timed): 3,091 -> ~450-750 us a sub-block and layer, **~50-57 us a token = +7.5-8.5%
prefill** (1,471 -> ~1,590-1,600 tok/s at 24.5k), plus ~0.5-1 ms a decode round (v1 `_expand` is 109 us a call at
1-8 rows).

**GPU plan:**
1. `tests/cuda/test_mla_expand_patches.py`: v2 == v1 bit for bit, q4 / q4mse / bf16, 1 to 8,192 rows, every tile,
   in a CUDA graph.
2. `tests/cuda/bench_mla_expand.py --sweep`.
3. `test_latent_patches.py` with v2.
4. A load with `GLM53_TF_MLA_EXPAND=v2`:
   - reply sha `8794a3463259cc2f`;
   - exact 10/10, batchexact 4/4;
   - prefill at 24.5k / 98k >= +5%.

Any bit difference in step 1: keep the knob off.

**W10 (2026-09-29): GPU plan run, adopted** (`GLM53_TF_MLA_EXPAND=v2`). `V2_TILES` retuned from the GPU sweep: 4 warps
everywhere, absorb 64 x k32 at > 32 rows, expand 64 x k16 (speed only). GPU bitwise 50 passed (before and after the
retune), `test_latent_patches.py` with v2 20 passed. Bench at 512 rows: absorb 677 -> 346 us, expand 2,560 -> 886 us
(0.38x); at 1 / 8 rows expand 114 -> 68 us. Production config + v2: reply sha 8794a3463259cc2f, exact 10/10,
batchexact 4/4, sessions resumed == cold, **prefill +7.1% / +6.7%** at 24.5k / 98k (1,607 / 1,603 tok/s), decode not
lower (1 / 4 streams +1.1% / +0.9% mean). docs/RESULTS.md "W10".

## 0400 — KDA recurrence v2, same bits (`GLM53_TF_KDA_V2`; `kda_v2.py`, `fastpf.py`, `engine.py`, `sessdisk.py`)

ROOFLINE gap 5: the fast-prefill KDA recurrence (`fast_kda`, patches/0081) takes 63.5 us a token against a 15.7 us
roof. A 512-row lean sub-block costs `_kda_prep` 568 us (256 programs, 255 registers, 1 an SM), `_kda_state` 334 us
(**64 programs** on 48 SMs) and `_kda_norm` 41 us. That is ~150 MB of DRAM traffic a call, mostly the fp32
workspace: the prep writes 46 MB, and the scan reads it back, the two value blocks of a head a wave apart, so ~65 MB.

**Change.** `kda_v2.kda_prefill_v2`, `fast_kda.kda_prefill_chunked`'s contract, dispatched by `fastpf.kda_chain` when
`GLM53_TF_KDA_V2` is set:

- `1` (split): `fast_kda._kda_prep` unchanged, then `_state2`:
  - grid (value blocks, heads), blocks of `_BV` = 32 value rows;
  - the key dimension of the scan's dots in two 64-wide halves, the second dot starting from the first one's
    accumulator (the same mma chain), each operand loaded right before its dot (24 KB of shared memory, not 64);
  - `maxnreg` 168: 3 programs an SM, all 128 resident.
- `2` (fused): `_fused`, a persistent kernel (one program an SM).
  - Work items by ticket: prep(0), then per chunk c: prep(c, every head) and the scan steps of chunk c - LAG
    (value blocks of a head adjacent).
  - A scan step waits for its chunk's prep and its block's previous step; a prep waits for the ring slot it
    overwrites to be read. Flags: CTA barrier + one thread's `atom.release.gpu`; one thread's `ld.acquire.gpu` spin +
    barrier.
  - The state passes between steps through `state_out` (fp32).
  - The workspace is a ring of `_RING` = 3 chunks (~14 MB, L2-resident); A / T scratch per program. Reads of other
    programs' data are `.cg` (L1 is not coherent).
- `_kda_norm` unchanged in both.
- Anything `fast_kda` would do differently (`save_replay`, other precision / bv / warps, |lower| > 5.33) goes to
  `fast_kda`.
- The engine parses the settings at load and prints them.
- `sessdisk.KNOB_SKIP_PREFIX` gets `GLM53_TF_KDA_V2`: same bits, so the NVMe store stays valid.

**Why the bits are `fast_kda`'s** (`docs/KDA-V2.md` section 2):

- Row j of the state meets only column j of U / E / O. The sums run over keys (128) or chunk rows (64), never over
  values, so value blocks of any width compute the same elements.
- A tf32 `tl.dot` on sm_12x is an `mma.sync` m16n8k8 chain in ascending K from the given accumulator, independent of
  the tile's M / N and warps. The K split is the same chain.
- Triton's combine rewrite gives `O = dot(P, E, acc = dot(Q~, S^T))` and `S = dot(E^T, K^, acc = S * e^{G_C})` in
  `_kda_state`. v2 has the same chains.
- The only layout-ordered ops (the q / k norms, the decay cumsum, the forward substitution's 30 sums) stay in the
  prep with its code and 8 warps.
- fp32 stored and reloaded is exact. No bf16 rounding is added or removed.
- A register cap changes only the PTX's `.maxnreg` line.

**Offline results** (Triton 3.7.1 = the image's, and 3.8; torch CPU; ptxas 12.9 sm_121a).

- `tests/test_kda_v2_compile.py`, **17 passed on 3.7.1 and on 3.8**:
  - `_prep_item` on `_kda_prep`'s grid compiles to `_kda_prep`'s PTX, instruction for instruction;
  - `_state2` bv 64 / 8 warps has `_kda_state`'s float ops and PTX float counts;
  - every `_state2` setting (bv 16 / 32 / 64, 2 / 4 / 8 warps, K split on / off) and both `_fused` branches have the
    reference's elementwise ops in order, the same dot chains (precision, accumulator source, total K), mma v2 m16n8
    with kWidth 1, and the prep's 33 reduce / scan ops with identical full layouts;
  - `_fused`'s PTX fma / ex2 / rcp / sqrt / cvt / mma counts == prep + one step (no FMA contraction gained or lost);
  - `maxnreg` changes only the directive.
  - Resources (3.7.1):

    | kernel | registers / spills / shared memory | programs an SM |
    | --- | --- | ---: |
    | `_kda_state` | 255 / 404 B / 64 KB | 1 |
    | `_state2` bv 32 w4 | 240 / 0 / 24 KB | 2 |
    | `_state2` bv 32 w4, maxnreg 168 | 168 / 296 B | 3 |
    | `_fused` | 255 / ~1.1 KB / 64 KB | 1 |

    `num_stages` 2 needs 120-160 KB: not offered.
- `tests/test_kda_v2_interpreter.py`, **95 passed**. GPU-like interpreter: RNE bf16 casts; `tl.dot` as an
  emulated mma chain; Triton's combine rewrite modelled. All bitwise against `fast_kda`:
  - both modes, 11 settings, 1 to 8,192 rows, aligned and unaligned starts;
  - resume == fresh over 1-4 calls cut on the 64-row grid, with each cut's state == `fast_kda`'s;
  - aliasing; determinism;
  - the ticket order's waits all point backwards.
  - Inputs keep a large state: slow decay chosen per channel, small v. With per-element random decay the state was
    gone in a few rows and a swapped dot order went unnoticed.
  - Controls: the scan redone in numpy == the kernel; with the key halves swapped != the kernel; without the combine
    rewrite the K split != `fast_kda`.
  - Mutations run by hand are caught: swapped K halves; the fused step reading `state_in` at every chunk.
- Applies on 0001-0390.
- **Not checkable offline:** the tensor cores' internal rounding inside one mma, real concurrency, timing.

**Estimate** (arithmetic, not timed):

- split: -7..-11 us a token (+1.0-1.6% prefill);
- fused: -20..-35 us a token (+3-5%: 1,471 -> ~1,515-1,545 tok/s at 24.5k).

**GPU plan** (`docs/KDA-V2.md` section 7):

1. `tests/cuda/test_kda_v2_patches.py`: every mode and setting bitwise == `fast_kda`; 50 repeated runs plus 20 beside
   a busy stream; resume == fresh; aliasing; the engine entry. Plus the fast_kda / cindep / lean / b12x / session
   regressions.
2. `tests/cuda/bench_kda_v2.py 512 1024 2048 8192`, then `--sweep`: it prints the fastest bit-identical setting.
   Go on at <= 0.85x of fast_kda at 512 rows.
3. Engine gates with it on: reply sha 8794a3463259cc2f, exact 10/10, batchexact 4/4, resume == fresh (slot / RAM /
   NVMe).
4. `ab.py` 24.5k / 98k off / on / off / on. Adopt at >= +2% with every gate green.

Revert: unset the knob.


**W10 (2026-09-29): GPU plan run, not adopted** (stays off). GPU bitwise 345 passed (every mode and setting). Bench at
512 rows: the best split setting 0.845 ms vs fast_kda 0.913 (0.93x; the plan's bar 0.85x), every fused setting slower
(1.25-4.0 ms). Engine (split, bench's best = defaults): exact 10/10, batchexact 4/4, sessions == cold, same reply sha,
prefill **+1.2% / +1.0%**, under the +2% bar. docs/RESULTS.md "W10".

## 0410 — sparse latent attention v2, same bits (`GLM53_TF_SPARSE_V2`; `sparse_v2.py`, `b12x_attn.py`, `engine.py`, `sessdisk.py`)

ROOFLINE gap 2, on the path production runs since W9: b12x bit 4's one-pass kernel (`b12x_attn._lsparse_one`,
5.0-5.4 ms per 1,024 rows x 2,051 tokens, 25-27.5 TFLOP/s; the chunked kernel it replaced: 9.1 ms).

**What the one-pass kernel compiles to** (3.7.1, sm_121):

- Both dots and the softmax sit on `#mma` [1, 8]. The chained-dot heuristic gives QK the PV layout, and the
  [32 heads, 32 keys] scores tile is half as wide as 8 warps x n8, so **warps 4-7 recompute warps 0-3's scores**:
  768 mma a tile, 256 of them duplicates.
- Every warp reads all of q from shared memory each tile: ~430 KB of shared-memory traffic a tile, 256 KB of it q.
- The gathers are synchronous: token ids, then 16.9 KB of rows, then use, with one 8-warp CTA an SM (178 registers,
  66 KB).
- Measured ~8,170 clocks a tile against a static bound of ~3,360: latency-bound, which is why its time does not move
  with context.
- Heads-as-rows was already the case: one program, all 32 heads, one gather.

**Change.** `sparse_v2._lsparse_v2`, `sparse_latent_one`'s contract, run by `sparse_latent_one` when
`GLM53_TF_SPARSE_V2=1` (32 heads x 512 latent, default tile; `v2=` overrides for benches and tests). A Gluon kernel
(`triton.experimental.gluon`, shipped with the image's Triton 3.7.1), because the reference's bits are pinned to the
layouts Triton chose for it and Gluon states layouts explicitly. Per 32-token tile:

- the selected rows (FP8: 512 bytes + the scale) are gathered with `cp.async` (16 B, masked lanes zero-filled) into a
  ring of `stages` slots, `stages - 1` tiles ahead, through the page table, the token ids a tile earlier still;
- the FP8 tile is dequantized once (the reference's exact e4m3 -> f32 x 2^k -> bf16) into a bf16 tile both dots read;
- `qkl` 1: QK on [2, 4] (one m16n8 tile a warp, no duplicates), the 4 KB scores tile moved to [1, 8] through the
  consumed ring slot, where the softmax runs with `latent._ltile`'s ops in its order; p moved to the PV operand the
  same way;
- `qreg` 1: K 0-255 of q stays in registers for the row (QK as two chained dots), the rest in shared memory: 16 KB
  less shared memory, so 3 slots fit (99,200 B of 101,376);
- 5 barriers a tile, the second placed before every operand load (Triton's barrier analysis would add one there
  anyway, since it cannot tell ring slots apart; placed by Triton it left 256 registers live and the bf16 variant
  spilled).

Settings (`GLM53_TF_SPARSE_V2_CFG`, speed only, all the same bits):

| setting | registers (3.7.1) | shared memory | mma / smem traffic a tile |
| --- | ---: | ---: | --- |
| `3,1,1` (FP8 default) | 252-254 | 99,200 | 512 / ~286 KB |
| `2,1,1` | 254 | 82,688 | 512 / ~286 KB |
| `2,1,0` | 172-174 | 99,072 | 512 / ~352 KB |
| `2,0,0` (bf16 default) | 170-178 | 100,352-100,608 | 768 / ~475 KB |

`config()` rejects settings over GB10's 99 KB. There are no spills on 3.7.1 or 3.8. Also: the engine parses and
prints the setting at load (a Triton without Gluon fails the load, not a request), and `sessdisk.KNOB_SKIP_PREFIX`
gets `GLM53_TF_SPARSE_V2`.

**Why the bits are the one-pass kernel's** (`docs/SPARSE-V2.md` section 3):

- MMAv2's lowering chains each output over K in ascending k16 steps from the dot's c, whatever the warp layout, with
  K placed by kWidth (2 everywhere). v2's dots have the reference's operands and c: zero for QK (with qreg, the second
  dot continues the first), `o * alpha` for PV (Triton's Combine fold).
- The sum is not associative, and its tree follows the source layout (and changed between 3.7.1 and 3.8). Both
  reductions therefore run on the reference's own [1, 8] layout. The max is exact anyway.
- Every element-wise op is the same op on the same value in the same order.
- Masked keys are zero in both kernels.

**Offline results** (3.7.1 and 3.8; torch CPU; ptxas from each Triton):

- `tests/test_sparse_v2_compile.py`, **36 passed on both**:
  - the reference's single `#mma` [1, 8] appears verbatim in v2; kWidth 2 on every operand;
  - the float op sequence from the key loop to the end is identical (types resolved; QK compared by kWidth and total
    K) for FP8 / bf16, contiguous / paged and every setting;
  - chains: QK from zero (two dots with qreg, linked); PV from `mulf(o)`;
  - PTX float instruction counts equal, class by class (fma / mul / add / sub / div / ex2 / max / cvt / selp / setp,
    packed forms), mma 64 or 96 against 96; no rz / ftz;
  - the gathers are `cp.async.cg ... 16, src_size`;
  - shared memory == `smem_need()`; the defaults do not spill; dispatch and knob parsing.
- `tests/test_sparse_v2_emulator.py` (+ `tests/sparse_v2_emu.py`), **25 passed**:
  - Method: v2's own source runs on a numpy model of its Gluon ops (cp.async groups land at `wait_group`, shared
    memory as bytes with the slot reinterpretations aliasing, and a hazard model that raises on reading in-flight or
    unsynchronized bytes or overwriting unsynchronized reads). The reference runs in Triton's interpreter with the
    same model (nearest-even bf16, order-sensitive k16 dot chains, Combine fold).
  - Bitwise: FP8 / bf16, settings 3,1,1 / 2,1,1 / 2,1,0 / 2,0,0 / 4,1,1, counts 0 .. 2,051, contexts 700 to 6,000,
    paged == reference == contiguous, rows == alone / subsets, FP8 == dequantized.
  - Controls: an unfolded reference differs; the chunked kernel differs; a missing wait and each missing barrier
    raise; source mutations (token ids a tile early, the wrong slot) fail.
- The reference's PTX is identical before and after the patch (4 variants, debug info stripped).
- The 0360 offline tests still pass (6 + 10).
- The GPU test's own code was dry-run on CPU (emulator + interpreter): 7 cases pass.
- Applies on 0001-0400.
- **Not checkable offline:** ptxas / hardware treatment of the same arithmetic, and timing.

**Estimate** (arithmetic, not timed):

- the kernel 1.8-3.1x the one-pass kernel;
- -24..-36 us a prefill token (11 DSA layers) of ~638;
- **+3.5-6% prefill** (1,566 -> ~1,620-1,665 tok/s at 24.5k, similar at 98k);
- decode / verify / MTP unaffected (chunked kernel).

**GPU plan** (`docs/SPARSE-V2.md` section 7):

1. `tests/cuda/test_sparse_v2_patches.py`: bitwise == one pass for every setting, FP8 / bf16, paged, 1-2,048 rows,
   overlapping lists, extreme logits.
2. `tests/cuda/bench_sparse_v2.py 512 2048 8192` (contexts 10.7k / 85.8k, `--random`, `--bf16`): `same bits True`
   on every line, the best setting. Go on at <= 0.7x of the one-pass time.
3. Load V (production + the knob), with the gates: reply sha 8794a3463259cc2f, exact 10/10, batchexact 4/4, sessions
   cached > 0 == cold, prefill 24.5k / 98k >= +3%, decode unchanged. Plus a knob-off control load.

Revert: unset the knob.

**W10 (2026-09-29): GPU plan run, not adopted** (stays off). **Fixed a syntax error in this patch's `engine.py` hunk**
(the load message split an f-string expression across two literals: `engine.py` did not import). GPU bitwise: every
FP8 case passed; 6 bf16 cases hit `OutOfResources` because the test forced 3 stages on bf16 (fixed in the test: 30
passed, 3 skipped). Bench (FP8, 10.7k / 85.8k context): 1.16-1.25x the one-pass kernel (best cfg 2,1,1; plan bar
1.43x), 1.3-1.6% projected. Engine: exact 10/10, batchexact 4/4, sessions == cold, same reply sha, prefill **+0.0%**
(under the +3% bar); with 0390 and 0400 split together +1.5% over 0390 alone. docs/RESULTS.md "W10".

## 0420 — trimmed draft vocabulary (`GLM53_TF_DRAFT_VOCAB`; `draftvocab.py`, `mtp.py`, `decode.py`, `batch.py`, `dflash2.py`, `graphs.py`, `engine.py`)

**Question.** DECODE-PLAN E7: a draft step's last matmul is the 77,440-row head a rank (178 MB, ~0.9 ms of a
1.7 ms MTP step). Scoring only the ~32k most frequent tokens (FR-Spec) would save most of it, if the tokens a
drafter must propose are covered. Full write-up, coverage study, estimate and GPU plan: `docs/DRAFT-VOCAB.md`.

**Change** (default off = upstream code and graphs).
- `GLM53_TF_DRAFT_VOCAB=N`: each rank's drafters score the first N / 2 ids of its vocabulary half in the shipped
  ranking (`glm5_next/cuda/draft_vocab.txt`; public release: built from public text, `bench/draftvocab_public.py`,
  held-out coverage on the private sessions 78.8% / 87.1% / 91.3% at N = 16k / 32k / 48k, docs/DRAFT-VOCAB.md), then the half's lowest unlisted ids, a multiple of 64 rows, the same
  count on both ranks (a draft step lasts as long as the larger half). A path: that file's ids (`PATH:N`: the first
  N / 2 a half).
- The trimmed head is the same 4-bit words, scales and biases, gathered once at load (`take_rows`). With 12,288+ rows
  its logits are the full head's listed columns bit for bit.
- `_ARMS`: `mtp` (default), `dflash`, `all`. `_FALLBACK=RATE,WINDOW` (default `0.03,256`): a request drafts over
  the whole vocabulary while more than RATE of the last WINDOW tokens its MTP head absorbed (prompt, then committed)
  are unlisted. Whole-vocabulary MTP graphs (and a DFlash2 block) are captured next to the trimmed ones. In
  batched MTP one pass serves every slot: full if any slot's rule says so.
- The ranks compare the settings and the list's digest; the calibration key includes them.

**Exactness.** The target's sampling path is untouched (`sample_rows` without `draft`, `sample_multi`); a draft is
kept exactly when it equals the target's keyed sample. Draft columns map to token ids before the exchange, so draft
draws use the target's keyed Gumbel noise per (seed, position, token id); the noise is a stateless hash (no RNG
stream a draft could advance). With top-k / top-p, the draft's top-k over listed ids can take in tail tokens outside
the target's, an acceptance effect only. No Triton source changed (`tests/kvpool_ptx.py`: 33 / 33 identical).

**Offline results** (`bench/draftvocab.py`, `results/draftvocab/`): 826 GLM-5.3-Flash agent steps, 385k reply
tokens, held out by session and by time.
- Coverage at 24,576 rows a rank: prose 98.2-98.5%, code 97.8-98.8%, tool calls 98.3-99.2%; CJK (a stand-in) 63-64%;
  at 16,384 a rank 95.6-96.9% overall. A per-rank list beats a global top-N list at equal cost.
- Net (upper-leaning loss): MTP rounds +1.2-1.7% at 24,576 a rank (bound +3.1%); ~+0.5-0.8% for 1-stream decode
  overall; DFlash2 rounds -0.5% to -3.7% even with the fallback (one head pass a round saves ~0.6 ms); CJK -28% to
  -38% without the fallback, 0 with it.
- Tests: `tests/cuda/test_draft_vocab_patches.py` 27 passed on the CPU (drafted == serial through the real MTP loop,
  greedy / sampled, with / without fallback, rank 0 == rank 1; the target's sampling identical with and without the
  list), `tests/test_draft_vocab_interpreter.py` 3 passed; 19 existing suites unchanged (226 passed).

**GPU plan** (`docs/DRAFT-VOCAB.md` section 7): the unit tests in the image, then production +
`GLM53_TF_DRAFT_VOCAB=49152` against a control load: exact 10/10, batchexact 4/4, reply sha, `glmbench --suites
tf,kit,edit`, concurrent 1 / 4 streams, and acceptance by position per drafter (`bench/acceptpos.py`). Adopt at
≥ +0.5% 1-stream decode with MTP position-1 acceptance within -0.02.

Revert: unset the knob.

## 0430 — draft-training records and a distilled MTP head (`GLM53_TF_DRAFT_DUMP`, `GLM53_TF_MTP_WEIGHTS`; `dump.py`, `mtpw.py`, `decode.py`, `batch.py`, `engine.py`)

**Question.** DECODE-PLAN T1 / T2: train the MTP head (FastMTP self-distillation) and re-fit a block drafter on the
abliterated target's own outputs. Both need, per token, what the drafters read (the final-normed hidden row, the
DFlash2 taps) and what they should predict (the target's next-token distribution), dumped at serving speed. The
plan, data and time budget, reference, trainers and licences are in `docs/DRAFTER-TRAINING.md`.

**Change** (default off: no hook runs, `mtpw.load` is `weights.load`).
- `GLM53_TF_DRAFT_DUMP=<dir>` (`dump.py`), load-time on both ranks (checked at start):
  - **prefill:** each chunk of `decode._prefill` (exact / fast / lean / row-split / pipelined), before its commit.
    The head runs on all rows, 256 at a time (the same `qmm.matmul` as a verify window).
  - **decode:** `Batcher._verify` after the accept loop. The kept rows of every slot, whose logits the round
    already computed.
  - Each rank takes the top-k and the log-sum-exp of its vocabulary half. One all-gather on the NCCL control path,
    then a merge (log-probabilities over the whole vocabulary). Rank 0 writes; rank 1 does not.
  - Files: `<dir>/<host>-<pid>-<t>/`: `meta.json`, `index.jsonl` (a segment a line: kind, start, n, prefix hashes
    `h0` / `h1`, array offsets) and `shard-NNNNN.bin` (4 GiB). A background writer with ≤ 4 segments queued. It
    stops at `_GIB`.
  - Records start at `arm()` (the engine ready): calibration and capture passes are not written.
  - `_TAP_LAYERS` taps other layers than the served drafter's (the engine's tap buffers take them), with the DFlash2
    drafter off while they differ.
- `GLM53_TF_MTP_WEIGHTS=<dir>` (`mtpw.py`, called from `GlmEngine.__init__` in place of `weights.load`):
  - `train/mtp_distill.py`'s export: full BF16 tensors under the checkpoint's names, MTP layer only. They replace the
    checkpoint's through a `split.RankReader` overlay (this rank's slice by `split.rule`, stored as
    `GLM53_TF_NONEXPERT` stores the checkpoint's).
  - Unknown or unused names are refused.
  - The prepared tree's key includes the folder's files. `weights.py` is untouched, so production's prepared
    folders stay valid with the knob unset.

**Exactness.**
- The dump only reads buffers the next forward overwrites (after the drafters absorbed them, before the commit /
  the next round), and runs its own head into its own scratch. Replies are unchanged.
- A distilled MTP head only drafts: drafted == serial holds with any head.

**Offline results.**
- `tests/test_draft_dump.py`: a fake two-rank engine through `dump.Dumper`:
  - the merged top-k == top-k of the full log-softmax;
  - a resumed request's rows link to the first request's prompt segments; every row owned once;
  - tokens / hidden / taps / lp round-trip (fp8 taps within e4m3), a tap subset.
- `tests/test_drafter_train.py`: dump → `train/mtp_distill.py` → export → reload → `train/eval_accept.py` →
  `train/dflash_distill.py` → export → reload, on CPU.
- 17 CPU tests pass with `tests/test_drafter_ref.py`.
- Estimate: prefill +4-6% (the extra head, ~25 µs a row a rank), decode < 1% (~0.2-0.4 ms a round). 8,388 B a token
  without taps, 28,888 with 5 fp8 taps, 45,288 with 9.

**GPU plan** (`docs/DRAFTER-TRAINING.md` §8):
- `tests/cuda/test_drafter_ref_patches.py` on the synthetic checkpoint:
  - the reference MTP / DFlash2 == the engine;
  - the override loads an export;
  - replies identical with the dump on or off;
  - greedy decode rows' dumped top-1 == the emitted token.
- A 300-prompt smoke with overhead A/B (canary, multiturn 1 / 4, dump on vs off).
- Gate 1 on real dumps: the reference reproduces the engine's stock MTP acceptance within ±0.08.

Revert: unset both knobs.

## 0440 — streaming decode kernels, same bits (`GLM53_TF_DEC_EXPERTS`, `GLM53_TF_DEC_QMM`, `GLM53_TF_DEC_PDL`; `exl3_stream.cu`, `q4_stream.cu`, `decode_stream.py`, `exl3_mm.py`, `qmm.py`, `forward.py`, `sessdisk.py`)

**Question.** DECODE-PLAN E1 / E2: the two bandwidth kernels of a decode round run at ~193 GB/s (routed experts, 44%
of a 1-stream round) and ~190 GB/s (dense 4-bit, 27%). W11's probe streams the same bytes at 233-237 / 230 GB/s.
Can they get close to that with today's bits? SFXNZ-AUDIT adds two bars: Marlin's dense GEMV at 210-229 GB/s on our
shapes, and Marlin MoE at 213 GB/s. It also adds a caution, their PDL gate on sm_12x.

**Change** (default off: the hooks are `None`).

- **`GLM53_TF_DEC_EXPERTS=1`.** `exl3_mm.routed` (not fast) for member tables of <= 64 columns: decode, verify,
  MTP and DFlash2 windows and 4-slot batched windows; prefill chunks keep `grouped_loop`. Three launches:
  exl3.cu's `rot_in`, then `exl3_stream.cu` for gate/up and for down.
  - **Grid:** persistent, SMs x resident CTAs.
  - **Items:** every (member tile, expert, matrix, K split, column block), striped over the CTAs. Dead pairs are
    never visited: extra member tiles come from a per-CTA list.
  - **Rings:** 4 warps a CTA, each running exl3.cu's K range for its warp index. Each warp owns a ring of 4-8
    shared-memory k steps (trellis words and the live member rows, 16-byte `cp.async.cg`), kept full across item
    ends.
  - **Arithmetic:** `decode_tile` / `mma16816` verbatim. The warps are added in order by warp 0, and Z is stored
    for live rows.
  - **Epilogues:** a ticket on (member tile, expert, Hadamard block). The last item runs exl3_dec.cu's
    `gateup_epilogue` / `down_epilogue` (verbatim) for the tile's rows. The counters are left at zero.
  - `GLM53_TF_DEC_EXPERTS_CFG` places work only.
- **`GLM53_TF_DEC_QMM=1`.** `qmm.matmul` for Q4 weights and 1-64 rows through `q4_stream.cu`.
  - A persistent CTA ring 4-8 groups deep of words, scales, biases, input rows and group sums.
  - The nibbles go straight into the bf16 mma B fragments (`sub.rn.bf16x2` of `0x4300 | q`, exact).
  - The chain is `_qmm`'s: four chained m16n8k16 a group from +0.0, then `fma(xs, b, fma(p, s, acc))`.
  - Items are (64-column tile, K slice), the slices added in order by the last to finish. `SERIAL`: whole-tile
    items, the slices added in registers in order, the default for >= 96 tiles. Either way it is `_reduce`'s sum.
  - Refused (qmm's kernels run) under a Triton that does not fuse `_qmm`'s epilogue on every shape (3.8.0 does
    not). `force` skips the check, for measurements only.
- **`GLM53_TF_DEC_PDL=1`.** Both kernels launched as programmatic dependents.
  - Each CTA prefetches its first item's weights into L2, then waits.
  - Before `griddepcontrol.wait` only L2 prefetches (and E1's `ld.global.cg` of the grouping for their addresses,
    never kept) are issued.
- **`GLM53_TF_DEC_CTAS=N`** caps the grid.
- The knobs are skipped in the NVMe compat hash (same bits).

**Exactness.**
- Every output element gets the instruction sequence of the kernel it replaces: exl3.cu's warp K ranges, mma chain,
  warp order, split order and epilogue code; `_qmm`'s mma chain and fused epilogue as Triton 3.7.1 compiles it, and
  `_reduce`'s slice order. NT, stages, item order, CTA count and which CTA reduces only place work.
- So a row's bits do not depend on the window, the other rows, the configuration or the knob. Drafted == serial,
  batched == alone and resumed == fresh hold as before, even with the ranks set differently.
- **Found on the way:** under Triton 3.8.0, `_qmm` itself leaves 24-36 product-adds a thread unfused for 12288 x 4096
  and 77440 x 4096 (and 4096 x 128 at 64 rows), in bucket-dependent slots. An image moving to 3.8 would change
  today's bits and could break batched == alone there. `tests/test_decode_kernels_compile.py` checks it.

**Offline results** (no GPU; stack 0001-0430 + 0440; Triton 3.7.1 and 3.8.0; nvcc 13.4):
- `tests/test_decode_kernels_emulator.py` (61 tests, ~3 min). Lane-level CPU ports of both kernels against
  matrix-level models of exl3.cu's routed path and `_qmm` + `_reduce`. Setup: an order-sensitive mma model, NaN
  garbage in shared memory, copies landing at random before their wait, random CTA interleavings.
  - E1, bit for bit for Xd and Y: 1-16-row windows, 4-slot mixes, skewed 20 / 40-row windows with extra member
    tiles, every configuration, grids of 1-50; plus real-K Z checks and the real-shape walk.
  - E2, bit for bit: 1-64 rows, K slices 1-8, N not a multiple of 64, bf16 / fp32, both item modes.
  - Rows alone == rows in the window.
  - 12 negative controls caught: a missing wait, swizzle, warp order, a dropped live row, a moved k half, the fma
    order, the chunk order, the slice order.
- `tests/test_decode_kernels_compile.py` (6 tests).
  - sm_121: E1 100-142 registers, 20-45 KB, 2-4 CTAs an SM; E2 68-166 registers, 19-47 KB, 2-5 CTAs an SM; 0 spills.
  - PDL prologues clean in the PTX; E1's helpers are exl3.cu's text.
  - `_qmm`'s TTIR / PTX sequence for 11 shapes x 3 buckets.
- Both extensions compile host + device against torch's headers.
- Estimate (W11 ceilings; DECODE-KERNELS.md §7):
  - E1: -2.3 ms (1 stream) / -9.7 ms (4 streams);
  - E2: -2.1 / -3.0 ms;
  - PDL: -0.4 / -0.8 ms;
  - total **+9.7% 1 stream, +12.8% 4 streams** (range +5-16% / +7-19%).

**GPU plan** (DECODE-KERNELS.md §8, ~45 minutes).
1. `tests/cuda/test_decode_stream_patches.py -k "experts or qmm or pdl"`: bitwise on == off at the real shapes,
   every configuration, graphs, rows alone. Includes the PDL race test: a late writer releases its dependents, then
   writes their poisoned inputs 0.3 ms later, beside a busy stream.
2. `tests/cuda/bench_decode_kernels.py`: old vs new µs / GB/s at 1 / 2 / 4 / 8 / 16 rows and 4-slot mixes, a roof
   probe beside them, probes 1 / 2. Pick the configurations; gate ≥ 1.05x on U = 8-22 and the five largest dense
   shapes.
3. Engine tests, then the existing suites with the knobs on.
4. Production A/B with exact / batchexact and reply SHAs equal. Adopt at ≥ +3% 1 stream; PDL separately at ≥ +1%.

Revert: unset the knobs.

**W12 (2026-09-29): GPU plan run; not adopted** (docs/RESULTS.md "W12"). Offline suites in the image pass (emulator 61,
compile 6: Triton 3.7.1 fuses every `_qmm` epilogue). E1 / E2 bitwise at the real shapes and the **PDL race test: 16
passed** (the race test's `load_inline` helper needed a C++ declaration: test fix). Engine tests 9 passed; the existing
suites with both knobs on pass (deep verify: W10's known coverage assert only). **`bench_decode_kernels.py` failed the
gate** (every variant same bits): E1's best configuration (8,3,8,3 + PDL) streams **~160-163 GB/s at every window**
against the old kernel's 201-218 (0.74-0.80x), and probes 1 (no decode) / 2 (no mma) run at the same speed, so the
ring's loads in flight, not the ALU, are the limit. E2 wins only the small shapes, which the bench keeps in L2 (3
copies < 24 MB; 1.1-2.1x), and loses the five largest 10-18% (head 0.82-0.86x, KDA projection 0.85-0.89x, MLP gate/up
0.89-0.91x, DSA o 0.87-0.91x, MLP down 0.96-1.01x). One engine load of `DEC_QMM=1`: exact everywhere (every reply hash
equal), **1 stream -3.9%, 4 streams -3.1%**. E1 and PDL were not loaded. Knobs stay off.

## 0450 — a GPU-resident decode round (`GLM53_TF_GPU_ROUND`, `GLM53_TF_GPU_ROUND_PEEK`; `gpusample.py`, `gpuround.py`, `resident.py`, `batch.py`, `decode.py`, `dflash2.py`, `kda.cu`, `kda.cpp`, `kda.py`, `engine.py`)

**Question.** DECODE-PLAN E6 / F3 / F4: take the host off the decode round's critical path and batch the per-slot
fixed cost.

- A round reads the device from the host several times:
  - the sampler's candidates (`batch.py:333`);
  - every MTP draft (`decode.py:54`, `batch.py:379`);
  - DFlash2's candidates (`dflash2.py:526`).
- Each read is followed by numpy and Python, then per-slot commit launches.
- At 4 streams, each slot costs ~6.6 ms a round on top of its rows (W9). W11 splits that into: drafting passes 2.4,
  KDA 1.5, idle 1.1, attention 0.9, exchanges 0.6 (`docs/GPU-ROUND.md` §1).

**Change** (default off: every path is today's).

- **`sample`** (`gpusample.py`): `_choose_kernel` draws each row from its gathered candidates with the host's float64
  operations in the host's order:
  - ranks by (value desc, id asc);
  - the division by max(T, 1e-6);
  - the splitmix64 uniform;
  - `log(-log(u))`, and the top-p cut with numpy's pairwise sum and a sequential cumsum;
  - the first maximum.

  log / exp:

  - They are glibc 2.39's aarch64 `__log` / `__exp` as Ubuntu builds them (GCC contracted glibc's C into
    fmadd / fmsub): ported from the disassembly, with glibc's own tables (read from the binary, == e_log_data.c /
    e_exp_data.c).
  - Every path is covered, including `log(+-0)` (`u` rounds to 1.0 once in 2^53 draws). Compiled with
    `enable_fp_fusion=False`.
  - `_probability` (the drafts' confidence) the same way.

  Where it is used:

  - `batch.sample_multi` / `sample_drafts` and `decode.sample_rows` read tokens (and probabilities, rank 0's rider
    words) instead of the candidates. Rows past 128 kept / 256 gathered candidates stay on the host.
  - At load, both ranks compare the kernels with their own numpy (libm on 3 x 32,768 values, 4,096 draws). A
    difference on either turns the device draw off on both.
- **`resident`** (`resident.py`, `gpuround.py`; needs `GLM53_TF_DECODE_OVERLAP`'s `plan`).
  - When rank 0's plan rider rode empty and every slot can, both ranks run GPU-resident rounds after the normal round:
    - `_stage_kernel` (window ids, tie routes for padded rows, sampler positions);
    - `compute_multi` (the batcher's graph policy per key);
    - one exchange + `_choose_kernel`;
    - `_accept_kernel` (`Stepper.accept`'s rule, `done`, counters, the MTP token backlog, a record);
    - `kda.replay_slots` (every slot's replay in one launch, keeps on the device);
    - `_conv_shift_dev`;
    - `_backlog_kernel`;
    - the next window's drafts on the device: batched MTP head passes with device-sampled drafts and
      `DepthOptimizer.mtp_next`; DFlash2 `add_taps_dev`, the block graph, `dflash2._chain_kernel`, `f_depth`;
    - `_finish_kernel`, then one pinned record.
  - `PEEK=1` (default):
    - The host does round k - 1's `Stepper` bookkeeping (records, emits, depth calibration) while round k's forward
      runs.
    - It waits only on small pinned copies of the counters, after each MTP chain step and after the drafting. Chains
      run exactly their steps and windows are never padded.
  - `PEEK=0`: the host launches a whole round ahead (no reads while launching). Bounds are one round stale: MTP steps
    past a chain's end run for nothing, and windows are padded, tie-routed and never kept.
  - Drains (the round in flight processed, host state materialized, `_finish` for finished requests):
    - a finished request;
    - rank 0's drain flag (queue, cancel, stopping) riding the sampler exchange;
    - a lookup match;
    - an uncertain context mode (2,051, a pool bucket);
    - the cache's end or a pool shortage.
  - Every check comes before a round is launched.
- **`kda`**: `kda.chain_slots` runs every slot's chain in one launch a layer.
  - Its prologue replays the previous window's kept rows (keeps read on the device: `replay_kernel`'s loop) and writes
    the committed state.
  - Then it runs the real rows only, saving into a second 16-row scratch set per slot (~27 MB a rank).
  - The separate replay pass (71 MB read + written a slot a rejected window) and two copies a slot a layer go.
  - Drains materialize with `replay_slots`.

**Exactness.**
- The draw is the host's arithmetic op for op, and the accept rule is `Stepper.accept`'s.
- Commits keep the update routine and row order.
- Drafts only propose. So replies, committed states and snapshots are today's: drafted == serial, batched == alone,
  resumed == fresh.
- Both ranks run the same kernels on the same gathered data, make the same host decisions from equal device records,
  and drain after the same round.

**Offline results** (`docs/GPU-ROUND.md` §4).
- The sampler == host (`choose_rows`, greedy lexsort, `_probability`) on 1,900+ random row sets, in Triton 3.7.1 and
  3.8's interpreter.
- The Triton port == Ubuntu 24.04's aarch64 libm executed in unicorn on 5.4M inputs, and the scalar spec on 10.5M.
- Glue kernels == the host rules (the accept rule, `mtp_next` / `f_depth` / the confidence cuts on random calibrated
  histories).
- The real `Batcher` / `Stepper` on a hostile fake model:
  - replies and committed states == serial, the MTP cache and the DFlash2 context position by position;
  - peek and lag, MTP batched or not, `kda`;
  - two ranks in threads with identical logs;
  - no device read while launching (lag) / only the peek records (peek).
- sm_121:
  - the sampler's float64 exactly as written;
  - `chain_slots` = `chain` + `replay_layers` per PTX opcode (nvcc 13.4), <= 64 registers.
- Existing CPU suites unchanged with the knob unset.
- Applies after 0440; 0460 / 0470 / 0490 apply after it.
- Estimate (W11 round): 1 stream -0.6..-1.2 ms (+1-2%), 4 streams -3.5..-7 ms (+3-6%).

**GPU plan** (`docs/GPU-ROUND.md` §6):
- the KDA kernels bitwise;
- the sampler over 10^8 draws against the node's numpy;
- the synthetic checkpoint with each part == serial;
- loads A (prod) / `sample` / `resident` / `1` / `1` + `PEEK=0` through exact, batchexact, transcripts, tf / kit /
  edit hashes, 1 / 4 streams x 5 reps, cancel, arrival, resume;
- nsys of the idle outside the forward and the KDA family.

Revert: unset `GLM53_TF_GPU_ROUND`.

**W12 (2026-09-29): GPU plan run; not adopted** (docs/RESULTS.md "W12"). **KDA fold bitwise** (`replay_slots` ==
`replay_layers`, `chain_slots` == replay + `chain`): 8 passed. **Sampler at scale** (`results/W12/sampler.py`,
`self_check`'s two halves at size): 104,857,600 keyed draws and 2.5 x 10^9 libm values against the node's numpy /
`choose_rows`, **0 differences**. The synthetic-checkpoint GPU tests (run with `TRITON_INTERPRET=0`: the module
defaults to the CPU interpreter, which made the first run crawl) fail for every part, and `test_batch_parallel` /
`test_batch_sessions` with `GPU_ROUND=1` fail their sampled cases: **a harness artifact**. The one-GPU fakes
(`_TwoCopies`, `_Gather`) give rank 1 a copy of rank 0's candidates, so every id appears twice with the same value;
the host's lexsort keeps one, the device pick adds the tied winners (instrumented: id 88 -> 176). Real ranks hold
disjoint vocabulary halves, and the **engine is exact in every mode** on the model (exact, batchexact, W9
transcripts, 13 glmbench cells' hashes). Still a latent difference between the device pick and the host rule on exact
duplicates (fix the kernel's tie rule or the fakes). Engine A/B vs three controls: `sample` -0.1% / -0.4% (1 / 4
streams); `resident` -2.3% / +0.5% (tf code greedy -18%, hashmap -8%); `1` -2.3% / +1.2%; `1` + `PEEK=0` -4.3% /
-4.3%. Under the bars (4 streams >= +3%, 1 stream not lower): off. No nsys was taken, so the resident rounds' loss
is not decomposed.

## 0460 — L2 prefetch in every decode path, and the RoCE exchange's residual latency (`GLM53_TF_L2PF`, `GLM53_TF_ROCE_STRIPE_KB` / `_INLINE` / `_LAZY_CQ` / `_LEAN` / `_TRACE`; `l2pf.py`, `l2pf.cu`, `l2pf.cpp`, `forward.py`, `batch.py`, `latent.py`, `decode.py`, `overlap.py`, `roce.py`, `roce.cpp`, `roce.cu`, `roce_common.h`)

**Question.** DECODE-PLAN E3 and E8.
- E3: 0040's prefetch sets no site in `batch.compute_multi`, the Batcher's multi-slot forward (production runs
  `GLM53_TF_BATCH=4`). Port it and extend it to the other latency-bound windows.
- E8: cut the per-exchange latency and the exchange count where the math allows.

**Change** (every knob off by default; off, each hook is one dict lookup).

- **`GLM53_TF_L2PF=1` (`bulk`).** A side-stream kernel issues `cp.async.bulk.prefetch.L2.global` over a static
  per-site table. Sites (`_SITES`):
  - `a` at a layer's attention all-gather: fn, post norm, router, bias, shared gate/up (or the dense MLP's gate/up);
  - `f` at its FFN all-gather: the next layer's fn, input norm, input projection (after the last layer: the head);
  - `o` after the attention projections: KDA `o`, or latent DSA `kv_v` + `o`, while the chains / attention core run;
  - `e` (opt-in) after the router: `_EXPERT_KB` of each picked expert's gate and up trellis, indexed on the device
    from `qmm.Group`;
  - the MTP head's own `a` / `f` / `o` / `e`.
- **Budget:** `_MB` a site (4 MiB, capped at half of GB10's 24 MB L2).
- **4-bit matrices:** scales and biases first. Then, for a one-wave launch (<= 192 programs), the head of every
  (64-column tile, K slice) chunk; for multi-wave launches, the first bytes.
- **Tagged paths:** `forward.compute`, `batch.compute_multi` (new: tags and join), `mtp_compute`, `mtp_multi`,
  `batch._kda` / `_dsa`, latent `dsa_block_latent` / `dsa_multi`.
- **Joins and limits:** each forward joins the side stream; an eager fork never joins a capture. Windows over `_ROWS`
  (64) skip every site.
- **Mechanisms:** `lines` (`prefetch.global.L2`) and `touch` (0040's loads) as alternatives.
- **The shared expert** runs before the routed experts while a prefetcher is installed (0040's order).
- **Exclusive with 0040:** `GLM53_TF_COMM=prefetch` is refused together with `GLM53_TF_L2PF`.
- **RoCE** (extension renamed `tensorfold_glm_roce_v3`):
  - **`_STRIPE_KB`:** padded shards below it travel on one HCA (seq % HCAs): one post, one flag. The kernel, the
    proxy and the test loop share `roce_common.h`'s rule. It is agreed at load.
  - **`_LEAN=1`:** the doorbell is an `atom.acq_rel.sys` arrival (none for one block) plus a `st.release.sys`
    sequence, instead of two `fence.sc.sys`; the tail has no fences.
  - **`_INLINE=N`:** stripes up to N bytes (capped by the QP's granted inline size) are posted inline.
  - **`_LAZY_CQ=1`:** completions are reaped in the idle loop.
  - **`_TRACE=N`:** a device ring of kernel stamps plus the proxy's stamps. `Runtime.trace_rows`,
    `roce.trace_summary`, `bench_roce.py --trace`. The stats gain `inline_posts`, `single_hca_ops`, `inline_cap`.

**Exactness.**
- The prefetch kernels' PTX holds only loads and L2 prefetches. There is no store, atomic or reduction (checked).
  Every model kernel keeps its arguments and stream order, the shared-expert reorder aside (disjoint buffers).
- The RoCE changes move the same bytes in the same rank order. Each used HCA keeps payload-then-flag on one RC QP.
  The lean ordering keeps 0230's happens-before edges.
- **The count stays 100 / 115 a round.** Fusion and deferral are impossible exactly: hc is nonlinear and attention
  and FFN are sequential. Micro-batching re-reads the weights (+12 ms to hide ≤ 4.2). No model exchange falls to
  NCCL at 4 streams (W11).
- **Zero-copy** (NIC DMA from the tensors): not built. The device memory is not registrable without GPUDirect, and
  host-pinned tensors would need a send-completion wait and parity-keyed graphs for ≤ 1 us an exchange.

**Offline results** (no GPU; stack 0001-0440 + 0460; nvcc 13.4, Triton 3.7.1, torch 2.14 CPU):
- `tests/test_prefetch_comm.py`: 22 passed.
  - The port: compute_multi and mtp_multi reach the lone forward's sites in order, with rows, and join. With the tags
    removed the test fails (checked).
  - Also: plans, heads, expert plans, knobs, `agreed()`, `trace_summary`.
- `tests/test_roce_protocol_model.py`: 166 passed, 76 of them new (one-HCA / inline under random schedules, 3 ranks,
  two controls that time out).
- `tests/test_prefetch_comm_compile.py`: 4 passed.
  - `l2pf.cu`: 22-30 registers, no spills.
  - `roce.cu`: still 64 registers, 0 spills; lean forms present.
  - Both `.cpp` files pass the syntax check.
- The RoCE / batch / overlap / sessions / draft-vocab host tests pass. The series 0001-0440 + 0460 applies in order.
- Estimate (W11 rounds): E3 `a,f,o` -1.05 ms (0.65-2.1), `e` -0.15 / -0.3, E8 knobs -0.1. **Total +2.4% 1 stream,
  +1.2% 4 streams (mid).** DECODE-PLAN's E8 shrinks: of 2.3 ms exposed at 1 stream, ~0.9 is peer skew.

**GPU plan** (PREFETCH-COMM.md §6, ~70 minutes).
1. `tests/cuda/test_prefetch_comm_patches.py` plus the regressions (`test_roce_patches.py`, `test_comm_patches.py`,
   `test_batch_parallel_patches.py`, `test_decode_stream_patches.py -k pdl`).
2. `bench_roce.py bench --trace 4096` per RoCE knob.
3. stress / fault / soak with the chosen knobs.
4. Engine A/B: L2PF (a,f,o / +e / 8 MiB) then + RoCE knobs.
   - exact 10/10, batchexact 4/4, transcripts byte-identical;
   - 1 / 4-stream decode;
   - one nsys capture.

Adopt L2PF at ≥ +1.5% single stream with 4 streams not lower. Adopt the RoCE knobs only after a clean stress and a 1 h
soak.

Revert: unset the knobs.

**W12 (2026-09-29): L2PF adopted at 8 MiB a site; the RoCE knobs not** (docs/RESULTS.md "W12").
**Two bugs fixed in this patch** (found by its GPU unit tests; the patch file now carries the fixes):

- `roce.py` `_ext()`: the new `# patches/0460 ...` comment was written on the `load(name="tensorfold_glm_roce_v3", ...)`
  line **before** `sources=[...]`, so the call lost its sources and the extension never built. Every image with 0460
  would have fallen back from RoCE to NCCL at load, knobs off included (`test_roce_patches.py` 6 failed / 19 errors ->
  25 passed).
- `l2pf.cu`: `cp.async.bulk.prefetch.L2` needs sm_90+, and the image's `TORCH_CUDA_ARCH_LIST` also builds compute_80,
  so `GLM53_TF_L2PF=1` could not build (the offline compile test built sm_121 only). Now under `__CUDA_ARCH__ >= 900`,
  with a line-prefetch loop for older targets (never run on GB10).

GPU tests after the fixes: `test_prefetch_comm_patches.py` 25 passed (L2PF bulk / lines / touch at sites a f o e,
the Batcher, the RoCE loop runtime with `STRIPE_KB` 0 / 32 x `LEAN` 0 / 1); `test_roce_patches` 25, `test_comm_patches`
15, `test_batch_parallel_patches` with L2PF 26, `-k pdl` with L2PF 1 passed. Engine A/B vs three controls (every
reply hash equal): **`L2PF=1` (4 MiB) +0.8% 1 stream / -0.1% 4 streams**; sites a,f,o,e +0.4% / -0.1%; **`L2PF_MB=8`
+2.4% 1 stream (prose +2.9, code +2.5, edit +1.7), +1.2% 4 streams, lone requests +2.8%**: over the +1.5% bar, adopted
with the combined candidate (+ `BATCH_CAPTURE_AFTER=8`: 1 stream +2.5%, 4 streams +2.8%, all gates). RoCE knobs (bench
`--trace 4096`, two nodes, bits equal): base-to-base spread 0.5-1.4 us a op; only `LEAN=1` comes near the 0.3 us bar
(-0.2 to -1.4 us at 16-64 KiB, inside that spread, <= 0.1 ms a round): no stress / soak / engine load, knobs off. The
trace's `notice` component prints ~1.8 x 10^15 us (a clock-domain mix-up in that stamp).

## 0470 — 8-bit non-expert weights and a per-class precision map (`GLM53_TF_NONEXPERT=q8`, `GLM53_TF_NONEXPERT_MAP`; `qmm.py`, `fast_qmm.py`, `latent.py`, `weights.py`, `fastboot.py`, `l2pf.py`, `overlap.py`, `fastpf.py`, `dump.py`, `engine.py`)

**Problem.** q4mse (0001) stores every non-expert matrix at 4.5 bpw.

- The sfxnz kit's data (SFXNZ-AUDIT §4 item 5) says 4.5-bpw non-expert weights on this model cost ~6x the A/A KL.
  Top-1 agreement there was 95.6%.
- Our only quality gate so far is MMLU-200.
- The goal: higher-precision non-experts, paid for by the decode kernel gains.
- `bf16` (upstream) is the only other mode. It is ~3.6x q4's bytes and does not fit production memory.

**Change.**

- **`qmm.Q8`: int8, symmetric, one bf16 scale per (column, 128 inputs).**
  - The scale is the smallest bf16 >= amax / 127, so nothing clips.
  - Chosen on real `zai-org/GLM-5.3-Flash` tensors over per-channel, g64 and zero-point variants
    (docs/QUALITY-PLAN.md §1.2).
  - `quantize8` is elementwise torch: the same bits on the CPU and the GPU.
- **Kernels.** Every one sums a group as one mma chain (two 64-input halves), then one FMA with the group's scale,
  groups in order:
  - `_q8mm`: decode / verify / MTP, with qmm's split-K slices (`q8_split_k`, never more partial rows than Q4's);
  - `_fq8`: prefill, one accumulator in fast chunks; `exact` keeps qmm's slice order and bits;
  - the latent absorb / expand tile loaders (`IS_Q4 = 2`, exact `s * q`).
- **Weight loads.** Weights are read as int32 words and unpacked in registers (`_q8_unpack`). Triton then streams
  them with `cp.async` like the 4-bit words. Plain int8 tensors compiled to a `ld.global.b8` a byte, unpipelined.
- **Tiles.** Chosen offline to compile without spills for sm_121 on Triton 3.7.1 and 3.8.0 at every per-rank shape;
  not yet timed. Env overrides: `GLM53_TF_Q8_DECODE_CFG`, `GLM53_TF_Q8_TILE`.
- **`weights.py`.**
  - Builds each non-expert matrix with its class: `kda_proj kda_fb kda_gb kda_out dsa_proj dsa_qb dsa_kvb dsa_out
    idx_proj idx_qb dense_gu dense_down shared_gu shared_down mtp_eh lm_head`.
  - `GLM53_TF_NONEXPERT_MAP` picks a mode per class, a group (`o_proj`, `down`, `out`, `attn`, `mlp`, `kv_b`,
    `head`, ...) or `default`, optionally `mtp.`-prefixed.
  - The most specific key wins; conflicting groups at one level are an error.
  - An 8-bit head gets the 4-bit draft copy a BF16 head gets.
- **Prepared folders.** Keyed by `weights.precision_key()`: the plain mode when every class agrees, else the resolved
  table. Each precision is its own folder (~82-88 GB a node).

**Exactness.** Within a configuration, every guarantee Q4 has:

- a row's bits never depend on the row tile, the call's rows or the tile tables;
- drafted == serial, batched == alone, resumed == fresh;
- exact prefill == decode bits.

An 8-bit or mixed configuration is a different model from q4mse (new reply hashes), checked by
`bench/divergence.py` (KL / top-1 against `bf16`) and `bench/mmlu_full.py`.

**Result (offline).**

- Tests:
  - `tests/test_q8_interpreter.py`: 21 passed;
  - `tests/test_q8_compile.py`: 90 passed on 3.7.1 and 3.8.0;
  - `tests/test_nonexpert_map.py`, `tests/test_fastboot_prepared.py` (q8 added), `tests/test_divergence.py`: 42
    passed, CPU.
- 0440 / 0460 / 0430 CPU suites unchanged.
- GPU: `tests/cuda/test_q8_patches.py` (bits, and `-k timing -s` for GB/s and tile sweeps).
- Estimates and the GPU plan: docs/QUALITY-PLAN.md.

## 0490 — the context limit in the API, OpenAI-shaped context errors, `/tokenize` and token-ID prompts (`cuda/server.py`, `glm5_next/cuda/app.py`)

Why: a user of the public kit reported "only ~50K max context". Nothing told a client (or the user) what the server
served: `/v1/models` had no context field, the over-limit 400 used wording no agent client recognises, and the boot
log never said the limit. Host-side only, on by default (no knob): the engine, the kernels and every reply's bits are
unchanged.

- `GET /v1/models`: `max_model_len` and `context_length` (vLLM's and OpenRouter's keys; Hermes Agent and others read
  them), `created`, `root`, when the engine has a `limit` (GLM: `--context` / `CONTEXT`; 2,051 when unset). Engines
  without one list the entry as before.
- `GlmApp.check` answers a request whose prompt plus `max_tokens` (1 without) exceeds the limit with HTTP 400
  `{"error": {"message", "type": "invalid_request_error", "code": "context_length_exceeded", "param": "messages" |
  "prompt"}}` and OpenAI's / vLLM's wording: "This model's maximum context length is N tokens. However, you requested
  M tokens (P in the messages, R in the completion). Please reduce the length of the messages or completion." plus
  "(This TensorFold server was started with --context N (CONTEXT); restart both ranks with CONTEXT=M or more to serve
  it.)". opencode (`packages/llm/src/provider-error.ts`) then treats the turn as a context overflow and compacts;
  Hermes Agent parses the limit and the prompt / completion split. Streams: still refused before any header.
  (`check` may return `server.Problem`, a `str` with `code` / `param`; `server.error_body` builds the body.)
- A `ValueError` raised by the engine inside a stream (e.g. 0290's "more than the whole pool") is sent as an
  `invalid_request_error` event, as the non-streamed response already answered it with a 400.
- The load prints `[tensorfold] context: N tokens a request (max_model_len; set by CONTEXT / --context); KV cache:
  latent fp8 | per-head ...; K request slot(s); KV pool: T tokens shared by the slots`, with a pointer to the
  long-context config when N <= 65,536.
- RigMark (docs/RIGMARK.md, gap 1; proposed there as "0470"): `POST /tokenize` or `/v1/tokenize` returns
  `{"count", "max_model_len", "tokens", "token_strs": null}` for a string `prompt` (`add_special_tokens`, default true
  as in vLLM; GLM's post-processor adds nothing either way) or `messages` (rendered as a chat request renders them,
  0150's effort applied); anything else is a 400. It runs on the HTTP thread: no engine lock, no GPU, nothing sent
  to rank 1. `/v1/completions` takes one flat list of token ids as `prompt`: `App.check` requires ints (not bools)
  in `[0, vocab)`, a list of lists or strings is "one prompt a request"; `GlmApp.check` counts `len(ids)` for the
  context check; `App.run` passes the ids through `given_ids` (`GlmApp.given_ids` starts 0300's log ticket as
  `prompt_ids` does). Rank sharing, sessions, the prefix share and 0210 work on ids already.
  `GLM53_TF_REASONING_FIELDS` keeps its default (RigMark runs set it at start).

- 0290 pool sizing (`kvpool.pool_tokens`, `kvpool.build(..., limit)`, `decode.Engine`, the load line in `engine.py`):
  admission reserves `min(prompt + max_tokens + 64, capacity)` tokens of pages, capacity being CONTEXT + the verify
  window's rows (16), while the pool held exactly CONTEXT: a request with prompt + max_tokens within 64 tokens of
  CONTEXT passed the context check and was then refused as "more than the whole pool" (1,048,576: 4,097 pages needed,
  4,096 held). A pool set to at least the context limit now grows to the capacity rounded up to a page (4,097 pages
  at 1M, ~2 MB more a rank); a pool set smaller stays as set. The load line no longer says "one request alone cannot
  reach it" for production's pool. Which page holds a row never changes a bit.

Launcher (not in the patch): `scripts/serve.sh` stops with a pointer to `config/prod.env.example` when the config
file is missing, logs `CONFIG: CONTEXT=... tokens a request, KV cache ..., GLM53_TF_BATCH=..., GLM53_TF_KV_POOL_TOKENS=...`
before a start, warns when `CONTEXT` is unset (2,051) or <= 65,536, refuses `CONTEXT` > 131,072 on the per-head KV
cache (`GLM53_TF_LATENT_KV` unset: ~390 KB a token a rank does not fit next to the weights; `FORCE_CONTEXT=1`
overrides), and after the load logs "serving up to N tokens a request (max_model_len)".

Tests (host only): `tests/test_api_context.py` (19: the wording against copies of opencode's and Hermes Agent's
patterns, `/v1/models` with and without a limit, 400s with the code chat / stream / completions, plain errors keep
their shape, a stream refusal event, `/tokenize` string / special tokens / messages / 400, token-ID prompts streamed
and not with exact `prompt_tokens`, bad ids, a 65,536-id prompt within 1,048,576 and past 65,536, the boot line);
`tests/test_prompt_tokens.py` (the check's answer is now `context_problem`'s, code included);
`tests/test_serve_ops.py` (+4: missing config, the context line and warnings, the per-head refusal and
`FORCE_CONTEXT`, the max_model_len line). On the Sparks (new image): RigMark's `preflight.py` should pass, and the
usual exact / batchexact / reply-sha gates (the check and run paths changed).

**W12 (2026-09-29): in production** with image b5 (docs/RESULTS.md "W12"). `tests/test_api_context.py` +
`test_prompt_tokens.py` in the image: 20 passed, 1 skipped (no tokenizer dir). On the served model: `/v1/models`
lists `max_model_len` = `context_length` = 1,048,576; 13 + 1,048,576 tokens -> HTTP 400 `context_length_exceeded`
with OpenAI's wording; the pool edge, 13 + 1,048,531 = 1,048,544 tokens (reservation 64 past CONTEXT), is admitted
and answers; the boot line reads `KV pool: 1048832 tokens` (4,097 pages). b5 with every new knob off equals b4 bit for
bit (reply sha, exact, batchexact, W9 transcripts, 13 glmbench cells, N1's 12 long replies).

## 0500 — image input (`GLM53_TF_VISION`; `vision_prep.py`, `vision.py`, `glue.py`, `engine.py`, `batch.py`, `app.py`, `lookup.py`, `draftvocab.py`)

**Question.** Serve images in OpenAI chat messages for GLM-5.3-Flash, whose checkpoint carries a vision tower that
the engine ignored, without disturbing text serving or any cache.

**Findings** (`docs/VISION.md` §2):

- The 347 `model.visual.*` tensors are BF16 (1.127 GB, one shard).
- The LM uses **no positions**, so there is no M-RoPE. transformers' `glm5_next`:
  - asserts `qk_rope_head_dim == 0`;
  - calls every layer with `position_embeddings=None` and plain `arange` ids;
  - has no rope index.
- An image is `masked_scatter` of the tower's rows at the `<|image|>` positions, so the engine needs only the
  embedding step changed.
- This checkpoint's `chat_template.jinja` renders image parts as a "you cannot process images" reminder. The vision
  template (the vLLM kit's, MikeRoz's) emits `<|begin_of_image|><|image|><|end_of_image|>`.
- Upstream 0.3.6.2 has no vision: it skips the tensors and answers media with 400. The vLLM kit serves vision on by
  default (`LIMIT_MM` image 4, its own template).

**Change** (default off: the knob unset serves exactly as before).

- **Host** (`vision_prep.py`):
  - content parts (`image_url` / `input_image` / `image`);
  - `data:` and http(s) fetch with size / time / pixel limits;
  - RGB conversion as vLLM (alpha on white);
  - the processor, bit for bit (smart resize to the 16-8,000-token budget, bicubic antialiased uint8 resize, zero
    padding, fused normalize, merge-window-major patches, the frame duplicated over the temporal pair).
- **Rendering:** each image part becomes a sentinel before the checkpoint's template runs, then the vision
  template's placeholder. The prompt equals `Glm5NextProcessor`'s ids.
- **Virtual ids:** each image's `<|image|>` becomes its N = grid / 4 rows as virtual ids.
  - Ids are >= 2^24 and < 2^31. Each row's id is a hash of (the SHA-256 of the canvas, grid and settings; the row
    index).
  - A registry on rank 0 re-salts any clash within the process.
  - A typed-in `<|image|>` is refused.
  - `usage.prompt_tokens` and the context check count the rows. `/tokenize` returns them as `<|image|>`.
- **Engine** (`vision.py`):
  - the tower in PyTorch BF16 on rank 0 (cuBLAS + flash SDPA, the MLP row-chunked, its own CUDA stream, one
    request at a time);
  - a cache of rows by digest;
  - `exchange`: sorted ids and rows as int32 words through the all-gather, right after the prompt, in both the lone
    and the batch paths, data-driven so text requests exchange nothing;
  - `glue.embed` inside `vision.active(table)`, which wraps `_run`'s and `_piece`'s prefill: virtual ids read row 0,
    then `Table.fixup` writes the tower rows in every stream copy (device-only, never inside a graph capture).
- **Drafters:**
  - the MTP head's absorb gets the image rows (as vLLM's drafter does);
  - DFlash2 reads only vocabulary ids;
  - `SuffixIndex.propose` cuts lookup drafts at a virtual id (0450's resident rounds use it too);
  - the draft-vocabulary fallback window skips them.

**Caches:** nothing keyed by token ids changed, and none needs to. Two different images with the same text differ at
every image row. Covered:

- the lone and batch snapshots, the session store's entries, pages, fork marks and `lcp`, the prefix share;
- the NVMe tier's index and int32 packing.

These resume up to the image, never past it. The same pixels (any encoding) resume everything, including a
conversation's next turn. A false share needs a 31-bit collision per image row of the cached prefix.

**Cost:**

- rank 0: +1.13 GB tower, ~0.1 GB transient a screenshot (~0.5 GB at the 8,000-row maximum), and up to 2 x 256 MB of
  host caches;
- rank 1: the rows in flight (8 KB a row).
- Tower FLOPs: 5.37 T for 1024x768, 20.9 T for 1920x1080, 128 T for 3840x2160 (78% attention). That is 49 ms / 190 ms
  / 1.17 s at 110 TFLOP/s, estimated 75-90 ms / 0.3 s / ~2 s. The image rows' LM prefill (~0.9 s for 1,036 rows)
  dominates.

**Tests** (CPU, 89 checks: `tests/test_vision_prep.py`, `tests/test_vision.py`, `tests/test_vision_interpreter.py`,
`tests/test_vision_server.py`):

- processor parity with transformers bit for bit (16 sizes, plus 3,000 `smart_resize` draws and 63 token counts);
- the tower == transformers' `Glm5NextVisionModel` (fp32 1e-5, bf16 close);
- loading; the table; the two-rank exchange;
- `glue.embed` in the interpreter (BF16 / 4-bit tables);
- the HTTP flow and its 400s;
- every cache index for two images and the same image;
- the lookup cut;
- the whole prompt == `Glm5NextProcessor` on the real tokenizer.

Existing host and interpreter tests pass on the patched tree. `tests/cuda/test_vision_patches.py` (GPU, the next
Spark round): image rows set to real tokens' embedding rows reply exactly as the tokens across prefill modes,
drafters and resume; plus the real tower's latency and memory.

The GPU plan (text 10/10 unchanged, real images against the vLLM kit, cache exactness, the 4 x 250k memory, latency)
is in `docs/VISION.md` §7.

- Applies after 0490 (and 0450).
- Revert: unset `GLM53_TF_VISION`.

## 0510 — lone-only batched graphs, the verify graph in pieces, a launch probe (`GLM53_TF_BATCH_GRAPHS=lone`, `GLM53_TF_VERIFY_SPLIT`, `GLM53_TF_GRAPH_PROBE`, `GLM53_TF_RESIDENT_REPORT`; `batchplan.py`, `batch.py`, `resident.py`, `vsplit.py`, `graphs.py`, `decode.py`)

Why (docs/THEORY-2.md §4 items 1, 2 and 9): batched graphs off (`GLM53_TF_BATCH_GRAPHS=0`, load G0) measured +2.6%
(W11) / +1.9% (W12) at 4 streams, because rounds of 2-4 slots meet many (slots, rows, parity) keys and each capture
costs a forward, but a request alone in slots 1-3 lost ~3% (its rounds are the batcher's, graphed under `1`). And at
one stream ~1.1 ms a round of GPU idle sat inside `cudaGraphLaunch` of the ~1,650-node verify graph (measured under
node-level nsys tracing, which inflates launches: the uncaptured size is 0.1-1.1 ms).

- `batchplan.graph_policy` / `graphs_for`: `1` (unset, as before), `0`, `lone` (also `single`, `one`). `Batcher.graphs_on(n)`
  answers for a round of n active slots; `_forward` (padding, buckets, capture / replay), 0450's `RealOps.forward` and
  `_pass` (batched MTP head) ask it. Under `lone` a round of 2+ slots takes the eager path of `0` (the same
  `compute_multi(..., nchs, host_pos)` call); a round of one slot captures / replays as `1`; slot 0 alone still replays
  the engine's own graphs. The policy is in the ranks' batch-settings exchange (checked equal). Boot line:
  `batch graphs (patches/0510): lone rounds only`.
- `vsplit.compute_pieces(w, st, b, R, n, npb=)`: `forward._compute` (logits on, the capturable path) as n callables
  over `split_bounds` layer ranges (the first also embeds, the last takes the final stream mean, norm and head; each
  ends with `forward._join`, the prefetch side stream's join: a wait only). `Graphs.__init__` captures the main graphs
  that way when `GLM53_TF_VERIFY_SPLIT` >= 2 (or the probe is on), `long_step` does the same for 0050's ("main", R,
  parity, npb) keys; `vsplit.Split.replay` runs the pieces in capture order (a device sync before each piece's capture,
  as `roce._order` assumes). `GLM53_TF_VERIFY_SPLIT_FIRST=L` sizes the first piece.
- `GLM53_TF_GRAPH_PROBE=N`: a Triton `%globaltimer` stamp as node 0 of the first piece, an eager stamp before the replay;
  per replay the host ns of each piece's `replay()` and the host gap before the first; every N replays (one sync)
  `[tensorfold] graph probe (patches/0510) r0 ('main', R, parity) pieces=P n=..: replay host us p50 .. p90 .. (first
  piece p50 ..); first node +us p50 .. p10 .. p90 ..` on stderr. First-node delay = node-0 stamp - eager stamp - the
  host gap: the launch's exposure on an idle stream.
- `GLM53_TF_RESIDENT_REPORT=N`: `[tensorfold] resident rounds (patches/0510 report): rounds .. entries .. drains
  {reason: n} ; batch rounds {...}` every N drains (item 9's attribution of 0450's -18% on greedy code).

Same bits: the policy picks between two paths that run the same kernels (G0 passed exact / batchexact / reply hashes
in W11 / W12); the pieces launch `_compute`'s kernels in its order (CPU test with every kernel wrapper recorded); the
stamps write only the probe's own buffer. Tests: `tests/test_batch_graphs_lone.py` (44, CPU: parsing, the dispatch of
`_forward` / `RealOps.forward` / `_pass` on fakes under each policy, `split_bounds`, the pieces' call sequence ==
`_compute`'s for 1-8 pieces with and without `npb` and taps, capture / replay order on fake graphs, the probe
arithmetic); `tests/cuda/test_batch_graphs_lone_patches.py` (GPU: Split objects of N pieces, window logits / hidden of
1-8 rows and replies == the default engine's with split 2 / 3 / 8 and the probe, lone batched replies == serial with
no multi-slot graph and a lone-slot graph captured). GPU plan and gates: results/THEORY2-SESSION (W13 loads L, CT, VS).

## 0520 — one hyper-connection boundary in one CUDA kernel, same bits (`GLM53_TF_HC_CUDA`, `_ROWS`, `_RG`, `_PDL`; `hc_fused.cu`, `hc_fused.cpp`, `hc_cuda.py`, `forward.py`, `batch.py`, `engine.py`)

Why (docs/THEORY-2.md §1.5, §4 item 3): every layer boundary runs `_hc_post` -> `_hc_partial` -> `_hc_finish`, 15.6 us
of latency-bound kernels (1-96 CTAs), 90 times a 1-stream round (~1.4 ms). 0190's Triton fusion was bitwise but 7-8x
slower. docs/HC-FUSED.md has the design, the bit-identity argument and the GPU plan. (`GLM53_TF_HC_FUSED` is 0190's
per-request prefill knob, hence `HC_CUDA`.)

- `hc_fused.cu`: 32 step CTAs a group of `GLM53_TF_HC_CUDA_RG` rows (one per 128-column step of a quarter of D): fn
  slice, X' of their columns by `_hc_post`'s expression (staged, and to a scratch XP), and `_hc_partial`'s per-step
  dot values (8-column chain, 16-lane butterfly) to a scratch TS; a ticket per group. One finisher CTA a row spins on
  its group's 32 tickets (acquire), then sums the steps in order (PART), the square sums (same chains and trees), writes
  X from XP (every step CTA has read X), and runs `_hc_finish` with its thread mapping. The last finisher resets the
  tickets (graphs). Arithmetic: only `.rn` intrinsics (ptxas contracts none) plus rcp / sqrt / ex2 `.approx.ftz` with
  the scaling ptxas wraps around Triton's `div.full` / `ex2.approx` written out; the fma where ptxas fused Triton's
  multiply with the next add (div + hc_eps, x / 2^k + eps, the square chains); Triton's mixed bf16 instructions.
  sm_121: 80 registers, no spills, 18.9 KB shared, 3 CTAs an SM.
- `hc_cuda.py`: knobs, `fits` (decode shapes, not fast chunks / b12x, <= ROWS rows), `post_pre` (the kernel or False:
  the caller runs the Triton pair), scratch per Buffers (allocated on the first eager use; never inside a capture),
  the layer-to-layer mark (`mark_ready` / `take_ready`: consumed by the very next layer_forward; a stale one costs a
  recomputation only), `self_check` (1 / 3 / 16 random rows, every output bitwise vs glue's kernels).
- Call sites: `forward.layer_forward` and `batch.compute_multi` (both boundaries of a layer), `engine.py` (ranks must
  agree on `GLM53_TF_HC_CUDA`; with 1, the self-check at load, before calibration and graph capture; on only if both
  ranks pass). Off: the same Triton calls in the same order (unit-tested).

Same bits: the specification (`tests/hc_fused_emu.py`) is Triton 3.7.1's executed arithmetic on sm_121: the Triton
kernels' own PTX, run in a PTX interpreter through a model of ptxas's contractions (checked against the SASS: FFMA /
FADD / MUFU counts exact for all three kernels), equals it; so do hc_fused.cu's own PTX and a lane-level port (1-16
rows, rg 1-8, random CTA orders, 7 input kinds incl. signed zeros, the unit sequences' scaling paths, subnormals,
inf / NaN, and an adversarial input for the collapse's per-slot fusion). 7 specification variants and 6 source
mutations are caught. Under Triton 3.8 the Triton kernels compile to other arithmetic (the self-check keeps the knob
off there). Not proven offline: the production image's ptxas (analysed: 13.1; the GPU test prints it), hardware unit
results and the ticket ordering (GPU test). Tests: `tests/test_hc_fused.py` (68, CPU), `tests/test_hc_fused_compile.py`
(33: sm_121 compile, IR / PTX / SASS checks, the interpreter runs, full extension compile),
`tests/cuda/test_hc_fused_patches.py` (GPU: fused == Triton for 1-16 / 24-64 rows, edge inputs, graphs, 2,112 CTAs,
PDL late writer, self-check), `tests/cuda/bench_hc_fused.py` (cold in-graph bench, GATE line). Offline estimate
7.5-9.5 us vs 15.6 (0.48-0.61: borderline on the 50% gate), +1.0-1.4% at 1 stream. Not measured yet.

## 0530 — rank 0's HTTP threads off the engine's cores, and the RoCE trace from a running server (`GLM53_TF_CPU_PIN=http`, `GLM53_TF_ROCE_TRACE_DUMP`; `cpupin.py`, `rocedump.py`, `roce.py`)

Why (docs/THEORY-2.md §1.4, §4 item 6): each rank waits 1.3-1.5 ms a round (one stream) and 2.2-3.0 ms (four) inside
all-gathers for the other; at four streams rank 0, the HTTP node, is late on 62% of exchanges. Hypothesis: rank 0's
host threads (HTTP, streaming) share cores (and LPDDR5x) with the engine's launch thread and RoCE proxy. 0370's
`CPU_PIN=auto` dedicates cores to every role (W10: +0.4%, not adopted); this moves only the HTTP threads.

- `GLM53_TF_CPU_PIN=http`: `cpupin.plan` mode `http`; `http_thread()` (rank 0's request threads, named `tf-http`) moves
  its thread to the http cpus (default: allowed minus fast cores = the A725s on GB10; else the lower half);
  `classify` goes by thread name; nothing else is pinned unless `engine=<cpus>|fast` (then the main thread, the round
  loop and every other thread there, by the 30 s sweep). Without batching `serving_request` runs the decoding request
  thread on the engine's cpus (or all allowed) and puts it back. `http=` and `engine=` must not overlap and must be
  allowed. 0370's other modes are unchanged.
- `GLM53_TF_ROCE_TRACE_DUMP=<prefix>` (with `GLM53_TF_ROCE_TRACE=N`, 0460's ring): `rocedump.tick` (from `cpupin.tick`,
  once a round on both ranks) appends `{"t", "rank", "completed", "rows"}` to `<prefix>-r<rank>.jsonl` whenever
  `GLM53_TF_ROCE_TRACE_EVERY` exchanges (default 16 x the ring) completed since the last dump; `roce.COMMS` lists the
  process's `RoceComm`s so it finds the runtime. Three failures turn it off; NCCL or no ring: nothing. The two ranks'
  rows pair by `seq`: `results/THEORY2-SESSION/rocetrace.py` gives transport (the shorter wait), skew and the waiting
  beyond transport per rank, and the item 6 gate against a control.

Same bits: affinity and a host read of the trace ring only. Tests: `tests/test_http_pin.py` (plans on a fake GB10
sysfs, errors, classify by name, real affinity calls on this host's threads, the dump on a fake runtime).

**W19 (2026-09-30, image b10): adopted** (`GLM53_TF_CPU_PIN=http`). Boot line `http: HTTP threads on 0-4,10-14`;
during a streamed reply the request's HTTP thread ran on cpu 1 and the round loop stayed on 0-19. GPU idle changed
-0.05 / +0.05 / +0.08 ms a token (prose / code / 4 streams): within noise, kept because it costs nothing and every gate
passed with it (docs/RESULTS.md "W19").

## 0540 — warm replay and multi-arrival TTFT (`GLM53_TF_SNAPSHOT_BEFORE_END`, `GLM53_TF_EMIT_FIRST`; `pfgrid.py`, `decode.py`, `engine.py`, `batch.py`)

Why (RigMark W13, `docs/REPLAY-TTFT.md`): the immediate replays resumed only 0 / 16,384 / 49,152 of 8,192 / 32,768 /
65,536 tokens (6,304 tok/s at 64K vs vLLM's 11,364), because a prompt ending on the 64-grid kept its only snapshot at
n and a resume needs a strict prefix; and C4's per-stream TTFT was 1.91 s (vLLM 0.81) because a round's short prompts
prefill one after another and 0370's `emit` held each first token until the round's verify launch.

- `pfgrid.before_end()` (`GLM53_TF_SNAPSHOT_BEFORE_END`, default 1) and `pfgrid.plan(before=True)`: S = (n - 1) // G *
  G for every n (= the old S off the grid), with 0085's tail exception; exact plans (grid 0) cut at the last 64-multiple
  before n and snapshot there (none for a fresh prompt of <= 64 tokens). `decode._prefill` applies it when
  `e.snap_before == len(prompt)` (set by `GlmEngine._run` for the prompt and by `Batcher._piece` with the whole
  prompt's length, so only the piece that ends at n uses it; consumed once) and leaves the snapshot in `e.fast_snap`
  (exact prefills too); `_run` / `_piece` store and remember it instead of the snapshot at n. The load checks the knob
  on both ranks (it moves a cut, hence a chunk's collectives).
- `Batcher.emit_first` (`GLM53_TF_EMIT_FIRST`, default 1): `_piece` flushes the held tokens right after emitting a
  prompt's first token (only this round's earlier first tokens are held there).

Same bits: a cut and a snapshot on the 64-grid are what off-grid prompts, marks and piece bounds already get
(resumed == fresh, 0085); emitting earlier is host scheduling on rank 0. Memory: the snapshot moves, none is added
(<= 2 a slot, one a save, the store under budget). Cost: one more chunk (~0.2 s) on a cold prompt that ends on the
grid. Estimates: replay 8K / 32K / 64K ~23k-33k / 94k-131k / 187k-262k tok/s; cold -4 / -1 / -0.5%; C4 TTFT median
~1.2 s. Multi-slot prefill (several prompts in one fast chunk) is exact in principle but needs segment-aware lean /
0084 / 0320 loops and GPU validation: analysed in REPLAY-TTFT.md (est. C4 TTFT ~0.6 s), not implemented. Also in this
change: `GLM53_TF_REASONING_FIELDS=reasoning` recommended for prod.env (the vLLM kit sends `reasoning` only; not
edited), and `test_vision_patches.py`'s bf16 bound 2% -> 8% (W14). Offline only (no GPU run yet): GPU plan in
REPLAY-TTFT.md.

## 0550 — memory safety: the prefill selection's scratch, allocator trims, page cache at admission (`GLM53_TF_SELECT_SCRATCH`, `GLM53_TF_ALLOC_TRIM_GB`, `GLM53_TF_ADMIT_MEM`, `GLM53_TF_DROP_OWN_CACHE`; `memsafe.py`, `sparse.py`, `decode.py`, `batch.py`, `sessions.py`, `sessdisk.py`)

Why (W15 §3, `docs/MEMORY-SAFETY.md`): during a lone ~300k prefill MemAvailable fell by 3-4.5 GiB (6.39 / 6.42 GiB
after the stress + MMLU, under the 8 GiB floor) and came back when the prefill ended; and while a 36 GB file was
copied on the head node the batcher admitted C4 requests one at a time because `cudaMemGetInfo` free (= MemFree on GB10)
counts reclaimable page cache as used.

- Cause of the dip: `sparse.select_pools_blocked` (0065) allocates each DSA call's int32 key block `[n, npc]`,
  ~512 B a prompt position at 512-row sub-blocks (153 MiB at 314k, 256 MiB = `GLM53_TF_SELECT_MB` from 524k). The
  size grows every sub-block; torch's caching allocator cannot reuse the previous (smaller) freed segment, takes a new
  one every ~4,096 tokens and keeps the old: reserved memory grows quadratically (`memsafe.keys_growth`: 5.78 GiB
  bound at 314k, measured 4.5; 4.04 vs 3.0 in B7b; 16 GiB at 524k-1M) until `torch.cuda.graph`'s `empty_cache` (the
  MTP head's long-context graph for a new pool bucket, captured with the first draft after the prompt) releases it.
  Nothing else in a lone prefill scales with the prompt beyond MiB (inventory in MEMORY-SAFETY.md §2).
- `sparse.reserve_scratch` / `reserve_for_prefill` / `_keys_block` (`GLM53_TF_SELECT_SCRATCH`, default `grow`): the key
  blocks are views of one int32 buffer per (device, stream), grown on the host before a prefill
  (`decode._prefill`, for positions begin .. n with the DSA call rows) to the largest block rounded up to 16 MiB (<=
  256 MiB; old buffer dropped, device synced, cache emptied: nothing left behind; <= 16 growths a process). Inside a
  capture, or when a caller did not reserve, the old allocation / an in-place growth (`SCRATCH_STATS`). `max`:
  `SELECT_MB` at the first use; `off`: 0065.
- `memsafe.Trimmer` (`GLM53_TF_ALLOC_TRIM_GB`, default 2; `GLM53_TF_ALLOC_TRIM_S`, 10): before a batch prefill piece
  (both ranks, each for itself), `empty_cache` when the allocator holds more than 2 GiB unused, at most every 10 s; a
  trim that frees < 256 MiB (live graph pools) raises the trigger above what stayed.
- Admission (`Batcher._mem_ok`, rank 0) and the store (`SessionStore._can_grow`) use `memsafe.view`: CUDA free +
  allocator cache + the page cache credit MemAvailable - MemFree - Dirty - Writeback - max(Mapped,
  `GLM53_TF_ADMIT_CACHE_KEEP_GB` = 2) (`GLM53_TF_ADMIT_MEM=available`, default; `free` = the old measure). Beside
  running requests a request also needs `min(GLM53_TF_ADMIT_FREE_FLOOR_GB, ADMIT_GB)` (1 GiB) immediately free, and
  `ADMIT_GB` + the scratch its prompt still grows by. `memsafe.AdmitLog` prints `admission waits for memory` (numbers,
  once an episode and every 30 s) and `admission resumed after N s`; `counts["mem_deferred"]`, `stats["mem_waits"]`.
  The load-time slot rule is unchanged. Why counting clean page cache is safe on GB10 (device memory is kernel pages;
  reclaim hands clean cache to the driver's allocation; every "MemFree" observation so far was a check of the
  reported figure), the margins, and the probe that verifies it: MEMORY-SAFETY.md §4.
- `sessdisk._drop_cache` (`GLM53_TF_DROP_OWN_CACHE`, default 1): entry headers and buffered-fallback writes are
  `fdatasync`ed and `POSIX_FADV_DONTNEED`ed (the tier is otherwise O_DIRECT).
- A boot line (rank 0): `memory safety (patches/0550): selection scratch grow (key blocks up to 256 MiB), admission
  available (...), allocator trim over 2 GiB unused`.

Same bits: a key block view has the shape, strides and 512-byte-aligned base `torch.empty` gives, and `_scores_rows`
writes every element of it before `torch.topk` / `_gather_sel` read it; blocks run in stream order; no chunk, piece,
row block or grid changes. A trim changes addresses only (every graph capture already empties the cache); admission
changes scheduling only (batched == alone). Expected (per rank): the needle's minimum at 314k from 6.4 to ~10.7 GiB
(start 10.9 less a 160 MiB scratch), a lone ~1M prompt ~10.6 GiB (before: exhaustion), ~900k beside three busy slots
>= ~8.1 GiB (likely 9-10; levers `GLM53_TF_SELECT_MB=128`, `ALLOC_TRIM_GB=1`), the stress minimum up; C4 under a copy
admitted together. Offline only (CPU tests; no GPU run): the GPU plan (unit tests, the page cache probe
`bench/pagecache_probe.py`, gates with reply sha 8794a3463259cc2f, stress + needle 314k, ~900k with 3 busy slots, C4
during a 36 GB copy) is MEMORY-SAFETY.md §5.

**W18 (2026-09-30, image b9):** loads A (off) / T (trim off) / S (scratch only) after W17's heavy warm-up all
prefilled at 1,600-1,613 tok/s; every slow prefill in W17 had `GLM53_TF_ALLOC_TRIM_GB=2`, so the trim is the cause.
**W19 (image b10): the scratch adopted** (`GLM53_TF_SELECT_SCRATCH=grow`, `GLM53_TF_ALLOC_TRIM_GB=0`,
`GLM53_TF_ADMIT_MEM=free`): 314k needle minimum 6.58 / 6.31 -> 8.68 / 8.51 GiB and its lone prefill +1.6%; the 4 x 250k
stress minimum 7.75 / 7.61 -> 8.34 / 8.09 GiB together with the NCCL change. 0550 v2 (trims only as planned rounds at
idle or under pressure, one `empty_cache` after the boot warm-up) is proposed in MEMORY-SAFETY.md §6.

## 0560 — multi-slot prefill: a round's pieces in one forward (`GLM53_TF_MULTI_PREFILL`, `_ROWS`, `_WAIT_MS`; `mpf.py`, `batch.py`, `reqlog.py`)

Why (RigMark W13 / W15, `docs/MULTI-PREFILL.md`, `docs/REPLAY-TTFT.md` §2): a short prompt's prefill is weight-read
bound (a 130-row chunk reads ~97% of the routed experts, ~74 GB a rank), and a round prefilled its short prompts one
after another (0200's `BATCH_SHORT`): C4 paid four expert reads, per-stream TTFT 1.9 s with thinking on (0540's
`EMIT_FIRST` helps only when the first token is text: 0.93 s median); vLLM 0.8 s.

- `mpf.compute`: the lean chunk (0082) over the members' rows, stacked (member i at `[off_i, off_i + R_i)`) and cut into
  sub-blocks from each member's own start (`mpf.blocks`: never two members in one, a member's sub-blocks are its lone
  ones). Per sub-block the attention half as alone with the member's State, position and conv carry (`Seg.tail`), the
  FFN half's per-sub-block work as alone, the router / top-k / grouping / routed experts once over all rows, the head
  per member's last row. Loops: `_plain` (0082), `_MultiChunk` (0084's pipeline; `attn_pre` per member, final sums per
  row), `_MultiSplit` (0320, when a chunk of T rows would split). `lean.commit` per member with its carry.
- `mpf.run` (both ranks): `decode._prefill` cut at its forwards: each member's setup (reset / restore, pending MTP
  rows, `pfgrid.plan` with marks and 0540's rule, room and pages; 0550's scratch reservation when present), then the
  members' chunks in lockstep (one group forward per chunk index), and per member after each: last row, MTP absorb,
  DFlash2 taps, commit, the end snapshot. `Batcher._piece` then runs per member with `mpf.take`'s result in place of
  `decode.prefill` (one changed line), so saves, remembered snapshots, stats, the first token's emit, the Stepper are
  the lone code. A member's own error before the forward (ValueError) fails that request only.
- `mpf.groups` (both ranks, `_execute`): the round's pieces in order, packed by equal `knobs.HEADER` values within
  `GLM53_TF_MULTI_PREFILL_ROWS` (default the lean rows); only fast lean prefills without image rows, profile / dump
  off, not under FAST_EXPERTS auto / once with min rows. The last member leaves its group when 0335's solo rule could
  move its bound (it may be alone at its turn) and no other request holds a slot.
- `mpf.more_pieces` (rank 0 plan): with the fair share allowing a piece, more waiting prompts' pieces, fewest tokens
  left first, within the row budget. `mpf.coalesce` (rank 0, `GLM53_TF_MULTI_PREFILL_WAIT_MS`, default 10): idle and
  fewer waiting than slots, the plan waits until 10 ms after the oldest arrival.
- Load: both ranks' `[on, rows]` checked; boot line; request log `multi` (members of the request's group); counts
  `multi_groups`, `multi_members`.

Same bits: every call that reads per-sequence state or positions is the member's lone call on the same rows; every
other call is per-row glue, or a fast kernel 0085 already proved row-independent (the chunking-independence of fast
prefill rests on it); no new kernel. Memory: 2.5 MB of carries a member. Estimates (§6 there): C4 group ~0.55-0.6 s,
first tokens ~0.6-0.7 s (thinking off) / ~0.7-0.8 s (on), C2 ~0.5 / ~0.6 s; bursts of 4-8 short prompts 2-2.8x,
4 x 1,000 tokens ~1.7x prefill tok/s; lone requests unchanged (<= 10 ms wait when idle). Limits: attention / shared /
dense weights are still read once a member sub-block (~15 ms a member; sharing them is the follow-up). Offline only:
the GPU plan in MULTI-PREFILL.md §9.

## 0570 — the dense size switch (`GLM53_TF_DEC_QMM_MAXMB`, `_TABLE`, `_EXCLUDE`; `decode_stream.py`)

Why (THEORY-2 item 7, RESULTS W16 §1, `results/W16/probes-head/cold.log`): 0440's `q4_stream.cu` lost -3.9% end to
end (W12) because it is 9-18% slower cold on the large dense shapes that carry a round's bytes, while it is 1.1-1.5x
faster cold on the small, latency-bound ones. W16's cold bench: every shape up to index q_b (3.54 MB) wins in both cold
modes; the gate's "candidate 3.5" is 3.54 MB rounded.

- `GLM53_TF_DEC_QMM_MAXMB=X` (MiB, so 3.5 = 3.67 MB takes index q_b and leaves shared gate/up's 4.5 MiB out;
  `GLM53_TF_DEC_QMM_MAX_MB` reads the same): 0440's dense hook, turned on by the size knob alone, sends a 4-bit matmul
  of <= 64 rows to `q4_stream.cu` only if `Q4.nbytes()` <= X MiB (cached on the matrix); `qmm.matmul` runs `_qmm` +
  `_reduce` otherwise. 0440's Triton guard stays (3.7.x only).
- `SMALL_PLACE` (`GLM53_TF_DEC_QMM_TABLE=1`): per switched shape and PDL setting, W16's best cold placement (groups a
  stage, stages, whole-tile or split items; the best geomean over 1-16 rows in flush AND rotate with no window below
  ~0.95x) and a row cap: index k <= 4 rows (0.76 / 0.72x at 16 rows), DSA kv_b only with PDL (0.93-0.98x at 1-2 rows
  without; it has no decode call anyway). Unmeasured shapes (the (1,16,8) 1024 x 4096 grid) get 0440's CFG / auto.
- `GLM53_TF_DEC_QMM_EXCLUDE="NxK,..."` keeps shapes on `_qmm`; the NVMe compat hash already skips `GLM53_TF_DEC_QMM*`.

Same bits: no new kernel; `q4_stream.cu` gives `_qmm`'s bits for every shape and row count (0440's emulator, W12's
GPU bitwise), so a choice by (shape, rows) cannot change a row. Estimate (DECODE-KERNELS-2.md §4, W11 calls a round x
in-situ µs x W16 cold speed-ups): 1 stream -0.15 to -0.23 ms a round (+0.3-0.4%; up to -0.37 ms with
`GLM53_TF_DEC_PDL=1`), 4 streams -0.17 to -0.30 ms (+0.15-0.25%). Offline results: `tests/test_decode_size_switch.py`
(16 passed: knobs, the Triton guard, which of 15 per-rank shapes switch at 3.5, row caps, the table off, exclude, the
dispatch with / without PDL, the compat hash); 0440's `test_decode_kernels_emulator.py` (61) and
`test_theory2_probes.py` (19) pass with it (updated for the new CFG keys and the extra cold shape). GPU plan:
DECODE-KERNELS-2.md §7 (bitwise on 15 shapes x 19 row counts, `bench_decode_cold.py` on the switched shapes + 1024 x
4096, then loads).

**W19 (2026-09-30, image b10): not adopted** (`GLM53_TF_DEC_QMM_MAXMB` unset). Bitwise equal to `_qmm` per shape and the
cold gate passed (geomean 1.04-1.70x per switched shape), but the traces showed the switched shapes at 0.44 ms a token
in situ where `_qmm` + `_reduce` took 0.26 (dense decode +3-5%); the same config without it ran +1.2% at 1 stream and
+0.7% at 4 streams (docs/RESULTS.md "W19" §3).

## 0580 — the routed experts' load path, same bits (`GLM53_TF_DEC_EXPERT_LOADS`, `_CFG`, `_PDL`; `exl3_ld.cu`, `exl3_ld.cpp`, `expert_loads.py`, `exl3_mm.py`, `forward.py`, `sessdisk.py`)

Why (THEORY-2 item 8, RESULTS W16 §1, `results/W16/probes-head/littles.log`): the routed experts are 27.8 of a 53.3 ms
1-stream round (W11) and stream at 203 GB/s in the 1-stream windows (U 8-22) against a 233-237 plain read. `littles.cu`
answers how many bytes must be in flight on GB10: **`ld.global.nc.v4` into registers, 4 KB an SM -> 232.7 GB/s**
(cp.async 6 KB, TMA bulk 8-16 KB). It also ran 0440 E1's exact ring shape alone at 234 GB/s: E1's ~158 was its
persistent walk (per-item fences, tickets, barriers and fused epilogues, all CTAs at once), not the memory system.
grouped_kernel holds 1 KB a warp only at the top of each k step (0 during the decode / mma) and starts every CTA behind
three dependent round trips; short launches (3.6 waves at U 8) are ramp and tail.

- `ld_kernel<NT, PD, LD, PROBE>` replaces `ext.grouped` in `exl3_mm.routed` (not `grouped_loop`) when
  `GLM53_TF_DEC_EXPERT_LOADS=1` (`exl3_mm.LOADS`, installed by `expert_loads.configure`). Kept: grouped_kernel's grid,
  work item, W = 4 warps and K splits (so each warp's k range), the mma chain per accumulator from +0.0 on
  `decode_tile` (verbatim) / `load_pair`'s values, `red` and the warp-order sum, the Z rows and the three early exits.
- Changed: ucount / uids / member codes / the tile's first code in one round trip (`ld.global.cg`); the first PD k
  steps of trellis words issued before the rows reach shared memory; a PD-deep register ring a warp: `nc` = NT / 4
  `ld.global.nc.L1::no_allocate.v4` a lane a step, stored to a 1-KB warp staging area (in the warp's own slice of
  `red`, written with partials only after the last step) and read back as word `lane` of each tile; `w32` = 32-bit
  loads straight into the ring; `cpa` = a cp.async ring of PD + 1 steps in the same slice. The A fragments ride PD
  steps ahead. Settings (`_CFG`, gate/up / down): nc 8,1 / 8,2 / 8,4 / 4,2 / 4,4, w32 8,2 / 4,4, cpa 8,2 / 4,4; probes
  at 8,2: nc 1 (no decode) / 2 (no mma) / 3 (load path alone), w32 3, cpa 3.
- `GLM53_TF_DEC_EXPERT_LOADS_PDL=1` (sm_90+): programmatic dependent launch; the prologue (grouping reads + the first
  weights) before `griddepcontrol.wait`, X / Xd reads and Z writes after it. Safe because the launch before each grouped
  launch (rot_in / gateup_epilogue) is a plain launch and the grouping is written before `routed`.
- The NVMe compat hash skips `GLM53_TF_DEC_EXPERT_LOADS*` (same bits). No new buffers.

Same bits: every mma sees the same operands in the same chain position, the warps are summed in the same order, the
same rows are stored; only when the bytes arrive changes. So Z, Xd and Y are exl3.cu's for every row count, setting
and window (drafted == serial, batched == alone, resumed == fresh; ranks may differ).

**Offline results** (no GPU; stack = b9's list + 0570 + 0580; nvcc 13.4, torch 2.14 CPU, Triton 3.8 for the host tests):
- `tests/test_decode_loads_emulator.py` (58 passed, ~4.5 min): a lane-level port of `ld_kernel` (v4 lane mapping,
  staging store / read-back, register ring and refill order, cp.async ring slots with copies landing at random before
  their wait, NaN shared memory aliased with `red`, A fragments ahead, prologue exits, warp-order reduction) == the
  matrix-level grouped_kernel model with the order-sensitive mma model: Z bit for bit (never-stored elements too) for
  all 9 settings on gate/up-like (1-4 K splits) and down-like layers, 1-16-row windows, 4-slot mixes, 20 / 40-row
  skewed windows with 2-3 member tiles, stale slots past ucount; routed Xd / Y; the real K ranges (4,096 / 4 splits,
  1,024 / 1 split) at PD 2 / 4; rows alone == in the window; 8 negative controls caught (staging read lane-major, slot
  refilled before read x2, cp.async slot / wait off by one, wrong-k A fragments, reversed warp order, rotated k order);
  host: knobs, `fits` at the real shapes, the hook dispatch (<= 16 rows -> ld_kernel, 17+ -> grouped_loop), the compat
  hash, the bench gate logic.
- `tests/test_decode_loads_compile.py` (5 passed): 14 instantiations for sm_121, 0 spills; default nc,8,2 163 registers
  (grouped_kernel<8,4>: 109), 32,832 B static shared memory (= grouped_kernel's), 3 CTAs an SM, 24 KB an SM in flight
  (nominal); nt 4 settings 103-126 registers, 4 CTAs an SM; the PTX has the v4 / u32 no_allocate loads, cp.async +
  wait_group, the f16 mma, no atomics; PDL prologue clean (only `ld.global.cg` and trellis loads before the wait); the
  shared helpers are exl3.cu's text; SASS: the 128-bit refills sit between the HMMAs of the step being computed.
- `exl3_ld.cu` compiles host + device against torch's headers; `exl3_ld.cpp` passes a syntax check.
- Estimate (DECODE-KERNELS-2.md §5, W11 round): 203 -> 220 / 225 / 230 GB/s in situ = -2.0 / -2.6 / -3.1 ms a
  1-stream prose round (+4.0 / +5.1 / +6.2%); code +3.2-5.8%; 4 streams (already 219 GB/s) +0.4-3.1%.
- GPU plan: DECODE-KERNELS-2.md §7 -- bitwise + PDL race (`tests/cuda/test_decode_loads_patches.py`), the engine
  suites with the knobs, `bench_decode_kernels.py --loads` with its two GATE lines (probe 3 >= 220 GB/s; one config
  >= 1.05x grouped_kernel cold on U 8-22), loads C / E / EQ (/ EQP), one nsys capture, the adoption gates.

**W19 (2026-09-30, image b10): adopted** (`GLM53_TF_DEC_EXPERT_LOADS=1`, `GLM53_TF_DEC_EXPERT_LOADS_CFG=nc,8,1`).
Bitwise tests 20/20; probe 3 (load path alone) 226-231 GB/s (gate >= 220); the kernel against `grouped_kernel` 1.03-1.19x
flush, 0.98-1.09x rotate (the strict 1.05x on every window missed, so the in-situ trace decided): routed-expert decode
-4.1..-4.6% a token at 1 stream, -2.9..-3.1% at 4 streams, the bulk of W19's +3.3% / +3.0% decode (docs/RESULTS.md
"W19").

## 0590 — routed experts in prefill, v2: one pipelined kernel, same bits (`GLM53_TF_FAST_EXPERTS=fat2`; `exl3_fat2.cu`, `exl3_fat2.cpp`, `exl3_mm.py`)

Why (ROOFLINE gap 1, KINDLING-AUDIT adoption #2): at 4,096-8,192-row chunks `fat` costs about the SUM of its MMA and
DRAM roofs (1.36x the DRAM floor at 2,048 / 4,096 rows, 1.76x at 8,192; W8). Its pipeline drains at every
(expert, 64-member pass, 128-column block) item, and each trellis tile is decoded per 64 members. The audited fused
prefill MoE kernel (NVFP4) reaches ~84% of the bf16 roof with 256 threads, a 4-stage `cp.async` ring, expert-sorted
128-row tiles from a device list and the gather inside the A loads. 0590 re-implements that structure for EXL3 x fp16 with
fat's arithmetic (no code taken: that repository has no licence).

- **Kernel** (`fat2::expert_kernel<KIND, NG, NSA, PROBE>`, `__maxnreg__(168)`):
  - 8 warps, one CTA an SM, a persistent grid;
  - items (expert, 128-member pass, column block; down: a pair of column blocks) from fast2's device plan, claimed by
    ticket NSA stages before the current item ends, into a second item buffer;
  - the ring (NSA stages of 2 k tiles: 8 KB of swizzled member rows, gathered by pair index, zero-filled only to the
    last member's 16-row group; 4 KB of trellis words) keeps loading the next item's first stages under the current
    item's tail and epilogue;
  - a warp holds 2 column tiles x 128 members, so each decoded tile feeds 16 mma;
  - the epilogue is fat's code in its own 33 KB buffer;
  - block-uniform barriers only (no mbarriers, nothing to hang on).
- **Configurations** (`GLM53_TF_FAT2_CFG`): 0 = 128 members / 4 stages (82.0 KB), 1 = 64 / 4 (65.5 KB), 2 = 128 / 3
  (70.0 KB). Probes: 1 = no decode, 2 = no mma. The same bits every way.
- **Dispatch.** Inside the fat family (`fat_experts = 1`). Needs gate / up sharing their sign vector and K >= 128
  (fat otherwise). Its own extension, built at first use for the device's architecture. Nothing picks by the call's
  rows, so 0560's groups stay allowed.
- **Combine fusion (adoption #6) is not in 0590.** Inside the expert kernel it needs cross-CTA ordering to keep the
  bits (a row's 8 slots come from 8 items), and it saves at most 1/8 of Y's write. The audit's exact form
  (`_combine_s` into 0084's send buffer) is independent of this kernel: its own patch.

Same bits: every stored element is fat's chain of m16n8k16 mma (the same `decode_tile` A fragment, the same fp16 B
values at the same k, ascending k tiles from +0.0, all of K in one warp, the same tile position), then fat's
epilogue. Only which CTA / warp / item holds it, and when bytes arrive, change. So fat2 == fat == fast2 == v1:
row-independent (0085), deterministic, snapshots shared.

**Offline results** (no GPU; stack = b9's list + 0590, which also applies on b9 + 0570 + 0580 and in the full sorted
list; nvcc 13.4, torch 2.14 CPU, cuobjdump 13.1):

- `tests/test_fat2_emulator.py` (49 passed, ~8.5 min): lane-level ports of fat and fat2 (plan, ticket / stride under
  random CTA interleavings, NaN-initialised ring with copies landing at random before their `wait_group` and between
  warps, swizzle, `ldmatrix.x4`, `decode_tile`, fat2's claim-ahead and cross-item stages, epilogue rounds) with an
  order-sensitive mma model. Results:
  - fat == the matrix-level reference;
  - fat2 == the reference with every output written once (3 configurations, 1-7 CTAs, 1-4,096-row windows with 1-43
    passes, odd block counts, stale grouping entries, the real per-rank shapes);
  - fat2 == fat under a position-sensitive mma model;
  - rows alone == rows in the window;
  - 12 planted mutations caught, and the neutral one (no zero-fill past the last member) keeps the bits.
- `tests/test_fat2_compile.py` (21 passed, 17 s):
  - every instantiation at 168 registers, 0 spills, <= 82.0 KB, one CTA an SM, leaving 22,528 registers and 16-32 KB
    for an overlap-stream CTA (fat's gate/up leaves ~3.5 K registers);
  - 11 helpers verbatim, and the epilogue lines are fat's;
  - **PTX:** every stored value's expression tree == fat's (opcodes with rounding modifiers, fma contractions,
    constants, operand order), and every mma's A / B operand trees == fat's; the atomics are the ticket and the member
    count only;
  - **SASS:** the epilogue's FP instruction census at 2 rounds == fat's exactly, and at 4 rounds == 2 + fat's per-round
    delta; no local memory;
  - 3 planted source changes caught.
- `tests/test_fat2_bench.py` (3 passed): the bench's GATE line logic and the DRAM floor numbers.
- `exl3_fat2.cu` compiles host + device against torch's headers; `exl3_fat2.cpp` passes a syntax check.
- `tests/cuda/test_fat2_patches.py`, host part: 6 passed (env switch, dispatch with fake extensions,
  `mpf.kernels_ok`).
- Regression host suites (tc / auto / once / multi-prefill / lean / solo): green, except 3 solo-piece tests that fail
  identically on b9 without 0590.

**Expected** (EXPERT-PREFILL-V2.md section 6; not timed):

- 2,048 rows: 10.2-11.4 ms against fat's 13.18 (0.77-0.87x). The DRAM floor is 9.71 ms at 235 GB/s.
- 4,096 rows: 12.3-13.8 ms against 15.96 (floor 11.71).
- 8,192 rows: 18-21 ms against 27.70.
- **The 0.75x gate at 2,048 / 4,096 needs ~98% of the DRAM peak**: expect FAIL on the letter there. A floor-based gate
  is proposed for the user.
- End to end: +4-7% prefill at kernel value, +2-4% after the W5 / W8 transfer discount.

GPU plan: EXPERT-PREFILL-V2.md section 7:

- `test_fat2_patches.py` under `timeout`;
- `bench_experts.py 2048 4096 8192 --fat2 --contend` (bits, isolated / contended / makespan, the GATE line);
- the engine tests;
- load A (b10, knob off) vs load B (`fat2`) with ab.py 24.5k / 98k x2, exact / batchexact, 4-stream decode, one nsys
  capture;
- adopt at >= +2% prefill, same sha, decode not lower.

**W19 (2026-09-30, image b10): not adopted** (stays `fat`). Bitwise 22/22 and every bench cell same bits, but fat2 is slower
than fat: isolated 1.16x / 1.19x, contended 1.12x / 1.20x, makespan 1.07x / 1.09x at 2,048 / 4,096 rows (1.76x /
1.88x the DRAM floor); both the floor gate and the letter gate fail (docs/RESULTS.md "W19" §1).

## 0600 — upstream ports: client disconnect, request hardening, USR1, image URL hardening, small bundle, /health totals (`GLM53_TF_DISCONNECT`, `GLM53_TF_VISION_FETCH_*`, `GLM53_TF_VISION_PREP_*`; `server/cancellation.py`, `cuda/server.py`, `cuda/health.py`, `cli.py`, `batch.py`, `app.py`, `vision_prep.py`, `image_fetch.py`, `engine.py`)

Why: docs/UPSTREAM-050-AUDIT.md's ranked port list. Items 1-4, 6 and 9 are host-side fixes and small features that
keep every reply's bits; this patch takes them in one go. Full description, behaviour changes, knobs and the GPU check
plan: **docs/UPSTREAM-PORTS.md**. Upstream TensorFold is MIT-licensed; the ported code says where it came from
(`server/cancellation.py`, `image_fetch.py`, `closed_json`, `parse_numbers`, the `/health` totals) and NOTICE lists it.

- **Client disconnect** (upstream 24afe5e; ours on the batcher). `server/cancellation.py` (upstream's
  `socket_cancellation`, `poll` instead of `select`); the handler passes `cancelled=` to `App.run` for every
  completion (`GLM53_TF_DISCONNECT=0`: off). `App.run`: a request whose client left while it waited raises
  `RequestCancelled` without an engine call; the callback checks the socket every call, never raises into the engine
  (a failure is raised after `generate`), and a stopped request writes nothing more. `Batcher._collect`: the HTTP
  thread polls `on_tokens.cancelled` every `GLM53_TF_DISCONNECT_POLL_MS` (250) while no token arrives and between
  tokens, and on a departed client or a callback exception sets `job.cancel` and wakes the loop: `_plan` drops a
  queued job, and cancels an admitted one (prefilling or decoding) through the plan rank 1 receives, 0370's riders
  included, as every cancel since 0120. 0300 logs it `cancelled`. The lone path (`BATCH=1`) refuses requests whose
  client left while they waited for the lock and no longer lets a callback failure desync rank 1; a running lone
  request still decodes on unheard (0160's design). Half-closed client sockets count as gone (upstream's rule).
- **Request hardening** (2bc35c4). `do_POST` catches `check` / `tokenize` exceptions: template refusals are 400 "the
  chat template rejected the request: ...", anything else a 400 with its message and a redacted traceback in the log;
  `App.check` refuses non-object `chat_template_kwargs` and malformed `temperature` / `top_p` / `top_k` / `seed` /
  `max_tokens` / `max_completion_tokens` (booleans, non-finite, non-integral) before a stream's headers;
  `sampling_for` reads the same values unclamped (the same `Sampling` as before for every valid request); non-UTF-8
  bodies are "not JSON"; 500s and stream error events are logged. `Problem.status` (503 for capacity).
- **`kill -USR1`** (a586f3d): `faulthandler.register` before `cmd_serve` hands over to `_serve_cuda` (both ranks).
- **Image URL hardening** (391713e, 05cbb54, b4042e8): `image_fetch.py` (upstream's `images_http.py`) replaces
  0500's `urllib` fetch: HTTPS on 443, no credentials / fragments / control characters, every resolved address public
  (plus NAT64 / IPv4-compatible refused), the connection pinned to the checked address with TLS verified, each redirect
  re-checked (<= 3), identity encoding, a declared JPEG / PNG / WebP / GIF, `GLM53_TF_VISION_MAX_BYTES`, per-download
  and per-request deadlines; `vision_prep.Host.request` holds one of `GLM53_TF_VISION_PREP_SLOTS` (16) with
  `_PREP_WAITERS` (128) waiting up to `_PREP_WAIT_S` (60 s), then `VisionBusy` = HTTP 503. `GLM53_TF_VISION_FETCH_HTTP`
  / `_FETCH_PRIVATE` for local testing only. Messages carry no URL.
- **Small bundle**: `return_token_ids` (e6be5ff: the counted reply ids in the `tensorfold` block); the token that
  completes a stop string found per token (88407d1's rule on 0160's visible-answer scope), so `completion_tokens` is
  the same drafted or serial; `closed_json` (c30a1a9 / #87) in 0002's GLM parser for `array` / `object`-typed
  arguments that only stop short of their closing brackets.
- **`/health`** (51b098d, c4bf25f): upstream's `backend`, `busy`, `requests_running`, `requests_total`,
  `prompt_tokens_total`, `completion_tokens_total` (live), `prefill_seconds_total`, `decode_seconds_total`,
  `cached_tokens_total`, `rounds_total`, `drafted_total`, `accepted_total`, `streams`, `context_length` beside 0150's
  fields (unchanged, as is `/metrics`); the batch and lone stats now carry `drafted` / `accepted`.

Tests (host): `tests/test_upstream_ports.py` (169) and `tests/cuda/test_disconnect_patches.py` (8: queued /
prefilling / decoding / callback failure on both ranks in lockstep, 0370 riders off and on); `tests/test_vision_prep.py`
and `tests/test_vision_server.py` use the local-testing knobs for their `http://127.0.0.1` image servers. **Offline
only: GPU check plan in docs/UPSTREAM-PORTS.md §4** (exact 10/10, batchexact, reply sha `8794a3463259cc2f`, canary, a
non-streamed long request freeing its slot, errors / USR1 smoke, an image URL over https, `/health`).

**W19 (2026-09-30, image b10): in production (on by default).** Host suites 169 + 8 passed in the image. A
non-streamed 32k request whose client closed freed its slot 0.18 s after the close; four at once 0.26-0.31 s; a queued
~100k request whose client left never ran; bad `temperature` / `chat_template_kwargs` and a non-UTF-8 body: 400s;
`USR1` stacks on both ranks; an https image URL equals the same image as `data:`; `http://`, loopback and metadata
addresses refused; `/health` token totals grow during a reply (docs/RESULTS.md "W19" §2).

## 0610 — structured output, exact under drafting and batching (`GLM53_TF_GRAMMAR`, `_THREADS`; `grammar.py`, `batch.py`, `decode.py`, `engine.py`, `app.py`, `resident.py`)

Why (docs/UPSTREAM-050-AUDIT.md 2.1): a request's `response_format` was silently ignored. Upstream TensorFold 0.5.0
(98fb8f9 / 7ae9da2 / 1da4b7e) enforces it with xgrammar on GLM's one-request loops only; production runs the batcher.

- `grammar.py` (new; request fields, compile rules, the cut with rollback, the bitmask unpacking and `advance` ported
  from upstream's `engine/grammar.py`, MIT, attributed): `request_spec` (400s: malformed, not compiling, an output
  format beside a required call); two xgrammar compilers built at load on both ranks from `tokenizer.json` through
  `tokenizers` (text view: every added token excluded, upstream lets them spell text inside JSON; tools view: the six
  tool markup tokens kept; logits width from `config.json`, checked against the head); tool calls through xgrammar's
  `glm_4_7` structural tag (`glm_xml` arguments: our 0002 parser's reading); the think state read from the prompt (the
  last of `<think>` / `</think>`). `Constraint.cut` (a chain window: rejected / stop-token / added-token drafts dropped
  with the rest) before the forward; `Constraint.fill` (accept along the kept chain, fill, roll back) on
  `GLM53_TF_GRAMMAR_THREADS` worker threads while the forward runs (the fill releases the GIL); `apply` (-inf on the
  rank's columns only, a new tensor); `advance`; `pack` / `split` (the grammar in front of the header: a negative first
  int no header has).
- Batcher: `_admit` a fresh constraint per admission (both ranks); `_piece` the last piece's first token under
  `first_token` (also 0560's group finish); `_verify` `_grammar_cut` before the forward, `_grammar_masks` right after
  its launch (the bits staged on the device behind the forward from pinned memory: no host wait), `sample_multi(..., masks=)`, `advance` after `accept`, `_grammar_fail` ends a failing slot alone (both
  ranks at the same point: the same walk); `_header` / `follow` carry the grammar; `Batcher.generate` reads
  `request.grammar_bound` and raises a GrammarError itself. `sample_multi`: masks, and a masked round draws on the
  host. `resident.eligible`: `"grammar"`.
- Lone path: `decode.serial_decode` / `mtp_decode` / `dflash_decode` / `auto_decode` through `grammar.lone(e)` (cut,
  masks after the launch, masked host draw, advance); `sample_rows(host=True)`; `engine.generate` packs the grammar in
  front of the header, `follow` compiles it and ends a failing reply as rank 0 does; `engine._run`'s prefill under
  `first_token`.
- App: `GlmApp.check` answers the 400s (knob on), `run` hands `(spec, compiled)` to the engine; knob off: every field
  ignored as before.
- Dockerfile: `pip install -c torch==<base image's> xgrammar==0.2.8` (apache-tvm-ffi, pydantic, transformers).

Same bits for everything else: no forward, kernel, graph, collective, drafter or snapshot changes; an unconstrained
request creates no grammar object and its header is unchanged. Constrained requests: drafted == serial, batched ==
alone, 2 ranks == 1, resumed == fresh (docs/STRUCTURED-OUTPUT.md §3). Cost (§5): *measured* on a laptop core with GLM's
154,880-column vocabulary, a row's fill is ~13-80 us (p50; up to ~0.3 ms inside JSON strings), the cut ~1.5-2 us a
draft, a 4 x 16-row round 1.2-2.8 ms on 4 threads (3-7 ms inline), all during a 60-120 ms forward: est. ~0.1-0.3% of a
constrained round (the GPU mask), 0 for other rounds. Not done: masking the drafters' own logits (acceptance on JSON),
`thinking_budget`. Tests: `tests/cuda/test_grammar_patches.py` (129 CPU), `bench/grammar_bench.py`; the GPU gates:
`bench/structured.py` (docs/STRUCTURED-OUTPUT.md §8).

**W19 (2026-09-30, image b10): adopted** (`GLM53_TF_GRAMMAR=1`). The image installs xgrammar 0.2.8 with `--no-deps`
(docker/Dockerfile). Unconstrained replies' hashes equal the knob-off reference; schemas 32/32 (8 schemas x greedy /
sampled x thinking on / off; drafted == `"draft": false` == 4 concurrent), tools 6/6; a 50-object `response_format`
task 71.9 vs 72.2 tok/s unconstrained (-0.4%). The exposed mask wait was ~0.7 ms a round, above the 0.3 ms estimate
(follow-up: `GLM53_TF_GRAMMAR_THREADS=8`) (docs/RESULTS.md "W19" §2).

## 0620 — tool-calling fixes for multi-step chains (`GLM53_TF_TOOL_FIXES`; `toolfix.py`, `cuda/server.py`, `app.py`, `grammar.py`)

Why: an independent tester's tool-eval-bench (multi-step chains 75%) and spark-bench (agentic 82.6) runs (issue #6);
the audit of our request path is **docs/TOOL-CALLING.md**, which also has the benchmark notes and the test plan.

- `cuda/server.py`: `tool_fixes()` (the knob, validated at load), `schema_types` / `typed_value` (the `args` reading,
  used by 0002's `_glm_call` when on), `App.reply_hook` (identity; called with a finished chat reply that had tools,
  before `finish_reason` is chosen, with the reply's last token).
- `glm5_next/cuda/toolfix.py` (new): `ToolFixes.prepare` (the request, in place and idempotent: `GlmApp.check`,
  `tokenize` and `run` all call it before rendering; a bad named `tool_choice` is a 400), `ToolFixes.reply` (the
  reply: `thinkcalls`, `parallel_tool_calls`, the reasoning memory), `ReasoningMemory` (LRU by entries and
  characters, thread-safe), `trailing_calls`, `signature`.
- `app.py`: builds `ToolFixes` at load (prints one `[tensorfold] tool calling (patches/0620)` line when on) and
  overrides `reply_hook`. `grammar.py`: `_tool_spec` treats auto tool requests as strict ones with `grammar`.

Exactness: host only. Off (default), every request renders, parses and answers as before; on, requests without
tools are untouched (no GPU gate changes: exact, batchexact, reply sha, glmbench hashes are tool-free), and tool
requests render a different prompt only where the fix says so. Streaming: `thinkcalls` has already streamed the
calls' text as reasoning before it returns them as calls; `parallel_tool_calls: false` drops the later calls.
Tests (host): `tests/test_tool_fixes.py` (46; the real chat template and xgrammar with `$GLM53_TF_TOKENIZER_DIR`).

**W20 (2026-09-30, image b11): adopted** (`GLM53_TF_TOOL_FIXES=all`; `grammar` stays off). Host tests 46 (42 without
a tokenizer dir). With it on, W20's gates passed: exact, batchexact, reply sha, all 13 glmbench hashes equal to b10's, structured
schemas / tools. tool-eval-bench against production (thinking off, temperature 0, one run): final score 90, category C
(multi-step chains) 8/8; spark-bench was stopped early and is not a score (docs/RESULTS.md "W20").

## Tests

| File | What it checks |
| --- | --- |
| `tests/cuda/test_patches.py` | on TensorFold's synthetic GLM checkpoint (one GPU playing rank 0 of two): non-expert weights really are 4-bit; q4 drafted == serial (sampled and greedy); 64- vs 512-row prefill give the same reply and the same resume; the MSE clip is never worse than min/max; vectorized sparse selection equals the per-row loop at the dense/sparse boundary; a 3,000-token prompt (past the 2,051 dense limit) gives the same reply with 64- and 512-row chunks and with drafting |
| `tests/cuda/test_longctx_patches.py` | 0050: bounded and device selection == unbounded (random, tied, signed-zero scores); the device selection replays in a graph across its bucket; long-context graph steps (rows 1-8, both parities, MTP 1-8) == upstream eager logits at 2.5k-7k tokens (contexts 4,096 / 8,192); replies with graphs == `GLM53_TF_LONGCTX_GRAPHS=0`, serial and drafted (8 policies), greedy and sampled, including a prompt crossing 2,051 during decode; resume after a long reply == fresh prefill |
| `tests/cuda/test_latent_patches.py` | 0060: latent vs expanded kernels and an fp32 reference on the real shapes (32 heads, 256, latent 512; 4-bit and BF16 kv_b); window rows == serial rows bit for bit (dense, sparse, graph-style chunk counts); scratch independent of capacity; with `GLM53_TF_LATENT_KV=1`: latent caches and bytes a token, drafted == serial (11 policies), resumed == fresh, 64 vs 512-row prefill same state, BF16 kv_b (EXL3), a 3,000-token prompt past the dense limit (chunks, drafting, resume), batched == alone; latent vs expanded prefill within bf16 tolerance |
| `tests/cuda/test_1m_patches.py` | 0065: row-block scores == `_scores` bits; blocked selection == sorted (random, ties, -0.0, NaN, dense-limit rows, bucket boundary, up to 1,024 rows, 1M pool counts, small blocks); bounded selection scratch; index ring pool keys == full; DFlash2 ring == linear (wrapped, rewound); replies with 0065 == all-off (serial, drafted, 64/512 rows); resume behind both rings == fresh; memory accounting at a 1M capacity |
| `tests/cuda/test_knob_patches.py` | 0090: host-only validation (unknown / load-only / out-of-range / typed knobs, batch mode), header block round trip, the server's 400 message; on the synthetic EXL3 checkpoint: 14 knob sets == serial (greedy, sampled, drafted and serial, fresh and resumed across chunk sizes), knobs revert after each request (also a failing one) and are echoed, invalid ones fail before rank 1 hears, lookup/depth/auto_fdrafts reach the header and bound the windows, a replaying follower runs rank 0's knobs and windows with other defaults of its own, `longctx_graphs` 0 vs 1 per request past 2,051 tokens |
| `tests/cuda/test_batch2_patches.py` | 0120 (replaces 0030's `test_batch_patches.py`). Host only: request headers and round plans round trip; piece bounds (exact, fast grid); the prefill time share; piece order, admission order, which background request steps aside; the title heuristic; the round cost model; knobs in batch mode. GPU (synthetic EXL3 checkpoint): per-slot states / DFlash2 contexts / graphs; batched rows == lone rows (logits, MTP rows, taps; 2-4 sequences; graphs first round and replay, eager); 2 and 4 requests == serial for every policy (MTP, DFlash2, auto, o / om / of, thresholds, serial, lookup), sampled and greedy, graphs and eager; uneven lengths and queued requests; per-slot prefix reuse; a 700-token prompt in pieces while another decodes; fast and lean prefill admissions == the lone fast / lean engine (grid snapshots only, resume, exact requests never resume fast snapshots); per-request knobs per sequence == lone with the same knobs; latent and expanded KV past 2,051 tokens (pieces across the limit, a crossing window, bucket-keyed graphs, resume); a client gone mid-reply; a background request stepping aside; concurrent streaming callers; slots trimmed to memory; the MLX checkpoint; a follower replaying rank 0's plans makes the same decisions |
| `tests/cuda/test_fastk_patches.py` | 0081: fast_qmm vs qmm on the per-rank GLM shapes, M = 256/1024/1500 (close, deterministic, bitwise), BF16, strided rows, timings; fast_kda vs `kda.chain` (32 heads, aligned/unaligned pos, tf32 tolerance), the fp32 variant vs the fp32 reference, determinism, 64-aligned split == one call, `save_replay` feeds `kda.replay`, timing |
| `tests/test_gpu_sampler_interpreter.py` | 0450 in Triton's CPU interpreter (with `tests/gpuround_interp.py`: a correctly rounded fma, launches serialized): the libm port == numpy bit for bit (uniforms, their Gumbel, near 1, every exp range); the device draw == `choose_rows` / the greedy lexsort / `_probability` (1-2 ranks, top_k 0-120, top_p, temperatures, ties); numpy's pairwise sum; the samplers with the knob == the host path; the load check |
| `tests/libm_a64.py` | 0450, optional (`GLM53_TF_LIBM_A64`): the libm port == Ubuntu's aarch64 `libm.so.6` executed in unicorn |
| `tests/test_gpu_round_compile.py` | 0450 on sm_121 without a GPU: the sampler's float64 is exactly as written (no contraction; fma counts), the glue and the DFlash2 chain compile; with nvcc, `kda.cu`'s slot kernels have the reference kernels' float ops per opcode and fit 64 registers |
| `tests/cuda/test_gpu_round_patches.py` | 0450: knob; the glue kernels == the host rules (accept, stage, `mtp_next`, `f_depth`, the confidence cuts, conv shift, backlogs). GPU: `replay_slots` / `chain_slots` bitwise, the load check on GB10, the synthetic checkpoint with each part == serial, a lone request + resume |
| `tests/cuda/test_gpu_round_resident.py` | 0450: the real `Batcher` / `Stepper` / `Resident` on a hostile fake model (CPU, interpreter): replies and committed states == serial, the MTP cache and DFlash2 context position by position, peek and lag, `kda`, context-mode drains, two ranks in threads, no device read while launching |
| `tests/test_fastk_interpreter.py` | 0081 in Triton's CPU interpreter (no GPU): the same checks on small shapes, tf32 operand rounding emulated |
| `tests/cuda/test_fp8_kv_patches.py` | 0220 (GPU): the FP8 writer == torch's e4m3 conversion (every finite bf16 value <= 448, random rows over 11 decades, zeros / -0.0), rows independent, error <= half an e4m3 step; FP8 dense / sparse latent attention (BM 16 / 32) == the bf16 kernels on the dequantized rows bit for bit; absorb -> attend -> expand vs the fp32 expanded reference (prints bf16 vs fp8 error); window rows == serial rows; the knob's validation. Engine with fp8 KV: layout and bytes a token, drafted == serial (11 policies), resumed == fresh, 64 / 512-row chunks same state, fp8 vs bf16 KV prefill within tolerance (quality hook), snapshots never cross formats, 3,000 tokens past the dense limit (chunks, drafting, 0050 graphs, resume), batched == alone (2 slots; 4 slots with 0200's on-set), sessions resume FP8 rows (keys carry the format), 0180 batch sessions == alone |
| `tests/test_fp8kv_interpreter.py` | 0220 in Triton's CPU interpreter: writer == torch conversion (exhaustive bf16), rows independent, fp8 attention == bf16 on dequantized rows (dense, sparse, BM 16 / 32); the knob, session key tag, snapshot format check |
| `tests/cuda/test_expert_once_patches.py` | 0260. Host only: the env switch (`once` is a fat-family mode, `ONCE_PAIR`, `ONCE_MIN_ROWS`), `routed`'s dispatch with fake extensions, and a Python model of the `once` kernel: the plan's units cover every (expert, pass, column block) once (pair on / off, odd column-block counts, stride and ticket walks) with ceil(P / 2) decodes a tile; every warp reads fat's member-row bytes and trellis words (both unit kinds, all four tilings); the fragment exchange hands over the right tile and never overwrites a slot before it is read; the shared-memory budget; the decode-count table. GPU: once == fast2 == fat bit for bit (Xd, Y; shared / own input; pair and ticket on / off; 300-8192 rows uniform / skewed; a 384-wide shape with 3 column blocks); row subsets and permutations; repeatable; the timing probes launch. Engine: committed state once == fast2 (lean and not, C = 256 / 1024 / 8192), drafted == serial, resumed == fresh, snapshots shared with fast2 both ways. `tests/cuda/bench_experts.py`: v1 / fast2 / fat / once (+ split-only, no-decode and no-mma probes) per kernel at 1024-8192 rows, uniform / skewed, with bit checks |
| `tests/cuda/test_fast_experts_auto_patches.py` | 0270. Host only: `auto` is a fat-family mode, `GLM53_TF_FAST2_ROWS` parsed / refused, `family` / `set_family` and the knob's 0-2 range through the header, `routed`'s dispatch with fake extensions (fast2 inside the window on the shared input, fat / once outside, both rotations when the sign vectors differ). GPU: auto == fast2 == fat bit for bit (64-4,160 rows, uniform / skewed, shared / own input, windows around the row count); row subsets of a fat chunk run by fast2 give the same rows; engine: committed state auto == fast2 with a threshold inside the chunk sizes (C = 256 / 1024 / 8192), drafted == serial, resumed == fresh, auto / fast2 / fat resume each other's snapshots |
| `tests/cuda/test_batch_buckets_patches.py` | 0280. Host only: `bucket_mask` / `bucket_rows` / `route_sources`. GPU: settings; bucketed rounds give every real row its lone bits (logits, final normed rows, DFlash2 taps) eager / captured / replayed, two window sets of one bucket share one graph, padded rows pick their source row's experts; 4 requests with buckets (alone and with every 0200 knob) == serial, sampled and greedy, mixed policies; buckets without the tie still exact |
| `tests/cuda/test_fastpf_patches.py` | 0080/0091. Host only: the grid, the switches, the knob, and the rule through the engine's real prefill / snapshot / resume code on a fake model whose chunks depend on their start and length (random conversations, serial and MTP-drafted, fast and exact). GPU: the fused expert kernels vs a float64 reference and the row-invariant path (synthetic and real shapes, multi-pass experts), deterministic and row-independent; fast engine: deterministic state, drafted == serial (7 policies), resumed == fresh (within the last chunk, across grid points, after a reply, chains, on the grid, short prompts), only grid snapshots kept, fast and exact snapshots never mix, off-grid resumes refused; the same with adversarial chunk-dependent stand-ins for 0081 (and the control that they do depend on the chunks); fast vs exact prefill within bf16 tolerance; a 3,000-token prompt on the latent cache past the dense limit |
| `tests/cuda/test_lean_patches.py` | 0082. Host only: the lean orchestration on a hash model (every kernel exact integer arithmetic with the real row / position / KDA-state / cache dependencies) equals 0080's fast chunk bit for bit (lengths 1-256 around 64/128-row sub-blocks, two positions, commit included, EXL3 and MLX MoE), the routed experts run once a layer, a planted carry bug is caught; a last sub-block of 1 / 17 / 63 rows runs the fast matmuls like the whole chunk (with fast_qmm's < 64-row qmm fallback emulated; control without the fix); the fast rule with lean chunks that depend on their sub-blocks and a grid 4x the buffers (fake model, real prefill / snapshot / resume code); the lean set's bytes on the real shapes (linear, ~397 KiB a row). GPU: lean (64-row buffers, 256-row chunks) vs non-lean fast engine: same committed state (3-700 tokens), same replies (7 policies, greedy and sampled), drafted == serial, resumed == fresh (chains too), deterministic, knobs (prefill_rows up to the lean max, exact requests clamped), memory; 3,000 tokens on the latent cache past the dense limit |
| `tests/cuda/test_fp8_patches.py` | 0083/0092. Host only: the switch, the tag, the knob; the fast rule with FP8 and bf16 fast requests mixed in random conversations on a fake model whose fast chunks depend on the mode (real prefill / snapshot / resume code). Kernels (GPU on the real shapes at M = 1024 / 2048 / 8192; Triton's interpreter on small shapes without a GPU): `matmul_fp8` == its host model to fp32 summation order, vs fp32 within the e4m3 bound, deterministic, row-independent (slices, permutations, the first 1024 rows alone), strided rows, zero rows; "cvt" == "bits" weight encoding and every tile == the same bits; small-shape / short-call fallbacks; BF16 weights; tensor-core absorb / expand vs a float64 reference; 32- vs 16-query latent attention. GPU engine (every matmul in fp8): runs and differs from bf16 fast (control), deterministic, drafted == serial (7 policies), resumed == fresh (chains too), FP8 / bf16 / exact snapshots never mix, lean == non-lean within FP8, 3,000 tokens on the latent cache; quality bound vs bf16 fast; timing prints (fp8 vs bf16 per shape and M, tile sweep, absorb / expand) |
| `tests/cuda/test_overlap_patches.py` | 0084/0093. Host only: on 0082's hash model, every pipeline variant (direct, gather, slab, all; slabs of 64 and whole) equals the unpipelined lean chunk bit for bit (lengths 1-256, two positions, blocks 64/128, EXL3 and MLX MoE, fp32 gathers, one rank). A deferred exchange runs only when waited for. Planted bugs are caught: a post that skips its exchange, a missing drain, routing before the last attention post. The pipelined order is checked, and the collectives equal 0082's in size and order. Also: knob parsing, slab rows, the 0093 knob. GPU kernels: hc_post on bf16 partials (and row slices) == on the fp32 copy; bf16 combine == fp32 combine rounded; the fast matmul's bf16 store == fp32 store rounded (o_proj / down shapes, Q4 and BF16, 1-2048 rows); an hc timing print on real shapes. GPU engine (64-row buffers, 256-row chunks), variants switched on one engine: committed state == unpipelined (3-700 tokens), also with an exchange stand-in that sleeps on the comm stream; fp32 gathers; replies (5 policies, greedy and sampled); resumed == fresh; deterministic; the per-request knob; 3,000 tokens on the latent cache |
| `tests/cuda/test_cindep_patches.py` | 0085. Host only: `pfgrid` (tags, resumable, rows parsing, auto rows, plans: snapshot point, cut, tail rule, marks; with the grid at C exactly 0080's rule; 2,000 random plans); the engine's real `_run` / `_prefill` / snapshot / resume code on a fake whose fast rows depend on their whole 64-row block, the call's offset in it and the mode (not on C): 60-turn random conversations with a random C per request (auto included), FP8 / bf16 mixed, marks, serial and MTP-drafted, every reply and state == a fresh prefill with another C, follow-ups resume at the old prompt's last 64-point (or within the tail); the control (a C-dependent fake) is caught; misaligned resumes and fast KDA calls refused; `fastpf.chunk` keeps only the head on qmm. GPU kernels, bit for bit: `matmul_fast` (4-bit, BF16, bf16 / FP8 prefill, real shapes) on 1-300-row slices and permutations, head's 1-row call == qmm; `hc_partial`; `absorb_tc` / `expand_tc`; the fused experts (row subsets, permutations, skewed routing past a pass); `fast_kda` split anywhere on 64 == one call. GPU engine: committed state identical for C = 64 / 1024 / 4096 / 8192, non-lean and lean (256 sub-blocks), overlap on / off, bf16 and FP8, 3-1,000 tokens and 3,000 / 9,000 tokens on the latent cache with 32 index heads; resumed from the 64-grid snapshot with another C (64 / 1024 / auto) == fresh with a third (MTP, DFlash2, auto drafts; FP8; chains); drafted == serial with auto; the auto knob and `GLM53_TF_PREFILL_ROWS=auto` |
| `tests/cuda/test_mia_prefill_patches.py` | 0170. Host only: the knob in the header block, `kda_proj_bf16` refused per request, `fastpf.settings` / parsing / the memory estimate (98.25 MiB a layer), the copy's dispatch (fast chunks only, FP8 off), a Python model of the fat kernel's index arithmetic (swizzled stages read back what was written, conflict-free phases; the weight ring hands each warp fast2's words; the expert walk == fast2's binary search; every item claimed once). GPU kernels: fat == fast2 bit for bit (Xd, Y; shared and own inputs; 3 / 4 stages; ticket on / off; 300 / 1024 / 4160 rows, uniform and skewed); `rot_in1` == `rot_in`; row subsets and permutations; timings of fat vs fast2 and of the KDA copy vs 4-bit with a tile sweep (`-s`); the copy row-independent (1-63-row and unaligned slices, permutations) and within 5e-3 of the 4-bit path. GPU engine (gate / up sign vectors made equal): committed state fat == fast2 (lean and not, 3-1,000 tokens); with the KDA copies the state is the same for C = 64 / 1024 / 8192, fat or not; drafted == serial and resumed == fresh (fat, KDA copies); a fast2 request resumes a fat request's snapshot |
| `tests/cuda/test_session_patches.py` | 0110. Host only: page keys and counts (exact / fast / MTP), tail bytes; longest fitting prefix, common prefix, marks; shared pages with reference counts, LRU eviction within the budget, slabs reused and released, oversize entries skipped; rank 1 replaying rank 0's decisions (300 random saves) ends with the same store, divergence detected; the plan message. With torch on any device: the tensor store on fake caches whose rows are functions of their prefix (random forks, follow-ups, junk past every request, evictions): every restore gives the entry's rows and drafter window. GPU: sessions A B A C B A interleaved (shared system prompt, fork mark) == fresh prefill + serial, pages shared and not recopied; eviction under a tiny budget and an oversize entry stay exact; mixed drafter policies; 2,300-token system prompt past the dense limit on the latent cache, exact and fast; rank 1 following rank 0's messages (same restores, saves, evictions, replies; divergence refused) |
| `tests/cuda/test_batch_sessions_patches.py` | 0180. Host only: piece bounds for fast resumes between chunk-grid bounds, the piece grid, entry placement, admission memory, the switch. Torch on the CPU: the store bound to several slots (save on one slot, restore into another, per-slot live maps, eviction after a restore leaves the slot's copy, other layouts refused); the real `Batcher._plan` / `_execute` / `_admit` / `_piece` / `_finish` / `follow` with the real store on a hostile fake model, 4 sessions over a shared system prompt on 3 slots, exact and fast, roomy and tiny budgets: every reply and the slot's whole state at the end == fresh prefill + serial decode; a replaying follower ends with the same store; the 2026-09-28 fixes (a follower that says rank 0 replays save decisions by role; sessions admitted together leave a fork mark). GPU: off by default; 4 concurrent sessions on 3 / 4 slots == alone (mixed policies, sampled and greedy), fork mark, shared pages; tiny budget; fast prefill with resumes between chunk bounds; rank 1 replaying plans, session plans and save decisions |
| `tests/cuda/test_session_disk_patches.py` | 0250. Host only: settings, the compat hash (moves with bit-deciding knobs, image, model, layout, KV format; not with session / batch / path knobs), the disk index (shared pages, pinned pages, LRU, oversize, replay determinism), header size and chunk plan, the plan message with the disk entry and digest. Torch on the CPU: round trip bit for bit with live pages skipped; a corrupted page / entry chunk, truncated entry, missing page, failed write fail the read and drop the entry; both ranks fall back when one fails; restart indexing (debris removed, budget trimmed, another compat sees nothing, only entries both ranks hold); evict policy; in-place writes past the queue; the lone engine's real `_run` and 0180's real `Batcher` on hostile fake models with tiny RAM stores: every reply and state == fresh, resumes from disk, after a restart too, damaged files fall back cold; a replaying follower ends with the same RAM and disk indexes and files. GPU: evicted sessions resume from disk == fresh (sampled, greedy), restart + damaged files, rank 1 replay, batch slots from disk and after a restart, FP8 latent + fast prefill past the dense limit, `bench` on real shapes (`GLM53_TF_SESSION_DISK_BENCH=<dir>` times the NVMe) |
| `tests/cuda/test_glue_patches.py` | 0190. Host only: the knobs in the header, the MTP window's start rule, `group_sorted` == a Python model of `_group` (every cell, ids past the count untouched), the switch rules, the MTP absorb with a window on a fake state (zeroed rows / index keys / pools, `mtp_len`, drafted rows dropped), the TC tag and load-only refusal. GPU kernels: `group_sorted` == `_group` (64-8192 rows, uniform / skewed / local), `select` on == off, `_router_fused` == partials + sum (and row slices), `_combine_s` == copy + `_combine`, fused hc == separate kernels bit for bit (1-1000 rows, bf16 / fp32 partials, 1 / 2 ranks, modes 1-3, slab slices), fused tiles of 16 / 32 rows reported, 32- vs 16-query latent attention bit for bit (dense, sparse). GPU engine: state with `moe_glue` on == off (non-lean, lean, 3-1000 tokens), every same-bits knob on at C = 1024 / 8192 (pipelined or not) == all off at C = 64, `hc_fused` 1-3 in the pipeline, `attn_bm32` on the latent cache past the dense limit; MTP window: main state unchanged, head rows zero below lo, history-independent, replies == window off == serial (MTP, DFlash2, auto; greedy, sampled), resumed == fresh; knobs echoed and restored; `GLM53_TF_LATENT_TC`: C-independent, differs from the FMA path (control), drafted == serial, resumed == fresh |
| `tests/cuda/test_b12x_patches.py` | 0240. Host only: the knob in the header, tags (every mode x bits distinct, `is_fp8`, session grid), env parsing, effective bits, module switches. GPU kernels: `b12x_kda` vs `kda.chain` (slow / fast / mixed decays, aligned and short calls; no worse than `fast_kda`), bf16 policy report, deterministic, split at any multiple of 64 == one call, first k rows == a k-row call, `fastpf.kda_chain` dispatch and |lower| > 5.3 fallback; `b12x_mhc` partials vs float64, fused == unfused bit for bit (1-1000 rows, bf16 / fp32 partials, 1 / 2 ranks), slab slices / subsets / permutations, close to today's path (new streams identical); `b12x_attn` vs float64 and the chunked kernel, row subsets / permutations bitwise, FP8 == dequantized. GPU engine: committed state C- and pipeline-independent with bits 1-3 (and 4 / 7 on the latent cache past the dense limit), deterministic, differs from bits 0 (control) but close; drafted == serial and resumed == fresh (4 policies, greedy / sampled); snapshots never cross bits; the knob echoed and restored |
| `tests/test_b12x_interpreter.py` | 0240 in Triton's CPU interpreter (run it on its own: TRITON_INTERPRET must be set before triton is imported): KDA16 vs the serial chain (fp32 1e-7; tf32-rounded no worse than `fast_kda`), its float64 tile model, split == one call; mhc partials vs float64, fused == unfused; one-pass attention vs float64 and the chunked kernel, FP8 == dequantized. Fixes the interpreter's truncating bf16 casts and bf16 dots for the run |
| `tests/cuda/test_batch_parallel_patches.py` | 0200. Host only: `pick_pieces`, `pad_mask` / `padded`, `Sightings`, plans with several pieces, the row-cost floor in `verify_ms` / `RoundCosts`. Torch on the CPU (0180's hostile fake model, the real `_plan` / `_execute` / `_piece` / `_verify` / `follow`): short prompts prefill several a round and every reply and slot state == fresh prefill + serial decode, a follower replays the same rounds; a padding forward keeps replies and states exact (offsets, commit rows); `MtpChains` on a fake head == `decode.draft` alone (drafts, head-cache length, chained entries, optimizer calls; confidence and cost-depth stops, several rounds); `sample_drafts` == `sample_rows` row by row (greedy / sampled, with / without probability, 1 and 2 ranks). GPU: padded batched rows == lone rows (eager, capture on the 2nd sighting, replay); 4 requests with every knob == serial (mixed policies, sampled / greedy), parity-keyed graphs; short prompts share their admission round; one MTP head pass over 2-4 slots == each slot's own; batched MTP drafting == serial with per-slot drafting's keeps |
| `tests/cuda/test_adapt_patches.py`, `bench/draftsim.py` | 0340. Host only: `extra_ms` / `drafter_ms` vs `lookup.round_ms` and `RoundCosts.table`; `SlotChoice` (each arm once, best surplus, serial only when allowed / past the margin, probes, alone = the base choice, only shared rounds build the surplus, deterministic); env knobs; the simulator's `DrafterChoice` copy == decode's (torch) and the simulator deterministic. Torch on the CPU, the REAL `Stepper` / `_plan` / `_execute` / `_verify` on 0180's hostile fake model with fake MTP / DFlash2 drafters (true continuation or a wrong token): 0340 off / on / on with serial / arms forced at random per slot and round, requests arriving alone and together: every reply == serial; a follower decides the same arms and keeps. GPU: 4 requests (o / of / om) with 0340 + serial, chosen and forced arms, greedy and sampled == serial |
| `tests/test_roce_logic.py`, `tests/cuda/test_roce_patches.py`, `tests/cuda/bench_roce.py` | 0230. Host only: GID detection (fake sysfs, moved index, forced index), subnet pairing, knobs, collective `select` on two fake ranks (mismatch refused, marker / failed setup -> NCCL on both, `error`), `RoceComm` dispatch. GPU (no RDMA; a host thread plays the peer): the kernel for 1 B-256 KiB, unaligned, dtypes, both ranks, graph replays, two streams, timeout -> poison / diagnosis / marker; the engine through the kernel == `_TwoCopies` (windows 1-8, drafted == serial, resume). Two Sparks: `bench_roce.py` bench / fault / soak, `loopback` on one node |
| `tests/cuda/test_kv_pool_patches.py` | 0290. Torch on the CPU: the real `Batcher` (`_plan` / `_execute` / `_admit` / `_piece` / `_finish` / `follow`), `SessionStore` and 0250's `DiskTier` on 0180's hostile fake model with every slot's caches paged views of one pool: roomy and tight pools (admissions wait, idle slots spilled, tables fragmented), exact / fast prefill, 256 / 512-token pages, every reply and slot state == fresh prefill + serial decode on contiguous caches, null page clean every round, no page or reservation leaked; a request larger than the pool refused without blocking the queue; a follower replaying rank 0's plans (with spill lists) ends with the same stores and page tables; store copies between scrambled paged slots; NVMe restores into paged slots. GPU: compiled kernels paged == contiguous on the real shapes (bf16 / FP8, BM 16 / 32, b12x one-pass, indexer, three selection paths); a captured graph replayed after every page moved; the synthetic engine with the pool (fragmented table, pages 256 / 512) == without: committed rows and state past the dense limit, replies serial / drafted / greedy / sampled, resumed == fresh; 4 slots on a 1.5-slot pool == alone with waits and spills; disk sessions into pooled slots; bytes |
| `tests/test_kvpool_interpreter.py` | 0290 in Triton's CPU interpreter: `latent_write` (bf16 / FP8) across page bounds, dense / sparse / b12x one-pass latent attention, `index_update` (full and ring keys, paged pool keys), `select_tokens` / `select_tokens_dev` / `select_pools_blocked` on paged pool keys: all bit for bit == contiguous with scrambled tables and poisoned unmapped pages; the allocator, slot tables (ensure / truncate / quota / over-quota), `Paged` view rules, `pool_spills`, settings |
| `tests/test_roce_protocol_model.py`, `tests/test_roce_watch.py` (+ `tests/cpp/roce_watch_test.cpp`), `tests/test_bench_roce_stress.py` | 0350, host only: the executable model of 0230's protocol and loopback harness (W2's failure reproduced number for number, the fixed harness completes, invariants under random schedules with 1-3 ranks and 1-2 HCAs, two controls caught); `roce_watch.h` compiled with g++ (idle / not_seen / late / never / out of range / racing records) and `roce.classify`; `bench_roce.py stress` on fake runtimes (a stale slot is counted, eager and in graph replays) |
| `tests/test_b12x_onepass_interpreter.py`, `tests/test_b12x_onepass_compile.py`, `tests/cuda/test_b12x_attn_patches.py` | 0360: the one-pass kernel in the interpreter (rows == alone / subsets / orders / duplicates bitwise, FP8 == dequantized, paged, vs float64); compiled for sm_121 without a GPU (FP8 dots == bf16 dots, same loop compute, control: 0240's kWidth 4); host: the lookup tag carries the bits, follow-ups with bits resume and equal fresh on the fake model (control: 0240 never resumed), bits never cross, compat hash; GPU: real-shape FP8 == dequantized and row independence, engine bit 4 C-independent, drafted == serial, resumed == fresh with cached > 0 |
| `tests/test_kda_v2_compile.py`, `tests/test_kda_v2_interpreter.py`, `tests/cuda/test_kda_v2_patches.py`, `tests/cuda/bench_kda_v2.py` | 0400: compiled for sm_121 without a GPU (the prep helper's PTX == `_kda_prep`'s; every scan setting and the fused kernel's branches == the reference's float ops, dot chains, reduce layouts and PTX float counts; `maxnreg` only the directive); the interpreter with emulated mma chains and Triton's combine rewrite (every mode / setting == `fast_kda` bitwise at 1-8,192 rows, resume == fresh, aliasing, the ticket order; controls); GPU: bitwise == `fast_kda` on the real shape for every mode / setting, repeated and busy-GPU runs, resume == fresh, the engine entry; bench: fast_kda's three kernels vs every setting with a bitwise flag and the best bit-identical env lines |
| `tests/test_sparse_v2_compile.py`, `tests/test_sparse_v2_emulator.py` (+ `tests/sparse_v2_emu.py`), `tests/cuda/test_sparse_v2_patches.py`, `tests/cuda/bench_sparse_v2.py` | 0410: compiled for sm_121 without a GPU (v2's softmax layout == the reference's `#mma`, kWidth 2, the loop's float op sequence and chains == the one-pass kernel's, equal PTX float counts, cp.async zero-fill gathers, shared memory / no spills, dispatch and knob); v2's own Gluon source on a CPU emulator (cp.async ring, byte-level shared memory, barrier / hazard model) == the one-pass kernel in the interpreter bitwise (FP8 / bf16, paged, every setting, counts 0-2,051, subsets), controls (unfolded / chunked differ, missing wait / barriers caught); GPU: bitwise == one pass on the real shapes for every setting, paged, 1-2,048 rows, extreme logits, the knob; bench: chunked / one pass / every v2 setting, TF/s, same-bits flag, projected tok/s |
| `tests/cuda/test_deep_verify_patches.py`, `bench/lookupsim.py` | 0380 (the file re-runs itself with `GLM53_TF_MAX_DRAFT_ROWS=16` in a child process: the knob is read at import). Host: knobs, caps at 8 (upstream's) and 16, the two-line calibration table, cost depths past 7 only for confident drafts, 15-token lookup copies. Torch on the CPU (0180's hostile fake model): the real `Stepper` / `Batcher` (`auto` + lookup, `l15:3`, `of15`, `om15`, mixed; alone and shared) and the lone `auto_decode` with windows of 9-16 rows == serial, `depths` stats, a follower deciding the same windows. GPU: dense windows 1-16 == serial rows, long-context windows / MTP steps 1-16 through graphs == eager, lone and batched drafted replies with a 16-row DFlash2 block == serial. At 16 rows five older tests that assert the 7-draft cap fail by design (run them at the default) |
| `tests/test_mla_expand_interpreter.py`, `tests/test_mla_expand_compile.py`, `tests/cuda/test_mla_expand_patches.py`, `tests/cuda/bench_mla_expand.py` | 0390: in the interpreter with the GPU's dot (per-element fp32 FMA chain, exact fma emulation) and nearest-even bf16, v2 == v1 (Combine-folded source) == a numpy chain for 1-8,192 rows, rows == alone, every tile / BN; compiled for sm_121: v1's dot folded, v2's dots chained, FMA path only, no unfused fp32 op in the PTX, no spills; GPU: v2 == v1 bitwise (q4 / q4mse / bf16, 1-8,192 rows, 3 seeds incl. extreme magnitudes, windows, every tile, in a CUDA graph), control vs the tensor-core kernels; bench: v1 vs v2 us a call, tile sweep |
| `tests/cuda/test_draft_vocab_patches.py`, `tests/test_draft_vocab_interpreter.py`, `bench/draftvocab.py`, `bench/acceptpos.py` | 0420. Host only: knobs and refusals, the list file, per-rank rows (order, fill, multiple of 64, equal counts), the shipped ranking, the settings the ranks compare, the fallback window. Torch on the CPU, two ranks in two threads over a fake all-gather: `take_rows` / `attach`; the target's sampling identical with and without the list (`sample_rows`, `sample_multi`); trimmed draft draws equal on both ranks, listed, greedy coupling; `sample_drafts` == per row; `MtpChains` full iff any slot falls back; drafted == serial through the real `mtp_decode` / `draft` / `absorb` / `Engine.mtp` / `Engine.sample` (greedy / sampled; list with and without fallback, untrimmed). Compile: `qmm._qmm` at the trimmed shapes for sm_121. Interpreter: the trimmed head through `qmm.matmul` == the full head's listed columns bit for bit. GPU (synthetic checkpoint): trimmed MTP logits == listed columns (graphs incl. `mtp_full`), drafted == serial == untrimmed, lone and 4 batched, arms `mtp` / `all`, fallback off / forced |
| `tests/test_draft_dump.py`, `tests/test_drafter_ref.py`, `tests/test_drafter_train.py`, `tests/cuda/test_drafter_ref_patches.py`, `train/`, `bench/gendata.py` | 0430 and the drafter pipeline (docs/DRAFTER-TRAINING.md). CPU (`TF_SRC=<patched tree>/src`): the dump's records, merge and prefix linking on a fake two-rank engine; the reference's `quantize4` / clip search == the engine's source, EXL3 dequantization == the engine's float64 decoder, FP8 rows == 0220's reference, the training-time chain == step-by-step drafting, DSA selection == dense below the limit, shards, export round trip, DFlash2 block independence and window; dump -> both trainers -> exports -> evaluator end to end. GPU (mount `train/` next to `tests/`): reference MTP / DFlash2 == engine on the synthetic checkpoint (rank 0's shard), `GLM53_TF_MTP_WEIGHTS` loads an export, replies identical with the dump on / off and decode rows line up; env-gated real-model acceptance parity |
| `tests/test_decode_kernels_emulator.py` (+ `tests/decode_kernels_emu.py`), `tests/test_decode_kernels_compile.py`, `tests/cuda/test_decode_stream_patches.py`, `tests/cuda/bench_decode_kernels.py` | 0440. CPU (`PYTHONPATH=<tree>/src`): lane-level ports of `exl3_stream.cu` / `q4_stream.cu` (rings, swizzles, cp.async groups landing at random before their wait, NaN-initialised shared memory, random CTA interleavings, tickets) against matrix-level models of exl3.cu's routed path and `_qmm` + `_reduce` with an order-sensitive mma model and exact fma: bit for bit for 1-16-row windows, 4-slot mixes, skewed windows with extra member tiles, every configuration and grid, real K, the real-shape walk; rows alone == in the window; 12 negative controls; knobs, dispatch, the Triton guard, the compat hash. Compile (no GPU): every instantiation for sm_121 (spills, registers, shared memory, CTAs an SM), the PDL prologues, the verbatim helpers, `_qmm`'s TTIR / PTX sequence per shape and bucket (fully fused under 3.7.x; the unfused set under 3.8 printed). GPU: on == off bitwise (real / synthetic shapes, every configuration, PDL, graphs, CTA caps, rows alone, probes), the PDL race test (late writer + busy stream), engines (windows 1-16 graphs / eager, drafted == serial greedy / sampled, resumed == fresh); bench: old vs new µs and GB/s beside a roof probe, every configuration, with a same-bits flag |
| `tests/test_prefetch_comm.py`, `tests/test_prefetch_comm_compile.py`, `tests/test_roce_protocol_model.py` (extended), `tests/cuda/test_prefetch_comm_patches.py`, `tests/cuda/bench_roce.py --trace` | 0460. CPU (`PYTHONPATH=<tree>/src`, Triton importable): L2PF knobs, plans on fake weights (inside weights, aligned, within budget, read order; chunk heads for one-wave 4-bit launches), expert plans, MTP keys, **the port** (stubbed kernels: `compute_multi` / `mtp_multi` reach the lone forward's a / o / e / f sites in order and join), 0040 compatibility, the one-HCA rule, RoCE knobs and `agreed()`, `trace_summary`; the protocol model with one-HCA ops and inline posts under random schedules (+ two timeout controls). sm_121 (nvcc): the prefetch kernels hold only loads / prefetches, `roce.cu` 64 registers with the lean forms; bindings syntax (`TF_CXX_INC`). GPU: every L2PF mode == off (windows 1-8, graphs / eager, replies for every policy, the Batcher with graph knobs), no weight written; the RoCE kernel with `STRIPE_KB` / `LEAN` over the loop peer, trace stamps ordered |
| `tests/kvpool_ptx.py` | 0290, a tool (not pytest; no GPU): compiles the touched kernels for sm_121 from a tree; `--against` the output of a tree without 0290 checks the pool-off PTX is identical, and every paged variant compiles |
| `tests/test_api_context.py`, `tests/test_prompt_tokens.py` (updated), `tests/test_serve_ops.py` (+4) | 0490, host only: `/v1/models`' `max_model_len` / `context_length`; the `context_length_exceeded` 400 (code, param, wording checked against opencode's and Hermes Agent's patterns) for chat, streamed chat and completions; engine refusals in a stream as `invalid_request_error`; `/tokenize`; token-ID prompts (exact `prompt_tokens`, bad ids refused); the boot line; serve.sh's missing-config message, context line, warnings and per-head refusal |
| `tests/test_vision_prep.py`, `tests/test_vision.py`, `tests/test_vision_interpreter.py`, `tests/test_vision_server.py` | 0500, CPU (89): the processor == transformers' `Glm5NextImageProcessor` bit for bit, `smart_resize` / token counts == transformers'; fetch / decode limits, part shapes, prompt placement and refusals, virtual ids a function of the pixels, the registry; the tower == transformers' `Glm5NextVisionModel` (tiny config, fp32 1e-5, bf16), loading, `Encoder.table`, `Table.fixup`, the two-rank `exchange` in threads; `glue.embed` substitution in the interpreter (BF16 / 4-bit, 4 / 1 copies, never during capture); the HTTP flow (off by default, 400s, `usage`, `/tokenize`), two images vs the same image through `_resume`, `Batcher._resume`, `SessionIndex`, `DiskIndex`, page and entry keys; the lookup cut; the prompt == `Glm5NextProcessor` on the real tokenizer (`GLM53_TF_TOKENIZER_DIR`, `GLM53_TF_VISION_TEMPLATE`) |
| `tests/cuda/test_vision_patches.py` | 0500 on a GPU: image rows set to real tokens' embedding rows == those tokens (serial / drafted, greedy / sampled, prefill rows 1 / 16 / default / fast, mid-prompt and last chunk), resumed == fresh, another image never resumes past the text, the compiled embed and graph capture; with `GLM53_TF_MODEL`: the checkpoint's tower on a screenshot (1,036 rows, deterministic, bf16 vs fp32, latency, peak memory) |
| `tests/test_request_log.py` | 0300, host only: settings (off by default), role ids, the prefix compare; a line's fields (sizes, cache source slot / RAM / disk / none, timings, effort, finish), no text in the file (prompt, reply, reasoning, store description), head / system hashes, conversation keys, lcp any / same / other; requests in flight count as earlier partners, later arrivals do not; the window; errors and cancels logged, a failing log never raises (reported once); rotation keeps whole lines; concurrent writers; `GlmApp.run` writes one line a request (also on failure) and nothing when off; `scripts/traffic-report.py` numbers on a synthetic 12k-system-prompt log |
| `tests/cuda/test_replay_ttft_patches.py` | 0540 (32). Host only: `pfgrid.plan(before=True)` (on-grid snapshot at n - 64 with the cut, RigMark's last pieces, the resend's plan, marks deduplicated; off-grid plans == 0085's for every length / resume / chunk size tried, every snapshot a strict prefix within tail + G; exact plans; the knob). Torch on the CPU, `test_batch_sessions_patches`' hostile fake model: resume at end - grid == fresh (first token, state, reply; exact with cut chunks, exact, fast); the real batcher + store: a resent prompt resumes at (n - 1) // 64 * 64 (0 without the rule), replies == fresh, no more snapshots kept; 4 short prompts together: one round, fewest first, each first token out before the next piece (held without `EMIT_FIRST`), batched == alone. Also: `test_decode_overlap_patches.py`'s emit test with / without `EMIT_FIRST` |
| `tests/cuda/test_multi_prefill_patches.py` | 0560 (83 host / CPU + 4 GPU). Host only: knob / settings, `more_pieces`, `pack`, sub-blocks per member, the solo guard. Torch on the CPU, test_lean_patches' hash model: a group forward == each member's lone lean chunk bit for bit (last logits, final rows, taps, streams, KDA state / conv, caches) for 2-5 members of 1-600 rows at positions 0-192, 0082's loop and every 0084 variant, EXL3 / MLX, any member order; experts once a group; controls (shared carry, group positions, straddling sub-blocks) caught; the driver's real hooks; two real processes over gloo with 0320's split == lone on both ranks. The real `Batcher` on the hostile fake model: C4 in one group (fewest first, each first token out in turn, replies and final states == alone, 1-3 lockstep chunks); resends resume at n - 64 from the group's snapshots; 4 interleaved sessions with groups (roomy / tiny store) == fresh; a follower forms the same groups; other knobs / image rows / a member's error / decoders in the round; more pieces past BATCH_SHORT; the last member and the solo rule; the idle wait. GPU (synthetic EXL3): 4 prompts grouped == alone, lean and pipelined, greedy and sampled, next turns resumed |
| `tests/test_memory_safety.py`, `tests/cuda/test_memory_safety_patches.py` | 0550. Host (20; the selection in Triton's interpreter): the size model == `sparse`'s rules and == the blocks `select_pools_blocked` allocates; the growth bound vs W15 (0.74 / 0.78), quadratic, 16 GiB at 1M; the scratch <= 256 MiB at every piece to 1M, admission's reservation; scratch (`grow` / `max`, junk-filled, grown, too small) == `off` == sorted on random / tied / -0.0 / NaN scores; a reserved scratch allocates no key block; meminfo, the page cache credit and margins; W15's copy admits with `available`, waits with `free` (== the old rule on 2,000 cases), the floor, the log; `Batcher._mem_ok` / `SessionStore._can_grow` on fakes; the trimmer (trigger, interval, hysteresis); `drop_file_cache`. GPU: scratch == off == sorted at 70k-1M with 262,148 pools; `memory_reserved` grows by >= half the bound over a 100k -> 230k sweep without the scratch and <= 160 MiB with it; a capture empties the cache; the trim frees; engine replies with scratch + a trim before every chunk == off, serial and drafted |
| `tests/cuda/test_prefix_share_patches.py` | 0310. Host only: settings, role-token lookup, boundary points, `marks(extra=)` on the exact / fast grid, the `prefix` entry cap replayed by rank 1. Torch on the CPU (0180's hostile fake model, real `Batcher` / `SessionStore`, 0290's pooled slots): today's miss vs the switch's resume, bursts with / without the wait, a cancelled partner, random agent traffic (exact / fast, roomy / tiny, keep 4 / 1, pool roomy / tight) all == fresh + serial, a replaying follower. GPU (synthetic checkpoint): a new session resumes at the system end (exact / fast, greedy / sampled) with replies == alone, a burst of 4 waits and resumes, a follower replays |
| `tests/cuda/test_expert_tc_patches.py` | 0330. Host only (numpy; the patched source via `TF_SRC`): the helpers copied from `exl3_fast.cu` are the same text, and the epilogue arithmetic lines are fat's; the env switch (tc is a fat-family mode; CFG / TICKET / CTAS parsed and refused); `routed`'s dispatch (tc on the shared input, fat when the sign vectors differ, fast2 for knob 0 and auto's window). A Python model of `exl3_tc.cu` for its 6 configurations and 1 / 3 / 8 / 32 column blocks: items cover every (expert, pass, block group) once (stride and ticket walks); every B value an mma receives is the right member row and k through an emulated ldmatrix.x4 and the m16n8k16 fragment layout (zero-filled rows past the count); every A word is fat's trellis word; the epilogue writes each output of each present block once from the right accumulator; shared-memory and register budgets. The mbarrier protocol simulated with random interleavings and copy completion times: no deadlock, no stage read before it lands or refilled while read, item info never overwritten in use; the control (one arrival too few on "empty") is caught. GPU: tc == fast2 == fat bit for bit (Xd, Y; cfg 0-3; ticket on / off; a CTA cap; 64-8,192 rows, uniform / skewed; a 384-wide shape with 3 blocks and many passes); row subsets and permutations; repeatable; probes launch. Engine: committed state tc == fast2 (lean and not), drafted == serial, resumed == fresh, snapshots shared with fast2 both ways. `tests/cuda/bench_experts.py --tc [--contend] [--variants ...]`: tc configurations, static stride, CTA caps and probes beside fast2 / fat, the DRAM floor column, and every kernel timed again beside a busy side stream |
| `tests/cuda/test_solo_piece_patches.py` | 0335. Host only: `batchplan.solo_piece`; piece bounds under 3,000 random mixes of solo and normal pieces (on the grid, increasing, ending at the prompt, never more pieces alone); `Batcher._piece` on a fake engine: solo bound only when alone, re-evaluated at the next boundary, `solo_pieces` in the stats; the refused values; lazy Xu (allocated on first use); the memory table. GPU (synthetic EXL3 checkpoint): a fast lean prompt alone prefills in solo pieces == the lone engine; a request admitted mid-prefill makes the next pieces normal, both replies exact; a follower replays the same rounds; lazy Xu engine same reply and smaller lean set |
| `tests/cuda/test_prefill_pp_patches.py` | 0320. Host only: knob / settings / split / `applies`; two real processes over gloo, each running 0082's hash model as its rank with rank-specific output projections, experts and head: the row-split chunk == 0084's pipelined chunk bit for bit on both ranks (last-row logits, final rows, taps, KDA state and conv, caches, own-row streams; 4 variants, R 65-256 on 64-row sub-blocks, 2 positions, EXL3 / MLX), one share a sub-block and site; the swap's [rank 0, rank 1] layout; the padded all-gather fallback; controls caught (routing without the share wait, the wrong half swapped, no split for one sub-block); the piece order. GPU: hc_pre / hc_post on the slot view / hc_post_pre (fused 0 / 3) / stream_mean / rmsnorm on half a sub-block == the whole call's rows (real shapes, 100 / 512 / 1024 rows); `PP_TWO_PROC=1`: rank 0 and rank 1 engines on one GPU through a host-staged gloo communicator, committed state split == not (3-700 tokens) |
| `tests/test_health.py`, `tests/test_effort.py` | 0150, host only: `/health` modes (fatal, stalls with the prefill allowance), `/metrics` counters, JSON 500 / SSE error events, 503 refusals in `strict`, the engine wrapper's transparency; `reasoning_effort` mapping and the default effort |
| `tests/test_serve_ops.py` | host only (fake docker / ssh / nvidia-smi / journalctl, a fake OpenAI server): `serve.sh` start (preflight, memory gate, canary warn / strict, retries, log rotation, NCCL passthrough, the start lock), parallel stop, `xid`, `watch` (absent, loading grace, bad ticks, alert, heal, unreachable worker, drafter rate alert); `canary.py` (degenerate replies, dead drafter, warmup); `xid.py` classes |
| `tests/test_glm_tool_calls.py` | GLM tool calls: schema-typed values, a string that looks like JSON stays text, two calls plus an unknown tool, a call without arguments, the Qwen format still parses |
| `tests/test_decode_size_switch.py` | 0570, CPU (`PYTHONPATH=<tree>/src`, torch CPU): the size knob (default off, MiB, the `_MAX_MB` alias, bad values refuse, the Triton guard), which of 15 per-rank decode shapes switch at 3.5 MiB, `SMALL_PLACE`'s row caps and PDL columns, the table off, the exclude list, `qmm.matmul`'s dispatch through the hook, the compat hash |
| `tests/test_decode_loads_emulator.py` (+ `tests/decode_loads_emu.py`), `tests/test_decode_loads_compile.py`, `tests/cuda/test_decode_loads_patches.py`, `tests/cuda/bench_decode_kernels.py --loads` | 0580. CPU: a lane-level port of `ld_kernel` (every setting, cp.async copies landing at random, NaN shared memory) == the matrix-level grouped_kernel model, Z / Xd / Y bit for bit, real K ranges, rows alone, 8 negative controls; knobs, `fits`, the hook dispatch, the compat hash, the bench gate. Compile (no GPU): 14 sm_121 instantiations (spills, registers, shared memory, CTAs, in-flight KB), PTX (loads, cp.async, mma, no atomics), the PDL prologue, verbatim helpers, SASS loads between HMMAs. GPU: on == off bitwise (real / synthetic, every setting, graphs, rows alone, raw Z, probes), the PDL late-writer race, 0570 == `_qmm` on 15 shapes x 19 row counts; bench: grouped_kernel vs every setting cold (rotate / flush), probes, whole routed layer, two GATE lines |
| `tests/test_fat2_emulator.py` (+ `tests/fat2_emu.py`), `tests/test_fat2_compile.py` (+ `tests/fat2_ptx.py`), `tests/test_fat2_bench.py`, `tests/cuda/test_fat2_patches.py`, `tests/cuda/bench_experts.py --fat2` | 0590. CPU: lane-level ports of fat and fat2 == the matrix reference with an order-sensitive mma model (and fat2 == fat under a position-sensitive one), every output written once, rows alone == in the window, 12 planted mutations caught; sm_121 compile (168 registers, 0 spills, 82 KB, one CTA an SM), helpers verbatim, PTX dataflow of every stored value and mma operand == fat's, SASS epilogue census, 3 planted source changes caught; the bench's GATE logic. GPU: fat2 == fat == fast2 bit for bit (64-8,192 rows, cfgs 0-2, ticket / stride, CTA caps), odd shapes, row subsets, repeatability, probes; engine state fat2 == fat, drafted == serial, resumed == fresh, shared snapshots. Bench: isolated / contended / makespan and the GATE line (docs/EXPERT-PREFILL-V2.md) |
| `tests/test_upstream_ports.py` | 0600, host only (`PYTHONPATH=<tree>/src`; torch CPU for the batcher cases): the socket check; `App.run` refusing a request whose client left, stopping at the next round with nothing more written, a callback failure raised after `generate`; over HTTP a non-streamed request whose client closes stopping within a few rounds, a request cancelled while prefilling through the real `Batcher._collect` (streamed and not), a half-closed stream without a final chunk; `_collect`'s poll, interval check and failure path; `GLM53_TF_DISCONNECT`; 68 malformed-field 400s before headers (App / GlmApp, streamed / not), valid fields == the 0.3.4 `Sampling`, template refusals (check, run, `/tokenize`), a redacted 400 log, non-UTF-8, 500 / stream error; `cmd_serve` registers USR1 before `_serve_cuda`; `check_url`, 28 blocked / 5 public addresses, mixed DNS answers, a DNS-rebinding-style redirect to a private address never connected, same-host rebinding, 6 refused redirect targets, redirect limits, a local image host (types, gzip, sizes, redirects, 404, a stalled server's deadline), no URL in messages, the knobs, the 503 bound; `closed_json`, the #87 repair, the stop count with 5 chunkings, `return_token_ids`, `/health` live totals, failed requests, `streams` |
| `tests/test_tool_fixes.py` | 0620, host only (`PYTHONPATH=<tree>/src`; `GLM53_TF_TOKENIZER_DIR` for the real template and the xgrammar case): the knob; off changes nothing; `history` (the checkpoint template renders `None` before, not after; empty arguments render instead of raising; idempotent); the reasoning memory by server ids and by signature when ids are renumbered, not for other arguments / another user message, never over a client's own reasoning, bounds; `tool_choice` none / named / required, `parallel_tool_calls: false` through the real `App.run`; `thinkcalls` only on `<\|observation\|>` with trailing calls; `typed_value` over 25 schema / value cases; 0610's auto tool grammar (free text free, bad argument types and unknown names rejected) |
| `tests/cuda/test_disconnect_patches.py` | 0600 (`PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda`, torch CPU): 0180's hostile fake model under the real `_plan` / `_execute` / `_finish` / `follow`, every HTTP thread in the real `_collect`, rank 0's message stream replayed on rank 1 (0370 riders off / on): a client gone while queued (never admitted on either rank), prefilling (cancelled before a token), decoding, a callback that raises; rank 1's log / trace / slot states == rank 0's, the other replies == fresh prefill + serial |
| `/src/TensorFold/tests/cuda/test_glm_*.py` | upstream's own GLM CUDA tests, run against the patched tree |
| `tests/cuda/test_grammar_patches.py` | 0610 (`PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda`, xgrammar 0.2.8, torch CPU, jsonschema; `GLM53_TF_TOKENIZER_DIR` for the real-tokenizer part): request fields and 400s; `cut` / `fill` == a fresh xgrammar matcher per row on random paths and corrupted draft chains (stop tokens, `</think>`, invalid ids), the think gate, threads == inline; `apply` on a vocabulary half == the whole row sliced; a masked keyed draw == the rule over the allowed tokens only; two ranks exchanging candidates == one rank (`sample_multi`, `sample_rows`); the real `Batcher` / `Stepper` / `sample_multi` / `prefill` on a hostile float-logit fake model with fake MTP / DFlash2 drafters drafting the constrained continuation with corruptions, and lookup: drafted == `"draft": false` == an independent serial reference (6 policies, greedy / sampled), 4 mixed slots == alone, plain requests knob on == off, random nested schemas valid (jsonschema), thinking on / off, stop / EOS / `ignore_eos` / `max_tokens`, a failure ends only its slot, a follower compiles and decides the same, the lone loops, sessions resumed == fresh, `Batcher.generate`'s thread path; the lone engine's header and rank 1's follow; `setup`; the app; GLM's real tokenizer (no added token inside JSON, stop tokens only at the end, tool markup and calls that parse and validate) |
| `bench/grammar_bench.py`, `bench/structured.py` | 0610: the mask computation microbenchmark on GLM's real vocabulary (build, compile, cut / fill / advance per row, 16-row windows, 4 x 16-row rounds inline and on 2 / 4 / 8 threads, `apply`); the GPU gates against a running server (schemas drafted == serial == 4 concurrent and valid, RigMark-like 50-object array with and without a schema, tool calls, unconstrained hashes vs knob off) |

Run inside the image (needs one GPU; does not load the real checkpoint):

```bash
docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $PWD/tests:/work/tests --entrypoint bash \
    glm53-tensorfold:dev -c "pip install -q pytest; cd /work && \
    PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda python -m pytest -q \
    tests/cuda/test_patches.py tests/test_glm_tool_calls.py /src/TensorFold/tests/cuda/test_glm_*.py"
```

Expected: 50 passed.

On the real model, `bench/glmbench.py --suites exact` checks drafted == serial end to end (10/10 byte-identical
in every TensorFold run so far).

## Rebasing on a newer TensorFold

```bash
git -C vendor/TensorFold fetch && git -C vendor/TensorFold checkout <new-rev>
for p in patches/*.patch; do git -C vendor/TensorFold apply --check "../../$p" || echo "conflict: $p"; done
```

Refresh a conflicting patch against the new tree, rebuild the image, and re-run the tests and the `exact` suite.
