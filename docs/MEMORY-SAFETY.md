# Memory safety: the long-prefill dip and page cache at admission (patches/0550)

> **Update 2026-09-30 (W19, docs/RESULTS.md): the scratch is adopted.** Production (image b10) runs
> `GLM53_TF_SELECT_SCRATCH=grow`, `GLM53_TF_ALLOC_TRIM_GB=0`, `GLM53_TF_ADMIT_MEM=free`. W18 found the trim to be the
> cause of W17's slow first prefills (scratch only was as fast as off); in W19 the 314k needle's minimum went from
> 6.58 / 6.31 to 8.68 / 8.51 GiB (head / worker) and the 4 x 250k stress minimum from 7.75 / 7.61 to 8.34 / 8.09 GiB
> (with the NCCL channel change), so the 8 GiB gate is met again after the heavy warm-up. Section 6 has the W18
> measurements and the 0550 v2 proposal (trims only as planned rounds).
>
> Earlier (W17): measured in image b9 and not adopted: the needle's dip was 1.7 instead of 4.0 GiB, but the first long
> prefill after a burst of short requests ran up to 9.8% slower in 2 of 4 prefill pairs (the trim, per W18).

Status (2026-09-29): written and tested offline (CPU, Triton's interpreter; no GPU run). Production is unchanged
(image b7 = patches through 0490 + 0500 + 0540). 0550 applies to the b7 stack and to the whole stack through 0540.
Line numbers below are in the b7 tree with 0550 applied (`src/tensorfold/families/glm5_next/cuda/`).

| knob | default | what it does | can it change outputs? |
| --- | --- | --- | --- |
| `GLM53_TF_SELECT_SCRATCH` | `grow` | `grow`: the prefill selection's key blocks are views of one buffer per device and stream, grown before a prefill to its largest block (rounded up to 16 MiB, at most `GLM53_TF_SELECT_MB`); `max`: `GLM53_TF_SELECT_MB` from the first use; `off`: one allocation per block (0065) | no (same shapes, strides, alignment, values; §3) |
| `GLM53_TF_ALLOC_TRIM_GB` | `2` | before a batch prefill piece, `torch.cuda.empty_cache()` when the caching allocator holds more than this much it does not use (at most every `GLM53_TF_ALLOC_TRIM_S` = 10 s; a trim that frees < 256 MiB moves the trigger up); `0` off | no (addresses only; §3) |
| `GLM53_TF_ADMIT_MEM` | `available` | admission and the session store count reclaimable page cache: usable = CUDA free + allocator cache + (MemAvailable - MemFree - Dirty - Writeback - max(Mapped, `GLM53_TF_ADMIT_CACHE_KEEP_GB` = 2)); `free`: the rule before 0550 | no (scheduling; batched == alone) |
| `GLM53_TF_ADMIT_FREE_FLOOR_GB` | `1` | with `available`: a request beside running ones also needs this much immediately free (CUDA free + allocator cache), capped at `GLM53_TF_BATCH_ADMIT_GB` | no |
| `GLM53_TF_DROP_OWN_CACHE` | `1` | the session tier's buffered writes (entry headers, O_DIRECT fallbacks) are written back and dropped from the page cache | no |

Admission also reserves the selection scratch a prompt still needs (`memsafe.scratch_growth`, <= 256 MiB, 0 once the
scratch has grown) on top of `GLM53_TF_BATCH_ADMIT_GB`, and logs when it waits for memory:

    [tensorfold] admission waits for memory (3 queued): usable free 2.10 + allocator cache 0.00 + page cache credit 0.00 = 2.10 GiB, less the session store's unused 0.50 GiB, is under 2.00 GiB (GLM53_TF_BATCH_ADMIT_GB + scratch), or free now 2.10 GiB is under the floor
    [tensorfold] admission resumed after 12.4 s waiting for memory

(first line once per episode, again every 30 s while it lasts; `counts["mem_deferred"]`, `job.stats["mem_waits"]`).
The batcher prints its settings at load: `memory safety (patches/0550): selection scratch grow (key blocks up to 256
MiB), admission available (...), allocator trim over 2 GiB unused`.

## 1. The needle dip (W15 §3)

### What was measured

During a lone ~300k prefill MemAvailable fell by 3.0 GiB (B7b, fresh server, 298,388 tokens: 12.3 -> 9.3, flat for
the first ~46%, then steeper) and by 4.5 GiB after the stress + MMLU (314,305 tokens: 10.9 -> **6.39 / 6.42**), all of
it in MemFree (page cache, dirty, anon, shmem flat), then jumped back above the starting level (14.1 / 11.6) exactly
when the prefill ended. W10 / W12 saw the same shape (8.55 / 7.39, 8.27 / 6.58).

### What it is: the caching allocator's growth from the selection's key blocks

A fast chunk runs every DSA layer in 512-row sub-blocks (`GLM53_TF_LEAN_BLOCK`), and every DSA call past 2,051
tokens selects its rows' pools with patches/0065's blocked selection (`latent._attend`, latent.py:913-957 ->
`sparse.select_pools_blocked`, sparse.py:562): per row block it allocates an int32 key block `[n, npc]`,
`npc` = the block's last row's complete pools rounded up to 64, `n` = `block_rows(R, np_max)` <= `GLM53_TF_SELECT_MB`
(256 MiB) / (4 x pool bucket). At 512-row sub-blocks that is one block of 512 rows (two of 256 past 524,288
tokens), **~512 bytes a prompt position**: 64 MiB at 131k, 153 MiB at 314k, 256 MiB from 524k on.

The live size is bounded (0065 did that), but the size **grows by 256 KiB every sub-block**. torch's caching
allocator keeps every segment it ever took (nothing goes back to the system until `empty_cache`), places a request in
the smallest cached free block that holds it, and rounds new large segments up to 2 MiB. So every ~4,096 tokens the
next key block no longer fits the segment the previous one freed: it takes a **new** segment of the new size and the
old one stays cached (other allocations of a chunk are fixed-size and already have their own). The reserved memory
grows by the sum of all sizes: **quadratically in the prompt length**.

The release: `torch.cuda.graph.__enter__` calls `torch.cuda.empty_cache()` before every capture. When the prompt ends
the batcher proposes the first MTP drafts; at a new pool bucket (131,072 pools past 262k) patches/0050's long-context
graph for the MTP head is captured (`graphs.long_step`, graphs.py:74-97) -> the cache is emptied -> MemFree jumps
above where it started (the cache also held older segments). The same explains the boot log's MemFree rising during
calibration (captures) and much of the "drift" between gates (segments kept until the next new graph key).

The model (`memsafe.keys_growth`: a no-reuse upper bound, one layer standing for all 11 since they allocate the same
sizes) against W15:

| trace | bound | measured | ratio |
| --- | ---: | ---: | ---: |
| B7b, 140k -> 298k (the flat first 46%: the cache already held segments that large) | 4.04 GiB | 3.0 GiB | 0.74 |
| gated, 0 -> 314k (MMLU's captures had emptied the cache) | 5.78 GiB | 4.5 GiB | 0.78 |
| slope late in the gated needle (~290-314k) | ~2.8 GiB/min | ~2.5 GiB/min | |

Both nodes dip alike (both ranks allocate the same sizes). Nothing else in a lone prefill scales with the prompt (§2).

### The dip as a function of the prompt length (per rank)

"from 10.9" = the needle's MemAvailable at its start after the stress + MMLU (the worst measured start), minus 0.75 x
the bound (the W15 fit) before 0550, minus the scratch with 0550:

| prompt | largest key block | growth bound before 0550 | x 0.75 | MemAvailable min, before | 0550 scratch | MemAvailable min, 0550 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 32 MiB | 0.18 GiB | 0.13 | 10.8 | 32 MiB | 10.9 |
| 131,072 | 64 MiB | 0.94 GiB | 0.71 | 10.2 | 64 MiB | 10.8 |
| 200,000 | 98 MiB | 2.30 GiB | 1.73 | 9.2 | 112 MiB | 10.8 |
| 262,144 | 128 MiB | 3.97 GiB | 2.98 | 7.9 | 128 MiB | 10.8 |
| 298,388 | 146 MiB | 5.19 GiB | 3.89 | 7.0 | 160 MiB | 10.7 |
| 314,305 | 153 MiB | 5.78 GiB | 4.33 | 6.6 (measured 6.39 / 6.42) | 160 MiB | 10.7 |
| 400,000 | 195 MiB | 9.39 GiB | 7.04 | 3.9 | 208 MiB | 10.7 |
| 524,288 - 1,048,576 | 256 MiB | 16.04 GiB | 12.03 | **< 0: exhaustion** | 256 MiB | **10.7** |

Past 524,288 the blocks are 256 rows and fit the 256 MiB segment made at 524k, so the pre-0550 growth stops there,
but at ~12-16 GiB it is more than a node has free: a lone ~1M prompt (or any past ~420k after a busy session) would
have run the node out of memory (cudaMalloc failing -> torch releases its cache and retries if the driver returns an
error in time; otherwise the OOM killer or an NVRM OOM, as on 2026-09-28 at 524k). Unmeasured, never run.

With 0550 the only prompt-scaled item is the scratch: `S(P) = roundup16MiB(max key block) <= GLM53_TF_SELECT_MB`
(256 MiB), held once per rank and reused by every layer, sub-block, piece and slot.

### 1M prompts with the other three slots busy (the target)

The pool holds 1,048,832 tokens, so a ~1M request leaves room only for small conversations beside it (a 900k prompt
+ 32k max_tokens leaves ~116k tokens for three others). Per rank, relative to the measured 4 x ~250k stress minimum
(8.27 / 8.28 GiB, W15; four slots prefilling in 2,048-token pieces and decoding, store full, graphs, snapshots):

- the same per-slot costs (3 decoding slots' snapshots, round buffers, graphs; the store full; the pool preallocated);
- key blocks: the stress held up to 125 MiB live plus the pre-0550 growth of four interleaved prefills to 250k (up to
  ~2.7 GiB by the model, less whatever captures emptied); with 0550: the 256 MiB scratch, no growth;
- the allocator cache carried between pieces is bounded by the trim (2 GiB, and only while it is unused).

Estimate: >= 8.27 - 0.13 = **8.1 GiB**, likely 9-10 GiB (the stress's own growth removed). The margin is thin enough
that the GPU plan measures it (§5, load S). If it comes out under 8: `GLM53_TF_SELECT_MB=128` (scratch 128 MiB at
1M; same bits: the row blocks are a split of rows whose selections are per row, 0065), `GLM53_TF_ALLOC_TRIM_GB=1`.
A lone ~1M prompt: >= 10.9 - 0.26 = **10.6 GiB** from the worst measured start.

## 2. Every allocation of a lone prefill that depends on the prompt length

Per rank, production shapes (512-row sub-blocks, 4,096-row solo pieces, FP8 latent, KV pool). "Fixed" = sized by
rows or by the load, not by the prompt.

| allocation | size | prompt-scaled? | 0550 |
| --- | --- | --- | --- |
| blocked selection key blocks (`select_pools_blocked`) | `n x npc x 4`, ~512 B a position, <= `SELECT_MB` | **yes: live <= 256 MiB, cached growth quadratic** | one scratch, grown before the prefill (§3) |
| `torch.topk` over a key block (radix select, multi-block for 512 rows) | counts `num_blocks x 256 x 2 B` + small per-slice buffers: ~1.3 MiB at 300k, ~2 MiB at 1M | yes, MiB-scale (<= ~32 MiB of cached steps over a 1M prefill) | left (bounded; trimmed) |
| `_tokens`, `pools`, `top` / `need` | `[512, 2051]` int32/int64 temporaries ~30 MiB | fixed | - |
| one-pass sparse attention (0240, b12x bit 4) | none (no partials) | fixed | - |
| latent sparse partials when not one-pass | `5 x R x 32 x 512 x 4` (160 MiB at 512) | fixed | - |
| eager decode / verify selection (<= 8 rows: sorted path) | `[R, bucket]` fp32 + sort buffers: <= ~80 MiB at 1M, power-of-two buckets | yes, small, log-many sizes | left |
| eager verify windows 9-16 rows (blocked path) | <= 16 MiB at 1M | yes, small | served by the scratch (its 16 MiB minimum) |
| lean set, window buffers, KDA scratch, router partials, expert inputs | per chunk rows | fixed (preallocated or per-chunk constant) | - |
| KV pool pages, pool keys, MTP index keys | preallocated at load (7.44 GiB) | no new memory | - |
| MTP head in prefill (`MTP_PREFILL_CACHE=1`) | cache writes only, no selection | fixed | - |
| snapshots: a piece's resume snapshot, <= 2 kept a slot | rec 68 MiB + conv + DFlash2 window ~94 MB each | fixed | - |
| session marks (every 16,384 tokens) into the RAM store | pages ~125 MB + ~94 MB a mark, **within `SESSION_GIB` (2)**; growth only while CUDA free + cache (+ page cache credit with 0550) - n >= `SESSION_RESERVE_GIB` | bounded by the budget | counts page cache |
| NVMe tier write queue (0250) | packed device buffers <= `GLM53_TF_SESSION_DISK_QUEUE_GIB` (1); larger entries written synchronously in 8 MiB groups | bounded | own buffered writes dropped from the cache |

So a longer prompt needs more memory only through the key blocks (fixed by 0550) and a few MiB of top-k / eager
selection temporaries (bounded; the trim returns what the cache collects). The measured dip is not the store, the
NVMe tier or a leak (W15: page cache / anon flat; the store's own growth is inside its budget).

Alternatives considered: smaller solo pieces above a threshold (the key block does not depend on the piece: the DSA
calls are 512-row sub-blocks at every piece size), context-scaled chunking of the selection (the live block is
already bounded by `SELECT_MB`; the problem was the allocator, not the live size), admission-time reservation of the
whole growth (it is unbounded without the scratch). The scratch plus a trim fixes the cause; admission reserves what
is left (the scratch's growth).

## 3. The fix and why the outputs cannot change

**Scratch** (`sparse.reserve_scratch` / `reserve_for_prefill` / `_keys_block`, sparse.py:463-560; called from
`decode._prefill`, decode.py:514-520, for the prefill's positions `begin .. n` with the DSA call rows: the lean
sub-block, else the chunk). A key block is `scratch[:n * npc].view(n, npc)`:

- same shape and strides as `torch.empty((n, npc), int32)` (contiguous rows of `npc`), base = the buffer's start: the
  caching allocator's 512-byte alignment, so Triton's pointer specializations (16-byte divisibility) and every launch
  are the same;
- `_scores_rows` writes every element before anything reads it: its grid is `(cdiv(n, 16), cdiv(npc, 64))`, program
  (rb, pb) stores rows `row0 + rb*16 .. < row0 + n` at pools `p < npc` (`mask=p < NP`); `torch.topk` and
  `_gather_sel` read only `[n, npc]`. The junk left by earlier blocks or other prompts is never read (the CPU test fills
  the scratch with junk first);
- blocks are used one after another on one stream (the next block's scoring is stream-ordered after the previous
  block's `_gather_sel`), one scratch per (device, stream); while a stream captures a CUDA graph the block is allocated
  as before (no growth, no `empty_cache` inside a capture);
- no chunking changes: the row blocks, pieces, sub-blocks and chunk grid are exactly 0065 / 0082 / 0085's, so the
  C-independence / row-invariance arguments are untouched.

So keys, thresholds and selections are the same bits (tested against `off` and the sorted path on random, tied,
-0.0 / NaN scores, rows crossing 2,051, several blocks, junk-filled / too-small / grown scratch).

Growth happens on the host before the prefill: the old buffer is dropped, the device synchronized and the cache
emptied, then the new one allocated, so no segment is left behind. At most 16 growths a process (16 MiB steps to 256
MiB); a growth costs a sync plus re-allocating the next chunk's temporaries (milliseconds). If a DSA call still needs
more (a caller that did not reserve: the MTP head without `MTP_PREFILL_CACHE`, a verify window), the scratch grows in
place without emptying the cache (the next trim does) and `SCRATCH_STATS["fallback"]` counts it.

**Trim** (`memsafe.Trimmer`, `trim_torch`; `Batcher._piece`, batch.py:1941-1945). `empty_cache` moves nothing that is
in use and changes only which addresses later allocations get; kernels never depend on addresses beyond alignment,
which the allocator keeps at 512 bytes. It is exactly what every CUDA graph capture already does (`torch.cuda.graph`
calls it) at data-dependent points in every gate run so far (exact, batchexact, transcripts, N1 all equal across
captures). Local to each rank: no collective, no ordering constraint between ranks.

**Admission** changes only when a request starts (batched == alone and resumed == fresh are gated invariants).

## 4. Page cache and admission (W15: C4 serialized during a 36 GB copy)

### What happened

`Batcher._plan` admits a request beside running ones while `free - store headroom >= GLM53_TF_BATCH_ADMIT_GB` (2),
`free` = `cudaMemGetInfo` free + the allocator's unused cache (batch.py `_free`). On GB10 that free is the kernel's
MemFree. While `docker save` streamed 36 GB, head had MemFree 2.1 GiB and 16.8 GiB of page cache: 2.1 - headroom
< 2, so C4's requests went one at a time (queues 6.3 / 12.5 / 18.6 s); after `drop_caches` the same run was normal.
The session store's growth rule (`_can_grow`: free + cache - n >= `SESSION_RESERVE_GIB`) had the same blind spot.

### The rule now (`GLM53_TF_ADMIT_MEM=available`, `memsafe.view` / `admit_ok`, batch.py `_mem_ok`, sessions.py `_can_grow`)

    usable = CUDA free + allocator cache + credit
    credit = max(0, MemAvailable - MemFree - Dirty - Writeback - max(Mapped, GLM53_TF_ADMIT_CACHE_KEEP_GB))
    admit beside running requests iff  usable - store headroom >= ADMIT_GB + the prompt's scratch growth
                                  and  CUDA free + allocator cache >= min(GLM53_TF_ADMIT_FREE_FLOOR_GB, ADMIT_GB)

The W15 copy (MemFree 2.1 GiB, 16.8 GiB of page cache): MemAvailable - MemFree ~16.3 GiB (the cache less the kernel's low-watermark share), less Dirty (not recorded then; ~1 GiB assumed) and max(Mapped 2.8, 2): a credit of ~12.5 GiB -> admitted at once. Idle
production today (head: MemFree 6.2, MemAvailable 10.6, Mapped 2.8 GiB): credit ~1.6 GiB. With nothing
reclaimable and little free, it waits exactly as before. `GLM53_TF_ADMIT_MEM=free` is the old rule (the free-now floor
is then off; only the scratch reservation is added). The load-time slot rule (`GLM53_TF_BATCH_RESERVE_GB`) is not
changed: slots are permanent, and `serve.sh`'s memory gate drops caches before a start anyway. Rank 0 decides from
its own node, as before (rank 1 is not polled: that would add a collective to every round); both nodes dipped alike
in W15.

### Why counting page cache is safe on GB10, and the margins

- GB10 has no separate device memory: the driver backs `cudaMalloc` with pages from the kernel's page allocator
  (BOOT.md: "the page cache it fills is GPU memory on GB10"). A device allocation competes with the page cache like
  any other kernel allocation: under the zone watermarks kswapd and direct reclaim evict clean file pages and the
  allocation proceeds. `cudaMemGetInfo` reports the free pages only; it is a reporting limit, not an allocation limit.
- Every "MemFree, not MemAvailable" observation in this repo was a check of that reported figure, not a failed
  allocation: vLLM refusing to start (its own `gpu_memory_utilization` check), the 1-slot loads (the load rule reads
  free), C4 serialized (admission reads free). During the copy the server kept prefilling and decoding at MemFree
  ~2.1 GiB for minutes (piece temporaries re-allocated as needed) without an error. No failed allocation with
  reclaimable cache present is recorded. The GPU plan's probe (step P) measures it directly before anything relies on it.
- What cannot be handed over at once is not counted: Dirty and Writeback (they need I/O first; a copy can hold up to
  `dirty_ratio` 20% of available memory dirty), Shmem (not in MemAvailable anyway), and max(Mapped, 2 GiB): pages
  mapped by running programs, our own libraries included (reclaiming them trades an allocation for re-faults).
- The OOM protection never came from counting only MemFree (the 2026-09-28 OOM happened under a MemFree-based store
  rule): it comes from the margins and from bounded transients. Kept: `GLM53_TF_BATCH_ADMIT_GB` (2) now on usable
  memory, the free-now floor (1 GiB: the first chunk's buffers are allocated before reclaim can matter), the store's
  `SESSION_RESERVE_GIB` (6) on the same measure, and with 0550 a request's growth is predictable (the scratch, reserved
  at admission; the trim).
- Remaining risk: reclaim latency (several GB/s for clean pages: ~0.1-0.3 s for a GiB in the worst case, inside a
  piece) and a probe result showing cudaMalloc failing beyond MemFree. Either way `GLM53_TF_ADMIT_MEM=free` restores
  the old rule with one knob.

Our own page cache: the NVMe session tier reads and writes with O_DIRECT (reads already `POSIX_FADV_DONTNEED` when it
falls back); its entry headers and any buffered fallback write now `fdatasync` + `DONTNEED` (`sessdisk._drop_cache`).
Prepared weights are read with O_DIRECT. Dropping other programs' caches needs root (`serve.sh`'s
`MEM_GATE_DROP_CACHES` at start); the engine does not do it at run time.

## 5. Tests and the GPU plan

### CPU (done)

`tests/test_memory_safety.py` (20 passed; `TRITON_INTERPRET=1 PYTHONPATH=<tree>/src`, ~4.5 min, the bit tests ~55 s a
case): the model's rules == `sparse`'s (3,000 random cases); admission's fast `select_peak` == a scan of every call
(4,000 cases); `select_blocks` == the blocks `select_pools_blocked`
really allocates (recorded, interpreter); `keys_growth` vs W15 (0.74 / 0.78 of the bound), its quadratic shape and the
16 GiB at 1M; the scratch <= 256 MiB at every 4,096-token piece to 1,048,576 and admission's reservation; scratch
(`grow`, `max`; junk-filled, grown, too small) == `off` == the sorted path on random / tied / -0.0 / NaN scores;
a reserved scratch allocates no key block; meminfo parsing, the credit and its margins; W15's copy admits with
`available` and waits with `free`; `free` == the old rule on 2,000 random cases; the floor; the wait / resume log;
`Batcher._mem_ok` and `SessionStore._can_grow` on fakes; the trimmer's trigger, interval and hysteresis;
`drop_file_cache`. Existing host tests against b7 + 0550: `test_replay_ttft_patches`, `test_batch_sessions_patches`,
`test_decode_overlap_patches`, `test_session_patches`, `test_session_disk_patches`, `test_prefix_share_patches`,
`test_kv_pool_patches`, `test_batch2_patches`, `test_fastpf_patches`, `test_lean_patches`, `test_request_log` (227
passed, 119 GPU-only skipped) and `test_kvpool_interpreter` (16) pass (`test_solo_piece_patches`: the same 3 failures on b7 without 0550: its fake job lacks 0500's
`vision`); against the whole stack + 0550 also `test_http_pin`, `test_batch_graphs_lone(_patches)`.

### GPU (to do; prod stopped, both nodes, under `timeout`, W15's lease / watchdog / restore harness)

Build `glm53-tensorfold:b8` = b7's list + 0550 (`PATCHES="... 0490 0500 0540 0550"`), ship to the worker node. Loads: `B7`
(control, prod.env), `B8` (prod.env + IMAGE b8, 0550's defaults), `B8F` (B8 + `GLM53_TF_ADMIT_MEM=free`).

1. **Kernel / unit tests on the device** (`tests/cuda/test_memory_safety_patches.py`, both nodes): scratch == off ==
   sorted at production shapes (70k / 300k / 524k boundary / 1M, 262,148 pools); the mechanism: 512-row selections
   at a 100k -> 230k prefill's positions grow `memory_reserved` by >= half the bound with `off` and by <= 160 MiB with
   `grow` (the print gives both); a capture empties the cache; the trim frees; engine replies with the scratch
   (`grow` / `max`) and a trim before every chunk == `off`, serial and drafted. Also `test_1m_patches.py`,
   `test_lean_patches.py`, `test_batch_sessions_patches.py`, `test_session_disk_patches.py`, TensorFold's
   `test_glm_engine.py`.
2. **P: can cudaMalloc take page cache?** (node idle, server stopped): `bench/pagecache_probe.py OUT.json <a 30+ GB
   file> 20 6 1 8` on both nodes. Pass: `cache_reclaimed` true, no `failed_at`, per-GiB step time recorded. If it
   fails: ship 0550 with `GLM53_TF_ADMIT_MEM=free` (the scratch and trim still apply) and record it here.
3. **Gates on B8** (W12 / W15 `gates.sh` unchanged, MemAvailable + meminfo every 2 s with phase marks): exact 10/10
   twice, batchexact 4/4 twice, W9 transcripts, **reply sha 8794a3463259cc2f**, 13/13 glmbench hashes == W12, N1 12/12,
   prefill 24.5k / 98k within 1% of B7 (1,610 / 1,607), decode 1 / 4 streams within noise, MMLU-200 >= 87%.
4. **Stress + needle 314k** (the W15 sequence: ab, N1, 4 x ~250k stress, MMLU, needle 314,305 alone): stress minimum
   >= 8 (expect above B7's 8.27 / 8.28: its key-block growth is gone), **needle minimum >= 8 on both nodes** (expect
   ~10.5; B7 6.39 / 6.42), no step at the prefill's end (the next capture has nothing to release), `trims` and
   scratch size from the log. Control: the needle's trace on B7 exists (W15).
5. **~900k with 3 busy slots (load S)**: three looping conversations (~8k prompts, 8k max_tokens, thinking on) in
   slots 1-3, then a ~900,000-token needle (max_tokens 256; 3,517 pool pages, the others hold < 200): prefill in
   2,048-token pieces beside the decodes (expect ~40-60 min; `GLM53_TF_STALL_S` >= 3600). Pass: found, MemAvailable
   **>= 8 GiB on both nodes throughout**, no OOM / NVRM line, the three streams' decode gaps <= one piece. Then the
   same prompt resent (resumes at n - 64). If a lone 1M is wanted: ~1,015,000 tokens alone, expect >= 10.5 GiB.
   If S dips under 8: `GLM53_TF_SELECT_MB=128` and / or `GLM53_TF_ALLOC_TRIM_GB=1` and repeat step 5.
6. **C4 under a concurrent copy**: on the head node, `cp` a 36 GB file (or `docker save`) to local NVMe, and while it runs
   (MemFree ~2 GiB, Cached >= 15 GiB) run `results/W15/c4.py` (thinking off, 3 rounds) on B8 and on B8F. Pass: B8's
   rounds admitted together (queue_s ~0 for all four, as W15's `c4-b5ctl2` after `drop_caches`), B8F reproduces the
   serialization (6 / 12 / 18 s) and logs `admission waits for memory` with the numbers; no error, MemAvailable
   >= 8 GiB during it; one request's first piece time not worse than without the copy by more than the reclaim latency
   measured in P. Then exact / batchexact / reply sha again after the copy.
7. **Adopt** if 1-6 pass: prod.env `IMAGE=glm53-tensorfold:b8` (0550's knobs default on; nothing else changes).
   Revert: `IMAGE=glm53-tensorfold:b7`; partial on b8: `GLM53_TF_SELECT_SCRATCH=off`, `GLM53_TF_ALLOC_TRIM_GB=0`,
   `GLM53_TF_ADMIT_MEM=free`, each alone.

## 6. W17 / W18 on the GPU: the trim, and a follow-up (proposal, not implemented)

Measured (docs/RESULTS.md W17, W18; `results/W18/`). Per load: the W17 sequence (heavy warm-up = `mpf.py` + `c4.py`
x2, ab.sh set, N1, 4 x ~250k stress, MMLU-200, exact again, needle ~314k alone) with four ab.py pairs (24.5k / 98k).

| load | knobs | slow prefills (of 8) | stress dip / min | needle dip / min | where |
| --- | --- | ---: | --- | --- | --- |
| W17 B9 | 0550 full (scratch grow, admission available, trim 2 GiB) | **5** (pairs: 1,454 / 1,550, 1,608 / 1,597, 1,577 / 1,594, 1,512 / 1,541) | 2.0 / 7.37, 6.89 | 1.7 / 7.92, 7.48 | W17 |
| W17 M6, W18 A | 0550 off (prod) | 0 | 1.6-2.1 / 6.51, 6.14 (A) | **3.4-3.6** / 5.10, 4.75 (A) | W17, W18 |
| W18 T | scratch grow + admission available, **trim off** | **0** | 1.25 / 7.70, 7.58 | **0.8-1.1** / 8.10, 7.86 | W18 |
| W18 S | scratch grow only (admission free, trim off) | **0** | 1.15 / 7.84, 7.56 | **0.6-0.75** / 8.39, 8.01 (298k) | W18 |

(dip = MemAvailable at the phase start minus its minimum, GiB, head / worker; the minima also depend on the boot path,
see below.) The scratch removes ~2.5 GiB of the 314k needle's dip and ~0.4 GiB of the stress's, and the lone 314k
prefill is faster with it (1,371 / 1,380 against 1,290 tok/s: the pre-0550 path spends time in cudaMalloc for every
new key block size). Every prefill slowdown seen so far came with the trim on; without it 16 of 16 (T, S) and 16 of
16 (A, M6, C7) prefills are at 1,600-1,613 tok/s. T and S do not differ measurably (the admission rule only acts with
a lot of page cache present, which no load had). W18 did not re-run the full-on load (the plan changed), so the attribution is
by elimination: the trim is the only 0550 knob B9 had and T does not (T keeps the scratch and the admission rule).

### Why the trim costs time where it runs

- `Batcher._piece` calls it before every prefill piece. During a long lone prefill every piece frees its temporaries
  at its end, so "unused > 2 GiB" is true at almost every piece start and the 10 s interval is what limits it: a trim
  every ~10 s of prefill (1-2 in a 24.5k prompt, ~5 in a 98k one).
- `empty_cache` frees every unused segment with `cudaFree`, which synchronizes the device: the host loses its lead
  over the GPU (the launch queue drains). On GB10 the pages go back to the kernel, and the next piece's cudaMallocs
  (several GiB of piece temporaries) take them again (~0.04 s a GiB, step P).
- It is **rank-local**: each rank decides from its own allocator at its own moment, so a trim on one rank stalls the
  other at the next all-gather; the two ranks' costs add instead of overlapping.
- Whether it fires depends on the allocator's state (the hysteresis raises the trigger after a trim that frees
  < 256 MiB of graph-pool memory), which is why W17 saw it in 2 of 4 pairs and never in a fixed pattern.

### Proposal for the follow-up patch (0550 v2): trim only at planned, symmetric, idle points

1. **Idle trim.** Rank 0's `_plan`: when no slot is busy and the queue is empty for `GLM53_TF_ALLOC_TRIM_IDLE_S`
   (default 2 s), plan a **trim round** (a flag in the plan `batchplan` already shares with rank 1); both ranks run
   `empty_cache` at that round. Rank 1 cannot see idleness itself (it blocks in `_share`), so the flag has to ride the
   plan. Cost: none for a running request; a request arriving during the trim waits for it (tens to a few hundred ms).
2. **Pressure trim while busy.** Only when rank 0's usable memory (0550's `memsafe.view`) is under
   `GLM53_TF_BATCH_ADMIT_GB + GLM53_TF_SESSION_RESERVE_GIB` (or MemAvailable under 8 GiB) **and** the allocator holds
   more than `GLM53_TF_ALLOC_TRIM_GB` unused: a planned trim round on both ranks, between pieces, at most every 60 s.
   This is the 4 x 250k case, where returning cache is worth a stall; a lone prefill on a healthy node never trims.
3. **Never inside a lone prefill's pieces** unless (2) applies; the scratch already removes the growth the trim was
   meant to catch (§1), and the scratch's own growth (host side, <= 16 times a process, same prompt on both ranks) is
   already symmetric.
4. **Boot-time trim (independent, free).** W18 found that a start with a cached calibration ends its "engine ready"
   step (the batcher's warm-up, 6.5-7.6 s) holding ~1.2 GiB more than a start that re-measured it (engine ready
   MemAvailable 15.6 / 15.0 against 16.8 / 16.6 GiB, both ranks; b7 did not show it: 17.0 GiB, 0.4 s). Production
   restarts take the cached path, so prod starts ~1 GiB lower than the W17 loads did. One `torch.cuda.empty_cache()`
   on both ranks after the warm-up (before serving) should give it back. Check: engine-ready MemAvailable equal on the
   cached and the re-measured path.

Gates for v2: same bits (exact, batchexact, reply sha, N1); prefill 24.5k / 98k x4 within 0.5% of the control; stress
and needle minima >= T's; a planned trim visible in the log with its duration; C4 TTFT not worse.

Recommended now (no rebuild; not adopted in W18 after a plan change, for W19): `GLM53_TF_SELECT_SCRATCH=grow`,
`GLM53_TF_ALLOC_TRIM_GB=0`, `GLM53_TF_ADMIT_MEM=free` (S; `available` once the C4-under-a-copy test of §5 step 6
passes). Expected on prod's cached boot path: stress minimum ~6.9 / 6.6 GiB, 314k needle ~7.4 / 7.6 (A's starts minus
T's / S's dips) instead of 6.5 / 6.1 and 5.1 / 4.75 today; >= 8 needs item 4 and / or less 4-slot transient.
