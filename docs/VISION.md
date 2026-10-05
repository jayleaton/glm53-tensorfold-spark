# Image input for GLM-5.3-Flash (patches/0500, `GLM53_TF_VISION`)

Status: implemented, tested offline (CPU) and on the Sparks (image b6, docs/RESULTS.md W14: GPU tests, text
bit-identical with the knob on and off, real images, caches incl. the NVMe tier, 4 x 250k stress). **In production since
W15** (image b7, `GLM53_TF_VISION=1` with 64 / 64 MB caches in `config/prod.env`; docs/RESULTS.md W15: the full gate set,
stress 8.27 / 8.28 GiB with the tower on the head node). The GPU plan is in §7. Default off: with the knob unset, every request
is served as before, bit for bit.

## 1. What it does

`GLM53_TF_VISION=1` accepts images in OpenAI chat messages on `/v1/chat/completions` (and in `/tokenize`):

```json
{"role": "user", "content": [
  {"type": "text", "text": "What does this screenshot say?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR...", "detail": "auto"}}]}
```

- **Accepted part shapes:**
  - `image_url` with `{"url": ...}` or a bare URL string;
  - `input_image` (`image_url` / `url`);
  - `image` (`image` / `url`).
- **URLs:** `data:` (base64 or percent-encoded) and `http(s)`. Nothing else: `file:`, `ftp:` and the like are
  refused.
- **Video parts** keep the checkpoint template's "cannot process" reminder. Video is not implemented.
- Rank 0 fetches, decodes and preprocesses each image, then encodes it with the vision tower. Both ranks then
  prefill the prompt with the image's rows in place of its placeholder tokens. Decode is unchanged.
- **Errors:** any image problem is an HTTP 400 with `param: "messages"` before anything streams. Covered: fetch,
  size, decode, too many images, a placeholder typed into the text.
- **Token accounting:** `usage.prompt_tokens` counts the image rows, as vLLM does. The context check counts them too
  (`context_length_exceeded`).

| Knob | Default | Meaning |
| --- | --- | --- |
| `GLM53_TF_VISION` | `0` | `1`: image input on. Rank 0 loads the tower. Rank 1 needs nothing: it follows the data, whatever its own setting. |
| `GLM53_TF_VISION_MAX_IMAGES` | `8` | Most images a request may carry, counted over the whole `messages` history (earlier turns included). |
| `GLM53_TF_VISION_OVERFLOW` | `refuse` | patches/0660. `refuse`: more images than `_MAX_IMAGES` is a 400. `drop_oldest`: the oldest image parts become the text `[image omitted: over GLM53_TF_VISION_MAX_IMAGES]` and the newest `_MAX_IMAGES` are kept; a last message that alone holds more is still a 400. See "Long conversations with images" below. |
| `GLM53_TF_VISION_MAX_TOKENS` | `0` (the processor's 8,000) | Caps the rows an image takes. Below 8,000 this departs from the reference processor for large images. |
| `GLM53_TF_VISION_LOW_TOKENS` | `0` (ignored, as vLLM) | Rows for `detail: "low"` when set. |
| `GLM53_TF_VISION_FETCH` | `1` | `0` refuses `http(s)` URLs (only `data:`). The server fetches URLs from rank 0's network, so this is SSRF-relevant. |
| `GLM53_TF_VISION_FETCH_TIMEOUT` | `10` s | Per download. |
| `GLM53_TF_VISION_MAX_BYTES` | 20 MiB | Largest encoded image. |
| `GLM53_TF_VISION_MAX_PIXELS` | 64 M | Largest decoded image. Checked from the header, before decoding (decompression bombs). |
| `GLM53_TF_VISION_CACHE_MB` | `256` | Rank 0's cache of encoded rows by image digest (host memory). A conversation's earlier images are not re-encoded every turn. |
| `GLM53_TF_VISION_PREP_MB` | `256` | Rank 0's cache of preprocessed `data:` images (host memory). |

### Long conversations with images (patches/0660, issue #15)

The image limit counts every image part in the request, and an agent client resends its whole history each turn.
Once the history holds more than `GLM53_TF_VISION_MAX_IMAGES` images, every later request carrying that history is
refused the same way, including a client's compaction request. The 400 (`invalid_request_error`, `param:
"messages"`, sent before anything streams, also for `stream: true`) says so and names the ways out:

- **Raise the limit**, e.g. `GLM53_TF_VISION_MAX_IMAGES=32`. Each image takes up to 8,000 prompt tokens (cap it with
  `GLM53_TF_VISION_MAX_TOKENS`, e.g. `2000`, for screenshot-heavy sessions); the context and KV-pool checks price the
  rows, so a long history fails cleanly with `context_length_exceeded`.
- **`GLM53_TF_VISION_OVERFLOW=drop_oldest`** (opt-in): the oldest image parts beyond the limit become the text
  `[image omitted: over GLM53_TF_VISION_MAX_IMAGES]`, the newest `_MAX_IMAGES` are kept, and the request runs. The
  images of the last message are never dropped: if it alone holds more than the limit, the request is still a 400.
  Trade-offs: the model no longer sees the dropped images, and each new image moves the cut by one, so the prompt
  changes at the oldest kept image and the prefix caches (snapshots, session store, NVMe tier) reuse only the text
  before it. Rank 0 decides; rank 1 follows the data.

The default (`refuse`) is the behaviour before patches/0660 apart from the message text.

## 2. Findings about the model

### The checkpoint

`neko-legends/GLM-5.3-Flash-Uncensored-EXL3`, snapshot `07135ec`, inspected read-only on the head node.

- `Glm5NextForConditionalGeneration`.
- 347 `model.visual.*` tensors, **all BF16** (EXL3 covers only the routed experts), 1.127 GB, all in shard
  `model-00092.safetensors`:
  - `patch_embed.proj` (Conv3d [1024, 3, 2, 14, 14] + bias);
  - 24 blocks, each with `norm1`, `norm2`, `attn.qkv` (+ bias), `attn.proj` (+ bias), `attn.q_norm` / `attn.k_norm`
    ([64]: per-head RMSNorm) and `mlp.gate/up/down` (+ biases);
  - `post_layernorm`;
  - `downsample` (Conv2d [4096, 1024, 2, 2] + bias);
  - `merger.proj`, `merger.post_projection_norm` (LayerNorm + bias), `merger.gate/up/down` (4096 -> 10240 -> 4096).
- No learned position embedding and no post-conv norm (GLM-4.1V has both).
- `processor_config.json`:
  - `Glm5NextImageProcessor`, CLIP mean / std;
  - patch 14, temporal 2, merge 2;
  - `min_image_tokens` 16, `max_image_tokens` 8000 (video: 240,000, fps 2).
- `config.json`:
  - `image_start_token_id` 154830 (`<|begin_of_image|>`), `image_token_id` 154854 (`<|image|>`),
    `image_end_token_id` 154831;
  - video 154832 / 154855 / 154833.
- **Its `chat_template.jinja` does not render images.** Every image / video / audio part becomes the text
  `<reminder>You are unable to process this image because you don't have multi-modal input ability...</reminder>`.
- The vision template renders an image as `<|begin_of_image|><|image|><|end_of_image|>`. It ships with
  `MikeRoz/GLM-5.3-Flash-Uncensored-3.05bpw-h6-exl3` and as the vLLM kit's `files/chat_template.jinja` (macro
  `emit_image`).
- The patch does not replace the template. It replaces each image part with a sentinel text part, renders the
  checkpoint's own template, then puts the placeholder where the sentinel landed. The result is exactly the vision
  template's text; tested to be token-identical to transformers' processor with the kit template.

### M-RoPE: no

GLM-5.3-Flash's language model uses no positions at all, so an image is only its embeddings.

- transformers 5.17's `glm5_next` (identical on the head node's `vllm-exl3` venv and locally):
  - `Glm5NextTextConfig` asserts `qk_rope_head_dim == 0` ("Expecting NoPE for the DSA attention layers"). The
    checkpoint has `qk_rope_head_dim` 0, `mla_use_nope` true and no `rope_parameters`.
  - `Glm5NextTextModel.forward` builds `position_ids = arange(...)` and calls every layer with
    `position_embeddings=None` ("Key change using NoPE").
  - The MLA split gives `k_rot` of width 0. The DSA indexer has no rotary. The KDA layers take no positions.
  - `Glm5NextModel` has no `get_rope_index` / M-RoPE code. The `ndim == 3` position branch in
    `_expand_inputs_for_generation` is a GLM-4V leftover no caller reaches.
  - An image is `inputs_embeds.masked_scatter(image_mask, image_embeds)` at the `<|image|>` positions, which the
    model then expands over the 4 hyper-connection streams like any token.
- exllamav3's `glm5_next.py` agrees: it asserts `qk_rope_head_dim == 0`, has `rope_settings = None`, and takes its
  tower from GLM-4V with GLM-5.3 deltas.
- **What our engine needs:** nothing positional. The DSA / MLA layers (NoPE, latent cache) and the KDA layers
  consume the image rows as they consume any row. The indexer's k-pool compression APE is by position within a pool
  of 4, the same for any row.
- The only requirement is that the embedding step writes the image rows in place of the token rows, in every
  stream copy.

### The tower (transformers `Glm5NextVisionModel`, reproduced in `vision.py`)

- **Patch embedding:** the Conv3d over [3, 2, 14, 14] with stride = kernel, i.e. a matmul over 1,176 values. The
  processor duplicates the image's single frame over the temporal pair (`broadcast_to`).
- **2D RoPE:** axial.
  - 16 inverse frequencies of base 10000 over head_dim / 2 = 32.
  - The h angles, then the w angles, concatenated twice, rotating the whole 64-wide head (`rotate_half`).
  - Applied after the per-head q / k RMSNorm, in fp32.
- **Attention:** full bidirectional within each image (`cu_seqlens` per image). No windows.
- **MLP:** the gate is clamped at `swiglu_limit` 10, up to +-10, then `silu(gate) * up`. Biases throughout.
- **Merge:** a final RMSNorm, then the 2 x 2 window as a Conv2d (a matmul over [C, kh, kw] = 4,096 inputs).
  - Patches arrive merge-window-major from the processor, so each window is 4 consecutive rows.
- **Merger:**
  - `proj` (no bias);
  - LayerNorm (eps 1e-5, bias);
  - exact GELU;
  - the clamped SwiGLU 4096 -> 10240 -> 4096.
- **Result:** 1 LM row per 2 x 2 patches.

### The processor (transformers `Glm5NextImageProcessor`, torchvision backend)

This is the backend HF's `AutoProcessor` and vLLM load. The steps:

1. `smart_resize` to the token budget. It is a binary search on the aligned 28-pixel canvas; small images are scaled
   up to at least 16 tokens.
2. Content scale = min(target / size), never above 1 once the image meets the minimum.
3. Bicubic antialiased resize of the uint8 image (torch's native uint8 path on CPU).
4. **Zero padding** to the canvas: right and bottom, black before normalization. GLM-4V warps the aspect ratio
   instead.
5. Fused `(x - 255 mean) / (255 std)` in float32.
6. The patch layout.

vLLM's media loader converts RGBA to RGB on white before the processor. `vision_prep.decode` does the same.

### Upstream and the vLLM kit

- **TensorFold 0.3.6.2** (`71377a5`, read in `~/.cache/tf-upstream`) has no vision anywhere:
  - `cuda/geometry.py` and `glm5_next/cuda/split.py` skip `model.visual.*`;
  - `server/messages.py` lists `image_url` among the media types;
  - `docs/api.md`: "Image, audio and video input or output requests receive HTTP 400".
- **The vLLM kit** on the head node (`~/glm53-exl3-2x-kit`, read only) serves with vision **on** by default:
  - `LANGUAGE_MODEL_ONLY=0`, `LIMIT_MM='{"image":4,"video":1}'`, `SKIP_MM_PROFILING=1`;
  - `CHAT_TEMPLATE=/opt/glm53/chat_template.jinja` (the `emit_image` template);
  - an overlay `patch_glm_video_placeholders.py` for video timestamps.
- Its vLLM `glm5_next` model file lives inside the kit's docker image and was not read (no containers were started
  for it). transformers' implementation is the reference used here. vLLM 0.29 on the head node's `vllm-exl3` venv has no
  `glm5_next` model file.

## 3. Design

### Virtual ids

Each image row in the prompt is a **virtual token id** at or above `VBASE` = 2^24. The vocabulary has 154,880 ids,
and ids travel as int32 below 2^31.

- `digest` = SHA-256 of:
  - a version tag;
  - the processor settings and the `vision_config`;
  - the canvas size and grid;
  - **the preprocessed uint8 canvas**.
- Row k's id = `VBASE + blake2b(sha256(digest || salt) || k) mod (2^31 - 2^24)`.

Consequences:

- **Every cache keyed by token ids is automatically correct**, with no change to any of them. Covered:
  - the engine's live snapshots (`_resume`) and the batcher's per-slot snapshots;
  - the session store (entry keys, chained page keys, `find`, `lcp`, fork marks), the 0310 prefix share, the NVMe
    tier (`DiskIndex`: the same lookups, `ids_b64` int32 packing);
  - the request log's hashes, `seed_for`.
- **Two different images with the same text share state up to the image and never past it.** They differ at every
  row, not only the first. The same pixels share state everywhere, whether the image arrived as PNG, BMP or a
  re-encoded data URL.
- A follow-up turn resumes past the image, because the client resends the same image and it gets the same ids.
- **False sharing** needs every row of a cached prefix inside an image to collide. A prefix ending j rows into an
  image collides with probability 2^-31j, so about 5e-10 at j = 1, the worst case, which needs a snapshot boundary
  right after an image's first row.
  - Rank 0 also keeps a registry of the last 262,144 ids it handed out. An image whose ids meet an earlier image's
    is derived again with the next salt, so within a process no two images share an id.
  - Across restarts (the NVMe tier), the 2^-31j bound is the guarantee.
- Nothing reads a virtual id as a token:
  - the embedding substitutes the rows (below);
  - the lookup drafter cuts a proposal at the first virtual id (in `SuffixIndex.propose`, so 0450's resident rounds
    get the same cut);
  - the draft-vocabulary fallback window (0420) skips them;
  - the sampler only ever emits vocabulary ids.
- Known leak: 0430's draft dump records them as `tokens`. It is off in production, and a trainer reading image
  requests must mask ids >= 2^24.

### Where the rows enter

`glue.embed` is the single place every forward path embeds token rows:

- the exact and fast prefill chunks, lean, pipelined (0084) and row-split (0320);
- the batcher's pieces;
- the MTP head's absorb (`mtp.py`, `pfglue.absorb`);
- DFlash2's block (vocabulary ids only).

Inside `vision.active(table)`, which wraps exactly the prefill calls (`GlmEngine._run`, `Batcher._piece`), it:

1. clamps virtual ids to 0 so the kernel reads a valid row;
2. runs the unchanged kernel;
3. overwrites the matching rows in every stream copy with the tower's rows (`Table.fixup`: `searchsorted` +
   `where`, no host sync).

Outside the block, or while a CUDA graph is being captured, nothing changes. The main model's graphs only hold
decode windows, whose ids are sampled tokens. The MTP head's graphs also serve prefill's absorb, so `Engine.mtp`
(`decode.py`) runs eager when the block is active and its rows carry virtual ids (a replay would read them as
vocabulary rows: an illegal memory access, found by the GPU test in W14). Text-only requests pay one attribute check.

The MTP head also absorbs the image rows, as vLLM feeds multimodal embeddings to its drafter. Replies never depend
on it: drafted == serial.

### Rank 1

Rank 1 gets rank 0's rows right after the prompt (`vision.exchange`), in both paths: `generate` / `follow` and the
batcher's `_plan` / `follow`.

- The order: the count and width, the sorted ids, then the rows as int32 words through the engine's all-gather, in
  chunks of 1,024 rows (8 MB).
- Both ranks decide from the prompt: a prompt without virtual ids exchanges nothing. Text requests keep the exact
  protocol they had, and rank 1 needs no knob.
- Rank 1 checks the ids against its prompt.
- Rank 0 encodes before the header travels. In batch mode that happens on the HTTP thread, on the tower's own CUDA
  stream, one request at a time. The batch loop keeps running rounds meanwhile.
- A background job's re-admission ships its rows again.

### Why the tower is on rank 0 only

A 1024 x 768 screenshot is ~5.4 TFLOP, about 50-100 ms on one GB10, against ~1 s of LM prefill for its 1,036 rows.
Splitting the tower would halve the smaller term and need an exchange inside every block. Not worth it.

## 4. Cost

**Memory on rank 0:**

- the tower: 1.127 GB BF16 (1.05 GiB), loaded after the LM weights and before the KV / session budgets are sized;
- transients while encoding:
  - a screenshot: ~0.1 GB;
  - the 8,000-token maximum (31,724 patches): ~0.5 GB (activations 64-200 MB, the row-chunked MLP 8,192 rows at a
    time, flash attention with no N^2 buffer);
- the caches: up to 256 + 256 MB of host memory, which is the same unified pool on GB10.

**Memory on rank 1:** only the rows of the requests in flight (8 KB a row: 8.3 MB for a screenshot, 64 MB for a
maximum image), plus 16 MB of exchange buffer.

**Under the 4 x 250k stress** (docs/MEMORY-4x256k.md):

- Rank 0 is not the binding node: it has ~2 GiB more MemTotal than rank 1. After the canary warm-up it had MemFree
  5.9 GiB against rank 1's 5.1.
- With the tower and the caches, rank 0 lands at ~4.3 GiB, still above rank 1. It sits closer to
  `GLM53_TF_BATCH_RESERVE_GB` (4), so admissions may be refused ~1.5 GiB earlier on rank 0.
- For stress configs, set `GLM53_TF_VISION_CACHE_MB=64 GLM53_TF_VISION_PREP_MB=64`, which saves 0.4 GB.
- Measure this in GPU step 4.

**Rows and FLOPs** (`Tower.flops`: matmuls + attention; 110 TFLOP/s is GB10's dense BF16 peak):

| Image | Grid (patches) | Rows | ViT TFLOP | Attention share | Ideal at 110 TFLOP/s | Est. at 55-65% | LM prefill of the rows (~1,000-1,300 tok/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 x 512 | 38 x 38 | 361 | 1.49 | 14% | 14 ms | 20-25 ms | ~0.35 s |
| 800 x 600 | 44 x 58 | 638 | 2.90 | 22% | 26 ms | 40-50 ms | ~0.6 s |
| **1024 x 768** | 56 x 74 | **1,036** | **5.37** | 31% | **49 ms** | **75-90 ms** | **~0.9 s** |
| 1280 x 720 | 52 x 92 | 1,196 | 6.50 | 35% | 59 ms | 90-110 ms | ~1.0 s |
| 1920 x 1080 | 78 x 138 | 2,691 | 20.9 | 54% | 190 ms | 0.3-0.35 s | ~2.3 s |
| 2560 x 1440 | 104 x 184 | 4,784 | 53.0 | 68% | 0.48 s | 0.75-0.9 s | ~4 s |
| 3840 x 2160 | 134 x 238 | 7,973 | 128 | 78% | 1.17 s | 1.8-2.2 s | ~6.5 s |

- A 1024 x 768 screenshot is padded, not resized: 1024 x 768 in a 1036 x 784 canvas.
- The tower is 5-10% of an image's time to first token. The image rows' LM prefill dominates.
- The rank-0 row cache makes a conversation's later turns free of tower work. The session store makes them free of
  prefill too.

## 5. Files

- **New** `glm5_next/cuda/vision_prep.py` (host): settings, the processor, fetch / decode, content parts, prompt
  placement, virtual ids, the registry, `Host`.
- **New** `glm5_next/cuda/vision.py` (engine): `Tower`, `Encoder`, `Table`, `active` / `embed_ids`, `exchange`.
- `glue.py`: `embed` substitutes image rows (`vision.ACTIVE`).
- `engine.py`:
  - loads the tower (rank 0);
  - `_vision_table`;
  - `exchange` after the prompt in `generate` / `follow`;
  - `active` around `_run`'s prefill;
  - `stats["vision"]`.
- `batch.py`: `Job.vision`, the exchange in `_plan` / `follow`, `active` around `_piece`'s prefill.
- `app.py`:
  - `VisionTemplate` (the sentinel rendering);
  - `check` (fetch + preprocess -> 400s; the context with the rows);
  - `prompt_ids` (placeholders -> virtual ids, `request.vision`);
  - `/tokenize` (rows as `<|image|>`).
- `decode.py`: `Engine.mtp` skips the head's graphs for rows with virtual ids (W14).
- `lookup.py`: `SuffixIndex.propose` stops at a virtual id.
- `draftvocab.py`: the fallback window skips virtual ids.

## 6. Tests (offline, CPU; 89 checks)

Run each with `PYTHONPATH=<patched tree>/src`. The real-tokenizer checks also need
`GLM53_TF_TOKENIZER_DIR=<checkpoint files>` and optionally `GLM53_TF_VISION_TEMPLATE=<the kit's template>`.

- **`tests/test_vision_prep.py` (43).**
  - **Processor parity:** `preprocess` == transformers' `Glm5NextImageProcessor` **bit for bit** on the patches and
    grid over 16 sizes, from 1 x 1 to 12000 x 9000 and 5000 x 20.
  - `smart_resize` == transformers' on 3,000 random (size, budget) draws.
  - `image_tokens` == `Glm5NextProcessor._get_num_multimodal_tokens` on 63 sizes.
  - Without torchvision the same bits.
  - The budget knobs.
  - Part shapes.
  - Data URLs; http fetch from a local server (404, size cap, `FETCH=0`); refused schemes.
  - Decode limits; RGBA on white; grey; GIF first frame.
  - Sentinel order; `expand` and its refusals.
  - Virtual ids: a function of the pixels (PNG == BMP; one pixel changes every row; a budget the image fits changes
    nothing).
  - The registry's re-salting.
- **`tests/test_vision.py` (14).**
  - `Tower` == transformers' `Glm5NextVisionModel` (tiny random config with every structural feature, clamps
    exercised): fp32 within 1e-5 relative, eager and SDPA, 5 grids; bf16 within bf16 rounding.
  - MLP chunking exact.
  - Loading a sharded safetensors checkpoint; refusing a text-only one.
  - `Encoder.table`: ids sorted with their rows, duplicates encoded once, the cache.
  - `Table.fixup` / `clamp`; `active` nesting.
  - `exchange` between two ranks in threads over a fake all-gather: bit-for-bit rows in chunks (NaN payloads
    included); nothing exchanged without image rows; mismatches fail on either side.
  - The FLOP counts above.
- **`tests/test_vision_interpreter.py` (5).** `glue.embed` in Triton's interpreter, BF16 and 4-bit tables, 4 copies
  and 1:
  - image rows in every copy, vocabulary rows elsewhere;
  - unchanged outside `active` and during graph capture.
- **`tests/test_vision_server.py` (27).**
  - Off by default (the reminder, no rows).
  - The prompt layout; multiple and duplicate images; `usage.prompt_tokens`; `/tokenize`; a URL fetched once a
    request.
  - HTTP 400s streamed and not: bad base64, undecodable, scheme, too many, a placeholder typed in the text,
    malformed part, context counting the rows.
  - **The caches:**
    - same text with a different image differs exactly at the image rows;
    - `GlmEngine._resume` / `Batcher._resume` stop before the image for another image and resume the whole prompt
      for the same one (and for its next turn);
    - `SessionIndex.find` / `lcp` / `marks`;
    - page keys equal before the image and different from it on;
    - entry keys;
    - `DiskIndex.find`;
    - `ids_b64` round trip.
  - The lookup drafter never proposes an image row.
  - With the real tokenizer: the whole prompt == `Glm5NextProcessor(images=...)` with the kit's template or the
    checkpoint's (1-2 images, 17 x 900 to 2000 x 2000).
- **Regression runs against the patched tree:** `test_api_context`, `test_request_log`, `test_openai_compat`,
  `test_serve_ops`, `test_prompt_tokens`, `test_glm_lookup`, `test_effort`, `test_glm_tool_calls`, `test_health`,
  `test_fastboot_*`, `test_draft_vocab_interpreter`, `test_fp8kv_interpreter`, `test_kvpool_interpreter`,
  `test_b12x_interpreter`, `test_kda_v2_interpreter`, `test_nonexpert_map`, `test_prefetch_comm`,
  `test_gpu_sampler_interpreter`, `test_gpu_round_compile`: all pass.
- **`tests/cuda/test_vision_patches.py` (GPU, not yet run):**
  - on the synthetic checkpoint, image rows set to real tokens' embedding rows reply exactly as those tokens do. It
    covers serial / drafted, greedy / sampled, prefill rows 1 / 16 / default / fast, image mid-prompt and in the
    last chunk;
  - resumed == fresh, and another image never resumes past the text;
  - the compiled embed and graph capture;
  - with `GLM53_TF_MODEL`: the real tower on a screenshot (1,036 rows, deterministic, bf16 vs fp32 < 2%, latency,
    peak memory).

## 7. GPU test plan (for the next Spark round; one agent owns the Sparks)

Build the image with the stack through 0500 (`PATCHES=""`). Every step records the image id and the knobs.

1. **Unit tests on the GPU** (either node, no server):
   `PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda pytest -q -s tests/cuda/test_vision_patches.py` with
   `GLM53_TF_MODEL=/root/.cache/huggingface/hub/models--neko-legends--GLM-5.3-Flash-Uncensored-EXL3/snapshots/07135ec082f8f11f7a71e4244a4e5167a0f96277`.
   - Pass: all green; screenshot tower latency printed (expect 75-90 ms); peak +0.1-0.2 GB; bf16 vs fp32 < 2%.
   - Also re-run the host tests in the image (`scripts/run_tests_in_image.sh`, the `test_vision*` files) to confirm
     torchvision 0.28 / torch 2.13 give the same processor bits.
2. **Text unchanged (exactness gate).** Start production (`config/prod.env`) with `GLM53_TF_VISION=1` added.
   - Run the canary and the exact 10/10 suite (`scripts/canary.py`, the usual 10 prompts, greedy and seeded): all
     replies byte-identical to the same image with `GLM53_TF_VISION=0`, and to the last recorded production run.
   - Repeat with `GLM53_TF_VISION=0` on the new image: identical (the knob off is upstream's path).
   - Boot lines: `[boot] vision tower 1.13 GB`, the `vision (patches/0500)` line; load time +1-3 s.
3. **Correctness on real images**, greedy, `reasoning_effort` low and max; and the same through the vLLM kit (vision
   on by default there: `LIMIT_MM` image 4) on the other stack slot, never both loaded at once:
   - **(a) Describe** a desktop screenshot (1920 x 1080) and a photo (1024 x 768 JPEG): the named objects, the
     application / window titles.
   - **(b) Read text:** a screenshot of a terminal with a known 3-line command and output, a receipt photo, a
     rendered paragraph at 12 px. Pass: exact strings reproduced (character error rate reported).
   - **(c) Count objects:** 3 / 7 / 12 coloured circles on white (generated), a grid of 5 x 4 icons. Pass: right
     counts on the generated ones.
   - **(d) Two images in one message:** "which is larger" on two generated charts; image-first vs text-first order.
   - **(e) Transparency:** a PNG with alpha (text on transparent), which must read as text on white.
   - **(f) URL input** from a local `python -m http.server` on the head, and `data:`.

   For each: TensorFold vs the vLLM kit. Same meaning required, not byte equality (different kernels). Also record
   `usage.prompt_tokens`: it must be equal on both (same processor and template; any difference is a bug).
4. **Caching exactness on the GPU.**
   - The same conversation (image + question) twice: the second has `cached` >= its prompt minus the generation tail
     (session store with `GLM53_TF_SESSION_GIB` > 0 and in batch mode) and a byte-identical reply (greedy).
   - Two different images with the same text back to back: the second's `cached` <= the text before the image, and
     its reply is the same as when it is sent first on a fresh server.
   - A 3-turn conversation with an image in turn 1: turns 2-3 resume past the image. The tower does not run again
     (`stats.vision.encoded` 0).
   - Restart with the NVMe tier (0250) and repeat turn 3: restored from disk, byte-identical.
5. **Memory and latency.**
   - The 4 x 250k stress (`scripts/rigmark` / the stress bench of docs/MEMORY-4x256k.md) with `GLM53_TF_VISION=1`
     and no images: MemFree / MemAvailable on both ranks against the last stress run. Expect rank 0 -1.1 GiB (tower)
     -0.5 GiB after the first images (caches); rank 1 unchanged. Admissions unchanged or a stated delta.
   - Then with one image request among the four streams: no OOM, the others' decode tok/s dip only during the tower
     (< 0.1 s) and the image's prefill.
   - Latency per image: TTFT for 512², 1024x768, 1920x1080 and 3840x2160, split into `stats.vision.vision_s` and
     prefill. Compare with the table in §4 and with the vLLM kit's TTFT on the same image.
6. **Failure paths:** a 30 MB image (400 `MAX_BYTES`), a 20k x 20k PNG (400 `MAX_PIXELS` before decoding), a dead
   URL (400 after the timeout), 9 images (400). The server stays healthy (`/health` ok), and rank 1 never desyncs:
   the next text request is served.

## 8. Limits and follow-ups

- **Video:** the template's reminder stays. The processor, `get_video_features` and timestamps are not ported.
- **Placeholders in text:** `<|image|>` typed by a user in a request with images is refused (HF would mis-align).
  In a text-only request it is left as the token, as before.
- **Across restarts** a restored session relies on the tower being deterministic: the same kernels on the same image
  give the same rows. Step 4 checks it.
- **The tower is plain PyTorch** (cuBLAS + flash SDPA). A fused Triton path is not needed at 5-10% of TTFT.
- **0430 dumps** record virtual ids in `tokens`.
- **In-flight patches that touch `glue.embed`, the lookup or the batch admission** must keep the `vision.active` /
  `exchange` calls. 0500 applies after 0490, and after 0450.
