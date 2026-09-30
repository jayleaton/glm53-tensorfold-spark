# Upstream TensorFold 0.5.0 against our GLM engine (0.3.4 + patches 0001-0580, prod image b9), 2026-09-30

> Offline audit: no GPU, the Sparks were not touched, the submodule pin (`2f8e514`, 0.3.4) is unchanged. Upstream was
> read in a separate clone (`~/.cache/tf-upstream-050` at `9cd52ab`, 0.5.0). It covers the 55 commits
> `71377a5..9cd52ab` (0.3.6.3, 0.3.7, 0.4.0, 0.5.0) that follow the 0.3.6.2 audit (`docs/UPSTREAM-0362-AUDIT.md`).
> "Ours" means 0.3.4 with `patches/*.patch` applied in order, as the Dockerfile applies them (built at
> `~/.cache/tf-ours-b9`). `patches/` ends at 0580 at `e52b0d6`: there is no 0590 in the tree yet, so this audit is
> against 0001-0580. Speed numbers are **estimates** from our W11-W17 measurements unless marked as upstream's.

## 0. Bottom line

- **Almost nothing in 0.5.0 reaches our GLM CUDA decode or prefill.** In `71377a5..9cd52ab`, `glm5_next/cuda/`
  changes only in `app.py`, `decode.py`, `drafter_choice.py`, `engine.py` (grammar hooks, `min_p`, a sampler branch
  for `top_k` 0), and `split.py` / `weights.py` (checkpoint load path). `forward.py`, `exl3.cu`, `qmm.py`, `latent.py`,
  `sparse.py`, `mtp.py`, `dflash2.py`, `kda.cu` are untouched. The `glm5_next` `runtime.py`, `weights.py`, `mlp.py`,
  `prompts.py` and `__init__.py` changes are the Mac (MLX) lane engine. **Decode 0%, prefill 0%** from porting
  upstream code (section 5).
- **The one new feature worth building is structured output** (`response_format` json_object / json_schema,
  `guided_json`, `structured_outputs`). Upstream's design (a per-row xgrammar mask on each verify window; drafts the
  grammar rejects cut before the forward) **is exact under our speculative decoding**: a row's logits do not depend on
  the other rows (our row-invariance contract), its mask depends only on its own path, and our draw is keyed by
  (seed, position, token). So a constrained drafted reply equals its constrained `"draft": false` reply, and a
  constrained slot in a 4-slot round equals itself alone. Upstream wires it only into GLM's one-request loops (and has
  no GLM test); **production runs the batcher (`GLM53_TF_BATCH=4`), which upstream does not have**, so we write our own
  port on `batch.py`. Effort ~1.5-2.5 days offline + ~1-2 h GPU. Cost: 0 for unconstrained requests (no code path
  changes), est. <= 1-2% a round for constrained ones, ~0 if the bitmask fills overlap the forward (section 2.1).
- **Fixes worth taking are all host-side and keep today's bits:**
  - client disconnect (24afe5e, adapted to the batcher): a non-streamed request whose client left today runs to
    `max_tokens` (32,768 in prod: ~5-14 min of a slot), and a queued / prefilling one is detected only at its first
    written delta;
  - request hardening (2bc35c4): an exception in `App.check` (a template `raise_exception`, a non-dict
    `chat_template_kwargs`, a non-UTF-8 body) is uncaught in our `do_POST`, so the client gets a dropped
    connection, not a 400;
  - `kill -USR1` stack dumps on the CUDA server (a586f3d): our `cli.cmd_serve` returns into `_serve_cuda` before
    registering the handler, exactly the bug upstream fixed;
  - image URL hardening (391713e): our 0500 fetch takes `http` and any port, follows redirects and does not check
    the address, which is a blind SSRF from any prompt an agent forwards. Upstream's HTTPS-443-only, public-address,
    re-validated-redirect fetcher ports nearly verbatim.
- **Already ours, or superseded by ours:**
  - 7d27a19 (stale extension-build lock): `docker/entrypoint.sh` already deletes stale locks at start.
  - da8a5df (pinned staging released): 0140's `_release_pinned` already calls `torch._C._host_emptyCache()`.
  - 8ae247f for restarts: 0140 prepared folders already load at ~9 GB/s over O_DIRECT; 8ae247f only speeds a first
    start from the checkpoint.
  - 0768f8d (Qwen prompt-end entry one token early): same idea as our 0540 (snapshot strictly before the end, on the
    grid).
  - 88407d1 (stop strings) for GLM: upstream's GLM decodes on to EOS and only cuts the text; our 0160 stops the batch
    engine at the next round.
  - 603d3bc "replies keep decoding while prompts prefill": Mac only; our 0120 / 0335 / 0560 already do it on CUDA.
- **A future rebase grows a little.** The squashed series merged onto 0.5.0 conflicts in 22 files / **115 regions**,
  against 21 / 106 on 0.3.6.2 (same series 0001-0580). New: a `cuda/health.py` add/add (0150 vs c4bf25f); the server
  split into `cuda/http.py` + `cuda/chat_template.py`; the engine's rank-1 header (`min_p`, grammar flag); and
  `decode.sample_rows`. **Recommendation unchanged: no rebase; port individually** (section 6).

## 1. Every change touching the GLM CUDA path or the shared CUDA server

Import trace, as in the 0.3.6.2 audit: our two-Spark server is `tensorfold serve` -> `cli._serve_cuda` ->
`tensorfold/cuda/server.py` (ours: 0.3.4 + 0150 / 0160 / 0210 / 0490) + `glm5_next/cuda/app.py` + the engine. At 0.3.4
the shared `tensorfold/cuda/` package holds only `server.py` and `__init__.py` (our 0150 adds `health.py`). Every other
shared CUDA module upstream (`build.py`, `direct_read.py`, `kernels/*`, `sampling.py`, `http.py`, `turns.py`, ...)
exists only after a rebase or a port.

Legend for the last column:

- **GLM CUDA**: code upstream's own `glm5_next` CUDA engine runs.
- **ours**: whether it would change anything for our b9 stack.

| commit(s) | what it does | GLM CUDA / ours |
| --- | --- | --- |
| **98fb8f9, 7ae9da2, f41bcd3, 7f9c8d3, 1da4b7e, 996e38b** | Structured output. `engine/grammar.py` (xgrammar, optional extra `tensorfold[grammar]` = `xgrammar>=0.2.8,<0.3`) reads `response_format` (json_object / json_schema), `guided_json/regex/choice/grammar` and `structured_outputs`, and compiles them (cached by text, 256 MiB). A `Constraint` walks a verify window depth first, drops drafts no accepted path can hold, and fills each kept row's allowed-token bitmask. `mask()` sets the other tokens to -inf before sampling; `advance()` follows the chosen tokens. With thinking on, it starts after `</think>`. 400 for malformed / non-compiling grammars, or one beside a required tool call; a mid-reply grammar failure is a 500 / stream error. 1da4b7e wires it into GLM's `decode.py` (`Engine.sample` masks, `verify_window` cuts, `follow` advances; prefill's first token masked), `drafter_choice.auto_decode`, and `engine.py` (header flag, `pack()`ed grammar shared to rank 1, which compiles the same) | **GLM CUDA: yes, one-request path only** (serial / MTP / DFlash2 / auto loops). No GLM test upstream (tests cover the 27B, Flash Next and Nemotron). **Ours: not present; a request's `response_format` is silently ignored today.** Our prod path (the batcher) needs its own port (section 2.1) |
| 88407d1 (+ bb69403, 914f2d9) | Stop strings on every CUDA engine, cut token by token (`StopStrings.hit`), so drafted and serial replies give the same content, usage and `token_sha`; matched on the whole generated text, reasoning included. `ignore_eos` and `stop` validated before a stream opens. GLM's app sets `reads_ignore_eos`. **For GLM, upstream decodes on to EOS or the limit and only cuts the returned text** | **Ours already stops**: 0160 cuts at a stop string and returns True from the callback, and the batcher cancels the slot at its next round. Two differences remain: ours matches only the visible answer (not reasoning, not tool-call text), and our `completion_tokens` under a stop counts the tokens received when the match was seen (round-granular; drafted and serial can differ by up to one round). Port upstream's per-token count (section 3) |
| 2bc35c4 (+ d448640) | The CUDA server answers malformed requests with 400 and failed ones with 500 or a stream error event. It validates `chat_template_kwargs` (object or null), turns a template refusal into "the chat template rejected the request: ...", parses temperature / top_p / top_k / seed strictly (no booleans, finite, integral) **before** a stream's headers, and takes a non-UTF-8 body as "not JSON" | **Ours partly**: 0150 / 0490 already give JSON 500s, SSE error events and 400s for `ValueError` from `run`. **Gap:** `do_POST` calls `app.check(body)` outside any `try`, and `GlmApp.check` renders the template, so a template `raise_exception`, a string `chat_template_kwargs` (`dict("x")` raises) or a non-UTF-8 body (`UnicodeDecodeError` is not `JSONDecodeError`) drops the connection with no response. A streamed request with a bad `temperature` gets 200 + role chunk + error event instead of a 400 |
| 24afe5e family (in 0.3.6.2, not ported yet) | Stop a request when its client has gone: a socket check every round (streamed and not), queued requests of departed clients never start, the per-token callback never raises into the engine | **Ours partly** (section 3.1) |
| 51b098d, c4bf25f | Live token totals on `/health` (requests / prompt / completion tokens, prefill / decode seconds, cached, rounds, drafted, accepted; completion tokens move as emitted), a new `cuda/health.py` | Server only. **Ours: 0150 has the same counters on `GET /metrics`** (Prometheus), finished requests only. Upstream's JSON field names on `/health` would help pollers written for upstream (MiaAI dashboards). Low value; a rebase add/add |
| 1330709 | Chat template renderer moved to `cuda/chat_template.py` | Refactor, no behaviour. Rebase conflict with our 0160 template code in `server.py` |
| 9cd52ab (release) | `cuda/http.py` (handler split out; `/v1/responses`); `server/responses*.py` (**OpenAI Responses API**, translated to a chat completion, `previous_response_id` store); `min_p` in `Sampling` / `choose_rows` / the GLM header; **`top_k` 0 draws the whole nucleus across ranks** (`cuda/sampling.nucleus_rows`: fixed-point mass so shard sums are exact, a second all-gather of 1,024 candidates a row, whole shards when they don't cover it), wired into GLM `decode.sample_rows`; `thinking_budget` (`call_gate.ThinkBudget`) and `reasoning_effort` read as on the Mac; `priority: background` yields between rounds (`cuda/turns.py`); `glm5_next/prompts.thinking_off` (the same rule as our `ThinkingOffTemplate`); prefill attention CUDA kernel (`kernels/prefill_attention.cu`, same bits as the Triton `_attend`) | Server parts: ours lacks Responses, `min_p`, `thinking_budget`. **`top_k` 0 is a real semantic gap in ours:** `sample_rows` takes `k = top_k + MARGIN = 8` candidates a rank when `top_k` is 0 (also for `-1`, clamped to 0), so the draw is over the top 16 tokens, not the nucleus. Deterministic and exact, just truncated. `prefill_attention.cu` is GLM's per-head (non-latent) prompt path only: not ours (`GLM53_TF_LATENT_KV=1`). Background yielding: ours has it (0120) |
| 8ae247f, da8a5df | CUDA startup / loading: `cuda/direct_read.py` (O_DIRECT reads, 8 in flight, neighbouring tensors in 128 MiB runs, uploads on a side stream), GLM `split.RankReader` rewritten on it (a rank's rows read as one run, device-side splits), GLM `weights.load` prefetching two layers of expert tensors, `empty_cache` every 8 layers instead of every layer; da8a5df gives the pinned staging back (`_host_emptyCache`) | GLM CUDA: yes, **checkpoint path only**. Ours restarts from 0140 prepared folders (O_DIRECT, 8 threads, pinned double buffers, ~9 GB/s, `_release_pinned` already there). Only a first start or a prepared-key miss reads the checkpoint (8+ min, dominated by q4mse re-quantization, calibration and writing the folder). Est. saving <= ~1 min of that, rarely. Conflicts with 0001 / 0140 in `weights.py` / `split.py`. **Skip** |
| 7d27a19, 5fd5bf7, a927ba5 | `cuda/build.load`: "building extension X" lines, and a named stale `lock` with the fix | Ours builds 10 extensions with `cpp_extension.load` directly; **`docker/entrypoint.sh` already deletes stale locks** at start (the container is the volume's only compiler). Only the "building X" line is new: cosmetic |
| a586f3d | `kill -USR1` registers `faulthandler` before the CUDA branch of `cmd_serve` | **Ours has the bug** (`cli.py`: `faulthandler.register` sits after `return _serve_cuda(...)`). 15-minute port, useful for a hung rank (RoCE wait, NCCL) under the watchdog |
| 391713e (+ 05cbb54, b4042e8) | Image URLs HTTPS on port 443 only, no fragment / credentials, public addresses only (DNS checked and the connection pinned to the checked address), redirects re-validated, `Content-Type` JPEG / PNG / WebP only, identity encoding; at most 16 image preparations at once, 128 waiting, then 503 (`CapacityError`); request log redacts images | Upstream's vision is Qwen's (`vision/`); ours is 0500's own GLM vision. **Our fetch** (`vision_prep._image_bytes`): `urllib.request.urlopen` with `http`/`https`, any port, redirects followed, no address check. Port the fetcher (section 3). Our 0300 log records no text or image data, so the redaction is moot |
| fe2b514, c30a1a9 (tool args part) | Typed tool parameters by schema; #87: an object / array argument one closing bracket short is closed (`closed_json`) if it then parses | GLM's parser upstream (`parse_glm_tool_call_block` -> `decode_parameter`) gets #87. Ours (0002 `_glm_call`): JSON-first for non-string types, no bracket repair. 30-minute port |
| b29df86 | Resume points for templates that drop a reply's think block from history | `engine/prefill_plan` via `cuda/markers.py`: Qwen CUDA engines and the Mac only. Commit says GLM-4.5-5.3 markers unchanged. Not ours (0110 / 0310 / 0540 mark and resume) |
| 0768f8d, a1e2815, 7121d76 (+ tests) | Qwen 27B prompt-end cache entry at `len(prompt) - 1`; GDN state freeing; the 27B's concurrent drafter context | `qwen3_5` only. The idea is our 0540 (`(n - 1) // G * G`), which also covers the case 0768f8d targets (a turn sent back without its reasoning differs in its last token) |
| 603d3bc | Flash Next NVFP4 on CUDA; Mac prompts in planned chunks with decode rounds between (`--decode-share`); `glm5_next/mlp.py` sorted expert gather on M1-M4 | Mac / Flash Next. Shared `kernels/qmm*` gained an FP8-weight prefill entry (`qmm_prefill8w`, NVFP4); the lane matmul itself is unchanged |
| 76ad10f, 293752c, 2dc3454, 3655360, 1c5ea53, 1663756, f925559, 9b8bc3a, 2c984cf, 756a125, 7399e7c, fe941b6, 95e5b9a, a2dba9c, ac3de87, af9639f, 05cbb54, 1911880, 6b2e4c4, 7a00336 | NVFP4 (Flash Next, 27B), Qwen3.6 `--parallel`, Flash Next QSA tiles, DeepSeek-V4 on Macs, Bonsai, MLX 0.32.3, capacity estimates (`cuda/capacity.py`: GLM's upstream admission, which we replace), releases | Not on our path |

## 2. Features worth porting as our own patches

### 2.1 Structured output for GLM, exact under our speculative decoding

#### How upstream does it for drafted rows

The unit is a **verify window**: row 0 is the pending token, rows 1.. are drafts, each with a parent (GLM: a chain,
`parents = [-1, 0, 1, ...]`). Upstream handles it in four steps:

1. **Before the forward, `Constraint.window(tokens, parents)` walks the window depth first on xgrammar's matcher.**
   - At each row whose path is inside the grammar, `fill_next_token_bitmask` writes that row's allowed tokens (int32
     words, V bits).
   - For each child, `accept_token(draft)`. A draft the grammar rejects is dropped with everything under it: the
     parent row's masked distribution could never choose it. A draft that terminates the grammar (its stop token) ends
     the path.
   - It rolls back on the way up, so the matcher ends where it started (the committed state).
   - With thinking on, rows before `</think>` are not constrained; the grammar starts at the row after it.
2. **The forward runs on the kept rows only.** `drafts = tokens[1:]` is replaced by the cut window.
3. **After the forward, `mask(logits, window, vocab_offset)` sets every constrained row's disallowed columns to
   -inf.** On two ranks each rank masks its own vocabulary half. Both ranks compiled the same grammar: rank 0 `pack`s
   (kind, think end, text) into the request header stream, and rank 1 compiles it with `Grammars.follow`. Then the
   usual keyed draw runs.
4. **The kept tokens go to `advance()`**, including the bonus / correction token.

The first token after prefill uses a one-row window `window([0], [-1])`.

#### Why it is exact for us

The contract is drafted == serial, batched == alone, resumed == fresh. It holds for four reasons:

- **A row's logits do not depend on the other rows of its window or round.** This is our row-invariance contract:
  0085 / 0200 / 0290 tests, 0440 / 0520 / 0580 "same bits".
- **A row's mask depends only on its path** (committed tokens + the drafts above it). For every row that is actually
  kept, that path is the serial prefix at that position, so the mask equals the serial step's mask. Rows past the
  first mismatch get wrong-path masks, but they are discarded anyway.
- **The draw is keyed by (seed, absolute position, token id)** (`choose_rows`), so equal masked logits give the same
  token.
- **Cutting a window changes only R (its row count).** Our kernels and CUDA graph keys are row-count invariant, and
  0280's padded rows are never kept.

Two details that hold but need a test:

- **`torch.topk` over a masked shard.** When a rank's shard has fewer than `k = top_k + 8` allowed tokens, topk
  returns -inf entries whose ids are arbitrary (and may depend on R). They never win:
  - their Gumbel score is -inf;
  - their top_p probability is 0, and they sort last;
  - greedy's lexsort picks the finite maximum.

  At least one allowed token exists on some rank while the grammar is live (xgrammar allows the stop tokens at its
  end).
- **A resumed request's first token.** It must come from a masked row. With 0540, a whole-prompt snapshot sits
  strictly before the end, so the first token always comes from a fresh prefill row. Test resumed == fresh under a
  grammar anyway, including the NVMe tier and prefix share.

#### What our port must add (upstream does not have the batcher)

Prod is `GLM53_TF_BATCH=4` (0120 / 0200 / 0340 / 0560).

| site | change |
| --- | --- |
| server (`cuda/server.py`, `glm5_next/cuda/app.py`) | `grammar.request_spec` + compile in `check()` (400 before headers: malformed, xgrammar missing, compile error, beside `tool_choice` required); think end = `</think>`'s id when thinking is on; the `Constraint` handed to `engine.generate(..., constraint=)` |
| `engine/grammar.py` | upstream's module verbatim minus `_mask_mlx`; `for_model` from `tokenizer.json`, logits width = `config.json` `vocab_size` (154,880), stop ids = the engine's eos |
| rank 1 | batch: after a job's header (`batchplan.encode_header` gets a grammar flag), `g._share(pack(c))` beside 0500's `vision_mod.exchange`; `_job_from` compiles it. Lone (`BATCH=1`) path: upstream's header layout (a flag + one more `_share`) |
| `batch.Stepper` | after `propose()` (and after `MtpChains.run` when the chains are batched), cut `self.drafts` through `constraint.window`, before `_verify` stages the windows; keep the `Window`; `accept()` then `advance(sampled[:keep])` |
| `batch.sample_multi` | before each spec's `topk`, `constraint.mask(logits[off:off + R], window, w.vocab_offset)`; the rider / 0370 plan-ahead exchange is unchanged (masking is local to each rank) |
| first tokens | `batch._piece`'s last piece, `mpf` (0560: `mpf.py` `e.sample(self.last, ...)`), lone `decode.prefill`: a one-row window |
| lone loops | `decode.serial_decode / mtp_decode / dflash_decode`, `drafter_choice.auto_decode`, `decode_v2` (0130): upstream's hooks |
| 0450 GPU round | resident rounds run several rounds with no host in the loop, so a constrained slot is not eligible (`resident.eligible`). The device sampler is fine after the mask (check its -inf handling). Both are off in prod |
| errors | a `GrammarError` on rank 0 cancels that slot through the next plan (both ranks walk the same matcher; rank 1 never ends a request on its own). The other slots go on |
| knob | `GLM53_TF_GRAMMAR=0\|1` (default 1 once validated; 0 = today's behaviour: the fields are ignored). It enters the 0140 calibration key: one re-measure (~100 s) on the first start |

Two refinements over upstream, both keeping the bits:

- **Hide the host cost.** Only the accept / rollback walk (the cut) must precede the forward. The bitmask fills can run
  on the host while the GPU runs the forward, because the forward does not read them; only `sample_multi` does.
  Upstream fills before the forward.
- **Mask the drafters too (phase 2).** Mask the MTP / DFlash2 draft logits with the draft path's bitmask. Drafts that
  the grammar would cut are then never proposed, which raises acceptance on JSON bodies. Drafts never change a reply's
  bits.

#### Cost, effort, exactness

- **Unconstrained requests:** no code path changes (the constraint is None). Reply SHA `8794a3463259cc2f` must not
  move; gate it.
- **Constrained requests:**
  - **Host work.** xgrammar's JSON-schema mask fill is tens of us a row (its published figure is under ~40 us at a
    128k vocabulary; measure on GB10's cores). A 16-row window is ~0.3-0.7 ms of fills plus the accept walk. Both
    ranks do it in parallel.
  - **Device work.** Upload 19.4 KB a row (4,840 words), then unpack and `masked_fill` in 2-4 small launches:
    ~0.05-0.1 ms.
  - **Total.** Against a 50-120 ms round that is **+0.5-1.5% as upstream does it, ~0.1-0.3% with the fills during
    the forward**. The first structured request pays 1-2 s to build the tokenizer info (cached); a new schema 10-100
    ms to compile (cached by text).
  - **Acceptance.** Cut drafts lower tokens a round slightly; phase 2's draft masking should recover it.
- **Thinking.** GLM thinks by default (effort `high`), so the grammar applies after `</think>`. The MiaAI-Lab result
  in our `app.py` docstring (thinking on at `low` effort is the reliable setting for structured output) still
  applies. `thinking_budget` (2.2) bounds the reasoning before the JSON.
- **Dependency.** `xgrammar>=0.2.8,<0.3` in the image. Check that an aarch64 cp312 wheel exists for the 26.07 base,
  else build from source in the image (C++ / nanobind).
- **Effort:**

  | part | hours |
  | --- | --- |
  | grammar module | ~1 |
  | server + app | 3-4 |
  | batcher, rank 1, lone loops | 5-7 |
  | CPU tests | 4-6 |
  | image | ~1 |
  | GPU window | ~1-2 |

  **Total ~1.5-2.5 days offline, ~1-2 h GPU.**
- **CPU tests.** Upstream's `tests/cuda/toy_grammar.py` idea on our hash-model fakes:
  - constrained drafted == `"draft": false`, greedy and sampled;
  - 4 slots mixing constrained and plain == each alone;
  - two gloo ranks;
  - resumed == fresh;
  - a grammar error ends only its own slot.
- **GPU gates:**
  - the exact suite and batchexact;
  - the reply SHA;
  - a JSON-schema suite (N schemas x greedy / sampled x draft / serial x alone / 4-slot), with `json.loads` +
    `jsonschema` validity 100%;
  - the round-time overhead.

### 2.2 Smaller features

| feature (upstream) | what we'd get | effort | exactness |
| --- | --- | --- | --- |
| `top_k` 0 = whole nucleus across ranks + `min_p` (9cd52ab: `cuda/sampling.nucleus_rows`, `exact_sampling` min_p) | `top_k` 0 / -1 (vLLM's "off") draws from the real top_p nucleus instead of our top-16 truncation; `min_p` for clients that send it (Open WebUI, SillyTavern). The nucleus gather must go through our `comm_mod.fast_gather` (0230). Its packed candidates are 16 rows x 3,074 int64 = 393 KB a rank, above `GLM53_TF_ROCE_MAX_KB=256`, so it takes the NCCL fallback unless trimmed. Est. +1-3 ms a round **only for `top_k` 0 requests** | 3-5 h (+ 0450's device sampler: refuse, i.e. host path, for top_k 0) | changes bits only for `top_k` 0 / `min_p` > 0 requests; drafted == serial kept (the same keyed rule on every path) |
| `thinking_budget` + `tool_choice: "required"` / named (0.3.6.2 `engine/call_gate`, 9cd52ab `ThinkBudget`) | cap GLM's reasoning at N tokens, then force `\n</think>\n\n` and continue; enforce a required call. Both run as `generate_gated`: the reply is cut and continued as a new request whose prompt is prompt + reply + forced tokens. On our batcher that is a finished job + a resubmit. **Caveat for prod:** 0180 saves a reply snapshot only for exact (grid 0) requests, and fast and exact snapshots never mix. So a prod (fast-prefill) continuation resumes from the prompt's snapshot and prefills the reply so far: est. +1-3 s for 2-8k reasoning tokens, once a gated reply | ~1 day | a gated reply is a well-defined sequence of requests, each equal to a fresh request with that prompt; ungated requests unchanged |
| OpenAI Responses API (`/v1/responses`, 9cd52ab) | Codex-style clients; runs as a chat completion, `previous_response_id` store | ~1 day (host; upstream's module targets its `http.py` handler shape) | host only |
| `return_token_ids` (e6be5ff, from the 0.3.6.2 audit, still not ported) | reply ids in the `tensorfold` block, for RigMark / exact harnesses through the OpenAI path | 0.5 h | host only |
| `/health` live totals (51b098d, c4bf25f) | upstream's JSON keys (`requests_running`, `completion_tokens_total` including live replies, `drafted_total` / `accepted_total`) beside 0150's `/metrics` | ~1 h | host only |

## 3. Fixes worth porting

### 3.1 Client disconnect (24afe5e, adapted to the batcher)

What ours does today (b9):

- **Streamed.** `emit` returns False on `BrokenPipeError` / `ConnectionResetError`; `on_tokens` returns True; the
  batcher's `_collect` sets `job.cancel`; the next plan cancels the slot.
- **Non-streamed.** `emit` is `lambda delta: True`: **never detected**. A request abandoned by a timed-out client
  runs to its end or `max_tokens` (prod 32,768: ~5-14 min of one of four slots at 40-100 tok/s).
- **Silent phases are not detected.**
  - While a request waits in the queue or prefills (minutes at 250k-1M tokens), no delta is written, so no write
    fails.
  - While `hide_tool_calls` holds back a tool call being written (no delta until it closes), the same.
- **An exception inside `on_tokens`** (the stop-string / tool / reasoning code, a write error other than the two
  caught) propagates out of `_collect` without setting `job.cancel`. The slot then decodes on unheard, and the job's
  queue fills.
- **What already works.** Rank 1 cannot desync from this: in batch mode the callback runs on the HTTP thread, not in
  the decode loop. Upstream's "never raises into the engine" part matters only for our lone path (`BATCH=1`), where
  0370's `Emitter` or the loop calls it.

The port:

- `server/cancellation.socket_cancellation` (upstream module, 40 lines: `select` + `MSG_PEEK`) for every request.
- `Batcher._collect` polls `job.out.get(timeout=0.25)` and sets `job.cancel` + notifies when the socket is gone. This
  covers queued and prefilling requests: `_plan` already drops cancelled queued jobs and cancels active slots,
  prefilling ones included.
- `on_tokens` wrapped: any exception is kept, `job.cancel` is set, and it is re-raised after `generate` returns.
- Streamed writes treat any `OSError` as a departed client.
- A departed client gets nothing more (no final chunk), as upstream.

**Effort 2-3 h + CPU tests** (upstream's `tests/test_cuda_server_disconnect.py` cases on our batcher fake). Same bits
(it only stops requests earlier).

### 3.2 400 / 500 handling (2bc35c4, the parts we lack)

- Wrap `app.check(body)` in `do_POST`:
  - a `jinja2` `TemplateError` becomes 400 "the chat template rejected the request: ...";
  - any other exception becomes 400 with its message, logged with the traceback (upstream's rule).
- `chat_template_kwargs`: object or null only (today a list of pairs is accepted, and a string raises, uncaught).
- Parse `temperature` / `top_p` / `top_k` / `seed` (and `min_p` if ported) in `check()` with upstream's
  `parse_numbers` rules (no booleans, finite, integral where integral), so a streamed request is refused before its
  headers. Today a bad value is a 200 + role chunk + error event; `True` is read as 1.0; NaN reaches the sampler.
- Non-UTF-8 body: catch `UnicodeDecodeError` beside `JSONDecodeError`.
- **Effort 2-3 h + tests** (port the relevant half of upstream's 202-case `tests/test_cuda_server_errors.py`). Same
  bits for every valid request.

### 3.3 Stop strings: token-exact length (88407d1, the part we lack)

- Our 0160 already cuts and stops, as described in section 1. Add upstream's per-token `StopStrings.hit` inside
  `on_tokens`, so `hit["at"]` is the token that completes the match, not the end of the round that delivered it. Then
  `usage.completion_tokens` is equal between drafted and `"draft": false` replies. The batch engine's own `sha256`
  (all decoded tokens) already is.
- Keep our matching scope (the visible answer). That is the OpenAI / vLLM-like behaviour our clients were tested
  with. Document the difference from upstream (which matches reasoning too).
- **Effort ~1 h.**

### 3.4 Image URL fetch hardening (391713e)

Replace `vision_prep._image_bytes`' `urllib` fetch with upstream's `vision/images_http.fetch_image` (185 lines, no
Qwen dependency):

- HTTPS on 443 only;
- no credentials, fragments or control characters;
- DNS on a bounded pool with every address required to be public (`is_global`, not multicast / reserved / 6to4 /
  Teredo / mapped, no cloud metadata);
- the connection pinned to the checked address with TLS verified for the host;
- each redirect re-validated;
- a whole-request deadline;
- identity encoding, a declared JPEG / PNG / WebP type and a byte cap.

Also take its bound on concurrent preparations (16, then 128 waiting, then 503), since our 0500 decodes and resizes
on HTTP threads. Keep `GLM53_TF_VISION_FETCH=0` as the off switch. **Effort 2-3 h.** Host only; images that fetch
today from public HTTPS hosts behave the same.

### 3.5 `kill -USR1` stacks (a586f3d)

Move `faulthandler.register(signal.SIGUSR1, all_threads=True)` above `if backend == "cuda": return _serve_cuda(...)`
in `cli.cmd_serve` (both ranks). Then `docker exec glm53-tf-rN kill -USR1 1` prints every thread's stack of a hung
rank (RoCE / NCCL waits, a stuck batch loop) into `logs N`. **15 min.**

### 3.6 Tool arguments one bracket short (#87, in c30a1a9)

In 0002's `_glm_call`, when `json.loads` fails for a non-string parameter whose schema type is `array` / `object`,
try upstream's `tool_parameters.closed_json` (close the open brackets outside strings) before falling back to the
text. **30 min.** Host only.

### 3.7 Not worth porting

- **8ae247f / da8a5df (load path):** section 1; 0140 covers restarts and already frees pinned staging.
- **7d27a19 (build lock):** our entrypoint deletes stale locks. Optionally copy the one-line "building CUDA extension
  X" notice into our 10 `cpp_extension.load` call sites (cosmetic).
- **0768f8d / b29df86 / 603d3bc / 7121d76:** other families or the Mac; our 0540 / 0120 / 0560 are the GLM CUDA
  equivalents.

## 4. What supersedes or conflicts with our patches

### 4.1 Supersession

- **Upstream supersedes nothing new of ours in 0.5.0.** The 0.3.6.2 list stands: 0002 by the shared GLM parser,
  0004 mostly, and 0050 / 0060 in design only.
- **Ours supersedes upstream here:**
  - 0140 over 8ae247f (restarts) and da8a5df;
  - the entrypoint over 7d27a19;
  - 0540 over 0768f8d's idea;
  - 0160 over 88407d1 for GLM (ours stops the engine; upstream's GLM decodes on);
  - 0120 / 0335 / 0560 over 603d3bc's decode-while-prefill (Mac only upstream);
  - 0150's `/metrics` over most of 51b098d.

### 4.2 Semantic differences to keep in mind

| topic | ours (b9) | upstream 0.5.0 |
| --- | --- | --- |
| stop strings match | the visible answer only | the whole generated text, reasoning included |
| `completion_tokens` at a stop | round-granular | token-exact |
| `top_k` 0 / -1 | top 8 a rank (16 candidates) | the whole top_p nucleus |
| default effort | `GLM53_TF_DEFAULT_EFFORT=high` (0150) | the template's own default |
| `response_format` | ignored | enforced, or 400 |
| images | 0500 GLM vision on the engine's `vision` attribute | the generic `App` takes `engine.vision` and routes images through Qwen's `prepare_images`, which would mis-route ours after a rebase |

### 4.3 Conflicts on a future rebase

Measured with `git merge-tree --write-tree` of the squashed series (0.3.4 + 0001-0580) against each base:

| base | files | conflict regions |
| --- | ---: | ---: |
| 0.3.6.2 (`71377a5`) | 21 | 106 |
| 0.5.0 (`9cd52ab`) | 22 | **115** |

Per file on 0.5.0: `forward.py` 22, `decode.py` 14 (+3), `engine.py` 13 (+4), `server.py` 11, `weights.py` 6,
`glue.py` / `latent.py` / `mtp.py` / `sparse.py` 5 each, `app.py` 4 (+1), `graphs.py` / `qmm.py` 4, `exl3.cu` /
`kda.cu` / `dflash2.py` 3, `exl3_mm.py` 2, **`cuda/health.py` 1 (new add/add)**, and 1 each in `attention.py`, the
GLM recipe, `pyproject.toml`, `THIRD_PARTY_NOTICES.md` and `tests/cuda/test_glm_engine.py`.

Beyond the text, five new semantic breaks add to the 0.3.6.2 list:

1. **The server split.** The handler moved to `cuda/http.py` and the template to `cuda/chat_template.py`. Our 0150 /
   0160 / 0490 / 0500 handler and template hunks land in files that no longer hold that code.
2. **`cuda/health.py` add/add.** 0150's liveness / stall / `/metrics` module against upstream's totals module.
3. **The rank-1 header.** Upstream appends `min_p` + a grammar flag (+ a packed grammar `_share`). Ours carries 0070's
   table, 0090's knobs and the cost flag, and the batcher has its own `batchplan` header. Both ranks must agree, so
   merge by hand.
4. **`decode.sample_rows`.** Upstream's nucleus branch against 0420 (draft vocab ids) and 0450 (device sampler).
5. **`split.RankReader` / `weights.load` rewrite** against 0001's q4mse loader and 0140. Any rebase also changes the
   0140 prepared key (it hashes the weight-building sources), so the first start after a rebase rebuilds the folders
   (8+ min, expected).

Estimate: the 0.3.6.2 figure (squash merge + fixes ~40-60 h + 15-25 h GPU) grows by ~6-10 h.

## 5. Speed-relevant changes for our GLM path

### 5.1 Direct speed from upstream code

**None.**

- **Decode.** No kernel on our decode path changed in `71377a5..9cd52ab`: experts, dense q4, latent MLA, sparse
  selection, KDA, MTP, DFlash2 and the head are the same. The `glm5_next/cuda/decode.py` changes are grammar hooks (no
  work when unconstrained) and the `top_k` 0 sampler branch.
- **Prefill.** The GLM prefill changes are Mac-only (`mlp.py`'s sorted gather on M1-M4). `prefill_attention.cu` serves
  GLM's per-head path, which production does not run.
- **The lane matmul.** `kernels/qmm.cu` (the 0.3.6.2 audit's "L1", est. +5% 1-stream if equal bits) is unchanged
  except a new FP8-weight prefill entry. Its bench arm from that audit was never run; 0440 E2 + 0570's size switch
  now hold that design space.

### 5.2 Indirect effects

| item | effect on our GLM path | estimate |
| --- | --- | --- |
| disconnect port (3.1) | frees slots held by departed clients (non-streamed, queued, prefilling, mid tool call) | 0 for a single well-behaved client; under agent churn (aborts, timeouts, retries) up to a whole slot's worth of throughput per abandoned 32k-token request (~5-14 min) and whole abandoned long prefills |
| structured output (2.1) | constrained requests only | -0.5-1.5% a round as upstream; ~-0.1-0.3% with fills during the forward; draft masking (phase 2) could raise JSON tokens a round |
| `top_k` 0 nucleus (2.2) | `top_k` 0 requests only | +1-3 ms a round (a second exchange, over NCCL at our sizes, and host loops) |
| 8ae247f (skipped) | first start from the checkpoint only | <= ~1 min of 8+ |

### 5.3 Ideas, not code

The 0.5.0 notes credit the 27B's long-context decode gain (18.4 -> 38.9 tok/s at 128k on one Spark, upstream's
number) to reading each key chunk once for a round's rows. The GLM analogue is our decode indexer:

- **Today.** `sparse.select_tokens(_dev)` launches `_scores` with grid (row, pool block), so each row's program
  loads the pool-key tile itself. Rows of one block are adjacent in launch order, so L2 likely absorbs most of it.
- **Evidence.** Decode at 314k was 58.9 tok/s against ~60-77 at short context, so it is not dominant at 300k.
- **The candidate change.** 0065's `_scores_rows` (BRB rows a program, one tile load, **same bits**, checked on the
  device) already exists for prefill and could serve decode / verify windows.
- **Estimate.** ~0 at short context, 0 to +3% at >= 256k, more toward 1M. Measure `_scores`' DRAM bytes with ncu at
  256k / R = 16 before doing anything. Not an upstream port: our own DECODE-PLAN item.

## 6. Recommendation: ranked port list

Each port is a patch of our own (numbers from 0600 up, in case a 0590 is in flight). Nothing needs a rebase; the pin
stays at 0.3.4.

| rank | port | upstream source | effort | bits | why this rank |
| ---: | --- | --- | --- | --- | --- |
| 1 | **Client disconnect on the batcher**: socket check while queued / prefilling / silent, non-streamed included; callback exceptions cancel the slot (3.1) | 24afe5e family | 2-3 h | same | real slot waste in prod today; small, host only |
| 2 | **Request hardening**: `check()` exceptions -> 400, template refusals, `chat_template_kwargs`, strict sampling fields before headers, non-UTF-8 (3.2) | 2bc35c4 | 2-3 h | same | dropped connections today on malformed input |
| 3 | **`kill -USR1` stacks on the CUDA server** (3.5) | a586f3d | 15 min | same | ops: a hung rank becomes diagnosable under the watchdog |
| 4 | **Image URL hardening**: HTTPS-443, public addresses only, pinned connection, redirects re-checked, types, bounded preparation (3.4) | 391713e | 2-3 h | same | closes a blind SSRF in 0500 (GLM53_TF_VISION=1 in prod) |
| 5 | **Structured output**: `response_format` / `guided_json` / `structured_outputs` on the batcher and the lone path, exact under drafting; fills overlapped with the forward; `GLM53_TF_GRAMMAR` (2.1) | 98fb8f9, 7ae9da2, 1da4b7e | 1.5-2.5 days + 1-2 h GPU | same for unconstrained; constrained drafted == serial == alone | the one substantial feature; its own GPU gate set |
| 6 | **Small compat bundle**: `return_token_ids`, token-exact stop length, #87 bracket repair (2.2, 3.3, 3.6) | e6be5ff, 88407d1, c30a1a9 | ~2 h | same | cheap correctness; `return_token_ids` helps RigMark / exact checks |
| 7 | **`top_k` 0 nucleus + `min_p`** over our `fast_gather` (2.2) | 9cd52ab | 3-5 h | changes only `top_k` 0 / `min_p` requests | fixes the silent top-16 truncation |
| 8 | **Call gate + `thinking_budget`** (`tool_choice` required / named, reasoning cap) on the batcher as cut + resubmit (prod re-prefills the reply once a gated reply, 2.2) | 0.3.6.2 call gate, 9cd52ab `ThinkBudget` | ~1 day | gated replies well-defined; others same | agent frameworks; bounds GLM's long thinking, pairs with 5 |
| 9 | `/health` live totals in upstream's JSON shape (2.2) | 51b098d, c4bf25f | ~1 h | same | compat with upstream-aware pollers; 0150's `/metrics` already has the data |
| 10 | Responses API (2.2) | 9cd52ab | ~1 day | same | only if a Responses-only client (Codex-style) is wanted |
| - | Do not port: 8ae247f / da8a5df (0140 covers them), 7d27a19 (entrypoint covers it), 0768f8d / b29df86 / 603d3bc (other families or the Mac; 0540 / 0120 / 0560 cover GLM), `prefill_attention.cu` (per-head path), NVFP4 / Flash Next / DeepSeek / Bonsai work | | | | |

GPU needs:

- Ranks 1-4, 6 and 9 are host-only. They can share one short GPU check: the exact suite, the reply SHA, the canary,
  and one disconnect / error smoke test on the live pair.
- Rank 5 needs its own window: section 2.1's gates.
- Rank 7 needs one sampled `top_k` 0 drafted == serial run.
- Rank 8 needs a gated-reply exactness run.

### Method

To reproduce:

- `git clone vendor/TensorFold ~/.cache/tf-upstream-050 && git checkout 9cd52ab`.
- The patched tree: `git checkout 2f8e514` + `git apply patches/*.patch` in name order, committed in
  `~/.cache/tf-ours-b9`.
- The conflict counts: `git merge-tree --write-tree --name-only --messages HEAD <base>`, then counting `<<<<<<<`
  lines in the resulting tree's conflicted files.
- Per-commit reach: `git log 71377a5..9cd52ab -- <path>` over `glm5_next/cuda/`, `cuda/`, `engine/` and `server/`.
