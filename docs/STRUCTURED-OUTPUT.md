# Structured output, exact under drafting and batching (patches/0610, `GLM53_TF_GRAMMAR`)

> **Update 2026-09-30 (W19, docs/RESULTS.md): adopted** (`GLM53_TF_GRAMMAR=1`, image b10). Unconstrained replies
> unchanged (hashes); 8 schemas x greedy / sampled x thinking on / off all valid and identical drafted, undrafted and
> 4 at once; tool calls 6/6; a 50-object `response_format` task -0.4% tok/s. The exposed mask wait measured ~0.7 ms a
> round, above section 8's 0.3 ms estimate (follow-up: `GLM53_TF_GRAMMAR_THREADS=8`).


> Offline work (no GPU; the Sparks were not touched). Stack: 0001-0600 + 0610 (0610 also applies on 0001-0590 without
> 0600). The design follows docs/UPSTREAM-050-AUDIT.md 2.1 (upstream TensorFold 0.5.0's 98fb8f9 / 7ae9da2 / 1da4b7e),
> ported to our batcher, which upstream does not have. Numbers marked *measured* come from a laptop (AMD Ryzen 7
> 6800U, x86-64, 16 threads); GB10 numbers are **estimates** until the GPU plan (section 8) runs.

## 0. Bottom line

- **What it does.** With `GLM53_TF_GRAMMAR=1` a request's OpenAI `response_format` (`json_object`, `json_schema`),
  vLLM's `guided_json` / `guided_regex` / `guided_choice` / `guided_grammar` / `structured_outputs`, and tool calls
  (`tool_choice` `"required"` or a named function, or tools marked `"strict": true`) are enforced token by token:
  the reply is valid by construction (JSON that parses and validates; tool calls in GLM's markup whose arguments
  validate against the tool's schema). Knob off (default): every one of those fields is ignored exactly as today.
- **Unconstrained requests take today's code path** with the knob on too: no grammar object exists for them, no
  header byte changes (the grammar travels only in front of a constrained request's header), no extra collective,
  no extra GPU work. Tested: knob on == knob off for plain requests, replies, keeps and drafters (section 6).
- **Exact.** A constrained drafted reply == the same request with `"draft": false`; a constrained slot among 4 ==
  itself alone; masks depend only on each row's own path; the keyed Gumbel draw is unchanged except that masked tokens
  are excluded; two ranks (each masking its vocabulary half, RoCE candidate gather unchanged) == one rank with the whole
  row. All proven on CPU against the real `Batcher` / `Stepper` / `sample_multi` / `decode.prefill` and the lone
  loops (129 tests).
- **Thinking.** The grammar applies to the visible answer only: when the prompt ends in an open `<think>` (thinking
  on) the rows are unconstrained until `</think>` is chosen, then the grammar starts. The reasoning tokens are the
  unconstrained reply's, bit for bit (tested).
- **Engine.** xgrammar 0.2.8 (Apache-2.0, aarch64 cp312 wheel, what upstream uses; it also ships a GLM-4.7 / GLM-5
  tool-call structural tag). One `RUN pip install` line in the Dockerfile (torch pinned to the base image's).
- **Cost** (section 5): only the draft cut (~1-2 us a draft token) precedes the forward; the masks are filled on
  worker threads during the forward (the fill releases the GIL): *measured* here a 4-slot x 16-row round's fills take
  ~1.3-2.7 ms on 4 threads against a 60-120 ms round. Est. **~0.1-0.3% a round** for rounds with constrained slots
  (upload + unpack + mask on the GPU, ~0.05-0.1 ms, and the host draw instead of the device draw, off in prod anyway);
  0 for rounds without them. Acceptance may drop slightly (drafts the grammar rejects are cut; drafter-side masking
  would recover it, not done).

## 1. What a request gets

| field | grammar |
| --- | --- |
| `response_format: {"type": "json_object"}` | any JSON object (`{"type": "object"}`, OpenAI's contract) |
| `response_format: {"type": "json_schema", "json_schema": {"schema": S}}` | JSON valid under S (xgrammar strict mode: `additionalProperties` false unless S says otherwise; properties in S's order) |
| `guided_json`, `structured_outputs.json` / `.json_object` | as above (vLLM clients) |
| `guided_regex`, `guided_choice`, `guided_grammar` (EBNF), `structured_outputs.regex` / `.choice` / `.grammar` | as upstream |
| `tools` + `tool_choice: "required"` | at least one call to one of the tools (GLM markup), arguments held to each tool's schema; parallel calls unless `parallel_tool_calls: false` |
| `tools` + `tool_choice: {"type": "function", "function": {"name": N}}` | exactly a call to N |
| `tools` with any `function.strict: true` (`tool_choice` auto / absent) | free text, or calls (after GLM's `<tool_call>` trigger) whose arguments follow the schemas (non-strict tools: any JSON value per argument) |
| `tools` otherwise, `tool_choice: "none"` | not constrained (today's behaviour) |

- **400 before any header** (as upstream): a malformed field, a grammar that does not compile, an output format beside
  `tool_choice` required / named, xgrammar missing on the server. An output format beside strict `auto` tools wins
  (the reply is JSON, no call can be made).
- **Mid-reply failure** (the matcher rejects, or a row allows no token: not expected with xgrammar): that request ends
  with HTTP 500 / an SSE error event; the other slots go on. Both ranks walk the same matcher, so both end the request
  at the same point without a message.
- **Stop / EOS.** The grammar allows the stop tokens (the engine's eos ids: `<|endoftext|>`, `<|user|>`,
  `<|observation|>`) only where the grammar may end, so with `stop_eos` the reply ends exactly when the JSON (or the tool
  call) is complete and never inside it. `"ignore_eos": true`: after the stop token the grammar has ended and decoding
  goes on unconstrained. `max_tokens` can cut a reply mid-JSON (`finish_reason` `length`), as OpenAI does. `stop`
  strings (0160) cut the visible text as before.
- **Added tokens.** Upstream's tokenizer view lets GLM's added tokens (`</think>`, `<|user|>`, `<|begin_of_image|>`,
  `<tool_call>`, ...) spell their text inside JSON strings. Ours never allows an added token inside a grammar (their
  vocabulary entries are empty for xgrammar); the tool grammars keep the six tool markup tokens (`<tool_call>`,
  `</tool_call>`, `<arg_key>`, `</arg_key>`, `<arg_value>`, `</arg_value>`), which the model emits as single tokens.
  Two compilers (text / tools views) are built at load.
- **Thinking** is read from the prompt: the last of `<think>` (154841) / `</think>` (154842) in it. `<think>` last
  (the chat template with thinking on ends the prompt with `<|assistant|><think>`): unconstrained until `</think>` is
  chosen. Otherwise (thinking off ends it with `</think>`, or a raw completion prompt without either): constrained
  from the first token. The MiaAI-Lab note in `app.py`
  (thinking on at `low` effort is the reliable setting for structured output) still applies; `reasoning_effort` bounds
  the reasoning before the JSON.
- The response's `tensorfold.grammar`: windows, drafts cut, rows masked, fill ms (worker time), wait ms (exposed on
  the round's thread), finished.

## 2. The grammar engine

| | xgrammar 0.2.8 | llguidance 1.6.1 | outlines-core 0.2.14 |
| --- | --- | --- | --- |
| licence | Apache-2.0 | MIT | Apache-2.0 |
| aarch64 cp312 wheel | yes (manylinux_2_28) | yes (abi3) | yes |
| upstream TensorFold 0.5.0 | **yes** (`tensorfold[grammar]`) | no | no |
| GLM tool calls | **built-in `glm_4_7` structural tag** (`glm_xml` argument style = our 0002 parser's reading) | would need our own grammar | would need our own grammar |
| rollback / fork | yes / yes | yes | no rollback (states by index) |
| threads | `BatchGrammarMatcher`; `fill_next_token_bitmask` releases the GIL (*measured*: a Python loop on the main thread ran at full speed while a thread filled masks) | Rust, releases the GIL | Rust |

xgrammar is used. Its imports need `apache-tvm-ffi`, `pydantic` and `transformers` (pure Python on aarch64: the
`triton` requirement is x86-64 only); the Dockerfile installs them with the base image's torch pinned
(`pip install -c torch==<installed>`), so a resolver that wanted another torch fails the build instead of replacing
it. Our code never builds a transformers tokenizer: the vocabulary comes from `tokenizer.json` through `tokenizers`
(xgrammar's `TokenizerInfo.from_huggingface` does the same with a fast tokenizer: the same encoded vocabulary and
`_detect_metadata_from_hf` on the backend JSON: byte-level, no prefix space).

## 3. Why it is exact

The contract: drafted == serial, batched == alone, resumed == fresh, knob on == knob off for unconstrained requests.

1. **A verify window is a chain** (row 0 the pending token, already followed by the matcher; rows 1.. drafts).
   `Constraint.cut` walks the drafts on the reply's matcher (accept, then roll back): a draft the grammar rejects, a
   draft that ends the grammar (its stop token), or an added token is dropped with everything after it. The parent
   row's masked distribution could never choose it, so the accept rule would have stopped there anyway. Only R changes,
   and every kernel is row-count invariant (0085 / 0200 / 0290 / 0440 / 0520 / 0580's contracts); 0280's padded rows are
   never kept.
2. **A row's mask depends only on its own path** (committed tokens + the drafts above it). For every row that is kept,
   that path is the serial prefix at that position, so its mask is the serial step's mask. `Constraint.fill` computes
   exactly those masks (accept along the kept chain, fill each constrained row, roll back); tests compare them with a
   fresh matcher per row.
3. **The keyed draw.** `grammar.apply` sets a constrained row's disallowed columns to -inf (a new tensor: the forward's
   logits are not written) before `sample_multi`'s / `sample_rows`' top-k. Then the same rule runs: top-k by (value,
   id), top-p, Gumbel keyed by (seed, absolute position, token id). -inf candidates never win (-inf score, zero
   probability, sorted last; greedy's lexsort takes the finite maximum), so the token is the rule applied to the
   allowed tokens only. Tested against a reference that draws from the allowed tokens alone, including rows with fewer
   allowed tokens than top_k, greedy and sampled.
4. **Two ranks.** Each rank holds half of the vocabulary (`w.vocab_offset`); `apply` unpacks only the bitmask words
   covering its columns (77,440 = 2,420 words exactly). The candidates' exchange (0230 RoCE / NCCL) is unchanged; both
   ranks draw the same token from the same gathered candidates. Tested with two threads exchanging their candidates as
   the all-gather does: == one rank with the whole row.
5. **Both ranks walk the same rows.** Rank 0 sends the grammar (kind, think state, text) in front of the admission's
   header; rank 1 compiles it with its own compiler. Cuts, fills and advances are deterministic functions of the same
   tokens, so the windows, keeps and replies are the same (tested with a follower batcher replaying rank 0's plans).
6. **The first token** is sampled from the prefill's last row under a one-row window (`grammar.first_token` wraps
   `e.sample` for the prefill's target row only; drafts never), in the batcher's last piece, in 0560's group finish and
   in the lone prefill. 0540 keeps a prompt's snapshot strictly before its end, so a resumed request's first token is
   always a fresh (masked) row: resumed == fresh (tested with the session store).
7. **Fills on threads** only move when the masks are computed (the matcher is not touched by the round's thread
   between the cut and the wait); tested: threads == inline, the same bits.

## 4. Where the hooks are

| site | change (only for a request with a grammar) |
| --- | --- |
| `app.GlmApp.check` / `run` | `grammar.check` (400s, compile, cached) when the knob is on; `request.grammar` = (spec, compiled) |
| `engine.GlmEngine.__init__` | `grammar.setup`: both ranks' knobs checked; the two compilers built at load (both ranks); fill threads |
| `engine.generate` (rank 0) | the grammar bound to the prompt's think state; batch: `request.grammar_bound`; lone path: `pack` in front of the header, a fresh `Constraint` on `e.constraint` around `_run`, its stats |
| `engine.follow` (rank 1, lone) | `split` the header, compile, a fresh `Constraint` on `e.constraint` around `_run`; a GrammarError ends the reply here as on rank 0 |
| `engine._run` | the prefill under `first_token` when `e.constraint` is set (otherwise unchanged) |
| `decode.serial_decode` / `mtp_decode` / `dflash_decode` / `auto_decode` | `grammar.lone(e)`: cut before the forward (fills start), masks after the launch, masked host draw, advance (upstream 1da4b7e's hooks) |
| `decode.sample_rows` | `host=True` keeps masked rows off 0450's device draw |
| `batch.Batcher._header` / `follow` | the grammar in front of the admission's header / split off and compiled |
| `batch.Batcher._admit` | a fresh `Constraint` per admission (a background request that runs again starts over) |
| `batch.Batcher._piece` | the last piece's first token under `first_token`, then `advance` |
| `batch.Batcher._verify` | `_grammar_cut` (cut each constrained window, fills start) before the forward; `_grammar_masks` (wait, then `grammar.stage`: the bits queued on the device behind the forward, pinned, no host wait) right after its launch; `sample_multi(..., masks=)`; `advance` after `accept`; a failing slot ends alone (`_grammar_fail`) |
| `batch.sample_multi` | `masks`: -inf before each constrained spec's top-k; a round with masks draws on the host |
| `resident.Resident.eligible` | a constrained slot is not eligible (`"grammar"`): resident rounds have no host in the loop |

Not changed: forward / kernels / graphs / collectives, the MTP and DFlash2 drafters, lookup, the depth optimizers,
snapshots and the session store, 0370's plan rider, the header format of unconstrained requests.

## 5. Cost

*Measured* on this laptop (`bench/grammar_bench.py`, GLM-5.3's real 154,880-column vocabulary, see the table in
section 6.3): per constrained row the fill is tens of microseconds (p50) and at most ~0.4 ms (inside JSON strings,
where nearly every token is allowed); the cut walk ~1-2 us a draft token; `advance` ~1-2 us a token. GB10's
Cortex-X925 cores are faster single-threaded than this Zen 3+ laptop core (est. 1-1.3x).

Per round with constrained slots (estimates for GB10):

| work | where | est. |
| --- | --- | --- |
| cut walk, up to 4 x 15 drafts | round thread, before the forward | 0.05-0.15 ms |
| fills, 4 slots x up to 16 rows | 4 worker threads, during the forward (60-120 ms) | 1-3 ms of worker time, ~0 exposed |
| upload 2,420 words a row (9.7 KB, pinned, queued right after the forward's launch: no host wait), unpack, mask | GPU, after the forward | ~0.05-0.1 ms |
| host draw instead of 0450's device draw | host | 0 in prod (the device draw is off) |
| `advance` of the kept tokens | round thread | < 0.05 ms |

**Total ~0.1-0.3% of a constrained round** (the audit's estimate for fills during the forward; serial fills before the
forward would be +0.5-1.5%). Rounds without a constrained slot: 0. The first constrained request of a new schema
pays its compile (*measured* 2-20 ms here, cached by text), on rank 0's HTTP thread and on rank 1's round loop (once).
Load: the two tokenizer views take ~3.6 s a rank (*measured* here; both ranks in parallel), only with the knob on.
Memory: two `TokenizerInfo`s (~tens of MB a rank) and the compile caches (<= 128 MiB each).

Acceptance: drafts the grammar rejects are cut, so a JSON body's windows are shorter where a drafter proposed
non-JSON; the kept rows' tokens are the ones serial decoding makes. Masking the drafters' own logits with the draft
path's masks (the audit's phase 2) would raise acceptance on JSON; not done (drafts only: replies would not change).

## 6. Tests (CPU)

### 6.1 `tests/cuda/test_grammar_patches.py` (129 passed, ~20 s; 4 skip without `$GLM53_TF_TOKENIZER_DIR`)

Run: `PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda GLM53_TF_TOKENIZER_DIR=<dir> pytest -q
tests/cuda/test_grammar_patches.py` (xgrammar 0.2.8, torch 2.14 CPU, jsonschema).

- **Fields** (23): every accepted shape, every 400, tools opt-in only, knobs, `pack` / `split` of any text.
- **Masks vs the library** (41): random committed paths x random chains with corruptions (invalid tokens, stop
  tokens, `</think>`, random ids) on a nested schema, `json_object` and a choice grammar: `cut` keeps exactly what
  a fresh matcher accepts (to the first rejection / stop token); `fill`'s bits == a fresh matcher's at every kept
  row's path; the matcher ends where it started; `advance` == accepting; the think gate (rows before `</think>`
  unconstrained, drafts too); threads == inline; failures are GrammarErrors.
- **The draw** (21): `apply` on a vocabulary half == the whole row's mask sliced (offsets 0 / 32 / 80 / 77 / 150,
  padded columns never allowed); a masked row's token == the keyed rule over the allowed tokens only (greedy, three
  samplings, densities 2% / 30% / 100%); two ranks exchanging candidates == one rank, `sample_multi` (3 sequences,
  one unconstrained) and `sample_rows`.
- **The real Batcher on a hostile fake model** (36): float logits over a 160-token JSON vocabulary from a hash of the
  row's state (which reads every KV entry before it); fake MTP / DFlash2 drafters drafting the *constrained*
  continuation with deterministic corruptions (invalid tokens, stop tokens, `</think>`, wrong valid tokens); lookup:
  - drafted == `"draft": false` == an independent serial reference, 6 policies x greedy / sampled x schema / object /
    thinking;
  - 4 slots mixing constrained (schema, object, thinking) and plain requests arriving at different rounds == each
    alone == the reference; plain requests with the knob on == knob off (replies, keeps, drafters; no grammar stats);
    fills on 0 and 4 threads;
  - random nested schemas (objects, arrays, enums, bounds), 32 replies: every reply that ended parses and validates
    (jsonschema), the stop token is its last token, drafted == reference;
  - thinking on: the reasoning == the unconstrained reply's up to and including `</think>`, then valid JSON; thinking
    off: JSON from the first token;
  - stop / EOS: the stop token exactly where the JSON completes; `ignore_eos` == the same tokens then unconstrained;
    `max_tokens` mid-JSON cuts without an error;
  - drafts cut before the forward (the verified windows are the cut ones);
  - an injected fill failure ends only its slot (GrammarError to its caller), the others == reference;
  - a follower batcher with its own compiler replaying rank 0's plans: the same windows, keeps, replies;
  - the lone loops (serial / MTP / DFlash2 / auto+lookup, fills inline and on threads) == the reference;
  - the session store: the same request again resumes (`cached` > 0) and gives the same reply; a next turn too;
  - `Batcher.generate` on an HTTP thread takes the thread's bound grammar; a failure reaches the caller as the
    GrammarError itself.
- **Wiring** (4): the lone engine's header prefix and rank 1's `follow` (its own compile, the same bound grammar; a
  plain header unchanged); `setup` (knob mismatch refused, xgrammar missing / a vocab mismatch -> 400s); the app
  (knob off ignores every field, knob on answers the 400s and hands the compiled grammar to the engine).
- **The real GLM tokenizer** (optional, 4): 154,880 columns; no added token ever allowed inside JSON (start and
  inside a string, >100k tokens allowed there); the stop tokens exactly at the end; the think ids; the tool grammar
  takes the markup tokens, rejects a wrongly typed argument, and its call parses (`server.parse_tool_calls`) into
  arguments that validate; a JSON grammar never takes `<tool_call>`.

### 6.2 Regression (CPU host suites, 0001-0600 without / with 0610)

The same suites on the stack without 0610 (0001-0600) and with it, torch 2.14 CPU (`PYTHONPATH=<tree>/src:<tree>/
tests/cuda:tests/cuda`): **identical results**. Passing on both: `test_batch_sessions_patches` 17, `test_deep_verify_patches`
5 (+ its 16-row child), `test_batch_parallel_patches` 18, `test_decode_overlap_patches` 30, `test_kv_pool_patches` 13,
`test_adapt_patches` 11, `test_replay_ttft_patches` 32, `test_prefix_share_patches` 24, `test_session_disk_patches` 24,
`test_session_patches` 21, `test_disconnect_patches` 8, `test_multi_prefill_patches` 83, `test_draft_vocab_patches` 27,
`test_openai_compat` 22, `test_api_context` 20, `test_health` 10, `test_upstream_ports` 166, `test_request_log` 12,
`test_effort` 14, `test_glm_tool_calls` 5, `test_glm_lookup` 28, `test_http_pin` 4 (GPU-only cases skipped). Failing
identically on both (pre-existing, not 0610's): `test_solo_piece_patches` 3, `test_upstream_ports` 3 errors,
`test_vision_server` 1 error (this host's environment). `GlmEngine._run` keeps its signature (other suites bind it to
stand-ins): the grammar is set on `e` by its callers.

### 6.3 Mask computation microbenchmark (`bench/grammar_bench.py`, real tokenizer)

`GLM53_TF_TOKENIZER_DIR=<GLM-5 tokenizer.json + config.json> PYTHONPATH=<tree>/src python bench/grammar_bench.py
--repeat 5` (*measured*, laptop Ryzen 7 6800U under ~3.5 load from other jobs; GLM-5.3's 154,880 columns). Replies: a
pretty-printed 129-token JSON object (`json_object`, long free strings), a 95-token nested schema reply, a 24-token
GLM tool call (`tool_choice` required, strict schema).

| | json_object | json_schema | tool call |
| --- | ---: | ---: | ---: |
| compile, cold / cached (ms) | 20.1 / 0.04 | 5.9 / 0.03 | 27.8 / 0.28 |
| fill a row, p50 / p90 / max (us) | 82 / 237 / 383 | 47 / 78 / 145 | 14 / 43 / 125 |
| cut walk, a draft (us) | 2.0 | 1.5 | 1.8 |
| advance, a token (us) | 2.0 | 1.6 | 1.8 |
| cut + fill of a 16-row window (ms) | 1.73 | 0.65 | 0.32 |

A round of 4 slots x 16 rows (the slots at 4 points of the reply), fills only (ms):

| threads | 0 (inline) | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| json_schema | 2.87 | 1.74 | 1.32 | 1.20 |
| json_object | 6.65 | 4.06 | 2.72 | 3.04 |

Build of the two tokenizer views: 3.6 s (load time, knob on only). `apply` on the host for 16 x 77,440: 0.39 ms (the
GPU does it in prod: ~0.05-0.1 ms est.). Reading: the fills stay far below a 60-120 ms forward even in the worst case
(4 constrained slots with 16 rows each, inside free strings), so with 4 threads nothing is exposed; inline (threads 0)
they would still fit under the forward's GPU time after its launch, but compete with the round thread's own work.
Draft chains longer than the grammar allows are cut in ~2 us a draft.

## 7. Knobs, limits, follow-ups

- `GLM53_TF_GRAMMAR=0|1` (load-time, both ranks, checked): default 0. `GLM53_TF_GRAMMAR_THREADS=N` (rank-local, 0..64,
  default 4; 0 = fill on the round's thread right after the forward's launch). Same masks either way. The workers
  (`glm-grammar_*`) are made at load by the loading thread, so with 0370's `GLM53_TF_CPU_PIN` they inherit its cpus
  (`rest`), not the round loop's dedicated core.
- Not in the 0140 calibration key: the knob changes no forward, graph or cost.
- Constrained slots never run 0450's resident rounds and draw on the host (both off in prod).
- Not done: masking the MTP / DFlash2 drafters' logits (acceptance on JSON; drafts only), `thinking_budget` (2.2 of the
  audit: a separate patch), `top_k` 0 / `min_p` (separate).
- xgrammar logs a warning when asked to accept a special token; drafts of added / padded tokens are cut before asking
  it, so production logs stay quiet.

## 8. GPU test plan

Image `glm53-tensorfold:b10` = b9's list + 0590 (if adopted) + 0600 + 0610, built with the Dockerfile's xgrammar
layer. Prod down ~1.5-2 h. `$B=http://127.0.0.1:8000`, `$M=GLM-5.3-Flash-EXL3`, `$R` a results folder.

1. **Image.** `docker run --rm $IMAGE python -c "import xgrammar, torch; print(xgrammar.__version__, torch.__version__)"`:
   0.2.8 and the base image's torch (unchanged from b9: compare `torch.__version__`). In the image:
   `run_tests_in_image.sh $R/tests -- tests/cuda/test_grammar_patches.py` with `GLM53_TF_TOKENIZER_DIR` pointing at the
   checkpoint's snapshot (all pass), plus the batch suites (`test_batch_sessions_patches.py`,
   `test_deep_verify_patches.py`, `test_decode_overlap_patches.py`, `test_multi_prefill_patches.py`): all pass.
2. **Knob off (b10 as is).** Gates as for every image: `glmbench.py --suites exact` 10/10, `multiturn.py --modes
   batchexact` 4/4, reply sha `8794a3463259cc2f` (`results/W5/ab.py`), decode and prefill equal to b9 within noise.
   `bench/structured.py --suites plain --write-ref $R/plain-off.json` (the reference hashes).
3. **Knob on** (`GLM53_TF_GRAMMAR=1` in `config/prod.env`, restart). Boot line `structured output (patches/0610): ...
   built in ~2-4 s`, no rank mismatch.
   - **Unconstrained unchanged:** exact 10/10, batchexact 4/4, sha `8794a3463259cc2f`, `bench/structured.py --suites
     plain --ref $R/plain-off.json` (the same hashes; no `grammar` stats); ab.py decode tok/s == step 2 within noise.
   - **Structured exactness and validity:** `bench/structured.py --suites schemas,tools --out $R/structured.json`:
     8 schemas x greedy / sampled x thinking on / off, each drafted alone == `"draft": false` == 4 concurrent (the same
     `tensorfold.sha256`); every finished reply parses and validates; tool calls (required, named, strict auto) parse
     into known tools with valid arguments and no leaked markup.
   - **RigMark-like structured task:** `bench/structured.py --suites rigmark --reps 3`: RigMark's decode task shape (a
     50-object JSON array, temperature 0, thinking on, 4,096 tokens) without and with `response_format` json_schema:
     the schema run gives 50 valid objects every time (gated), both have stable hashes across runs. Also the real RigMark decode phase (`scripts/rigmark/run.sh
     tensorfold`, new `COMPARISON_ID`) with the knob on: its structured gate (exact JSON) passes as on b9 (the harness
     sends no `response_format`: the knob must not change it).
4. **Speed impact** (knob on): from step 3's rigmark suite, decode tok/s and `tokens_per_round` with the schema vs
   without (same prompt): expect -0-3% tok/s (the cut drafts; masks ~0.1-0.3% a round); the response's
   `tensorfold.grammar.wait_ms` / rounds (exposed fill time a round, expect < 0.05 ms) and `fill_ms` / rounds (worker
   time, expect ~0.5-1.5 ms). 4 concurrent constrained streams (4 x the rigmark request with the schema, sent
   together) against 4 without it: aggregate tok/s and round times from the request log. With
   `GLM53_TF_GRAMMAR_THREADS=0` once: the exposed time should grow by the fill time (shows the overlap works). If
   `wait_ms` a round exceeds ~0.3 ms, raise the threads to 8.
5. **Rank-1 compile stall:** a burst of 8 new schemas at once (`structured.py --suites schemas` right after a restart):
   the first rounds' `queued_s` / round times include rank 1's compiles (est. 5-50 ms each, once per schema); note it.
6. **Adopt** when 2-4 pass: `GLM53_TF_GRAMMAR=1` in `config/prod.env`. Revert: drop the knob (b10 without it == b10
   knob off; the fields are ignored as on b9).
