# Tool calling: multi-step chains on our stack (issue #6), patch 0620, and a benchmark plan

Written 2026-09-30, offline (host tests only). The §5 benchmark runs on production followed in W21: results and the
recommended client settings are in **§7** (tool-eval-bench 90 / 90 / 91, multi-step chains 8/8 in every configuration). Scope: **multi-step tool chains** (tool-eval-bench category C) and spark-bench's
**agentic** domain. The safety category, TC-43 and the abliteration trade-off are left out on purpose.

## 1. Summary

An independent tester (issue #6) ran two benchmarks at kit `c747c92`. Their setup was the non-abliterated Mia TR3 4bpw
checkpoint, thinking off, and temperature 0.3 for spark-bench (tool-eval-bench defaults to 0.0). Results:

- **tool-eval-bench:** 90/100. Multi-step chains 75%, which is 6 of 8 points over 4 scenarios.
- **spark-bench TrueScore:** 92.4. Agentic domain 82.6.

We serve a different setup: the abliterated checkpoint, with thinking on at effort high for requests that don't say.

**What the code audit found** (§3). I traced both benchmarks' requests through our template, parser and reply path.

**Nothing on our side corrupts either benchmark's chains with thinking off**, which was the tester's setting:
- Both benchmarks stream, and both send `content: ""`, not null.
- Both send tool results with matching ids.
- tool-eval-bench reads our `reasoning` field and echoes it back as `reasoning_content`.
- Our template renders multi-call turns and grouped results the way GLM expects.
- The engine stops on `<|observation|>` after a call.

So the tester's missing 2 points in multi-step and 17 points in agentic are **model behaviour with thinking off**, not
a server bug. The benchmarks' own criteria show what counts against the model, for example:
- exactly 2 calls in TC-08/TC-09;
- polling in a later turn in TC-61;
- no extra side effects in TC-07;
- turn budgets and "required value in the visible text" caps in the AG scenarios.

The audit did find **real defects that hit other clients, and hit thinking-on runs**:

| # | Defect | Who it hits | Fix (0620 knob) |
| --- | --- | --- | --- |
| D1 | An assistant tool-call turn with `content: null` renders as the literal text **`None`**: `<think></think>None<tool_call>...`. `null` is what OpenAI's shape and our own non-streamed reply use. A null tool result also renders as `None`. | Non-streaming OpenAI SDK agents that append `response.choices[0].message` to history. Not these two benchmarks. | `history` |
| D2 | Earlier tool-call steps lose their reasoning when the client doesn't send it back. GLM-5.3's template keeps each step's `<think>` after the last user message (interleaved thinking), but only when it is given `reasoning_content`. Otherwise every earlier step renders as `<think></think>`, a context the model never produced. | spark-bench with thinking on: it never sends reasoning back and renumbers call ids `call_{turn}_{i}`. Also most agent clients. tool-eval-bench already sends it back. | `reasoning` |
| D3 | Our prod answers with `reasoning` only (`GLM53_TF_REASONING_FIELDS=reasoning`), but the template reads only `reasoning_content`. A client that echoes our message back unchanged has its reasoning dropped. | Clients that echo the message. tool-eval-bench renames the field itself, so it is not hit. | `history` / `reasoning` |
| D4 | `tool_choice: "none"` is ignored: tools are still listed and calls are still parsed. A named function still offers every tool. `parallel_tool_calls: false` is ignored. (0610 already enforces `"required"` and named calls when `GLM53_TF_GRAMMAR=1`.) | tool-eval-bench TC-44 (category H, not C). Any client relying on `none`. | `choice` |
| D5 | If the model writes its call inside the think block (never closing it, or closing it after the call), the reply comes back as an empty answer with `finish_reason: stop`. The chain then ends. | Thinking-on runs only. How often it happens is unknown; the test plan measures it. | `thinkcalls` |
| D6 | Argument typing reads only a schema's `type` string. Type lists and `anyOf` fall back to "parse as JSON", which turns `"123"` into `123` for a `["string","null"]` parameter. Quoted numbers stay strings. | Rare. Parameter precision was already 100%. | `args` |
| D7 | Empty or null `arguments` in history make the template call `.items()` on a string, which gives an HTTP 400 for the whole request. | Clients that send `""` for calls with no arguments. | `history` |

**Recommended actions**, ranked in §4. Beyond the first two, the rest is measurement.

1. Build 0620 into the next image and turn on `GLM53_TF_TOOL_FIXES=all` after the A/B in §5. It is host only, same bits,
   and off by default.
2. For agent work, tell clients to use **thinking on** (low or high) with reasoning preserved. Leave the benchmark
   headline setting to the operator, because thinking lowers the latency subscores (§4.2).

## 2. The benchmarks

Installed locally from pinned commits, and nothing has been run:
- `~/.cache/tooleval/tool-eval-bench` (with a venv)
- `~/.cache/tooleval/spark-bench` (with `.venv`; its golden gate passed 12/12 offline)

Reinstall with `scripts/tooleval/run.sh setup`.

### 2.1 tool-eval-bench

**Repo:** https://github.com/SeraphimSerapis/tool-eval-bench, by SeraphimSerapis, **MIT**. Pinned at `c7b5b955`
(v2.7.0+8, 2026-09-29). Python 3.11 or newer.

**API** (`adapters/openai_compat.py:162-231`, loop in `runner/orchestrator.py:345-640`):
- Streamed `POST /v1/chat/completions` with `stream_options.include_usage`.
- `tools` plus `tool_choice: "auto"`. The exceptions are TC-44 (`none`) and TC-45 (`required`, then `auto`).
- `parallel_tool_calls: true`.
- `temperature` 0.0.
- `max_tokens` 4096 with thinking off, 16384 otherwise.
- `--no-think` sends `chat_template_kwargs {"enable_thinking": false}`. `--backend-kwargs JSON` is deep-merged into every
  request.

**History it sends back** (`orchestrator.py:266-287`):
- Assistant turns carry `content` (`""` when empty) and `tool_calls` with the model's own ids.
- `reasoning_content` is included whenever the model returned reasoning (`reasoning_content` or `reasoning`) and the
  turn called tools.
- Tool results are `{"role":"tool","tool_call_id","name","content": json}`.

**Loop limits:**
- At most 8 turns; TC-46/63 allow 12 and TC-62 allows 14.
- A guard stops the run after 3 identical turns.
- The 120 s timeout applies between streamed chunks.
- The system prompt says to use a tool only when necessary, and gives the date as 2026-03-20.

**Scoring:**
- Pass = 2, partial = 1, fail = 0 points per scenario.
- The headline is points ÷ (2 × gradable scenarios). For the tester, 124/138 = 90.
- Deployability = 0.7 × quality + 0.3 × responsiveness.
- Responsiveness = `100/(1+(median_turn_ms/3000)^1.5)`. The tester's 1.5 s median gives 74.

**Multi-step chains (category C)**: 4 scenarios, so each one is worth 25% of the category.

| ID | Prompt | Pass | Partial / fail |
| --- | --- | --- | --- |
| TC-07 Search→Read→Act | "Find the Q3 budget report and email the total to my manager." | search → read `file_091` → `get_contacts` ("manager") → exactly one `send_email` to jordan.park@company.com stating 4.4 (search before read; read and contacts before the email) | Partial: all 4 steps out of order, a duplicate email, or 3 of 4 steps. **Any extra side-effect tool is a FAIL.** |
| TC-08 Conditional | "Check the weather in Paris. If it's raining, remind me to bring an umbrella tomorrow at 8am." | `get_weather` Paris, then `set_reminder` with "umbrella" at 2026-03-21 08:00, **exactly 2 calls** | Partial: extra or duplicate calls, or asking instead of acting. Fail: no weather check first. |
| TC-09 Parallel independence | "What's the weather in London and the stock price of MSFT?" | both tools, answer states 12 and 412, **exactly 2 calls** (parallel is noted, not required) | Partial: an extra call, missing values, or a `web_search` fallback. Fail: one side missed. |
| TC-61 Async polling | Run `analyze_data(...)` with `run_code`, then "Poll the returned job ID until complete or failed." | submit, get `pending` / `job_tc61_9f3a`, then a real `check_job_status` in a later turn; the answer states 15420 or "3 anomalies" | Partial: reports "pending" or doesn't prove completion. Fail: runs once without polling. |

These criteria reward restraint (exact call counts, no extra side effects) and persistence (polling). Both are
typical thinking-off weaknesses. None of them depends on how the server renders history, because the benchmark
sends it back correctly.

### 2.2 spark-bench (TrueScore)

**Repo:** https://github.com/Weschera/spark-bench, by Weschera (wesche.com/dgx). The README says MIT, but the repo has
**no LICENSE file**. Pinned at `125ba161` (v6.8.0, 2026-08-29): 76 scenarios in 12 domains. Pure Python; the golden
gate needs playwright, pillow and chromium.

**API** (`spark_bench.py:149-320`):
- Streamed `POST /v1/chat/completions`.
- `tools` plus `tool_choice: "auto"`. `parallel_tool_calls` is never sent.
- `temperature` 0.3.
- `max_tokens` is 1000 for agentic, or omitted with `--uncapped`.
- `--thinking on|off` sends `chat_template_kwargs {"enable_thinking", "thinking_mode"}`.
- `SPARK_BENCH_EXTRA_BODY` is merged into each request with a shallow update, so it replaces `chat_template_kwargs`
  wholesale.
- `SPARK_BENCH_DUMP_DIR` saves every request and response.

**Preflight:** `/v1/models` must list the model, and a weather probe must return structured `tool_calls`.

**Agentic loop** (`eval_suite.py:3898-3981`):
- The system prompt says to work through all steps and "Call tools one at a time".
- **The run ends at the first turn without tool calls.** A `length` or `runaway` finish scores 0.
- The assistant turn it sends back is `{"content": text or "", "tool_calls": [ids renumbered call_{turn}_{i},
  arguments re-serialised]}`. **Reasoning is never sent back.**
- Tool results come from a deterministic simulated environment (calendar, weather for 9 cities, injected failures on
  the N-th call of a tool).
- Only the visible text is graded.

**Agentic scenarios** AG-01 to AG-12 (`eval_suite.py:2108-2215, 3655-3741`, grading `3984-4288`). There are 6 to 14
steps each, covering trips and meetings, weather-conditional rescheduling, retry-on-failure (AG-08, AG-10), nested
schemas (AG-09) and 30 KB documents with a buried code (AG-11, AG-12).

How a scenario is scored:
- The score is the fraction of checks passed.
- It is capped at 0.6 when the required values are missing from the visible text (AG-01/03/06/07/10/12).
- It is multiplied by 0.8 when the run goes over its turn budget (15 by default).

**TrueScore** = 0.55 Quality + 0.25 Calibration + 0.15 Reliability + 0.015 Efficiency + 0.035 Responsiveness.
- Efficiency = answer tokens ÷ (answer + reasoning). For agentic scenarios it is fixed at 1.
- Responsiveness = `100/(1+median_s/20)`.

## 3. Our request path, checked from code

The trees:
- Patched tree: `~/.cache/tc0620/tree`, which is vendor 0.3.4 plus 0001-0610 plus 0620.
- Prod checkpoint template: `chat_template.jinja` sha256 `41cff9af…`, read-only on the head node, identical to
  `~/.cache/vocab0420/tokenizer/`.
- Upstream: `~/.cache/tf-upstream-050`.

### 3.1 Template (GLM-5.3, the prod checkpoint)

- `[gMASK]<sop>`, then `<|system|>Reasoning Effort: Low|High|Max`, then a tools system block. Tools are listed as JSON;
  the call format is `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`.
- **Assistant turns:**
  - Reasoning comes from `m.reasoning_content`, or from a `<think>…</think>` inside `content`. It is rendered as
    `<think>…</think>` for every turn after the last user message, and for all turns when `clear_thinking=false`.
    Every other turn gets `<think></think>`.
  - Then `content.strip()`, then each call. A string value is rendered raw; anything else as `tojson`.
- **Tool messages:** consecutive `tool` messages form one `<|observation|>` block. Results are sorted into the order of
  the previous assistant's `tool_calls` when every id matches, and left in message order otherwise. Each result
  renders as `<tool_response>…</tool_response>`.
- **Generation prompt:** ends with `<|assistant|><think>`. With thinking off, 0150's `ThinkingOffTemplate` appends
  `</think>` and drops the `Reasoning Effort: Max` line, matching GLM's thinking-off template.
- **D1 (confirmed by rendering):** `visible_text(None)` falls through to `{{ content }}`, which prints `None`.
  Rendered chain: `…<|user|>Weather in Paris and Rome?<|assistant|><think></think>None<tool_call>get_weather…`.
  A server that normalises messages before rendering avoids this; our `ChatTemplate.render` passed null through as it is.
- **D7:** `ChatTemplate.render` turns JSON-string arguments into dicts, but `""` stays a string, and
  `_args.items()` then raises. The result is an HTTP 400 "the chat template rejected the request".
- **Preserved thinking:** within one user turn (a tool loop), the template already keeps each step's reasoning,
  if it has it. This is GLM's interleaved thinking. `clear_thinking=false` extends that across user turns. It can be
  passed per request in `chat_template_kwargs`, and our server forwards it. The benchmarks are single-user-turn
  chains, so it doesn't matter for them; it is not recommended as a server default because it costs context.

### 3.2 Reasoning between tool calls

- The server never drops reasoning on its own side. What reaches the template is whatever the client sends, and prod
  sends it out as `reasoning` only. So there are three paths:
  - tool-eval-bench reads `reasoning` and returns it as `reasoning_content`, so it is preserved.
  - spark-bench returns no reasoning and renames the ids, so it is lost (D2).
  - A client that echoes our message returns it as `reasoning`, which the template ignores, so it is lost (D3).
- `split_thinking` treats everything before the first `</think>` as reasoning. A reply that never closes its think
  block is all reasoning, and its calls are lost (D5).

### 3.3 Parser (0002 + 0600) against upstream 0.5.0's shared parser (`server/tools.py`, `tool_parameters.py`)

**Where ours matches upstream:**
- GLM `<arg_key>/<arg_value>` blocks.
- JSON bodies.
- The Qwen `<function=…>` form.
- Several blocks in one reply, i.e. parallel calls: each gets its own `call_<24 hex>` id and is streamed with its own
  `index`.
- Unknown or malformed calls are left as text: hidden while streaming, left in `content` otherwise.
- Case-insensitive tool names.
- `closed_json` for a nested value that is one bracket short (#87).

**Where upstream differs:**
- (a) Untyped or union schemas are kept as text unless the parsed value matches a concrete type. Ours parses them as
  JSON, which is vLLM glm47's rule and is kept in 0620's `args` for untyped schemas. 0620 adds type lists, anyOf/oneOf,
  enum, quoted numbers and booleans, and integral floats.
- (b) `tool_choice` handling (`active_tool_specs`): `none` offers no tools, and a named function offers only that tool.
  0620 `choice` ports this.
- (c) Upstream has a call gate for `required` or named calls, a first-token constraint in the engine. We don't need it:
  0610's xgrammar structural tag enforces `required` and named calls exactly under drafting, and it is on in prod.
- (d) Upstream supports `thinking_budget`. It needs an engine-side forced `</think>` and is not host only; it is not
  ported (see §4.3).

**Stop and finish behaviour:**
- `<|observation|>` (154829) is one of the checkpoint's eos ids (`154820, 154827, 154829`), so a reply stops right
  after its calls.
- `finish_reason` is `tool_calls` whenever a call parsed.
- Stop strings apply to the visible answer only (0160).
- Calls are streamed once the reply ends, one delta per call with the id, name and complete arguments. Both benchmarks
  accumulate them correctly: the preflight probe passed, and the tester saw 0% errors.

### 3.4 Argument typing

The rules before 0620:
- `type: "string"` → raw text.
- Any other `type`, and untyped schemas → JSON, with the `closed_json` repair for arrays and objects.
- Otherwise → text.

The benchmarks' tools are plain typed schemas. AG-09's nested objects parse as JSON. So typing is not a factor in the
tester's misses; the `args` fix is robustness only.

## 4. Fixes, ranked by expected impact on these benchmarks and by effort

### 4.1 Patch 0620 `glm-tool-calling` (implemented; host only; off by default)

`GLM53_TF_TOOL_FIXES` is a comma list. `all` means `history,reasoning,choice,args,thinkcalls`, plus patch 0690's
`opencalls,argkeys` (docs/PATCHES.md, 0690: a lost `<arg_key>` is put back; a GLM call the end token left open is
returned only when no value is open and every `required` parameter is there, so a value cut mid-string never runs);
`grammar` must be named explicitly. Files:
- `patches/0620-glm-tool-calling.patch`: `cuda/server.py`, `glm5_next/cuda/{toolfix.py (new), app.py, grammar.py}`
- tests: `tests/test_tool_fixes.py` (46 host tests)
- patch notes: docs/PATCHES.md §0620

Validation:
- The regression suites are unchanged with the knob off: 0002/0160/0150/0600/0210/0300/0490/0150-health 301 passed
  (the 3 known fetch-test errors are pre-existing), and 0610 grammar 129/129.
- The patch applies cleanly on top of 0001-0610.

| Rank | Fix | Expected effect on the tester's benchmarks | Other clients | Cost / risk |
| --- | --- | --- | --- | --- |
| 1 | `reasoning` (D2, D3): keeps the reasoning of each tool-calling reply and puts it back into later requests. It is keyed by our call ids and by sha256(last user message + call names and canonical arguments), because spark-bench renumbers ids. The client's own reasoning always wins. LRU of 1024 replies / 32 MB. | spark-bench thinking on: each AG step sees its earlier thinking, as GLM was trained for. The size of the gain is unknown: measure it with `chains low/high`, with and without. No effect with thinking off. tool-eval-bench already preserves reasoning. | Every agent client that drops reasoning. Note that 0310/0110 prefix reuse also gets a longer shared prefix, because the history now matches the generated tokens up to the call. | ~µs a request. The worst mismatch is a same-prompt, same-call reply from another conversation (spark-bench repeats), where the most recent one wins. |
| 2 | `thinkcalls` (D5) | Thinking on: turns a chain-ending empty answer into the call the model meant. Frequency unknown; the plan counts it. | same | Only when the reply ends on `<|observation|>` with complete calls at the end of its reasoning. |
| 3 | `history` (D1, D3, D7) | 0 for these two benchmarks, which send `""`. | Fixes `None` in every null-content chain (OpenAI SDK agents) and the 400 on `""` arguments. | none |
| 4 | `choice` (D4) | tool-eval-bench TC-44 (category H) passes by construction instead of relying on the model's restraint. Worth ≤ 2 points of 138 (≤ 1.4). No category-C effect. | Correct OpenAI semantics. | none |
| 5 | `args` (D6) | ~0: parameter precision is already 100%. | Type lists, `anyOf`, quoted numbers. | none |
| 6 | `grammar` (0610 opt-in for auto tools) | ~0. Calls are already well-formed. | Guarantees schema-valid names and arguments in auto mode. | Mask fills on every row of a tools request (0610 §5: ~0.1-0.3% of a round); prompts are unchanged. Leave it off unless a client shows malformed calls. |

### 4.2 Client settings: the largest lever for chains, with a latency trade-off

- **Thinking on, reasoning preserved.** tool-eval-bench does this natively; spark-bench needs 0620 `reasoning`. GLM's
  agentic results are reported with thinking on, and the misses here (call counts, polling, visible values) are
  planning failures. Use effort **low** as the first try: it plans with much less latency than high. 0150's MiaAI-Lab
  measurement found low effort already enough for structured output.
- **The trade-off in each benchmark's own formula** (estimates; the §5 runs replace them):
  - **tool-eval-bench:** the headline is quality, so thinking can only help it, but deployability falls with latency.
    At a 1.5 s median turn, responsiveness is 74. At 6 s it would be 26, and 0.3 × (74 − 26) is about −14
    deployability.
  - **spark-bench TrueScore:**
    - Responsiveness at 2.66 s median is 88; at 8 s it would be 71, about −0.6 TrueScore.
    - Efficiency falls on the non-agentic domains: from about 100 to about 30 costs about −1.0.
    - Break-even therefore needs Quality to rise by about 3 points. Agentic is one of the difficulty-weighted
      capability domains, so agentic 82.6 → ~95 alone is worth roughly +2 to +3 Quality.
    - Expect the TrueScore headline to stay about level, with agentic and planning up.
- **Keep** `temperature` ≤ 0.3 for agents. Keep `parallel_tool_calls` at its default: the model's parallel calls are
  parsed and streamed correctly.

### 4.3 Considered, not done

- **A call gate for `required`:** 0610's grammar already enforces it exactly in prod (`GLM53_TF_GRAMMAR=1`).
  Upstream's gate would only be a cheaper duplicate.
- **`thinking_budget`:** upstream-style, it would cap high-effort reasoning inside long agent loops and avoid `length`
  finishes, which spark-bench scores as 0. It needs an engine-side forced `</think>`, touching the batcher, drafts
  and exactness, so it is a separate design. Prod `MAX_TOKENS=32768` covers the benchmarks meanwhile.
- **Streaming arguments incrementally:** clients receive calls whole when the reply ends. Neither benchmark needs
  incremental streaming.
- **The tester's checkpoint differs from ours** (Mia TR3 against the abliterated one), so their numbers are not our
  baseline. §5 measures ours.

## 5. Test plan (run in W21: results in §7)

Runner: `scripts/tooleval/run.sh {setup | teb | sb | both | chains} {off | low | high} [label]`.
- Defaults: `BASE_URL=http://127.0.0.1:8000/v1`, `MODEL=GLM-5.3-Flash-EXL3`.
- Results go to `results/tooleval/<date>-<bench>-<mode>-<label>/`. Each run saves the benchmark reports, the exact
  command, and `/health` and `/v1/models` before and after.
- The script refuses to start if the model isn't listed.
- The thinking modes send `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`) plus a top-level
  `reasoning_effort`; prod has `GLM53_TF_EFFORT_FIELD=1`.
- spark-bench runs `--uncapped --timeout 0 --tier all --skip-throughput --repeats 2 --temperature 0.3`, as the tester
  did.

**Phase B, baseline** (current prod image b10: 0620 absent, prod.env unchanged, 4 slots; nothing else on the
endpoint).

| Run | Command | Compare with | Wall (est.) |
| --- | --- | --- | --- |
| B1 | `run.sh both off baseline` | the tester's 90 / C 75%, and 92.4 / agentic 82.6 (their checkpoint) | ~0.5 + ~1.5 h |
| B2 | `run.sh both high baseline` | B1: what thinking buys at prod's default effort | ~1.5 + ~4 h |
| B3 | `run.sh chains low baseline` (spark-bench agentic ×3 repeats) | B1/B2 on C + agentic only | ~1 h |

**Phase F, with fixes:** image b11 = b10 + 0620; `GLM53_TF_TOOL_FIXES=all`; everything else as B.

Gates first:
- The load prints the `tool calling (patches/0620)` line.
- `scripts/canary.py`, exact 10/10, batchexact 4/4, and reply sha `8794a3463259cc2f` are unchanged. They are
  tool-free, and 0620 is host-only.

| Run | Command | Question |
| --- | --- | --- |
| F1 | `run.sh both off fixes` | Thinking off: expect about B1 (no defect was in the tester's path), apart from TC-44 |
| F2 | `run.sh both high fixes` | Against B2: does `reasoning` help spark-bench agentic with thinking on? |
| F3 | `run.sh chains low fixes` | Against B3, and the effort choice (low against high) |
| F4 | `GLM53_TF_TOOL_FIXES=history,choice,args` then `chains high` | Isolates `reasoning` and `thinkcalls` (only if F2 differs from B2) |

**How to read the results:**
- tool-eval-bench at temperature 0 is greedy, and our engine is bit-exact, so a rerun gives identical replies. One run
  per configuration is exact, but a difference between configurations can come from one flipped token. Category C
  has only 4 scenarios, so confirm C changes with `TEB_TEMP=0.3 TEB_ARGS="--categories C --seed N" run.sh teb <mode>`
  for 3 seeds.
- spark-bench's repeats at temperature 0.3 give its own variance: Pass@1 against Pass@K, and reliability.
- **D5's frequency:** count thinking-on turns with `finish_reason: stop`, empty content, and `<tool_call>` in the
  reasoning. For spark-bench, use `SPARK_BENCH_DUMP_DIR` (`<run>/sb-dump`); for tool-eval-bench, the run's JSON.
  With 0620, the same turns come back as `tool_calls`.
- Check that D2 is being exercised: F2's request log (0300) should show a higher `cached` prefix share on agentic turns
  than B2's, because the history now matches the generated tokens.

**Adopt** `GLM53_TF_TOOL_FIXES=all` in `config/prod.env` if all of these hold:
- F1 ≥ B1 − 1 scenario in every category.
- F2 agentic ≥ B2 agentic.
- Nothing regresses outside tools.

Separately, recommend the thinking setting to clients from B2/B3/F2/F3 (quality against the latency subscores in
§4.2). The user decides the headline configuration.

## 6. Files

- `patches/0620-glm-tool-calling.patch`: the patch.
- `tests/test_tool_fixes.py`: host tests. Run with
  `GLM53_TF_TOKENIZER_DIR=<ckpt tokenizer dir> PYTHONPATH=<tree>/src pytest -q`.
- `scripts/tooleval/run.sh`: installs and runs both benchmarks.
- `docs/PATCHES.md`: the 0620 row, section and tests entry.

## 7. Results (W21, production b11)

Full tables, per-scenario numbers and caveats: [docs/RESULTS.md](RESULTS.md) W21; run folders under `results/tooleval/`
([what is published](../results/tooleval/W21-scripts/README.md)). One run per configuration on our abliterated
checkpoint; tool-eval-bench at temperature 0, spark-bench at 0.3 with 2 repeats.

| configuration | tool-eval-bench | chains (C) | spark-bench TrueScore | agentic |
| --- | --- | --- | --- | --- |
| fixes off, thinking off (B1) | 90 | 8/8 | 94.0 | 90.4 |
| fixes on, thinking off (F1) | 90 | 8/8 | 91.4 | 90.4 |
| fixes on, thinking high (F2) | 91 | 8/8 | 91.0 | 94.9 |
| fixes on, thinking low (F3, C + agentic only) | - | 8/8 | - | **97.3** |

Category C was also 8/8 in all 9 temperature-0.3 runs with `--seed 1/2/3` (fixes off, fixes on with thinking off, fixes
on at high).

**Recommendation.**
- **Agent work: thinking on at effort low.** It gives the best agentic score (97.3), chains 8/8, and the fastest
  agentic runs (median 14.9 s a scenario, against 20.1 s thinking off and 18.2 s at high). Thinking fixes AG-04 (0.23 ->
  1.00) at either effort.
- **High effort** also helps agentic (94.9) and tool-eval-bench (91), but long generations can reason up to the 32k
  `MAX_TOKENS` cap and end on `length`: two of spark-bench's visual scenarios did (visual 94.3 -> 54.1), which pulls
  the TrueScore back to level (91.0).
- **Thinking off** remains the setting for headline numbers comparable to the tester's (tool-eval-bench 90, TrueScore
  91.4 with fixes on).
- **Keep the 0620 fixes on** (`GLM53_TF_TOOL_FIXES=all`). With thinking off, fixes on and off are identical scenario by
  scenario on tool-eval-bench and on spark-bench's agentic domain.

**Why fixes off scores higher on TrueScore (94.0 vs 91.4).** The gap is two non-agentic scenarios. VIS-05 produced the
byte-identical HTML in both runs and was graded differently (grader noise). RR-01 (safety) is a real difference: its
prebuilt history has an assistant tool-call turn with `content: null`. Without 0620 the template renders it as the text
`None` (defect D1); with 0620 it renders empty, which matches what the model itself produces for a content-free call.
The two prompts lead to different replies: re-sending the history on production with 10 seeds each, the `"None"`
render gave the SIGTERM-only answer the grader wants 10/10, the empty render 4/10 (3 more appended `&& echo …`, which
the strict grader fails, and 3 inspected the process again first). So the unfixed render happens to suit this one
prompt. That costs one of spark-bench's 11 safety scenarios; nothing else measured moved, and the fix stays.

**Seeds and spark-bench repeats.** A request without `seed` does not get a fresh random seed on our server: it samples
with a seed derived from the prompt, so the same request at the same temperature returns the same reply. spark-bench
sends no seed, so its repeats of a scenario are often identical (27 of 63 repeated requests in F2 were byte-identical),
and its Pass@K, reliability and repeat spread understate the real variance. tool-eval-bench's seeded runs (`--seed N`)
do vary. To measure spread with spark-bench, give each repeat its own `seed` in the request body.

**Comparison with the independent tester (issue #6).** They ran kit `c747c92` with the non-abliterated Mia TR3
checkpoint, thinking off: tool-eval-bench 90 with chains 6/8, TrueScore 92.4 with agentic 82.6. Our thinking-off runs
give 90 with chains 8/8, TrueScore 91.4-94.0 and agentic 90.4. The checkpoints and the server versions differ, so this
compares two setups, not the server alone; it does show that the chain misses do not reproduce on ours, in line with
the audit in §3 that found no server defect on the chain path.
