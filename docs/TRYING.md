# Ways to run and try it

> Work in progress: knobs and defaults may change. Everything here was run on one pair of DGX Sparks.

All commands run on the head node (rank 0) from the repo checkout. `scripts/serve.sh` reads `config/prod.env` (the
production config, copied from `config/prod.env.example`) unless `CONFIG=path` is set; a non-empty environment
variable overrides the same key in the file for one start (`MAX_TOKENS=16384 scripts/serve.sh start`). Setting up
from scratch: the README's Quickstart, or [`AGENTS.md`](../AGENTS.md) (step by step, with checks). Placeholders: `<worker-ssh>` (ssh target of the worker node, e.g.
`user@<worker CX7 address>`), `<head-ip>` (the head's address on the CX7 link), `<head HF cache>` / `<worker HF cache>`
(absolute paths of each node's `~/.cache/huggingface`).

Only one serving stack fits on the pair at a time: `scripts/serve.sh start` refuses to run while any other CUDA
process is up on either node. Stop vLLM (or anything else) first.

## 1. Pick a configuration

| Config | Copy from | Context | Concurrency | KV | Use it for |
| --- | --- | --- | --- | --- | --- |
| **Production** (4 requests, shared pool) | `config/prod.env.example` | up to 1,048,576 a request, 1,048,576 shared by the 4 | 4 | FP8 latent, paged pool | the default: agents with subagents, several sessions at once (~78 tok/s aggregate at 4 streams), the fastest prefill (~1,600 tok/s), NVMe session tier |
| Single-stream, long context (earlier) | `config/prod-single.env.example` | 524,288 | 1 (others queue) | bf16 latent | one request at a time with bf16 KV (1,266 tok/s prefill at 28k) |
| 4 x 256k batch (earlier) | `config/prod-batch.env.example` | 262,144 a slot | 4 | FP8 latent | the pre-pool batch config, 4 GiB session store; only if your nodes have more free memory (worker hit 7.3 GiB in stress) |
| Debugging baseline (upstream-like) | `config/minimal.env.example` | 32,768 | 1 | per-head K/V (upstream) | not for normal use: a baseline with few patches active; check a problem against it |

```bash
cp config/prod.env.example config/prod.env
$EDITOR config/prod.env
scripts/serve.sh build       # image on the head node, copied to the worker
scripts/serve.sh preflight   # both nodes: ssh, docker, image, link, weights, memory
scripts/serve.sh start
scripts/serve.sh status      # logs 0 | logs 1 | stop | restart
```

The other configs are files next to it; pass them with `CONFIG=`, e.g. `CONFIG=config/prod-single.env
scripts/serve.sh start` (and on every other `serve.sh` command while it runs).

### Single-stream (524k)

`config/prod-single.env.example`. One request runs at a time; the session store keeps other conversations'
states so switching back to one costs ~0.4-1 s instead of a re-prefill. Keep `GLM53_TF_SESSION_GIB` at 4-6: with
12 GiB at 524k the store filled under agent traffic and the pair died of unified-memory OOM. `stop` strings end the
reply, but the engine keeps decoding silently to EOS / `max_tokens` before the next queued request starts.

### Production: 4 requests over a shared 1M-token pool (FP8 KV)

`config/prod.env.example`. Four requests decode together, and their latent KV lives in one paged pool of 1,048,576
tokens (patch 0290): one request can grow to 1M tokens, and admission reserves prompt + `max_tokens` pages, spills
idle slots' sessions to the store when the pool is short, or waits. A request alone prefills in 4,096-row chunks
(0335); beside decoders, prompts prefill in 2048-token pieces between the others' rounds, so a long prompt delays the
decoders by a few seconds at most but takes longer itself. What else the config turns on, each measured in
`docs/RESULTS.md` W1-W10:

- the NVMe session tier (0250, `GLM53_TF_SESSION_DISK=/sessions`, up to 64 GiB a node): an evicted ~39k-token session
  comes back in ~0.4 s instead of a 31 s re-prefill, also after a restart. `serve.sh` mounts
  `HEAD_SESSIONS` / `WORKER_SESSIONS` (default `<HF cache>/../glm53-tf/sessions`) at `/sessions`;
- shared system-prompt reuse (0310, `GLM53_TF_PREFIX_SHARE=1`): a new session resumes at the end of a system prompt
  another session already prefilled; a burst of 4 over an ~18k-token system prompt takes 28 s instead of 72 s;
- the RoCE all-gather (0230/0350, `GLM53_TF_COMM_BACKEND=roce`): decode +4-11%. It needs `/dev/infiniband` in the
  containers (serve.sh passes it) and falls back to NCCL on a setup failure; a run-time failure writes
  `/cache/roce-failed` and the next start uses NCCL (delete the file to retry). `GLM53_TF_COMM_BACKEND=nccl` turns it
  off. `docs/ROCE-FIX.md` has the validation stages to run on a new pair first;
- row-split prefill (0320), b12x bit 4 one-pass attention (0360), MLA expand v2 (0390): prefill, same bits;
- decode overlap (0370) and verify windows of up to 16 rows (0380): decode, same bits;
- the request log (0300, `GLM53_TF_REQUEST_LOG=/sessions/requests.jsonl`, no text): `python3
  scripts/traffic-report.py <head sessions dir>/requests.jsonl` summarizes reuse, sizes and speeds.

The same config at `CONTEXT=262144` without `GLM53_TF_KV_POOL_TOKENS` is the earlier 4 x 256k batch config (every slot's
caches allocated at full size). KV is stored as FP8 (`GLM53_TF_KV_DTYPE=fp8`): greedy replies differ from bf16 KV from the first tokens on;
quality checks held (MMLU-200 88.0%, refusals 0/10, needle 9/9 at 28k and 3/3 at 112k). To go back to bf16 KV
without a rebuild: `GLM53_TF_KV_DTYPE=bf16` with `GLM53_TF_SESSION_GIB=0` / `GLM53_TF_BATCH_SESSIONS=0`, or 3
slots, or the single-stream config. The memory gate (`MEM_GATE_GIB=108`, `MEM_GATE_DROP_CACHES=1`) needs
passwordless `sudo -n` on both nodes to drop page caches; without it the 4th slot may not fit at load.

### Debugging baseline

`config/minimal.env.example` (copy to `config/minimal.env`, then `CONFIG=config/minimal.env scripts/serve.sh ...`)
turns on only the decode-side patches (4-bit non-expert weights `q4mse`, deeper
DFlash2 drafts, real-text draft calibration, prompt-lookup drafts, `prefill_rows=auto`) at `CONTEXT=32768`, and leaves
fast prefill, the latent / FP8 KV cache, sessions and batching off. For upstream TensorFold behaviour exactly, also
set `GLM53_TF_NONEXPERT=bf16`, `GLM53_TF_PREFILL_ROWS=64`, `GLM53_TF_AUTO_FDRAFTS=5`, `GLM53_TF_CALIB=random`
and `GLM53_TF_LOOKUP=0`, or build an image with only some patches:

```bash
docker build -f docker/Dockerfile --build-arg PATCHES="0001 0002 0003 0004" -t glm53-tensorfold:min .
CONFIG=config/minimal.env IMAGE=glm53-tensorfold:min scripts/serve.sh start
```

(`serve.sh build` copies the image to the worker; with a manual `docker build`, build it on both nodes or copy it
with `docker save | ssh <worker-ssh> docker load`.)

## 2. Talk to it

```bash
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3     # every example config serves this port and name
curl -s $B/v1/models
curl -s $B/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'$M'", "messages": [{"role": "user", "content": "Summarize the CAP theorem in 3 bullets."}],
  "max_tokens": 2048, "stream": false
}'
```

- Streaming (`"stream": true`), tools (`tools`, `tool_choice`) and reasoning follow the OpenAI API. Reasoning comes
  as `reasoning` in the production config (`GLM53_TF_REASONING_FIELDS=reasoning`, as the vLLM kit); the other
  configs and `GLM53_TF_REASONING_FIELDS=both` also send `reasoning_content`.
- Images (production config, `GLM53_TF_VISION=1`, patch 0500): `image_url` content parts with a `data:` or `http(s)`
  URL, up to 8 a request; example in the README ([image input](../README.md#image-input)), limits and knobs in
  [`VISION.md`](VISION.md).
- The response has a `tensorfold` object (decode tok/s, TTFT, prefill seconds, the knobs used) and a `speculative`
  object (rounds, drafted and accepted tokens). `usage.prompt_tokens_details.cached_tokens` shows a session hit.
- `/health` and `/metrics` (Prometheus) on the same port (patch 0150).

### Reasoning effort

Thinking is on by default. In the prod configs `GLM53_TF_DEFAULT_EFFORT=high` and `GLM53_TF_EFFORT_FIELD=1`:

| Request | Effect |
| --- | --- |
| nothing | thinking on, effort high |
| `"reasoning_effort": "none"` or `"minimal"` | thinking off |
| `"reasoning_effort": "low"` | low effort (good for long structured output: tables, many numbers) |
| `"reasoning_effort": "medium"` / `"high"` | high |
| `"reasoning_effort": "max"` | max |
| `"chat_template_kwargs": {"enable_thinking": false}` | thinking off (always works, without the field mapping) |
| `"chat_template_kwargs": {"reasoning_effort": "low"}` | low effort (template-level) |

For structured output use thinking on at low effort rather than thinking off: the MiaAI-Lab kit measured 0/6 garbled
long tables that way against 5-6/6 with thinking off.

### Sessions

Nothing to do: every request resumes from the longest stored prefix of the same conversation (the stored states
are keyed by the token prefix). With the single-stream config the store holds `GLM53_TF_SESSION_GIB` of other
sessions; in the batch configs each slot keeps its own conversation, the 2 GiB RAM store holds a few more, and in the
production config the NVMe tier keeps every evicted session (up to 64 GiB a node) and survives restarts. Check `cached_tokens` in `usage`. The same prompt sent again (a regenerate, a retry) resumes all
but its last <= 64 tokens (patch 0540); the resumed state is bit-identical to a fresh prefill. `"priority": "background"` (and opencode's
session-title requests, recognised automatically) waits behind foreground requests.

## 3. A/B a knob without a restart (`tf_knobs`)

Most speed knobs switch per request; the environment only sets their defaults. The response echoes the values
used in `tensorfold.tf_knobs`.

```bash
curl -s $B/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'$M'", "messages": [{"role": "user", "content": "..."}], "max_tokens": 256,
  "tf_knobs": {"lookup": 0, "auto_fdrafts": 5, "depth": "threshold"}
}'
```

| Key | Values | Patch |
| --- | --- | --- |
| `lookup`, `lookup_min` | 0/1, 1-64 | 0020 prompt-lookup drafts |
| `auto_fdrafts` | 1-7 | 0010 DFlash2 depth of `auto` |
| `depth` | `cost`, `threshold` | 0071 |
| `calib_online` | 0/1 (refused in batch mode) | 0070 |
| `prefill_rows` | 1 to `GLM53_TF_PREFILL_ROWS_MAX`, or auto via the env | 0003 / 0085 |
| `expert_loop`, `longctx_graphs`, `profile` | 0/1 | 0006, 0050, 0005 |
| `fast_prefill`, `fp8_prefill`, `prefill_overlap` | 0/1 | 0091-0093 |
| `fat_experts` | 0/1 | 0170 |
| `moe_glue`, `mtp_window`, `hc_fused`, `attn_bm32` | see `docs/PATCHES.md` 0190 | 0190 |
| `b12x` | bit mask; 4 = one-pass sparse latent attention (production), 1 / 2 not adopted | 0240 / 0360 |

Load-time settings (`GLM53_TF_NONEXPERT`, `GLM53_TF_LATENT_KV`, `GLM53_TF_KV_DTYPE`, `GLM53_TF_BATCH`, the prefill
buffer size, calibration) answer HTTP 400 if sent per request. Every bench script takes `--extra '{"tf_knobs": {...}}'`.

## 4. Draft policies (`model@policy`)

Append a policy to the model name, e.g. `"model": "GLM-5.3-Flash-EXL3@f7"`. `"draft": false` decodes that request
serially (one token a round, no drafts): the reference every drafted reply must equal.

| Policy | Meaning |
| --- | --- |
| `auto` | the default: per round, MTP (`c3:0.35`) or DFlash2 (`fc7:0.3` with patch 0010), whichever has committed more tokens per ms in this request; prompt-lookup drafts when they pay (0020); sampled requests use `a:0.6:0.85` |
| `N` | N MTP drafts a round |
| `a` / `a:LOW:HIGH` | 1-3 MTP drafts from the running acceptance |
| `cN:P` | up to N MTP drafts while the drafts' probability product stays >= P |
| `fN`, `fcN:P`, `fa:...` | the same with DFlash2 drafts (needs the drafter on both nodes) |
| `lN` / `lN:M` | prompt-lookup: up to N tokens that followed an earlier occurrence of the last M tokens |
| `o`, `om[N]`, `of[N]` | cost-derived depth (0071): each round the depth with the most expected tokens net of cost |

Measured (tok/s, single stream, same load): `@f7` is fastest on structured text (kit structured 103 vs 86 for
`auto` then) but slow on prose (essay 31 vs 42); `@c3:0.35` is best on sampled chat. `auto` is the balanced default.

## 5. Fast boot

`scripts/prepare.sh` writes each node's prepared weight folder once (split, re-quantized, tiled exactly as the load
builds them, ~83 GB a node); later starts read it with an O_DIRECT reader and reuse the cached calibration: ready in
~35 s instead of ~8 min. `serve.sh start` also writes the folder after a full load (`GLM53_TF_PREPARED_WRITE=1`).
`scripts/prepare.sh status` lists the folders; `--force` rewrites them. A folder is keyed by the checkpoint, the rank,
`GLM53_TF_NONEXPERT`, torch and the engine source, so a new image or weight format prepares again.

## 6. Benchmark it

```bash
B=http://127.0.0.1:8000; M=GLM-5.3-Flash-EXL3
# decode cells (tf / kit / tweet / edit) and prefill (ctx), median of 5
python3 bench/glmbench.py --base $B --model $M --suites tf,kit,tweet,edit --out results/mine-decode.json
python3 bench/glmbench.py --base $B --model $M --suites ctx --ctx 8000,32000,128000 --out results/mine-ctx.json
# sessions, follow-ups, concurrency, prefill stall, batch slots
python3 bench/multiturn.py --base $B --model $M --modes sessions,followup --doc 32000 --out results/mine-sess.json
python3 bench/multiturn.py --base $B --model $M --modes concurrent,stall,slots --streams 1,2,4 --out results/mine-conc.json
# quality: MMLU-200 + refusals
python3 bench/quality.py --base $B --model $M --label mine --out results/mine-quality.json
# opencode-shaped tool calls (21 cases; corruption = leaked GLM markup or bad JSON)
GLM_URL=$B/v1/chat/completions GLM_MODEL=$M python3 bench/toolcall_harness.py --reps 5 --out results/mine-tools.json
# shared system-prompt reuse (0310): sessions and a burst of 4 over one ~18k-token system prompt
python3 bench/prefixshare.py --base $B --model $M --system 12000 --out results/mine-prefix.json
# memory worst case (batch configs): 4 conversations grown to ~250k, then a 32k turn beside 3 decoders
python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 250000 --stress-step 60000 \
    --stress-final 32000 --long-tokens 512 --mem-hosts local,$WORKER_SSH --out results/mine-stress.json
# A/B a knob on reply agreement and needle retrieval
python3 bench/fp8ab.py --base $B --model $M --modes agree,needle --knob fast_prefill
```

`glmbench.py` works against any OpenAI-compatible server (`--key` for a bearer token), so the same cells run against
vLLM for a fair comparison. Use one stack at a time and nothing else on the GPUs. The first long prompt after a load
compiles kernels; the configs' `WARMUP_LENGTHS="4096 16384"` pays that at start, otherwise run a warm-up first.

## 7. Check exactness

```bash
python3 bench/glmbench.py --base $B --model $M --suites exact              # drafted == serial, 10 cases
python3 bench/multiturn.py --base $B --model $M --modes batchexact         # batched == alone (batch config)
```

Both should report every case identical. They compare SHA-256 of replies decoded with drafts against `"draft": false`
(and 4 concurrent requests against the same requests alone). Settings that change the arithmetic (q4mse, fast
prefill, FP8 KV) change replies against other settings, but exactness holds within each setting.

## 8. Operations

```bash
scripts/serve.sh preflight   # CONTEXT, ssh, docker, image, CX7 netdev / HEAD_IP / RDMA port, weights, MemFree, sudo -n,
                             # CUDA processes, RoCE marker, driver and vm.min_free_kbytes parity, GPU state (both nodes)
scripts/serve.sh canary      # the post-load probes on demand (fails on degenerate output or a dead drafter)
scripts/serve.sh xid 2h      # NVIDIA Xid events on both nodes
scripts/serve.sh watch       # watchdog loop; or the systemd user units in scripts/systemd/
scripts/serve.sh gpucheck    # GB10 clock / power-clamp / slow-state check of both nodes (strict: gate benchmarks on it)
python3 scripts/gpuwatch.py status                           # the GPU watch's last samples (docs/OPS-GPUWATCH.md)
python3 scripts/traffic-report.py <head sessions dir>/requests.jsonl   # the request log (0300), no text
```

`preflight` also runs the GPU check (`GPUWATCH_PREFLIGHT=off|on|strict`, default `on`): a clock- or power-clamped
GB10 (it survives warm reboots; a full power drain clears it) is a start problem. The GPU watch has its own user
units (`scripts/systemd/glm53-gpuwatch.{service,timer}`). `CPUSET` (or `HEAD_CPUSET` / `WORKER_CPUSET`) passes
`--cpuset-cpus` to both containers; it is not used in production (pinning measured +0.4%).

The watchdog timer (`scripts/systemd/glm53-tf-watchdog.{service,timer}`, edit `WorkingDirectory` and `CONFIG`)
checks `/health` every minute and, with `WATCH_HEAL=1`, restarts both ranks after repeated failures. Watch
`MemAvailable` on the worker node during the first days of a new config; it is the binding node (2 GiB less memory).

## 9. Roll back

| To undo | Do |
| --- | --- |
| a per-request knob | drop it from the request; nothing persists |
| an env knob | remove it from the config and `scripts/serve.sh restart` (defaults are upstream behaviour) |
| FP8 KV | `GLM53_TF_KV_DTYPE=bf16` plus the memory changes in section 1 |
| batching | use `config/prod-single.env.example` |
| the shared KV pool | drop `GLM53_TF_KV_POOL_TOKENS` and set `CONTEXT=262144` (the earlier 4 x 256k config) |
| RoCE | `GLM53_TF_COMM_BACKEND=nccl` (or leave `/cache/roce-failed` in place) |
| a W8-W10 knob (`PREFILL_PP`, `PREFIX_SHARE`, `B12X`, `MLA_EXPAND`, `DECODE_OVERLAP`, `MAX_DRAFT_ROWS`) | remove it from the config and restart; each is off by default and was adopted on its own A/B |
| a patch | build with `--build-arg PATCHES="..."` listing the ones to keep, and `IMAGE=<tag>` |
| everything | `scripts/serve.sh stop`; start your previous stack (vLLM kit or upstream TensorFold). Old images stay tagged; `IMAGE=<old tag> scripts/serve.sh start` |

Prepared folders and the kernel cache volume (`glm53-tf-cache`) are safe to delete; the next start rebuilds them.

## 10. Context smaller than expected

A request can use `CONTEXT` tokens of the config the pair was started with: prompt **plus** `max_tokens` (the reply
is reserved up front). Nothing truncates silently; a request past the limit is refused with HTTP 400. Check, in order:

1. **What the server serves.** `scripts/serve.sh start` logs `<config>: CONTEXT=... tokens a request` before it
   starts and `serving up to N tokens a request` once ready. On a running pair:
   `docker inspect glm53-tf-r0 | grep -o 'CONTEXT=[0-9]*'`. The production config serves 1,048,576.
2. **Which config.** `serve.sh` reads `config/prod.env` unless `CONFIG=` is set. `config/minimal.env.example` is the
   **debugging baseline: `CONTEXT=32768`**, one request at a time and the upstream per-head KV cache (~390 KB a token
   a rank). Raising its `CONTEXT` does not get far: past ~100k tokens that cache does not fit next to the weights
   (`serve.sh` refuses more than 131,072 on it). For long context use `config/prod.env` (`CONTEXT=1048576`, FP8
   latent KV at 7.4 KB a token, the shared pool) and leave `CONFIG` unset. An older checkout's `config/tensorfold.env`
   is not read any more unless named.
3. **The client.** Tell the client the real window, and keep its reply budget (`max_tokens`) well under it: see
   [Client settings](#client-settings) below. If long sessions fail only in one client, its context setting is the
   first suspect.
4. **Memory.** A node short of free memory at load (a desktop session, other containers, page cache: the load's
   rules read MemFree, not MemAvailable) makes the production config start with fewer slots (the log says
   `only N sequence(s) fit`): fewer concurrent requests, not a shorter context; each request can still grow to
   `CONTEXT` through the shared pool. A load that does not fit fails ("rank 0 exited"); it never shrinks `CONTEXT`.
   Stop what else runs on the Sparks, and keep `MEM_GATE_GIB` / `MEM_GATE_DROP_CACHES` from the example.

Measured on production (2026-09-29, `config/prod.env`): single prompts of 63k and 120k tokens, and two 9-turn
conversations growing by ~15k tokens a turn to 143k (thinking off, and thinking on with `max_tokens` 32,000 as
opencode sends it): every turn answered correctly, resumed all but the last turn's new tokens from the session
store (~11-12 s a turn), no refusals or waits.

Coming (a server patch in testing, not in this tree yet): `/v1/models` reporting the context as `max_model_len`, so
clients that read it configure themselves, and over-long requests answered with OpenAI's `context_length_exceeded`
error code and message ("This model's maximum context length is N tokens. However, you requested M tokens ...").
Until then the 400 (`invalid_request_error`) carries the engine's own message: "this request needs a M-token context
(P prompt tokens plus max_tokens R), and this server was started for N: ...". If R is large, the client's reply
budget is what hit the limit.

### Client settings

Production serves `http://127.0.0.1:8000/v1`, model `GLM-5.3-Flash-EXL3`, a **1,048,576-token** context window.
Suggested maximum output: **32,768** tokens (the config's `MAX_TOKENS`). What to set, and what a client shows when a
request does not fit:

| Client | Where | Set | If a request exceeds the window |
| --- | --- | --- | --- |
| opencode | `~/.config/opencode/opencode.json`, the model's `limit` (README Quickstart has the whole entry) | `"limit": {"context": 1048576, "output": 32768}` | without `limit.context` opencode never compacts a long session; it sends `max_tokens` = `limit.output` (capped at 32,000), which counts against the context. Past the window the session stops with the server's 400 message |
| Continue | the model entry in `config.yaml` (`defaultCompletionOptions`) or `config.json` | `contextLength: 1048576`, `maxTokens: 32768` | an unknown model defaults to a 32,768-token context: Continue drops older messages to fit that, so long sessions lose context without an error. Past the server window: the server's 400 message |
| Cline / Roo Code | provider "OpenAI Compatible", model info | context window 1048576, max output tokens 32768 | the default (128,000) makes Cline condense or truncate the conversation early; past the server window the task shows an API request failed with the 400 message |
| Open WebUI | Admin, Connections, OpenAI API: `http://127.0.0.1:8000/v1` | nothing required (its context options are for Ollama); set Max Tokens in the model's advanced params if you want a shorter reply budget | the chat shows the server's 400 error text |
| Anything OpenAI-compatible | base URL, API key any string | context window 1,048,576; max output 32,768 or less | HTTP 400 with an `error.message` naming the requested and allowed tokens |

**Clients that send neither effort nor `max_tokens`** get the server defaults: thinking at effort `high`
(`GLM53_TF_DEFAULT_EFFORT=high` in production) and up to `MAX_TOKENS` (32,768) tokens a turn. On a hard prompt that
can be one long reply that runs for minutes (issue #11). Set the client's max output tokens and, for agents,
`reasoning_effort: "low"` (README: best agentic score, fastest runs; "Reasoning effort" above), or lower
`GLM53_TF_DEFAULT_EFFORT` / `MAX_TOKENS` in the config for every client.

The four requests of the production config share one 1,048,576-token pool: one request can use all of it; several
long ones at once wait for pages or spill idle sessions to the store (they are not refused).
