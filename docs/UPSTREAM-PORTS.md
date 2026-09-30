# Upstream TensorFold ports: patch 0600 (`glm-upstream-ports`), 2026-09-30

> **Update 2026-09-30 (W19, docs/RESULTS.md): in production** (image b10, on by default). Every check in section 4
> passed: a departed client's slot freed 0.18 s after the close (4 at once: 0.26-0.31 s), a queued request whose client
> left never ran, the 400s, `USR1` on both ranks, image URL hardening, `/health` token totals.


> Offline work: no GPU, the Sparks were not touched. Items 1, 2, 3, 4, 6 and 9 of the ranked port list in
> `docs/UPSTREAM-050-AUDIT.md` section 6, ported from upstream TensorFold (read-only clone at
> `~/.cache/tf-upstream-050`, `9cd52ab`, 0.5.0; the disconnect module from 0.3.6.2) onto our 0.3.4 + 0001-0590 as one
> host-only patch, `patches/0600-glm-upstream-ports.patch`. **Bits unchanged**: no kernel, forward, sampler or plan
> encoding changes; the engine sees the same prompt, sampling and knobs for every valid request. Upstream TensorFold is
> MIT-licensed (`LICENSE` at `9cd52ab`, "Copyright (c) 2026 TensorFold contributors"); the ported files say where
> each piece came from, and `NOTICE` lists the port.

## 1. What changed, by item

### 1.1 Client disconnect on the batcher (upstream 24afe5e, audit 3.1)

Before 0600, only a failed streamed write stopped a request. A **non-streamed** request whose client timed out ran to
its end or `max_tokens` (prod 32,768: ~5-14 min of one of the 4 slots); a queued or prefilling request (no delta
written yet) and a stream inside a held-back tool call were never checked; an exception in the per-token callback left
`_collect` without cancelling the job, so the slot decoded on unheard.

Now:

- `tensorfold/server/cancellation.py` (upstream's module, `PrefillGuard` left out): `socket_cancellation(sock)` is
  true once the client closed its end (`poll` + a zero-byte `MSG_PEEK` read; a pipelined next request is data, not a
  close). Ours uses `poll` instead of `select` (select() refuses descriptors past 1023).
- The handler builds it for every completion (streamed or not) and passes `cancelled=` to `App.run`
  (`GLM53_TF_DISCONNECT=0` turns the check off; validated at load).
- `App.run` (upstream's rules): a client gone before the request starts raises `RequestCancelled` without an engine
  call; the callback checks `cancelled()` every call, text or not, and returns True; after a stop / failure it
  returns True at once; any exception inside it is kept and raised after `generate` returns (never into the engine);
  a stopped request raises `RequestCancelled` after `generate`, so **a departed client gets nothing more** (no final
  chunk, no `[DONE]`, no JSON reply). Streamed writes treat any `OSError` as a departed client.
- **The batcher** (`batch.Batcher._collect`, ours): the check rides on the callback (`on_tokens.cancelled`, forwarded
  by 0150's `TrackedEngine`), and the HTTP thread waiting for its request polls it every
  `GLM53_TF_DISCONNECT_POLL_MS` (default 250 ms; 10-10,000, checked at load) while no token arrives, and at most as
  often between tokens. When it turns true, or the callback raises, the thread sets `job.cancel` and wakes the loop:
  a **queued** job is dropped by the next `_plan` (rank 1 never hears of it); an admitted one, **prefilling** or
  decoding, is cancelled through the round plan rank 1 receives (the same path as every cancel since 0120, including
  0370's plan riders). A callback failure is raised to the caller once the job's end marker came.
- `GlmApp.run` logs a cancelled request in 0300's request log as `finish: "cancelled"` with what it did, not as an
  error.
- **Lone path (`GLM53_TF_BATCH=1`, not prod)**: a request waiting for the engine lock is refused before it starts
  (new), and a callback failure no longer raises into the decode loop (it used to leave rank 1 out of step). A running
  lone request still decodes on unheard to its end: the lone loops ignore the callback's return value by design (0160;
  stopping would need a stop both ranks share). Unchanged from before.

Behaviour change for clients: a client that half-closes its socket (`shutdown(SHUT_WR)`) after sending the request is
treated as gone (upstream's rule). No known client does this; `GLM53_TF_DISCONNECT=0` restores the old behaviour.

### 1.2 Request hardening (2bc35c4, audit 3.2)

- `do_POST` wraps `app.check(body)` and `app.tokenize(body)`: a jinja2 `TemplateError` (the template's
  `raise_exception`) is a 400 "the chat template rejected the request: ...", a `ValueError` a 400 with its message, any
  other exception a 400 with its message, logged with its traceback. `App.run` turns a template refusal found only
  there (an app whose `check` does not render) into the same 400.
- `App.check` validates, before a stream's headers: `chat_template_kwargs` must be an object or null; `temperature` /
  `top_p` finite numbers; `top_k` / `seed` / `max_tokens` / `max_completion_tokens` integers (`20.0` and `"20"` are
  fine, `20.5` is not); no booleans anywhere (upstream's `parse_numbers` rules). `sampling_for` reads the same parsed
  values, **unclamped**, so every valid request gets exactly the `Sampling` it got before (tested against the 0.3.4
  formula).
- A non-UTF-8 body (and a malformed `Content-Length`) is "the request body is not JSON" (400).
- Engine failures: a 500 JSON body / a `server_error` stream event, as 0150 already did, now also logged with the
  traceback. `Problem` gained a `status` (503 for a capacity refusal, below).
- Logs are redacted (`server.redact`): URL credentials, query strings / fragments and `data:` payloads become
  `<redacted>`.

Behaviour change for clients (all were accepted before, most by accident): a boolean in a sampling field (read as 1 /
0), a non-finite `temperature` / `top_p` (NaN reached the sampler), a non-integral `top_k` / `seed` (truncated), a
malformed `top_p` / `top_k` / `seed` on a greedy request (not read), `chat_template_kwargs` as `[]`, `""`, `false`,
`0` (read as absent) or a list of pairs (read as a dict): each is now a 400. A streamed request with a bad field gets
a 400 instead of 200 + role chunk + error event. No knob: these are malformed requests.

### 1.3 `kill -USR1` stack dumps (a586f3d, audit 3.5)

`cli.cmd_serve` registers `faulthandler.register(SIGUSR1, all_threads=True)` before it hands over to
`_serve_cuda`, so both CUDA ranks dump every thread's Python stack on `docker exec glm53-tf-rN kill -USR1 1` (PID 1 is
the `tensorfold` process: `docker/entrypoint.sh` execs it), read with `scripts/serve.sh logs N`. The process keeps
running.

### 1.4 Image URL hardening (391713e + 05cbb54, b4042e8; audit 3.4) for 0500's vision

0500 fetched `http(s)` URLs with `urllib` on any port, followed redirects unchecked and never looked at the address: a
blind SSRF from any prompt an agent forwards (`GLM53_TF_VISION=1` in prod). Now `vision_prep.load_bytes` calls
`image_fetch.fetch_image` (upstream's `vision/images_http.py`):

- **URL**: `https` on port 443 only, a host, no credentials, fragment, backslash, whitespace or control characters,
  at most 4,096 characters; `localhost`, `metadata.google.internal`, `instance-data`, `metadata` refused by name.
- **Addresses**: the host is resolved on a bounded pool (4 resolver threads) and **every** address must be public:
  `is_global`, not multicast / reserved, not 192.0.0.0/24 (incl. Oracle's 192.0.0.192) or Azure's 168.63.129.16; IPv6
  not IPv4-mapped / 6to4 / Teredo, and (ours) not NAT64 `64:ff9b::/96` / `64:ff9b:1::/48` or IPv4-compatible
  `::/96`. That excludes RFC 1918, loopback, link-local (169.254.169.254), CGNAT 100.64/10 (Alibaba's 100.100.100.200),
  0.0.0.0, fc00::/7 (AWS's fd00:ec2::254), fe80::/10.
- **Connection**: to the checked address itself (no second lookup a rebinding DNS could answer differently), TLS
  verified for the host name, `Accept-Encoding: identity`.
- **Redirects** (301/302/303/307/308, at most 3): each `Location` goes through the URL check, the resolution and the
  address check from the start.
- **Reply**: status 200, identity encoding, a declared `image/jpeg` / `png` / `webp` (upstream) or `gif` (ours: 0500
  decodes a GIF's first frame), at most `GLM53_TF_VISION_MAX_BYTES` by `Content-Length` and by bytes read.
- **Time**: one deadline for all steps of a download (`GLM53_TF_VISION_FETCH_TIMEOUT`, 10 s) inside one for all of a
  request's downloads (`GLM53_TF_VISION_FETCH_TOTAL_S`, 30 s); a watchdog closes the socket if a TLS handshake or a
  header read stalls.
- **Bounded preparation**: at most `GLM53_TF_VISION_PREP_SLOTS` (16) requests fetch / decode / preprocess images at
  once, at most `GLM53_TF_VISION_PREP_WAITERS` (128) wait up to `GLM53_TF_VISION_PREP_WAIT_S` (60 s); past that, HTTP
  503 `server_error` (retry shortly). Decoded pixels stay bounded by 0500's `GLM53_TF_VISION_MAX_PIXELS`.
- **No URL in any message**: fetch errors say what failed, not where; 0300's request log never held text or URLs;
  server logs of refused / failed requests are redacted (1.2).
- **Local testing only**: `GLM53_TF_VISION_FETCH_HTTP=1` allows `http://` and any port, `GLM53_TF_VISION_FETCH_PRIVATE=1`
  any address (both default 0; neither may be set in prod). `GLM53_TF_VISION_FETCH=0` still turns URL fetching off.
- Ours, beyond upstream: a reply with `Connection: close` / HTTP/1.0 is read to its end (upstream's per-read
  `settimeout` hits a socket `http.client` already closed); a resolver slot is waited for until the deadline instead
  of failing at once.

Behaviour change for clients: image URLs over `http://`, on a non-443 port, to private / local hosts, or answering
with another type (`image/bmp`, `application/octet-stream`, none) are now 400s. Public HTTPS images and `data:` URLs
behave as before (same bytes, same preprocessing, same rows).

### 1.5 Small bundle (e6be5ff, 88407d1, c30a1a9; audit 2.2, 3.3, 3.6)

- **`return_token_ids`**: `"return_token_ids": true` adds the reply's token ids (the ones `usage.completion_tokens`
  counts) to the response's `tensorfold` block; streamed: the final chunk's `tensorfold` (ours always sends it).
  Nothing changes for a request that does not ask.
- **Token-exact length at a stop string**: our 0160 scope is kept (a stop string ends the visible answer, not the
  reasoning or a tool call's text: the OpenAI / vLLM behaviour our clients were tested with; upstream matches the whole
  text). When a round's tokens complete a stop, the callback now finds the token that completes it (upstream's
  per-token rule), so `completion_tokens` (and `token_ids`) are the same however the tokens arrived: drafted == serial.
  Before, it was the tokens received in the round that showed the stop (drafted and serial could differ by up to a
  round). Content was already exact; the engine's `sha256` (all decoded tokens) is untouched.
- **Tool arguments one bracket short (#87)**: in 0002's GLM parser, a value typed `array` / `object` by the tool's
  schema that is not JSON as written and only stops short of its closing brackets (outside strings) is closed with
  upstream's `closed_json` and kept if it then parses to that type; otherwise the text, as before. Valid JSON, string
  parameters, untyped parameters and over-closed values are unchanged. This changes only the parsed arguments of a
  malformed call (which previously reached the client as a string where an array / object was declared); the model's
  output is untouched.

### 1.6 Live token totals on `/health` (51b098d, c4bf25f; audit 2.2)

`GET /health` keeps every 0150 field (`ok`, `mode`, `uptime_s`, `inflight`, `oldest_s`, `idle_s`, `requests`,
`errors`, `fatal`, `stalled`) and adds upstream's JSON shape: `backend: "tensorfold"`, `busy`, `requests_running`,
`requests_total`, `prompt_tokens_total`, `completion_tokens_total` (ended replies plus the running replies' tokens so
far: it moves as tokens are emitted), `prefill_seconds_total`, `decode_seconds_total`, `cached_tokens_total`,
`rounds_total`, `drafted_total`, `accepted_total` (engine stats folded when a request ends; the batch and lone paths
now report `drafted` / `accepted` in the stats), `streams` (`decoding` / `prefilling` / `max` slots of the batch
engine) and `context_length`. As upstream, a failed request still counts its prompt and the tokens it emitted.
`/metrics` is unchanged. Merged into 0150's `cuda/health.py` (upstream's own module of that name is an add/add
conflict on a rebase: section 5).

## 2. Knobs

| knob | default | where | meaning |
| --- | --- | --- | --- |
| `GLM53_TF_DISCONNECT` | `1` | rank 0, load | `0`: no socket check (only a failed streamed write stops a request) |
| `GLM53_TF_DISCONNECT_POLL_MS` | `250` | rank 0 (`Batcher.__init__` on both ranks checks it) | how often a waiting HTTP thread checks its client, 10-10,000 |
| `GLM53_TF_VISION_FETCH_HTTP` | `0` | rank 0 | `1`: `http://` and any port (local testing only) |
| `GLM53_TF_VISION_FETCH_PRIVATE` | `0` | rank 0 | `1`: any resolved address (local testing only) |
| `GLM53_TF_VISION_FETCH_TOTAL_S` | `30` | rank 0 | all image downloads of one request |
| `GLM53_TF_VISION_PREP_SLOTS` / `_WAITERS` / `_WAIT_S` | `16` / `128` / `60` | rank 0 | concurrent image preparations, waiting requests, the wait; then 503 |

None enters the 0140 calibration key or a rank header; rank 1 reads none of them for its work.

## 3. Tests (host only)

| file | cases | what |
| --- | ---: | --- |
| `tests/test_upstream_ports.py` | 169 | every item above on fake engines, a toy tokenizer and local sockets (module docstring), incl. a non-streamed request whose client closes stopping within a few rounds, a request cancelled while prefilling (the real `Batcher._collect`) streamed and not, 68 malformed-field cases (17 fields x streamed / not x App / GlmApp), the pre-0600 sampling formula, `cmd_serve`'s registration order, 28 blocked + 5 public addresses, a DNS-rebinding-style redirect to 10.0.0.7 that never connects, a same-host rebinding caught at the redirect, 6 redirect targets refused, a local image host (types, gzip, sizes, redirects, loops, 404, a stalled server's deadline), the 503 bound, `closed_json`, the #87 repair, the stop count with 5 chunkings, `return_token_ids`, `/health` live totals |
| `tests/cuda/test_disconnect_patches.py` | 8 | both ranks in lockstep (0180's hostile fake model, the real `_plan` / `_execute` / `_finish` / `follow`, rank 0's message stream replayed on rank 1, 0370 riders off and on): a client gone while **queued** (never admitted on either rank), **prefilling** (cancelled before a token), **decoding**, and a callback that raises; rank 1's log (slot, sha, keeps, cancelled), trace and slot states == rank 0's; the other replies == fresh prefill + serial |
| `tests/test_vision_prep.py`, `tests/test_vision_server.py` | (changed) | their local `http://127.0.0.1` image servers now declare `image/png` and the tests set the local-testing knobs; the default refusing that URL is checked |

Existing suites pass on a fresh tree with patches 0001-0600 applied (section 6).

## 4. GPU check plan (short; one window, new image = b9's list + 0600)

The patch is host-only, so one short session on the live pair covers it. Build the image with b9's patch list plus
0600 (prod.env unchanged; the new knobs keep their defaults), start with `scripts/serve.sh start`, then:

1. **Exactness gates** (the handler, `check`, `run` and `_collect` changed), W17's scripts
   (`results/W17/ab.sh` runs the first three):
   - `python3 bench/glmbench.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --suites exact`: **10/10**
     identical;
   - `python3 bench/multiturn.py --base http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --modes batchexact`: 4/4
     batched == alone;
   - the reply sha (`results/W5/ab.py`, as `ab.sh` runs it): **`8794a3463259cc2f`** (unchanged);
   - `scripts/serve.sh canary`: ok (tokens a round and decode tok/s as b9).
   ~15 min.
2. **Disconnect, non-streamed long request frees its slot**:
   - send a non-streamed chat request with `max_tokens` 32768 and a prompt that makes a long reply ("count from 1 to
     100000, one number a line"), `thinking off`, from a client with a 5 s timeout (e.g. `curl -m 5 ...`);
   - after curl exits, within ~1 s: `GET /health` shows `requests_running` 0 / `streams.decoding` 0 and `inflight` 0;
     `logs 0` has no error; the request log line (if `GLM53_TF_REQUEST_LOG` is set) says `finish: "cancelled"` with a
     few hundred `decode_tokens`, not 32,768;
   - the same with 4 such requests at once, then 4 normal requests: all 4 slots admit at once (no slot left decoding
     for a departed client);
   - a queued / prefilling one: start 4 long streams, then a 5th request with a ~100k-token prompt, kill its client
     after 2 s: it never admits (or its prefill stops at the next round); the 4 streams' replies are unchanged
     (their `token_sha` against a run without the 5th).
3. **Errors** (smoke): a request with `"temperature": true` streamed -> 400 JSON before headers; `chat_template_kwargs:
   "x"` -> 400; a non-UTF-8 body -> 400 "not JSON"; `docker exec glm53-tf-r1 kill -USR1 1` and `r0` -> stacks in
   `logs 1` / `logs 0`, both ranks keep serving (run one exact request after).
4. **Image URL over https** (vision on in prod): a chat request with a public HTTPS image URL (e.g. a PNG / JPEG on
   `https://upload.wikimedia.org/...`) answers as with the same image as a `data:` URL (identical `prompt_tokens` and
   reply, greedy); `http://` of the same image, `https://127.0.0.1/...`, `https://169.254.169.254/latest/meta-data/`
   and a URL redirecting to a private address are 400s with no URL in the message; the Sparks must reach the internet
   for this item (skip it otherwise; the host tests cover the rules).
5. **/health**: during a long reply `completion_tokens_total` grows between two polls; after it, the totals include
   its `rounds_total` / `drafted_total` / `accepted_total`; 0150's fields are still there (`scripts/serve.sh watch`
   reads `ok`).
6. Adopt if 1 passes unchanged and 2-5 behave as described; revert = the b9 image (no knob to undo: the patch keeps
   bits and the defaults are the new behaviour; `GLM53_TF_DISCONNECT=0` turns off only the socket check).

## 5. Rebase notes

On a rebase onto upstream >= 0.4.0 most of this collapses into upstream's own code: `server/cancellation.py` becomes
upstream's (ours adds `poll`), 2bc35c4's `parse_numbers` / template refusal live in `cuda/http.py` / `server/request_options.py`,
`image_fetch.py` maps onto `vision/images_http.py` (keep our NAT64 / GIF / closed-socket differences), `closed_json`
onto `tool_parameters.py`, and upstream's `cuda/health.py` against our 0150 module is an add/add merge (keep both
sets of keys). What stays ours: `Batcher._collect`'s poll (upstream has no batcher) and the stop-count rule on our 0160
scope.

## 6. Validation done offline

- Applied: 0001-0590 then 0600 with `git apply` on `2f8e514` (the Dockerfile's order); also 0001-0580 + 0600
  (independent of 0590).
- Suites on that tree against the same tree without 0600 (torch 2.14 CPU, Python 3.14, `PYTHONPATH=<tree>/src`):
  - `tests/cuda/`: 660 passed / 26 failed with 0600, 652 / 26 without: the **same 26** fail on both
    (`test_gpu_round_resident.py`, `test_solo_piece_patches.py`: fakes that predate 0500's `vision` attribute);
  - `tests/` (host, emulators): 1,304 passed / 28 failed with 0600, 1,134 / 29 without: the same 28 fail on both
    (`test_fastk_interpreter.py`'s Triton-interpreter cases, `test_serve_ops.py::test_context_line_and_short_context_warning`),
    plus, without 0600 only, the updated `test_vision_prep.py::test_http_fetch` (it now expects 0600's defaults);
  - the new tests three times in a row: 177 passed each time.
  GPU-only tests skip.
