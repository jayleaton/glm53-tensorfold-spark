# AGENTS.md: setting this up on 2x DGX Spark

Instructions for AI coding agents (and people) bringing up GLM-5.3-Flash on TensorFold across two DGX Sparks. The goal
is to run **exactly the author's production config** (`config/prod.env.example`): 4 concurrent requests, a
1,048,576-token context, FP8 latent KV, the RoCE all-gather. Follow the steps in order; run each check and stop on
a failure. Ask the user before anything that changes system state outside this repo (installing packages, editing
network config, `sudo`, deleting files).

Terms: **head** = the Spark you run commands on (rank 0, serves the API). **worker** = the other Spark (rank 1),
reached from the head over ssh.

## 1. Check both nodes

Run on the head; run the `ssh <worker>` lines to check the worker too.

| Check | Command | Expect |
| --- | --- | --- |
| GPU, driver | `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` (both) | `NVIDIA GB10`, the **same** driver on both; 580.x is known good (590.x: a CUDA-graph deadlock is reported on GB10) |
| Nothing else on the GPUs | `nvidia-smi --query-compute-apps=pid,name --format=csv,noheader` (both) | empty. Stop vLLM or any other stack first: only one fits |
| Docker + NVIDIA runtime | `docker info --format '{{.Runtimes}}'` (both) | contains `nvidia`; runs without `sudo`. With `CONTAINER_RT=podman`: `podman info` plus a CDI spec (once: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, regenerate after a driver/toolkit upgrade) and `podman run --device nvidia.com/gpu=all`; must be rootful (`podman info --format '{{.Host.Security.Rootless}}'` is `false`) |
| Passwordless ssh | `ssh -o BatchMode=yes <worker> docker ps` (podman: `ssh -o BatchMode=yes <worker> podman ps`, as a user that can use rootful podman) | no password prompt; the worker user can run the container runtime |
| Passwordless sudo (memory gate) | `sudo -n true` (both) | exit 0. Without it `MEM_GATE_DROP_CACHES` cannot drop page caches and the 4th slot may not fit |
| Free memory | `grep -E 'MemTotal\|MemFree' /proc/meminfo` (both) | MemFree >= 108 GiB when idle (`MEM_GATE_GIB=108`). Close desktop sessions, browsers, other containers |
| Disk | `df -h ~/.cache` (both) | ~165 GB for the checkpoint + ~84 GB prepared weights + up to 64 GiB session tier |
| Submodule | `ls vendor/TensorFold` | not empty; else `git submodule update --init` |

## 2. Find the link settings

The two Sparks are cabled QSFP to QSFP on their ConnectX-7 ports. On each node:

```bash
ibdev2netdev          # e.g. "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)": RDMA device ==> netdev, for the cabled port
ip -br addr           # the IPv4 address on that netdev
cat /sys/class/infiniband/<RDMA device>/ports/1/state    # "4: ACTIVE"
```

- `NCCL_IB_HCA` = the RDMA device(s) of the cabled port (preset: `rocep1s0f1,roceP2p1s0f1`, the two functions of a
  Spark's CX7 port; `ibdev2netdev` lists both; if yours shows only one, set that one and drop `NCCL_PASSTHROUGH` /
  `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` from the config).
- `NCCL_SOCKET_IFNAME` = its netdev (preset: `enp1s0f1np1`). Must be the same name on both nodes.
- `HEAD_IP` = the head's IPv4 address on that netdev (not its LAN address).
- Check the link: `ping -c 3 -I <netdev> <worker link address>` from the head.
- If the port is `DOWN` or has no address, the link is not configured. Do not reconfigure networking without the
  user's approval; tell them what is missing.
- The RoCE all-gather uses both CX7 functions of the port (`rocep1s0f1`, `roceP2p1s0f1` on a Spark). If its setup
  fails, the server falls back to NCCL by itself (slower, still correct): see section 6.

## 3. Weights and config

Weights, on **both** nodes, same revision (the model repo is gated: the user must accept its terms on Hugging Face
and be logged in with `hf auth login`):

```bash
hf download neko-legends/GLM-5.3-Flash-Uncensored-EXL3 --revision 07135ec082f8f11f7a71e4244a4e5167a0f96277
# optional drafter (CC BY-NC-ND 4.0, non-commercial only; ask the user):
hf download incoai/GLM-5.3-Flash-DFlash2 --revision 7d74cdd881ed7e32c31175984a67823127b66cfe
```

Config, on the head:

```bash
cp config/prod.env.example config/prod.env
```

Edit **only** these fields in `config/prod.env`:

| Field | Value | How to find it |
| --- | --- | --- |
| `WORKER_SSH` | `user@<worker link address>` (or an ssh alias) | the target that worked in step 1 |
| `HEAD_IP` | the head's IPv4 on the CX7 netdev | step 2 |
| `HEAD_HF` | absolute path of the head's HF cache | `echo ~/.cache/huggingface` on the head (or `$HF_HOME`) |
| `WORKER_HF` | absolute path of the worker's HF cache | `ssh <worker> 'echo ~/.cache/huggingface'` |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | only if step 2 found other names | step 2 |
| `DRAFTER` | set it empty (`DRAFTER=`) only if the drafter was not downloaded | |

`serve.sh` refuses a config that still has `<placeholders>`. It reads `config/prod.env` by default: do not pass
`CONFIG=` for production.

## 4. Build, preflight, start

```bash
scripts/serve.sh build       # the first build pulls nvcr.io/nvidia/pytorch:26.07-py3 (large) and applies the patches,
                             # then ships the image to the worker (docker save | ssh docker load): allow a while
scripts/serve.sh preflight   # must end with "[glm53-tf] preflight ok"; fix every line it reports first
scripts/serve.sh start
```

`preflight` checks, on both nodes: the config keys and `CONTEXT`; ssh; docker; the image; the CX7 netdev has an
address and `HEAD_IP` is the head's; the RDMA port is ACTIVE; the weights (and drafter) snapshots exist; MemFree
against `MEM_GATE_GIB`; `sudo -n` when the memory gate drops caches; no CUDA process running; the RoCE failure marker;
driver and `vm.min_free_kbytes` parity; GB10 clock / power state. Lines with `warning:` do not fail it; read them.

What `start` prints, in order (head's terminal):

```
[glm53-tf] config/prod.env: CONTEXT=1048576 tokens a request (prompt + max_tokens), KV cache latent fp8, GLM53_TF_BATCH=4, GLM53_TF_KV_POOL_TOKENS=1048576
[glm53-tf] MemFree 113 / 112 GiB (head / worker) >= 108
[glm53-tf] waiting for rank 0 on :8000 (first start compiles the kernels)
[glm53-tf] ready after 34 s
{"object": "list", "data": [{"id": "GLM-5.3-Flash-EXL3", ...}]}
[glm53-tf] serving up to 1048576 tokens a request (...): set your client's context window to it
[canary] ok: 3 probes, tokens/round 5.71, decode 76.7 tok/s
```

Timings: the **first** start loads from the checkpoint, compiles kernels into the `glm53-tf-cache` volume, measures
the drafter costs and writes the prepared weight folders (~83 GB a node): expect 8 minutes or more with little
output. Do not interrupt it; follow it with `scripts/serve.sh logs 0` (and `logs 1`). Later starts: 20-40 s
(~100 s when the calibration is re-measured after an image change).

Rank 0's log (`scripts/serve.sh logs 0`, or `docker logs glm53-tf-r0 2>&1 | grep -E '^\[(boot|tensorfold)\]'`) on a
healthy production start includes:

```
[boot] r0 +17.6s weights: 82.2 GB on the GPU (11.2s) | MemFree ...
[tensorfold] roce: all-gathers of up to 256 KiB a rank over RoCE on roceP2p1s0f1 (...), rocep1s0f1 (...); ...
[tensorfold] latent KV cache (fp8 rows): 7.4 KB a token a rank, in the KV pool: 1048576 tokens in pages of 256 ...
[tensorfold] batching 4 requests: 3 extra sequence(s), ...
[tensorfold] KV pool (patches/0290): 1048576 tokens in 4096 pages of 256, 7.44 GiB on this rank, shared by the 4 slots ...
[tensorfold] session store: 2.0 GiB, pages of 256 tokens ...
[tensorfold] serving GLM-5.3-Flash-EXL3 at http://127.0.0.1:8000/v1 on CUDA, rank 0 of 2 (...; loaded in ...s)
```

With `GLM53_TF_VISION=1` (production) rank 0 also loads the vision tower (`vision tower: 1.13 GB on the GPU`,
`image_url parts on`). The first start of a new image can take ~200 s more while kernels compile.

## 5. Verify

```bash
scripts/serve.sh status      # glm53-tf-r0 and glm53-tf-r1 "Up", then /v1/models and /health JSON
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3", "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}]}'
```

The reply has `choices[0].message.content` (and `reasoning`: thinking is on by default, effort high; production sends
it in `reasoning` only, `GLM53_TF_REASONING_FIELDS=both` adds `reasoning_content`) and a `tensorfold` object with
decode tok/s (~40-100 depending on the text).

Image input (production has it on): make a small PNG (any screenshot; or `python3 -c "from PIL import Image,
ImageDraw; im=Image.new('RGB',(400,120),'white'); ImageDraw.Draw(im).text((20,40),'17 x 23 = 391',fill='black');
im.save('/tmp/t.png')"`) and ask about it:

```bash
IMG=$(base64 -w0 /tmp/t.png)
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3", "max_tokens": 512,
  "messages": [{"role": "user", "content": [{"type": "text", "text": "What text is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}}]}]}'
```

The reply quotes the text; `usage.prompt_tokens` includes the image's rows. A malformed image is an HTTP 400 with
`param: "messages"`. `docs/VISION.md` has every limit and knob. Optional: `python3 bench/glmbench.py --base
http://127.0.0.1:8000 --model GLM-5.3-Flash-EXL3 --suites exact` must report 10/10 identical.

Client settings: base URL `http://127.0.0.1:8000/v1`, model `GLM-5.3-Flash-EXL3`, **context window 1,048,576**,
**max output 32,768**. opencode: the README Quickstart has the provider entry (`"limit": {"context": 1048576,
"output": 32768}`); Continue, Cline, Open WebUI: [`docs/TRYING.md`, client settings](docs/TRYING.md#client-settings).
The API has no auth and listens on 127.0.0.1 only; do not change `HOST` to `0.0.0.0` without a reverse proxy with
auth in front, and ask the user first.

## 6. Common failures and fixes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `no config/prod.env` | the config was not created | step 3 |
| `config/prod.env: set WORKER_SSH (still '<worker-ssh>')` | placeholder left | step 3 |
| Long prompts refused (HTTP 400 "this request needs a N-token context ... started for 32768") or a client cuts context short | wrong config (the 32k `config/minimal.env` debugging baseline, or an old `CONFIG=config/tensorfold.env`), or the client's context window setting | `start`'s first line must say `CONTEXT=1048576`; unset `CONFIG`; set the client's window (section 5). [docs/TRYING.md section 10](docs/TRYING.md#10-context-smaller-than-expected) |
| `a CUDA process is running on a node; stop the other stack first` | vLLM or another server holds a GPU | stop it (ask the user), then start again |
| `waiting for memory: MemFree ... want 108` for minutes, or `GLM53_TF_BATCH=4: only 3 sequence(s) fit` in `logs 0` | other workloads or page cache use memory | stop them; make `sudo -n` work for page-cache drops; restart. Fewer slots = fewer concurrent requests, not a shorter context |
| `rank 0 exited` / `rank 1 exited` with `CUDA out of memory` or a killed process | not enough free memory at load | as above; do not raise `CONTEXT`, `GLM53_TF_SESSION_GIB` or the prefill rows to "fix" it |
| NCCL timeout minutes into the load, `rank 1 exited` | link down, wrong `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` / `HEAD_IP`, or a firewall on `MASTER_PORT` (29551) | `scripts/serve.sh preflight`; step 2 |
| `[tensorfold] roce: setup failed, serving on NCCL: ...` or `roce: a RoCE failure marker is present ...: serving on NCCL` | RoCE could not start, or failed at run time earlier (a marker in the cache volume) | it keeps serving on NCCL (decode ~4-11% slower). Fix the link, then delete the marker on both nodes: `docker run --rm -v glm53-tf-cache:/cache --entrypoint rm glm53-tensorfold:dev -f /cache/roce-failed`, and restart. Validation stages: `docs/ROCE-FIX.md`. If it keeps failing, `GLM53_TF_COMM_BACKEND=nccl` (ask the user) |
| First start seems stuck at `waiting for rank 0` | kernel compile + checkpoint load + calibration | wait (8+ min); watch `scripts/serve.sh logs 0`. `READY_TIMEOUT` is 0 (no limit) on purpose |
| `no image glm53-tensorfold:dev on the worker` | the image was built but not shipped | `scripts/serve.sh build` again (it ships it) |
| `the weights are not in the worker's HF cache (...)` | download on one node only, or another revision | the `hf download` line it prints, on that node |
| `a node's GPU is degraded` (preflight) | a GB10 clock / power clamp; survives warm reboots | a full power drain of that node (docs/OPS-GPUWATCH.md); ask the user |
| Image request: `400 ... image ...` | the URL cannot be fetched from the head node, the image is over 20 MiB / 64 M pixels, more than 8 images, or `GLM53_TF_VISION` is off | send it as a `data:` URL; resize it; check the knob in `config/prod.env` (restart after a change) |
| Thinking missing in a client | production sends thinking in `reasoning` only | `GLM53_TF_REASONING_FIELDS=both` in `config/prod.env`, restart |
| Concurrent requests run one at a time while a large file is copied on the head node | page cache counted as used memory at admission (docs/RESULTS.md W15) | let the copy finish or drop caches. `scripts/serve.sh start` drops the page cache before a start and checks rank 0's slot count (a start with fewer than `GLM53_TF_BATCH` slots restarts, then exits 3). 0550's admission fix is in the image but off (docs/RESULTS.md W17) |
| Canary `FAIL` / tokens a round ~1.0 | the drafter is missing or mismatched | check `DRAFTER` exists on both nodes, or set `DRAFTER=` empty (MTP drafts only) |

## 7. Do not change these knobs

Every value in `config/prod.env.example` other than the fields in step 3 is the tested production setting (docs/
RESULTS.md W1-W17: exactness, memory stress at 4 x 250k tokens, quality, images). Changing one means an untested config.
In particular, leave these as they are unless the user asks for an experiment:

- `CONTEXT=1048576`, `MAX_TOKENS=32768`, `GLM53_TF_KV_POOL_TOKENS=1048576`, `GLM53_TF_BATCH=4`
- `GLM53_TF_LATENT_KV=1`, `GLM53_TF_KV_DTYPE=fp8` (the per-head KV cache cannot hold long contexts)
- `GLM53_TF_NONEXPERT=q4mse` and the prefill sizes (`GLM53_TF_PREFILL_ROWS_MAX=4096`, `GLM53_TF_SOLO_PIECE=4096`,
  `GLM53_TF_LEAN_BLOCK=512`): 8,192 was measured and rejected for memory
- `GLM53_TF_SESSION_GIB=2`, `GLM53_TF_BATCH_RESERVE_GB=11`, `GLM53_TF_BATCH_ADMIT_GB=2`,
  `GLM53_TF_SESSION_RESERVE_GIB=6`, `MEM_GATE_GIB=108`, `MEM_GATE_DROP_CACHES=1`: the memory margins
- `GLM53_TF_COMM_BACKEND=roce` + `GLM53_TF_ROCE_MARK` (falls back to NCCL by itself)
- `GLM53_TF_L2PF=1`, `GLM53_TF_L2PF_MB=8`, `GLM53_TF_BATCH_CAPTURE_AFTER=8` (decode, W12)
- `GLM53_TF_VISION=1` with `GLM53_TF_VISION_CACHE_MB=64` / `GLM53_TF_VISION_PREP_MB=64` (sized for the memory stress;
  the default 256 MB caches were not stress-tested). Turning vision off is safe (text is bit-identical) and gives
  the head ~1.1-1.9 GiB back; `GLM53_TF_VISION_FETCH=0` refuses `http(s)` image URLs (only `data:`), which you
  should set if anything but the user's own clients can reach the API
- 0540's `GLM53_TF_SNAPSHOT_BEFORE_END` / `GLM53_TF_EMIT_FIRST`: on by default, leave them unset
- `GLM53_TF_MULTI_PREFILL=1` (0560, W17) and 0550's three knobs set off (`GLM53_TF_ADMIT_MEM=free`,
  `GLM53_TF_SELECT_SCRATCH=off`, `GLM53_TF_ALLOC_TRIM_GB=0`): 0550 on slowed long prefills in W17
- knobs documented as measured and not adopted (`docs/PATCHES.md`): do not turn them on
- `HOST=127.0.0.1` (no auth on the API)

Other configs exist for specific cases (`config/prod-single.env.example`, `config/prod-batch.env.example`), and
`config/minimal.env.example` is a **32k debugging baseline**: use it only to check whether a problem also happens
without the production knobs (`CONFIG=config/minimal.env scripts/serve.sh ...` on every command while it runs).

## 8. Operate

```bash
scripts/serve.sh status | logs 0 | logs 1 | stop | restart
scripts/serve.sh canary      # post-load probes on demand
scripts/serve.sh xid 2h      # NVIDIA Xid events on both nodes
```

Watchdog (optional, restarts a dead pair): `scripts/systemd/glm53-tf-watchdog.{service,timer}`, see its header;
docs/TRYING.md section 8. Before leaving the server unattended, watch both nodes' MemAvailable during the first
long sessions: with the vision tower on the head node both nodes bind (W17, after a heavy warm-up: 7.50 / 6.92 GiB
at the 4 x 250k stress and 6.04 / 5.46 GiB during a 314k prompt after it, under the 8 GiB target, no OOM;
docs/MEMORY-SAFETY.md). A prompt far beyond 314k beside 3 busy slots has not been memory-tested (0550 bounds it; it is off).

To reproduce the RigMark numbers: `scripts/rigmark/install.sh`, then `scripts/rigmark/run.sh tensorfold` on the head
node while nothing else uses the server (docs/RIGMARK.md).
