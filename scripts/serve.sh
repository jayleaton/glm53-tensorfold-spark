#!/usr/bin/env bash
# Start / stop / inspect the two TensorFold ranks. Run on the head Spark (rank 0).
#
#   scripts/serve.sh build      build the image here and ship it to the worker
#   scripts/serve.sh start      preflight, memory gate, rank 1 on the worker, then rank 0 here; waits for /v1/models,
#                               then runs the canary (scripts/canary.py)
#   scripts/serve.sh restart    stop, then start
#   scripts/serve.sh preflight  read-only checks of both nodes before a start (AGENTS.md): config, ssh, docker, image,
#                               CX7 netdev / address / RDMA port, weights, memory, CUDA processes, RoCE marker
#   scripts/serve.sh stop | status | logs [0|1] | canary | xid [SINCE] | watch [--once]
#   scripts/serve.sh gpucheck   GB10 clock / power / slow-state check of both nodes (scripts/gpuwatch.py check);
#                               non-zero when a node is degraded or was recently slow: gate benchmark runs on it
#
# Configuration: config/prod.env, the production config (cp config/prod.env.example config/prod.env and fill it in).
# CONFIG=path reads another file, e.g. CONFIG=config/minimal.env for the 32k debugging baseline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG="${CONFIG:-config/prod.env}"
if [[ ! -f "$CONFIG" ]]; then
    if [[ "$CONFIG" == config/prod.env ]]; then
        echo "[glm53-tf] no config/prod.env. Create it from the production example: cp config/prod.env.example" \
             "config/prod.env, then set WORKER_SSH, HEAD_IP, HEAD_HF and WORKER_HF (README Quickstart, AGENTS.md)" >&2
        [[ ! -f config/tensorfold.env ]] || echo "[glm53-tf] config/tensorfold.env (the old default, usually the 32k" \
             "baseline) is no longer read unless CONFIG=config/tensorfold.env is set" >&2
    else
        echo "[glm53-tf] no config file $CONFIG. The production config: cp config/prod.env.example config/prod.env" \
             "and leave CONFIG unset (README Quickstart, AGENTS.md)" >&2
    fi
    exit 2
fi
export CONFIG
# A non-empty caller export wins over the same key in the config file:
#   CONTEXT=32768 GLM53_TF_NONEXPERT=q4mse scripts/serve.sh start
caller_env=$(env | grep -E '^(CONTAINER_RT|HEAD_PREPARED|WORKER_PREPARED|HEAD_SESSIONS|WORKER_SESSIONS|CONTEXT|MTP_DRAFTS|NO_DRAFTS|IMAGE|PORT|HOST|DRAFTER|MODEL_PATH|EXTRA_ARGS|SERVED_NAME|MAX_TOKENS|CPUSET|HEAD_CPUSET|WORKER_CPUSET|CANARY[A-Z_]*|MEM_GATE_[A-Z_]+|START_ATTEMPTS|READY_TIMEOUT|LOG_MAX_[A-Z]+|WATCH_[A-Z_]+|GPUWATCH_[A-Z_]+|WARMUP_LENGTHS|PREFLIGHT|NCCL_PASSTHROUGH|PATCHES|GLM53_TF_[A-Z0-9_]+)=.' || true)
# shellcheck disable=SC1090
set -a; source "$CONFIG"; set +a
while IFS= read -r kv; do [[ -n "$kv" ]] && export "${kv?}"; done <<< "$caller_env"
# refuse a config that still holds the example's <placeholders>
for _k in WORKER_SSH HEAD_IP HEAD_HF WORKER_HF MODEL_PATH; do
    if [[ "${!_k:-}" == *"<"*">"* ]]; then echo "[glm53-tf] $CONFIG: set $_k (still '${!_k}'; AGENTS.md step 3)" >&2; exit 2; fi
done

NAME="${NAME:-glm53-tf}"
IMAGE="${IMAGE:-glm53-tensorfold:dev}"
PORT="${PORT:-8080}"
BASE="http://127.0.0.1:$PORT"
# Container runtime: docker (default, stock DGX OS) or podman (rootful). The serving pod
# needs --network host so NCCL / the RoCE proxy own the CX7 NIC, so this is never pasta/rootless.
# podman's GPU and device flags differ slightly; see run_args below.
CONTAINER_RT="${CONTAINER_RT:-docker}"
case "$CONTAINER_RT" in
    docker|podman) ;;
    *) echo "[glm53-tf] CONTAINER_RT=$CONTAINER_RT: expected docker or podman" >&2; exit 2 ;;
esac
rt() { printf '%s' "$CONTAINER_RT"; }
# Runtime answers on the current node, and for podman that it is rootful (rootless has a separate image store, and
# GPU / RDMA devices, --ulimit memlock=-1 and IPC_LOCK can fail). 0: ok; 1: not answering or (podman) rootless.
rt_ok() {
    $(rt) info >/dev/null 2>&1 || return 1
    [[ "$CONTAINER_RT" == podman ]] || return 0
    [[ "$($(rt) info --format '{{.Host.Security.Rootless}}' 2>/dev/null)" == false ]]
}
# Every ops feature below is off unless set (config or environment); `start` / `stop` / `status` behave as before.
# post-load canary (scripts/canary.py): off | warn (log a failure, keep serving) | strict (stop both ranks, exit 1)
CANARY="${CANARY:-off}"
CANARY_MIN_TPR="${CANARY_MIN_TPR:-1.3}"       # least tokens a decode round over the probes (drafter alive)
CANARY_MIN_TPS="${CANARY_MIN_TPS:-0}"         # least decode tok/s over the probes (0: no floor)
# wait before launch until MemFree >= MEM_GATE_GIB on both nodes (0: off), at most MEM_GATE_TIMEOUT s, then go on.
# MemFree, not MemAvailable: on the Spark's unified memory CUDA cannot use reclaimable page cache right away, and
# MemAvailable counts it (the vLLM kit measured MemAvailable 111.9 GiB green-lighting starts the engine then refused)
MEM_GATE_GIB="${MEM_GATE_GIB:-0}"
MEM_GATE_TIMEOUT="${MEM_GATE_TIMEOUT:-600}"
MEM_GATE_DROP_CACHES="${MEM_GATE_DROP_CACHES:-0}"   # 1: while waiting, `sync; echo 1 > drop_caches` (sudo -n) on both nodes
WARMUP_LENGTHS="${WARMUP_LENGTHS:-}"          # e.g. "4096 16384": after the canary, prefill prompts of about these lengths
START_ATTEMPTS="${START_ATTEMPTS:-1}"         # a failed start (a rank exited, not ready, strict canary) tears down and retries
# W16/W17: page cache left by a big file operation (docker load / save of an image) can make the engine's fit check
# find less free device memory at load and silently start with fewer batch slots than GLM53_TF_BATCH. Before a
# launch, drop the page cache (`sync; echo 3 > drop_caches`, sudo -n) on a node whose MemAvailable - MemFree is at
# least START_DROP_CACHES_GIB (0: never; default 4 when MEM_GATE_DROP_CACHES=1, else 0). After /v1/models answers,
# rank 0's `context: ... N request slot(s)` line must show N >= GLM53_TF_BATCH; if it is short, stop, drop the caches
# and start again, at most SLOT_RETRIES times (default 2), then fail (exit 3, the short server left serving).
START_DROP_CACHES_GIB="${START_DROP_CACHES_GIB:-$([[ "$MEM_GATE_DROP_CACHES" == 1 ]] && echo 4 || echo 0)}"
SLOT_RETRIES="${SLOT_RETRIES:-2}"
READY_TIMEOUT="${READY_TIMEOUT:-0}"           # seconds to wait for /v1/models (0: no limit; first start compiles kernels)
LOG_MAX_SIZE="${LOG_MAX_SIZE:-}"              # e.g. 200m: container log rotation per container (empty: the runtime's default)
LOG_MAX_FILE="${LOG_MAX_FILE:-3}"
PREFLIGHT="${PREFLIGHT:-off}"                 # before start: off | warn (log problems) | strict (refuse to start on one)
NCCL_PASSTHROUGH="${NCCL_PASSTHROUGH:-0}"     # 1: every other NCCL_* variable set here reaches both ranks
# patches/0370: docker --cpuset-cpus for the containers (empty: every cpu, as before). GB10's Cortex-X925 cores are
# 5-9,15-19 on both Sparks; CPUSET="5-9,15-19" is the container-level form of GLM53_TF_CPU_PIN=fast. HEAD_ / WORKER_
# override it per node
CPUSET="${CPUSET:-}"
HEAD_CPUSET="${HEAD_CPUSET:-$CPUSET}"
WORKER_CPUSET="${WORKER_CPUSET:-$CPUSET}"
# preflight's GPU state check (scripts/gpuwatch.py check, docs/OPS-GPUWATCH.md): off | on (a degraded node -- clock or
# power clamp, a step-time regression past the transient window -- is a preflight problem; warnings are logged) |
# strict (a warning is a problem too: a recent slow state, an idle clock asymmetry; use for benchmark windows)
GPUWATCH_PREFLIGHT="${GPUWATCH_PREFLIGHT:-on}"
STATE_DIR="${STATE_DIR:-${XDG_STATE_HOME:-${HOME:-/tmp}/.local/state}/glm53-tf}"
# patches/0140: prepared rank folders (scripts/prepare.sh), next to each node's HF cache unless set
HEAD_PREPARED="${HEAD_PREPARED:-${HEAD_HF%/*}/glm53-tf/prepared}"
WORKER_PREPARED="${WORKER_PREPARED:-${WORKER_HF%/*}/glm53-tf/prepared}"
# patches/0250: the session store's NVMe tier, mounted at /sessions in both ranks (used when GLM53_TF_SESSION_DISK=/sessions;
# up to GLM53_TF_SESSION_DISK_GIB a node, default 64), next to each node's HF cache unless set
HEAD_SESSIONS="${HEAD_SESSIONS:-${HEAD_HF%/*}/glm53-tf/sessions}"
WORKER_SESSIONS="${WORKER_SESSIONS:-${WORKER_HF%/*}/glm53-tf/sessions}"
IMAGE_ID="${IMAGE_ID:-}"
log() { echo "[glm53-tf] $*"; }
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }

# The context a request can use comes from the config (docs/TRYING.md, "Context smaller than expected"). One line
# at start says what this config serves; a short CONTEXT warns; a long CONTEXT on the per-head KV cache
# (GLM53_TF_LATENT_KV unset or 0: ~390 KB a token a rank) cannot load next to the weights and is refused unless
# FORCE_CONTEXT=1. Returns 1 to refuse.
context_check() {
    local ctx="${CONTEXT:-}" latent="${GLM53_TF_LATENT_KV:-0}"
    if [[ -n "$ctx" && ! "$ctx" =~ ^[0-9]+$ ]]; then log "CONTEXT=$ctx: expected a token count"; return 1; fi
    log "$CONFIG: CONTEXT=${ctx:-unset} tokens a request (prompt + max_tokens), KV cache $(
        [[ "$latent" == 1 ]] && echo "latent ${GLM53_TF_KV_DTYPE:-bf16}" || echo "per-head (GLM53_TF_LATENT_KV=0)"),"\
        "GLM53_TF_BATCH=${GLM53_TF_BATCH:-1}, GLM53_TF_KV_POOL_TOKENS=${GLM53_TF_KV_POOL_TOKENS:-off}"
    local long="for long context use the production config: cp config/prod.env.example config/prod.env and leave CONFIG unset (CONTEXT=1048576)"
    if [[ -z "$ctx" ]]; then
        log "warning: CONTEXT is not set: the engine serves 2,051 tokens a request; $long"
    elif (( ctx <= 65536 )); then
        log "warning: CONTEXT=$ctx: a request past $ctx tokens (prompt + max_tokens) gets HTTP 400; $long"
    fi
    if [[ "$latent" != 1 && -n "$ctx" ]] && (( ctx > 65536 )); then
        log "CONTEXT=$ctx with the per-head KV cache (GLM53_TF_LATENT_KV=0): ~390 KB a token a rank, about" \
            "$(( ctx * 390 / 1048576 )) GiB a rank of KV cache; past ~100k tokens it does not fit next to the weights." \
            "Set GLM53_TF_LATENT_KV=1 and GLM53_TF_KV_DTYPE=fp8 (7.4 KB a token, as config/prod.env.example does)"
        if (( ctx > 131072 )) && [[ "${FORCE_CONTEXT:-0}" != 1 ]]; then
            log "not starting: CONTEXT=$ctx cannot load with the per-head KV cache (FORCE_CONTEXT=1 starts anyway)"
            return 1
        fi
    fi
    return 0
}

run_args() { # $1 = rank, $2 = host HF cache dir
    local rank=$1 hf=$2 prep sess
    # patches/0140: this node's prepared rank folders (scripts/prepare.sh) at /prepared; the image id and the launch
    # time key the calibration cache and start the [boot] timeline
    if [[ "$rank" == 0 ]]; then prep="$HEAD_PREPARED"; else prep="$WORKER_PREPARED"; fi
    if [[ "$rank" == 0 ]]; then sess="$HEAD_SESSIONS"; else sess="$WORKER_SESSIONS"; fi   # patches/0250
    local cpuset="$HEAD_CPUSET"; [[ "$rank" == 0 ]] || cpuset="$WORKER_CPUSET"             # patches/0370
    # GPU access: docker --gpus all; podman --device nvidia.com/gpu=all (nvidia-container-toolkit wrapper).
    local gpu=(--gpus all); [[ "$CONTAINER_RT" == podman ]] && gpu=(--device nvidia.com/gpu=all)
    echo --name "$NAME-r$rank" -d "${gpu[@]}" --ipc=host --network host \
        ${cpuset:+--cpuset-cpus "$cpuset"} \
        -v "$prep:/prepared" -e GLM53_TF_PREPARED=/prepared -e GLM53_TF_PREPARED_WRITE="${GLM53_TF_PREPARED_WRITE:-1}" \
        -v "$sess:/sessions" \
        -e GLM53_TF_CALIB="${GLM53_TF_CALIB:-cached}" -e GLM53_TF_IMAGE_ID="$IMAGE_ID" \
        -e GLM53_TF_LAUNCH_T0="$(date +%s.%N)" \
        --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
        ${LOG_MAX_SIZE:+--log-opt max-size="$LOG_MAX_SIZE" --log-opt max-file="$LOG_MAX_FILE"} \
        -v "$hf:/root/.cache/huggingface" -v "$NAME-cache:/cache" \
        -e RANK="$rank" -e MASTER="$HEAD_IP" -e MASTER_PORT="${MASTER_PORT:-29551}" \
        -e MODEL_PATH="$MODEL_PATH" -e DRAFTER="${DRAFTER:-}" -e CONTEXT="${CONTEXT:-}" \
        -e SERVED_NAME="${SERVED_NAME:-}" -e MAX_TOKENS="${MAX_TOKENS:-}" -e MTP_DRAFTS="${MTP_DRAFTS:-}" \
        -e NO_DRAFTS="${NO_DRAFTS:-0}" -e HOST="${HOST:-127.0.0.1}" -e PORT="$PORT" \
        -e EXTRA_ARGS="${EXTRA_ARGS:-}" -e GLM53_TF_NONEXPERT="${GLM53_TF_NONEXPERT:-bf16}" \
        -e GLM53_TF_PREFILL_ROWS="${GLM53_TF_PREFILL_ROWS:-auto}" -e GLM53_TF_AUTO_FDRAFTS="${GLM53_TF_AUTO_FDRAFTS:-7}" \
        -e GLM53_TF_PROFILE="${GLM53_TF_PROFILE:-0}" -e GLM53_TF_EXPERT_LOOP="${GLM53_TF_EXPERT_LOOP:-1}" \
        -e GLM53_TF_EXPERT_LOOP_CFG="${GLM53_TF_EXPERT_LOOP_CFG:-4,2}" \
        $(env | grep -E '^GLM53_TF_[A-Z0-9_]+=' | sed 's/^/-e /' | tr '\n' ' ') \
        $([[ "$NCCL_PASSTHROUGH" == 1 ]] && env | grep -E '^NCCL_[A-Z0-9_]+=' | grep -vE '^NCCL_(SOCKET_IFNAME|IB_HCA|PASSTHROUGH)=' | sed 's/^/-e /' | tr '\n' ' ') \
        -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" -e NCCL_IB_HCA="$NCCL_IB_HCA" \
        -e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" "$IMAGE"
}

gpu_busy() { # a foreign CUDA process on either node means the other stack is still up
    # grep reads everything (not -q): with pipefail, -q's early exit SIGPIPEs the writer and a busy node read as free
    { nvidia-smi --query-compute-apps=pid --format=csv,noheader; wssh nvidia-smi --query-compute-apps=pid --format=csv,noheader; } \
        | grep '[0-9]' >/dev/null
}

running() { # $1 = rank: prints true | false | absent | unreachable (the worker's ssh failed)
    local out rc
    if [[ "$1" == 0 ]]; then
        out=$($(rt) inspect -f '{{.State.Running}}' "$NAME-r0" 2>/dev/null) || out=absent
    else
        out=$(wssh $(rt) inspect -f "'{{.State.Running}}'" "$NAME-r1" 2>/dev/null) && rc=0 || rc=$?
        if [[ $rc == 255 ]]; then out=unreachable; elif [[ $rc != 0 ]]; then out=absent; fi
    fi
    echo "${out:-absent}"
}

# MemFree in GiB: head, worker (the worker's is empty if ssh fails)
memfree() { awk '/^MemFree:/ {printf "%d", $2/1048576}' /proc/meminfo; }
memfree_worker() { wssh "awk '/^MemFree:/ {printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null || true; }

mem_gate() {
    [[ "$MEM_GATE_GIB" -gt 0 ]] || return 0
    local t0 h w
    t0=$(date +%s)
    while :; do
        h=$(memfree); w=$(memfree_worker)
        if [[ "${h:-0}" -ge "$MEM_GATE_GIB" && "${w:-0}" -ge "$MEM_GATE_GIB" ]]; then
            log "MemFree ${h} / ${w} GiB (head / worker) >= $MEM_GATE_GIB"; return 0
        fi
        if (( $(date +%s) - t0 >= MEM_GATE_TIMEOUT )); then
            log "MemFree ${h} / ${w:-?} GiB still under $MEM_GATE_GIB after ${MEM_GATE_TIMEOUT}s; starting anyway"
            return 0
        fi
        log "waiting for memory: MemFree ${h} / ${w:-?} GiB (head / worker), want $MEM_GATE_GIB"
        if [[ "$MEM_GATE_DROP_CACHES" == 1 ]]; then   # page cache only (1), never swap: other processes own it
            sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null || log "drop_caches: sudo -n refused on the head"
            wssh "sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches'" 2>/dev/null || log "drop_caches: refused on the worker"
        fi
        sleep 10
    done
}

# page cache (GiB, integer): MemAvailable - MemFree, on the head / the worker
cache_gib() { awk '/^MemFree:/ {f=$2} /^MemAvailable:/ {a=$2} END {printf "%d", (a-f)/1048576}' /proc/meminfo; }
cache_gib_worker() { wssh "awk '/^MemFree:/ {f=\$2} /^MemAvailable:/ {a=\$2} END {printf \"%d\", (a-f)/1048576}' /proc/meminfo" 2>/dev/null || true; }

drop_caches_if_needed() { # $1 = force: drop on both nodes regardless of the threshold
    local force=${1:-0} h w
    [[ "$force" == 1 || "$START_DROP_CACHES_GIB" -gt 0 ]] || return 0
    h=$(cache_gib); w=$(cache_gib_worker)
    if [[ "$force" == 1 || "${h:-0}" -ge "$START_DROP_CACHES_GIB" ]]; then
        sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null && log "dropped the head's page cache (${h} GiB)" \
            || log "drop_caches: sudo -n refused on the head"
    fi
    if [[ "$force" == 1 || "${w:-0}" -ge "$START_DROP_CACHES_GIB" ]]; then
        wssh "sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null && log "dropped the worker's page cache (${w:-?} GiB)" \
            || log "drop_caches: refused on the worker"
    fi
}

# rank 0's slot count from its boot line `context: ... N request slot(s)` (empty until it is printed)
slots_r0() { $(rt) logs "$NAME-r0" 2>&1 | grep -oE '[0-9]+ request slot\(s\)' | tail -1 | grep -oE '^[0-9]+' || true; }

check_slots() { # 0: rank 0 has >= GLM53_TF_BATCH slots (or no batching / no line within 60 s); 3: fewer
    local want=${GLM53_TF_BATCH:-1} n="" i
    [[ "$want" -gt 1 ]] || return 0
    for i in $(seq 1 60); do n=$(slots_r0); [[ -n "$n" ]] && break; sleep 1; done
    if [[ -z "$n" ]]; then log "no 'request slot(s)' line from rank 0 within 60 s; not checked"; return 0; fi
    if [[ "$n" -lt "$want" ]]; then log "rank 0 started with $n request slot(s), GLM53_TF_BATCH=$want"; return 3; fi
    log "request slots: $n (GLM53_TF_BATCH=$want)"
}

on_node() { # $1 = head | worker, then a command line (one string): run it on that node
    local n=$1; shift
    if [[ "$n" == head ]]; then bash -c "$*"; else wssh "$*"; fi
}

link_addrs() { # $1 = head | worker: the IPv4 addresses (a.b.c.d/nn) on NCCL_SOCKET_IFNAME, space-separated
    on_node "$1" "ip -o -4 addr show dev '$NCCL_SOCKET_IFNAME'" 2>/dev/null | awk '{printf "%s ", $4}' || true
}

weights_missing() { # $1 = head | worker, $2 = that node's HF cache, $3 = container path, $4 = file to look for:
    # prints the host path when the snapshot is not in that node's cache (paths outside the HF mount: not checked)
    local p="$3"
    [[ "$p" == /root/.cache/huggingface/* ]] || return 0
    local host="$2${p#/root/.cache/huggingface}"
    on_node "$1" "test -e '$host/$4'" 2>/dev/null || echo "$host"
}

hf_hint() { # $1 = container path of a snapshot: the matching `hf download` command
    local repo="${1#*/models--}" rev="${1##*/snapshots/}"
    repo="${repo%%/*}"; echo "hf download ${repo/--//} --revision ${rev%%/*}"
}

preflight() { # read-only checks (AGENTS.md lists them); non-zero on a problem a retry cannot fix
    local bad=0 st wst h w a node miss mark
    for v in WORKER_SSH HEAD_IP NCCL_SOCKET_IFNAME NCCL_IB_HCA MODEL_PATH HEAD_HF WORKER_HF; do
        [[ -n "${!v:-}" ]] || { log "preflight: $v is not set in $CONFIG"; bad=1; }
    done
    [[ $bad == 0 ]] || return 1
    wssh true || { log "preflight: cannot ssh to $WORKER_SSH: passwordless ssh from the head is needed (ssh-copy-id $WORKER_SSH)"; return 1; }
    local rt_hint  # accurate "how to run the runtime" hint per runtime (podman has no podman group)
    if [[ "$CONTAINER_RT" == podman ]]; then rt_hint="service running? run as root or via sudo on both nodes"; else rt_hint="service running? user in the docker group?"; fi
    if ! rt_ok; then log "preflight: the container runtime does not answer (or is rootless) on the head ($rt_hint)"; bad=1; fi
    # worker: same check, built from the concrete binary name so it runs on the worker (rt_ok is a local function)
    local rtb="$CONTAINER_RT" w_cmd="$CONTAINER_RT info >/dev/null 2>&1"
    [[ "$CONTAINER_RT" == podman ]] && w_cmd="$w_cmd && [[ \"\$($rtb info --format '{{.Host.Security.Rootless}}' 2>/dev/null)\" == false ]]"
    if ! wssh "$w_cmd"; then log "preflight: the container runtime does not answer (or is rootless) on the worker as $WORKER_SSH ($rt_hint)"; bad=1; fi
    $(rt) image inspect "$IMAGE" >/dev/null 2>&1 || { log "preflight: no image $IMAGE here (scripts/serve.sh build)"; bad=1; }
    wssh $(rt) image inspect "$IMAGE" >/dev/null 2>&1 || { log "preflight: no image $IMAGE on the worker (scripts/serve.sh build ships it)"; bad=1; }
    [[ -n "$(ls -A vendor/TensorFold 2>/dev/null)" ]] \
        || log "preflight: warning: vendor/TensorFold is empty: git submodule update --init (serve.sh build needs it)"
    # the CX7 link: NCCL_SOCKET_IFNAME carries an address on both nodes, HEAD_IP is the head's
    if command -v ip >/dev/null 2>&1; then
        for node in head worker; do
            a=$(link_addrs "$node")
            if [[ -z "$a" ]]; then
                log "preflight: no IPv4 address on NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME on the $node: set it to the CX7" \
                    "netdev that carries the link (ibdev2netdev: '$NCCL_IB_HCA port 1 ==> <netdev> (Up)'; ip -br addr)"
                bad=1
            elif [[ "$node" == head && "$HEAD_IP" =~ ^[0-9]+(\.[0-9]+){3}$ && " $a" != *" $HEAD_IP/"* ]]; then
                log "preflight: HEAD_IP=$HEAD_IP is not an address of $NCCL_SOCKET_IFNAME on the head (it has: ${a% })"
                bad=1
            fi
        done
    else
        log "preflight: warning: no 'ip' command here; cannot check NCCL_SOCKET_IFNAME / HEAD_IP"
    fi
    # the RDMA port of the link must be up on both nodes (a down port shows as an NCCL timeout minutes into the load)
    # (NCCL_IB_HCA may list several devices, comma-separated: each one is checked)
    IFS=',' read -ra hcas <<< "$NCCL_IB_HCA"
    for hca in "${hcas[@]}"; do
        hca="${hca%%:*}"
        st=$(cat "/sys/class/infiniband/$hca/ports/1/state" 2>/dev/null || echo "")
        wst=$(wssh cat "/sys/class/infiniband/$hca/ports/1/state" 2>/dev/null || echo "")
        for pair in "head:$st" "worker:$wst"; do
            case "${pair#*:}" in
                *ACTIVE*) ;;
                "") log "preflight: warning: cannot read $hca port state on the ${pair%%:*}" ;;
                *) log "preflight: $hca port 1 on the ${pair%%:*} is '${pair#*:}', not ACTIVE"; bad=1 ;;
            esac
        done
    done
    # vm.min_free_kbytes is taken from the GPU's share of unified memory; a mismatch gives the ranks different room
    h=$(cat /proc/sys/vm/min_free_kbytes 2>/dev/null || echo "?"); w=$(wssh cat /proc/sys/vm/min_free_kbytes 2>/dev/null || echo "?")
    [[ "$h" == "$w" ]] || log "preflight: warning: vm.min_free_kbytes differs (head $h, worker $w)"
    # the same driver on both nodes; the vLLM kit holds 580.x (590.x: a CUDA-graph deadlock reported on GB10)
    h=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    w=$(wssh nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    [[ "$h" == "$w" ]] || log "preflight: warning: NVIDIA driver differs (head ${h:-?}, worker ${w:-?})"
    [[ "$h $w" != *590.* ]] || log "preflight: warning: driver 590.x (a GB10 CUDA-graph deadlock is reported on 590; 580.x is the known-good line)"
    # the weights (and the drafter, when set) in both nodes' HF caches, same snapshot
    for node in head worker; do
        [[ "$node" == head ]] && h="$HEAD_HF" || h="$WORKER_HF"
        miss=$(weights_missing "$node" "$h" "$MODEL_PATH" config.json)
        [[ -z "$miss" ]] || { log "preflight: the weights are not in the $node's HF cache ($miss): $(hf_hint "$MODEL_PATH") on the $node"; bad=1; }
        if [[ -n "${DRAFTER:-}" ]]; then
            miss=$(weights_missing "$node" "$h" "$DRAFTER" .)
            [[ -z "$miss" ]] || { log "preflight: DRAFTER is not in the $node's HF cache ($miss): $(hf_hint "$DRAFTER")" \
                "on the $node (CC BY-NC-ND 4.0, non-commercial), or set DRAFTER= empty for MTP drafts only"; bad=1; }
        fi
    done
    # memory: the load's slot rule reads MemFree; other workloads or page cache cost the 4th batch slot
    h=$(memfree); w=$(memfree_worker)
    if [[ "$MEM_GATE_GIB" -gt 0 ]] && [[ "${h:-0}" -lt "$MEM_GATE_GIB" || "${w:-0}" -lt "$MEM_GATE_GIB" ]]; then
        log "preflight: warning: MemFree ${h} / ${w:-?} GiB (head / worker) under MEM_GATE_GIB=$MEM_GATE_GIB: stop other" \
            "workloads (vLLM, other containers, a desktop session); start waits up to ${MEM_GATE_TIMEOUT}s for it"
    fi
    if [[ "$MEM_GATE_DROP_CACHES" == 1 ]]; then
        sudo -n true 2>/dev/null || log "preflight: warning: no passwordless sudo -n on the head: the memory gate cannot drop page caches"
        wssh sudo -n true 2>/dev/null || log "preflight: warning: no passwordless sudo -n on the worker: the memory gate cannot drop page caches"
    fi
    if gpu_busy; then log "preflight: a CUDA process is running on a node (nvidia-smi): stop the other stack (vLLM, ...) first"; bad=1; fi
    # a RoCE failure at run time leaves a marker in the cache volume; the next start then serves on NCCL
    mark="${GLM53_TF_ROCE_MARK:-}"
    if [[ "${GLM53_TF_COMM_BACKEND:-}" == roce && "$mark" == /cache/* ]]; then
        local probe="$(rt) run --rm --network none -v $NAME-cache:/cache --entrypoint test $IMAGE -e $mark"
        for node in head worker; do
            if on_node "$node" "$probe" >/dev/null 2>&1; then
                log "preflight: warning: RoCE failure marker $mark on the $node: the next start serves on NCCL (decode" \
                    "~4-11% slower). After fixing the link, delete it on both nodes: $(rt) run --rm -v $NAME-cache:/cache" \
                    "--entrypoint rm $IMAGE -f $mark (docs/ROCE-FIX.md)"
            fi
        done
    fi
    gpu_state "$GPUWATCH_PREFLIGHT" || bad=1
    return $bad
}

gpu_state() { # $1 = off | on | strict: the GB10 clock / power / slow-state check of both nodes; non-zero on a problem
    [[ "$1" != off && -f scripts/gpuwatch.py ]] || return 0
    local out rc=0 args=(check --state-dir "$STATE_DIR" --worker "$WORKER_SSH" --port "$PORT")
    [[ "$1" == strict ]] && args+=(--strict)
    out=$(python3 scripts/gpuwatch.py "${args[@]}" 2>&1) || rc=$?
    [[ -z "$out" ]] || while IFS= read -r l; do log "preflight: $l"; done <<< "$out"
    case $rc in
        0) return 0 ;;
        1) if [[ "$1" == strict ]]; then log "preflight: GPU warning above (GPUWATCH_PREFLIGHT=strict)"; return 1; fi
           log "preflight: warning: GPU state (above)"; return 0 ;;
        2) log "preflight: a node's GPU is degraded (docs/OPS-GPUWATCH.md: power-drain a clamped node)"; return 1 ;;
        *) log "preflight: warning: cannot check the GPU state (rc $rc)"; return 0 ;;
    esac
}

stop_both() { # both ranks at once
    $(rt) rm -f "$NAME-r0" >/dev/null 2>&1 &
    local p0=$!
    wssh $(rt) rm -f "$NAME-r1" >/dev/null 2>&1 &
    local p1=$!
    wait "$p0" || true; wait "$p1" || true
}

run_canary() { # 0: pass or off; 1: failed
    [[ "$CANARY" == off ]] && return 0
    local args=(--base "$BASE" --min-tpr "$CANARY_MIN_TPR" --min-tps "$CANARY_MIN_TPS")
    [[ -n "${SERVED_NAME:-}" ]] && args+=(--model "$SERVED_NAME")
    [[ "${NO_DRAFTS:-0}" == 1 ]] && args+=(--no-draft)
    [[ -n "$WARMUP_LENGTHS" ]] && args+=(--warm-lengths "$WARMUP_LENGTHS")
    if python3 scripts/canary.py "${args[@]}"; then return 0; fi
    if [[ "$CANARY" == strict ]]; then return 1; fi
    log "canary failed (CANARY=warn: still serving; CANARY=strict fails the start)"
    return 0
}

start_once() { # 0: ready; 1: failed (containers left for the caller to inspect or remove); 2: the canary failed;
    # 3: fewer request slots than GLM53_TF_BATCH (serving)
    stop_both
    drop_caches_if_needed "${FORCE_DROP:-0}"
    mem_gate
    IMAGE_ID=$($(rt) image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo "$IMAGE")   # same id after save | load
    # both ranks at once (rank 1 waits for rank 0's TCP store either way); patches/0140
    # shellcheck disable=SC2046
    wssh "mkdir -p '$WORKER_PREPARED' '$WORKER_SESSIONS' && $(rt) run $(run_args 1 "$WORKER_HF")" >/dev/null &
    local wpid=$!
    mkdir -p "$HEAD_PREPARED" "$HEAD_SESSIONS"
    # shellcheck disable=SC2046
    $(rt) run $(run_args 0 "$HEAD_HF") >/dev/null
    wait "$wpid" || { log "rank 1 did not start"; return 1; }
    log "waiting for rank 0 on :$PORT (first start compiles the kernels)"
    local t0 w_fail=0 r1 tick=0
    t0=$(date +%s)
    until curl -sf -m 5 "$BASE/v1/models" >/dev/null; do
        [[ "$(running 0)" == true ]] || { log "rank 0 exited"; $(rt) logs --tail 40 "$NAME-r0" 2>&1; return 1; }
        tick=$((tick + 1))
        # 1 s polling (patches/0140); the worker over ssh every 10th tick
        if (( tick % 10 )); then sleep 1; continue; fi
        r1=$(running 1)
        case "$r1" in
            true) w_fail=0 ;;
            unreachable) w_fail=$((w_fail + 1)); log "worker unreachable over ssh ($w_fail)" ;;
            *) log "rank 1 exited"; wssh $(rt) logs --tail 40 "$NAME-r1" 2>&1; return 1 ;;
        esac
        [[ $w_fail -lt 3 ]] || { log "worker unreachable 3 times in a row"; return 1; }
        (( READY_TIMEOUT <= 0 || $(date +%s) - t0 < READY_TIMEOUT )) || { log "not ready after ${READY_TIMEOUT}s"; return 1; }
        sleep 1
    done
    log "ready after $(( $(date +%s) - t0 )) s"
    local models mml
    models=$(curl -s "$BASE/v1/models" || true); echo "$models"
    # an image that reports the context in /v1/models (max_model_len) is believed; otherwise the config's CONTEXT
    mml=$(grep -oE '"max_model_len": *[0-9]+' <<< "$models" | grep -oE '[0-9]+$' || true)
    if [[ -n "$mml" ]]; then
        log "serving up to $mml tokens a request (max_model_len): set your client's context window to it"
    else
        log "serving up to ${CONTEXT:-2051} tokens a request (CONTEXT in $CONFIG): set your client's context window to it"
    fi
    check_slots || return 3
    run_canary || { log "canary failed (CANARY=strict)"; return 2; }
}

run_preflight() { # PREFLIGHT=off: skip; warn: log and go on; strict: non-zero on a problem
    case "$PREFLIGHT" in
        off) return 0 ;;
        strict) preflight ;;
        *) preflight || log "preflight: problems above (PREFLIGHT=warn: starting anyway)"; return 0 ;;
    esac
}

take_lock() { # one start at a time; the watchdog stands down while it is held. Best effort: no lock if unavailable
    [[ "${GLM53_TF_LOCKED:-0}" == 1 ]] && return 0
    GLM53_TF_LOCKED=1
    command -v flock >/dev/null 2>&1 && mkdir -p "$STATE_DIR" 2>/dev/null && { exec 8>"$STATE_DIR/lock"; } 2>/dev/null \
        || { log "no start lock ($STATE_DIR not writable or no flock); going on"; return 0; }
    flock -n 8 || { log "another start (or a watchdog heal) is running"; exit 1; }
}

cmd_start() {
    take_lock
    if gpu_busy; then log "a CUDA process is running on a node; stop the other stack first"; exit 1; fi
    context_check || exit 1
    run_preflight || { log "preflight failed; not starting (PREFLIGHT=strict)"; exit 1; }
    local attempt=1 slot_try=0
    while :; do
        local rc=0
        start_once || rc=$?
        [[ $rc != 0 ]] || return 0
        if [[ $rc == 3 ]]; then
            if (( slot_try >= SLOT_RETRIES )); then
                log "still short of request slots after $slot_try restart(s); leaving it serving, start failed"; exit 3
            fi
            slot_try=$((slot_try + 1))
            log "too few request slots: dropping the page caches on both nodes and restarting ($slot_try of $SLOT_RETRIES)"
            FORCE_DROP=1; continue
        fi
        if (( attempt >= START_ATTEMPTS )); then
            if [[ $rc == 2 ]]; then     # a degenerate engine must not keep serving
                log "stopping both ranks: they loaded but failed the canary (last logs in $STATE_DIR/canary-fail-r{0,1}.log)"
                $(rt) logs --tail 200 "$NAME-r0" > "$STATE_DIR/canary-fail-r0.log" 2>&1 || true
                wssh $(rt) logs --tail 200 "$NAME-r1" > "$STATE_DIR/canary-fail-r1.log" 2>&1 || true
                stop_both
            else
                log "start failed (attempt $attempt of $START_ATTEMPTS); leaving the containers for 'logs', 'stop' removes them"
            fi
            exit 1
        fi
        log "start failed (attempt $attempt of $START_ATTEMPTS); tearing down and retrying"
        stop_both
        run_preflight || { log "preflight failed; not retrying (PREFLIGHT=strict)"; exit 1; }
        attempt=$((attempt + 1))
    done
}

kernel_log() { # $1 = since (journalctl syntax); the head's kernel log, else dmesg
    journalctl -k -q --no-pager --since "$1" 2>/dev/null || dmesg 2>/dev/null || true
}

cmd_xid() { # Xid events on both nodes since $1 (default: the last hour); non-zero if a fatal one was seen
    local since="${1:--1h}" rc=0
    kernel_log "$since" | python3 scripts/xid.py --label "head: " || rc=1
    wssh "journalctl -k -q --no-pager --since '$since' 2>/dev/null || dmesg 2>/dev/null || true" \
        | python3 scripts/xid.py --label "worker: " || rc=1
    return $rc
}

# -- watchdog --------------------------------------------------------------------------------------------------
# One tick: decide whether the pair is healthy. Stands down while a start runs (lock), when both containers are
# absent (stopped on purpose) and while rank 0 is younger than WATCH_GRACE. A tick is bad when a rank exited, or
# /health fails (connection refused, or 503 with GLM53_TF_HEALTH=strict: a fatal engine error or a stall).
# WATCH_FAILS bad ticks in a row heal (WATCH_HEAL=1: stop + start in the background) at most once every
# WATCH_MIN_HEAL s. A fatal Xid never heals (an off-the-bus GPU needs a power cycle): it is logged, and alerted.
WATCH_GRACE="${WATCH_GRACE:-1800}"
WATCH_FAILS="${WATCH_FAILS:-3}"
WATCH_HEAL="${WATCH_HEAL:-0}"
WATCH_MIN_HEAL="${WATCH_MIN_HEAL:-1800}"
WATCH_INTERVAL="${WATCH_INTERVAL:-60}"
WATCH_ALERT="${WATCH_ALERT:-}"                # a command run with the message as $1 (e.g. a notify script)
# drafter health from /metrics (patches/0150) between checks: alert (never heal) when the decode tokens a round over at
# least WATCH_TPR_ROUNDS rounds fall under WATCH_MIN_TPR (0: off). A dead/mismatched drafter decodes 1 token a round.
WATCH_MIN_TPR="${WATCH_MIN_TPR:-0}"
WATCH_TPR_ROUNDS="${WATCH_TPR_ROUNDS:-500}"

metric() { # $1 = metrics text, $2 = name: the value summed over label sets (0 when absent)
    awk -v n="$2" '$1 ~ "^"n"([{]|$)" {s += $NF} END {printf "%.0f", s + 0}' <<< "$1"
}

tpr_check() { # compare /metrics with the last stored sample; alert on a low tokens-a-round rate
    [[ "$WATCH_MIN_TPR" != 0 ]] || return 0
    local m tok rounds reqs f="$STATE_DIR/tpr" pt pr pq dt dr dq
    m=$(curl -s -m 10 "$BASE/metrics") || return 0
    [[ "$m" == *tensorfold_decode_rounds_total* ]] || return 0
    tok=$(metric "$m" tensorfold_completion_tokens_total); rounds=$(metric "$m" tensorfold_decode_rounds_total)
    reqs=$(metric "$m" tensorfold_requests_total)
    read -r pt pr pq < "$f" 2>/dev/null || { echo "$tok $rounds $reqs" > "$f"; return 0; }
    dt=$((tok - pt)); dr=$((rounds - pr)); dq=$((reqs - pq))
    if (( dt < 0 || dr < 0 || dq < 0 )); then echo "$tok $rounds $reqs" > "$f"; return 0; fi   # restarted: reseed
    (( dr >= WATCH_TPR_ROUNDS )) || return 0          # not enough rounds yet: keep the old sample
    echo "$tok $rounds $reqs" > "$f"
    # the first token of each request comes from the prefill, not a round
    if awk -v t="$dt" -v q="$dq" -v r="$dr" -v m="$WATCH_MIN_TPR" 'BEGIN {exit !((t - q) / r < m)}'; then
        alert "drafter: $(awk -v t="$dt" -v q="$dq" -v r="$dr" 'BEGIN {printf "%.2f", (t - q) / r}') tokens a round over $dr rounds < $WATCH_MIN_TPR"
    fi
}

alert() { log "ALERT: $*"; [[ -z "$WATCH_ALERT" ]] || "$WATCH_ALERT" "$*" || true; }

watch_tick() {
    mkdir -p "$STATE_DIR"
    local fails_f="$STATE_DIR/fails" heal_f="$STATE_DIR/last_heal" xid_f="$STATE_DIR/xid_since"
    local fails r0 r1 age code body now
    fails=$(cat "$fails_f" 2>/dev/null || echo 0)
    now=$(date +%s)
    # a start (by hand or a heal) holds the lock for its whole run
    if ! flock -n "$STATE_DIR/lock" true; then log "watch: a start is in progress; standing down"; return 0; fi
    # Xid events since the last tick (both nodes), alert only
    local since
    since=$(cat "$xid_f" 2>/dev/null || echo "-${WATCH_INTERVAL}s")
    date '+%F %T' > "$xid_f"
    local xids
    if ! xids=$(cmd_xid "$since" 2>&1); then alert "fatal Xid: $xids"; elif [[ -n "$xids" ]]; then log "watch: $xids"; fi
    r0=$(running 0); r1=$(running 1)
    if [[ "$r0" == absent && "$r1" == absent ]]; then
        echo 0 > "$fails_f"; log "watch: both ranks absent (stopped on purpose); standing down"; return 0
    fi
    local bad=""
    if [[ "$r0" != true ]]; then bad="rank 0 $r0"
    elif [[ "$r1" == unreachable ]]; then log "watch: worker unreachable over ssh (not counted)"
    elif [[ "$r1" != true ]]; then bad="rank 1 $r1"
    else
        # podman's .State.StartedAt ends with a zone word ('... +0000 UTC' or '... +0200 CEST') that `date -d`
        # rejects; strip it (docker returns a bare ISO UTC time, so the sed is a no-op there).
        age=$(( now - $(date -d "$($(rt) inspect -f '{{.State.StartedAt}}' "$NAME-r0" | sed -E 's/ [A-Za-z]+$//')" +%s) ))
        body=$(curl -s -m 10 -w '\n%{http_code}' "$BASE/health" || true)
        code=${body##*$'\n'}; body=${body%$'\n'*}
        if [[ "$code" == 200 ]]; then :
        elif [[ "$code" == 000 && $age -lt $WATCH_GRACE ]]; then log "watch: loading (${age}s)"; echo 0 > "$fails_f"; return 0
        else bad="/health $code ${body:0:300}"
        fi
    fi
    if [[ -z "$bad" ]]; then echo 0 > "$fails_f"; tpr_check; return 0; fi
    fails=$((fails + 1)); echo "$fails" > "$fails_f"
    log "watch: bad tick $fails of $WATCH_FAILS: $bad"
    (( fails >= WATCH_FAILS )) || return 0
    local last
    last=$(cat "$heal_f" 2>/dev/null || echo 0)
    if [[ "$WATCH_HEAL" != 1 ]]; then alert "unhealthy ($bad); WATCH_HEAL=0, not restarting"; return 1; fi
    if (( now - last < WATCH_MIN_HEAL )); then alert "unhealthy ($bad); healed $((now - last))s ago, waiting"; return 1; fi
    echo "$now" > "$heal_f"; echo 0 > "$fails_f"
    alert "unhealthy ($bad); restarting both ranks"
    # detached: a start takes minutes; the lock makes later ticks stand down until it is done
    setsid env GLM53_TF_LOCKED=1 flock "$STATE_DIR/lock" "$0" restart >>"$STATE_DIR/heal.log" 2>&1 < /dev/null &
    return 1
}

case "${1:-}" in
build)
    # PATCHES="0001 0002 ...": only these patch number prefixes (docker/Dockerfile); unset = every patch in patches/
    $(rt) build -f docker/Dockerfile ${PATCHES:+--build-arg PATCHES="$PATCHES"} -t "$IMAGE" .
    log "shipping $IMAGE to $WORKER_SSH"
    $(rt) save "$IMAGE" | wssh $(rt) load
    ;;
start)
    cmd_start ;;
restart)
    take_lock; stop_both; log "stopped"; cmd_start ;;
stop)
    stop_both
    log "stopped"
    ;;
status)
    $(rt) ps -a --filter "name=$NAME" --format '{{.Names}} {{.Status}}'
    wssh "$(rt) ps -a --filter name=$NAME --format '{{.Names}} {{.Status}}'"   # one string: the remote shell re-splits args
    curl -s "$BASE/v1/models" || true; echo
    curl -s "$BASE/health" || true; echo
    ;;
logs)
    if [[ "${2:-0}" == 1 ]]; then wssh $(rt) logs --tail "${TAIL:-80}" "$NAME-r1"; else $(rt) logs --tail "${TAIL:-80}" "$NAME-r0"; fi
    ;;
canary)
    CANARY=strict run_canary ;;
xid)
    cmd_xid "${2:--1h}" ;;
preflight)
    rc=0; context_check || rc=1; preflight || rc=1
    if [[ $rc == 0 ]]; then log "preflight ok"; else log "preflight: problems above (AGENTS.md, common failures)"; exit 1; fi ;;
gpucheck)   # for benchmark runs: strict unless GPUWATCH_PREFLIGHT says otherwise; exits 1 on any problem
    mode="${GPUWATCH_PREFLIGHT:-on}"; [[ "$mode" == on ]] && mode=strict
    gpu_state "$mode" && log "gpucheck ok" ;;
watch)
    if [[ "${2:-}" == --once ]]; then watch_tick; exit $?; fi
    while :; do watch_tick || true; sleep "$WATCH_INTERVAL"; done
    ;;
*)
    sed -n '2,12p' "$0"; exit 2 ;;
esac
