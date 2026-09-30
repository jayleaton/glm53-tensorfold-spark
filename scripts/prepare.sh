#!/usr/bin/env bash
# Write each node's prepared rank folder (patches/0140) once, so every later start reads its weights (~82 GB a rank)
# with the O_DIRECT reader instead of slicing the full checkpoint and re-quantizing. Run on the head Spark while
# the server is stopped (it needs each node's GPU for a few minutes: the quantizers run there, so the prepared bits
# are the serving bits).
#
#   scripts/prepare.sh            rank 0 here and rank 1 on the worker, in parallel; skips a valid folder
#   scripts/prepare.sh --force    rewrite them
#   scripts/prepare.sh status     list the prepared folders on both nodes
#
# Same config as scripts/serve.sh (config/prod.env, CONFIG=path, caller exports win). A folder is keyed by the
# checkpoint, the rank, GLM53_TF_NONEXPERT, torch and the source of the weight-building code, so prepare with the
# IMAGE and GLM53_TF_NONEXPERT you serve with; a start with another key ignores the folder (and, with
# GLM53_TF_PREPARED_WRITE=1, serve.sh's default, writes its own after loading from the checkpoint).
# Folders: HEAD_PREPARED / WORKER_PREPARED (default: <HF cache>/../glm53-tf/prepared), ~82 GB + ~1 GB (drafter) a
# key; serve/prepare keep the newest two keys a rank.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG="${CONFIG:-config/prod.env}"
if [[ ! -f "$CONFIG" ]]; then
    echo "[glm53-tf] no config file $CONFIG. The production config: cp config/prod.env.example config/prod.env" \
         "and fill it in (README Quickstart, AGENTS.md)" >&2
    exit 2
fi
caller_env=$(env | grep -E '^(CONTAINER_RT|HEAD_PREPARED|WORKER_PREPARED|IMAGE|DRAFTER|MODEL_PATH|GLM53_TF_[A-Z0-9_]+)=.' || true)
# shellcheck disable=SC1090
set -a; source "$CONFIG"; set +a
while IFS= read -r kv; do [[ -n "$kv" ]] && export "${kv?}"; done <<< "$caller_env"
# refuse a config that still holds the example's <placeholders>
for _k in WORKER_SSH HEAD_IP HEAD_HF WORKER_HF MODEL_PATH; do
    if [[ "${!_k:-}" == *"<"*">"* ]]; then echo "[glm53-tf] $CONFIG: set $_k (still '${!_k}')" >&2; exit 2; fi
done

NAME="${NAME:-glm53-tf}"
IMAGE="${IMAGE:-glm53-tensorfold:dev}"
# Container runtime: docker (default, stock DGX OS) or podman (rootful). Same contract as serve.sh.
CONTAINER_RT="${CONTAINER_RT:-docker}"
case "$CONTAINER_RT" in
    docker|podman) ;;
    *) echo "[glm53-tf] CONTAINER_RT=$CONTAINER_RT: expected docker or podman" >&2; exit 2 ;;
esac
rt() { printf '%s' "$CONTAINER_RT"; }
HEAD_PREPARED="${HEAD_PREPARED:-${HEAD_HF%/*}/glm53-tf/prepared}"
WORKER_PREPARED="${WORKER_PREPARED:-${WORKER_HF%/*}/glm53-tf/prepared}"
log() { echo "[glm53-tf prepare] $(date '+%F %T') $*"; }
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }

gpu_busy() {
    { nvidia-smi --query-compute-apps=pid --format=csv,noheader; wssh nvidia-smi --query-compute-apps=pid --format=csv,noheader; } \
        | grep -q '[0-9]'
}

run_args() { # $1 = rank, $2 = host HF cache, $3 = host prepared dir, $4.. = extra args for `prepare`
    local rank=$1 hf=$2 prep=$3; shift 3
    local drafter=()
    local gpu=(--gpus all); [[ "$CONTAINER_RT" == podman ]] && gpu=(--device nvidia.com/gpu=all)
    [[ -n "${DRAFTER:-}" && "${NO_DRAFTS:-0}" != 1 ]] && drafter=(--drafter "$DRAFTER")
    echo --rm --name "$NAME-prepare-r$rank" "${gpu[@]}" --ipc=host --network host --ulimit memlock=-1 \
        -v "$hf:/root/.cache/huggingface:ro" -v "$prep:/prepared" -v "$NAME-cache:/cache" \
        -e GLM53_TF_NONEXPERT="${GLM53_TF_NONEXPERT:-bf16}" \
        $(env | grep -E '^GLM53_TF_[A-Z0-9_]+=' | sed 's/^/-e /' | tr '\n' ' ') \
        --entrypoint python "$IMAGE" -m tensorfold.families.glm5_next.cuda.fastboot prepare "$MODEL_PATH" \
        --rank "$rank" --root /prepared "${drafter[@]}" "$@"
}

case "${1:-}" in
status)
    for side in head worker; do
        if [[ $side == head ]]; then
            find "$HEAD_PREPARED" -name manifest.json -printf '%h\n' 2>/dev/null | while read -r d; do
                echo "head   $(du -sh "$d" | cut -f1)  $d"; done
        else
            wssh "find '$WORKER_PREPARED' -name manifest.json -printf '%h\n' 2>/dev/null | while read -r d; do echo \"worker \$(du -sh \"\$d\" | cut -f1)  \$d\"; done"
        fi
    done
    ;;
""|--force)
    extra=()
    [[ "${1:-}" == --force ]] && extra=(--force)
    if gpu_busy; then log "a CUDA process is running on a node; stop the server first (scripts/serve.sh stop)"; exit 1; fi
    mkdir -p "$HEAD_PREPARED"
    t0=$(date +%s)
    log "rank 1 on $WORKER_SSH -> $WORKER_PREPARED; rank 0 here -> $HEAD_PREPARED (GLM53_TF_NONEXPERT=${GLM53_TF_NONEXPERT:-bf16}, $IMAGE)"
    # shellcheck disable=SC2046
    wssh "mkdir -p '$WORKER_PREPARED' && $(rt) run $(run_args 1 "$WORKER_HF" "$WORKER_PREPARED" "${extra[@]}")" \
        > >(sed 's/^/[r1] /') 2>&1 &
    wpid=$!
    rc=0
    # shellcheck disable=SC2046
    $(rt) run $(run_args 0 "$HEAD_HF" "$HEAD_PREPARED" "${extra[@]}") 2>&1 | sed 's/^/[r0] /' || rc=1
    wait "$wpid" || rc=1
    log "done in $(( $(date +%s) - t0 )) s (rc=$rc)"
    exit $rc
    ;;
*)
    sed -n '2,17p' "$0"; exit 2 ;;
esac
