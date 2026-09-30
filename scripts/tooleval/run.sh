#!/usr/bin/env bash
# Tool-calling benchmarks against our OpenAI endpoint (docs/TOOL-CALLING.md §5): tool-eval-bench (SeraphimSerapis, MIT)
# and spark-bench TrueScore (Weschera, README says MIT). Installs live in ~/.cache/tooleval (see `setup`).
#
#   scripts/tooleval/run.sh setup                         # clone the pinned commits + venvs (no server needed)
#   scripts/tooleval/run.sh teb  off|low|high [label]     # tool-eval-bench, 69 scenarios (the tester's suite)
#   scripts/tooleval/run.sh sb   off|low|high [label]     # spark-bench TrueScore, 76 scenarios x 2, temp 0.3
#   scripts/tooleval/run.sh both off|low|high [label]
#   scripts/tooleval/run.sh chains off|low|high [label]   # quick: tool-eval-bench C (multi-step) + spark-bench agentic
#
# Modes: off  = chat_template_kwargs {"enable_thinking": false} (the tester's setting)
#        low  = thinking on, template "Reasoning Effort: Low"
#        high = thinking on, "Reasoning Effort: High" (our prod default for thinking requests)
# Env: BASE_URL (http://127.0.0.1:8000/v1), MODEL (GLM-5.3-Flash-EXL3), API_KEY, OUT (results/tooleval),
#      TEB_ARGS / SB_ARGS (extra flags), TEB_TEMP (0.0, the bench default), SB_TEMP (0.3), SB_REPEATS (2),
#      TEB_TIMEOUT (off: 300, thinking: 900 s between streamed chunks).
# Each run writes <OUT>/<date>-<bench>-<mode>[-label]/ with the bench's own reports, the server's /health and
# /v1/models before and after, and the exact command line. Never run it while a GPU campaign owns the Sparks.
set -euo pipefail

CACHE=${TOOLEVAL_CACHE:-$HOME/.cache/tooleval}
TEB_REPO=https://github.com/SeraphimSerapis/tool-eval-bench.git
TEB_REV=c7b5b9550                      # v2.7.0+8 (2026-09-29), the commit this doc's reading is from
SB_REPO=https://github.com/Weschera/spark-bench.git
SB_REV=125ba161                        # v6.8.0 (2026-08-29)
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
MODEL=${MODEL:-GLM-5.3-Flash-EXL3}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$ROOT/results/tooleval}

die() { echo "run.sh: $*" >&2; exit 1; }

setup() {
    mkdir -p "$CACHE"
    if [[ ! -d $CACHE/tool-eval-bench/.git ]]; then git clone -q "$TEB_REPO" "$CACHE/tool-eval-bench"; fi
    git -C "$CACHE/tool-eval-bench" fetch -q origin && git -C "$CACHE/tool-eval-bench" checkout -q "$TEB_REV"
    [[ -x $CACHE/tool-eval-bench/.venv/bin/python ]] || python3 -m venv "$CACHE/tool-eval-bench/.venv"
    "$CACHE/tool-eval-bench/.venv/bin/pip" install -q -e "$CACHE/tool-eval-bench"
    if [[ ! -d $CACHE/spark-bench/.git ]]; then git clone -q "$SB_REPO" "$CACHE/spark-bench"; fi
    git -C "$CACHE/spark-bench" fetch -q origin && git -C "$CACHE/spark-bench" checkout -q "$SB_REV"
    [[ -x $CACHE/spark-bench/.venv/bin/python ]] || python3 -m venv "$CACHE/spark-bench/.venv"
    # the golden gate renders pages (VIS scenarios): playwright + pillow and a chromium
    "$CACHE/spark-bench/.venv/bin/pip" install -q playwright pillow
    "$CACHE/tool-eval-bench/.venv/bin/tool-eval-bench" --help >/dev/null
    (cd "$CACHE/spark-bench" && .venv/bin/python spark_bench.py eval --help >/dev/null)
    echo "installed: tool-eval-bench $(git -C "$CACHE/tool-eval-bench" rev-parse --short HEAD)," \
         "spark-bench $(git -C "$CACHE/spark-bench" rev-parse --short HEAD) in $CACHE"
}

# thinking mode -> request fields. Both benches merge extra body fields into every request; ours reads
# chat_template_kwargs (enable_thinking, reasoning_effort) and, with GLM53_TF_EFFORT_FIELD=1, top-level reasoning_effort.
kwargs_for() {
    case $1 in
        off)  echo '{"enable_thinking": false}' ;;
        low)  echo '{"enable_thinking": true, "reasoning_effort": "low"}' ;;
        high) echo '{"enable_thinking": true, "reasoning_effort": "high"}' ;;
        *) die "mode must be off, low or high (got '$1')" ;;
    esac
}

snapshot() {       # $1 = dir, $2 = tag: what the server was (patch knobs show in the log, not here)
    local host=${BASE_URL%/v1}
    curl -fsS -m 10 "$host/health" -o "$1/health-$2.json" 2>/dev/null || echo '{"unreachable": true}' > "$1/health-$2.json"
    curl -fsS -m 10 "$BASE_URL/models" -o "$1/models-$2.json" 2>/dev/null || true
}

preflight() {
    curl -fsS -m 10 "$BASE_URL/models" >/dev/null || die "no server at $BASE_URL (start prod first; never during a GPU campaign)"
    curl -fsS -m 10 "$BASE_URL/models" | grep -q "\"$MODEL\"" || die "$BASE_URL/models does not list $MODEL"
}

run_teb() {        # $1 = mode, $2 = dir, $3.. = extra args (scenario / category filters)
    local mode=$1 dir=$2; shift 2
    local bin=$CACHE/tool-eval-bench/.venv/bin/tool-eval-bench
    [[ -x $bin ]] || die "tool-eval-bench not installed: scripts/tooleval/run.sh setup"
    local args=(--base-url "$BASE_URL" --model "$MODEL" --no-live --temperature "${TEB_TEMP:-0.0}"
                --output-dir "$dir/teb" --json-file "$dir/teb.json")
    [[ -n ${API_KEY:-} ]] && args+=(--api-key "$API_KEY")
    if [[ $mode == off ]]; then
        args+=(--no-think --timeout "${TEB_TIMEOUT:-300}")
    else
        # max_tokens: the bench's 16384 for thinking; our prod MAX_TOKENS is 32768
        args+=(--timeout "${TEB_TIMEOUT:-900}"
               --backend-kwargs "{\"chat_template_kwargs\": $(kwargs_for "$mode"), \"reasoning_effort\": \"$mode\"}")
    fi
    # shellcheck disable=SC2206
    args+=("$@" ${TEB_ARGS:-})
    printf '%q ' "$bin" "${args[@]}" > "$dir/teb.cmd"; echo >> "$dir/teb.cmd"
    "$bin" "${args[@]}" 2>&1 | tee "$dir/teb.log"
}

run_sb() {         # $1 = mode, $2 = dir, $3.. = extra args
    local mode=$1 dir=$2; shift 2
    [[ -x $CACHE/spark-bench/.venv/bin/python ]] || die "spark-bench not installed: scripts/tooleval/run.sh setup"
    local args=(eval --endpoint "$BASE_URL" --model "$MODEL" --label "GLM53-TF-$mode${LABEL:+-$LABEL}"
                --out-dir "$dir/sb" --repeats "${SB_REPEATS:-2}" --temperature "${SB_TEMP:-0.3}"
                --uncapped --timeout 0 --tier all --skip-throughput)
    local extra
    if [[ $mode == off ]]; then
        args+=(--thinking off)
        extra=""
    else
        # the bench's --thinking on sends chat_template_kwargs {enable_thinking, thinking_mode}; SPARK_BENCH_EXTRA_BODY
        # replaces that dict (shallow update), so it carries enable_thinking and the effort itself
        args+=(--thinking on)
        extra="{\"chat_template_kwargs\": $(kwargs_for "$mode"), \"reasoning_effort\": \"$mode\"}"
    fi
    # shellcheck disable=SC2206
    args+=("$@" ${SB_ARGS:-})
    { [[ -n $extra ]] && printf 'SPARK_BENCH_EXTRA_BODY=%q ' "$extra"; printf '%q ' .venv/bin/python spark_bench.py "${args[@]}"; echo; } > "$dir/sb.cmd"
    (cd "$CACHE/spark-bench" && SPARK_BENCH_EXTRA_BODY="$extra" SPARK_BENCH_API_KEY="${API_KEY:-}" \
        SPARK_BENCH_DUMP_DIR="$dir/sb-dump" .venv/bin/python spark_bench.py "${args[@]}") 2>&1 | tee "$dir/sb.log"
}

cmd=${1:-}; mode=${2:-}; LABEL=${3:-}
case $cmd in
    setup) setup; exit 0 ;;
    teb|sb|both|chains) kwargs_for "$mode" >/dev/null ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
preflight
dir=$OUT/$(date +%Y%m%d-%H%M)-$cmd-$mode${LABEL:+-$LABEL}
mkdir -p "$dir"
snapshot "$dir" before
case $cmd in
    teb)    run_teb "$mode" "$dir" ;;
    sb)     run_sb "$mode" "$dir" ;;
    both)   run_teb "$mode" "$dir"; run_sb "$mode" "$dir" ;;
    chains) run_teb "$mode" "$dir" --categories C
            SB_REPEATS=${SB_REPEATS:-3} run_sb "$mode" "$dir" --domains agentic ;;
esac
snapshot "$dir" after
echo "results: $dir"
