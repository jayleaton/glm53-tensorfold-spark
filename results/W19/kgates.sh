#!/usr/bin/env bash
# W19 step 2: standalone kernel gates in image b10, prod stopped, one node each, every run under timeout, clocks as
# they are (unlocked; each gate compares old vs new in the same process).
#   kgates.sh head | worker     -> results/W19/kg-NODE/ (one log a run + SUMMARY)
# head: 0580 bench --loads (probe 3 >= 220 GB/s, best cfg >= 1.05x grouped_kernel flush, >= 1.00x rotate; Z bitwise
#        before timing), 0570 cold per shape (bench_decode_cold, both modes), 0570 / 0580 bitwise (test_decode_loads_patches)
# worker: 0590 bitwise (test_fat2_patches -k "not engine"), 0590 floor gate + contended makespan (fat2gate.py), then the
#        0590 engine tests if time is left
# The glm53-tf-cache volume is mounted at /cache so the JIT-built extensions (TORCH_EXTENSIONS_DIR) are ready for the loads.
node=$1
IMG=glm53-tensorfold:b10
if [[ $node == *02 ]]; then REPO=$HOME/glm53-tensorfold-spark; HF=/root/.cache/huggingface; else REPO=$HOME/glm53-tensorfold-spark; HF=$HOME/.cache/huggingface; fi
cd $REPO; O=results/W19/kg-$node; mkdir -p $O
P="python -m pytest -q -p no:cacheprovider -rs"
run() { # name timeout cmd...
  local n=$1 t=$2; shift 2; local t0=$(date +%s)
  docker run --rm --name w19-kg-$n --gpus all --network host --ipc host -e PYTHONDONTWRITEBYTECODE=1 \
    -v $HF:/root/.cache/huggingface:ro -v glm53-tf-cache:/cache -v $REPO/tests:/work/tests -v $REPO/bench:/work/bench \
    -v $REPO/results/W19:/w19 --entrypoint bash $IMG -c \
    "pip install -q pytest >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda:/work/tests timeout -k 30 $t $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w19-kg-$n >/dev/null 2>&1
  echo "$(date +%T) $node $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'GATE|passed|failed|error|BITS' $O/$n.log | tail -1 | cut -c1-240)" | tee -a $O/SUMMARY
}
echo "$(date +%T) start $node" | tee -a $O/SUMMARY
case $node in
head)
  run loads-bench 1500 python tests/cuda/bench_decode_kernels.py --loads --json /w19/kg-head/loads.json
  grep -E "^GATE" $O/loads-bench.log | tee -a $O/SUMMARY
  run cold570 600 python tests/cuda/bench_decode_cold.py --mode both --no-hot --shapes 1024x4096 4096x128 160x4096 4096x1024 4096x1536 8192x512 --json /w19/kg-head/cold570.json
  run loads-bits 900 $P tests/cuda/test_decode_loads_patches.py ;;
worker)
  run fat2-bits 900 $P -x tests/cuda/test_fat2_patches.py -k "'not engine'"
  run fat2-gate 1800 python /w19/fat2gate.py /w19/kg-worker/fat2gate.json 2048 4096 8192
  grep -E "^GATE|BITS" $O/fat2-gate.log | tee -a $O/SUMMARY
  [[ -n "${KG_ENGINE:-}" ]] && run fat2-engine 900 $P tests/cuda/test_fat2_patches.py -k engine ;;
esac
echo "$(date +%T) done $node" | tee -a $O/SUMMARY
