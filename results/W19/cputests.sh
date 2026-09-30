#!/usr/bin/env bash
# W19 step 1: host-only suites in image b10 (no GPU, prod may be serving: nice 19 on 4 little cores), on the worker node.
#   cputests.sh  -> results/W19/tests-cpu/ (one log a file + SUMMARY)
REPO=$HOME/glm53-tensorfold-spark; HF=/root/.cache/huggingface; IMG=glm53-tensorfold:b10
TOK=/root/.cache/huggingface/hub/models--neko-legends--GLM-5.3-Flash-Uncensored-EXL3/snapshots/07135ec082f8f11f7a71e4244a4e5167a0f96277
cd $REPO; O=results/W19/tests-cpu; mkdir -p $O
run() { # name timeout cmd...
  local n=$1 t=$2; shift 2; local t0=$(date +%s)
  nice -n 19 docker run --rm --name w19-cpu-$n --cpuset-cpus 0-3 --network host -e PYTHONDONTWRITEBYTECODE=1 \
    -e GLM53_TF_TOKENIZER_DIR=$TOK -e NVCC=/usr/local/cuda/bin/nvcc -v $HF:/root/.cache/huggingface:ro \
    -v $REPO/tests:/work/tests -v $REPO/bench:/work/bench --entrypoint bash $IMG -c \
    "pip install -q pytest jsonschema >/dev/null 2>&1; cd /work && PYTHONPATH=/src/TensorFold/src:/src/TensorFold/tests/cuda:/work/tests/cuda:/work/tests timeout -k 30 $t python -m pytest -q -p no:cacheprovider -rs $*" > $O/$n.log 2>&1
  local rc=$?; docker rm -f w19-cpu-$n >/dev/null 2>&1
  echo "$(date +%T) $n rc=$rc $(( $(date +%s) - t0 ))s :: $(grep -E 'passed|failed|error|no tests ran' $O/$n.log | tail -1 | cut -c1-200)" | tee -a $O/SUMMARY
}
echo "$(date +%T) start" | tee -a $O/SUMMARY
run upstream-ports 1200 tests/test_upstream_ports.py
run disconnect 1200 tests/cuda/test_disconnect_patches.py
run grammar 1200 tests/cuda/test_grammar_patches.py
run http-pin 600 tests/test_http_pin.py
run size-switch 900 tests/test_decode_size_switch.py
run loads-compile 1200 tests/test_decode_loads_compile.py
run fat2-compile 1200 tests/test_fat2_compile.py
run fat2-bench 600 tests/test_fat2_bench.py
run loads-emu 1800 tests/test_decode_loads_emulator.py
run fat2-emu 1800 tests/test_fat2_emulator.py
echo "$(date +%T) done" | tee -a $O/SUMMARY
