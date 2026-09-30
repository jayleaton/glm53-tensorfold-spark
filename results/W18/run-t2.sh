#!/usr/bin/env bash
# W18 task 2: clean A/B of GLM53_TF_CPU_PIN=http (patches/0530) on image b10 (= b9's list + 0530), normal (prod) clocks.
#   run-t2.sh [build] [K=V ...]   K=V: extra knobs for BOTH loads (the prod config's own values otherwise)
# CT = b10 + RoCE trace + dump (the control, same knobs, same length), P = CT + GLM53_TF_CPU_PIN=http. Each: ab.sh set
# (warm-up, exact 10/10, batchexact 4/4, transcripts, ab.py 24.5k / 98k (reply sha), glmbench 13 cells x3 = 1 stream,
# 4 streams x3 twice = 6 reps, lone slots); P also records its threads' cpus during a 4-stream burst. Order CT, P, CT2
# (a second control if time allows, to bound drift). Window-start / deadman / restore as run-window.sh.
cd $HOME/glm53-tensorfold-spark
R=results/W18
LIMIT_MIN=${LIMIT_MIN:-68}
BUILD=0; [[ "${1:-}" == build ]] && { BUILD=1; shift; }
KN=("$@")
TRACE=(GLM53_TF_ROCE_TRACE=4096 GLM53_TF_ROCE_TRACE_EVERY=65536)
for i in $(seq 1 36); do
  inf=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c "import json,sys;print(json.load(sys.stdin).get('inflight',0))" 2>/dev/null)
  [[ "${inf:-0}" == 0 ]] && break; sleep 5
done
echo "prod inflight before the window: ${inf:-?} ($(date +%T))"
t0=$(date +%s)
left() { echo $(( LIMIT_MIN * 60 - ($(date +%s) - t0) )); }
bash $R/window-start.sh T2 100
nohup bash $R/deadman.sh "$(date -d '+78 min' +%H%M)" > /dev/null 2>&1 & dm=$!
if [[ $BUILD == 1 ]]; then
  echo "=== build b10 $(date +%T)"
  L="$(cat results/W17/build-patches.txt | tr -s ' ') 0530"; echo "$L" > $R/build-patches.txt
  timeout 1200 docker build -f docker/Dockerfile --build-arg PATCHES="$L" -t glm53-tensorfold:b10 . > $R/build.log 2>&1; echo "build rc=$?"
  grep -E "applying.*05|skipping.*05" $R/build.log | cut -c1-120
  s=$(date +%s); docker save glm53-tensorfold:b10 | ssh -o BatchMode=yes $WORKER_SSH docker load > $R/ship.log 2>&1
  echo "ship rc=$? $(( $(date +%s) - s ))s" | tee -a $R/ship.log
  docker run --rm -v $PWD/tests:/t --entrypoint bash glm53-tensorfold:b10 -c "cd /t && python -m pytest -q test_http_pin.py 2>&1 | tail -2" > $R/test-http-pin.log 2>&1; cat $R/test-http-pin.log
fi
collect() { local n=$1
  docker exec glm53-tf-r0 sh -c "cat /sessions/w18-roce-$n-r0.jsonl 2>/dev/null; rm -f /sessions/w18-roce-$n-r0.jsonl" > "$R/roce-$n-r0.jsonl"
  ssh -o BatchMode=yes $WORKER_SSH "docker exec glm53-tf-r1 sh -c 'cat /sessions/w18-roce-$n-r1.jsonl 2>/dev/null; rm -f /sessions/w18-roce-$n-r1.jsonl'" > "$R/roce-$n-r1.jsonl"
  echo "$n: RoCE dumps $(wc -l < $R/roce-$n-r0.jsonl) / $(wc -l < $R/roce-$n-r1.jsonl) lines"
  docker logs glm53-tf-r0 > $R/run-$n-r0.log 2>&1
  grep -ciE 'traceback|error' $R/run-$n-r0.log | sed "s/^/$n r0 error lines: /"
}
threads() { # the engine container's threads and their allowed cpus / last cpu, while 4 streams run
  python3 bench/multiturn.py --base http://127.0.0.1:8001 --model GLM-5.3-Flash-EXL3 --modes concurrent --streams 4 --reps 1 --long-tokens 256 --out /tmp/w18-thr.json > /dev/null 2>&1 &
  local p=$!; sleep 12
  docker exec glm53-tf-r0 sh -c 'for t in /proc/[0-9]*/task/*; do echo "$(cat $t/comm) | allowed $(grep Cpus_allowed_list $t/status | cut -f2) | on $(awk "{print \$39}" $t/stat)"; done | sort | uniq -c | sort -rn' > "$R/threads-$1-r0.txt" 2>&1
  wait $p; head -30 "$R/threads-$1-r0.txt"
}
one() { local n=$1; shift
  (( $(left) > 1500 )) || { echo "=== $n skipped: $(left) s left in the window"; return 1; }
  bash $R/load.sh $n IMAGE=glm53-tensorfold:b10 "${KN[@]}" "${TRACE[@]}" GLM53_TF_ROCE_TRACE_DUMP=/sessions/w18-roce-$n "$@" || { echo "LOAD $n FAILED"; return 1; }
  docker logs glm53-tf-r0 2>&1 | grep -iE "cpu pin|cpupin|trace dump" | cut -c1-200
  [[ $n == P* ]] && threads $n
  bash $R/ab.sh $n
  collect $n
}
setsid bash -c "$(declare -f left collect threads one); t0=$t0; LIMIT_MIN=$LIMIT_MIN; R=$R; KN=(${KN[*]}); TRACE=(${TRACE[*]})
one CT; one P GLM53_TF_CPU_PIN=http; one CT2" > $R/run-T2.out 2>&1 &
gp=$!
while kill -0 $gp 2>/dev/null; do
  if (( $(left) < 0 )); then echo "=== T2: window limit reached $(date +%T), killing" | tee -a $R/run-T2.out; kill -- -$gp; sleep 3; kill -9 -- -$gp 2>/dev/null; break; fi
  sleep 10
done
wait $gp 2>/dev/null
python3 $R/rocetrace.py $R/roce-CT --json $R/rocetrace-CT.json > $R/rocetrace-CT.txt 2>&1
python3 $R/rocetrace.py $R/roce-P --vs $R/roce-CT --json $R/rocetrace-P.json > $R/rocetrace-P.txt 2>&1; grep GATE $R/rocetrace-P.txt
bash $R/restore.sh T2 > $R/restore-T2.out 2>&1
kill $dm 2>/dev/null; pkill -f "results/W18/deadman.sh" 2>/dev/null
echo "=== T2 window closed $(date +%T), $(( ($(date +%s) - t0) / 60 )) min" | tee -a $R/run-T2.out
