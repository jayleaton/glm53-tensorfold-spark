#!/usr/bin/env bash
# W18: one test window = one load through the W17 memory / prefill sequence, then prod restored.
#   run-window.sh NAME [K=V ...]      (IMAGE defaults to b9 in load.sh)
# window-start (lease + refresher, watchdog timer stopped, prod stopped), then gates.sh with W17's PRE (mpf.py group +
# c4.py thinking low and off = the "heavy warm-up"), then restore.sh. Hard limit: the gates run in their own process
# group and are killed at LIMIT_MIN (default 66) minutes after the window opened; restore always runs. deadman.sh at
# +78 min restores prod if this script itself dies.
cd $HOME/glm53-tensorfold-spark
R=results/W18; N=$1; shift
LIMIT_MIN=${LIMIT_MIN:-66}
# let a running client request finish first (up to 3 min)
for i in $(seq 1 36); do
  inf=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c "import json,sys;print(json.load(sys.stdin).get('inflight',0))" 2>/dev/null)
  [[ "${inf:-0}" == 0 ]] && break; sleep 5
done
echo "prod inflight before the window: ${inf:-?} ($(date +%T))"
t0=$(date +%s)
bash $R/window-start.sh $N 100
nohup bash $R/deadman.sh "$(date -d '+78 min' +%H%M)" > /dev/null 2>&1 & dm=$!
# optional per-window extra before the load (prod already stopped): results/W18/prewindow-NAME.sh, at most 15 min
[ -f $R/prewindow-$N.sh ] && { echo "=== prewindow-$N $(date +%T)"; timeout 900 bash $R/prewindow-$N.sh > $R/prewindow-$N.out 2>&1; echo "prewindow rc=$? $(date +%T)"; }
tag=W18$(echo $N | tr 'A-Z' 'a-z')
PRE="python3 $R/mpf.py group $R/mpf-$N.json $tag 10 > $R/mpf-$N.log 2>&1; tail -3 $R/mpf-$N.log | cut -c1-300; \
python3 $R/c4.py $R/c4-$N.json 3 > $R/c4-$N.log 2>&1; grep SUMMARY $R/c4-$N.log; \
C4_THINK=0 python3 $R/c4.py $R/c4-$N-nothink.json 3 > $R/c4-$N-nothink.log 2>&1; grep SUMMARY $R/c4-$N-nothink.log"
export PRE BASE=http://127.0.0.1:8001
setsid bash $R/${GATES:-gates.sh} $N "$@" > $R/run-$N.out 2>&1 &
gp=$!
while kill -0 $gp 2>/dev/null; do
  if (( $(date +%s) - t0 > LIMIT_MIN * 60 )); then
    echo "=== $N: window limit ${LIMIT_MIN} min reached $(date +%T), killing the gates" | tee -a $R/run-$N.out
    kill -- -$gp 2>/dev/null; sleep 3; kill -9 -- -$gp 2>/dev/null
    for f in mem meminfo memfast0 memfast1; do [ -f $R/$f.pid ] && kill $(cat $R/$f.pid) 2>/dev/null; rm -f $R/$f.pid; done
    docker logs glm53-tf-r0 > $R/run-$N-r0.log 2>&1
    break
  fi
  sleep 10
done
wait $gp 2>/dev/null
bash $R/restore.sh $N > $R/restore-$N.out 2>&1
kill $dm 2>/dev/null; pkill -f "results/W18/deadman.sh" 2>/dev/null
echo "=== $N window closed $(date +%T), $(( ($(date +%s) - t0) / 60 )) min" | tee -a $R/run-$N.out
