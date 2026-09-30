#!/usr/bin/env bash
# W19: one test window on the head node (from W18's run-window.sh):
#   win.sh WIN NAME [K=V ...]
# window-start (lease + refresher, watchdog timer stopped, prod stopped), deadman (+100 min), then, in its own process
# group with a hard limit of LIMIT_MIN minutes from the window's start (default 92):
#   KGATES=1: kgates.sh on both nodes in parallel (at most KG_MIN minutes, default 24; leftovers removed)
#   NCCL=1:   ncclbw2.sh (NCCL sweep + gate, at most 10 min)
#   the load NAME traced: load.sh NAME-T TRACE=1 GLM53_TF_BATCH_RESERVE_GB=6 K=V ... (nsys launch costs ~4 GiB, so the
#     4th slot needs the lower admission reserve; this start is only for the capture) + cap.sh NAME (one nsys window)
#   the load NAME plain: gates.sh NAME K=V ... (PRE = mpf.py grouped == alone + c4.py thinking low; POST_AB from the
#     environment; then W18's memory sequence), CFG from the environment (default config/prod.env)
# then restore.sh (prod from config/prod.env, 4 slots, canary, https, 17*23, watchdog on, lease gone).
cd $HOME/glm53-tensorfold-spark
R=results/W19; WIN=$1; N=$2; shift 2
LIMIT_MIN=${LIMIT_MIN:-92}; KG_MIN=${KG_MIN:-24}
for i in $(seq 1 36); do
  inf=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c "import json,sys;print(json.load(sys.stdin).get('inflight',0))" 2>/dev/null)
  [[ "${inf:-0}" == 0 ]] && break; sleep 5
done
echo "prod inflight before the window: ${inf:-?} ($(date +%T))"
t0=$(date +%s)
bash $R/window-start.sh $WIN 130
nohup bash $R/deadman.sh "$(date -d '+100 min' +%H%M)" > /dev/null 2>&1 & dm=$!
KV=("$@")
cat > /tmp/w19-win-$WIN.sh <<EOS
cd $HOME/glm53-tensorfold-spark; R=$R
if [[ "${KGATES:-0}" == 1 ]]; then
  echo "=== kgates \$(date +%T)"
  timeout $(( KG_MIN * 60 )) bash \$R/kgates.sh head > \$R/kg-head.out 2>&1 &
  ssh -o BatchMode=yes $WORKER_SSH "cd $HOME/glm53-tensorfold-spark && timeout $(( KG_MIN * 60 )) bash results/W19/kgates.sh worker" > \$R/kg-worker.out 2>&1 &
  wait
  docker ps -q --filter name=w19-kg | xargs -r docker rm -f >/dev/null
  ssh -o BatchMode=yes $WORKER_SSH "docker ps -q --filter name=w19-kg | xargs -r docker rm -f" >/dev/null
  scp -q -r $WORKER_SSH:$HOME/glm53-tensorfold-spark/results/W19/kg-worker \$R/ 2>/dev/null
  echo "=== kgates done \$(date +%T)"; cat \$R/kg-head/SUMMARY \$R/kg-worker/SUMMARY 2>/dev/null
fi
if [[ "${NCCL:-0}" == 1 ]]; then
  echo "=== ncclbw2 \$(date +%T)"; timeout 600 bash \$R/ncclbw2.sh glm53-tensorfold:b10 > \$R/ncclbw2.out 2>&1
  docker ps -q --filter ancestor=glm53-tensorfold:b10 | xargs -r docker rm -f >/dev/null
  cat \$R/ncclgate.txt 2>/dev/null; echo "=== ncclbw2 done \$(date +%T)"
fi
echo "=== $N traced \$(date +%T)"
if bash \$R/load.sh $N-T TRACE=1 GLM53_TF_BATCH_RESERVE_GB=6 ${KV[*]}; then
  timeout 900 bash \$R/cap.sh $N > \$R/cap-$N.out 2>&1; tail -4 \$R/cap-$N.out
  docker logs glm53-tf-r0 > \$R/run-$N-T-r0.log 2>&1; ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > \$R/run-$N-T-r1.log 2>&1
else
  echo "TRACED LOAD $N-T FAILED"; docker logs glm53-tf-r0 > \$R/run-$N-T-r0.log 2>&1
fi
echo "=== $N plain \$(date +%T)"
bash \$R/gates.sh $N ${KV[*]}
EOS
PRE="python3 $R/mpf.py group $R/mpf-$N.json w19$(echo $N | tr 'A-Z' 'a-z') 10 > $R/mpf-$N.log 2>&1; tail -3 $R/mpf-$N.log | cut -c1-300; \
python3 $R/c4.py $R/c4-$N.json 3 > $R/c4-$N.log 2>&1; grep SUMMARY $R/c4-$N.log"
export PRE BASE=http://127.0.0.1:8001 POST_AB CFG
setsid bash /tmp/w19-win-$WIN.sh > $R/run-$N.out 2>&1 &
gp=$!
while kill -0 $gp 2>/dev/null; do
  if (( $(date +%s) - t0 > LIMIT_MIN * 60 )); then
    echo "=== $WIN: window limit ${LIMIT_MIN} min reached $(date +%T), killing" | tee -a $R/run-$N.out
    kill -- -$gp 2>/dev/null; sleep 3; kill -9 -- -$gp 2>/dev/null
    for f in mem meminfo memfast0 memfast1; do [ -f $R/$f.pid ] && kill $(cat $R/$f.pid) 2>/dev/null; rm -f $R/$f.pid; done
    docker logs glm53-tf-r0 > $R/run-$N-r0.log 2>&1
    break
  fi
  sleep 10
done
wait $gp 2>/dev/null
bash $R/restore.sh $WIN > $R/restore-$WIN.out 2>&1
kill $dm 2>/dev/null; pkill -f "results/W19/deadman.sh" 2>/dev/null
echo "=== $WIN window closed $(date +%T), $(( ($(date +%s) - t0) / 60 )) min" | tee -a $R/run-$N.out
