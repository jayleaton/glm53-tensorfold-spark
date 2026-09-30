#!/usr/bin/env bash
# W18 (from W17): start a test load from config/prod.env with overrides: load.sh NAME [K=V ...]; IMAGE defaults to b9.
# Page cache dropped on both nodes before the start; serve.sh start (2a1f017) re-checks the slot count and restarts
# on fewer than GLM53_TF_BATCH; this script checks it once more (exit 3 if short).
cd $HOME/glm53-tensorfold-spark
R=results/W18; name=$1; shift
for kv in "$@"; do export "$kv"; done
export PORT=${PORT:-8001}  # W18: tests on :8001, away from live client traffic (prod :8000 is down in a window)
export IMAGE=${IMAGE:-glm53-tensorfold:b9}
echo "load $name: IMAGE=$IMAGE $* ($(date +%T))" | tee -a $R/loads.log
CONFIG=config/prod.env scripts/serve.sh stop >/dev/null 2>&1
bash $R/dropc.sh | tee -a $R/loads.log
CONFIG=${CFG:-config/prod.env} timeout 1800 scripts/serve.sh start > "$R/start-$name.log" 2>&1
rc=$?
echo "start rc=$rc $(date +%T)" | tee -a $R/loads.log; tail -3 "$R/start-$name.log"
docker logs glm53-tf-r0 > "$R/boot-$name-r0.log" 2>&1
ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > "$R/boot-$name-r1.log" 2>&1
grep -iE 'roce conn|roce:|context:|vision|l2pf|batching|session store|snapshot|multi|memsafe|scratch|admission|admit|trim|reasoning' "$R/boot-$name-r0.log" | cut -c1-240 | head -20
grep -iE 'multi|scratch|error|Traceback' "$R/boot-$name-r1.log" | cut -c1-240 | head -6
bash $R/slotcheck.sh | tee -a $R/loads.log || { echo "SLOTS SHORT on $name"; exit 3; }
exit $rc
