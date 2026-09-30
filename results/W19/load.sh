#!/usr/bin/env bash
# W19 (from W18 / W17 / W11): start a test load: load.sh NAME [K=V ...]; IMAGE defaults to b10, PORT to 8001, CFG to
# config/prod.env. Page cache dropped on both nodes before the start; serve.sh start (2a1f017) re-checks the slot count
# and restarts on fewer than GLM53_TF_BATCH; this script checks it once more (exit 3 if short).
# TRACE=1: the W11 nsys method -- scripts/serve.sh copied to results/W19/serve-w19.sh with ${W19_DOCKER} added to both
# ranks' docker run: the shim (bin/tensorfold under `nsys launch`, idle session "w19"), SYS_ADMIN (RmProfilingAdminOnly),
# /var/tmp/w19 (entry.sh, bin/, out/) on both nodes. One `nsysctl.sh start/stop` per server start.
cd $HOME/glm53-tensorfold-spark
R=results/W19; name=$1; shift
for kv in "$@"; do export "$kv"; done
export PORT=${PORT:-8001}  # W18: tests on :8001, away from live client traffic (prod :8000 is down in a window)
export IMAGE=${IMAGE:-glm53-tensorfold:b10}
echo "load $name: IMAGE=$IMAGE CFG=${CFG:-config/prod.env} TRACE=${TRACE:-0} $* ($(date +%T))" | tee -a $R/loads.log
S=scripts/serve.sh
if [[ "${TRACE:-0}" == 1 ]]; then
  S=$R/serve-w19.sh
  python3 - <<'PY'
s = open("scripts/serve.sh").read()
a = 'cd "$(dirname "${BASH_SOURCE[0]}")/.."'
b = '-e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" "$IMAGE"'
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, 'cd $HOME/glm53-tensorfold-spark   # W19 copy').replace(
    b, '-e GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" ${W19_DOCKER:-} "$IMAGE"   # W19')
open("results/W19/serve-w19.sh", "w").write(s)
PY
  chmod +x $S
  mkdir -p /var/tmp/w19/out /var/tmp/w19/bin; cp $R/entry.sh /var/tmp/w19/; cp $R/bin/tensorfold /var/tmp/w19/bin/
  ssh -o BatchMode=yes $WORKER_SSH mkdir -p /var/tmp/w19/out /var/tmp/w19/bin
  scp -q $R/entry.sh $WORKER_SSH:/var/tmp/w19/; scp -q $R/bin/tensorfold $WORKER_SSH:/var/tmp/w19/bin/
  export W19_DOCKER="--cap-add SYS_ADMIN -v /var/tmp/w19:/w19 --entrypoint /w19/entry.sh -e W19_NSYS=1"
fi
CONFIG=config/prod.env scripts/serve.sh stop >/dev/null 2>&1
bash $R/dropc.sh | tee -a $R/loads.log
CONFIG=${CFG:-config/prod.env} timeout 1800 $S start > "$R/start-$name.log" 2>&1
rc=$?
echo "start rc=$rc $(date +%T)" | tee -a $R/loads.log; tail -3 "$R/start-$name.log"
docker logs glm53-tf-r0 > "$R/boot-$name-r0.log" 2>&1
ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > "$R/boot-$name-r1.log" 2>&1
grep -iE 'roce conn|roce:|context:|batching|memsafe|scratch|admission|trim|multi|expert loads|0570|0580|0590|fat2|size switch|grammar|structured|cpu pin|cpupin|NET/IB : Using|nChannels|calibration|nsys|disconnect' "$R/boot-$name-r0.log" | cut -c1-240 | head -30
grep -iE 'error|Traceback|calibration' "$R/boot-$name-r1.log" | cut -c1-240 | head -6
bash $R/slotcheck.sh | tee -a $R/loads.log || { echo "SLOTS SHORT on $name"; exit 3; }
exit $rc
