#!/usr/bin/env bash
# W18 (from W17): open a test window on the head node: lease + refresher, watchdog timer stopped, prod stopped. window-start.sh NAME [MAXMIN]
cd $HOME/glm53-tensorfold-spark
R=results/W18; n=${1:-w}
touch $HOME/.test-window-lease
[ -f $R/lease.pid ] && kill $(cat $R/lease.pid) 2>/dev/null
nohup bash $R/lease.sh ${2:-240} > /dev/null 2>&1 &
echo $! > $R/lease.pid
systemctl --user stop glm53-tf-watchdog.timer
echo "watchdog timer: $(systemctl --user is-active glm53-tf-watchdog.timer)"
CONFIG=config/prod.env scripts/serve.sh stop > $R/stop-$n.log 2>&1
echo "$n start $(date +%T) lease pid $(cat $R/lease.pid)" | tee -a $R/windows.log
docker ps --format '{{.Names}} {{.Image}}'; ssh -o BatchMode=yes $WORKER_SSH "docker ps --format '{{.Names}} {{.Image}}'"
