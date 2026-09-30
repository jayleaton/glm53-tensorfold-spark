#!/usr/bin/env bash
# W18 (from W17): close a window: page cache dropped, prod from config/prod.env, slots == 4 verified, local + https
# /v1/models, 17*23 chat, canary (serve.sh start), watchdog timer on, lease refresher killed, lease deleted. restore.sh NAME
cd $HOME/glm53-tensorfold-spark
R=results/W19; n=${1:-w}
docker ps -a --format '{{.Names}}' | grep -E '^w19-' | xargs -r docker rm -f
ssh -o BatchMode=yes $WORKER_SSH "docker ps -a --format '{{.Names}}' | grep -E '^w19-' | xargs -r docker rm -f"
CONFIG=config/prod.env scripts/serve.sh stop > /dev/null 2>&1
for try in 1 2 3; do
  bash $R/dropc.sh
  CONFIG=config/prod.env timeout 1500 scripts/serve.sh start > $R/start-prod-$n.log 2>&1
  echo "start rc=$? (try $try)"; grep -iE "canary|ready|decode|slot" $R/start-prod-$n.log | tail -5
  bash $R/slotcheck.sh && break
  CONFIG=config/prod.env scripts/serve.sh stop > /dev/null 2>&1
done
curl -s -m 10 http://127.0.0.1:8000/v1/models | cut -c1-200; echo
curl -s -m 20 https://<head-host>/v1/models | cut -c1-200; echo
curl -s -m 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash-EXL3","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":64,"reasoning_effort":"none"}' | python3 -c "import json,sys;d=json.load(sys.stdin);print('chat:',repr(d['choices'][0]['message']['content']))"
CONFIG=config/prod.env scripts/serve.sh canary > $R/canary-$n.log 2>&1; echo "canary rc=$? $(tail -1 $R/canary-$n.log | cut -c1-200)"
systemctl --user start glm53-tf-watchdog.timer; echo "watchdog timer: $(systemctl --user is-active glm53-tf-watchdog.timer)"
[ -f $R/lease.pid ] && kill $(cat $R/lease.pid) 2>/dev/null; rm -f $R/lease.pid
[ -f $R/mem.pid ] && kill $(cat $R/mem.pid) 2>/dev/null; rm -f $R/mem.pid
rm -f $HOME/.test-window-lease; ls $HOME/.test-window-lease 2>&1 | tail -1
echo "$n restored $(date +%T)" | tee -a $R/windows.log
