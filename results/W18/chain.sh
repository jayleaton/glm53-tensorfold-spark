#!/usr/bin/env bash
# W18: run windows one after another with prod up >= GAP_MIN (default 21) minutes between them:
#   chain.sh "A|" "T|GLM53_TF_ADMIT_MEM=available GLM53_TF_SELECT_SCRATCH=grow GLM53_TF_ALLOC_TRIM_GB=0" ...
# Each argument = NAME|K=V K=V. Before a window: prod must answer /v1/models on :8000 and show 4 request slots, the
# watchdog timer active and no lease; otherwise the chain stops (prod left as it is for a human / the watchdog).
cd $HOME/glm53-tensorfold-spark
R=results/W18; GAP_MIN=${GAP_MIN:-21}
last=${FIRST_AFTER:-0}
for spec in "$@"; do
  N=${spec%%|*}; KV=${spec#*|}
  while (( $(date +%s) < last + GAP_MIN * 60 )); do sleep 20; done
  ok=1
  curl -s -m 10 http://127.0.0.1:8000/v1/models | grep -q GLM-5.3 || ok=0
  bash $R/slotcheck.sh > /dev/null || ok=0
  [[ "$(systemctl --user is-active glm53-tf-watchdog.timer)" == active ]] || ok=0
  [[ -f $HOME/.test-window-lease ]] && ok=0
  if [[ $ok != 1 ]]; then echo "chain: prod not healthy before $N ($(date +%T)); stopping" | tee -a $R/windows.log; exit 1; fi
  echo "chain: window $N ($KV) $(date +%T)" | tee -a $R/windows.log
  bash $R/run-window.sh $N $KV > $R/window-$N.out 2>&1
  last=$(date +%s)
  grep -q "restored" $R/restore-$N.out || { echo "chain: restore after $N not confirmed; stopping" | tee -a $R/windows.log; exit 1; }
done
echo "chain done $(date +%T)" | tee -a $R/windows.log
