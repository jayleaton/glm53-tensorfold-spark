#!/usr/bin/env bash
# W18: a window of short loads (no memory sequence): run-multi.sh WIN "NAME|K=V ..." ...   (CFG=<config> as a K=V
# selects another config file for that load). Per load: ab.sh set (warm-up, exact 10/10, batchexact 4/4, transcripts,
# ab.py 24.5k / 98k (reply sha), glmbench 13 cells x3 = 1 stream, 4 streams x3 twice, lone slots) + ab.py x3 more
# (4 prefill pairs). Optional results/W18/prewindow-WIN.sh first (model stopped). A load starts only with >= 17 min
# left of LIMIT_MIN (68). window-start / deadman / restore as run-window.sh.
cd $HOME/glm53-tensorfold-spark
R=results/W18; WIN=$1; shift
LIMIT_MIN=${LIMIT_MIN:-68}
for i in $(seq 1 36); do
  inf=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c "import json,sys;print(json.load(sys.stdin).get('inflight',0))" 2>/dev/null)
  [[ "${inf:-0}" == 0 ]] && break; sleep 5
done
echo "prod inflight before the window: ${inf:-?} ($(date +%T))"
t0=$(date +%s)
bash $R/window-start.sh $WIN 100
nohup bash $R/deadman.sh "$(date -d '+78 min' +%H%M)" > /dev/null 2>&1 & dm=$!
[ -f $R/prewindow-$WIN.sh ] && { echo "=== prewindow-$WIN $(date +%T)"; timeout 600 bash $R/prewindow-$WIN.sh > $R/prewindow-$WIN.out 2>&1; echo "prewindow rc=$? $(date +%T)"; }
cat > /tmp/w18-multi-$WIN.sh <<EOS
cd $HOME/glm53-tensorfold-spark; R=$R
for spec in $(printf '"%s" ' "$@"); do
  N=\${spec%%|*}; KV=\${spec#*|}
  left=\$(( $LIMIT_MIN * 60 - (\$(date +%s) - $t0) ))
  (( left > 1020 )) || { echo "=== \$N skipped: \$left s left"; continue; }
  cfg=config/prod.env; kvs=()
  for kv in \$KV; do [[ \$kv == CFG=* ]] && cfg=\${kv#CFG=} || kvs+=("\$kv"); done
  echo "=== \$N (\$cfg \${kvs[*]}) \$(date +%T)"
  CFG=\$cfg bash \$R/load.sh \$N "\${kvs[@]}" || { echo "LOAD \$N FAILED"; continue; }
  docker logs glm53-tf-r0 2>&1 | grep -E "NET/IB : Using|Channel [0-9]+/[0-9]+|nChannels" | sort | uniq -c | sort -rn | head -4 | cut -c1-200
  bash \$R/ab.sh \$N
  for k in 2 3 4; do python3 \$R/ab-p.py \$R/ab-\$N-\$k.json 24500,98000 '{"prod":{}}' > \$R/ab-\$N-\$k.log 2>&1; cut -c1-110 \$R/ab-\$N-\$k.log | tail -2; done
  docker logs glm53-tf-r0 > \$R/run-\$N-r0.log 2>&1
  grep -ciE 'traceback|error' \$R/run-\$N-r0.log | sed "s/^/\$N r0 error lines: /"
done
EOS
setsid bash /tmp/w18-multi-$WIN.sh > $R/run-$WIN.out 2>&1 &
gp=$!
while kill -0 $gp 2>/dev/null; do
  if (( $(date +%s) - t0 > LIMIT_MIN * 60 )); then echo "=== $WIN: window limit reached $(date +%T), killing" | tee -a $R/run-$WIN.out; kill -- -$gp; sleep 3; kill -9 -- -$gp 2>/dev/null; break; fi
  sleep 10
done
wait $gp 2>/dev/null
bash $R/restore.sh $WIN > $R/restore-$WIN.out 2>&1
kill $dm 2>/dev/null; pkill -f "results/W18/deadman.sh" 2>/dev/null
echo "=== $WIN window closed $(date +%T), $(( ($(date +%s) - t0) / 60 )) min" | tee -a $R/run-$WIN.out
