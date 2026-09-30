#!/usr/bin/env bash
# W18 (from W17): full gates on a candidate: gates.sh NAME [K=V ...] = load, ab.sh set (exact 10/10, batchexact 4/4, transcripts,
# reply sha, prefill once, glmbench 1 stream, 4 streams x6, slots), ab.py 24.5k / 98k again (prefill), N1 long exactness,
# 4 x 250k stress (MemAvailable >= 8 GiB both nodes), MMLU-200 >= 87%, exact / batchexact again, needle ~314k alone,
# OOM counts, /health.
cd $HOME/glm53-tensorfold-spark
B=http://127.0.0.1:8001; export BASE=$B; M=GLM-5.3-Flash-EXL3; R=results/W18
N=$1; shift
mark() { echo "$1 $(date +%T)" >> $R/mem-marks-$N.txt; }
rmmark() { for h in local $WORKER_SSH; do c="docker run --rm -v glm53-tf-cache:/cache --entrypoint bash glm53-tensorfold:b9 -c 'cat /cache/roce-failed 2>/dev/null; rm -f /cache/roce-failed'"; if [ $h = local ]; then eval "$c"; else ssh -o BatchMode=yes $h "$c"; fi; done; }
oom() { echo "$(dmesg -T 2>/dev/null | grep -ciE 'out of memory|oom-kill') $(ssh -o BatchMode=yes $WORKER_SSH "dmesg -T | grep -ciE 'out of memory|oom-kill'") nvrm-nomem $(dmesg -T 2>/dev/null | grep -c NV_ERR_NO_MEMORY) $(ssh -o BatchMode=yes $WORKER_SSH "dmesg -T | grep -c NV_ERR_NO_MEMORY")"; }
rmmark > $R/roce-marks-before-$N.txt 2>&1
oom > $R/oom-before-$N.txt
bash $R/load.sh $N "$@" || { echo "LOAD $N FAILED"; exit 1; }
bash $R/mem.sh > $R/mem-$N.log 2>&1 & echo $! > $R/mem.pid
bash $R/meminfo.sh > $R/meminfo-$N.log 2>&1 & echo $! > $R/meminfo.pid
bash $R/memfast.sh local > $R/memfast-$N-r0.log 2>&1 & echo $! > $R/memfast0.pid
bash $R/memfast.sh worker > $R/memfast-$N-r1.log 2>&1 & echo $! > $R/memfast1.pid
# W18 (from W17): optional PRE commands (the vision checks) before W12's sequence, so the tower's caches are warm for the stress
if [ -n "${PRE:-}" ]; then mark pre-start; bash -c "$PRE"; mark pre-end; fi
mark ab-start
bash $R/ab.sh $N
mark ab2-start; python3 $R/ab-p.py $R/ab-$N-2.json 24500,98000 '{"prod":{}}' > $R/ab-$N-2.log 2>&1; cut -c1-110,200- $R/ab-$N-2.log | tail -2
mark n1-start; python3 bench/longexact.py --base $B --corpus $R/n1-corpus.txt --out $R/n1-$N.json > $R/n1-$N.log 2>&1; grep '^N1' $R/n1-$N.log
mark stress-start; python3 bench/multiturn.py --base $B --model $M --modes stress --stress-target 250000 --stress-step 100000 --stress-final 32000 \
  --long-tokens 512 --mem-hosts local,$WORKER_SSH --mem-log $R/stress-mem-$N.log --out $R/stress-$N.json > $R/stress-$N.log 2>&1
tail -3 $R/stress-$N.log; curl -s $B/health > $R/health-$N-stress.json
mark mmlu-start; python3 bench/quality.py --base $B --model $M --label $N --out $R/quality-$N.json > $R/quality-$N.log 2>&1
tail -3 $R/quality-$N.log
python3 bench/glmbench.py --base $B --model $M --suites exact --out $R/exact-${N}2.json > $R/exact-${N}2.log 2>&1
python3 bench/multiturn.py --base $B --model $M --modes batchexact --out $R/batchexact-${N}2.json > $R/batchexact-${N}2.log 2>&1
echo "exact again: $(tail -1 $R/exact-${N}2.log | grep -o true | wc -l)/10"; grep -E 'batched == alone' $R/batchexact-${N}2.log
mark needle-start; python3 $R/needle-p.py $R/needle-$N.json 350000 0.4 > $R/needle-$N.log 2>&1
cut -c1-300 $R/needle-$N.log | tail -4
mark needle-end
# W18: two more ab.py pairs after the needle (W17 B9's slow pair came here): 4 prefill pairs a load in all
for k in 3 4; do mark ab$k-start; python3 $R/ab-p.py $R/ab-$N-$k.json 24500,98000 '{"prod":{}}' > $R/ab-$N-$k.log 2>&1; cut -c1-110,200- $R/ab-$N-$k.log | tail -2; done
mark end; curl -s $B/health > $R/health-$N-end.json; cat $R/health-$N-end.json; echo
docker logs glm53-tf-r0 2>&1 | grep -iE "roce.*(timed out|failed|fallback)" | head -3
echo "oom / nvrm lines after: $(oom) (before: $(cat $R/oom-before-$N.txt))" | tee $R/oom-$N.txt
for f in mem meminfo memfast0 memfast1; do kill $(cat $R/$f.pid) 2>/dev/null; rm -f $R/$f.pid; done
docker logs glm53-tf-r0 > $R/run-$N-r0.log 2>&1; ssh -o BatchMode=yes $WORKER_SSH docker logs glm53-tf-r1 > $R/run-$N-r1.log 2>&1
grep -ciE 'traceback|error' $R/run-$N-r0.log | sed 's/^/r0 error lines: /'
awk 'NR>1{if(a==""||$2<a)a=$2; if(b==""||$3<b)b=$3} END{print "MemAvailable min over the gates: " a " / " b " GiB"}' $R/mem-$N.log
python3 $R/memsum.py $N | tee $R/memsum-$N.txt
# the request log lines of this load (slot, pieces, prefill_s, queue_s of every request, ours only: :8001)
python3 - $N <<'PY2' > $R/reqlog-$N.jsonl
import json, sys, time, os
t0 = os.path.getmtime(f"results/W18/start-{sys.argv[1]}.log") - 600
for l in open("$HOME/.cache/glm53-tf/sessions/requests.jsonl"):
    try:
        d = json.loads(l)
    except ValueError:
        continue
    if d.get("start", 0) >= t0:
        print(json.dumps({k: d.get(k) for k in ("start", "ts", "prompt", "cached", "prefill_s", "queue_s", "first_s", "pieces", "slot", "marks", "decode_tokens", "decode_s", "tools", "messages")}))
PY2
echo "$N gates done $(date +%T)"
