#!/usr/bin/env bash
# W19 step 4 re-run: the culprit off. COMBINED2 = COMBINED without GLM53_TF_DEC_QMM_MAXMB (0570: q4_kernel slower in
# situ than the _qmm + _reduce it replaced). Only the gates 0570 can move: bits (exact, batchexact, transcripts, reply
# sha, 13 glmbench hashes) and decode (glmbench 1 stream x3, 4 streams x6, lone slots) via ab.sh (+ one prefill pair),
# then the 0530 thread placement during a streamed reply. Prod stays down (back to back).
cd $HOME/glm53-tensorfold-spark; R=results/W19
export CFG=$R/combined.env
echo "=== COMBINED2 $(date +%T)"
bash $R/load.sh COMBINED2 GLM53_TF_SELECT_SCRATCH=grow GLM53_TF_ALLOC_TRIM_GB=0 GLM53_TF_ADMIT_MEM=free GLM53_TF_DEC_EXPERT_LOADS=1 \
  GLM53_TF_DEC_EXPERT_LOADS_CFG=nc,8,1 GLM53_TF_CPU_PIN=http GLM53_TF_GRAMMAR=1 NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 || { echo "LOAD FAILED"; exit 1; }
bash $R/ab.sh COMBINED2
# 0530: thread placement while a streamed reply is being written
python3 - <<'PY' &
import json, urllib.request
b = {"model": "GLM-5.3-Flash-EXL3", "stream": True, "max_tokens": 600, "temperature": 0,
     "messages": [{"role": "user", "content": "Write a detailed plain-prose explanation of the causes of ocean tides."}],
     "chat_template_kwargs": {"enable_thinking": False}}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=json.dumps(b).encode(),
                           headers={"Content-Type": "application/json"}), timeout=300)
for _ in r:
    pass
PY
sleep 4
docker exec glm53-tf-r0 sh -c 'for t in /proc/1/task/*; do echo "$(cat $t/comm) | allowed $(grep Cpus_allowed_list $t/status | cut -f2) | on $(awk "{print \$39}" $t/stat)"; done | sort | uniq -c | sort -rn' > $R/threads-COMBINED2-stream.txt
docker exec glm53-tf-r0 sh -c 'ps -L -o tid,psr,pcpu,comm -p 1' > $R/threads-COMBINED2-ps.txt 2>&1
wait
echo "threads during a streamed reply:"; grep -E "tf-http" $R/threads-COMBINED2-stream.txt; grep -c . $R/threads-COMBINED2-stream.txt
docker logs glm53-tf-r0 > $R/run-COMBINED2-r0.log 2>&1
echo "=== COMBINED2 done $(date +%T)"
