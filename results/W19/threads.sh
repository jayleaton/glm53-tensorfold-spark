#!/usr/bin/env bash
# W19: the engine container's threads (name | allowed cpus | last cpu) on rank 0 while 4 streams run (0530 CPU_PIN=http
# check: tf-http threads on the little cores 0-4,10-14, the rest unpinned). threads.sh NAME
cd $HOME/glm53-tensorfold-spark; R=results/W19
python3 bench/multiturn.py --base http://127.0.0.1:8001 --model GLM-5.3-Flash-EXL3 --modes concurrent --streams 4 --reps 1 --long-tokens 256 --out /tmp/w19-thr.json > /dev/null 2>&1 &
p=$!; sleep 12
docker exec glm53-tf-r0 sh -c 'for t in /proc/[0-9]*/task/*; do echo "$(cat $t/comm) | allowed $(grep Cpus_allowed_list $t/status | cut -f2) | on $(awk "{print \$39}" $t/stat)"; done | sort | uniq -c | sort -rn' > $R/threads-$1-r0.txt 2>&1
docker exec glm53-tf-r0 sh -c 'ps -L -o tid,psr,pcpu,comm -p 1 2>/dev/null | head -60' > $R/threads-$1-ps.txt 2>&1
wait $p
echo "threads $1 (rank 0, during 4 streams):"; grep -E "tf-http|tf-serve|glm-grammar" $R/threads-$1-r0.txt | head -8; head -4 $R/threads-$1-r0.txt
docker logs glm53-tf-r0 2>&1 | grep -iE "cpu pin|cpupin" | head -3 | cut -c1-200
