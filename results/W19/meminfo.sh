#!/usr/bin/env bash
# W18 (from W17): /proc/meminfo breakdown on both nodes every 2 s (GiB): time, then per node MemAvailable MemFree Cached Dirty
# Writeback AnonPages Shmem (rank 0 = head first, then worker). Attributes the needle's MemAvailable dip.
f='/^(MemAvailable|MemFree|Cached|Dirty|Writeback|AnonPages|Shmem):/{v[$1]=$2/1048576} END{printf "%.2f %.2f %.2f %.2f %.2f %.2f %.2f", v["MemAvailable:"], v["MemFree:"], v["Cached:"], v["Dirty:"], v["Writeback:"], v["AnonPages:"], v["Shmem:"]}'
echo "time | r0: avail free cached dirty wb anon shmem | r1: same"
while true; do
  a=$(awk "$f" /proc/meminfo)
  b=$(ssh -o BatchMode=yes $WORKER_SSH "awk '$f' /proc/meminfo")
  echo "$(date +%T) | $a | $b"; sleep 2
done
