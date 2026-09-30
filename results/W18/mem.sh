#!/usr/bin/env bash
# sample MemAvailable (GiB) on both nodes every 2 s: time r0 r1
while true; do
  a=$(awk '/MemAvailable/{printf "%.2f",$2/1048576}' /proc/meminfo)
  b=$(ssh -o BatchMode=yes $WORKER_SSH "awk '/MemAvailable/{printf \"%.2f\",\$2/1048576}' /proc/meminfo")
  echo "$(date +%T) $a $b"; sleep 2
done
