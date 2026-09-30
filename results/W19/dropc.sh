#!/usr/bin/env bash
# W17: drop the page cache on both nodes (W16 finding: page cache at a start -> fewer batch slots); prints MemFree after
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' || echo "drop_caches refused on the head node"
ssh -o BatchMode=yes $WORKER_SSH "sync; echo 3 > /proc/sys/vm/drop_caches" || echo "drop_caches failed on the worker node"
echo "drop_caches $(date +%T): MemFree head $(awk '/^MemFree/{printf "%.1f",$2/1048576}' /proc/meminfo) / worker $(ssh -o BatchMode=yes $WORKER_SSH "awk '/^MemFree/{printf \"%.1f\",\$2/1048576}' /proc/meminfo") GiB"
