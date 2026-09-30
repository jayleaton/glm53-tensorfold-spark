#!/usr/bin/env bash
# W18 (from W17): refresh the test-window lease every 5 min while it exists, for at most MAXMIN minutes (default 240);
# deleting the lease ends it. lease.sh [MAXMIN]
L=$HOME/.test-window-lease
n=$(( ${1:-240} / 5 ))
for i in $(seq 1 $n); do [ -f "$L" ] || exit 0; touch "$L"; sleep 300; done
