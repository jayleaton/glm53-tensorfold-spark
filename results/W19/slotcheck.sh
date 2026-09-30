#!/usr/bin/env bash
# W17: rank 0's request slot count (boot line `context: ... N request slot(s)`); exit 1 if under 4
n=$(docker logs glm53-tf-r0 2>&1 | grep -oE '[0-9]+ request slot\(s\)' | tail -1 | grep -oE '^[0-9]+')
echo "request slots (rank 0): ${n:-none}"
[[ "${n:-0}" -ge 4 ]]
