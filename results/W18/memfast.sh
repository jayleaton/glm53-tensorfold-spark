#!/usr/bin/env bash
# W18: MemFree / MemAvailable (GiB) every 0.5 s on one node, sampled locally (no ssh per sample): memfast.sh [local|worker]
# lines: epoch.ms free avail. A trim (patches/0550 GLM53_TF_ALLOC_TRIM_GB) shows as a step up in MemFree of >= 1 GiB
# outside a graph capture; used to tell trims apart from scratch growth during the ab.py prefills.
loop='while :; do awk -v t="$(date +%s.%N | cut -c1-14)" '"'"'/^MemFree:/{f=$2} /^MemAvailable:/{a=$2} END{printf "%s %.2f %.2f\n", t, f/1048576, a/1048576}'"'"' /proc/meminfo; sleep 0.5; done'
if [[ "${1:-local}" == worker ]]; then exec ssh -o BatchMode=yes $WORKER_SSH "$loop"; else exec bash -c "$loop"; fi
