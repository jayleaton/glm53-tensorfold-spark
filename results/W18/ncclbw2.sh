#!/usr/bin/env bash
# W18 (K9): NCCL bandwidth sweep with the model stopped (inside a window, after window-start). ncclbw.sh [IMAGE]
# One pair of containers (b9 image) per setting; rank 1 on the worker node over ssh. Output: results/W18/ncclbw.jsonl + .log
cd $HOME/glm53-tensorfold-spark
R=results/W18; IMG=${1:-glm53-tensorfold:b9}; W=$WORKER_SSH
ssh -o BatchMode=yes $W "mkdir -p /root/w18" && scp -q $R/ncclbw.py $W:/root/w18/ncclbw.py
base="NCCL_SOCKET_IFNAME=enp1s0f1np1 GLOO_SOCKET_IFNAME=enp1s0f1np1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
one() { # one LABEL K=V ...
  local label=$1; shift; local env="$base $* MASTER=$HEAD_IP PORT=$((29600 + RANDOM % 300))"
  local e0="" e1=""; for kv in $env; do e0="$e0 -e $kv"; done
  local run="docker run --rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --entrypoint python"
  ssh -o BatchMode=yes $W "timeout 180 $run $e0 -e RANK=1 -v /root/w18:/w $IMG /w/ncclbw.py $label /w/out.jsonl" > $R/ncclbw-$label-r1.log 2>&1 &
  timeout 180 $run $e0 -e RANK=0 -v $PWD/$R:/w $IMG /w/ncclbw.py $label /w/ncclbw.jsonl > $R/ncclbw-$label-r0.log 2>&1
  wait
  grep -E "^$label " $R/ncclbw-$label-r0.log || echo "$label FAILED: $(tail -2 $R/ncclbw-$label-r0.log | cut -c1-200)"
  grep -hE "NET/IB : Using|Channel 0[0-9]/|via NET" $R/ncclbw-$label-r0.log | head -4 | cut -c1-200
}
HCA1=rocep1s0f1; HCA2=rocep1s0f1,roceP2p1s0f1
# W18 window N: the channel counts on one function (attribution: channels vs the second function), a repeat of the
# best two-function setting, and the Ring vs Tree health check on all-reduce (NCCL has no Tree all-gather)
one n-two-ch4     NCCL_IB_HCA=$HCA2 NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
grep -q '"label": "n-two-ch4"' $R/ncclbw.jsonl 2>/dev/null || { echo "first NCCL run failed: sweep aborted"; exit 1; }
one n-one-ch2     NCCL_IB_HCA=$HCA1 NCCL_MIN_NCHANNELS=2 NCCL_MAX_NCHANNELS=2
one n-one-ch4     NCCL_IB_HCA=$HCA1 NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
one n-one-ch8     NCCL_IB_HCA=$HCA1 NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8
one n-one-ar-ring AR_ONLY=1 NCCL_IB_HCA=$HCA1 NCCL_ALGO=Ring
one n-one-ar-tree AR_ONLY=1 NCCL_IB_HCA=$HCA1 NCCL_ALGO=Tree
one n-two-ar-ring AR_ONLY=1 NCCL_IB_HCA=$HCA2 NCCL_ALGO=Ring
one n-two-ar-tree AR_ONLY=1 NCCL_IB_HCA=$HCA2 NCCL_ALGO=Tree
one n-two-ch4-b   NCCL_IB_HCA=$HCA2 NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
