#!/usr/bin/env python3
"""W18 (KINDLING-AUDIT K9): NCCL all-gather / all-reduce bandwidth between the two Sparks, model stopped.
Run one process a node (rank from RANK, master MASTER:PORT). ncclbw.py LABEL OUT.jsonl
Sizes = the all-gather OUTPUT bytes (1..32 MiB; the prefill's are ~4 MiB); per size 10 warm-up + 50 timed ops
(CUDA events), busbw = algbw x (n - 1) / n (nccl-tests' definition). NCCL settings come from the environment
(NCCL_IB_HCA, NCCL_PROTO, NCCL_ALGO, NCCL_MIN/MAX_NCHANNELS). Rank 0 appends one JSON line."""
import json, os, sys, time
import torch, torch.distributed as dist

label, out = sys.argv[1], sys.argv[2]
rank = int(os.environ["RANK"]); world = 2
torch.cuda.set_device(0)
dist.init_process_group("nccl", init_method=f"tcp://{os.environ['MASTER']}:{os.environ.get('PORT', '29600')}",
                        rank=rank, world_size=world)
res = {"label": label, "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}, "ag": {}, "ar": {}}
AR_ONLY = os.environ.get("AR_ONLY") == "1"      # Ring vs Tree health check: all-reduce only (no Tree all-gather)
for mib in (1, 2, 4, 8, 16, 32):
    if AR_ONLY and mib not in (4, 16):
        continue
    n = mib * 2**20 // 2                        # bf16 elements of the output
    outt = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    inp = torch.randn(n // world, dtype=torch.bfloat16, device="cuda")
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(0 if AR_ONLY else 10):
        dist.all_gather_into_tensor(outt, inp)
    torch.cuda.synchronize()
    s.record()
    for _ in range(0 if AR_ONLY else 50):
        dist.all_gather_into_tensor(outt, inp)
    e.record(); torch.cuda.synchronize()
    t = max(s.elapsed_time(e), 1e-3) / 50 / 1e3
    if not AR_ONLY:
      res["ag"][mib] = {"us": round(t * 1e6, 1), "algbw_GBs": round(mib * 2**20 / t / 1e9, 2),
                      "busbw_GBs": round(mib * 2**20 / t / 1e9 * (world - 1) / world, 2)}
    if mib in (4, 16):
        x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize(); s.record()
        for _ in range(20):
            dist.all_reduce(x)
        e.record(); torch.cuda.synchronize()
        t = s.elapsed_time(e) / 20 / 1e3
        res["ar"][mib] = {"us": round(t * 1e6, 1), "busbw_GBs": round(mib * 2**20 / t / 1e9 * 2 * (world - 1) / world, 2)}
if rank == 0:
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(label, "ag 4 MiB", res["ag"].get(4), "| 16 MiB", res["ag"].get(16), "| ar 4 MiB", res["ar"][4], "| ar 16 MiB", res["ar"][16], flush=True)
dist.destroy_process_group()
