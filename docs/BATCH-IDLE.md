# Sleeping between batches

`patches/0630-glm-batch-idle-doorbell.patch` adapts the CPU-store wake-up pattern discussed in
[TensorFold #132](https://github.com/ashhart/TensorFold/pull/132) to this recipe's **Batcher**.
An engine-only backport does not cover the batched `GlmEngine.follow()` early return.

When every sequence is empty, rank 1 waits on the existing rendezvous TCPStore before entering the plan
collective. Rank 0 rings the matching numbered key immediately before sharing its next plan. Notifications
survive an early arrival, and rank 1 deletes each consumed key. The RoCE wrapper exposes its NCCL base's store,
so both transports use the same mechanism. Active rounds and sampler-carried plans do not access the store.
Only an actual idle-wait timeout retries; connection loss and other store errors propagate.

No checkpoint, sampling, numerical kernel, session-cache, KV-pool or memory-margin setting changes.
The existing disk-tier compatibility hash includes the image identity: a newly built image starts a
new namespace, without deleting the old one. A restart of the same image reuses its compatible disk tier.
The patch concerns resident batched serving, not stale GPU telemetry after container shutdown or the
separate single-request engine path.

## Heat and power scope

An idle worker blocked in a GPU collective consumes power and generates heat without doing inference.
On the installation used for this check, board/ACPI temperatures repeatedly reached about 95–98°C
and one machine turned itself off. The shutdown cause remains unproven; heat preceding it is not
proof that the idle collective caused it or that this patch prevents another shutdown.

The checks below establish that both resident GPUs return to 0% utilization. They do not measure
temperature or wattage reduction against an otherwise identical unpatched run. Active prefill and
decode remain GPU workloads and can still run hot. This patch makes the empty batch sleep; it does
not change cooling, clocks, power limits or active-load thermal behavior.

## CPU regression checks

With the patched image and pytest available:

```bash
docker run --rm --network none --pids-limit 512 -e BASH_ENV= -e ENV= \
  -v "$PWD:/work:ro" --entrypoint bash <patched-image> -c '
  PYTHONPATH=/src/TensorFold/tests/cuda:/work/tests/cuda \
  python -m pytest -p no:cacheprovider -q \
    /work/tests/cuda/test_batch_idle_bell.py \
    /work/tests/cuda/test_batch_sessions_patches.py \
    /work/tests/cuda/test_batch_parallel_patches.py \
    /work/tests/test_serve_ops.py'
```

These use real TCPStore objects and the actual Batcher plan/follow methods on a CPU fixture. They cover
blocked idle waits, early and repeated notifications, consumed-key removal, active rounds, store-less
test communicators and peer-error propagation. Existing cache and batch regressions run alongside them.
GPU-only cases are skipped in this command; it is not a numerical GPU qualification.

Clearing `BASH_ENV`/`ENV` affects only this isolated test command: NVIDIA's shell initialization otherwise
calls the test's fake `nvidia-smi` recursively. Do not change the production image environment for this.

## Minimal two-node acceptance

On the production checkpoint and unchanged production knobs, check an empty resident pair, then four
overlapping completed decoding streams, then idle again. Revisit alternating independent long histories
and verify actual cached-token counts and time to first token. Stop and start the pair once and repeat
the wake-up check. Record measured values separately; configured slots or `/health` alone are not proof.

## Initial two-GB10 check (2026-10-01)

On the production uncensored checkpoint with DFlash2, four slots and unchanged memory settings:

- CPU checks above: 87 passed; 19 GPU-only cases skipped.
- Image audit: dependency versions identical, installed source matches the patched tree, and `batch.py`
  is the only changed production source file.
- Authenticated public SSE: one 256-token stream completed at 51.6 decode tok/s, TTFT 0.394 s. Four
  independent 256-token streams all completed, overlapped decoding for 10.66 s, and delivered 85.76
  aggregate tok/s over their combined streaming window; TTFT 0.507-0.946 s.
- Resident idle before and after serving: both GPUs measured 0% utilization. Initial idle container
  CPU samples were about 2% each. Under real prefill, both GPUs measured 96% utilization.
- Three independent histories around 125.5K tokens, visited A-B-A-C-B-A: cold TTFT 79.01-79.80 s;
  revisits retained 125,440-125,504 cached tokens and reached first token in 0.415-0.474 s at the origin.
- Stop/start: both GPUs measured 0% after stopping. Restart loaded in 35 s, indexed the same 26
  disk-tier entries, and completed another public single/four-stream check: single 53.7 decode tok/s,
  0.363 s TTFT; four-stream aggregate 83.40 tok/s with all four complete and 10.82 s decode overlap.
  Both resident GPUs again returned to 0% utilization afterward; no persistence-daemon reset was used.

These are bounded acceptance measurements, not a matched unpatched/patched speed comparison,
a long soak, or proof of 30 concurrent long-context requests. Longer owner testing precedes PR submission.
