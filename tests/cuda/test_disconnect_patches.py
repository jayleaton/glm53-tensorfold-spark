"""patches/0600: a client that leaves cancels its request on both ranks in lockstep (upstream 24afe5e on our batcher).

Checked (torch on the CPU; no GPU): patches/0180's hostile fake model under the real ``Batcher._plan`` / ``_execute`` /
``_finish`` / ``follow`` (test_decode_overlap_patches' rank-0 message stream replayed through ``follow`` on a batcher
playing rank 1), with every HTTP thread waiting in the real ``Batcher._collect`` and a silent (non-streamed) callback
whose ``cancelled`` check turns true while the request is

- **queued** (all slots busy): it is dropped by the next plan, never admitted, rank 1 never hears of it;
- **prefilling** (a 900-token prompt in 256-token pieces): the plan cancels the slot before its first token;
- **decoding**: the plan cancels it, with and without 0370's plan riders (``GLM53_TF_DECODE_OVERLAP``: prod runs 1);
- or when its callback **raises** (the job is cancelled, the exception reaches the caller after the end marker).

In every case rank 1 ends the same requests the same way (slot, sha, keeps, cancelled) through the same rounds with
identical slot states, and the other requests' replies equal a fresh prefill + serial decoding.

Run: PYTHONPATH=<tree>/src:<tree>/tests/cuda:tests/cuda pytest -q tests/cuda/test_disconnect_patches.py
"""

from __future__ import annotations

import threading
import time

import pytest

try:
    import torch
except ImportError:          # pragma: no cover
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch")


def _wait(cond, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.002)
    return False


def _run(monkeypatch, part: str, phase: str, *, raise_in_callback: bool = False):
    import numpy as np
    import test_batch_sessions_patches as tb
    import test_decode_overlap_patches as to

    from tensorfold.families.glm5_next.cuda import batch

    if not hasattr(batch, "_poll_s"):
        pytest.skip("patches/0600 not applied")
    rng = np.random.default_rng(17)
    sizes = {"queued": (300, 200, 250, 280), "prefilling": (300, 200, 900), "decoding": (300, 200, 250)}[phase]
    target = {"queued": 3, "prefilling": 2, "decoding": 1}[phase]
    prompts = [[int(t) for t in rng.integers(0, 1000, size=s)] for s in sizes]
    r0 = to._overlap_batcher(monkeypatch, to.PARTS[part])
    r1 = to._overlap_batcher(monkeypatch, to.PARTS[part], rank=1)
    wire = to._Wire(monkeypatch)
    wire.attach(r0, r1)
    r0.poll_s = 0.005
    jobs = [tb._job(p, 40, False, 64) for p in prompts]
    flags = [False] * len(jobs)
    results: list = [None] * len(jobs)

    def listener(i: int):
        def on_tokens(new):
            if raise_in_callback and i == target and flags[i]:
                raise RuntimeError("the tool-call parser failed")
            return False                        # a non-streamed reply: nothing is written, no write can fail

        on_tokens.cancelled = (lambda: False) if raise_in_callback else (lambda: flags[i])
        return on_tokens

    def collect(i: int):
        try:
            results[i] = r0._collect(jobs[i], listener(i))
        except BaseException as exc:            # noqa: BLE001
            results[i] = exc

    threads = [threading.Thread(target=collect, args=(i,), daemon=True) for i in range(len(jobs))]
    for t in threads:
        t.start()
    r0.queue.extend(jobs)
    flipped = {"round": None, "state": None}
    rounds = 0
    while r0.queue or any(s is not None for s in r0.seqs) or r0._ahead is not None:
        plan = r0._take_ahead() or r0._plan()
        r0._execute(*plan)
        rounds += 1
        assert rounds < 5000
        if flipped["round"] is not None:
            continue
        job = jobs[target]
        seq = next((s for s in r0.seqs if s is not None and s.job is job), None)
        if phase == "queued":
            ready = job in r0.queue and rounds >= 2
        elif phase == "prefilling":
            ready = seq is not None and seq.stepper is None and rounds >= 1
        else:
            ready = seq is not None and seq.stepper is not None and len(seq.stepper.out) >= 5
        if ready:
            flipped.update(round=rounds, state="queued" if seq is None else
                           ("prefilling" if seq.stepper is None else "decoding"))
            if raise_in_callback:
                flags[target] = True            # the next tokens it receives make the callback raise (a later round)
            else:
                flags[target] = True            # the client left: the HTTP thread's poll sees it
                assert _wait(lambda: job.cancel), "the poll never cancelled the job"
    for t in threads:
        t.join(10)
    assert flipped["state"] == phase, flipped
    return r0, r1, wire, jobs, prompts, results, target


@needs_torch
@pytest.mark.parametrize("part", ["off", "all"])
@pytest.mark.parametrize("phase", ["queued", "prefilling", "decoding"])
def test_client_gone_cancels_on_both_ranks(monkeypatch, part, phase):
    import test_batch_sessions_patches as tb
    import test_decode_overlap_patches as to

    r0, r1, wire, jobs, prompts, results, target = _run(monkeypatch, part, phase)
    assert jobs[target].stats.get("cancelled") is True
    got = results[target]
    assert isinstance(got, list)
    want = tb._reference(prompts[target], 40, 64, False)[0]
    if phase in ("queued", "prefilling"):
        assert got == []                        # not one token was decoded for it
    else:
        assert 5 <= len(got) < 40 and got == want[:len(got)]
    for i, p in enumerate(prompts):
        if i != target:
            assert results[i] == tb._reference(p, 40, 64, False)[0]
    wire.replaying = True
    with pytest.raises(to._Wire.Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log]
    assert [t[1:] for t in r1.trace] == [t[1:] for t in r0.trace]
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]
    ended = [d for d in r0.log if d["cancelled"]]
    if phase == "queued":
        assert ended == [] and len(r0.log) == len(prompts) - 1      # never admitted, on either rank
    else:
        assert len(ended) == 1


@needs_torch
@pytest.mark.parametrize("part", ["off", "all"])
def test_callback_failure_cancels_the_slot_on_both_ranks(monkeypatch, part):
    import test_batch_sessions_patches as tb
    import test_decode_overlap_patches as to

    r0, r1, wire, jobs, prompts, results, target = _run(monkeypatch, part, "decoding", raise_in_callback=True)
    assert isinstance(results[target], RuntimeError) and jobs[target].stats.get("cancelled") is True
    for i, p in enumerate(prompts):
        if i != target:
            assert results[i] == tb._reference(p, 40, 64, False)[0]
    wire.replaying = True
    with pytest.raises(to._Wire.Done):
        r1.follow()
    key = lambda d: (d["slot"], d["sha256"], tuple(d["keeps"]), d["cancelled"])      # noqa: E731
    assert [key(d) for d in r1.log] == [key(d) for d in r0.log]
    assert [st.state() for st in r0.states] == [st.state() for st in r1.states]
