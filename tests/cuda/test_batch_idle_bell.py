"""Batched idle doorbell: actual Batcher methods, real TCPStore, no checkpoint or GPU."""
from datetime import timedelta
import queue
import socket
import threading
from types import SimpleNamespace

import pytest
import torch
from torch.distributed import TCPStore, DistStoreError

from tensorfold.families.glm5_next.cuda.batch import Batcher
from tensorfold.families.glm5_next.cuda.roce import RoceComm


def ranks():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    master = TCPStore('127.0.0.1', port, 2, True, timeout=timedelta(seconds=10), wait_for_workers=False)
    worker = TCPStore('127.0.0.1', port, 2, False, timeout=timedelta(seconds=10))
    bats = []
    for rank, store in enumerate((master, worker)):
        # The production RoCE wrapper must expose the same rendezvous store as plain NCCL.
        comm = RoceComm(SimpleNamespace(rank=rank, world=2, store=store), None, 0)
        bat = object.__new__(Batcher)
        bat.g = SimpleNamespace(comm=comm)
        bat.seqs = [None] * 4
        bats.append(bat)
    return master, bats[0], bats[1]


def test_idle_blocks_then_wakes_repeatedly_without_lost_or_retained_keys():
    store, head, worker = ranks()
    initial_keys = store.num_keys()
    woke = queue.Queue()

    def follow():
        for _ in range(3):
            worker._idle_bell(wait=True)
            woke.put(worker._idle_bell_seq)

    thread = threading.Thread(target=follow, daemon=True)
    thread.start()
    with pytest.raises(queue.Empty):
        woke.get(timeout=0.15)
    head._idle_bell(wait=False)
    assert woke.get(timeout=5) == 1
    # Early notifications are durable even while the follower is still processing its previous round.
    head._idle_bell(wait=False)
    head._idle_bell(wait=False)
    assert [woke.get(timeout=5), woke.get(timeout=5)] == [2, 3]
    thread.join(5)
    assert not thread.is_alive()
    assert head._idle_bell_seq == worker._idle_bell_seq == 3
    assert store.num_keys() == initial_keys


def test_active_rounds_do_not_touch_the_store_or_advance_the_bell():
    class ForbiddenStore:
        def __getattr__(self, name):
            raise AssertionError(f'Active rounds must not use TCPStore.{name}')

    bat = object.__new__(Batcher)
    bat.g = SimpleNamespace(comm=SimpleNamespace(store=ForbiddenStore()))
    bat.seqs = [None, object(), None, None]
    bat._idle_bell(wait=False)
    bat._idle_bell(wait=True)
    assert not hasattr(bat, '_idle_bell_seq')


@pytest.mark.parametrize('comm', [None, SimpleNamespace()])
def test_storeless_test_communicators_keep_the_original_exchange(comm):
    bat = object.__new__(Batcher)
    bat.g = SimpleNamespace(comm=comm)
    bat.seqs = [None] * 4
    bat._idle_bell(wait=False)
    bat._idle_bell(wait=True)
    assert not hasattr(bat, '_idle_bell_seq')


def test_only_idle_wait_timeouts_retry_and_peer_or_other_errors_escape():
    class Store:
        def __init__(self, errors):
            self.errors = list(errors)
            self.deleted = []

        def wait(self, keys, timeout):
            if self.errors:
                raise self.errors.pop(0)

        def delete_key(self, key):
            self.deleted.append(key)

    bat = object.__new__(Batcher)
    bat.seqs = [None] * 4
    store = Store([DistStoreError('wait timeout after 3600000ms, keys: /idle')])
    bat.g = SimpleNamespace(comm=SimpleNamespace(store=store))
    bat._idle_bell(wait=True)
    assert store.deleted == ['tf_glm_batch_idle_1']
    for error in (RuntimeError('Connection reset by peer'), DistStoreError('socket timeout')):
        bat.g.comm.store = Store([error])
        with pytest.raises(type(error), match=str(error)):
            bat._idle_bell(wait=True)
        assert bat._idle_bell_seq == 1


def test_real_plan_wakes_real_follow_before_its_first_collective(monkeypatch):
    # Reuse the recipe's existing CPU batch fixture; test the real _plan and follow call sites, not a copy.
    from test_batch_sessions_patches import _fake_batcher, _job

    store, minimal_head, minimal_worker = ranks()
    head = _fake_batcher(monkeypatch, n=4, rows=64, fast=False, piece=256, budget_pages=20)
    worker = _fake_batcher(monkeypatch, n=4, rows=64, fast=False, piece=256, budget_pages=20, rank=1)
    head.g.comm, worker.g.comm = minimal_head.g.comm, minimal_worker.g.comm
    messages = queue.Queue()
    received = threading.Event()
    errors = []

    def share(values):
        messages.put(list(values))
        return list(values)

    def read(values):
        assert values is None
        received.set()
        return messages.get(timeout=5)

    class EndTest(Exception):
        pass

    def execute(cancels, jobs, pieces):
        assert not cancels and len(jobs) == 1
        assert jobs[0][2].prompt == [1, 2, 3, 4]
        raise EndTest

    head.g._share, worker.g._share = share, read
    worker._execute = execute

    def follow():
        try:
            worker.follow()
        except EndTest:
            pass
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=follow, daemon=True)
    thread.start()
    assert not received.wait(0.15)  # No GPU collective can be entered while the batch is empty.
    head.queue.append(_job([1, 2, 3, 4], 1, False, 64))
    plan = head._plan()
    assert plan is not None
    thread.join(5)
    assert not thread.is_alive() and not errors
    assert received.is_set()
    assert head._idle_bell_seq == worker._idle_bell_seq == 1
