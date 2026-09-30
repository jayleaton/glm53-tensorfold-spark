"""patches/0590: the GATE line of ``tests/cuda/bench_experts.py --fat2 --contend`` (host only: torch CPU + numpy).

PASS needs fat2's bits "same" and its contended time <= 0.75x fat s3's at BOTH 2,048 and 4,096 rows; the line also
carries the makespan and isolated ratios and the DRAM floor. Without --contend, or without both row counts, it says what
is missing instead of passing.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cuda"))
pytest.importorskip("torch")

import bench_experts as B  # noqa: E402


def _cell(fat, fat2, same=True, span=(20.0, 15.0)):
    res = {"fat s3": (fat * 0.6, fat * 0.4), "fat2": (fat2 * 0.6, fat2 * 0.4)}
    under = {"fat s3": fat * 1.1, "fat2": fat2 * 1.1}
    return res, {"fat s3": True, "fat2": same}, under, {"fat s3": span[0], "fat2": span[1]}


def test_gate_pass_and_fail():
    ok = {2048: _cell(13.0, 9.5), 4096: _cell(16.0, 11.9)}
    line = B.gate_line(ok, True)
    assert line.startswith("GATE 0590: PASS") and "2048: contended 0.731x" in line and "makespan 0.750x" in line
    slow = {2048: _cell(13.0, 10.2), 4096: _cell(16.0, 11.9)}
    assert B.gate_line(slow, True).startswith("GATE 0590: FAIL")
    bits = {2048: _cell(13.0, 9.0, same=False), 4096: _cell(16.0, 11.0)}
    line = B.gate_line(bits, True)
    assert line.startswith("GATE 0590: FAIL") and "BITS DIFFER" in line


def test_gate_needs_contend_and_both_rows():
    assert "needs --contend" in B.gate_line({2048: _cell(13, 9), 4096: _cell(16, 11)}, False)
    assert "needs rows [4096]" in B.gate_line({2048: _cell(13, 9)}, True)


def test_floor_is_below_the_gate_only_barely_at_2048():
    """The layer's DRAM floor at the measured 235 GB/s against W8's fat s3 (13.18 / 15.96 ms isolated): 2,048 rows
    ~0.74x, 4,096 rows ~0.73x -- the 0.75x gate asks for ~98% of the DRAM roof with fat's dataflow."""

    assert 0.72 < B._floor_ms(2048, 235e9) / 13.18 < 0.75
    assert 0.71 < B._floor_ms(4096, 235e9) / 15.96 < 0.75
