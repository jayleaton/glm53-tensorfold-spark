#!/usr/bin/env python3
"""GB10 slow-state / clock watch for the two Sparks: catch a degraded node before it silently skews production or a
benchmark. Standard library only; runs on the head (rank 0) and reaches the worker over ssh.

What goes wrong on GB10 (docs/OPS-GPUWATCH.md has the evidence and the playbook):
  clamp       the SM clock pinned at 507-890 MHz and ~12-14 W under full load, with NO clock event reason set;
              survives warm reboots, clears only after a full power drain (the head node, 2026-09-11: 507 MHz, 23.8 vs
              94.8 TFLOP/s on the worker). Visible here: clock under the floor / power under the clamp line while
              the GPU is busy, and an idle clock far below the other node's.
  slow state  a hidden state that drops memory bandwidth from ~230 to 66-80 GB/s for 7-32 s at a time; nothing in
              nvidia-smi changes (decode step 63 -> 94 ms). Visible here only through the server's own timing:
              the decode ms a round from /metrics (patches/0150) against a rolling baseline.
  lockstep    tensor parallel runs at the slower node's pace, so one bad node degrades the pair.

Signals, per node every --interval s (default 5):
  nvidia-smi --query-gpu: SM / graphics / memory clock, power, temperature, utilization, pstate, the active clock
  event reasons bitmask and the reasons' cumulative microsecond counters (the SW power cap share of each interval);
  the head's /metrics: tensorfold_decode_seconds_total / tensorfold_decode_rounds_total -> ms a decode round over
  each window of >= --step-min-rounds rounds, compared with the median of past windows at the same concurrency.
  Passive first: nothing here launches GPU work. The optional bandwidth probe (--probe, off by default) is a
  ~6-20 ms device-to-device copy in its own process at the lowest stream priority, at most once a minute a node,
  and only while the node is idle (--probe idle) or right after a step-time regression to say which node is slow
  (--probe on-alert). Why it is not the default: /metrics already times every real decode round for free, and a
  second CUDA context shares the GPU by time-slicing (stream priority does not preempt another process), so a probe
  under load both disturbs the server and measures the shared bandwidth; it is an attribution tool, not a monitor.

Output: one CSV row a node a sample, an atomically replaced state file (gpuwatch.json: last samples, active
conditions, recent cleared ones, step baselines) that ``check`` and ``serve.sh preflight`` read, and alert lines
("ALERT [crit] ..." / "CLEARED ...") on stdout; each alert also goes to GPUWATCH_WEBHOOK (JSON POST) and to the
command in WATCH_ALERT / --alert-cmd (message as $1), when set.

    scripts/gpuwatch.py watch [--interval 5] [--count N]      # the loop (systemd: scripts/systemd/glm53-gpuwatch.*)
    scripts/gpuwatch.py check [--max-age 60] [--strict]       # exit 0 ok, 1 warning, 2 degraded, 3 cannot tell
    scripts/gpuwatch.py status                                # print the state file
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

# (our name, nvidia-smi field). Counters: cumulative microseconds a reason held the clocks down since boot.
FIELDS: list[tuple[str, str]] = [
    ("sm_mhz", "clocks.sm"),
    ("gr_mhz", "clocks.gr"),
    ("mem_mhz", "clocks.mem"),
    ("max_sm_mhz", "clocks.max.sm"),
    ("app_gr_mhz", "clocks.applications.gr"),
    ("power_w", "power.draw"),
    ("temp_c", "temperature.gpu"),
    ("util_pct", "utilization.gpu"),
    ("pstate", "pstate"),
    ("reasons", "clocks_event_reasons.active"),
    ("ctr_sw_power_cap_us", "clocks_event_reasons_counters.sw_power_cap"),
    ("ctr_sw_thermal_us", "clocks_event_reasons_counters.sw_thermal_slowdown"),
    ("ctr_hw_thermal_us", "clocks_event_reasons_counters.hw_thermal_slowdown"),
    ("ctr_hw_power_brake_us", "clocks_event_reasons_counters.hw_power_brake_slowdown"),
]
# older drivers: no counters, the pre-rename reasons field
FALLBACK_FIELDS: list[tuple[str, str]] = [(k, f) for k, f in FIELDS if not k.startswith("ctr_")]
FALLBACK_FIELDS = [(k, "clocks_throttle_reasons.active" if k == "reasons" else f) for k, f in FALLBACK_FIELDS]

# nvmlClocksEventReasons bits
REASON_BITS: dict[int, str] = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}
THROTTLE_BAD = {"hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown"}
RANK = {"ok": 0, "warn": 1, "crit": 2}


def log(msg: str) -> None:
    print(f"[gpuwatch] {time.strftime('%F %T')} {msg}", flush=True)


# -- parsing -----------------------------------------------------------------------------------------------------

def _value(raw: str) -> Any:
    s = raw.strip()
    if not s or s.strip("[]").lower() in ("n/a", "not supported", "unknown error", "not found", "insufficient permissions"):
        return None
    if s.lower().startswith("0x"):
        try:
            return int(s, 16)
        except ValueError:
            return s
    try:
        return float(s)
    except ValueError:
        return s


def parse_smi(text: str, fields: list[tuple[str, str]] = FIELDS) -> dict[str, Any]:
    """The first GPU's line of ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` -> {our name: value}.
    Raises ValueError on output that is not one value a field (an error message, an empty reply)."""

    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p for p in line.split(",")]
        if len(parts) != len(fields):
            raise ValueError(f"expected {len(fields)} fields, got {len(parts)}: {line.strip()[:200]}")
        return {name: _value(p) for (name, _), p in zip(fields, parts)}
    raise ValueError("no output")


def reason_names(mask: Any) -> list[str]:
    if not isinstance(mask, int):
        return []
    return [name for bit, name in REASON_BITS.items() if mask & bit]


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text -> {metric name: value summed over label sets}."""

    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_part, _, value = line.rpartition(" ")
        name = name_part.split("{", 1)[0].strip()
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


# -- nodes -------------------------------------------------------------------------------------------------------

# ssh binds the ControlPath as a Unix socket, "<dir>/ssh-" + %C (40 hex) + ".<16 random>" while the master starts, and
# that has to fit sun_path (107 usable bytes on Linux). Under a long home directory the default state dir is too long
# and ssh refuses the master ("too long for Unix domain socket"), so the worker is never sampled.
CTL_SUFFIX_LEN = len("/ssh-") + 40 + 1 + 16
SUN_PATH_MAX = 107
CTL_FALLBACK_BASE = Path("/tmp")


def control_dir_for(state_dir: Path) -> Path | None:
    """The ssh ControlMaster socket directory: <state_dir>/gpuwatch, or a private per-user directory under /tmp when
    that path is too long for a socket. None means run without connection sharing."""
    ctl = state_dir / "gpuwatch"
    if len(str(ctl)) + CTL_SUFFIX_LEN <= SUN_PATH_MAX:
        try:
            ctl.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return ctl
    ctl = CTL_FALLBACK_BASE / f"glm53-gw-{os.getuid()}"
    try:
        ctl.mkdir(mode=0o700, exist_ok=True)
        st = ctl.lstat()
    except OSError:
        return None
    # a shared /tmp: only a directory we own and nobody else can write to may hold the socket
    if not ctl.is_dir() or ctl.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o077:
        return None
    return ctl


@dataclass
class Node:
    name: str
    target: str = "local"          # "local" or an ssh destination (user@host)
    control_dir: str = ""          # ssh ControlMaster sockets: one TCP connection a node for the whole watch

    def command(self, argv: list[str]) -> list[str]:
        if self.target == "local":
            return argv
        opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
        if self.control_dir:
            opts += ["-o", "ControlMaster=auto", "-o", f"ControlPath={self.control_dir}/ssh-%C",
                     "-o", "ControlPersist=120"]
        return ["ssh", *opts, self.target, shlex.join(argv)]

    def run(self, argv: list[str], timeout: float = 15, stdin: str | None = None) -> tuple[int, str]:
        try:
            p = subprocess.run(self.command(argv), input=stdin, capture_output=True, text=True, timeout=timeout,
                               check=False)
        except subprocess.TimeoutExpired:
            return 124, "timed out"
        except OSError as exc:
            return 127, str(exc)
        return p.returncode, (p.stdout if p.returncode == 0 else (p.stderr.strip() or p.stdout.strip()))


def parse_nodes(spec: str) -> list[Node]:
    """``head=local,worker=ssh:root@worker-cx7`` -> nodes."""

    nodes = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, target = item.partition("=")
        target = target or "local"
        nodes.append(Node(name.strip(), target.removeprefix("ssh:")))
    return nodes


def smi_query(node: Node) -> dict[str, Any]:
    """One sample of ``node``: the fields, or {"error": ...}."""

    last = "no output"
    for fields in (FIELDS, FALLBACK_FIELDS):
        rc, out = node.run(["nvidia-smi", f"--query-gpu={','.join(f for _, f in fields)}",
                            "--format=csv,noheader,nounits"])
        if rc == 0:
            try:
                return parse_smi(out, fields)
            except ValueError as exc:
                last = str(exc)
                continue
        last = out.strip()[:200] or f"exit {rc}"
        if rc == 255 or rc == 124 or rc == 127:          # ssh / timeout / missing binary: a retry will not help
            break
    return {"error": last}


def fetch_metrics(url: str, timeout: float = 2.0) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return parse_metrics(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


# -- the optional bandwidth probe ----------------------------------------------------------------------------------
# Run as its own process (python3 - MIB REPS, source on stdin): the driver API through ctypes, no torch. The primary
# context, a lowest-priority non-blocking stream, REPS device-to-device copies of MIB MiB between two buffers timed
# with events. 64 MiB x 10: 1.34 GB of traffic, ~6 ms at a healthy ~230 GB/s, ~20 ms in the 66-80 GB/s slow state.
PROBE_SRC = r'''
import ctypes, json, sys
mib = int(sys.argv[1]) if len(sys.argv) > 1 else 64
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
out = {"ok": False}
try:
    cu = ctypes.CDLL("libcuda.so.1")
    u64, vp = ctypes.c_uint64, ctypes.c_void_p
    cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(u64), ctypes.c_size_t]
    cu.cuMemsetD8_v2.argtypes = [u64, ctypes.c_ubyte, ctypes.c_size_t]
    cu.cuMemcpyDtoDAsync_v2.argtypes = [u64, u64, ctypes.c_size_t, vp]
    cu.cuMemFree_v2.argtypes = [u64]
    cu.cuEventRecord.argtypes = [vp, vp]
    cu.cuEventSynchronize.argtypes = [vp]
    cu.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), vp, vp]
    cu.cuStreamSynchronize.argtypes = [vp]
    def ck(r, what):
        if r != 0:
            raise RuntimeError("%s -> CUresult %d" % (what, r))
    ck(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = vp()
    ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    size = mib << 20
    src, dst = u64(), u64()
    ck(cu.cuMemAlloc_v2(ctypes.byref(src), size), "cuMemAlloc")
    ck(cu.cuMemAlloc_v2(ctypes.byref(dst), size), "cuMemAlloc")
    least, greatest = ctypes.c_int(), ctypes.c_int()
    ck(cu.cuCtxGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest)), "priority range")
    stream = vp()
    ck(cu.cuStreamCreateWithPriority(ctypes.byref(stream), 1, least), "stream")   # 1: non-blocking; least: lowest
    ck(cu.cuMemsetD8_v2(src, 1, size), "memset")
    ck(cu.cuMemcpyDtoDAsync_v2(dst, src, size, stream), "warm-up copy")
    ck(cu.cuStreamSynchronize(stream), "sync")
    e0, e1 = vp(), vp()
    ck(cu.cuEventCreate(ctypes.byref(e0), 0), "event")
    ck(cu.cuEventCreate(ctypes.byref(e1), 0), "event")
    ck(cu.cuEventRecord(e0, stream), "record")
    for _ in range(reps):
        ck(cu.cuMemcpyDtoDAsync_v2(dst, src, size, stream), "copy")
    ck(cu.cuEventRecord(e1, stream), "record")
    ck(cu.cuEventSynchronize(e1), "sync")
    ms = ctypes.c_float()
    ck(cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1), "elapsed")
    out = {"ok": True, "ms": round(ms.value, 3), "gbs": round(2 * size * reps / (ms.value / 1e3) / 1e9, 1)}
    cu.cuMemFree_v2(src); cu.cuMemFree_v2(dst)
    cu.cuDevicePrimaryCtxRelease_v2(dev)
except Exception as exc:
    out = {"ok": False, "error": str(exc)}
print(json.dumps(out))
'''


def run_probe(node: Node, mib: int = 64, reps: int = 10) -> dict[str, Any]:
    """Memory bandwidth of ``node`` in GB/s (read + write of a device-to-device copy), or {"ok": False, ...}."""

    rc, out = node.run(["nice", "-n", "19", "timeout", "20", "python3", "-", str(mib), str(reps)],
                       timeout=30, stdin=PROBE_SRC)
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "error": (out or f"exit {rc}")[:200]}


# -- step time from /metrics -------------------------------------------------------------------------------------

@dataclass
class StepTracker:
    """Decode ms a round over windows of at least ``min_rounds`` rounds (the counters move when requests finish),
    against the median of the last windows with the same peak concurrency (a batched round costs more)."""

    min_rounds: int = 50
    max_window_s: float = 600.0
    history_len: int = 30
    min_history: int = 5
    factor: float = 1.3
    prev: dict[str, float] | None = None
    prev_t: float = 0.0
    peak_inflight: int = 0
    history: dict[str, deque] = field(default_factory=dict)

    def update(self, m: dict[str, float] | None, now: float) -> dict[str, Any] | None:
        if not m or "tensorfold_decode_rounds_total" not in m:
            return None
        cur = {"s": m.get("tensorfold_decode_seconds_total", 0.0), "r": m.get("tensorfold_decode_rounds_total", 0.0),
               "up": m.get("tensorfold_uptime_seconds", 0.0)}
        self.peak_inflight = max(self.peak_inflight, int(m.get("tensorfold_requests_inflight", 0)))
        p = self.prev
        if p is None or cur["r"] < p["r"] or cur["s"] < p["s"] or cur["up"] < p["up"]:   # first sample / restart
            self._reset(cur, now, m)
            return None
        dr, ds = cur["r"] - p["r"], cur["s"] - p["s"]
        if dr < self.min_rounds:
            if now - self.prev_t > self.max_window_s:            # a trickle over a long idle time: start over
                self._reset(cur, now, m)
            return None
        bucket = str(min(max(self.peak_inflight, 1), 4))
        step = 1000.0 * ds / dr
        hist = self.history.setdefault(bucket, deque(maxlen=self.history_len))
        base = statistics.median(hist) if len(hist) >= self.min_history else None
        hist.append(step)
        out = {"step_ms": round(step, 2), "rounds": int(dr), "window_s": round(now - self.prev_t, 1),
               "bucket": int(bucket), "baseline_ms": round(base, 2) if base else None,
               "ratio": round(step / base, 3) if base else None}
        out["regression"] = bool(base and step >= self.factor * base)
        self._reset(cur, now, m)
        return out

    def _reset(self, cur: dict[str, float], now: float, m: dict[str, float]) -> None:
        self.prev, self.prev_t = cur, now
        self.peak_inflight = int(m.get("tensorfold_requests_inflight", 0))

    def dump(self) -> dict[str, list[float]]:
        return {k: [round(x, 3) for x in v] for k, v in self.history.items()}

    def load(self, d: dict[str, list[float]]) -> None:
        for k, v in (d or {}).items():
            self.history[str(k)] = deque((float(x) for x in v), maxlen=self.history_len)


# -- detection -----------------------------------------------------------------------------------------------------

@dataclass
class Thresholds:
    load_util: float = 50.0          # utilization.gpu % at or above which a node counts as loaded
    clock_floor: float = 1500.0      # MHz under load; a clamp sits at 507-890
    power_clamp: float = 18.0        # W under load; the clamp draws ~12-14 W, a working GB10 42-92 W
    idle_floor: float = 1000.0       # MHz while idle and the idle reason is not set (0: off)
    swcap_frac: float = 0.5          # share of an interval under the SW power cap, while loaded
    hot_c: float = 85.0
    asym_ratio: float = 0.85         # loaded: slower / faster SM clock
    asym_power_ratio: float = 0.5    # loaded: lower / higher power
    idle_asym_ratio: float = 0.6     # idle: slower / faster SM clock (the 2026-09-11 clamp: 507 vs 2411 idle)
    probe_floor_gbs: float = 120.0   # probe copy bandwidth under which a node is flagged (and < 0.6 x its median)


# key -> (severity, samples in a row before it alerts)
PERSIST = {"unreachable": 2, "clock_floor": 2, "power_clamp": 2, "util_spin": 3, "clock_low_idle": 6,
           "throttle": 2, "sw_power_cap": 3, "hot": 3, "asymmetry": 3, "idle_asymmetry": 6, "step_regression": 1,
           "probe_slow": 1, "probe_asymmetry": 1}


@dataclass
class Condition:
    key: str                  # "<node>:<kind>" or "pair:<kind>"
    severity: str             # warn | crit
    message: str

    @property
    def kind(self) -> str:
        return self.key.split(":", 1)[1]


def loaded(s: dict[str, Any], th: Thresholds) -> bool:
    return (s.get("util_pct") or 0) >= th.load_util


def evaluate(samples: dict[str, dict[str, Any]], step: dict[str, Any] | None, th: Thresholds) -> list[Condition]:
    """The conditions a set of per-node samples (and the pair's step window) show right now."""

    out: list[Condition] = []
    good: dict[str, dict[str, Any]] = {}
    for name, s in samples.items():
        if "error" in s:
            out.append(Condition(f"{name}:unreachable", "warn", f"{name}: no sample ({s['error']})"))
            continue
        good[name] = s
        sm, pw, busy = s.get("sm_mhz"), s.get("power_w"), loaded(s, th)
        reasons = s.get("reason_names") or reason_names(s.get("reasons"))
        desc = f"{name}: {sm:.0f} MHz, {pw if pw is not None else '?'} W, {s.get('util_pct')}% busy" \
            if sm is not None else f"{name}: clock unreadable"
        if busy and sm is not None and sm < th.clock_floor:
            out.append(Condition(f"{name}:clock_floor", "crit",
                                 f"{desc}: SM clock under {th.clock_floor:.0f} MHz under load (clamp)"))
        if busy and pw is not None and pw < th.power_clamp:
            if sm is not None and sm >= th.clock_floor:
                # The clamp pins the SM clock (507-890 MHz). Full clocks at a few watts and ~96 % utilization is a
                # kernel waiting on a peer: an idle batched follower's control collective, or a context left
                # behind on the node after its rank stopped (issues #8, #10). Real work at 2 GHz draws 42-92 W.
                out.append(Condition(f"{name}:util_spin", "warn",
                                     f"{desc}: {s.get('util_pct')}% utilization at under {th.power_clamp:.0f} W "
                                     "with a healthy clock: a kernel spinning on a wait, not the clamp "
                                     "(docs/OPS-GPUWATCH.md)"))
            else:
                out.append(Condition(f"{name}:power_clamp", "crit",
                                     f"{desc}: power under {th.power_clamp:.0f} W under load (power clamp)"))
        if (not busy and th.idle_floor > 0 and sm is not None and sm < th.idle_floor
                and "gpu_idle" not in reasons):
            out.append(Condition(f"{name}:clock_low_idle", "warn",
                                 f"{desc}: idle clock under {th.idle_floor:.0f} MHz (a clamp reads like this; "
                                 "confirm under load)"))
        bad = sorted(THROTTLE_BAD.intersection(reasons))
        if bad:
            out.append(Condition(f"{name}:throttle", "warn", f"{desc}: clock event reasons {','.join(bad)}"))
        frac = s.get("swcap_frac")
        if busy and frac is not None and frac >= th.swcap_frac:
            out.append(Condition(f"{name}:sw_power_cap", "warn",
                                 f"{desc}: SW power cap held the clocks {frac:.0%} of the interval"))
        if (s.get("temp_c") or 0) >= th.hot_c:
            out.append(Condition(f"{name}:hot", "warn", f"{desc}: {s['temp_c']:.0f} C"))
        pr = s.get("probe")
        if isinstance(pr, dict) and pr.get("ok") and pr.get("slow"):
            out.append(Condition(f"{name}:probe_slow", "warn",
                                 f"{name}: probe {pr['gbs']} GB/s (floor {th.probe_floor_gbs:.0f}, "
                                 f"median {pr.get('median')}) -- the hidden slow state or a clamp"))
    if len(good) >= 2:
        names = sorted(good, key=lambda n: good[n].get("sm_mhz") or 0)
        lo, hi = good[names[0]], good[names[-1]]
        lo_sm, hi_sm = lo.get("sm_mhz"), hi.get("sm_mhz")
        both_busy = all(loaded(s, th) for s in good.values())
        none_busy = not any(loaded(s, th) for s in good.values())
        if lo_sm is not None and hi_sm:
            if both_busy and lo_sm / hi_sm < th.asym_ratio:
                out.append(Condition("pair:asymmetry", "warn",
                                     f"{names[0]} {lo_sm:.0f} MHz vs {names[-1]} {hi_sm:.0f} MHz under load: "
                                     f"the pair runs at {names[0]}'s pace"))
            if none_busy and lo_sm / hi_sm < th.idle_asym_ratio:
                out.append(Condition("pair:idle_asymmetry", "warn",
                                     f"idle clocks {names[0]} {lo_sm:.0f} MHz vs {names[-1]} {hi_sm:.0f} MHz "
                                     f"(the 2026-09-11 clamp showed 507 vs 2411 idle; confirm under load)"))
        if both_busy:
            pw = {n: s.get("power_w") for n, s in good.items() if s.get("power_w")}
            if len(pw) >= 2 and min(pw.values()) / max(pw.values()) < th.asym_power_ratio:
                low = min(pw, key=pw.__getitem__)
                high = max(pw, key=pw.__getitem__)
                if not any(c.key in (f"{low}:power_clamp", f"{low}:util_spin") for c in out):
                    out.append(Condition("pair:asymmetry", "warn",
                                         f"{low} draws {pw[low]:.0f} W vs {high} {pw[high]:.0f} W under load"))
        probes = {n: s["probe"]["gbs"] for n, s in good.items()
                  if isinstance(s.get("probe"), dict) and s["probe"].get("ok")}
        if len(probes) >= 2 and min(probes.values()) / max(probes.values()) < 0.7:
            low = min(probes, key=probes.__getitem__)
            out.append(Condition("pair:probe_asymmetry", "warn",
                                 f"probe: {low} {probes[low]} GB/s vs "
                                 + ", ".join(f"{n} {v}" for n, v in probes.items() if n != low)
                                 + f" GB/s -- {low} is the slow node"))
    if step and step.get("regression"):
        out.append(Condition("pair:step_regression", "warn",
                             f"decode {step['step_ms']} ms a round vs baseline {step['baseline_ms']} "
                             f"(x{step['ratio']}, {step['rounds']} rounds over {step['window_s']} s, "
                             f"concurrency {step['bucket']}): the slow state, or a clamp"))
    return out


# -- alerting ------------------------------------------------------------------------------------------------------

@dataclass
class Alerter:
    """Debounce (``PERSIST`` samples in a row), re-alert every ``realert_s`` while active, report clears. A
    step regression that stays past ``escalate_s`` is no longer the transient slow state: it escalates to crit."""

    notify: Callable[[str, str, str], None] | None = None     # (severity, key, message)
    realert_s: float = 600.0
    escalate_s: float = 120.0
    persist_scale: float = 1.0
    streak: dict[str, int] = field(default_factory=dict)
    active: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent: list[dict[str, Any]] = field(default_factory=list)

    def feed(self, conds: list[Condition], now: float, step_seen: bool = True) -> None:
        seen = {c.key: c for c in conds}
        for key, c in seen.items():
            self.streak[key] = self.streak.get(key, 0) + 1
            need = max(1, round(PERSIST.get(c.kind, 2) * self.persist_scale))
            a = self.active.get(key)
            if a is None:
                if self.streak[key] >= need:
                    self.active[key] = {"severity": c.severity, "since": now, "message": c.message,
                                        "last_alert": now}
                    self._emit(c.severity, key, "ALERT", c.message)
                continue
            a["message"] = c.message
            sev = c.severity
            if c.kind == "step_regression" and now - a["since"] >= self.escalate_s:
                sev = "crit"
            if RANK[sev] > RANK[a["severity"]]:
                a["severity"] = sev
                a["last_alert"] = now
                self._emit(sev, key, "ALERT", f"{c.message} (for {now - a['since']:.0f} s)")
            elif now - a["last_alert"] >= self.realert_s:
                a["last_alert"] = now
                self._emit(a["severity"], key, "STILL", f"{c.message} (for {now - a['since']:.0f} s)")
        for key in list(self.streak):
            if key not in seen:
                self.streak.pop(key)
        for key in list(self.active):
            if key in seen:
                continue
            # the step window only reports when enough rounds finished: keep its condition until a clean window
            if key == "pair:step_regression" and not step_seen:
                continue
            a = self.active.pop(key)
            dur = now - a["since"]
            self.recent.append({"key": key, "severity": a["severity"], "since": a["since"], "until": now,
                                "message": a["message"]})
            self.recent = self.recent[-50:]
            self._emit("ok", key, "CLEARED", f"{key} after {dur:.0f} s")

    def _emit(self, severity: str, key: str, verb: str, message: str) -> None:
        log(f"{verb} [{severity}] {key}: {message}")
        if self.notify:
            try:
                self.notify(severity, key, f"{verb} [{severity}] {key}: {message}")
            except Exception as exc:          # alerting must never stop the watch
                log(f"notify failed: {exc}")

    def worst(self) -> str:
        return max((a["severity"] for a in self.active.values()), key=RANK.__getitem__, default="ok")


def make_notify(webhook: str, alert_cmd: str) -> Callable[[str, str, str], None] | None:
    if not webhook and not alert_cmd:
        return None

    def notify(severity: str, key: str, message: str) -> None:
        if webhook:
            body = json.dumps({"text": f"gpuwatch: {message}", "content": f"gpuwatch: {message}",
                               "severity": severity, "key": key}).encode()
            req = urllib.request.Request(webhook, data=body, headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=5).read()
            except Exception as exc:
                log(f"webhook failed: {exc}")
        if alert_cmd:
            try:
                subprocess.run([alert_cmd, message], timeout=15, capture_output=True, check=False)
            except Exception as exc:
                log(f"alert command failed: {exc}")

    return notify


# -- the watch -----------------------------------------------------------------------------------------------------

CSV_COLUMNS = ["ts", "node", "ok", "sm_mhz", "gr_mhz", "mem_mhz", "max_sm_mhz", "app_gr_mhz", "power_w", "temp_c",
               "util_pct", "pstate", "reasons", "reason_names", "swcap_frac", "loaded", "probe_gbs", "step_ms",
               "step_baseline_ms", "step_rounds", "inflight", "conditions", "error"]


@dataclass
class Watch:
    nodes: list[Node]
    metrics_url: str = ""
    th: Thresholds = field(default_factory=Thresholds)
    state_dir: Path = Path(".")
    alerter: Alerter = field(default_factory=Alerter)
    steps: StepTracker = field(default_factory=StepTracker)
    probe_mode: str = "off"            # off | idle | on-alert
    probe_every: float = 60.0
    write_files: bool = True
    keep_days: int = 14
    prev: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict)
    last_probe: dict[str, float] = field(default_factory=dict)
    probe_hist: dict[str, deque] = field(default_factory=dict)
    last_step: dict[str, Any] | None = None
    prober: Callable[[Node], dict[str, Any]] = run_probe

    def sample(self, now: float | None = None) -> dict[str, dict[str, Any]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(self.nodes), 1)) as ex:
            got = dict(zip((n.name for n in self.nodes), ex.map(smi_query, self.nodes)))
        now = time.time() if now is None else now
        for name, s in got.items():
            if "error" in s:
                continue
            s["reason_names"] = reason_names(s.get("reasons"))
            s["loaded"] = loaded(s, self.th)
            ctr = s.get("ctr_sw_power_cap_us")
            if name in self.prev and ctr is not None:
                t0, p = self.prev[name]
                c0 = p.get("ctr_sw_power_cap_us")
                if c0 is not None and now > t0 and ctr >= c0:
                    s["swcap_frac"] = round(min((ctr - c0) / ((now - t0) * 1e6), 1.0), 3)
            self.prev[name] = (now, s)
        return got

    def maybe_probe(self, samples: dict[str, dict[str, Any]], metrics: dict[str, float] | None,
                    regression: bool, now: float) -> None:
        if self.probe_mode == "off":
            return
        every = max(self.probe_every, 60.0)                      # never more than once a minute a node
        if self.probe_mode == "idle":
            if metrics and metrics.get("tensorfold_requests_inflight", 0) > 0:
                return
            todo = [n for n in self.nodes if "error" not in samples[n.name]
                    and (samples[n.name].get("util_pct") or 0) <= 5
                    and now - self.last_probe.get(n.name, 0) >= every]
        else:                                                    # on-alert: every node at once, to compare them
            if not regression or any(now - self.last_probe.get(n.name, 0) < every for n in self.nodes):
                return
            todo = [n for n in self.nodes if "error" not in samples[n.name]]
        if not todo:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(todo)) as ex:
            results = dict(zip((n.name for n in todo), ex.map(self.prober, todo)))
        for name, r in results.items():
            self.last_probe[name] = now
            if not r.get("ok"):
                log(f"probe on {name} failed: {r.get('error')}")
                continue
            hist = self.probe_hist.setdefault(name, deque(maxlen=30))
            med = statistics.median(hist) if len(hist) >= 3 else None
            r["median"] = round(med, 1) if med else None
            r["slow"] = r["gbs"] < self.th.probe_floor_gbs and (med is None or r["gbs"] < 0.6 * med)
            hist.append(r["gbs"])
            samples[name]["probe"] = r

    def tick(self, now: float | None = None) -> list[Condition]:
        samples = self.sample(now)
        now = time.time() if now is None else now
        metrics = fetch_metrics(self.metrics_url) if self.metrics_url else None
        step = self.steps.update(metrics, now)
        if step:
            self.last_step = dict(step, ts=now)
        self.maybe_probe(samples, metrics, bool(step and step["regression"]), now)
        conds = evaluate(samples, step, self.th)
        self.alerter.feed(conds, now, step_seen=step is not None)
        if self.write_files:
            self.write_csv(samples, step, metrics, conds, now)
            self.write_state(samples, metrics, now)
        return conds

    def write_csv(self, samples, step, metrics, conds, now) -> None:
        d = self.state_dir / "gpuwatch"
        d.mkdir(parents=True, exist_ok=True)
        path = d / time.strftime("gpuwatch-%Y%m%d.csv", time.localtime(now))
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(CSV_COLUMNS)
                self._prune(d, keep=path)
            for name, s in samples.items():
                mine = [c.kind for c in conds if c.key.startswith(f"{name}:") or c.key.startswith("pair:")]
                reasons = s.get("reasons")
                row = {"ts": time.strftime("%FT%T", time.localtime(now)), "node": name, "ok": int("error" not in s),
                       **{k: s.get(k) for k in ("sm_mhz", "gr_mhz", "mem_mhz", "max_sm_mhz", "app_gr_mhz",
                                                "power_w", "temp_c", "util_pct", "pstate", "swcap_frac")},
                       "reasons": f"0x{reasons:x}" if isinstance(reasons, int) else "",
                       "reason_names": "|".join(s.get("reason_names") or []),
                       "loaded": int(bool(s.get("loaded"))),
                       "probe_gbs": (s.get("probe") or {}).get("gbs", ""),
                       "step_ms": step["step_ms"] if step else "",
                       "step_baseline_ms": (step or {}).get("baseline_ms") or "",
                       "step_rounds": step["rounds"] if step else "",
                       "inflight": int(metrics.get("tensorfold_requests_inflight", 0)) if metrics else "",
                       "conditions": "|".join(mine), "error": s.get("error", "")}
                w.writerow([_cell(row[c]) for c in CSV_COLUMNS])

    def _prune(self, d: Path, keep: Path) -> None:
        now = time.time()
        for p in d.glob("gpuwatch-*.csv"):
            try:
                if p != keep and now - p.stat().st_mtime > self.keep_days * 86400:
                    p.unlink()
            except OSError:
                pass

    def write_state(self, samples, metrics, now) -> None:
        d = self.state_dir / "gpuwatch"
        d.mkdir(parents=True, exist_ok=True)
        state = {"ts": now, "time": time.strftime("%F %T", time.localtime(now)), "nodes": samples,
                 "metrics_ok": metrics is not None, "step": self.last_step, "active": self.alerter.active,
                 "recent": self.alerter.recent, "baselines": self.steps.dump(),
                 "probe_hist": {k: list(v) for k, v in self.probe_hist.items()}}
        tmp = d / "gpuwatch.json.tmp"
        tmp.write_text(json.dumps(state, indent=1, default=str))
        os.replace(tmp, d / "gpuwatch.json")

    def restore(self) -> None:
        """Carry the step baselines and probe history over a restart of the watch."""

        st = load_state(self.state_dir)
        if st:
            self.steps.load(st.get("baselines") or {})
            for k, v in (st.get("probe_hist") or {}).items():
                self.probe_hist[k] = deque(v, maxlen=30)
            self.alerter.recent = list(st.get("recent") or [])


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def load_state(state_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((state_dir / "gpuwatch" / "gpuwatch.json").read_text())
    except (OSError, ValueError):
        return None


# -- check (preflight) ---------------------------------------------------------------------------------------------

def verdict_from_state(st: dict[str, Any], now: float, recent_s: float) -> tuple[str, list[str]]:
    lines, worst = [], "ok"
    for key, a in sorted((st.get("active") or {}).items()):
        lines.append(f"{a['severity']}: {a['message']} (for {now - a['since']:.0f} s)")
        worst = max(worst, a["severity"], key=RANK.__getitem__)
    for r in st.get("recent") or []:
        if now - r.get("until", 0) <= recent_s:
            what = "the transient slow state? wait it out and re-check" if r["key"].endswith("step_regression") \
                else "cleared since"
            lines.append(f"warn: {r['key']} {time.strftime('%T', time.localtime(r['since']))}-"
                         f"{time.strftime('%T', time.localtime(r['until']))} ({what}): {r['message']}")
            worst = max(worst, "warn", key=RANK.__getitem__)
    return worst, lines


def describe(samples: dict[str, dict[str, Any]]) -> str:
    parts = []
    for name, s in samples.items():
        if "error" in s:
            parts.append(f"{name}: unreadable")
            continue
        parts.append(f"{name}: {s.get('sm_mhz') or 0:.0f} MHz {s.get('power_w') or 0:.1f} W {s.get('temp_c') or 0:.0f} C "
                     f"{s.get('util_pct') or 0:.0f}%" + (" busy" if s.get("loaded") else " idle")
                     + (f" [{','.join(s.get('reason_names') or [])}]" if s.get("reason_names") else ""))
    return " | ".join(parts)


def cmd_check(args, nodes: list[Node], th: Thresholds, state_dir: Path) -> int:
    """0 ok, 1 warning (a strict caller refuses too), 2 degraded, 3 cannot tell."""

    now = time.time()
    st = load_state(state_dir)
    if st and now - float(st.get("ts", 0)) <= args.max_age:
        worst, lines = verdict_from_state(st, now, args.recent)
        print(f"gpuwatch: {worst} (watch state {now - st['ts']:.0f} s old) {describe(st.get('nodes') or {})}")
        for line in lines:
            print(f"gpuwatch:   {line}")
    else:
        # no fresh watch: a few live samples here (the same rules, debounced over them)
        w = Watch(nodes=nodes, th=th, state_dir=state_dir, write_files=False,
                  alerter=Alerter(persist_scale=0.0))
        samples: dict[str, dict[str, Any]] = {}
        counts: dict[str, int] = {}
        msgs: dict[str, Condition] = {}
        for i in range(args.samples):
            samples = w.sample()
            for c in evaluate(samples, None, th):
                counts[c.key] = counts.get(c.key, 0) + 1
                msgs[c.key] = c
            if i + 1 < args.samples:
                time.sleep(args.sample_gap)
        need = args.samples // 2 + 1
        hits = [msgs[k] for k, n in counts.items() if n >= need]
        if samples and all("error" in s for s in samples.values()):
            print(f"gpuwatch: cannot read any node ({describe(samples)})")
            return 3
        worst = max((c.severity for c in hits), key=RANK.__getitem__, default="ok")
        lines = [f"{c.severity}: {c.message}" for c in hits]
        if st:                          # a stale state still tells about a recent slow state
            rw, recent = verdict_from_state({"recent": st.get("recent")}, now, args.recent)
            lines += recent
            worst = max(worst, rw, key=RANK.__getitem__)
        age = f"{now - st['ts']:.0f} s old" if st else "none"
        print(f"gpuwatch: {worst} (live, {args.samples} samples; watch state {age}) {describe(samples)}")
        for line in lines:
            print(f"gpuwatch:   {line}")
    code = RANK[worst]
    return 1 if args.strict and code == 1 else code


# -- CLI -----------------------------------------------------------------------------------------------------------

def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("cmd", choices=["watch", "check", "status"])
    ap.add_argument("--config", default=os.environ.get("CONFIG", str(ROOT / "config" / "prod.env")),
                    help="serve.sh config file: WORKER_SSH and PORT are read from it")
    ap.add_argument("--nodes", default=os.environ.get("GPUWATCH_NODES", ""),
                    help="name=local|name=ssh:user@host,... (default: head=local,worker=ssh:$WORKER_SSH)")
    ap.add_argument("--worker", default="", help="the worker's ssh destination (overrides the config)")
    ap.add_argument("--port", default="", help="the server port for /metrics (overrides the config)")
    ap.add_argument("--metrics-url", default=os.environ.get("GPUWATCH_METRICS_URL", ""))
    ap.add_argument("--state-dir", default=os.environ.get("STATE_DIR") or os.path.join(
        os.environ.get("XDG_STATE_HOME") or os.path.join(os.environ.get("HOME") or "/tmp", ".local/state"),
        "glm53-tf"))
    ap.add_argument("--interval", type=float, default=float(os.environ.get("GPUWATCH_INTERVAL", "5")))
    ap.add_argument("--count", type=int, default=0, help="watch: stop after N samples (0: forever)")
    ap.add_argument("--webhook", default=os.environ.get("GPUWATCH_WEBHOOK", ""))
    ap.add_argument("--alert-cmd", default=os.environ.get("GPUWATCH_ALERT_CMD") or os.environ.get("WATCH_ALERT", ""))
    ap.add_argument("--realert", type=float, default=600.0)
    ap.add_argument("--probe", choices=["off", "idle", "on-alert"], default=os.environ.get("GPUWATCH_PROBE", "off"))
    ap.add_argument("--probe-every", type=float, default=60.0, help="seconds between probes of a node (min 60)")
    ap.add_argument("--keep-days", type=int, default=14)
    th = Thresholds()
    for k, v in vars(th).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v, dest=f"th_{k}")
    ap.add_argument("--step-factor", type=float, default=1.3, help="a window this much slower than baseline alerts")
    ap.add_argument("--step-min-rounds", type=int, default=50)
    ap.add_argument("--escalate", type=float, default=120.0,
                    help="a step regression lasting this long is not the transient slow state: crit")
    # check
    ap.add_argument("--max-age", type=float, default=60.0, help="check: use the watch's state if this fresh")
    ap.add_argument("--recent", type=float, default=600.0, help="check: cleared conditions this recent still warn")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--sample-gap", type=float, default=1.0)
    ap.add_argument("--strict", action="store_true", help="check: a warning exits 1 too (benchmarks)")
    args = ap.parse_args(argv)

    th = Thresholds(**{k: getattr(args, f"th_{k}") for k in vars(Thresholds())})
    state_dir = Path(args.state_dir)
    cfg = read_env_file(Path(args.config))
    worker = args.worker or os.environ.get("WORKER_SSH") or cfg.get("WORKER_SSH", "")
    port = args.port or os.environ.get("PORT") or cfg.get("PORT", "8000")
    nodes = parse_nodes(args.nodes) if args.nodes else [Node("head")] + ([Node("worker", worker)] if worker else [])
    ctl = control_dir_for(state_dir)
    if ctl:
        for n in nodes:
            n.control_dir = str(ctl)

    if args.cmd == "status":
        st = load_state(state_dir)
        if not st:
            print("gpuwatch: no state yet")
            return 3
        print(json.dumps({k: st.get(k) for k in ("time", "active", "recent", "step", "nodes")}, indent=1,
                         default=str))
        return RANK[max((a["severity"] for a in (st.get("active") or {}).values()), key=RANK.__getitem__,
                        default="ok")]
    if args.cmd == "check":
        return cmd_check(args, nodes, th, state_dir)

    w = Watch(nodes=nodes, metrics_url=args.metrics_url or f"http://127.0.0.1:{port}/metrics", th=th,
              state_dir=state_dir,
              alerter=Alerter(notify=make_notify(args.webhook, args.alert_cmd), realert_s=args.realert,
                              escalate_s=args.escalate),
              steps=StepTracker(min_rounds=args.step_min_rounds, factor=args.step_factor),
              probe_mode=args.probe, probe_every=args.probe_every, keep_days=args.keep_days)
    w.restore()
    log(f"watching {', '.join(f'{n.name}={n.target}' for n in nodes)} every {args.interval:g} s; "
        f"metrics {w.metrics_url}; probe {args.probe}; state {state_dir / 'gpuwatch'}")
    n = 0
    while True:
        t0 = time.monotonic()
        try:
            w.tick()
        except Exception as exc:              # one bad tick (a full disk, a parse surprise) must not end the watch
            log(f"tick failed: {exc!r}")
        n += 1
        if args.count and n >= args.count:
            return RANK[w.alerter.worst()]
        time.sleep(max(0.0, args.interval - (time.monotonic() - t0)))


if __name__ == "__main__":
    sys.exit(main())
