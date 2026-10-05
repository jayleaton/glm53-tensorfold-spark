"""Host-only tests of scripts/gpuwatch.py (the GB10 clock / slow-state watch) and its serve.sh preflight hook.

No GPU: ``nvidia-smi`` is a shell fake printing per-node canned ``--query-gpu`` lines (the head's is "local", the
worker's is reached through a fake ``ssh`` that runs the command here with FAKE_NODE set), and the server's /metrics is
a small HTTP server whose decode counters the tests advance at a chosen ms a round.

    python -m pytest -q tests/test_gpuwatch.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gpuwatch as gw  # noqa: E402

# what the head node printed on 2026-09-28 (idle, server loaded, -lgc 300,2250), in the FIELDS order, nounits
REAL_IDLE = "2229, 2229, [N/A], 3003, 2418, 13.00, 57, 0, P0, 0x0000000000000000, 6869275397, 0, 0, 0"


def smi_line(sm=2229, power=13.0, util=0, reasons=0, temp=57, swcap=6869275397, counters=True) -> str:
    vals = [sm, sm, "[N/A]", 3003, 2418, f"{power:.2f}", temp, util, "P0", f"0x{reasons:016x}"]
    if counters:
        vals += [swcap, 0, 0, 0]
    return ", ".join(str(v) for v in vals)


HEALTHY_LOAD = dict(sm=2236, power=71.0, util=96, temp=68)
CLAMP_LOAD = dict(sm=507, power=12.1, util=96, temp=43)            # the head node, 2026-09-11
CLAMP_IDLE = dict(sm=507, power=5.1, util=0, temp=42)
SPIN_IDLE = dict(sm=2093, power=10.3, util=96, temp=51)            # issue #10: the worker after a stop


# -- parsing -------------------------------------------------------------------------------------------------------

def test_parse_real_line():
    s = gw.parse_smi(REAL_IDLE)
    assert s["sm_mhz"] == 2229 and s["mem_mhz"] is None and s["power_w"] == 13.0
    assert s["pstate"] == "P0" and s["reasons"] == 0 and s["ctr_sw_power_cap_us"] == 6869275397
    with pytest.raises(ValueError):
        gw.parse_smi("Field \"clocks_event_reasons_counters.sw_power_cap\" is not a valid field to query.")
    with pytest.raises(ValueError):
        gw.parse_smi("")


def test_reason_names():
    assert gw.reason_names(0) == []
    assert gw.reason_names(0x1 | 0x4 | 0x80) == ["gpu_idle", "sw_power_cap", "hw_power_brake_slowdown"]
    assert gw.reason_names(None) == []


def test_parse_metrics_sums_labels():
    m = gw.parse_metrics('# HELP x\ntensorfold_decode_rounds_total{model="a"} 10\n'
                         'tensorfold_decode_rounds_total{model="b"} 5\ntensorfold_uptime_seconds{model="a"} 1.5\n')
    assert m == {"tensorfold_decode_rounds_total": 15.0, "tensorfold_uptime_seconds": 1.5}


def test_parse_nodes():
    nodes = gw.parse_nodes("head=local, worker=ssh:root@worker-cx7")
    assert [(n.name, n.target) for n in nodes] == [("head", "local"), ("worker", "root@worker-cx7")]
    assert nodes[1].command(["nvidia-smi", "-q"])[-2:] == ["root@worker-cx7", "nvidia-smi -q"]


# -- detection -----------------------------------------------------------------------------------------------------

def sample(**kw) -> dict:
    s = gw.parse_smi(smi_line(**kw))
    s["reason_names"] = gw.reason_names(s["reasons"])
    return s


def kinds(conds) -> set[str]:
    return {c.key for c in conds}


def test_control_path_length_math():
    # the path ssh refused on a node whose home is /home/<19-char user> (2026-09-30): 112 bytes, over sun_path
    real = "/home/" + "u" * 19 + "/.local/state/glm53-tf/gpuwatch"
    assert len(real) + gw.CTL_SUFFIX_LEN > gw.SUN_PATH_MAX
    # the fallback always fits, whatever the uid's digit count
    assert len(str(gw.CTL_FALLBACK_BASE / "glm53-gw-4294967294")) + gw.CTL_SUFFIX_LEN <= gw.SUN_PATH_MAX


def test_control_dir_short_state_dir_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(gw, "SUN_PATH_MAX", 10_000)
    ctl = gw.control_dir_for(tmp_path / "state")
    assert ctl == tmp_path / "state" / "gpuwatch" and ctl.is_dir()


def test_control_dir_long_state_dir_falls_back(tmp_path, monkeypatch):
    base = tmp_path / "t"
    base.mkdir()
    monkeypatch.setattr(gw, "CTL_FALLBACK_BASE", base)
    monkeypatch.setattr(gw, "SUN_PATH_MAX", len(str(base)) + 20 + gw.CTL_SUFFIX_LEN)
    long_state = tmp_path / ("s" * 120)
    ctl = gw.control_dir_for(long_state)
    assert ctl == base / f"glm53-gw-{os.getuid()}"
    assert ctl.stat().st_mode & 0o777 == 0o700
    assert not (long_state / "gpuwatch").exists()
    assert gw.Node("worker", "user@host", str(ctl)).command(["true"])[:8] == [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ControlMaster=auto", "-o"]


def test_control_dir_refuses_an_unsafe_fallback(tmp_path, monkeypatch):
    base = tmp_path / "t"
    base.mkdir()
    monkeypatch.setattr(gw, "CTL_FALLBACK_BASE", base)
    monkeypatch.setattr(gw, "SUN_PATH_MAX", len(str(base)) + 20 + gw.CTL_SUFFIX_LEN)
    long_state = tmp_path / ("s" * 120)
    fallback = base / f"glm53-gw-{os.getuid()}"
    fallback.mkdir()
    fallback.chmod(0o777)                      # writable by others: someone could plant the socket
    assert gw.control_dir_for(long_state) is None
    fallback.rmdir()
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    fallback.symlink_to(target)                # a symlink planted in a shared /tmp
    assert gw.control_dir_for(long_state) is None


def test_healthy_pair_is_quiet():
    th = gw.Thresholds()
    assert gw.evaluate({"head": sample(), "worker": sample(sm=2236, power=10.9)}, None, th) == []
    assert gw.evaluate({"head": sample(**HEALTHY_LOAD), "worker": sample(**HEALTHY_LOAD)}, None, th) == []


def test_clamp_under_load_is_critical():
    conds = gw.evaluate({"head": sample(**CLAMP_LOAD), "worker": sample(**HEALTHY_LOAD)}, None, gw.Thresholds())
    assert {"head:clock_floor", "head:power_clamp", "pair:asymmetry"} <= kinds(conds)
    assert all(c.severity == "crit" for c in conds if c.key.startswith("head:"))
    assert "head 507 MHz vs worker 2236 MHz" in next(c.message for c in conds if c.key == "pair:asymmetry")


def test_clamp_at_idle_warns():
    conds = gw.evaluate({"head": sample(**CLAMP_IDLE), "worker": sample(sm=2411, power=12.1)}, None,
                        gw.Thresholds())
    assert kinds(conds) == {"head:clock_low_idle", "pair:idle_asymmetry"}
    assert all(c.severity == "warn" for c in conds)
    # parked by the driver (the idle reason set): no idle-floor warning for that node
    conds = gw.evaluate({"head": sample(sm=507, reasons=0x1)}, None, gw.Thresholds())
    assert conds == []


def test_spin_at_full_clock_is_not_the_clamp():
    # issues #8 / #10: ~96 % utilization at full clocks and a few watts is a kernel waiting on a peer, not the clamp
    for head, worker in ((sample(), sample(**SPIN_IDLE)),                          # #10: after a stop
                         (sample(sm=2411, power=13.6), sample(sm=2411, power=14.7, util=96))):   # #8: idle server
        conds = gw.evaluate({"head": head, "worker": worker}, None, gw.Thresholds())
        assert kinds(conds) == {"worker:util_spin"}
        assert conds[0].severity == "warn" and "not the clamp" in conds[0].message
    # beside a node under real load: no power asymmetry on top of the spin
    conds = gw.evaluate({"head": sample(**HEALTHY_LOAD), "worker": sample(**SPIN_IDLE)}, None, gw.Thresholds())
    assert kinds(conds) == {"worker:util_spin"}
    # the clamp's own signature (pinned clock) stays critical
    conds = gw.evaluate({"head": sample(**CLAMP_LOAD)}, None, gw.Thresholds())
    assert {"head:clock_floor", "head:power_clamp"} <= kinds(conds) and "head:util_spin" not in kinds(conds)


def test_throttle_power_cap_heat_and_unreachable():
    th = gw.Thresholds()
    s = sample(**dict(HEALTHY_LOAD, temp=88), reasons=0x40 | 0x4)
    s["swcap_frac"] = 0.8
    conds = gw.evaluate({"head": s, "worker": {"error": "ssh: connect timed out"}}, None, th)
    assert kinds(conds) == {"head:throttle", "head:sw_power_cap", "head:hot", "worker:unreachable"}
    assert "hw_thermal_slowdown" in next(c.message for c in conds if c.key == "head:throttle")


def test_power_asymmetry_under_load():
    conds = gw.evaluate({"head": sample(**HEALTHY_LOAD), "worker": sample(sm=2236, power=25.0, util=96)}, None,
                        gw.Thresholds())
    assert kinds(conds) == {"pair:asymmetry"} and "worker draws 25 W" in conds[0].message


# -- step time ----------------------------------------------------------------------------------------------------

def metrics(rounds, secs, inflight=1, up=100.0):
    return {"tensorfold_decode_rounds_total": rounds, "tensorfold_decode_seconds_total": secs,
            "tensorfold_requests_inflight": inflight, "tensorfold_uptime_seconds": up}


def test_step_tracker_baseline_and_regression():
    t = gw.StepTracker(min_rounds=100)
    r, s, now = 0.0, 0.0, 0.0
    assert t.update(metrics(r, s), now) is None                  # first sample seeds
    outs = []
    for ms in [63] * 6 + [94] + [63]:
        r += 100
        s += 100 * ms / 1000
        now += 10
        outs.append(t.update(metrics(r, s, up=100 + now), now))
    assert outs[0]["baseline_ms"] is None and not outs[0]["regression"]
    assert outs[5]["baseline_ms"] == 63.0 and not outs[5]["regression"]
    assert outs[6]["step_ms"] == 94.0 and outs[6]["regression"] and outs[6]["ratio"] == pytest.approx(1.492, 0.01)
    assert not outs[7]["regression"]                             # one slow window does not move the median


def test_step_tracker_waits_for_rounds_and_resets_on_restart():
    t = gw.StepTracker(min_rounds=100)
    t.update(metrics(1000, 60), 0)
    assert t.update(metrics(1050, 63), 5) is None                # too few rounds: keep accumulating
    out = t.update(metrics(1100, 66.3), 10)
    assert out["rounds"] == 100 and out["step_ms"] == pytest.approx(63.0)
    assert t.update(metrics(10, 0.6, up=3), 15) is None          # the server restarted: reseed, no bogus window


def test_step_tracker_concurrency_buckets():
    t = gw.StepTracker(min_rounds=10, min_history=2)
    t.update(metrics(0, 0, inflight=1), 0)
    t.update(metrics(10, 0.6, inflight=1), 1)
    t.update(metrics(20, 1.2, inflight=1), 2)
    out = t.update(metrics(30, 2.4, inflight=4), 3)              # 4 streams at 120 ms: its own bucket, no baseline
    assert out["bucket"] == 4 and out["baseline_ms"] is None and not out["regression"]


# -- alerting ------------------------------------------------------------------------------------------------------

def test_alerter_debounce_clear_and_escalation():
    sent = []
    a = gw.Alerter(notify=lambda sev, key, msg: sent.append((sev, key, msg)), realert_s=100, escalate_s=60)
    clamp = gw.Condition("head:clock_floor", "crit", "head: 507 MHz")
    a.feed([clamp], 0)
    assert not a.active and not sent                              # one sample is not enough
    a.feed([clamp], 5)
    assert a.active["head:clock_floor"]["severity"] == "crit" and sent[-1][2].startswith("ALERT [crit]")
    a.feed([clamp], 50)
    assert len(sent) == 1                                         # no repeat inside realert_s
    a.feed([clamp], 110)
    assert sent[-1][2].startswith("STILL")
    a.feed([], 115)
    assert not a.active and a.recent[-1]["key"] == "head:clock_floor" and sent[-1][2].startswith("CLEARED")
    assert a.worst() == "ok"

    slow = gw.Condition("pair:step_regression", "warn", "decode 94 ms")
    a.feed([slow], 200)
    assert a.active["pair:step_regression"]["severity"] == "warn"
    a.feed([], 205, step_seen=False)                              # no window this tick: still active
    assert "pair:step_regression" in a.active
    a.feed([slow], 270)                                           # 70 s > escalate_s: not the transient state
    assert a.active["pair:step_regression"]["severity"] == "crit" and a.worst() == "crit"


def test_notify_webhook_and_command(tmp_path):
    got = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cmd = tmp_path / "notify"
    out = tmp_path / "notified"
    cmd.write_text(f"#!/bin/sh\necho \"$1\" >> {out}\n")
    cmd.chmod(0o755)
    try:
        n = gw.make_notify(f"http://127.0.0.1:{srv.server_address[1]}/hook", str(cmd))
        n("crit", "head:power_clamp", "ALERT [crit] head:power_clamp: 12 W")
    finally:
        srv.shutdown()
        srv.server_close()
    assert got and got[0]["severity"] == "crit" and "12 W" in got[0]["text"]
    assert "12 W" in out.read_text()
    gw.make_notify("http://127.0.0.1:9/nothing", "")("warn", "k", "m")   # a dead webhook is logged, not raised
    assert gw.make_notify("", "") is None


# -- the watch against fake tools --------------------------------------------------------------------------------

FAKE_SMI = r"""#!/usr/bin/env bash
node="${FAKE_NODE:-head}"; f="$FAKE_SMI_DIR/$node"
case "$*" in
    *driver_version*) echo "580.178.04"; exit 0 ;;
    *--query-compute-apps*) exit 0 ;;
esac
[[ -f "$f.fail" ]] && { echo "Unable to determine the device handle for GPU0000:01:00.0: Unknown Error" >&2; exit 9; }
if [[ -f "$f.old" && "$*" == *clocks_event_reasons_counters* ]]; then
    echo 'Field "clocks_event_reasons_counters.sw_power_cap" is not a valid field to query.'; exit 2
fi
cat "$f"
"""

FAKE_SSH = r"""#!/usr/bin/env bash
echo "ssh $*" >> "$FAKE_SMI_DIR/calls"
while [[ "$1" == -o ]]; do shift 2; done
[[ -f "$FAKE_SMI_DIR/ssh.fail" ]] && { echo "ssh: connect to host $1 port 22: No route to host" >&2; exit 255; }
FAKE_NODE="$1"; export FAKE_NODE; shift
exec bash -c "$*"
"""


class Server:
    """/metrics with decode counters the test advances."""

    def __init__(self):
        self.rounds, self.secs, self.inflight, self.up = 0.0, 0.0, 1, 1000.0
        state = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = (f'tensorfold_decode_rounds_total{{model="m"}} {state.rounds}\n'
                        f'tensorfold_decode_seconds_total{{model="m"}} {state.secs:.6f}\n'
                        f'tensorfold_requests_inflight{{model="m"}} {state.inflight}\n'
                        f'tensorfold_uptime_seconds{{model="m"}} {state.up}\n').encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/metrics"

    def decode(self, rounds, ms):
        self.rounds += rounds
        self.secs += rounds * ms / 1000
        self.up += 5


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, text in (("nvidia-smi", FAKE_SMI), ("ssh", FAKE_SSH)):
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    smi = tmp_path / "smi"
    smi.mkdir()
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SMI_DIR", str(smi))

    class F:
        dir = tmp_path
        state = tmp_path / "state"

        @staticmethod
        def set(node, **kw):
            (smi / node).write_text(smi_line(**kw) + "\n")

        @staticmethod
        def flag(name, on=True):
            p = smi / name
            p.touch() if on else p.unlink(missing_ok=True)

    F.set("head")
    F.set("fake-worker", sm=2236, power=10.9)
    srv = Server()
    F.server = srv
    yield F
    srv.srv.shutdown()
    srv.srv.server_close()


def make_watch(fakes, **kw) -> gw.Watch:
    nodes = [gw.Node("head"), gw.Node("worker", "fake-worker")]
    return gw.Watch(nodes=nodes, metrics_url=fakes.server.url, state_dir=fakes.state,
                    steps=gw.StepTracker(min_rounds=50), **kw)


def test_watch_writes_csv_and_state(fakes):
    w = make_watch(fakes)
    t = 1_800_000_000.0
    for i in range(3):
        fakes.server.decode(100, 63)
        assert w.tick(t + 5 * i) == []
    d = fakes.state / "gpuwatch"
    rows = next(d.glob("gpuwatch-*.csv")).read_text().splitlines()
    assert rows[0].split(",") == gw.CSV_COLUMNS and len(rows) == 1 + 3 * 2
    assert ",worker,1,2236," in rows[2]
    st = json.loads((d / "gpuwatch.json").read_text())
    assert st["nodes"]["head"]["sm_mhz"] == 2229 and st["active"] == {} and st["step"]["step_ms"] == 63.0
    assert st["baselines"]["1"] == [63.0, 63.0]


def test_watch_power_clamp_alert_and_clear(fakes, capsys):
    w = make_watch(fakes)
    t = 1_800_000_000.0
    fakes.set("head", **HEALTHY_LOAD)
    fakes.set("fake-worker", **CLAMP_LOAD)
    w.tick(t)
    assert not w.alerter.active                                   # debounced
    w.tick(t + 5)
    assert w.alerter.active["worker:power_clamp"]["severity"] == "crit"
    assert w.alerter.active["worker:clock_floor"]["severity"] == "crit"
    out = capsys.readouterr().out
    assert "ALERT [crit] worker:power_clamp" in out
    fakes.set("fake-worker", **HEALTHY_LOAD)
    w.tick(t + 10)
    assert not w.alerter.active and "CLEARED" in capsys.readouterr().out


def test_watch_transient_slow_state(fakes, capsys):
    """The hidden slow state: nvidia-smi unchanged, decode 63 -> 94 ms a round for one window, then back."""

    w = make_watch(fakes)
    t = 1_800_000_000.0
    fakes.set("head", **HEALTHY_LOAD)
    fakes.set("fake-worker", **HEALTHY_LOAD)
    w.tick(t)
    for i in range(1, 7):
        fakes.server.decode(100, 63)
        w.tick(t + 5 * i)
    assert not w.alerter.active
    fakes.server.decode(100, 94)
    conds = w.tick(t + 40)
    assert kinds(conds) == {"pair:step_regression"} and "94.0 ms a round vs baseline 63.0" in conds[0].message
    assert w.alerter.active["pair:step_regression"]["severity"] == "warn"
    w.tick(t + 45)                                                # no new rounds: stays active, no clear
    assert "pair:step_regression" in w.alerter.active
    fakes.server.decode(100, 63)
    w.tick(t + 50)
    assert not w.alerter.active and w.alerter.recent[-1]["key"] == "pair:step_regression"


def test_watch_swcap_fraction_from_counters(fakes):
    w = make_watch(fakes)
    t = 1_800_000_000.0
    fakes.set("head", **HEALTHY_LOAD, swcap=1_000_000_000)
    w.tick(t)
    fakes.set("head", **HEALTHY_LOAD, swcap=1_004_000_000)       # 4 s of power capping in a 5 s interval
    w.tick(t + 5)
    assert w.prev["head"][1]["swcap_frac"] == pytest.approx(0.8)
    w.tick(t + 10)
    assert "head:sw_power_cap" not in w.alerter.active            # 0 in the last interval: streak broken
    fakes.set("head", **HEALTHY_LOAD, swcap=1_008_000_000)
    for i in range(3):
        fakes.set("head", **HEALTHY_LOAD, swcap=1_008_000_000 + 4_000_000 * (i + 1))
        w.tick(t + 15 + 5 * i)
    assert "head:sw_power_cap" in w.alerter.active


def test_old_driver_fallback_and_unreachable(fakes):
    fakes.flag("head.old")
    fakes.set("head", counters=False)
    s = gw.smi_query(gw.Node("head"))
    assert s["sm_mhz"] == 2229 and "ctr_sw_power_cap_us" not in s
    fakes.flag("ssh.fail")
    s = gw.smi_query(gw.Node("worker", "fake-worker"))
    assert "No route to host" in s["error"]
    fakes.flag("ssh.fail", False)
    fakes.flag("fake-worker.fail")
    assert "Unknown Error" in gw.smi_query(gw.Node("worker", "fake-worker"))["error"]


# -- probe gating (no GPU: the prober is a stub) ------------------------------------------------------------------

def test_probe_source_compiles_and_parse():
    compile(gw.PROBE_SRC, "probe", "exec")

    class N(gw.Node):
        def run(self, argv, timeout=15, stdin=None):
            assert argv[:3] == ["nice", "-n", "19"] and "python3" in argv and stdin == gw.PROBE_SRC
            return 0, 'noise\n{"ok": true, "ms": 5.8, "gbs": 231.4}\n'

    assert gw.run_probe(N("head")) == {"ok": True, "ms": 5.8, "gbs": 231.4}


def test_probe_idle_mode_gating(fakes):
    calls = []

    def prober(node):
        calls.append(node.name)
        return {"ok": True, "ms": 6.0, "gbs": 230.0 if node.name == "head" else 70.0}

    w = make_watch(fakes, probe_mode="idle", prober=prober)
    t = 1_800_000_000.0
    fakes.server.inflight = 0
    conds = w.tick(t)
    assert sorted(calls) == ["head", "worker"]
    assert "pair:probe_asymmetry" in kinds(conds) and "worker:probe_slow" in kinds(conds)
    w.tick(t + 5)
    assert len(calls) == 2                                        # at most once a minute a node
    w.tick(t + 61)
    assert len(calls) == 4
    fakes.server.inflight = 2                                     # serving: no probe
    w.tick(t + 200)
    assert len(calls) == 4
    fakes.server.inflight = 0
    fakes.set("head", **HEALTHY_LOAD)                             # busy node: not probed
    w.tick(t + 300)
    assert calls[-1] == "worker" and len(calls) == 5


def test_probe_on_alert_only_after_regression(fakes):
    calls = []
    w = make_watch(fakes, probe_mode="on-alert",
                   prober=lambda n: calls.append(n.name) or {"ok": True, "ms": 6, "gbs": 230.0})
    t = 1_800_000_000.0
    w.tick(t)
    for i in range(1, 7):
        fakes.server.decode(100, 63)
        w.tick(t + 5 * i)
    assert calls == []
    fakes.server.decode(100, 94)
    w.tick(t + 40)
    assert sorted(calls) == ["head", "worker"]


# -- check (the preflight) ----------------------------------------------------------------------------------------

def run_check(fakes, *extra) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(ROOT / "scripts/gpuwatch.py"), "check", "--state-dir", str(fakes.state),
                        "--worker", "fake-worker", "--sample-gap", "0", "--config", "/nonexistent", *extra],
                       capture_output=True, text=True, env=dict(os.environ), timeout=60)
    return r.returncode, r.stdout + r.stderr


def test_check_live(fakes):
    rc, out = run_check(fakes)
    assert rc == 0 and "gpuwatch: ok (live" in out and "head: 2229 MHz" in out
    fakes.set("head", **CLAMP_IDLE)
    fakes.set("fake-worker", sm=2411, power=12.1)
    rc, out = run_check(fakes)
    assert rc == 1 and "idle clocks head 507 MHz vs worker 2411 MHz" in out
    fakes.flag("head.fail")
    fakes.flag("fake-worker.fail")
    rc, out = run_check(fakes)
    assert rc == 3 and "cannot read any node" in out


def test_check_uses_fresh_state(fakes):
    w = make_watch(fakes)
    import time as _t
    now = _t.time()
    fakes.set("fake-worker", **CLAMP_LOAD)
    fakes.set("head", **HEALTHY_LOAD)
    w.tick(now - 10)
    w.tick(now - 5)
    rc, out = run_check(fakes)
    assert rc == 2 and "watch state" in out and "power under 18 W under load" in out
    # the clamp clears; the state remembers it for --recent seconds
    fakes.set("fake-worker", **HEALTHY_LOAD)
    w.tick(now - 1)
    rc, out = run_check(fakes)
    assert rc == 1 and "cleared since" in out
    rc, _ = run_check(fakes, "--recent", "0")
    assert rc == 0


def test_check_stale_state_falls_back_to_live_but_keeps_recent_slow_state(fakes):
    import time as _t
    d = fakes.state / "gpuwatch"
    d.mkdir(parents=True)
    now = _t.time()
    (d / "gpuwatch.json").write_text(json.dumps({
        "ts": now - 3600, "nodes": {}, "active": {"head:clock_floor": {"severity": "crit", "since": now - 4000,
                                                                       "message": "stale-clamp"}},
        "recent": [{"key": "pair:step_regression", "severity": "warn", "since": now - 200, "until": now - 180,
                    "message": "decode 94 ms"}]}))
    rc, out = run_check(fakes)
    assert rc == 1 and "(live" in out and "wait it out" in out and "stale-clamp" not in out


# -- serve.sh preflight / gpucheck ----------------------------------------------------------------------------------

FAKE_DOCKER = "#!/usr/bin/env bash\nexit 0\n"


@pytest.fixture
def kit(tmp_path, fakes):
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "config").mkdir()
    for f in ("serve.sh", "xid.py", "canary.py", "gpuwatch.py"):
        shutil.copy(ROOT / "scripts" / f, repo / "scripts" / f)
    (repo / "config" / "prod.env").write_text(
        "WORKER_SSH=fake-worker\nHEAD_IP=head-cx7\nNCCL_SOCKET_IFNAME=eth9\nNCCL_IB_HCA=fakehca0\n"
        "HEAD_HF=/tmp/hf\nWORKER_HF=/tmp/hf\nMODEL_PATH=/m\nIMAGE=fake:img\nPORT=9\n")
    docker = tmp_path / "bin" / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    ip = tmp_path / "bin" / "ip"                                  # the preflight's link check: eth9 has an address
    ip.write_text("#!/usr/bin/env bash\necho \"5: eth9    inet 198.51.100.7/24 scope global eth9\"\n")
    ip.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GLM53_TF_", "GPUWATCH_", "WATCH_", "PREFLIGHT"))}
    env.update(STATE_DIR=str(fakes.state))

    def run(*args, **extra):
        return subprocess.run(["bash", str(repo / "scripts/serve.sh"), *args], env=dict(env, **extra), text=True,
                              capture_output=True, timeout=120)

    return run


def test_preflight_passes_healthy_nodes(kit):
    r = kit("preflight")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "preflight: gpuwatch: ok" in r.stdout and "preflight ok" in r.stdout


def test_preflight_refuses_a_clamped_node(kit, fakes):
    fakes.set("head", **HEALTHY_LOAD)
    fakes.set("fake-worker", **CLAMP_LOAD)
    r = kit("preflight")
    assert r.returncode != 0 and "GPU is degraded" in r.stdout and "power clamp" in r.stdout
    r = kit("preflight", GPUWATCH_PREFLIGHT="off")
    assert r.returncode == 0 and "gpuwatch" not in r.stdout


def test_preflight_warns_on_a_spinning_node(kit, fakes):
    fakes.set("fake-worker", **SPIN_IDLE)                         # issue #10: no longer refused as a clamp
    r = kit("preflight")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "warning: GPU state" in r.stdout and "not the clamp" in r.stdout and "GPU is degraded" not in r.stdout


def test_gpucheck_is_strict_for_benchmarks(kit, fakes):
    fakes.set("head", **CLAMP_IDLE)
    fakes.set("fake-worker", sm=2411, power=12.1)
    r = kit("preflight")                                          # a warning: preflight goes on
    assert r.returncode == 0 and "warning: GPU state" in r.stdout
    r = kit("gpucheck")                                           # the benchmark gate refuses
    assert r.returncode != 0 and "GPUWATCH_PREFLIGHT=strict" in r.stdout
    fakes.set("head")
    r = kit("gpucheck")
    assert r.returncode == 0 and "gpucheck ok" in r.stdout
