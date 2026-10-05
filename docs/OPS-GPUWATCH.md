# GB10 slow-state / clock watch (`scripts/gpuwatch.py`)

A GB10 can lose most of its speed without an error, an Xid or a clock event reason. Tensor parallel runs in
lockstep, so one degraded Spark slows the pair, and a benchmark taken then measures the fault instead of the code.
`gpuwatch` samples both nodes, alerts on a degraded node, and gates `serve.sh preflight` and benchmark runs.

## What it catches

| Failure | What it looks like | How the watch sees it |
|---|---|---|
| **Clock/power clamp** | SM clock pinned at 507-890 MHz, ~12-14 W at 96 % utilisation, **no clock event reason set**. Survives warm reboots. Seen on the head node, 2026-09-11: 507 MHz, 23.8 vs 94.8 TFLOP/s on the worker (clock logs kept with the earlier vLLM kit setup, not in this repo). NVIDIA forum: "DGX Spark GB10 GPU clock pinned at 721 MHz" (after watchdog reboots). | `clock_floor` / `power_clamp` under load (crit); `idle_asymmetry` / `clock_low_idle` while idle (warn: confirm under load) |
| **Hidden slow state** | GEMV bandwidth falls from 224-233 to 66-80 GB/s for 7-32 s at a time. Decode goes from 63 to 94 ms a step. nvidia-smi shows nothing (tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark issue #1). | `step_regression`: decode ms a round from the server's `/metrics` against a rolling baseline (warn; crit after `--escalate` 120 s) |
| **Lockstep drag** | One node is slower, so the pair runs at its pace. | `asymmetry`: SM clock or power of the two nodes under load |
| Thermal / power throttling | HW slowdown, thermal or power-brake reasons set, or a long SW power cap | `throttle`, `sw_power_cap` (share of the interval from the reason counters), `hot` |
| Unreachable node / nvidia-smi failing | | `unreachable` |

## Signals and cost

Every `--interval` seconds (default 5), in parallel on both nodes:

* `nvidia-smi --query-gpu=clocks.sm,clocks.gr,clocks.mem,clocks.max.sm,clocks.applications.gr,power.draw,temperature.gpu,utilization.gpu,pstate,clocks_event_reasons.active,clocks_event_reasons_counters.{sw_power_cap,sw_thermal_slowdown,hw_thermal_slowdown,hw_power_brake_slowdown}`.
  The worker is queried over one multiplexed ssh connection (ControlMaster). The memory clock and power limits read
  `N/A` on GB10, so they are logged but not used.
* The head's `GET /metrics` (patches/0150). The watch divides the change in `tensorfold_decode_seconds_total` by
  the change in `tensorfold_decode_rounds_total`, over windows of at least `--step-min-rounds` (50) rounds. It keys
  the baseline by the peak `tensorfold_requests_inflight` in the window (1, 2, 3, 4+), because a batched round
  costs more. Baseline = median of the last 30 windows in the same bucket, once 5 exist. The baseline survives a
  restart of the watch (it is kept in the state file).
  * Granularity: the server adds a request's decode time when the request **finishes**. A 7-32 s slow state inside
    one long request is averaged over that request. Short agent turns show it clearly; a 30k-token reply dilutes it.
    Per-round timing inside the server would fix this (a later patch: a histogram of round wall time in `/metrics`).
  * Per-request `round_kinds` (patches/0200) is only in the completion response, so the watch cannot see it.

The watch launches no GPU work. It costs two `nvidia-smi` calls and one local HTTP GET every 5 s. The unit caps it at
10 % CPU and 256 MB.

### The optional bandwidth probe (`--probe`, default `off`)

`/metrics` shows **that** the pair is slow but not **which** node, because the ranks run in lockstep. The probe can
answer that. It runs in its own process at `nice 19` through the CUDA driver API (ctypes, no torch), on a
lowest-priority non-blocking stream. It does 10 device-to-device copies of 64 MiB. That is 1.34 GB of traffic:
about 6 ms at a healthy ~230 GB/s and about 20 ms in the slow state, so well under the 50 ms budget. Each node is
probed at most once a minute, with a 20 s hard timeout.

* `idle`: probe a node only while it is idle (`utilization.gpu` <= 5 % and no request in flight). This catches the
  slow state between requests and needs no server traffic.
* `on-alert`: probe both nodes together right after a `step_regression`, then compare them (`probe_asymmetry`).

Why it is off by default: a second CUDA context shares the GPU by **time-slicing**. Stream priority only orders work
within one process and does not preempt the server. Under load, a probe slows production and measures only the
shared bandwidth. It also adds a CUDA context and 128 MiB to unified memory, where the memory gate is tight
(`MEM_GATE_GIB=108`). Use it for attribution after `/metrics` has flagged a regression, not for monitoring. It has
**not been run on the Sparks yet**, because the GPUs were in use. Before relying on the absolute floor, run it once on
an idle node and record its healthy GB/s (a copy is read + write, so the number differs from the GEMV figures above).

## Thresholds (flags; defaults)

| Condition | Default | Severity | Samples in a row before an alert |
|---|---|---|---|
| loaded (a node counts as busy) | `--load-util 50` % | | |
| `clock_floor` | SM < `--clock-floor 1500` MHz while loaded | crit | 2 |
| `power_clamp` | power < `--power-clamp 18` W while loaded (healthy under load: 42-92 W; 60-62 W on 2026-09-28) | crit | 2 |
| `util_spin` | power < `--power-clamp` W while loaded but SM >= `--clock-floor` MHz: ~96 % utilization from a kernel waiting on a peer (an idle batched follower before patches/0630, or a context left on a node after its rank stopped), not the clamp | warn | 3 |
| `clock_low_idle` | idle SM < `--idle-floor 1000` MHz and the idle reason not set (`0` turns it off) | warn | 6 |
| `idle_asymmetry` | idle, slower / faster SM < `--idle-asym-ratio 0.6` | warn | 6 |
| `asymmetry` | loaded, slower / faster SM < `--asym-ratio 0.85`, or power ratio < `--asym-power-ratio 0.5` | warn | 3 |
| `throttle` | HW slowdown / HW thermal / HW power brake / SW thermal active | warn | 2 |
| `sw_power_cap` | SW power cap held the clocks >= `--swcap-frac 0.5` of the interval, while loaded | warn | 3 |
| `hot` | >= `--hot-c 85` C | warn | 3 |
| `step_regression` | window >= `--step-factor 1.3` x baseline | warn, crit after `--escalate 120` s | 1 window |
| `probe_slow` | probe < `--probe-floor-gbs 120` and < 0.6 x the node's median | warn | 1 |

An active condition re-alerts every `--realert 600` s. When it clears, the watch logs `CLEARED ... after N s`, which
records how long a slow state lasted.

Reference readings on 2026-09-28 (driver 580.178.04, `-lgc 300,2250`):
* Idle with the server loaded: 2229 / 2236 MHz, 11-13 W, 53-57 C.
* Under load: 2223-2236 MHz, 60-62 W, 79-83 C, 96 %.
* No reason bits set.

## Outputs

* Log lines on stdout (journald under systemd):
  * `ALERT [crit|warn] <node>:<kind>: ...`
  * `STILL ...`
  * `CLEARED ...`
* Webhook: set `GPUWATCH_WEBHOOK=<url>` to POST `{"text", "content", "severity", "key"}` for each alert.
  Both `text` and `content` are sent, so Slack- and Discord-style hooks both accept it.
* Alert command: set `WATCH_ALERT` (shared with the serve.sh watchdog) or `GPUWATCH_ALERT_CMD`. The command runs
  with the message as `$1`.
* `$STATE_DIR/gpuwatch/gpuwatch-YYYYMMDD.csv`: one row per node per sample, including clocks, power, reasons, SW cap
  share, step ms and baseline, and the conditions. Files older than 14 days are deleted.
* `$STATE_DIR/gpuwatch/gpuwatch.json`: the last samples, active conditions, the last 50 cleared ones, and the
  baselines. `STATE_DIR` defaults to `~/.local/state/glm53-tf`, the same as serve.sh.

## Preflight and benchmark gate

`scripts/gpuwatch.py check` exit codes:

| Code | Meaning |
|---|---|
| 0 | ok |
| 1 | warning |
| 2 | degraded |
| 3 | cannot read any node |

If the watch's state is at most `--max-age 60` s old, `check` uses it. Otherwise it takes 3 live samples; a condition
counts when it appears in 2 of the 3. With a live check, the last 10 minutes of cleared conditions from an old state
file still count as warnings (`--recent 600`).

* `serve.sh preflight` (and `start` with `PREFLIGHT=warn|strict`) runs it when `GPUWATCH_PREFLIGHT` is set:
  * `on` (default): **degraded refuses**; a warning is logged.
  * `strict`: a warning refuses too.
  * `off`: skip the check.
* `serve.sh gpucheck` is the benchmark gate. It is strict unless `GPUWATCH_PREFLIGHT=off`, so a recent slow state
  or an idle clock asymmetry also refuses:

  ```bash
  scripts/serve.sh gpucheck && python3 bench/glmbench.py --base http://127.0.0.1:8000 ...
  ```

An idle check cannot prove a clamp is absent: the 2026-09-11 clamp read 507 MHz idle as well as loaded, but a
clamp that shows only under load passes an idle check. With the watch running during the benchmark, a clamp under
load alerts within about 10 s. When in doubt, look at the CSV for the benchmark window afterwards; rows whose
`conditions` column is not empty invalidate that window.

## What to do

**`util_spin` (warn)**: not the clamp. With the server up and idle, it is the batched follower waiting in a control
collective (fixed by patches/0630). With both ranks stopped and no process on the GPU (`nvidia-smi
--query-compute-apps=pid --format=csv,noheader` empty), a context was left behind on that node: `sudo systemctl
restart nvidia-persistenced` there clears it in a few seconds (re-apply the clock cap afterwards: the restart resets
it). No power drain needed.

**`power_clamp` / `clock_floor` (crit)**: the clamp. It does not clear by itself, and a **warm reboot does not
clear it** (proven 2026-09-11).
1. Stop serving: `scripts/serve.sh stop`. Note the node, its boot id (`cat /proc/sys/kernel/random/boot_id`), and
   the CSV rows.
2. **Power-drain the clamped node:**
   1. `sudo shutdown -h now`.
   2. Once the LEDs are off, unplug the USB-C power supply from the wall (not only from the Spark).
   3. Wait at least 60 s (NVIDIA forum advice ranges from 30 s to a few minutes).
   4. Reconnect and power on.

   If both nodes went through watchdog reboots, drain both.
3. After boot, check that `dgx-gpu-clock-cap.service` is active (`nvidia-smi -lgc 300,2250`), then run
   `scripts/serve.sh gpucheck`. Then start the server and confirm under load, with the watch running: expect
   >= 2000 MHz and >= 40 W at >= 90 % utilisation.
4. If the clamp returns after a drain, check the power supply and cable first (the 240 W USB-C PD input cannot be
   read from software), then contact NVIDIA support / RMA.

**`step_regression` (warn)**: probably the transient slow state. **Wait it out**: episodes last 7-32 s. Do not
restart anything; a restart costs minutes and does not help.
* For a benchmark, discard the affected window and re-run once `CLEARED` is logged. `gpucheck` refuses for 10
  minutes after it.
* If it **escalates to crit** (still there after 120 s), treat it as a clamp and check the `clock_floor` /
  `power_clamp` / `asymmetry` rows.
* If clocks and power look normal, turn on `GPUWATCH_PROBE=on-alert` for a while to find the slow node.
* Frequent episodes: record the timestamps from the CSV and add them to the upstream issue.

**`asymmetry` (warn)**: one node is slower under the same load. If the slower node's power is low, see the clamp
steps. If its temperature is high, check its airflow. Persistent asymmetry with normal power and temperature:
compare `clocks.applications.gr` and the clock lock (`systemctl status dgx-gpu-clock-cap`) on both nodes.

**`idle_asymmetry` / `clock_low_idle` (warn)**: this looks like the clamp but can also be a legitimately parked
GPU (the clock lock allows 300 MHz). Confirm under load before draining. If it is a false positive, set
`--idle-floor 0`.

**`throttle` / `sw_power_cap` / `hot` (warn)**: look at airflow and ambient temperature. SW power capping
accumulates on both nodes normally (about 7 % of the time over a day); only a sustained share while loaded is
reported.

**`unreachable`**: the worker's ssh or nvidia-smi is failing. Run `scripts/serve.sh xid` to check for Xid events
(an Xid 79 "fallen off the bus" needs a power cycle).

## Install (head node, user unit)

```bash
cd ~/glm53-tensorfold-spark
python3 scripts/gpuwatch.py check                                      # one read-only check first
cp scripts/systemd/glm53-gpuwatch.{service,timer} ~/.config/systemd/user/
$EDITOR ~/.config/systemd/user/glm53-gpuwatch.service                  # CONFIG, webhook
loginctl enable-linger "$USER"                                         # already on if the watchdog is installed
systemctl --user daemon-reload && systemctl --user enable --now glm53-gpuwatch.timer
journalctl --user -u glm53-gpuwatch -f
```

The unit reads `WORKER_SSH` and `PORT` from `CONFIG`. Keep `CONFIG` in line with the config that is serving
(`config/prod-batch.env` now): the watch needs the right port to find `/metrics`, and without `/metrics` the
slow-state signal is gone.

Tests: `python -m pytest -q tests/test_gpuwatch.py` (host only; fake nvidia-smi, ssh and `/metrics`).
