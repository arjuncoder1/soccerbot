"""G1 hardware health check: exhaustive, read-only, run before inference.

Subscribes ``rt/lowstate`` only (never publishes), samples every message for a
few seconds, then runs a large battery of checks. Exit code 0 = all pass,
1 = at least one FAIL.

Per motor (all 29 joints, ~14 checks each, ~400 total):
  - error flags (``motorstate``) and sensor fault words
  - q / dq / tau finite in every sampled message
  - winding temperature vs warn/fail thresholds + temperature RISE during the window
  - bus voltage vs minimum
  - position inside URDF joint limits (+ near-limit warning)
  - encoder glitch (impossible q jump between consecutive messages)
  - stuck encoder (dq says moving but q frozen)
  - excessive velocity / torque at rest

IMU: finite rpy/gyro/accel, gyro not railed, gyro vibration (std), gravity
magnitude, quaternion norm, tilt, IMU temperature.

System: rt/lowstate rate, tick monotonicity and max gap, motor array length,
motion-switcher mode and loco FSM (same interpretation as ``diag_state.py``).

Usage:

    ./local-vla-inference/run.sh diag_health.py --iface enp5s0
    ./local-vla-inference/run.sh diag_health.py --iface enp5s0 --watch
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time

from diag_state import FSM_NAMES
from g1_arms import ARM_JOINT_INDEX, LEG_JOINT_INDEX

# Remaining joints on the 29-DoF G1 not covered by the arm/leg maps.
OTHER_JOINT_INDEX: dict[str, int] = {
    "kWaistRoll": 13,
    "kWaistPitch": 14,
}

ALL_JOINTS: dict[str, int] = {**LEG_JOINT_INDEX, **OTHER_JOINT_INDEX, **ARM_JOINT_INDEX}

# Joint limits (rad) from the Unitree g1_description 29-DoF URDF. Approximate;
# used with a small margin so firmware calibration offsets don't false-alarm.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "kLeftHipPitch": (-2.5307, 2.8798),
    "kLeftHipRoll": (-0.5236, 2.9671),
    "kLeftHipYaw": (-2.7576, 2.7576),
    "kLeftKnee": (-0.087267, 2.8798),
    "kLeftAnklePitch": (-0.87267, 0.5236),
    "kLeftAnkleRoll": (-0.2618, 0.2618),
    "kRightHipPitch": (-2.5307, 2.8798),
    "kRightHipRoll": (-2.9671, 0.5236),
    "kRightHipYaw": (-2.7576, 2.7576),
    "kRightKnee": (-0.087267, 2.8798),
    "kRightAnklePitch": (-0.87267, 0.5236),
    "kRightAnkleRoll": (-0.2618, 0.2618),
    "kWaistYaw": (-2.618, 2.618),
    "kWaistRoll": (-0.52, 0.52),
    "kWaistPitch": (-0.52, 0.52),
    "kLeftShoulderPitch": (-3.0892, 2.6704),
    "kLeftShoulderRoll": (-1.5882, 2.2515),
    "kLeftShoulderYaw": (-2.618, 2.618),
    "kLeftElbow": (-1.0472, 2.0944),
    "kLeftWristRoll": (-1.9722, 1.9722),
    "kLeftWristPitch": (-1.6144, 1.6144),
    "kLeftWristYaw": (-1.6144, 1.6144),
    "kRightShoulderPitch": (-3.0892, 2.6704),
    "kRightShoulderRoll": (-2.2515, 1.5882),
    "kRightShoulderYaw": (-2.618, 2.618),
    "kRightElbow": (-1.0472, 2.0944),
    "kRightWristRoll": (-1.9722, 1.9722),
    "kRightWristPitch": (-1.6144, 1.6144),
    "kRightWristYaw": (-1.6144, 1.6144),
}

TEMP_WARN_C = 60.0
TEMP_FAIL_C = 80.0
TEMP_RISE_WARN_C = 3.0  # heating this fast during a short idle window is wrong
VOLTAGE_MIN_V = 40.0  # G1 13S pack: ~58.8 V full, ~40 V deeply discharged
LIMIT_MARGIN_RAD = 0.05  # tolerance beyond URDF limit before FAIL
NEAR_LIMIT_RAD = 0.05  # within this of a limit -> WARN
ENCODER_JUMP_RAD = 0.5  # impossible q step between consecutive 500 Hz messages
STUCK_DQ_RAD_S = 0.3  # dq above this with a frozen q -> encoder suspect
DQ_WARN_RAD_S = 2.0  # robot should be ~still during a health check
DQ_FAIL_RAD_S = 8.0
TAU_WARN_NM = 50.0
TAU_FAIL_NM = 80.0
GYRO_MAX_RAD_S = 8.0
GYRO_STD_WARN = 0.5  # sustained vibration/oscillation while standing
ACCEL_G_TOL = 3.0  # |accel| must be within this of 9.81 m/s^2
TILT_WARN_RAD = 0.35
RATE_MIN_HZ = 100.0  # rt/lowstate nominally streams at 500 Hz
TICK_GAP_WARN_MS = 50.0

OK, WARN, FAIL = "OK", "WARN", "FAIL"

Row = tuple[str, str, str]  # (status, item, detail)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 exhaustive hardware health check (read-only).")
    p.add_argument("--iface", default=None, help="Network interface on the robot's network (e.g. enp5s0).")
    p.add_argument("--window", type=float, default=2.0, help="Seconds of lowstate to sample (default 2).")
    p.add_argument("--watch", action="store_true", help="Re-run the check every few seconds until Ctrl+C.")
    p.add_argument("--temp-warn", type=float, default=TEMP_WARN_C, help="Motor temp warning threshold, C.")
    p.add_argument("--temp-fail", type=float, default=TEMP_FAIL_C, help="Motor temp failure threshold, C.")
    p.add_argument("--verbose", action="store_true", help="Print every motor row, not just problems.")
    return p.parse_args()


def motor_temp(m) -> float:
    """hg MotorState_ temperature is int16[2] (winding/board); take the max."""
    t = m.temperature
    try:
        return float(max(t))
    except TypeError:
        return float(t)


def extract(msg) -> dict:
    """Pull everything we check out of one LowState_ message (cheap copies)."""
    sample: dict = {
        "tick": int(getattr(msg, "tick", 0)),
        "n_motors": len(msg.motor_state),
        "q": {}, "dq": {}, "tau": {}, "temp": {}, "vol": {}, "flags": {}, "sensor": {},
    }
    for name, idx in ALL_JOINTS.items():
        m = msg.motor_state[idx]
        sample["q"][name] = float(m.q)
        sample["dq"][name] = float(m.dq)
        sample["tau"][name] = float(m.tau_est)
        sample["temp"][name] = motor_temp(m)
        sample["vol"][name] = float(getattr(m, "vol", 0.0))
        sample["flags"][name] = int(getattr(m, "motorstate", 0))
        s = getattr(m, "sensor", None)
        sample["sensor"][name] = tuple(int(v) for v in s) if s is not None else ()
    imu = msg.imu_state
    sample["rpy"] = [float(v) for v in imu.rpy]
    sample["gyro"] = [float(v) for v in imu.gyroscope]
    sample["accel"] = [float(v) for v in getattr(imu, "accelerometer", (0.0, 0.0, 9.81))]
    sample["quat"] = [float(v) for v in getattr(imu, "quaternion", (1.0, 0.0, 0.0, 0.0))]
    sample["imu_temp"] = float(getattr(imu, "temperature", 0.0))
    return sample


class Tally:
    """Collects check rows and counts how many individual checks ran."""

    def __init__(self) -> None:
        self.rows: list[Row] = []
        self.n_checks = 0

    def check(self, ok: bool, item: str, detail: str, level: str = FAIL) -> bool:
        self.n_checks += 1
        if not ok:
            self.rows.append((level, item, detail))
        return ok


def check_motor(t: Tally, name: str, samples: list[dict], args: argparse.Namespace) -> bool:
    """~14 checks for one motor over the whole sample window. True if no FAIL."""
    short = name.removeprefix("k")
    qs = [s["q"][name] for s in samples]
    dqs = [s["dq"][name] for s in samples]
    taus = [s["tau"][name] for s in samples]
    temps = [s["temp"][name] for s in samples]
    vols = [s["vol"][name] for s in samples]
    fails_before = sum(1 for r in t.rows if r[0] == FAIL)

    flags = {s["flags"][name] for s in samples} - {0}
    t.check(not flags, short, f"error flags {[hex(f) for f in flags]}")
    sensors = {s["sensor"][name] for s in samples} - {(), (0,), (0, 0)}
    t.check(not sensors, short, f"sensor fault words {sensors}")

    finite = True
    finite &= t.check(all(math.isfinite(q) for q in qs), short, "non-finite q in window")
    finite &= t.check(all(math.isfinite(d) for d in dqs), short, "non-finite dq in window")
    finite &= t.check(all(math.isfinite(x) for x in taus), short, "non-finite tau in window")
    if not finite:
        return False  # garbage numbers; skip the numeric checks below

    tmax, trise = max(temps), temps[-1] - temps[0]
    t.check(tmax < args.temp_fail, short, f"temp {tmax:.0f}C >= {args.temp_fail:.0f}C")
    t.check(tmax < args.temp_warn or tmax >= args.temp_fail, short,
            f"temp {tmax:.0f}C >= {args.temp_warn:.0f}C", WARN)
    t.check(trise < TEMP_RISE_WARN_C, short, f"temp rose {trise:.1f}C during {len(samples)}-msg window", WARN)

    vmin = min(vols)
    t.check(not (0.0 < vmin < VOLTAGE_MIN_V), short, f"bus voltage {vmin:.1f}V < {VOLTAGE_MIN_V:.0f}V", WARN)

    lo, hi = JOINT_LIMITS[name]
    qmin, qmax = min(qs), max(qs)
    t.check(qmin >= lo - LIMIT_MARGIN_RAD and qmax <= hi + LIMIT_MARGIN_RAD, short,
            f"q [{qmin:+.3f},{qmax:+.3f}] outside limits [{lo:+.3f},{hi:+.3f}]")
    t.check(qmin >= lo + NEAR_LIMIT_RAD and qmax <= hi - NEAR_LIMIT_RAD, short,
            f"q [{qmin:+.3f},{qmax:+.3f}] within {NEAR_LIMIT_RAD} rad of limit [{lo:+.3f},{hi:+.3f}]", WARN)

    max_jump = max((abs(b - a) for a, b in zip(qs, qs[1:])), default=0.0)
    t.check(max_jump < ENCODER_JUMP_RAD, short, f"encoder glitch: q jumped {max_jump:.3f} rad between messages")
    peak_dq = max(abs(d) for d in dqs)
    t.check(not (peak_dq > STUCK_DQ_RAD_S and (qmax - qmin) < 1e-5), short,
            f"stuck encoder? dq up to {peak_dq:.2f} rad/s but q frozen", WARN)

    t.check(peak_dq < DQ_FAIL_RAD_S, short, f"velocity {peak_dq:.1f} rad/s >= {DQ_FAIL_RAD_S} at rest")
    t.check(peak_dq < DQ_WARN_RAD_S or peak_dq >= DQ_FAIL_RAD_S, short,
            f"velocity {peak_dq:.1f} rad/s >= {DQ_WARN_RAD_S} at rest", WARN)
    peak_tau = max(abs(x) for x in taus)
    t.check(peak_tau < TAU_FAIL_NM, short, f"torque {peak_tau:.0f} Nm >= {TAU_FAIL_NM} at rest")
    t.check(peak_tau < TAU_WARN_NM or peak_tau >= TAU_FAIL_NM, short,
            f"torque {peak_tau:.0f} Nm >= {TAU_WARN_NM} at rest", WARN)

    return sum(1 for r in t.rows if r[0] == FAIL) == fails_before


def check_imu(t: Tally, samples: list[dict]) -> None:
    last = samples[-1]
    rpy, gyros = last["rpy"], [s["gyro"] for s in samples]
    flat = [v for g in gyros for v in g] + rpy + last["accel"] + last["quat"]
    if not t.check(all(math.isfinite(v) for v in flat), "imu", "non-finite IMU values"):
        return

    peak = max(abs(v) for g in gyros for v in g)
    t.check(peak < GYRO_MAX_RAD_S, "imu gyro", f"max|gyro|={peak:.2f} rad/s (railed?)")
    for axis in range(3):
        vals = [g[axis] for g in gyros]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        t.check(std < GYRO_STD_WARN, "imu gyro", f"axis {axis} vibrating: std {std:.2f} rad/s", WARN)

    amag = math.sqrt(sum(v * v for v in last["accel"]))
    t.check(abs(amag - 9.81) < ACCEL_G_TOL, "imu accel", f"|accel|={amag:.1f} m/s^2 (gravity missing?)")
    qnorm = math.sqrt(sum(v * v for v in last["quat"]))
    t.check(abs(qnorm - 1.0) < 0.05, "imu quat", f"quaternion norm {qnorm:.3f} != 1")
    t.check(abs(rpy[0]) < TILT_WARN_RAD and abs(rpy[1]) < TILT_WARN_RAD, "imu tilt",
            f"roll={rpy[0]:+.2f} pitch={rpy[1]:+.2f} rad — robot not upright?", WARN)
    t.check(last["imu_temp"] < TEMP_WARN_C, "imu temp", f"{last['imu_temp']:.0f}C", WARN)


def check_stream(t: Tally, samples: list[dict], window_s: float) -> None:
    hz = len(samples) / window_s if window_s > 0 else 0.0
    t.check(hz >= RATE_MIN_HZ, "lowstate rate", f"{hz:.0f} Hz over {window_s:.1f}s (min {RATE_MIN_HZ:.0f})")

    ticks = [s["tick"] for s in samples]
    diffs = [b - a for a, b in zip(ticks, ticks[1:])]
    if any(d != 0 for d in diffs):  # some firmware leaves tick at 0
        t.check(all(d >= 0 for d in diffs), "lowstate tick", "tick went backwards (duplicate/reordered messages)")
        t.check(max(diffs) <= TICK_GAP_WARN_MS, "lowstate tick", f"gap of {max(diffs)} ms between messages", WARN)
    t.check(min(s["n_motors"] for s in samples) >= 30, "lowstate shape",
            f"motor_state has {samples[-1]['n_motors']} entries (< 30)")


def check_mode(t: Tally, msc, loco) -> str:
    status, result = msc.CheckMode()
    name = (result or {}).get("name", "") if status == 0 else "?"
    t.check(bool(name), "motion switcher", "NONE (debug mode — re-enable ai_sport)", WARN)
    code, fsm_id = loco.GetFsmId()
    if not t.check(code == 0, "loco FSM", f"GetFsmId failed (code {code})", WARN):
        return f"switcher={name or 'NONE'} fsm=?"
    desc = FSM_NAMES.get(fsm_id, "unknown")
    t.check(fsm_id != 1, "loco FSM", "robot is in DAMP — stand up before inference", WARN)
    return f"switcher={name or 'NONE'} fsm={fsm_id} ({desc})"


def run_check(msc, loco, collector: dict, lock: threading.Lock, args: argparse.Namespace) -> bool:
    with lock:
        collector["samples"] = []
        collector["on"] = True
    time.sleep(args.window)
    with lock:
        collector["on"] = False
        samples = collector["samples"]

    print(f"\n=== G1 health @ {time.strftime('%H:%M:%S')} ===")
    if not samples:
        print("  [FAIL] lowstate            no messages during sample window")
        return False

    t = Tally()
    mode_desc = check_mode(t, msc, loco)
    check_stream(t, samples, args.window)
    check_imu(t, samples)
    motors_ok = 0
    for name in ALL_JOINTS:
        if check_motor(t, name, samples, args):
            motors_ok += 1

    for status, item, detail in t.rows:
        print(f"  [{status:<4}] {item:<20} {detail}")

    last = samples[-1]
    hot = max(last["temp"], key=last["temp"].get)
    fast = max(last["dq"], key=lambda k: abs(last["dq"][k]))
    strong = max(last["tau"], key=lambda k: abs(last["tau"][k]))
    print(f"  {mode_desc}")
    print(f"  {len(samples)} msgs sampled | motors {motors_ok}/{len(ALL_JOINTS)} OK")
    print(
        f"  hottest {hot.removeprefix('k')} {last['temp'][hot]:.0f}C | "
        f"max|dq| {abs(last['dq'][fast]):.2f} rad/s ({fast.removeprefix('k')}) | "
        f"max|tau| {abs(last['tau'][strong]):.1f} Nm ({strong.removeprefix('k')})"
    )
    if args.verbose:
        for name in ALL_JOINTS:
            print(
                f"    {name.removeprefix('k'):<20} q={last['q'][name]:+.4f}  dq={last['dq'][name]:+.3f}  "
                f"tau={last['tau'][name]:+.2f}  temp={last['temp'][name]:.0f}C  vol={last['vol'][name]:.1f}V"
            )

    n_fail = sum(1 for s, _, _ in t.rows if s == FAIL)
    n_warn = sum(1 for s, _, _ in t.rows if s == WARN)
    print(f"  total: {t.n_checks} checks run, {n_fail} failure(s), {n_warn} warning(s)")
    if n_fail:
        print("  VERDICT: FAIL — do NOT run inference")
    elif n_warn:
        print(f"  VERDICT: OK with {n_warn} warning(s)")
    else:
        print("  VERDICT: all checks passed — OK to run inference")
    return n_fail == 0


def main() -> None:
    args = parse_args()

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    msc = MotionSwitcherClient()
    msc.SetTimeout(3.0)
    msc.Init()

    loco = LocoClient()
    loco.SetTimeout(3.0)
    loco.Init()

    lock = threading.Lock()
    collector: dict = {"on": False, "samples": []}
    first = threading.Event()

    def on_state(msg) -> None:
        with lock:
            if collector["on"]:
                collector["samples"].append(extract(msg))
        first.set()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    print("Waiting for rt/lowstate...", flush=True)
    if not first.wait(timeout=10.0):
        sys.exit("FAIL: no rt/lowstate within 10s — check --iface and robot network.")

    try:
        while True:
            ok = run_check(msc, loco, collector, lock, args)
            if not args.watch:
                sys.exit(0 if ok else 1)
            time.sleep(3.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
