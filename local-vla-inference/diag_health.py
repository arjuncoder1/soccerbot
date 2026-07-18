"""G1 hardware health check: actuator errors, temperatures, voltage, IMU, DDS rate.

Read-only (subscribes ``rt/lowstate`` only, never publishes). Run it before
inference to confirm every actuator is alive and healthy. Exit code 0 = all
checks pass, 1 = at least one FAIL.

Checks per motor (all 29 joints, arms highlighted):
  - error flags (``motorstate`` / ``sensor`` words nonzero)
  - winding temperature vs warn/fail thresholds
  - bus voltage vs minimum
  - state freshness: q must be finite and the joint must actually report

Plus: rt/lowstate message rate, IMU sanity (finite rpy, gyro not railed,
IMU temperature), motion-switcher mode and loco FSM (same interpretation as
``diag_state.py``).

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

TEMP_WARN_C = 60.0
TEMP_FAIL_C = 80.0
VOLTAGE_MIN_V = 40.0  # G1 13S pack: ~58.8 V full, ~40 V is deeply discharged
GYRO_MAX_RAD_S = 8.0  # standing still; anything near sensor limits is wrong
RATE_MIN_HZ = 100.0  # rt/lowstate nominally streams at 500 Hz

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 hardware health check (read-only).")
    p.add_argument("--iface", default=None, help="Network interface on the robot's network (e.g. enp5s0).")
    p.add_argument("--window", type=float, default=2.0, help="Seconds of lowstate to sample (default 2).")
    p.add_argument("--watch", action="store_true", help="Re-run the check every few seconds until Ctrl+C.")
    p.add_argument("--temp-warn", type=float, default=TEMP_WARN_C, help="Motor temp warning threshold, C.")
    p.add_argument("--temp-fail", type=float, default=TEMP_FAIL_C, help="Motor temp failure threshold, C.")
    return p.parse_args()


def motor_temp(m) -> float:
    """hg MotorState_ temperature is int16[2] (winding/board); take the max."""
    t = m.temperature
    try:
        return float(max(t))
    except TypeError:
        return float(t)


def check_motors(msg, temp_warn: float, temp_fail: float) -> list[tuple[str, str, str]]:
    """Return (status, joint, detail) rows for all 29 joints."""
    rows: list[tuple[str, str, str]] = []
    for name, idx in ALL_JOINTS.items():
        m = msg.motor_state[idx]
        problems: list[str] = []
        status = OK

        err = int(getattr(m, "motorstate", 0))
        if err != 0:
            status = FAIL
            problems.append(f"error flags=0x{err:08x}")
        sensor = getattr(m, "sensor", None)
        if sensor is not None and any(int(s) != 0 for s in sensor):
            status = FAIL
            problems.append(f"sensor flags={[hex(int(s)) for s in sensor]}")

        if not math.isfinite(m.q) or not math.isfinite(m.dq):
            status = FAIL
            problems.append(f"non-finite state q={m.q} dq={m.dq}")

        temp = motor_temp(m)
        if temp >= temp_fail:
            status = FAIL
            problems.append(f"temp {temp:.0f}C >= {temp_fail:.0f}C")
        elif temp >= temp_warn:
            if status == OK:
                status = WARN
            problems.append(f"temp {temp:.0f}C >= {temp_warn:.0f}C")

        vol = float(getattr(m, "vol", 0.0))
        if 0.0 < vol < VOLTAGE_MIN_V:
            if status == OK:
                status = WARN
            problems.append(f"bus voltage {vol:.1f}V < {VOLTAGE_MIN_V:.0f}V")

        detail = "; ".join(problems) if problems else f"temp {temp:.0f}C  q={m.q:+.3f}  tau={m.tau_est:+.2f}"
        rows.append((status, name.removeprefix("k"), detail))
    return rows


def check_imu(msg) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    imu = msg.imu_state
    rpy = [float(v) for v in imu.rpy]
    gyro = [float(v) for v in imu.gyroscope]

    if not all(math.isfinite(v) for v in rpy + gyro):
        rows.append((FAIL, "imu", f"non-finite values rpy={rpy} gyro={gyro}"))
        return rows

    worst_gyro = max(abs(v) for v in gyro)
    status = FAIL if worst_gyro > GYRO_MAX_RAD_S else OK
    rows.append(
        (status, "imu", f"rpy=({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f})  max|gyro|={worst_gyro:.2f} rad/s")
    )

    imu_temp = getattr(imu, "temperature", None)
    if imu_temp is not None:
        t = float(imu_temp)
        rows.append((WARN if t >= TEMP_WARN_C else OK, "imu temp", f"{t:.0f}C"))
    return rows


def check_rate(count: int, window_s: float) -> tuple[str, str, str]:
    hz = count / window_s if window_s > 0 else 0.0
    status = OK if hz >= RATE_MIN_HZ else FAIL
    return (status, "lowstate rate", f"{hz:.0f} Hz over {window_s:.1f}s (min {RATE_MIN_HZ:.0f})")


def check_mode(msc, loco) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    status, result = msc.CheckMode()
    name = (result or {}).get("name", "") if status == 0 else "?"
    rows.append((OK if name else WARN, "motion switcher", name or "NONE (debug mode — re-enable ai_sport)"))
    code, fsm_id = loco.GetFsmId()
    if code != 0:
        rows.append((WARN, "loco FSM", f"GetFsmId failed (code {code})"))
    else:
        desc = FSM_NAMES.get(fsm_id, "unknown")
        rows.append((WARN if fsm_id == 1 else OK, "loco FSM", f"{fsm_id} ({desc})"))
    return rows


def run_check(msc, loco, latest: dict, lock: threading.Lock, counter: dict, args: argparse.Namespace) -> bool:
    with lock:
        counter["n"] = 0
    time.sleep(args.window)
    with lock:
        msg = latest["msg"]
        n = counter["n"]

    headline: list[tuple[str, str, str]] = [check_rate(n, args.window)]
    headline.extend(check_mode(msc, loco))
    headline.extend(check_imu(msg))
    motors = check_motors(msg, args.temp_warn, args.temp_fail)

    print(f"\n=== G1 health @ {time.strftime('%H:%M:%S')} ===")
    for status, item, detail in headline:
        print(f"  [{status:<4}] {item:<20} {detail}")
    # Motors: print only the problematic ones; a healthy robot stays terse.
    for status, item, detail in motors:
        if status != OK:
            print(f"  [{status:<4}] {item:<20} {detail}")

    rows = headline + motors
    n_fail = sum(1 for s, _, _ in rows if s == FAIL)
    n_warn = sum(1 for s, _, _ in rows if s == WARN)
    n_motor_ok = sum(1 for s, _, _ in motors if s == OK)
    print(f"  motors: {n_motor_ok}/{len(motors)} OK")

    hottest = max(
        ((motor_temp(msg.motor_state[i]), n) for n, i in ALL_JOINTS.items()),
        key=lambda x: x[0],
    )
    print(f"  hottest motor: {hottest[1].removeprefix('k')} at {hottest[0]:.0f}C")
    if n_fail:
        print(f"  VERDICT: FAIL ({n_fail} failure(s), {n_warn} warning(s)) — do NOT run inference")
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
    latest: dict = {"msg": None}
    counter = {"n": 0}
    first = threading.Event()

    def on_state(msg) -> None:
        with lock:
            latest["msg"] = msg
            counter["n"] += 1
        first.set()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    print("Waiting for rt/lowstate...", flush=True)
    if not first.wait(timeout=10.0):
        sys.exit("FAIL: no rt/lowstate within 10s — check --iface and robot network.")

    try:
        while True:
            ok = run_check(msc, loco, latest, lock, counter, args)
            if not args.watch:
                sys.exit(0 if ok else 1)
            time.sleep(3.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
