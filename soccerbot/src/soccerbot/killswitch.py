"""G1 killswitch — CLI (default) and optional Tk GUI.

Actions (same as physical pendant where applicable):

  stop   — LocoClient.StopMove() (stay standing)
  damp   — LocoClient.Damp()           (pendant L2+B)
  zero   — LocoClient.ZeroTorque()     (pendant L2+A)
  start  — LocoClient.Start()          (re-engage balancer)
  home   — slew-clamped go-to-home arm pose (zeros / home_pose.json)

Usage:

    # CLI interactive (no GUI / no tkinter required)
    ./killswitch.sh --iface enp5s0
    python -m soccerbot.killswitch --iface enp5s0

    # CLI one-shot
    ./killswitch.sh --iface enp5s0 stop
    ./killswitch.sh --iface enp5s0 home

    # Tk GUI
    ./killswitch.sh --gui --iface enp5s0
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

from soccerbot.home import go_home
from soccerbot.safety import balance_stand, enter_damp, enter_zero_torque, init_loco, stop_loco

logger = logging.getLogger("soccerbot.killswitch")

FSM_NAMES = {
    0: "zero torque",
    1: "damp",
    2: "squat",
    3: "sit",
    4: "stand (locked)",
    200: "start / balance stand",
    500: "advanced (main operation)",
}

ACTIONS = ("stop", "damp", "zero", "start", "home", "status")


def _fsm_status(loco) -> str:
    try:
        code, fsm_id = loco.GetFsmId()
        name = FSM_NAMES.get(int(fsm_id), f"fsm {fsm_id}")
        return f"FSM={fsm_id} ({name}) rpc={code}"
    except Exception as exc:  # noqa: BLE001
        return f"FSM read failed: {exc}"


def run_action(action: str, *, iface: str | None, loco=None) -> None:
    """Execute one killswitch / home action."""
    action = action.lower().strip()
    if action == "stop":
        stop_loco(loco, iface=iface)
    elif action == "damp":
        enter_damp(iface=iface, loco=loco)
    elif action == "zero":
        enter_zero_torque(iface=iface, loco=loco)
    elif action == "start":
        balance_stand(iface=iface, loco=loco)
    elif action == "home":
        go_home(iface=iface, release_after=False)
    elif action == "status":
        client = loco or init_loco(iface)
        print(_fsm_status(client), flush=True)
    else:
        raise ValueError(f"unknown action {action!r}; choose from {ACTIONS}")


# ---------------------------------------------------------------------------
# CLI (no GUI)
# ---------------------------------------------------------------------------


def run_cli(iface: str | None, action: str | None = None) -> int:
    """One-shot action, or interactive prompt loop."""
    loco = None
    try:
        loco = init_loco(iface)
        print(f"killswitch CLI armed  iface={iface or '(default)'}  {_fsm_status(loco)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DDS connect deferred (%s); will retry on first command", exc)

    if action:
        run_action(action, iface=iface, loco=loco)
        return 0

    print(
        "Commands: stop | damp | zero | start | home | status | quit\n"
        "  stop  = StopMove (stay standing)\n"
        "  damp  = Damp (L2+B)\n"
        "  zero  = ZeroTorque (L2+A — limp)\n"
        "  start = balancer Start\n"
        "  home  = arms go home (slew-clamped)\n",
        flush=True,
    )
    while True:
        try:
            line = input("killswitch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            return 130
        if not line:
            continue
        cmd = line.split()[0].lower()
        if cmd in ("q", "quit", "exit"):
            return 0
        if cmd in ("help", "?"):
            print("Commands:", ", ".join(ACTIONS), "| quit", flush=True)
            continue
        if cmd == "damp":
            confirm = input("Enter Damp? [y/N] ").strip().lower()
            if confirm not in ("y", "yes"):
                continue
        if cmd == "zero":
            confirm = input("Enter ZeroTorque (limp)? Spotter ready? [y/N] ").strip().lower()
            if confirm not in ("y", "yes"):
                continue
        try:
            if loco is None:
                loco = init_loco(iface)
            run_action(cmd, iface=iface, loco=loco)
            if loco is not None and cmd != "status":
                print(_fsm_status(loco), flush=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed", cmd)
            print(f"ERROR: {exc}", flush=True)


# ---------------------------------------------------------------------------
# GUI (tkinter)
# ---------------------------------------------------------------------------


def _require_tkinter():
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "tkinter is required for --gui.\n"
            "Install: sudo apt-get install -y python3-tk\n"
            "Or use the CLI (default): ./killswitch.sh --iface enp5s0\n"
            f"Original error: {exc}"
        ) from exc
    return tk, messagebox, ttk


class KillswitchApp:
    def __init__(self, iface: str | None) -> None:
        tk, messagebox, ttk = _require_tkinter()
        self._tk = tk
        self._messagebox = messagebox
        self.iface = iface
        self._loco = None
        self._lock = threading.Lock()

        self.root = tk.Tk()
        self.root.title("G1 KILLSWITCH")
        self.root.configure(bg="#1a1a1a")
        self.root.geometry("520x640")
        self.root.minsize(480, 600)

        tk.Label(
            self.root,
            text="G1 KILLSWITCH",
            font=("Helvetica", 28, "bold"),
            fg="#ff4444",
            bg="#1a1a1a",
        ).pack(pady=(16, 4))

        tk.Label(
            self.root,
            text=f"iface: {iface or '(default DDS iface)'}",
            font=("Helvetica", 11),
            fg="#aaaaaa",
            bg="#1a1a1a",
        ).pack()

        self.status = tk.Label(
            self.root,
            text="Connecting…",
            font=("Helvetica", 13),
            fg="#eeeeee",
            bg="#1a1a1a",
            wraplength=480,
            justify="center",
        )
        self.status.pack(pady=12)

        btn_frame = tk.Frame(self.root, bg="#1a1a1a")
        btn_frame.pack(fill="both", expand=True, padx=24, pady=8)

        self._mk_button(btn_frame, "STOP MOVE\n(stay standing)", "#cc8800", self._on_stop).pack(
            fill="x", pady=6
        )
        self._mk_button(btn_frame, "GO HOME\n(arms → home pose)", "#2266aa", self._on_home).pack(
            fill="x", pady=6
        )
        self._mk_button(btn_frame, "DAMP\n(L2+B)", "#dd6622", self._on_damp).pack(fill="x", pady=6)
        self._mk_button(btn_frame, "ZERO TORQUE\n(L2+A — limp)", "#cc2222", self._on_zero).pack(
            fill="x", pady=6
        )
        self._mk_button(btn_frame, "START / STAND\n(re-engage balancer)", "#227744", self._on_start).pack(
            fill="x", pady=6
        )

        ttk.Separator(self.root).pack(fill="x", padx=24, pady=8)
        tk.Label(
            self.root,
            text="Keep this window open during demos.\n"
            "CLI mode (no GUI): ./killswitch.sh --iface …\n"
            "Physical pendant still works in parallel.",
            font=("Helvetica", 10),
            fg="#888888",
            bg="#1a1a1a",
            justify="center",
        ).pack(pady=(0, 12))

        self.root.after(100, self._connect_async)
        self.root.after(1000, self._poll_fsm)

    def _mk_button(self, parent, text: str, color: str, command):
        return self._tk.Button(
            parent,
            text=text,
            font=("Helvetica", 16, "bold"),
            fg="#ffffff",
            bg=color,
            activebackground=color,
            activeforeground="#ffffff",
            relief="raised",
            bd=4,
            height=2,
            command=command,
        )

    def _connect_async(self) -> None:
        def worker() -> None:
            try:
                loco = init_loco(self.iface)
                with self._lock:
                    self._loco = loco
                self._set_status(f"DDS connected — {_fsm_status(loco)}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("killswitch connect failed")
                self._set_status(f"CONNECT FAILED: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _set_status(self, text: str) -> None:
        def apply() -> None:
            self.status.config(text=text)

        self.root.after(0, apply)

    def _with_loco(self, fn, label: str) -> None:
        def worker() -> None:
            with self._lock:
                loco = self._loco
            try:
                fn(loco)
                with self._lock:
                    loco = self._loco
                extra = f"  {_fsm_status(loco)}" if loco is not None else ""
                self._set_status(f"{label} OK @ {time.strftime('%H:%M:%S')}{extra}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s failed", label)
                self._set_status(f"{label} FAILED: {exc}")
                self.root.after(
                    0, lambda: self._messagebox.showerror("Killswitch", f"{label} failed:\n{exc}")
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_stop(self) -> None:
        self._with_loco(lambda loco: stop_loco(loco, iface=self.iface), "STOP MOVE")

    def _on_home(self) -> None:
        if not self._messagebox.askokcancel(
            "GO HOME",
            "Move arms to home pose (slew-clamped)?\nSpotter should be ready.",
        ):
            return
        self._with_loco(lambda _loco: go_home(iface=self.iface, release_after=False), "GO HOME")

    def _on_damp(self) -> None:
        if not self._messagebox.askokcancel("DAMP", "Enter Damp mode? Robot will go passive."):
            return
        self._with_loco(lambda loco: enter_damp(iface=self.iface, loco=loco), "DAMP")

    def _on_zero(self) -> None:
        if not self._messagebox.askokcancel(
            "ZERO TORQUE",
            "Enter ZeroTorque? Motors go limp — spotter must be ready.",
        ):
            return
        self._with_loco(lambda loco: enter_zero_torque(iface=self.iface, loco=loco), "ZERO TORQUE")

    def _on_start(self) -> None:
        self._with_loco(lambda loco: balance_stand(iface=self.iface, loco=loco), "START")

    def _poll_fsm(self) -> None:
        def worker() -> None:
            with self._lock:
                loco = self._loco
            if loco is None:
                return
            self._set_status(_fsm_status(loco))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(1000, self._poll_fsm)

    def run(self) -> None:
        self.root.mainloop()


def run_gui(iface: str | None) -> int:
    KillswitchApp(iface).run()
    return 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="G1 killswitch (CLI by default; --gui for Tk panel).",
    )
    p.add_argument(
        "--iface",
        default=None,
        help="DDS network interface (e.g. enp5s0). Omit for SDK default.",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Open the Tk killswitch window (needs python3-tk).",
    )
    p.add_argument(
        "action",
        nargs="?",
        default=None,
        choices=list(ACTIONS),
        help="Optional one-shot CLI action (stop/damp/zero/start/home/status).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    try:
        if args.gui:
            if args.action:
                logger.warning("Ignoring one-shot action %r in --gui mode", args.action)
            return run_gui(args.iface)
        return run_cli(args.iface, action=args.action)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
