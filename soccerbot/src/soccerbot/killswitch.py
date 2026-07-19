"""Headed killswitch panel for the Unitree G1.

Big on-screen buttons (tkinter) that map to the same safety actions as the
physical pendant:

  STOP / STAND  — StopMove + (optional) leave balancer standing
  DAMP          — LocoClient.Damp()           (pendant L2+B)
  ZERO TORQUE   — LocoClient.ZeroTorque()     (pendant L2+A)
  START         — LocoClient.Start()          (re-engage balancer)

Run separately from the demo so it stays usable even if the policy process
hangs:

    ./killswitch.sh --iface enp5s0
    python -m soccerbot.killswitch --iface enp5s0
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

from soccerbot.safety import balance_stand, enter_damp, enter_zero_torque, stop_loco

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


def _require_tkinter():
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "tkinter is required for the headed killswitch.\n"
            "Install: sudo apt-get install -y python3-tk\n"
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
        self.root.geometry("520x560")
        self.root.minsize(480, 520)

        title = tk.Label(
            self.root,
            text="G1 KILLSWITCH",
            font=("Helvetica", 28, "bold"),
            fg="#ff4444",
            bg="#1a1a1a",
        )
        title.pack(pady=(16, 4))

        iface_text = iface or "(default DDS iface)"
        tk.Label(
            self.root,
            text=f"iface: {iface_text}",
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
            text="Keep this window open during demos.\nPhysical pendant still works in parallel.",
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
                from soccerbot.safety import init_loco

                loco = init_loco(self.iface)
                with self._lock:
                    self._loco = loco
                self._set_status("DDS connected — killswitch armed")
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
                self._set_status(f"{label} OK @ {time.strftime('%H:%M:%S')}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s failed", label)
                self._set_status(f"{label} FAILED: {exc}")
                self.root.after(
                    0, lambda: self._messagebox.showerror("Killswitch", f"{label} failed:\n{exc}")
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_stop(self) -> None:
        self._with_loco(lambda loco: stop_loco(loco, iface=self.iface), "STOP MOVE")

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
            try:
                code, fsm_id = loco.GetFsmId()
                name = FSM_NAMES.get(int(fsm_id), f"fsm {fsm_id}")
                self._set_status(f"FSM={fsm_id} ({name})  rpc={code}")
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(1000, self._poll_fsm)

    def run(self) -> None:
        self.root.mainloop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headed G1 killswitch panel.")
    p.add_argument(
        "--iface",
        default=None,
        help="DDS network interface (e.g. enp5s0). Omit for SDK default.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    try:
        KillswitchApp(args.iface).run()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
