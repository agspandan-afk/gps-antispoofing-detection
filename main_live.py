#!/usr/bin/env python3
"""
GPS Anti-Spoofing & Jamming Detection — LIVE HARDWARE MODE
===========================================================
main_live.py

Replaces the simulator with real GPS receiver(s) and IMU.
The DetectionEngine is identical to the simulation version.

Usage examples
--------------
# Minimum — one USB GPS receiver, no IMU (mock IMU mode):
python main_live.py --gps1 /dev/ttyUSB0 --no-imu

# One GPS + MPU-6050 on Raspberry Pi I2C:
python main_live.py --gps1 /dev/ttyACM0 --imu-mode i2c

# Dual receiver (best spoofing detection):
python main_live.py --gps1 /dev/ttyUSB0 --gps2 /dev/ttyUSB1 --imu-mode i2c

# Serial IMU (Pixhawk-style CSV stream):
python main_live.py --gps1 /dev/ttyACM0 --imu-mode serial --imu-port /dev/ttyUSB1

# Different baud / I2C address:
python main_live.py --gps1 /dev/ttyUSB0 --baud1 9600 --imu-address 0x69

# Save alerts to file:
python main_live.py --gps1 /dev/ttyUSB0 --log /var/log/gps_integrity.jsonl

Hardware wiring quick reference
--------------------------------
u-blox GPS → USB → /dev/ttyACM0 or /dev/ttyUSB0  (115200 baud default)
MPU-6050   → I2C → SDA=GPIO2, SCL=GPIO3 (Raspberry Pi), address 0x68

Enable I2C on Raspberry Pi:
  sudo raspi-config → Interface Options → I2C → Enable
  pip install smbus2

Find your GPS port:
  ls /dev/tty* before and after plugging in the receiver
  dmesg | tail -20  (shows new device)
"""

import sys
import time
import signal
import argparse
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from gps_antispoofing import DetectionEngine
from gps_antispoofing.core import ThreatLevel, ThreatType
from hardware import GPSReader, IMUReader


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
R = "\033[31m"; Y = "\033[93m"; G = "\033[32m"
C = "\033[36m"; W = "\033[97m"; DIM = "\033[2m"; RESET = "\033[0m"; BOLD = "\033[1m"

LEVEL_COLOR = {
    ThreatLevel.CLEAR:    G,
    ThreatLevel.ADVISORY: Y,
    ThreatLevel.CAUTION:  Y,
    ThreatLevel.WARNING:  R,
    ThreatLevel.CRITICAL: R,
}


# ─────────────────────────────────────────────────────────────────────────────
# Live runner
# ─────────────────────────────────────────────────────────────────────────────

class LiveRunner:
    """
    Wires together GPSReader(s) + IMUReader → DetectionEngine.
    Prints status to terminal and optionally writes alerts to a log file.
    """

    def __init__(self, args):
        self.args = args

        self.engine = DetectionEngine(
            baseline_window_s  = 30.0,
            detection_window_s = 3.0,
            log_alerts         = bool(args.log),
            log_path           = args.log or "/tmp/gps_integrity.jsonl",
        )

        # Primary GPS reader
        self.gps1 = GPSReader(
            port        = args.gps1,
            baud        = args.baud1,
            receiver_id = "RX1",
        )

        # Optional secondary receiver
        self.gps2 = None
        if args.gps2:
            self.gps2 = GPSReader(
                port        = args.gps2,
                baud        = args.baud2 or args.baud1,
                receiver_id = "RX2",
            )

        # IMU reader
        imu_mode = "mock" if args.no_imu else args.imu_mode
        self.imu = IMUReader(
            mode         = imu_mode,
            i2c_bus      = args.imu_bus,
            i2c_address  = int(args.imu_address, 16),
            serial_port  = args.imu_port,
            rate_hz      = 200.0,
        )

        # Latest frame from RX2 for cross-check
        self._rx2_frame   = None
        self._rx2_lock    = threading.Lock()

        # State
        self._running     = False
        self._epoch       = 0
        self._start_time  = time.time()

    # ──────────────────────────────────────────────
    # Start / stop
    # ──────────────────────────────────────────────

    def start(self) -> None:
        print(f"\n{BOLD}{W}GPS Anti-Spoofing & Jamming Detection — LIVE MODE{RESET}")
        print(f"{DIM}Primary: {self.args.gps1} @ {self.args.baud1} baud{RESET}")
        if self.gps2:
            print(f"{DIM}Secondary: {self.args.gps2}{RESET}")
        print(f"{DIM}IMU: {self.imu.mode} mode{RESET}")
        if self.args.log:
            print(f"{DIM}Alert log: {self.args.log}{RESET}")
        print()

        # Wire callbacks
        self.gps1.on_frame = self._on_gps1_frame
        if self.gps2:
            self.gps2.on_frame = self._on_gps2_frame

        self.imu.on_frame = self._on_imu_frame

        # Start hardware readers
        self.imu.start()

        if not self.args.no_imu and self.args.imu_mode == "i2c":
            print(f"Calibrating IMU... keep still for 5 seconds")
            time.sleep(1.0)  # let I2C settle
            self.imu.calibrate(duration_s=5.0)

        self.gps1.start()
        if self.gps2:
            self.gps2.start()

        self._running = True
        self._print_header()

    def stop(self) -> None:
        self._running = False
        self.gps1.stop()
        if self.gps2:
            self.gps2.stop()
        self.imu.stop()
        print(f"\n{G}Detection engine stopped.{RESET}")

    def run_forever(self) -> None:
        """Block until Ctrl-C."""
        self.start()
        try:
            while self._running:
                time.sleep(0.5)
                self._print_status_line()
        except KeyboardInterrupt:
            print(f"\n{Y}Interrupted by user.{RESET}")
        finally:
            self.stop()

    # ──────────────────────────────────────────────
    # Callbacks — called from reader threads
    # ──────────────────────────────────────────────

    def _on_imu_frame(self, frame) -> None:
        """200 Hz — propagate EKF with each IMU sample."""
        self.engine.ingest_imu(frame)
        # Keep mock IMU updated with latest GPS velocity
        if self.imu.mode == "mock":
            latest = self.engine.monitor.get_latest_frame("RX1")
            if latest is not None:
                self.imu.update_mock_velocity(latest.velocity_ecef[:3])

    def _on_gps2_frame(self, frame) -> None:
        """Store RX2 frame for cross-check on next RX1 update."""
        with self._rx2_lock:
            self._rx2_frame = frame

    def _on_gps1_frame(self, frame) -> None:
        """
        1 Hz — main detection update.
        Called from GPSReader background thread.
        """
        if not self._running:
            return

        self._epoch += 1

        # Collect secondary receiver if available
        frames = [frame]
        with self._rx2_lock:
            if self._rx2_frame is not None:
                frames.append(self._rx2_frame)

        # Run detection
        alert = self.engine.ingest_gps(frames)

        # Print to terminal
        self._print_epoch(frame, alert)

        if alert:
            self._print_alert(alert)

    # ──────────────────────────────────────────────
    # Terminal output
    # ──────────────────────────────────────────────

    def _print_header(self) -> None:
        print(
            f"  {'Epoch':>5}  {'SVs':>4}  {'C/N0':>6}  {'AGC':>6}  "
            f"{'PosRes':>7}  {'INN':>6}  {'Jam':>5}  {'Spoof':>5}  "
            f"{'INS':>5}  {'Level':<12}"
        )
        print("  " + "─" * 72)

    def _print_epoch(self, frame, alert) -> None:
        status = self.engine.get_status()
        tel    = status["telemetry"][-1] if status["telemetry"] else {}
        lvl    = self.engine.alert_manager.get_current_level()
        lc     = LEVEL_COLOR.get(lvl, W)
        flag   = f"  {lc}◄ ALERT!{RESET}" if alert else ""

        sv   = tel.get("sv_count", 0)
        cn0  = tel.get("mean_cn0", 0.0)
        agc  = tel.get("agc_db", 0.0)
        pr   = tel.get("pos_residual_m", 0.0)
        inn  = tel.get("innovation_norm", 0.0)
        jc   = tel.get("jamming_conf", 0.0)
        sc   = tel.get("spoofing_conf", 0.0)
        ic   = tel.get("ins_conf", 0.0)

        print(
            f"  {self._epoch:>5}  {sv:>4}  {cn0:>6.1f}  {agc:>6.1f}  {pr:>7.1f}  "
            f"{inn:>6.1f}  {jc:>5.2f}  {sc:>5.2f}  {ic:>5.2f}  "
            f"{lc}{lvl.name:<12}{RESET}{flag}"
        )

    def _print_status_line(self) -> None:
        """Called from main thread every 0.5 s — only used for heartbeat if needed."""
        pass

    def _print_alert(self, alert) -> None:
        lc = LEVEL_COLOR.get(alert.threat_level, W)
        print(f"\n  {lc}{BOLD}[{alert.threat_level.name}] {alert.alert_id}{RESET}")
        print(f"  {lc}Type: {alert.threat_type.name}  "
              f"Confidence: {alert.fused_confidence:.1%}{RESET}")
        print(f"  Detectors: {', '.join(alert.contributing_detectors)}")
        print(f"  → {alert.mitigation_action}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="GPS Anti-Spoofing Detection — Live Hardware Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Laptop with USB GPS, no IMU:
  python main_live.py --gps1 /dev/ttyUSB0 --no-imu

  # Raspberry Pi with GPS + MPU-6050:
  python main_live.py --gps1 /dev/ttyACM0 --imu-mode i2c

  # Dual receiver setup:
  python main_live.py --gps1 /dev/ttyUSB0 --gps2 /dev/ttyUSB1 --imu-mode i2c

  # Windows COM port:
  python main_live.py --gps1 COM3 --no-imu
        """
    )

    # GPS
    p.add_argument("--gps1",  required=True,
                   help="Primary GPS receiver serial port (e.g. /dev/ttyUSB0, COM3)")
    p.add_argument("--baud1", type=int, default=115200,
                   help="Primary receiver baud rate (default: 115200)")
    p.add_argument("--gps2",  default=None,
                   help="Secondary GPS receiver port (optional, improves spoofing detection)")
    p.add_argument("--baud2", type=int, default=None,
                   help="Secondary receiver baud rate (defaults to --baud1)")

    # IMU
    imu_group = p.add_mutually_exclusive_group()
    imu_group.add_argument("--no-imu", action="store_true",
                           help="Disable IMU — run with GPS-only detection (mock IMU mode)")
    imu_group.add_argument("--imu-mode", default="i2c",
                           choices=["i2c", "serial", "mock"],
                           help="IMU interface mode (default: i2c)")

    p.add_argument("--imu-bus",     type=int,   default=1,
                   help="I2C bus number for MPU-6050 (default: 1)")
    p.add_argument("--imu-address", type=str,   default="0x68",
                   help="I2C address for MPU-6050 (default: 0x68)")
    p.add_argument("--imu-port",    type=str,   default="/dev/ttyUSB1",
                   help="Serial port for serial IMU mode")

    # Output
    p.add_argument("--log", default=None,
                   help="Write JSON-lines alert log to this file")

    args = p.parse_args()
    runner = LiveRunner(args)

    # Handle Ctrl-C cleanly
    def _sig(sig, frame):
        runner.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    runner.run_forever()


if __name__ == "__main__":
    main()
