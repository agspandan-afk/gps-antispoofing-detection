#!/usr/bin/env python3
"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
main.py — Demo runner

Runs all 6 simulation scenarios and prints a terminal summary.
Also generates JSON telemetry output for the dashboard.

Usage:
  python main.py                            # Run all scenarios
  python main.py --scenario SPOOFING_PULLOFF
  python main.py --scenario CLEAN --epochs 30
  python main.py --output telemetry.json    # Save JSON for dashboard
  python main.py --list-scenarios
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ─── Make package importable from this directory ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from gps_antispoofing import DetectionEngine
from gps_antispoofing.core import ThreatLevel
from gps_antispoofing.simulator import ScenarioSimulator, Scenario


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
R = "\033[31m"   # Red
Y = "\033[93m"   # Yellow
G = "\033[32m"   # Green
C = "\033[36m"   # Cyan
B = "\033[34m"   # Blue
W = "\033[97m"   # White
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

LEVEL_COLOR = {
    0: G,    # CLEAR
    1: Y,    # ADVISORY
    2: Y,    # CAUTION
    3: R,    # WARNING
    4: R,    # CRITICAL (blinking would be ideal)
}

BAR_CHARS = "▏▎▍▌▋▊▉█"


def conf_bar(conf: float, width: int = 20) -> str:
    """Render a confidence score as a compact ASCII bar."""
    filled = int(conf * width)
    bar = "█" * filled + "░" * (width - filled)
    color = G if conf < 0.3 else (Y if conf < 0.6 else R)
    return f"{color}[{bar}]{RESET} {conf:.0%}"


def run_scenario(
    scenario: Scenario,
    epochs: int = 45,
    imu_rate: int = 200,
    verbose: bool = True,
) -> dict:
    """
    Run one simulation scenario and return telemetry.

    Parameters
    ----------
    scenario  : Which attack scenario to simulate.
    epochs    : Number of GPS epochs (1 Hz → seconds of simulated flight).
    imu_rate  : IMU frames per GPS epoch.
    verbose   : Print per-epoch output.
    """
    sep = "─" * 70

    if verbose:
        print(f"\n{BOLD}{C}{'═' * 70}{RESET}")
        print(f"{BOLD}{W}  SCENARIO: {scenario.name}  [{epochs} epochs × {imu_rate} IMU/GPS]{RESET}")
        print(f"{C}{'═' * 70}{RESET}")

        descriptions = {
            Scenario.CLEAN:            "Normal flight. No interference expected.",
            Scenario.JAMMING_WEAK:     "Broadband noise jammer activates at T+15s. Gradual C/N0 degradation.",
            Scenario.JAMMING_STRONG:   "High-power jammer. AGC collapse + rapid satellite dropout.",
            Scenario.SPOOFING_NAIVE:   "Amateur spoofer. Sudden 300m position jump at T+15s.",
            Scenario.SPOOFING_PULLOFF: "Sophisticated pull-off attack. 3 m/epoch drift from T+15s.",
            Scenario.MEACONING:        "Signal re-broadcast with 2s delay. Subtle position lag.",
        }
        print(f"  {DIM}{descriptions.get(scenario, '')}{RESET}\n")
        print(f"  {'Epoch':>5}  {'SVs':>4}  {'C/N0':>6}  {'AGC':>6}  {'PosRes':>7}  "
              f"{'INN':>6}  {'Jam':>5}  {'Spoof':>5}  {'INS':>5}  {'Level':<10}")
        print(f"  {sep}")

    sim    = ScenarioSimulator(scenario, num_receivers=2)
    engine = DetectionEngine(log_alerts=False)
    all_telemetry = []
    alerts_fired = 0

    for epoch in range(1, epochs + 1):
        # IMU frames at high rate
        imu_frames = sim.generate_imu_frames(imu_rate)
        for imu in imu_frames:
            engine.ingest_imu(imu)

        # GPS frames (both receivers)
        gps_frames = sim.generate_gps_frame()
        alert = engine.ingest_gps(gps_frames)

        if alert:
            alerts_fired += 1

        status = engine.get_status()
        tel = status["telemetry"][-1] if status["telemetry"] else {}
        all_telemetry.append(tel)

        if verbose:
            level_val = status["threat_level_val"]
            level_name = status["threat_level"]
            lc = LEVEL_COLOR.get(level_val, W)
            flag = f"  {lc}◄ ALERT!{RESET}" if alert else ""

            sv  = tel.get("sv_count", 0)
            cn0 = tel.get("mean_cn0", 0.0)
            agc = tel.get("agc_db", 0.0)
            pos = tel.get("pos_residual_m", 0.0)
            inn = tel.get("innovation_norm", 0.0)
            jc  = tel.get("jamming_conf", 0.0)
            sc  = tel.get("spoofing_conf", 0.0)
            ic  = tel.get("ins_conf", 0.0)

            level_str = f"{lc}{level_name:<10}{RESET}"

            print(
                f"  {epoch:>5}  {sv:>4}  {cn0:>6.1f}  {agc:>6.1f}  {pos:>7.1f}  "
                f"{inn:>6.1f}  {jc:>5.2f}  {sc:>5.2f}  {ic:>5.2f}  {level_str}{flag}"
            )

    # ── Summary ───────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  {sep}")
        status = engine.get_status()
        print(f"  {BOLD}FINAL STATUS: {LEVEL_COLOR.get(status['threat_level_val'], W)}"
              f"{status['threat_level']} ({status['threat_type']}){RESET}")
        print(f"  Total alerts fired: {alerts_fired}")
        det = status.get("detectors", {})
        for name, d in det.items():
            conf = d.get("confidence", 0)
            lv   = d.get("threat_level", "CLEAR")
            print(f"  {DIM}{name:<22}{RESET} conf={conf_bar(conf, 15)}  level={lv}")
        print()

    return {
        "scenario": scenario.name,
        "epochs": epochs,
        "telemetry": all_telemetry,
        "final_status": engine.get_status(),
        "alerts_fired": alerts_fired,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPS Anti-Spoofing & Jamming Detection Module — Demo Runner"
    )
    parser.add_argument(
        "--scenario", "-s",
        default=None,
        choices=[s.name for s in Scenario],
        help="Run a single scenario (default: run all).",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=45,
        help="Number of GPS epochs per scenario (default: 45).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save telemetry JSON to this file for dashboard loading.",
    )
    parser.add_argument(
        "--list-scenarios", "-l",
        action="store_true",
        help="List available scenarios and exit.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-epoch output.",
    )
    args = parser.parse_args()

    if args.list_scenarios:
        print("\nAvailable scenarios:")
        for s in Scenario:
            print(f"  {s.name}")
        return

    print(f"\n{BOLD}{W}GPS Anti-Spoofing & Jamming Detection Module{RESET}")
    print(f"{DIM}Tata / L3 Navigation Systems — Detection Framework v1.0{RESET}\n")

    scenarios_to_run = (
        [Scenario[args.scenario]] if args.scenario
        else list(Scenario)
    )

    all_results = []
    for scenario in scenarios_to_run:
        result = run_scenario(
            scenario,
            epochs=args.epochs,
            verbose=not args.quiet,
        )
        all_results.append(result)

    # ── Final comparison table ────────────────────────────────────────────
    if not args.quiet and len(all_results) > 1:
        print(f"\n{BOLD}{W}{'─' * 60}")
        print(f"  SCENARIO COMPARISON SUMMARY")
        print(f"{'─' * 60}{RESET}")
        print(f"  {'Scenario':<22}  {'Alerts':>6}  {'Final Level':<12}  {'Final Type'}")
        print(f"  {'─' * 56}")
        for r in all_results:
            s   = r["scenario"]
            a   = r["alerts_fired"]
            lvl = r["final_status"]["threat_level"]
            typ = r["final_status"]["threat_type"]
            lv  = r["final_status"]["threat_level_val"]
            lc  = LEVEL_COLOR.get(lv, W)
            print(f"  {s:<22}  {a:>6}  {lc}{lvl:<12}{RESET}  {typ}")
        print()

    # ── Save JSON output ──────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"{G}✓ Telemetry saved to {output_path}{RESET}")


if __name__ == "__main__":
    main()
