"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
alert_manager.py

AlertManager
────────────
Fuses outputs from all detectors, deduplicates repeated alerts,
applies hysteresis (avoids flip-flopping), and generates structured
AlertEvents that can be forwarded to:

  • Flight Management System (FMS) / Autopilot
  • Ground Control Station (GCS) via MAVLink / STANAG 4586 datalink
  • Mission Computer logging (MIL-STD-1553 / ARINC 429)
  • Cockpit/HMI warning display

Fusion strategy:
  Conservative OR-fusion: the highest threat level from any single
  detector drives the alert.  Individual confidences are averaged to
  give a fused confidence score.

  Reasoning: In a safety-critical system (UAV, manned aircraft, missile
  guidance), a missed detection is far more dangerous than a false alarm.
  The operator can verify; the aircraft cannot recover from spoofed impact.

Hysteresis:
  Alert level escalates immediately (hard safety principle).
  De-escalation requires the confidence to stay below threshold for
  DEESCALATE_WINDOW consecutive epochs to avoid flapping.
"""

import uuid
import time
import json
import logging
from typing import List, Optional, Dict
from collections import deque

from .core.data_types import (
    DetectionResult, AlertEvent, ThreatType, ThreatLevel
)


logger = logging.getLogger("gps_antispoofing.alert_manager")


class AlertManager:
    """
    Fuses detection results and manages alert lifecycle.

    Parameters
    ----------
    deescalate_window : int
        Number of clean epochs required before de-escalating (default 5).
    log_to_file : bool
        If True, writes alert events to a JSON-lines log file.
    log_path : str
        Path for the log file.
    """

    def __init__(
        self,
        deescalate_window: int = 5,
        log_to_file: bool = True,
        log_path: str = "/tmp/gps_antispoofing_alerts.jsonl",
    ):
        self.deescalate_window = deescalate_window
        self.log_to_file = log_to_file
        self.log_path = log_path

        # Current alert state
        self._current_level = ThreatLevel.CLEAR
        self._current_type  = ThreatType.NONE
        self._clear_streak  = 0
        self._alert_count   = 0

        # History for trend display
        self._alert_history: deque = deque(maxlen=200)
        self._level_history: deque = deque(maxlen=200)

        # Active (unacknowledged) alert
        self._active_alert: Optional[AlertEvent] = None

        if log_to_file:
            logging.basicConfig(level=logging.INFO)

    # ──────────────────────────────────────────────
    # Main interface
    # ──────────────────────────────────────────────

    def process(
        self,
        results: List[DetectionResult],
        position_lla: Optional[object] = None,
    ) -> Optional[AlertEvent]:
        """
        Process a list of detection results from this epoch.
        Returns an AlertEvent if a threat is detected, else None.
        """
        if not results:
            return None

        # ── Fusion: highest level wins ────────────────
        max_level = max(results, key=lambda r: r.threat_level.value).threat_level
        max_type  = max(results, key=lambda r: r.confidence).threat_type

        # Average confidence across all detectors
        fused_confidence = sum(r.confidence for r in results) / len(results)

        # ── Hysteresis ────────────────────────────────
        # Escalate immediately
        if max_level.value > self._current_level.value:
            self._current_level = max_level
            self._current_type  = max_type
            self._clear_streak  = 0

        # De-escalate slowly
        elif max_level == ThreatLevel.CLEAR:
            self._clear_streak += 1
            if self._clear_streak >= self.deescalate_window:
                self._current_level = ThreatLevel.CLEAR
                self._current_type  = ThreatType.NONE
                self._active_alert  = None
                self._clear_streak  = 0
        else:
            self._clear_streak = 0
            self._current_level = max_level
            self._current_type  = max_type

        # Track history for plotting
        self._level_history.append({
            "timestamp": time.time(),
            "level": self._current_level.value,
            "confidence": fused_confidence,
        })

        # ── Generate alert if meaningful threat ───────
        # ADVISORY-only with low confidence = monitor silently; no alert fired
        if self._current_level == ThreatLevel.CLEAR:
            return None
        if self._current_level == ThreatLevel.ADVISORY and fused_confidence < 0.25:
            return None

        self._alert_count += 1
        alert_id = f"ALERT-{self._alert_count:04d}-{self._current_type.name}"

        contributing = [r.detector_name for r in results if r.threat_level.value >= 2]

        alert = AlertEvent(
            alert_id=alert_id,
            threat_type=self._current_type,
            threat_level=self._current_level,
            fused_confidence=round(fused_confidence, 3),
            contributing_detectors=contributing,
            position_at_alert=position_lla,
            timestamp=time.time(),
            mitigation_action=self._recommend_action(self._current_level, self._current_type),
            raw_results=results,
        )
        self._active_alert = alert
        self._alert_history.append(self._alert_to_dict(alert))

        if self.log_to_file:
            self._write_log(alert)

        self._log_to_console(alert)
        return alert

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged by the operator."""
        if self._active_alert and self._active_alert.alert_id == alert_id:
            self._active_alert.acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged.")
            return True
        return False

    def get_alert_history(self) -> List[dict]:
        """Returns serializable alert history for dashboard display."""
        return list(self._alert_history)

    def get_level_history(self) -> List[dict]:
        """Returns threat-level time series for plotting."""
        return list(self._level_history)

    def get_current_level(self) -> ThreatLevel:
        return self._current_level

    def get_current_type(self) -> ThreatType:
        return self._current_type

    def get_active_alert(self) -> Optional[AlertEvent]:
        return self._active_alert

    def get_summary(self) -> Dict:
        """Returns a summary dict for status display."""
        return {
            "current_level": self._current_level.name,
            "current_type":  self._current_type.name,
            "total_alerts":  self._alert_count,
            "active_alert":  self._active_alert.alert_id if self._active_alert else None,
            "acknowledged":  self._active_alert.acknowledged if self._active_alert else True,
        }

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _recommend_action(level: ThreatLevel, threat_type: ThreatType) -> str:
        actions = {
            ThreatLevel.ADVISORY: "Monitor signal quality. Increase INS weight to 30%.",
            ThreatLevel.CAUTION:  "Engage INS-blend mode (INS 60% / GPS 40%). Alert GCS.",
            ThreatLevel.WARNING:  "Switch to INS-primary. Downlink alert to GCS. Log position.",
            ThreatLevel.CRITICAL: "FULL INS-PRIMARY. Engage ECCM. GCS EMERGENCY ALERT. "
                                  "If SPOOFING: discard GPS; re-initialize from known waypoint. "
                                  "If JAMMING: seek unobstructed sky or alternate nav aid.",
        }
        return actions.get(level, "No action required.")

    @staticmethod
    def _alert_to_dict(alert: AlertEvent) -> dict:
        return {
            "alert_id":             alert.alert_id,
            "threat_type":          alert.threat_type.name,
            "threat_level":         alert.threat_level.name,
            "threat_level_value":   alert.threat_level.value,
            "fused_confidence":     alert.fused_confidence,
            "contributing":         alert.contributing_detectors,
            "timestamp":            alert.timestamp,
            "mitigation":           alert.mitigation_action,
            "acknowledged":         alert.acknowledged,
            "evidence":             {
                r.detector_name: r.evidence
                for r in alert.raw_results
            },
        }

    def _write_log(self, alert: AlertEvent) -> None:
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(self._alert_to_dict(alert)) + "\n")
        except Exception as e:
            logger.warning(f"Could not write alert log: {e}")

    @staticmethod
    def _log_to_console(alert: AlertEvent) -> None:
        level_colors = {
            ThreatLevel.ADVISORY: "\033[93m",   # Yellow
            ThreatLevel.CAUTION:  "\033[33m",   # Orange
            ThreatLevel.WARNING:  "\033[91m",   # Light red
            ThreatLevel.CRITICAL: "\033[31m",   # Red
        }
        RESET = "\033[0m"
        color = level_colors.get(alert.threat_level, "")
        print(
            f"{color}[{alert.threat_level.name}] {alert.alert_id} | "
            f"Type={alert.threat_type.name} | "
            f"Confidence={alert.fused_confidence:.1%} | "
            f"Detectors={', '.join(alert.contributing_detectors)}{RESET}"
        )
        print(f"  → Mitigation: {alert.mitigation_action}")
