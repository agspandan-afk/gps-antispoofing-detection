"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
engine.py

DetectionEngine
───────────────
The main orchestrator that wires together:
  SignalQualityMonitor  →  buffers signal metrics
  ExtendedKalmanFilter  →  INS/GPS fusion + innovation monitoring
  JammingDetector       →  RF interference detection
  SpoofingDetector      →  Signal manipulation detection
  INSValidator          →  INS vs GPS cross-validation
  AlertManager          →  Fusion, deduplication, notification

Usage
─────
  engine = DetectionEngine()

  # Feed IMU data (typically 200 Hz between GPS epochs)
  engine.ingest_imu(imu_frame)

  # Feed GPS data (typically 1 Hz, one frame per receiver)
  alert = engine.ingest_gps([rx1_frame, rx2_frame])

  # Get current system status
  status = engine.get_status()

Design decisions
────────────────
  1. STATEFUL: The engine maintains all state between calls.
     Callers just push data in and read alerts out.

  2. PRIMARY RECEIVER: The first receiver in the list is primary.
     All others are secondary (used for multi-receiver cross-check).
     This mirrors Tata/L3 dual-receiver architectures where RX1 is
     the certified primary and RX2 is the integrity monitor.

  3. EKF is per-engine (single vehicle). For multi-vehicle scenarios,
     instantiate one DetectionEngine per vehicle.
"""

from typing import List, Optional
import time

from .core.data_types import GPSFrame, IMUFrame, AlertEvent, ThreatLevel, ThreatType
from .core.signal_monitor import SignalQualityMonitor
from .core.kalman_filter import ExtendedKalmanFilter
from .detectors.jamming_detector import JammingDetector
from .detectors.spoofing_detector import SpoofingDetector
from .detectors.ins_validator import INSValidator
from .alert_manager import AlertManager


class DetectionEngine:
    """
    Top-level GPS Anti-Spoofing & Jamming Detection Engine.

    Parameters
    ----------
    baseline_window_s : float
        Signal quality baseline window (default 30 s).
    detection_window_s : float
        Short anomaly detection window (default 3 s).
    log_alerts : bool
        Write alerts to JSON-lines log (default True).
    log_path : str
        Alert log file path.
    """

    def __init__(
        self,
        baseline_window_s: float = 30.0,
        detection_window_s: float = 3.0,
        log_alerts: bool = True,
        log_path: str = "/tmp/gps_antispoofing_alerts.jsonl",
    ):
        # Signal quality monitor (shared across all receivers)
        self.monitor = SignalQualityMonitor(
            baseline_window_s=baseline_window_s,
            detection_window_s=detection_window_s,
        )

        # EKF — one per vehicle
        self.ekf = ExtendedKalmanFilter()

        # Detectors
        self.jamming_detector  = JammingDetector()
        self.spoofing_detector = SpoofingDetector()
        self.ins_validator     = INSValidator()

        # Alert manager
        self.alert_manager = AlertManager(
            log_to_file=log_alerts,
            log_path=log_path,
        )

        # Internal state
        self._epoch = 0
        self._start_time = time.time()
        self._last_results = []

        # Telemetry cache for dashboard
        self._telemetry_history: List[dict] = []
        self._max_history = 120  # 2 minutes at 1 Hz

    # ──────────────────────────────────────────────
    # Data ingestion
    # ──────────────────────────────────────────────

    def ingest_imu(self, imu: IMUFrame) -> None:
        """
        Feed a single IMU frame into the EKF.
        Call this at the IMU rate (e.g., 200 Hz) between GPS updates.
        """
        self.ekf.propagate(imu)

    def ingest_gps(self, frames: List[GPSFrame]) -> Optional[AlertEvent]:
        """
        Feed one epoch of GPS frames (one per receiver) into the engine.
        
        Parameters
        ----------
        frames : list of GPSFrame
            frames[0] is the primary receiver.
            frames[1:] are secondary (cross-check) receivers.

        Returns
        -------
        AlertEvent if a threat is detected, else None.
        """
        if not frames:
            return None

        self._epoch += 1
        primary = frames[0]
        others  = frames[1:] if len(frames) > 1 else []

        # ── 1. Ingest all frames into signal monitor ─
        for f in frames:
            self.monitor.ingest(f)

        # ── 2. EKF update (primary receiver) ─────────
        ekf_result = self.ekf.update(primary)

        # ── 3. Run detectors ──────────────────────────
        jamming_result = self.jamming_detector.detect(primary, self.monitor)
        spoofing_result = self.spoofing_detector.detect(
            primary, self.monitor, self.ekf, others if others else None
        )
        ins_result = self.ins_validator.validate(primary, self.ekf)

        results = [jamming_result, spoofing_result, ins_result]
        self._last_results = results

        # ── 4. Fuse and alert ─────────────────────────
        alert = self.alert_manager.process(results, primary.position_lla)

        # ── 5. Record telemetry ───────────────────────
        self._record_telemetry(primary, ekf_result, results, alert)

        return alert

    # ──────────────────────────────────────────────
    # Status & telemetry
    # ──────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Returns a comprehensive status dictionary for dashboard display.
        """
        cn0_stats = self.monitor.get_cn0_stats("RX1", window_s=5.0)
        agc_stats = self.monitor.get_agc_stats("RX1")
        sv_stats  = self.monitor.get_sv_count_stats("RX1")
        dop_stats = self.monitor.get_doppler_residuals("RX1")
        ekf_hist  = self.ekf.innovation_history

        latest_ekf = ekf_hist[-1] if ekf_hist else {}

        return {
            "epoch":             self._epoch,
            "uptime_s":         round(time.time() - self._start_time, 1),
            "threat_level":     self.alert_manager.get_current_level().name,
            "threat_level_val": self.alert_manager.get_current_level().value,
            "threat_type":      self.alert_manager.get_current_type().name,
            "alert_summary":    self.alert_manager.get_summary(),
            "signal": {
                "mean_cn0":    round(cn0_stats.get("mean", 0), 2),
                "min_cn0":     round(cn0_stats.get("min", 0), 2),
                "cn0_delta":   round(self.monitor.get_cn0_delta("RX1"), 2),
                "agc_current": round(agc_stats.get("current", 40), 2),
                "agc_drop":    round(agc_stats.get("drop_db", 0), 2),
                "sv_count":    sv_stats.get("current", 0),
                "sv_drop":     round(sv_stats.get("drop", 0), 1),
                "doppler_mean": round(dop_stats.get("mean", 0), 1),
            },
            "ekf": {
                "innovation_norm": round(latest_ekf.get("innovation_norm", 0), 2),
                "pos_residual_m":  round(latest_ekf.get("pos_residual_m", 0), 2),
                "is_anomalous":    latest_ekf.get("is_anomalous", False),
                "trend_slope":     round(self.ekf.get_innovation_trend(), 3),
            },
            "detectors": {
                r.detector_name: {
                    "confidence":   round(r.confidence, 3),
                    "threat_level": r.threat_level.name,
                    "threat_type":  r.threat_type.name,
                }
                for r in self._last_results
            },
            "alert_history": self.alert_manager.get_alert_history()[-10:],
            "level_history":  self.alert_manager.get_level_history()[-60:],
            "telemetry":      self._telemetry_history[-60:],
        }

    def get_telemetry_history(self) -> List[dict]:
        return self._telemetry_history

    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────

    def _record_telemetry(
        self,
        frame: GPSFrame,
        ekf_result: dict,
        results: list,
        alert: Optional[AlertEvent],
    ) -> None:
        """Cache telemetry point for dashboard history."""
        entry = {
            "timestamp": frame.timestamp,
            "epoch":     self._epoch,
            "pos_lla":   frame.position_lla.tolist() if frame.position_lla is not None else None,
            "sv_count":  frame.num_svs,
            "mean_cn0":  round(frame.mean_cn0(), 2),
            "agc_db":    round(frame.agc_gain_db, 2),
            "pos_residual_m": round(ekf_result.get("pos_residual_m", 0), 2),
            "innovation_norm": round(ekf_result.get("innovation_norm", 0), 2),
            "jamming_conf": round(results[0].confidence if results else 0, 3),
            "spoofing_conf": round(results[1].confidence if len(results) > 1 else 0, 3),
            "ins_conf":     round(results[2].confidence if len(results) > 2 else 0, 3),
            "threat_level": self.alert_manager.get_current_level().value,
            "alert_id":     alert.alert_id if alert else None,
        }
        self._telemetry_history.append(entry)
        if len(self._telemetry_history) > self._max_history:
            self._telemetry_history.pop(0)
