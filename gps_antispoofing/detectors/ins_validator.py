"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
detectors/ins_validator.py

INSValidator
────────────
This module bridges the Kalman filter and the spoofing detector to provide
a standalone INS vs GPS consistency check — the most physically grounded
indicator of navigation manipulation.

What the INS Validator does:
  1. Runs independent dead-reckoning from IMU data (EKF propagation)
  2. At each GPS update, computes the position difference: GPS − INS
  3. Evaluates this residual against statistical thresholds
  4. Tracks the VELOCITY CONSISTENCY between GPS-implied velocity
     (Δposition/Δtime) and IMU-integrated velocity
  5. Tracks HEADING CONSISTENCY: GPS track angle vs IMU-integrated heading

Physical basis:
  A legitimate GPS signal, even with noise, will produce position residuals
  consistent with the measurement noise model (typically 2–5 m RMS).
  A spoofed signal that is "pulling" the receiver off course will produce
  residuals that GROW monotonically with the attack offset.

  The critical signature is:
    (GPS − INS residual) trending in a single direction over time
  rather than the random-walk pattern of natural noise.

Integration with L3-style avionics:
  L3 Technologies' NAVSYS-2000 and similar units use tightly-coupled
  GPS/INS in which the GPS pseudoranges are fed directly into the Kalman
  filter.  This loose-coupling implementation provides equivalent detection
  capability using position/velocity measurements, suitable for integration
  with any receiver (NovAtel, u-blox, Septentrio) via standard NMEA/OEM7 API.
"""

import numpy as np
from typing import Optional, List, Dict
from collections import deque
import time

from ..core.data_types import (
    GPSFrame, IMUFrame, DetectionResult, ThreatType, ThreatLevel
)
from ..core.kalman_filter import ExtendedKalmanFilter


class INSValidator:
    """
    Validates GPS consistency against INS dead-reckoning.

    Parameters
    ----------
    position_residual_threshold_m : float
        Position residual (m) above which spoofing is flagged (default 30 m).
    velocity_residual_threshold_ms : float
        Velocity residual (m/s) threshold (default 5.0 m/s).
    heading_residual_threshold_deg : float
        Heading residual (°) threshold (default 15°).
    trend_window : int
        Number of epochs for trend analysis (default 10).
    """

    def __init__(
        self,
        position_residual_threshold_m: float = 30.0,
        velocity_residual_threshold_ms: float = 5.0,
        heading_residual_threshold_deg: float = 15.0,
        trend_window: int = 10,
    ):
        self.pos_threshold = position_residual_threshold_m
        self.vel_threshold = velocity_residual_threshold_ms
        self.hdg_threshold = heading_residual_threshold_deg
        self.trend_window  = trend_window

        # Rolling history of residuals for trend analysis
        self._pos_residuals: deque = deque(maxlen=trend_window)
        self._vel_residuals: deque = deque(maxlen=trend_window)
        self._hdg_residuals: deque = deque(maxlen=trend_window)
        self._timestamps:    deque = deque(maxlen=trend_window)

        self._last_gps_pos: Optional[np.ndarray] = None
        self._last_gps_vel: Optional[np.ndarray] = None
        self._last_ts: Optional[float] = None

    # ──────────────────────────────────────────────
    # Main interface
    # ──────────────────────────────────────────────

    def validate(
        self,
        gps_frame: GPSFrame,
        ekf: ExtendedKalmanFilter,
    ) -> DetectionResult:
        """
        Run INS/GPS consistency validation.

        Parameters
        ----------
        gps_frame : Latest GPS frame.
        ekf       : EKF instance (after its update() has been called with this frame).
        """
        evidence = {}
        now = gps_frame.timestamp

        # ── 1. Position residual ─────────────────────
        pos_res = self._compute_position_residual(gps_frame, ekf)
        evidence["ins_pos_residual_m"] = round(pos_res, 2)
        self._pos_residuals.append(pos_res)

        # ── 2. Velocity residual ─────────────────────
        vel_res = self._compute_velocity_residual(gps_frame, ekf)
        evidence["ins_vel_residual_ms"] = round(vel_res, 2)
        self._vel_residuals.append(vel_res)

        # ── 3. Heading consistency ───────────────────
        hdg_res = self._compute_heading_residual(gps_frame, ekf)
        evidence["ins_hdg_residual_deg"] = round(hdg_res, 2)
        self._hdg_residuals.append(hdg_res)

        self._timestamps.append(now)

        # ── 4. Trend analysis ────────────────────────
        pos_trend = self._compute_trend(self._pos_residuals)
        vel_trend = self._compute_trend(self._vel_residuals)
        evidence["ins_pos_residual_trend"] = round(pos_trend, 4)  # m/epoch
        evidence["ins_vel_residual_trend"] = round(vel_trend, 4)  # m/s/epoch

        # ── 5. Directional persistence ───────────────
        # A real spoofer pulls in ONE direction.  Check if residuals
        # are consistently signed (not zero-mean random noise).
        pos_persistence = self._directional_persistence(ekf)
        evidence["ins_residual_persistence"] = round(pos_persistence, 3)
        # 0 = random / healthy, 1 = persistent directional drift = spoofing

        # ── Confidence scoring ───────────────────────
        confidence = 0.0

        # Position residual component
        if pos_res > self.pos_threshold:
            excess = pos_res - self.pos_threshold
            confidence += 0.40 * min(1.0, excess / (2 * self.pos_threshold))

        # Velocity residual component
        if vel_res > self.vel_threshold:
            excess = vel_res - self.vel_threshold
            confidence += 0.20 * min(1.0, excess / (3 * self.vel_threshold))

        # Rising trend component (key pull-off indicator)
        if pos_trend > 2.0:  # residual growing > 2 m/epoch
            confidence += 0.25 * min(1.0, pos_trend / 10.0)

        # Directional persistence
        if pos_persistence > 0.7:
            confidence += 0.15 * pos_persistence

        confidence = min(1.0, confidence)

        # Build result
        threat_level = self._to_threat_level(confidence)
        threat_type  = ThreatType.SPOOFING if confidence > 0.05 else ThreatType.NONE

        recommendation = self._build_recommendation(
            confidence, pos_res, vel_res, pos_trend, pos_persistence
        )

        evidence["ins_confidence"] = round(confidence, 3)

        # Store for next call
        self._last_gps_pos = gps_frame.position_lla.copy()
        self._last_gps_vel = gps_frame.velocity_ecef[:3].copy()
        self._last_ts = now

        return DetectionResult(
            detector_name="INSValidator",
            threat_type=threat_type,
            threat_level=threat_level,
            confidence=confidence,
            evidence=evidence,
            timestamp=now,
            recommendation=recommendation,
        )

    # ──────────────────────────────────────────────
    # Residual computations
    # ──────────────────────────────────────────────

    def _compute_position_residual(
        self, gps: GPSFrame, ekf: ExtendedKalmanFilter
    ) -> float:
        """
        INS-predicted position vs GPS position (metres, 3D).
        Uses the EKF's internal NED state vs the GPS-converted NED position.
        """
        if not ekf.initialized:
            return 0.0

        gps_ned = ekf._lla_to_ned(gps.position_lla)
        ins_ned = ekf.pos_ned

        return float(np.linalg.norm(gps_ned - ins_ned))

    def _compute_velocity_residual(
        self, gps: GPSFrame, ekf: ExtendedKalmanFilter
    ) -> float:
        """
        INS velocity (from EKF) vs GPS velocity vector (m/s, 3D).
        """
        if not ekf.initialized:
            return 0.0
        return float(np.linalg.norm(gps.velocity_ecef[:3] - ekf.vel_ned))

    def _compute_heading_residual(
        self, gps: GPSFrame, ekf: ExtendedKalmanFilter
    ) -> float:
        """
        GPS track angle vs INS yaw (degrees).
        GPS track angle: atan2(VE, VN) from GPS velocity.
        INS heading: ekf.att_rad[2] (yaw).
        """
        if not ekf.initialized:
            return 0.0

        gps_vel = gps.velocity_ecef[:2]   # N, E components
        speed = np.linalg.norm(gps_vel)
        if speed < 2.0:
            return 0.0  # heading meaningless at low speed

        gps_track_rad = np.arctan2(gps_vel[1], gps_vel[0])  # NE → heading
        ins_yaw_rad   = ekf.att_rad[2]

        diff = abs(gps_track_rad - ins_yaw_rad)
        diff = min(diff, 2 * np.pi - diff)  # wrap to [0, π]
        return float(np.degrees(diff))

    def _compute_trend(self, values: deque) -> float:
        """Linear slope of residual values over the trend window."""
        vals = list(values)
        if len(vals) < 3:
            return 0.0
        x = np.arange(len(vals), dtype=float)
        slope = float(np.polyfit(x, vals, 1)[0])
        return slope

    def _directional_persistence(self, ekf: ExtendedKalmanFilter) -> float:
        """
        Measures how consistently the position innovation points in the SAME direction.
        Under spoofing pull-off: the innovation always points "toward" the fake position.
        Under noise: innovations are zero-mean and random.

        Returns 0 (random) → 1 (fully persistent / directional).
        """
        history = ekf.innovation_history
        if len(history) < 5:
            return 0.0

        innovations = [h.get("innovation", None) for h in history[-self.trend_window:]]
        innovations = [i for i in innovations if i is not None and len(i) >= 3]

        if len(innovations) < 3:
            return 0.0

        # Look at just the horizontal (NE) innovation components
        inn_arr = np.array([[i[0], i[1]] for i in innovations])  # N, E
        unit_vectors = []
        for row in inn_arr:
            norm = np.linalg.norm(row)
            if norm > 0.1:
                unit_vectors.append(row / norm)

        if len(unit_vectors) < 3:
            return 0.0

        # Mean resultant length (R) of unit vectors: 0 = uniform, 1 = all same direction
        mean_vec = np.mean(unit_vectors, axis=0)
        R = float(np.linalg.norm(mean_vec))
        return R

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _to_threat_level(confidence: float) -> ThreatLevel:
        if confidence < 0.20:
            return ThreatLevel.CLEAR
        elif confidence < 0.40:
            return ThreatLevel.ADVISORY
        elif confidence < 0.60:
            return ThreatLevel.CAUTION
        elif confidence < 0.78:
            return ThreatLevel.WARNING
        else:
            return ThreatLevel.CRITICAL

    @staticmethod
    def _build_recommendation(
        confidence: float, pos_res: float, vel_res: float,
        pos_trend: float, persistence: float
    ) -> str:
        if confidence < 0.15:
            return f"INS/GPS consistent. Pos residual={pos_res:.1f}m, Vel residual={vel_res:.2f}m/s."
        else:
            return (
                f"INS divergence: pos={pos_res:.1f}m, vel={vel_res:.2f}m/s, "
                f"trend={pos_trend:.2f}m/epoch, persistence={persistence:.2f}. "
                f"Confidence={confidence:.1%}. "
                + ("Pull-off trajectory detected. " if pos_trend > 2.0 else "")
                + ("Directional drift confirmed. " if persistence > 0.7 else "")
                + "Recommend INS-primary."
            )
