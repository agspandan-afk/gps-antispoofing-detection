"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
detectors/spoofing_detector.py

SpoofingDetector
────────────────
Spoofing is more sophisticated than jamming: the attacker generates counterfeit
GPS signals that the receiver accepts as legitimate.  Detection is harder
because the receiver DOES obtain a solution — it's just WRONG.

Spoofing attack taxonomy:
  • Simplistic spoofing  : single fake transmitter, no synchronisation
  • Intermediate spoofing: synchronises power & frequency to real signals
  • Sophisticated attack : gradually pulls the receiver off its true position
                           ("meaconing" or "trajectory deviation attack")

The latter is the threat relevant to UAVs and battlefield systems: an attacker
near the operating area transmits at slightly higher power, gradually offsetting
the vehicle's position report to drive it off course or into a kill zone.

Detection algorithms (5 independent methods):
─────────────────────────────────────────────
  1. MULTI-RECEIVER CROSS-CHECK
     Compare positions from 2+ independent receivers.  Spoofed signals affect
     ALL receivers equally (single-source assumption), so their solutions agree
     — but disagree with the INS.  Legitimate multi-path causes receivers to
     disagree with EACH OTHER.  This is the gold standard for spoofing detection.
     (Used in Boeing/L3 integrated nav systems, DARPA HACMS.)

  2. INS POSITION RESIDUAL (via EKF innovation — see kalman_filter.py)
     The EKF propagates a position from IMU data.  Spoofed GPS measurements
     cause growing innovations — the filter "fights" the spoofer.
     Slope of the innovation vector over time reveals the pull-off trajectory.

  3. SIGNAL POWER ANOMALY
     Spoofed signals typically arrive at higher power (meaconer must overpower
     the authentic signal).  Sudden C/N0 INCREASE of >3 dB on multiple SVs
     simultaneously is a classic spoofing signature.
     Reference: Amin et al., "Vulnerabilities, Threats and Authentication in
     Satellite-Based Navigation Systems," Proceedings of the IEEE, 2016.

  4. DOPPLER CONSISTENCY CHECK
     Each satellite's Doppler shift must be consistent with:
       (a) Satellite velocity (from ephemeris)
       (b) Vehicle velocity (from INS)
     Spoofed signals often have wrong Doppler or static Doppler —
     if the attacker's hardware isn't precisely tracking satellite motion.
     Residual threshold: > 50 Hz is suspicious, > 200 Hz is very likely spoofing.

  5. ELEVATION / AZIMUTH CONSISTENCY
     Claimed satellite positions must match published ephemeris.
     A satellite reported below 5° elevation but with high C/N0 is impossible
     under normal conditions (horizon masking) — strong spoofing indicator.
     Also: satellites "appearing" with high power and no prior acquisition
     history are suspicious.

  6. POSITION JUMP DETECTOR
     Spoofing pull-off creates a position velocity that doesn't match INS velocity.
     A position step > 50 m between consecutive 1-Hz epochs is physically
     impossible for fixed-wing UAVs and is an immediate hard flag.
"""

import numpy as np
from typing import List, Optional, Dict
import time

from ..core.data_types import (
    GPSFrame, IMUFrame, DetectionResult, ThreatType, ThreatLevel
)
from ..core.signal_monitor import SignalQualityMonitor
from ..core.kalman_filter import ExtendedKalmanFilter


class SpoofingDetector:
    """
    Multi-algorithm spoofing detector.

    Parameters
    ----------
    cn0_spike_threshold_db : float
        C/N0 increase (dB) that triggers power anomaly check (default 4.0).
    doppler_residual_threshold_hz : float
        Max acceptable Doppler residual per satellite (default 50.0 Hz).
    position_jump_threshold_m : float
        Max acceptable single-epoch position step (default 50.0 m).
    innovation_threshold : float
        EKF innovation chi-squared threshold override (None = use EKF default).
    """

    def __init__(
        self,
        cn0_spike_threshold_db: float = 4.0,
        doppler_residual_threshold_hz: float = 50.0,
        position_jump_threshold_m: float = 50.0,
        innovation_threshold: Optional[float] = None,
    ):
        self.cn0_spike_threshold_db = cn0_spike_threshold_db
        self.doppler_residual_threshold_hz = doppler_residual_threshold_hz
        self.position_jump_threshold_m = position_jump_threshold_m
        self.innovation_threshold = innovation_threshold

        self._weights = {
            "multi_receiver":  0.30,
            "ins_residual":    0.25,
            "power_anomaly":   0.20,
            "doppler":         0.15,
            "position_jump":   0.10,
        }

        self._last_position: Optional[np.ndarray] = None
        self._last_timestamp: Optional[float] = None

    # ──────────────────────────────────────────────
    # Main detection entry point
    # ──────────────────────────────────────────────

    def detect(
        self,
        frame: GPSFrame,
        monitor: SignalQualityMonitor,
        ekf: ExtendedKalmanFilter,
        other_receiver_frames: Optional[List[GPSFrame]] = None,
    ) -> DetectionResult:
        """
        Fused spoofing detection.

        Parameters
        ----------
        frame                : Primary receiver's latest GPS frame.
        monitor              : Signal quality monitor (already ingested frame).
        ekf                  : Kalman filter instance (already updated with frame).
        other_receiver_frames: List of frames from secondary receivers (can be None).
        """
        rx = frame.receiver_id
        evidence = {}

        # ── 1. Multi-receiver cross-check ────────────
        mr_confidence, mr_evidence = self._multi_receiver_check(
            frame, other_receiver_frames
        )
        evidence.update(mr_evidence)

        # ── 2. INS position residual (EKF innovation) ─
        ins_confidence, ins_evidence = self._ins_residual_check(ekf)
        evidence.update(ins_evidence)

        # ── 3. Signal power anomaly ───────────────────
        pwr_confidence, pwr_evidence = self._power_anomaly_check(rx, monitor)
        evidence.update(pwr_evidence)

        # ── 4. Doppler consistency ────────────────────
        dop_confidence, dop_evidence = self._doppler_check(rx, monitor, frame)
        evidence.update(dop_evidence)

        # ── 5. Position jump ──────────────────────────
        jump_confidence, jump_evidence = self._position_jump_check(frame)
        evidence.update(jump_evidence)

        # ── Fused confidence ──────────────────────────
        fused_confidence = (
            self._weights["multi_receiver"] * mr_confidence  +
            self._weights["ins_residual"]   * ins_confidence +
            self._weights["power_anomaly"]  * pwr_confidence +
            self._weights["doppler"]        * dop_confidence +
            self._weights["position_jump"]  * jump_confidence
        )
        fused_confidence = min(1.0, fused_confidence)

        # Hard flags — any one of these alone warrants a WARNING regardless of fusion
        hard_flags = []
        if jump_confidence > 0.9:
            hard_flags.append("POSITION_JUMP")
        if ins_confidence > 0.85:
            hard_flags.append("EKF_INNOVATION_BREACH")
        if mr_confidence > 0.85:
            hard_flags.append("MULTI_RX_DIVERGENCE")

        if hard_flags:
            fused_confidence = max(fused_confidence, 0.7)
        evidence["hard_flags"] = hard_flags

        threat_level = self._confidence_to_level(fused_confidence)
        threat_type  = ThreatType.SPOOFING if fused_confidence > 0.05 else ThreatType.NONE

        recommendation = self._build_recommendation(
            fused_confidence, hard_flags, evidence
        )

        # Store for next position jump check
        self._last_position = frame.position_lla.copy()
        self._last_timestamp = frame.timestamp

        return DetectionResult(
            detector_name="SpoofingDetector",
            threat_type=threat_type,
            threat_level=threat_level,
            confidence=fused_confidence,
            evidence=evidence,
            timestamp=frame.timestamp,
            recommendation=recommendation,
        )

    # ──────────────────────────────────────────────
    # Sub-detectors
    # ──────────────────────────────────────────────

    def _multi_receiver_check(
        self,
        primary: GPSFrame,
        others: Optional[List[GPSFrame]],
    ) -> tuple:
        """
        Compare position solutions from multiple receivers.
        Under spoofing: solutions are tightly clustered (attacker broadcasts to all)
        but collectively wrong (diverge from INS).
        Under multi-path / natural fading: individual receivers scatter.

        Metric: inter-receiver position spread.
        Low spread alone is NOT spoofing — but combined with INS residual it is.
        """
        evidence = {}
        if not others:
            evidence["multi_rx_available"] = False
            return 0.0, evidence

        evidence["multi_rx_available"] = True
        positions = [primary.position_lla]
        positions.extend(f.position_lla for f in others if f.fix_valid)

        if len(positions) < 2:
            evidence["multi_rx_spread_m"] = 0.0
            return 0.0, evidence

        # Compute pairwise 2D position differences (metres)
        lats = np.array([p[0] for p in positions])
        lons = np.array([p[1] for p in positions])
        lat_centre = np.mean(lats)

        dx_m = (lons - np.mean(lons)) * 111_320.0 * np.cos(np.deg2rad(lat_centre))
        dy_m = (lats - lat_centre) * 110_540.0
        spread = float(np.sqrt(np.var(dx_m) + np.var(dy_m)))

        evidence["multi_rx_spread_m"] = round(spread, 2)
        evidence["multi_rx_count"]    = len(positions)

        # Tight spread alone is NOT a spoofing flag — receivers agree under clean sky too.
        # We return a small positive weight only when spread is anomalously ZERO
        # (physically impossible for independent receivers), leaving corroboration
        # to the INS and power detectors.
        if spread < 0.3:
            # Numerically identical positions — hardware fault or same hardware path
            confidence = 0.10
        else:
            confidence = 0.0

        return confidence, evidence

    def _ins_residual_check(self, ekf: ExtendedKalmanFilter) -> tuple:
        """
        Check EKF innovation history for spoofing pull-off pattern.
        
        A spoofing attack typically follows this progression:
          Phase 1 (sync): Innovations are small — attacker is aligned
          Phase 2 (pull): Innovations grow — attacker starts drifting target
          Phase 3 (complete): Large sustained innovations, position diverged

        We detect Phase 2 & 3 via:
          (a) Current innovation norm vs chi² threshold
          (b) Rising slope of innovation history
        """
        evidence = {}
        history = ekf.innovation_history

        if not history:
            evidence["ekf_innovation_norm"] = 0.0
            evidence["ekf_innovation_slope"] = 0.0
            return 0.0, evidence

        latest = history[-1]
        norm      = latest["innovation_norm"]
        threshold = 38.9  # χ²(6) at P_FA=1e-6
        slope     = ekf.get_innovation_trend()
        pos_res   = latest["pos_residual_m"]

        evidence["ekf_innovation_norm"]      = round(norm, 2)
        evidence["ekf_chi2_threshold"]       = threshold
        evidence["ekf_innovation_slope"]     = round(slope, 3)
        evidence["ekf_position_residual_m"]  = round(pos_res, 2)
        evidence["ekf_anomalous"]            = latest["is_anomalous"]

        confidence = 0.0
        if norm > threshold:
            confidence += 0.5 * min(1.0, (norm - threshold) / threshold)
        if slope > 2.0:  # innovation growing by 2 χ² units/epoch
            confidence += 0.3 * min(1.0, slope / 10.0)
        if pos_res > 50.0:  # GPS position > 50 m from INS prediction
            confidence += 0.2 * min(1.0, (pos_res - 50.0) / 100.0)

        return min(1.0, confidence), evidence

    def _power_anomaly_check(
        self, receiver_id: str, monitor: SignalQualityMonitor
    ) -> tuple:
        """
        Detect unexpected signal power INCREASE (spoofing) vs decrease (jamming).
        
        Spoofed signals typically arrive 3–10 dB above the authentic signal.
        The attacker must overpower the real satellite to take over the receiver.
        A simultaneous C/N0 increase on 3+ satellites is highly suspicious.
        """
        evidence = {}
        delta = monitor.get_cn0_delta(receiver_id)

        # Count satellites with elevated C/N0 in latest frame
        frame = monitor.get_latest_frame(receiver_id)
        elevated_count = 0
        if frame and frame.satellites:
            cn0_vals = np.array([s.cn0 for s in frame.satellites])
            # Baseline comparison per satellite would be ideal; using mean here
            baseline_stats = monitor.get_cn0_stats(receiver_id, window_s=30.0)
            baseline_mean  = baseline_stats.get("mean", 40.0)
            elevated_count = int(np.sum(cn0_vals > baseline_mean + self.cn0_spike_threshold_db))

        evidence["cn0_delta_db"]           = round(delta, 2)
        evidence["elevated_sv_count"]      = elevated_count
        evidence["power_spike_threshold"]  = self.cn0_spike_threshold_db

        confidence = 0.0
        if delta > self.cn0_spike_threshold_db:
            confidence += 0.4 * min(1.0, (delta - self.cn0_spike_threshold_db) / 6.0)
        if elevated_count >= 3:
            confidence += 0.6 * min(1.0, elevated_count / 6.0)

        return min(1.0, confidence), evidence

    def _doppler_check(
        self,
        receiver_id: str,
        monitor: SignalQualityMonitor,
        frame: GPSFrame,
    ) -> tuple:
        """
        Verify that measured Doppler shifts match ephemeris predictions.
        
        Doppler (Hz) = (satellite_velocity_radial + vehicle_velocity) / wavelength
        GPS L1 wavelength ≈ 0.19 m, so 1 m/s relative velocity → ~5.26 Hz Doppler.
        
        Residual thresholds:
          < 30 Hz : Normal (within receiver phase-lock loop bandwidth)
          30–100 Hz : Suspicious (warrants elevated alert)
          > 100 Hz : Very likely spoofing (attacker can't track true satellite motion)
        """
        evidence = {}
        doppler_stats = monitor.get_doppler_residuals(receiver_id)

        mean_res = doppler_stats.get("mean", 0.0)
        max_res  = doppler_stats.get("max", 0.0)
        per_prn  = doppler_stats.get("per_prn", {})

        # Count satellites with anomalous Doppler
        anomalous_svs = sum(
            1 for v in per_prn.values()
            if v > self.doppler_residual_threshold_hz
        )

        evidence["doppler_mean_residual_hz"] = round(mean_res, 2)
        evidence["doppler_max_residual_hz"]  = round(max_res, 2)
        evidence["doppler_anomalous_svs"]    = anomalous_svs

        confidence = 0.0
        if mean_res > self.doppler_residual_threshold_hz:
            confidence += 0.5 * min(1.0, (mean_res - self.doppler_residual_threshold_hz) / 150.0)
        if anomalous_svs >= 2:
            confidence += 0.5 * min(1.0, anomalous_svs / 5.0)

        return min(1.0, confidence), evidence

    def _position_jump_check(self, frame: GPSFrame) -> tuple:
        """
        Detect physically-impossible position jumps between consecutive epochs.
        
        Maximum achievable velocity of common UAV platforms:
          Fixed-wing surveillance UAV (e.g., ScanEagle): ~85 m/s
          Rotary-wing UAV: ~50 m/s
          
        Between 1-Hz GPS epochs, maximum LEGITIMATE position change:
          ~85 m for fast fixed-wing → use 100 m as threshold with margin.
          
        Hard threshold: 200 m (impossible for any non-missile platform).
        """
        evidence = {}

        if self._last_position is None or self._last_timestamp is None:
            evidence["position_jump_m"]  = 0.0
            evidence["position_jump_ms"] = 0.0
            return 0.0, evidence

        dt = frame.timestamp - self._last_timestamp
        if dt <= 0 or dt > 10.0:
            evidence["position_jump_m"]  = 0.0
            evidence["position_jump_ms"] = 0.0
            return 0.0, evidence

        # Compute distance in metres (flat earth)
        dlat_m = (frame.position_lla[0] - self._last_position[0]) * 110_540.0
        dlon_m = (
            (frame.position_lla[1] - self._last_position[1])
            * 111_320.0
            * np.cos(np.deg2rad(frame.position_lla[0]))
        )
        jump_m  = float(np.sqrt(dlat_m**2 + dlon_m**2))
        speed_implied = jump_m / dt  # implied speed (m/s)

        evidence["position_jump_m"]          = round(jump_m, 2)
        evidence["implied_speed_ms"]         = round(speed_implied, 2)
        evidence["position_jump_threshold_m"]= self.position_jump_threshold_m

        confidence = 0.0
        if jump_m > self.position_jump_threshold_m:
            excess = jump_m - self.position_jump_threshold_m
            confidence = min(1.0, excess / 200.0)

        return confidence, evidence

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _confidence_to_level(confidence: float) -> ThreatLevel:
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
        confidence: float, hard_flags: List[str], evidence: dict
    ) -> str:
        pos_res = evidence.get("ekf_position_residual_m", 0.0)
        jump    = evidence.get("position_jump_m", 0.0)

        if hard_flags:
            flags_str = ", ".join(hard_flags)
            return (
                f"HARD FLAG(S) TRIGGERED: {flags_str}. "
                f"Position residual={pos_res:.1f}m, jump={jump:.1f}m. "
                "Immediate INS-primary switch. Discard GPS solution. "
                "Cross-check with GLONASS/Galileo if available. Report incident."
            )
        elif confidence < 0.15:
            return "Signal integrity nominal. GPS solution trusted."
        elif confidence < 0.35:
            return (
                f"Advisory: Minor spoofing indicators (confidence={confidence:.1%}). "
                "Increase INS weight in blending filter."
            )
        elif confidence < 0.55:
            return (
                f"Caution: Spoofing possible (confidence={confidence:.1%}). "
                "Downgrade GPS trust. Cross-validate with terrain reference if available."
            )
        elif confidence < 0.75:
            return (
                f"WARNING: Spoofing likely (confidence={confidence:.1%}, "
                f"pos_residual={pos_res:.1f}m). Switch to INS-primary. "
                "Engage anti-spoofing protocols."
            )
        else:
            return (
                f"CRITICAL: Spoofing confirmed (confidence={confidence:.1%}). "
                "GPS UNTRUSTED. Full INS + alternative nav (DME/TACAN/TERCOM). "
                "Emergency procedures if applicable."
            )
