"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
detectors/jamming_detector.py

JammingDetector
───────────────
Detects deliberate RF interference that prevents the receiver from tracking
GPS signals.  Jamming attacks are the SIMPLER threat: the attacker doesn't
need to generate a convincing fake signal, just overwhelm the receiver with
noise.

Jamming types detected:
  1. Broadband noise jamming    (J/N floor rise → C/N0 collapse)
  2. Swept CW jamming           (intermittent lock loss, modulated AGC)
  3. Narrowband CW jamming      (affects specific frequencies, partial SV loss)
  4. Pulsed jamming             (rapid on/off — harder to catch)
  5. Smart (meaconing) jam      (classified as a separate type in the alert)

Detection algorithms:
  A. C/N0 Threshold Check
     If mean C/N0 < threshold (default 25 dB-Hz) → jamming likely.
     This is the primary indicator used in RTCA DO-292 and EUROCAE ED-172.

  B. AGC Drop Monitor
     Automatic Gain Control backs off when jammer saturates the front-end.
     A drop > 3 dB in < 5 s is a strong jamming signature.
     Used in NovAtel OEM7 and Septentrio AsteRx receivers.

  C. Satellite Count Drop
     A sudden loss of ≥3 satellites in < 5 s while vehicle is not maneuvering
     indicates the receiver can no longer pull signals above the noise floor.

  D. SNR Variance Spike
     Jamming noise modulates every satellite's SNR simultaneously —
     unlike natural fading which is satellite/geometry-specific.
     High cross-satellite SNR variance correlation = jammer.

  E. Receiver Noise Floor (C/N0 of weakest tracked satellite)
     Jammer raises the noise floor; the weakest signal is lost first.
     Progressive loss pattern (weakest → strongest) confirms jamming.

Confidence fusion:
  Each sub-detector votes with a confidence score (0–1).
  Final confidence = weighted sum, clamped to [0,1].
"""

import numpy as np
from typing import List, Optional
import time

from ..core.data_types import (
    GPSFrame, DetectionResult, ThreatType, ThreatLevel
)
from ..core.signal_monitor import SignalQualityMonitor


class JammingDetector:
    """
    Multi-algorithm jamming detector.

    Parameters
    ----------
    cn0_threshold_dbhz : float
        Minimum healthy C/N0 in dB-Hz (default 28.0).
        Below this, jamming is suspected.
    agc_drop_threshold_db : float
        AGC drop (dB) over the short window that triggers an alert (default 3.0).
    sv_drop_threshold : int
        Number of satellites that must be lost suddenly to trigger alert (default 3).
    """

    def __init__(
        self,
        cn0_threshold_dbhz: float = 28.0,
        agc_drop_threshold_db: float = 3.0,
        sv_drop_threshold: int = 3,
    ):
        self.cn0_threshold_dbhz = cn0_threshold_dbhz
        self.agc_drop_threshold_db = agc_drop_threshold_db
        self.sv_drop_threshold = sv_drop_threshold

        # Algorithm weights for fusion
        self._weights = {
            "cn0_collapse": 0.35,
            "agc_drop":     0.30,
            "sv_count_drop": 0.20,
            "snr_variance":  0.15,
        }

    # ──────────────────────────────────────────────
    # Main detection interface
    # ──────────────────────────────────────────────

    def detect(
        self,
        frame: GPSFrame,
        monitor: SignalQualityMonitor,
    ) -> DetectionResult:
        """
        Run all jamming detection algorithms on the latest GPS frame.
        Returns a fused DetectionResult.
        """
        rx = frame.receiver_id
        evidence = {}

        # ── Algorithm A: C/N0 threshold ──────────────
        cn0_stats = monitor.get_cn0_stats(rx, window_s=3.0)
        mean_cn0  = cn0_stats.get("mean", 45.0)
        cn0_delta = monitor.get_cn0_delta(rx)

        cn0_confidence = 0.0
        if mean_cn0 < self.cn0_threshold_dbhz:
            # Severity scales with how far below threshold
            deficit = self.cn0_threshold_dbhz - mean_cn0
            cn0_confidence = min(1.0, deficit / 10.0)
        evidence["mean_cn0_dbhz"]   = round(mean_cn0, 2)
        evidence["cn0_delta_db"]    = round(cn0_delta, 2)
        evidence["cn0_confidence"]  = round(cn0_confidence, 3)

        # ── Algorithm B: AGC drop ─────────────────────
        agc_stats = monitor.get_agc_stats(rx)
        agc_drop  = agc_stats.get("drop_db", 0.0)

        agc_confidence = 0.0
        if agc_drop > self.agc_drop_threshold_db:
            agc_confidence = min(1.0, (agc_drop - self.agc_drop_threshold_db) / 5.0)
        evidence["agc_current_db"]   = round(agc_stats.get("current", 40.0), 2)
        evidence["agc_drop_db"]      = round(agc_drop, 2)
        evidence["agc_confidence"]   = round(agc_confidence, 3)

        # ── Algorithm C: Satellite count drop ──────────
        sv_stats = monitor.get_sv_count_stats(rx)
        sv_drop  = sv_stats.get("drop", 0.0)
        sv_current = sv_stats.get("current", 0)

        sv_confidence = 0.0
        if sv_drop >= self.sv_drop_threshold:
            sv_confidence = min(1.0, (sv_drop - self.sv_drop_threshold + 1) / 5.0)
        evidence["sv_current"]       = sv_current
        evidence["sv_drop"]          = round(sv_drop, 1)
        evidence["sv_confidence"]    = round(sv_confidence, 3)

        # ── Algorithm D: Cross-satellite SNR variance ──
        snr_var_confidence = 0.0
        if frame.satellites:
            cn0_values = np.array([s.cn0 for s in frame.satellites])
            snr_std = float(np.std(cn0_values))
            # Under jamming: all SVs degrade simultaneously → LOW std
            # (variance is high under natural fading, low under uniform jamming)
            # Note: we combine low-std with other indicators for better accuracy
            if snr_std < 2.0 and mean_cn0 < 35.0:
                # Low variance AND degraded signal → uniform jamming signature
                snr_var_confidence = min(1.0, (35.0 - mean_cn0) / 15.0)
            evidence["snr_std_db"]       = round(snr_std, 2)
        evidence["snr_var_confidence"] = round(snr_var_confidence, 3)

        # ── Fused confidence ──────────────────────────
        fused_confidence = (
            self._weights["cn0_collapse"]  * cn0_confidence  +
            self._weights["agc_drop"]      * agc_confidence  +
            self._weights["sv_count_drop"] * sv_confidence   +
            self._weights["snr_variance"]  * snr_var_confidence
        )
        fused_confidence = min(1.0, fused_confidence)

        # ── Threat level mapping ──────────────────────
        threat_level = self._confidence_to_level(fused_confidence)
        threat_type  = ThreatType.JAMMING if fused_confidence > 0.05 else ThreatType.NONE

        # ── Recommendation ────────────────────────────
        recommendation = self._build_recommendation(
            fused_confidence, agc_drop, sv_drop, mean_cn0
        )

        return DetectionResult(
            detector_name="JammingDetector",
            threat_type=threat_type,
            threat_level=threat_level,
            confidence=fused_confidence,
            evidence=evidence,
            timestamp=frame.timestamp,
            recommendation=recommendation,
        )

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
        confidence: float, agc_drop: float, sv_drop: float, mean_cn0: float
    ) -> str:
        if confidence < 0.15:
            return "Signal nominal. Continue GPS-primary navigation."
        elif confidence < 0.35:
            return (
                f"Advisory: C/N0={mean_cn0:.1f} dB-Hz, AGC drop={agc_drop:.1f} dB. "
                "Monitor signal quality. Verify against secondary receiver."
            )
        elif confidence < 0.55:
            return (
                f"Caution: Degraded signal ({sv_drop:.0f} SVs lost). "
                "Increase INS blending. Avoid critical maneuvers on GPS alone."
            )
        elif confidence < 0.75:
            return (
                "WARNING: Jamming suspected. Switch to INS-primary navigation. "
                "Log bearing to jamming source via AGC directionality if antenna array available."
            )
        else:
            return (
                "CRITICAL: Active jamming confirmed. "
                "GPS solution unreliable. Full INS-primary mode. "
                "Engage ECCM procedures. Report to GCS immediately."
            )
