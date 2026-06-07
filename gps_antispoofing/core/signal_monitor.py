"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
core/signal_monitor.py

SignalQualityMonitor
────────────────────
Maintains a rolling time-window of per-satellite and per-receiver signal
quality metrics.  Every new GPSFrame is ingested here; the monitor computes:

  • Per-satellite C/N0 statistics (mean, variance, delta from baseline)
  • AGC gain trend (jammer signature: gain drops as front-end saturates)
  • Satellite count trend
  • Lock-time tracking (sudden lock loss on multiple SVs = jamming or spoofing)
  • Doppler residuals (|measured − predicted| per satellite)

Why rolling windows?
  A single noisy sample means nothing.  Real attacks cause SUSTAINED changes.
  We track a 30-second baseline window and a 3-second detection window, then
  compare.  This mirrors the approach in "GPS Anti-Spoofing Techniques" (Psiaki
  & Humphreys 2016) and NATO STANAG 4246 signal quality monitoring.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import time

from .data_types import GPSFrame, GPSSatellite


class SignalQualityMonitor:
    """
    Ingests GPS frames and maintains rolling statistics for downstream detectors.

    Parameters
    ----------
    baseline_window_s : float
        Duration of the rolling baseline window (default 30 s).
        Used to compute 'normal' signal conditions for each satellite.
    detection_window_s : float
        Short window for anomaly detection (default 3 s).
    max_receivers : int
        Max number of independent receivers being monitored.
    """

    def __init__(
        self,
        baseline_window_s: float = 30.0,
        detection_window_s: float = 3.0,
        max_receivers: int = 4,
    ):
        self.baseline_window_s = baseline_window_s
        self.detection_window_s = detection_window_s
        self.max_receivers = max_receivers

        # Per-receiver rolling frame buffers
        # key: receiver_id → deque of GPSFrame
        self._frame_buffers: Dict[str, deque] = {}

        self._latest_timestamp: float = 0.0  # updated every ingest() call
        # key: (receiver_id, prn) → deque of (timestamp, cn0)
        self._cn0_history: Dict[Tuple[str, int], deque] = {}

        # AGC history per receiver
        # key: receiver_id → deque of (timestamp, agc_db)
        self._agc_history: Dict[str, deque] = {}

        # Satellite count history
        self._sv_count_history: Dict[str, deque] = {}

        # Doppler residual history: key (receiver_id, prn) → deque of residuals
        self._doppler_residuals: Dict[Tuple[str, int], deque] = {}

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def ingest(self, frame: GPSFrame) -> None:
        """
        Feed a new GPS frame into the monitor.
        All buffers are automatically pruned to their window durations.
        """
        rx = frame.receiver_id
        now = frame.timestamp
        self._latest_timestamp = max(self._latest_timestamp, now)

        # Initialise buffers for new receiver
        if rx not in self._frame_buffers:
            self._frame_buffers[rx] = deque()
            self._agc_history[rx] = deque()
            self._sv_count_history[rx] = deque()

        # Store full frame (baseline window)
        self._frame_buffers[rx].append(frame)
        self._agc_history[rx].append((now, frame.agc_gain_db))
        self._sv_count_history[rx].append((now, len(frame.satellites)))

        # Per-satellite histories
        for sat in frame.satellites:
            key = (rx, sat.prn)
            if key not in self._cn0_history:
                self._cn0_history[key] = deque()
                self._doppler_residuals[key] = deque()

            self._cn0_history[key].append((now, sat.cn0))
            residual = abs(sat.doppler_hz - sat.predicted_doppler_hz)
            self._doppler_residuals[key].append((now, residual))

        # Prune old data
        self._prune(rx, now)

    def get_cn0_stats(self, receiver_id: str, window_s: Optional[float] = None) -> Dict:
        """
        Returns C/N0 statistics across all satellites for a given receiver
        within the requested window (defaults to baseline window).
        
        Returns dict with keys: mean, std, min, max, delta_from_baseline
        """
        if window_s is None:
            window_s = self.baseline_window_s

        now = self._latest_timestamp if self._latest_timestamp > 0 else time.time()
        cutoff = now - window_s
        values = []

        for (rx, prn), history in self._cn0_history.items():
            if rx != receiver_id:
                continue
            values.extend(v for t, v in history if t >= cutoff)

        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n_samples": 0}

        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_samples": len(arr),
        }

    def get_cn0_delta(self, receiver_id: str) -> float:
        """
        Compare mean C/N0 in the recent detection window vs the baseline.
        Negative delta = signal weakening (jamming / obstruction).
        Large positive delta = power increase (spoofing / meaconing).
        """
        now = self._latest_timestamp if self._latest_timestamp > 0 else time.time()
        # Baseline: older half of the buffer (pre-attack period)
        baseline_cutoff = now - self.baseline_window_s
        detect_cutoff   = now - self.detection_window_s

        baseline_vals, recent_vals = [], []
        for (rx, prn), history in self._cn0_history.items():
            if rx != receiver_id:
                continue
            for t, v in history:
                if t >= baseline_cutoff:
                    baseline_vals.append(v)
                if t >= detect_cutoff:
                    recent_vals.append(v)

        if not baseline_vals or not recent_vals:
            return 0.0
        return float(np.mean(recent_vals)) - float(np.mean(baseline_vals))

    def get_agc_stats(self, receiver_id: str) -> Dict:
        """
        AGC monitoring — critical jamming indicator.

        How it works:
          The receiver's RF front-end has a variable-gain amplifier.
          Normally AGC holds a constant gain to keep the ADC input level stable.
          When a jammer floods the band, the AGC backs off (gain drops) to prevent
          saturation.  A drop of >3 dB over seconds is a strong jamming signature.
        """
        history = self._agc_history.get(receiver_id, deque())
        if not history:
            return {"current": 0.0, "mean": 0.0, "drop_db": 0.0}

        now = self._latest_timestamp if self._latest_timestamp > 0 else time.time()
        detect_cutoff = now - self.detection_window_s

        values = [v for _, v in history]
        recent_values = [v for t, v in history if t >= detect_cutoff]
        current = values[-1]
        # Use first half as baseline (before any attack)
        baseline_values = values[: max(1, len(values) // 2)]
        baseline_mean = float(np.mean(baseline_values))

        return {
            "current": current,
            "mean": float(np.mean(values)),
            "baseline_mean": baseline_mean,
            "drop_db": baseline_mean - current,   # positive = AGC backed off
            "std": float(np.std(values)),
        }

    def get_sv_count_stats(self, receiver_id: str) -> Dict:
        """
        Track satellite count over time.
        Sudden drop (e.g., 8 → 3 SVs in <5 s) is a strong jamming or
        geometry-manipulation indicator.
        """
        history = self._sv_count_history.get(receiver_id, deque())
        if not history:
            return {"current": 0, "baseline_mean": 0.0, "drop": 0}

        now = self._latest_timestamp if self._latest_timestamp > 0 else time.time()
        detect_cutoff = now - self.detection_window_s

        values = [v for _, v in history]
        current = values[-1]
        baseline_mean = float(np.mean(values[: max(1, len(values) // 2)]))

        return {
            "current": current,
            "baseline_mean": baseline_mean,
            "drop": baseline_mean - current,  # positive = lost satellites
            "min_recent": min(values[-5:]) if len(values) >= 5 else current,
        }

    def get_doppler_residuals(self, receiver_id: str) -> Dict:
        """
        Doppler consistency check.

        For each satellite:  residual = |measured_Doppler − predicted_Doppler|
        Predicted Doppler comes from orbital mechanics + vehicle velocity (INS).
        Spoofed signals typically have wrong Doppler because the attacker's
        hardware can't perfectly replicate true satellite motion.

        Returns mean, max, and per-PRN breakdown.
        """
        per_prn = {}
        all_residuals = []

        for (rx, prn), history in self._doppler_residuals.items():
            if rx != receiver_id or not history:
                continue
            recent = [v for _, v in history][-10:]   # last 10 epochs
            mean_res = float(np.mean(recent))
            per_prn[prn] = mean_res
            all_residuals.append(mean_res)

        if not all_residuals:
            return {"mean": 0.0, "max": 0.0, "per_prn": {}}

        return {
            "mean": float(np.mean(all_residuals)),
            "max": float(np.max(all_residuals)),
            "std": float(np.std(all_residuals)),
            "per_prn": per_prn,
        }

    def get_latest_frame(self, receiver_id: str) -> Optional[GPSFrame]:
        buf = self._frame_buffers.get(receiver_id)
        if not buf:
            return None
        return buf[-1]

    def get_all_receiver_ids(self) -> List[str]:
        return list(self._frame_buffers.keys())

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _prune(self, receiver_id: str, now: float) -> None:
        """Remove data older than baseline_window_s to bound memory usage."""
        cutoff = now - self.baseline_window_s

        buf = self._frame_buffers.get(receiver_id)
        if buf:
            while buf and buf[0].timestamp < cutoff:
                buf.popleft()

        agc = self._agc_history.get(receiver_id)
        if agc:
            while agc and agc[0][0] < cutoff:
                agc.popleft()

        sv = self._sv_count_history.get(receiver_id)
        if sv:
            while sv and sv[0][0] < cutoff:
                sv.popleft()

        for key, history in self._cn0_history.items():
            if key[0] == receiver_id:
                while history and history[0][0] < cutoff:
                    history.popleft()

        for key, history in self._doppler_residuals.items():
            if key[0] == receiver_id:
                while history and history[0][0] < cutoff:
                    history.popleft()
