"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
core/data_types.py

Defines all data structures used across the module:
  - GPSSatellite       : per-satellite measurement (SNR, Doppler, azimuth/elevation)
  - GPSFrame           : full receiver snapshot (position, time, satellite list)
  - IMUFrame           : inertial measurement unit snapshot
  - DetectionResult    : output of any detector (threat type + confidence + evidence)
  - AlertEvent         : escalated, timestamped alert for the alert manager

Design note:
  Using dataclasses + numpy arrays keeps the math clean while staying
  serializable for logging / IPC to a flight computer.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any
import numpy as np
import time


# ─────────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────────

class ThreatType(Enum):
    NONE      = auto()   # Clean signal
    JAMMING   = auto()   # Deliberate RF noise / power flooding
    SPOOFING  = auto()   # Counterfeit navigation signals
    MEACONING = auto()   # Re-broadcast with delay (replay attack)
    UNKNOWN   = auto()   # Anomaly that doesn't fit a known pattern


class ThreatLevel(Enum):
    CLEAR    = 0   # No anomaly detected
    ADVISORY = 1   # Weak indication; monitor
    CAUTION  = 2   # Moderate confidence; degrade trust
    WARNING  = 3   # High confidence; switch to INS-only
    CRITICAL = 4   # Confirmed attack; emergency procedures


class ConstellationID(Enum):
    GPS     = "GPS"      # US – L1 C/A 1575.42 MHz, L2C, L5
    GLONASS = "GLONASS"  # Russia
    GALILEO = "GALILEO"  # EU – has OSNMA authentication
    BEIDOU  = "BEIDOU"   # China
    NAVIC   = "NAVIC"    # India – relevant for Tata/DRDO systems


# ─────────────────────────────────────────────────
#  Per-satellite measurement
# ─────────────────────────────────────────────────

@dataclass
class GPSSatellite:
    """
    One satellite's worth of data in a single receiver epoch.

    SNR / C/N0 explained:
      C/N0 (carrier-to-noise density) is measured in dB-Hz.
      Healthy open-sky values: 35–50 dB-Hz.
      Spoofing typically arrives 3–10 dB ABOVE normal (meaconers pump power).
      Jamming collapses values below 25 dB-Hz.

    Doppler shift:
      Expected from satellite motion + vehicle velocity.
      Spoofed signals often don't match predicted Doppler from orbital ephemeris.
    """
    prn: int                         # Satellite pseudo-random number (ID)
    constellation: ConstellationID
    snr: float                       # Signal-to-noise ratio (dB)
    cn0: float                       # Carrier-to-noise density (dB-Hz)
    doppler_hz: float                # Measured Doppler frequency (Hz)
    predicted_doppler_hz: float      # Predicted from ephemeris + vehicle state
    azimuth_deg: float               # Azimuth in sky (°)
    elevation_deg: float             # Elevation above horizon (°)
    pseudorange_m: float             # Raw pseudorange measurement (m)
    carrier_phase_cycles: float      # Phase measurement (cycles)
    lock_time_ms: float              # Time tracker has held lock (ms)
    healthy: bool = True             # Receiver health flag


# ─────────────────────────────────────────────────
#  GPS receiver frame (one epoch)
# ─────────────────────────────────────────────────

@dataclass
class GPSFrame:
    """
    Complete receiver snapshot at one epoch (~1 Hz for nav-grade).

    position_ecef: Earth-Centered Earth-Fixed (X,Y,Z) metres
    velocity_ecef: ECEF velocity (Vx,Vy,Vz) m/s
    hdop/vdop: dilution of precision — high values mean poor geometry
    agc_gain_db: Automatic Gain Control level — drops when jammer floods RF front-end
    """
    timestamp: float                         # UNIX epoch seconds
    receiver_id: str                         # "RX1", "RX2", etc.
    position_ecef: np.ndarray                # shape (3,) metres
    velocity_ecef: np.ndarray                # shape (3,) m/s
    position_lla: np.ndarray                 # (lat_deg, lon_deg, alt_m)
    satellites: List[GPSSatellite] = field(default_factory=list)
    hdop: float = 1.0
    vdop: float = 1.0
    pdop: float = 1.0
    agc_gain_db: float = 40.0                # Healthy ≈ 38–45 dB
    clock_bias_ns: float = 0.0              # Receiver clock offset (ns)
    num_svs: int = 0                         # Number of tracked satellites
    fix_valid: bool = True

    def mean_cn0(self) -> float:
        if not self.satellites:
            return 0.0
        return float(np.mean([s.cn0 for s in self.satellites]))

    def min_cn0(self) -> float:
        if not self.satellites:
            return 0.0
        return float(np.min([s.cn0 for s in self.satellites]))


# ─────────────────────────────────────────────────
#  IMU frame (one epoch)
# ─────────────────────────────────────────────────

@dataclass
class IMUFrame:
    """
    Inertial Measurement Unit snapshot.

    In a tightly-coupled INS/GPS system (as used in L3 NAVSYS, Tata SkyNav):
      - Accelerometer + gyro run at 200–1000 Hz
      - GPS updates arrive at 1–10 Hz
      - Between GPS fixes, position is propagated via mechanization equations
      - Any divergence between GPS and INS dead-reckoning is a key spoofing indicator

    accel_body: specific force in body frame (m/s²)
    gyro_body:  angular rate in body frame (rad/s)
    """
    timestamp: float
    accel_body: np.ndarray    # shape (3,) — forward/right/down or NED
    gyro_body: np.ndarray     # shape (3,) — roll/pitch/yaw rates
    temperature_c: float = 25.0
    imu_health: bool = True


# ─────────────────────────────────────────────────
#  Detection result from any individual detector
# ─────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """
    Output of a single detection algorithm.

    confidence: 0.0 (no indication) → 1.0 (certain attack)
    evidence:   dict of metric name → observed value,
                used for diagnostics and alert messages.
    """
    detector_name: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    confidence: float                        # 0.0 – 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    recommendation: str = ""


# ─────────────────────────────────────────────────
#  Alert event (escalated, deduplicated)
# ─────────────────────────────────────────────────

@dataclass
class AlertEvent:
    """
    Fused, deduplicated alert sent to the cockpit / GCS / flight controller.

    In UAV context this feeds into the flight management system.
    In battlefield nav context this drives mode switching on the nav computer.
    """
    alert_id: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    fused_confidence: float
    contributing_detectors: List[str]
    position_at_alert: Optional[np.ndarray]  # LLA at time of detection
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    mitigation_action: str = ""
    raw_results: List[DetectionResult] = field(default_factory=list)
