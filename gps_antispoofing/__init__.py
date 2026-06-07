"""
GPS Anti-Spoofing & Jamming Detection Module — Live Hardware Build
==================================================================
gps_antispoofing/__init__.py

Quick start:
    from gps_antispoofing import DetectionEngine
    engine = DetectionEngine()

    # Feed IMU at 200 Hz:
    engine.ingest_imu(imu_frame)

    # Feed GPS at 1 Hz:
    alert = engine.ingest_gps([rx1_frame, rx2_frame])
    if alert:
        print(alert.mitigation_action)
"""

from .engine import DetectionEngine
from .alert_manager import AlertManager
from .core import ThreatType, ThreatLevel, ConstellationID

__all__ = [
    "DetectionEngine",
    "AlertManager",
    "ThreatType",
    "ThreatLevel",
    "ConstellationID",
]

__version__ = "1.0.0-live"
