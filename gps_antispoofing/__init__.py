"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
gps_antispoofing/__init__.py

Top-level package. Exposes the main DetectionEngine for easy use.

Quick start:
    from gps_antispoofing import DetectionEngine
    from gps_antispoofing.simulator import ScenarioSimulator, Scenario

    engine = DetectionEngine()
    sim    = ScenarioSimulator(Scenario.SPOOFING_PULLOFF)

    for epoch in range(60):
        imu_frames = sim.generate_imu_frames(200)
        gps_frames = sim.generate_gps_frame()

        for imu in imu_frames:
            engine.ingest_imu(imu)

        alert = engine.ingest_gps(gps_frames)
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

__version__ = "1.0.0"
__author__  = "GPS Anti-Spoofing Module"
