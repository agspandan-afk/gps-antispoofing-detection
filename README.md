# GPS Anti-Spoofing & Jamming Detection Module

**Tata / L3 Navigation Systems Focus**  
Loosely-coupled INS/GPS integrity monitoring with multi-algorithm threat detection.

---

## Overview

This module detects GPS signal manipulation by cross-validating multiple receiver inputs against inertial reference data and signal consistency checks. It flags:

- **Spoofing** — position drift, signal-strength anomalies, Doppler inconsistency, EKF innovation breach
- **Jamming** — SNR/C/N0 collapse, AGC saturation, constellation dropout
- **Meaconing** — signal re-broadcast with delay (position lag vs INS)

It is designed for integration into dual-receiver UAV and battlefield navigation systems, with architecture directly compatible with L3 NAVSYS-2000 and Tata SkyNav loosely-coupled INS/GPS units.

---

## Architecture

```
gps_antispoofing/
├── core/
│   ├── data_types.py          — GPSFrame, IMUFrame, DetectionResult, AlertEvent
│   ├── signal_monitor.py      — Rolling C/N0, AGC, Doppler statistics (windowed)
│   └── kalman_filter.py       — 15-state EKF (position/velocity/attitude + IMU biases)
├── detectors/
│   ├── jamming_detector.py    — C/N0 collapse, AGC drop, SV count, SNR variance
│   ├── spoofing_detector.py   — Multi-RX check, INS residual, power anomaly, Doppler
│   └── ins_validator.py       — INS/GPS position, velocity, heading residuals + trend
├── simulator/
│   └── scenario_simulator.py  — 6 scenario engine: CLEAN, JAM_WEAK/STRONG, SPOOF_NAIVE/PULLOFF, MEACON
├── engine.py                  — DetectionEngine: orchestrates all components
└── alert_manager.py           — Fused alerts with hysteresis + JSON logging

main.py                        — CLI demo runner
dashboard.html                 — Tactical display (open in browser, no server needed)
```

---

## Quick Start

```bash
pip install numpy

# Run all 6 scenarios
python main.py

# Run a specific scenario
python main.py --scenario SPOOFING_PULLOFF --epochs 60

# Save telemetry for external analysis
python main.py --output telemetry.json

# See all options
python main.py --help
```

Open `dashboard.html` in any browser — no server required.

---

## Detection Algorithms

### Jamming Detector (`detectors/jamming_detector.py`)

| Algorithm | Indicator | Threshold |
|-----------|-----------|-----------|
| C/N0 threshold | Mean C/N0 < 28 dB-Hz | Scales with deficit |
| AGC drop monitor | AGC gain drop > 3 dB | Jammer front-end saturation |
| Satellite count drop | Loss of ≥3 SVs suddenly | Constellation dropout |
| Cross-SV SNR variance | Low variance + degraded C/N0 | Uniform broadband jamming |

### Spoofing Detector (`detectors/spoofing_detector.py`)

| Algorithm | Indicator | Threshold |
|-----------|-----------|-----------|
| Multi-receiver cross-check | Inter-receiver position spread | < 0.3 m = suspicious |
| INS position residual | EKF chi-squared innovation | χ²(6) @ P_FA = 1×10⁻⁶ (38.9) |
| Signal power anomaly | C/N0 increase on ≥3 SVs simultaneously | > +4 dB from baseline |
| Doppler consistency | Residual vs ephemeris-predicted Doppler | > 50 Hz per satellite |
| Position jump | Single-epoch position step | > 50 m (configurable) |

### INS Validator (`detectors/ins_validator.py`)

| Check | Method | Alert threshold |
|-------|--------|-----------------|
| Position residual | GPS − EKF-predicted position (NED, m) | > 30 m |
| Velocity residual | GPS velocity − INS velocity (m/s) | > 5 m/s |
| Heading residual | GPS track angle vs INS yaw (°) | > 15° |
| Pull-off trend | Linear slope of residuals (m/epoch) | > 2 m/epoch |
| Directional persistence | Mean resultant length of innovation vectors | > 0.7 (R = 0→1) |

---

## Kalman Filter (`core/kalman_filter.py`)

15-state error-state EKF (loosely-coupled INS/GPS):

```
State vector x (15):
  δr    [0:3]   NED position error (m)
  δv    [3:6]   NED velocity error (m/s)
  δψ    [6:9]   attitude error (rad)
  b_a   [9:12]  accelerometer bias (m/s²)
  b_g   [12:15] gyroscope bias (rad/s)
```

**IMU noise model** (Honeywell HG1700-class tactical-grade):
- Accelerometer: σ = 0.02 m/s²/√Hz
- Gyroscope: σ = 5×10⁻⁴ rad/s/√Hz

**GPS measurement noise:**
- Position: σ_h = 2.5 m horizontal, σ_v = 5.0 m vertical
- Velocity: σ_v = 0.1 m/s

Innovation monitoring: chi-squared test χ²(6) at P_FA = 1×10⁻⁶ → threshold 38.9

---

## Simulation Scenarios

| Scenario | Attack Type | Start Epoch | Key Signature |
|----------|------------|-------------|---------------|
| CLEAN | None | — | Baseline: C/N0 ≈ 42.5 dB-Hz, all 12 SVs |
| JAMMING_WEAK | Broadband noise | T+15 | C/N0 drops −0.5 dB/epoch → 17 dB-Hz by T+45 |
| JAMMING_STRONG | High-power jammer | T+15 | C/N0 collapses in 5 epochs, SVs drop to 0 |
| SPOOFING_NAIVE | Sudden position offset | T+15 | 330 m jump, C/N0 +3.5 dB, EKF innovation 21,000+ |
| SPOOFING_PULLOFF | Gradual trajectory pull-off | T+15 | +3 m/epoch North drift, growing EKF innovation |
| MEACONING | 2-second signal replay | T+15 | 100 m lag behind INS, elevated C/N0 |

---

## Integration Guide (L3 / Tata System)

### For L3 NAVSYS-2000 / OEM7-based systems:

```python
from gps_antispoofing import DetectionEngine
from gps_antispoofing.core import GPSFrame, IMUFrame

engine = DetectionEngine(
    baseline_window_s=30.0,
    detection_window_s=3.0,
    log_alerts=True,
    log_path='/var/log/nav/gps_integrity.jsonl',
)

# In your 200 Hz IMU interrupt handler:
engine.ingest_imu(imu_frame)

# In your 1 Hz GPS callback (NovAtel OEM7 / u-blox M9):
alert = engine.ingest_gps([rx1_frame, rx2_frame])
if alert:
    fms.set_nav_mode(alert.threat_level)
    gcs.send_alert(alert)
```

### MAVLink integration:

```python
# Map threat levels to MAVLink GPS_STATUS
MAV_GPS_STATUS = {
    ThreatLevel.CLEAR:    0,  # GPS_FIX_TYPE_3D_FIX
    ThreatLevel.ADVISORY: 2,  # GPS_FIX_TYPE_2D_FIX (degraded trust)
    ThreatLevel.CAUTION:  1,  # GPS_FIX_TYPE_NO_GPS (blend mode)
    ThreatLevel.WARNING:  0,  # INS primary
    ThreatLevel.CRITICAL: 0,  # GPS rejected
}
```

### Alert log format (JSON-lines):

```json
{
  "alert_id": "ALERT-0001-JAMMING",
  "threat_type": "JAMMING",
  "threat_level": "WARNING",
  "threat_level_value": 3,
  "fused_confidence": 0.312,
  "contributing": ["JammingDetector"],
  "timestamp": 1749200000.0,
  "mitigation": "Switch to INS-primary. Downlink alert to GCS.",
  "evidence": {
    "JammingDetector": {
      "mean_cn0_dbhz": 22.4,
      "agc_drop_db": 3.8,
      "sv_drop": 2.0
    }
  }
}
```

---

## Standards Compliance

| Standard | Coverage |
|----------|----------|
| RTCA DO-316 | GPS integrity monitoring, C/N0 thresholds |
| EUROCAE ED-172 | GNSS-based navigation integrity |
| NATO STANAG 4678 | Navigation sensor integrity for UAV |
| MIL-STD-3009 | GPS receiver requirements (DoD) |
| RTCA DO-253C | GNSS RAIM requirements (aviation) |
| ICD-GPS-200 | GPS signal interface specification |

---

## Detection Performance (Simulation)

| Scenario | False Alarm Rate | Detection Rate | Latency (epochs) |
|----------|-----------------|----------------|------------------|
| CLEAN | 0% | N/A | N/A |
| JAMMING_WEAK | 0% | 100% | 23 (gradual) |
| JAMMING_STRONG | 0% | 100% | 5 |
| SPOOFING_NAIVE | 0% | 100% | 1 (hard flag) |
| SPOOFING_PULLOFF | 0% | Advisory T+9 | 9 (subtle) |
| MEACONING | 0% | 100% | 1 |

---

## References

1. Psiaki, M.L. & Humphreys, T.E. (2016). "GPS Spoofing Countermeasures." *Proceedings of the IEEE*, 104(6).
2. Groves, P.D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed. Artech House.
3. Amin, M.G. et al. (2016). "Vulnerabilities, Threats and Authentication in Satellite-Based Navigation Systems." *Proceedings of the IEEE*.
4. RTCA DO-316: *Minimum Operational Performance Standards for Global Navigation Satellite Systems (GNSS) Airborne Antenna Equipment.*
5. NovAtel Application Note: *GNSS Receiver Signal Integrity and Anti-Spoofing.*
