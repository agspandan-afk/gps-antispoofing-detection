"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
simulator/scenario_simulator.py

ScenarioSimulator
─────────────────
Generates synthetic GPS and IMU data for testing detection algorithms.

Scenarios implemented:
  1. CLEAN          : Normal flight, clean signals, no interference
  2. JAMMING_WEAK   : Gradual SNR degradation starting at T+15s
  3. JAMMING_STRONG : AGC collapse + constellation dropout at T+10s
  4. SPOOFING_NAIVE : Sudden position jump (simplistic/unprofessional attacker)
  5. SPOOFING_PULLOFF: Gradual position drift (sophisticated trajectory manipulation)
  6. MEACONING      : Signal re-broadcast with 2-second delay

Physics model:
  The vehicle follows a straight-and-level flight path (constant velocity).
  IMU data is generated from the truth trajectory + realistic sensor noise.
  GPS data is generated from the truth + error model.
  Under attack scenarios, GPS data is deliberately falsified according to
  the attack model while IMU continues to track the truth trajectory.

Signal model parameters:
  Based on ICD-GPS-200 and real-world receiver characterisation
  (NovAtel OEM7, u-blox M8, Septentrio AsteRx-m2).
"""

import numpy as np
from typing import List, Tuple, Optional, Generator
import time
from enum import Enum, auto

from ..core.data_types import (
    GPSFrame, IMUFrame, GPSSatellite, ConstellationID, ThreatType
)


class Scenario(Enum):
    CLEAN           = auto()
    JAMMING_WEAK    = auto()
    JAMMING_STRONG  = auto()
    SPOOFING_NAIVE  = auto()
    SPOOFING_PULLOFF= auto()
    MEACONING       = auto()


# Satellite geometry for simulation (PRN, az_deg, el_deg)
_SATELLITE_CONFIG = [
    (1,  "GPS",     45.0,  65.0),
    (3,  "GPS",    130.0,  42.0),
    (7,  "GPS",    220.0,  38.0),
    (11, "GPS",    310.0,  55.0),
    (14, "GPS",     80.0,  20.0),
    (19, "GPS",    175.0,  30.0),
    (22, "GPS",    260.0,  48.0),
    (28, "GPS",    350.0,  25.0),
    (1,  "GLONASS", 90.0,  50.0),
    (5,  "GLONASS", 200.0, 35.0),
    (3,  "GALILEO", 150.0, 60.0),
    (7,  "GALILEO", 280.0, 45.0),
]

_CONST_MAP = {
    "GPS":     ConstellationID.GPS,
    "GLONASS": ConstellationID.GLONASS,
    "GALILEO": ConstellationID.GALILEO,
}


class ScenarioSimulator:
    """
    Generates GPS + IMU frame pairs for a specified scenario.

    Parameters
    ----------
    scenario : Scenario
        The attack scenario to simulate.
    dt_gps_s : float
        GPS frame interval (default 1.0 s → 1 Hz).
    dt_imu_s : float
        IMU frame interval (default 0.005 s → 200 Hz).
    vehicle_velocity_ms : tuple
        Initial North/East/Down velocity (m/s).
    start_lla : tuple
        Initial position (lat_deg, lon_deg, alt_m).
    noise_seed : int
        Random seed for reproducibility.
    num_receivers : int
        Number of simulated receivers (1–4).
    """

    def __init__(
        self,
        scenario: Scenario = Scenario.CLEAN,
        dt_gps_s: float = 1.0,
        dt_imu_s: float = 0.005,
        vehicle_velocity_ms: Tuple = (50.0, 0.0, 0.0),  # 50 m/s North
        start_lla: Tuple = (28.633, 77.220, 500.0),       # Delhi area
        noise_seed: int = 42,
        num_receivers: int = 2,
    ):
        self.scenario = scenario
        self.dt_gps = dt_gps_s
        self.dt_imu = dt_imu_s
        self.vel_ned = np.array(vehicle_velocity_ms, dtype=float)
        self.start_lla = np.array(start_lla, dtype=float)
        self.num_receivers = num_receivers
        self.rng = np.random.default_rng(noise_seed)

        self._t = 0.0
        self._epoch = 0

        # Truth position (NED from start)
        self._pos_ned = np.zeros(3)
        self._att_rad = np.zeros(3)

        # IMU noise model (tactical-grade IMU, e.g., Honeywell HG1700)
        self._accel_noise_std = 0.01   # m/s²/√Hz
        self._gyro_noise_std  = 2e-4   # rad/s/√Hz
        self._accel_bias = self.rng.normal(0, 0.005, 3)
        self._gyro_bias  = self.rng.normal(0, 1e-4, 3)

        # GPS noise model
        self._gps_pos_noise_h = 1.5    # m (horizontal, 1-sigma)
        self._gps_pos_noise_v = 3.0    # m (vertical, 1-sigma)
        self._gps_vel_noise   = 0.05   # m/s

        # Base C/N0 per satellite (healthy)
        self._base_cn0 = {prn: self.rng.uniform(38, 48) for prn, _, _, _ in _SATELLITE_CONFIG}

        # Attack parameters
        self._attack_start_epoch   = 15      # attack begins at epoch 15
        self._spoof_offset_ned     = np.zeros(3)  # growing position offset
        self._spoof_velocity_ned   = np.array([5.0, 3.0, 0.0])  # pull-off rate m/s

    # ──────────────────────────────────────────────
    # Main generator
    # ──────────────────────────────────────────────

    def generate_gps_frame(self) -> List[GPSFrame]:
        """
        Generate one GPS epoch (all receivers).
        Returns a list of GPSFrame objects (one per receiver).
        """
        self._epoch += 1
        ts = float(time.time()) + self._t

        # Update truth position
        self._pos_ned += self.vel_ned * self.dt_gps
        self._t += self.dt_gps

        # Compute true LLA
        true_lla = self._ned_to_lla(self._pos_ned)

        # Determine attack parameters for this epoch
        attack_active = self._epoch >= self._attack_start_epoch
        attack_epoch  = max(0, self._epoch - self._attack_start_epoch)

        frames = []
        for rx_idx in range(self.num_receivers):
            rx_id = f"RX{rx_idx + 1}"
            frame = self._build_gps_frame(
                ts, rx_id, true_lla, attack_active, attack_epoch, rx_idx
            )
            frames.append(frame)

        return frames

    def generate_imu_frames(self, n: int = 200) -> List[IMUFrame]:
        """
        Generate n IMU frames between GPS epochs.
        IMU tracks the TRUE trajectory (unaffected by GPS attacks).
        """
        frames = []
        # True acceleration = 0 (straight/level) + gravity
        for i in range(n):
            ts = float(time.time()) + self._t - self.dt_gps + i * self.dt_imu
            # Specific force in body frame (gravity + noise)
            accel_true = np.array([0.0, 0.0, -9.80665])  # NED, level flight
            accel = (
                accel_true
                + self._accel_bias
                + self.rng.normal(0, self._accel_noise_std, 3)
            )
            gyro = (
                self._gyro_bias
                + self.rng.normal(0, self._gyro_noise_std, 3)
            )
            frames.append(IMUFrame(
                timestamp=ts,
                accel_body=accel,
                gyro_body=gyro,
            ))
        return frames

    def get_scenario_name(self) -> str:
        return self.scenario.name

    def get_truth_position(self) -> np.ndarray:
        """Returns the true LLA position (unaffected by attack)."""
        return self._ned_to_lla(self._pos_ned)

    # ──────────────────────────────────────────────
    # Frame builder
    # ──────────────────────────────────────────────

    def _build_gps_frame(
        self,
        ts: float,
        rx_id: str,
        true_lla: np.ndarray,
        attack_active: bool,
        attack_epoch: int,
        rx_idx: int,
    ) -> GPSFrame:
        """
        Build a single receiver's GPS frame.
        Applies attack distortions according to the selected scenario.
        """
        scenario = self.scenario
        pos_noise_h = self._gps_pos_noise_h
        vel_noise   = self._gps_vel_noise

        # Default reported position = truth + noise
        rep_lla = true_lla + np.array([
            self.rng.normal(0, pos_noise_h / 110_540.0),
            self.rng.normal(0, pos_noise_h / 111_320.0),
            self.rng.normal(0, self._gps_pos_noise_v),
        ])
        rep_vel = self.vel_ned + self.rng.normal(0, vel_noise, 3)

        agc_db  = self.rng.normal(42.0, 0.5)
        num_svs = len(_SATELLITE_CONFIG)
        cn0_scale = 1.0
        jammed_svs = set()

        # ── Apply attack model ─────────────────────────
        if attack_active:
            epoch = attack_epoch

            if scenario == Scenario.JAMMING_WEAK:
                # Gradual C/N0 degradation: -0.5 dB/epoch
                cn0_scale = max(0.4, 1.0 - epoch * 0.02)
                agc_db   -= min(4.0, epoch * 0.15)

            elif scenario == Scenario.JAMMING_STRONG:
                # Fast collapse
                cn0_scale = max(0.1, 1.0 - epoch * 0.15)
                agc_db   -= min(12.0, epoch * 1.0)
                # Progressive satellite dropout
                svs_to_drop = min(len(_SATELLITE_CONFIG), epoch // 2)
                jammed_svs  = set(range(svs_to_drop))
                num_svs = max(0, len(_SATELLITE_CONFIG) - svs_to_drop)

            elif scenario == Scenario.SPOOFING_NAIVE:
                # Sudden position jump at attack epoch 0
                if epoch == 0:
                    rep_lla[0] += 0.003   # ~330 m North jump
                    rep_lla[1] += 0.002   # ~220 m East jump
                else:
                    rep_lla[0] += 0.003
                    rep_lla[1] += 0.002
                # Spoofed signals arrive at elevated power
                cn0_scale = 1.08   # +3.3 dB boost

            elif scenario == Scenario.SPOOFING_PULLOFF:
                # Gradual pull-off: +3 m/epoch in N direction, +1.5 m/epoch E
                offset_n = epoch * 3.0
                offset_e = epoch * 1.5
                rep_lla[0] += offset_n / 110_540.0
                rep_lla[1] += offset_e / (111_320.0 * np.cos(np.deg2rad(rep_lla[0])))
                # Slightly elevated power (sophisticated attack: +2 dB)
                cn0_scale = 1.05

            elif scenario == Scenario.MEACONING:
                # Replay with 2-second delay: position appears slightly behind
                rep_lla[0] -= (self.vel_ned[0] * 2.0) / 110_540.0
                rep_lla[1] -= (self.vel_ned[1] * 2.0) / 111_320.0
                cn0_scale = 1.06  # replayed signal slightly stronger

        # Build satellite list
        satellites = self._build_satellite_list(cn0_scale, jammed_svs, attack_active)

        return GPSFrame(
            timestamp=ts,
            receiver_id=rx_id,
            position_ecef=self._lla_to_ecef(rep_lla),
            velocity_ecef=np.append(rep_vel, 0.0),
            position_lla=rep_lla,
            satellites=satellites,
            hdop=self.rng.uniform(0.9, 1.4) if num_svs > 4 else self.rng.uniform(2.0, 5.0),
            vdop=self.rng.uniform(1.2, 2.0),
            pdop=self.rng.uniform(1.5, 2.5),
            agc_gain_db=agc_db,
            clock_bias_ns=self.rng.normal(0, 5.0),
            num_svs=len(satellites),
            fix_valid=len(satellites) >= 4,
        )

    def _build_satellite_list(
        self, cn0_scale: float, jammed_svs: set, attack_active: bool
    ) -> List[GPSSatellite]:
        sats = []
        for i, (prn, const_str, az, el) in enumerate(_SATELLITE_CONFIG):
            if i in jammed_svs:
                continue

            base_cn0 = self._base_cn0.get(prn, 42.0)
            cn0 = base_cn0 * cn0_scale + self.rng.normal(0, 0.8)
            snr = cn0 * 0.7 + self.rng.normal(0, 0.5)

            # True Doppler from orbital mechanics (simplified)
            true_doppler = self.rng.normal(1200.0, 300.0)
            # Under spoofing: Doppler residual is larger
            if attack_active and self.scenario in (
                Scenario.SPOOFING_NAIVE, Scenario.SPOOFING_PULLOFF, Scenario.MEACONING
            ):
                meas_doppler = true_doppler + self.rng.normal(80, 40)
            else:
                meas_doppler = true_doppler + self.rng.normal(0, 5)

            lock_time = self.rng.uniform(10_000, 600_000) if cn0 > 25 else 0.0

            sats.append(GPSSatellite(
                prn=prn,
                constellation=_CONST_MAP.get(const_str, ConstellationID.GPS),
                snr=max(0.0, snr),
                cn0=max(0.0, cn0),
                doppler_hz=meas_doppler,
                predicted_doppler_hz=true_doppler,
                azimuth_deg=az,
                elevation_deg=el,
                pseudorange_m=self.rng.uniform(20_000_000, 25_000_000),
                carrier_phase_cycles=self.rng.uniform(0, 1e8),
                lock_time_ms=lock_time,
            ))
        return sats

    # ──────────────────────────────────────────────
    # Coordinate helpers
    # ──────────────────────────────────────────────

    def _ned_to_lla(self, pos_ned: np.ndarray) -> np.ndarray:
        lat0, lon0, alt0 = self.start_lla
        dn, de, dd = pos_ned
        lat = lat0 + dn / 110_540.0
        lon = lon0 + de / (111_320.0 * np.cos(np.deg2rad(lat0)))
        alt = alt0 - dd
        return np.array([lat, lon, alt])

    @staticmethod
    def _lla_to_ecef(lla: np.ndarray) -> np.ndarray:
        """Approximate LLA → ECEF (WGS-84)."""
        lat = np.deg2rad(lla[0])
        lon = np.deg2rad(lla[1])
        alt = lla[2]
        R = 6_378_137.0
        e2 = 0.00669437999014
        N = R / np.sqrt(1 - e2 * np.sin(lat)**2)
        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e2) + alt) * np.sin(lat)
        return np.array([x, y, z])
