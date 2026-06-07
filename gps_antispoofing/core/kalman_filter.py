"""
GPS Anti-Spoofing & Jamming Detection Module
============================================
core/kalman_filter.py

ExtendedKalmanFilter  (loosely-coupled INS/GPS mode)
─────────────────────────────────────────────────────
This is the mathematical heart of the detection system.

Why an EKF for spoofing detection?
────────────────────────────────────
A Kalman filter fuses IMU (high-rate, drifts over time) with GPS (lower-rate,
absolute position).  At each GPS measurement update, the filter computes an
INNOVATION — the difference between the measured GPS position and the position
the filter PREDICTED from IMU data alone.

Under clean conditions:  innovation ≈ small Gaussian noise
Under spoofing:          innovation GROWS as the fake GPS pulls position away
                         from the true INS-predicted trajectory

This "innovation monitoring" (also called RAIM — Receiver Autonomous Integrity
Monitoring) is a standard technique in aviation (RTCA DO-316, DO-253C) and
military navigation (MIL-STD-3009, NATO STANAG 4678).

State vector (15-state error model):
  δr   (3) : position error (m) in NED frame
  δv   (3) : velocity error (m/s) in NED frame  
  δψ   (3) : attitude error (rad)   [roll, pitch, yaw]
  b_a  (3) : accelerometer bias (m/s²)
  b_g  (3) : gyroscope bias (rad/s)

References:
  Groves, P.D. (2013) "Principles of GNSS, Inertial, and Multisensor
  Integrated Navigation Systems", 2nd ed., Artech House.
"""

import numpy as np
from typing import Optional, Tuple
from .data_types import IMUFrame, GPSFrame


# ─────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────

EARTH_RADIUS_M = 6_371_000.0   # WGS-84 mean radius (m)
GRAVITY_MPS2   = 9.80665       # Standard gravity (m/s²)
EARTH_RATE     = 7.2921150e-5  # Earth rotation rate (rad/s)
DEG2RAD        = np.pi / 180.0
RAD2DEG        = 180.0 / np.pi


class ExtendedKalmanFilter:
    """
    15-state loosely-coupled INS/GPS error-state EKF.

    Usage
    -----
    ekf = ExtendedKalmanFilter()
    ekf.initialize(initial_gps_frame)

    # At 200 Hz — propagate with IMU
    for imu_frame in imu_stream:
        ekf.propagate(imu_frame)

    # At 1 Hz — update with GPS, check innovation
    result = ekf.update(gps_frame)
    if result['innovation_norm'] > result['chi2_threshold']:
        # Potential spoofing / measurement fault
    """

    N_STATES = 15

    def __init__(self):
        # State estimate: 15-element error vector (initially zero)
        self.x = np.zeros(self.N_STATES)

        # State covariance
        self.P = np.diag([
            5.0, 5.0, 10.0,          # position errors (m²)
            0.1, 0.1, 0.1,           # velocity errors (m/s)²
            (1e-3)**2, (1e-3)**2, (5e-3)**2,  # attitude errors (rad²)
            (0.05)**2, (0.05)**2, (0.05)**2,  # accel biases (m/s²)²
            (1e-4)**2, (1e-4)**2, (1e-4)**2,  # gyro biases (rad/s)²
        ])

        # Process noise covariance (tuned for tactical-grade IMU)
        accel_noise  = 0.02    # m/s²/√Hz  (e.g., Honeywell HG1700)
        gyro_noise   = 5e-4    # rad/s/√Hz
        accel_bias_instability = 1e-4  # m/s²
        gyro_bias_instability  = 1e-6  # rad/s

        self.Q = np.diag([
            1e-4, 1e-4, 1e-4,
            accel_noise**2, accel_noise**2, accel_noise**2,
            gyro_noise**2,  gyro_noise**2,  gyro_noise**2,
            accel_bias_instability**2, accel_bias_instability**2, accel_bias_instability**2,
            gyro_bias_instability**2,  gyro_bias_instability**2,  gyro_bias_instability**2,
        ])

        # GPS measurement noise (loosely-coupled: position + velocity)
        # position σ = 2.5 m (horizontal), 5.0 m (vertical)
        # velocity σ = 0.1 m/s
        pos_sigma = 2.5
        vel_sigma = 0.1
        self.R = np.diag([
            pos_sigma**2, pos_sigma**2, (pos_sigma * 2)**2,
            vel_sigma**2, vel_sigma**2, vel_sigma**2,
        ])

        # Nominal (truth) navigation state
        self.pos_ned = np.zeros(3)   # NED position (m) from reference
        self.vel_ned = np.zeros(3)   # NED velocity (m/s)
        self.att_rad = np.zeros(3)   # [roll, pitch, yaw] (rad)
        self.C_bn = np.eye(3)        # Body-to-NED rotation matrix

        self.initialized = False
        self.last_imu_timestamp = None

        # Reference origin for NED frame (set at initialization)
        self.origin_lla = None

        # Innovation monitoring history for spoofing detection
        self.innovation_history = []
        self.max_history = 60  # last 60 GPS updates ≈ 60 seconds at 1 Hz

    # ──────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────

    def initialize(self, gps_frame: GPSFrame) -> None:
        """
        Seed the filter with the first valid GPS fix.
        Sets the NED reference origin at the initial position.
        """
        self.origin_lla = gps_frame.position_lla.copy()
        self.pos_ned = np.zeros(3)
        self.vel_ned = gps_frame.velocity_ecef.copy()[:3]
        self.att_rad = np.zeros(3)
        self.C_bn = np.eye(3)
        self.x = np.zeros(self.N_STATES)
        self.last_imu_timestamp = gps_frame.timestamp
        self.initialized = True

    # ──────────────────────────────────────────────
    # IMU propagation (runs at high rate: 100–1000 Hz)
    # ──────────────────────────────────────────────

    def propagate(self, imu: IMUFrame) -> None:
        """
        Dead-reckoning: integrate IMU to propagate position/velocity/attitude.

        This is the core of INS mechanization:
          1. Rotate specific force to NED using current attitude
          2. Subtract gravity
          3. Integrate to get velocity delta
          4. Integrate velocity to get position delta
          5. Update attitude using gyro data (Rodrigues rotation formula)

        Then propagate the error covariance using the system Jacobian F.
        """
        if not self.initialized:
            return

        if self.last_imu_timestamp is None:
            self.last_imu_timestamp = imu.timestamp
            return

        dt = imu.timestamp - self.last_imu_timestamp
        if dt <= 0 or dt > 1.0:
            self.last_imu_timestamp = imu.timestamp
            return

        # Correct IMU for estimated biases
        accel_b = imu.accel_body - self.x[9:12]
        gyro_b  = imu.gyro_body  - self.x[12:15]

        # Rotate specific force to NED frame
        f_ned = self.C_bn @ accel_b

        # Apply gravity (NED: gravity points Down = positive)
        # Specific force f = a_true - g_ned  →  a_true = f_ned + g_ned
        gravity_ned = np.array([0.0, 0.0, GRAVITY_MPS2])
        accel_ned = f_ned + gravity_ned  # correct NED mechanization

        # Update velocity
        self.vel_ned += accel_ned * dt

        # Update position
        self.pos_ned += self.vel_ned * dt

        # Update attitude via gyro integration (small angle)
        dtheta = gyro_b * dt
        # Build skew-symmetric matrix for rotation update
        self.att_rad += dtheta  # simplified; full impl uses quaternions

        # Update rotation matrix (small angle approximation)
        dC = np.array([
            [1.0,        -dtheta[2],  dtheta[1]],
            [dtheta[2],   1.0,       -dtheta[0]],
            [-dtheta[1],  dtheta[0],  1.0      ],
        ])
        self.C_bn = self.C_bn @ dC

        # ── Covariance propagation ──
        # Build system Jacobian F (15×15 linearised dynamics matrix)
        F = self._build_F(dt)
        self.P = F @ self.P @ F.T + self.Q * dt

        self.last_imu_timestamp = imu.timestamp

    # ──────────────────────────────────────────────
    # GPS measurement update (runs at 1 Hz typically)
    # ──────────────────────────────────────────────

    def update(self, gps: GPSFrame) -> dict:
        """
        Measurement update using GPS position and velocity.

        Returns a dict with:
          innovation         : 6-element innovation vector (pos + vel)
          innovation_norm    : Mahalanobis distance (chi-squared statistic)
          chi2_threshold     : Detection threshold (chi² with 6 dof at P_FA=1e-6)
          is_anomalous       : True if innovation exceeds threshold
          pos_residual_m     : Position residual magnitude (m)
          correction         : Applied state correction
        """
        if not self.initialized:
            self.initialize(gps)
            return {"innovation_norm": 0.0, "is_anomalous": False, "pos_residual_m": 0.0}

        # Convert GPS position to NED relative to origin
        gps_pos_ned = self._lla_to_ned(gps.position_lla)
        gps_vel_ned = gps.velocity_ecef[:3].copy()

        # Predicted measurement from current state
        pred_pos = self.pos_ned + self.x[0:3]
        pred_vel = self.vel_ned + self.x[3:6]

        # Innovation: measured − predicted
        z = np.concatenate([
            gps_pos_ned - pred_pos,
            gps_vel_ned - pred_vel,
        ])

        # Measurement matrix H (6×15): maps state to measurement space
        H = np.zeros((6, self.N_STATES))
        H[0:3, 0:3] = np.eye(3)   # position
        H[3:6, 3:6] = np.eye(3)   # velocity

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Mahalanobis distance (chi-squared statistic)
        # Under clean conditions: z^T * S^{-1} * z ~ χ²(6)
        # At P_FA = 1e-6: threshold ≈ 38.9
        S_inv = np.linalg.inv(S)
        innovation_norm = float(z @ S_inv @ z)
        chi2_threshold = 38.9  # χ²(6) at false alarm rate 1e-6 (MIL-SPEC grade)

        # Kalman gain
        K = self.P @ H.T @ S_inv

        # State update
        self.x = self.x + K @ z

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Apply correction to nominal state
        self.pos_ned += self.x[0:3]
        self.vel_ned += self.x[3:6]
        self.att_rad += self.x[6:9]
        # Reset error state (feedback loop)
        self.x[:9] = 0.0

        pos_residual_m = float(np.linalg.norm(z[:3]))
        is_anomalous = innovation_norm > chi2_threshold

        # Record for trend analysis
        self.innovation_history.append({
            "timestamp": gps.timestamp,
            "innovation_norm": innovation_norm,
            "pos_residual_m": pos_residual_m,
            "is_anomalous": is_anomalous,
        })
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)

        return {
            "innovation": z,
            "innovation_norm": innovation_norm,
            "chi2_threshold": chi2_threshold,
            "is_anomalous": is_anomalous,
            "pos_residual_m": pos_residual_m,
            "ins_position_ned": self.pos_ned.copy(),
            "gps_position_ned": gps_pos_ned,
        }

    def get_innovation_trend(self) -> float:
        """
        Returns the slope of the innovation norm over recent history.
        Rising slope under a sustained spoofing attack = pull-off trajectory.
        """
        if len(self.innovation_history) < 5:
            return 0.0
        vals = [h["innovation_norm"] for h in self.innovation_history[-10:]]
        x = np.arange(len(vals), dtype=float)
        slope = float(np.polyfit(x, vals, 1)[0])
        return slope

    # ──────────────────────────────────────────────
    # Coordinate helpers
    # ──────────────────────────────────────────────

    def _lla_to_ned(self, lla: np.ndarray) -> np.ndarray:
        """
        Convert LLA (lat, lon, alt) to NED (North, East, Down) metres
        relative to the stored reference origin.
        
        Uses flat-earth approximation (valid within ~50 km of origin).
        """
        if self.origin_lla is None:
            return np.zeros(3)

        lat0 = self.origin_lla[0] * DEG2RAD
        lat  = lla[0] * DEG2RAD
        lon0 = self.origin_lla[1] * DEG2RAD
        lon  = lla[1] * DEG2RAD

        dN = (lat  - lat0) * EARTH_RADIUS_M
        dE = (lon  - lon0) * EARTH_RADIUS_M * np.cos(lat0)
        dD = -(lla[2] - self.origin_lla[2])

        return np.array([dN, dE, dD])

    def _build_F(self, dt: float) -> np.ndarray:
        """
        Linearised dynamics Jacobian for error-state propagation.
        Simplified form (full form includes Coriolis, gravity gradient, etc.)
        """
        F = np.eye(self.N_STATES)
        # Position = position + velocity * dt
        F[0:3, 3:6] = np.eye(3) * dt
        # Velocity affected by accel bias
        F[3:6, 9:12] = -self.C_bn * dt
        # Attitude affected by gyro bias
        F[6:9, 12:15] = -np.eye(3) * dt
        return F
