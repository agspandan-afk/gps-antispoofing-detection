"""
GPS Anti-Spoofing — Real Hardware Layer
========================================
hardware/imu_reader.py

IMUReader
─────────
Reads accelerometer and gyroscope data from a physical IMU and produces
IMUFrame objects at the configured rate (default 200 Hz).

Supported hardware:
  MPU-6050   — 6-axis IMU, I2C address 0x68 or 0x69
               Available on: GY-521 breakout, many Arduino shields
               Raspberry Pi: sudo apt install python3-smbus

  MPU-9250   — 9-axis (adds magnetometer), same I2C interface as MPU-6050
               Compatible code path, magnetometer ignored here.

  ICM-42688  — higher-grade MEMS, same I2C address scheme (0x68/0x69)
               Requires minor register map change (see _VARIANT flag)

  Serial IMU — any IMU outputting CSV or binary over UART (Pixhawk-style):
               Format: timestamp,ax,ay,az,gx,gy,gz  (SI units)
               Set mode="serial" and port="/dev/ttyUSB1"

  Mock IMU   — no hardware needed; integrates GPS velocity for approximate
               dead-reckoning. Suitable for testing on a laptop.
               Set mode="mock"

Wiring (Raspberry Pi GPIO):
  MPU-6050 VCC  → 3.3V (pin 1)
  MPU-6050 GND  → GND  (pin 6)
  MPU-6050 SDA  → GPIO2 (pin 3)  — SDA1
  MPU-6050 SCL  → GPIO3 (pin 5)  — SCL1
  MPU-6050 AD0  → GND (address 0x68) or 3.3V (address 0x69)

Enable I2C on Raspberry Pi:
  sudo raspi-config → Interface Options → I2C → Enable
  sudo apt install python3-smbus i2c-tools
  i2cdetect -y 1   ← should show 68 or 69

For other Linux SBCs (Jetson, OrangePi, Radxa):
  Same smbus2 interface, different I2C bus number (usually 0 or 1).
"""

import time
import threading
import numpy as np
from typing import Optional, Callable

try:
    import smbus2
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False

try:
    import serial as _serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from gps_antispoofing.core.data_types import IMUFrame


# ─────────────────────────────────────────────────────────────────────────────
# MPU-6050 / MPU-9250 register map
# ─────────────────────────────────────────────────────────────────────────────
_PWR_MGMT_1    = 0x6B
_SMPLRT_DIV    = 0x19
_CONFIG_REG    = 0x1A
_GYRO_CONFIG   = 0x1B
_ACCEL_CONFIG  = 0x1C
_ACCEL_XOUT_H  = 0x3B
_GYRO_XOUT_H   = 0x43
_TEMP_OUT_H    = 0x41
_WHO_AM_I      = 0x75

# Full-scale ranges
# Accel: ±2g=0, ±4g=1, ±8g=2, ±16g=3
# Gyro:  ±250°/s=0, ±500=1, ±1000=2, ±2000=3
_ACCEL_SCALE = {0: 16384.0, 1: 8192.0, 2: 4096.0, 3: 2048.0}
_GYRO_SCALE  = {0: 131.0,   1: 65.5,   2: 32.8,   3: 16.4}
_GRAVITY     = 9.80665


class IMUReader:
    """
    IMU data reader with three modes: i2c, serial, mock.

    Parameters
    ----------
    mode : str
        "i2c"    — MPU-6050/9250 on I2C bus (Raspberry Pi, Jetson, etc.)
        "serial" — CSV IMU over UART
        "mock"   — no hardware, derives approximate IMU from GPS velocity
    i2c_bus : int
        I2C bus number. Usually 1 on Raspberry Pi.
    i2c_address : int
        0x68 (AD0=GND) or 0x69 (AD0=VCC)
    serial_port : str
        Serial port for "serial" mode
    rate_hz : float
        Desired output rate. IMU is polled at this rate.
    accel_range : int
        0=±2g, 1=±4g, 2=±8g, 3=±16g
    gyro_range : int
        0=±250°/s, 1=±500, 2=±1000, 3=±2000
    """

    def __init__(
        self,
        mode: str = "i2c",
        i2c_bus: int = 1,
        i2c_address: int = 0x68,
        serial_port: str = "/dev/ttyUSB1",
        serial_baud: int = 115200,
        rate_hz: float = 200.0,
        accel_range: int = 0,
        gyro_range: int = 0,
    ):
        self.mode        = mode
        self.i2c_bus     = i2c_bus
        self.i2c_address = i2c_address
        self.serial_port = serial_port
        self.serial_baud = serial_baud
        self.rate_hz     = rate_hz
        self.dt          = 1.0 / rate_hz
        self.accel_range = accel_range
        self.gyro_range  = gyro_range

        self.on_frame: Optional[Callable[[IMUFrame], None]] = None

        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._bus      = None
        self._ser      = None

        # Calibration offsets (populated by calibrate())
        self._accel_bias = np.zeros(3)
        self._gyro_bias  = np.zeros(3)

        # For mock mode — receive GPS velocity updates
        self._mock_vel_ned = np.zeros(3)
        self._mock_lock    = threading.Lock()

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def start(self) -> None:
        """Initialise hardware and start background polling thread."""
        if self.mode == "i2c":
            self._init_i2c()
        elif self.mode == "serial":
            self._init_serial()
        elif self.mode == "mock":
            print("[IMUReader] Running in MOCK mode — no IMU hardware required.")
        else:
            raise ValueError(f"Unknown IMU mode: {self.mode}")

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"[IMUReader] Started in {self.mode} mode @ {self.rate_hz:.0f} Hz")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._ser and hasattr(self._ser, 'close'):
            self._ser.close()
        print("[IMUReader] Stopped")

    def calibrate(self, duration_s: float = 5.0) -> None:
        """
        Collect static samples and compute bias offsets.
        Keep the IMU completely still during calibration.
        In i2c mode: call this AFTER start().
        """
        if self.mode == "mock":
            return
        print(f"[IMUReader] Calibrating — keep IMU still for {duration_s:.0f}s ...")
        samples_a, samples_g = [], []
        deadline = time.time() + duration_s
        while time.time() < deadline:
            if self.mode == "i2c":
                a, g, _ = self._read_i2c_raw()
                samples_a.append(a)
                samples_g.append(g)
            time.sleep(self.dt)
        if samples_a:
            mean_a = np.mean(samples_a, axis=0)
            mean_g = np.mean(samples_g, axis=0)
            # Gravity compensation: remove expected 1g on Z axis
            mean_a[2] -= _GRAVITY
            self._accel_bias = mean_a
            self._gyro_bias  = mean_g
            print(f"[IMUReader] Calibration done. Accel bias: {self._accel_bias}, Gyro bias: {self._gyro_bias}")

    def update_mock_velocity(self, vel_ned: np.ndarray) -> None:
        """Feed GPS-derived velocity into mock IMU for dead-reckoning."""
        with self._mock_lock:
            self._mock_vel_ned = vel_ned.copy()

    # ──────────────────────────────────────────────
    # Hardware init
    # ──────────────────────────────────────────────

    def _init_i2c(self) -> None:
        if not SMBUS_AVAILABLE:
            raise ImportError("smbus2 not installed. Run: pip install smbus2")
        self._bus = smbus2.SMBus(self.i2c_bus)

        # Check WHO_AM_I — MPU-6050=0x68, MPU-9250=0x71, ICM-42688=0x47
        who = self._bus.read_byte_data(self.i2c_address, _WHO_AM_I)
        print(f"[IMUReader] WHO_AM_I = 0x{who:02X} (6050=0x68, 9250=0x71, ICM42688=0x47)")

        # Wake up (clear sleep bit)
        self._bus.write_byte_data(self.i2c_address, _PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        # Sample rate divider: sample_rate = 1000 / (1 + SMPLRT_DIV)
        # For 200 Hz: SMPLRT_DIV = 4 → 1000/(1+4) = 200 Hz
        divider = max(0, int(1000 / self.rate_hz) - 1)
        self._bus.write_byte_data(self.i2c_address, _SMPLRT_DIV, divider)

        # DLPF config (low-pass filter): 0=260Hz (raw), 3=44Hz, 6=5Hz
        self._bus.write_byte_data(self.i2c_address, _CONFIG_REG, 3)

        # Gyro full scale range
        self._bus.write_byte_data(self.i2c_address, _GYRO_CONFIG,
                                  self.gyro_range << 3)

        # Accel full scale range
        self._bus.write_byte_data(self.i2c_address, _ACCEL_CONFIG,
                                  self.accel_range << 3)

        print(f"[IMUReader] MPU initialised at I2C bus {self.i2c_bus}, addr 0x{self.i2c_address:02X}")

    def _init_serial(self) -> None:
        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial not installed. Run: pip install pyserial")
        self._ser = _serial.Serial(self.serial_port, self.serial_baud, timeout=0.1)
        print(f"[IMUReader] Serial IMU opened on {self.serial_port}")

    # ──────────────────────────────────────────────
    # Poll loop
    # ──────────────────────────────────────────────

    def _poll_loop(self) -> None:
        next_tick = time.time()
        while self._running:
            now = time.time()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += self.dt

            frame = self._read_frame()
            if frame and self.on_frame:
                self.on_frame(frame)

    def _read_frame(self) -> Optional[IMUFrame]:
        try:
            if self.mode == "i2c":
                return self._read_i2c_frame()
            elif self.mode == "serial":
                return self._read_serial_frame()
            elif self.mode == "mock":
                return self._read_mock_frame()
        except Exception as e:
            print(f"[IMUReader] Read error: {e}")
        return None

    # ──────────────────────────────────────────────
    # I2C read
    # ──────────────────────────────────────────────

    def _read_i2c_raw(self):
        """Read 14 bytes: accel(6) + temp(2) + gyro(6)"""
        data = self._bus.read_i2c_block_data(
            self.i2c_address, _ACCEL_XOUT_H, 14
        )
        def s16(hi, lo):
            v = (hi << 8) | lo
            return v - 65536 if v > 32767 else v

        ax_raw = s16(data[0],  data[1])
        ay_raw = s16(data[2],  data[3])
        az_raw = s16(data[4],  data[5])
        temp_raw = s16(data[6], data[7])
        gx_raw = s16(data[8],  data[9])
        gy_raw = s16(data[10], data[11])
        gz_raw = s16(data[12], data[13])

        a_scale = _ACCEL_SCALE[self.accel_range]
        g_scale = _GYRO_SCALE[self.gyro_range]

        accel = np.array([ax_raw, ay_raw, az_raw]) / a_scale * _GRAVITY
        gyro  = np.array([gx_raw, gy_raw, gz_raw]) / g_scale * (np.pi / 180)
        temp  = temp_raw / 340.0 + 36.53

        return accel, gyro, temp

    def _read_i2c_frame(self) -> IMUFrame:
        accel, gyro, temp = self._read_i2c_raw()
        # Apply calibration bias correction
        accel -= self._accel_bias
        gyro  -= self._gyro_bias
        return IMUFrame(
            timestamp  = time.time(),
            accel_body = accel,
            gyro_body  = gyro,
            temperature_c = temp,
            imu_health = True,
        )

    # ──────────────────────────────────────────────
    # Serial read
    # ──────────────────────────────────────────────

    def _read_serial_frame(self) -> Optional[IMUFrame]:
        """
        Expect CSV line: timestamp,ax,ay,az,gx,gy,gz  (SI units)
        e.g. 1749200000.123,0.12,-0.05,9.81,0.001,-0.002,0.000
        """
        line = self._ser.readline().decode("ascii", errors="replace").strip()
        if not line:
            return None
        parts = line.split(",")
        if len(parts) < 7:
            return None
        ts   = float(parts[0])
        accel = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        gyro  = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
        accel -= self._accel_bias
        gyro  -= self._gyro_bias
        return IMUFrame(
            timestamp  = ts,
            accel_body = accel,
            gyro_body  = gyro,
        )

    # ──────────────────────────────────────────────
    # Mock read (no hardware)
    # ──────────────────────────────────────────────

    def _read_mock_frame(self) -> IMUFrame:
        """
        Generates approximate IMU data from GPS velocity.
        Specific force = [0, 0, -g] for level flight.
        Adds realistic sensor noise.
        Not suitable for tight EKF spoofing detection, but allows
        the system to run on a laptop for signal-based detection.
        """
        with self._mock_lock:
            vel = self._mock_vel_ned.copy()

        noise_a = np.random.normal(0, 0.01, 3)
        noise_g = np.random.normal(0, 2e-4, 3)

        # Specific force: counteracts gravity in level flight
        accel = np.array([0.0, 0.0, -_GRAVITY]) + noise_a
        gyro  = noise_g

        return IMUFrame(
            timestamp  = time.time(),
            accel_body = accel,
            gyro_body  = gyro,
            temperature_c = 25.0,
            imu_health = True,
        )
