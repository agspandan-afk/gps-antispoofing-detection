# GPS Anti-Spoofing — Live Hardware Mode

Real GPS receiver + IMU → DetectionEngine → Live alerts.
The simulator is completely removed. All detection algorithms are identical.

---

## What's in this zip

```
gps_antispoofing/        ← detection engine (unchanged from simulation version)
hardware/
  gps_reader.py          ← NMEA serial reader → GPSFrame
  imu_reader.py          ← MPU-6050/9250 I2C reader → IMUFrame
  nmea_parser.py         ← NMEA 0183 sentence parser
main_live.py             ← live hardware entry point (replaces main.py)
requirements_real.txt
README_REAL.md
```

---

## Hardware you need

**Minimum (laptop, no IMU):**
- Any USB GPS receiver outputting NMEA 0183
- Tested: u-blox M8 (NEO-M8N/M8U), u-blox M9 (ZED-F9P), Quectel LC29H
- ~£15–30 on Amazon/AliExpress

**Full setup (Raspberry Pi / Jetson, best detection):**
- GPS receiver (above)
- MPU-6050 or MPU-9250 IMU breakout board (~£3)
- Jumper wires

---

## Install

```bash
pip install numpy pyserial

# Only if using MPU-6050 IMU on Raspberry Pi:
sudo apt install python3-smbus
pip install smbus2

# Enable I2C on Raspberry Pi:
sudo raspi-config → Interface Options → I2C → Enable
```

---

## Find your GPS port

**Linux:**
```bash
ls /dev/tty* | grep -E "USB|ACM"   # before and after plugging in
# Usually: /dev/ttyUSB0  or  /dev/ttyACM0
dmesg | tail -5                     # shows exact device name
```

**Windows:** Device Manager → Ports (COM & LPT) → note COM number

**macOS:**
```bash
ls /dev/cu.*
# Usually: /dev/cu.usbserial-XXXX  or  /dev/cu.usbmodem-XXXX
```

---

## Run

```bash
# Laptop / desktop — USB GPS only, no IMU:
python main_live.py --gps1 /dev/ttyUSB0 --no-imu

# Raspberry Pi — GPS + MPU-6050:
python main_live.py --gps1 /dev/ttyACM0 --imu-mode i2c

# Dual receiver (best spoofing detection):
python main_live.py --gps1 /dev/ttyUSB0 --gps2 /dev/ttyUSB1 --imu-mode i2c

# Windows:
python main_live.py --gps1 COM3 --no-imu

# Save alerts to log file:
python main_live.py --gps1 /dev/ttyUSB0 --no-imu --log alerts.jsonl

# Different baud rate (older receivers often 9600):
python main_live.py --gps1 /dev/ttyUSB0 --baud1 9600 --no-imu

# MPU-6050 at alternative I2C address (AD0 pin pulled high):
python main_live.py --gps1 /dev/ttyACM0 --imu-mode i2c --imu-address 0x69
```

---

## Receiver setup

### u-blox (recommended)
Default NMEA output works out of the box at 9600 or 115200 baud.
For better C/N0 resolution, enable UBX-NAV-SAT in u-center:
`View → Messages → UBX → CFG → MSG → enable NAV-SAT at 1 Hz`

### NovAtel OEM7
```
LOG GPGGA ONTIME 1
LOG GPGSV ONTIME 1
LOG GPRMC ONTIME 1
LOG GPVTG ONTIME 1
SERIALCONFIG COM1 115200
```

### Any other NMEA receiver
Set output rate to 1 Hz, enable GGA + GSV + RMC sentences.
Baud 115200 preferred (9600 also works).

---

## MPU-6050 wiring (Raspberry Pi)

```
MPU-6050 Pin → Raspberry Pi Pin
VCC          → Pin 1  (3.3V)
GND          → Pin 6  (GND)
SDA          → Pin 3  (GPIO2, I2C SDA)
SCL          → Pin 5  (GPIO3, I2C SCL)
AD0          → GND    (address 0x68)  or  3.3V (address 0x69)
```

Verify connection:
```bash
i2cdetect -y 1
# Should show "68" in the grid
```

---

## What each mode detects

| Mode | Jamming | Spoofing (signal) | Spoofing (INS) | Notes |
|------|---------|-------------------|----------------|-------|
| GPS only, no IMU (`--no-imu`) | ✓ Full | ✓ Partial | ✗ | No position/velocity cross-check |
| GPS + mock IMU | ✓ Full | ✓ Partial | ~ Approximate | Approx dead-reckoning from GPS vel |
| GPS + real IMU | ✓ Full | ✓ Full | ✓ Full | Best — all 5 algorithms active |
| Dual GPS + IMU | ✓ Full | ✓ Full | ✓ Full | Adds multi-receiver cross-check |

---

## AGC limitation

NMEA 0183 does not carry AGC (Automatic Gain Control) values.
A constant fallback of 42.0 dB is used for the AGC field, which means
the AGC-based jamming sub-detector is inactive.

To get real AGC:
- **u-blox:** Parse UBX-MON-RF (binary protocol, use pyubx2 library)
- **NovAtel:** Parse `RXSTATUS` or `HWMONITOR` logs
- **Septentrio:** Use SBF binary protocol

The C/N0 collapse and satellite dropout detectors work fully with NMEA.

---

## Troubleshooting

**No data / timeout:**
- Wrong port — re-check with `ls /dev/tty*`
- Wrong baud — try 9600 with `--baud1 9600`
- Receiver not outputting NMEA — check receiver config

**I2C error / IMU not found:**
- Run `i2cdetect -y 1` — should show 68 or 69
- Check wiring and 3.3V supply
- Try `--imu-address 0x69` if AD0 is pulled high

**Zero C/N0 values:**
- GPS antenna needs open sky view — move outdoors or to window
- Receiver needs 30–60 seconds to acquire satellites cold-start

**All confidence scores stay at 0:**
- Normal for first 15–30 seconds while baseline window fills
- Detection requires ~30 seconds of clean signal before it can flag anomalies
