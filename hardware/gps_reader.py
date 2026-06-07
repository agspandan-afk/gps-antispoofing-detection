"""
GPS Anti-Spoofing — Real Hardware Layer
========================================
hardware/gps_reader.py

GPSReader
─────────
Reads NMEA 0183 sentences from a physical GPS receiver over a serial port
and assembles them into GPSFrame objects identical to what the simulator
produced — so the DetectionEngine receives the same data structure.

Supported receivers (tested / known working):
  • u-blox M8 / M9 / F9P (USB or UART, 9600–460800 baud)
  • NovAtel OEM7 (NMEA mode — use 115200 baud, enable GPGGA/GPGSV/GPRMC logs)
  • SiRFstar IV / V
  • Quectel LC29H / LC86L
  • Any receiver outputting standard NMEA 0183 v2.3+

Typical serial ports:
  Linux:   /dev/ttyUSB0   (USB-to-serial)
           /dev/ttyACM0   (CDC-ACM, u-blox USB native)
           /dev/ttyAMA0   (Raspberry Pi GPIO UART)
  Windows: COM3, COM4, ...
  macOS:   /dev/cu.usbserial-...

For u-blox receivers, recommended UBX config:
  Enable: GGA at 1 Hz, GSV at 1 Hz, RMC at 1 Hz, VTG at 1 Hz
  Baud: 115200
  Set with u-center or: ubxtool -p CFG-MSG ...

For NovAtel OEM7:
  LOG GPGGA ONTIME 1
  LOG GPGSV ONTIME 1
  LOG GPRMC ONTIME 1
  LOG GPVTG ONTIME 1
  SERIALCONFIG COM1 115200
"""

import time
import threading
import numpy as np
from collections import defaultdict
from typing import Optional, List, Callable

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from hardware.nmea_parser import parse_sentence
from gps_antispoofing.core.data_types import (
    GPSFrame, GPSSatellite, ConstellationID
)


# Talker ID → constellation
_TALKER_TO_CONST = {
    "GP": ConstellationID.GPS,
    "GL": ConstellationID.GLONASS,
    "GA": ConstellationID.GALILEO,
    "GB": ConstellationID.BEIDOU,
    "BD": ConstellationID.BEIDOU,
    "GN": ConstellationID.GPS,   # multi-constellation, tag as GPS (split later)
}


class GPSReader:
    """
    Serial NMEA GPS reader.  Accumulates sentences within each 1-second epoch
    and calls `on_frame(GPSFrame)` whenever a complete frame is ready.

    Usage
    -----
        def handle(frame):
            engine.ingest_gps([frame])

        reader = GPSReader("/dev/ttyUSB0", baud=115200, receiver_id="RX1")
        reader.on_frame = handle
        reader.start()          # non-blocking background thread
        ...
        reader.stop()

    Or synchronous (blocking) mode:
        for frame in reader.iter_frames():
            engine.ingest_gps([frame])
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        receiver_id: str = "RX1",
        timeout_s: float = 2.0,
        agc_fallback_db: float = 42.0,
    ):
        if not SERIAL_AVAILABLE:
            raise ImportError(
                "pyserial not installed. Run: pip install pyserial"
            )
        self.port        = port
        self.baud        = baud
        self.receiver_id = receiver_id
        self.timeout_s   = timeout_s
        self.agc_fallback_db = agc_fallback_db  # AGC not in NMEA; use this placeholder

        self.on_frame: Optional[Callable[[GPSFrame], None]] = None

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Accumulator for the current epoch
        self._gga  = None
        self._rmc  = None
        self._vtg  = None
        self._gsa  = None
        self._sats: dict = {}   # prn → satellite dict from GSV
        self._gsv_complete = False

        self._last_frame_time = 0.0
        self._frame_queue: List[GPSFrame] = []

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def start(self) -> None:
        """Open serial port and start background reading thread."""
        self._ser = serial.Serial(
            self.port, self.baud,
            timeout=self.timeout_s,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"[GPSReader] {self.receiver_id}: opened {self.port} @ {self.baud} baud")

    def stop(self) -> None:
        """Stop reader and close serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._ser and self._ser.is_open:
            self._ser.close()
        print(f"[GPSReader] {self.receiver_id}: closed")

    def iter_frames(self):
        """
        Synchronous generator — yields GPSFrame objects one by one.
        Blocks until a complete frame arrives.
        Use in a simple for-loop instead of start()/on_frame.
        """
        if not self._running:
            self.start()
        while self._running:
            frame = self._get_queued_frame()
            if frame:
                yield frame
            else:
                time.sleep(0.01)

    # ──────────────────────────────────────────────
    # Background read loop
    # ──────────────────────────────────────────────

    def _read_loop(self) -> None:
        while self._running:
            try:
                raw = self._ser.readline().decode("ascii", errors="replace")
                if not raw.startswith("$"):
                    continue
                parsed = parse_sentence(raw)
                if parsed:
                    self._handle_sentence(parsed)
            except serial.SerialException as e:
                print(f"[GPSReader] {self.receiver_id} serial error: {e}")
                time.sleep(1.0)
            except Exception as e:
                print(f"[GPSReader] {self.receiver_id} parse error: {e}")

    def _handle_sentence(self, s: dict) -> None:
        stype = s.get("type")

        if stype in ("GGA", "GNS"):
            # GGA/GNS arriving = new epoch boundary for most receivers
            if self._gga is not None:
                # We had a previous epoch — emit it first
                frame = self._build_frame()
                if frame:
                    self._emit(frame)
                self._reset_epoch()
            self._gga = s

        elif stype == "RMC":
            self._rmc = s

        elif stype == "VTG":
            self._vtg = s

        elif stype == "GSA":
            self._gsa = s

        elif stype == "GSV":
            talker = s.get("talker", "GP")
            for sat in s.get("satellites", []):
                prn = sat["prn"]
                sat["talker"] = talker
                self._sats[prn] = sat
            # Detect last GSV message in sequence
            if s.get("msg_num") == s.get("total_msgs"):
                self._gsv_complete = True

    # ──────────────────────────────────────────────
    # Frame assembly
    # ──────────────────────────────────────────────

    def _build_frame(self) -> Optional[GPSFrame]:
        if self._gga is None:
            return None

        lat = self._gga.get("latitude")
        lon = self._gga.get("longitude")
        alt = self._gga.get("altitude_m")

        if lat is None or lon is None:
            return None

        alt = alt or 0.0
        lla = np.array([lat, lon, alt])

        # Velocity from RMC/VTG
        vel_ned = self._compute_velocity_ned()

        # Build satellite list
        satellites = self._build_satellite_list()

        fix_quality = self._gga.get("fix_quality", 0)
        num_svs     = self._gga.get("num_svs") or len(satellites)
        hdop        = self._gga.get("hdop") or 1.5
        pdop        = self._gsa.get("pdop") if self._gsa else None
        vdop        = self._gsa.get("vdop") if self._gsa else None

        return GPSFrame(
            timestamp     = time.time(),
            receiver_id   = self.receiver_id,
            position_ecef = self._lla_to_ecef(lla),
            velocity_ecef = vel_ned,
            position_lla  = lla,
            satellites    = satellites,
            hdop          = hdop,
            vdop          = vdop or (hdop * 1.4),
            pdop          = pdop or (hdop * 1.8),
            agc_gain_db   = self.agc_fallback_db,  # not available in NMEA
            num_svs       = num_svs,
            fix_valid     = fix_quality >= 1,
        )

    def _compute_velocity_ned(self) -> np.ndarray:
        """Derive NED velocity from RMC speed + course or VTG."""
        speed_ms = 0.0
        course_deg = 0.0

        if self._vtg:
            speed_ms   = self._vtg.get("speed_ms") or 0.0
            course_deg = self._vtg.get("course_true_deg") or 0.0
        elif self._rmc:
            speed_ms   = self._rmc.get("speed_ms") or 0.0
            course_deg = self._rmc.get("course_deg") or 0.0

        course_rad = np.deg2rad(course_deg)
        vN = speed_ms * np.cos(course_rad)
        vE = speed_ms * np.sin(course_rad)
        return np.array([vN, vE, 0.0])

    def _build_satellite_list(self) -> List[GPSSatellite]:
        sats = []
        for prn, s in self._sats.items():
            talker   = s.get("talker", "GP")
            const    = _TALKER_TO_CONST.get(talker, ConstellationID.GPS)
            snr      = s.get("snr_db") or 0.0
            cn0      = snr  # NMEA SNR ≈ C/N0 on u-blox; may need offset on other receivers
            el       = s.get("elevation_deg") or 0.0
            az       = s.get("azimuth_deg") or 0.0

            # Doppler: NMEA does not provide Doppler directly.
            # We set measured = predicted = 0 here.
            # If your receiver outputs UBX-NAV-SAT or proprietary log with Doppler,
            # parse it separately and inject it via inject_doppler().
            sats.append(GPSSatellite(
                prn                   = prn,
                constellation         = const,
                snr                   = snr,
                cn0                   = cn0,
                doppler_hz            = 0.0,
                predicted_doppler_hz  = 0.0,
                azimuth_deg           = az,
                elevation_deg         = el,
                pseudorange_m         = 0.0,   # not in standard NMEA
                carrier_phase_cycles  = 0.0,
                lock_time_ms          = 0.0,
                healthy               = (snr > 10.0),
            ))
        return sats

    def _reset_epoch(self) -> None:
        self._gga = None
        self._rmc = None
        self._vtg = None
        self._gsa = None
        self._sats = {}
        self._gsv_complete = False

    def _emit(self, frame: GPSFrame) -> None:
        if self.on_frame:
            self.on_frame(frame)
        with self._lock:
            self._frame_queue.append(frame)

    def _get_queued_frame(self) -> Optional[GPSFrame]:
        with self._lock:
            return self._frame_queue.pop(0) if self._frame_queue else None

    @staticmethod
    def _lla_to_ecef(lla: np.ndarray) -> np.ndarray:
        lat = np.deg2rad(lla[0])
        lon = np.deg2rad(lla[1])
        alt = lla[2]
        R  = 6_378_137.0
        e2 = 0.00669437999014
        N  = R / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        x  = (N + alt) * np.cos(lat) * np.cos(lon)
        y  = (N + alt) * np.cos(lat) * np.sin(lon)
        z  = (N * (1 - e2) + alt) * np.sin(lat)
        return np.array([x, y, z])
