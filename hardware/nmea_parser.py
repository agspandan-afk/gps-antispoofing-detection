"""
GPS Anti-Spoofing — Real Hardware Layer
========================================
hardware/nmea_parser.py

Parses raw NMEA 0183 sentences from any GPS receiver into structured dicts.

Supported sentences:
  GGA  — position, altitude, fix quality, satellite count
  RMC  — position, speed, course (redundant but useful for velocity)
  GSV  — satellites in view: PRN, elevation, azimuth, SNR (≈ C/N0)
  VTG  — course and speed over ground
  GNS  — multi-constellation fix data
  GSA  — active satellites and DOP

Works with any NMEA 0183 receiver: u-blox, SiRF, MTK, Trimble, NovAtel
(NovAtel also supports proprietary OEM7 binary — use gps_reader.py directly
for that).

Talker IDs handled:
  $GP — GPS only
  $GN — multi-constellation (GPS + GLONASS + Galileo etc.)
  $GL — GLONASS only
  $GA — Galileo only
  $GB / $BD — BeiDou
"""

import re
from typing import Optional, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Checksum
# ─────────────────────────────────────────────────────────────────────────────

def verify_checksum(sentence: str) -> bool:
    """Return True if the NMEA checksum is valid."""
    try:
        if "*" not in sentence:
            return False
        body, chk = sentence.strip().lstrip("$").rsplit("*", 1)
        computed = 0
        for c in body:
            computed ^= ord(c)
        return computed == int(chk[:2], 16)
    except Exception:
        return False


def parse_sentence(raw: str) -> Optional[Dict]:
    """
    Parse one NMEA sentence and return a dict, or None if unrecognised / bad checksum.

    Keys depend on sentence type — see individual parsers below.
    All parsers include 'type' and 'talker' keys.
    """
    raw = raw.strip()
    if not raw.startswith("$"):
        return None
    if not verify_checksum(raw):
        return None

    body = raw.lstrip("$").split("*")[0]
    fields = body.split(",")
    if not fields:
        return None

    tag = fields[0]           # e.g. "GPGGA", "GNGSA"
    talker = tag[:2]          # "GP", "GN", "GL", "GA", "GB"
    sentence_type = tag[2:]   # "GGA", "RMC", "GSV", etc.

    parsers = {
        "GGA": _parse_gga,
        "RMC": _parse_rmc,
        "GSV": _parse_gsv,
        "VTG": _parse_vtg,
        "GSA": _parse_gsa,
        "GNS": _parse_gns,
    }

    parser = parsers.get(sentence_type)
    if parser is None:
        return None

    result = parser(fields[1:])
    if result is None:
        return None

    result["type"]   = sentence_type
    result["talker"] = talker
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Individual sentence parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gga(f: List[str]) -> Optional[Dict]:
    """
    GGA — Global Positioning System Fix Data
    $GPGGA,hhmmss.ss,llll.ll,a,yyyyy.yy,a,x,xx,x.x,x.x,M,x.x,M,x.x,xxxx*hh

    fix_quality:
      0 = invalid, 1 = GPS fix, 2 = DGPS, 4 = RTK fixed, 5 = RTK float
    """
    try:
        return {
            "utc_time":    _str(f, 0),
            "latitude":    _lat(f, 1, 2),
            "longitude":   _lon(f, 3, 4),
            "fix_quality": _int(f, 5),
            "num_svs":     _int(f, 6),
            "hdop":        _float(f, 7),
            "altitude_m":  _float(f, 8),
            "geoid_sep_m": _float(f, 10),
        }
    except Exception:
        return None


def _parse_rmc(f: List[str]) -> Optional[Dict]:
    """
    RMC — Recommended Minimum Specific GPS/Transit Data
    Provides speed over ground and true course.
    """
    try:
        return {
            "utc_time":    _str(f, 0),
            "status":      _str(f, 1),   # A=active, V=void
            "latitude":    _lat(f, 2, 3),
            "longitude":   _lon(f, 4, 5),
            "speed_knots": _float(f, 6),
            "speed_ms":    (_float(f, 6) or 0.0) * 0.51444,  # knots → m/s
            "course_deg":  _float(f, 7),
            "date":        _str(f, 8),
        }
    except Exception:
        return None


def _parse_gsv(f: List[str]) -> Optional[Dict]:
    """
    GSV — Satellites in View
    Returns one dict per message (may need multiple messages for full constellation).

    Each satellite entry: {prn, elevation_deg, azimuth_deg, snr_db}
    snr_db ≈ C/N0 for most receivers (some report S/N ratio instead).

    Note: SNR in NMEA GSV is typically 0–99 dB. On u-blox M8/M9 it is
    reported in dB-Hz (carrier-to-noise density), matching the C/N0 field
    in our GPSSatellite dataclass. On older receivers it may be dBHz - 30.
    """
    try:
        total_msgs  = _int(f, 0)
        msg_num     = _int(f, 1)
        total_svs   = _int(f, 2)
        satellites  = []

        i = 3
        while i + 3 < len(f):
            prn = _int(f, i)
            el  = _float(f, i + 1)
            az  = _float(f, i + 2)
            snr = _float(f, i + 3)
            if prn is not None:
                satellites.append({
                    "prn":           prn,
                    "elevation_deg": el or 0.0,
                    "azimuth_deg":   az or 0.0,
                    "snr_db":        snr or 0.0,
                })
            i += 4

        return {
            "total_msgs":  total_msgs,
            "msg_num":     msg_num,
            "total_svs":   total_svs,
            "satellites":  satellites,
        }
    except Exception:
        return None


def _parse_vtg(f: List[str]) -> Optional[Dict]:
    """VTG — Course and Speed Over Ground"""
    try:
        return {
            "course_true_deg":  _float(f, 0),
            "course_mag_deg":   _float(f, 2),
            "speed_knots":      _float(f, 4),
            "speed_kmh":        _float(f, 6),
            "speed_ms":         (_float(f, 6) or 0.0) / 3.6,
        }
    except Exception:
        return None


def _parse_gsa(f: List[str]) -> Optional[Dict]:
    """GSA — GNSS DOP and Active Satellites"""
    try:
        active_prns = [_int(f, i) for i in range(2, 14) if _int(f, i)]
        return {
            "mode":        _str(f, 0),   # M=manual, A=auto
            "fix_type":    _int(f, 1),   # 1=none, 2=2D, 3=3D
            "active_prns": active_prns,
            "pdop":        _float(f, 14),
            "hdop":        _float(f, 15),
            "vdop":        _float(f, 16),
        }
    except Exception:
        return None


def _parse_gns(f: List[str]) -> Optional[Dict]:
    """GNS — GNSS Fix Data (multi-constellation)"""
    try:
        return {
            "utc_time":   _str(f, 0),
            "latitude":   _lat(f, 1, 2),
            "longitude":  _lon(f, 3, 4),
            "mode":       _str(f, 5),
            "num_svs":    _int(f, 6),
            "hdop":       _float(f, 7),
            "altitude_m": _float(f, 8),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper extractors
# ─────────────────────────────────────────────────────────────────────────────

def _str(f, i):
    return f[i].strip() if i < len(f) and f[i].strip() else None

def _int(f, i):
    try:
        return int(f[i]) if i < len(f) and f[i].strip() else None
    except ValueError:
        return None

def _float(f, i):
    try:
        return float(f[i]) if i < len(f) and f[i].strip() else None
    except ValueError:
        return None

def _lat(f, vi, di):
    """Parse NMEA latitude: ddmm.mmmm + N/S"""
    raw = _str(f, vi)
    hem = _str(f, di)
    if not raw or not hem:
        return None
    try:
        deg = int(raw[:2])
        mins = float(raw[2:])
        dec = deg + mins / 60.0
        return -dec if hem == "S" else dec
    except Exception:
        return None

def _lon(f, vi, di):
    """Parse NMEA longitude: dddmm.mmmm + E/W"""
    raw = _str(f, vi)
    hem = _str(f, di)
    if not raw or not hem:
        return None
    try:
        deg = int(raw[:3])
        mins = float(raw[3:])
        dec = deg + mins / 60.0
        return -dec if hem == "W" else dec
    except Exception:
        return None
