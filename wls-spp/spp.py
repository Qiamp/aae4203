"""
Single Point Positioning (SPP) port of RTKLIB's pntpos.c.
Reads RINEX observation/navigation files and outputs CSV/KML solutions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import georinex as gr  # type: ignore
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants and configuration mirrors
# ---------------------------------------------------------------------------

CLIGHT = 299792458.0
OMGE = 7.2921151467e-5
MU_GPS = 3.986005e14
MU_CMP = 3.986004418e14
MU_GAL = 3.986004418e14
MU_QZS = 3.986005e14
MU_IRN = 3.986004418e14
MU_GLO = 3.9860044e14
MU_SBAS = MU_GPS

FREQ_GPS_L1 = 1575.42e6
FREQ_GPS_L2 = 1227.60e6
FREQ_GPS_L5 = 1176.45e6

FREQ_GAL_E1 = 1575.42e6
FREQ_GAL_E5a = 1176.45e6
FREQ_GAL_E5b = 1207.14e6

FREQ_CMP_B1 = 1575.42e6
FREQ_CMP_B1I = 1561.098e6
FREQ_CMP_B2 = 1207.14e6
FREQ_CMP_B2I = 1207.14e6
FREQ_CMP_B3 = 1268.52e6

FREQ_QZS_L1 = 1575.42e6
FREQ_QZS_L2 = 1227.60e6
FREQ_QZS_L5 = 1176.45e6
FREQ_QZS_LEX = 1278.75e6

FREQ_IRN_L5 = 1176.45e6
FREQ_IRN_S = 2492.028e6

FREQ_GLO_G1 = 1602.0e6
FREQ_GLO_G2 = 1246.0e6
FREQ_GLO_G3 = 1202.025e6
DFRQ1_GLO = 0.5625e6
DFRQ2_GLO = 0.4375e6

RE_WGS84 = 6378137.0
FE_WGS84 = 1.0 / 298.257223563

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

ERR_ION = 5.0
ERR_TROP = 3.0
ERR_SAAS = 0.3
ERR_BRDCI = 0.5
ERR_CBIAS = 0.3
REL_HUMI = 0.7
MIN_EL = 5.0 * D2R
MAX_GDOP = 30.0
MAXITR = 10

SQR = lambda x: x * x

GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)

SYS_GPS = 1
SYS_GLO = 2
SYS_GAL = 4
SYS_CMP = 8
SYS_QZS = 16
SYS_SBS = 32
SYS_IRN = 64

PMODE_SINGLE = 0
IONOOPT_OFF = 0
IONOOPT_BRDC = 1
IONOOPT_SBAS = 2
IONOOPT_IFLC = 3
IONOOPT_EST = 4
IONOOPT_TEC = 5
IONOOPT_QZS = 6

TROPOPT_OFF = 0
TROPOPT_SAAS = 1
TROPOPT_SBAS = 2
TROPOPT_EST = 3
TROPOPT_ESTG = 4

EPHOPT_BRDC = 0
EPHOPT_SBAS = 1
EPHOPT_PREC = 2
EPHOPT_SSRC = 3

SOLQ_NONE = 0
SOLQ_SINGLE = 1
SOLQ_SBAS = 2

SVH_OK = 0
SVH_UNHEALTHY = 1

MAXSAT = 256
MAXOBS = 64
NX = 9  # position+clock+inter-system offset state (with QZSS)
NFREQ = 3

CHISQR = np.array([
    0.0, 0.0, 5.9915, 7.8147, 9.4877, 11.0705, 12.5916, 14.0671, 15.5073,
    16.9189, 18.3070, 19.6751, 21.0261, 22.3620, 23.6848, 24.9958, 26.2962,
    27.5871, 28.8693, 30.1435, 31.4104, 32.6706, 33.9245, 35.1725, 36.4150,
    37.6525, 38.8851, 40.1133, 41.3372, 42.5570, 43.7730, 44.9853, 46.1942,
    47.3999, 48.6024, 49.8018, 50.9985, 52.1922, 53.3832, 54.5716, 55.7574,
    56.9406, 58.1215, 59.2999, 60.4760, 61.6498, 62.8212, 63.9904, 65.1574,
    66.3222, 67.4849, 68.6455, 69.8040, 70.9605, 72.1149, 73.2673, 74.4177,
    75.5662, 76.7128, 77.8574, 79.0002, 80.1411, 81.2802, 82.4174, 83.5529,
    84.6867, 85.8187, 86.9490, 88.0777, 89.2047, 90.3300, 91.4538, 92.5759,
    93.6965, 94.8156, 95.9331, 97.0491, 98.1637, 99.2767, 100.3883, 101.4985,
    102.6073, 103.7146, 104.8205, 105.9250, 107.0282, 108.1299, 109.2303,
    110.3294, 111.4271, 112.5235, 113.6186, 114.7124, 115.8049, 116.8961
])

# ---------------------------------------------------------------------------
# Data structures mirroring RTKLIB
# ---------------------------------------------------------------------------

@dataclass
class SNRMask:
    ena: Tuple[int, int] = (0, 0)
    mask: np.ndarray = field(
        default_factory=lambda: np.zeros((NFREQ, 11), dtype=float)
    )

@dataclass
class ProcOptions:
    mode: int = PMODE_SINGLE
    ionoopt: int = IONOOPT_BRDC
    tropopt: int = TROPOPT_SAAS
    sateph: int = EPHOPT_BRDC
    posopt: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    elmin: float = MIN_EL
    err: np.ndarray = field(default_factory=lambda: np.array(
        [100.0, 0.003, 0.003, 0.0, 0.003, 30.0, 0.3, 0.0], dtype=float))
    eratio: np.ndarray = field(default_factory=lambda: np.array([300.0, 300.0, 300.0]))
    snrmask: SNRMask = field(default_factory=SNRMask)
    nf: int = 2

@dataclass
class Observation:
    time: dt.datetime
    sat: int
    code: List[str]
    P: np.ndarray
    L: np.ndarray
    D: np.ndarray
    SNR: np.ndarray
    Pstd: np.ndarray
    eventime: float = 0.0

@dataclass
class BroadcastEphemeris:
    sat: int
    toe: float
    toc: float
    ttr: float
    A: float
    e: float
    i0: float
    OMG0: float
    omg: float
    M0: float
    deltan: float
    OMGd: float
    idot: float
    Cuc: float
    Cus: float
    Crc: float
    Crs: float
    Cic: float
    Cis: float
    af0: float
    af1: float
    af2: float
    tgd: Tuple[float, float, float, float]
    iode: int
    iodc: int
    week: int
    svh: int
    ura: float
    fit: int
    prn: int
    sys: int

@dataclass
class NavigationData:
    eph: Dict[int, List[BroadcastEphemeris]] = field(default_factory=dict)
    ion_gps: np.ndarray = field(default_factory=lambda: np.zeros(8))
    ion_qzs: np.ndarray = field(default_factory=lambda: np.zeros(8))
    cbias: Dict[int, np.ndarray] = field(default_factory=dict)
    leaps: int = 18

@dataclass
class Solution:
    time: dt.datetime = GPS_EPOCH
    rr: np.ndarray = field(default_factory=lambda: np.zeros(6))
    qr: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    qv: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    dtr: np.ndarray = field(default_factory=lambda: np.zeros(6))
    type: int = 0
    stat: int = SOLQ_NONE
    ns: int = 0
    age: float = 0.0
    ratio: float = 0.0
    eventime: float = 0.0

@dataclass
class SatelliteStatus:
    azel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    resp: np.ndarray = field(default_factory=lambda: np.zeros(1))
    resc: np.ndarray = field(default_factory=lambda: np.zeros(1))
    snr_rover: np.ndarray = field(default_factory=lambda: np.zeros(NFREQ))
    snr_base: np.ndarray = field(default_factory=lambda: np.zeros(NFREQ))
    vs: int = 0

# ---------------------------------------------------------------------------
# Satellite numbering helpers
# ---------------------------------------------------------------------------

SAT_OFFSETS = {
    SYS_GPS: 0,
    SYS_SBS: 32,
    SYS_GLO: 32 + 32,
    SYS_GAL: 32 + 32 + 27,
    SYS_CMP: 32 + 32 + 27 + 36,
    SYS_QZS: 32 + 32 + 27 + 36 + 63,
    SYS_IRN: 32 + 32 + 27 + 36 + 63 + 7,
}

SYS_CHAR = {
    'G': SYS_GPS,
    'R': SYS_GLO,
    'E': SYS_GAL,
    'C': SYS_CMP,
    'J': SYS_QZS,
    'S': SYS_SBS,
    'I': SYS_IRN,
}

def satno(sys: int, prn: int) -> int:
    return SAT_OFFSETS.get(sys, 0) + prn

def satsys(sat: int) -> Tuple[int, int]:
    for sys, off in sorted(SAT_OFFSETS.items(), key=lambda x: x[1], reverse=True):
        if sat > off:
            return sys, sat - off
    return SYS_GPS, sat

# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def mat(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=float)

def zeros(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=float)

def dot(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))

def dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(v))

def lsq(H: np.ndarray, v: np.ndarray, nx: int, nv: int) -> Tuple[np.ndarray, float, np.ndarray]:
    sol, residuals, rank, s = np.linalg.lstsq(H[:nv, :nx], v[:nv], rcond=None)
    Q = np.linalg.inv(H[:nv, :nx].T @ H[:nv, :nx])
    return sol, residuals[0] if residuals.size else 0.0, Q

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def datetime_to_gps_seconds(time: dt.datetime) -> float:
    if time.tzinfo is None:
        time = time.replace(tzinfo=dt.timezone.utc)
    return (time - GPS_EPOCH).total_seconds()

def gps_seconds_to_datetime(seconds: float) -> dt.datetime:
    return GPS_EPOCH + dt.timedelta(seconds=seconds)

def timeadd(time: dt.datetime, sec: float) -> dt.datetime:
    return time + dt.timedelta(seconds=sec)

def timediff(t1: dt.datetime, t2: dt.datetime) -> float:
    return (t1 - t2).total_seconds()

# ---------------------------------------------------------------------------
# Geodetic conversions and geometry
# ---------------------------------------------------------------------------

def ecef2pos(r: Sequence[float]) -> np.ndarray:
    x, y, z = r
    e2 = FE_WGS84 * (2.0 - FE_WGS84)
    r2 = x * x + y * y
    v = RE_WGS84
    lat = math.atan2(z, math.sqrt(r2))
    lon = math.atan2(y, x)
    h = 0.0
    for _ in range(5):
        sinp = math.sin(lat)
        v = RE_WGS84 / math.sqrt(1.0 - e2 * sinp * sinp)
        h = r2 ** 0.5 / math.cos(lat) - v
        lat = math.atan2(z, r2 ** 0.5 * (1.0 - e2 * v / (v + h)))
    return np.array([lat, lon, h], dtype=float)

def pos2ecef(pos: Sequence[float]) -> np.ndarray:
    lat, lon, h = pos
    sinp = math.sin(lat)
    cosp = math.cos(lat)
    sinl = math.sin(lon)
    cosl = math.cos(lon)
    e2 = FE_WGS84 * (2.0 - FE_WGS84)
    v = RE_WGS84 / math.sqrt(1.0 - e2 * sinp * sinp)
    x = (v + h) * cosp * cosl
    y = (v + h) * cosp * sinl
    z = (v * (1.0 - e2) + h) * sinp
    return np.array([x, y, z], dtype=float)

def xyz2enu(pos: Sequence[float]) -> np.ndarray:
    lat, lon = pos[0], pos[1]
    sinp, cosp = math.sin(lat), math.cos(lat)
    sinl, cosl = math.sin(lon), math.cos(lon)
    return np.array([
        [-sinl, cosl, 0.0],
        [-sinp * cosl, -sinp * sinl, cosp],
        [cosp * cosl, cosp * sinl, sinp]
    ], dtype=float)

def geodist(rs: Sequence[float], rr: Sequence[float]) -> Tuple[float, np.ndarray]:
    e = np.array(rs[:3], dtype=float) - np.array(rr[:3], dtype=float)
    r = norm(e)
    if r <= 0.0:
        return 0.0, np.zeros(3, dtype=float)
    return r, e / r

def satazel(pos: Sequence[float], e: Sequence[float]) -> float:
    E = xyz2enu(pos)
    enu = E @ e
    az = math.atan2(enu[0], enu[1])
    if az < 0.0:
        az += 2.0 * math.pi
    el = math.asin(enu[2])
    return el

def dops(n: int, azel: np.ndarray, elmin: float, dop: np.ndarray) -> None:
    H = []
    for i in range(n):
        el = azel[1 + 2 * i]
        if el < elmin:
            continue
        az = azel[2 * i]
        s = math.sin(el)
        c = math.cos(el)
        H.append([-c * math.sin(az), -c * math.cos(az), -s, 1.0])
    if len(H) < 4:
        dop[:] = 0.0
        return
    H = np.array(H)
    Q = np.linalg.inv(H.T @ H)
    dop[0] = math.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2] + Q[3, 3])
    dop[1] = math.sqrt(Q[0, 0] + Q[1, 1])
    dop[2] = math.sqrt(Q[2, 2])
    dop[3] = math.sqrt(Q[3, 3])

# ---------------------------------------------------------------------------
# Atmospheric models
# ---------------------------------------------------------------------------

def ionmodel(time: dt.datetime, ion: np.ndarray, pos: Sequence[float], azel: Sequence[float]) -> float:
    if ion.size < 8:
        return 0.0
    az, el = azel
    psi = 0.0137 / (el / math.pi + 0.11) - 0.022
    phi = pos[0] / math.pi + psi * math.cos(az)
    phi = min(max(phi, -0.416), 0.416)
    lam = pos[1] / math.pi + psi * math.sin(az) / math.cos(phi * math.pi)
    t = 43200.0 * lam + datetime_to_gps_seconds(time) % 86400.0
    t %= 86400.0
    s_arg = 1.57 - 1.634 * el / math.pi
    a = ion[0] + ion[1] * phi + ion[2] * phi**2 + ion[3] * phi**3
    a = max(a, 0.0)
    b = ion[4] + ion[5] * phi + ion[6] * phi**2 + ion[7] * phi**3
    b = max(b, 72000.0)
    x = 2.0 * math.pi * (t - 50400.0) / b
    if abs(x) < 1.57:
        ionodelay = (5e-9 + a * (1.0 - x**2 / 2.0 + x**4 / 24.0)) * (1.0 + 0.1 * (0.53 - s_arg)**2)
    else:
        ionodelay = 5e-9 * (1.0 + 0.1 * (0.53 - s_arg)**2)
    return CLIGHT * ionodelay

def tropmodel(time: dt.datetime, pos: Sequence[float], azel: Sequence[float], rel_humi: float) -> float:
    lat, _, h = pos
    el = azel[1]
    P0 = 1013.25
    T0 = 15.0 + 273.15
    e0 = 6.108 * rel_humi * math.exp((17.15 * (T0 - 273.15) - 38.25) / (T0 - 273.15 + 273.15))
    z = math.pi / 2.0 - el
    trop = 0.0022768 * P0 / (1.0 - 0.00266 * math.cos(2.0 * lat) - 0.00028 * h / 1000.0) / math.cos(z)
    wet = 0.002277 * (1255.0 / T0 + 0.05) * e0 / math.cos(z)
    return trop + wet

def sbsioncorr(*args, **kwargs) -> bool:
    return False

def iontec(*args, **kwargs) -> bool:
    return False

def sbstropcorr(*args, **kwargs) -> float:
    return 0.0

# ---------------------------------------------------------------------------
# Auxiliary helpers
# ---------------------------------------------------------------------------

def testsnr(mode: int, freq: int, el: float, snr: float, mask: SNRMask) -> bool:
    if mask.ena[mode] == 0:
        return False
    idx = min(int(max((el * R2D) // 5, 0)), 10)
    threshold = mask.mask[freq, idx]
    if threshold <= 0.0:
        return False
    return snr < threshold

def sat2freq(sat: int, code: str, nav: NavigationData) -> float:
    sys, prn = satsys(sat)
    if sys == SYS_GPS or sys == SYS_SBS:
        if code.startswith(("C1", "P1", "L1", "S1", "D1")):
            return FREQ_GPS_L1
        if code.startswith(("C2", "P2", "L2", "S2", "D2")):
            return FREQ_GPS_L2
        if code.startswith(("C5", "L5", "D5", "S5")):
            return FREQ_GPS_L5
    if sys == SYS_GAL:
        if code.startswith(("C1", "L1", "D1", "S1")):
            return FREQ_GAL_E1
        if code.startswith(("C5", "L5", "D5", "S5")):
            return FREQ_GAL_E5a
        if code.startswith(("C7", "L7", "D7", "S7")):
            return FREQ_GAL_E5b
    if sys == SYS_CMP:
        if code.startswith(("C2", "L2", "D2", "S2")):
            return FREQ_CMP_B1I
        if code.startswith(("C7", "L7", "D7", "S7")):
            return FREQ_CMP_B2I
        if code.startswith(("C6", "L6", "D6", "S6")):
            return FREQ_CMP_B3
    if sys == SYS_QZS:
        if code.startswith(("C1", "L1", "D1", "S1")):
            return FREQ_QZS_L1
        if code.startswith(("C2", "L2", "D2", "S2")):
            return FREQ_QZS_L2
        if code.startswith(("C5", "L5", "D5", "S5")):
            return FREQ_QZS_L5
        if code.startswith(("C6", "L6", "D6", "S6")):
            return FREQ_QZS_LEX
    if sys == SYS_IRN:
        if code.startswith(("C5", "L5", "D5", "S5")):
            return FREQ_IRN_L5
        if code.startswith(("C9", "L9", "D9", "S9")):
            return FREQ_IRN_S
    if sys == SYS_GLO:
        freq_slot = prn - 1
        if code.startswith(("C1", "L1", "D1", "S1")):
            return FREQ_GLO_G1 + freq_slot * DFRQ1_GLO
        if code.startswith(("C2", "L2", "D2", "S2")):
            return FREQ_GLO_G2 + freq_slot * DFRQ2_GLO
        if code.startswith(("C3", "L3", "D3", "S3")):
            return FREQ_GLO_G3
    raise ValueError(f"Unsupported code {code} for sat {sat}")

def seliflc(nf: int, sys: int) -> int:
    if nf < 2:
        return 0
    return 1

def code2bias_ix(sys: int, code: str) -> int:
    return 0

def getseleph(sys: int) -> bool:
    return sys == SYS_GAL

def satexclude(sat: int, vare: float, svh: int, opt: ProcOptions) -> bool:
    return svh & SVH_UNHEALTHY

# ---------------------------------------------------------------------------
# Range and variance helpers
# ---------------------------------------------------------------------------

def varerr(opt: ProcOptions, obs: Observation, el: float, sys: int) -> float:
    fact = {
        SYS_GPS: 1.0,
        SYS_GLO: 1.5,
        SYS_SBS: 3.0,
        SYS_CMP: 1.1,
        SYS_QZS: 1.2,
        SYS_IRN: 1.0
    }.get(sys, 1.0)
    el = max(el, MIN_EL)
    varr = SQR(opt.err[1]) + SQR(opt.err[2]) / math.sin(el)
    if opt.err[6] > 0.0:
        snr = obs.SNR[0] if obs.SNR.size else 0.0
        varr += SQR(opt.err[6]) * 10 ** (0.1 * max(opt.err[5] - snr, 0.0))
    varr *= SQR(opt.eratio[0])
    if opt.err[7] > 0.0:
        pstd = obs.Pstd[0] if obs.Pstd.size else 0.0
        varr += SQR(opt.err[7] * pstd)
    if opt.ionoopt == IONOOPT_IFLC:
        varr *= SQR(3.0)
    return SQR(fact) * varr

def gettgd(sat: int, nav: NavigationData, type_idx: int) -> float:
    if sat not in nav.cbias:
        return 0.0
    arr = nav.cbias[sat]
    if type_idx < arr.size:
        return arr[type_idx]
    return 0.0

# ---------------------------------------------------------------------------
# Measurement models
# ---------------------------------------------------------------------------

def snrmask(obs: Observation, azel: Sequence[float], opt: ProcOptions) -> bool:
    if testsnr(0, 0, azel[1], obs.SNR[0] if obs.SNR.size else 0.0, opt.snrmask):
        return False
    if opt.ionoopt == IONOOPT_IFLC and opt.nf > 1:
        f2 = seliflc(opt.nf, satsys(obs.sat)[0])
        if obs.SNR.size > f2 and testsnr(0, f2, azel[1], obs.SNR[f2], opt.snrmask):
            return False
    return True

def prange(obs: Observation, nav: NavigationData, opt: ProcOptions) -> Tuple[float, float]:
    P1 = obs.P[0] if obs.P.size else 0.0
    var = 0.0
    if P1 == 0.0:
        return 0.0, var
    sys, _ = satsys(obs.sat)
    if opt.ionoopt == IONOOPT_IFLC and opt.nf > 1:
        f2 = seliflc(opt.nf, sys)
        if obs.P.size <= f2 or obs.P[f2] == 0.0:
            return 0.0, var
        P2 = obs.P[f2]
        freq1 = sat2freq(obs.sat, obs.code[0], nav)
        freq2 = sat2freq(obs.sat, obs.code[f2], nav)
        gamma = SQR(freq1 / freq2)
        if abs(1.0 - gamma) < 1e-12:
            return 0.0, var
        return ((P2 - gamma * P1) / (1.0 - gamma)), var
    var = SQR(ERR_CBIAS)
    if sys in (SYS_GPS, SYS_QZS):
        return P1 - gettgd(obs.sat, nav, 0), var
    if sys == SYS_GAL:
        return P1 - gettgd(obs.sat, nav, 0), var
    if sys == SYS_CMP:
        return P1 - gettgd(obs.sat, nav, 0), var
    if sys == SYS_IRN:
        freq = sat2freq(obs.sat, obs.code[0], nav)
        return P1 - (freq / FREQ_IRN_L5) * gettgd(obs.sat, nav, 0), var
    if sys == SYS_GLO:
        return P1, var
    return P1, var

def ionocorr(time: dt.datetime, nav: NavigationData, sat: int, pos: Sequence[float],
             azel: Sequence[float], ionoopt: int) -> Tuple[float, float, bool]:
    err = False
    if ionoopt == IONOOPT_SBAS and sbsioncorr(time, nav, pos, azel, None, None):
        return 0.0, 0.0, True
    if ionoopt == IONOOPT_TEC and iontec(time, nav, pos, azel, 1, None, None):
        return 0.0, 0.0, True
    if ionoopt == IONOOPT_QZS and np.linalg.norm(nav.ion_qzs) > 0.0:
        ion = ionmodel(time, nav.ion_qzs, pos, azel)
        return ion, SQR(ion * ERR_BRDCI), True
    if ionoopt in (IONOOPT_BRDC, IONOOPT_SBAS, IONOOPT_TEC, IONOOPT_QZS):
        ion = ionmodel(time, nav.ion_gps, pos, azel)
        return ion, SQR(ion * ERR_BRDCI), True
    return 0.0, (SQR(ERR_ION) if ionoopt == IONOOPT_OFF else 0.0), True

def tropcorr(time: dt.datetime, nav: NavigationData, pos: Sequence[float],
             azel: Sequence[float], tropopt: int) -> Tuple[float, float, bool]:
    if tropopt in (TROPOPT_SAAS, TROPOPT_EST, TROPOPT_ESTG):
        trp = tropmodel(time, pos, azel, REL_HUMI)
        return trp, SQR(ERR_SAAS / (math.sin(azel[1]) + 0.1)), True
    if tropopt == TROPOPT_SBAS:
        trp = sbstropcorr(time, pos, azel, None)
        return trp, SQR(0.12), True
    return 0.0, (SQR(ERR_TROP) if tropopt == TROPOPT_OFF else 0.0), True

# ---------------------------------------------------------------------------
# Satellite position from ephemeris
# ---------------------------------------------------------------------------

def check_t(t: float) -> float:
    half_week = 302400.0
    if t > half_week:
        t -= 604800.0
    elif t < -half_week:
        t += 604800.0
    return t

def eph2pos(time: dt.datetime, eph: BroadcastEphemeris) -> Tuple[np.ndarray, np.ndarray, float]:
    tk = check_t(datetime_to_gps_seconds(time) - eph.toe)
    n0 = math.sqrt(MU_GPS / eph.A**3)
    n = n0 + eph.deltan
    M = eph.M0 + n * tk
    E = M
    for _ in range(10):
        Ek = E
        E = M + eph.e * math.sin(E)
        if abs(E - Ek) < 1e-12:
            break
    sinE, cosE = math.sin(E), math.cos(E)
    v = math.atan2(math.sqrt(1.0 - eph.e**2) * sinE, cosE - eph.e)
    phi = v + eph.omg
    du = eph.Cus * math.sin(2.0 * phi) + eph.Cuc * math.cos(2.0 * phi)
    dr = eph.Crs * math.sin(2.0 * phi) + eph.Crc * math.cos(2.0 * phi)
    di = eph.Cis * math.sin(2.0 * phi) + eph.Cic * math.cos(2.0 * phi)
    u = phi + du
    r = eph.A * (1.0 - eph.e * cosE) + dr
    i = eph.i0 + di + eph.idot * tk
    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)
    OMG = eph.OMG0 + (eph.OMGd - OMGE) * tk - OMGE * eph.toe
    sinOMG, cosOMG = math.sin(OMG), math.cos(OMG)
    sinI, cosI = math.sin(i), math.cos(i)
    pos = np.array([
        x_orb * cosOMG - y_orb * cosI * sinOMG,
        x_orb * sinOMG + y_orb * cosI * cosOMG,
        y_orb * sinI
    ], dtype=float)
    rel = -2.0 * math.sqrt(MU_GPS * eph.A) * eph.e * sinE / CLIGHT
    dt_clk = eph.af0 + eph.af1 * tk + eph.af2 * tk * tk + rel - eph.tgd[0]
    tk_dot = 1e-3
    pos2, _, _ = eph2pos_raw(time + dt.timedelta(seconds=tk_dot), eph)
    vel = (pos2 - pos) / tk_dot
    return pos, vel, dt_clk

def eph2pos_raw(time: dt.datetime, eph: BroadcastEphemeris) -> Tuple[np.ndarray, float, float]:
    tk = check_t(datetime_to_gps_seconds(time) - eph.toe)
    n0 = math.sqrt(MU_GPS / eph.A**3)
    n = n0 + eph.deltan
    M = eph.M0 + n * tk
    E = M
    for _ in range(10):
        Ek = E
        E = M + eph.e * math.sin(E)
        if abs(E - Ek) < 1e-12:
            break
    sinE, cosE = math.sin(E), math.cos(E)
    v = math.atan2(math.sqrt(1.0 - eph.e**2) * sinE, cosE - eph.e)
    phi = v + eph.omg
    du = eph.Cus * math.sin(2.0 * phi) + eph.Cuc * math.cos(2.0 * phi)
    dr = eph.Crs * math.sin(2.0 * phi) + eph.Crc * math.cos(2.0 * phi)
    di = eph.Cis * math.sin(2.0 * phi) + eph.Cic * math.cos(2.0 * phi)
    u = phi + du
    r = eph.A * (1.0 - eph.e * cosE) + dr
    i = eph.i0 + di + eph.idot * tk
    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)
    OMG = eph.OMG0 + (eph.OMGd - OMGE) * tk - OMGE * eph.toe
    sinOMG, cosOMG = math.sin(OMG), math.cos(OMG)
    sinI, cosI = math.sin(i), math.cos(i)
    pos = np.array([
        x_orb * cosOMG - y_orb * cosI * sinOMG,
        x_orb * sinOMG + y_orb * cosI * cosOMG,
        y_orb * sinI
    ], dtype=float)
    rel = -2.0 * math.sqrt(MU_GPS * eph.A) * eph.e * sinE / CLIGHT
    dt_clk = eph.af0 + eph.af1 * tk + eph.af2 * tk * tk + rel - eph.tgd[0]
    return pos, dt_clk, tk

def satposs(time: dt.datetime, obs: List[Observation], nav: NavigationData,
            sateph: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(obs)
    rs = np.zeros((n, 6))
    dts = np.zeros((n, 2))
    var = np.zeros(n)
    svh = np.zeros(n, dtype=int)
    for i, ob in enumerate(obs):
        ephs = nav.eph.get(ob.sat, [])
        if not ephs:
            svh[i] = SVH_UNHEALTHY
            continue
        eph = min(ephs, key=lambda e: abs(check_t(datetime_to_gps_seconds(time) - e.toe)))
        pos, vel, clk = eph2pos(ob.time, eph)
        rs[i, :3] = pos
        rs[i, 3:] = vel
        dts[i, 0] = clk
        dts[i, 1] = eph.af1
        var[i] = 1e-6
        svh[i] = eph.svh
    return rs, dts, var, svh

# ---------------------------------------------------------------------------
# Residuals and estimation
# ---------------------------------------------------------------------------

def rescode(iteration: int, obs: List[Observation], rs: np.ndarray, dts: np.ndarray,
            vare: np.ndarray, svh: np.ndarray, nav: NavigationData, x: np.ndarray,
            opt: ProcOptions, ssat: Dict[int, SatelliteStatus], v: np.ndarray,
            H: np.ndarray, var: np.ndarray, azel: np.ndarray, vsat: np.ndarray,
            resp: np.ndarray) -> Tuple[int, int]:
    rr = x[:3]
    dtr = x[3]
    pos = ecef2pos(rr)
    nv = 0
    ns = 0
    mask = np.zeros(NX - 3, dtype=int)
    for i, ob in enumerate(obs):
        vsat[i] = 0
        azel[2 * i:2 * i + 2] = 0.0
        resp[i] = 0.0
        sat = ob.sat
        sys, _ = satsys(sat)
        if satexclude(sat, vare[i], svh[i], opt):
            continue
        r, e = geodist(rs[i], rr)
        if r <= 0.0:
            continue
        el = satazel(pos, e)
        az = math.atan2(e[0], e[1])
        azel[2 * i] = az
        azel[2 * i + 1] = el
        if el < opt.elmin:
            continue
        if iteration > 0:
            if not snrmask(ob, azel[2 * i:2 * i + 2], opt):
                continue
            dion, vion, ok = ionocorr(ob.time, nav, sat, pos, azel[2 * i:2 * i + 2], opt.ionoopt)
            if not ok:
                continue
            freq = sat2freq(sat, ob.code[0], nav)
            dion *= SQR(FREQ_GPS_L1 / freq)
            vion *= SQR(SQR(FREQ_GPS_L1 / freq))
            dtrp, vtrp, ok = tropcorr(ob.time, nav, pos, azel[2 * i:2 * i + 2], opt.tropopt)
            if not ok:
                continue
        else:
            dion = vion = dtrp = vtrp = 0.0
        P, vmeas = prange(ob, nav, opt)
        if P == 0.0:
            continue
        v[nv] = P - (r + dtr - CLIGHT * dts[i, 0] + dion + dtrp)
        for j in range(NX):
            H[nv, j] = (-e[j] if j < 3 else (1.0 if j == 3 else 0.0))
        if sys == SYS_GLO:
            v[nv] -= x[4]; H[nv, 4] = 1.0; mask[1] = 1
        elif sys == SYS_GAL:
            v[nv] -= x[5]; H[nv, 5] = 1.0; mask[2] = 1
        elif sys == SYS_CMP:
            v[nv] -= x[6]; H[nv, 6] = 1.0; mask[3] = 1
        elif sys == SYS_IRN:
            v[nv] -= x[7]; H[nv, 7] = 1.0; mask[4] = 1
        elif sys == SYS_QZS:
            v[nv] -= x[8]; H[nv, 8] = 1.0; mask[5] = 1
        else:
            mask[0] = 1
        vsat[i] = 1
        resp[i] = v[nv]
        var[nv] = vare[i] + vmeas + vion + vtrp + varerr(opt, ob, el, sys)
        nv += 1
        ns += 1
    for i in range(NX - 3):
        if mask[i]:
            continue
        H[nv, :] = 0.0
        H[nv, i + 3] = 1.0
        v[nv] = 0.0
        var[nv] = 0.01
        nv += 1
    return nv, ns

def valsol(azel: np.ndarray, vsat: np.ndarray, n: int, opt: ProcOptions,
           v: np.ndarray, nv: int, nx: int) -> bool:
    vv = dot(v[:nv], v[:nv])
    if nv > nx and nv - nx - 1 < len(CHISQR):
        if vv > CHISQR[nv - nx - 1]:
            pass
    azels = []
    for i in range(n):
        if vsat[i]:
            azels.extend(azel[2 * i:2 * i + 2])
    if len(azels) < 2:
        return False
    dop = np.zeros(4)
    dops(len(azels) // 2, np.array(azels), opt.elmin, dop)
    if dop[0] <= 0.0 or dop[0] > MAX_GDOP:
        return False
    return True

def estpos(obs: List[Observation], rs: np.ndarray, dts: np.ndarray, vare: np.ndarray,
           svh: np.ndarray, nav: NavigationData, opt: ProcOptions, ssat: Dict[int, SatelliteStatus],
           sol: Solution, azel_out: np.ndarray, vsat: np.ndarray, resp: np.ndarray,
           msg: List[str]) -> int:
    x = np.zeros(NX)
    x[:3] = sol.rr[:3]
    v = np.zeros(len(obs) + NX)
    H = np.zeros((len(obs) + NX, NX))
    var = np.zeros(len(obs) + NX)
    ns_sum = 0
    for itr in range(MAXITR):
        nv, ns = rescode(itr, obs, rs, dts, vare, svh, nav, x, opt, ssat, v, H, var, azel_out, vsat, resp)
        if nv < NX:
            msg.append(f"lack of valid sats ns={nv}")
            break
        for j in range(nv):
            sig = math.sqrt(var[j])
            v[j] /= sig
            H[j, :] /= sig
        dx, _, Q = lsq(H, v, NX, nv)
        x[:NX] += dx
        if norm(dx) < 1e-4:
            sol.type = 0
            sol.time = timeadd(obs[0].time, -x[3] / CLIGHT)
            sol.dtr[:6] = x[3:9] / CLIGHT
            sol.rr[:3] = x[:3]
            sol.rr[3:] = 0.0
            sol.qr[0] = Q[0, 0]
            sol.qr[1] = Q[1, 1]
            sol.qr[2] = Q[2, 2]
            sol.qr[3] = Q[0, 1]
            sol.qr[4] = Q[1, 2]
            sol.qr[5] = Q[0, 2]
            sol.ns = ns
            sol.age = 0.0
            sol.ratio = 0.0
            if valsol(azel_out, vsat, len(obs), opt, v, nv, NX):
                sol.stat = SOLQ_SINGLE if opt.sateph != EPHOPT_SBAS else SOLQ_SBAS
                return 1
            return 0
        ns_sum = ns
    msg.append("iteration divergent")
    return 0

def resdop(obs: List[Observation], rs: np.ndarray, dts: np.ndarray, nav: NavigationData,
           rr: np.ndarray, x: np.ndarray, azel: np.ndarray, vsat: np.ndarray,
           err: float, v: np.ndarray, H: np.ndarray) -> int:
    pos = ecef2pos(rr[:3])
    E = xyz2enu(pos)
    nv = 0
    for i, ob in enumerate(obs):
        if ob.D.size == 0 or not vsat[i]:
            continue
        freq = sat2freq(ob.sat, ob.code[0], nav)
        if freq == 0.0 or norm(rs[i, 3:]) <= 0.0:
            continue
        az = azel[2 * i]
        el = azel[2 * i + 1]
        cosel = math.cos(el)
        a = np.array([math.sin(az) * cosel, math.cos(az) * cosel, math.sin(el)])
        e = E.T @ a
        vs = rs[i, 3:] - x[:3]
        rate = dot3(vs, e) + OMGE / CLIGHT * (
            rs[i, 4] * rr[0] + rs[i, 1] * x[0] - rs[i, 3] * rr[1] - rs[i, 0] * x[1]
        )
        sig = err * CLIGHT / freq if err > 0.0 else 1.0
        v[nv] = (-ob.D[0] * CLIGHT / freq - (rate + x[3] - CLIGHT * dts[i, 1])) / sig
        H[nv, :3] = -e / sig
        H[nv, 3] = 1.0 / sig
        nv += 1
    return nv

def estvel(obs: List[Observation], rs: np.ndarray, dts: np.ndarray, nav: NavigationData,
           opt: ProcOptions, sol: Solution, azel: np.ndarray, vsat: np.ndarray) -> None:
    x = np.zeros(4)
    v = np.zeros(len(obs))
    H = np.zeros((len(obs), 4))
    err = opt.err[4]
    for _ in range(MAXITR):
        nv = resdop(obs, rs, dts, nav, sol.rr, x, azel, vsat, err, v, H)
        if nv < 4:
            break
        dx, _, Q = lsq(H, v, 4, nv)
        x[:4] += dx[:4]
        if norm(dx) < 1e-6:
            sol.rr[3:6] = x[:3]
            sol.qv[0] = Q[0, 0]
            sol.qv[1] = Q[1, 1]
            sol.qv[2] = Q[2, 2]
            sol.qv[3] = Q[0, 1]
            sol.qv[4] = Q[1, 2]
            sol.qv[5] = Q[0, 2]
            break

def raim_fde(*args, **kwargs) -> int:
    return 0  # Full RAIM FDE requires replicated subset; placeholder keeps flow identical.

def pntpos(obs: List[Observation], nav: NavigationData, opt: ProcOptions,
           sol: Solution, azel: np.ndarray, ssat: Dict[int, SatelliteStatus],
           msg: List[str]) -> int:
    if not obs:
        msg.append("no observation data")
        print("No observation data available")
        return 0
    sol.time = obs[0].time
    sol.eventime = obs[0].eventime
    rs, dts, var, svh = satposs(sol.time, obs, nav, opt.sateph)
    vsat = np.zeros(len(obs), dtype=int)
    resp = np.zeros(len(obs))
    stat = estpos(obs, rs, dts, var, svh, nav, opt, ssat, sol, azel, vsat, resp, msg)
    if not stat and len(obs) >= 6 and opt.posopt[4]:
        print(f"Positioning failed: {msg}")
        stat = raim_fde(obs, len(obs), rs, dts, var, svh, nav, opt, ssat, sol, azel, vsat, resp, msg)
    if stat:
        estvel(obs, rs, dts, nav, opt, sol, azel, vsat)
    return stat

# ---------------------------------------------------------------------------
# RINEX parsing
# ---------------------------------------------------------------------------

def parse_rinex_obs(path: pathlib.Path) -> List[Observation]:
    ds = gr.load(path)
    time_values = ds.time.values
    obs_list: List[Observation] = []
    for t_idx, t_val in enumerate(time_values):
        epoch = pd.to_datetime(str(t_val)).to_pydatetime().replace(tzinfo=dt.timezone.utc)
        for sv in ds.sv.values:
            if np.all(np.isnan(ds.sel(sv=sv).to_array().values[:, t_idx])):
                continue
            system_char = sv[0]
            prn = int(sv[1:])
            sys = SYS_CHAR.get(system_char)
            if sys is None:
                continue
            sat = satno(sys, prn)
            codes = []
            P = []
            L = []
            D = []
            SNR = []
            Pstd = []
            for obs_type in ds.data_vars:
                field = ds[obs_type]
                val = field.sel(time=t_val, sv=sv).item()
                if np.isnan(val):
                    continue
                if obs_type.startswith('C'):
                    codes.append(obs_type)
                    P.append(val)
                elif obs_type.startswith('L'):
                    L.append(val)
                elif obs_type.startswith('D'):
                    D.append(val)
                elif obs_type.startswith('S'):
                    SNR.append(val)
            P_arr = np.array(P) if P else np.zeros(0)
            if P_arr.size == 0:
                continue
            while len(codes) < opt_default().nf:
                codes.append(codes[-1])
                P_arr = np.pad(P_arr, (0, 1), 'edge')
            obs_list.append(Observation(
                time=epoch,
                sat=sat,
                code=codes,
                P=P_arr,
                L=np.array(L) if L else np.zeros(0),
                D=np.array(D) if D else np.zeros(0),
                SNR=np.array(SNR) if SNR else np.zeros(opt_default().nf),
                Pstd=np.array(Pstd) if Pstd else np.zeros(opt_default().nf),
                eventime=datetime_to_gps_seconds(epoch)
            ))
    obs_list.sort(key=lambda o: (o.time, o.sat))
    return obs_list

def parse_rinex_nav(path: pathlib.Path) -> NavigationData:
    ds = gr.load(path)
    nav = NavigationData()
    if 'iono' in ds.attrs:
        ion = ds.attrs['iono']
        if 'ALPHA' in ion:
            nav.ion_gps[:4] = np.array(ion['ALPHA'])
        if 'BETA' in ion:
            nav.ion_gps[4:] = np.array(ion['BETA'])
    for sv in ds.sv.values:
        system_char = sv[0]
        prn = int(sv[1:])
        sys = SYS_CHAR.get(system_char)
        if sys is None:
            continue
        sat = satno(sys, prn)
        ephs: List[BroadcastEphemeris] = []
        for t_val in ds.time.values:
            block = ds.sel(time=t_val, sv=sv)
            epoch = pd.to_datetime(str(t_val)).to_pydatetime().replace(tzinfo=dt.timezone.utc)
            sqrtA = float(block['sqrtA'])
            A = sqrtA * sqrtA
            eph = BroadcastEphemeris(
                sat=sat,
                toe=datetime_to_gps_seconds(epoch),
                toc=datetime_to_gps_seconds(epoch),
                ttr=datetime_to_gps_seconds(epoch),
                A=A,
                e=float(block['e']),
                i0=float(block['i0']),
                OMG0=float(block['OMEGA']),
                omg=float(block['omega']),
                M0=float(block['M0']),
                deltan=float(block['DeltaN']),
                OMGd=float(block['OMEGADOT']),
                idot=float(block['IDOT']),
                Cuc=float(block['Cuc']),
                Cus=float(block['Cus']),
                Crc=float(block['Crc']),
                Crs=float(block['Crs']),
                Cic=float(block['Cic']),
                Cis=float(block['Cis']),
                af0=float(block['af0']),
                af1=float(block['af1']),
                af2=float(block['af2']),
                tgd=(float(block.get('TGD', 0.0)), 0.0, 0.0, 0.0),
                iode=int(block.get('IODE', 0)),
                iodc=int(block.get('IODC', 0)),
                week=int(block.get('week', 0)),
                svh=int(block.get('svHealth', 0)),
                ura=float(block.get('svAccuracy', 0.0)),
                fit=int(block.get('fitInterval', 0)),
                prn=prn,
                sys=sys
            )
            ephs.append(eph)
        nav.eph[sat] = ephs
    return nav

def opt_default() -> ProcOptions:
    return ProcOptions()

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(path: pathlib.Path, records: List[Dict[str, float]]) -> None:
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)

def write_kml(path: pathlib.Path, records: List[Dict[str, float]]) -> None:
    header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<Placemark>
<LineString>
<tessellate>1</tessellate>
<coordinates>
"""
    footer = """</coordinates>
</LineString>
</Placemark>
</Document>
</kml>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for rec in records:
            f.write(f"{rec['lon']:.9f},{rec['lat']:.9f},{rec['height']:.3f}\n")
        f.write(footer)

# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def run_spp(obs_path: pathlib.Path, nav_path: pathlib.Path,
            csv_path: pathlib.Path, kml_path: pathlib.Path) -> None:
    print(f"Processing observation file: {obs_path}")
    print(f"Processing navigation file: {nav_path}")
    obs_all = parse_rinex_obs(obs_path)
    print(f"Number of observations parsed: {len(obs_all)}")
    nav = parse_rinex_nav(nav_path)
    print(f"Number of ephemerides parsed: {sum(len(e) for e in nav.eph.values())}")
    opt = opt_default()
    solutions: List[Dict[str, float]] = []
    ssat: Dict[int, SatelliteStatus] = {i: SatelliteStatus() for i in range(MAXSAT)}
    current_time: Optional[dt.datetime] = None
    epoch_obs: List[Observation] = []
    for ob in obs_all:
        if current_time is None:
            current_time = ob.time
        if ob.time != current_time:
            process_epoch(epoch_obs, nav, opt, ssat, solutions)
            epoch_obs = []
            current_time = ob.time
        epoch_obs.append(ob)
    if epoch_obs:
        process_epoch(epoch_obs, nav, opt, ssat, solutions)
    write_csv(csv_path, solutions)
    write_kml(kml_path, solutions)
    
    print(f"Number of solutions generated: {len(solutions)}")
    if not solutions:
        print("No solutions to write")
        return
    write_csv(csv_path, solutions)
    write_kml(kml_path, solutions)

def process_epoch(obs_epoch: List[Observation], nav: NavigationData,
                  opt: ProcOptions, ssat: Dict[int, SatelliteStatus],
                  solutions: List[Dict[str, float]]) -> None:
    print(f"Processing epoch with {len(obs_epoch)} observations")
    sol = Solution()
    azel = np.zeros(2 * len(obs_epoch))
    msg: List[str] = []
    stat = pntpos(obs_epoch, nav, opt, sol, azel, ssat, msg)
    if not stat:
        print(f"Epoch processing failed: {msg}")
        return
    if stat:
        pos = ecef2pos(sol.rr[:3])
        lat = pos[0] * R2D
        lon = pos[1] * R2D
        h = pos[2]
        solutions.append({
            "gps_week": datetime_to_gps_seconds(sol.time) / 604800.0,
            "time": sol.time.isoformat(),
            "lat": lat,
            "lon": lon,
            "height": h,
            "stat": sol.stat,
            "ns": sol.ns
        })

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_BASE_DIR = pathlib.Path("/Users/jay/Documents/Bachelor/aae4203")
DEFAULT_OBS_PATH = DEFAULT_BASE_DIR / "rinex_data" / "20250527_PolyU_X.obs"
DEFAULT_NAV_PATH = DEFAULT_BASE_DIR / "rinex_data" / "20250527_PolyU_X.nav"
DEFAULT_OUT_CSV = DEFAULT_BASE_DIR / "output" / "solution.csv"
DEFAULT_OUT_KML = DEFAULT_BASE_DIR / "output" / "solution.kml"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single Point Positioning (SPP) solver.")
    parser.add_argument("--obs", type=pathlib.Path, default=DEFAULT_OBS_PATH,
                        help="RINEX observation file")
    parser.add_argument("--nav", type=pathlib.Path, default=DEFAULT_NAV_PATH,
                        help="RINEX navigation file")
    parser.add_argument("--out-csv", type=pathlib.Path, default=DEFAULT_OUT_CSV,
                        help="Output CSV path")
    parser.add_argument("--out-kml", type=pathlib.Path, default=DEFAULT_OUT_KML,
                        help="Output KML path")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run_spp(args.obs, args.nav, args.out_csv, args.out_kml)

if __name__ == "__main__":
    main()
