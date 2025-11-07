"""
RINEX 3.0.x GPS-only SPP with WLS
现在: 直接在代码中指定 OBS / NAV 路径, 不再需要命令行参数。
- Ionosphere: IF (C1/C2) if available; else Klobuchar (GPSA/GPSB)
- Troposphere: Saastamoinen + Niell mapping; optional ZWD estimation
- Outputs per-epoch ENU (relative to reference), 2D/3D errors, DOPs, and summary stats

Author: ChatGPT (GPT-5)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# Constants
C = 299792458.0
MU_E = 3.986005e14
OMEGA_E = 7.2921151467e-5
F_REL = -4.442807633e-10
F_L1 = 1575.42e6
F_L2 = 1227.60e6

# ----------------------------
# Data classes
# ----------------------------
@dataclass
class ObsEpoch:
    t_gps: float
    week: int
    sats: List[str]
    types_G: List[str]
    data: Dict[str, List[float]]  # prn -> list aligned with types_G

@dataclass
class Eph:
    prn: str
    toc: float
    week: int
    af0: float
    af1: float
    af2: float
    crs: float
    delta_n: float
    m0: float
    cuc: float
    e: float
    cus: float
    sqrt_a: float
    toe: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    w: float
    omega_dot: float
    idot: float
    health: int
    iode: int
    iodc: int
    ura: float

@dataclass
class NavHeader:
    ion_alpha: Optional[List[float]]
    ion_beta: Optional[List[float]]

# ----------------------------
# Time utilities
# ----------------------------
def ymd_to_jd(y,m,d):
    a=(14-m)//12
    y2=y+4800-a
    m2=m+12*a-3
    jd=d+(153*m2+2)//5+365*y2+y2//4-y2//100+y2//400-32045
    return jd+0.5

def ymdhms_to_gpsweek(y, m, d, hh, mm, ssf) -> Tuple[int, float]:
    jd = ymd_to_jd(y,m,d) + (hh-12)/24 + mm/1440 + ssf/86400
    jd_gps0 = 2444244.5
    sow_total = (jd - jd_gps0) * 86400.0
    week = int(sow_total // 604800)
    sow = sow_total - week*604800
    return week, sow

def time_diff_gps(t1, t0):
    dt = t1 - t0
    if dt > 302400: dt -= 604800
    if dt < -302400: dt += 604800
    return dt

# ----------------------------
# RINEX 3 NAV parser (GPS)
# ----------------------------
def parse_rinex3_nav(path:str) -> Tuple[NavHeader, Dict[str, List[Eph]]]:
    ion_alpha=None; ion_beta=None
    ephs: Dict[str, List[Eph]] = {}
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        lines=f.readlines()
    i=0
    # Header
    while i < len(lines):
        line = lines[i]
        if "IONOSPHERIC CORR" in line and "GPSA" in line:
            vals=[float(line[5:17]),float(line[17:29]),float(line[29:41]),float(line[41:53])]
            ion_alpha=vals
        if "IONOSPHERIC CORR" in line and "GPSB" in line:
            vals=[float(line[5:17]),float(line[17:29]),float(line[29:41]),float(line[41:53])]
            ion_beta=vals
        if "END OF HEADER" in line:
            i+=1
            break
        i+=1
    header = NavHeader(ion_alpha=ion_alpha, ion_beta=ion_beta)

    def to_float(s): 
        return float(s.replace('D','E').replace('d','E')) if s.strip() else 0.0

    # Body (RINEX 3 GPS: 8 lines per record)
    while i < len(lines):
        if not lines[i].strip():
            i+=1; continue
        line0 = lines[i]
        prn = line0[0:3].strip()
        if not prn or prn[0] != 'G':
            # Skip non-GPS
            # but consume 8 lines in case
            i += 8
            continue
        year = int(line0[3:7]); month=int(line0[8:10]); day=int(line0[11:13])
        hour=int(line0[14:16]); minute=int(line0[17:19]); sec=float(line0[20:22])
        af0 = to_float(line0[22:41]); af1 = to_float(line0[41:60]); af2 = to_float(line0[60:79])
        p=[]
        for k in range(7):
            i+=1
            l=lines[i]
            # RINEX3 fields 4 per line after 3-char indent
            p.extend([to_float(l[3+19*j:3+19*(j+1)]) for j in range(4)])
        # Assign
        crs, delta_n, m0 = p[0], p[1], p[2]
        cuc, e, cus, sqrt_a = p[4], p[5], p[6], p[7]
        toe, cic, omega0, cis = p[8], p[9], p[10], p[11]
        i0, crc, w, omega_dot = p[12], p[13], p[14], p[15]
        idot = p[16]
        gps_week = int(round(p[18])) if len(p) > 19 else 0
        ura = p[20] if len(p) > 21 else 0.0
        health = int(round(p[21])) if len(p) > 22 else 0
        iodc = int(round(p[23])) if len(p) > 24 else 0
        iode = int(round(p[3])) if len(p) > 4 else 0

        week, sow = ymdhms_to_gpsweek(year,month,day,hour,minute,sec)
        eph = Eph(prn=prn, toc=sow, week=week, af0=af0, af1=af1, af2=af2,
                  crs=crs, delta_n=delta_n, m0=m0, cuc=cuc, e=e, cus=cus, sqrt_a=sqrt_a,
                  toe=toe, cic=cic, omega0=omega0, cis=cis, i0=i0, crc=crc, w=w,
                  omega_dot=omega_dot, idot=idot, health=health, iode=iode, iodc=iodc, ura=ura)
        ephs.setdefault(prn, []).append(eph)
        i+=1
    # sort by toe desc
    for k in ephs:
        ephs[k].sort(key=lambda e: e.toe, reverse=True)
    return header, ephs

# ----------------------------
# RINEX 3 OBS parser (GPS)
# ----------------------------
def parse_rinex3_obs(path:str) -> Tuple[List[str], List[ObsEpoch]]:
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        lines=f.readlines()
    i=0
    types_G: List[str] = []
    # Header
    while i < len(lines):
        line=lines[i]
        if "SYS / # / OBS TYPES" in line and line[0]=='G':
            n = int(line[3:6])
            types = [line[7+4*k:11+4*k].strip() for k in range(min(n,13))]
            while len(types) < n:
                i+=1
                line=lines[i]
                types += [line[7+4*k:11+4*k].strip() for k in range(min(n-len(types),13))]
            types_G = types
        if "END OF HEADER" in line:
            i+=1
            break
        i+=1
    if not types_G:
        raise ValueError("No GPS observation types found in header (SYS / # / OBS TYPES for G).")
    epochs: List[ObsEpoch] = []
    # Body
    while i < len(lines):
        line = lines[i]
        if not line.strip(): i+=1; continue
        if line[0] != '>':
            i+=1; continue
        year=int(line[2:6]); month=int(line[7:9]); day=int(line[10:12])
        hour=int(line[13:15]); minute=int(line[16:18]); sec=float(line[19:29])
        flag=int(line[31:32]) if len(line)>=32 else 0
        nsat=int(line[32:35])
        week, sow = ymdhms_to_gpsweek(year,month,day,hour,minute,sec)
        sats=[]; data={}
        for k in range(nsat):
            i+=1
            l=lines[i]
            prn=l[0:3].strip()
            sats.append(prn)
            vals=[]
            pos=3
            for t in types_G:
                if pos+16 <= len(l):
                    vstr=l[pos:pos+14].strip()
                    v=float(vstr.replace('D','E')) if vstr else float('nan')
                    vals.append(v)
                else:
                    vals.append(float('nan'))
                pos+=16
            data[prn]=vals
        epochs.append(ObsEpoch(t_gps=sow, week=week, sats=sats, types_G=types_G, data=data))
        i+=1
    return types_G, epochs

# ----------------------------
# Coordinate transforms
# ----------------------------
def llh_to_ecef(lat, lon, h):
    a=6378137.0; f=1/298.257223563; e2=f*(2-f)
    N=a/math.sqrt(1-e2*math.sin(lat)**2)
    x=(N+h)*math.cos(lat)*math.cos(lon)
    y=(N+h)*math.cos(lat)*math.sin(lon)
    z=(N*(1-e2)+h)*math.sin(lat)
    return np.array([x,y,z],dtype=float)

def ecef_to_llh(xyz):
    a=6378137.0; f=1/298.257223563; e2=f*(2-f)
    x,y,z=xyz
    lon=math.atan2(y,x)
    r=math.hypot(x,y)
    lat=math.atan2(z, r*(1-e2))
    for _ in range(5):
        N=a/math.sqrt(1-e2*math.sin(lat)**2)
        h=r/math.cos(lat)-N
        lat=math.atan2(z, r*(1-e2*N/(N+h)))
    N=a/math.sqrt(1-e2*math.sin(lat)**2)
    h=r/math.cos(lat)-N
    return lat,lon,h

def ecef_to_enu_matrix(lat, lon):
    sl=math.sin(lat); cl=math.cos(lat)
    slon=math.sin(lon); clon=math.cos(lon)
    return np.array([
        [-slon,        clon,      0],
        [-sl*clon, -sl*slon,    cl],
        [ cl*clon,  cl*slon,    sl],
    ])

def az_el(rx_ecef, sv_ecef):
    lat,lon,h = ecef_to_llh(rx_ecef)
    E = ecef_to_enu_matrix(lat,lon)
    rho = sv_ecef - rx_ecef
    enu = E @ rho
    e,n,u = enu
    az = math.atan2(e,n) % (2*math.pi)
    el = math.atan2(u, math.hypot(e,n))
    rng = np.linalg.norm(rho)
    return az, el, rng

# ----------------------------
# Dynamics and models
# ----------------------------
def solve_kepler(M,e,tol=1e-12,maxit=30):
    E=M
    for _ in range(maxit):
        f=E-e*math.sin(E)-M
        d=1-e*math.cos(E)
        dE=-f/d
        E+=dE
        if abs(dE)<tol: break
    return E

def kepler_orbit(eph:Eph, t_tx:float) -> Tuple[np.ndarray, float]:
    tk = time_diff_gps(t_tx, eph.toe)
    a = eph.sqrt_a**2
    n0 = math.sqrt(MU_E/a**3)
    n = n0 + eph.delta_n
    M = eph.m0 + n*tk
    E = solve_kepler(M, eph.e)
    sinE=math.sin(E); cosE=math.cos(E)
    v = math.atan2(math.sqrt(1-eph.e**2)*sinE, cosE - eph.e)
    phi = v + eph.w
    sin2=math.sin(2*phi); cos2=math.cos(2*phi)
    du = eph.cus*sin2 + eph.cuc*cos2
    dr = eph.crs*sin2 + eph.crc*cos2
    di = eph.cis*sin2 + eph.cic*cos2
    u = phi + du
    r = a*(1 - eph.e*cosE) + dr
    i = eph.i0 + di + eph.idot*tk
    x_orb = r*math.cos(u)
    y_orb = r*math.sin(u)
    omega = eph.omega0 + (eph.omega_dot - OMEGA_E)*tk - OMEGA_E*eph.toe
    cosO=math.cos(omega); sinO=math.sin(omega)
    cosi=math.cos(i); sini=math.sin(i)
    x = x_orb*cosO - y_orb*cosi*sinO
    y = x_orb*sinO + y_orb*cosi*cosO
    z = y_orb*sini
    r_ecef = np.array([x,y,z],dtype=float)
    dt_rel = F_REL * eph.e * eph.sqrt_a * sinE
    dts = eph.af0 + eph.af1*(t_tx - eph.toc) + eph.af2*(t_tx - eph.toc)**2 + dt_rel
    return r_ecef, dts

def sagnac_correction(r_tx, r_rx):
    return OMEGA_E / C * (r_tx[0]*r_rx[1] - r_tx[1]*r_rx[0])

def iono_free(C1:float, C2:float) -> Optional[float]:
    if not (np.isfinite(C1) and np.isfinite(C2)): return None
    f1=F_L1; f2=F_L2; g=(f1/f2)**2
    return (g*C1 - C2)/(g-1.0)

def klobuchar_delay(nav_header:NavHeader, t_gps:float, lat:float, lon:float, az:float, el:float, freq=F_L1) -> float:
    if not nav_header.ion_alpha or not nav_header.ion_beta or el<=0: return 0.0
    alpha=nav_header.ion_alpha; beta=nav_header.ion_beta
    psi = 0.0137/(el/math.pi + 0.11) - 0.022
    phi_i = lat/math.pi + psi*math.cos(az); phi_i = max(-0.416, min(0.416, phi_i))
    lam_i = lon/math.pi + psi*math.sin(az)/math.cos(phi_i*math.pi)
    phi_m = phi_i + 0.064*math.cos((lam_i-1.617)*math.pi)
    t = (t_gps%86400.0) + 43200.0*lam_i/0.5  # 4.32e4*lam_i
    t = t % 86400.0
    AMP = alpha[0] + alpha[1]*phi_m + alpha[2]*phi_m**2 + alpha[3]*phi_m**3
    PER = beta[0] + beta[1]*phi_m + beta[2]*phi_m**2 + beta[3]*phi_m**3
    AMP=max(0.0,AMP); PER=max(72000.0,PER)
    x = 2*math.pi*(t-50400.0)/PER
    Fm = 1.0 + 16.0*(0.53 - el/math.pi)**3
    if abs(x) < 1.57:
        iono = Fm * (5e-9 + AMP*(1 - x*x/2 + x**4/24.0))
    else:
        iono = Fm * 5e-9
    iono_sec = iono * (F_L1/freq)**2
    return iono_sec * C

def saastamoinen_zenith_dry(lat:float, h:float) -> float:
    # standard atmosphere
    p = 1013.25 * (1 - 2.2557e-5 * h)**5.2568
    return 0.0022768 * p / (1 - 0.00266*math.cos(2*lat) - 0.00028e-3*h)

def saastamoinen_zenith_wet(lat:float, h:float, rh=0.5) -> float:
    T_c = 15.0 - 0.0065*h; T_k=T_c+273.15
    es = 6.108*math.exp((17.15*T_c)/(234.7+T_c))
    e = rh*es
    return 0.002277 * (1255.0/T_k + 0.05) * e

def niell_mapping(lat:float, h:float, el:float) -> Tuple[float,float]:
    sin_el=math.sin(el)
    ah=0.0012769934; bh=0.0029153695; ch=0.062610505
    aw=0.0005; bw=0.001; cw=0.04391
    mfh=(1+ah/(1+bh/(1+ch)))/(sin_el + ah/(sin_el+bh/(sin_el+ch)))
    mfw=(1+aw/(1+bw/(1+cw)))/(sin_el + aw/(sin_el+bw/(sin_el+cw)))
    return mfh, mfw

# ----------------------------
# DOP computation
# ----------------------------
def compute_dops(H_geo: np.ndarray) -> Dict[str, float]:
    # H_geo: Nx4 with columns [ux, uy, uz, 1] (clock)
    Q = np.linalg.inv(H_geo.T @ H_geo)
    GDOP = math.sqrt(np.trace(Q))
    PDOP = math.sqrt(Q[0,0] + Q[1,1] + Q[2,2])
    HDOP = math.sqrt(Q[0,0] + Q[1,1])
    VDOP = math.sqrt(Q[2,2])
    TDOP = math.sqrt(Q[3,3])
    return dict(GDOP=GDOP, PDOP=PDOP, HDOP=HDOP, VDOP=VDOP, TDOP=TDOP)

# ----------------------------
# Core SPP epoch solver
# ----------------------------
def spp_epoch_wls(epoch:ObsEpoch, nav_header:NavHeader, ephs:Dict[str,List[Eph]],
                  x0:np.ndarray, elev_mask_deg=10.0, est_zwd=False) -> Tuple[np.ndarray, float, Dict[str,float], Dict[str,float], np.ndarray]:
    """
    Returns:
      x_est (state [X,Y,Z,dt,(ZWD)]),
      rms [m],
      residuals_by_sat {prn: v},
      dops {GDOP,PDOP,HDOP,VDOP,TDOP},
      H_geo (design for DOP)
    """
    x = x0.copy()
    for it in range(10):
        A=[]; L=[]; W=[]; res_by_sat={}
        H_geo_rows=[]
        lat,lon,h = ecef_to_llh(x[:3])
        for prn in epoch.sats:
            if prn[0] != 'G': continue
            if prn not in ephs: continue
            eph = min(ephs[prn], key=lambda e: abs(time_diff_gps(epoch.t_gps, e.toe)))
            # pick observations
            types = epoch.types_G
            vals = epoch.data.get(prn, [])
            def pick(names):
                for name in names:
                    if name in types:
                        idx = types.index(name)
                        return vals[idx]
                return float('nan')
            C1 = pick(['C1C','C1W','C1P','C1X','C1S','C1L','C1Z'])
            C2 = pick(['C2W','C2P','C2C','C2X','C2S','C2L','C2Z'])
            # Build pseudorange
            P_if = iono_free(C1,C2) if (np.isfinite(C1) and np.isfinite(C2)) else None
            use_IF = P_if is not None
            P = P_if if use_IF else C1
            if not np.isfinite(P): continue
            # iterate transmit time
            # initial tx
            r_sv, dt_sv = kepler_orbit(eph, epoch.t_gps)
            rho = np.linalg.norm(r_sv - x[:3])
            t_tx = epoch.t_gps - P/C
            r_sv, dt_sv = kepler_orbit(eph, t_tx)
            sag = sagnac_correction(r_sv, x[:3])
            rho = np.linalg.norm(r_sv - x[:3])
            # elevation mask
            az, el, _ = az_el(x[:3], r_sv)
            if math.degrees(el) < elev_mask_deg: continue
            # atmos
            mfh, mfw = niell_mapping(lat,h,el)
            trop = saastamoinen_zenith_dry(lat,h)*mfh
            if est_zwd:
                trop += x[4]*mfw
            else:
                trop += saastamoinen_zenith_wet(lat,h)*mfw
            iono = 0.0 if use_IF else klobuchar_delay(nav_header, epoch.t_gps, lat, lon, az, el, F_L1)
            modeled = rho + C*(x[3] - dt_sv) + sag + trop + iono
            v = P - modeled
            u = (x[:3] - r_sv)/rho
            H = np.zeros(5 if est_zwd else 4)
            H[0:3] = u
            H[3] = C
            if est_zwd: H[4] = -mfw
            # weight by elevation
            sigma = 0.5 + 3.0/(max(math.sin(el),1e-3)**2)
            w = 1.0/sigma**2
            A.append(H); L.append(v); W.append(w)
            res_by_sat[prn]=v
            # H for DOP (geometry-only)
            H_geo_rows.append([u[0],u[1],u[2],1.0])
        mreq = 5 if est_zwd else 4
        if len(A) < mreq:  # not enough
            return x, float('nan'), {}, {}, np.zeros((0,4))
        A = np.vstack(A)
        L = np.array(L).reshape(-1,1)
        Wm = np.diag(W)
        N = A.T @ Wm @ A
        U = A.T @ Wm @ L
        try:
            dx = np.linalg.solve(N,U).flatten()
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(N,U,rcond=None)[0].flatten()
        x[:4] += dx[:4]
        if est_zwd and len(dx)==5:
            x[4] += dx[4]
        if np.linalg.norm(dx[:4]) < 1e-4 and (not est_zwd or abs(dx[-1])<1e-3):
            # RMS of residuals
            v = (L - A @ dx.reshape(-1,1))
            dof = max(1, len(L) - mreq)
            rms = math.sqrt(float((v.T @ Wm @ v)/dof))
            H_geo = np.array(H_geo_rows)
            dops = compute_dops(H_geo) if H_geo.shape[0] >= 4 else {}
            return x, rms, res_by_sat, dops, H_geo
    # not converged
    H_geo = np.array(H_geo_rows) if 'H_geo_rows' in locals() else np.zeros((0,4))
    dops = compute_dops(H_geo) if H_geo.shape[0] >= 4 else {}
    return x, float('nan'), res_by_sat, dops, H_geo

# ----------------------------
# ENU errors and summary
# ----------------------------
def enu_from_ref(x_ecef:np.ndarray, ref_llh:Tuple[float,float,float]) -> np.ndarray:
    lat,lon,h = ref_llh
    E = ecef_to_enu_matrix(lat,lon)
    ref_ecef = llh_to_ecef(lat,lon,h)
    enu = E @ (x_ecef - ref_ecef)
    return enu

# ----------------------------
# Driver
# ----------------------------
def run(obs_path:str, nav_path:str, elev_mask:float=10.0, est_trop:bool=False,
        ref_lat:Optional[float]=None, ref_lon:Optional[float]=None, ref_h:Optional[float]=None):
    nav_header, ephs = parse_rinex3_nav(nav_path)
    types_G, epochs = parse_rinex3_obs(obs_path)

    # Initial state
    x_state = np.zeros(5 if est_trop else 4)
    x_state[:3] = llh_to_ecef(0.0,0.0,0.0)
    x_state[3] = 0.0
    if est_trop: x_state[4] = 0.1

    ref_llh = None
    if ref_lat is not None and ref_lon is not None and ref_h is not None:
        ref_llh = (math.radians(ref_lat), math.radians(ref_lon), ref_h)

    enu_list=[]; err2d_list=[]; err3d_list=[]; dop_list=[]
    first_ref_locked=False
    for k, epoch in enumerate(epochs):
        x_state, rms, res, dops, H_geo = spp_epoch_wls(epoch, nav_header, ephs, x_state, elev_mask_deg=elev_mask, est_zwd=est_trop)
        if not np.isfinite(rms):
            print(f"Epoch {k+1} W{epoch.week} {epoch.t_gps:8.1f}s: solution not available (nsat < req).")
            continue
        # Establish reference if not provided: use first successful solution
        if ref_llh is None and not first_ref_locked:
            ref_llh = ecef_to_llh(x_state[:3])
            first_ref_locked=True
        enu = enu_from_ref(x_state[:3], ref_llh)
        e,n,u = enu
        err2d = math.hypot(e,n)
        err3d = math.sqrt(e*e + n*n + u*u)
        enu_list.append(enu); err2d_list.append(err2d); err3d_list.append(err3d)
        dop_list.append(dops)
        lat,lon,h = ecef_to_llh(x_state[:3])
        print(f"Epoch {k+1} W{epoch.week} {epoch.t_gps:8.1f}s:"
              f" lat={math.degrees(lat):.8f} lon={math.degrees(lon):.8f} h={h:.3f} m,"
              f" ENU=({e:.3f},{n:.3f},{u:.3f}) m, 2D={err2d:.3f} m, 3D={err3d:.3f} m,"
              f" RMS={rms:.3f} m,"
              f" DOPs={{{'GDOP':{dops.get('GDOP',float('nan')):.2f},'PDOP':{dops.get('PDOP',float('nan')):.2f},'HDOP':{dops.get('HDOP',float('nan')):.2f},'VDOP':{dops.get('VDOP',float('nan')):.2f}}}}")
    if enu_list:
        enu_arr = np.vstack(enu_list)
        e_rms = math.sqrt(np.mean(enu_arr[:,0]**2))
        n_rms = math.sqrt(np.mean(enu_arr[:,1]**2))
        u_rms = math.sqrt(np.mean(enu_arr[:,2]**2))
        err2d_arr = np.array(err2d_list)
        err3d_arr = np.array(err3d_list)
        print("\nSummary (relative to reference):")
        print(f"RMS ENU [m]: E={e_rms:.3f}, N={n_rms:.3f}, U={u_rms:.3f}")
        print(f"2D error: mean={np.mean(err2d_arr):.3f} m, RMS={math.sqrt(np.mean(err2d_arr**2)):.3f} m, 95th={np.percentile(err2d_arr,95):.3f} m")
        print(f"3D error: mean={np.mean(err3d_arr):.3f} m, RMS={math.sqrt(np.mean(err3d_arr**2)):.3f} m, 95th={np.percentile(err3d_arr,95):.3f} m")
        if ref_llh is not None:
            print(f"Reference (LLH): lat={math.degrees(ref_llh[0]):.9f}, lon={math.degrees(ref_llh[1]):.9f}, h={ref_llh[2]:.3f} m")

# ----------------------------
# 可配置的硬编码路径 (请修改为实际文件路径)
# ----------------------------
DEFAULT_OBS_PATH = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/20250527_PolyU_X.obs"
DEFAULT_NAV_PATH = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/20250527_PolyU_X.nav"
DEFAULT_ELEV_MASK = 10.0
DEFAULT_EST_TROP = False
# 若需要参考点, 填写 (度 / 米); 否则设为 None
DEFAULT_REF_LAT = None   # 例如 30.0
DEFAULT_REF_LON = None   # 例如 114.0
DEFAULT_REF_H   = None   # 例如 50.0

# ----------------------------
# CLI (已改为直接调用)
# ----------------------------
def main():
    # 直接使用上面定义的常量
    run(
        DEFAULT_OBS_PATH,
        DEFAULT_NAV_PATH,
        elev_mask=DEFAULT_ELEV_MASK,
        est_trop=DEFAULT_EST_TROP,
        ref_lat=DEFAULT_REF_LAT,
        ref_lon=DEFAULT_REF_LON,
        ref_h=DEFAULT_REF_H,
    )

if __name__ == "__main__":
    main()