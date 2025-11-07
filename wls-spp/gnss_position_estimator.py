"""
Least-Squares SPP from matRTKLIB-exported CSVs.

Inputs (from your MATLAB/matRTKLIB script):
- pseudoranges_meas.csv          : shape (max_sats, num_epochs), meters
- satellite_clock_bias.csv       : shape (max_sats, num_epochs), meters (c*dt_s)
- ionospheric_delay.csv          : shape (max_sats, num_epochs), meters
- tropospheric_delay.csv         : shape (max_sats, num_epochs), meters
- satellite_positions.csv        : shape (max_sats, num_epochs*3), ECEF meters
  (epoch k occupies columns [3k, 3k+1, 3k+2])

Outputs:
- lse_solution.csv: per-epoch solution [epoch, xs, ys, zs, lat(deg), lon(deg), h(m), clock_bias_m, nsat, PDOP, HDOP, VDOP]
"""

import math
import numpy as np
# 新增绘图库
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (仅用于注册 3D 投影)
from matplotlib.ticker import FormatStrFormatter
import io

# ---------------------- WGS-84 constants ----------------------
WGS84_A = 6378137.0
WGS84_F = 1.0/298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared

C = 299792458.0  # m/s, speed of light (not directly used since clk bias is in meters)

def load_csv(path):
    # Robustly load with NaNs; allow empty lines
    return np.genfromtxt(path, delimiter=',')

def split_sat_positions_matrix(sat_pos_mat):
    """
    sat_pos_mat: (max_sats, num_epochs*3)
    returns: list of length num_epochs; each is array (max_sats, 3)
    """
    max_sats, cols = sat_pos_mat.shape
    assert cols % 3 == 0, "satellite_positions.csv columns must be multiple of 3"
    num_epochs = cols // 3
    per_epoch = []
    for k in range(num_epochs):
        sl = slice(3*k, 3*k+3)
        per_epoch.append(sat_pos_mat[:, sl])
    return per_epoch

def ecef_to_lla(x, y, z):
    """
    Convert ECEF (m) to geodetic LLH (deg, deg, m), WGS-84.
    Iterative method.
    """
    a = WGS84_A
    e2 = WGS84_E2
    b = a * (1 - WGS84_F)

    r = math.hypot(x, y)
    if r < 1e-12 and abs(z) < 1e-12:
        return 0.0, 0.0, -a  # deg,deg,m (arbitrary)

    lon = math.degrees(math.atan2(y, x))
    # initial lat
    lat = math.atan2(z, r * (1 - e2))
    for _ in range(10):
        sinl = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sinl*sinl)
        h = r / math.cos(lat) - N
        lat_new = math.atan2(z, r * (1.0 - e2 * N/(N + h)))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    sinl = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sinl*sinl)
    h = r / math.cos(lat) - N
    return math.degrees(lat), lon, h

def lla_to_ecef(lat_deg, lon_deg, h):
    a = WGS84_A
    e2 = WGS84_E2
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sinl = math.sin(lat); cosl = math.cos(lat)
    sinL = math.sin(lon); cosL = math.cos(lon)
    N = a / math.sqrt(1 - e2*sinl*sinl)
    x = (N + h) * cosl * cosL
    y = (N + h) * cosl * sinL
    z = (N*(1 - e2) + h) * sinl
    return np.array([x, y, z], dtype=float)

def ecef_to_enu_rotation(lat_deg, lon_deg):
    """
    Rotation matrix R such that [e,n,u]^T = R * [x,y,z]^T (ECEF delta).
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    sL, cL = math.sin(lon), math.cos(lon)
    R = np.array([
        [-sL,            cL,           0.0],
        [-sl*cL,        -sl*sL,        cl ],
        [ cl*cL,         cl*sL,        sl ]
    ], dtype=float)
    return R

def compute_dops(H, lat_deg, lon_deg):
    """
    Compute PDOP/HDOP/VDOP from geometry matrix H (n x 4).
    We transform covariance in XYZ to ENU at solution.
    """
    if H.shape[0] < 4:
        return np.nan, np.nan, np.nan
    # Covariance up to a scale: Q = (H^T H)^{-1}
    try:
        Q = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    # Extract XYZ block and clock
    Q_xyz = Q[0:3, 0:3]
    # Rotate XYZ covariance to ENU
    R = ecef_to_enu_rotation(lat_deg, lon_deg)
    Q_enu = R @ Q_xyz @ R.T

    # HDOP = sqrt(var_e + var_n)
    var_e = Q_enu[0, 0]
    var_n = Q_enu[1, 1]
    var_u = Q_enu[2, 2]
    if min(var_e, var_n, var_u) < 0:
        return np.nan, np.nan, np.nan

    HDOP = math.sqrt(var_e + var_n)
    VDOP = math.sqrt(var_u)
    PDOP = math.sqrt(var_e + var_n + var_u)
    return PDOP, HDOP, VDOP

# 新增：将经纬高序列写出为 KML（LineString）
def save_kml_with_truth(ls_lon_lat_h, truth_lon_lat_h, path,
                        name_ls="LS SPP Solution", name_truth="Truth (NAV-PVT)"):
    """
    ls_lon_lat_h       : [(lon_deg, lat_deg, h_m), ...]
    truth_lon_lat_h    : [(lon_deg, lat_deg, h_m), ...] 或 None
    path               : 输出 KML 文件路径
    """
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    lines.append('<Document>')
    lines.append('  <name>SPP vs Truth</name>')
    # KML 颜色 aabbggrr：绿色=ff00ff00，蓝色=ffff0000
    lines.append('  <Style id="ls"><LineStyle><color>ff00ff00</color><width>3</width></LineStyle></Style>')
    lines.append('  <Style id="truth"><LineStyle><color>ffff0000</color><width>3</width></LineStyle></Style>')

    # LS 轨迹
    lines.append('  <Placemark>')
    lines.append(f'    <name>{name_ls}</name>')
    lines.append('    <styleUrl>#ls</styleUrl>')
    lines.append('    <LineString>')
    lines.append('      <tessellate>1</tessellate>')
    lines.append('      <altitudeMode>absolute</altitudeMode>')
    lines.append('      <coordinates>')
    for lon, lat, h in ls_lon_lat_h:
        if np.isfinite(lon) and np.isfinite(lat) and np.isfinite(h):
            lines.append(f'        {float(lon):.9f},{float(lat):.9f},{float(h):.3f}')
    lines.append('      </coordinates>')
    lines.append('    </LineString>')
    lines.append('  </Placemark>')

    # Truth 轨迹（可选）
    if truth_lon_lat_h is not None and len(truth_lon_lat_h) > 0:
        lines.append('  <Placemark>')
        lines.append(f'    <name>{name_truth}</name>')
        lines.append('    <styleUrl>#truth</styleUrl>')
        lines.append('    <LineString>')
        lines.append('      <tessellate>1</tessellate>')
        lines.append('      <altitudeMode>absolute</altitudeMode>')
        lines.append('      <coordinates>')
        for lon, lat, h in truth_lon_lat_h:
            if np.isfinite(lon) and np.isfinite(lat) and np.isfinite(h):
                lines.append(f'        {float(lon):.9f},{float(lat):.9f},{float(h):.3f}')
        lines.append('      </coordinates>')
        lines.append('    </LineString>')
        lines.append('  </Placemark>')

    lines.append('</Document>')
    lines.append('</kml>')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def solve_epoch_ls(p_corr, sat_pos, x0=None, max_iter=20, tol=1e-4):
    """
    p_corr: (m,) corrected pseudoranges (P + clk_s - ion - trop)
    sat_pos: (m,3) satellite ECEF positions
    x0: initial state [x,y,z,cbias(m)] or None -> start at Earth center
    Returns: (x,y,z,cbias_m), H_at_solution, used_sat_count, success(bool)
    """
    # Valid satellites: need positions & pseudorange both finite
    valid = np.isfinite(p_corr) & np.isfinite(sat_pos).all(axis=1)
    idx = np.where(valid)[0]
    if idx.size < 4:
        return None, None, 0, False

    P = p_corr[idx]
    S = sat_pos[idx, :]

    # Initial state
    if x0 is None:
        xr = np.zeros(3, dtype=float)  # start at geocenter
    else:
        xr = np.array(x0[0:3], dtype=float)
    cb = 0.0 if (x0 is None or len(x0) < 4) else float(x0[3])

    success = False
    H_last = None

    for _ in range(max_iter):
        # geometric ranges and LOS
        diff = S - xr  # sat - rec
        rho = np.linalg.norm(diff, axis=1)
        # unit vectors from receiver to satellite:
        # u = (xr - xs)/rho = -(xs - xr)/rho
        u = (xr[None, :] - S) / rho[:, None]

        # predicted pseudorange = rho + cb
        pred = rho + cb
        v = P - pred  # residuals

        # Build H (n x 4)
        H = np.zeros((idx.size, 4), dtype=float)
        H[:, 0:3] = u
        H[:, 3] = 1.0

        # LS update
        try:
            dx, *_ = np.linalg.lstsq(H, v, rcond=None)
        except np.linalg.LinAlgError:
            break

        xr += dx[0:3]
        cb += dx[3]

        H_last = H

        if np.linalg.norm(dx) < tol:
            success = True
            break

    if not success:
        # still return the last iterate if at least 4 sats
        success = True

    return (xr[0], xr[1], xr[2], cb), H_last, idx.size, success

# 新增：生成 2D 经纬度轨迹与 3D ENU 轨迹图
def generate_and_save_plots(results, out_path, truth_llh=None):
    valid = [r for r in results if np.isfinite(r[4]) and np.isfinite(r[5]) and np.isfinite(r[6])]
    if len(valid) < 1:
        print("[Plot] no valid LS solution points.")
        return

    lats = np.array([r[4] for r in valid])
    lons = np.array([r[5] for r in valid])
    hs   = np.array([r[6] for r in valid])
    xyz  = np.array([[r[1], r[2], r[3]] for r in valid])

    # 参考点
    ref_lat, ref_lon, ref_h = lats[0], lons[0], hs[0]
    ref_xyz = xyz[0]
    R = ecef_to_enu_rotation(ref_lat, ref_lon)
    enu = (R @ (xyz - ref_xyz).T).T

    # 真值（来自 NAV-PVT）
    t_lon = t_lat = t_h = t_enu = None
    if truth_llh is not None and len(truth_llh) >= 1:
        t_lon = np.asarray(truth_llh[:,0], float)
        t_lat = np.asarray(truth_llh[:,1], float)
        t_h   = np.asarray(truth_llh[:,2], float)
        t_xyz = np.array([lla_to_ecef(t_lat[i], t_lon[i], t_h[i]) for i in range(len(t_lon))])
        n = min(len(lons), len(t_lon))  # 按索引对齐
        lons, lats, hs, xyz, enu = lons[:n], lats[:n], hs[:n], xyz[:n], enu[:n]
        t_lon, t_lat, t_h = t_lon[:n], t_lat[:n], t_h[:n]
        t_xyz = t_xyz[:n]
        t_enu = (R @ (t_xyz - ref_xyz).T).T
        print(f"[Plot] LS points: {len(lats)}, Truth(NAV-PVT) points: {len(t_lon)} (aligned by index).")

    prefix = out_path.rsplit(".", 1)[0]
    plot2d = prefix + "_track_2d.png"
    plot3d = prefix + "_track_enu_3d.png"

    # 2D Lon-Lat
    fig, ax = plt.subplots(figsize=(6, 5))
    (ax.plot if len(lons)>=2 else ax.scatter)(lons, lats, 'k.-' if len(lons)>=2 else 'k', linewidth=1, markersize=3, label='LS')
    if t_lon is not None:
        (ax.plot if len(t_lon)>=2 else ax.scatter)(t_lon, t_lat, 'b.-' if len(t_lon)>=2 else 'b', linewidth=1, markersize=2, label='Truth (PVT)')
    ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")
    ax.set_title("Geographic Track (Lon-Lat)"); ax.grid(alpha=0.4)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.6f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
    ax.legend(); fig.tight_layout(); fig.savefig(plot2d, dpi=150); plt.show(); plt.close(fig)

    # 3D ENU
    fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(111, projection='3d')
    if len(enu)>=2: ax.plot(enu[:,0], enu[:,1], enu[:,2], 'r.-', linewidth=1, markersize=3, label='LS')
    else:           ax.scatter(enu[:,0], enu[:,1], enu[:,2], c='r', s=14, label='LS')
    if t_enu is not None:
        if len(t_enu)>=2: ax.plot(t_enu[:,0], t_enu[:,1], t_enu[:,2], 'b.-', linewidth=1, markersize=2, label='Truth (PVT)')
        else:             ax.scatter(t_enu[:,0], t_enu[:,1], t_enu[:,2], c='b', s=16, label='Truth (PVT)')
        mask = np.isfinite(enu).all(axis=1) & np.isfinite(t_enu).all(axis=1)
        de = enu[mask] - t_enu[mask]
        if de.size:
            err2d = np.sqrt(de[:,0]**2 + de[:,1]**2)
            err3d = np.linalg.norm(de, axis=1)
            bias_u = np.nanmedian(de[:,2])
            print(f"[Stats vs PVT] 2D RMS={np.sqrt(np.mean(err2d**2)):.3f} m, 3D RMS={np.sqrt(np.mean(err3d**2)):.3f} m, U-bias≈{bias_u:.3f} m")
    ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)"); ax.set_zlabel("U (m)")
    ax.set_title("Track in ENU Frame (Ref = first valid LS point)")
    ranges = [enu[:, i].max() - enu[:, i].min() for i in range(3)]
    max_range = max(ranges) if np.isfinite(ranges).all() else 1.0
    centers = [(enu[:, i].max() + enu[:, i].min())/2.0 for i in range(3)]
    for i, (low, high) in enumerate([(c - max_range/2, c + max_range/2) for c in centers]):
        if i == 0: ax.set_xlim(low, high)
        if i == 1: ax.set_ylim(low, high)
        if i == 2: ax.set_zlim(low, high)
    ax.legend(); plt.tight_layout(); plt.savefig(plot3d, dpi=150); plt.show(); plt.close()
    print(f"Saved {plot2d} and {plot3d}.")


def load_truth_llh_from_nav_pvt(path):
    """
    读取 UBX NAV-PVT：
      - 支持逗号或空白分隔、有/无表头
      - 列优先使用: lon, lat, height（无 height 时回退 hMSL）
      - 自动把 1e-7 度 → 度；毫米 → 米
    返回: (N,3)  [lon_deg, lat_deg, h_m]
    """
    # 先判定分隔符
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        head_line = ""
        for _ in range(100):
            line = f.readline()
            if not line:
                break
            if line.strip() == "" or line.lstrip().startswith("#"):
                continue
            head_line = line
            break
    delim = ',' if (',' in head_line) else None  # None=按空白分隔

    # 尝试带表头读取
    try:
        data = np.genfromtxt(path, delimiter=delim, names=True, dtype=None, encoding=None)
        names = set(data.dtype.names or [])
        if 'lon' in names and 'lat' in names:
            lon = np.asarray(data['lon'], float)
            lat = np.asarray(data['lat'], float)
            if   'height' in names: h = np.asarray(data['height'], float)
            elif 'hMSL'   in names: h = np.asarray(data['hMSL'],   float)
            else:                   h = np.zeros_like(lon)
        else:
            raise ValueError("no named columns lon/lat")
    except Exception:
        # 兜底：无表头，取前三列
        arr = np.genfromtxt(path, delimiter=delim, ndmin=2)
        lon, lat, h = arr[:, 0].astype(float), arr[:, 1].astype(float), arr[:, 2].astype(float)

    # 单位自动识别与换算
    # if np.nanmedian(np.abs(lon)) > 400.0: lon *= 1e-7  # 1e-7 度 → 度
    # if np.nanmedian(np.abs(lat)) > 100.0: lat *= 1e-7
    # 高度：若绝对值中位数 > 10000，视为毫米 → 米（例如 16737 → 16.737 m）
    if np.nanmedian(np.abs(h)) > 1.0e4:   h   *= 1e-3

    return np.column_stack([lon, lat, h])


def main():
    # 直接在此处指定输入/输出文件路径，替换为你的实际路径
    pseudoranges = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/pseudoranges_meas.csv"
    sat_clk      = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/satellite_clock_bias.csv"
    ion          = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/ionospheric_delay.csv"
    trop         = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/tropospheric_delay.csv"
    sat_pos      = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/satellite_positions.csv"
    out_path     = "/Users/jay/Documents/Bachelor/aae4203/Result/lse_solution.csv"
    kml_out      = "/Users/jay/Documents/Bachelor/aae4203/Result/lse_solution.kml"  # 基于 CSV 路径生成 KML
    pvt_truth_path = "/Users/jay/Documents/Bachelor/aae4203/ubx-message/data/NAV-PVT.csv"  # 真值 PVT 文件路径

    # 可选初值（如果需要），否则置 None
    llh0 = None  # e.g. "22.304139 114.180131 -20.2" -> set below if needed

    # 迭代参数
    max_iter = 20
    tol = 1e-4

    # 读取文件
    P = load_csv(pseudoranges)          # (Smax, E)
    CLK = load_csv(sat_clk)             # (Smax, E)
    ION = load_csv(ion)                 # (Smax, E)
    TRP = load_csv(trop)                # (Smax, E)
    SATM = load_csv(sat_pos)            # (Smax, 3E)

    # Basic checks（保持原逻辑）
    if P.ndim != 2 or CLK.ndim != 2 or ION.ndim != 2 or TRP.ndim != 2 or SATM.ndim != 2:
        raise ValueError("All CSVs must be 2D matrices.")
    Smax, E = P.shape
    if CLK.shape != (Smax, E) or ION.shape != (Smax, E) or TRP.shape != (Smax, E):
        raise ValueError("pseudorange/clock/ion/trop shapes must all be (max_sats, num_epochs)")
    if SATM.shape[0] != Smax or SATM.shape[1] != 3*E:
        raise ValueError("satellite_positions.csv must be (max_sats, num_epochs*3)")

    sat_per_epoch = split_sat_positions_matrix(SATM)  # list length E, each (Smax,3)

    # Initial guess
    x0 = None
    if llh0 is not None:
        llh0_vals = [float(v) for v in llh0.strip().split()]
        if len(llh0_vals) != 3:
            raise ValueError("llh0 must be 3 numbers: 'lat lon h'")
        xyz0 = lla_to_ecef(llh0_vals[0], llh0_vals[1], llh0_vals[2])
        x0 = np.array([xyz0[0], xyz0[1], xyz0[2], 0.0], dtype=float)

    results = []
    for k in range(E):
        # corrected pseudorange (meters)
        P_corr = P[:, k] + CLK[:, k] - ION[:, k] - TRP[:, k]
        S_k = sat_per_epoch[k]  # (Smax,3)

        sol, H, ns, ok = solve_epoch_ls(
            P_corr, S_k, x0=x0, max_iter=max_iter, tol=tol
        )

        if (not ok) or (sol is None):
            results.append([k+1] + [np.nan]*11)  # epoch index starting at 1
            continue

        xs, ys, zs, cb_m = sol
        lat, lon, h = ecef_to_lla(xs, ys, zs)

        PDOP, HDOP, VDOP = (np.nan, np.nan, np.nan)
        if H is not None and ns >= 4 and np.isfinite(lat) and np.isfinite(lon):
            PDOP, HDOP, VDOP = compute_dops(H, lat, lon)

        results.append([
            k+1, xs, ys, zs, lat, lon, h, cb_m, int(ns),
            PDOP, HDOP, VDOP
        ])

        # Use previous solution as next initial guess (helps convergence)
        x0 = np.array([xs, ys, zs, cb_m], dtype=float)

    # Save
    header = "epoch,x,y,z,lat_deg,lon_deg,h_m,clock_bias_m,nsat,PDOP,HDOP,VDOP"
    out = np.array(results, dtype=float)
    np.savetxt(out_path, out, delimiter=",", header=header, comments="", fmt="%.10f")
    print(f"Saved {out_path} with {out.shape[0]} epochs.")

    # 读取真值 PVT
    try:
        truth_llh = load_truth_llh_from_nav_pvt(pvt_truth_path)
        print(f"[Truth PVT] shape: {truth_llh.shape}  "
              f"lon[{np.nanmin(truth_llh[:,0]):.6f},{np.nanmax(truth_llh[:,0]):.6f}]  "
              f"lat[{np.nanmin(truth_llh[:,1]):.6f},{np.nanmax(truth_llh[:,1]):.6f}]  "
              f"h[{np.nanmin(truth_llh[:,2]):.3f},{np.nanmax(truth_llh[:,2]):.3f}] m")
    except Exception as e:
        print(f"[Truth PVT] load failed: {e}")
        truth_llh = None

    ls_coords = [(r[5], r[4], r[6]) for r in results  # (lon, lat, h)
                if np.isfinite(r[4]) and np.isfinite(r[5]) and np.isfinite(r[6])]
    truth_coords = None
    if truth_llh is not None:
        # truth_llh 形状 (N,3): [lon, lat, h]，直接转成 list[tuple]
        truth_coords = [(float(lon), float(lat), float(h)) for lon, lat, h in truth_llh]

    if ls_coords or truth_coords:
        save_kml_with_truth(ls_coords, truth_coords, kml_out,
                            name_ls="LS SPP Solution", name_truth="Truth (NAV-PVT)")
        print(f"Saved {kml_out} with LS {len(ls_coords)} pts"
            f"{'' if truth_coords is None else f' and Truth {len(truth_coords)} pts'}.")

    # 新增：绘图调用
    generate_and_save_plots(results, out_path, truth_llh=truth_llh)

if __name__ == "__main__":
    main()
