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
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

# ---------------------- WGS-84 constants ----------------------
WGS84_A = 6378137.0
WGS84_F = 1.0/298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared

C = 299792458.0  # m/s, speed of light (not directly used since clk bias is in meters)
# 地球自转角速度
OMEGA_E = 7.2921151467e-5  # rad/s

# 地球自转改正函数（Sagnac）
def correct_for_earth_rotation(S, xr):
    """
    根据当前接收机位置 xr 对卫星坐标 S 进行地球自转改正。
    返回:
        S_corr: 修正后的卫星坐标 (n,3)
        rho0  : 未改正前几何距离 (n,)
        tau   : 传播时间估计 (n,)
    """
    diff0 = S - xr
    rho0 = np.linalg.norm(diff0, axis=1)
    tau = rho0 / C
    S_corr = S.copy()
    for i, ang in enumerate(OMEGA_E * tau):
        ca = math.cos(ang); sa = math.sin(ang)
        # 旋转矩阵 R3(+ωτ)（将卫星从接收时刻旋回发射时刻参考系）
        R3 = np.array([[ ca,  sa, 0],
                       [-sa,  ca, 0],
                       [  0,   0, 1]], dtype=float)
        S_corr[i] = R3 @ S[i]
    return S_corr, rho0, tau

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
                        name_ls="LS SPP Solution", name_truth="Truth (NAV-ECEF)"):
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


def solve_epoch_ls(p_corr, sat_pos, x0=None, max_iter=20, tol=1e-4, earth_rotation=True):
    """
    p_corr: (m,) corrected pseudoranges (P + clk_s - ion - trop)
    sat_pos: (m,3) satellite ECEF positions (接收历元)
    x0: initial state [x,y,z,cbias(m)] or None -> start at Earth center
    earth_rotation: 是否应用地球自转(Sagnac)改正（旋转卫星坐标）
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
        # 几何距离及地球自转改正
        if earth_rotation:
            S_iter, rho_pre, tau = correct_for_earth_rotation(S, xr)
        else:
            S_iter = S
        diff = S_iter - xr
        rho = np.linalg.norm(diff, axis=1)
        # 单位方向向量（从接收机指向卫星）
        u = (xr[None, :] - S_iter) / rho[:, None]

        # predicted pseudorange = rho + cb
        pred = rho + cb
        v = P - pred

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

# 误差分析计算函数
def compute_error_statistics(ls_xyz, truth_xyz, ref_lat, ref_lon):
    """
    计算 LS 解相对于真值的误差统计(ENU坐标系)
    
    参数:
        ls_xyz: (N, 3) LS解的ECEF坐标
        truth_xyz: (N, 3) 真值的ECEF坐标
        ref_lat, ref_lon: 参考点纬度经度（度）
    
    返回:
        dict 包含:
            'e_err', 'n_err', 'u_err': (N,) 各方向误差
            'err_2d', 'err_3d': (N,) 2D和3D误差
            'stats': 统计量字典
    """
    # 转换到ENU
    R = ecef_to_enu_rotation(ref_lat, ref_lon)
    ref_xyz = truth_xyz[0]  # 使用第一个真值点作为参考
    
    ls_enu = (R @ (ls_xyz - ref_xyz).T).T
    truth_enu = (R @ (truth_xyz - ref_xyz).T).T
    
    # 计算各方向误差
    delta = ls_enu - truth_enu
    e_err = delta[:, 0]
    n_err = delta[:, 1]
    u_err = delta[:, 2]
    
    # 2D和3D误差
    err_2d = np.sqrt(e_err**2 + n_err**2)
    err_3d = np.linalg.norm(delta, axis=1)
    
    # 统计量
    valid_mask = np.isfinite(err_3d)
    if valid_mask.sum() > 0:
        stats = {
            'e_mean': np.mean(e_err[valid_mask]),
            'e_std': np.std(e_err[valid_mask]),
            'e_rms': np.sqrt(np.mean(e_err[valid_mask]**2)),
            'n_mean': np.mean(n_err[valid_mask]),
            'n_std': np.std(n_err[valid_mask]),
            'n_rms': np.sqrt(np.mean(n_err[valid_mask]**2)),
            'u_mean': np.mean(u_err[valid_mask]),
            'u_std': np.std(u_err[valid_mask]),
            'u_rms': np.sqrt(np.mean(u_err[valid_mask]**2)),
            '2d_mean': np.mean(err_2d[valid_mask]),
            '2d_std': np.std(err_2d[valid_mask]),
            '2d_rms': np.sqrt(np.mean(err_2d[valid_mask]**2)),
            '3d_mean': np.mean(err_3d[valid_mask]),
            '3d_std': np.std(err_3d[valid_mask]),
            '3d_rms': np.sqrt(np.mean(err_3d[valid_mask]**2)),
        }
    else:
        stats = {}
    
    return {
        'e_err': e_err,
        'n_err': n_err,
        'u_err': u_err,
        'err_2d': err_2d,
        'err_3d': err_3d,
        'stats': stats
    }

# 绘图函数，包含误差分析
def generate_and_save_plots(results, out_path, truth_llh=None):
    valid = [r for r in results if np.isfinite(r[4]) and np.isfinite(r[5]) and np.isfinite(r[6])]
    if len(valid) < 1:
        print("[Plot] no valid LS solution points.")
        return

    lats = np.array([r[4] for r in valid])
    lons = np.array([r[5] for r in valid])
    hs   = np.array([r[6] for r in valid])
    xyz  = np.array([[r[1], r[2], r[3]] for r in valid])
    epochs = np.array([r[0] for r in valid])

    # 参考点
    ref_lat, ref_lon, ref_h = lats[0], lons[0], hs[0]
    ref_xyz = xyz[0]
    R = ecef_to_enu_rotation(ref_lat, ref_lon)
    enu = (R @ (xyz - ref_xyz).T).T

    # 真值
    t_lon = t_lat = t_h = t_enu = None
    error_data = None
    if truth_llh is not None and len(truth_llh) >= 1:
        t_lon = np.asarray(truth_llh[:,0], float)
        t_lat = np.asarray(truth_llh[:,1], float)
        t_h   = np.asarray(truth_llh[:,2], float)
        t_xyz = np.array([lla_to_ecef(t_lat[i], t_lon[i], t_h[i]) for i in range(len(t_lon))])
        n = min(len(lons), len(t_lon))  # 按索引对齐
        lons, lats, hs, xyz, enu = lons[:n], lats[:n], hs[:n], xyz[:n], enu[:n]
        epochs = epochs[:n]
        t_lon, t_lat, t_h = t_lon[:n], t_lat[:n], t_h[:n]
        t_xyz = t_xyz[:n]
        t_enu = (R @ (t_xyz - ref_xyz).T).T
        
        # 计算误差统计
        error_data = compute_error_statistics(xyz, t_xyz, ref_lat, ref_lon)
        
        print(f"[Plot] LS points: {len(lats)}, Truth(NAV-ECEF) points: {len(t_lon)} (aligned by index).")
        if error_data['stats']:
            st = error_data['stats']
            print(f"[Error Stats]")
            print(f"  East  - Mean: {st['e_mean']:7.3f} m, Std: {st['e_std']:6.3f} m, RMS: {st['e_rms']:6.3f} m")
            print(f"  North - Mean: {st['n_mean']:7.3f} m, Std: {st['n_std']:6.3f} m, RMS: {st['n_rms']:6.3f} m")
            print(f"  Up    - Mean: {st['u_mean']:7.3f} m, Std: {st['u_std']:6.3f} m, RMS: {st['u_rms']:6.3f} m")
            print(f"  2D    - Mean: {st['2d_mean']:7.3f} m, Std: {st['2d_std']:6.3f} m, RMS: {st['2d_rms']:6.3f} m")
            print(f"  3D    - Mean: {st['3d_mean']:7.3f} m, Std: {st['3d_std']:6.3f} m, RMS: {st['3d_rms']:6.3f} m")

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
    
    # 误差分析图
    if error_data is not None:
        # 1. ENU误差时间序列
        plot_enu_err = prefix + "_error_enu_timeseries.png"
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        axes[0].plot(epochs, error_data['e_err'], 'b.-', linewidth=0.8, markersize=2)
        axes[0].axhline(0, color='k', linestyle='--', linewidth=0.5)
        axes[0].set_ylabel('East Error (m)')
        axes[0].grid(alpha=0.3)
        axes[0].set_title('Position Errors in ENU Frame')
        
        axes[1].plot(epochs, error_data['n_err'], 'g.-', linewidth=0.8, markersize=2)
        axes[1].axhline(0, color='k', linestyle='--', linewidth=0.5)
        axes[1].set_ylabel('North Error (m)')
        axes[1].grid(alpha=0.3)
        
        axes[2].plot(epochs, error_data['u_err'], 'r.-', linewidth=0.8, markersize=2)
        axes[2].axhline(0, color='k', linestyle='--', linewidth=0.5)
        axes[2].set_ylabel('Up Error (m)')
        axes[2].set_xlabel('Epoch')
        axes[2].grid(alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(plot_enu_err, dpi=150)
        plt.show()
        plt.close(fig)
        print(f"Saved {plot_enu_err}.")
        
        # 2. 2D/3D误差时间序列
        plot_2d3d_err = prefix + "_error_2d3d_timeseries.png"
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        axes[0].plot(epochs, error_data['err_2d'], 'm.-', linewidth=0.8, markersize=2)
        axes[0].set_ylabel('2D Error (m)')
        axes[0].grid(alpha=0.3)
        axes[0].set_title('2D and 3D Position Errors')
        
        axes[1].plot(epochs, error_data['err_3d'], 'c.-', linewidth=0.8, markersize=2)
        axes[1].set_ylabel('3D Error (m)')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(plot_2d3d_err, dpi=150)
        plt.show()
        plt.close(fig)
        print(f"Saved {plot_2d3d_err}.")
        
        # 3. 误差分布直方图
        plot_hist = prefix + "_error_histogram.png"
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        
        # E/N/U误差分布
        for idx, (err, name, color) in enumerate([
            (error_data['e_err'], 'East', 'blue'),
            (error_data['n_err'], 'North', 'green'),
            (error_data['u_err'], 'Up', 'red')
        ]):
            valid = err[np.isfinite(err)]
            axes[0, idx].hist(valid, bins=30, color=color, alpha=0.7, edgecolor='black')
            axes[0, idx].axvline(0, color='k', linestyle='--', linewidth=1)
            axes[0, idx].set_xlabel(f'{name} Error (m)')
            axes[0, idx].set_ylabel('Count')
            axes[0, idx].set_title(f'{name} Error Distribution')
            axes[0, idx].grid(alpha=0.3)
        
        # 2D/3D误差分布
        for idx, (err, name, color) in enumerate([
            (error_data['err_2d'], '2D', 'magenta'),
            (error_data['err_3d'], '3D', 'cyan')
        ]):
            valid = err[np.isfinite(err)]
            axes[1, idx].hist(valid, bins=30, color=color, alpha=0.7, edgecolor='black')
            axes[1, idx].set_xlabel(f'{name} Error (m)')
            axes[1, idx].set_ylabel('Count')
            axes[1, idx].set_title(f'{name} Error Distribution')
            axes[1, idx].grid(alpha=0.3)
        
        # 隐藏空白子图
        axes[1, 2].axis('off')
        
        fig.tight_layout()
        fig.savefig(plot_hist, dpi=150)
        plt.show()
        plt.close(fig)
        print(f"Saved {plot_hist}.")
        
        # 4. 误差统计汇总图
        if error_data['stats']:
            plot_stats = prefix + "_error_statistics.png"
            fig, ax = plt.subplots(figsize=(10, 6))
            
            components = ['East', 'North', 'Up', '2D', '3D']
            means = [error_data['stats'][f'{k}_mean'] for k in ['e', 'n', 'u', '2d', '3d']]
            stds = [error_data['stats'][f'{k}_std'] for k in ['e', 'n', 'u', '2d', '3d']]
            rms = [error_data['stats'][f'{k}_rms'] for k in ['e', 'n', 'u', '2d', '3d']]
            
            x = np.arange(len(components))
            width = 0.25
            
            ax.bar(x - width, means, width, label='Mean', color='skyblue', edgecolor='black')
            ax.bar(x, stds, width, label='Std Dev', color='lightcoral', edgecolor='black')
            ax.bar(x + width, rms, width, label='RMS', color='lightgreen', edgecolor='black')
            
            ax.set_xlabel('Component')
            ax.set_ylabel('Error (m)')
            ax.set_title('Error Statistics Summary')
            ax.set_xticks(x)
            ax.set_xticklabels(components)
            ax.legend()
            ax.grid(alpha=0.3, axis='y')
            
            fig.tight_layout()
            fig.savefig(plot_stats, dpi=150)
            plt.show()
            plt.close(fig)
            print(f"Saved {plot_stats}.")


# ============================================================================
# 真值数据加载函数
# ============================================================================

# 注释：原 NAV-PVT 加载函数（保留以备将来使用）
# def load_truth_llh_from_nav_pvt(path):
#     """
#     读取 UBX NAV-PVT：
#       - 支持逗号或空白分隔、有/无表头
#       - 列优先使用: lon, lat, height（无 height 时回退 hMSL）
#       - 自动把 1e-7 度 → 度；毫米 → 米
#     返回: (N,3)  [lon_deg, lat_deg, h_m]
#     """
#     # 先判定分隔符
#     with open(path, 'r', encoding='utf-8', errors='ignore') as f:
#         head_line = ""
#         for _ in range(100):
#             line = f.readline()
#             if not line:
#                 break
#             if line.strip() == "" or line.lstrip().startswith("#"):
#                 continue
#             head_line = line
#             break
#     delim = ',' if (',' in head_line) else None  # None=按空白分隔

#     # 尝试带表头读取
#     try:
#         data = np.genfromtxt(path, delimiter=delim, names=True, dtype=None, encoding=None)
#         names = set(data.dtype.names or [])
#         if 'lon' in names and 'lat' in names:
#             lon = np.asarray(data['lon'], float)
#             lat = np.asarray(data['lat'], float)
#             if   'height' in names: h = np.asarray(data['height'], float)
#             elif 'hMSL'   in names: h = np.asarray(data['hMSL'],   float)
#             else:                   h = np.zeros_like(lon)
#         else:
#             raise ValueError("no named columns lon/lat")
#     except Exception:
#         # 兜底：无表头，取前三列
#         arr = np.genfromtxt(path, delimiter=delim, ndmin=2)
#         lon, lat, h = arr[:, 0].astype(float), arr[:, 1].astype(float), arr[:, 2].astype(float)

#     # 单位自动识别与换算
#     # if np.nanmedian(np.abs(lon)) > 400.0: lon *= 1e-7  # 1e-7 度 → 度
#     # if np.nanmedian(np.abs(lat)) > 100.0: lat *= 1e-7
#     # 高度：若绝对值中位数 > 10000，视为毫米 → 米（例如 16737 → 16.737 m）
#     if np.nanmedian(np.abs(h)) > 1.0e4:   h   *= 1e-3

#     return np.column_stack([lon, lat, h])


def load_truth_from_nav_hpposecef(path):
    """
    读取 UBX NAV-HPPOSECEF.csv：
      - 支持逗号或空白分隔、有/无表头
      - 读取列: ecefX, ecefY, ecefZ (ECEF坐标，单位：厘米)
      - 将 ECEF 坐标转换为 LLH (经纬高)
    返回: (N,3)  [lon_deg, lat_deg, h_m]
    """
    # 判定分隔符
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
        
        # 检查是否有 ecefX, ecefY, ecefZ 列
        if 'ecefX' in names and 'ecefY' in names and 'ecefZ' in names:
            ecef_x = np.asarray(data['ecefX'], float)
            ecef_y = np.asarray(data['ecefY'], float)
            ecef_z = np.asarray(data['ecefZ'], float)
        else:
            raise ValueError("NAV-HPPOSECEF.csv must contain ecefX, ecefY, ecefZ columns")
    except Exception as e:
        # 兜底：无表头，假设前三列为 ecefX, ecefY, ecefZ
        print(f"[Warning] Failed to read with header: {e}. Trying without header...")
        arr = np.genfromtxt(path, delimiter=delim, ndmin=2)
        if arr.shape[1] < 3:
            raise ValueError("NAV-HPPOSECEF.csv must have at least 3 columns")
        ecef_x = arr[:, 0].astype(float)
        ecef_y = arr[:, 1].astype(float)
        ecef_z = arr[:, 2].astype(float)

    # 单位转换：NAV-HPPOSECEF 通常以厘米为单位，转换为米
    # 如果数值很大（>1e6），说明单位是厘米
    # if np.nanmedian(np.abs(ecef_x)) > 1e6:
    ecef_x *= 0.01  # cm -> m
    ecef_y *= 0.01
    ecef_z *= 0.01

    # 转换 ECEF 到 LLH
    n = len(ecef_x)
    llh_data = np.zeros((n, 3), dtype=float)
    
    for i in range(n):
        if np.isfinite(ecef_x[i]) and np.isfinite(ecef_y[i]) and np.isfinite(ecef_z[i]):
            lat, lon, h = ecef_to_lla(ecef_x[i], ecef_y[i], ecef_z[i])
            llh_data[i, 0] = lon  # 经度
            llh_data[i, 1] = lat  # 纬度
            llh_data[i, 2] = h    # 高度
        else:
            llh_data[i, :] = np.nan

    return llh_data


def print_summary_table(results, error_data, truth_llh):
    """
    打印格式化的统计摘要表格，类似于图像中的格式
    """
    # 统计有效历元数
    valid_results = [r for r in results if np.isfinite(r[4]) and np.isfinite(r[5]) and np.isfinite(r[6])]
    total_epochs = len(valid_results)
    
    print("\n" + "="*80)
    print("GNSS SPP POSITIONING RESULTS SUMMARY".center(80))
    print("="*80)
    print(f"\nTotal Epochs Processed: {total_epochs}")
    print("="*80)
    
    # 定位精度统计
    if error_data is not None and error_data['stats']:
        st = error_data['stats']
        print("\nPOSITIONING ACCURACY STATISTICS".center(80))
        print("="*80)
        print(f"{'Metric':<25} {'2D Error (m)':<20} {'3D Error (m)':<20}")
        print("-" * 80)
        print(f"{'Mean':<25} {st['2d_mean']:>18.3f}  {st['3d_mean']:>18.3f}")
        print(f"{'RMS':<25} {st['2d_rms']:>18.3f}  {st['3d_rms']:>18.3f}")
        print(f"{'Standard Deviation':<25} {st['2d_std']:>18.3f}  {st['3d_std']:>18.3f}")
        
        # 计算最大值、最小值和95百分位
        valid_2d = error_data['err_2d'][np.isfinite(error_data['err_2d'])]
        valid_3d = error_data['err_3d'][np.isfinite(error_data['err_3d'])]
        
        if len(valid_2d) > 0 and len(valid_3d) > 0:
            max_2d = np.max(valid_2d)
            min_2d = np.min(valid_2d)
            p95_2d = np.percentile(valid_2d, 95)
            
            max_3d = np.max(valid_3d)
            min_3d = np.min(valid_3d)
            p95_3d = np.percentile(valid_3d, 95)
            
            print(f"{'Maximum':<25} {max_2d:>18.3f}  {max_3d:>18.3f}")
            print(f"{'Minimum':<25} {min_2d:>18.3f}  {min_3d:>18.3f}")
            print(f"{'95th Percentile':<25} {p95_2d:>18.3f}  {p95_3d:>18.3f}")
    
    # ENU 分量统计
    if error_data is not None and error_data['stats']:
        st = error_data['stats']
        print("\n" + "="*80)
        print("ENU COMPONENT ACCURACY".center(80))
        print("="*80)
        print(f"{'Component':<15} {'Mean (m)':<15} {'RMS (m)':<15} {'Std Dev (m)':<15}")
        print("-" * 80)
        print(f"{'East':<15} {st['e_mean']:>13.3f}  {st['e_rms']:>13.3f}  {st['e_std']:>13.3f}")
        print(f"{'North':<15} {st['n_mean']:>13.3f}  {st['n_rms']:>13.3f}  {st['n_std']:>13.3f}")
        print(f"{'Up':<15} {st['u_mean']:>13.3f}  {st['u_rms']:>13.3f}  {st['u_std']:>13.3f}")
    
    # 坐标范围
    if len(valid_results) > 0:
        lats = np.array([r[4] for r in valid_results])
        lons = np.array([r[5] for r in valid_results])
        hs = np.array([r[6] for r in valid_results])
        
        print("\n" + "="*80)
        print("COORDINATE RANGES".center(80))
        print("="*80)
        print(f"Latitude Range:  {np.min(lats):.6f}°  to  {np.max(lats):.6f}°")
        print(f"Longitude Range: {np.min(lons):.6f}°  to  {np.max(lons):.6f}°")
        print(f"Altitude Range:  {np.min(hs):.3f} m  to  {np.max(hs):.3f} m")
    
    # DOP 统计
    pdops = [r[9] for r in valid_results if np.isfinite(r[9])]
    hdops = [r[10] for r in valid_results if np.isfinite(r[10])]
    vdops = [r[11] for r in valid_results if np.isfinite(r[11])]
    
    if pdops and hdops and vdops:
        print("\n" + "="*80)
        print("DOP STATISTICS".center(80))
        print("="*80)
        print(f"{'Metric':<15} {'PDOP':<15} {'HDOP':<15} {'VDOP':<15}")
        print("-" * 80)
        print(f"{'Mean':<15} {np.mean(pdops):>13.3f}  {np.mean(hdops):>13.3f}  {np.mean(vdops):>13.3f}")
        print(f"{'Median':<15} {np.median(pdops):>13.3f}  {np.median(hdops):>13.3f}  {np.median(vdops):>13.3f}")
        print(f"{'Minimum':<15} {np.min(pdops):>13.3f}  {np.min(hdops):>13.3f}  {np.min(vdops):>13.3f}")
        print(f"{'Maximum':<15} {np.max(pdops):>13.3f}  {np.max(hdops):>13.3f}  {np.max(vdops):>13.3f}")
    
    # 卫星数统计
    nsats = [int(r[8]) for r in valid_results if np.isfinite(r[8])]
    if nsats:
        print("\n" + "="*80)
        print("SATELLITE VISIBILITY".center(80))
        print("="*80)
        print(f"Average Satellites:  {np.mean(nsats):.1f}")
        print(f"Minimum Satellites:  {np.min(nsats)}")
        print(f"Maximum Satellites:  {np.max(nsats)}")
    
    print("="*80 + "\n")

def main():
    # 直接在此处指定输入/输出文件路径，替换为你的实际路径
    pseudoranges = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/pseudoranges_meas.csv"
    sat_clk      = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/satellite_clock_bias.csv"
    ion          = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/ionospheric_delay.csv"
    trop         = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/tropospheric_delay.csv"
    sat_pos      = "/Users/jay/Documents/Bachelor/aae4203/rinex_data/satellite_positions.csv"
    out_path     = "/Users/jay/Documents/Bachelor/aae4203/Result/lse_solution.csv"
    kml_out      = "/Users/jay/Documents/Bachelor/aae4203/Result/lse_solution.kml"  # 基于 CSV 路径生成 KML
    
    # 真值文件路径：使用 NAV-HPPOSECEF.csv（ECEF坐标）
    hpposecef_truth_path = "/Users/jay/Documents/Bachelor/aae4203/ubx-message/data/NAV-HPPOSECEF.csv"
    
    # 注释：原 NAV-PVT 真值路径（保留以备将来使用）
    # pvt_truth_path = "/Users/jay/Documents/Bachelor/aae4203/ubx-message/data/NAV-PVT.csv"

    # 可选初值（如果需要），否则置 None
    llh0 = None  # e.g. "22.304139 114.180131 -20.2" -> set below if needed

    # 迭代参数
    max_iter = 20
    tol = 1e-7

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
            P_corr, S_k, x0=x0, max_iter=max_iter, tol=tol, earth_rotation=True
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

    # # 读取真值：从 NAV-HPPOSECEF.csv (ECEF 数据)
    try:
        truth_llh = load_truth_from_nav_hpposecef(hpposecef_truth_path)
        print(f"[Truth HPPOSECEF] shape: {truth_llh.shape}  "
              f"lon[{np.nanmin(truth_llh[:,0]):.6f},{np.nanmax(truth_llh[:,0]):.6f}]  "
              f"lat[{np.nanmin(truth_llh[:,1]):.6f},{np.nanmax(truth_llh[:,1]):.6f}]  "
              f"h[{np.nanmin(truth_llh[:,2]):.3f},{np.nanmax(truth_llh[:,2]):.3f}] m")
    except Exception as e:
        print(f"[Truth HPPOSECEF] load failed: {e}")
        truth_llh = None

    # 注释：原 NAV-PVT 真值加载（保留以备将来使用）
    # try:
    #     truth_llh = load_truth_llh_from_nav_pvt(pvt_truth_path)
    #     print(f"[Truth PVT] shape: {truth_llh.shape}  "
    #           f"lon[{np.nanmin(truth_llh[:,0]):.6f},{np.nanmax(truth_llh[:,0]):.6f}]  "
    #           f"lat[{np.nanmin(truth_llh[:,1]):.6f},{np.nanmax(truth_llh[:,1]):.6f}]  "
    #           f"h[{np.nanmin(truth_llh[:,2]):.3f},{np.nanmax(truth_llh[:,2]):.3f}] m")
    # except Exception as e:
    #     print(f"[Truth PVT] load failed: {e}")
    #     truth_llh = None

    ls_coords = [(r[5], r[4], r[6]) for r in results  # (lon, lat, h)
                if np.isfinite(r[4]) and np.isfinite(r[5]) and np.isfinite(r[6])]
    truth_coords = None
    if truth_llh is not None:
        # truth_llh 形状 (N,3): [lon, lat, h]，直接转成 list[tuple]
        truth_coords = [(float(lon), float(lat), float(h)) for lon, lat, h in truth_llh]

    if ls_coords or truth_coords:
        save_kml_with_truth(ls_coords, truth_coords, kml_out,
                            name_ls="LS SPP Solution", name_truth="Truth (HPPOSECEF)")
        print(f"Saved {kml_out} with LS {len(ls_coords)} pts"
            f"{'' if truth_coords is None else f' and Truth {len(truth_coords)} pts'}.")

    # 绘图调用（修改为保存error_data）
    valid = [r for r in results if np.isfinite(r[4]) and np.isfinite(r[5]) and np.isfinite(r[6])]
    error_data = None
    
    if len(valid) >= 1 and truth_llh is not None and len(truth_llh) >= 1:
        lats = np.array([r[4] for r in valid])
        lons = np.array([r[5] for r in valid])
        hs = np.array([r[6] for r in valid])
        xyz = np.array([[r[1], r[2], r[3]] for r in valid])
        
        t_lon = np.asarray(truth_llh[:, 0], float)
        t_lat = np.asarray(truth_llh[:, 1], float)
        t_h = np.asarray(truth_llh[:, 2], float)
        t_xyz = np.array([lla_to_ecef(t_lat[i], t_lon[i], t_h[i]) for i in range(len(t_lon))])
        
        n = min(len(lons), len(t_lon))
        xyz = xyz[:n]
        t_xyz = t_xyz[:n]
        
        ref_lat, ref_lon = lats[0], lons[0]
        error_data = compute_error_statistics(xyz, t_xyz, ref_lat, ref_lon)
    
    generate_and_save_plots(results, out_path, truth_llh=truth_llh)

    # 打印统计表格
    print_summary_table(results, error_data, truth_llh)

    # 导出误差和DOP数据到CSV
    error_dop_csv = out_path.rsplit('.', 1)[0] + '_error_dop.csv'
    save_error_and_dop_statistics(results, error_data, error_dop_csv)

def save_error_and_dop_statistics(results, error_data, out_path):
    """
    将误差分析数据和DOP数据保存到独立的CSV文件
    
    参数:
        results: 定位结果列表
        error_data: 误差分析数据字典
        out_path: 输出CSV文件路径
    """
    valid_results = [r for r in results if np.isfinite(r[4]) and np.isfinite(r[5]) and np.isfinite(r[6])]
    
    if len(valid_results) == 0:
        print("[Warning] No valid results to export error/DOP statistics.")
        return
    
    # 准备数据
    epochs = [r[0] for r in valid_results]
    nsats = [r[8] for r in valid_results]
    pdops = [r[9] for r in valid_results]
    hdops = [r[10] for r in valid_results]
    vdops = [r[11] for r in valid_results]
    
    # 如果有误差数据，添加误差列
    if error_data is not None:
        # 对齐长度（误差数据可能比结果少）
        n = min(len(epochs), len(error_data['e_err']))
        
        data_dict = {
            'epoch': epochs[:n],
            'nsat': nsats[:n],
            'PDOP': pdops[:n],
            'HDOP': hdops[:n],
            'VDOP': vdops[:n],
            'east_error_m': error_data['e_err'][:n],
            'north_error_m': error_data['n_err'][:n],
            'up_error_m': error_data['u_err'][:n],
            '2d_error_m': error_data['err_2d'][:n],
            '3d_error_m': error_data['err_3d'][:n]
        }
    else:
        data_dict = {
            'epoch': epochs,
            'nsat': nsats,
            'PDOP': pdops,
            'HDOP': hdops,
            'VDOP': vdops
        }
    
    # 转换为numpy数组并保存
    df = pd.DataFrame(data_dict)
    df.to_csv(out_path, index=False, float_format='%.6f')
    
    print(f"Saved error and DOP statistics to {out_path}")
    
    # 如果有误差统计，保存统计摘要到另一个文件
    if error_data is not None and error_data['stats']:
        stats_path = out_path.rsplit('.', 1)[0] + '_summary.csv'
        
        stats_data = []
        st = error_data['stats']
        
        # 误差统计
        for component, prefix in [('East', 'e'), ('North', 'n'), ('Up', 'u'), 
                                   ('2D', '2d'), ('3D', '3d')]:
            stats_data.append({
                'Component': component,
                'Mean_m': st[f'{prefix}_mean'],
                'RMS_m': st[f'{prefix}_rms'],
                'Std_m': st[f'{prefix}_std']
            })
        
        # 添加最大值、最小值、95百分位
        valid_2d = error_data['err_2d'][np.isfinite(error_data['err_2d'])]
        valid_3d = error_data['err_3d'][np.isfinite(error_data['err_3d'])]
        
        if len(valid_2d) > 0 and len(valid_3d) > 0:
            stats_data.append({
                'Component': '2D_max',
                'Mean_m': np.max(valid_2d),
                'RMS_m': np.nan,
                'Std_m': np.nan
            })
            stats_data.append({
                'Component': '2D_min',
                'Mean_m': np.min(valid_2d),
                'RMS_m': np.nan,
                'Std_m': np.nan
            })
            stats_data.append({
                'Component': '2D_95percentile',
                'Mean_m': np.percentile(valid_2d, 95),
                'RMS_m': np.nan,
                'Std_m': np.nan
            })
            stats_data.append({
                'Component': '3D_max',
                'Mean_m': np.max(valid_3d),
                'RMS_m': np.nan,
                'Std_m': np.nan
            })
            stats_data.append({
                'Component': '3D_min',
                'Mean_m': np.min(valid_3d),
                'RMS_m': np.nan,
                'Std_m': np.nan
            })
            stats_data.append({
                'Component': '3D_95percentile',
                'Mean_m': np.percentile(valid_3d, 95),
                'RMS_m': np.nan,
                'Std_m': np.nan
            })
        
        # DOP统计
        pdops_valid = [p for p in pdops if np.isfinite(p)]
        hdops_valid = [h for h in hdops if np.isfinite(h)]
        vdops_valid = [v for v in vdops if np.isfinite(v)]
        
        if pdops_valid and hdops_valid and vdops_valid:
            stats_data.append({
                'Component': 'PDOP_mean',
                'Mean_m': np.mean(pdops_valid),
                'RMS_m': np.median(pdops_valid),
                'Std_m': np.std(pdops_valid)
            })
            stats_data.append({
                'Component': 'HDOP_mean',
                'Mean_m': np.mean(hdops_valid),
                'RMS_m': np.median(hdops_valid),
                'Std_m': np.std(hdops_valid)
            })
            stats_data.append({
                'Component': 'VDOP_mean',
                'Mean_m': np.mean(vdops_valid),
                'RMS_m': np.median(vdops_valid),
                'Std_m': np.std(vdops_valid)
            })
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_csv(stats_path, index=False, float_format='%.6f')
        print(f"Saved summary statistics to {stats_path}")

if __name__ == "__main__":
    main()
