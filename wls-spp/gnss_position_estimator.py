"""
WLS GNSS Positioning

Expected CSVs:
  Required:
    pseudoranges_meas.csv        [max_sats, num_epochs]
    satellite_clock_bias.csv     [max_sats, num_epochs] (meters or seconds; auto-detect)
    ionospheric_delay.csv        [max_sats, num_epochs] (meters, L1-equivalent)
    tropospheric_delay.csv       [max_sats, num_epochs] (meters)
    satellite_positions.csv      [max_sats, num_epochs*3] (meters; columns per epoch = [Ex,Ey,Ez])

Outputs:
  wls_ecef_clk.csv  -> [x(m), y(m), z(m), dtr(m)]
  wls_llh.csv       -> [lat(deg), lon(deg), h(m)]
  wls_iters_ok.csv  -> [iters, ok(1/0)]
  trajectory.kml    -> KML trajectory file
  trajectory_2d.png -> 2D geographic view
  trajectory_3d_enu.png -> 3D ENU frame trajectory

Author: ZHAO Jiaqi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import simplekml

# ---------------------------
# Config
# ---------------------------
FILE_PSEUDO = 'rinex_data/pseudoranges_meas.csv'
FILE_CLK    = 'rinex_data/satellite_clock_bias.csv'
FILE_IONO   = 'rinex_data/ionospheric_delay.csv'
FILE_TROPO  = 'rinex_data/tropospheric_delay.csv'
FILE_POS    = 'rinex_data/satellite_positions.csv'

MAX_ITERS   = 100
TOL         = 1e-3    # meters
MIN_SATS    = 10
USE_LAST_AS_INIT = True  # use previous epoch as init
C_MPS       = 299792458.0

# If you want to force units instead of auto-detection, set these:
CLK_UNITS = 'meters'  # None|'seconds'|'meters'

# ---------------------------
# Helpers
# ---------------------------

def load_csv_or_none(path):
    try:
        arr = np.genfromtxt(path, delimiter=',')
        if arr.ndim == 0:  # empty or not found
            return None
        return arr
    except Exception:
        return None

def load_csv_required(path, shape_hint=None):
    arr = np.genfromtxt(path, delimiter=',')
    if arr.ndim != 2:
        raise RuntimeError(f"{path} must be a 2D matrix (got ndim={arr.ndim}).")
    if shape_hint is not None and arr.shape != shape_hint:
        raise RuntimeError(f"{path} has shape {arr.shape}, expected {shape_hint}")
    return arr

def ecef_to_lla(ecef):
    x, y, z = ecef
    a  = 6378137.0
    f  = 1/298.257223563
    b  = a*(1 - f)
    e2 = 1 - (b*b)/(a*a)

    lon = np.arctan2(y, x)
    r   = np.hypot(x, y)

    E2  = a*a - b*b
    F   = 54.0*b*b*z*z
    G   = r*r + (1 - e2)*z*z - e2*E2
    c   = (e2*e2*F*r*r)/(G*G*G)
    s   = np.cbrt(1 + c + np.sqrt(c*c + 2*c))
    P   = F/(3*(s + 1/s + 1)**2 * G*G)
    Q   = np.sqrt(1 + 2*e2*e2*P)
    r0  = -(P*e2*r)/(1+Q) + np.sqrt(0.5*a*a*(1+1.0/Q) - P*(1-e2)*z*z/(Q*(1+Q)) - 0.5*P*r*r)
    U   = np.sqrt((r - e2*r0)**2 + z*z)
    V   = np.sqrt((r - e2*r0)**2 + (1 - e2)*z*z)
    Z0  = (b*b*z)/(a*V)
    h   = U*(1 - (b*b)/(a*V))
    lat = np.arctan2(z + (e2*Z0), r)
    return np.degrees(lat), np.degrees(lon), h

def get_epoch_block(mat_pos, epoch_idx):
    start = 3*epoch_idx
    return mat_pos[:, start:start+3]


def build_weights(n):
    # equal weights placeholder; replace if you add elevation/SNR
    return np.ones(n, dtype=float)

def solve_epoch_wls(P, dts_m, ion, trp, sat_xyz, x0=None):
    """
    P, dts_m, ion, trp: [Ns]
    sat_xyz: [Ns,3]
    Returns x_hat[4], n_it, ok
    """
    # corrected equivalent geometric range
    y = P + dts_m - ion - trp

    good = np.isfinite(y) & np.all(np.isfinite(sat_xyz), axis=1)
    if good.sum() < MIN_SATS:
        return None, 0, False

    yv = y[good]
    sv = sat_xyz[good, :]

    x = np.array([0.0, 0.0, 0.0, 0.0]) if x0 is None else x0.copy()
    W = build_weights(len(yv))

    for it in range(1, MAX_ITERS+1):
        rx, ry, rz, dtr = x
        rcvr = np.array([rx, ry, rz])
        diff = sv - rcvr[None, :]
        rho  = np.linalg.norm(diff, axis=1)
        los  = diff / rho[:, None]
        H    = np.hstack([-los, np.ones((los.shape[0], 1))])
        res  = yv - rho - dtr

        sqrtW = np.sqrt(W)[:, None]
        H_w = H * sqrtW
        r_w = res * sqrtW[:, 0]

        try:
            dx, *_ = np.linalg.lstsq(H_w, r_w, rcond=None)
        except np.linalg.LinAlgError:
            return None, it, False

        x += dx
        if np.linalg.norm(dx[:3]) < TOL and abs(dx[3]) < TOL:
                # if it > 5:
                    return x, it, True

    return x, MAX_ITERS, False

def ecef_to_enu(ecef, ref_lla):
    """
    Convert ECEF coordinates to ENU (East-North-Up) frame.
    
    Args:
        ecef: [N, 3] array of ECEF coordinates (x, y, z) in meters
        ref_lla: [3] reference point (lat_deg, lon_deg, h_m)
    
    Returns:
        [N, 3] array of ENU coordinates (east, north, up) in meters
    """
    lat0, lon0, h0 = np.radians(ref_lla[0]), np.radians(ref_lla[1]), ref_lla[2]
    
    # Convert reference LLA to ECEF
    a = 6378137.0
    f = 1/298.257223563
    b = a * (1 - f)
    e2 = 1 - (b*b)/(a*a)
    
    N0 = a / np.sqrt(1 - e2 * np.sin(lat0)**2)
    x0 = (N0 + h0) * np.cos(lat0) * np.cos(lon0)
    y0 = (N0 + h0) * np.cos(lat0) * np.sin(lon0)
    z0 = (N0 * (1 - e2) + h0) * np.sin(lat0)
    ref_ecef = np.array([x0, y0, z0])
    
    # Compute delta ECEF
    dx = ecef[:, 0] - ref_ecef[0]
    dy = ecef[:, 1] - ref_ecef[1]
    dz = ecef[:, 2] - ref_ecef[2]
    
    # Rotation matrix ECEF to ENU
    sin_lat, cos_lat = np.sin(lat0), np.cos(lat0)
    sin_lon, cos_lon = np.sin(lon0), np.cos(lon0)
    
    east  = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up    =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return np.column_stack([east, north, up])

def generate_kml(llh_data, output_file="trajectory.kml"):

    # Filter valid data
    valid = np.all(np.isfinite(llh_data), axis=1)
    if valid.sum() == 0:
        print("No valid data for KML generation.")
        return
    
    llh_valid = llh_data[valid]
    
    kml = simplekml.Kml()
    
    # Add line string for trajectory
    linestring = kml.newlinestring(name="GNSS Trajectory")
    coords = [(lon, lat, h) for lat, lon, h in llh_valid]
    linestring.coords = coords
    linestring.altitudemode = simplekml.AltitudeMode.absolute
    linestring.style.linestyle.color = simplekml.Color.red
    linestring.style.linestyle.width = 3
    
    # Add start point
    start_point = kml.newpoint(name="Start", coords=[coords[0]])
    start_point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/grn-circle.png'
    
    # Add end point
    end_point = kml.newpoint(name="End", coords=[coords[-1]])
    end_point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-circle.png'
    
    kml.save(output_file)
    print(f"KML file saved: {output_file}")

def plot_2d_geographic(llh_data, output_file="trajectory_2d.png"):
    """
    Plot 2D geographic view of trajectory.
    
    Args:
        llh_data: [N, 3] array of (lat_deg, lon_deg, h_m)
        output_file: output PNG filename
    """
    valid = np.all(np.isfinite(llh_data), axis=1)
    if valid.sum() == 0:
        print("No valid data for 2D plot.")
        return
    
    llh_valid = llh_data[valid]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Lat-Lon plot
    ax1.plot(llh_valid[:, 1], llh_valid[:, 0], 'b-', linewidth=1.5, label='Trajectory')
    ax1.plot(llh_valid[0, 1], llh_valid[0, 0], 'go', markersize=10, label='Start')
    ax1.plot(llh_valid[-1, 1], llh_valid[-1, 0], 'ro', markersize=10, label='End')
    ax1.set_xlabel('Longitude (deg)', fontsize=12)
    ax1.set_ylabel('Latitude (deg)', fontsize=12)
    ax1.set_title('2D Geographic Trajectory', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # Height profile
    epochs = np.arange(len(llh_valid))
    ax2.plot(epochs, llh_valid[:, 2], 'b-', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title('Height Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"2D plot saved: {output_file}")
    plt.close()

def plot_3d_enu(ecef_data, llh_data, output_file="trajectory_3d_enu.png"):
    """
    Plot 3D trajectory in ENU frame.
    
    Args:
        ecef_data: [N, 3] array of ECEF coordinates (x, y, z)
        llh_data: [N, 3] array of (lat_deg, lon_deg, h_m)
        output_file: output PNG filename
    """
    valid = np.all(np.isfinite(ecef_data), axis=1) & np.all(np.isfinite(llh_data), axis=1)
    if valid.sum() == 0:
        print("No valid data for 3D ENU plot.")
        return
    
    ecef_valid = ecef_data[valid]
    llh_valid = llh_data[valid]
    
    # Use first valid point as reference
    ref_lla = llh_valid[0]
    
    # Convert to ENU
    enu = ecef_to_enu(ecef_valid, ref_lla)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(enu[:, 0], enu[:, 1], enu[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(enu[0, 0], enu[0, 1], enu[0, 2], c='g', s=100, marker='o', label='Start')
    ax.scatter(enu[-1, 0], enu[-1, 1], enu[-1, 2], c='r', s=100, marker='o', label='End')
    
    ax.set_xlabel('East (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('North (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Up (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'3D ENU Trajectory\nReference: ({ref_lla[0]:.6f}°, {ref_lla[1]:.6f}°, {ref_lla[2]:.2f}m)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"3D ENU plot saved: {output_file}")
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    # Load required
    P_mat   = load_csv_required(FILE_PSEUDO)
    dts_raw = load_csv_required(FILE_CLK)
    ion_mat = load_csv_required(FILE_IONO, shape_hint=P_mat.shape)
    trp_mat = load_csv_required(FILE_TROPO, shape_hint=P_mat.shape)
    pos_mat = np.genfromtxt(FILE_POS, delimiter=',')
    if pos_mat.shape[0] != P_mat.shape[0] or pos_mat.shape[1] != P_mat.shape[1]*3:
        raise RuntimeError(f"{FILE_POS} has shape {pos_mat.shape}, expected {(P_mat.shape[0], P_mat.shape[1]*3)}")

    max_sats, num_epochs = P_mat.shape

    # Outputs
    X_est   = np.full((num_epochs, 4), np.nan)
    LLH_est = np.full((num_epochs, 3), np.nan)
    iters   = np.zeros(num_epochs, dtype=int)
    flags   = np.zeros(num_epochs, dtype=bool)

    x_prev = None

    for e in range(num_epochs):
        # Build epoch data robustly
        Sxyz = get_epoch_block(pos_mat, e)  # [max_sats,3]
        Pcol = P_mat[:, e]
        Dcol = dts_raw[:, e]
        Icol = ion_mat[:, e]
        Tcol = trp_mat[:, e]


        # No map: rely on all fields finite (old behavior)
        good = np.isfinite(Pcol) & np.all(np.isfinite(Sxyz), axis=1) & \
        np.isfinite(Dcol) & np.isfinite(Icol) & np.isfinite(Tcol)

        if good.sum() < MIN_SATS:
            iters[e] = 0
            flags[e] = False
            print(f"[Epoch {e+1:4d}/{num_epochs}] insufficient valid sats: {good.sum()}")
            continue

        P_use    = Pcol[good]
        D_use    = Dcol[good]
        I_use    = Icol[good]
        T_use    = Tcol[good]
        S_use    = Sxyz[good, :]

        x0 = x_prev if (USE_LAST_AS_INIT and x_prev is not None) else None
        x_hat, n_it, ok = solve_epoch_wls(P_use, D_use, I_use, T_use, S_use, x0=x0)

        iters[e] = n_it
        flags[e] = ok
        if ok:
            X_est[e, :] = x_hat
            LLH_est[e, :] = ecef_to_lla(x_hat[:3])
            x_prev = x_hat
            print(f"[Epoch {e+1:4d}/{num_epochs}] iters={n_it:3d}  "
                  f"ECEF=({x_hat[0]:.3f},{x_hat[1]:.3f},{x_hat[2]:.3f})  dtr={x_hat[3]:.3f}")
        else:
            print(f"[Epoch {e+1:4d}/{num_epochs}] WLS failed (iters={n_it})")

    # Save results
    np.savetxt("wls_ecef_clk.csv", X_est, delimiter=',', header="x(m),y(m),z(m),dtr(m)", comments='')
    np.savetxt("wls_llh.csv", LLH_est, delimiter=',', header="lat(deg),lon(deg),h(m)", comments='')
    np.savetxt("wls_iters_ok.csv", np.vstack([iters, flags.astype(int)]).T, delimiter=',', header="iters,ok(1/0)", comments='')

    print("\nDone. Files written:\n  - wls_ecef_clk.csv\n  - wls_llh.csv\n  - wls_iters_ok.csv")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_kml(LLH_est, "trajectory.kml")
    plot_2d_geographic(LLH_est, "trajectory_2d.png")
    plot_3d_enu(X_est[:, :3], LLH_est, "trajectory_3d_enu.png")
    
    print("\nAll outputs completed!")
    print("  - trajectory.kml")
    print("  - trajectory_2d.png")
    print("  - trajectory_3d_enu.png\n")

if __name__ == "__main__":
    main()
