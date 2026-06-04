from dataclasses import dataclass
from typing import Optional, Tuple
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive windows
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
C_LIGHT   = 2.99792458e8     # m/s
K_BOLTZMANN = 1.380649e-23   # J/K

# ============================================================================
#  LINK BUDGET INPUTS
# ============================================================================
# Core link parameters
POWER_TX_DBW = 10.0 * np.log10(20)              # dBW, transmitter power (2 W default)
BITRATE_HZ = 500.0e6                            # Hz, data rate (500 Mbps)
FREQUENCY_HZ = 2.2e9                            # Hz, S-band default

# Antenna configuration
N_TX_ANTENNAS = 3                               # number of transmitter antennas
TX_ANTENNA = None                               # AntennaPattern instance (set below)
GAIN_TX_DBI = 0.0                               # dBi, transmit peak gain

# Fixed link budget parameters (engineering standard for sounding rockets)
T_SYSTEM_K = 290.0                              # K, system noise temperature (fixed)
L_LINE_DB = 1.0                                 # dB, transmitter line/feed loss (fixed)
L_A_ZENITH_DB = 0.04                            # dB, zenith atmospheric loss S-band (fixed)
L_POL_DB = 0.0                                  # dB, polarization mismatch
EL_MIN_DEG = 2.0                                # deg, elevation angle clamp near horizon


# ============================================================================
#  3D ROCKET VISUALIZATION
# ============================================================================
def _rocket_local_frame(phi, psi, theta=0.0):
    """
    Build local frame (rotation matrix) from attitude angles.
    Returns 3x3 rotation matrix from body frame to ENU.
    phi: pitch (angle from vertical), psi: yaw, theta: roll about spin axis
    """
    u_z = np.array([
        np.sin(psi) * np.sin(phi),
        np.cos(psi) * np.sin(phi),
        np.cos(phi),
    ])

    if abs(u_z[0]) < 0.9:
        u_x_temp = np.array([1.0, 0.0, 0.0])
    else:
        u_x_temp = np.array([0.0, 1.0, 0.0])

    u_y = np.cross(u_z, u_x_temp)
    u_y = u_y / (np.linalg.norm(u_y) + 1e-10)
    u_x = np.cross(u_y, u_z)

    u_x_rolled = np.cos(theta) * u_x + np.sin(theta) * u_y
    u_y_rolled = -np.sin(theta) * u_x + np.cos(theta) * u_y

    return np.stack([u_x_rolled, u_y_rolled, u_z], axis=1)


def _create_rocket_mesh():
    """Create detailed rocket mesh (cylinder + cone + 4 fins) in local coordinates (Z up along nose)."""
    cyl_radius = 0.25
    cyl_height = 4.0
    cone_height = 1.5
    cone_tip_z = cyl_height + cone_height

    # Cylinder surface
    theta = np.linspace(0, 2*np.pi, 20)
    z_cyl = np.linspace(0, cyl_height, 10)
    Theta_cyl, Z_cyl = np.meshgrid(theta, z_cyl)
    X_cyl = cyl_radius * np.cos(Theta_cyl)
    Y_cyl = cyl_radius * np.sin(Theta_cyl)

    # Cone surface
    z_cone = np.linspace(cyl_height, cone_tip_z, 10)
    Theta_cone, Z_cone = np.meshgrid(theta, z_cone)
    cone_progress = (Z_cone - cyl_height) / cone_height
    X_cone = cyl_radius * (1 - cone_progress) * np.cos(Theta_cone)
    Y_cone = cyl_radius * (1 - cone_progress) * np.sin(Theta_cone)

    # Four fins at base
    fin_height = 1.2
    fin_width = 0.6
    fin_z_start = 0.5
    fin_z_end = fin_z_start + fin_height
    fin_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    fins = []

    for angle in fin_angles:
        x_base = cyl_radius * np.cos(angle)
        y_base = cyl_radius * np.sin(angle)
        fin_radius = cyl_radius + fin_width
        x_tip = fin_radius * np.cos(angle)
        y_tip = fin_radius * np.sin(angle)

        fin_t = np.linspace(0, 1, 5)
        fin_z = np.linspace(fin_z_start, fin_z_end, 5)
        Fin_t, Fin_z = np.meshgrid(fin_t, fin_z)

        Fin_x = x_base + Fin_t * (x_tip - x_base)
        Fin_y = y_base + Fin_t * (y_tip - y_base)

        fins.append((Fin_x, Fin_y, Fin_z))

    return {
        'cyl': (X_cyl, Y_cyl, Z_cyl),
        'cone': (X_cone, Y_cone, Z_cone),
        'tip': (0.0, 0.0, cone_tip_z),
        'fins': fins
    }


def _draw_rocket_3d(ax, phi, psi, theta=0.0, center=None):
    """Draw rocket in 3D axes at given attitude (phi, psi control pointing; theta is roll)."""
    if center is None:
        center = np.array([0.0, 0.0, 0.0])

    R = _rocket_local_frame(phi, psi, theta)
    mesh = _create_rocket_mesh()

    def rotate_and_translate(X, Y, Z):
        """Rotate mesh and translate to center."""
        result_X = np.zeros_like(X)
        result_Y = np.zeros_like(Y)
        result_Z = np.zeros_like(Z)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pt = R @ np.array([X[i, j], Y[i, j], Z[i, j]]) + center
                result_X[i, j] = pt[0]
                result_Y[i, j] = pt[1]
                result_Z[i, j] = pt[2]
        return result_X, result_Y, result_Z

    # Draw cylinder
    X_cyl, Y_cyl, Z_cyl = mesh['cyl']
    X_cyl_r, Y_cyl_r, Z_cyl_r = rotate_and_translate(X_cyl, Y_cyl, Z_cyl)
    ax.plot_surface(X_cyl_r, Y_cyl_r, Z_cyl_r, color='steelblue', alpha=0.75, edgecolor='none')

    # Draw cone
    X_cone, Y_cone, Z_cone = mesh['cone']
    X_cone_r, Y_cone_r, Z_cone_r = rotate_and_translate(X_cone, Y_cone, Z_cone)
    ax.plot_surface(X_cone_r, Y_cone_r, Z_cone_r, color='orangered', alpha=0.85, edgecolor='none')

    # Draw fins
    for Fin_x, Fin_y, Fin_z in mesh['fins']:
        Fin_x_r, Fin_y_r, Fin_z_r = rotate_and_translate(Fin_x, Fin_y, Fin_z)
        ax.plot_surface(Fin_x_r, Fin_y_r, Fin_z_r, color='darkslateblue', alpha=0.65, edgecolor='navy', linewidth=0.5)

    # Draw nose tip
    X_t, Y_t, Z_t = mesh['tip']
    tip_local = np.array([[X_t], [Y_t], [Z_t]])
    tip_enu = R @ tip_local + center[:, None]
    ax.scatter(tip_enu[0], tip_enu[1], tip_enu[2], c='red', s=60, alpha=1.0, edgecolor='darkred', linewidth=1)

    # Draw axes with angle labels
    axis_len = 1.5
    origin = center
    x_end = center + axis_len * R[:, 0]
    y_end = center + axis_len * R[:, 1]
    z_end = center + axis_len * R[:, 2]

    ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], 'r-', lw=2.5, alpha=0.9)
    ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], 'g-', lw=2.5, alpha=0.9)
    ax.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], 'b-', lw=2.5, alpha=0.9)

    ax.text(x_end[0], x_end[1], x_end[2], 'X', color='red', fontsize=9, weight='bold')
    ax.text(y_end[0], y_end[1], y_end[2], 'Y', color='green', fontsize=9, weight='bold')
    ax.text(z_end[0], z_end[1], z_end[2], 'Z', color='blue', fontsize=9, weight='bold')


# ============================================================================
#  ANTENNA PATTERN
# ============================================================================
class AntennaPattern:
    """
    Nose-mounted directional antenna described by two principal-plane cuts
    (Azimuth / H-plane and Elevation / E-plane) read off the datasheet.

    Gain values are absolute dBi.  The pattern is symmetric about boresight in
    each plane, so samples spanning 0..180 deg are folded to cover the full
    sphere.  `composite_gain_db` returns the linear (roll-)average of the two
    cuts at a given polar angle, which approximates the time-averaged gain a
    ground station sees while the rocket rolls about its spin axis.
    """

    def __init__(self, sample_angles_rad, azimuth_cut_db, elevation_cut_db):
        self.sample_angles_rad = np.asarray(sample_angles_rad, dtype=float)
        self.azimuth_cut_db = np.asarray(azimuth_cut_db, dtype=float)
        self.elevation_cut_db = np.asarray(elevation_cut_db, dtype=float)

    @classmethod
    def from_datasheet(cls):
        """Default pattern built from the digitised datasheet polar plots."""
        # 40 linearly spaced angles, 0-180 deg (symmetric about boresight).
        sample_angles_rad = np.linspace(0.0, np.pi, 40)
        azimuth_cut_db = np.array([
             8.8,  8.8,  9.0,  8.8,  8.8,  8.3,  8.2,  7.8,  7.3,  6.8,
             6.3,  5.9,  5.0,  4.8,  4.0,  3.2,  2.5,  1.7,  1.0,  0.5,
            -0.8, -1.8, -3.0, -4.5, -5.7, -8.2,-10.0,-11.8,-12.3,-12.3,
           -10.5, -8.8, -8.5, -7.7, -6.7, -6.0, -6.2, -6.2, -6.2, -6.2,
        ])
        elevation_cut_db = np.array([
             9.9, 10.1,  9.8,  9.5,  8.8,  8.5,  7.4,  6.7,  5.2,  4.5,
             3.5,  2.1,  0.4, -0.8, -2.9, -6.0, -9.9, -9.9, -9.9, -9.9,
            -9.9, -9.6, -9.6, -9.9, -9.5, -9.5, -9.5, -9.8, -9.9,-10.0,
           -10.0, -9.9,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0, -9.9, -9.9,
        ])
        return cls(sample_angles_rad, azimuth_cut_db, elevation_cut_db)

    @staticmethod
    def _fold_to_half(angle_rad):
        """Map any angle to [0, pi] using even symmetry about 0 and pi."""
        a = np.mod(angle_rad, 2.0 * np.pi)
        return np.where(a > np.pi, 2.0 * np.pi - a, a)

    def _cut_db(self, angle_rad, cut_db):
        a = self._fold_to_half(angle_rad)
        return np.interp(a, self.sample_angles_rad, cut_db)

    def azimuth_gain_db(self, angle_rad):
        """H-plane absolute gain (dBi) vs angle from boresight."""
        return self._cut_db(angle_rad, self.azimuth_cut_db)

    def elevation_gain_db(self, angle_rad):
        """E-plane absolute gain (dBi) vs angle from boresight."""
        return self._cut_db(angle_rad, self.elevation_cut_db)

    def composite_gain_db(self, theta_off_rad):
        """
        Roll-averaged absolute gain (dBi) vs off-boresight angle: the linear
        mean of the E- and H-plane cuts.  Drop-in for LinkBudget's tx_pattern.
        """
        g_az_lin = 10.0 ** (self.azimuth_gain_db(theta_off_rad) / 10.0)
        g_el_lin = 10.0 ** (self.elevation_gain_db(theta_off_rad) / 10.0)
        return 10.0 * np.log10(0.5 * (g_az_lin + g_el_lin))

    def plot(self, show=True):
        """Render the two datasheet cuts and the roll-averaged composite."""
        fig, axes = plt.subplots(
            1, 3, subplot_kw={"projection": "polar"}, figsize=(15, 5)
        )
        phi = np.linspace(0.0, 2.0 * np.pi, 721)

        for ax in axes:
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.arange(0, 360, 30))

        axes[0].plot(phi, self.azimuth_gain_db(phi), color="crimson", lw=2)
        axes[0].set_title("Azimuth Pattern")
        axes[1].plot(phi, self.elevation_gain_db(phi), color="crimson", lw=2)
        axes[1].set_title("Elevation Pattern")
        axes[2].plot(phi, self.composite_gain_db(phi), color="navy", lw=2)
        axes[2].set_title("Roll-averaged composite\n(used by LinkBudget)")

        for ax in axes:
            ax.set_rlim(-25, 15)
            ax.set_rticks([-20, -10, 0, 10])

        fig.suptitle("Nose-mounted antenna pattern model", y=1.02)
        fig.tight_layout()
        if show:
            plt.show()
        return fig


# ============================================================================
#  TRAJECTORY LOADING
# ============================================================================
def load_trajectory_from_csv(csv_filepath: str = "trajectory.csv", roll_rate_deg_per_s: float = 30.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load trajectory data from CSV file generated by trajectoryCodev2.m.

    The CSV should contain columns: Time_s, Downrange_km, Altitude_km, Mach, etc.

    Args:
        csv_filepath: Path to the trajectory CSV file
        roll_rate_deg_per_s: Roll rate in degrees per second (default 30 deg/s)

    Returns:
        trajectory: 6xN array [x, y, z, theta, phi, psi]
                    x = downrange distance (m)
                    y = lateral distance (m, set to 0)
                    z = altitude (m)
                    theta = roll angle (rad, calculated from roll rate)
                    phi = angle of attack (rad, set to 0)
                    psi = sideslip angle (rad, set to 0)
        time_array: Time in seconds (1D array, length N)
        mach_array: Mach number (1D array, length N)

    Raises:
        FileNotFoundError: If the CSV file does not exist
    """
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"Trajectory file '{csv_filepath}' not found. "
                              f"Please run trajectoryCodev2.m first to generate it.")

    df = pd.read_csv(csv_filepath)

    # Extract required columns
    time = df['Time_s'].values
    downrange_km = df['Downrange_km'].values
    altitude_km = df['Altitude_km'].values
    mach = df['Mach'].values

    n = len(time)

    # Build trajectory array: 6xN with [x, y, z, theta, phi, psi]
    x = downrange_km * 1000.0          # Convert to meters
    y = np.zeros(n)                    # No lateral motion (assumed straight downrange)
    z = altitude_km * 1000.0           # Convert to meters
    roll_rate_rad_per_s = np.radians(roll_rate_deg_per_s)
    theta = roll_rate_rad_per_s * time # Roll angle from constant roll rate
    phi = np.zeros(n)                  # Angle of attack (set to zero)
    psi = np.zeros(n)                  # Sideslip angle (set to zero)

    x = x + 9000
    trajectory = np.array([x, y, z, theta, phi, psi], dtype=float)

    return trajectory, time, mach


# ============================================================================
#  LINK BUDGET RESULT
# ============================================================================
@dataclass
class LinkResult:
    """Per-timestep link-budget outputs (each a length-N ndarray)."""
    EbN0: np.ndarray  # dB
    EIRP: np.ndarray  # dBW
    GT: np.ndarray    # dB/K
    CN0: np.ndarray   # dB-Hz


# ============================================================================
#  LINK BUDGET (SIMPLIFIED)
# ============================================================================
class LinkBudget:
    """
    Per-timestep link budget for a sounding-rocket downlink (S-band default).

    Frame: ENU with the receiver at the origin. Units: P in dBW, G_t / G_r in dBi.
    Uses industry-standard fixed values for atmospheric and system parameters.
    """

    R_EARTH = 6371e3              # m, mean Earth radius (horizon model)

    def __init__(
        self,
        P,                         # dBW, transmitter power
        G_r,                       # dBi, receiver gain
        R,                         # Hz, bit rate
        f=2.2e9,                   # Hz, frequency (S-band default)
        *,
        n_tx_antennas=1,           # number of transmitter antennas
        tx_antenna=None,           # AntennaPattern; None = flat gain at G_t
        G_t=0.0,                   # dBi, transmit peak/flat gain
        T_s=290.0,                 # K, system noise temperature (fixed)
        L_l=1.0,                   # dB, transmitter line/feed loss (fixed)
        L_a_zenith=0.04,           # dB, zenith atmospheric loss (fixed)
        L_pol=0.0,                 # dB, polarization mismatch
        el_min_deg=2.0,            # deg, elevation angle clamp
    ):
        self.P = P
        self.G_r = G_r
        self.R = R
        self.f = f
        self.n_tx_antennas = n_tx_antennas
        self.tx_antenna = tx_antenna
        self.G_t = G_t
        self.T_s = T_s
        self.L_l = L_l
        self.L_a_zenith = L_a_zenith
        self.L_pol = L_pol
        self.el_min_deg = el_min_deg

    @staticmethod
    def trajectory_vel_to_enu(trajectory):
        """
        Convert trajectory from velocity-frame angles to ENU-frame angles.

        Input trajectory [x, y, z, theta, phi_vel, psi_vel]:
          phi_vel, psi_vel = angle of attack and sideslip (local alphas w.r.t. velocity)

        Output trajectory [x, y, z, theta, phi_enu, psi_enu]:
          phi_enu = pitch angle from vertical, psi_enu = yaw angle from north (ENU frame)
        """
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim != 2 or 6 not in traj.shape:
            raise ValueError("trajectory must be shape (6, N) or (N, 6)")
        if traj.shape[0] != 6:
            traj = traj.T

        x, y, z, theta, phi_vel, psi_vel = traj
        n = len(x)

        # Compute velocity direction from position gradient (vectorized)
        pos = np.stack([x, y, z], axis=0)
        vel = np.zeros_like(pos)
        vel[:, 1:-1] = (pos[:, 2:] - pos[:, :-2]) / 2.0
        vel[:, 0] = pos[:, 1] - pos[:, 0]
        vel[:, -1] = pos[:, -1] - pos[:, -2]

        vel_norm = np.linalg.norm(vel, axis=0, keepdims=True)
        u_vel = vel / (vel_norm + 1e-10)  # (3, N)

        # Determine reference direction (vertical or East based on velocity direction)
        # Default to vertical [0, 0, 1]
        ref = np.broadcast_to(np.array([0.0, 0.0, 1.0])[:, None], (3, n)).copy()
        # Switch to East [1, 0, 0] where flying nearly vertically
        near_vertical = np.abs(u_vel[2, :]) > 0.999
        ref[:, near_vertical] = np.array([1.0, 0.0, 0.0])[:, None]

        # E-pitch: vertical projected perpendicular to velocity (vectorized)
        # e_pitch = ref - (ref · u_vel) * u_vel
        ref_dot_uvel = np.sum(ref * u_vel, axis=0)
        e_pitch = ref - ref_dot_uvel[None, :] * u_vel
        e_pitch_norm = np.linalg.norm(e_pitch, axis=0)
        e_pitch = e_pitch / (e_pitch_norm[None, :] + 1e-10)

        # E-yaw: u_vel × e_pitch (vectorized)
        e_yaw = np.cross(u_vel, e_pitch, axis=0)

        # Nose direction in ENU: velocity + AoA/sideslip tilt
        # u_nose = cos(phi) * u_vel + sin(phi) * (cos(psi) * e_pitch + sin(psi) * e_yaw)
        cos_phi = np.cos(phi_vel)
        sin_phi = np.sin(phi_vel)
        cos_psi = np.cos(psi_vel)
        sin_psi = np.sin(psi_vel)

        u_nose_enu = (cos_phi[None, :] * u_vel
                     + sin_phi[None, :] * (cos_psi[None, :] * e_pitch + sin_psi[None, :] * e_yaw))

        # Extract ENU angles from nose direction (vectorized)
        phi_enu = np.arccos(np.clip(u_nose_enu[2, :], -1.0, 1.0))
        psi_enu = np.arctan2(u_nose_enu[1, :], u_nose_enu[0, :])

        return np.array([x, y, z, theta, phi_enu, psi_enu], dtype=float)

    @staticmethod
    def _spin_axis_enu(theta, phi, psi):
        """
        Rocket spin-axis unit vector(s) in ENU, given the 6-DoF attitude angles.

        Convention:
            - Spin axis = body-Z (rocket nose direction).
            - At zero attitude (phi = psi = 0) the spin axis is +ENU-Z (up).
            - phi = pitch (tips the axis from vertical, +X direction at psi=0).
            - psi = yaw (rotates the pitch plane about vertical).
            - theta = roll about the spin axis, so it does not affect pointing.

        Returns shape (3, N) where columns are unit vectors in ENU.
        """
        del theta  # roll about the spin axis leaves the axis fixed
        return np.stack([
            np.cos(psi) * np.sin(phi),
            np.sin(psi) * np.sin(phi),
            np.cos(phi),
        ], axis=0)

    @staticmethod
    def parabolic_dish_gain_db(diameter_m, efficiency, f):
        """Peak gain (dBi) of a parabolic dish from diameter, efficiency, freq."""
        wavelength = 299792458.0 / f
        return 10.0 * np.log10(efficiency * (np.pi * diameter_m / wavelength) ** 2)

    def compute(self, trajectory):
        """
        Evaluate the link budget along a trajectory.

        Trajectory: 6xN (or Nx6) array of [x, y, z, theta, phi, psi].
        Position in meters, angles in radians.
        """
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim != 2 or 6 not in traj.shape:
            raise ValueError("trajectory must be shape (6, N) or (N, 6)")
        if traj.shape[0] != 6:
            traj = traj.T

        # Convert from velocity-frame angles to ENU-frame angles
        traj = self.trajectory_vel_to_enu(traj)

        x, y, z, theta, phi, psi = traj

        # Geometry
        d = np.sqrt(x**2 + y**2 + z**2)              # slant range, m
        r_g = np.sqrt(x**2 + y**2)
        el = np.arctan2(z, r_g)                       # rad, elevation at receiver
        u_r2r = -np.stack([x, y, z], axis=0) / d      # unit vector rocket -> receiver

        # Off-axis angle between rocket spin axis and the LOS to the receiver
        u_spin = self._spin_axis_enu(theta, phi, psi)
        cos_off = np.clip(np.sum(u_spin * u_r2r, axis=0), -1.0, 1.0)
        theta_off = np.arccos(cos_off)                # rad

        # Effective gains
        G_t_eff = self.G_t
        if self.tx_antenna is not None:
            G_t_eff = self.G_t + self.tx_antenna.composite_gain_db(theta_off)
        G_r_eff = self.G_r

        # Free-space path loss (d in m, f in Hz); 147.55 = 20*log10(c / 4pi)
        L_s = 20.0 * np.log10(d) + 20.0 * np.log10(self.f) - 147.55

        # Fixed atmospheric loss (zenith scaled by 1/sin(elevation))
        L_a = self.L_a_zenith / np.sin(np.maximum(el, np.deg2rad(self.el_min_deg)))

        # Link-budget assembly (all in dB)
        k_dB = 10.0 * np.log10(K_BOLTZMANN)          # ~ -228.6 dBW/K/Hz
        EIRP = self.P - self.L_l + G_t_eff            # dBW

        # Multiple transmit antennas: incoherent power addition
        if self.n_tx_antennas > 1:
            EIRP = EIRP + 10.0 * np.log10(self.n_tx_antennas)

        GT = G_r_eff - 10.0 * np.log10(self.T_s)     # dB/K
        CN0 = EIRP + GT - L_s - L_a - self.L_pol - k_dB  # dB-Hz
        EbN0 = CN0 - 10.0 * np.log10(self.R)         # dB

        # Broadcast scalars to length-N for a consistent return shape
        EIRP = np.broadcast_to(np.asarray(EIRP, dtype=float), d.shape).copy()
        GT = np.broadcast_to(np.asarray(GT, dtype=float), d.shape).copy()

        return LinkResult(EbN0=EbN0, EIRP=EIRP, GT=GT, CN0=CN0)

    def _horizon_altitude(self, downrange_m, k_factor=1.0):
        """
        Altitude (m) of the ground station's visible horizon at ground arc
        distance `downrange_m`:  h = R/cos(d/R) - R, with R = k_factor * R_EARTH.
        k_factor = 4/3 gives the refraction-corrected radio horizon.
        """
        R = k_factor * self.R_EARTH
        return R / np.cos(downrange_m / R) - R

    def plot_trajectory(self, trajectory, result=None, show=True,
                       time_array: Optional[np.ndarray] = None,
                       mach_array: Optional[np.ndarray] = None):
        """
        Interactive side-view (downrange vs altitude) of the flight.

        The trajectory is colour-coded by Eb/N0 (red = worst, green = best); a
        slider scrubs along the flight and the right-hand panel reads out the
        link metrics at the selected point.  Two buttons jump to the apogee and
        the worst-link point, and two horizon curves (geometric + 4/3 radio)
        show when the rocket clears the ground station's line of sight.

        The rocket visualization shows the attitude in ENU (earth-centered) frame.
        Input trajectory phi/psi are interpreted as local angles of attack/sideslip.

        Args:
            trajectory: 6xN trajectory array
            result: LinkResult from compute() (auto-computed if None)
            show: Whether to display the plot
            time_array: Optional time values (length N) for display
            mach_array: Optional Mach number values (length N) for display

        Needs a GUI matplotlib backend (TkAgg/QtAgg) for the slider/buttons.
        Requires at least 2 trajectory samples.
        """
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim != 2 or 6 not in traj.shape:
            raise ValueError("trajectory must be shape (6, N) or (N, 6)")
        if traj.shape[0] != 6:
            traj = traj.T

        # Convert from velocity-frame angles (local alphas) to ENU-frame angles
        traj = self.trajectory_vel_to_enu(traj)

        x, y, z, theta, phi, psi = traj
        n = x.size
        if n < 2:
            raise ValueError("plot_trajectory needs at least 2 trajectory samples")

        if result is None:
            result = self.compute(traj)

        # Plot geometry (mirrors compute()).
        xyz = np.stack([x, y, z], axis=0)
        slant = np.sqrt(x**2 + y**2 + z**2)            # m
        downrange = np.sqrt(x**2 + y**2)               # m, horizontal range
        altitude = z                                   # m
        elevation = np.degrees(np.arctan2(z, downrange))
        u_r2r = -xyz / slant
        u_spin = self._spin_axis_enu(theta, phi, psi)
        u_spin = np.broadcast_to(u_spin, xyz.shape)
        cos_off = np.clip(np.sum(u_spin * u_r2r, axis=0), -1.0, 1.0)
        theta_off = np.degrees(np.arccos(cos_off))

        dr_km = downrange / 1e3
        alt_km = altitude / 1e3

        apogee_idx = int(np.argmax(altitude))
        worst_idx = int(np.argmin(result.EbN0))

        # ---- Figure layout -------------------------------------------------
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_axes([0.07, 0.35, 0.58, 0.58])      # trajectory
        ax_txt = fig.add_axes([0.70, 0.58, 0.28, 0.35])  # metrics panel
        ax_txt.axis("off")
        ax_3d = fig.add_axes([0.70, 0.27, 0.28, 0.28], projection='3d')  # 3D rocket view
        ax_slider = fig.add_axes([0.10, 0.18, 0.55, 0.03])
        ax_worst = fig.add_axes([0.10, 0.03, 0.22, 0.055])
        ax_apogee = fig.add_axes([0.43, 0.03, 0.22, 0.055])

        # ---- Eb/N0-coloured trajectory
        max_line_pts = 2000
        if n > max_line_pts:
            disp_idx = np.unique(np.concatenate([
                np.linspace(0, n - 1, max_line_pts).astype(int),
                [apogee_idx, worst_idx],
            ]))
        else:
            disp_idx = np.arange(n)
        dr_disp, alt_disp = dr_km[disp_idx], alt_km[disp_idx]
        ebn0_disp = result.EbN0[disp_idx]
        points = np.array([dr_disp, alt_disp]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Calculate 4/3 radar horizon for each point
        horizon_alt_disp = self._horizon_altitude(dr_disp * 1e3, 4.0 / 3.0) / 1e3

        # Determine which segments are below the 4/3 horizon
        below_horizon = (alt_disp[:-1] < horizon_alt_disp[:-1]) & (alt_disp[1:] < horizon_alt_disp[1:])

        # Create colors: black for below horizon, Eb/N0-based for above
        norm = Normalize(vmin=float(np.min(result.EbN0)),
                         vmax=float(np.max(result.EbN0)))
        cmap = plt.cm.RdYlGn
        ebn0_colors = cmap(norm(0.5 * (ebn0_disp[:-1] + ebn0_disp[1:])))
        colors = np.array(ebn0_colors)
        colors[below_horizon] = [0, 0, 0, 1]  # Black for below horizon

        lc = LineCollection(segments, colors=colors, lw=2.5)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label="Eb/N0 (dB)", pad=0.01)

        # ---- Horizon curves
        d_max = max(float(downrange.max()), 1.0)
        d_line = np.linspace(0.0, d_max, 400)
        ax.plot(d_line / 1e3, self._horizon_altitude(d_line) / 1e3,
                "--", color="tab:blue", lw=1.3, label="Visible horizon")
        ax.plot(d_line / 1e3, self._horizon_altitude(d_line, 4.0 / 3.0) / 1e3,
                "--", color="tab:purple", lw=1.3, label="Radio horizon (4/3)")

        # ---- Apogee / worst annotations
        ax.annotate("apogee", (dr_km[apogee_idx], alt_km[apogee_idx]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9, color="black",
                    arrowprops=dict(arrowstyle="->", color="black"))
        ax.annotate("worst link", (dr_km[worst_idx], alt_km[worst_idx]),
                    textcoords="offset points", xytext=(0, -18), ha="center",
                    fontsize=9, color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred"))

        # ---- Movable marker
        marker, = ax.plot([dr_km[0]], [alt_km[0]], "o", ms=11,
                          mfc="white", mec="black", mew=2, zorder=5)

        ax.set_xlabel("Downrange (km)")
        ax.set_ylabel("Altitude (km)")
        ax.set_title("Trajectory coloured by Eb/N0")
        ax.margins(x=0.02, y=0.15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        panel = ax_txt.text(0.0, 1.0, "", va="top", ha="left",
                            family="monospace", fontsize=10,
                            transform=ax_txt.transAxes)

        def _panel_text(i):
            text = (
                f"Time {i + 1} / {n}\n"
                f"{'-' * 28}\n"
                f"Downrange   : {downrange[i] / 1e3:10.2f} km\n"
                f"Altitude    : {altitude[i] / 1e3:10.2f} km\n"
                f"Elevation   : {elevation[i]:10.2f} deg\n"
                f"Slant range : {slant[i] / 1e3:10.2f} km\n"
                f"Off-bore    : {theta_off[i]:10.2f} deg\n"
            )
            if time_array is not None:
                text += f"Time        : {time_array[i]:10.2f} s\n"
            if mach_array is not None:
                text += f"Mach        : {mach_array[i]:10.3f}\n"
            if time_array is not None or mach_array is not None:
                text += f"{'-' * 28}\n"
            text += (
                f"Theta (roll): {np.degrees(theta[i]):10.2f} deg\n"
                f"Phi (pitch) : {np.degrees(phi[i]):10.2f} deg\n"
                f"Psi (yaw)   : {np.degrees(psi[i]):10.2f} deg\n"
                f"{'-' * 28}\n"
                f"Eb/N0       : {result.EbN0[i]:10.2f} dB\n"
                f"EIRP        : {result.EIRP[i]:10.2f} dBW\n"
                f"C/N0        : {result.CN0[i]:10.2f} dB-Hz\n"
                f"G/T         : {result.GT[i]:10.2f} dB/K\n"
            )
            return text

        slider = Slider(ax_slider, "Time", 0, n - 1, valinit=0, valstep=1)
        btn_worst = Button(ax_worst, "Worst link time")
        btn_apogee = Button(ax_apogee, "Apogee")

        # ---- Fast updates via blitting
        slider.drawon = False
        animated = [marker, panel]
        animated += [a for a in (getattr(slider, "poly", None),
                                 getattr(slider, "_handle", None),
                                 getattr(slider, "valtext", None)) if a is not None]
        for a in animated:
            a.set_animated(True)

        _bg = {"region": None}

        def _draw_animated():
            for a in animated:
                a.axes.draw_artist(a)

        def _on_draw(_event):
            _bg["region"] = fig.canvas.copy_from_bbox(fig.bbox)
            _draw_animated()
            fig.canvas.blit(fig.bbox)

        def _update(val):
            i = int(val)
            marker.set_data([dr_km[i]], [alt_km[i]])
            panel.set_text(_panel_text(i))

            ax_3d.clear()
            _draw_rocket_3d(ax_3d, phi[i], psi[i], theta[i])

            phi_deg = np.degrees(phi[i])
            psi_deg = np.degrees(psi[i])

            ax_3d.set_xlabel(f'Pitch φ: {phi_deg:.1f}°', fontsize=9, weight='bold')
            ax_3d.set_ylabel(f'Yaw ψ: {psi_deg:.1f}°', fontsize=9, weight='bold')
            ax_3d.set_title('Rocket Attitude (Isometric)', fontsize=10, weight='bold')
            ax_3d.set_xlim([-3, 3])
            ax_3d.set_ylim([-3, 3])
            ax_3d.set_zlim([-1, 6])
            ax_3d.view_init(elev=30, azim=45)
            ax_3d.tick_params(labelsize=7)
            ax_3d.set_zticks([])

            fig.canvas.draw()

        slider.on_changed(_update)
        btn_worst.on_clicked(lambda _evt: slider.set_val(worst_idx))
        btn_apogee.on_clicked(lambda _evt: slider.set_val(apogee_idx))
        fig.canvas.mpl_connect("draw_event", _on_draw)

        # Initial marker / panel state
        marker.set_data([dr_km[0]], [alt_km[0]])
        panel.set_text(_panel_text(0))

        # Initial 3D rocket view
        _draw_rocket_3d(ax_3d, phi[0], psi[0], theta[0])

        phi_deg_0 = np.degrees(phi[0])
        psi_deg_0 = np.degrees(psi[0])

        ax_3d.set_xlabel(f'Pitch φ: {phi_deg_0:.1f}°', fontsize=9, weight='bold')
        ax_3d.set_ylabel(f'Yaw ψ: {psi_deg_0:.1f}°', fontsize=9, weight='bold')
        ax_3d.set_title('Rocket Attitude (Isometric)', fontsize=10, weight='bold')
        ax_3d.set_xlim([-3, 3])
        ax_3d.set_ylim([-3, 3])
        ax_3d.set_zlim([-1, 6])
        ax_3d.view_init(elev=30, azim=45)
        ax_3d.tick_params(labelsize=7)
        ax_3d.set_zticks([])

        # Keep widget references alive
        fig._link_widgets = (slider, btn_worst, btn_apogee)

        if show:
            plt.show()
        return fig


if __name__ == "__main__":
    pattern = AntennaPattern.from_datasheet()

    # Try to load trajectory from CSV file
    try:
        traj, time_data, mach_data = load_trajectory_from_csv("trajectory.csv")
        print("Trajectory loaded from trajectory.csv")
        print(f"  Number of points: {traj.shape[1]}")
        print(f"  Downrange range: {(traj[0, :].min() / 1e3):.1f} to {(traj[0, :].max() / 1e3):.1f} km")
        print(f"  Altitude range: {(traj[2, :].min() / 1e3):.1f} to {(traj[2, :].max() / 1e3):.1f} km")
        print(f"  Time range: {time_data.min():.1f} to {time_data.max():.1f} s")
        print(f"  Mach range: {mach_data.min():.2f} to {mach_data.max():.2f}\n")
    except FileNotFoundError:
        print("No trajectory.csv found, using default parabolic trajectory")
        x = np.arange(0.5, 400000.5, 0.5)
        y = np.zeros_like(x)
        z = (-x / 4e5) * (x - 400000.0)
        theta = np.linspace(0.0, 18.0 * np.pi, num=len(x))
        phi = np.zeros_like(x)
        psi = np.zeros_like(x)
        x = x + 9000
        traj = np.array([x, y, z, theta, phi, psi], dtype=float)
        time_data = None
        mach_data = None

    # Tracking parabolic-dish ground station
    G_r = LinkBudget.parabolic_dish_gain_db(diameter_m=1.7, efficiency=0.8, f=2.2e9)

    # Link budget with simplified fixed values
    link = LinkBudget(
        P=10.0 * np.log10(20),  # 2 W transmitter
        G_r=G_r,
        R=500.0e6,              # 500 Mbps
        f=2.2e9,                # S-band
        n_tx_antennas=1,
        tx_antenna=pattern,
        G_t=0.0,
        T_s=290.0,              # Fixed system noise temperature
        L_l=1.0,                # Fixed feeder loss
        L_a_zenith=0.04,        # Fixed zenith attenuation
    )
    result = link.compute(traj)

    print("\n" + "="*60)
    print("Link Budget Summary")
    print("="*60)
    print(f"Worst Eb/N0: {result.EbN0.min():.2f} dB at index {np.argmin(result.EbN0)}")
    print(f"Best Eb/N0:  {result.EbN0.max():.2f} dB at index {np.argmax(result.EbN0)}")
    print(f"Mean Eb/N0:  {result.EbN0.mean():.2f} dB")
    print(f"Min C/N0:    {result.CN0.min():.2f} dB-Hz")
    print(f"Mean C/N0:   {result.CN0.mean():.2f} dB-Hz")
    print("="*60 + "\n")

    print("Creating antenna pattern plot...")
    pattern.plot(show=True)
    print("Antenna pattern plot created and displayed")

    print("\nCreating trajectory plot with 3D rocket visualization...")
    link.plot_trajectory(traj, result, show=True,
                        time_array=time_data, mach_array=mach_data)
    print("Trajectory plot created and displayed")
    print("="*60)
