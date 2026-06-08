from dataclasses import dataclass
from typing import Optional
import numpy as np
from constants import C_LIGHT, K_BOLTZMANN
from antenna import AntennaPattern


@dataclass
class LinkResult:
    """Per-timestep link-budget outputs (each a length-N ndarray)."""
    EbN0: np.ndarray  # dB
    EIRP: np.ndarray  # dBW
    GT: np.ndarray    # dB/K
    CN0: np.ndarray   # dB-Hz


class LinkBudget:
    """
    Per-timestep link budget for a sounding-rocket downlink (S-band default).

    Frame: ENU with the receiver at the origin. Units: P in dBW, G_t / G_r in dBi.
    Polarization: Circular (RHCP or LHCP) with constant loss modeling antenna
    axial ratio degradation and nominal ionospheric Faraday rotation effects.
    Uses industry-standard fixed values for atmospheric and system parameters.
    """

    R_EARTH = 6371e3              # m, mean Earth radius (horizon model)

    @staticmethod
    def estimate_zenith_loss_db(f_hz):
        """
        Estimate zenith atmospheric loss (dB) for a given frequency.
        Based on typical clear-sky conditions with standard humidity.
        Uses piecewise linear interpolation from empirical data points.
        """
        f_ghz = f_hz / 1e9
        freq_ref = np.array([1.0, 2.2, 10.0, 35.0])
        loss_ref = np.array([0.015, 0.04, 0.15, 1.0])
        return float(np.interp(f_ghz, freq_ref, loss_ref))

    def __init__(
        self,
        P,                         # dBW, transmitter power
        G_r,                       # dBi, receiver gain
        R,                         # Hz, bit rate
        f,                         # Hz, frequency (required, no default)
        *,
        n_tx_antennas=1,           # number of transmitter antennas
        tx_antenna=None,           # AntennaPattern; None = flat gain at G_t
        G_t=0.0,                   # dBi, transmit peak/flat gain
        T_s=290.0,                 # K, system noise temperature (fixed)
        L_l=1.0,                   # dB, transmitter line/feed loss (fixed)
        L_a_zenith=None,           # dB, zenith atmospheric loss; auto-estimated if None
        L_cp=0.5,                  # dB, circular polarization loss (antenna AR + Faraday margin)
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
        self.L_a_zenith = L_a_zenith if L_a_zenith is not None else self.estimate_zenith_loss_db(f)
        self.L_cp = L_cp
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
    def _body_frame_from_enu(u_spin, theta):
        """
        Compute orthonormal body-frame axes in ENU coordinates.

        Args:
            u_spin: spin axis unit vector(s) in ENU, shape (3,) or (3, N)
            theta: roll angle(s) about spin axis, scalar or shape (N,)

        Returns:
            (z_body, x_body, y_body): unit vectors in ENU frame
                z_body: points along boresight (spin axis)
                x_body: perpendicular, oriented by zero-roll reference
                y_body: completes right-handed frame (z_body × x_body)
        """
        # Ensure u_spin is 2D: (3, N)
        u_spin = np.asarray(u_spin, dtype=float)
        if u_spin.ndim == 1:
            u_spin = u_spin[:, None]
        n = u_spin.shape[1]

        # Normalize spin axis
        z_body = u_spin / (np.linalg.norm(u_spin, axis=0, keepdims=True) + 1e-10)

        # Choose reference direction perpendicular to z_body
        # If nearly vertical (z ≈ ±1), use East [1, 0, 0]; else use Up [0, 0, 1]
        ref = np.zeros((3, n))
        near_vertical = np.abs(z_body[2, :]) > 0.999
        ref[2, ~near_vertical] = 1.0
        ref[0, near_vertical] = 1.0

        # Gram-Schmidt: x_body_0 = ref - (ref · z_body) z_body
        ref_dot_z = np.sum(ref * z_body, axis=0, keepdims=True)
        x_body_0 = ref - ref_dot_z * z_body
        x_body_0 = x_body_0 / (np.linalg.norm(x_body_0, axis=0, keepdims=True) + 1e-10)

        # y_body_0 perpendicular to both (right-hand rule: z × x)
        y_body_0 = np.cross(z_body, x_body_0, axis=0)

        # Apply roll rotation about z_body
        # x_body = cos(θ) x_body_0 + sin(θ) y_body_0
        # y_body = -sin(θ) x_body_0 + cos(θ) y_body_0
        theta = np.asarray(theta, dtype=float)
        if np.isscalar(theta):
            theta = np.full(n, theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_body = cos_theta[None, :] * x_body_0 + sin_theta[None, :] * y_body_0
        y_body = -sin_theta[None, :] * x_body_0 + cos_theta[None, :] * y_body_0

        return z_body, x_body, y_body

    @staticmethod
    def _los_to_body_angles(u_r2r, z_body, x_body, y_body):
        """
        Transform receiver line-of-sight to body-frame spherical coordinates.

        Args:
            u_r2r: rocket-to-receiver unit vector in ENU, shape (3,) or (3, N)
            z_body, x_body, y_body: body frame axes in ENU

        Returns:
            theta_off: off-boresight angle (0 to π)
            phi_body: azimuthal angle in body frame (0 to 2π)
                phi_body = 0 → x_body direction (azimuth plane)
                phi_body = π/2 → y_body direction (elevation plane)
        """
        # Project LOS onto body axes
        los_x = np.sum(u_r2r * x_body, axis=0)
        los_y = np.sum(u_r2r * y_body, axis=0)
        los_z = np.sum(u_r2r * z_body, axis=0)

        # Polar angle from boresight
        theta_off = np.arccos(np.clip(los_z, -1.0, 1.0))

        # Azimuthal angle in body frame
        phi_body = np.arctan2(los_y, los_x)
        phi_body = np.mod(phi_body, 2.0 * np.pi)

        return theta_off, phi_body

    def _multi_antenna_best_gain(self, u_r2r, z_body, x_body, y_body):
        """
        Compute the best gain from multiple antennas evenly spaced around the rocket.

        Args:
            u_r2r: rocket-to-receiver unit vector in ENU, shape (3, N)
            z_body, x_body, y_body: body frame axes in ENU, each shape (3, N)

        Returns:
            best_gain: maximum gain across all antennas, shape (N,)
        """
        # Project LOS onto body axes
        los_x = np.sum(u_r2r * x_body, axis=0)
        los_y = np.sum(u_r2r * y_body, axis=0)
        los_z = np.sum(u_r2r * z_body, axis=0)

        # Off-boresight angle (same for all antennas)
        theta_off = np.arccos(np.clip(los_z, -1.0, 1.0))

        # Compute gain from each antenna at evenly-spaced positions
        gains = []
        for i in range(self.n_tx_antennas):
            phi_ant = i * 2.0 * np.pi / self.n_tx_antennas

            # Azimuthal angle in the antenna-centered frame
            # Receiver direction is at atan2(los_y, los_x) in the body frame
            # Antenna i is at phi_ant, so rotate the coordinate system
            phi_body_ant = np.arctan2(los_y, los_x) - phi_ant
            phi_body_ant = np.mod(phi_body_ant, 2.0 * np.pi)

            # Get gain from this antenna
            gain = self.tx_antenna.gain_3d_db(theta_off, phi_body_ant)
            gains.append(gain)

        gains = np.array(gains)
        best_gain = np.max(gains, axis=0)

        return best_gain

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

        # Compute body-frame axes from attitude angles
        u_spin = self._spin_axis_enu(theta, phi, psi)
        z_body, x_body, y_body = self._body_frame_from_enu(u_spin, theta)

        # Effective gains
        G_t_eff = self.G_t
        if self.tx_antenna is not None:
            if self.n_tx_antennas == 1:
                # Single antenna case
                theta_off, phi_body = self._los_to_body_angles(u_r2r, z_body, x_body, y_body)
                G_t_eff = self.G_t + self.tx_antenna.gain_3d_db(theta_off, phi_body)
            else:
                # Multiple antennas: compute gain from each and select the best
                G_t_eff = self.G_t + self._multi_antenna_best_gain(u_r2r, z_body, x_body, y_body)
        G_r_eff = self.G_r

        # Free-space path loss (d in m, f in Hz); 147.55 = 20*log10(c / 4pi)
        L_s = 20.0 * np.log10(d) + 20.0 * np.log10(self.f) - 147.55

        # Fixed atmospheric loss (zenith scaled by 1/sin(elevation))
        L_a = self.L_a_zenith / np.sin(np.maximum(el, np.deg2rad(self.el_min_deg)))

        # Link-budget assembly (all in dB)
        k_dB = 10.0 * np.log10(K_BOLTZMANN)          # ~ -228.6 dBW/K/Hz
        EIRP = self.P - self.L_l + G_t_eff            # dBW
        GT = G_r_eff - 10.0 * np.log10(self.T_s)     # dB/K
        CN0 = EIRP + GT - L_s - L_a - self.L_cp - k_dB  # dB-Hz
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
