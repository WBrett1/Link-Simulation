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
        self.L_a_zenith = L_a_zenith if L_a_zenith is not None else self.estimate_zenith_loss_db(f)
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
