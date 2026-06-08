import numpy as np


class AntennaPattern:
    """
    Antenna pattern with azimuth and elevation cuts loaded from CSV.
    Supports 2D interpolation via gain_3d_db method and roll-averaged composite gain.
    """

    def __init__(self, angles_rad, azimuth_cut_db, elevation_cut_db):
        self.angles_rad = np.asarray(angles_rad, dtype=float)
        self.azimuth_cut_db = np.asarray(azimuth_cut_db, dtype=float)
        self.elevation_cut_db = np.asarray(elevation_cut_db, dtype=float)

    @classmethod
    def from_csv(cls, filename='AntennaPattern.csv'):
        """Load pattern from CSV file with columns: angle, azimuth_db, elevation_db"""
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        angles_deg = data[:, 0]
        azimuth_cut_db = data[:, 1]
        elevation_cut_db = data[:, 2]
        angles_rad = np.deg2rad(angles_deg)
        return cls(angles_rad, azimuth_cut_db, elevation_cut_db)

    @staticmethod
    def _fold_to_half(angle_rad):
        """Map any angle to [0, π] using even symmetry about 0 and π."""
        a = np.mod(angle_rad, 2.0 * np.pi)
        return np.where(a > np.pi, 2.0 * np.pi - a, a)

    def _cut_db(self, angle_rad, cut_db):
        a = self._fold_to_half(angle_rad)
        return np.interp(a, self.angles_rad, cut_db)

    def azimuth_gain_db(self, angle_rad):
        """H-plane absolute gain (dBi) vs angle from boresight."""
        return self._cut_db(angle_rad, self.azimuth_cut_db)

    def elevation_gain_db(self, angle_rad):
        """E-plane absolute gain (dBi) vs angle from boresight."""
        return self._cut_db(angle_rad, self.elevation_cut_db)

    def gain_3d_db(self, theta_off_rad, phi_body_rad):
        """
        3D antenna gain (dBi) vs off-boresight angle and azimuthal angle in body frame.

        Interpolates between azimuth (H-plane) and elevation (E-plane) cuts based on
        phi_body. The azimuth cut is at phi_body = 0, π (side-to-side); the elevation
        cut is at phi_body = ±π/2 (up-down).

        Args:
            theta_off_rad: angle from boresight (0 to π), scalar or ndarray
            phi_body_rad: azimuthal angle in body frame (0 to 2π), same shape as theta_off

        Returns:
            Gain in dBi, same shape as inputs
        """
        phi_body_rad = np.mod(phi_body_rad, 2.0 * np.pi)

        g_az = self.azimuth_gain_db(theta_off_rad)
        g_el = self.elevation_gain_db(theta_off_rad)

        phi_normalized = np.mod(phi_body_rad, np.pi)
        weight_az = np.cos(phi_normalized) ** 2
        weight_el = np.sin(phi_normalized) ** 2

        g_az_lin = 10.0 ** (g_az / 10.0)
        g_el_lin = 10.0 ** (g_el / 10.0)
        g_combined_lin = weight_az * g_az_lin + weight_el * g_el_lin

        return 10.0 * np.log10(g_combined_lin)
