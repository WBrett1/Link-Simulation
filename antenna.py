import numpy as np


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
        # Normalize phi_body to [0, 2π]
        phi_body_rad = np.mod(phi_body_rad, 2.0 * np.pi)

        # Get gains from both plane cuts at this off-boresight angle
        g_az = self.azimuth_gain_db(theta_off_rad)
        g_el = self.elevation_gain_db(theta_off_rad)

        # Compute blending weights based on azimuthal angle
        # phi_body = 0, π → all azimuth (side-to-side)
        # phi_body = π/2, 3π/2 → all elevation (up-down)
        # Use cos²/sin² for smooth interpolation
        phi_normalized = np.mod(phi_body_rad, np.pi)
        weight_az = np.cos(phi_normalized) ** 2
        weight_el = np.sin(phi_normalized) ** 2

        # Convert to linear, weight, and convert back to dB
        g_az_lin = 10.0 ** (g_az / 10.0)
        g_el_lin = 10.0 ** (g_el / 10.0)
        g_combined_lin = weight_az * g_az_lin + weight_el * g_el_lin

        return 10.0 * np.log10(g_combined_lin)
