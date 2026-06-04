import numpy as np

# Physical constants
C_LIGHT = 2.99792458e8      # m/s
K_BOLTZMANN = 1.380649e-23  # J/K

# ============================================================================
#  LINK BUDGET INPUTS
# ============================================================================
# Core link parameters
POWER_TX_DBW = 10.0 * np.log10(20)   # dBW, transmitter power (2 W default)
BITRATE_HZ = 500.0e6                 # Hz, data rate (500 Mbps)
FREQUENCY_HZ = 2.2e9                 # Hz, S-band default

# Antenna configuration
N_TX_ANTENNAS = 3            # number of transmitter antennas
TX_ANTENNA = None            # AntennaPattern instance (set below)
GAIN_TX_DBI = 0.0            # dBi, transmit peak gain

# Fixed link budget parameters (engineering standard for sounding rockets)
T_SYSTEM_K = 290.0           # K, system noise temperature (fixed)
L_LINE_DB = 1.0              # dB, transmitter line/feed loss (fixed)
L_A_ZENITH_DB = 0.04         # dB, zenith atmospheric loss S-band (fixed)
L_POL_DB = 0.0               # dB, polarization mismatch
EL_MIN_DEG = 2.0             # deg, elevation angle clamp near horizon
