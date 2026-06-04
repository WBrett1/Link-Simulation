import numpy as np
from antenna import AntennaPattern
from trajectory import load_trajectory_from_csv
from link_budget import LinkBudget
from plotting import plot_antenna_pattern, plot_trajectory


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
    plot_antenna_pattern(pattern, show=True)
    print("Antenna pattern plot created and displayed")

    print("\nCreating trajectory plot with 3D rocket visualization...")
    plot_trajectory(link, traj, result, show=True,
                    time_array=time_data, mach_array=mach_data)
    print("Trajectory plot created and displayed")
    print("="*60)
