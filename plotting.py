from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive windows
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from antenna import AntennaPattern
from link_budget import LinkBudget, LinkResult
from rocket_viz import _draw_rocket_3d


def plot_antenna_pattern(pattern: AntennaPattern, show=True):
    """Render the antenna pattern's two datasheet cuts and the roll-averaged composite."""
    fig, axes = plt.subplots(
        1, 3, subplot_kw={"projection": "polar"}, figsize=(15, 5)
    )
    phi = np.linspace(0.0, 2.0 * np.pi, 721)

    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 30))

    axes[0].plot(phi, pattern.azimuth_gain_db(phi), color="crimson", lw=2)
    axes[0].set_title("Azimuth Pattern")
    axes[1].plot(phi, pattern.elevation_gain_db(phi), color="crimson", lw=2)
    axes[1].set_title("Elevation Pattern")
    axes[2].plot(phi, pattern.composite_gain_db(phi), color="navy", lw=2)
    axes[2].set_title("Roll-averaged composite\n(used by LinkBudget)")

    for ax in axes:
        ax.set_rlim(-25, 15)
        ax.set_rticks([-20, -10, 0, 10])

    fig.suptitle("Nose-mounted antenna pattern model", y=1.02)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_trajectory(
    link: LinkBudget,
    trajectory,
    result: Optional[LinkResult] = None,
    show=True,
    time_array: Optional[np.ndarray] = None,
    mach_array: Optional[np.ndarray] = None
):
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
        link: LinkBudget instance
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
    traj = link.trajectory_vel_to_enu(traj)

    x, y, z, theta, phi, psi = traj
    n = x.size
    if n < 2:
        raise ValueError("plot_trajectory needs at least 2 trajectory samples")

    if result is None:
        result = link.compute(traj)

    # Plot geometry (mirrors compute()).
    xyz = np.stack([x, y, z], axis=0)
    slant = np.sqrt(x**2 + y**2 + z**2)            # m
    downrange = np.sqrt(x**2 + y**2)               # m, horizontal range
    altitude = z                                   # m
    elevation = np.degrees(np.arctan2(z, downrange))
    u_r2r = -xyz / slant
    u_spin = link._spin_axis_enu(theta, phi, psi)
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
    horizon_alt_disp = link._horizon_altitude(dr_disp * 1e3, 4.0 / 3.0) / 1e3

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
    ax.plot(d_line / 1e3, link._horizon_altitude(d_line) / 1e3,
            "--", color="tab:blue", lw=1.3, label="Visible horizon")
    ax.plot(d_line / 1e3, link._horizon_altitude(d_line, 4.0 / 3.0) / 1e3,
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
