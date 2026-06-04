import numpy as np


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
