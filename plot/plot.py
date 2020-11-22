import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from dynamics.wind_models import *

plot_location = "./results/plots/"


def plot_wind_profile(ax, wind_function, h_max=20):
    dh_arrows = 2.5

    h = np.arange(0.03, h_max, 0.05)
    ax.plot(wind_function(h), h, color="black")

    arrow_start = np.arange(dh_arrows, h_max, dh_arrows)
    wind_strengths = wind_function(arrow_start)
    zeros = np.zeros(arrow_start.shape[0])
    ax.quiver(
        zeros, arrow_start, wind_strengths, zeros, units="xy", scale=1, color="tab:blue"
    )
    ax.set_aspect("equal")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, h_max)
    # ax.grid()


def plot_wind_profiles():
    fig, axs = plt.subplots(1, 4, constrained_layout=True)
    fig.set_size_inches(10, 3)
    wind_profiles = [
        linear_wind_model,
        log_wind_model,
        exp_wind_model,
        logistic_wind_model,
    ]
    wind_profile_names = [
        "Linear",
        "Logarithmic",
        "Exponential",
        "Logistic",
    ]
    for i in range(len(wind_profiles)):
        ax = axs[i]
        plot_wind_profile(ax, wind_profiles[i])
        ax.set_title(wind_profile_names[i])
        if i == 0:
            ax.set_xlabel("Wind strength [m/s]")
            ax.set_ylabel("Height [m]")

    plt.savefig(plot_location + "wind_models.eps", bbox_inches="tight")
    return


def plot_glider_input(t, u_trj, c_l_trj, phi_trj, n_trj):
    plt.subplots(figsize=(5, 4))

    plt.subplot(3, 1, 1)
    plt.plot(t, c_l_trj)
    plt.xlabel("time [s]")
    plt.title("Lift coeff")
    plt.ylabel("c_L")

    plt.subplot(3, 1, 2)
    plt.plot(t, phi_trj * 180 / np.pi)
    plt.xlabel("time [s]")
    plt.title("Bank angle")
    plt.ylabel("deg")

    plt.subplot(3, 1, 3)
    plt.plot(t, n_trj)
    plt.xlabel("time [s]")
    plt.title("Load factor")
    plt.ylabel("")

    return


def plot_glider_pos(zhukovskii_glider, x_trj, u_trj, travel_angle):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_title("Glider position")

    pos_trj = x_trj[:, 0:3]
    draw_pos_trajectory(pos_trj, travel_angle, ax)
    draw_gliders(zhukovskii_glider, x_trj, u_trj, ax)
    return


# Params:
# x_trj.shape = (N, 3)
# x_trj = [x, y, z]
# TODO continue with adding figure of glider
# TODO continue with only plotting wind at one point
def draw_pos_trajectory(pos_trj, travel_angle, ax):
    # Plot trajectory
    traj_plot = ax.plot(
        pos_trj[:, 0],
        pos_trj[:, 1],
        pos_trj[:, 2],
        label="Flight path",
        color="tab:red",
        linewidth=1,
    )

    # Calculate axis properties
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = limits
    x_diff = np.abs(x_min - x_max)
    y_diff = np.abs(y_min - y_max)
    z_diff = np.abs(z_min - z_max)

    N_arrows = 3
    dx = x_diff / N_arrows - 1
    dy = y_diff / N_arrows - 1
    dz = z_diff / N_arrows - 1

    # Plot start position
    x0 = pos_trj[0, :]
    ax.scatter(x0[0], x0[1], x0[2], color="maroon")

    # Plot direction vector
    dir_vector = np.array([np.sin(travel_angle), np.cos(travel_angle)])
    ax.quiver(
        x0[0],
        x0[1],
        x0[2],
        dir_vector[0],
        dir_vector[1],
        0,
        color="tab:green",
        label="Desired direction",
        length=np.sqrt(dx ** 2 + dy ** 2) * 0.4,
        linewidth=1,
        arrow_length_ratio=0.2,
    )

    # Plot wind field
    xs = (np.arange(x_min, x_max, dx),)
    ys = np.ones(N_arrows + 1) * y_max
    zs = (np.arange(z_min, z_max, dz),)

    X, Y, Z = np.meshgrid(xs, ys, zs)
    u, v, w = get_wind_field(X, Y, Z)
    ax.quiver(
        X,
        Y,
        Z,
        u,
        v,
        w,
        length=1,  # np.sqrt(dx ** 2 + dy ** 2) / 15,
        linewidth=0.5,
        arrow_length_ratio=0.1,
        color="tab:blue",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    # FIX ASPECT RATIO
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    x, y, z = origin
    ax.set_xlim3d([x - x_diff / 2, x + x_diff / 2])
    ax.set_ylim3d([y - y_diff / 2, y + y_diff / 2])
    ax.set_zlim3d([0, z_diff])

    ax.set_box_aspect([x_diff, y_diff, z_diff])
    return


def draw_gliders(zhukovskii_glider, x_trj, u_trj, ax):
    # plot only start first
    x = x_trj[0, :]
    c = u_trj[0, :]
    F, RF, RB, LF, LB = get_glider_corners(zhukovskii_glider, x, c)

    return


def get_glider_corners(zhukovskii_glider, x, c):
    sweep = 0.3
    tip_chord = 0.7
    b = 3.03
    dist_cg_front = 0.5
    # TODO chord =

    # Extract values
    p = x[0:3]
    v_r = x[3:6]
    h = p[2]

    # Define glider corners
    com_to_front = np.array([dist_cg_front, 0, 0])
    com_to_wing_front = np.array([dist_cg_front - sweep, b / 2, 0])
    com_to_wing_back = np.array([dist_cg_front - sweep - tip_chord, b / 2, 0])

    # Calculate heading
    psi = zhukovskii_glider.calc_heading(h, v_r)
    # Calculate bank angle
    phi = zhukovskii_glider.calc_bank_angle(v_r, c)
    # Calculate angle of attack from lift coeff
    c_l = zhukovskii_glider.calc_lift_coeff(v_r, c, A)
    alpha = c_l  # TODO fix this

    # Create rotation matrix
    R = np.array(
        [
            [
                np.cos(psi) * np.cos(alpha),
                np.cos(psi) * np.sin(alpha) * np.sin(phi) - np.sin(psi) * np.cos(phi),
                np.cos(phi) * np.sin(alpha) * np.cos(phi) + np.sin(psi) * np.sin(phi),
            ],
            [
                np.sin(psi) * np.cos(alpha),
                np.sin(psi) * np.sin(alpha) * np.sin(phi) + np.cos(psi) * np.cos(phi),
                np.sin(psi) * np.sin(alpha) * np.cos(phi) - np.cos(psi) * np.sin(phi),
            ],
            [-np.sin(alpha), np.cos(alpha) * np.sin(phi), np.cos(alpha) * np.cos(phi)],
        ]
    )

    # Rotate glider vectors by rotation matrix
    rotated_com_to_front = R.dot(com_to_front)
    rotated_com_to_wing_front = R.dot(com_to_wing_front)
    rotated_com_to_wing_back = R.dot(com_to_wing_back)

    # Calculate glider corners
    F = p + rotated_com_to_front  # Front
    RF = p + rotated_com_to_wing_front  # Right front
    RB = p + rotated_com_to_wing_back  # Right back
    LF = p - rotated_com_to_wing_front  # Left front
    LB = p - rotated_com_to_wing_back  # Left back

    # Plot all corners as vectors without arrowheads
    return F, RF, RB, LF, LB


def polar_plot_avg_velocities(avg_velocities):
    lists = sorted(avg_velocities.items())
    x, y = zip(*lists)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(x, y)
    ax.set_title("Achievable speeds")
    fig.savefig("./animations/polar_plot.png")

    return


# TODO OUTDATED with new relative model
def save_trajectory_gif(zhukovskii_glider, traj, travel_angle):
    ## ANIMATION FILE SETTINGS
    filepath = "./animations/"
    filename = "glider_psi_{0}_degs.mp4".format(int(travel_angle * (180 / np.pi)))
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=int(1 / dt), metadata=dict(artist="Me"), bitrate=1800)

    # SETUP FIGURE
    fig = plt.figure(figsize=(13, 10))
    ax = fig.gca(projection="3d")

    t, x_trj, u_trj = traj
    N = x_trj.shape[0]
    dt = times[1] - times[0]
    T = 1 / dt

    x0 = x_trj[0, :]

    x_min = min(x_trj[:, 0])
    x_max = max(x_trj[:, 0])
    y_min = min(x_trj[:, 1])
    y_max = max(x_trj[:, 1])
    z_min = 0
    z_max = max(x_trj[:, 2])

    # Spacing for wind field
    dx = np.abs(x_min - x_max)
    dy = np.abs(y_min - y_max)
    dz = np.abs(z_min - z_max)
    max_axis = max([dx, dy, dz])

    x, y, z = np.meshgrid(
        # (-min, max, step_length)
        np.arange(x_min, x_min + max_axis, max_axis / 5 - 1),
        np.arange(y_min, y_min + max_axis, max_axis / 5 - 1),
        np.arange(z_min, z_min + max_axis, max_axis / 5 - 1),
    )
    u, v, w = get_wind_field(x, y, z)

    # Define three points on glider, defined in body frame
    scale = 8
    com_to_wing_vec = np.array([0, zhukovskii_glider.b / 2, 0]) * scale
    com_to_front_vec = np.array([zhukovskii_glider.glider_length, 0, 0]) * scale

    pos = ax.scatter([], [], [])
    w1 = ax.quiver([], [], [], [], [], [])
    w2 = ax.quiver([], [], [], [], [], [])
    w3 = ax.quiver([], [], [], [], [], [])

    def init():
        ax.set_zlim(0, max_axis)
        ax.set_xlim(x_min, x_min + max_axis)
        ax.set_ylim(y_min, y_min + max_axis)
        ax.plot(x_trj[:, 0], x_trj[:, 1], x_trj[:, 2], linewidth=0.7)  # Plot trajectory
        ax.scatter(x0[0], x0[1], x0[2])  # Plot initial position

        # plot wind field
        ax.quiver(
            x,
            y,
            z,
            u,
            v,
            w,
            length=1,
            linewidth=0.5,
            arrow_length_ratio=0.2,
            pivot="middle",
            color="tab:blue",
        )

        # Plot direction vector
        dir_vector = np.array([np.sin(travel_angle), np.cos(travel_angle)])
        ax.quiver(
            x0[0],
            x0[1],
            x0[2],
            dir_vector[0],
            dir_vector[1],
            0,
            color="green",
            label="Desired direction",
            length=10,
            arrow_length_ratio=0.1,
        )

        return (
            pos,
            w1,
            w2,
            w3,
        )

    def update(frame):
        plt.cla()
        init()
        time, x, u = frame

        com = x[0:3]  # Center of mass
        c = u[:]
        pos = ax.scatter(
            com[0], com[1], com[2], color="red", s=0.5
        )  # plot current position

        vel_rel = zhukovskii_glider.get_vel_rel(x[:])
        alpha = zhukovskii_glider.get_angle_of_attack(x[:], u[:])

        j_body = c / np.linalg.norm(c)  # j unit vector in body frame

        i_stability = vel_rel / np.linalg.norm(vel_rel)  # i unit vec in stability frame
        i_body = np.array(
            [
                [np.cos(alpha), 0, np.sin(alpha)],
                [0, 1, 0],
                [-np.sin(alpha), 0, np.cos(alpha)],
            ]
        ).dot(
            i_stability
        )  # Rotate i_stab by alpha around y axis to get i_body
        # TODO which way rotate by alpha here??
        k_body = np.cross(j_body, i_body)

        R_ned_to_body = np.stack((i_body, j_body, k_body), axis=1)
        curr_com_to_wing_vec = R_ned_to_body.dot(com_to_wing_vec)
        curr_com_to_front_vec = R_ned_to_body.dot(com_to_front_vec)

        # Draw glider
        # wing line
        w1 = ax.quiver(
            com[0] - curr_com_to_wing_vec[0],
            com[1] - curr_com_to_wing_vec[1],
            com[2] - curr_com_to_wing_vec[2],
            curr_com_to_wing_vec[0] * 2,
            curr_com_to_wing_vec[1] * 2,
            curr_com_to_wing_vec[2] * 2,
            linewidth=2,
            arrow_length_ratio=0.0,
            color="black",
        )
        # left wing to front
        w2 = ax.quiver(
            com[0] - curr_com_to_wing_vec[0],
            com[1] - curr_com_to_wing_vec[1],
            com[2] - curr_com_to_wing_vec[2],
            curr_com_to_wing_vec[0] + curr_com_to_front_vec[0],
            curr_com_to_wing_vec[1] + curr_com_to_front_vec[1],
            curr_com_to_wing_vec[2] + curr_com_to_front_vec[2],
            linewidth=2,
            arrow_length_ratio=0.0,
            color="black",
        )
        # left wing to front
        w3 = ax.quiver(
            com[0] + curr_com_to_wing_vec[0],
            com[1] + curr_com_to_wing_vec[1],
            com[2] + curr_com_to_wing_vec[2],
            -curr_com_to_wing_vec[0] + curr_com_to_front_vec[0],
            -curr_com_to_wing_vec[1] + curr_com_to_front_vec[1],
            -curr_com_to_wing_vec[2] + curr_com_to_front_vec[2],
            linewidth=2,
            arrow_length_ratio=0.0,
            color="black",
        )

        # Plot x axis
        ax.quiver(
            com[0],
            com[1],
            com[2],
            i_body[0],
            i_body[1],
            i_body[2],
            linewidth=2,
            color="red",
            length=scale * 3,
        )
        # Plot y axis
        ax.quiver(
            com[0],
            com[1],
            com[2],
            j_body[0],
            j_body[1],
            j_body[2],
            linewidth=2,
            color="yellow",
            length=scale * 3,
        )
        # Plot z axis
        ax.quiver(
            com[0],
            com[1],
            com[2],
            -k_body[0],
            -k_body[1],
            -k_body[2],
            linewidth=2,
            color="green",
            length=scale * 3,
        )

        return (
            pos,
            w1,
            w2,
            w3,
        )

    ani = FuncAnimation(
        fig, update, frames=list(zip(times, x_trj, u_trj)), init_func=init, blit=True
    )

    ## SAVE ANIMATION
    ani.save(filepath + filename, writer=writer)
    print("Saved animation as: {0}".format(filename))
    plt.close()

    return
