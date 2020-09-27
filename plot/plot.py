import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dynamics.wind_models import wind_model, ddt_wind_model, get_wind_field


def animate_trajectory(zhukovskii_glider, traj):  # TODO add travel angle
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    times, x_trj, u_trj = traj
    N = x_trj.shape[0]
    dt = times[1] - times[0]

    x_min = min(x_trj[:, 0])
    x_max = max(x_trj[:, 0])
    y_min = min(x_trj[:, 1])
    y_max = max(x_trj[:, 1])
    z_min = 0
    z_max = max(x_trj[:, 2]) * 5

    # Spacing for wind field
    dx = np.abs((x_min - x_max) / 5) - 1
    dy = np.abs((y_min - y_max) / 5) - 1
    dz = np.abs((z_min - z_max) / 3) - 1
    x, y, z = np.meshgrid(
        # (-min, max, step_length)
        np.arange(x_min, x_max, dx),
        np.arange(y_min, y_max, dy),
        np.arange(z_min, z_max, dz),
    )
    u, v, w = get_wind_field(x, y, z)

    x0 = x_trj[0, :]

    for n in range(N):
        plt.cla()
        ax.plot(x_trj[:, 0], x_trj[:, 1], x_trj[:, 2], linewidth=0.7)  # Plot trajectory
        ax.scatter(x0[0], x0[1], x0[2])  # Plot initial position
        axis3d_equal(x_trj[:, 0], x_trj[:, 1], x_trj[:, 2], ax)

        # plot wind field
        ax.quiver(
            x,
            y,
            z,
            u,
            v,
            w,
            length=1,
            linewidth=0.4,
            arrow_length_ratio=0.1,
            pivot="middle",
            color="blue"
        )

        com = x_trj[n, 0:3]  # Center of mass
        c = u_trj[n, :]
        ax.scatter(com[0], com[1], com[2], color="red", s=0.5)  # plot current position

        vel_rel = zhukovskii_glider.get_vel_rel(x_trj[n, :])
        alpha = zhukovskii_glider.get_angle_of_attack(x_trj[n, :], u_trj[n, :])

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

        ax.quiver(
            com[0],
            com[1],
            com[2],
            i_body[0],
            i_body[1],
            i_body[2],
            color="red",
            linewidth=0.9,
            length=10,
        )  # Body x-axis

        ax.quiver(
            com[0],
            com[1],
            com[2],
            j_body[0],
            j_body[1],
            j_body[2],
            color="red",
            linewidth=0.9,
            length=10,
        )  # Body x-axis

        ax.set_title("Flight trajectory")

        plt.pause(dt)

    return


# Borrowed from here: https://github.com/AtsushiSakai/PythonRobotics/blob/135e33af88128c68c0420282775bbfeacc327f77/AerialNavigation/rocket_powered_landing/rocket_powered_landing.py
def axis3d_equal(X, Y, Z, ax):

    max_range = np.array(
        [X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]
    ).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (
        X.max() + X.min()
    )
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (
        Y.max() + Y.min()
    )
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (
        Z.max() + Z.min()
    )
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], "w")

    return


def plot_trajectories(trajectories):
    N = len(trajectories.items())
    fig = plt.figure(figsize=(3, int(N * 3 / 2)))

    i = 1
    for travel_angle, traj in trajectories.items():
        times, x_trj, u_trj = traj
        ax_x = fig.add_subplot(N, 2, 2 * i - 1, projection="3d")
        plot_x_trj(x_trj, travel_angle, ax_x)

        ax_u = fig.add_subplot(N, 2, 2 * i)
        plot_u_trj(times, u_trj, ax_u)

        i = i + 1

    return


def plot_u_trj(t, u_trj, ax):
    ax.set_title("input (circulation)")

    ax.plot(t, u_trj[:, 0], label="x-axis")
    ax.plot(t, u_trj[:, 1], label="y-axis")
    ax.plot(t, u_trj[:, 2], label="z-axis")
    ax.set_ylim(-7, 7)
    ax.legend()
    return


def plot_x_trj(x_trj, travel_angle, ax):
    x0 = x_trj[0, :]

    x_min = min(x_trj[:, 0])
    x_max = max(x_trj[:, 0])
    y_min = min(x_trj[:, 1])
    y_max = max(x_trj[:, 1])
    z_min = 0
    z_max = max(x_trj[:, 2])

    dx = np.abs((x_min - x_max) / 2) - 1
    dy = np.abs((y_min - y_max) / 2.0) - 1
    dz = np.abs((z_min - z_max) / 2.0) - 1

    # Plot wind field
    x, y, z = np.meshgrid(
        # (-min, max, step_length)
        np.arange(x_min, x_max, dx),
        np.arange(y_min, y_max, dy),
        np.arange(z_min, z_max, dz),
    )
    u, v, w = get_wind_field(x, y, z)

    ax.quiver(
        x,
        y,
        z,
        u,
        v,
        w,
        length=1,  # np.sqrt(dx ** 2 + dy ** 2) / 15,
        linewidth=0.7,
        arrow_length_ratio=0.1,
        pivot="middle",
    )

    # Plot trajectory
    traj_plot = ax.plot(
        x_trj[:, 0],
        x_trj[:, 1],
        x_trj[:, 2],
        label="Flight path",
        color="red",
        linewidth=1,
    )

    # Plot start position
    ax.scatter(x0[0], x0[1], x0[2])

    dir_vector = np.array([np.sin(travel_angle), np.cos(travel_angle)])
    # Plot direction vector
    ax.quiver(
        x0[0],
        x0[1],
        x0[2],
        dir_vector[0],
        dir_vector[1],
        0,
        color="green",
        label="Desired direction",
        length=np.sqrt(dx ** 2 + dy ** 2),
        arrow_length_ratio=0.1,
    )

    ax.set_zlim(0, z_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("{0}".format(travel_angle))

    return traj_plot


def polar_plot_avg_velocities(avg_velocities):
    lists = sorted(avg_velocities.items())
    x, y = zip(*lists)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(x, y)
    ax.set_title("Achievable speeds")

    return


def plot_input_slotine_glider(t, u_trj):
    plt.subplots(figsize=(5, 3))
    plt.title("Input")

    plt.subplot(2, 1, 1)
    plt.plot(t, u_trj[:, 0])
    plt.xlabel("time (seconds)")
    plt.ylabel("Lift coefficient")

    plt.subplot(2, 1, 2)
    plt.plot(t, u_trj[:, 1])
    plt.xlabel("time (seconds)")
    plt.ylabel("Roll (radians)")

    return


def plot_circulation(t, u_trj):
    plt.subplots(figsize=(5, 3))
    plt.title("Circulation")

    plt.subplot(3, 1, 1)
    plt.plot(t, u_trj[:, 0])
    plt.xlabel("time (seconds)")
    plt.ylabel("x-component")

    plt.subplot(3, 1, 2)
    plt.plot(t, u_trj[:, 1])
    plt.xlabel("time (seconds)")
    plt.ylabel("y-component")

    plt.subplot(3, 1, 3)
    plt.plot(t, u_trj[:, 2])
    plt.xlabel("time (seconds)")
    plt.ylabel("z-component")

    return


def plot_trj_3_wind(x_trj, dir_vector=np.array([0, 0, 0])):
    # To make the plot function general:
    #   x_trj.shape = (N, 3)
    #   x_trj = [x, y, z]
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    x0 = x_trj[0, :]

    x_min = min(x_trj[:, 0])
    x_max = max(x_trj[:, 0])
    y_min = min(x_trj[:, 1])
    y_max = max(x_trj[:, 1])
    z_min = 0
    z_max = max(x_trj[:, 2])

    dx = np.abs((x_min - x_max) / 5.0) - 1
    dy = np.abs((y_min - y_max) / 5.0) - 1
    dz = np.abs((z_min - z_max) / 3.0) - 1

    # Plot wind field
    x, y, z = np.meshgrid(
        # (-min, max, step_length)
        np.arange(x_min, x_max, dx),
        np.arange(y_min, y_max, dy),
        np.arange(z_min, z_max, dz),
    )
    u, v, w = get_wind_field(x, y, z)

    ax.quiver(
        x,
        y,
        z,
        u,
        v,
        w,
        length=1,  # np.sqrt(dx ** 2 + dy ** 2) / 15,
        linewidth=0.7,
        arrow_length_ratio=0.1,
        pivot="middle",
    )

    # Plot trajectory
    traj_plot = ax.plot(
        x_trj[:, 0],
        x_trj[:, 1],
        x_trj[:, 2],
        label="Flight path",
        color="red",
        linewidth=1,
    )

    # Plot start position
    ax.scatter(x0[0], x0[1], x0[2])

    # Plot direction vector
    ax.quiver(
        x0[0],
        x0[1],
        x0[2],
        dir_vector[0],
        dir_vector[1],
        0,
        color="green",
        label="Desired direction",
        length=np.sqrt(dx ** 2 + dy ** 2),
        arrow_length_ratio=0.1,
    )

    ax.set_zlim(0, z_max)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Trajectory")

    return traj_plot
