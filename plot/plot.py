import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from dynamics.wind_models import wind_model, ddt_wind_model, get_wind_field


def animate_trajectory_gif(
    zhukovskii_glider, traj, travel_angle
):  # TODO add travel angle
    fig = plt.figure(figsize=(13, 10))
    ax = fig.gca(projection="3d")

    times, x_trj, u_trj = traj
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
            color="blue",
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

        return ([pos, w1, w2, w3],)

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

        return ((pos, w1, w2, w3),)

    ani = FuncAnimation(
        fig, update, frames=list(zip(times, x_trj, u_trj)), init_func=init, blit=False
    )
    # TODO figure out a way to set blit to true

    # Set up formatting for the movie files
    filepath = "./animations/"
    filename = "glider_psi_{0}_degs.mp4".format(int(travel_angle * (180 / np.pi)))
    print("Saving animation as: {0}".format(filename))
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=int(1 / dt), metadata=dict(artist="Me"), bitrate=1800)
    ani.save(filepath + filename, writer=writer)
    print("Saved animation")

    return


# TODO save to gif
# TODO move COM to middle of glider


def animate_trajectory(zhukovskii_glider, traj):  # TODO add travel angle
    fig = plt.figure(figsize=(13, 10))
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

    # Define three points on glider, defined in body frame
    scale = 10
    com_to_wing_vec = np.array([0, zhukovskii_glider.b / 2, 0]) * scale
    com_to_front_vec = np.array([zhukovskii_glider.glider_length, 0, 0]) * scale

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
            linewidth=0.5,
            arrow_length_ratio=0.2,
            pivot="middle",
            color="blue",
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
        # TODO which way rotate by alpha here??
        k_body = np.cross(j_body, i_body)

        R_ned_to_body = np.stack((i_body, j_body, k_body), axis=1)
        curr_com_to_wing_vec = R_ned_to_body.dot(com_to_wing_vec)
        curr_com_to_front_vec = R_ned_to_body.dot(com_to_front_vec)

        # Draw glider
        # wing line
        ax.quiver(
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
        ax.quiver(
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
        ax.quiver(
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
    fig.savefig("./animations/polar_plot.png")

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
