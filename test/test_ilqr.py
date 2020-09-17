import numpy as np
import pdb
from ilqr.ilqr import run_ilqr

from plot.plot import plot_trj_3_wind
import matplotlib.pyplot as plt
from dynamics.slotine_dynamics import get_wind_field

def test_ilqr():
    N = 5000
    n_x = 6
    n_u = 2

    x0 = np.array([50, -np.pi / 2, 0, 5, 0, 0])

    max_iter = 10
    regu_init = 1000
    x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = run_ilqr(
        x0, n_x, n_u, N, max_iter, regu_init
    )

    ###########
    # Plotting
    ###########

    # Slotine dynamics: x = [airspeed, heading, flight_path_angle, z, x, y]
    z = x_trj[:, 3]
    x = x_trj[:, 4]
    y = x_trj[:, 5]

    plot_trj_3_wind(np.vstack((x, y, z)).T, get_wind_field)


    ####
    # Analysis

    plt.subplots(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(cost_trace)
    plt.xlabel("# Iteration")
    plt.ylabel("Total cost")
    plt.title("Cost trace")

    plt.subplot(2, 2, 2)
    delta_opt = np.array(cost_trace) - cost_trace[-1]
    plt.plot(delta_opt)
    plt.yscale("log")
    plt.xlabel("# Iteration")
    plt.ylabel("Optimality gap")
    plt.title("Convergence plot")

    plt.subplot(2, 2, 3)
    plt.plot(redu_ratio_trace)
    plt.title("Ratio of actual reduction and expected reduction")
    plt.ylabel("Reduction ratio")
    plt.xlabel("# Iteration")

    plt.subplot(2, 2, 4)
    plt.plot(regu_trace)
    plt.title("Regularization trace")
    plt.ylabel("Regularization")
    plt.xlabel("# Iteration")
    plt.tight_layout()
    plt.show()

    ####
    # 3D plot

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    x, y, z = np.meshgrid(
        # (-min, max, num steps)
        np.arange(-10, 40, 10),
        np.arange(-10, 40, 10),
        np.arange(0, 15, 3),
    )
    u, v, w = get_wind_field(x, y, z)
    ax.quiver(x, y, z, u, v, w, length=1, linewidth=1)
    ax.plot(
        x_trj[:, 0],
        x_trj[:, 1],
        x_trj[:, 2],
        label="Flight path",
        color="red",
        linewidth=1,
    )
    ax.scatter(x0[0], x0[1], x0[2])
    ax.scatter(10, 2, 8)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Plot input
    X = x_trj[:-1:40, 0]
    Y = x_trj[:-1:40, 1]
    Z = x_trj[:-1:40, 2]
    U = u_trj[::40, 0]
    V = u_trj[::40, 1]
    W = u_trj[::40, 2]

    # ax.quiver(X, Y, Z, U, V, W, length=1, linewidth=1, color="green")
    # ax.set_xlim([-10, 40])
    # ax.set_ylim([-10, 40])
    # ax.set_zlim([0, 15])

    dt = 0.001
    t = np.linspace(0, dt * N, num=N - 1)
    fig, axs = plt.subplots(3)
    fig.suptitle("Input")
    axs[0].plot(t, u_trj[:, 0])
    axs[0].set_title("Input x")
    axs[1].plot(t, u_trj[:, 1])
    axs[1].set_title("Input y")
    axs[2].plot(t, u_trj[:, 2])
    axs[2].set_title("Input z")
    plt.show()

    return 0
