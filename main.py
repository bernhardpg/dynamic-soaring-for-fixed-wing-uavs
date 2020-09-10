import numpy as np
from pydrake.all import eq, MathematicalProgram, Solve, Variable, Expression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ilqr.ilqr import *
from dynamics.glider import get_wind_field

import pdb


def main():
    N = 1000
    n_x = 6
    n_u = 3

    x0 = np.array([0, 0, 10, 100, 0, 50])

    max_iter = 10
    regu_init = 100
    x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = run_ilqr(
        x0, n_x, n_u, N, max_iter, regu_init
    )
    ###########
    # Plotting
    ###########
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
    # goal = np.array([15, 0, 10])
    ax.scatter(30, 0, 10)
    ax.legend()
    plt.show()

    dt = 0.001
    t = np.linspace(0, dt * N, num=N - 1)
    breakpoint()
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


if __name__ == "__main__":
    main()
