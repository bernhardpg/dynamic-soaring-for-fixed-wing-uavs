import numpy as np
from pydrake.all import eq, MathematicalProgram, Solve, Variable, Expression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ilqr.ilqr import *
from dynamics.glider import get_wind_field

import pdb


def main():
    N = 10000
    n_x = 6
    n_u = 3

    u_trj = np.zeros((N - 1, n_u))
    x0 = np.array([0, 0, 10, 100, 0, 50])
    x_trj = rollout(x0, u_trj)

    derivs = derivatives(discrete_dynamics, cost_stage, cost_final, n_x, n_u)

    ###########
    # Test rollout and costs
    ###########

    costs = np.zeros((N - 1, 1))
    for i in range(N - 1):
        costs[i] = cost_stage(x_trj[i], u_trj[i])

    total_cost = cost_trj(x_trj, u_trj)
    print(total_cost)

    plt.plot(costs)
    plt.title("costs")
    plt.show()

    ###########
    # Plotting
    ###########

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    x, y, z = np.meshgrid(
        # (-min, max, num steps)
        np.arange(-10, 100, 10),
        np.arange(-10, 100, 10),
        np.arange(-10, 15, 5),
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
    ax.legend()
    plt.show()

    #    t = np.linspace(0, dt * N, num=N - 1)
    #    fig, axs = plt.subplots(3)
    #    fig.suptitle("Input")
    #    axs[0].plot(t, u_sol[0, :])
    #    axs[0].set_title("Input x")
    #    axs[1].plot(t, u_sol[1, :])
    #    axs[1].set_title("Input y")
    #    axs[2].plot(t, u_sol[2, :])
    #    axs[2].set_title("Input z")
    #    plt.show()

    return 0


if __name__ == "__main__":
    main()
