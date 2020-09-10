import numpy as np
from pydrake.all import eq, MathematicalProgram, Solve, Variable, Expression

from dynamics.glider import continuous_dynamics

# Discretize using forward Euler
def discrete_dynamics(x, u):
    dt = 0.001
    x_next = x + dt * continuous_dynamics(x, u)
    return x_next


def rollout(x0, u_trj):
    N = u_trj.shape[0] + 1
    m = x0.shape[0]
    x_trj = np.zeros((N, m))
    x_trj[0] = x0

    for i in range(1, N):
        x_trj[i] = discrete_dynamics(x_trj[i - 1], u_trj[i - 1])

    return x_trj


def cost_stage(x, u):
    m = sym if x.dtype == object else np  # Check type for autodiff

    goal = np.array([15, 0, 10])
    c_pos = (x[0:3] - goal).dot(x[0:3] - goal)
    c_control = (u[0] ** 2 + u[1] ** 2) * 0.1
    return c_pos + c_control


# No control penalty on final cost
def cost_final(x):
    m = sym if x.dtype == object else np  # Check type for autodiff

    goal = np.array([15, 0, 10])
    c_pos = (x[0:3] - goal).dot(x[0:3] - goal)
    return c_pos


def cost_trj(x_trj, u_trj):
    total_cost = 0
    N = x_trj.shape[0]
    for i in range(N - 1):
        total_cost += cost_stage(x_trj[i, :], u_trj[i, :])

    total_cost += cost_final(x_trj[N - 1, :])
    return total_cost
