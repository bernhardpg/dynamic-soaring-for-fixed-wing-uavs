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
