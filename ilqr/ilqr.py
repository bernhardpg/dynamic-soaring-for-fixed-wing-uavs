import numpy as np
from pydrake.all import eq, MathematicalProgram, Solve, Variable, Expression
import pydrake.symbolic as sym

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


#######
# Costs
#######


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


################
# Numerical diff
################


class derivatives:
    def __init__(self, discrete_dynamics, cost_stage, cost_final, n_x, n_u):
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(n_u)])
        x = self.x_sym
        u = self.u_sym

        l = cost_stage(x, u)
        self.l_x = sym.Jacobian([l], x).ravel()
        self.l_u = sym.Jacobian([l], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)

        l_final = cost_final(x)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)

        f = discrete_dynamics(x, u)
        self.f_x = sym.Jacobian(f, x)
        self.f_u = sym.Jacobian(f, u)

    def stage(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})

        l_x = sym.Evaluate(self.l_x, env).ravel()
        l_u = sym.Evaluate(self.l_u, env).ravel()
        l_xx = sym.Evaluate(self.l_xx, env)
        l_ux = sym.Evaluate(self.l_ux, env)
        l_uu = sym.Evaluate(self.l_uu, env)

        f_x = sym.Evaluate(self.f_x, env)
        f_u = sym.Evaluate(self.f_u, env)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

    def final(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}

        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)

        return l_final_x, l_final_xx
