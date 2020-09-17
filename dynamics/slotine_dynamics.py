import numpy as np
import pydrake.symbolic as sym
from pydrake.all import eq, MathematicalProgram, Solve, Variable, Expression


# Dynamics taken from Slotine: successive shallow arcs
def continuous_dynamics(state, u):
    me = sym if state.dtype == object else np  # Check type for autodiff
    # me = math engine

    # Constants
    rho = 1.255  # air_density
    S = 2  # wing_area
    c_d0 = 1  # parasitic drag
    k = 1  # See Slotine for this, no idea
    # TODO find a good value for k
    wingspan = 2
    m = 2  # mass
    g = 9.81  # gravity

    V, psi, gamma, z_pos, x_pos, y_pos = state
    # glider_speed, heading, flight_path_angle, z, x, y
    c_l, phi = u
    # u = [lift_coeff, roll_angle]

    # D = drag
    # L = lift
    # W = wind speed

    c_d = c_d0 + k * c_l ** 2
    L = 0.5 * c_l * rho * S * V ** 2
    D = 0.5 * c_d * rho * S * V ** 2

    W = get_wind(z_pos)
    W_dot = 0  # TODO I think this is correct

    V_dot = (1 / m) * (
        -D - m * g * me.sin(gamma) - m * W_dot * me.cos(gamma) * me.sin(psi)
    )

    gamma_dot = (1 / (m * V)) * (
        L * me.cos(phi)
        - m * g * me.cos(gamma)
        - m * W_dot * me.sin(gamma) * me.sin(psi)
    )

    psi_dot = (1 / (m * V * me.cos(gamma))) * (
        L * me.sin(phi) + m * W_dot * me.cos(psi)
    )  # NOTE gamma != pi/2

    z_dot = V * me.sin(gamma)
    x_dot = V * me.cos(gamma) * me.cos(psi)
    y_dot = V * me.cos(gamma) * me.sin(psi) - W

    # Dynamics
    state_dot = np.empty(6, dtype=Expression)
    state_dot[0] = V_dot
    state_dot[1] = psi_dot
    state_dot[2] = gamma_dot
    state_dot[3] = z_dot
    state_dot[4] = x_dot
    state_dot[5] = y_dot

    return state_dot


def get_wind(z):
    W0 = 2  # Free stream wind speed
    delta = 3  # wind_shear_layer thickness
    W = W0 / (1 + np.exp(-z / delta))
    return W


# Assume wind blows from north to south, i.e. along -y axis
def get_wind_field(x, y, z):
    u = np.zeros(y.shape)
    v = -get_wind(z)
    w = np.zeros(z.shape)

    return u, v, w
