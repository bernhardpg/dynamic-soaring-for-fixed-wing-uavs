import numpy as np
from pydrake.all import eq, MathematicalProgram, Solve, Variable, Expression


def continuous_dynamics(x, u):
    # Constants
    air_density = 1.255
    wing_area = 1.5
    parasitic_drag = 1
    wingspan = 2
    mass = 2

    # Dynamics
    x_dot = np.empty(6, dtype=Expression)

    circulation = u
    pos = x[0:3]
    height = pos[2]
    vel = x[3:6]

    wind = np.array([get_wind(height), 0, 0])
    #     if height <= 0: # No wind below ground
    #         wind = np.array([0,0,0])
    rel_vel = vel - wind

    circ_squared_norm = (
        pow(circulation[0], 2) + pow(circulation[1], 2) + pow(circulation[2], 2)
    )

    x_dot[0:3] = vel
    x_dot[3:6] = (1 / mass) * (
        air_density * np.cross(circulation, rel_vel)
        - 0.5
        * air_density
        * wing_area
        * parasitic_drag
        * np.sqrt(rel_vel.T.dot(rel_vel) + 0.001)
        * rel_vel
        - (2 * air_density / np.pi)
        * (circ_squared_norm / wingspan ** 2)
        * rel_vel
        / np.sqrt(rel_vel.T.dot(rel_vel) + 0.001)
        + mass * np.array([0, 0, -9.81])
    )

    return x_dot


def get_wind(height):
    # TODO set parameters here
    ref_height = 10
    alpha = 2

    # TODO replace u0
    u0 = 5

    return u0 * (height / ref_height) ** alpha


def get_wind_field(x, y, z):
    u = get_wind(z)
    v = np.zeros(y.shape)
    w = np.zeros(z.shape)

    return u, v, w
