import numpy as np


# Wind models
def linear_wind_model(z):  # Taken from Deittert et al.
    W0 = 16  # m/s
    p = 0.143
    h_r = 5  # ref height, can be set arbritrary
    W = W0 * (z / h_r) ** p  # wind strength

    return W


def ddt_linear_wind_model(z, z_dot):
    W0 = 16  # m/s
    p = 0.143
    h_r = 5  # ref height, can be set arbritrary
    W_dot = ((p * W0) / z) * (z / h_r) ** p * z_dot
    return W_dot


def exp_wind_model(z):  # Taken from slotine
    W0 = 16  # Free stream wind speed
    delta = 5  # wind_shear_layer thickness
    W = W0 / (1 + np.exp(-z / delta))
    return W


def ddt_exp_wind_model(z, z_dot):
    W0 = 16  # Free stream wind speed
    delta = 5  # wind_shear_layer thickness
    W_dot = (W0 * np.exp(-z / delta) * z_dot) / (delta * (1 + np.exp(-z / delta)) ** 2)
    return W_dot


# Assume wind blows from north to south, i.e. along negative y axis
def get_wind_field(x, y, z):
    u = np.zeros(x.shape)
    v = -wind_model(z)
    w = np.zeros(z.shape)

    return u, v, w


def get_wind_vector(z):
    W_vec = np.array([0, -wind_model(z), 0])
    return W_vec


#wind_model = exp_wind_model
#ddt_wind_model = ddt_exp_wind_model

wind_model = lambda x: 0
ddt_wind_model = lambda x, y: 0
