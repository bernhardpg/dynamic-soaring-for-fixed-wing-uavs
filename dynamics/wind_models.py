import numpy as np


def exp_wind_model(z):  # Taken from Deittert et al.
    w_ref = 16  # m/s
    p = 0.143
    h_r = 5  # ref height, can be set arbritrary
    w = w_ref * (z / h_r) ** p  # wind strength

    return w


def ddt_exp_wind_model(z, z_dot):
    w_ref = 16  # m/s
    p = 0.143
    h_r = 5  # ref height, can be set arbritrary
    w_dot = ((p * w_ref) / z) * (z / h_r) ** p * z_dot
    return w_dot


def log_wind_model(z):
    w_ref = 10  # m/s
    h_ref = 10  # m
    h_0 = 0.03  # m
    if z < h_0: return 0 # NOTE zero wind below ground
    w = w_ref * (np.log(z / h_0)) / (np.log(h_ref / h_0))
    return w


def ddt_log_wind_model(z, z_dot):
    dw_dz = w_ref / (np.log(h_ref / h_0) * z)
    if z < h_0: dw_dz = 0 # NOTE zero wind below ground
    w = w_ref * (np.log(z / h_0)) / (np.log(h_ref / h_0))
    dw_dt = dw_dz * z_dot
    return dw_dt


def logistic_wind_model(z):  # Taken from slotine
    w_freestream = 16  # Free stream wind speed
    delta = 5  # wind_shear_layer thickness
    w = w_freestream / (1 + np.exp(-z / delta))
    return w


def ddz_logistic_wind_model(z):
    w_freestream = 16  # Free stream wind speed
    delta = 5  # wind_shear_layer thickness
    dw_dz = (w_freestream * np.exp(-z / delta)) / (delta * (1 + np.exp(-z / delta)) ** 2)
    return dw_dz


def ddt_logistic_wind_model(z, z_dot):
    dw_dz = ddz_logistic_wind_model(z)
    w_dot = dw_dz * z_dot
    return w_dot


# TODO currently only used for plotting
# Assume wind blows from north to south, i.e. along negative y axis
def get_wind_field(x, y, z):
    u = np.zeros(x.shape)
    v = np.zeros(y.shape)
    #v = -wind_model(z)
    w = np.zeros(z.shape)

    return u, v, w


def get_wind_vector(z):
    w_vec = np.array([0, -wind_model(z), 0])
    return w_vec


def get_wind_jacobian(z):
    dw_dz = ddz_logistic_wind_model(z)
    dw_dx = np.array([[0, 0, 0], [0, 0, dw_dz], [0, 0, 0]])
    return dw_dx


#wind_model = logistic_wind_model
#ddt_wind_model = ddt_logistic_wind_model

wind_model = log_wind_model
ddt_wind_model = ddt_log_wind_model

#wind_model = exp_wind_model
#ddt_wind_model = ddt_exp_wind_model

def get_dimless_wind_vector(z, L, V_l):
    return get_wind_vector(L * z) / V_l
