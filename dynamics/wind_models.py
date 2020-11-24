import numpy as np

w_ref = 10  # m/s
w_freestream = w_ref
h_ref = 10  # m
h_0 = 0.03  # m


########
# Linear wind model
########

def linear_wind_model(z):
    w = w_ref / h_ref * z
    return w



########
# Exponential wind model
########


def exp_wind_model(z):  # Taken from Deittert et al.
    p = 0.143
    w = w_ref * (z / h_ref) ** p  # wind strength

    return w


def ddt_exp_wind_model(z, z_dot):
    p = 0.143
    w_dot = ((p * w_ref) / z) * (z / h_ref) ** p * z_dot
    return w_dot


########
# Logarithmic wind model
########


def log_wind_model(z):
    # Check for type for plotting. NOTE will fail if plotter has z<h_0
    if (not type(z) == type(np.array(1))) and z < h_0:
        return 0  # NOTE zero wind below ground
    w = w_ref * (np.log(z / h_0)) / (np.log(h_ref / h_0))
    return w


def ddz_log_wind_model(z):
    dw_dz = w_ref / (np.log(h_ref / h_0) * z)
    if z < h_0:
        dw_dz = 0  # NOTE zero wind below ground
    return dw_dz


def ddt_log_wind_model(z, z_dot):
    dw_dz = ddz_log_wind_model
    dw_dt = dw_dz * z_dot
    return dw_dt


########
# Logistic wind model
########


def logistic_wind_model(z):  # Taken from slotine
    delta = 3  # wind_shear_layer thickness
    w = w_freestream / (1 + np.exp(-z / delta))
    return w


def ddz_logistic_wind_model(z):
    delta = 3  # wind_shear_layer thickness
    dw_dz = (w_freestream * np.exp(-z / delta)) / (
        delta * (1 + np.exp(-z / delta)) ** 2
    )
    return dw_dz


def ddt_logistic_wind_model(z, z_dot):
    dw_dz = ddz_logistic_wind_model(z)
    w_dot = dw_dz * z_dot
    return w_dot


########
# General functions
########


def get_wind_vector(z):
    w_vec = np.array([0, -wind_model(z), 0])
    return w_vec


def get_wind_jacobian(z):
    dw_dz = ddz_wind_model(z)
    dw_dx = np.array([[0, 0, 0], [0, 0, -dw_dz], [0, 0, 0]])
    return dw_dx


wind_model = log_wind_model
ddz_wind_model = ddz_log_wind_model
ddt_wind_model = ddt_log_wind_model

# PLOTTING FUNCTIONs

# Assume wind blows from north to south, i.e. along negative y axis
def get_wind_field(x, y, z):
    u = np.zeros(x.shape)
    v = -wind_model(z)
    w = np.zeros(z.shape)

    return u, v, w
