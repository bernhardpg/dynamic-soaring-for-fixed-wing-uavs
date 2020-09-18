import numpy as np
from plot.plot import plot_trj_3_wind


def rollout_slotine_dynamics():
    from dynamics.slotine_dynamics import get_wind_field
    from dynamics.slotine_dynamics import continuous_dynamics

    N = 500
    m = 6

    x_trj = np.zeros((N, m))
    x0 = np.array([10, -np.pi / 2, 0, 40, 0, 0])
    x_trj[0, :] = x0
    u = np.array([0.5, 0])

    dt = 0.01
    for n in range(N - 1):
        x_next = x_trj[n, :] + dt * continuous_dynamics(x_trj[n, :], u)
        if x_next[3] < 0:
            break
        x_trj[n + 1, :] = x_next

    # Slotine dynamics: x = [airspeed, heading, flight_path_angle, z, x, y]
    z = x_trj[:, 3]
    x = x_trj[:, 4]
    y = x_trj[:, 5]

    plot_trj_3_wind(np.vstack((x, y, z)).T, get_wind_field)
    return
