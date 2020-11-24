import numpy as np
from dynamics.wind_models import wind_model, ddt_wind_model, get_wind_vector
from plot.plot import *


def _calc_abs_vel(h, v_r, w):
    v = v_r + w
    return v


def _calc_winds(h):
    N = h.shape[0]
    w_value = wind_model(h)
    w = np.vstack((np.zeros(N), -w_value, np.zeros(N))).T
    return w


def _calc_ddt_winds(h, h_dot):
    N = h.shape[0]
    ddt_w_value = ddt_wind_model(h, h_dot)
    ddt_w = np.vstack((np.zeros(N), -ddt_w_value, np.zeros(N))).T
    return ddt_w


def _generate_finite_diff_matrix_third_order(N, dt):
    D = (
        -11 / 6 * np.diag(np.ones(N), 0)
        + 3 * np.diag(np.ones((N - 1)), 1)
        - 3 / 2 * np.diag(np.ones((N - 2)), 2)
        + 1 / 3 * np.diag(np.ones((N - 3)), 3)
    ) / dt
    return D


def _calc_energy(h, v, m, g):
    N = h.shape[0]
    v_squared = np.diag(v.dot(v.T))
    E_kin = 0.5 * m * v_squared
    E_pot = m * g * h
    return E_kin, E_pot


def _calc_drag_param(v_r, c, c_Dp, A, AR):
    v_r_norm = np.sqrt(np.diag(v_r.dot(v_r.T)))
    c_squared = np.diag(c.dot(c.T))
    d = 0.5 * A * v_r_norm * c_Dp + (2 * c_squared) / (np.pi * AR * A * v_r_norm)
    return d


def _calc_dissipation_power(v_r, d, rho):
    v_r_squared = np.diag(v_r.dot(v_r.T))
    P_dissipated = -rho * d * v_r_squared
    return P_dissipated


def _calc_gained_power(v, w, ddt_w, D, m):
    v_T_w = np.diag(v.dot(w.T))
    S_dyn_passive = m * D.dot(v_T_w)
    S_dyn_active = -m * np.diag(v.dot(ddt_w.T))

    return S_dyn_passive, S_dyn_active


def _calc_integral(graph, dt):
    N = graph.shape[0]
    Riemann_matrix = np.tril(np.ones(N)) * dt
    integral = Riemann_matrix.dot(graph)  # Use Riemann sums to calculate integral
    return integral


def energy_analysis(times, x_traj, u_traj, phys_params):
    print("### Running energy analysis")

    (m, c_Dp, A, b, rho, g, AR) = phys_params
    dt = times[1] - times[0]
    N = x_traj.shape[0]

    # Derivative matrix
    D = _generate_finite_diff_matrix_third_order(N, dt)

    # Generate all needed trajectories
    h = x_traj[:, 2]
    v_r = x_traj[:, 3:6]
    c = u_traj

    w = _calc_winds(h)
    v = _calc_abs_vel(h, v_r, w)
    h_dot = v[:, 2]
    ddt_w = _calc_ddt_winds(h, h_dot)
    d = _calc_drag_param(v_r, c, c_Dp, A, AR)

    # Calculate energies
    E_kin, E_pot = _calc_energy(h, v, m, g)
    E_tot = E_kin + E_pot

    # Calculate powers
    P_tot = D.dot(E_tot)
    P_dissipated = _calc_dissipation_power(v_r, d, rho)
    S_dyn_passive, S_dyn_active = _calc_gained_power(v, w, ddt_w, D, m)
    P_gained = S_dyn_passive + S_dyn_active

    # Integrate powers to look at energy
    E_dissipated = _calc_integral(P_dissipated, dt)
    E_dyn_active = _calc_integral(S_dyn_active, dt)
    E_dyn_passive = _calc_integral(S_dyn_passive, dt)

    # PLOTTING
    plot_energies(times, E_tot, E_kin, E_pot)
    plot_powers(times[:-3], P_tot[:-3], P_dissipated[:-3], P_gained[:-3])
    plot_power_terms(
        times[:-3],
        P_dissipated[:-3],
        S_dyn_active[:-3],
        S_dyn_passive[:-3],
        E_dissipated[:-3],
        E_dyn_active[:-3],
        E_dyn_passive[:-3],
    )
    plt.show()


# TODO Old shit, remove!
def _calc_energy_analysis(zhukovskii_glider, dt, phys_params, x_knots, u_knots):
    N = x_knots.shape[0]
    (m, c_Dp, A, b, rho, g, AR) = phys_params
    c_knots = u_knots  # Circulation
    # Calculate total energy
    E_knots = np.zeros((N, 1))
    for k in range(N):
        h = x_knots[k, 2]
        v_r = x_knots[k, 3:6]
        c = u_knots[k, :]
        E = zhukovskii_glider.calc_total_energy(h, v_r, m, g)
        E_knots[k] = E

    # Calculate dissipated energy (Power)
    D_knots = np.zeros((N, 1))
    for k in range(N):
        v_r = x_knots[k, 3:6]
        c = u_knots[k, :]
        D = zhukovskii_glider.calc_dissipated_energy(v_r, c, c_Dp, A, AR, rho)
        D_knots[k] = D

    # Calculate gained dynamic soaring energy (Power)
    S_dyn_2_knots = np.zeros((N, 1))
    for k in range(N):
        v_r = x_knots[k, 3:6]
        c = u_knots[k, :]
        S_dyn = zhukovskii_glider.calc_dynamic_soaring_energy_gain(h, v_r, m)
        S_dyn_2_knots[k] = S_dyn

    # Calculate soaring power
    wind_knots = np.zeros((N, 3))
    for k in range(N):
        h = x_knots[k, 2]
        w = get_wind_vector(h)
        wind_knots[k] = w

    abs_vel_knots = np.zeros((N, 3))
    for k in range(N):
        h = x_knots[k, 2]
        v_r = x_knots[k, 3:6]
        v = zhukovskii_glider.calc_abs_vel(h, v_r)
        abs_vel_knots[k, :] = v

    v_T_w = np.diag(abs_vel_knots.dot(wind_knots.T)).reshape((N, 1))
    N = E_knots.shape[0]
    third_order_finite_diff_matrix = (
        -11 / 6 * np.diag(np.ones(N), 0)
        + 3 * np.diag(np.ones((N - 1)), 1)
        - 3 / 2 * np.diag(np.ones((N - 2)), 2)
        + 1 / 3 * np.diag(np.ones((N - 3)), 3)
    ) / dt

    S_dyn_1_knots = m * third_order_finite_diff_matrix.T.dot(v_T_w)

    return E_knots, D_knots, S_dyn_1_knots, S_dyn_2_knots
