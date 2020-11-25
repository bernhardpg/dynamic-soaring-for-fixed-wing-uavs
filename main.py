import sys

from dynamics.zhukovskii_glider import *
from trajopt.nonlin_trajopt import *
from trajopt.fourier_collocation import *
from plot.plot import *
from trajopt.direct_collocation import *
from dynamics.wind_models import *
from analysis.energy_analysis import energy_analysis


def main(argv):
    # Parse command line args
    travel_angle = float(argv[1]) * np.pi / 180 if len(argv) > 1 else np.pi
    period_guess = float(argv[2]) if len(argv) > 1 else 4
    avg_vel_scale_guess = float(argv[3]) if len(argv) > 1 else 1
    plot_axis = argv[4] if len(argv) > 1 else "x"

    # Physical parameters
    m = 8.5
    c_Dp = 0.033
    A = 0.65
    b = 3.306
    rho = 1.255  # g/m**3 Air density
    g = 9.81
    AR = b ** 2 / A
    phys_params = (m, c_Dp, A, b, rho, g, AR)

    #    calc_trajectory(
    #        phys_params,
    #        travel_angle,
    #        period_guess,
    #        avg_vel_scale_guess,
    #        plot_axis,
    #    )

    sweep_calculation(phys_params)
    return 0


def sweep_calculation(phys_params):
    (m, c_Dp, A, b, rho, g, AR) = phys_params
    zhukovskii_glider = RelativeZhukovskiiGlider(m, c_Dp, A, b, rho, g)
    V_l = zhukovskii_glider.calc_opt_level_glide_speed(AR, c_Dp, m, A, b, rho, g)

    n_angles = 360
    angle_increment = 2 * np.pi / n_angles
    psi_start = np.pi / 2

    travel_angles = np.hstack(
        [
            np.arange(psi_start, 2 * np.pi, angle_increment),
            np.arange(0, psi_start, angle_increment),
        ]
    )

    solution_avg_speeds = dict()
    solution_periods = dict()

    # File handling
    import json

    # f.read(json.loads(all_avg_velocities))

    # Obtain an initial guess
    period = 4
    avg_speed = 2 * V_l
    (
        solution_details,
        solution_trajectory,
        next_initial_guess,
    ) = direct_collocation_relative(
        zhukovskii_glider,
        travel_angles[0],
        period_guess=period,
        avg_vel_guess=avg_speed,
    )
    if solution_details == None:
        solution_avg_speeds[travel_angles[0]] = -1
        solution_periods[travel_angles[0]] = -1
    else:
        avg_speed, period = solution_details
        solution_avg_speeds[travel_angles[0]] = avg_speed
        solution_periods[travel_angles[0]] = period

    # Run a sweep search
    for travel_angle in travel_angles[1:]:
        (
            solution_details,
            solution_trajectory,
            next_initial_guess,
        ) = direct_collocation_relative(
            zhukovskii_glider,
            travel_angle,
            period_guess=period,
            avg_vel_guess=avg_speed,
            initial_guess=next_initial_guess,
        )
        if solution_details == None:
            solution_avg_speeds[travel_angle] = -1
            solution_periods[travel_angle] = -1
        else:
            avg_speed, period = solution_details
            solution_avg_speeds[travel_angle] = avg_speed
            solution_periods[travel_angle] = period

        with open("./results/sweep_results_speeds", "w+") as f:
            f.write(json.dumps(solution_avg_speeds))
            f.close()

        with open("./results/sweep_results_periods", "w+") as f:
            f.write(json.dumps(solution_periods))
            f.close()

    return


def calc_trajectory(
    phys_params,
    travel_angle=0,
    period_guess=4,
    avg_vel_scale_guess=1,
    plot_axis="x",
):

    (m, c_Dp, A, b, rho, g, AR) = phys_params
    zhukovskii_glider = RelativeZhukovskiiGlider(m, c_Dp, A, b, rho, g)

    # Print performance params
    Lam = zhukovskii_glider.calc_opt_glide_ratio(AR, c_Dp)
    Th = zhukovskii_glider.calc_opt_glide_angle(AR, c_Dp)
    V_opt = zhukovskii_glider.calc_opt_glide_speed(AR, c_Dp, m, A, b, rho, g)
    V_l = zhukovskii_glider.calc_opt_level_glide_speed(AR, c_Dp, m, A, b, rho, g)

    print("Running dircol with:")
    print("\tLam: {0}\n\tTh: {1}\n\tV_opt: {2}\n\tV_l: {3}".format(Lam, Th, V_opt, V_l))

    avg_speed, traj = direct_collocation_relative(
        zhukovskii_glider,
        travel_angle,
        period_guess=period_guess,
        avg_vel_scale_guess=avg_vel_scale_guess,
    )

    times, x_knots, u_knots = traj

    (
        phi_knots,
        gamma_knots,
        psi_knots,
        c_l_knots,
        n_knots,
    ) = _calc_phys_values_from_traj(zhukovskii_glider, phys_params, x_knots, u_knots)

    soaring_power = energy_analysis(times, x_knots, u_knots, phys_params)

    plot_glider_pos(
        x_knots,
        u_knots,
        travel_angle,
        draw_soaring_power=True,
        soaring_power=soaring_power,
        plot_axis=plot_axis,
    )
    plt.show()
    plot_glider_angles(times, gamma_knots, phi_knots, psi_knots)
    plot_glider_input(times, u_knots, c_l_knots, phi_knots, n_knots)

    plt.close()
    return


def _calc_phys_values_from_traj(zhukovskii_glider, phys_params, x_knots, u_knots):
    (m, c_Dp, A, b, rho, g, AR) = phys_params
    c_knots = u_knots  # Circulation

    # Calculate bank angle
    phi_knots = np.zeros((x_knots.shape[0], 1))
    for k in range(x_knots.shape[0]):
        v_r = x_knots[k, 3:6]
        c = u_knots[k, :]
        phi = zhukovskii_glider.calc_bank_angle(v_r, c)
        phi_knots[k] = phi

    # Calculate relative flight path angle
    gamma_knots = np.zeros((x_knots.shape[0], 1))
    for k in range(x_knots.shape[0]):
        h = x_knots[k, 2]
        v_r = x_knots[k, 3:6]
        gamma = zhukovskii_glider.calc_rel_flight_path_angle(v_r)
        gamma_knots[k] = gamma

    # Calculate heading angle
    psi_knots = np.zeros((x_knots.shape[0], 1))
    for k in range(x_knots.shape[0]):
        h = x_knots[k, 2]
        v_r = x_knots[k, 3:6]
        psi = zhukovskii_glider.calc_heading(h, v_r)
        psi_knots[k] = psi

    # Calculate lift coeff
    c_l_knots = np.zeros((x_knots.shape[0], 1))
    for k in range(x_knots.shape[0]):
        v_r = x_knots[k, 3:6]
        c = u_knots[k, :]
        c_l = zhukovskii_glider.calc_lift_coeff(v_r, c, A)
        c_l_knots[k] = c_l

    # Calculate load factor
    n_knots = np.zeros((x_knots.shape[0], 1))
    for k in range(x_knots.shape[0]):
        v_r = x_knots[k, 3:6]
        c = u_knots[k, :]
        n = zhukovskii_glider.calc_load_factor(v_r, c, m, g, rho)
        n_knots[k] = n

    return (
        phi_knots,
        gamma_knots,
        psi_knots,
        c_l_knots,
        n_knots,
    )


# TODO OLD from here


def do_sweep_dircol():
    zhukovskii_glider = ZhukovskiiGlider()

    # Program parameters
    SAVE_ANIMATION = False
    N_ANGLES = 100
    START_ANGLE = 0

    # Save trajectories and values
    avg_velocities = dict()
    trajectories = dict()

    # First run with straight line as initial guess
    travel_angles = np.linspace(START_ANGLE, START_ANGLE + 2 * np.pi, N_ANGLES)

    print("### Running with straight line as initial guess")
    for psi in travel_angles:
        avg_speed, traj, curr_solution = direct_collocation(
            zhukovskii_glider, psi, PLOT_SOLUTION=False
        )
        trajectories[psi] = traj
        avg_velocities[psi] = avg_speed

    # polar_plot_avg_velocities(avg_velocities)

    if True:
        print("### Running twice with proximate solution as initial guess")
        double_travel_angles = np.concatenate((travel_angles, travel_angles))
        prev_solution = None

        for psi in double_travel_angles:
            avg_speed, traj, curr_solution = direct_collocation(
                zhukovskii_glider, psi, initial_guess=prev_solution
            )

            if avg_speed > avg_velocities[psi]:
                avg_velocities[psi] = avg_speed
                trajectories[psi] = traj

            prev_solution = curr_solution

        print("### Running twice with proximate solution as initial guess (other way)")

        for psi in np.flip(double_travel_angles):
            avg_speed, traj, curr_solution = direct_collocation(
                zhukovskii_glider, psi, initial_guess=prev_solution
            )

            if avg_speed > avg_velocities[psi]:
                avg_velocities[psi] = avg_speed
                trajectories[psi] = traj

            prev_solution = curr_solution

    print("### Finished!")
    # plot_trajectories(trajectories)
    polar_plot_avg_velocities(avg_velocities)
    if SAVE_ANIMATION:
        for travel_angle in travel_angles:
            save_trajectory_gif(
                zhukovskii_glider, trajectories[travel_angle], travel_angle
            )

    return 0


def do_collocation_w_fourier():
    zhukovskii_glider = ZhukovskiiGlider()
    prog = FourierCollocationProblem(
        zhukovskii_glider.continuous_dynamics_dimless,
        zhukovskii_glider.get_constraints_dimless(),
    )
    prog.get_solution()
    return


if __name__ == "__main__":
    main(sys.argv)
