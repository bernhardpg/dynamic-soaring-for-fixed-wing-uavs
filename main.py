from trajopt.nonlin_trajopt import *
from trajopt.fourier_collocation import *
from plot.plot import *
from trajopt.direct_collocation import *


def main():
    #single_dircol_w_real_values()
    do_single_dircol(psi=1 * np.pi)
    # do_dircol()
    return 0

def calc_glider_values():
    # Physical parameters
    m = 8.5
    c_Dp = 0.033
    A = 0.65
    b = 3.306
    rho = 1.255  # g/m**3 Air density
    g = 9.81
    AR = b ** 2 / A

    zhukovskii_glider = ZhukovskiiGlider()
    Lam = zhukovskii_glider.calc_opt_glide_ratio(AR, c_Dp)
    Th = zhukovskii_glider.calc_opt_glide_angle(AR, c_Dp)
    V_opt = zhukovskii_glider.calc_opt_glide_speed(AR, c_Dp, m, A, b, rho, g)

    print("Lam: {0}\nTh: {1}\nV_opt: {2}".format(Lam, Th, V_opt))
    return


def single_dircol_w_real_values():
    # Physical parameters
    m = 8.5
    c_Dp = 0.033
    A = 0.65
    b = 3.306
    rho = 1.255  # g/m**3 Air density
    g = 9.81
    AR = b ** 2 / A

    zhukovskii_glider = ZhukovskiiGlider()
    Lam = zhukovskii_glider.calc_opt_glide_ratio(AR, c_Dp)
    Th = zhukovskii_glider.calc_opt_glide_angle(AR, c_Dp)
    V_opt = zhukovskii_glider.calc_opt_glide_speed(AR, c_Dp, m, A, b, rho, g)

    print("Running dircol with:")
    print("\tLam: {0}\n\tTh: {1}\n\tV_opt: {2}".format(Lam, Th, V_opt))

    psi = np.pi

    avg_speed, traj, curr_solution = direct_collocation(
        zhukovskii_glider, psi, PLOT_SOLUTION=False
    )

    return


def do_single_dircol(psi):
    PLOT_SOLUTION = True
    zhukovskii_glider = ZhukovskiiGlider()
    avg_speed, traj, curr_solution = direct_collocation(
        zhukovskii_glider, psi, PLOT_SOLUTION=False
    )

    times, x_knots, u_knots = traj

    if PLOT_SOLUTION:
        plot_glider_pos(x_knots[:, 0:3], psi)
        plot_glider_input(times, u_knots)
        plt.show()

        if False:
            # TODO continue on this later: ensure that c_l and AoA is realistic!
            # TODO continue on this after asking morten
            c_l_knots = np.zeros((200,))
            for k in range(len(times)):
                c_l = zhukovskii_glider.get_lift_coeff(x_knots[k, :], u_knots[k, :])
                c_l_knots[k] = c_l * (180 / np.pi)
    return


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
    main()
