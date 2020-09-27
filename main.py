from trajopt.nonlin_trajopt import *
from dynamics.zhukovskii_glider import get_sim_params
from plot.plot import *


def main():
    N_angles = 20
    travel_angles = np.linspace(0, 2 * np.pi, N_angles)

    sim_params = get_sim_params()

    avg_velocities = dict()
    solutions = dict()
    print("### Running with straight line as initial guess")
    for psi in travel_angles:
        avg_speed, curr_solution = direct_collocation(psi, sim_params)
        solutions[psi] = curr_solution
        avg_velocities[psi] = avg_speed

    polar_plot_avg_velocities(avg_velocities)

    print("### Running twice with proximate solution as initial guess")
    double_travel_angles = np.concatenate((travel_angles, travel_angles))
    prev_solution = None
    for psi in travel_angles:
        avg_speed, curr_solution = direct_collocation(
            psi, sim_params, initial_guess=prev_solution
        )

        if avg_speed > avg_velocities[psi]:
            avg_velocities[psi] = avg_speed
            solutions[psi] = curr_solution

        prev_solution = curr_solution

    polar_plot_avg_velocities(avg_velocities)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
