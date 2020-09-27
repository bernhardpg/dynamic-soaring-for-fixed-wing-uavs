from trajopt.nonlin_trajopt import *
from plot.plot import *


def main():
    N_angles = 1
    travel_angles = np.linspace(0, 2 * np.pi, N_angles)

    avg_velocities = dict()
    trajectories = dict()
    print("### Running with straight line as initial guess")
    for psi in travel_angles:
        avg_speed, traj, curr_solution = direct_collocation(
            psi, plot_solution=False
        )
        trajectories[psi] = traj
        avg_velocities[psi] = avg_speed

    # polar_plot_avg_velocities(avg_velocities)

    if False:
        print("### Running twice with proximate solution as initial guess")
        double_travel_angles = np.concatenate((travel_angles, travel_angles))
        prev_solution = None
        for psi in travel_angles:
            avg_speed, traj, curr_solution = direct_collocation(
                psi, initial_guess=prev_solution
            )

            if avg_speed > avg_velocities[psi]:
                avg_velocities[psi] = avg_speed
                trajectories[psi] = traj

            prev_solution = curr_solution

        # polar_plot_avg_velocities(avg_velocities)

    print("### Finished!")
    #plot_trajectories(trajectories)
    plt.show()
    animate_trajectory(trajectories[0.0])

    return 0


if __name__ == "__main__":
    main()
