from trajopt.nonlin_trajopt import *
from trajopt.dircol_fourier import *
from plot.plot import *

def main():
    #zhukovskii_glider_w_dircol()
    prog = DirColFourierProblem()


    return 0

def zhukovskii_glider_w_dircol():
    zhukovskii_glider = ZhukovskiiGlider()

    N_angles = 100
    travel_angles = np.linspace(0, 2 * np.pi, N_angles)
    #travel_angles = np.array([np.pi/2 + 0.3])

    avg_velocities = dict()
    trajectories = dict()
    print("### Running with straight line as initial guess")
    for psi in travel_angles:
        avg_speed, traj, curr_solution = direct_collocation(
            zhukovskii_glider, psi, plot_solution=False
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
    for travel_angle in travel_angles:
        animate_trajectory_gif(zhukovskii_glider, trajectories[travel_angle], travel_angle)

    return 0


if __name__ == "__main__":
    main()
