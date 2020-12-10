import sys
import json

from trajopt.trajectory_generator import *
from plot.plot import *


def main(argv):
    # TODO clean up command line parsing
    # Parse command line args
    travel_angle = float(argv[1]) * np.pi / 180 if len(argv) > 1 else None
    period_guess = float(argv[2]) if len(argv) > 2 else 8
    avg_vel_scale_guess = float(argv[3]) if len(argv) > 3 else 1
    plot_axis = argv[4] if len(argv) > 4 else ""
    run_once = not travel_angle == None

    # Physical parameters
    m = 8.5
    c_Dp = 0.033
    A = 0.65
    b = 3.306
    rho = 1.255  # g/m**3 Air density
    g = 9.81
    AR = b ** 2 / A
    phys_params = (m, c_Dp, A, b, rho, g, AR)

    if run_once:
        calc_and_plot_trajectory(
            phys_params,
            travel_angle,
            period_guess,
            avg_vel_scale_guess,
            plot_axis,
        )

    else:
        sweep_calculation_for_period(phys_params, 8, n_angles=9)
        # sweep_calculation_old(phys_params)

        show_sweep_result()
        plt.show()
    return 0


if __name__ == "__main__":
    main(sys.argv)
