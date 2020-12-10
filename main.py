import sys, getopt
from trajopt.trajectory_generator import *


def main(argv):
    # Default program arguments
    travel_angle = np.pi / 2
    period_guess = 8
    avg_vel_scale_guess = 2
    run_once = True

    # Command line parsing
    try:
        opts, args = getopt.getopt(
            argv, "a:p:v:s", ["angle=", "period=", "velocity=", "sweep"]
        )
    except getopt.GetoptError:
        print(
            "main.py -a <travel_angle> -p <period_guess> -v <velocity_guess> -s <do_a_sweep?>"
        )
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(
                "main.py -a <travel_angle> -p <period_guess> -v <velocity_guess> -s <do_a_sweep?>"
            )
            sys.exit()
        elif opt in ("-a", "--angle"):
            travel_angle = float(arg) * np.pi / 180
        elif opt in ("-p", "--period"):
            period_guess = float(arg)
        elif opt in ("-v", "--velocity"):
            avg_vel_scale_guess = float(arg)
        elif opt in ("-s", "--sweep"):
            run_once = False

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
            plot_axis="",
        )

    else:
        sweep_calculation_for_period(phys_params, travel_angle, period_guess, n_angles=9)

        show_sweep_result()
        plt.show()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
