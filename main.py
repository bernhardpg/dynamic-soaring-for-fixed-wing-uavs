from test.test_dynamics import *
from trajopt.nonlin_trajopt import *


def main():
    #    N_angles = 7
    #    splice_point = int(N_angles * (2 / 4))
    #    travel_angles = np.linspace(0, 2 * np.pi, N_angles)
    #    travel_angles = np.concatenate(
    #        (
    #            travel_angles[splice_point:N_angles],
    #            travel_angles[0:splice_point],
    #        )
    #    )
    #    breakpoint()

    N_angles = 20
    travel_angles = np.linspace(0, 2 * np.pi, N_angles)

    travel_speeds = dict()
    initial_guess = None
    print("### Running with straight line as initial guess")
    for psi in travel_angles:
        print("*** Solving DirCol for travel_angle: {0}".format(psi))
        avg_speed, initial_guess = direct_collocation(
            psi, initial_guess=None, plot_solution=False
        )
        travel_speeds[psi] = avg_speed

    lists = sorted(travel_speeds.items())
    x, y = zip(*lists)
    ax = plt.subplot(111, projection="polar")
    ax.plot(x, y)

    print("### Running with proximate solution as initial guess")
    for psi in travel_angles:
        print("*** Solving DirCol for travel_angle: {0}".format(psi))
        avg_speed, initial_guess = direct_collocation(
            psi, initial_guess, plot_solution=False
        )
        travel_speeds[psi] = max(avg_speed, travel_speeds[psi])
    print("### Running again with proximate solution as initial guess")
    for psi in travel_angles:
        print("*** Solving DirCol for travel_angle: {0}".format(psi))
        avg_speed, initial_guess = direct_collocation(
            psi, initial_guess, plot_solution=False
        )
        travel_speeds[psi] = max(avg_speed, travel_speeds[psi])

    # PLOTTING
    lists = sorted(travel_speeds.items())
    x, y = zip(*lists)
    ax = plt.subplot(111, projection="polar")
    ax.plot(x, y)
    ax.set_title("Achievable speeds")
    plt.show()

    return 0


if __name__ == "__main__":
    main()
