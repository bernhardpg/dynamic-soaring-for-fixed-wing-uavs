from test.test_dynamics import *
from trajopt.nonlin_trajopt import *
from dynamics.zhukovskii_glider import ZhukovskiiGliderDimless


def main():
    # direct_collocation()
    # plant = ZhukovskiiGliderDimless()
    # simulate_drake_system(plant)
    # direct_collocation_slotine_glider()
    # direct_collocation_zhukovskii_glider()

    travel_angles = np.linspace(0, 2 * np.pi, 41)
    travel_distances = []
    for psi in travel_angles:
        travel_distances.append(direct_collocation(psi))

    ax = plt.subplot(111, projection='polar')
    ax.plot(travel_angles, travel_distances)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
