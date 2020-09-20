from test.test_dynamics import *
from trajopt.nonlin_trajopt import *


def main():
    #direct_collocation()
    #plant = ZhukovskiiGlider()
    #simulate_drake_system(plant)
    direct_collocation_zhukovskii_glider()


    return 0


if __name__ == "__main__":
    main()

