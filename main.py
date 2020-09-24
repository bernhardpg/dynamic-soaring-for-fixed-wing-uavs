from test.test_dynamics import *
from trajopt.nonlin_trajopt import *
from dynamics.zhukovskii_glider import ZhukovskiiGliderDimless


def main():
    #direct_collocation()
    #plant = ZhukovskiiGliderDimless()
    #simulate_drake_system(plant)
    direct_collocation()
    #direct_collocation_zhukovskii_glider()


    return 0


if __name__ == "__main__":
    main()

