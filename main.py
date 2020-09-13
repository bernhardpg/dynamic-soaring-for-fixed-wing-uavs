import numpy as np
from pydrake.all import eq, MathematicalProgram, Solve, Variable, Expression
from test.test_dynamics import *

def main():
    rollout_slotine_dynamics()

    return 0


if __name__ == "__main__":
    main()
