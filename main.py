import numpy as np
from pydrake.all import eq, MathematicalProgram, Solve, Variable, Expression
from test.test_dynamics import *
from test.test_ilqr import test_ilqr

def main():
    test_ilqr()
    return 0


if __name__ == "__main__":
    main()

