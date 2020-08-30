import numpy as np
import pdb
import matplotlib.pyplot as plt

from pydrake.all import eq, MathematicalProgram, Solve, Variable


def f(x, u):
    f = np.array(x)
    f[0:3] += u

    return f


def get_wind(height):
    ref_height = 10
    alpha = 2

    # TODO replace u0
    u0 = 5

    return u0 * (height / ref_height) ** alpha


def main():
    print("Starting trajopt")

    prog = MathematicalProgram()

    N = 1000
    dt = 0.01

    # Create decision variables
    x = np.empty((6, N), dtype=Variable)
    u = np.empty((3, N - 1), dtype=Variable)  # Circulation vector
    for n in range(N - 1):
        x[:, n] = prog.NewContinuousVariables(6, "x" + str(n))
        u[:, n] = prog.NewContinuousVariables(3, "u" + str(n))
    x[:, N - 1] = prog.NewContinuousVariables(6, "x" + str(N))

    # Add constraints
    x0 = [0, 0, 0, 0, 0, 0]
    prog.AddBoundingBoxConstraint(x0, x0, x[:, 0])
    for n in range(N - 1):
        # Dynamics
        prog.AddConstraint(eq(x[:, n + 1], x[:, n] + dt * f(x[:, n], u[:, n])))

        # Force circulation to be normal to relative wind
        height = x[2, n]
        vel = x[3:6, n]
        # TODO Assume only wind in x direction for now
        wind = np.array([get_wind(height), 0, 0])
        rel_vel = vel - wind
        prog.AddConstraint(u[:, n].dot(rel_vel) == 0)

        prog.AddBoundingBoxConstraint([-1, -1, -1], [1, 1, 1], u[:, n])
    xf = [0, 10, 0, 0, 0, 0]
    prog.AddBoundingBoxConstraint(xf, xf, x[:, N - 1])

    print("Initialized opt problem")
    result = Solve(prog)

    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    assert result.is_success(), "Optimization failed"

    print("Solution")
    #    plt.figure()
    #    plt.plot(x_sol[0, :], x_sol[1, :])
    #    plt.xlabel("x")
    #    plt.ylabel("y")
    #    plt.show()
    t = np.linspace(0, dt * N, num=N - 1)

    fig, axs = plt.subplots(3)
    fig.suptitle("Solution")
    axs[0].plot(x_sol[0, :], x_sol[1, :])
    axs[0].set_title("Position")
    axs[1].plot(t, u_sol[0, :])
    # axs[1].plot(t, u_sol[1, :])
    axs[1].set_title("Input x")
    axs[2].plot(t, u_sol[1, :])
    axs[2].set_title("Input y")
    plt.show()


# Drake example
def example():
    # Discrete-time approximation of the double integrator.
    dt = 0.01
    A = np.eye(2) + dt * np.mat("0 1; 0 0")
    B = dt * np.mat("0; 1")

    prog = MathematicalProgram()

    N = 284  # Note: I had to do a manual "line search" to find this.

    # Create decision variables
    u = np.empty((1, N - 1), dtype=Variable)
    x = np.empty((2, N), dtype=Variable)
    for n in range(N - 1):
        u[:, n] = prog.NewContinuousVariables(1, "u" + str(n))
        x[:, n] = prog.NewContinuousVariables(2, "x" + str(n))
    x[:, N - 1] = prog.NewContinuousVariables(2, "x" + str(N))

    # Add constraints
    x0 = [-2, 0]
    prog.AddBoundingBoxConstraint(x0, x0, x[:, 0])
    for n in range(N - 1):
        # Will eventually be prog.AddConstraint(x[:,n+1] == A@x[:,n] + B@u[:,n])
        # See drake issues 12841 and 8315
        prog.AddConstraint(eq(x[:, n + 1], A.dot(x[:, n]) + B.dot(u[:, n])))
        prog.AddBoundingBoxConstraint(-1, 1, u[:, n])
    xf = [0, 0]
    prog.AddBoundingBoxConstraint(xf, xf, x[:, N - 1])

    result = Solve(prog)

    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    assert result.is_success(), "Optimization failed"

    breakpoint()
    print(x_sol)

    plt.figure()
    plt.plot(x_sol[0, :], x_sol[1, :])
    plt.xlabel("q")
    plt.ylabel("qdot")
    plt.show()


if __name__ == "__main__":
    main()
    # example()
