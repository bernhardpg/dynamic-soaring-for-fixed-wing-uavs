import numpy as np
import pdb
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from pydrake.all import eq, MathematicalProgram, Solve, Variable, Expression


def continous_dynamics(x, u):
    # Constants
    air_density = 1
    wing_area = 1
    parasitic_drag = 1
    wingspan = 1
    mass = 1

    # Dynamics
    x_dot = np.empty(6, dtype=Expression)

    circulation = u
    pos = x[0:3]
    height = pos[2]
    vel = x[3:6]

    wind = np.array([get_wind(height), 0, 0])
    rel_vel = vel - wind

    x_dot[0:3] = vel
    x_dot[3:6] = (1 / mass) * (
        air_density * np.cross(circulation, rel_vel)
        - 0.5 * air_density * wing_area * parasitic_drag
        # * np.linalg.norm(rel_vel)
        * rel_vel
        - (2 * air_density / np.pi)
        # * (np.linalg.norm(circulation) / wingspan) ** 2
        * rel_vel
        # / np.linalg.norm(rel_vel)
        + mass * np.array([0, 0, -9.81])
    )

    return x_dot


def get_wind(height):
    ref_height = 10
    alpha = 2

    # TODO replace u0
    u0 = 5

    return u0 * (height / ref_height) ** alpha


def get_wind_field(x, y, z):
    u = get_wind(z)
    v = np.zeros(y.shape)
    w = np.zeros(z.shape)

    return u, v, w


def main():
    print("Starting trajopt")

    prog = MathematicalProgram()

    N = 500
    dt = 0.01

    # Create decision variables
    x = np.empty((6, N), dtype=Variable)
    u = np.empty((3, N - 1), dtype=Variable)  # Circulation vector
    for n in range(N - 1):
        x[:, n] = prog.NewContinuousVariables(6, "x" + str(n))
        u[:, n] = prog.NewContinuousVariables(3, "u" + str(n))
    x[:, N - 1] = prog.NewContinuousVariables(6, "x" + str(N))

    # Add constraints
    x0 = [0, 0, 10, 0, 0, 0]
    prog.AddBoundingBoxConstraint(x0, x0, x[:, 0])
    for n in range(N - 1):
        # Dynamics
        prog.AddConstraint(
            eq(x[:, n + 1], x[:, n] + dt * continous_dynamics(x[:, n], u[:, n]))
        )

        # Force circulation to be normal to relative wind
        height = x[2, n]
        vel = x[3:6, n]
        # TODO Assume only wind in x direction for now
        wind = np.array([get_wind(height), 0, 0])
        rel_vel = vel - wind
        prog.AddConstraint(u[:, n].dot(rel_vel) == 0)

    xf = [0, 10, 10, 0, 0, 0]
    prog.AddBoundingBoxConstraint(xf, xf, x[:, N - 1])

    print("Initialized opt problem")
    result = Solve(prog)

    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    assert result.is_success(), "Optimization failed"
    print("Found solution")

    # Plotting
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    x, y, z = np.meshgrid(
        np.arange(-2, 6, 1), np.arange(-2, 10, 1), np.arange(0, 15, 3)
    )
    u, v, w = get_wind_field(x, y, z)
    ax.quiver(x, y, z, u, v, w, length=0.1)
    ax.plot(x_sol[0, :], x_sol[1, :], x_sol[2, :], label="Flight path", color="red")
    ax.legend()
    plt.show()

    t = np.linspace(0, dt * N, num=N - 1)
    fig, axs = plt.subplots(3)
    fig.suptitle("Input")
    axs[0].plot(t, u_sol[0, :])
    axs[0].set_title("Input x")
    axs[1].plot(t, u_sol[1, :])
    axs[1].set_title("Input y")
    axs[2].plot(t, u_sol[2, :])
    axs[2].set_title("Input z")
    plt.show()


if __name__ == "__main__":
    main()
    # example()
