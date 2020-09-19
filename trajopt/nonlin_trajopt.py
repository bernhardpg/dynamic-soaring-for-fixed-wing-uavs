import numpy as np
import pdb
from pydrake.all import (
    eq,
    MathematicalProgram,
    DirectCollocation,
    Solve,
    Variable,
    Expression,
    SnoptSolver,
)
import matplotlib.pyplot as plt

from plot.plot import plot_trj_3_wind, plot_input_slotine_glider
from dynamics.slotine_dynamics import continuous_dynamics, get_wind_field, SlotineGlider


def direct_collocation():
    print("Running direct collocation")

    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.systems.primitives import LogOutput

    plant = SlotineGlider()
    context = plant.CreateDefaultContext()

    N = 21
    max_dt = 0.5
    max_tf = N * max_dt
    dircol = DirectCollocation(
        plant,
        context,
        num_time_samples=N,
        minimum_timestep=0.05,
        maximum_timestep=max_dt,
    )

    # Constrain all timesteps, $h[k]$, to be equal, so the trajectory breaks are evenly distributed.
    dircol.AddEqualTimeIntervalsConstraints()

    # Add input constraints
    u = dircol.input()
    dircol.AddConstraintToAllKnotPoints(0 <= u[0])
    dircol.AddConstraintToAllKnotPoints(u[0] <= 2)

    # Add state constraints
    x = dircol.state()
    min_speed = 5
    dircol.AddConstraintToAllKnotPoints(x[0] >= min_speed)
    min_height = 0.5
    dircol.AddConstraintToAllKnotPoints(x[3] >= min_height)

    # Add initial state
    V0 = 10
    travel_angle = np.pi * (3/2)
    h0 = 20

    dir_vector = np.array([np.cos(travel_angle), np.sin(travel_angle)])

    x0 = np.array([V0, travel_angle, 0, h0, 0, 0])
    dircol.AddBoundingBoxConstraint(x0, x0, dircol.initial_state())

    # x0_pos = np.array([h0, 0, 0])
    # # NOTE Much slower once I remove constraints on the first three states
    # dircol.AddBoundingBoxConstraint(x0_pos, x0_pos, dircol.initial_state()[3:6])

    # Maximize distance travelled in desired direction
    p0 = dircol.initial_state()
    p1 = dircol.final_state()

    Q_dist = 1
    dist_travelled = np.array([p1[4] - p0[4], p1[5] - p0[5]])
    dircol.AddFinalCost(-dir_vector.T.dot(dist_travelled * Q_dist))

    # Minimize lost height
    Q_height = 1
    lost_height = Q_height * (p1[3] - p0[3]) ** 2
    dircol.AddFinalCost(lost_height)

    # Cost on input effort
    R = 0.1
    dircol.AddRunningCost(R * (u[0] - 1) ** 2 + R * u[1] ** 2)

    result = Solve(dircol)
    # assert result.is_success()
    print("Found solution: {0}".format(result.is_success()))

    # PLOTTING
    N_plot = 100

    # Plot trajectory
    x_trajectory = dircol.ReconstructStateTrajectory(result)
    times = np.linspace(x_trajectory.start_time(), x_trajectory.end_time(), N_plot)
    x_knots = np.hstack([x_trajectory.value(t) for t in times])
    z = x_knots[3, :]
    x = x_knots[4, :]
    y = x_knots[5, :]
    plot_trj_3_wind(np.vstack((x, y, z)).T, get_wind_field, dir_vector)

    # Plot input
    u_trajectory = dircol.ReconstructInputTrajectory(result)
    u_knots = np.hstack([u_trajectory.value(t) for t in times])

    plot_input_slotine_glider(times, u_knots.T)

    plt.show()
    return 0


def simulate_drake_system():
    print("Running direct collocation")

    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.systems.primitives import LogOutput

    plant = SlotineGlider()

    # Create a simple block diagram containing our system.
    builder = DiagramBuilder()
    system = builder.AddSystem(plant)
    logger = LogOutput(system.get_output_port(0), builder)
    diagram = builder.Build()

    # Set the initial conditions, x(0).
    context = diagram.CreateDefaultContext()
    context.SetContinuousState([10, 0, 0, 20, 0, 0])

    # Create the simulator, and simulate for 10 seconds.
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(10)

    # Plotting
    x_sol = logger.data().T
    z = x_sol[:, 3]
    x = x_sol[:, 4]
    y = x_sol[:, 5]

    plot_trj_3_wind(np.vstack((x, y, z)).T, get_wind_field)

    return 0


def direct_transcription():
    prog = MathematicalProgram()

    N = 500
    dt = 0.01

    # Create decision variables
    n_x = 6
    n_u = 2

    x = np.empty((n_x, N), dtype=Variable)
    u = np.empty((n_u, N - 1), dtype=Variable)

    for n in range(N - 1):
        x[:, n] = prog.NewContinuousVariables(n_x, "x" + str(n))
        u[:, n] = prog.NewContinuousVariables(n_u, "u" + str(n))
    x[:, N - 1] = prog.NewContinuousVariables(n_x, "x" + str(N))
    T = N - 1

    # Add constraints
    # x0 = np.array([10, -np.pi / 2, 0, 40, 0, 0])
    # Slotine dynamics: x = [airspeed, heading, flight_path_angle, z, x, y]

    for n in range(N - 1):
        # Dynamics
        prog.AddConstraint(
            eq(x[:, n + 1], x[:, n] + dt * continuous_dynamics(x[:, n], u[:, n]))
        )

        # Never below sea level
        prog.AddConstraint(x[3, n + 1] >= 0 + 0.5)

        # Always positive velocity
        prog.AddConstraint(x[0, n + 1] >= 0)

    # TODO use residuals
    # Add periodic constraints
    prog.AddConstraint(x[0, 0] - x[0, T] == 0)
    prog.AddConstraint(x[1, 0] - x[1, T] == 0)
    prog.AddConstraint(x[2, 0] - x[2, T] == 0)
    prog.AddConstraint(x[3, 0] - x[3, T] == 0)

    # Start at 20 meter height
    prog.AddConstraint(x[4, 0] == 0)
    prog.AddConstraint(x[5, 0] == 0)
    h0 = 20
    prog.AddConstraint(x[3, 0] == h0)

    # Add cost function
    p0 = x[4:6, 0]
    p1 = x[4:6, T]

    travel_angle = np.pi  # TODO start along y direction
    dir_vector = np.array([np.sin(travel_angle), np.cos(travel_angle)])
    prog.AddCost(dir_vector.T.dot(p1 - p0))  # Maximize distance travelled

    print("Initialized opt problem")

    # Initial guess
    V0 = 10
    x_guess = np.zeros((n_x, N))
    x_guess[:, 0] = np.array([V0, travel_angle, 0, h0, 0, 0])

    for n in range(N - 1):
        # Constant airspeed, heading and height
        x_guess[0, n + 1] = V0
        x_guess[1, n + 1] = travel_angle
        x_guess[3, n + 1] = h0

        # Interpolate position
        avg_speed = 10  # conservative estimate
        x_guess[4:6, n + 1] = dir_vector * avg_speed * n * dt

        # Let the rest of the variables be initialized to zero

    prog.SetInitialGuess(x, x_guess)

    # solve mathematical program
    solver = SnoptSolver()
    result = solver.Solve(prog)

    # be sure that the solution is optimal
    # assert result.is_success()

    # retrieve optimal solution
    thrust_opt = result.GetSolution(x)
    state_opt = result.GetSolution(u)

    result = Solve(prog)

    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)

    breakpoint()

    # Slotine dynamics: x = [airspeed, heading, flight_path_angle, z, x, y]
    z = x_sol[:, 3]
    x = x_sol[:, 4]
    y = x_sol[:, 5]

    plot_trj_3_wind(np.vstack((x, y, z)).T, get_wind_field)

    return
