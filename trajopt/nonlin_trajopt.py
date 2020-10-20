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
    PiecewisePolynomial,
    Simulator,
    DiagramBuilder,
    LogOutput,
)
import matplotlib.pyplot as plt

from plot.plot import plot_trj_3_wind, plot_input_slotine_glider, plot_circulation
from dynamics.slotine_dynamics import continuous_dynamics, SlotineGlider
from dynamics.zhukovskii_glider import ZhukovskiiGlider


# TODO DONE
# * Use time scaled dynamics
# * Redefine heading from wind angle
# * Make opt problem work always
# * Get period
# * Plot input
# * make sure right direction
# * Get avg velocity in direction
# * Discretize and get polar plot
# * Better initial guess, use previous angle
# * Rename wind models


# TODO
# * Find good max/min values for speed and height.
# * Penalize too big cost changes
# * Use Mortens wind model
# * Constrain input difference
# * Constrain AoA within +- 5 degrees


def direct_collocation(
    zhukovskii_glider,
    travel_angle,
    initial_guess=None,
    PLOT_SOLUTION=False,
    PRINT_GLIDER_DETAILS=False,
    PLOT_INITIAL_GUESS=False,
):

    # Get model parameters
    M, rho, AR, Lambda, efficiency, V_l, G, L, T, C = zhukovskii_glider.get_params()
    (
        min_height,
        max_height,
        min_vel,
        max_vel,
        h0,
        min_travelled_distance,
    ) = zhukovskii_glider.get_constraints()

    if PRINT_GLIDER_DETAILS:
        print("Running direct collocation with:")
        print("Dimensionless Zhukovskii Glider")
        print(
            "Lambda: {0}\nV_l: {1} (m/s)\nL: {2} (m)\nT: {3} (s)\npsi: {4} (rad)".format(
                Lambda, V_l, L, T, travel_angle
            )
        )

    # Optimization params
    N = 21  # Collocation points
    min_dt = 0.05
    max_dt = 0.5

    # Initial guess
    avg_vel_guess = V_l * 0.5  # TODO tune this
    end_time_guess = N * max_dt
    total_dist_travelled_guess = avg_vel_guess * end_time_guess

    # Make all values dimless
    min_height /= L
    max_height /= L
    min_travelled_distance /= L
    h0 /= L
    total_dist_travelled_guess /= L
    min_vel /= V_l
    max_vel /= V_l
    avg_vel_guess /= V_l
    end_time_guess /= T

    ######
    # DEFINE TRAJOPT PROBLEM
    ######

    plant = zhukovskii_glider.get_drake_plant()
    context = plant.CreateDefaultContext()

    dircol = DirectCollocation(
        plant,
        context,
        num_time_samples=N,
        minimum_timestep=min_dt,
        maximum_timestep=max_dt,
        # TODO add/find solution treshold
    )
    # Constrain all timesteps, h[k], to be equal,
    # so the trajectory breaks are evenly distributed.
    dircol.AddEqualTimeIntervalsConstraints()

    ######
    # ADD CONSTRAINTS
    ######

    ## Add input constraints
    u = dircol.input()
    # TODO ?

    ## Add state constraints
    x = dircol.state()

    # Height constraints
    dircol.AddConstraintToAllKnotPoints(min_height <= x[2])
    dircol.AddConstraintToAllKnotPoints(x[2] <= max_height)

    # Velocity constraints
    dircol.AddConstraintToAllKnotPoints(min_vel ** 2 <= x[3:6].T.dot(x[3:6]))
    dircol.AddConstraintToAllKnotPoints(x[3:6].T.dot(x[3:6]) <= max_vel ** 2)

    # Initial state constraint
    x0_pos = np.array([0, 0, h0])
    dircol.AddBoundingBoxConstraint(x0_pos, x0_pos, dircol.initial_state()[0:3])

    ## Periodicity constraints

    # Periodic height
    dircol.AddLinearConstraint(dircol.final_state()[2] == dircol.initial_state()[2])

    # Periodic velocities
    dircol.AddLinearConstraint(dircol.final_state()[3] == dircol.initial_state()[3])
    dircol.AddLinearConstraint(dircol.final_state()[4] == dircol.initial_state()[4])
    dircol.AddLinearConstraint(dircol.final_state()[5] == dircol.initial_state()[5])

    # Final position constraint in terms of travel angle
    if travel_angle % np.pi == 0:
        # Travel along y-axis, constrain x values to be equal
        dircol.AddConstraint(dircol.final_state()[0] == dircol.initial_state()[0])
    elif travel_angle % ((1 / 2) * np.pi) == 0:
        # Travel along x-axis, constrain y values to be equal
        dircol.AddConstraint(dircol.final_state()[1] == dircol.initial_state()[1])
    else:
        dircol.AddConstraint(
            dircol.final_state()[0] == dircol.final_state()[1] * np.tan(travel_angle)
        )

    # Make sure covered distance along travel angle is positive
    xy_pos_final = dircol.final_state()[0:2]
    dir_vector = np.array([np.sin(travel_angle), np.cos(travel_angle)])
    dircol.AddConstraintToAllKnotPoints(
        dir_vector.T.dot(xy_pos_final) >= min_travelled_distance
    )

    ## Objective function

    # Cost on input effort
    R = 5
    # TODO replace 0.125
    dircol.AddRunningCost(R * (u[0] ** 2 + (u[1] - 0.125) ** 2 + u[2] ** 2))

    # Maximize distance travelled in desired direction
    Q = 2
    dircol.AddFinalCost(-(dir_vector.T.dot(xy_pos_final)) * Q)

    ######
    # PROVIDE INITIAL GUESS
    ######

    # If no initial guess provided, use a straight line
    if initial_guess == None:
        print("Running with straight line as initial guess")
        x0_guess = np.array(
            [0, 0, h0, avg_vel_guess * dir_vector[0], avg_vel_guess * dir_vector[1], 0]
        )

        xf_guess = np.array(
            [
                dir_vector[0] * total_dist_travelled_guess,
                dir_vector[1] * total_dist_travelled_guess,
                h0,
                avg_vel_guess * dir_vector[0],
                avg_vel_guess * dir_vector[1],
                0,
            ]
        )
        # Linear interpolation
        initial_x_trajectory = PiecewisePolynomial.FirstOrderHold(
            [0.0, end_time_guess], np.column_stack((x0_guess, xf_guess))
        )
        dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

    # Use provided initial_guess
    else:
        print("Running with provided initial guess")
        dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_guess)

    if PLOT_INITIAL_GUESS:
        times = np.linspace(
            initial_x_trajectory.start_time(), initial_x_trajectory.end_time(), N
        )
        x0_knots = np.hstack([initial_x_trajectory.value(t) for t in times]).T
        traj_plt = plot_trj_3_wind(x0_knots[:, 0:3], dir_vector)
        plt.show()


    #######
    # SOLVE TRAJOPT PROBLEM
    #######

    print("*** Solving DirCol for travel_angle: {0}".format(travel_angle))
    result = Solve(dircol)
    # assert result.is_success()

    if result.is_success():
        print("Found a solution!")
        x_traj_dimless = dircol.ReconstructStateTrajectory(result)

        N_plot = 200

        # Reconstruct trajectory
        times_dimless = np.linspace(
            x_traj_dimless.start_time(), x_traj_dimless.end_time(), N_plot
        )

        x_knots_dimless = np.hstack([x_traj_dimless.value(t) for t in times_dimless]).T
        x_knots = x_knots_dimless * L
        times = times_dimless * T

        # Plot input
        u_traj_dimless = dircol.ReconstructInputTrajectory(result)
        u_knots_dimless = np.hstack([u_traj_dimless.value(t) for t in times_dimless]).T
        u_knots = u_knots_dimless * C

        if PLOT_SOLUTION:  # TODO remove

            plot_trj_3_wind(x_knots[:, 0:3], dir_vector)
            plot_circulation(times, u_knots)

            plt.show()

        # Calculate solution properties
        solution_period = x_traj_dimless.end_time() * T
        solution_cost = result.get_optimal_cost()
        solution_distance = dir_vector.T.dot(x_knots[-1, 0:2])
        solution_avg_vel = solution_distance / solution_period

        print(
            "\tSolution period: {0} (s)\n\tSolution cost: {1}\n\tSolution distance: {2} (m) \n\tSolution avg. vel: {3} (m/s)".format(
                solution_period, solution_cost, solution_distance, solution_avg_vel
            )
        )

        return solution_avg_vel, (times, x_knots, u_knots), x_traj_dimless

    else:  # No solution
        print("ERROR: Did not find a solution")
        return -1, None, None


def direct_collocation_slotine_glider():
    print("Running direct collocation")

    plant = SlotineGlider()
    context = plant.CreateDefaultContext()

    N = 21
    initial_guess = True
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
    dircol.AddConstraintToAllKnotPoints(u[0] <= 3)
    dircol.AddConstraintToAllKnotPoints(-np.pi / 2 <= u[1])
    dircol.AddConstraintToAllKnotPoints(u[1] <= np.pi / 2)

    # Add state constraints
    x = dircol.state()
    min_speed = 5
    dircol.AddConstraintToAllKnotPoints(x[0] >= min_speed)
    min_height = 0.5
    dircol.AddConstraintToAllKnotPoints(x[3] >= min_height)

    # Add initial state
    travel_angle = (3 / 2) * np.pi
    h0 = 10
    dir_vector = np.array([np.cos(travel_angle), np.sin(travel_angle)])

    # Start at initial position
    x0_pos = np.array([h0, 0, 0])
    dircol.AddBoundingBoxConstraint(x0_pos, x0_pos, dircol.initial_state()[3:6])

    # Periodicity constraints
    dircol.AddLinearConstraint(dircol.final_state()[0] == dircol.initial_state()[0])
    dircol.AddLinearConstraint(dircol.final_state()[1] == dircol.initial_state()[1])
    dircol.AddLinearConstraint(dircol.final_state()[2] == dircol.initial_state()[2])
    dircol.AddLinearConstraint(dircol.final_state()[3] == dircol.initial_state()[3])

    # Always end in right direction
    # NOTE this assumes that we always are starting in origin
    if travel_angle % np.pi == 0:  # Travel along x-axis
        dircol.AddConstraint(dircol.final_state()[5] == dircol.initial_state()[5])
    elif travel_angle % ((1 / 2) * np.pi) == 0:  # Travel along y-axis
        dircol.AddConstraint(dircol.final_state()[4] == dircol.initial_state()[4])
    else:
        dircol.AddConstraint(
            dircol.final_state()[5] == dircol.final_state()[4] * np.tan(travel_angle)
        )

    # Maximize distance travelled in desired direction
    p0 = dircol.initial_state()
    p1 = dircol.final_state()
    Q = 1
    dist_travelled = np.array([p1[4], p1[5]])  # NOTE assume starting in origin
    dircol.AddFinalCost(-(dir_vector.T.dot(dist_travelled)) * Q)

    if True:
        # Cost on input effort
        R = 0.1
        dircol.AddRunningCost(R * (u[0]) ** 2 + R * u[1] ** 2)

    # Initial guess is a straight line from x0 in direction
    if initial_guess:
        avg_vel_guess = 10  # Guess for initial velocity
        x0_guess = np.array([avg_vel_guess, travel_angle, 0, h0, 0, 0])

        guessed_total_dist_travelled = 200
        xf_guess = np.array(
            [
                avg_vel_guess,
                travel_angle,
                0,
                h0,
                dir_vector[0] * guessed_total_dist_travelled,
                dir_vector[1] * guessed_total_dist_travelled,
            ]
        )
        initial_x_trajectory = PiecewisePolynomial.FirstOrderHold(
            [0.0, 4.0], np.column_stack((x0_guess, xf_guess))
        )
        dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

    # Solve direct collocation
    result = Solve(dircol)
    assert result.is_success()
    print("Found a solution!")

    # PLOTTING
    N_plot = 200

    # Plot trajectory
    x_trajectory = dircol.ReconstructStateTrajectory(result)
    times = np.linspace(x_trajectory.start_time(), x_trajectory.end_time(), N_plot)
    x_knots = np.hstack([x_trajectory.value(t) for t in times])
    z = x_knots[3, :]
    x = x_knots[4, :]
    y = x_knots[5, :]
    plot_trj_3_wind(np.vstack((x, y, z)).T, dir_vector)

    # Plot input
    u_trajectory = dircol.ReconstructInputTrajectory(result)
    u_knots = np.hstack([u_trajectory.value(t) for t in times])

    plot_input_slotine_glider(times, u_knots.T)

    plt.show()
    return 0


def simulate_drake_system(plant):
    # Create a simple block diagram containing our system.
    builder = DiagramBuilder()
    system = builder.AddSystem(plant)
    logger = LogOutput(system.get_output_port(0), builder)
    diagram = builder.Build()

    # Set the initial conditions, x(0).
    context = diagram.CreateDefaultContext()
    context.SetContinuousState([0, 0, 20, 10, 0, 0])

    # Create the simulator, and simulate for 10 seconds.
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(30)

    # Plotting
    x_sol = logger.data().T

    plot_trj_3_wind(x_sol[:, 0:3], np.array([0, 0, 0]))
    plt.show()

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
