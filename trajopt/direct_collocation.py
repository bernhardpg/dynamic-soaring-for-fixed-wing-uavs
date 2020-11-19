import time
import numpy as np
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


def direct_collocation_relative(
    zhukovskii_glider,
    travel_angle,
    initial_guess=None,
    PRINT_GLIDER_DETAILS=False,
    PLOT_INITIAL_GUESS=False,
):

    start_time = time.time()

    # Get model parameters
    V_l, L, T, C = zhukovskii_glider.get_char_values()
    # TODO neater way of passing params
    (
        max_bank_angle,
        min_height,
        max_height,
        min_vel,
        max_vel,
        h0,
        min_travelled_distance,
    ) = zhukovskii_glider.get_constraints()

    print("*** Running DirCol for travel_angle: {0}".format(travel_angle))

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
    min_dt = 0.25
    max_dt = 0.5

    # Initial guess
    avg_vel_guess = V_l * 2 # TODO tune this
    end_time_guess = N * min_dt
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

    plant = zhukovskii_glider.create_drake_plant()
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

    ## Add input constraint
    u = dircol.input()

    # Brake param
    enable_brake_param = u.shape[0] > 3
    if enable_brake_param:
        dircol.AddConstraintToAllKnotPoints(0 <= u[3])

    # TODO Add more input constraints?

    ## Add state constraints
    x = dircol.state()

    # Height constraints
    dircol.AddConstraintToAllKnotPoints(min_height <= x[2])
    dircol.AddConstraintToAllKnotPoints(x[2] <= max_height)

    # Velocity constraints
    dircol.AddConstraintToAllKnotPoints(min_vel ** 2 <= x[3:6].T.dot(x[3:6]))
    dircol.AddConstraintToAllKnotPoints(x[3:6].T.dot(x[3:6]) <= max_vel ** 2)
    # TODO change max_vel

    # TODO come back to this
    # Bank angle constraint
    sin_bank_angle_squared = np.sin(max_bank_angle) ** 2
    temp = u[2] ** 2 / (u.T.dot(u) * (1 - x[5] ** 2 / (x[3:6].T.dot(x[3:6]))))
    dircol.AddConstraintToAllKnotPoints(temp <= sin_bank_angle_squared)
    dircol.AddConstraintToAllKnotPoints(temp >= -sin_bank_angle_squared)

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
    dircol.AddRunningCost(R * (u[0] ** 2 + (u[1] - 1) ** 2 + u[2] ** 2))
    if enable_brake_param:
        dircol.AddRunningCost(R * u[3] ** 2)

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

    formulate_time = time.time()
    print("Formulated trajopt in: {0} s".format(formulate_time - start_time))
    result = Solve(dircol)
    # assert result.is_success()

    if result.is_success():
        solution_time = time.time()
        print("Found a solution in: {0} s".format(solution_time - formulate_time))
        x_traj_dimless = dircol.ReconstructStateTrajectory(result)

        N_plot = 200

        ## Reconstruct and re-scale trajectory
        times_dimless = np.linspace(
            x_traj_dimless.start_time(), x_traj_dimless.end_time(), N_plot
        )

        x_knots_dimless = np.hstack([x_traj_dimless.value(t) for t in times_dimless]).T
        x_knots = x_knots_dimless * L
        times = times_dimless * T

        u_traj_dimless = dircol.ReconstructInputTrajectory(result)
        u_knots_dimless = np.hstack([u_traj_dimless.value(t) for t in times_dimless]).T
        u_knots = u_knots_dimless * C

        # Calculate solution properties
        solution_period = x_traj_dimless.end_time() * T
        solution_cost = result.get_optimal_cost()
        solution_distance = dir_vector.T.dot(x_knots[-1, 0:2])
        solution_avg_vel = solution_distance / solution_period

        print("Solution details:")
        print(
            "\tperiod: {0} (s)\n\tcost: {1}\n\tdistance: {2} (m) \n\tavg. vel: {3} (m/s)".format(
                solution_period, solution_cost, solution_distance, solution_avg_vel
            )
        )

        return solution_avg_vel, (times, x_knots, u_knots), x_traj_dimless

    else:  # No solution
        print("ERROR: Did not find a solution")
        return -1, None, None



# TODO this is the old dircol
def direct_collocation(
    zhukovskii_glider,
    travel_angle,
    initial_guess=None,
    PLOT_SOLUTION=False,
    PRINT_GLIDER_DETAILS=False,
    PLOT_INITIAL_GUESS=False,
):

    start_time = time.time()

    # Get model parameters
    V_l, L, T, C = zhukovskii_glider.get_char_values()
    # TODO neater way of passing params
    (
        min_height,
        max_height,
        min_vel,
        max_vel,
        h0,
        min_travelled_distance,
    ) = zhukovskii_glider.get_constraints()

    print("*** Running DirCol for travel_angle: {0}".format(travel_angle))

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

    plant = zhukovskii_glider.create_drake_plant(diff_flat=False)
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

    ## Add input constraint
    u = dircol.input()
    enable_brake_param = u.shape[0] > 3

    if enable_brake_param:
        dircol.AddConstraintToAllKnotPoints(0 <= u[3])

    # TODO Add more input constraints?

    ## Add state constraints
    x = dircol.state()
    # TODO add bank angle constraint

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
    dircol.AddRunningCost(R * (u[0] ** 2 + (u[1] - 1) ** 2 + u[2] ** 2))
    if enable_brake_param:
        dircol.AddRunningCost(R * u[3] ** 2)

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

    formulate_time = time.time()
    print("Formulated trajopt in: {0} s".format(formulate_time - start_time))
    result = Solve(dircol)
    # assert result.is_success()

    if result.is_success():
        solution_time = time.time()
        print("Found a solution in: {0} s".format(solution_time - formulate_time))
        x_traj_dimless = dircol.ReconstructStateTrajectory(result)

        N_plot = 200

        ## Reconstruct and re-scale trajectory
        times_dimless = np.linspace(
            x_traj_dimless.start_time(), x_traj_dimless.end_time(), N_plot
        )

        x_knots_dimless = np.hstack([x_traj_dimless.value(t) for t in times_dimless]).T
        x_knots = x_knots_dimless * L
        times = times_dimless * T

        u_traj_dimless = dircol.ReconstructInputTrajectory(result)
        u_knots_dimless = np.hstack([u_traj_dimless.value(t) for t in times_dimless]).T
        u_knots = u_knots_dimless * C

        # Calculate solution properties
        solution_period = x_traj_dimless.end_time() * T
        solution_cost = result.get_optimal_cost()
        solution_distance = dir_vector.T.dot(x_knots[-1, 0:2])
        solution_avg_vel = solution_distance / solution_period

        print("Solution details:")
        print(
            "\tperiod: {0} (s)\n\tcost: {1}\n\tdistance: {2} (m) \n\tavg. vel: {3} (m/s)".format(
                solution_period, solution_cost, solution_distance, solution_avg_vel
            )
        )

        return solution_avg_vel, (times, x_knots, u_knots), x_traj_dimless

    else:  # No solution
        print("ERROR: Did not find a solution")
        return -1, None, None

