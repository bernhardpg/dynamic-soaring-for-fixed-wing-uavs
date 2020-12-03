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
    period_guess=4,
    avg_vel_scale_guess=1,
    avg_vel_guess=None,
    initial_guess=None,
    PRINT_GLIDER_DETAILS=False,
    PLOT_INITIAL_GUESS=False,
):

    start_time = time.time()

    # Get model parameters
    V_l, L, T, C = zhukovskii_glider.get_char_values()
    A = zhukovskii_glider.get_wing_area()
    # TODO neater way of passing params
    (
        max_bank_angle,
        max_lift_coeff,
        min_lift_coeff,
        max_load_factor,
        min_height,
        max_height,
        h0,
        min_travelled_distance,
    ) = zhukovskii_glider.get_constraints()

    # Initial guess
    if avg_vel_guess == None:
        avg_vel_guess = V_l * avg_vel_scale_guess
    total_dist_travelled_guess = avg_vel_guess * period_guess

    print(
        "*** Running DirCol for travel_angle: {0} deg".format(
            travel_angle * 180 / np.pi
        )
    )


    # Make all values dimless
    max_lift_coeff *= V_l / C
    min_height /= L
    max_height /= L
    min_travelled_distance /= L
    h0 /= L
    total_dist_travelled_guess /= L
    avg_vel_guess /= V_l
    period_guess /= T

    ######
    # DEFINE TRAJOPT PROBLEM
    # Implemented as dimensionless
    ######

    # Optimization params
    N = 21  # Collocation points
    min_dt = (period_guess / N) * 0.75
    max_dt = (period_guess / N) * 1.25

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

    ## Add state constraints
    x = dircol.state()

    # Lift coefficient constraint
    lift_coeff_squared = u.T.dot(u) / ((0.5 * A) ** 2 * x[3:6].T.dot(x[3:6]))
    dircol.AddConstraintToAllKnotPoints(lift_coeff_squared <= max_lift_coeff ** 2)
    dircol.AddConstraintToAllKnotPoints(min_lift_coeff ** 2 <= lift_coeff_squared)

    # Load factor constraint
    load_factor_squared = x[3:6].T.dot(x[3:6]) * u.T.dot(u)
    dircol.AddConstraintToAllKnotPoints(load_factor_squared <= max_load_factor ** 2)

    # Height constraints
    dircol.AddConstraintToAllKnotPoints(min_height <= x[2])
    dircol.AddConstraintToAllKnotPoints(x[2] <= max_height)

    # Bank angle constraint
    max_sin_bank_angle_squared = np.sin(max_bank_angle) ** 2
    sin_bank_angle_squared = u[2] ** 2 / (
        u.T.dot(u) * (1 - x[5] ** 2 / (x[3:6].T.dot(x[3:6])))
    )
    dircol.AddConstraintToAllKnotPoints(
        sin_bank_angle_squared <= max_sin_bank_angle_squared
    )
# TODO remove, this is never active
#    dircol.AddConstraintToAllKnotPoints(
#        sin_bank_angle_squared >= -max_sin_bank_angle_squared
#    )

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

    # Periodic inputs
    dircol.AddLinearConstraint(dircol.input(0)[0] == dircol.input(N - 1)[0])
    dircol.AddLinearConstraint(dircol.input(0)[1] == dircol.input(N - 1)[1])
    dircol.AddLinearConstraint(dircol.input(0)[2] == dircol.input(N - 1)[2])

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

    #    # Constraint covered distance along travel angle to be positive
    hor_pos_final = dircol.final_state()[0:2]
    dir_vector = np.array([np.sin(travel_angle), np.cos(travel_angle)])
    dircol.AddConstraintToAllKnotPoints(
        min_travelled_distance <= dir_vector.T.dot(hor_pos_final)
    )

    ## Objective function
    # Maximize average velocity travelled in desired direction
    Q = 1

    def average_speed(vars):
        hor_pos_final = vars[0:2]
        time_step = vars[2]
        avg_speed = dir_vector.T.dot(hor_pos_final) / (time_step * N)
        return -Q * avg_speed

    time_step = dircol.timestep(0)[0]
    dircol.AddCost(average_speed, vars=hor_pos_final.tolist() + [time_step])

    # Cost on input effort
    R = 0.5

    # Constrain input rates
    # Using 2nd order forward finite differences for first derivative
    first_order_finite_diff_matrix = -1 * np.diag(np.ones(N), 0) + 1 * np.diag(
        np.ones((N - 1)), 1
    )
    second_order_finite_diff_matrix = (
        -3 / 2 * np.diag(np.ones(N), 0)
        + 2 * np.diag(np.ones((N - 1)), 1)
        - 1 / 2 * np.diag(np.ones((N - 2)), 2)
    )
    third_order_finite_diff_matrix = (
        -11 / 6 * np.diag(np.ones(N), 0)
        + 3 * np.diag(np.ones((N - 1)), 1)
        - 3 / 2 * np.diag(np.ones((N - 2)), 2)
        + 1 / 3 * np.diag(np.ones((N - 3)), 3)
    )
    finite_diff_matrix = first_order_finite_diff_matrix

    def input_rate(vars):
        h = vars[0]
        u = np.array(vars[1:]).reshape(N, 3)
        u_change = finite_diff_matrix.dot(u)
        u_change_squared = np.sum(np.diag(u_change.T.dot(u_change)))

        return R * u_change_squared / h

    input_vars = (
        np.vstack([dircol.input(i).reshape((3, 1)) for i in range(N)])
        .flatten()
        .tolist()
    )
    dircol.AddCost(input_rate, vars=[time_step] + input_vars)

    ######
    # PROVIDE INITIAL GUESS
    ######

    # If no initial guess provided, use a straight line
    if initial_guess == None:
        print("\tRunning with straight line as initial guess")
        print("\t\tperiod_guess: {0}, avg_vel_guess: {1}".format(period_guess * T, avg_vel_guess * V_l))
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
            [0.0, period_guess], np.column_stack((x0_guess, xf_guess))
        )
        dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

    # Use provided initial_guess
    else:
        print("\tRunning with provided initial guess")
        initial_x_traj, initial_u_traj = initial_guess
        dircol.SetInitialTrajectory(initial_u_traj, initial_x_traj)

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
    print("\tFormulated trajopt in: {0} s".format(formulate_time - start_time))
    result = Solve(dircol)
    solve_time = time.time()
    print("\t! Finished trajopt in: {0} s".format(solve_time - formulate_time))
    # assert result.is_success()

    if result.is_success():
        x_traj_dimless = dircol.ReconstructStateTrajectory(result)
        sample_times = dircol.GetSampleTimes(result)
        N_plot = 200

        ## Reconstruct and re-scale trajectory
        times_dimless = np.linspace(
            x_traj_dimless.start_time(), x_traj_dimless.end_time(), N_plot
        )

        x_knots_dimless = np.hstack([x_traj_dimless.value(t) for t in times_dimless]).T

        p_knots = x_knots_dimless[:, 0:3] * L
        v_r_knots = x_knots_dimless[:, 3:6] * V_l
        x_knots = np.hstack((p_knots, v_r_knots))

        times = times_dimless * T

        u_traj_dimless = dircol.ReconstructInputTrajectory(result)
        u_knots_dimless = np.hstack([u_traj_dimless.value(t) for t in times_dimless]).T
        u_knots = u_knots_dimless * C

        # Calculate solution properties
        solution_period = x_traj_dimless.end_time() * T
        solution_cost = result.get_optimal_cost()
        solution_distance = dir_vector.T.dot(x_knots[-1, 0:2])
        solution_avg_vel = solution_distance / solution_period

        print("\t** Solution details:")
        print(
            "\t\tperiod: {0} (s)\n\t\tcost: {1}\n\t\tdistance: {2} (m) \n\t\tavg. vel: {3} (m/s)".format(
                solution_period, solution_cost, solution_distance, solution_avg_vel
            )
        )
        time_step = sample_times[1] - sample_times[0]
        print("\t\tTime step: {0}, max time step: {1}".format(time_step, max_dt))

        solution_details = (solution_avg_vel, solution_period)
        solution_trajectory = (times, x_knots, u_knots)
        next_initial_guess = (x_traj_dimless, u_traj_dimless)

        return (
            True,
            solution_details,
            solution_trajectory,
            next_initial_guess,
        )

    else:  # No solution
        print("!!! ERROR: Did not find a solution")
        return False, (-1, -1), None, None


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
    period_guess = N * max_dt
    total_dist_travelled_guess = avg_vel_guess * period_guess

    # Make all values dimless
    min_height /= L
    max_height /= L
    min_travelled_distance /= L
    h0 /= L
    total_dist_travelled_guess /= L
    min_vel /= V_l
    max_vel /= V_l
    avg_vel_guess /= V_l
    period_guess /= T

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
    hor_pos_final = dircol.final_state()[0:2]
    dir_vector = np.array([np.sin(travel_angle), np.cos(travel_angle)])
    dircol.AddConstraintToAllKnotPoints(
        dir_vector.T.dot(hor_pos_final) >= min_travelled_distance
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
    dircol.AddFinalCost(-(dir_vector.T.dot(hor_pos_final)) * Q)

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
            [0.0, period_guess], np.column_stack((x0_guess, xf_guess))
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
