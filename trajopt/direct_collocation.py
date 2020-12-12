import time
import logging as log
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
    period_guess=4,  # TODO a warning should be given when no period guess is given
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

    log.info(
        " *** Running DirCol for travel_angle: {0} deg".format(
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
    N = 31  # Collocation points
    min_dt = (period_guess / N) * 0.5
    max_dt = (period_guess / N) * 3
    # min_dt = 0.1
    # max_dt = 0.7

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

    # Max velocity constraint
    max_vel = 40  # m/s
    max_vel /= V_l
    airspeed_squared = x[3:6].T.dot(x[3:6])
    dircol.AddConstraintToAllKnotPoints(airspeed_squared <= max_vel ** 2)

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
    R = 0.01

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
        time_step = vars[0]
        u = np.array(vars[1:]).reshape(N, 3)
        u_change = finite_diff_matrix.dot(u)
        u_change_squared = np.sum(np.diag(u_change.T.dot(u_change)))

        return R * u_change_squared / time_step

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
        log.debug("\tRunning with straight line as initial guess")
        log.debug(
            "\tperiod_guess: {0}, avg_vel_guess: {1}".format(
                period_guess * T, avg_vel_guess * V_l
            )
        )
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
        log.debug(
            "\tRunning with provided initial guess\n"
            + "\t\tperiod_guess: {0}".format(period_guess * T)
        )
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
    log.debug("\tFormulated trajopt in: {0} s".format(formulate_time - start_time))
    result = Solve(dircol)
    solve_time = time.time()
    log.debug("\t! Finished trajopt in: {0} s".format(solve_time - formulate_time))
    # assert result.is_success()
    found_solution = result.is_success()

    if found_solution:
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

        log.info(
            "\t** Solution details:\n"
            + "\t\tperiod: {0} (s)\n\t\tcost: {1}\n\t\tdistance: {2} (m) \n\t\tavg. vel: {3} (m/s)".format(
                solution_period, solution_cost, solution_distance, solution_avg_vel
            )
        )

        # Check time step
        time_step = sample_times[1] - sample_times[0]
        log.debug("\tTime step: {0}".format(time_step))
        log.debug("\tmin_dt: {0}, max_dt: {1}".format(min_dt, max_dt))

        tol = 0.0001
        limited_by_time_step = "false"
        if abs(time_step - min_dt) < tol:
            limited_by_time_step = "lower"
        if abs(time_step - max_dt) < tol:
            limited_by_time_step = "upper"

        solution_details = (solution_avg_vel, solution_period, limited_by_time_step)
        solution_trajectory = (times, x_knots, u_knots)
        next_initial_guess = (x_traj_dimless, u_traj_dimless)

        return (
            found_solution,
            solution_details,
            solution_trajectory,
            next_initial_guess,
        )

    else:  # No solution
        log.error(" Did not find a solution")
        return found_solution, (-1, -1, -1), None, None

