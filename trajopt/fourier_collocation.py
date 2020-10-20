import time
import numpy as np
from pydrake.all import (
    eq,
    MathematicalProgram,
    DirectCollocation,
    Solve,
    Expression,
    SnoptSolver,
    PiecewisePolynomial,
    Simulator,
    DiagramBuilder,
    LogOutput,
)
import pydrake.symbolic as sym

class FourierCollocationProblem:
    def __init__(self, system_dynamics, constraints):
        start_time = time.time()

        self.prog = MathematicalProgram()
        self.N = 50  # Number of collocation points
        self.M = 10  # Number of frequencies
        self.system_dynamics = system_dynamics

        self.psi = np.pi * (0.7)  # TODO change

        (
            min_height,
            max_height,
            min_vel,
            max_vel,
            h0,
            min_travelled_distance,
            t_f_min,
            t_f_max,
            avg_vel_min,
            avg_vel_max,
        ) = constraints

        initial_pos = np.array([0, 0, h0])

        # Add state trajectory parameters as decision variables
        self.coeffs = self.prog.NewContinuousVariables(
            3, self.M + 1, "c"
        )  # (x,y,z) for every frequency
        self.phase_delays = self.prog.NewContinuousVariables(3, self.M, "eta")
        self.t_f = self.prog.NewContinuousVariables(1, 1, "t_f")[0, 0]
        self.avg_vel = self.prog.NewContinuousVariables(1, 1, "V_bar")[0, 0]

        # Enforce initial conditions
        residuals = self.get_pos_fourier(collocation_time=0) - initial_pos
        for residual in residuals:
            self.prog.AddConstraint(residual == 0)

        # Enforce dynamics at collocation points
        for n in range(self.N):
            collocation_time = (n / self.N) * self.t_f
            pos = self.get_pos_fourier(collocation_time)
            vel = self.get_vel_fourier(collocation_time)
            vel_dot = self.get_vel_dot_fourier(collocation_time)

            residuals = self.continuous_dynamics(pos, vel, vel_dot)
            for residual in residuals[3:6]:  # TODO only these last three are residuals
                self.prog.AddConstraint(residual == 0)

            if True:
                # Add velocity constraints
                squared_vel_norm = vel.T.dot(vel)
                self.prog.AddConstraint(min_vel ** 2 <= squared_vel_norm)
                self.prog.AddConstraint(squared_vel_norm <= max_vel ** 2)

                # Add height constraints
                self.prog.AddConstraint(pos[2] <= max_height)
                self.prog.AddConstraint(min_height <= pos[2])

        # Add constraint on min travelled distance
        self.prog.AddConstraint(min_travelled_distance <= self.avg_vel * self.t_f)

        # Add constraints on coefficients
        if False:
            for coeff in self.coeffs.T:
                lb = np.array([-500, -500, -500])
                ub = np.array([500, 500, 500])
                self.prog.AddBoundingBoxConstraint(lb, ub, coeff)

        # Add constraints on phase delays
        if False:
            for etas in self.phase_delays.T:
                lb = np.array([0, 0, 0])
                ub = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])
                self.prog.AddBoundingBoxConstraint(lb, ub, etas)

        # Add constraints on final time and avg vel
        self.prog.AddBoundingBoxConstraint(t_f_min, t_f_max, self.t_f)
        self.prog.AddBoundingBoxConstraint(avg_vel_min, avg_vel_max, self.avg_vel)

        # Add objective function
        self.prog.AddCost(-self.avg_vel)

        # Set initial guess
        coeffs_guess = np.zeros((3, self.M + 1))
        coeffs_guess += np.random.rand(*coeffs_guess.shape) * 100

        self.prog.SetInitialGuess(self.coeffs, coeffs_guess)

        phase_delays_guess = np.zeros((3, self.M))
        phase_delays_guess += np.random.rand(*phase_delays_guess.shape) * 1e-1

        self.prog.SetInitialGuess(self.phase_delays, phase_delays_guess)
        self.prog.SetInitialGuess(self.avg_vel, (avg_vel_max - avg_vel_min) / 2)
        self.prog.SetInitialGuess(self.t_f, (t_f_max - t_f_min) / 2)

        print(
            "Finished formulating the optimization problem in: {0} s".format(
                time.time() - start_time
            )
        )

        start_solve_time = time.time()
        self.result = Solve(self.prog)
        print("Found solution: {0}".format(self.result.is_success()))
        print("Found a solution in: {0} s".format(time.time() - start_solve_time))

        # TODO input costs
        # assert self.result.is_success()

        return

    def get_solution(self):
        coeffs_opt = self.result.GetSolution(self.coeffs)
        phase_delays_opt = self.result.GetSolution(self.phase_delays)
        t_f_opt = self.result.GetSolution(self.t_f)
        avg_vel_opt = self.result.GetSolution(self.avg_vel)

        sim_N = 100
        dt = t_f_opt / sim_N
        times = np.arange(0, t_f_opt, dt)
        pos_traj = np.zeros((3, sim_N))
        # TODO remove vel traj
        vel_traj = np.zeros((3, sim_N))
        for i in range(sim_N):
            t = times[i]
            pos = self.evaluate_pos_traj(
                coeffs_opt, phase_delays_opt, t_f_opt, avg_vel_opt, t
            )
            pos_traj[:, i] = pos
            vel = self.evaluate_vel_traj(
                coeffs_opt, phase_delays_opt, t_f_opt, avg_vel_opt, t
            )
            vel_traj[:, i] = vel

        # TODO move plotting out
        plot_trj_3_wind(pos_traj.T, np.array([np.sin(self.psi), np.cos(self.psi), 0]))

        plt.show()
        breakpoint()

        return

    def evaluate_pos_traj(self, coeffs, phase_delays, t_f, avg_vel, t):
        pos_traj = np.copy(coeffs[:, 0])
        for m in range(1, self.M):
            sine_terms = np.array(
                [
                    np.sin((2 * np.pi * m * t) / t_f + phase_delays[0, m]),
                    np.sin((2 * np.pi * m * t) / t_f + phase_delays[1, m]),
                    np.sin((2 * np.pi * m * t) / t_f + phase_delays[2, m]),
                ]
            )
            pos_traj += coeffs[:, m] * sine_terms

        direction_term = np.array(
            [
                avg_vel * np.sin(self.psi) * t,
                avg_vel * np.cos(self.psi) * t,
                0,
            ]
        )
        pos_traj += direction_term
        return pos_traj

    def evaluate_vel_traj(self, coeffs, phase_delays, t_f, avg_vel, t):
        vel_traj = np.array([0, 0, 0], dtype=object)
        for m in range(1, self.M):
            cos_terms = np.array(
                [
                    (2 * np.pi * m)
                    / t_f
                    * np.cos((2 * np.pi * m * t) / t_f + phase_delays[0, m]),
                    (2 * np.pi * m)
                    / t_f
                    * np.cos((2 * np.pi * m * t) / t_f + phase_delays[1, m]),
                    (2 * np.pi * m)
                    / t_f
                    * np.cos((2 * np.pi * m * t) / t_f + phase_delays[2, m]),
                ]
            )
            vel_traj += coeffs[:, m] * cos_terms

        direction_term = np.array(
            [
                avg_vel * np.sin(self.psi),
                avg_vel * np.cos(self.psi),
                0,
            ]
        )
        vel_traj += direction_term
        return vel_traj

    def get_pos_fourier(self, collocation_time):
        pos_traj = np.copy(self.coeffs[:, 0])
        for m in range(1, self.M):
            a = (2 * np.pi * m) / self.t_f
            sine_terms = np.array(
                [
                    np.sin(a * collocation_time + self.phase_delays[0, m]),
                    np.sin(a * collocation_time + self.phase_delays[1, m]),
                    np.sin(a * collocation_time + self.phase_delays[2, m]),
                ]
            )
            pos_traj += self.coeffs[:, m] * sine_terms

        direction_term = np.array(
            [
                self.avg_vel * np.sin(self.psi) * collocation_time,
                self.avg_vel * np.cos(self.psi) * collocation_time,
                0,
            ]
        )
        pos_traj += direction_term

        return pos_traj

    def get_vel_fourier(self, collocation_time):
        vel_traj = np.array([0, 0, 0], dtype=object)
        for m in range(1, self.M):
            a = (2 * np.pi * m) / self.t_f
            cos_terms = np.array(
                [
                    a * np.cos(a * collocation_time + self.phase_delays[0, m]),
                    a * np.cos(a * collocation_time + self.phase_delays[1, m]),
                    a * np.cos(a * collocation_time + self.phase_delays[2, m]),
                ]
            )
            vel_traj += self.coeffs[:, m] * cos_terms

        direction_term = np.array(
            [
                self.avg_vel * np.sin(self.psi),
                self.avg_vel * np.cos(self.psi),
                0,
            ]
        )
        vel_traj += direction_term
        return vel_traj

    def get_vel_dot_fourier(self, collocation_time):
        vel_dot_traj = np.array([0, 0, 0], dtype=object)
        for m in range(1, self.M):
            a = (2 * np.pi * m) / self.t_f
            sine_terms = np.array(
                [
                    -(a ** 2) * np.sin(a * collocation_time + self.phase_delays[0, m]),
                    -(a ** 2) * np.sin(a * collocation_time + self.phase_delays[1, m]),
                    -(a ** 2) * np.sin(a * collocation_time + self.phase_delays[2, m]),
                ]
            )
            vel_dot_traj += self.coeffs[:, m] * sine_terms
        return vel_dot_traj

    def continuous_dynamics(self, pos, vel, vel_dot):
        x = np.concatenate((pos, vel))
        x_dot = np.concatenate((vel, vel_dot))

        # TODO need to somehow implement input to make this work

        return x_dot - self.system_dynamics(x, u)
