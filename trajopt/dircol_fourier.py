import numpy as np
import pdb
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
import matplotlib.pyplot as plt

from plot.plot import plot_trj_3_wind, plot_input_slotine_glider, plot_circulation
from dynamics.slotine_dynamics import continuous_dynamics, SlotineGlider
from dynamics.zhukovskii_glider import ZhukovskiiGlider


class DirColFourierProblem:
    def __init__(self, system_dynamics):
        self.prog = MathematicalProgram()
        self.N = 11  # Number of collocation points
        self.M = 15  # Number of frequencies
        self.psi = np.pi * (0.7)  # TODO change
        initial_pos = np.array([0, 0, 10])  # TODO generalize

        self.system_dynamics = system_dynamics

        # Add state trajectory parameters as decision variables
        self.coeffs = self.prog.NewContinuousVariables(
            3, self.M + 1, "c"
        )  # (x,y,z) for every frequency
        self.phase_delays = self.prog.NewContinuousVariables(3, self.M, "eta")
        self.t_f = self.prog.NewContinuousVariables(1, 1, "t_f")[0, 0]
        self.avg_vel = self.prog.NewContinuousVariables(1, 1, "V_bar")[0, 0]

        # Add input values as decision variables
        # TODO or use differential flatness property??

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
            for residual in residuals[3:6]:  # TODO only thee last three are residuals
                self.prog.AddConstraint(residual == 0)

        # Add constraints on coefficients
        for coeff in self.coeffs.T:
            lb = np.array([-500, -500, -500])
            ub = np.array([500, 500, 500])
            self.prog.AddBoundingBoxConstraint(lb, ub, coeff)

        # Add constraints on phase delays
        for etas in self.phase_delays.T:
            lb = np.array([0, 0, 0])
            ub = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])
            self.prog.AddBoundingBoxConstraint(lb, ub, etas)

        # Add constraints on final time and avg vel
        self.prog.AddBoundingBoxConstraint(0.5, 200, self.t_f)  # TODO better limit?
        self.prog.AddBoundingBoxConstraint(0.5, 150, self.avg_vel)

        # TODO add constraints on state

        # Add objective function
        self.prog.AddCost(-self.avg_vel)

        self.result = Solve(self.prog)
        assert self.result.is_success()

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
        for i in range(sim_N):
            t = times[i]
            pos = self.evaluate_pos_traj(
                coeffs_opt, phase_delays_opt, t_f_opt, avg_vel_opt, t
            )
            pos_traj[:, i] = pos

        # TODO move plotting out
        plot_trj_3_wind(pos_traj.T, np.array([np.sin(self.psi), np.cos(self.psi), 0]))
        plt.show()
        breakpoint()
        # TODO make this work


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

    def get_pos_fourier(self, collocation_time):
        pos_traj = np.copy(self.coeffs[:, 0])
        for m in range(1, self.M):
            sine_terms = np.array(
                [
                    np.sin(
                        (2 * np.pi * m * collocation_time) / self.t_f
                        + self.phase_delays[0, m]
                    ),
                    np.sin(
                        (2 * np.pi * m * collocation_time) / self.t_f
                        + self.phase_delays[1, m]
                    ),
                    np.sin(
                        (2 * np.pi * m * collocation_time) / self.t_f
                        + self.phase_delays[2, m]
                    ),
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
            cos_terms = np.array(
                [
                    (2 * np.pi * m / self.t_f)
                    * np.cos(
                        (2 * np.pi * m * collocation_time) / self.t_f
                        + self.phase_delays[0, m]
                    ),
                    (2 * np.pi * m / self.t_f)
                    * np.cos(
                        (2 * np.pi * m * collocation_time) / self.t_f
                        + self.phase_delays[1, m]
                    ),
                    (2 * np.pi * m / self.t_f)
                    * np.cos(
                        (2 * np.pi * m * collocation_time) / self.t_f
                        + self.phase_delays[2, m]
                    ),
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
            sine_terms = np.array(
                [
                    -((2 * np.pi * m / self.t_f) ** 2)
                    * np.cos(
                        (2 * np.pi * m * collocation_time) / self.t_f
                        + self.phase_delays[0, m]
                    ),
                    -((2 * np.pi * m / self.t_f) ** 2)
                    * np.cos(
                        (2 * np.pi * m * collocation_time) / self.t_f
                        + self.phase_delays[1, m]
                    ),
                    -((2 * np.pi * m / self.t_f) ** 2)
                    * np.cos(
                        (2 * np.pi * m * collocation_time) / self.t_f
                        + self.phase_delays[2, m]
                    ),
                ]
            )
            vel_dot_traj += self.coeffs[:, m] * sine_terms
        return vel_dot_traj

    def continuous_dynamics(self, pos, vel, vel_dot):
        x = np.concatenate((pos, vel))
        x_dot = np.concatenate((vel, vel_dot))
        u = np.array([0, 0, 0])

        return x_dot - self.system_dynamics(x, u)
