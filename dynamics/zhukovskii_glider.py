import numpy as np
from pydrake.all import (
    eq,
    MathematicalProgram,
    Solve,
    Variable,
    Expression,
    BasicVector_,
    TemplateSystem,
    LeafSystem_,
)
from math import sqrt

from dynamics.wind_models import (
    wind_model,
    ddt_wind_model,
    get_wind_vector,
    get_wind_jacobian,
)


class RelativeZhukovskiiGlider:
    def __init__(self, b=3.306, A=0.65, m=8.5, c_Dp=0.033, rho=1.255, g=9.81):
        # Set model params
        self.set_params(b, A, m, c_Dp, rho, g)
        self.e_z = np.array([0, 0, 1])  # Unit vector along z axis

        # Optimization constraints
        self.max_bank_angle = 80 * np.pi / 180  # radians
        self.max_lift_coeff = 1.5
        self.min_lift_coeff = 0
        self.max_load_factor = 3
        self.min_height = 0.5  # m
        self.max_height = 100  # m
        self.min_travelled_distance = 5  # m
        self.t_f_min = 0.5  # s
        self.t_f_max = 200  # s
        self.avg_vel_min = 2  # s
        self.avg_vel_max = 100  # s
        self.h0 = 20  # m
        # TODO remove unused constraints
        return

    def set_params(self, b, A, m, c_Dp, rho, g):
        self.b = b
        self.A = A
        self.m = m
        self.c_Dp = c_Dp
        self.rho = rho
        self.g = g

        self.AR = self.b ** 2 / self.A  # Aspect ratio, b**2 / wing_area

        # Performance params
        self.Lam = self.calc_opt_glide_ratio(self.AR, self.c_Dp)  # Optimal glide ratio
        self.Th = self.calc_opt_glide_angle(self.AR, self.c_Dp)  # Optimal glide angle
        self.V_opt = self.calc_opt_glide_speed(
            self.AR, self.c_Dp, self.m, self.A, self.b, self.rho, self.g
        )  # Optimal glide speed
        self.V_l = self.calc_opt_level_glide_speed(
            self.AR, self.c_Dp, self.m, self.A, self.b, self.rho, self.g
        )  # Optimal level glide speed

        # Characteristic values
        self.L = self.V_l ** 2 / self.g  # Characteristic length
        self.T = self.V_l / self.g  # Characteristic time
        self.C = (self.m * self.g) / (
            self.rho * self.V_l
        )  # Norm of circulation vector in steady flight
        return

    def calc_opt_glide_ratio(self, AR, c_Dp):
        glide_ratio = 0.5 * sqrt(np.pi * AR / c_Dp)
        return glide_ratio

    def calc_opt_glide_angle(self, AR, c_Dp):
        opt_glide_ratio = self.calc_opt_glide_ratio(AR, c_Dp)
        opt_glide_angle = np.arctan(1 / opt_glide_ratio)
        return opt_glide_angle

    def calc_opt_glide_speed(self, AR, c_Dp, m, A, b, rho, g):
        opt_glide_angle = self.calc_opt_glide_angle(AR, c_Dp)
        opt_glide_speed = sqrt(
            2 * m * g * np.cos(opt_glide_angle) / (sqrt(np.pi * c_Dp * A) * rho * b)
        )
        return opt_glide_speed

    def calc_opt_level_glide_speed(self, AR, c_Dp, m, A, b, rho, g):
        opt_glide_angle = self.calc_opt_glide_angle(AR, c_Dp)
        opt_glide_speed = self.calc_opt_glide_speed(AR, c_Dp, m, A, b, rho, g)
        opt_level_glide_speed = opt_glide_speed / sqrt(np.cos(opt_glide_angle))
        return opt_level_glide_speed

    # NOTE for reconstructing trajectories.
    # This is meant to be used for dimensionalized inputs and outputs
    # TODO currently unused
    def calc_vel(self, x):
        p = x[0:3]
        v_r = x[3:6]
        w = get_wind_vector(pos[2])
        v = v_r + w
        return v

    def calc_bank_angle(self, v_r, c):
        v_r_norm = np.linalg.norm(v_r)
        gamma = np.arcsin(-v_r[2] / v_r_norm) # Relative flight path angle
        c_norm = np.linalg.norm(c)

        phi = np.arcsin(c[2] / (c_norm) * np.cos(gamma))

#        temp = c[2] / (
#            np.linalg.norm(c) * np.sqrt(1 - (v_r[2] ** 2) / (v_r.T.dot(v_r)))
#        )
#        if temp > 1 or temp < -1:
#            breakpoint()
        #phi = np.arcsin(temp)
        return phi

    def calc_lift_coeff(self, v_r, c, A):
        c_norm = np.linalg.norm(c)
        v_r_norm = np.linalg.norm(v_r)

        c_l = c_norm / (0.5 * A * v_r_norm)
        return c_l

    def calc_load_factor(self, v_r, c, m, g, rho):
        c_norm = np.linalg.norm(c)
        v_r_norm = np.linalg.norm(v_r)
        lift = rho * c_norm * v_r_norm
        weight = m * g

        n = lift / weight
        return n

    def get_char_values(self):
        return (
            self.V_l,
            self.L,
            self.T,
            self.C,
        )

    def get_wing_area(self):
        return self.A

    def get_constraints(self):
        constraints = (
            self.max_bank_angle,
            self.max_lift_coeff,
            self.min_lift_coeff,
            self.max_load_factor,
            self.min_height,
            self.max_height,
            self.h0,
            self.min_travelled_distance,
        )
        return constraints

    def get_constraints_dimless(self):
        constraints_dimless = (
            self.min_height / self.L,
            self.max_height / self.L,
            self.min_vel / self.V_l,
            self.max_vel / self.V_l,
            self.h0 / self.L,
            self.min_travelled_distance / self.L,
            self.t_f_min / self.T,
            self.t_f_max / self.T,
            self.avg_vel_min / self.V_l,
            self.avg_vel_max / self.V_l,
        )

        return constraints_dimless

    def create_drake_plant(self):
        return DrakeSysWrapper(3, self.continuous_dynamics_dimless)

    def continuous_dynamics_dimless(self, x, u):
        # x = [p, v_r]
        # u = c
        c = u
        p = x[0:3]
        v_r = x[3:6]

        w = get_wind_vector(self.L * p[2]) / self.V_l  # Nondimensionalized wind
        dw_dx = get_wind_jacobian(self.L * p[2]) * (
            self.L / self.V_l
        )  # Nondimenionalized wind jacobian

        # NOTE necessary to add a small epsilon to deal
        # with gradients of vector norms being horrible
        epsilon = 0.001
        l_term = (v_r.T.dot(v_r) + c.T.dot(c)) / (2 * np.sqrt(v_r.T.dot(v_r) + epsilon))

        v_r_dot = -self.e_z - (
            1 / self.Lam * l_term * np.eye(3) + dw_dx + skew_matrix(c)
        ).dot(v_r)
        p_dot = v_r + w

        x_dot = np.concatenate((p_dot, v_r_dot))
        return x_dot


# NOTE Currently unused, as the relative formulation allows for much easier constraint handling
class ZhukovskiiGlider:
    def __init__(self, b=3.306, A=0.65, m=8.5, c_Dp=0.033, rho=1.255, g=9.81):
        # Set model params
        self.set_params(b, A, m, c_Dp, rho, g)
        self.e_z = np.array([0, 0, 1])  # Unit vector along z axis

        # Optimization constraints
        self.max_bank_angle = 50 * np.pi / 180  # radians
        self.min_height = 0.5  # m
        self.max_height = 100  # m
        self.min_vel = 5  # m/s
        self.max_vel = 60  # m/s
        self.h0 = 20  # m
        self.min_travelled_distance = 5  # m
        self.t_f_min = 0.5  # s
        self.t_f_max = 200  # s
        self.avg_vel_min = 2  # s
        self.avg_vel_max = 100  # s
        return

    def set_params(self, b, A, m, c_Dp, rho, g):
        self.b = b
        self.A = A
        self.m = m
        self.c_Dp = c_Dp
        self.rho = rho
        self.g = g

        self.AR = self.b ** 2 / self.A  # Aspect ratio, b**2 / wing_area

        # Performance params
        self.Lam = self.calc_opt_glide_ratio(self.AR, self.c_Dp)  # Optimal glide ratio
        self.Th = self.calc_opt_glide_angle(self.AR, self.c_Dp)  # Optimal glide angle
        self.V_opt = self.calc_opt_glide_speed(
            self.AR, self.c_Dp, self.m, self.A, self.b, self.rho, self.g
        )  # Optimal glide speed
        self.V_l = self.calc_opt_level_glide_speed(
            self.AR, self.c_Dp, self.m, self.A, self.b, self.rho, self.g
        )  # Optimal level glide speed

        # Characteristic values
        self.L = self.V_l ** 2 / self.g  # Characteristic length
        self.T = self.V_l / self.g  # Characteristic time
        self.C = (self.m * self.g) / (
            self.rho * self.V_l
        )  # Norm of circulation vector in steady flight
        return

    def calc_opt_glide_ratio(self, AR, c_Dp):
        glide_ratio = 0.5 * sqrt(np.pi * AR / c_Dp)
        return glide_ratio

    def calc_opt_glide_angle(self, AR, c_Dp):
        opt_glide_ratio = self.calc_opt_glide_ratio(AR, c_Dp)
        opt_glide_angle = np.arctan(1 / opt_glide_ratio)
        return opt_glide_angle

    def calc_opt_glide_speed(self, AR, c_Dp, m, A, b, rho, g):
        opt_glide_angle = self.calc_opt_glide_angle(AR, c_Dp)
        opt_glide_speed = sqrt(
            2 * m * g * np.cos(opt_glide_angle) / (sqrt(np.pi * c_Dp * A) * rho * b)
        )
        return opt_glide_speed

    def calc_opt_level_glide_speed(self, AR, c_Dp, m, A, b, rho, g):
        opt_glide_angle = self.calc_opt_glide_angle(AR, c_Dp)
        opt_glide_speed = self.calc_opt_glide_speed(AR, c_Dp, m, A, b, rho, g)
        opt_level_glide_speed = opt_glide_speed / sqrt(np.cos(opt_glide_angle))
        return opt_level_glide_speed

    def get_char_values(self):
        return (
            self.V_l,
            self.L,
            self.T,
            self.C,
        )

    def get_constraints(self):
        constraints = (
            self.max_bank_angle,
            self.min_height,
            self.max_height,
            self.min_vel,
            self.max_vel,
            self.h0,
            self.min_travelled_distance,
        )
        return constraints

    def get_constraints_dimless(self):
        constraints_dimless = (
            self.min_height / self.L,
            self.max_height / self.L,
            self.min_vel / self.V_l,
            self.max_vel / self.V_l,
            self.h0 / self.L,
            self.min_travelled_distance / self.L,
            self.t_f_min / self.T,
            self.t_f_max / self.T,
            self.avg_vel_min / self.V_l,
            self.avg_vel_max / self.V_l,
        )

        return constraints_dimless

    def create_drake_plant(self, diff_flat):
        if diff_flat:
            return DrakeSysWrapper(4, self.continuous_dynamics_diff_flat_dimless)
        return DrakeSysWrapper(3, self.continuous_dynamics_dimless)

    def calc_lift_coeff(self, x, c, A):
        v_r = self.get_vel_rel(x)
        c_norm = np.linalg.norm(c)
        v_r_norm = np.linalg.norm(v_r)

        c_l = c_norm / (0.5 * A * v_r_norm)
        return c_l

    def calc_rel_flight_path_angle(self, x):
        # TODO unused
        v_r = self.get_vel_rel(x)
        v_rz = v_r[2]
        v_r_norm = sqrt(v_r.T.dot(v_r))
        gamma = np.arcsin(-v_rz / v_r_norm)
        return gamma

    def calc_bank_angle():
        # TODO implement
        phi = None
        return phi

    def get_vel_rel(self, x):
        pos = x[0:3]
        vel = x[3:6]
        return vel - get_wind_vector(pos[2])  # TODO Should not this be dimensionless??

    def continuous_dynamics_dimless(self, x, u):
        # x = [x, y, z, xdot, ydot, zdot]
        # u = circulation vector
        c = u
        pos = x[0:3]
        vel = x[3:6]

        dimless_wind = get_dimless_wind_vector(pos[2], self.L, self.V_l)
        vel_rel = vel - dimless_wind

        # NOTE necessary to add a small epsilon to deal
        # with gradients of vector norms being horrible
        epsilon = 0.001
        l_term = (vel_rel.T.dot(vel_rel) + c.T.dot(c)) / (
            2 * np.sqrt(vel_rel.T.dot(vel_rel) + epsilon)
        )

        vel_dot = -self.e_z - (1 / self.Lam * l_term * np.eye(3) + skew_matrix(c)).dot(
            vel_rel
        )

        x_dot = np.concatenate((vel, vel_dot))

        return x_dot

    def continuous_dynamics_diff_flat_dimless(self, x, u):
        # TODO implement
        # x = [x, y, z, xdot, ydot, zdot]
        # u = circulation vector, brake param
        c = u[0:3]
        b = u[3]
        pos = x[0:3]
        vel = x[3:6]

        dimless_wind = get_dimless_wind_vector(pos[2], self.L, self.V_l)
        vel_rel = vel - dimless_wind

        # NOTE necessary to add a small epsilon to deal
        # with gradients of vector norms being horrible
        epsilon = 0.001
        l_term = (vel_rel.T.dot(vel_rel) + c.T.dot(c)) / (
            2 * np.sqrt(vel_rel.T.dot(vel_rel) + epsilon)
        )

        vel_dot = -self.e_z - (
            b + self.efficiency * l_term * np.eye(3) + skew_matrix(c)
        ).dot(vel_rel)

        x_dot = np.concatenate((vel, vel_dot))

        return x_dot


@TemplateSystem.define("DrakeSysWrapper_")
def DrakeSysWrapper_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, num_inputs, continuous_dynamics, converter=None):
            LeafSystem_[T].__init__(self, converter)

            self.DeclareVectorInputPort("u", BasicVector_[T](num_inputs))
            # Six outputs (full state)
            self.DeclareVectorOutputPort("x", BasicVector_[T](6), self.CopyStateOut)
            # Three positions, three velocities
            self.DeclareContinuousState(3, 3, 0)
            self.continuous_dynamics = continuous_dynamics
            self.num_inputs = num_inputs

        def _construct_copy(self, other, converter=None):
            Impl._construct(
                self, other.num_inputs, other.continuous_dynamics, converter=converter
            )

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            u = self.EvalVectorInput(context, 0).CopyToVector()
            x_dot = self.continuous_dynamics(x, u)
            derivatives.get_mutable_vector().SetFromVector(x_dot)

        # y = x
        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            output.SetFromVector(x)

    return Impl


DrakeSysWrapper = DrakeSysWrapper_[None]


def skew_matrix(v):
    S = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return S
