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
    def __init__(
        self,
        m=8.5,
        c_Dp=0.033,
        A=0.65,
        b=3.306,
        rho=1.255,
        g=9.81,
        max_bank_angle=80 * np.pi / 180,
        max_lift_coeff=1.5,
        min_lift_coeff=0,
        max_load_factor=3,
        min_height=0.5,
        max_height=100,
        h0=5,
    ):
        # Set model params
        self.set_params(b, A, m, c_Dp, rho, g)
        self.e_z = np.array([0, 0, 1])  # Unit vector along z axis

        # Optimization constraints
        self.max_bank_angle = max_bank_angle
        self.max_lift_coeff = max_lift_coeff
        self.min_lift_coeff = min_lift_coeff
        self.max_load_factor = max_load_factor
        self.min_height = min_height
        self.max_height = max_height
        self.min_travelled_distance = self.L * 0.67  # m TODO is this good?
        self.h0 = h0
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

    def get_char_time(self):
        return self.T

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

    # TODO unused
    # NOTE This is meant to be used for dimensionalized inputs and outputs
    def calc_abs_vel(self, h, v_r):
        w = get_wind_vector(h)
        v = v_r + w
        return v

    # NOTE This is meant to be used for dimensionalized inputs and outputs
    def calc_heading(self, h, v_r):
        # Assumes psi meaured from x-axis in NED frame (i.e. heading is measured from the north)
        psi = np.arctan2(v_r[1], v_r[0])
        return psi

    def calc_rel_flight_path_angle(self, v_r):
        v_r_norm = np.linalg.norm(v_r)
        gamma = np.arcsin(-v_r[2] / v_r_norm)  # Relative flight path angle
        return gamma

    def calc_bank_angle(self, v_r, c):
        gamma = self.calc_rel_flight_path_angle(v_r)
        c_norm = np.linalg.norm(c)

        phi = np.arcsin(c[2] / (c_norm) * np.cos(gamma))
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
        # NOTE This actually uses ENU frame, not NED. i.e., z is positive upwards
        # somehow this is better for numerics
        # x = [x, y, h, [v_r]]
        # h = -z
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
            1 / self.Lam * l_term * np.eye(3) + dw_dx - skew_matrix(c)
        ).dot(v_r)
        p_dot = v_r + w

        x_dot = np.concatenate((p_dot, v_r_dot))
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
