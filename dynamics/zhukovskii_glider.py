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

from dynamics.wind_models import (
    wind_model,
    ddt_wind_model,
    get_wind_vector,
    get_dimless_wind_vector,
)


class ZhukovskiiGlider:
    def __init__(self):
        self.rho = 1.255  # g/m**3 Air density

        self.e_z = np.array([0, 0, 1])  # Unit vector along z axis
        self.b = 3  # m Wing span
        self.A = 0.5  # m**2 Wing area
        self.glider_length = 2  # m NOTE only used for visualization
        self.AR = self.b ** 2 / self.A  # Aspect ratio, b**2 / wing_area
        self.M = 4.5  # kg Mass

        # TODO the efficiency etc must be related to the area etc in some way
        self.Lambda = 40  # Lift-to-drag ratio
        self.efficiency = 1 / self.Lambda  # Small efficiency parameter
        self.V_l = 15  # m/s Optimal glide vel
        self.G = 9.81  # m/s**2 Graviational constant
        self.L = self.V_l ** 2 / self.G  # Characteristic length
        self.T = self.V_l / self.G  # Characteristic time
        self.C = (self.M * self.G) / (
            self.rho * self.V_l
        )  # Norm of circulation vector in steady flight

        # Optimization constraints
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

        self.drake_plant = DrakeZhukovskiiGlider(3, self.continuous_dynamics_dimless)
        self.drake_plant_diff_flat = DrakeZhukovskiiGlider(
            4, self.continuous_dynamics_diff_flat_dimless
        )
        return

    def get_params(self):
        params = (
            self.M,
            self.rho,
            self.AR,
            self.Lambda,
            self.efficiency,
            self.V_l,
            self.G,
            self.L,
            self.T,
            self.C,
        )
        return params

    def get_constraints(self):
        constraints = (
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

    def get_drake_plant(self, diff_flat):
        if diff_flat:
            return self.drake_plant_diff_flat
        return self.drake_plant

    def get_lift_coeff(self, x, u):
        c = u
        vel_rel = self.get_vel_rel(x)
        c_l = np.linalg.norm(c) / ((1 / 2) * self.A * np.linalg.norm(vel_rel))

        return c_l

    def get_roll(self, u):

        return

    # TODO something is strange with this
    def get_angle_of_attack(self, x, u):
        c = u
        v_r = self.get_vel_rel(x)

        alpha = np.arcsin(
            (1 + (2 / self.AR))
            / (np.pi * self.A)
            * (np.linalg.norm(c) / np.linalg.norm(v_r))
        )
        return alpha

    def get_vel_rel(self, x):
        pos = x[0:3]
        vel = x[3:6]
        return vel - get_wind_vector(pos[2])

    def continuous_dynamics_dimless(self, x, u):
        # NOTE Only depends on efficiency parameter
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

        vel_dot = -self.e_z - (
            self.efficiency * l_term * np.eye(3) + skew_matrix(c)
        ).dot(vel_rel)

        x_dot = np.concatenate((vel, vel_dot))

        return x_dot

    def continuous_dynamics_diff_flat_dimless(self, x, u):
        # TODO implement
        # x = [x, y, z, xdot, ydot, zdot]
        # u = circulation vector, brake param
        # Only depends on efficiency parameter
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

@TemplateSystem.define("DrakeZhukovskiiGlider_")
def DrakeZhukovskiiGlider_(T):
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


DrakeZhukovskiiGlider = DrakeZhukovskiiGlider_[None]


def skew_matrix(v):
    S = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return S
