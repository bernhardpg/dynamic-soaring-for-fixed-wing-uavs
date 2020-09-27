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

def get_sim_params():
    M = 4.5  # kg Mass
    rho = 1.255  # g/m**3 Air density

    Lambda = 40  # Lift-to-drag ratio
    efficiency = 1 / Lambda  # Small efficiency parameter
    V_l = 15  # m/s Optimal glide speed
    G = 9.81  # m/s**2 Graviational constant
    L = V_l ** 2 / G  # Characteristic length
    T = V_l / G  # Characteristic time
    C = (M * G) / (rho * V_l)  # Norm of circulation vector in steady flight

    sim_params = (M, rho, Lambda, efficiency, V_l, G, L, T, C)
    return sim_params

# From Mortens notes
@TemplateSystem.define("ZhukovskiiGliderDimless_")
def ZhukovskiiGliderDimless_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, converter=None):
            LeafSystem_[T].__init__(self, converter)

            # Three inputs
            self.DeclareVectorInputPort("u", BasicVector_[T](3))
            # Six outputs (full state)
            self.DeclareVectorOutputPort("x", BasicVector_[T](6), self.CopyStateOut)
            # Three positions, three velocities
            self.DeclareContinuousState(3, 3, 0)
            # State = [x, y, z, xdot, ydot, zdot]

            # Constants
            self.e_z = np.array([0, 0, 1])  # Unit vector along z axis
            self.Gamma = 40  # Optimal lift to drag ratio
            self.efficiency = 1 / self.Gamma  # Small efficiency number
            self.V_l = 20
            self.G = 9.81  # Gravitational constant
            self.L = self.V_l ** 2 / self.G  # Characteristic length
            self.T = self.V_l / self.G  # Characteristic time

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def DoCalcTimeDerivatives(self, context, derivatives):
            # NOTE all variabled dimless here
            # x, y, z, xdot, ydot, zdot
            x = context.get_continuous_state_vector().CopyToVector()
            u = self.EvalVectorInput(context, 0).CopyToVector()

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

            derivatives.get_mutable_vector().SetFromVector(
                np.concatenate((vel, vel_dot))
            )

        # y = x
        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            output.SetFromVector(x)

    return Impl


ZhukovskiiGliderDimless = ZhukovskiiGliderDimless_[None]


@TemplateSystem.define("ZhukovskiiGlider_")
def ZhukovskiiGlider_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, converter=None):
            LeafSystem_[T].__init__(self, converter)

            # two inputs (thrust)
            self.DeclareVectorInputPort("u", BasicVector_[T](3))
            # six outputs (full state)
            self.DeclareVectorOutputPort("x", BasicVector_[T](6), self.CopyStateOut)
            # three positions, three velocities
            self.DeclareContinuousState(3, 3, 0)
            # state = [x, y, z, xdot, ydot, zdot]

            # Constants
            # Values taken from Deittert et al.
            self.rho = 1.255  # g/m**3 air_density
            self.S = 0.473  # m**2 wing_area
            self.c_d0 = 0.0173  # parasitic drag
            self.c_d2 = 0.0517  # lift induced drag constant
            self.wingspan = 3  # m
            self.m = 4.5  # kg
            self.g = 9.81  # gravity
            self.g_vec = np.array([0, 0, -self.g])
            self.Gamma = 45  # Optimal lift to drag ratio
            self.V_l = 20  # Level flight speed that achieves LDR

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def DoCalcTimeDerivatives(self, context, derivatives):
            # x, y, z, xdot, ydot, zdot
            x = context.get_continuous_state_vector().CopyToVector()
            u = self.EvalVectorInput(context, 0).CopyToVector()

            c = u
            pos = x[0:3]
            vel = x[3:6]

            wind = get_wind_vector(pos[2])
            vel_rel = vel - wind

            # NOTE Original expression which is numerically bad
            # d = ((self.m * self.g) / (self.rho * self.Gamma * self.V_l)) * l(
            #    np.linalg.norm(vel_rel) / self.V_l,
            #    (self.rho * self.V_l * np.linalg.norm(c)) / (self.m * self.g),
            # )

            # NOTE necessary rewrite to deal with gradients of vector norms being horrible
            epsilon = 0.001
            l_term = (
                (vel_rel.T.dot(vel_rel)) / (self.V_l ** 2)
                + (self.rho ** 2 * self.V_l ** 2 * c.T.dot(c)) / (self.m * self.g) ** 2
            ) / (2 * (np.sqrt(vel_rel.T.dot(vel_rel) + epsilon) / self.V_l))

            d = ((self.m * self.g) / (self.rho * self.Gamma * self.V_l)) * l_term

            vel_dot = (1 / self.m) * (
                self.m * self.g_vec
                - self.rho * (d * np.eye(3) + skew_matrix(c)).dot(vel_rel)
            )

            derivatives.get_mutable_vector().SetFromVector(
                np.concatenate((vel, vel_dot))
            )

        # y = x
        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            output.SetFromVector(x)

    return Impl


ZhukovskiiGlider = ZhukovskiiGlider_[None]


# TODO currently unused
def l(w, c):
    return (w ** 2 + c ** 2) / (2 * w)


def skew_matrix(v):
    S = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return S
