import numpy as np
import pydrake.symbolic as sym
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

from dynamics.wind_models import wind_model, ddt_wind_model

# NOTE not used for final results. Left simply as a reference

# See this example for example implementation:
# https://github.com/RussTedrake/underactuated/blob/master/underactuated/quadrotor2d.py#L44

# Drake system for slotine dynamics
@TemplateSystem.define("ZhaoGlider_")
def ZhaoGlider_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, converter=None):
            LeafSystem_[T].__init__(self, converter)

            # two inputs (thrust)
            self.DeclareVectorInputPort("u", BasicVector_[T](2))
            # six outputs (full state)
            self.DeclareVectorOutputPort("x", BasicVector_[T](6), self.CopyStateOut)
            # three positions, three velocities
            self.DeclareContinuousState(6)

            # Constants
            # Values taken from Deittert et al.
            self.rho = 1.255  # g/m**3 air_density
            self.S = 0.473  # m**2 wing_area
            self.c_d0 = 0.0173  # parasitic drag
            self.c_d2 = 0.0517  # lift induced drag constant
            self.wingspan = 3  # m
            self.m = 4.5  # kg
            self.g = 9.81  # gravity

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            u = self.EvalVectorInput(context, 0).CopyToVector()

            # glider_speed, heading, flight_path_angle, z, x, y
            V, psi, gamma, height, pos_x, pos_y = x
            # lift_coeff, roll_angle
            c_l, phi = u

            c_d = self.c_d0 + self.c_d2 * c_l ** 2
            L = 0.5 * c_l * self.rho * self.S * V ** 2
            D = 0.5 * c_d * self.rho * self.S * V ** 2

            height_dot = V * np.sin(gamma)

            W = wind_model(height)
            W_dot = ddt_wind_model(height, height_dot)

            V_dot = (1 / self.m) * (
                -D
                - self.m * self.g * np.sin(gamma)
                - self.m * W_dot * np.cos(gamma) * np.sin(psi)
            )

            gamma_dot = (1 / (self.m * V)) * (
                L * np.cos(phi)
                - self.m * self.g * np.cos(gamma)
                - self.m * W_dot * np.sin(gamma) * np.sin(psi)
            )

            psi_dot = (1 / (self.m * V * np.cos(gamma))) * (
                L * np.sin(phi) + self.m * W_dot * np.cos(psi)
            )  # NOTE gamma != pi/2

            vel_x = V * np.cos(gamma) * np.cos(psi)
            vel_y = V * np.cos(gamma) * np.sin(psi) - W

            # Dynamics
            xdot = np.empty(6, dtype=Expression)
            xdot[0] = V_dot
            xdot[1] = psi_dot
            xdot[2] = gamma_dot
            xdot[3] = height_dot
            xdot[4] = vel_x
            xdot[5] = vel_y

            derivatives.get_mutable_vector().SetFromVector(xdot)

        # y = x
        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            output.SetFromVector(x)

    return Impl


ZhaoGlider = ZhaoGlider_[None]


# TODO remove this
# Dynamics taken from Slotine: successive shallow arcs
def continuous_dynamics(state, u):
    me = sym if state.dtype == object else np  # check type for autodiff
    # me = math engine

    # Constants
    # Values taken from Deittert et al.
    rho = 1.255  # g/m**3 air_density
    S = 0.473  # m**2 wing_area
    c_d0 = 0.0173  # parasitic drag
    c_d2 = 0.0517  # lift induced drag constant
    wingspan = 3  # m
    m = 4.5  # kg
    g = 9.81  # gravity

    V, psi, gamma, z_pos, x_pos, y_pos = state
    # Heading defined s.t. psi = 0 is along positive x-axis
    # glider_speed, heading, flight_path_angle, z, x, y
    c_l, phi = u
    # u = [lift_coeff, roll_angle]

    # D = drag
    # L = lift
    # W = wind speed

    c_d = c_d0 + c_d2 * c_l ** 2
    L = 0.5 * c_l * rho * S * V ** 2
    D = 0.5 * c_d * rho * S * V ** 2

    z_dot = V * me.sin(gamma)

    W = linear_wind_model(z_pos)
    W_dot = ddt_linear_wind_model(z_pos, z_dot)

    V_dot = (1 / m) * (
        -D - m * g * me.sin(gamma) - m * W_dot * me.cos(gamma) * me.sin(psi)
    )

    gamma_dot = (1 / (m * V)) * (
        L * me.cos(phi)
        - m * g * me.cos(gamma)
        - m * W_dot * me.sin(gamma) * me.sin(psi)
    )

    psi_dot = (1 / (m * V * me.cos(gamma))) * (
        L * me.sin(phi) + m * W_dot * me.cos(psi)
    )  # NOTE gamma != pi/2

    x_dot = V * me.cos(gamma) * me.cos(psi)
    y_dot = V * me.cos(gamma) * me.sin(psi) - W

    # Dynamics
    state_dot = np.empty(6, dtype=Expression)
    state_dot[0] = V_dot
    state_dot[1] = psi_dot
    state_dot[2] = gamma_dot
    state_dot[3] = z_dot
    state_dot[4] = x_dot
    state_dot[5] = y_dot

    return state_dot


