from collections.abc import Callable
from typing import ClassVar
from typing_extensions import TypeAlias
import jax

import jax.numpy as jnp
import optimistix as optx
from equinox.internal import ω

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y, Z
from .._heuristics import is_sde
from .._local_interpolation import LocalLinearInterpolation, LocalLinearInterpolationDAE
from .._root_finder import with_stepsize_controller_tols
from .._solution import RESULTS
from .._term import AbstractTerm, AbstractTermDAE
from .basedae import AbstractAdaptiveSolver, AbstractImplicitSolverDAE
from .._root_finder._verychord import VeryChord

_SolverState: TypeAlias = None

jax.config.update("jax_debug_nans", True)

def _implicit_relation(f, nonlinear_solve_args):
    f1, f2 = f
    vf_prod, t1, y0, z0, args, control = nonlinear_solve_args
    f1_0 = f1, 0
    f2_0 = 0, f2

    z_i = z0[0]
    x_i = y0[0]
    y_i = y0[1]
    v_i = y0[2]
    w_i = y0[3]

    z_f = (f2)**ω
    x_f = (f1[0])**ω
    y_f= (f1[1])**ω
    v_f = (f1[2])**ω
    w_f= (f1[3])**ω

    # f1_0 = f1, jnp.array([0], dtype=float)
    # f2_0 = jnp.array([0]), f2
    # df = (vf_prod(t1, (y0**ω + f1**ω).ω, (z0**ω).ω, args, control) ** ω - f1**ω).ω
    # dz = (vf_prod(t1, (y0**ω).ω, (z0**ω + f2**ω).ω, args, control) ** ω - f2**ω).ω
    # jax.debug.print("f1_0: {}, y0: {}, z0: {}, t1: {}", f1_0, y0, z0, t1)
    # return dy, dz
    # df1, df2 = (vf_prod(t1, (y0**ω + f1**ω).ω, (z0**ω).ω, args, control) ** ω - f**ω).ω
    # dg1, dg2 = (vf_prod(t1, (y0**ω).ω, (z0**ω).ω, args, control) ** ω - f**ω).ω
    # diff1 = df1
    # diff2 = dg2

    # dy = (vf_prod(t1, (y0**ω + f1**ω).ω, (z0**ω).ω, args, control) ** ω - f1_0**ω).ω
    # diff1 = (x_term_f1).ω
    # diff2 = ((y_term_f1 + y_term_y0) - v_term_f1 + y_term_f1).ω
    # diff3 = (x_term_f1*(z_term_z0 + 1) + u_term_f1 + z_term_f1*(x_term_y0 + x_term_f1)).ω
    # diff4 = (y_term_f1*(-1)*(z_term_z0 + z_term_f1) + v_term_f1 + z_term_f1*(y_term_y0 + y_term_f1)).ω
    # diff5 = (2*x_term_f1*(x_term_y0 + x_term_f1) + 2*y_term_f1*(y_term_y0 + y_term_f1)).ω

    # one_zero = jnp.array([0,])
    # f1, one_zero = f1_0
    # four_zeroes = jnp.array([0, 0, 0, 0,])
    # four_zeroes, f2 = f2_0 

    df_dy, dg_dy = (vf_prod(t1, (y0**ω + f1**ω).ω, (z0**ω).ω, args, control) ** ω - f1_0**ω).ω
    df_dz, dg_dz = (vf_prod(t1, (y0**ω).ω, (z0**ω + f2**ω).ω, args, control) ** ω - f1_0**ω).ω
    diff1 = df_dy + df_dz
    diff2 = dg_dy + dg_dz
    return diff1, diff2

    # vf = (vf_prod(t1, (y0**ω + f1**ω).ω, (z0**ω).ω, args, control)**ω - f**ω).ω
    # vf_x = vf[0][0]
    # vf_y = vf[0][1]
    # vf_v = vf[0][2]
    # vf_w = vf[0][3]
    # vf_z = vf[0]

    # diff1 = (vf_x - x_f).ω
    # diff2 = (vf_y - y_f).ω
    # diff3 = (vf_v - v_f).ω
    # diff4 = (vf_w - w_f).ω
    # diff5 = (vf_z - z_f).ω
    # # jax.debug.print("f: {}, y0: {}, z0: {}, t1: {}", f, y0, z0, t1)
    # return diff1, diff2, diff3, diff4, diff5
    # return vf

    # diff1 = (-x_f + t1*(v_i + v_f)).ω
    # diff2 = (-y_f + t1*(w_i + w_f)).ω
    # diff3 = (-v_f + t1*(z_i + z_f)*(x_i + x_f)).ω
    # diff4 = (-w_f + t1*(z_i + z_f)*(y_i + y_f)).ω
    # diff5 = (-z_f + t1*(x_i + x_f)**2 + t1*(y_i + y_f)**2 - 1).ω
    # jax.debug.print("f: {}, y0: {}, z0: {}, t1: {}", f, y0, z0, t1)
    # return diff1, diff2, diff3, diff4, diff5

class Implicit_Euler_DAE(AbstractImplicitSolverDAE, AbstractAdaptiveSolver):
    r"""Implicit Euler method.

    A-B-L stable 1st order SDIRK method. Has an embedded 2nd order Heun method for
    adaptive step sizing. Uses 1 stage. Uses a 1st order local linear interpolation for
    dense/ts output.
    """

    term_structure: ClassVar = AbstractTermDAE
    # We actually have enough information to use 3rd order Hermite interpolation.
    #
    # We don't use it as this seems to be quite a bad choice for low-order solvers: it
    # produces very oscillatory interpolations.
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolationDAE]
    ] = LocalLinearInterpolationDAE

    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(VeryChord)()
    root_find_max_steps: int = 10

    def order(self, terms):
        return 1

    def error_order(self, terms):
        if is_sde(terms):
            return None
        else:
            return 2

    def init(
        self,
        terms: AbstractTermDAE,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        z0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTermDAE,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        z0 : Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        del made_jump
        control = terms.contr(t0, t1)
        # Could use FSAL here but that would mean we'd need to switch to working with
        # `f0 = terms.vf(t0, y0, args)`, and that gets quite hairy quite quickly.
        # (C.f. `AbstractRungeKutta.step`.)
        # If we wanted FSAL then really the correct thing to do would just be to
        # write out a `ButcherTableau` and use `AbstractSDIRK`.
        k0 = terms.vf_prod(t0, y0, z0, args, control)
        k1, k2 = k0
        args = (terms.vf_prod, t1, y0, z0, args, control)
        nonlinear_sol = optx.root_find(_implicit_relation, self.root_finder, k0, args, throw=False, max_steps=self.root_find_max_steps)
        # jax.debug.print("k0:{}", k0)
        # jax.debug.print("value: {}", nonlinear_sol.value)
        # jax.debug.print("stats:{}", nonlinear_sol.stats)
        # jax.debug.print("result:{}", nonlinear_sol.result)
        # jax.debug.print("state:{}", nonlinear_sol.state)
        c0 = nonlinear_sol.value
        c1, c2 = c0
        y1 = (y0**ω + c1**ω).ω
        z1 = (z0**ω + c2**ω).ω
        # Use the trapezoidal rule for adaptive step sizing.
        y_error = (0.5 * (c1**ω - k1**ω)).ω
        z_error = (0.5 * (c2**ω - k2**ω)).ω
        dense_info = dict(y0=y0, y1=y1, z0=z0, z1=z1)
        solver_state = None
        result = RESULTS.promote(nonlinear_sol.result)
        return y1, y_error, z1, z_error, dense_info, solver_state, result

    def func(
        self,
        terms: AbstractTermDAE,
        t0: RealScalarLike,
        y0: Y,
        z0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, z0, args)