from collections.abc import Callable
from typing import ClassVar
from typing_extensions import TypeAlias

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


_SolverState: TypeAlias = None


def _implicit_relation(f, nonlinear_solve_args):
    f1, f2 = f
    vf_prod, t1, y0, z0, args, control = nonlinear_solve_args
    diff = (vf_prod(t1, (y0**ω + f1**ω).ω, (z0**ω).ω, args, control) ** ω - f**ω).ω
    return diff


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

    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(optx.Newton)()
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
        c0 = nonlinear_sol.value
        c1, c2 = c0
        y1 = (y0**ω + k1**ω).ω
        z1 = (z0**ω + k2**ω).ω
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