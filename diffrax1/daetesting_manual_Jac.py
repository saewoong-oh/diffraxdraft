import sys
import os
sys.path.append('/root/diffrax1/diffrax1')
import matplotlib.pyplot as plt

import diffrax1 as dfx

import jax.numpy as jnp
from diffrax1.diffrax1._integrate import diffeqsolve
from ._term import ODETerm
from diffrax1.diffrax1._solver.implicit_euler_dae_diffeqsolve import Implicit_Euler_DAE_diffeqsolve
from diffrax1.diffrax1._saveat import SaveAt, SaveAtDAE
from diffrax1.diffrax1._step_size_controller.adaptive import PIDController, PIDControllerDAE
from diffrax1.diffrax1._adjoint import RecursiveCheckpointAdjointDAE, RecursiveCheckpointAdjoint


# pendulum with length constraint DAE

def test(t, y, args):
    x, y, u, v, lambd = y
    args = args
    d_state = jnp.array([x, y, u, v, lambd])
    return d_state

terms = ODETerm(test)
lambd_init = (1 + 9.8)/2
t0 = 0
t1 = 6
y0 = jnp.array([0, -1, 1, 0, lambd_init])
solver = Implicit_Euler_DAE_diffeqsolve()
saveat = SaveAt(ts=jnp.linspace(t0, t1, 1000))
stepsize_controller = PIDController(rtol=1e-3, atol=1e-3)
max_steps = 4096 * 2

sol = diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=0.1, y0=y0, saveat=saveat,
                  stepsize_controller=stepsize_controller, max_steps=max_steps)

plt.plot(sol.ts, sol.ys[:, 0], label="x")
plt.plot(sol.ts, sol.ys[:, 1], label="y")
plt.plot(sol.ts, sol.ys[:, 2], label="u")
plt.plot(sol.ts, sol.ys[:, 3], label="v")
plt.plot(sol.ts, sol.ys[:, 4], label="lambd")
plt.legend()
plt.show()