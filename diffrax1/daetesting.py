import sys
import os
sys.path.append('/root/diffrax1/diffrax1')
import jax
import logging
logging.getLogger("jax").setLevel(logging.DEBUG)
jax.debug.print("potato")
import diffrax1 as dfx
import matplotlib.pyplot as plt

import jax.numpy as jnp
from diffrax1.diffrax1._integrate import daesolve
from ._term import DAETerm
from diffrax1.diffrax1._solver.implicit_euler_dae import Implicit_Euler_DAE
from diffrax1.diffrax1._saveat import SaveAt, SaveAtDAE
from diffrax1.diffrax1._step_size_controller.adaptive import PIDController, PIDControllerDAE
from diffrax1.diffrax1._adjoint import RecursiveCheckpointAdjointDAE, RecursiveCheckpointAdjoint


# pendulum with length constraint DAE

jax.config.update("jax_debug_nans", True)

def test(t, state, z, args):
    x, y, v, w = state
    lam, = z
    args = args
    # g = x**2 + y**2 - 1
    g = v**2 + w**2 - 9.8*y - 2*lam*(x**2+y**2)
    # g = x * v + y * w
    d_x = v
    d_y = w
    d_v = -2*lam*x
    d_w = -2*lam*y - 9.8
    d_state = jnp.array([d_x, d_y, d_v, d_w])
    # d_state1 = jnp.array([x, y, v, w])
    return d_state, g

terms = DAETerm(test)
lambd_init = (1 + 9.8)/2
y0 = jnp.array([0, -1, 1, 0])
z0 = jnp.array([lambd_init])
t0 = 0
t1 = 6
solver = Implicit_Euler_DAE()
saveat = SaveAtDAE(ts=jnp.linspace(t0, t1, 1000))
stepsize_controller = PIDControllerDAE(rtol=1e-3, atol=1e-3)
max_steps = 4096 * 2

sol = daesolve(terms, solver, t0=t0, t1=t1, dt0=0.1, y0=y0, z0=z0, saveat=saveat,
                  stepsize_controller=stepsize_controller, adjoint = RecursiveCheckpointAdjointDAE(checkpoints = None), max_steps=max_steps)

# print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
# print(sol.ys) 
# print(sol.zs)
# for t,y,z in zip(sol.ts, sol.ys, sol.zs):
#     vals = dict(zip(["x","y","u","v"],y))
#     print(f"{t}] state={vals} lambda={z}")
plt.plot(sol.ts, sol.ys[:, 0], label="x")
plt.plot(sol.ts, sol.ys[:, 1], label="y")
plt.plot(sol.ts, sol.ys[:, 2], label="u")
plt.plot(sol.ts, sol.ys[:, 3], label="v")
plt.plot(sol.ts, sol.zs, label="lambd")
plt.legend()
plt.show()