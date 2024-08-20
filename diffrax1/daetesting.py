import sys
import os
sys.path.append('/root/diffrax1/diffrax1')

import diffrax1 as dfx

import jax.numpy as jnp
from diffrax1.diffrax1._integrate import daesolve
from ._term import DAETerm
from diffrax1.diffrax1._solver.implicit_euler_dae import Implicit_Euler_DAE
from diffrax1.diffrax1._saveat import SaveAt, SaveAtDAE
from diffrax1.diffrax1._step_size_controller.adaptive import PIDController, PIDControllerDAE
from diffrax1.diffrax1._adjoint import RecursiveCheckpointAdjointDAE, RecursiveCheckpointAdjoint


# pendulum with length constraint DAE

def test(t, state, z, args):
    x, y, u, v = state
    lambd, = z
    args = args
    g = x**2 + y**2 - 1
    d_x = u
    d_y = v
    d_u = -1*lambd*x
    d_v = -1*lambd*y - 9.8
    d_state = jnp.array([d_x, d_y, d_u, d_v])
    # output_g = jnp.array([g])
    # output_z = z
    # output = jnp.append(d_y, g)
    return d_state, g

terms = DAETerm(test)
y0 = jnp.array([0, 1, 0, 0])
z0 = jnp.array([0])
solver = Implicit_Euler_DAE()
saveat = SaveAtDAE(ts=[0, 1, 2])
stepsize_controller = PIDControllerDAE(rtol=1e-5, atol=1e-5)

sol = daesolve(terms, solver, t0=0, t1=2, dt0=0.1, y0=y0, z0=z0, saveat=saveat,
                  stepsize_controller=stepsize_controller, adjoint = RecursiveCheckpointAdjointDAE(checkpoints = None))

print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
print(sol.ys) 
print(sol.zs)
for t,y,z in zip(sol.ts, sol.ys, sol.zs):
    vals = dict(zip(["x","y","u","v"],y))
    print(f"{t}] state={vals} lambda={z}")