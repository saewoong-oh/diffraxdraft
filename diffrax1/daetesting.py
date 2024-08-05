import jax.numpy as jnp
import diffrax as dfx
from diffrax1.diffrax1._integrate import daesolve
from ._term import DAETerm
from diffrax1.diffrax1._solver.implicit_euler_dae import Implicit_Euler_DAE
from diffrax1.diffrax1._saveat import SaveAt
from diffrax1.diffrax1._step_size_controller.adaptive import PIDController

def vector_field(t, y, z, args):
    x1, x2, x3, x4 = y
    x5 = z
    g = x1**2 + x2**2 + 1
    d_x1 = x3
    d_x2 = x4
    d_x3 = -x5*x1
    d_x4 = -x5*x2 - g
    d_y = jnp.array([d_x1, d_x2, d_x3, d_x4])
    g = jnp.array([g])
    return jnp.array([g, d_y])

terms = DAETerm(vector_field)
y0 = jnp.array([0, 0, 0, 0])
z0 = jnp.array([0])
solver = Implicit_Euler_DAE()
saveat = SaveAt(ts=[0, 1, 2, 3])
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

sol = daesolve(terms, solver, t0=0, t1=3, dt0=0.1, y0=y0, z0=z0, saveat=saveat,
                  stepsize_controller=stepsize_controller)

print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
print(sol.ys) 