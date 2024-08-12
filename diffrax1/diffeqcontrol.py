import jax.numpy as jnp
import diffrax as dfx
from diffrax1.diffrax1._integrate import diffeqsolve
from diffrax1.diffrax1._term import ODETerm
from diffrax1.diffrax1._solver.tsit5 import Tsit5
from diffrax1.diffrax1._solver.implicit_euler import ImplicitEuler
from diffrax1.diffrax1._saveat import SaveAt
from diffrax1.diffrax1._step_size_controller.adaptive import PIDController

def vector_field(t, y, args):
    x1, x2, x3 = y
    d_x1 = -(1/2) * x2
    d_x2 = (1/2) * x1 - (1/4) * x2
    d_x3 = (1/4) * x2 - (1/6) * x3
    d_y = jnp.array([d_x1, d_x2, d_x3])
    return d_y

terms = ODETerm(vector_field)
y0 = jnp.array([0, 0, 0])
solver = ImplicitEuler()
saveat = SaveAt(ts=[0, 1, 2, 3])
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

sol = diffeqsolve(terms, solver, t0=0, t1=3, dt0=0.1, y0=y0, saveat=saveat,
                  stepsize_controller=stepsize_controller)

print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
print(sol.ys) 