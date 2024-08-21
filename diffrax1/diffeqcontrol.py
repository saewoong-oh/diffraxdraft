import jax.numpy as jnp
import diffrax as dfx
from diffrax1.diffrax1._integrate import diffeqsolve
from diffrax1.diffrax1._term import ODETerm
from diffrax1.diffrax1._solver.tsit5 import Tsit5
from diffrax1.diffrax1._solver.implicit_euler import ImplicitEuler
from diffrax1.diffrax1._saveat import SaveAt
from diffrax1.diffrax1._step_size_controller.adaptive import PIDController
from diffrax1.diffrax1._adjoint import RecursiveCheckpointAdjointDAE, RecursiveCheckpointAdjoint

def vector_field(t, y, args):
    x, y, u, v, lambd = y
    lambd, = -3*9.8
    args = args
    g = x**2 + y**2 - 1
    d_x = u
    d_y = v
    d_u = -1*lambd*x
    d_v = -1*lambd*y - 9.8
    d_state = jnp.array([d_x, d_y, d_u, d_v])
    return d_state, g
    return d_y

terms = ODETerm(vector_field)
y0 = jnp.array([0, 0, 0])
solver = ImplicitEuler()
saveat = SaveAt(ts=[0, 1, 2, 3])
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
breakpoint()
sol = diffeqsolve(terms, solver, t0=0, t1=3, dt0=0.1, y0=y0, saveat=saveat,
                  stepsize_controller=stepsize_controller, adjoint = RecursiveCheckpointAdjoint(checkpoints = None))

print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
print(sol.ys) 