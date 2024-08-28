import jax.numpy as jnp
import diffrax as dfx
from diffrax1.diffrax1._integrate import diffeqsolve
from diffrax1.diffrax1._term import ODETerm
from diffrax1.diffrax1._solver.tsit5 import Tsit5
from diffrax1.diffrax1._solver.implicit_euler import ImplicitEuler
from diffrax1.diffrax1._saveat import SaveAt
from diffrax1.diffrax1._step_size_controller.adaptive import PIDController
from diffrax1.diffrax1._adjoint import RecursiveCheckpointAdjointDAE, RecursiveCheckpointAdjoint
import matplotlib.pyplot as plt

def vector_field(t, y, args):
    x, y, v, w, lambd = y
    args = args
    d_x = v
    d_y = w
    d_v = -2*lambd*x
    d_w = -2*lambd*y - 9.8
    d_lambd = (-4*lambd*(x*v + y*w) - 1.5*9.8*w)/(x**2 + y**2)
    # d_lambd = x**2 + y**2 - 1
    d_y = jnp.array([d_x, d_y, d_v, d_w, d_lambd])
    return d_y

terms = ODETerm(vector_field)
t0 = 0
t1= 6
lambd_init = (1 + 9.8)/2
d_lambd_init = (-4*lambd_init*(0*0 + -1*0) - 1.5*9.8*0)/(0**2 + (-1)**2)
y0 = jnp.array([0, -1, 1, 0, lambd_init])
solver = ImplicitEuler()
saveat = SaveAt(ts=jnp.linspace(t0, t1, 1000))
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
max_steps = 4096
sol = diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=0.1, y0=y0, saveat=saveat,
                  stepsize_controller=stepsize_controller,
                  max_steps = max_steps)

print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
print(sol.ys)
plt.plot(sol.ts, sol.ys[:, 0], label="x")
plt.plot(sol.ts, sol.ys[:, 1], label="y")
plt.plot(sol.ts, sol.ys[:, 2], label="u")
plt.plot(sol.ts, sol.ys[:, 3], label="v")
plt.plot(sol.ts, sol.ys[:, 4], label="lambd")
plt.legend()
plt.show()