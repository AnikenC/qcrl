import timeit
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit, config
from diffrax import (
    diffeqsolve,
    ODETerm,
    Tsit5,
    LinearInterpolation,
    SaveAt,
    PIDController,
)

from utils import get_params

config.update("jax_enable_x64", True)

# defining consistent dtypes to be used
float_dtype = jnp.float64
complex_dtype = jnp.complex128

# physics args
params = get_params()
KAPPA = params["kappa"]
CHI = params["chi"]
GAMMA = params["gamma"]
G = params["g"]
KERR = params["kerr"]
WA = params["wr"]
WB = params["wq"]
DELTA = params["delta"]
WC = 0.5 * (
    WA
    + WB
    - 1j * KAPPA / 2
    - 1j * GAMMA / 2
    + jnp.sqrt(
        DELTA**2
        + 1j * DELTA * (GAMMA - KAPPA)
        - 0.25 * (KAPPA + GAMMA) ** 2
        + 4 * G**2
        - KAPPA * GAMMA
    )
)
WQ = 0.5 * (
    WA
    + WB
    - 1j * KAPPA / 2
    - 1j * GAMMA / 2
    - jnp.sqrt(
        DELTA**2
        + 1j * DELTA * (GAMMA - KAPPA)
        - 0.25 * (KAPPA + GAMMA) ** 2
        + 4 * G**2
        - KAPPA * GAMMA
    )
)
ANHARM = params["anharm"]
RES_DRIVE_FREQ = WC.real
TRANS_DRIVE_FREQ = WQ.real - 0.95 * ANHARM

print(f"bare freqs, wa: {WA / (2 * jnp.pi)}, wb: {WB / (2 * jnp.pi)}")
print(f"exact dressed freqs, wc: {WC / (2 * jnp.pi)}, wq: {WQ / (2 * jnp.pi)}")
print(
    f"rough dressed freqs, wc: {(WA + G**2/DELTA - 1j * KAPPA / 2) / (2 * jnp.pi)}, wq: {(WB - G**2/DELTA - 1j * GAMMA / 2) / (2 * jnp.pi)}"
)

args = jnp.array(
    [
        0.5 * KAPPA,
        0.5 * GAMMA,
        G,
        ANHARM,
        WA,
        WB,
        RES_DRIVE_FREQ,
        TRANS_DRIVE_FREQ,
    ],
    dtype=complex_dtype,
)

# sim params
t0 = 0.0
t1 = 1.0
dt0 = 1e-4
ns_per_sample = 4
samples = int(1000.0 / ns_per_sample * (t1 - t0))
ts = jnp.linspace(t0, t1, samples + 1, dtype=float_dtype)

y0 = jnp.array([0.0 + 1j * 0.0, 0.0 + 1j * 0.0], dtype=complex_dtype)

solver = Tsit5()
saveat = SaveAt(ts=ts)
stepsize_controller = PIDController(rtol=1e-8, atol=1e-8, jump_ts=ts)
max_steps = int(1e6)

# defining drive
res_drive = jnp.zeros_like(ts, dtype=complex_dtype) + 0.0 * 2 * jnp.pi  # in MHz
trans_drive = jnp.zeros_like(ts, dtype=complex_dtype) + 10.0 * 2 * jnp.pi
drive_arr = jnp.vstack((res_drive, trans_drive), dtype=complex_dtype).T
control = LinearInterpolation(ts=ts, ys=drive_arr)


def vector_field(t, y, args):
    a, b = y
    (
        kappa_half,
        gamma_half,
        g,
        anharm,
        wa,
        wb,
        res_drive_freq,
        trans_drive_freq,
    ) = args
    drive_res, drive_trans = control.evaluate(t)

    d_a = (
        -kappa_half * a
        - 1j * wa * a
        - 1j * g * b
        - 1j * drive_res * jnp.exp(-1j * res_drive_freq * t)
    )
    d_b = (
        -gamma_half * b
        - 1j * (wb - 0.95 * anharm) * b
        - 1j * g * a
        - 1j * drive_trans * jnp.exp(-1j * trans_drive_freq * t)
        + 1j * 0.9 * anharm * jnp.absolute(b) ** 2 * b
        - 1j * anharm / 30 * jnp.absolute(b) ** 4 * b
    )
    return jnp.array([d_a, d_b], dtype=complex_dtype)


ode_term = ODETerm(vector_field)

sol = diffeqsolve(
    terms=ode_term,
    solver=solver,
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=y0,
    args=args,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=max_steps,
)


def time_func():
    _ = diffeqsolve(
        terms=ode_term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )


jitted_func = jit(time_func)
jitted_func()  # run once to compile the function

print(f"time taken per sim: {timeit.Timer(jitted_func).timeit(number=1000)/1000}")

fig, ax = plt.subplots(4, 2)

rot_res_freq = RES_DRIVE_FREQ
rot_trans_freq = TRANS_DRIVE_FREQ

rot_res = sol.ys[:, 0] * jnp.exp(1j * rot_res_freq * sol.ts)
rot_trans = sol.ys[:, 1] * jnp.exp(1j * rot_trans_freq * sol.ts)

ax[0, 0].plot(sol.ts, jnp.absolute(sol.ys[:, 0]) ** 2, label="res phot", color="red")
ax[0, 0].legend()

ax[1, 0].plot(sol.ts, jnp.absolute(sol.ys[:, 1]) ** 2, label="trans phot", color="blue")
ax[1, 0].legend()

ax[2, 0].plot(sol.ys[:, 0].real, sol.ys[:, 0].imag, label="res phase", color="orange")
ax[2, 0].legend()

ax[3, 0].plot(sol.ys[:, 1].real, sol.ys[:, 1].imag, label="trans phase", color="green")
ax[3, 0].legend()

# plotting rot frame

ax[0, 1].plot(sol.ts, jnp.absolute(rot_res) ** 2, label="rot res phot", color="red")
ax[0, 1].legend()

ax[1, 1].plot(
    sol.ts, jnp.absolute(rot_trans) ** 2, label="rot trans phot", color="blue"
)
ax[1, 1].legend()

ax[2, 1].plot(rot_res.real, rot_res.imag, label="rot res phase", color="orange")
ax[2, 1].legend()

ax[3, 1].plot(rot_trans.real, rot_trans.imag, label="rot trans phase", color="green")
ax[3, 1].legend()

plt.show()
