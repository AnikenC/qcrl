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

from utils import get_params, complex_plotter

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
RES_DRIVE_FREQ = WC.real - 0.9 * KERR - 0.475 * CHI
TRANS_DRIVE_FREQ = WQ.real - 0.45 * CHI - 0.95 * ANHARM + 0.05 * KERR
COS_ANGLE = jnp.cos(0.5 * jnp.arctan(2 * G / DELTA))
SIN_ANGLE = jnp.sin(0.5 * jnp.arctan(2 * G / DELTA))

print(f"bare freqs, wa: {WA / (2 * jnp.pi)}, wb: {WB / (2 * jnp.pi)}")
print(f"exact dressed freqs, wc: {WC / (2 * jnp.pi)}, wq: {WQ / (2 * jnp.pi)}")
print(
    f"rough dressed freqs, wc: {(WA + G**2/DELTA - 1j * KAPPA / 2) / (2 * jnp.pi)}, wq: {(WB - G**2/DELTA - 1j * GAMMA / 2) / (2 * jnp.pi)}"
)
print(f"rough cos: {1.}, sin: {G / DELTA}, exact cos: {COS_ANGLE}, sin: {SIN_ANGLE}")

og_sin = 0.052233502538071054
current_sin = 0.05202123087967161

diff_in_chi = ANHARM / 2 / jnp.pi * (og_sin**2 - current_sin**2)
diff_in_kerr = ANHARM / 2 / jnp.pi * (og_sin**4 - current_sin**4)

print(f"diff_in_chi in MHz: {diff_in_chi}")
print(f"diff_in_kerr in MHz: {diff_in_kerr}")

args = jnp.array(
    [
        CHI,
        KERR,
        ANHARM,
        WC,
        WQ,
        RES_DRIVE_FREQ,
        TRANS_DRIVE_FREQ,
        COS_ANGLE,
        SIN_ANGLE,
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
    c, q = y
    (
        chi,
        kerr,
        anharm,
        wc,
        wq,
        res_drive_freq,
        trans_drive_freq,
        cos_angle,
        sin_angle,
    ) = args
    drive_res, drive_trans = control.evaluate(t)

    c_squared = jnp.absolute(c) ** 2
    q_squared = jnp.absolute(q) ** 2
    c_fourth = jnp.absolute(c) ** 4
    q_fourth = jnp.absolute(q) ** 4
    d_c = (
        -1j * (wc - 0.9 * kerr - 0.475 * chi) * c
        - 1j * cos_angle * drive_res * jnp.exp(-1j * res_drive_freq * t)
        - 1j * sin_angle * drive_trans * jnp.exp(-1j * trans_drive_freq * t)
        - 1j * 0.2 * kerr * c_squared * q_squared * c
        + 1j * 0.9 * kerr * c_squared * c
        - 1j * 0.2 * kerr * q_squared * c
        + 1j * 0.9 * chi * q_squared * c
        - 1j * 0.05 * chi * q_fourth * c
    )
    d_q = (
        -1j * (wq - 0.45 * chi - 0.95 * anharm + 0.05 * kerr) * q
        - 1j * cos_angle * drive_trans * jnp.exp(-1j * trans_drive_freq * t)
        + 1j * sin_angle * drive_res * jnp.exp(-1j * res_drive_freq * t)
        - 1j * 0.2 * kerr * c_squared * q
        - 1j * 0.1 * kerr * c_fourth * q
        + 1j * 0.9 * anharm * q_squared * q
        - 1j * anharm / 30 * q_fourth * q
        - 1j * 0.1 * chi * c_squared * q_squared * q
        + 1j * 0.9 * chi * c_squared * q
        - 1j * 0.05 * chi * q_squared * q
    )
    return jnp.array([d_c, d_q], dtype=complex_dtype)


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


rot_res_freq = RES_DRIVE_FREQ
rot_trans_freq = TRANS_DRIVE_FREQ

rot_res = sol.ys[:, 0] * jnp.exp(1j * rot_res_freq * sol.ts)
rot_trans = sol.ys[:, 1] * jnp.exp(1j * rot_trans_freq * sol.ts)

og_res = COS_ANGLE * sol.ys[:, 0] - SIN_ANGLE * sol.ys[:, 1]
og_trans = SIN_ANGLE * sol.ys[:, 0] + COS_ANGLE * sol.ys[:, 1]

rot_og_res = og_res * jnp.exp(1j * rot_res_freq * sol.ts)
rot_og_trans = og_trans * jnp.exp(1j * rot_trans_freq * sol.ts)

fig1, ax1 = complex_plotter(
    ts=sol.ts,
    complex_1=sol.ys[:, 0],
    complex_2=sol.ys[:, 1],
    rot_1=rot_res,
    rot_2=rot_trans,
    name_1="res",
    name_2="trans",
    fig_name="Dressed Modes",
)

fig2, ax2 = complex_plotter(
    ts=sol.ts,
    complex_1=og_res,
    complex_2=og_trans,
    rot_1=rot_og_res,
    rot_2=rot_og_trans,
    name_1="res",
    name_2="trans",
    fig_name="Original Modes from Dressed",
)

plt.show()
