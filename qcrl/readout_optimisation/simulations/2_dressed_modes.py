import warnings

warnings.filterwarnings("ignore")

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

from utils import get_params, complex_plotter, timer_func

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
SQRT_RATIO = jnp.sqrt(2 * params["ratio"])
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
RES_DRIVE_FREQ = WC.real - 0.5 * CHI - KERR + 0.125 * SQRT_RATIO * CHI
TRANS_DRIVE_FREQ = (
    WQ.real - 0.5 * CHI - ANHARM + 0.25 * SQRT_RATIO * (CHI + KERR + ANHARM)
)
COS_ANGLE = jnp.cos(0.5 * jnp.arctan(2 * G / DELTA))
SIN_ANGLE = jnp.sin(0.5 * jnp.arctan(2 * G / DELTA))

WC_EFF = WC - 0.5 * CHI - KERR + 0.125 * SQRT_RATIO * CHI
WQ_EFF = WQ - 0.5 * CHI - ANHARM + 0.25 * SQRT_RATIO * (CHI + KERR + ANHARM)

print(f"bare freqs, wa: {WA / (2 * jnp.pi)}, wb: {WB / (2 * jnp.pi)}")
print(f"exact dressed freqs, wc: {WC / (2 * jnp.pi)}, wq: {WQ / (2 * jnp.pi)}")
print(
    f"rough dressed freqs, wc: {(WA + G**2/DELTA - 1j * KAPPA / 2) / (2 * jnp.pi)}, wq: {(WB - G**2/DELTA - 1j * GAMMA / 2) / (2 * jnp.pi)}"
)
print(f"rough cos: {1.}, sin: {G / DELTA}, exact cos: {COS_ANGLE}, sin: {SIN_ANGLE}")

print(f"wc eff: {WC_EFF}, wq eff: {WQ_EFF}")

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
        SQRT_RATIO,
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
y0_e = jnp.array([0.0 + 1j * 0.0, 1.0 + 1j * 0.0], dtype=complex_dtype)

solver = Tsit5()
saveat = SaveAt(ts=ts)
stepsize_controller = PIDController(rtol=1e-8, atol=1e-8, jump_ts=ts)
max_steps = int(1e6)

# defining drive
res_drive = jnp.zeros_like(ts, dtype=complex_dtype) + 5.0 * 2 * jnp.pi  # in MHz
trans_drive = jnp.zeros_like(ts, dtype=complex_dtype) + 0.0 * 2 * jnp.pi
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
        sqrt_ratio,
    ) = args
    drive_res, drive_trans = control.evaluate(t)

    c_squared = jnp.absolute(c) ** 2
    q_squared = jnp.absolute(q) ** 2

    d_c = (
        -1j
        * (
            wc
            - 0.5 * chi
            - kerr
            + 0.125 * sqrt_ratio * chi
            + c_squared
            * (-kerr + 0.5 * sqrt_ratio * kerr + sqrt_ratio * kerr * q_squared)
            + q_squared
            * (
                -chi
                + sqrt_ratio * kerr
                + 0.5 * sqrt_ratio * chi
                + 0.25 * sqrt_ratio * chi * q_squared
            )
        )
        * c
        - 1j * cos_angle * drive_res * jnp.exp(-1j * res_drive_freq * t)
        - 1j * sin_angle * drive_trans * jnp.exp(-1j * trans_drive_freq * t)
    )
    d_q = (
        -1j
        * (
            wq
            - 0.5 * chi
            - anharm
            + 0.25 * sqrt_ratio * (chi + kerr + anharm)
            + c_squared
            * (
                sqrt_ratio * kerr
                + 0.5 * sqrt_ratio * kerr * c_squared
                + 0.5 * sqrt_ratio * chi * q_squared
                + 0.5 * sqrt_ratio * chi
                - chi
            )
            + q_squared
            * (
                0.5 * sqrt_ratio * anharm
                + sqrt_ratio / 6 * anharm * q_squared
                + 0.25 * sqrt_ratio * chi
                - anharm
            )
        )
        * q
        - 1j * cos_angle * drive_trans * jnp.exp(-1j * trans_drive_freq * t)
        + 1j * sin_angle * drive_res * jnp.exp(-1j * res_drive_freq * t)
    )
    return jnp.array([d_c, d_q], dtype=complex_dtype)


def rot_vector_field(t, y, args):
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
        sqrt_ratio,
    ) = args
    drive_res, drive_trans = control.evaluate(t)

    c_squared = jnp.absolute(c) ** 2
    q_squared = jnp.absolute(q) ** 2

    d_c = (
        -1j
        * (
            wc
            - 0.5 * chi
            - kerr
            + 0.125 * sqrt_ratio * chi
            + c_squared
            * (-kerr + 0.5 * sqrt_ratio * kerr + sqrt_ratio * kerr * q_squared)
            + q_squared
            * (
                -chi
                + sqrt_ratio * kerr
                + 0.5 * sqrt_ratio * chi
                + 0.25 * sqrt_ratio * chi * q_squared
            )
        )
        * c
        - 1j * cos_angle * drive_res * jnp.exp(-1j * (res_drive_freq - wc) * t)
        - 1j * sin_angle * drive_trans * jnp.exp(-1j * trans_drive_freq * t)
    )
    d_q = (
        -1j
        * (
            wq
            - 0.5 * chi
            - anharm
            + 0.25 * sqrt_ratio * (chi + kerr + anharm)
            + c_squared
            * (
                sqrt_ratio * kerr
                + 0.5 * sqrt_ratio * kerr * c_squared
                + 0.5 * sqrt_ratio * chi * q_squared
                + 0.5 * sqrt_ratio * chi
                - chi
            )
            + q_squared
            * (
                0.5 * sqrt_ratio * anharm
                + sqrt_ratio / 6 * anharm * q_squared
                + 0.25 * sqrt_ratio * chi
                - anharm
            )
        )
        * q
        - 1j * cos_angle * drive_trans * jnp.exp(-1j * trans_drive_freq * t)
        + 1j * sin_angle * drive_res * jnp.exp(-1j * res_drive_freq * t)
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

sol_e = diffeqsolve(
    terms=ode_term,
    solver=solver,
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=y0_e,
    args=args,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    max_steps=max_steps,
)


def time_func():
    res_g = diffeqsolve(
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

    res_e = diffeqsolve(
        terms=ode_term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0_e,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    return res_g, res_e


jitted_func = jit(time_func)
res_ge = timer_func(
    jitted_func,
    num_reps=10,
    name="time taken for g + e",
    func_args=None,
    block=True,
    trials_in_each=1,
    units=1e-6,
)

res_g, res_e = res_ge

rot_res_freq = RES_DRIVE_FREQ
rot_trans_freq = TRANS_DRIVE_FREQ

plot_res_g = res_g.ys[:, 0]  # sol.ys[:, 0]
plot_res_e = res_e.ys[:, 0]  # sol_e.ys[:, 0]
plot_trans_g = res_g.ys[:, 1]  # sol.ys[:, 1]
plot_trans_e = res_e.ys[:, 1]  # sol_e.ys[:, 1]

rot_res = plot_res_g * jnp.exp(1j * rot_res_freq * sol.ts)
rot_trans = plot_trans_g * jnp.exp(1j * rot_trans_freq * sol.ts)

rot_res_e = plot_res_e * jnp.exp(1j * rot_res_freq * sol_e.ts)
rot_trans_e = plot_trans_e * jnp.exp(1j * rot_trans_freq * sol_e.ts)

fig1, ax1 = complex_plotter(
    ts=sol.ts,
    complex_1=plot_res_g,
    complex_2=plot_trans_g,
    rot_1=rot_res,
    rot_2=rot_trans,
    name_1="res",
    name_2="trans",
    fig_name="Ground Dressed",
)

fig2, ax2 = complex_plotter(
    ts=sol.ts,
    complex_1=plot_res_e,
    complex_2=plot_trans_e,
    rot_1=rot_res_e,
    rot_2=rot_trans_e,
    name_1="res",
    name_2="trans",
    fig_name="Excited Dressed",
)


plt.show()
