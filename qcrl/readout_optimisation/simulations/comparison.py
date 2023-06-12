import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import time
import timeit

import jax
import jax.numpy as jnp
from jax import jit, vmap, config
from diffrax import (
    diffeqsolve,
    ODETerm,
    LinearInterpolation,
    Tsit5,
    Heun,
    Dopri5,
    SaveAt,
    PIDController,
)

from utils import complex_plotter, timer_func, simple_plotter, simple_plotter_debug

config.update("jax_enable_x64", True)
float_dtype = jnp.float64
complex_dtype = jnp.complex128

G = 102.9 * 2.0 * jnp.pi
WA = 7062.0 * 2.0 * jnp.pi
WB = 5092.0 * 2.0 * jnp.pi
DELTA = WA - WB
KAPPA = 5.35
CHI = 0.16 * 2 * KAPPA * 2 * jnp.pi
GAMMA = 1 / 39.2
KERR = 0.5 * CHI * G**2 / DELTA**2
ANHARM = 314.0 * 2 * jnp.pi
SQRT_RATIO = jnp.sqrt(2 / 62.5)

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

WC_EFF = WC - 0.5 * CHI - KERR + 0.125 * SQRT_RATIO * CHI
WC_EFF_REAL = WC_EFF.real
WC_EFF_IMAG = WC_EFF.imag
WQ_EFF = WQ - 0.5 * CHI - ANHARM + 0.25 * SQRT_RATIO * (CHI + KERR + ANHARM)
WQ_EFF_REAL = WQ_EFF.real
WQ_EFF_IMAG = WQ_EFF.imag

RES_DRIVE_FREQ = WC.real - 0.5 * CHI - KERR + 0.125 * SQRT_RATIO * CHI
TRANS_DRIVE_FREQ = (
    WQ.real - 0.5 * CHI - ANHARM + 0.25 * SQRT_RATIO * (CHI + KERR + ANHARM)
)
COS_ANGLE = jnp.cos(0.5 * jnp.arctan(2 * G / DELTA))
SIN_ANGLE = jnp.sin(0.5 * jnp.arctan(2 * G / DELTA))

TS = jnp.linspace(0.0, 1.0, 50 + 1, dtype=float_dtype)

print_args = False
if print_args:
    print(f"G: {G}")
    print(f"WC: {WC}")
    print(f"WQ: {WQ}")
    print(f"CHI: {CHI}")
    print(f"KERR: {KERR}")
    print(f"ANHARM: {ANHARM}")
    print(f"res_Drive_freq: {RES_DRIVE_FREQ}")
    print(f"trans_drive_freq: {TRANS_DRIVE_FREQ}")
    print(f"sqrt(2 * ratio): {SQRT_RATIO}")
    print(f"cos(g/delta): {COS_ANGLE}")
    print(f"sin(g/delta): {SIN_ANGLE}")
    print(f"WC_EFF: {WC_EFF}")
    print(f"WQ_EFF: {WQ_EFF}")

    print(f"a1: {-KERR + 0.5 * SQRT_RATIO * KERR}")
    print(f"a2: {-CHI + SQRT_RATIO * KERR + 0.5 * SQRT_RATIO * CHI}")
    print(f"b1: {SQRT_RATIO * KERR + 0.5 * SQRT_RATIO * CHI - CHI}")
    print(f"b2: {0.5 * SQRT_RATIO * ANHARM + 0.25 * SQRT_RATIO * CHI - ANHARM}")

    print(f"wc imag: {WC_EFF_IMAG}")
    print(f"wq imag: {WQ_EFF_IMAG}")

    print(f"sqrt_kerr_b2: {SQRT_RATIO * KERR / 2}")
    print(f"sqrt_kerr_b2: {SQRT_RATIO * CHI / 2}")
    print(f"sqrt_kerr_b2: {SQRT_RATIO * ANHARM / 2}")


def debug_single_eval(init_state, res_drive, trans_drive, res_freq, trans_freq):
    ts = jnp.linspace(0.0, 1.0, 50 + 1, dtype=jnp.float64)
    in_complex_dtype = jnp.complex128
    state_g = jnp.array([init_state[0], 0.0], dtype=in_complex_dtype)
    state_e = jnp.array([init_state[0], 1.0], dtype=in_complex_dtype)
    drive_arr = jnp.vstack((res_drive, trans_drive), dtype=in_complex_dtype).T
    control = LinearInterpolation(ts=ts, ys=drive_arr)

    def vector_field(t, y, args):
        c, q = y
        drive_res, drive_trans = control.evaluate(t)

        wc_eff = 44400.38147835797 - 2.6677954118222127j
        wq_eff = 30070.71498952447 - 0.019959690218603887j
        res_drive_freq, trans_drive_freq = args
        sin_angle = 0.05202123150229454

        sqrt_kerr_b2 = 0.001312
        sqrt_chi_b4 = 0.4811
        sqrt_anharm_b6 = 58.82

        c_squared = jnp.absolute(c) ** 2
        q_squared = jnp.absolute(q) ** 2
        a = drive_res * jnp.exp(-1j * (res_drive_freq - wc_eff.real) * t)
        b = drive_trans * jnp.exp(-1j * (trans_drive_freq - wq_eff.real) * t)

        d_c = (
            -1j
            * (
                wc_eff.imag * 1j
                + c_squared * (-0.013361622621663578 + 2 * sqrt_kerr_b2 * q_squared)
                + q_squared * (-9.792069634763914 + sqrt_chi_b4 * q_squared)
            )
            * c
            - 1j * a
            - 1j * sin_angle * b
        )
        d_q = (
            -1j
            * (
                wq_eff.imag * 1j
                + c_squared
                * (
                    -9.792069634763916
                    + sqrt_kerr_b2 * c_squared
                    + 2 * sqrt_chi_b4 * q_squared
                )
                + q_squared * (+sqrt_anharm_b6 * q_squared - 1795.9757810978654)
            )
            * q
            - 1j * b
            + 1j * sin_angle * a
        )
        return jnp.array([d_c, d_q], dtype=complex_dtype)

    ode_term = ODETerm(vector_field)
    sol_g = diffeqsolve(
        terms=ode_term,
        solver=Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=1e-5,
        y0=state_g,
        args=jnp.array([res_freq[0], trans_freq[0]], dtype=in_complex_dtype),
        saveat=SaveAt(ts=ts, fn=lambda t, y, args: y[0]),
        stepsize_controller=PIDController(rtol=1e-7, atol=1e-9, jump_ts=ts),
        max_steps=int(1e5),
        throw=False,
    )
    sol_e = diffeqsolve(
        terms=ode_term,
        solver=Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=1e-5,
        y0=state_e,
        args=jnp.array([res_freq[0], trans_freq[0]], dtype=in_complex_dtype),
        saveat=SaveAt(ts=ts, fn=lambda t, y, args: y[0]),
        stepsize_controller=PIDController(rtol=1e-7, atol=1e-9, jump_ts=ts),
        max_steps=int(1e5),
        throw=False,
    )
    jax.debug.print("sol_g steps: {sol.stats}", sol=sol_g)
    jax.debug.print("sol_e steps: {sol.stats}", sol=sol_e)
    jax.debug.print(
        "g status: {sol_g.result}, e status: {sol_e.result}", sol_g=sol_g, sol_e=sol_e
    )
    return sol_g.ys, sol_e.ys, sol_g.stats, sol_e.stats


def single_eval(init_state, res_drive, trans_drive, res_freq, trans_freq):
    ts = jnp.linspace(0.0, 1.0, 50 + 1, dtype=jnp.float64)
    in_complex_dtype = jnp.complex128
    state_g = jnp.array([init_state[0], 0.0], dtype=in_complex_dtype)
    state_e = jnp.array([init_state[0], 1.0], dtype=in_complex_dtype)
    drive_arr = jnp.vstack((res_drive, trans_drive), dtype=in_complex_dtype).T
    control = LinearInterpolation(ts=ts, ys=drive_arr)

    def vector_field(t, y, args):
        c, q = y
        drive_res, drive_trans = control.evaluate(t)

        wc_eff = 44400.38147835797 - 2.6677954118222127j
        wq_eff = 30070.71498952447 - 0.019959690218603887j
        res_drive_freq, trans_drive_freq = args
        sin_angle = 0.05202123150229454

        sqrt_kerr_b2 = 0.001312
        sqrt_chi_b4 = 0.4811
        sqrt_anharm_b6 = 58.82

        c_squared = jnp.absolute(c) ** 2
        q_squared = jnp.absolute(q) ** 2
        a = drive_res * jnp.exp(-1j * (res_drive_freq - wc_eff.real) * t)
        b = drive_trans * jnp.exp(-1j * (trans_drive_freq - wq_eff.real) * t)

        d_c = (
            -1j
            * (
                wc_eff.imag * 1j
                + c_squared * (-0.013361622621663578 + 2 * sqrt_kerr_b2 * q_squared)
                + q_squared * (-9.792069634763914 + sqrt_chi_b4 * q_squared)
            )
            * c
            - 1j * a
            - 1j * sin_angle * b
        )
        d_q = (
            -1j
            * (
                wq_eff.imag * 1j
                + c_squared
                * (
                    -9.792069634763916
                    + sqrt_kerr_b2 * c_squared
                    + 2 * sqrt_chi_b4 * q_squared
                )
                + q_squared * (+sqrt_anharm_b6 * q_squared - 1795.9757810978654)
            )
            * q
            - 1j * b
            + 1j * sin_angle * a
        )
        return jnp.array([d_c, d_q], dtype=complex_dtype)

    ode_term = ODETerm(vector_field)
    sol_g = diffeqsolve(
        terms=ode_term,
        solver=Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=1e-5,
        y0=state_g,
        args=jnp.array([res_freq[0], trans_freq[0]], dtype=in_complex_dtype),
        saveat=SaveAt(ts=ts, fn=lambda t, y, args: y[0]),
        stepsize_controller=PIDController(rtol=1e-7, atol=1e-9, jump_ts=ts),
        max_steps=int(1e5),
        throw=False,
    )
    sol_e = diffeqsolve(
        terms=ode_term,
        solver=Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=1e-5,
        y0=state_e,
        args=jnp.array([res_freq[0], trans_freq[0]], dtype=in_complex_dtype),
        saveat=SaveAt(ts=ts, fn=lambda t, y, args: y[0]),
        stepsize_controller=PIDController(rtol=1e-7, atol=1e-9, jump_ts=ts),
        max_steps=int(1e5),
        throw=False,
    )
    return sol_g.ys, sol_e.ys, sol_g.stats, sol_e.stats


in_state = jnp.array([0.0, 0.0], dtype=complex_dtype)
in_drive_res = jnp.zeros(51, dtype=complex_dtype) + 2.0 * 2 * jnp.pi
in_drive_trans = jnp.zeros(51, dtype=complex_dtype) + 0.0
in_res_freq = [44400.38147835797]
in_trans_freq = [30070.71498952447]

n_times = 10
n_reps = 2

batched_state = jnp.zeros((2, n_times), dtype=complex_dtype)
batched_drive_res = jnp.zeros((51, n_times), dtype=complex_dtype) + 2.0 * 2 * jnp.pi
batched_drive_trans = jnp.zeros((51, n_times), dtype=complex_dtype) + 0.0
batched_res_freq = jnp.zeros((1, n_times), dtype=float_dtype) + 44400.3828125
batched_trans_freq = jnp.zeros((1, n_times), dtype=float_dtype) + 30070.71484375

jitted_debug = jit(debug_single_eval)
g, e, _, _ = jitted_debug(
    in_state, in_drive_res, in_drive_trans, in_res_freq, in_trans_freq
)
debug_res_g = g
debug_res_e = e

batched_eval = jit(vmap(single_eval, in_axes=1))

res = timer_func(
    batched_eval,
    num_reps=n_reps,
    name="Batched Evaluation",
    func_args=(
        batched_state,
        batched_drive_res,
        batched_drive_trans,
        batched_res_freq,
        batched_trans_freq,
    ),
    block=True,
    trials_in_each=n_times,
    units=1e-6,
)

res_vals_g = res[0][
    0
]  # 0th return (sol_g.ys), 0th index (the first element of the batch)
res_vals_e = res[1][
    0
]  # 1st return (sol_e.ys), 0th index (the first element of the batch)

print(f"status g: {res[2]}")
print(f"status e: {res[3]}")

fig, ax = simple_plotter_debug(
    ts=TS,
    complex_1=res_vals_g,
    complex_2=res_vals_e,
    name_1="ground state",
    name_2="excited state",
    fig_name="Dressed Response",
)
fig2, ax2 = simple_plotter_debug(
    ts=TS,
    complex_1=debug_res_g,
    complex_2=debug_res_e,
    name_1="ground state",
    name_2="excited state",
    fig_name="Debug Response",
)

plt.show()
