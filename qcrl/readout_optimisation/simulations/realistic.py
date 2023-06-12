import matplotlib.pyplot as plt
import timeit
import time

import jax.numpy as jnp
from jax import jit, vmap, config
from diffrax import (
    diffeqsolve,
    ODETerm,
    LinearInterpolation,
    Tsit5,
    EulerHeun,
    Dopri5,
    SaveAt,
    PIDController,
)

from utils import complex_plotter

config.update("jax_enable_x64", False)
float_dtype = jnp.float32
complex_dtype = jnp.complex64

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
RES_DRIVE_FREQ = WC.real - 0.5 * CHI - KERR + 0.125 * SQRT_RATIO * CHI
TRANS_DRIVE_FREQ = (
    WQ.real - 0.5 * CHI - ANHARM + 0.25 * SQRT_RATIO * (CHI + KERR + ANHARM)
)
COS_ANGLE = jnp.cos(0.5 * jnp.arctan(2 * G / DELTA))
SIN_ANGLE = jnp.sin(0.5 * jnp.arctan(2 * G / DELTA))

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

TS = jnp.linspace(0.0, 1.0, 50 + 1, dtype=jnp.float32)


def single_eval(init_state, res_drive, trans_drive, res_freq, trans_freq):
    ts = jnp.linspace(0.0, 1.0, 50 + 1, dtype=jnp.float32)
    drive_arr = jnp.vstack((res_drive, trans_drive), dtype=jnp.complex64).T
    control = LinearInterpolation(ts=ts, ys=drive_arr)

    def vector_field(t, y, args):
        c, q = y
        drive_res, drive_trans = control.evaluate(t)

        wc = 44405.53515625 - 2.667795181274414j
        wq = 31960.30078125 - 0.01995980739593506j
        chi = 10.756813245891452
        kerr = 0.014674115403763233
        anharm = 1972.92018645439
        res_drive_freq = res_freq[0]
        trans_drive_freq = trans_freq[0]
        sqrt_ratio = 0.17888544499874115
        cos_angle = 0.9986459612846375
        sin_angle = 0.05202123150229454

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
        return jnp.array([d_c, d_q], dtype=jnp.complex64)

    ode_term = ODETerm(vector_field)
    sol = diffeqsolve(
        terms=ode_term,
        solver=Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=1e-5,
        y0=init_state,
        saveat=SaveAt(ts=ts),
        # stepsize_controller=PIDController(rtol=1e-5, atol=1e-7, jump_ts=ts),
        max_steps=int(1e6),
    )
    return sol.ys


n_times = 10
n_reps = 5

in_state = jnp.array([0.0, 0.0], dtype=jnp.complex64)
in_drive_res = jnp.zeros(51, dtype=jnp.complex64) + 10.0
in_drive_trans = jnp.zeros(51, dtype=jnp.complex64) + 2.0
in_res_freq = [44400.3828125]
in_drive_freq = [30070.71484375]

batched_state = jnp.zeros((2, n_times), dtype=jnp.complex64)
batched_drive_res = jnp.zeros((51, n_times), dtype=jnp.complex64) + 10.0
batched_drive_trans = jnp.zeros((51, n_times), dtype=jnp.complex64) + 2.0
batched_res_freq = jnp.zeros((1, n_times), dtype=jnp.float32) + 44400.3828125
batched_trans_freq = jnp.zeros((1, n_times), dtype=jnp.float32) + 30070.71484375

jitted_single = jit(single_eval)
res = jitted_single(in_state, in_drive_res, in_drive_trans, in_res_freq, in_drive_freq)


def for_func():
    for i in range(n_times):
        res = jitted_single(
            batched_state[:, i],
            batched_drive_res[:, i],
            batched_drive_trans[:, i],
            batched_res_freq[:, i],
            batched_trans_freq[:, i],
        )


time_single_for = 0.0
for i in range(n_reps):
    start = time.time()
    for_func()
    end = time.time()
    time_single_for += (end - start) / n_reps
print(
    f"time taken for {n_times} evaluations averaged {n_reps} times: {time_single_for}"
)
print(f"average time per eval: {time_single_for / n_times}")

batched_eval = jit(vmap(single_eval, in_axes=1))
res = batched_eval(
    batched_state,
    batched_drive_res,
    batched_drive_trans,
    batched_res_freq,
    batched_trans_freq,
)


time_batched_for = 0.0
for i in range(n_reps):
    start = time.time()
    res = batched_eval(
        batched_state,
        batched_drive_res,
        batched_drive_trans,
        batched_res_freq,
        batched_trans_freq,
    ).block_until_ready()
    now = time.time()
    time_batched_for += (now - start) / n_reps
print(f"time taken for {n_times} in vmap averaged {n_reps} times: {time_batched_for}")
print(f"average time per eval: {time_batched_for / n_times}")

sol_1 = res[0]

rot_res_freq = RES_DRIVE_FREQ
rot_trans_freq = TRANS_DRIVE_FREQ

og_res = COS_ANGLE * sol_1[:, 0] - SIN_ANGLE * sol_1[:, 1]
og_trans = SIN_ANGLE * sol_1[:, 0] + COS_ANGLE * sol_1[:, 1]

rot_res = sol_1[:, 0] * jnp.exp(1j * rot_res_freq * TS)
rot_trans = sol_1[:, 1] * jnp.exp(1j * rot_trans_freq * TS)

og_rot_res = og_res * jnp.exp(1j * rot_res_freq * TS)
og_rot_trans = og_trans * jnp.exp(1j * rot_trans_freq * TS)

fig1, ax1 = complex_plotter(
    ts=TS,
    complex_1=sol_1[:, 0],
    complex_2=sol_1[:, 1],
    rot_1=rot_res,
    rot_2=rot_trans,
    name_1="res",
    name_2="trans",
    fig_name="Dressed",
)
fig2, ax2 = complex_plotter(
    ts=TS,
    complex_1=og_res,
    complex_2=og_trans,
    rot_1=og_rot_res,
    rot_2=og_rot_trans,
    name_1="res",
    name_2="trans",
    fig_name="Original",
)

plt.show()
