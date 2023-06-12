import time
import matplotlib.pyplot as plt

import jax
from jaxtyping import Array
import jax.numpy as jnp
from jax import jit, config, vmap
from diffrax import (
    diffeqsolve,
    ODETerm,
    Tsit5,
    LinearInterpolation,
    SaveAt,
    PIDController,
)

from updated_utils import get_params, simple_plotter_debug

config.update("jax_enable_x64", True)
float_dtype = jnp.float64
complex_dtype = jnp.complex128

physics_params = get_params(print_params=True)

T0 = 0.0
T1 = 2.0
NUM_ACTIONS = 50
NUM_SIM_SAMPLES = 2000
BATCH_SIZE = 2

TS_SIM = jnp.linspace(T0, T1, NUM_SIM_SAMPLES + 1, dtype=float_dtype)
TS_ACTION = jnp.linspace(T0, T1, NUM_ACTIONS + 1, dtype=complex_dtype)

WC_EFF = physics_params["wc_eff"]
WQ_EFF = physics_params["wq_eff"]
KERR_EFF = physics_params["kerr_eff"]
CHI_EFF = physics_params["chi_eff"]
SIN_ANGLE = physics_params["sin_angle"]
DELTA_EFF = physics_params["delta_eff"]

RES_DRIVE_AMP = 15.0 * 2.0 * jnp.pi
TRANS_DRIVE_AMP = 0.0

RES_COMPLEX_STATE = 0

drive_on_res_gaussian = jnp.asarray(
    (RES_DRIVE_AMP * jnp.exp(-((TS_ACTION - 0.5) ** 2) / (2 * 0.25**2))).reshape(
        NUM_ACTIONS + 1, 1
    ),
    dtype=complex_dtype,
)

drive_on_res_rectangular = (
    jnp.zeros_like(TS_ACTION, dtype=complex_dtype) + RES_DRIVE_AMP
).reshape(NUM_ACTIONS + 1, 1)

drive_on_trans_gaussian = jnp.asarray(
    (TRANS_DRIVE_AMP * jnp.exp(-((TS_ACTION - 0.5) ** 2) / (2 * 0.25**2))).reshape(
        NUM_ACTIONS + 1, 1
    ),
    dtype=complex_dtype,
)

drive_on_trans_rectangular = (
    jnp.zeros_like(TS_ACTION, dtype=complex_dtype) + TRANS_DRIVE_AMP
).reshape(NUM_ACTIONS + 1, 1)

batched_drive_res = drive_on_res_rectangular
batched_drive_trans = drive_on_trans_rectangular

for i in range(BATCH_SIZE - 1):
    batched_drive_res = jnp.concatenate(
        (batched_drive_res, drive_on_res_rectangular), axis=1
    )
    batched_drive_trans = jnp.concatenate(
        (batched_drive_trans, drive_on_trans_rectangular), axis=1
    )


def single_eval(
    res_state: Array,
    res_complex_drive: Array,
    trans_complex_drive: Array,
    delta_1r: Array,
    delta_2q: Array,
):
    in_float_dtype = jnp.float64
    in_complex_dtype = jnp.complex128

    t0 = T0
    t1 = T1
    dt0 = 1e-4
    num_actions = NUM_ACTIONS
    num_sim_samples = NUM_SIM_SAMPLES
    ts_action = jnp.linspace(t0, t1, num_actions + 1, dtype=in_float_dtype)
    ts_sim = jnp.linspace(t0, t1, num_sim_samples + 1, dtype=in_float_dtype)
    solver = Tsit5()
    saveat = SaveAt(ts=ts_sim, fn=lambda t, y, args: jnp.array([y[0], y[1]]))
    stepsize_controller = PIDController(
        rtol=1e-4, atol=1e-7, jump_ts=ts_action, pcoeff=0.4, icoeff=0.3, dcoeff=0.0
    )
    max_steps = int(16**4)

    state_g = jnp.array([res_state[0], 0.0], dtype=in_complex_dtype)
    state_e = jnp.array([res_state[0], 1.0], dtype=in_complex_dtype)

    args = jnp.array(
        [
            WC_EFF,
            WQ_EFF,
            KERR_EFF,
            CHI_EFF,
            SIN_ANGLE,
            delta_1r[0],
            delta_2q[0],
            DELTA_EFF,
        ],
        dtype=in_complex_dtype,
    )

    drive_arr = jnp.vstack(
        (res_complex_drive, trans_complex_drive), dtype=in_complex_dtype
    ).T
    control = LinearInterpolation(ts=ts_action, ys=drive_arr)

    def vector_field(t, y, args):
        c, q = y
        drive_res, drive_trans = control.evaluate(t)
        (
            wc_eff,
            wq_eff,
            kerr_eff,
            chi_eff,
            sin_angle,
            delta_1r,
            delta_2q,
            delta_eff,
        ) = args

        c_squared = jnp.absolute(c) ** 2
        q_squared = jnp.absolute(q) ** 2

        drive_res_freq = drive_res * jnp.exp(-1j * (delta_1r) * t)
        drive_res_freq_on_trans = drive_res * jnp.exp(-1j * (delta_1r + delta_eff) * t)
        drive_trans_freq = drive_trans * jnp.exp(-1j * delta_2q * t)
        drive_trans_freq_on_res = drive_trans * jnp.exp(
            -1j * (delta_2q - delta_eff) * t
        )

        d_c = -1j * (
            (
                wc_eff.imag * 1j
                + 0.5 * chi_eff
                - chi_eff * q_squared
                - kerr_eff * c_squared
            )
            * c
            + drive_res_freq
            + sin_angle * drive_trans_freq_on_res
        )
        d_q = -1j * (
            (wq_eff.imag * 1j - chi_eff * c_squared) * q
            + drive_trans_freq
            - sin_angle * drive_res_freq_on_trans
        )
        return jnp.array([d_c, d_q], dtype=in_complex_dtype)

    ode_term = ODETerm(vector_field)
    sol_g = diffeqsolve(
        terms=ode_term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=state_g,
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
        y0=state_e,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )

    return sol_g.ys, sol_e.ys, sol_g.stats, sol_e.stats


batched_eval = jit(vmap(single_eval, in_axes=1))

batched_res_state = jnp.zeros((1, BATCH_SIZE), dtype=complex_dtype) + RES_COMPLEX_STATE
batched_res_freq = jnp.zeros((1, BATCH_SIZE), dtype=float_dtype)
batched_trans_freq = jnp.zeros((1, BATCH_SIZE), dtype=float_dtype)

a_, b_, c_, d_ = batched_eval(
    batched_res_state,
    batched_drive_res,
    batched_drive_trans,
    batched_res_freq,
    batched_trans_freq,
)

start = time.time()

results_g, results_e, stats_g, stats_e = jax.block_until_ready(
    batched_eval(
        batched_res_state,
        batched_drive_res,
        batched_drive_trans,
        batched_res_freq,
        batched_trans_freq,
    )
)

now = time.time()
print(f"time taken to simulate g + e response: {(now - start) / BATCH_SIZE *1e6}us")

batch_element = 0

res_g = results_g[batch_element, :, 0]
trans_g = results_g[batch_element, :, 1]

res_e = results_e[batch_element, :, 0]
trans_e = results_e[batch_element, :, 1]

separation = jnp.absolute(res_g - res_e)
max_separation = jnp.max(jnp.absolute(res_g - res_e))
index = jnp.where(separation == max_separation)
print(f"max separation: {max_separation}")
print(f"time of max separation: {TS_SIM[index]}")

fig1, ax1 = simple_plotter_debug(
    ts=TS_SIM,
    complex_1=res_g,
    complex_2=trans_g,
    name_1="res",
    name_2="trans",
    fig_name=f"Ground, Max Separation: {max_separation}",
)

fig2, ax2 = simple_plotter_debug(
    ts=TS_SIM,
    complex_1=res_e,
    complex_2=trans_e,
    name_1="res",
    name_2="trans",
    fig_name=f"Excited, Max Separation: {max_separation}",
)

fig, ax = plt.subplots(1, num="Separation A")
ax.plot(TS_SIM, separation, label="separation", color="red")
ax.legend()

plt.show()

"""
delta_r = wd1 - wc
delta_q = wd2 - wq

c_dot = in terms of wc
q_dot = in terms of wq

c_dot in res = wd1 - wc, wd2 - wc
q_dot in trans = wd2 - wq, wd1 - wq

let delta_res_freq = wd1 - (wc - chi / 2)
let delta_trans_freq = wd2 - wq

then the trans_freq in res_dot = wd2 - (wc - chi / 2)
= delta_trans_freq - delta + chi / 2)

Actually,
Let us define Delta_1r = omega_1 - omega_cr + 0.5*chi_eff
Delta_2q = omega_2 - omega_qr
Then the drive terms in the Rotating Langevins are effectively
A * exp(-Delta_1r) + sin_angle * B * exp-(omega_2 - omega_cr + 0.5*chi_eff)
= A * exp(-Delta_1r) + sin_angle * B * exp-(omega_2 - omega_q + omega_q - omega_cr + 0.5*chi_eff)
= A * exp(-Delta_1r) + sin_angle * B * exp(-(Delta_2q - Delta_eff))
where Delta_eff = omega_cr - 0.5*chi_eff - omega_q

The transmon Langevin evaluates to
B * exp(-Delta_2q) - sin_angle * A * exp(-(omega_1 - omega_q))
B * exp(-Delta_2q) - sin_angle * A * exp(-(omega_1 - omega_cr + 0.5 * chi_eff + omega_cr - 0.5 * chi_eff - omega_q))
B * exp(-Delta_2q) - sin_angle * A * exp(-(Delta_1r + Delta_eff))

(bd * b) = exp(-gamma * t)
1/(bd * b) = exp(gamma * t)
ln(1 / (bd * b)) = gamma * t
ln(1 / (bd * b)) / 2 = gamma = 0.0315
2 / ln(1 / (bd * b)) = T1

0.933 with drive and two pi => 
0.989 with drive and no two pi

2pi/gamma = T1
or 1/gamma = T1

"""
