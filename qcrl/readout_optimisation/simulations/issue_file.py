import time
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit, config, vmap, block_until_ready
from diffrax import (
    diffeqsolve,
    ODETerm,
    Tsit5,
    LinearInterpolation,
    SaveAt,
    PIDController,
)

from jaxtyping import Array

config.update("jax_enable_x64", True)
out_float_dtype = jnp.float64
out_complex_dtype = jnp.complex128

n_trials = 100

TS_ACTIONS = jnp.linspace(0.0, 1.0, 51, dtype=out_float_dtype)
TS_SIM = jnp.linspace(0.0, 1.0, 201, dtype=out_float_dtype)

RES_DRIVE_AMP = (
    2.0 * 2.0 * jnp.pi * jnp.exp(-((TS_ACTIONS - 0.5) ** 2) / (2 * 0.25**2))
)
TRANS_DRIVE_AMP = (
    0.0 * 2.0 * jnp.pi * jnp.exp(-((TS_ACTIONS - 0.5) ** 2) / (2 * 0.25**2))
)

RES_DRIVE_AMP_alt = (
    2.0 * 2.0 * jnp.pi * jnp.exp(-((TS_ACTIONS - 0.5) ** 2) / (2 * 0.25**2))
).reshape((51, 1))
TRANS_DRIVE_AMP_alt = (
    0.0 * 2.0 * jnp.pi * jnp.exp(-((TS_ACTIONS - 0.5) ** 2) / (2 * 0.25**2))
).reshape((51, 1))

b_res = RES_DRIVE_AMP.reshape((51, 1))
b_trans = TRANS_DRIVE_AMP.reshape((51, 1))

for i in range(n_trials - 1):
    b_res = jnp.concatenate((b_res, RES_DRIVE_AMP_alt), axis=1)
    b_trans = jnp.concatenate((b_trans, TRANS_DRIVE_AMP_alt), axis=1)


def single_eval(
    res_state: Array,
    res_drive: Array,
    trans_drive: Array,
    delta_res_freq: Array,
    delta_trans_freq: Array,
):
    float_dtype = jnp.float64
    complex_dtype = jnp.complex128
    state_g = jnp.array([*res_state, 0.0, 0.0], dtype=float_dtype)
    state_e = jnp.array([*res_state, 1.0, 0.0], dtype=float_dtype)
    ts_actions = jnp.linspace(0.0, 1.0, 51, dtype=float_dtype)
    ts_sim = jnp.linspace(0.0, 1.0, 201, dtype=float_dtype)

    dt0 = 1e-3
    solver = Tsit5()
    max_steps = int(16**3)
    stepsize_controller = PIDController(
        rtol=1e-3, atol=1e-6, pcoeff=0.4, icoeff=0.3, dcoeff=0.0, jump_ts=ts_actions
    )
    saveat = SaveAt(
        ts=ts_sim,
        fn=lambda t, y, args: jnp.array([y[0], y[1]], dtype=float_dtype),
    )

    drive_arr = jnp.vstack((res_drive, trans_drive), dtype=complex_dtype).T
    control = LinearInterpolation(ts=ts_actions, ys=drive_arr)

    args = jnp.array(
        [
            44400.38147835797 - 2.6677954118222127j,
            30070.71498952447 - 0.019959690218603887j,
            -0.013361622621663578,
            -9.792069634763914,
            0.0013124927820996545,
            0.05202123087967161,
            delta_res_freq[0],
            delta_trans_freq[0],
        ],
        dtype=complex_dtype,
    )

    def vector_field(t, y, args):
        cr, ci, qr, qi = y
        drive_res, drive_trans = control.evaluate(t)
        (
            wc_eff,
            wq_eff,
            k1,
            k2,
            a1,
            a4,
            delta_res_freq,
            delta_trans_freq,
        ) = args
        c = cr + 1j * ci
        q = qr + 1j * qi
        c_squared = cr**2 + ci**2
        q_squared = qr**2 + qi**2
        c_freq_term = drive_res * jnp.exp(-1j * (delta_res_freq) * t)
        q_freq_term = drive_trans * jnp.exp(-1j * (delta_trans_freq) * t)
        d_c = -1j * (
            (
                wc_eff.imag * 1j
                + c_squared * (k1 + 2 * a1 * q_squared)
                + q_squared * (k2)  # + a2 * q_squared)
            )
            * c
            + c_freq_term
            + a4 * q_freq_term
        )
        d_q = -1j * (
            (
                wq_eff.imag * 1j
                + c_squared * (k2 + a1 * c_squared)  # + 2 * a2 * q_squared)
                # + q_squared * (a3 * q_squared + k3)
            )
            * q
            + q_freq_term
            - a4 * c_freq_term
        )
        return jnp.array([d_c.real, d_c.imag, d_q.real, d_q.imag], dtype=float_dtype)

    ode_term = ODETerm(vector_field)

    sol_g = diffeqsolve(
        terms=ode_term,
        solver=solver,
        t0=0.0,
        t1=1.0,
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
        t0=0.0,
        t1=1.0,
        dt0=dt0,
        y0=state_e,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    return sol_g.ys, sol_e.ys, sol_g.stats, sol_e.stats


in_state = jnp.array([0.0, 0.0], dtype=out_float_dtype)
# in_drive_res = jnp.zeros_like(TS_ACTIONS, dtype=out_complex_dtype) + 10.0
in_drive_trans = jnp.zeros_like(TS_ACTIONS, dtype=out_complex_dtype) + 0.0
in_res_freq = jnp.array([0.0], dtype=out_float_dtype)
in_trans_freq = jnp.array([0.0], dtype=out_float_dtype)

jitted_single = jit(single_eval)

b_state = jnp.zeros((2, n_trials), dtype=out_float_dtype)
# b_drive_res = jnp.zeros((51, n_trials), dtype=out_complex_dtype) + 10.0
# b_drive_trans = jnp.zeros((51, n_trials), dtype=out_complex_dtype) + 0.0
b_res_freq = jnp.zeros((1, n_trials), dtype=out_float_dtype)
b_trans_freq = jnp.zeros((1, n_trials), dtype=out_float_dtype)

single_g, single_e, g_stats, e_stats = jitted_single(
    in_state, RES_DRIVE_AMP, TRANS_DRIVE_AMP, in_res_freq, in_trans_freq
)

start = time.time()
second_g, second_e, g_stats, e_stats = block_until_ready(
    jitted_single(in_state, RES_DRIVE_AMP, TRANS_DRIVE_AMP, in_res_freq, in_trans_freq)
)
now = time.time()

batched_eval = jit(vmap(single_eval, in_axes=1))
batched_, batched__, a_, b_ = batched_eval(
    b_state, b_res, b_trans, b_res_freq, b_trans_freq
)
print(f"time taken for single g + e: {(now - start)*1e6}us")
print(f"for g")
print(f"max_steps: {g_stats['max_steps']}")
print(f"num_accepted_steps: {g_stats['num_accepted_steps']}")
print(f"num_rejected_steps: {g_stats['num_rejected_steps']}")
print(f"num_steps: {g_stats['num_steps']}")
print(f"for e")
print(f"max_steps: {e_stats['max_steps']}")
print(f"num_accepted_steps: {e_stats['num_accepted_steps']}")
print(f"num_rejected_steps: {e_stats['num_rejected_steps']}")
print(f"num_steps: {e_stats['num_steps']}")

start = time.time()
batched_g, batched_e, batched_g_stats, batched_e_stats = block_until_ready(
    batched_eval(b_state, b_res, b_trans, b_res_freq, b_trans_freq)
)

now = time.time()
print(f"time taken for single g + e after batching: {(now - start)/n_trials*1e6}us")
print(f"for g")
print(f"max_steps: {batched_g_stats['max_steps'][0]}")
print(f"num_accepted_steps: {batched_g_stats['num_accepted_steps'][0]}")
print(f"num_rejected_steps: {batched_g_stats['num_rejected_steps'][0]}")
print(f"num_steps: {batched_g_stats['num_steps'][0]}")
print(f"for e")
print(f"max_steps: {batched_e_stats['max_steps'][0]}")
print(f"num_accepted_steps: {batched_e_stats['num_accepted_steps'][0]}")
print(f"num_rejected_steps: {batched_e_stats['num_rejected_steps'][0]}")
print(f"num_steps: {batched_e_stats['num_steps'][0]}")


def simple_plotter_debug(ts, complex_1, complex_2, name_1, name_2, fig_name):
    fig, ax = plt.subplots(4, 2, num=fig_name)
    ax[0, 0].plot(ts, jnp.absolute(complex_1), label=f"{name_1} photons", color="red")
    ax[0, 0].legend()

    ax[1, 0].plot(
        complex_1.real, complex_1.imag, label=f"{name_1} phase", color="orange"
    )
    ax[1, 0].legend()

    ax[2, 0].plot(ts, complex_1.real, label=f"{name_1} real", color="blue")
    ax[2, 0].legend()

    ax[3, 0].plot(ts, complex_1.imag, label=f"{name_1} imag", color="green")
    ax[3, 0].legend()

    ax[0, 1].plot(ts, jnp.absolute(complex_2), label=f"{name_2} photons", color="red")
    ax[0, 1].legend()

    ax[1, 1].plot(
        complex_2.real, complex_2.imag, label=f"{name_2} phase", color="orange"
    )
    ax[1, 1].legend()

    ax[2, 1].plot(ts, complex_2.real, label=f"{name_2} real", color="blue")
    ax[2, 1].legend()

    ax[3, 1].plot(ts, complex_2.imag, label=f"{name_2} imag", color="green")
    ax[3, 1].legend()

    return fig, ax


batch_g_element = batched_g[0]
batch_e_element = batched_e[0]

res_g = batch_g_element[:, 0] + 1j * batch_g_element[:, 1]
res_e = batch_e_element[:, 0] + 1j * batch_e_element[:, 1]

fig1, ax1 = simple_plotter_debug(
    ts=TS_SIM, complex_1=res_g, complex_2=res_e, name_1="g", name_2="e", fig_name="Res"
)
"""
trans_g = batch_g_element[:, 2] + 1j * batch_g_element[:, 3]
trans_e = batch_e_element[:, 2] + 1j * batch_e_element[:, 3]

fig2, ax2 = simple_plotter_debug(
    ts=TS_SIM,
    complex_1=trans_g,
    complex_2=trans_e,
    name_1="g",
    name_2="e",
    fig_name="Trans",
)
"""
plt.show()
