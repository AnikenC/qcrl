import warnings

warnings.filterwarnings("ignore")

import time
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import config, jit, vmap, block_until_ready
from diffrax import (
    diffeqsolve,
    Tsit5,
    ODETerm,
    SaveAt,
    PIDController,
    LinearInterpolation,
)

from updated_utils import (
    get_params,
    handle_diffrax_stats,
    simple_plotter_debug,
    drive_plotter,
)

config.update("jax_enable_x64", True)

float_dtype = jnp.float64
complex_dtype = jnp.complex128

params = get_params(print_params=True)

# Physical Parameters for the Simulation
WC_EFF = -1.05j  # params["wc_eff"]
# WQ_EFF = #params["wq_eff"]
GAMMA_EFF = 0.025  # -2 * WQ_EFF.imag
CHI_EFF = 2.1  # params["chi_eff"]
KERR_EFF = 0.005  # params["kerr_eff"]

# Amplitude of 3.88
# Max Separation of 3.8816019794459597
# Time taken 2.95us
# Max Photons 7.968579718984644

QUBIT_STATE_G = 0.0
QUBIT_STATE_E = 1.0

ARGS = jnp.array([WC_EFF.imag, GAMMA_EFF, CHI_EFF, KERR_EFF], dtype=complex_dtype)
ARGS_G = jnp.array(
    [WC_EFF.imag, GAMMA_EFF, CHI_EFF, KERR_EFF, QUBIT_STATE_G], dtype=complex_dtype
)
ARGS_E = jnp.array(
    [WC_EFF.imag, GAMMA_EFF, CHI_EFF, KERR_EFF, QUBIT_STATE_E], dtype=complex_dtype
)

# Simulation and Solver Specific Parameters
T0 = 0.0
T1 = 4.0
NUM_ACTION = 61
NUM_SIM = 201
TS_SIM = jnp.linspace(T0, T1, NUM_SIM, dtype=float_dtype)
TS_ACTION = jnp.linspace(T0, T1, NUM_ACTION, dtype=float_dtype)

BATCH_SIZE = 2

# Constructing the Possible Drives and States for the Resonator
RES_DRIVE_AMP = 3.7
GAUSSIAN_MEAN = 0.5 * T1
GAUSSIAN_STD = 0.25

custom_drive = jnp.array(
    [
        [
            3.69491547 - 3.50931361e-01j,
            6.72930121 - 2.33867481e00j,
            3.77450049 + 2.27073148e00j,
            3.86820823 - 2.42574811e00j,
            4.45289493 + 2.48638123e00j,
            2.32496247 - 9.39155445e-01j,
            1.63034946 - 2.01874062e00j,
            4.76342171 + 2.35179633e00j,
            2.15763673 - 1.22603916e00j,
            3.66655231 + 9.48447138e-01j,
            2.76807547 - 6.43910468e-01j,
            3.01667899 + 4.59740572e-01j,
            3.65500301 - 5.02145477e-01j,
            2.56240875 - 8.71905871e-02j,
            4.02314842 - 1.52055481e-01j,
            2.75226533 + 5.46806678e-01j,
            4.29332882 - 2.15817951e-01j,
            2.21824199 + 3.83934230e-01j,
            4.11525279 + 2.88692117e-02j,
            3.41944844 + 3.11446749e-02j,
            3.61856848 + 4.68077064e-01j,
            3.61938685 + 2.71319374e-01j,
            3.58689874 - 2.53532045e-01j,
            3.4657228 + 2.12014150e-01j,
            3.52842987 - 2.22562235e-01j,
            3.26359183 + 2.87960079e-01j,
            2.51251429 + 1.68757588e00j,
            7.39478409 - 2.07404286e00j,
            3.20587873 + 1.74109936e-01j,
            -0.13940036 + 3.43552083e00j,
            -8.6601913 - 2.46023521e00j,
            -6.04055643 + 4.43706661e-01j,
            -9.43705618 - 1.16032735e-02j,
            -1.28904164 + 2.32014209e00j,
            3.82433832 + 4.37793970e00j,
            1.75555304 + 1.93925381e00j,
            2.42923975 - 3.42317373e00j,
            5.91170192 + 4.43493575e-02j,
            3.94437909 - 8.39746222e-01j,
            -4.3325758 + 1.68073833e00j,
            -6.02008879 + 4.49099630e00j,
            4.70876873 + 2.74956524e00j,
            -3.09303284 - 9.06136036e00j,
            5.66529632 - 5.84824860e00j,
            5.99182546 + 8.21442485e00j,
            -2.8857851 - 3.17872614e00j,
            0.70469052 + 3.67259413e00j,
            2.03827187 + 3.66764486e00j,
            0.57673417 - 7.12322295e00j,
            -0.82855508 + 2.23226726e00j,
            -2.79294163 + 6.64877295e00j,
            -4.40087378 + 3.64406586e00j,
            -2.15270057 + 3.97443205e00j,
            -2.73423105 + 2.20422372e00j,
            11.53959155 + 7.50876740e-01j,
            -4.38159525 - 5.25211871e00j,
            -7.82991171 - 1.30527616e01j,
            -0.51366128 - 4.52740371e00j,
            -3.31002951 + 4.18999612e00j,
            -5.09617686 - 4.79819894e00j,
            7.10769236 - 6.79059803e00j,
        ]
    ],
    dtype=complex_dtype,
).reshape(1, NUM_ACTION)
batched_custom_drive = custom_drive

rectangular_drive = (
    jnp.zeros_like(TS_ACTION, dtype=complex_dtype) + RES_DRIVE_AMP
).reshape(1, NUM_ACTION)
batched_rectangular_drive = rectangular_drive

gaussian_drive = jnp.asarray(
    (
        RES_DRIVE_AMP
        * jnp.exp(-((TS_ACTION - GAUSSIAN_MEAN) ** 2) / (2 * GAUSSIAN_STD**2))
    ),
    dtype=complex_dtype,
).reshape(1, NUM_ACTION)
batched_gaussian_drive = gaussian_drive

for i in range(BATCH_SIZE - 1):
    batched_rectangular_drive = jnp.concatenate(
        (batched_rectangular_drive, rectangular_drive), axis=0, dtype=complex_dtype
    )
    batched_gaussian_drive = jnp.concatenate(
        (batched_gaussian_drive, gaussian_drive), axis=0, dtype=complex_dtype
    )
    batched_custom_drive = jnp.concatenate(
        (batched_custom_drive, custom_drive), axis=0, dtype=complex_dtype
    )

batched_resonator_states = jnp.zeros((BATCH_SIZE, 1), dtype=complex_dtype) + 0.0
batched_custom_states = (
    jnp.zeros((BATCH_SIZE, 1), dtype=complex_dtype) - 0.06757781 - 2.39437953j
)


def single_eval(init_res_state, res_drive):
    control = LinearInterpolation(ts=TS_ACTION, ys=res_drive)

    solver = Tsit5()
    dt0 = 1e-4
    saveat = SaveAt(ts=TS_SIM)
    stepsize_controller = PIDController(
        rtol=1e-5, atol=1e-7, pcoeff=0.4, dcoeff=0.3, icoeff=0.0, jump_ts=TS_ACTION
    )
    max_steps = int(16**3)

    def vector_field(t, y, args):
        drive_res = control.evaluate(t)
        c = y[0]

        (
            wc_eff_imag,
            gamma_eff,
            chi_eff,
            kerr_eff,
            state,
        ) = args

        d_y = -1.0j * (
            c
            * (
                1.0j * wc_eff_imag
                + 0.5 * chi_eff
                - chi_eff * state * jnp.exp(-gamma_eff * t)
                - kerr_eff * jnp.absolute(c) ** 2
            )
            + drive_res
        )
        return jnp.array([d_y], dtype=complex_dtype)

    ode_term = ODETerm(vector_field)
    sol_g = diffeqsolve(
        solver=solver,
        terms=ode_term,
        t0=T0,
        t1=T1,
        dt0=dt0,
        y0=init_res_state,
        args=ARGS_G,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    sol_e = diffeqsolve(
        solver=solver,
        terms=ode_term,
        t0=T0,
        t1=T1,
        dt0=dt0,
        y0=init_res_state,
        args=ARGS_E,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )

    return sol_g.ys, sol_e.ys, sol_g.stats, sol_e.stats


def combined_eval(init_res_state, res_drive):
    control = LinearInterpolation(ts=TS_ACTION, ys=res_drive)

    y0 = jnp.array([*init_res_state, *init_res_state], dtype=complex_dtype)
    solver = Tsit5()
    dt0 = 1e-4
    saveat = SaveAt(ts=TS_SIM)
    stepsize_controller = PIDController(
        rtol=1e-5, atol=1e-7, pcoeff=0.4, dcoeff=0.3, icoeff=0.0, jump_ts=TS_ACTION
    )
    max_steps = int(16**3)

    def vector_field(t, y, args):
        drive_res = control.evaluate(t)
        res_g, res_e = y

        (
            wc_eff_imag,
            gamma_eff,
            chi_eff,
            kerr_eff,
        ) = args

        d_res_g = -1.0j * (
            res_g
            * (1.0j * wc_eff_imag + 0.5 * chi_eff - kerr_eff * jnp.absolute(res_g) ** 2)
            + drive_res
        )
        d_res_e = -1.0j * (
            res_e
            * (
                1.0j * wc_eff_imag
                + 0.5 * chi_eff
                - chi_eff * jnp.exp(-gamma_eff * t)
                - kerr_eff * jnp.absolute(res_e) ** 2
            )
            + drive_res
        )
        return jnp.array([d_res_g, d_res_e], dtype=complex_dtype)

    ode_term = ODETerm(vector_field)
    sol = diffeqsolve(
        solver=solver,
        terms=ode_term,
        t0=T0,
        t1=T1,
        dt0=dt0,
        y0=y0,
        args=ARGS,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )

    return sol.ys, sol.stats


batched_single_eval = jit(vmap(single_eval, in_axes=0))
batched_combined_eval = jit(vmap(combined_eval, in_axes=0))

a_, b_, c_, d_ = batched_single_eval(
    batched_resonator_states, batched_rectangular_drive
)

start = time.time()
single_res_g, single_res_e, single_stats_g, single_stats_e = block_until_ready(
    batched_single_eval(batched_resonator_states, batched_rectangular_drive)
)
time_taken = time.time() - start
print(
    f"total time taken for state separate eval: {time_taken*1e6}us with batchsize: {BATCH_SIZE}"
)
print(f"time taken per simulation of g + e: {time_taken/BATCH_SIZE*1e6}us")

a_, b_ = batched_combined_eval(batched_custom_states, batched_custom_drive)

start = time.time()
combined_res, combined_stats = block_until_ready(
    batched_combined_eval(batched_custom_states, batched_custom_drive)
)
time_taken = time.time() - start
print(
    f"total time taken for combined eval: {time_taken*1e6}us with batchsize: {BATCH_SIZE}"
)
print(f"time taken per simulation of g + e: {time_taken/BATCH_SIZE*1e6}us")

# The index of the batch results we want to compare
index = 0

single_g = single_res_g[index]
single_e = single_res_e[index]

separation_single = jnp.abs(single_g - single_e)
max_separation_single = jnp.max(separation_single)
index_single = jnp.where(separation_single == max_separation_single, size=1)[0]
time_of_peak_single = TS_SIM[index_single]

reshaped_g = single_g.reshape(NUM_SIM, 1)
reshaped_e = single_e.reshape(NUM_SIM, 1)
combined_single = jnp.concatenate((reshaped_g, reshaped_e), axis=1)

photons_ge = jnp.abs(combined_single) ** 2
max_photons_in_each = jnp.max(photons_ge, axis=1)
max_photons_single = jnp.max(max_photons_in_each)

combined_g = combined_res[index, :, 0]
combined_e = combined_res[index, :, 1]

separation_combined = jnp.abs(combined_g - combined_e)
max_separation_combined = jnp.max(separation_combined)
index_combined = jnp.where(separation_combined == max_separation_combined, size=1)[0]
time_of_peak_combined = TS_SIM[index_combined]

photons_ge = jnp.abs(combined_res[index]) ** 2
max_photons_in_each = jnp.max(photons_ge, axis=1)
max_photons_combined = jnp.max(max_photons_in_each)


print(f"max separation single: {max_separation_single}")
print(f"time of max separation single: {time_of_peak_single}")
print(f"max photons: {max_photons_single}")
print("###")
print(f"max separation combined: {max_separation_combined}")
print(f"time of max separation combined: {time_of_peak_combined}")
print(f"max photons: {max_photons_combined}")

handle_diffrax_stats(single_stats_g, name="Single Ground")
handle_diffrax_stats(single_stats_e, name="Single Excited")
handle_diffrax_stats(combined_stats, name="Combined")

fig1, ax1 = simple_plotter_debug(
    ts=TS_SIM,
    complex_1=single_g,
    complex_2=single_e,
    name_1="Ground",
    name_2="Excited",
    fig_name="Single Results",
)

fig2, ax2 = simple_plotter_debug(
    ts=TS_SIM,
    complex_1=combined_g,
    complex_2=combined_e,
    name_1="Ground",
    name_2="Excited",
    fig_name="Combined Results",
)

fig3, ax3 = drive_plotter(
    ts=TS_ACTION,
    complex_drive=custom_drive.reshape(NUM_ACTION),
    fig_name="Custom Drive",
)

plt.show()
