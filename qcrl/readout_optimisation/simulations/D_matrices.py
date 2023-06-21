import time
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, config, vmap, block_until_ready
from diffrax import (
    diffeqsolve,
    ODETerm,
    Tsit5,
    SaveAt,
    PIDController,
    LinearInterpolation,
)

from updated_utils import get_params, simple_plotter_debug, simple_plotter

config.update("jax_enable_x64", True)

float_dtype = jnp.float64
complex_dtype = jnp.complex128

physics_params = get_params(print_params=True)

WQ = physics_params["wq_eff"]

T0 = 0.0
T1 = 5.0
NUM_ACTIONS = 51
NUM_SIM = 201
TS_ACTION = jnp.linspace(T0, T1, NUM_ACTIONS, dtype=float_dtype)
TS_SIM = jnp.linspace(T0, T1, NUM_SIM, dtype=float_dtype)

BATCH_SIZE = 100

n_dims = 3

drive_on_trans = (jnp.zeros_like(TS_ACTION, dtype=complex_dtype) + 0.001).reshape(51, 1)

batched_drive_trans = drive_on_trans

for i in range(BATCH_SIZE - 1):
    batched_drive_trans = jnp.concatenate((batched_drive_trans, drive_on_trans), axis=1)

state_vector_g = jnp.array([1.0, 0.0, 0.0], dtype=complex_dtype).reshape(n_dims, 1)
state_vector_e = jnp.array([0.0, 1.0, 0.0], dtype=complex_dtype).reshape(n_dims, 1)
conjugate_state_g = jnp.conjugate(state_vector_g).reshape(1, 1, 1, n_dims)
conjugate_state_e = jnp.conjugate(state_vector_e).reshape(1, 1, 1, n_dims)

y0_sim = jnp.diag(jnp.sqrt(jnp.arange(1, n_dims, dtype=complex_dtype)), 1).reshape(
    n_dims, n_dims
)
y0 = y0_sim.reshape(1, 1, n_dims, n_dims)
grouped_operator = y0
identity = jnp.eye(n_dims, dtype=complex_dtype)

print(f"y0_sim: {y0_sim}")

grouped_conjugate_state_g = conjugate_state_g
grouped_conjugate_state_e = conjugate_state_e

for i in range(NUM_SIM - 1):
    grouped_conjugate_state_g = jnp.concatenate(
        (grouped_conjugate_state_g, conjugate_state_g), axis=1
    )
    grouped_conjugate_state_e = jnp.concatenate(
        (grouped_conjugate_state_e, conjugate_state_e), axis=1
    )
    grouped_operator = jnp.concatenate((grouped_operator, y0), axis=1)

grouped_conjugate_state_g_sim = grouped_conjugate_state_g[0]
grouped_conjugate_state_e_sim = grouped_conjugate_state_e[0]

batched_conjugate_state_g = grouped_conjugate_state_g
batched_conjugate_state_e = grouped_conjugate_state_e
batched_operator = grouped_operator

for i in range(BATCH_SIZE - 1):
    batched_conjugate_state_g = jnp.concatenate(
        (batched_conjugate_state_g, grouped_conjugate_state_g), axis=0
    )
    batched_conjugate_state_e = jnp.concatenate(
        (batched_conjugate_state_e, grouped_conjugate_state_e), axis=0
    )
    batched_operator = jnp.concatenate((batched_operator, grouped_operator), axis=0)


@jit
def batched_expectation_vals(b_operators, b_conjugate_state_g, b_conjugate_state_e):
    transformed_g = b_operators @ state_vector_g
    expectation_vals_g = b_conjugate_state_g @ transformed_g

    transformed_e = b_operators @ state_vector_e
    expectation_vals_e = b_conjugate_state_e @ transformed_e

    return expectation_vals_g, expectation_vals_e


a_, b_ = batched_expectation_vals(
    batched_operator, batched_conjugate_state_g, batched_conjugate_state_e
)


def single_eval(drive_trans):
    dt0 = 1e-4
    saveat = SaveAt(ts=TS_SIM)
    stepsize_controller = PIDController(
        rtol=1e-5, atol=1e-7, jump_ts=TS_ACTION, pcoeff=0.4, dcoeff=0.3, icoeff=0.0
    )

    control = LinearInterpolation(ts=TS_ACTION, ys=drive_trans)
    args = jnp.array([-0.00625])

    def vector_field(t, y, args):
        drive_trans = control.evaluate(t)
        kappa_half = args[0]
        d_q = kappa_half * y - 1j * drive_trans * identity
        return d_q

    ode_term = ODETerm(vector_field)
    sol = diffeqsolve(
        terms=ode_term,
        solver=Tsit5(),
        t0=T0,
        t1=T1,
        dt0=dt0,
        y0=y0_sim,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(16**3),
    )

    return sol.ys, sol.stats


batched_eval = jit(vmap(single_eval, in_axes=1))
a_, b_ = batched_eval(batched_drive_trans)

start = time.time()
(
    batched_results,
    batched_stats,
) = block_until_ready(batched_eval(batched_drive_trans))
now = time.time()
total_time = now - start
print(f"total time taken: {total_time*1e6}us")
print(f"time taken per sim: {total_time/(BATCH_SIZE)*1e6}us")

start = time.time()
batched_expectation_value_g, batched_expectation_value_e = batched_expectation_vals(
    batched_results, batched_conjugate_state_g, batched_conjugate_state_e
)
now = time.time()
total_time = now - start
print(f"total time taken: {total_time*1e6}us")

index = 0
results_0 = batched_results[index]
stats_0 = batched_stats
expectation_value_g = batched_expectation_value_g[index].reshape(NUM_SIM)
expectation_value_e = batched_expectation_value_e[index].reshape(NUM_SIM)

print(f"stats: {stats_0}")

fig1, ax1 = simple_plotter(
    ts=TS_SIM,
    complex_1=expectation_value_g,
    complex_2=expectation_value_e,
    name_1="Ground",
    name_2="Excited",
    fig_name="Matrix Results",
)

plt.show()
