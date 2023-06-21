import time
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, config, vmap
from jax.experimental import sparse
from diffrax import (
    diffeqsolve,
    Tsit5,
    ODETerm,
    LinearInterpolation,
    SaveAt,
    PIDController,
)

config.update("jax_enable_x64", True)

from updated_utils import *

FLOAT = jnp.float64
COMPLEX = jnp.complex128

N_DIMS = 3
BATCH_SIZE = 100

T0 = 0.0
T1 = 1.0
NUM_ACTIONS = 51
NUM_SIM = 201
TS_ACTIONS = jnp.linspace(T0, T1, NUM_ACTIONS, dtype=FLOAT)
TS_SIM = jnp.linspace(T0, T1, NUM_SIM, dtype=FLOAT)

INIT_KET = jnp.array([1.0, 0.0, 0.0], dtype=COMPLEX).reshape(N_DIMS, 1)
INIT_DM = state2dm(INIT_KET)

EYE = q_eye(N_DIMS, use_sparse=False, dtype=COMPLEX)

Q = annihilation_op(N_DIMS, use_sparse=False, dtype=COMPLEX)
QD = creation_op(N_DIMS, use_sparse=False, dtype=COMPLEX)
NQ = number_op(N_DIMS, use_sparse=False, dtype=COMPLEX)

print(f"INIT_DM: {INIT_DM}")
print(f"Q: {Q}")
print(f"QD: {QD}")
print(f"NQ: {NQ}")
print(f"EYE: {EYE}")


GAMMA = 1 / 80
TRANS_DRIVE_AMP = 2.0

ARGS = jnp.array([GAMMA], dtype=COMPLEX)

single_trans_drive = (
    jnp.zeros_like(TS_ACTIONS, dtype=COMPLEX).reshape(1, NUM_ACTIONS) + TRANS_DRIVE_AMP
)
batched_trans_drive = single_trans_drive

for i in range(BATCH_SIZE - 1):
    batched_trans_drive = jnp.concatenate(
        (batched_trans_drive, single_trans_drive), axis=0
    )


def single_eval(drive_trans):
    solver = Tsit5()
    dt0 = 1e-4
    saveat = SaveAt(ts=TS_SIM)
    stepsize_controller = PIDController(
        rtol=1e-5, atol=1e-7, pcoeff=0.4, dcoeff=0.3, icoeff=0.0, jump_ts=TS_ACTIONS
    )
    max_steps = int(16**4)

    control = LinearInterpolation(ts=TS_ACTIONS, ys=drive_trans)

    def dm_vector_field(t, y, args):
        """
        H0 = wq * q.dag() * q + B(t)*(q + q.dag())
        H1 = B(t) * (q + q.dag())
        p_dot(t) = -1j * [H, p(t)] + D[q]
        D[q] = kappa * (q * p(t) * q.dag() - 0.5 * {q.dag()*q, p(t)})
        """
        gamma = args[0]
        trans_drive = control.evaluate(t)
        d_y = -1j * trans_drive * ((Q + QD) @ y - y @ (Q + QD)) + gamma * (
            Q @ y @ QD - 0.5 * (NQ @ y + y @ NQ)
        )
        return d_y

    lmde_term = ODETerm(dm_vector_field)

    sol = diffeqsolve(
        terms=lmde_term,
        solver=solver,
        t0=T0,
        t1=T1,
        dt0=dt0,
        y0=INIT_DM,
        args=ARGS,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )

    return sol.ys, sol.stats


batched_eval = jit(vmap(single_eval, in_axes=0))

a_, b_ = batched_eval(batched_trans_drive)

start = time.time()
results, stats = batched_eval(batched_trans_drive)
now = time.time()
time_taken = now - start
print(f"time taken for batchsize: {BATCH_SIZE}: {time_taken*1e6}us")
print(f"time taken per simulation: {time_taken/BATCH_SIZE*1e6}us")
