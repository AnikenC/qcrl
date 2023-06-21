import time
import matplotlib.pyplot as plt

import numpy as np

import jax
from jax import jit, config, vmap, block_until_ready
import jax.numpy as jnp

from simple_resonator_env import SimpleResonatorEnv

config.update("jax_enable_x64", True)

float_dtype = jnp.float64
complex_dtype = jnp.complex128

env = SimpleResonatorEnv()

dict_params = env.get_params()
list_params = list(dict_params.items())
array_params = np.array(list_params)

t0, t1, num_actions, num_sim, ts_action, ts_sim, batch_size = array_params[:, 1]

res_drive_amp = 3.88
mean = 0.5 * (t1 - t0)
std = 0.25

single_gaussian_complex = jnp.asarray(
    res_drive_amp * jnp.cos(2000 * ts_sim),
    dtype=complex_dtype,
).reshape(1, num_sim)

single_gaussian_real = jnp.asarray(
    res_drive_amp * jnp.cos(2000 * ts_sim),
    dtype=float_dtype,
).reshape(1, num_sim)

custom_drive = jnp.array(
    [
        7.49172509,
        7.46409774,
        5.14059782,
        5.66096365,
        3.08803201,
        3.44390482,
        2.61656076,
        3.18328261,
        2.93183446,
        3.59811336,
        2.02354014,
        4.07287091,
        3.69312227,
        2.4635835,
        4.09731507,
        3.60129863,
        3.33313048,
        3.68065923,
        3.43288362,
        3.76958579,
        3.79945666,
        3.56833935,
        3.64312619,
        3.65251511,
        4.21696126,
        -0.11507809,
        -4.09716517,
        -8.40801835,
        -2.82565445,
        2.67126083,
        -4.01691645,
        3.39389145,
        11.03850484,
        1.03824936,
        2.02817351,
        5.38591266,
        -8.88300359,
        5.37384212,
        -3.63639116,
        1.13724604,
        -1.38552904,
        0.45808896,
        -1.46360978,
        -1.15160972,
        3.37208807,
        0.39334126,
        1.44626781,
        -8.36879969,
        0.91157705,
        -4.54192281,
        -1.27236828,
    ],
    dtype=float_dtype,
).reshape(1, num_actions)

batched_gaussian_complex = single_gaussian_complex
batched_gaussian_real = single_gaussian_real
batched_custom_drive = custom_drive

for i in range(batch_size - 1):
    batched_gaussian_complex = jnp.concatenate(
        (batched_gaussian_complex, single_gaussian_complex), axis=0, dtype=complex_dtype
    )
    batched_gaussian_real = jnp.concatenate(
        (batched_gaussian_real, single_gaussian_real), axis=0, dtype=float_dtype
    )
    batched_custom_drive = jnp.concatenate(
        (batched_custom_drive, custom_drive), axis=0, dtype=float_dtype
    )


def jax_fft(batched_array):
    freqs = jnp.fft.fft(batched_array, axis=-1)
    return freqs


vmapped_fft = jit(vmap(jax_fft))

start = time.time()
fft_res = vmapped_fft(batched_gaussian_complex)
time_taken = time.time() - start
print(f"time taken for fft batch of {batch_size}: {time_taken}")
print(f"time taken per: {time_taken / batch_size}")

fft_shifted = jnp.fft.fftshift(fft_res, axes=-1)


def jax_rfft(batched_array):
    freqs = jnp.fft.rfft(batched_array, axis=-1)
    return freqs


vmapped_rfft = jit(vmap(jax_rfft))

start = time.time()
rfft_res = vmapped_rfft(batched_gaussian_real)
time_taken = time.time() - start
print(f"time taken for rfft batch of {batch_size}: {time_taken}")
print(f"time taken per: {time_taken / batch_size}")

custom_rfft = vmapped_rfft(batched_custom_drive)

rfft_shifted = jnp.fft.fftshift(rfft_res, axes=-1)

custom_rfft_shifted = jnp.fft.fftshift(custom_rfft, axes=-1)

fft_freqs = jnp.fft.fftfreq(201, 1)

print(fft_freqs)

fig, ax = plt.subplots(3)
ax[0].plot(ts_sim, jnp.abs(fft_shifted[0]), label="fft", color="red")
ax[0].legend()

ax[1].plot(
    jnp.fft.rfftfreq(201, 0.08),
    jnp.abs(rfft_shifted[0]),
    label="rfft",
    color="blue",
)
ax[1].legend()

ax[2].plot(
    jnp.fft.rfftfreq(51, 0.08),
    jnp.abs(custom_rfft_shifted[0]),
    label="custom rfft",
    color="green",
)
ax[2].legend()

plt.show()
