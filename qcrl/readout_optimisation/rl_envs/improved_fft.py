import numpy as np

import matplotlib.pyplot as plt

from jax import jit, config, vmap
import jax.numpy as jnp
from jax.numpy.fft import *

from resonator_environment import ResonatorEnv

config.update("jax_enable_x64", True)

float_dtype = jnp.float64
complex_dtype = jnp.complex128

env = ResonatorEnv()

dict_params = env.get_params()
list_params = list(dict_params.items())
array_params = np.array(list_params)

t0, t1, num_actions, num_sim, ts_action, ts_sim, batch_size = array_params[:, 1]

res_drive_amp = 3.88
mean = 0.5 * (t1 - t0)
std = 0.25 * (t1 - t0)

gaussian_drive = jnp.asarray(
    (res_drive_amp * jnp.exp(-((ts_action - mean) ** 2) / (2 * std**2))),
    dtype=complex_dtype,
)
rectangular_drive = jnp.zeros((num_actions,), dtype=complex_dtype) + 3.685
custom_drive = jnp.array(
    [
        -1.76363215 + 7.47867465e00j,
        1.45123065 + 8.37511122e00j,
        2.16286659 + 6.81536973e00j,
        -1.89950794 + 9.86424163e-01j,
        -1.0951671 + 3.92279029e00j,
        1.29541174 + 2.89965451e00j,
        -0.59959225 + 2.82212377e00j,
        0.68357438 + 3.15843821e00j,
        -0.56981046 + 2.63452381e00j,
        1.80789351 + 3.20346475e00j,
        -1.11340985 + 2.66560107e00j,
        1.04290172 + 3.35500836e00j,
        -1.47978723 + 1.96030825e00j,
        1.12643905 + 3.21657717e00j,
        1.10523611 + 4.17771637e00j,
        -1.57069266 + 2.74708807e00j,
        0.78505881 + 2.87393808e00j,
        0.77856809 + 4.34716314e00j,
        -0.05674863 + 3.04446906e00j,
        0.35236437 + 3.30209911e00j,
        0.12937228 + 3.43653649e00j,
        -0.51999651 + 3.20424765e00j,
        0.83546005 + 3.86598527e00j,
        0.04282562 + 3.63748670e00j,
        0.25979817 + 3.16538900e00j,
        -0.22579031 + 3.72094691e00j,
        -0.15714848 + 3.32035333e00j,
        0.83990976 + 3.95251453e00j,
        -0.53462353 + 3.69702190e00j,
        0.45067154 + 3.94601554e00j,
        0.06876303 + 3.71294916e00j,
        0.34239724 + 3.54432136e00j,
        -0.28335668 + 3.87410223e00j,
        0.19973299 + 3.79095227e00j,
        0.35572954 + 3.68423730e00j,
        -0.21296047 + 4.05661076e00j,
        0.14307167 + 3.67210925e00j,
        0.01245901 + 4.02947456e00j,
        0.14908602 + 3.63136560e00j,
        -0.09474931 + 3.89942080e00j,
        0.05978119 + 3.87188375e00j,
        0.29105585 + 2.40469739e00j,
        -5.29986978 - 1.02324009e01j,
        1.03545338 - 6.22235239e-03j,
        9.95928288 - 1.03231955e01j,
        -2.08755597 - 5.96297503e00j,
        2.77124286 + 2.20843226e00j,
        2.02335715 - 2.97137439e00j,
        -2.10546806 - 5.14074028e00j,
        0.48477955 - 4.46932167e00j,
        -5.90931833 + 1.58012211e-01j,
        0.60663775 - 1.13317162e00j,
        0.8045283 - 7.67468333e00j,
        7.53545582 - 8.31820130e00j,
        3.55490983 - 8.39307785e00j,
        2.93335527 - 7.33340979e-01j,
        6.48033559 + 5.76431990e00j,
        1.95541501 - 2.40821600e00j,
        -4.08921331 + 2.97604501e00j,
        10.47285795 + 6.33377373e00j,
        7.94311106 - 2.83325016e00j,
    ],
    dtype=complex_dtype,
)

batched_gaussian = gaussian_drive
batched_rectangular = rectangular_drive
batched_custom = custom_drive

fft_gaussian = fft(batched_gaussian, axis=0)
fft_rectangular = fft(batched_rectangular, axis=0)
fft_custom = fft(batched_custom, axis=0)

shifted_gaussian = jnp.abs(fftshift(fft_gaussian))
shifted_rectangular = jnp.abs(fftshift(fft_rectangular))
shifted_custom = jnp.abs(fftshift(fft_custom))

normed_gaussian = shifted_gaussian / jnp.max(shifted_gaussian)
normed_rectangular = shifted_rectangular / jnp.max(shifted_rectangular)
normed_custom = shifted_custom / jnp.max(shifted_custom)

freqs = jnp.linspace(-5.0, 5.0, len(normed_custom), dtype=float_dtype)

threshold = 0.15

modded_gaussian = jnp.where(normed_gaussian - threshold > 0.0, size=num_actions)[0]
min_gaussian = modded_gaussian[0]
max_gaussian = jnp.max(modded_gaussian)

min_gaussian_freq_adjusted = 0.5 * (freqs[min_gaussian] + freqs[min_gaussian - 1])
max_gaussian_freq_adjusted = 0.5 * (freqs[max_gaussian] + freqs[max_gaussian + 1])

print(f"min gaussian freq: {min_gaussian_freq_adjusted}")
print(f"max gaussian freq: {max_gaussian_freq_adjusted}")

modded_custom = jnp.where((normed_custom - threshold) > 0.0, size=num_actions)[0]
min_custom = modded_custom[0]
max_custom = jnp.max(modded_custom)

min_custom_freq_adjusted = 0.5 * (freqs[min_custom] + freqs[min_custom - 1])
max_custom_freq_adjusted = 0.5 * (freqs[max_custom] + freqs[max_custom + 1])

print(f"min custom freq: {min_custom_freq_adjusted}")
print(f"max custom freq: {max_custom_freq_adjusted}")

modded_rectangular = jnp.where((normed_gaussian - threshold) > 0.0, size=num_actions)[0]
min_rectangular = modded_rectangular[0]
max_rectangular = jnp.max(modded_rectangular)

min_rectangular_freq_adjusted = 0.5 * (
    freqs[min_rectangular] + freqs[min_rectangular - 1]
)
max_rectangular_freq_adjusted = 0.5 * (
    freqs[max_rectangular] + freqs[max_rectangular + 1]
)

print(f"min rectangular freq: {min_rectangular_freq_adjusted}")
print(f"max rectangular freq: {max_rectangular_freq_adjusted}")


fig, ax = plt.subplots(3, num="FFT Plots")

ax[0].plot(freqs, normed_gaussian, label="gaussian", color="red")
ax[0].plot(freqs, jnp.zeros_like(freqs) + threshold, label="threshold", color="yellow")
ax[0].plot(
    freqs[min_gaussian], normed_gaussian[min_gaussian], label="min", color="blue"
)
ax[0].plot(
    freqs[max_gaussian], normed_gaussian[max_gaussian], label="max", color="blue"
)
ax[0].legend()

ax[1].plot(freqs, normed_rectangular, label="rectangular", color="red")
ax[1].plot(freqs, jnp.zeros_like(freqs) + threshold, label="threshold", color="yellow")
ax[1].plot(
    freqs[min_rectangular],
    normed_rectangular[min_rectangular],
    label="min",
    color="blue",
)
ax[1].plot(
    freqs[max_rectangular],
    normed_rectangular[max_rectangular],
    label="max",
    color="blue",
)
ax[1].legend()

ax[2].plot(freqs, normed_custom, label="custom", color="red")
ax[2].plot(freqs, jnp.zeros_like(freqs) + threshold, label="threshold", color="yellow")
ax[2].plot(freqs[min_custom], normed_custom[min_custom], label="min", color="blue")
ax[2].plot(freqs[max_custom], normed_custom[max_custom], label="max", color="blue")
ax[2].legend()

plt.show()
