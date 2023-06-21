from jax import config, jit
import numpy as np
import jax.numpy as jnp
from simple_resonator_env import SimpleResonatorEnv

config.update("jax_enable_x64", True)

float_dtype = jnp.float64
complex_dtype = jnp.complex128

env = SimpleResonatorEnv()
dict_info = env.get_params()

items_list = list(dict_info.items())
array = np.array(items_list)

T0, T1, NUM_ACTION, NUM_SIM, TS_ACTION, TS_SIM, batch_size = array[:, 1]

rectangular_drive = (
    0.368 * (jnp.cos(10.0 * TS_ACTION) + jnp.sin(500.0 * TS_ACTION))
).reshape(1, NUM_ACTION)

rectangular_drive = jnp.zeros((1, NUM_ACTION), dtype=float_dtype) + 3.54

rec = rectangular_drive

res_imag = -1.9

longitudinal_drive = -0.5 * (
    res_imag * env.chi_eff * (1 - jnp.exp(env.neg_kappa_half * TS_ACTION))
    + res_imag * env.chi_eff
).reshape(1, NUM_ACTION)

# for i in range(batch_size - 1):
#    rec = jnp.concatenate((rec, rectangular_drive), axis=0)

resonator_state = jnp.zeros((1, 1), dtype=float_dtype) + 0.0

action = jnp.concatenate((rec, resonator_state), axis=-1)

custom_drive = jnp.array(
    [
        1.87106952e00,
        1.08086371e01,
        4.33037490e00,
        4.21703935e00,
        4.42910910e00,
        2.39909038e00,
        3.31430882e00,
        3.18842441e00,
        2.17089742e00,
        2.90533185e00,
        3.81668806e00,
        2.62953877e00,
        4.39037353e00,
        1.55792624e00,
        3.94573838e00,
        2.10852429e00,
        3.26255262e00,
        4.99951452e00,
        2.68077374e00,
        3.84400010e00,
        3.17611903e00,
        3.62067938e00,
        2.24116772e00,
        4.24133927e00,
        3.60675514e00,
        3.63811165e00,
        3.68468434e00,
        3.00705731e00,
        3.72367442e00,
        2.93861061e00,
        4.33120668e00,
        3.92311394e00,
        3.73550713e00,
        3.63236547e00,
        3.46737444e00,
        3.94828618e00,
        3.58321875e00,
        3.95674676e00,
        3.71710867e00,
        3.69094133e00,
        3.94253254e00,
        3.62256527e00,
        3.63369375e00,
        4.04912114e00,
        3.75113249e00,
        3.92041683e00,
        3.75520855e00,
        3.65732461e00,
        4.08363551e00,
        3.56559902e00,
        4.17831689e00,
        3.65649104e00,
        3.86179179e00,
        4.09843981e00,
        2.81270653e00,
        -1.80459380e00,
        -4.95228678e00,
        2.06551224e00,
        4.93175507e00,
        -1.17217161e00,
        -7.81235695e-01,
        2.51236111e00,
        1.70588970e00,
        3.97604525e00,
        -8.44224915e-01,
        8.70972872e-03,
        5.73653877e-01,
        2.62105286e00,
        5.78078091e00,
        -1.19117066e00,
        6.62256658e00,
        1.40291229e00,
        8.93002748e-01,
        -4.79971766e-01,
        -7.96227634e00,
        8.22722673e00,
        2.17568964e00,
        2.06890896e00,
        7.78969377e-02,
        2.93003201e00,
        -3.59870136e00,
    ],
    dtype=float_dtype,
).reshape(1, NUM_ACTION)

custom_state = jnp.array([-0.0 + 1.0j * res_imag], dtype=complex_dtype).reshape(1, 1)

env.render(longitudinal_drive, custom_state, index=0)

"""
Vacuum Readout
Maximum Reward: 3.7704138044185282
Separation at Max Reward: [3.7704138]
Photon at Max: [7.97910092]
Bandwidth at Max: [[-3.33066907e-16]]
3.2us

Max Reward Obtained: 3.8339075569054306
Separation at Max Reward: [3.82332138]
Photon at Max Reward: [7.97148308]
Bandwidth at Max Reward: [[0.24691358]]
2.7us

-2.06014618j

A+R Readout
Max Reward Obtained: -96.25767052301387
Separation at Max Reward: [3.74232948]
Photon at Max: [8.00791206]
Bandwidth at Max: [[-3.33066907e-16]]
2.3us

A+L Readout wo Kerr and Gamma
Maximum Reward: 3.999817965012203
Separation at Max Reward: [3.99981797]
Photon at Max: [7.9996362]
Bandwidth at Max: [[0.]]
10us

A+L Readout w Kerr and Gamma
Maximum Reward: -96.09581105457289
Separation at Max Reward: [3.90418895]
Photon at Max: [11.0951808]
Bandwidth at Max: [[0.]]
Time taken 5.85us


"""
