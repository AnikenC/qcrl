import matplotlib.pyplot as plt

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


import jax.numpy as jnp
from jax import jit, config
from diffrax import (
    diffeqsolve,
    ODETerm,
    Tsit5,
    LinearInterpolation,
    SaveAt,
    PIDController,
)

config.update("jax_enable_x64", True)

from utils import get_params, complex_plotter

# defining consistent dtypes to be used
float_dtype = jnp.float64
complex_dtype = jnp.complex128

# physics args
params = get_params()
KAPPA = params["kappa"]
CHI = params["chi"]
GAMMA = params["gamma"]
G = params["g"]
KERR = params["kerr"]
WA = params["wr"]
WB = params["wq"]
DELTA = params["delta"]
SQRT_RATIO = jnp.sqrt(2 * params["ratio"])
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
ANHARM = params["anharm"]
RES_DRIVE_FREQ = WC.real - 0.5 * CHI - KERR + 0.125 * SQRT_RATIO * CHI
TRANS_DRIVE_FREQ = (
    WQ.real - 0.5 * CHI - ANHARM + 0.25 * SQRT_RATIO * (CHI + KERR + ANHARM)
)
COS_ANGLE = jnp.cos(0.5 * jnp.arctan(2 * G / DELTA))
SIN_ANGLE = jnp.sin(0.5 * jnp.arctan(2 * G / DELTA))

args = jnp.array(
    [
        CHI,
        KERR,
        ANHARM,
        WC,
        WQ,
        RES_DRIVE_FREQ,
        TRANS_DRIVE_FREQ,
        COS_ANGLE,
        SIN_ANGLE,
        SQRT_RATIO,
    ],
    dtype=complex_dtype,
)

# sim params
t0 = 0.0
t1 = 1.0
dt0 = 1e-4
ns_per_sample = 4
samples = int(1000.0 / ns_per_sample * (t1 - t0))
ts = jnp.linspace(t0, t1, samples + 1, dtype=float_dtype)

y0 = jnp.array([0.0 + 1j * 0.0, 1.0 + 1j * 0.0], dtype=complex_dtype)

solver = Tsit5()
saveat = SaveAt(ts=ts)
stepsize_controller = PIDController(rtol=1e-8, atol=1e-8, jump_ts=ts)
max_steps = int(1e6)


class ReadoutEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        state = {}
        self.n_actions = int(1000 * (t1 - t0) / 20) + 1
        # 2 pulses, complex_amplitudes so 2 amps per sample, n_actions num of samples
        # 2 freqs, one for res, one for trans
        # initial state, complex_valued so 2 values
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(4 * self.n_actions + 2 + 2,), dtype=float_dtype
        )
        # Large Observation Space
        # complex amplitudes at each given time
        # n_actions nums of samples x 2 because of complex
        # only the resonator is observable
        self.observation_space = Box(
            low=-100.0, high=100.0, shape=(2 * self.n_actions,), dtype=complex_dtype
        )

    def _get_obs(self):
        observation = np.zeros((2 * self.n_actions,), dtype=complex_dtype)
        return observation

    def _get_info(self):
        info = {}
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_obs(), self._get_info()

    def step(self, action):
        # seperate action into individual complex arrays
        real_res = action[0 : self.n_actions]
        imag_res = action[self.n_actions : 2 * self.n_actions]
        real_trans = action[2 * self.n_actions : 3 * self.n_actions]
        imag_trans = action[3 * self.n_actions : 4 * self.n_actions]
        freqs = action[4 * self.n_actions : 4 * self.n_actions + 2]
        init_state = action[4 * self.n_actions + 2 :]
        print(f"real res: {real_res}")
        print(real_res.shape)
        print(f"imag res: {imag_res}")
        print(imag_res.shape)
        print(f"real trans: {real_trans}")
        print(real_trans.shape)
        print(f"imag trans: {imag_trans}")
        print(imag_trans.shape)
        print(f"freqs: {freqs}")
        print(freqs.shape)
        print(f"init_state: {init_state}")
        print(init_state.shape)

        drive_res = real_res + 1j * imag_res
        drive_trans = real_trans + 1j * imag_trans
        drive_arr = jnp.vstack((drive_res, drive_trans), dtype=complex_dtype).T
        control = LinearInterpolation(ts=ts, ys=drive_arr)

        def vector_field(t, y, args):
            c, q = y
            (
                chi,
                kerr,
                anharm,
                wc,
                wq,
                res_drive_freq,
                trans_drive_freq,
                cos_angle,
                sin_angle,
                sqrt_ratio,
            ) = args
            drive_res, drive_trans = control.evaluate(t)

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
            return jnp.array([d_c, d_q], dtype=complex_dtype)

        ode_term = ODETerm(vector_field)
        sol = diffeqsolve(
            terms=ode_term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )

        # Calculate Reward from Observation
        reward = 0

        done = True

        return self._get_obs(), reward, done, False, self._get_info()
