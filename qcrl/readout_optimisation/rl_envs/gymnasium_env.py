import matplotlib.pyplot as plt

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import time

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

config.update("jax_enable_x64", False)

KAPPA = 5.35 * 2 * jnp.pi  # 33.6150
CHI = 0.16 * KAPPA * 2.0  # 10.7568
WA = 7062.0 * 2 * jnp.pi  # 44371.85
WB = 5092.0 * 2 * jnp.pi  # 31993.97
DELTA = WA - WB  # 12377.87
G = 102.9 * 2 * jnp.pi  # 646.54
GAMMA = 1 / 39.2 * 2 * jnp.pi  # 0.1603
ANHARM_1 = CHI * DELTA**2 / (2 * G**2)  # 1971.3
ANHARM_2 = CHI * DELTA**2 / (2 * G**2 + CHI * DELTA)
KERR = CHI * G**2 / DELTA**2 / 2  # 0.01467
RATIO = 1 / 62.5  # E_C/E_J


def complex_plotter(ts, complex_1, complex_2, rot_1, rot_2, name_1, name_2, fig_name):
    fig, ax = plt.subplots(4, 2, num=fig_name)
    ax[0, 0].plot(
        ts, jnp.absolute(complex_1) ** 2, label=f"{name_1} photons", color="red"
    )
    ax[0, 0].legend()

    ax[1, 0].plot(
        ts, jnp.absolute(complex_2) ** 2, label=f"{name_2} photons", color="blue"
    )
    ax[1, 0].legend()

    ax[2, 0].plot(
        complex_1.real, complex_1.imag, label=f"{name_1} phase", color="orange"
    )
    ax[2, 0].legend()

    ax[3, 0].plot(
        complex_2.real, complex_2.imag, label=f"{name_2} phase", color="green"
    )
    ax[3, 0].legend()

    ax[0, 1].plot(
        ts, jnp.absolute(rot_1) ** 2, label=f"rot {name_1} photons", color="red"
    )
    ax[0, 1].legend()

    ax[1, 1].plot(
        ts, jnp.absolute(rot_2) ** 2, label=f"rot {name_2} photons", color="blue"
    )
    ax[1, 1].legend()

    ax[2, 1].plot(rot_1.real, rot_1.imag, label=f"rot {name_1} phase", color="orange")
    ax[2, 1].legend()

    ax[3, 1].plot(rot_2.real, rot_2.imag, label=f"rot {name_2} phase", color="green")
    ax[3, 1].legend()

    return fig, ax


SQRT_RATIO = jnp.sqrt(2 * RATIO)  # 0.178885
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
ANHARM = ANHARM_1
RES_DRIVE_FREQ = WC.real - 0.5 * CHI - KERR + 0.125 * SQRT_RATIO * CHI
TRANS_DRIVE_FREQ = (
    WQ.real - 0.5 * CHI - ANHARM + 0.25 * SQRT_RATIO * (CHI + KERR + ANHARM)
)
COS_ANGLE = jnp.cos(0.5 * jnp.arctan(2 * G / DELTA))
SIN_ANGLE = jnp.sin(0.5 * jnp.arctan(2 * G / DELTA))


# defining consistent dtypes to be used
float_dtype = jnp.float32
complex_dtype = jnp.complex64

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
ns_per_sample = 20
samples = int(1000.0 / ns_per_sample * (t1 - t0))
ts = jnp.linspace(t0, t1, samples + 1, dtype=float_dtype)

y0 = jnp.array([0.0 + 1j * 0.0, 1.0 + 1j * 0.0], dtype=complex_dtype)

solver = Tsit5()
saveat = SaveAt(ts=ts)
stepsize_controller = PIDController(rtol=1e-8, atol=1e-8, jump_ts=ts)
max_steps = int(1e6)

prev_eval_1 = WC - CHI / 2 - KERR + 0.125 * SQRT_RATIO * CHI
prev_eval_2 = -KERR + 0.5 * SQRT_RATIO * KERR
prev_eval_3 = -CHI + SQRT_RATIO * KERR + 0.5 * SQRT_RATIO * CHI
prev_eval_4 = WQ - CHI / 2 - ANHARM + 0.25 * SQRT_RATIO * (CHI + KERR + ANHARM)
prev_eval_5 = SQRT_RATIO * KERR + 0.5 * SQRT_RATIO * CHI - CHI
prev_eval_6 = 0.5 * SQRT_RATIO * ANHARM + 0.25 * SQRT_RATIO * CHI - ANHARM
print(f"1: {prev_eval_1}")
print(f"2: {prev_eval_2}")
print(f"3: {prev_eval_3}")
print(f"4: {prev_eval_4}")
print(f"5: {prev_eval_5}")
print(f"6: {prev_eval_6}")

prev_eval_1 = 44400.3828125 - 16.762252807617188j
prev_eval_2 = -0.013361622579395771
prev_eval_3 = -9.792070388793945
prev_eval_4 = 30072.25 - 0.1254100799560547j
prev_eval_5 = -9.792069435119629
prev_eval_6 = -1794.5113525390625


def single_eval(
    init_state,
    drive_res,
    drive_trans,
    res_freq,
    trans_freq,
):
    # defining consistent dtypes to be used
    float_dtype = jnp.float32
    complex_dtype = jnp.complex64
    drive_arr = jnp.vstack((drive_res, drive_trans), dtype=complex_dtype).T
    ts = jnp.linspace(0.0, 1.0, 50 + 1, dtype=float_dtype)
    control = LinearInterpolation(ts=ts, ys=drive_arr)

    def vector_field(t, y, args):
        c, q = y
        sqrt_chi_b4 = 0.481
        sqrt_kerr_b2 = 0.001312
        sqrt_anharm_b6 = 58.77
        res_drive_freq = res_freq[0]
        trans_drive_freq = trans_freq[0]
        sin_angle = 0.05202123087967161

        drive_res, drive_trans = control.evaluate(t)

        c_squared = jnp.absolute(c) ** 2
        q_squared = jnp.absolute(q) ** 2
        a = drive_res * jnp.exp(-1j * res_drive_freq * t)
        b = drive_trans * jnp.exp(-1j * trans_drive_freq * t)
        d_c = -1j * (
            (
                44400.3828125
                - 16.762252807617188j
                + c_squared * (-0.013361622579395771 + 2 * sqrt_kerr_b2 * q_squared)
                + q_squared * (-9.792070388793945 + sqrt_chi_b4 * q_squared)
            )
            * c
            + a
            + sin_angle * b
        )
        d_q = -1j * (
            (
                30072.25
                - 0.1254100799560547j
                + c_squared
                * (
                    +sqrt_kerr_b2 * c_squared
                    + 2 * sqrt_chi_b4 * q_squared
                    - 9.792069435119629
                )
                + q_squared * (+sqrt_anharm_b6 * q_squared - 1794.5113525390625)
            )
            * q
            + b
            - sin_angle * a
        )
        return jnp.array([d_c, d_q], dtype=complex_dtype)

    ode_term = ODETerm(vector_field)
    sol = diffeqsolve(
        terms=ode_term,
        solver=Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=0.001,
        y0=init_state,
        saveat=SaveAt(ts=ts),
        max_steps=int(1e6),
    )
    return sol.ys


jitted_single = jit(single_eval)
res = jitted_single(
    init_state=jnp.array([0.0, 0.0], dtype=complex_dtype),
    drive_res=jnp.zeros(51, dtype=complex_dtype) + 31.4,
    drive_trans=jnp.zeros(51, dtype=complex_dtype) + 0.0,
    res_freq=[44400.381208231534],
    trans_freq=[30072.251654918105],
)
start = time.time()
res2 = jitted_single(
    init_state=jnp.array([0.0, 0.0], dtype=complex_dtype),
    drive_res=jnp.zeros(51, dtype=complex_dtype) + 31.4,
    drive_trans=jnp.zeros(51, dtype=complex_dtype) + 0.0,
    res_freq=[44400.381208231534],
    trans_freq=[30072.251654918105],
).block_until_ready()
print(f"time taken for 1: {time.time() - start}")

res_action = jnp.zeros((51, 100), dtype=complex_dtype) + 31.4
trans_action = jnp.zeros((51, 100), dtype=complex_dtype) + 0.0
init_state = jnp.zeros((2, 100), dtype=complex_dtype) + 1.0
res_freq = jnp.zeros((1, 100), dtype=float_dtype) + 44400.381208231534
trans_freq = jnp.zeros((1, 100), dtype=float_dtype) + 30072.251654918105


def single_time(res, trans, state, freq1, freq2):
    for i in range(100):
        _ = jitted_single(
            init_state=state[:, i],
            drive_res=res[:, i],
            drive_trans=trans[:, i],
            res_freq=freq1[:, i],
            trans_freq=freq2[:, i],
        )


start = time.time()
single_time(res_action, trans_action, init_state, res_freq, trans_freq)
now = time.time()
print(f"time taken for single x 100 + jit: {now - start}")
print(f"time taken per: {(now - start) / 100}")

batched_eval = jit(vmap(single_eval, in_axes=1))
_ = batched_eval(init_state, res_action, trans_action, res_freq, trans_freq)

start = time.time()
_ = batched_eval(
    init_state, res_action, trans_action, res_freq, trans_freq
).block_until_ready()
now = time.time()
print(f"time taken for batch of 100 with vmap + jit: {now - start}")
print(f"time taken per: {(now - start) / 100}")

print(_)


class ReadoutEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.state = {}
        self.n_actions = int(1000 * (t1 - t0) / 20) + 1
        # 2 pulses, complex_amplitudes so 2 amps per sample, n_actions num of samples
        # 2 freqs, one for res, one for trans
        # initial state, complex_valued so 2 values
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(2 * self.n_actions + 2 + 1,), dtype=complex_dtype
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
        drive_res = action[0 : self.n_actions]
        drive_trans = action[self.n_actions : 2 * self.n_actions]
        freqs = action[2 * self.n_actions : 2 * self.n_actions + 2]
        init_state = action[-1]
        print(f"drive res: {drive_res}, shape: {drive_res.shape}")
        print(f"drive trans: {drive_trans}, shape: {drive_trans.shape}")
        print(f"freqs: {freqs}, shape: {freqs.shape}")
        print(f"init_state: {init_state}")
        print(init_state.shape)

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

        observation = sol.ys[:, 0]

        # Calculate Reward from Observation
        reward = 0

        done = True

        return observation, reward, done, False, self._get_info()
