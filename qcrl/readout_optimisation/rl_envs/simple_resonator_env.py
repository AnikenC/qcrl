import warnings

warnings.filterwarnings("ignore")

import time
import matplotlib.pyplot as plt

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, config, vmap, block_until_ready
from diffrax import (
    diffeqsolve,
    Tsit5,
    LinearInterpolation,
    ODETerm,
    SaveAt,
    PIDController,
)

import gymnasium as gym
from gymnasium.spaces import Box

# from updated_utils import (
#    drive_plotter,
#    simple_plotter_debug,
# )

from updated_utils import drive_plotter, simple_plotter_debug, simple_plotter

config.update("jax_enable_x64", True)


class SimpleResonatorEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.float_dtype = jnp.float64
        self.complex_dtype = jnp.complex128
        self.n_actions = 81
        self.n_sim = 201
        self.batch_size = 512
        self.state = jnp.zeros((2 * 2 * (self.n_sim),), dtype=self.float_dtype)
        self.t0 = 0.0
        self.t1 = 10.0
        self.ts_action = jnp.linspace(
            self.t0, self.t1, self.n_actions, dtype=self.float_dtype
        )
        self.ts_sim = jnp.linspace(self.t0, self.t1, self.n_sim, dtype=self.float_dtype)
        self.chi_eff = 2.0
        self.kerr_eff = 0.0005
        self.neg_kappa_half = -0.5 * self.chi_eff
        self.gamma_eff = 1 / 500.0
        self.args = jnp.array(
            [self.neg_kappa_half, self.gamma_eff, self.chi_eff, self.kerr_eff],
            dtype=self.complex_dtype,
        )
        self.max_photons = 8.0
        self.separation_factor = 1.0
        self.photon_penalty = 50.0
        self.init_obs = np.zeros(1, dtype=np.float64)
        self.batched_default_obs = np.zeros((self.batch_size, 1), dtype=np.float64)
        self.max_drive_amplitude = 10.0
        self.max_init_state_amplitude = 5.0
        self.mean_reward = 0.0
        self.mean_max_photon = 0.0
        self.mean_max_separation = 0.0
        self.max_reward = -1000.0
        self.max_separation = -self.photon_penalty
        self.separation_at_max_reward = -self.photon_penalty
        self.photon_at_max_reward = -self.photon_penalty
        self.action_at_max_reward = 0.0
        self.res_state_at_max_reward = 0.0
        self.bandwidth_at_max_reward = 0.0

        self.threshold = 0.15
        self.max_bandwith = 1.5
        self.bandwidth_penalty = 50.0

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actions + 1,),
            dtype=self.float_dtype,
        )
        self.observation_space = Box(
            low=-10.0, high=10.0, shape=(1,), dtype=self.float_dtype
        )

        self.pre_compile(time_speed=True)

    def _get_info(self):
        return {
            "mean reward": self.mean_reward,
            "mean max photon": self.mean_max_photon,
            "mean max separation": self.mean_max_separation,
            "max reward": self.max_reward,
            "separation at max reward": self.separation_at_max_reward,
            "photon at max reward": self.photon_at_max_reward,
            "bandwidth at max reward": self.bandwidth_at_max_reward,
            "action at max reward": self.action_at_max_reward,
            "res state at max reward": self.res_state_at_max_reward,
        }

    def generate_resonator_state(self):
        batched_resonator_state = jnp.zeros(
            (self.batch_size, 1), dtype=self.complex_dtype
        )

        return batched_resonator_state

    def generate_batched_gaussian(self):
        res_drive_amp = 5.0
        gaussian_mean = 0.5 * self.t1
        gaussian_std = 0.25 * self.t1

        single_gaussian_complex = jnp.asarray(
            (
                res_drive_amp
                * jnp.exp(
                    -((self.ts_action - gaussian_mean) ** 2) / (2 * gaussian_std**2)
                )
            ),
            dtype=self.float_dtype,
        ).reshape(1, self.n_actions)

        batched_gaussian_complex = single_gaussian_complex

        for i in range(self.batch_size - 1):
            batched_gaussian_complex = jnp.concatenate(
                (batched_gaussian_complex, single_gaussian_complex),
                axis=0,
                dtype=self.float_dtype,
            )

        return batched_gaussian_complex

    def pre_compile(self, time_speed=True):
        batched_gaussian_pulse = self.generate_batched_gaussian()
        batched_resonator_state = self.generate_resonator_state()
        self.batched_eval = jit(vmap(self.single_eval, in_axes=0))
        self.batched_reward_func = jit(self.calculate_reward_and_stats)
        a_, x_ = self.batched_eval(batched_gaussian_pulse, batched_resonator_state)
        b_, c_, f_, g_, h_, i_, k_, n_, t = self.batched_reward_func(a_, x_)
        if time_speed:
            start = time.time()
            results, batched_bandwidths = block_until_ready(
                self.batched_eval(batched_gaussian_pulse, batched_resonator_state)
            )
            reward, a, c, d, e, f, k, h, t = block_until_ready(
                self.batched_reward_func(results, batched_bandwidths)
            )
            time_taken = (time.time() - start) * 1e6
            print(
                f"time taken for batch of {self.batch_size} simulations + calculations: {time_taken}us"
            )
            print(
                f"time taken per simulation of g + e and reward calculation: {time_taken/self.batch_size}us"
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.init_obs, self._get_info()

    def step(self, action):
        # Receives a 2D array of Actions since they are batched on the 0th axis
        batched_drive_res = self.max_drive_amplitude * jnp.array(
            action[:, 0 : self.n_actions],
            dtype=self.float_dtype,
        )
        batched_res_state = (
            1.0j
            * self.max_init_state_amplitude
            * jnp.array(
                action[:, -1].reshape(self.batch_size, 1),
                dtype=self.float_dtype,
            )
        )

        # 2D array with the g and e results, of shape (self.batch_size, self.n_sim, 2)
        results, batched_bandwidths = self.batched_eval(
            batched_drive_res, batched_res_state
        )

        (
            reward,
            self.mean_reward,
            self.mean_max_photon,
            self.mean_max_separation,
            max_reward_in_batch,
            separation_at_max,
            photon_at_max,
            bandwidth_at_max,
            max_reward_index,
        ) = self.batched_reward_func(results, batched_bandwidths)

        if max_reward_in_batch > self.max_reward:
            self.max_reward = max_reward_in_batch
            self.separation_at_max_reward = separation_at_max
            self.photon_at_max_reward = photon_at_max
            self.bandwidth_at_max_reward = bandwidth_at_max
            self.action_at_max_reward = jnp.array_repr(
                batched_drive_res[max_reward_index]
            )
            self.res_state_at_max_reward = jnp.array_repr(
                batched_res_state[max_reward_index]
            )

        return (
            self.batched_default_obs,
            np.asarray(reward),
            True,
            False,
            self._get_info(),
        )

    def single_eval(self, res_drive, res_state):
        control = LinearInterpolation(ts=self.ts_action, ys=res_drive)

        y0 = jnp.array([*res_state, *res_state], dtype=self.complex_dtype)
        solver = Tsit5()
        dt0 = 1e-3
        saveat = SaveAt(ts=self.ts_sim)
        stepsize_controller = PIDController(
            rtol=1e-4,
            atol=1e-7,
            pcoeff=0.4,
            dcoeff=0.3,
            icoeff=0.0,
            jump_ts=self.ts_action,
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
                * (
                    1.0j * wc_eff_imag
                    + 0.5 * chi_eff
                    - kerr_eff * jnp.absolute(res_g) ** 2
                )
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
            return jnp.array([d_res_g, d_res_e], dtype=self.complex_dtype)

        ode_term = ODETerm(vector_field)
        sol = diffeqsolve(
            terms=ode_term,
            solver=solver,
            t0=self.t0,
            t1=self.t1,
            dt0=dt0,
            y0=y0,
            args=self.args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )

        freqs = jnp.fft.fftfreq(self.n_actions, self.t1 / (self.n_actions - 1))
        rfft_vals = jnp.fft.rfft(res_drive)
        rfft_shifted = jnp.abs(jnp.fft.fftshift(rfft_vals))
        indices = jnp.where(
            rfft_shifted > self.threshold * jnp.max(rfft_shifted), size=self.n_actions
        )[0]
        min_index = indices[0]
        max_index = jnp.max(indices)
        bandwidth = jnp.array(
            [freqs[max_index] - freqs[min_index]], dtype=self.float_dtype
        )

        return sol.ys, bandwidth

    def calculate_reward_and_stats(self, results, batched_bandwidths):
        """
        Takes in an input array of shape (batch_size, num_actions, 2)
        where 2 is due to the different g and e results

        This function calculates the reward, and reformats the results to get the final observations
        """

        total_photons = jnp.absolute(results) ** 2
        batched_max_photons_in_each = jnp.max(total_photons, axis=1)
        batched_max_photons_in_ge = jnp.max(batched_max_photons_in_each, axis=-1)

        results_g = results[:, :, 0]
        results_e = results[:, :, 1]

        batched_separation = jnp.absolute(results_g - results_e)
        max_separation = jnp.max(batched_separation, axis=-1)

        sign = jnp.sign(self.max_photons - batched_max_photons_in_ge) - 1.0

        sign_bandwidth = jnp.sign(self.max_bandwith - batched_bandwidths) - 1.0

        batched_reward = (
            self.separation_factor * max_separation
            + self.photon_penalty * sign
            + self.bandwidth_penalty * sign_bandwidth
        )

        max_reward = jnp.max(batched_reward)

        max_reward_index = jnp.where(batched_reward == max_reward, size=1)[0]

        separation_at_max_reward = max_separation[max_reward_index]
        photon_at_max_reward = batched_max_photons_in_ge[max_reward_index]
        bandwidth_at_max_reward = batched_bandwidths[max_reward_index]

        mean_batch_reward = jnp.mean(batched_reward)
        mean_max_photon = jnp.mean(batched_max_photons_in_ge)
        mean_max_separation = jnp.mean(max_separation)

        return (
            batched_reward,
            mean_batch_reward,
            mean_max_photon,
            mean_max_separation,
            max_reward,
            separation_at_max_reward,
            photon_at_max_reward,
            bandwidth_at_max_reward,
            max_reward_index,
        )

    def render(self, res_drive, res_state, index=0):
        """
        res_drive should be in batched format, res_state should also be in batched format
        """
        results, bandwidths = self.batched_eval(res_drive, res_state)
        (
            reward,
            self.mean_reward,
            self.mean_max_photon,
            self.mean_max_separation,
            max_reward_in_batch,
            separation_at_max,
            photon_at_max,
            bandwith_at_max,
            max_reward_index,
        ) = self.batched_reward_func(results, bandwidths)

        plotting_results = results[index]
        results_g = plotting_results[:, 0]
        results_e = plotting_results[:, 1]

        print(f"Maximum Reward: {max_reward_in_batch}")
        print(f"Separation at Max Reward: {separation_at_max}")
        print(f"Photon at Max: {photon_at_max}")
        print(f"Bandwidth at Max: {bandwith_at_max}")

        fig1, ax1 = simple_plotter_debug(
            ts=self.ts_sim,
            complex_1=results_g,
            complex_2=results_e,
            name_1="Ground",
            name_2="Excited",
            fig_name="Custom Action Render",
        )

        fig2, ax2 = drive_plotter(
            ts=self.ts_action,
            complex_drive=res_drive[index],
            fig_name="Custom Drive",
        )

        separation = jnp.abs(results_g - results_e)
        max_separation = jnp.max(separation)
        max_sep_index = jnp.where(separation == max_separation, size=1)[0]
        time_of_sep = self.ts_sim[max_sep_index]
        fig3, ax3 = plt.subplots(1, num=f"Separation, max: {max_separation}")
        ax3.plot(
            self.ts_sim,
            jnp.abs(results_g - results_e),
            label=f"Separation at time: {time_of_sep}",
            color="red",
        )
        ax3.legend()

        plt.show()

    def get_params(self):
        return {
            "t0": self.t0,
            "t1": self.t1,
            "num_actions": self.n_actions,
            "num_sim": self.n_sim,
            "ts_action": self.ts_action,
            "ts_sim": self.ts_sim,
            "batch_size": self.batch_size,
        }
