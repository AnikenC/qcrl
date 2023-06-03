import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from .quantum_env import QuantumEnvironment


class QEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.q_environment = QuantumEnvironment()
        self.n_actions = 7
        self.n_qubits = 2
        self.N_in = self.n_qubits + 1
        self.average_fidelity = 0.0
        self.mean_rewards = 0.0
        self.max_reward = 0.0
        self.max_fidelity = 0.0
        self.index = 0.0

        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
        )
        self.observation_space = Box(
            low=-1e2, high=1e2, shape=(self.N_in,), dtype=np.float32
        )

    def _get_obs(self):
        observation = np.zeros((self.N_in,), dtype=np.float32)
        return observation

    def _get_info(self):
        info = {
            "average fidelity": self.average_fidelity,
            "mean rewards": self.mean_rewards,
            "max reward": self.max_reward,
            "max fidelity": self.max_fidelity,
        }
        return info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        if len(action.shape) == 1:
            action = np.reshape(action, (1, len(action)))
        reward, self.average_fidelity, self.index = self.q_environment.perform_action(
            action
        )
        self.mean_rewards = np.mean(reward)
        if self.max_reward < self.mean_rewards:
            self.max_reward = self.mean_rewards

        if self.max_fidelity < self.average_fidelity:
            self.max_fidelity = self.average_fidelity

        observation = self._get_obs()
        info = self._get_info()
        terminated = True

        return observation, reward, terminated, False, info
