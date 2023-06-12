import numpy as np
from gymnasium_env import ReadoutEnv

env = ReadoutEnv()
obs, info = env.reset()
action = np.arange(0.0, env.action_space.shape[0], 1)
print(action)
next_obs, reward, done, _, info = env.step(action)
print(next_obs)
