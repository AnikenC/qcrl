from gate_calibration.rl_envs import QEnv


def check_env(env_name):
    supported = False
    env = None
    supported_envs = {
        "gate_calibration": QEnv(),
    }
    if env_name in supported_envs:
        supported = True
        env = supported_envs[env_name]

    return supported, env
