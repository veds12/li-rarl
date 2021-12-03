# Ignore annoying warnings from imported envs
import warnings
warnings.filterwarnings("ignore", ".*Box bound precision lowered by casting")  # gym

import gym
import numpy as np

from .wrappers import *


def create_env(env_id, config):

    if env_id.startswith('MiniGrid-'):
        from .minigrid import MiniGrid
        env = MiniGrid(env_id)

    elif env_id.startswith('Atari-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=config["action_repeat"])

    elif env_id.startswith('AtariGray-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=config["action_repeat"], grayscale=config["grayscale"])

    elif env_id.startswith('MiniWorld-'):
        import gym_miniworld.wrappers as wrap
        env = gym.make(env_id)
        env = wrap.DictWrapper(env)
        env = wrap.MapWrapper(env)
        # env = wrap.PixelMapWrapper(env)
        env = wrap.AgentPosWrapper(env)

    elif env_id.startswith('DmLab-'):
        from .dmlab import DmLab
        env = DmLab(env_id.split('-')[1].lower(), num_action_repeats=config["action_repeat"])
        env = DictWrapper(env)

    elif env_id.startswith('MineRL'):
        from .minerl import MineRL
        env = MineRL(env_id, np.load('data/minerl_action_centroids.npy'), action_repeat=config["action_repeat"])

    elif env_id.startswith('DMC-'):
        from .dmc import DMC
        env = DMC(env_id.split('-')[1].lower(), action_repeat=config["action_repeat"])

    else:
        env = gym.make(env_id)
        env = DictWrapper(env)

    if hasattr(env.action_space, 'n'):
        env = OneHotActionWrapper(env)
    if config["time_limit"] > 0:
        env = TimeLimitWrapper(env, config["time_limit"])
    env = ActionRewardResetWrapper(env, config["no_terminal"])
    env = CollectWrapper(env)
    return env
