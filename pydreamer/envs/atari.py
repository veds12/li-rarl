import threading

import gym
import gym.envs.atari
import gym.wrappers
import numpy as np


class Atari(gym.Env):

    LOCK = threading.Lock()

    def __init__(self, name, config):
        assert config["frame_size"][0] == config["frame_size"][1]
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                game=name,
                obs_type='image',
                frameskip=1,
                repeat_action_probability=0.25 if config["sticky_actions"] else 0.0,
                full_action_space=config["all_actions"])
        # Avoid unnecessary rendering in inner env.
        env.get_obs = lambda: None  # type: ignore
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')  # type: ignore
        env = gym.wrappers.AtariPreprocessing(env, config["noops"], config["action_repeat"], config["frame_size"][0], config["life_done"], config["grayscale"])
        self.env = env
        self.grayscale = config["grayscale"]

    @property
    def observation_space(self):
        return gym.spaces.Dict({'image': self.env.observation_space})  # type: ignore

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        with self.LOCK:
            image: np.ndarray = self.env.reset()  # type: ignore
        if self.grayscale:
            image = image[..., None]
        obs = {'image': image}
        return obs

    def step(self, action):
        image, reward, done, info = self.env.step(action)
        if self.grayscale:
            image = image[..., None]
        obs = {'image': image}
        return obs, reward, done, info

    def render(self, mode):
        return self.env.render(mode)
