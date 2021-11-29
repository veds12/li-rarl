import os
from itertools import chain
import argparse
import ruamel.yaml as yaml

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as model

import gym.spaces as spaces

from agents import DQN
from selection import KMeansSelector
from buffers import VanillaReplayBuffer
from pydreamer.envs.atari import Atari

import wandb
os.environ["WANDB_SILENT"] = "true"

parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Path to config file', type=str, default='./config.yaml')
parser.add_argument('--suite', default='atari', help='Name of the benchmark suite')
parser.add_argument('--env', default='atari_pong', help='Name of the environment')
parser.add_argument('--seed', default=0, help='Random seed')
parser.add_argument('--run', default='', help='Name of this run')

if __name__ == '__main__':
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    suite = args.suite

    meta = yaml.safe_load(args.config)
    config = {}
    config.update(meta['defaults'], meta[suite])
    
    if suite == 'atari':
        assert 'atari' in args.env, 'Environment doesn\'t exist in the suite'
        env = Atari(args.env,
                    action_repeat=config['action_repeat'],
                    size=config['size'],
                    grayscale=config['grayscale'],
                    noops=config['noops'],
                    life_done=config['life_done'],
                    sticky_actions=config['sticky_actions'],
                    all_actions=config['all_actions'])

    elif suite == 'dmlab':
        pass # TODO

    elif suite == 'dmc':
        pass # TODO

    wandb.init(project="LI-RARL", name=config['run_name'], config=config)

    replay_buffer = VanillaReplayBuffer(config['buffer_size'])

    if config['raw_data'] is not None:
        print('Loading offline data into buffer....')
        replay_buffer.load(config['offline_trainfile'])

    if config['prefill']:
        print('Collecting raw experience....')
        steps = 0
        while True:
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_obs, reward, done, _ = env.step(action)
                replay_buffer.push(obs, action, reward, next_obs, done)
                obs = next_obs
                steps += 1
                if steps == config['buffer_size']:
                    print(f'{steps} steps prefilled')
                    break
    
    

    
        




    


    

    

