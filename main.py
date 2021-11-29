import os
from itertools import chain
import argparse
import ruamel.yaml as yaml
from threading import Thread

import numpy as np
import torch
import torch.nn as nn

import gym.spaces as spaces

from agents import DQN
from selection import KMeansSelector
from buffers import VanillaReplayBuffer
from pydreamer.envs.atari import Atari
from models import ConvEncoder
from pydreamer.models.dreamer import Dreamer

import wandb
os.environ["WANDB_SILENT"] = "true"

def collect_img_experience(module, init_state, config, img_trajs):
    _, _, _, _, dream_tensors = module.traning_step(init_state, do_dream_tensors=True)    # Need to fix the dreamer interface
    img_trajs.append(dream_tensors)         # Store in a buffer instead?

parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Path to config file', type=str, default='./config.yaml')
parser.add_argument('--suite', default='atari', help='Name of the benchmark suite')
parser.add_argument('--env', default='atari_pong', help='Name of the environment')
parser.add_argument('--selector', type=str, default='kmeans', help='Name of the selectoion process')
parser.add_argument('--seed', default=0, help='Random seed')
parser.add_argument('--run', default='', help='Name of this run')
parser.add_argument('--forward', default='dreamer', type=str, help='name of the forward module to use')

if __name__ == '__main__':
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    suite = args.suite

    meta = yaml.safe_load(args.config)
    config = {}
    config.update(meta['defaults'], meta[suite])
    config[args.selector] = meta[args.selector]
    config[args.forward] = meta[args.forward]
    
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
        raise NotImplementedError # TODO

    elif suite == 'dmc':
        raise NotImplementedError # TODO

    wandb.init(project="LI-RARL", name=config['run_name'], config=config)

    replay_buffer = VanillaReplayBuffer(config['buffer_size'])
    encoder = ConvEncoder(env.observation_space.shape, out_size=config['encoding_size'])

    if config['raw_data'] is not None:
        print('Loading offline data into buffer....')
        replay_buffer.load(config['offline_trainfile'])

    if config['prefill']:
        print('Collecting raw experience....')
        steps = 0
        done = True
        while True: 
            if done:
                obs = env.reset()
                done = False

            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            steps += 1
            if steps == config['buffer_size']:
                print(f'{steps} steps prefilled')
                break

    if args.selector == 'kmeans':
        selector = KMeansSelector(config['kmeans'])
    
    if args.forward == 'dreamer':
        img_modules = [Dreamer(config) for _ in range(config['similar'])]
    
    for cycle in range(config['cycles']):
        print(f'Cycle {cycle}')
        obs = env.reset()

        obs_enc = encoder(obs)
        exp = replay_buffer.sample(config['buffer_size'])
        exp_enc = encoder(exp.state)

        print('Selecting similar states....')
        selector.fit(exp_enc)
        states = selector.get_similar_states(config['similar'], obs_enc)  # fix return type
        
        states_enc = [encoder(state) for state in states]
        img_trajs = []
        threads = [Thread(collect_img_experience, args=(img_modules[i], states_enc[i], config[args.forward], img_trajs)) for i in range(config['similar'])]
        
        print('Training forward module / Imagining trajectories....')
        for i in range(config['similar']): threads[i].start()
        for i in range(config['similar']): threads[i].join()











    


