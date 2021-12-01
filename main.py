import os
from itertools import chain
import argparse
import ruamel.yaml as yaml
from threading import Thread        # Shift to PyTorch multiprocessing

import numpy as np
import torch
import torch.nn as nn

import gym.spaces as spaces

from agents import DQN
from selection import KMeansSelector
from buffers import VanillaBuffer
from retrieval import RolloutEncoder
from pydreamer.envs.atari import Atari
from models import ConvEncoder
from pydreamer.models.dreamer import Dreamer

import wandb
os.environ["WANDB_SILENT"] = "true"

def collect_img_experience(module, init_state, config, img_buffer):
    state = init_state
    for _ in range(config['img_horizon']):
        action, next_state, reward, done = module.dream(init_state, do_dream_tensors=True)    # Need to fix the dreamer interface
        img_buffer.push(state, action, next_state, reward, done)        # Complete

def encode_img_experience(img_buffer, encoder, code):
    code.append(encoder(img_buffer))



parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Path to config file', type=str, default='./config.yaml')
parser.add_argument('--suite', default='atari', help='Name of the benchmark suite')
parser.add_argument('--env', default='atari_pong', help='Name of the environment')
parser.add_argument('--selector', type=str, default='kmeans', help='Name of the selection process')
parser.add_argument('--forward', default='dreamer', type=str, help='name of the forward module to use')
parser.add_argument('--agent', default='dqn', type=str, help='name of the agent to use')
parser.add_argument('--seed', default=0, help='Random seed')
parser.add_argument('--run', default='', help='Name of this run')

if __name__ == '__main__':
    args = parser.parse_args()
    config = yaml.safe_load(args.config)

    meta = yaml.safe_load(args.config)
    config = {}
    config.update(meta['defaults'], meta[args.suite])
    config[args.selector] = meta[args.selector]
    config[args.forward] = meta[args.forward]
    config[args.agent] = meta[args.agent]
    
    if args.suite == 'atari':
        assert 'atari' in args.env, 'Environment doesn\'t exist in the suite'
        env = Atari(args.env,
                    action_repeat=config['action_repeat'],
                    size=config['size'],
                    grayscale=config['grayscale'],
                    noops=config['noops'],
                    life_done=config['life_done'],
                    sticky_actions=config['sticky_actions'],
                    all_actions=config['all_actions'])
    else:
        raise NotImplementedError

    if args.agent == 'dqn':
        agent = DQN(config[args.agent])      # Make more general - select agent class from args
    else:
        raise NotImplementedError

    wandb.init(project="LI-RARL", name=config['run_name'], config=config)

    replay_buffer = VanillaBuffer(config['buffer_size'])
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

    optimizer = torch.optim.Adam(chain(agent.parameters(), encoder.parameters()), lr=config['learning_rate'])
    total_steps = 0
    for i in range(config['num__train_episodes']):
        obs = env.reset()
        obs = torch.tensor(obs, device=config['device'], dtype=config['dtype'])
        episode_reward = 0
        
        for step in range(config['max_traj_length']):
            total_steps += 1
            obs_enc = encoder(obs)
            exp = replay_buffer.sample(config['buffer_size'])
            exp_enc = encoder(exp.state)

            print('Selecting similar states....')
            selector.fit(exp_enc)
            states = selector.get_similar_states(config['similar'], obs_enc)  # fix return type
            
            states_enc = [encoder(state) for state in states]
            img_buffers = [VanillaBuffer(config['buffer_size']) for _ in range(config['similar'])]
            img_threads = [Thread(collect_img_experience, args=(img_modules[i], states_enc[i], config[args.forward], img_buffers[i])) for i in range(config['similar'])]
            
            print('Training forward module / Imagining trajectories....')
            for i in range(config['similar']): img_threads[i].start()
            for i in range(config['similar']): img_threads[i].join()

            code = []

            rollout_encoders = [RolloutEncoder(config['encoding_size'], config['encoding_size']) for _ in range(config['similar'])]
            summ_threads = [Thread(encode_img_experience, args=(img_buffers[i], rollout_encoders[i], code)) for i in range(config['similar'])]

            for i in range(config['similar']): summ_threads[i].start()
            for i in range(config['similar']): summ_threads[i].join()

            print('Aggregating imagination encodings....')
            img_code = torch.cat(code, dim=1)

            print(f'Running the agent process....')
            in_agent = torch.cat([obs_enc, img_code], dim=1)

            action = agent.select_action(in_agent)
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            next_obs = torch.tensor(next_obs, device=config['device'], dtype=config['dtype'])
            reward = torch.tensor(reward, device=config['device'], dtype=config['dtype'])
            done = torch.tensor(done, device=config['device'], dtype=config['dtype'])

            replay_buffer.push(obs, action, reward, next_obs, done)

            sample = replay_buffer.sample(config['batch_size'])
            target_q_val, q_val = agent(sample)                   # Make agent agnostic
            loss = nn.MSELoss()(target_q_val, q_val)   
            loss.backward()
            optimizer.step()

            if total_steps % config['trg_update_freq'] == 0:
                agent.update_target_network()

            if done.item():
                print(f'Episode: {i} / Reward collected: {episode_reward} / No. of {steps}: {step}')
                break

        if i % config['model_save_freq'] == 0:
            agent.save_model(config['model_save_path'])

    env.close()