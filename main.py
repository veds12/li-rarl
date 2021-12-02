import os
from itertools import chain
import argparse
import yaml
from threading import Thread        # Shift to PyTorch multiprocessing

import numpy as np
import torch
import torch.nn as nn

import gym.spaces as spaces

from agents import get_agent
from selection import get_selector
from forward import get_forward_module
from buffers import VanillaBuffer
from pydreamer.envs.atari import Atari
from models import ConvEncoder, RolloutEncoder

import wandb
os.environ["WANDB_SILENT"] = "true"

def collect_img_experience(module, init_state, config, img_buffer, mode):
    state = init_state

    if mode == 'train':
        optimizers = module.init_optimizers(config["adam_lr"], config["adam_lr_actor"], config["adam_lr_critic"], config["adam_eps"])
    
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
    with open(args.config, 'r') as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    config = {**meta['general'], **meta[args.suite], **meta[args.forward], **meta[args.agent], **meta[args.selector]}
    del meta
    
    if args.suite == 'atari':
        assert 'atari' in args.env, 'Environment doesn\'t belong to the suite specified'
        env = Atari(args.env[6:], config)
    else:
        raise NotImplementedError

    config['action_dim'] = env.action_space.n
    config['obs_dim'] = env.observation_space['image'].shape[0]
    config['action_space_type'] = env.action_space.__class__.__name__

    agent_type = get_agent(args.agent)
    selector_type = get_selector(args.selector)
    forward_type = get_forward_module(args.forward)

    wandb.init(project="LI-RARL", name=config['run_name'], config=config)

    replay_buffer = VanillaBuffer(config)

    if config['raw_data']:
        print('Loading offline data into buffer....')
        replay_buffer.load(config['raw_data'])         # Implement

    if config['prefill'] is not None:
        print('\nCollecting raw experience....')
        prefill_steps = 0
        prefill_done = True
        while True: 
            if prefill_done:
                prefill_obs = env.reset()
                prefill_done = False

            prefill_action = env.action_space.sample()
            prefill_next_obs, prefill_reward, prefill_done, _ = env.step(prefill_action)
            replay_buffer.push(prefill_obs, prefill_action, prefill_reward, prefill_next_obs, prefill_done)
            prefill_obs = prefill_next_obs
            prefill_steps += 1
            if prefill_steps >= config['prefill']:
                print(f'{prefill_steps} steps added to buffer')
                break
    
    agent = agent_type(env.action_space, config)
    selector = selector_type(config)
    forward_modules = [forward_type(config) for _ in range(config['similar'])]    # See that dreamer doesn't use a separate encoder
    encoder = ConvEncoder(config)

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

            #print('Selecting similar states....')
            selector.fit(exp_enc)
            states = selector.get_similar_states(config['similar'], obs_enc)  # fix return type
            
            states_enc = [encoder(state) for state in states]
            img_buffers = [VanillaBuffer(config['buffer_size']) for _ in range(config['similar'])]
            img_threads = [Thread(collect_img_experience, args=(forward_modules[i], states_enc[i], config, img_buffers[i], 'train')) for i in range(config['similar'])]
            
            #print('Training forward module / Imagining trajectories....')
            for i in range(config['similar']): img_threads[i].start()
            for i in range(config['similar']): img_threads[i].join()

            code = []

            rollout_encoders = [RolloutEncoder(config['encoding_size'], config['encoding_size']) for _ in range(config['similar'])]
            summ_threads = [Thread(encode_img_experience, args=(img_buffers[i], rollout_encoders[i], code)) for i in range(config['similar'])]

            for i in range(config['similar']): summ_threads[i].start()
            for i in range(config['similar']): summ_threads[i].join()

            #print('Aggregating imagination encodings....')
            img_code = torch.cat(code, dim=1)

            #print(f'Running the agent process....')
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
                print(f'Episode: {i} / Reward collected: {episode_reward} / No. of steps: {step}')
                break

        if i % config['model_save_freq'] == 0:
            agent.save_model(config['model_save_path'])

        if i % config['write_data_freq'] == 0:
            replay_buffer.save(config['offline_trainfile'])

    env.close()