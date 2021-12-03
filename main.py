import os
from itertools import chain
import argparse
import yaml
from threading import Thread        # Shift to PyTorch multiprocessing
import pprint
import random

import numpy as np
import torch
import torch.nn as nn

import gym.spaces as spaces

from agents import get_agent
from selection import get_selector
from forward import get_forward_module
from buffers import VanillaBuffer
from pydreamer.envs import create_env
from models import ConvEncoder, RolloutEncoder

import wandb
os.environ["WANDB_SILENT"] = "true"

def collect_img_experience(module, seq, config, img_buffer, mode):
    obs =  {
        'image': torch.tensor(seq.obs, dtype=dtype, device=device),
        'action': torch.tensor(seq.action dtype=dtype, device=device),
        'reward': torch.tensor(seq.reward, dtype=dtype, device=device),
        'terminal': torch.tensor(seq.done, dtype=dtype, device=device),
        'reset': torch.tensor(seq.reset, dtype=dtype, device=device),
        }

    if mode == 'train':
        # not keeping states for now (config["keep_states"] = False)
        # should be module.init_state(config["batch_size"] * config["iwae_samples"]); 
        # batch_size = 1 in this case (for I2A style summarization)
        state = module.init_state(1 * config["iwae_samples"])
        losses, new_state, loss_metrics, tensors, dream_tensors = \
                            module.training_step(obs,
                                                state,                        # What is state?
                                                config,
                                                do_image_pred=False,
                                                do_dream_tensors=True)

        pprint.pprint(dream_tensors)
        
    #img_buffer.push(state, action, next_state, reward, done, )        # Complete

def encode_img_experience(img_buffer, encoder, code):
    code.append(encoder(img_buffer))

parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Path to config file', type=str, default='./config.yaml')
parser.add_argument('--suite', default='atari', help='Name of the benchmark suite')
parser.add_argument('--env', default='atari_pong', help='Name of the environment')
parser.add_argument('--selector', type=str, default='kmeans', help='Name of the selection process')
parser.add_argument('--forward', default='dreamer', type=str, help='name of the forward module to use')
parser.add_argument('--agent', default='dqn', type=str, help='name of the agent to use')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--run', default='', help='Name of this run')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    config = {**meta['general'], **meta[args.forward], **meta[args.agent], **meta[args.selector]}
    config.update(meta[args.suite])
    del meta

    dtype = torch.float32           # TODO: Set acc to config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    env = create_env(args.env, config)

    config['action_dim'] = env.action_space.n
    config['obs_dim'] = env.observation_space['image'].shape[0]
    config['action_space_type'] = env.action_space.__class__.__name__
    config['image_channels'] = env.observation_space['image'].shape[-1]
    config['image_size'] = env.observation_space['image'].shape[1]

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
        ep_steps = 0
        num_eps = 0
        prefill_obs = env.reset()
        while True:
            ep_steps += 1
            prefill_action = env.action_space.sample()
            prefill_next_obs, prefill_reward, prefill_done, prefill_info = env.step(prefill_action)
            prefill_obs = prefill_next_obs
            prefill_steps += 1
            if prefill_done:
                prefill_imgn_code = np.zeros((ep_steps+1, config["rollout_enc_size"]))
                replay_buffer.push(*list(prefill_info['episode'].values()), prefill_imgn_code)
                prefill_obs = env.reset()
                prefill_done = False
                num_eps += 1
                print(f'\tEpisode {num_eps} \ Number of steps: {ep_steps} \ Total steps added: {prefill_steps}')
                ep_steps = 0

            if prefill_steps >= config['prefill']:
                print(f'{prefill_steps} steps added to buffer')
                break
    
    agent = agent_type(config)
    selector = selector_type(config)
    forward_modules = [forward_type(config) for _ in range(config['similar'])]    # TODO: Use the same encoder for dreamer and env obs
    encoder = ConvEncoder(config).to(device).to(dtype)

    optimizer = torch.optim.Adam(chain(agent.parameters(), encoder.parameters()), lr=config['learning_rate'])
    total_steps = 0
    for i in range(config['num_train_episodes']):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(config['max_traj_length']):
            total_steps += 1
            proc_obs = torch.tensor(obs['image'], dtype=dtype, device=device).unsqueeze(0)
            obs_enc = encoder(proc_obs)
            sampled_seq = replay_buffer.sample_sequences(seq_len=config['seq_length'], n_sam_eps=config['n_sam_eps'], reset_interval=config["reset_interval"], skip_random=True)  # skip_random=True for training
            obsrvs = torch.cat([torch.tensor(sampled_seq[i].obs[0], dtype=dtype, device=device).unsqueeze(0) for i in range(len(sampled_seq))])
            obsrvs_enc = encoder(obsrvs)  

            #print('Selecting similar states....')
            selector.fit(obsrvs_enc)
            selected_seqs = selector.get_similar_seqs(config['similar'], obs_enc, sampled_seq)
            #print(selected_seqs[0]._fields)
            
            imgn_buffers = [VanillaBuffer(config) for _ in range(config['similar'])]
            imgn_threads = [Thread(target=collect_img_experience, args=(forward_modules[i], selected_seqs[i], config, imgn_buffers[i], 'train')) for i in range(config['similar'])]
            
            #print('Training forward module / Imagining trajectories....')
            for i in range(config['similar']): imgn_threads[i].start()
            for i in range(config['similar']): imgn_threads[i].join()

            code = []

            rollout_encoders = [RolloutEncoder(config) for _ in range(config['similar'])]
            summ_threads = [Thread(target=encode_img_experience, args=(imgn_buffers[i], rollout_encoders[i], code)) for i in range(config['similar'])]

            for i in range(config['similar']): summ_threads[i].start()
            for i in range(config['similar']): summ_threads[i].join()

            #print('Aggregating imagination encodings....')
            imgn_code = torch.cat(code, dim=1)

            #print(f'Running the agent process....')
            agent_in = torch.cat([obs_enc, imgn_code], dim=1)

            action = agent.select_action(agent_in)
            next_obs, reward, done, _ = env.step(action.item())
            episode_reward += reward

            proc_next_obs = torch.tensor(next_obs['image'], device=device, dtype=dtype).unsqueeze(0)
            proc_reward = torch.tensor([reward], device=device, dtype=dtype).unsqueeze(0)
            proc_action = torch.tensor(next_obs['action'], device=device, dtype=dtype).unsqueeze(0)  # make one hot
            proc_done = torch.tensor([bool(done)], device=device, dtype=dtype).unsqueeze(0)
            reset = torch.tensor([bool(next_obs['reset'])], device=device, dtype=dtype).unsqueeze(0)

            replay_buffer.push(proc_obs, proc_action, proc_reward, proc_next_obs, proc_done, reset, imgn_code)  # Fix input dtypes and add reset

            sample = replay_buffer.sample(config['dqn_batch_size'])
            obs_enc_update = encoder(sample.next_obs)
            input = torch.cat((obs_enc_update, sample.imgn_code), dim=1)
            print(f'Input shape is {input.shape}')
            target_q_val, q_val = agent(input)                   # Make agent agnostic
            loss = nn.MSELoss()(target_q_val, q_val)   
            loss.backward()
            optimizer.step()

            if total_steps % config['trg_update_freq'] == 0:
                agent.update_target_network()

            if done.item():
                print(f'Episode: {i} / Reward collected: {episode_reward} / No. of steps: {step}')
                break

        if i % config['model_save_freq'] == 0:
            agent.save_model(config['model_savepath'])

        if i % config['write_data_freq'] == 0:
            replay_buffer.save(config['offline_trainfile'])

    env.close()