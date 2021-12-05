import os
from itertools import chain
import argparse
import yaml
from threading import Thread        # Shift to PyTorch multiprocessing
import pprint
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import gym.spaces as spaces

from agents import get_agent
from selection import get_selector
from forward import get_forward_module
from buffers import SequenceBuffer, TransitionBuffer
from pydreamer.envs import create_env
from models import ConvEncoder, RolloutEncoder
from utils import *

import wandb
os.environ["WANDB_SILENT"] = "true"

def collect_img_experience(module, seq, config, dreams, mode, i):
    # TODO: 
    #   add options for categorical images
    #   convert to torch directly

    image = seq.obs / 255.0 - 0.5                        
    image = image.transpose(0, 3, 1, 2)
    image = np.expand_dims(image, axis=1).astype(np.float32)
    
    action = np.expand_dims(seq.action, axis=1)
    if len(action) == 2:
        action = to_onehot(seq.action, config["action_dim"]).astype(np.float32)
        print(action.shape)
    else:
        action = action.astype(np.float32)
    assert len(action.shape) == 3 
    
    if seq.done is not None:
        terminal = np.expand_dims(seq.done, axis=1).astype(np.float32)
    else:
        terminal = np.zeros((config["seq_lenghth"], 1)).astype(np.float32)
    
    if seq.reward is not None:
        reward = np.expand_dims(seq.reward, axis=1).astype(np.float32)
    else:
        np.zeros((config["seq_length"], 1)).astype(np.float32)

    if config["clip_rewards"] == 'tanh':
        reward = np.tanh(reward)  # type: ignore
    if config["clip_rewards"] == 'log1p':
        reward = np.log1p(reward)  # type: ignore
    
    reset = np.expand_dims(seq.reset, axis=1).astype(bool)
    vecobs = np.zeros((config["seq_length"], 1, 64)).astype(np.float32)
        
    obs =  {
        'image': torch.from_numpy(image).to(device),
        'action': torch.from_numpy(action).to(device),
        'reward': torch.from_numpy(reward).to(device),
        'terminal': torch.from_numpy(terminal).to(device),
        'reset': torch.from_numpy(reset).to(device),
        'vecobs': torch.from_numpy(vecobs).to(device),
        }

    #print(obs['image'].shape, obs['action'].shape, obs['reward'].shape, obs['terminal'].shape, obs['reset'].shape)

    if mode == 'train':
        # not keeping states for now (config["keep_states"] = False)
        # should be module.init_state(config["batch_size"] * config["iwae_samples"]); 
        # batch_size = 1 in this case (for I2A style summarization)
        # dream tensors contain traj dreamed from first state of the sequence
        state = module.init_state(1 * config["iwae_samples"])
        losses, new_state, loss_metrics, tensors, dream_tensors = \
                            module.training_step(obs,
                                                state,
                                                config,
                                                do_image_pred=False,
                                                do_dream_tensors=True)
    
    dreams[f'forward_{i}'] = dream_tensors

def encode_img_experience(dream, encoder, code, i):
    code[f'forward_{i}'] = encoder(dream)

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

    #wandb.init(project="LI-RARL", name=config['run_name'], config=config)

    sequence_buffer = SequenceBuffer(config)
    transition_buffer = TransitionBuffer(config)

    if config['raw_data']:
        print('Loading offline data into buffer....')
        sequence_buffer.load(config['raw_data'])         # Implement

    if config['prefill'] is not None:
        print('\nCollecting raw experience...\n')
        prefill_steps = 0
        ep_steps = 0
        num_eps = 0
        prefill_obs = env.reset()
        prefill_imgn_code = torch.zeros((1, config["similar"] * config["rollout_enc_size"]), dtype=dtype, device=device)
        while True:
            ep_steps += 1
            prefill_action = env.action_space.sample()
            prefill_next_obs, prefill_reward, prefill_done, prefill_info = env.step(prefill_action)

            proc_prefill_obs = torch.tensor(prefill_obs['image'], dtype=dtype, device=device).unsqueeze(0)
            proc_prefill_next_obs = torch.tensor(prefill_next_obs['image'], device=device, dtype=dtype).unsqueeze(0)
            proc_prefill_reward = torch.tensor([prefill_reward], device=device, dtype=dtype).unsqueeze(0)
            proc_prefill_action = torch.tensor([prefill_action], device=device, dtype=dtype).unsqueeze(0).long()
            proc_prefill_done = torch.tensor([bool(prefill_done)], device=device).unsqueeze(0)
            #print(f"Appending shapes {proc_prefill_obs.shape}, {proc_prefill_next_obs.shape}, {proc_prefill_reward.shape}, {proc_prefill_action.shape}, {proc_prefill_done.shape}, {prefill_imgn_code.shape}")
            transition_buffer.push(proc_prefill_obs, proc_prefill_action, proc_prefill_next_obs, proc_prefill_reward, proc_prefill_done, prefill_imgn_code)

            prefill_obs = prefill_next_obs
            prefill_steps += 1
            
            if prefill_done:
                sequence_buffer.push(*list(prefill_info['episode'].values()))
                prefill_obs = env.reset()
                prefill_done = False
                num_eps += 1
                checkpoint_prefill_steps = prefill_steps
                print(f'Episode {num_eps} \ Number of steps: {ep_steps} \ Total steps added: {prefill_steps}')
                ep_steps = 0

            if prefill_steps >= config['prefill']:
                break
    
    agent = agent_type(config).to(device).to(dtype)
    selector = selector_type(config)
    forward_modules = [forward_type(config).to(device).to(dtype) for _ in range(config['similar'])]    # TODO: Use the same encoder for dreamer and env obs
    encoder = ConvEncoder(config).to(device).to(dtype)
    rollout_encoders = [RolloutEncoder(config).to(device).to(dtype) for _ in range(config['similar'])]

    optimizer = torch.optim.Adam(chain(agent.parameters(), encoder.parameters()), lr=config['learning_rate'])
    total_steps = 0

    print('\nTraining...\n')
    for i in range(config['num_train_episodes']):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(config['max_traj_length']):
            total_steps += 1
            proc_obs = torch.tensor(obs['image'], dtype=dtype, device=device).unsqueeze(0)
            obs_enc = encoder(proc_obs)
            # will it see enough episode ends? - sampled i should be exactly n - seq_len
            sampled_seq = sequence_buffer.sample_sequences(seq_len=config['seq_length'], n_sam_eps=config['n_sam_eps'], reset_interval=config["reset_interval"], skip_random=True)  # skip_random=True for training
            obsrvs = torch.cat([torch.tensor(sampled_seq[i].obs[0], dtype=dtype, device=device).unsqueeze(0) for i in range(len(sampled_seq))])
            obsrvs_enc = encoder(obsrvs)  

            #print('Selecting similar states....')
            selector.fit(obsrvs_enc)                        # TODO: Fix the clustering problem
            selected_seqs = selector.get_similar_seqs(config['similar'], obs_enc, sampled_seq)
            #print(selected_seqs[0]._fields)
            
            dreams = {}
            #imgn_threads = [Thread(target=collect_img_experience, args=(forward_modules[i], selected_seqs[i], config, dreams, 'train', i)) for i in range(config['similar'])]
            
            #print('Training forward module / Imagining trajectories....')
            #for i in range(config['similar']): imgn_threads[i].start()
            #for i in range(config['similar']): imgn_threads[i].join()

            collect_img_experience(forward_modules[0], selected_seqs[0], config, dreams, 'train', 0)

            code = {}

            #summ_threads = [Thread(target=encode_img_experience, args=(imgn_buffers[i], rollout_encoders[i], code, i)) for i in range(config['similar'])]

            #for i in range(config['similar']): summ_threads[i].start()
            #for i in range(config['similar']): summ_threads[i].join()

            encode_img_experience(dreams['forward_0'], rollout_encoders[0], code, 0)

            #print('Aggregating imagination encodings....')
            imgn_code = torch.cat(([code[f'forward_{i}'] for i in range(config['similar'])]), dim=1).to(dtype).to(device)
            #print('Shape of imgn code: ', imgn_code.shape)

            #print(f'Running the agent process....')
            agent_in = torch.cat([obs_enc, imgn_code], dim=1)

            action = agent.select_action(agent_in, env.action_space.sample()).long()                   # Improve api
            #print(f"Shape of action sampled {action.shape}")
            next_obs, reward, done, info = env.step(action.item())
            episode_reward += reward

            proc_next_obs = torch.tensor(next_obs['image'], device=device, dtype=dtype).unsqueeze(0)
            proc_reward = torch.tensor([reward], device=device, dtype=dtype).unsqueeze(0)
            proc_action = action.int().to(device)
            proc_done = torch.tensor([bool(done)], device=device).unsqueeze(0)
            #print(f"Appending shapes {proc_obs.shape}, {proc_next_obs.shape}, {proc_reward.shape}, {proc_action.shape}, {proc_done.shape}, {imgn_code.shape}")

            #print(f"Pushing action of shape {proc_action.shape}")
            transition_buffer.push(proc_obs, proc_action, proc_next_obs, proc_reward, proc_done, imgn_code)

            sample = transition_buffer.sample(config['dqn_batch_size'])
            #print(f'Sampled transition {sample.obs.shape, sample.action.shape, sample.next_obs.shape, sample.reward.shape, sample.done.shape, sample.imgn_code.shape}')
            obs_enc_update = encoder(sample.obs)
            next_obs_enc_update = encoder(sample.next_obs)
            #print(obs_enc_update.shape, next_obs_enc_update.shape)
            input = torch.cat((obs_enc_update, sample.imgn_code), dim=1)
            trg_input = torch.cat((next_obs_enc_update, sample.imgn_code), dim=1)   # workaround - imgn_code abset for next_obs
            target_q_vals, q_vals = agent(input, trg_input, sample)                   # Make agent agnostic

            optimizer.zero_grad()
            agent_loss = nn.MSELoss()(target_q_vals, q_vals)   
            agent_loss.backward()
            optimizer.step()

            if total_steps % config['trg_update_freq'] == 0:
                agent.update_target_network()

            #if i % config['model_save_freq'] == 0:
            #    agent.save_model(config['model_savepath'])

            #if i % config['write_data_freq'] == 0:
            #    sequence_buffer.save(config['offline_trainfile'])

            if done or step == config['max_traj_length'] - 1:
                print(f'Episode: {i} / Reward collected: {episode_reward} / No. of steps: {step}')
                if done:
                    sequence_buffer.push(*list(info['episode'].values()))
                break

    env.close()