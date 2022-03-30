# Inspired from https://github.com/raillab/dqn
import random
import argparse
import os
import time
import copy
from threading import Thread

import numpy as np
import gym
import wandb

from buffers import TransitionBuffer
from agents import DQN
from wrappers import *
from selection import get_selector
from aggregate import AttentionModule
from forward import get_forward

import torch

def rollout(forward_model, init_state, steps, num_trajs, dreams, i):
    out = forward_model.rollout(init_state)
    dream = {
        'states': out[0],
        'rewards': out[1],
        'actions': out[2],
    }

    dreams[f'fwd_{i}'] = dream

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN Atari')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--logging', type=bool, default=False)
    parser.add_argument('--checkpointing', type=bool, default=False)
    parser.add_argument('--selection', type=str, default='knn')
    parser.add_argument('--name', type=str, default='DQN')
    parser.add_argument('--forward', type=str, default='SPR')
    parser.add_argument('--retrieval', type=bool, default=False)

    args = parser.parse_args()

    # If you have a checkpoint file, spend less time exploring
    if(args.load_checkpoint_file):
        eps_start= 0.01
    else:
        eps_start= 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    ############################### Hyperparameters ################################

    hyper_params = {
        "buffer_capacity": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "dqn_type":"neurips",
        "num-steps": int(1e6), # total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": eps_start,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
        "retrieval_batch": 64,          # size of the retrieval batch
        "n_retrieval": 1,         # no. of similar states to be selected
        "attn_topk": 4,         # topk for selector for attention based selection
        "val_topk": 16,        # topk for selector for value based selection
        "d_k": 128,           # size of key and query embeddings
        "enc_out_size": 2592,
        "in_channels": 4,
        "conv_channels": [32, 64, 64],
        "kernel_sizes": [8, 4, 3],
        "strides": [4, 2, 1],
        "paddings": [0, 0, 0],
        "use_maxpool": False,
        "head_sizes": None,
        "dropout": 0,
        "dyn_channels": 64,
        "num_actions": 6,
        "pixels": 49,
        "hidden_size": 64,
        "limit": 1,
        "blocks": 0,
        "norm_type": "bn",
        "renormalize": 1,
        "residual": 0.0,
        "rollout_steps": 5,
        "trajs_per_fwd": 8,
        "traj_enc_size": 1024,
    }

    ################################################################################

    ##################### Set up Environment #######################################

    assert "NoFrameskip" in args.env, "Require environment with no frameskip"
    env = gym.make(args.env)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)

    hyper_params['n_actions'] = env.action_space.n

    ####################################################################################

    ############################ Initializing components ###############################
    
    replay_buffer = TransitionBuffer(hyper_params)
    
    agent = DQN(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=hyper_params["use-double-dqn"],
        lr=hyper_params['learning-rate'],
        batch_size=hyper_params['batch-size'],
        gamma=hyper_params['discount-factor'],
    ).to(device).to(dtype)

    if args.retrieval:
        selector_type = get_selector(args.selection)
        selector = selector_type(hyper_params).to(device).to(dtype)
        forward_type = get_forward(args.forward)
        forward_modules = [forward_type(hyper_params).to(device).to(dtype) for _ in range(hyper_params['n_retrieval'])]
        
        dyn_chkpt_path = './models/dynamics_model.pt'
        enc_chkpt_path = './models/encoder_model.pt'

        for i in range(hyper_params['n_retrieval']):
            forward_modules[i].load_chkpts(dyn_chkpt_path, enc_chkpt_path)

        print('Loaded checkpoints in the forward model')

        aggregator = AttentionModule(hyper_params).to(device).to(dtype)

    #####################################################################################

    ############################# Checkpointing #########################################

    if(args.load_checkpoint_file):
        print(f"Loading a policy - { args.load_checkpoint_file } ")
        agent.policy_network.load_state_dict(
            torch.load(args.load_checkpoint_file))

    CHECKPOINT_PATH = os.path.expanduser('~') + '/scratch/li-rarl/DQN/'
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    #####################################################################################

    ############################### Logging #############################################

    if args.logging:
        wandb.init(project='LI-RARL', name=args.name, config=hyper_params)

    os.environ['WANDB_SILENT'] = 'true'

    #####################################################################################

    ############################## Set seeds ############################################

    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    #####################################################################################

    ################################ Training ###########################################
    
    agent.train()

    eps_timesteps = hyper_params["eps-fraction"] * \
        float(hyper_params["num-steps"])
    episode_rewards = [0.0]

    obs = env.reset()
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * \
            (hyper_params["eps-end"] - hyper_params["eps-start"])

        state = torch.from_numpy(np.array(obs) / 255.0).unsqueeze(0).to(device).to(dtype)    # Shape of state is (1, stack_size, 84, 84)
        sample = random.random()

        if sample > eps_threshold:
            # Exploit
            state = agent.get_state(state)               # Shape of state: (1, 2592)

            if args.retrieval and t > hyper_params["learning-starts"]:
                r_states, r_actions, _, _, _ = agent.memory.sample(hyper_params["retrieval_batch"])
                r_states = torch.from_numpy(r_states / 255.0).to(device).to(dtype)
                r_actions = torch.from_numpy(r_actions).to(device).to(dtype)
                r_states_enc = agent.get_state(r_states)
                
                # Selecting similar states from the retrieval batch
                sel_states = selector(q=state, k=r_states_enc, obs=r_states)

                # Calculating rollouts from the selected states using the forward model
                dreams = {}
                rollout_threads = [Thread(target=rollout, args=(forward_modules[i], sel_states[i], hyper_params['rollout_steps'], hyper_params['trajs_per_fwd'], dreams, i)) for i in range(hyper_params['n_retrieval'])]
                for t in range(hyper_params['n_retrieval']): rollout_threads[t].start()
                for u in range(hyper_params['n_retrieval']): rollout_threads[u].join()

                # Aggregating the imagined states
                fwd_states = [dreams[f'fwd_{i}']['states'] for i in range(hyper_params['n_retrieval'])]
                fwd_states = [dream.reshape((-1,)+dream.shape[2:]) for dream in fwd_states]
                fwd_states = torch.cat(fwd_states, 0).permute(1, 0, 2)

                # Residual attention between agent state and dreamed states
                attn_info, _ = aggregator(q=state, k=fwd_states)
                state = state + attn_info

            action = agent.select_action(state).item()
        
        else:
            # Explore
            action = env.action_space.sample()

        next_obs, reward, done, info = env.step(int(action))
        agent.memory.push(obs, action, next_obs, reward, float(done))
        obs = next_obs

        episode_rewards[-1] += reward
        if done:
            if args.logging:
                wandb.log({
                    'episode_reward': episode_rewards[-1],
                    'episodes': len(episode_rewards),
                    'total_steps': t,
                })
            obs = env.reset()
            episode_rewards.append(0.0)

        if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
            # states, next_states: (batch_size, stack_size, 84, 84)
            # actions, rewards, dones: (batch_size, )
            update_obs, update_actions, update_next_obs, update_rewards, update_dones = agent.memory.sample(hyper_params["batch-size"])
            update_states = torch.from_numpy(update_obs / 255.0).to(device).to(dtype)
            update_next_states = torch.from_numpy(update_next_obs / 255.0).to(device).to(dtype)
            update_states = agent.get_state(update_states)
            update_next_states = agent.get_state(update_next_states)

            if args.retrieval:
                r_update_states, _, _, _, _ = agent.memory.sample(hyper_params["retrieval_batch"])
                r_update_states = torch.from_numpy(r_update_states / 255.0).to(device).to(dtype)
                r_update_states_enc = agent.get_state(r_update_states)

                # Selecting similar states from the retrieval batch
                sel_states = selector(q=update_states, k=r_update_states_enc, obs=r_update_states)

                # Calculating Rollouts from the selected states
                updt_dreams = {}
                updt_rollout_threads = [Thread(target=rollout, args=(forward_modules[i], sel_states[i], hyper_params['rollout_steps'], hyper_params['trajs_per_fwd'], updt_dreams, i)) for i in range(hyper_params['n_retrieval'])]
                for t in range(hyper_params['n_retrieval']): updt_rollout_threads[t].start()
                for u in range(hyper_params['n_retrieval']): updt_rollout_threads[u].join()

                # Aggregating the imagined states
                updt_fwd_states = [updt_dreams[f'fwd_{i}']['states'] for i in range(hyper_params['n_retrieval'])]
                updt_fwd_states = [dream.reshape((-1,)+dream.shape[2:]) for dream in updt_fwd_states]
                updt_fwd_states = torch.cat(updt_fwd_states, 0).permute(1, 0, 2)

                # Residual attention to between states and next states in the sampled batch and the dreamed states
                attn_info_updt_states, _ = aggregator(q=update_states, k=updt_fwd_states)
                update_states = update_states + attn_info_updt_states
                attn_info_updt_next_states, _ = aggregator(q=update_next_states, k=updt_fwd_states)
                update_next_states = update_next_states + attn_info_updt_next_states

            sample = [update_states, update_actions, update_next_states, update_rewards, update_dones]
            loss = agent.optimize_td_loss(sample)

            if args.logging:
                wandb.log({
                    'loss': loss,
                })

        if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params[
                "print-freq"] == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            
            if args.checkpointing:
                agent.save_model(path=CHECKPOINT_PATH, name=f'{args.name}_{args.seed}.pt')
            # np.savetxt('rewards_per_episode.csv', episode_rewards,
            #            delimiter=',', fmt='%1.3f')

    ######################################################################################
