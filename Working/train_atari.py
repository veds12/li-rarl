# Inspired from https://github.com/raillab/dqn
import random
import argparse
import os
import time

import numpy as np
import gym
import wandb

from buffers import TransitionBuffer
from agents import DQN
from models import ConvEncoder
from wrappers import *
from selection import get_selector

import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN Atari')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--logging', type=bool, default=False)
    parser.add_argument('--checkpointing', type=bool, default=False)
    parser.add_argument('--selection', type=str, default=None)
    parser.add_argument('--name', type=str, default='DQN')

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
        "n_retrieval": 64,          # size of the retrieval batch
        "attn_topk": 4,         # topk for selector for attention based selection
        "val_topk": 16,        # topk for selector for value based selection
        "d_k": 128,           # size of key and query embeddings
        "enc_out_size": 2592,
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
    agent.train()

    if args.selection is not None:
        selector_type = get_selector(args.selection)
        selector = selector_type(hyper_params).to(device).to(dtype)

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

    #####################################################################################

    ############################## Set seeds ############################################

    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #####################################################################################

    ################################ Training ###########################################
    
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

            if args.selection is not None and t > hyper_params["learning-starts"]:
                r_states, _, _, _, _ = agent.memory.sample(hyper_params["n_retrieval"])
                r_states = torch.from_numpy(r_states / 255.0).to(device).to(dtype)
                r_states = agent.get_state(r_states)
                value_net = agent._network
                attn_state, _ = selector(q=state, k=r_states, value_net=value_net)
                state = state + attn_state

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

            if args.selection is not None:
                r_update_states, _, _, _, _ = agent.memory.sample(hyper_params["n_retrieval"])           # 0.006s
                r_update_states = torch.from_numpy(r_update_states / 255.0).to(device).to(dtype)         # 0.03s
                r_update_states = agent.get_state(r_update_states)                                     # 0.0003s
                value_net = agent._network
                attn_update_states, _ = selector(q=update_states, k=r_update_states, value_net=value_net)                     # 0.0008s
                attn_update_next_states, _ = selector(q=update_next_states, k=r_update_states, value_net=value_net)
                update_states = update_states + attn_update_states                                       # 0.00002s
                update_next_states = update_next_states + attn_update_next_states

            sample = [update_states, update_actions, update_next_states, update_rewards, update_dones]
            loss = agent.optimize_td_loss(sample)                                                         # 0.006s

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
