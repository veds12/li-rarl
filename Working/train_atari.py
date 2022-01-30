# Inspired from https://github.com/raillab/dqn
import random
import numpy as np
import gym
import wandb

from buffers import TransitionBuffer
from agents import DQN
from wrappers import *

import torch

import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN Atari')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--logging', type=bool, default=False)

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
        
        "env": "PongNoFrameskip-v4",  # name of the game
        "buffer_capacity": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "dqn_type":"neurips",
        # total number of steps to run the environment for
        "num-steps": int(1e6),
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": eps_start,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
    }

    ################################################################################

    ##################### Set up Environment #######################################

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"])

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)

    ####################################################################################

    ############################ Set up Agent ##########################################
    
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

    #####################################################################################

    ############################# Checkpointing #########################################

    if(args.load_checkpoint_file):
        print(f"Loading a policy - { args.load_checkpoint_file } ")
        agent.policy_network.load_state_dict(
            torch.load(args.load_checkpoint_file))

    CHECKPOINT_PATH = os.path.expanduser('~') + '/scratch/li-rarl/DQN'
    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)

    #####################################################################################

    ############################### Logging #############################################

    if args.logging:
        run_name = 'Vanilla_DQN_Pong'
        wandb.init(project='LI-RARL', name=run_name, config=hyper_params)

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

        state = torch.from_numpy(np.array(obs) / 255.0).unsqueeze(0).to(device).to(dtype)

        sample = random.random()

        if sample > eps_threshold:
            # Exploit
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
            loss = agent.optimize_td_loss()

            if args.logging:
                wandb.log({
                    'loss': t,
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
            agent.save_model(path=CHECKPOINT_PATH, name=f'{run_name}_{args.seed}.pt')
            # np.savetxt('rewards_per_episode.csv', episode_rewards,
            #            delimiter=',', fmt='%1.3f')
