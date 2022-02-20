import os
from itertools import chain
import argparse
import yaml
from threading import Thread  # Shift to PyTorch multiprocessing
import pprint
import random
from collections import deque
import time

import numpy as np
import torch
import torch.nn as nn

import gym.spaces as spaces

from agents import get_agent
from selection import get_selector
from summarize import get_summarizer
from buffers import SequenceBuffer, TransitionBuffer
from pydreamer.envs import create_env
from models import ConvEncoder
from pydreamer.models.dreamer import Dreamer
from utils import *

import wandb

os.environ["WANDB_SILENT"] = "true"
# torch.autograd.set_detect_anomaly(True)

def collect_img_experience(module, seq, config, dreams, mode, i):
    # TODO:
    #   add options for categorical images
    #   convert to torch directly

    image = seq.obs / 255.0 - 0.5
    image = image.transpose(0, 1, 4, 2, 3)
    image = image.astype(np.float32)

    action = seq.action
    if len(action) == 2:
        action = to_onehot(seq.action, config["action_dim"]).astype(np.float32)
    else:
        action = action.astype(np.float32)
    assert len(action.shape) == 3

    if seq.done is not None:
        terminal = seq.done.astype(np.float32)
    else:
        terminal = np.zeros((config["imgn_length"], 1)).astype(np.float32)

    if seq.reward is not None:
        reward = seq.reward.astype(np.float32)
    else:
        np.zeros((config["imgn_length"], 1)).astype(np.float32)

    if config["clip_rewards"] == "tanh":
        reward = np.tanh(reward)  # type: ignore
    if config["clip_rewards"] == "log1p":
        reward = np.log1p(reward)  # type: ignore

    reset = seq.reset.astype(bool)
    vecobs = np.zeros((config["imgn_length"], reward.shape[1], 64)).astype(np.float32)

    obs = {
        "image": torch.from_numpy(image).to(device),
        "action": torch.from_numpy(action).to(device),
        "reward": torch.from_numpy(reward).to(device),
        "terminal": torch.from_numpy(terminal).to(device),
        "reset": torch.from_numpy(reset).to(device),
        "vecobs": torch.from_numpy(vecobs).to(device),
    }

    if mode == "train":
        # not keeping states for now (config["keep_states"] = False)
        # should be module.init_state(config["batch_size"] * config["iwae_samples"]);
        # batch_size = 1 in this case (for I2A style summarization)
        # dream tensors contain traj dreamed from first state of the sequence
        state = module.init_state(1 * config["iwae_samples"])
        _, _, loss_metrics, tensors, dream_tensors = module.training_step(
            obs, state, config, do_image_pred=False, do_dream_tensors=True
        )

    losses_log = {f"fwd_{i}" + k: v.detach() for k, v in loss_metrics.items()}
    dream_tensors['reward_pred'] = dream_tensors['reward_pred'].unsqueeze(2)
    #dream_tensors = {k: v.detach() for k, v in dream_tensors.items()}
    dreams[f"forward_{i}"] = dream_tensors
    del dream_tensors, losses_log, loss_metrics, tensors

parser = argparse.ArgumentParser()

parser.add_argument("--config", help="Path to config file", type=str, default="./config.yaml")
parser.add_argument("--suite", default="atari", help="Name of the benchmark suite")
parser.add_argument("--env", default="atari_breakout", help="Name of the environment")
parser.add_argument("--selector", type=str, default="kmeans", help="Type of selector (supported: kmeans, attention")
parser.add_argument("--summarizer", type=str, default="i2a", help="Name of the summarizer (supported: i2a, self-attention")
parser.add_argument("--agent", default="dqn", type=str, help="name of the agent to use (supported: dqn")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--run", default="", help="Name of this run")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    config = {
        **meta["general"],
        **meta["dreamer"],
        **meta[args.agent],
        **meta[args.selector],
        **meta[args.summarizer]
    }
    config.update(meta[args.suite])
    del meta

    dtype = torch.float32  # TODO: Set acc to config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = create_env(args.env, config)

    config["action_dim"] = env.action_space.n
    config["obs_dim"] = env.observation_space["image"].shape[0]
    config["action_space_type"] = env.action_space.__class__.__name__
    config["image_channels"] = env.observation_space["image"].shape[-1]
    config["image_size"] = env.observation_space["image"].shape[1]
    config["seed"] = args.seed

    agent_type = get_agent(args.agent)
    selector_type = get_selector(args.selector)
    summarizer_type = get_summarizer(args.summarizer)

    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project="LI-RARL",
        name=args.run,
        config=config,
    )

    # ================================== INITIALIZING AND PREFILLING THE BUFFERS =====================================
    # ================================================================================================================
    # ================================================================================================================

    sequence_buffer = SequenceBuffer(config)
    transition_buffer = TransitionBuffer(config)

    if config["raw_data"]:
        print("Loading offline data into buffer....")
        sequence_buffer.load(config["offline_datafile"])  # Implement

    if config["prefill"] is not None:
        print("\nCollecting raw experience...\n")
        prefill_steps = 0
        ep_steps = 0
        num_eps = 0
        prefill_obs = env.reset()
        
        while True:
            ep_steps += 1
            prefill_action = env.action_space.sample()
            prefill_next_obs, prefill_reward, prefill_done, prefill_info = env.step(
                prefill_action
            )

            proc_prefill_obs = torch.tensor(
                prefill_obs["image"], dtype=dtype,
            ).unsqueeze(0)
            proc_prefill_next_obs = torch.tensor(
                prefill_next_obs["image"], dtype=dtype
            ).unsqueeze(0)
            proc_prefill_reward = torch.tensor(
                [prefill_reward], dtype=dtype
            ).unsqueeze(0)
            proc_prefill_action = (
                torch.tensor([prefill_action], dtype=dtype)
                .unsqueeze(0)
                .int()
            )
            proc_prefill_done = torch.tensor(
                [bool(prefill_done)],
            ).unsqueeze(0)

            transition_buffer.push(
                proc_prefill_obs,
                proc_prefill_action,
                proc_prefill_next_obs,
                proc_prefill_reward,
                proc_prefill_done,
            )

            prefill_obs = prefill_next_obs
            prefill_steps += 1

            if prefill_done:
                sequence_buffer.push(*list(prefill_info["episode"].values()))
                prefill_obs = env.reset()
                prefill_done = False
                num_eps += 1
                print(
                    f"Episode {num_eps} \ Number of steps: {ep_steps} \ Total steps added: {prefill_steps}"
                )
                ep_steps = 0

            if prefill_steps >= config["prefill"]:
                break

    # ================== INITIALIZING THE AGENT, FORWARD MODULES, ROLLOUT ENCODERS AND OPTIMIZER =====================
    # ================================================================================================================
    # ================================================================================================================

    agent = agent_type(config).to(device).to(dtype)
    selector = selector_type(config)
    summarizer = summarizer_type(config).to(device).to(dtype)
    forward_modules = [
        Dreamer(config).to(device).to(dtype) for _ in range(config["similar"])
    ]  # TODO: Use the same encoder for dreamer and env obs
    encoder = ConvEncoder(config).to(device).to(dtype)

    optimizer = torch.optim.Adam(
        chain(
            agent.parameters(),
            encoder.parameters(),
            summarizer.parameters(),
        ),
        lr=config["learning_rate"],
    )


    print("\nTraining...\n")

    print(f'Model checkpoint path: {config["model_savepath"]}')
    print(f'Buffer data save path: {config["data_savepath"]}\n')
    agent.train()
    encoder.train()

    for a in forward_modules:
        a.train()

    # =============================================== TRAINING LOOP ==================================================
    # ================================================================================================================
    # ================================================================================================================
    total_steps = 0
    for e in range(config["num_train_episodes"]):
        obs = env.reset()
        episode_reward = 0
        episode_loss = 0

        for step in range(config["max_traj_length"]):
            total_steps += 1
            # ================================== ENVIRONMENT INTERACTION =========================
            # ====================================================================================
            proc_obs = torch.tensor(obs["image"], dtype=dtype).unsqueeze(
                0
            )
            obs_enc = encoder(proc_obs.to(device))

            # ====== SELECTING config['similar'] SIMILAR SEQUENCES FROM THE SEQUECE BUFFER =======  
                 
            # begin_1 = time.time()
            sampled_seq = sequence_buffer.sample_sequences(                                 
                seq_len=config["imgn_length"],                                  # will it see enough episode ends? - sampled i should be exactly n - seq_len
                n_sam_eps=config["n_sam_eps"],
                reset_interval=config["reset_interval"],
                skip_random=True,
            )  # skip_random=True for training
            
            obsrvs = torch.cat(
                [
                    torch.tensor(
                        sampled_seq[k].obs[0], dtype=dtype, device=device
                    ).unsqueeze(0)
                    for k in range(len(sampled_seq))
                ]
            )
            obsrvs_enc = encoder(obsrvs)
            
            if args.selector == 'kmeans':
                selector.fit(obsrvs_enc)
            elif args.selector == 'attention':
                selector.to(device).to(dtype)
            else:
                raise ValueError(f'Selector {args.selector} not implemented')

            # try:
            selected_seqs = selector.get_similar_seqs(
            config["similar"], obs_enc, sampled_seq, obsrvs_enc
            )
            # except:
            #     print('Continuing (EI)')
            #     continue

            # end_1 = time.time()

            # ================ IMAGINING TRAJECTORIES FROM THE SELECTED SEQUENCES ================

            # begin_2 = time.time()
            dreams = {}
            imgn_threads = [Thread(target=collect_img_experience, args=(forward_modules[i], selected_seqs[i], config, dreams, 'train', i)) for i in range(config['similar'])]

            for t in range(config['similar']): imgn_threads[t].start()
            for u in range(config['similar']): imgn_threads[u].join()

            # end_2 = time.time()

            # ========================= SUMMARIZING THE IMAGINED TRAJECTORIES ========================

            # begin_3 = time.time()
            imgn_code = summarizer(dreams=dreams, state=obs_enc)
            # end_3 = time.time()

            # =========================== EXECUTING A STEP IN THE ENV ============================

            # begin_5 = time.time()
            if args.summarizer == 'i2a':
                agent_in = torch.cat([obs_enc, imgn_code], dim=1)
            elif args.summarizer == 'self-attention':
                agent_in = obs_enc + imgn_code
            else:
                raise ValueError(f'Summarizer {args.summarizer} not implemented')

            action = agent.select_action(
                agent_in, env.action_space.sample()
            ).long()
            next_obs, reward, done, info = env.step(action.item())
            episode_reward += reward
            # end_5 = time.time()

            # ================ ADDING THE TRANSITION TO THE TRANSITION BUFFER ====================

            # begin_6 = time.time()
            proc_next_obs = torch.tensor(
                next_obs["image"], dtype=dtype
            ).unsqueeze(0)
            proc_reward = torch.tensor([reward], dtype=dtype).unsqueeze(
                0
            )
            proc_action = action.int()
            proc_done = torch.tensor([bool(done)]).unsqueeze(0)

            transition_buffer.push(
                proc_obs.cpu(), proc_action.cpu(), proc_next_obs.cpu(), proc_reward.cpu(), proc_done.cpu()
            )
            # end_6 = time.time()

            obs = next_obs

            if total_steps % config["agent_updt_freq"] == 0:

                # =========================== UPDATING THE AGENT ======================================
                # =====================================================================================

                # ========= SAMPLING FROM THE TRANSITION BUFFER FOR THE AGENT TO TRAIN ON =============

                # begin_7 = time.time()
                sample = transition_buffer.sample(config["dqn_batch_size"])
                sampled_obs = sample.obs.to(device)
                sampled_next_obs = sample.next_obs.to(device)
                sampled_action = sample.action.to(device)
                sampled_reward = sample.reward.to(device)
                sampled_done = sample.done.to(device)

                updt_obs_enc = encoder(sampled_obs)
                nxt_updt_obs_enc = encoder(sampled_next_obs)
                # end_7 = time.time()

                # ==================== CALCULATING THE IMAGINATION CODE FOR OBS =======================

                # begin_8 = time.time()                                          
                updt_sampled_seq = sequence_buffer.sample_sequences(seq_len=config["imgn_length"], n_sam_eps=config["n_sam_eps"], reset_interval=config["reset_interval"], skip_random=True)
                updt_obsrvs = torch.cat([torch.tensor(updt_sampled_seq[i].obs[0], dtype=dtype, device=device).unsqueeze(0) for i in range(len(updt_sampled_seq))])
                updt_obsrvs_enc = encoder(updt_obsrvs)
                if args.selector == 'kmeans':
                    selector.fit(updt_obsrvs_enc)

                # try:
                updt_selected_seqs = selector.get_similar_seqs(config["similar"], updt_obs_enc, updt_sampled_seq, updt_obsrvs_enc)
                # except:
                #     print('Continuing (AU)')
                #     continue
                    
                updt_dreams = {}
                # end_8 = time.time()
                
                # begin_9 = time.time()
                updt_imgn_threads = [Thread(target=collect_img_experience, args=(forward_modules[i], updt_selected_seqs[i], config, updt_dreams, 'train', i)) for i in range(config['similar'])]
                
                for x in range(config['similar']): updt_imgn_threads[x].start()
                for y in range(config['similar']): updt_imgn_threads[y].join()
                # end_9 = time.time()
                
                # begin_10 = time.time()
                updt_obs_imgn_code = summarizer(dreams=updt_dreams, state=updt_obs_enc, nxt_state=nxt_updt_obs_enc)
                # end_10 = time.time()

                # ==================== CALCULATING THE IMAGINATION CODE FOR NEXT_OBS =======================

                # ================ CALCULATING THE INPUT AND TARGET INPUT FOR THE AGENT ================    

                # begin_14 = time.time()
                if args.summarizer == 'i2a':                            
                    input = torch.cat((updt_obs_enc, updt_obs_imgn_code), dim=1)
                    trg_input = torch.cat(
                        (nxt_updt_obs_enc, updt_obs_imgn_code), dim=1
                    )  # workaround - imgn_code absent for next_obs
                elif args.summarizer == 'self-attention':
                    input = updt_obs_enc + updt_obs_imgn_code
                    trg_input = nxt_updt_obs_enc + updt_obs_imgn_code
                else:
                    raise ValueError(f'Summarizer {args.summarizer} not implemented')
                
                # end_14 = time.time()

                # ======================== CALCULATING LOSS AND BACKPROPAGATING ========================

                # begin_15 = time.time()
                target_q_vals, q_vals = agent(
                    input, trg_input, sampled_action, sampled_reward, sampled_done
                )  # Make agent agnostic                                                  
                agent_loss = nn.MSELoss()(target_q_vals, q_vals)
                episode_loss += agent_loss.detach()

                optimizer.zero_grad()
                agent_loss.backward()
                optimizer.step()
                # end_15 = time.time()           
                
                if total_steps % config["trg_update_freq"] == 0:
                    agent.update_target_network()

            print(f'Step {total_steps} | Step Reward {reward}')
            
            if total_steps % config['model_save_freq'] == 0:
                agent.save_model(config['model_savepath']+f'/{args.run}/', f'model_{total_steps}_steps.pt')

            # if total_steps % config['write_data_freq'] == 0:
            #     sequence_buffer.save(config['data_savepath']+f'/{args.run}', f'final_sequences.h5')
            #     transition_buffer.save(config['data_savepath']+f'/{args.run}', f'final_transitions.h5')

            wandb.log({"total_steps": total_steps, "reward": reward})
            if done or step == config["max_traj_length"] - 1:
                print(
                    f"Episode: {e} / No. of steps: {step} / Loss: {episode_loss} / Reward collected: {episode_reward}"
                )
                wandb.log(
                    {
                        "episode_reward": episode_reward,
                        "episode_loss": episode_loss,
                        "episodes": e,
                    }
                )
                if done:
                    sequence_buffer.push(*list(info["episode"].values()))
                break

    env.close()