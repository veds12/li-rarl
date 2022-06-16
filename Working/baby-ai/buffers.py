import random
import string
import os
import json
from collections import deque

import numpy as np
import pickle
import torch

from babyai.rl.utils import DictList

class OfflineExperienceBuffer:
    def __init__(self, env, path, device):
        self.device = device
        self.env = env

        with open(path, 'rb') as f:
            self.data = pickle.load(f)
            f.close()

        self.size = len(self.data['images'])
        self.traj_len = self.data['images'][0].shape[0]

    def sample(self, sample_size):
        obs = np.concatenate(self.data['images'])
        action = np.concatenate(self.data['actions'])
        # masks = np.concatenate(self.data['masks'])

        ind = np.random.randint(0, self.size, sample_size)

        obs = obs[ind]
        image_obs = []

        for i in range(sample_size):
            obs_i = obs[i].transpose(1, 2, 0)
            obs_i = {'image': obs_i, 'mission': None}
            image_obs.append(self.env.observation(obs_i)['image'])

        obs = np.stack(image_obs, axis=0)

        sample = {
            'obs': torch.from_numpy(obs).to(self.device).to(torch.float32),
            'action': torch.from_numpy(action[ind]).to(self.device),
            # 'mask': torch.from_numpy(masks[ind]).to(self.device)
        }

        return sample

    def __len__(self):
        return self.size * self.traj_len

class ExperienceBuffer:
    def __init__(self, device, frames_per_process, num_processes):
        self.buffer = DictList()
        self.device = device
        self.num_frames_per_process = frames_per_process
        self.num_processes = num_processes
        
        self.buffer.obs = deque(maxlen=400)
        self.buffer.memory = deque(maxlen=400)
        self.buffer.mask = deque(maxlen=400)
        self.buffer.action = deque(maxlen=400)
        self.buffer.value = deque(maxlen=400)
        self.buffer.reward = deque(maxlen=400)
        self.buffer.advantage = deque(maxlen=400)
        self.buffer.returnn = deque(maxlen=400)
        self.buffer.log_prob = deque(maxlen=400)

    def push(self, exp):
        obs = exp.obs.image.cpu().numpy()
        memory = exp.memory.cpu().numpy()
        mask = exp.mask.cpu().numpy()
        action = exp.action.cpu().numpy()
        value = exp.value.cpu().numpy()
        reward = exp.reward.cpu().numpy()
        advantage = exp.advantage.cpu().numpy()
        returnn = exp.returnn.cpu().numpy()
        log_prob = exp.log_prob.cpu().numpy()

        self.buffer.obs.append(obs)
        self.buffer.memory.append(memory)
        self.buffer.mask.append(mask)
        self.buffer.action.append(action)
        self.buffer.value.append(value)
        self.buffer.reward.append(reward)
        self.buffer.advantage.append(advantage)
        self.buffer.returnn.append(returnn)
        self.buffer.log_prob.append(log_prob)

    def sample(self, sample_size):

        ind = np.random.permutation(len(self.buffer.obs) * self.num_frames_per_process * self.num_processes)[:sample_size]

        obs = np.concatenate(list(self.buffer.obs))
        memory = np.concatenate(list(self.buffer.memory))
        mask = np.concatenate(list(self.buffer.mask))
        action = np.concatenate(list(self.buffer.action))
        value = np.concatenate(list(self.buffer.value))
        reward = np.concatenate(list(self.buffer.reward))
        advantage = np.concatenate(list(self.buffer.advantage))
        returnn = np.concatenate(list(self.buffer.returnn))
        log_prob = np.concatenate(list(self.buffer.log_prob))

        sample = {
            'obs': torch.from_numpy(obs[ind]).to(self.device),
            'memory': torch.from_numpy(memory[ind]).to(self.device),
            'mask': torch.from_numpy(mask[ind]).to(self.device),
            'action': torch.from_numpy(action[ind]).to(self.device),
            'value': torch.from_numpy(value[ind]).to(self.device),
            'reward': torch.from_numpy(reward[ind]).to(self.device),
            'advantage': torch.from_numpy(advantage[ind]).to(self.device),
            'returnn': torch.from_numpy(returnn[ind]).to(self.device),
            'log_prob': torch.from_numpy(log_prob[ind]).to(self.device),
        }

        return sample
    
    def __len__(self):
        return len(self.buffer.obs) * self.num_frames_per_process * self.num_processes

class ExperienceBufferOnDisk:
    """
    This can be used for reading and writing the experiences on the disk rather than the memory.
    However, the use of this buffer is not recommended (until I find a superfast R/W alternative) since the R/W to the disk is painstakingly
    slow
    """
    def __init__(self, device, frames_per_process, num_processes, path='/network/scratch/v/vedant.shah/li-rarl/data/'):
        self.buffer = DictList()
        self.device = device
        self.num_frames_per_process = frames_per_process
        self.num_processes = num_processes
        self.len = 0
        
        self.path = os.path.join(path, ''.join(random.choices(string.ascii_uppercase + string.digits, k=7)))
        os.makedirs(self.path)

        self.obs_path = os.path.join(self.path, 'obs.json')
        with open(self.obs_path, 'w') as f:
            json.dump([], f)
            f.close()
        
        self.memory_path = os.path.join(self.path, 'memory.json')
        with open(self.memory_path, 'w') as f:
            json.dump([], f)
            f.close()

        self.mask_path = os.path.join(self.path, 'mask.json')
        with open(self.mask_path, 'w') as f:
            json.dump([], f)
            f.close()

        self.action_path = os.path.join(self.path, 'action.json')
        with open(self.action_path, 'w') as f:
            json.dump([], f)
            f.close()

        self.value_path = os.path.join(self.path, 'value.json')
        with open(self.value_path, 'w') as f:
            json.dump([], f)
            f.close()

        self.reward_path = os.path.join(self.path, 'reward.json')
        with open(self.reward_path, 'w') as f:
            json.dump([], f)
            f.close()

        self.advantage_path = os.path.join(self.path, 'advantage.json')
        with open(self.advantage_path, 'w') as f:
            json.dump([], f)
            f.close()

        self.returnn_path = os.path.join(self.path, 'returnn.json')
        with open(self.returnn_path, 'w') as f:
            json.dump([], f)
            f.close()

        self.log_prob_path = os.path.join(self.path, 'log_prob.json')
        with open(self.log_prob_path, 'w') as f:
            json.dump([], f)
            f.close()

    def push(self, exp):
        
        obs = exp.obs.image.cpu().numpy()
        memory = exp.memory.cpu().numpy()
        mask = exp.mask.cpu().numpy()
        action = exp.action.cpu().numpy()
        value = exp.value.cpu().numpy()
        reward = exp.reward.cpu().numpy()
        advantage = exp.advantage.cpu().numpy()
        returnn = exp.returnn.cpu().numpy()
        log_prob = exp.log_prob.cpu().numpy()
        
        with open(self.obs_path, 'r+') as f:
            obs_list = json.load(f)
            obs = obs.tolist()
            obs_list.append(obs)
            f.seek(0)
            json.dump(obs_list, f)
            self.len += 1
            f.close()

        with open(self.memory_path, 'r+') as f:
            memory_list = json.load(f)
            memory = memory.tolist()
            memory_list.append(memory)
            f.seek(0)
            json.dump(memory_list, f)
            f.close()

        with open(self.mask_path, 'r+') as f:
            mask_list = json.load(f)
            mask = mask.tolist()
            mask_list.append(mask)
            f.seek(0)
            json.dump(mask_list, f)
            f.close()

        with open(self.action_path, 'r+') as f:
            action_list = json.load(f)
            action = action.tolist()
            action_list.append(action)
            f.seek(0)
            json.dump(action_list, f)
            f.close()

        with open(self.value_path, 'r+') as f:
            value_list = json.load(f)
            value = value.tolist()
            value_list.append(value)
            f.seek(0)
            json.dump(value_list, f)
            f.close()
        
        with open(self.reward_path, 'r+') as f:
            reward_list = json.load(f)
            reward = reward.tolist()
            reward_list.append(reward)
            f.seek(0)
            json.dump(reward_list, f)
            f.close()

        with open(self.advantage_path, 'r+') as f:
            advantage_list = json.load(f)
            advantage = advantage.tolist()
            advantage_list.append(advantage)
            f.seek(0)
            json.dump(advantage_list, f)
            f.close()

        with open(self.returnn_path, 'r+') as f:
            returnn_list = json.load(f)
            returnn = returnn.tolist()
            returnn_list.append(returnn)
            f.seek(0)
            json.dump(returnn_list, f)
            f.close()
        
        with open(self.log_prob_path, 'r+') as f:
            log_prob_list = json.load(f)
            log_prob = log_prob.tolist()
            log_prob_list.append(log_prob)
            f.seek(0)
            json.dump(log_prob_list, f)
            f.close()

    def sample(self, sample_size):
        
        ind = np.random.permutation(len(self.buffer.obs) * self.num_frames_per_process * self.num_processes)[:sample_size]
        
        with open(self.obs_path, 'r') as f:
            obs_list = json.load(f)
            obs = np.concatenate(obs_list)
            obs = torch.from_numpy(obs[ind]).to(self.device)
            f.close()
        
        with open(self.memory_path, 'r') as f:
            memory_list = json.load(f)
            memory = np.concatenate(memory_list)
            memory = torch.from_numpy(memory[ind]).to(self.device)
            f.close()

        with open(self.mask_path, 'r') as f:
            mask_list = json.load(f)
            mask = np.concatenate(mask_list)
            mask = torch.from_numpy(mask[ind]).to(self.device)
            f.close()
        
        with open(self.action_path, 'r') as f:
            action_list = json.load(f)
            action = np.concatenate(action_list)
            action = torch.from_numpy(action[ind]).to(self.device)
            f.close()

        with open(self.value_path, 'r') as f:
            value_list = json.load(f)
            value = np.concatenate(value_list)
            value = torch.from_numpy(value[ind]).to(self.device)
            f.close()

        with open(self.reward_path, 'r') as f:
            reward_list = json.load(f)
            reward = np.concatenate(reward_list)
            reward = torch.from_numpy(reward[ind]).to(self.device)
            f.close()

        with open(self.advantage_path, 'r') as f:
            advantage_list = json.load(f)
            advantage = np.concatenate(advantage_list)
            advantage = torch.from_numpy(advantage[ind]).to(self.device)
            f.close()

        with open(self.returnn_path, 'r') as f:
            returnn_list = json.load(f)
            returnn = np.concatenate(returnn_list)
            returnn = torch.from_numpy(returnn[ind]).to(self.device)
            f.close()

        with open(self.log_prob_path, 'r') as f:
            log_prob_list = json.load(f)
            log_prob = np.concatenate(log_prob_list)
            log_prob = torch.from_numpy(log_prob[ind]).to(self.device)
            f.close()

        sample = {
            'obs': obs,
            'memory': memory,
            'mask': mask,
            'action': action,
            'value': value,
            'reward': reward,
            'advantage': advantage,
            'returnn': returnn,
            'log_prob': log_prob,
        }

        return sample
    
    def __len__(self):
        return  self.len * self.num_frames_per_process * self.num_processes