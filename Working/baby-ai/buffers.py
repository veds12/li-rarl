from babyai.rl.utils import DictList
import random
import numpy as np
import torch

class ExperienceBuffer:
    def __init__(self, device, frames_per_process, num_processes):
        self.buffer = DictList()
        self.device = device
        self.num_frames_per_process = frames_per_process
        self.num_processes = num_processes
        
        self.buffer.obs = []
        self.buffer.memory = []
        self.buffer.mask = []
        self.buffer.action = []
        self.buffer.value = []
        self.buffer.reward = []
        self.buffer.advantage = []
        self.buffer.returnn = []
        self.buffer.log_prob = []

    def push(self, exp):
        obs = exp.obs.image.cpu()
        memory = exp.memory.cpu()
        mask = exp.mask.cpu()
        action = exp.action.cpu()
        value = exp.value.cpu()
        reward = exp.reward.cpu()
        advantage = exp.advantage.cpu()
        returnn = exp.returnn.cpu()
        log_prob = exp.log_prob.cpu()
        
        self.buffer.obs.append(exp.obs.image)
        self.buffer.memory.append(memory)
        self.buffer.mask.append(mask)
        self.buffer.action.append(action)
        self.buffer.value.append(value)
        self.buffer.reward.append(reward)
        self.buffer.advantage.append(advantage)
        self.buffer.returnn.append(returnn)
        self.buffer.log_prob.append(log_prob)

    def sample(self, sample_size):
        
        obs = torch.cat(self.buffer.obs)
        memory = torch.cat(self.buffer.memory)
        mask = torch.cat(self.buffer.mask)
        action = torch.cat(self.buffer.action)
        value = torch.cat(self.buffer.value)
        reward = torch.cat(self.buffer.reward)
        advantage = torch.cat(self.buffer.advantage)
        returnn = torch.cat(self.buffer.returnn)
        log_prob = torch.cat(self.buffer.log_prob)

        sample = [obs, memory, mask, action, value, reward, advantage, returnn, log_prob]
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

        ind = torch.randperm(len(self.buffer.obs) * self.num_frames_per_process * self.num_processes)[:sample_size]

        sample = {key: sample[key][ind].to(self.device) for key in sample.keys()}

        return sample
    
    def __len__(self):
        return len(self.buffer.obs) * self.num_frames_per_process * self.num_processes