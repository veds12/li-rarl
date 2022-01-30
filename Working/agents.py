# Definitions for different agent processes

from pathlib import Path
import os
from buffers import TransitionBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym.spaces as spaces

from models import VanillaMLP

from copy import deepcopy
import random
from itertools import count
from models import DQN as DQN_neurips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class DQN(nn.Module):
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 replay_buffer: TransitionBuffer,
                 use_double_dqn,
                 lr,
                 batch_size,
                 gamma,
                 ):
        super(DQN, self).__init__()
        self.memory = replay_buffer
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self._gamma = gamma


        self._network = DQN_neurips(observation_space, action_space)
        self._target_network = DQN_neurips(observation_space, action_space)
        self.update_target_network()
        self._target_network.eval()

        self.optimizer = optim.RMSprop(self._network.parameters(), lr=lr)

    def optimize_td_loss(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        
        update_states = torch.from_numpy(states / 255.0).to(device).to(dtype)
        update_next_states = torch.from_numpy(next_states / 255.0).to(device).to(dtype)
        actions = torch.from_numpy(actions).unsqueeze(1).to(device).to(dtype)
        rewards = torch.from_numpy(rewards).unsqueeze(1).to(device).to(dtype)
        dones = torch.from_numpy(dones).unsqueeze(1).to(device).to(dtype)

        target_q_vals, q_vals = self.forward(update_states, update_next_states, actions.to(torch.int64), rewards, dones)
        loss = F.smooth_l1_loss(q_vals, target_q_vals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del states
        del next_states
        return loss.item()

    def select_action(self, state):
        return torch.argmax(self._network(state), dim=1)

    def forward(self, input, trg_input, action, reward, done):
        # print(sample.action.dtype)
        q_vals = self._network(input).gather(1, action)

        with torch.no_grad():
            if self.use_double_dqn:
                _, max_next_action = self._network(trg_input).max(1)
                max_next_q_values = self._target_network(trg_input).gather(1, max_next_action.unsqueeze(1))
            else:
                next_q_values = self._target_network(trg_input)
                max_next_q_values, _ = next_q_values.max(1)
            
            target_q_vals = reward + self._gamma * max_next_q_values * (1 - done)

        return target_q_vals, q_vals

    def update_target_network(self):
        self._target_network.load_state_dict(self._network.state_dict())

    def save_model(self, path, name, VERBOSE=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        path = os.path.join(path, name)
        torch.save(self._network.state_dict(), path)
        if VERBOSE:
            print(f"Model saved at: {path}")

    def load_model(self, path):
        self._network.load_state_dict(torch.load(path))
        self._target_network = deepcopy(self._network)


_agent_factory = {
    "dqn": DQN,
}

def get_agent(agent):
    return _agent_factory[agent]
