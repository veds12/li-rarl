# Definitions for different agent processes

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym.spaces as spaces

from models import VanillaMLP

from copy import deepcopy
import random
from itertools import count


class DQN(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(DQN, self).__init__()
        self._hidden_layers = config["hidden_layers"]
        self._gamma = config["dqn_gamma"]
        self._epsilon = config["dqn_epsilon"]
        self._batch_size = config["dqn_batch_size"]
        self._tau = config["dqn_tau"]
        self._device = torch.device('cuda')
        self._dtype = torch.float32
        _in_size = config['enc_out_size'] + config['similar'] * config['rollout_enc_size']
        
        assert config["action_space_type"] == 'Discrete', 'Action space should be discrete'
        self._layer_sizes = [_in_size, *self._hidden_layers, config["action_dim"]]

        self._network = VanillaMLP(self._layer_sizes, config["model_activation"], config["model_dropout"])
        self._target_network = deepcopy(self._network)

        for param in self._target_network.parameters():
            param.requires_grad = False

    def select_action(self, encoded_state, randn_action):
        if random.uniform(0, 1) < self._epsilon:
            return torch.tensor(randn_action, device=self._device, dtype=self._dtype).unsqueeze(0)
        else:
            return torch.argmax(self._network(encoded_state), dim=1).unsqueeze(0)

    def forward(self, input, trg_input, sample):
        q_vals = self._network(input).gather(1, sample.action)
        with torch.no_grad():
            target_q_vals = sample.reward + self._gamma * self._target_network(
                trg_input
            ).max(1).values.unsqueeze(1) * (~sample.done)

        return target_q_vals, q_vals

    def update_target_network(self):
        with torch.no_grad():
            for target_param, param in zip(self._target_network.parameters(), self._network.parameters()):
                target_param.data.copy_(target_param.data * (self._tau) + param.data * (1 - self._tau))

    def save_model(self, path):
        torch.save(self._network.state_dict(), path)
        print(f"Model saved at: {path}")

    def load_model(self, path):
        self._network.load_state_dict(torch.load(path))
        self._target_network = deepcopy(self._network)
        print(f"Model loaded from: {path}")

_agent_factory = {
    'dqn': DQN,
}

def get_agent(agent):
    return _agent_factory[agent]
