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
        self._device = config["device"]
        self._dtype = config["dtype"]
        _in_size = config['enc_out_size'] + config['similar'] * config['rollout_enc_size']
        
        assert config["action_space_type"] == 'Discrete', 'Action space should be discrete'
        self._layer_sizes = [_in_size, *self._hidden_layers, config["action_dim"]]

        self._network = VanillaMLP(self._layer_sizes, config["model_activation"], config["model_dropout"])
        self._target_network = deepcopy(self._network)

        for param in self._target_network.parameters():
            param.requires_grad = False

    def select_action(self, encoded_state):
        if random.uniform(0, 1) < self._epsilon:
            return torch.tensor(self._action_space.sample(), device=self._device, dtype=self._dtype)
        else:
            return torch.argmax(self._network(encoded_state), dim=1)

    def forward(self, sample):
        with torch.no_grad():
            target_q_val = sample.reward + self._gamma * torch.argmax(self._target_network(sample._next_state), dim=1) * (1 - sample.done)

        q_val = self._network(sample._state)[sample.action]

        return target_q_val, q_val

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
