# Definitions for different agent processes

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import gym.spaces as spaces

from networks import VanillaMLP

from copy import deepcopy
import random
from itertools import count


class DQN(nn.Module):
    def __init__(
        self,
        action_space,
        encoded_state_size,
        hidden_layers,
        model_activation,
        model_dropout,
        device,
        gamma,
        epsilon,
        sample_size,
        tau,
        dtype,
    ):
        super(DQN, self).__init__()
        self._action_space = action_space
        self._encoded_state_size = encoded_state_size,
        self._hidden_layers = hidden_layers
        self._device = torch.device(device)
        self._gamma = gamma
        self._epsilon = epsilon
        self._sample_size = sample_size
        self._tau = tau
        self._dtype = dtype
        
        assert isinstance(self._action_space, spaces.Discrete), 'Action space should be discrete'
        
        self._layer_sizes = [self._encoded_state_size, *self._hidden_layers, self._action_space.n]

        if device == 'cuda':
            assert torch.cuda.is_available()

        self._network = VanillaMLP(self._layer_sizes, model_activation, model_dropout)
        self._target_network = deepcopy(self._network)

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




        



        
