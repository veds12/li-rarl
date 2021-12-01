import os
import random
import h5py
from collections import deque, namedtuple

import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class VanillaBuffer:
    def __init__(
        self,
        capacity,
    ):
        self._capacity = capacity
        self._memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        transition = Transition(state, action, next_state, reward, done)
        self._memory.append(transition)

    def pop(self, end=None):
        if end == 'left':
            return self._memory.popleft()
        elif end == 'right' or end == None:
            return self._memory.pop()
        else:
            raise ValueError('end must be either left or right')

    def sample(self, batch_size):
        return Transition(torch.cat(i) for i in zip(*random.sample(self._memory, batch_size)))

    def save(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        experiences = [e._asdict() for e in self._memory]
        with h5py.File(filepath, 'w') as f:
            for i in range(len(experiences)):
                grp = f.create_group(str(i))
                for key in experiences[i].keys():
                    grp.create_dataset(key, data=experiences[i][key])

    def load(self, filename):
        raise NotImplementedError
            
    def __len__(self):
        return len(self._memory)


