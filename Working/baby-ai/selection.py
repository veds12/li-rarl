# Different methods for selecting similar states

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import random

class KNNSelector(nn.Module):
    def __init__(self, config):
        super(KNNSelector, self).__init__()
        self.knn = NearestNeighbors(n_neighbors=config['n_retrieval'])
        # self.W_vs = nn.Linear(config['n_actions'], config['n_actions'])
        self.n_actions = config['n_actions']

    def forward(self, q, k, obs, **kwargs):
        self.knn.fit(k.cpu().detach())
        distances, indices = self.knn.kneighbors(q.cpu().detach())
        distances = torch.tensor(distances, dtype=q.dtype, device=q.device)
        # attn = nn.Softmax(dim=-1)(distances)

        # actions = F.one_hot(k[1].to(torch.int64), num_classes=self.n_actions).to(q.dtype)

        # v = actions[indices[0]]
        # v = self.W_vs(v)
        

        # attn_info = torch.mm(attn, v)

        indices = torch.from_numpy(indices).to(q.device)
        selected_obs = obs[indices]

        if len(selected_obs.shape) != 5:
            selected_obs = selected_obs.unsqueeze(0)

        selected_obs = selected_obs.permute(1, 0, 2, 3, 4)
        return selected_obs

_selector_dict = {
                "knn": KNNSelector,
                }


def get_selector(selector):
    return _selector_dict[selector]
