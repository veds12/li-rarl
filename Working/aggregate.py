import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import random
from threading import Thread
from buffers import Sequence
from utils import Sparse_attention

class AttentionModule(nn.Module):
    def __init__(self, config):
        super(AttentionModule, self).__init__()
        self.topk=config["attn_topk"]
        d_k = config["d_k"]
        d_model = config["traj_enc_size"]
        self.temperature = np.power(d_k, 0.5)
        self.n_actions = config['n_actions']
        d_states = config['enc_out_size']

        self.W_qs = nn.Linear(d_states, d_k)        
        self.W_ks = nn.Linear(d_model, d_k)
        self.W_vs = nn.Linear(d_model, d_states)

        self.sa = Sparse_attention(top_k=self.topk)

    def forward(self, q, k, mask=None, **kwargs):
        # Shape of q = (batch_size, 2592)
        # Shape of k = (batch_size, #_imgn_states, 1024)

        k_init = k
        q = q.unsqueeze(1)                                            # Shape of q = (batch_size, 1, 2592)
        q = self.W_qs(q)                                            # Shape of q = (batch_size, 1, d_k)
        k = self.W_ks(k)                                         # Shape of k = (batch_size, #_imgn_states, d_k)

        # actions = F.one_hot(k_init[1].to(torch.int64), num_classes=self.n_actions).to(q.dtype)
        # value_net.eval()
        # values = value_net(k_init[0])                         
        v = self.W_vs(k_init)                                             # Shape of v = (batch_size, #_imgn_states, d_states)

        attn = torch.bmm(q, k.permute(0, 2, 1))
        attn = attn / self.temperature                                    # Shape of attn = (batch_size, 1, #_imgn_states)

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        # sparse_attn = self.sa(attn.squeeze(1)).unsqueeze(1)                            # Shape of sparse_attn = (batch_size, 1, #_imgn_states)
        # final_attn = torch.bmm(sparse_attn, v).squeeze(1)                             # Shape of final_attn = (batch_size, d_states)
        final_attn = torch.bmm(attn, v).squeeze(1)                             # Shape of final_attn = (batch_size, d_states)

        return final_attn, attn