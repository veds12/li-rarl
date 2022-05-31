import math
import numpy as np
import torch
import torch.nn as nn

def to_onehot(x: np.ndarray, n_categories) -> np.ndarray:
    e = np.eye(n_categories, dtype=np.float32)
    return e[x]  # Nice trick: https://stackoverflow.com/a/37323404

def get_angles(pos, i, d_model):
    # Code from https://www.tensorflow.org/tutorials/text/transformer
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    # Code from https://www.tensorflow.org/tutorials/text/transformer
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding

class Sparse_attention(nn.Module):
    '''
    Adapted from https://github.com/nke001/sparse_attentive_backtracking_release/
    '''
    def __init__(self, top_k = 5):
        super(Sparse_attention,self).__init__()
        top_k += 1
        self.top_k = top_k

    def forward(self, attn_s):
        eps = 10e-8
        delta = torch.topk(attn_s, self.top_k, dim=1)[0][:, -1].reshape(attn_s.shape[0], 1)  # (batch_size, 1)
        attn_w = attn_s - delta.repeat(1, attn_s.shape[1])                                   # (batch_size, sample_size)
        attn_w = torch.clamp(attn_w, min = 0)                            # (batch_size, sample_size)
        attn_w_sum = torch.sum(attn_w, dim = 1, keepdim=True)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, attn_s.shape[1])

        return attn_w_normalize         # (batch_size, sample_size)