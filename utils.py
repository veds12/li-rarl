import math
import numpy as np
import torch
import torch.nn as nn

def to_onehot(x: np.ndarray, n_categories) -> np.ndarray:
    e = np.eye(n_categories, dtype=np.float32)
    return e[x]  # Nice trick: https://stackoverflow.com/a/37323404
    
class Sparse_grad_attention(torch.autograd.Function):
    # def __init__(self, top_k):
    #     super(Sparse_grad_attention,self).__init__()
    
    #     self.sa = Sparse_attention(top_k=top_k)

    @staticmethod
    def forward(ctx, inp, sa):
        sparsified = sa(inp)
        ctx.save_for_backward(inp, sparsified)

        return inp

    @staticmethod
    def backward(ctx, grad_output):
        inp, sparsified = ctx.saved_tensors
        # print('sparsified', sparsified)
        return (grad_output) * (sparsified > 0.0).float()

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
        delta = torch.topk(attn_s, self.top_k, dim=2)[0][:, :, -1].permute(1, 0)
        attn_w = attn_s - delta.repeat(1, 1, attn_s.shape[-1])
        attn_w = torch.clamp(attn_w, min = 0).squeeze(0)
        attn_w_sum = torch.sum(attn_w, dim = 1, keepdim=True)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, attn_s.shape[-1])

        return attn_w_normalize

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, topk, grad_sparse=False, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        #self.dropout = nn.Dropout(attn_dropout)
        self.grad_sparse = grad_sparse
        self.softmax = nn.Softmax(dim=2)
        self.topk = topk
        self.sa = Sparse_attention(top_k=topk) #k=2

    def forward(self, q, k, mask=None):

        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward of Scaled Dot Product Attention ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("q: ", q.size())
        # print("k: ", k.size())
        # print("v: ", v.size())
        # print("k transpose: ", k.transpose(1,2).size())
        # input()

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        #print('in forward attn shape', attn.shape)

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)

        extra_loss = 0.0

        if self.grad_sparse:
            sga = Sparse_grad_attention(self.topk)
            sparse_attn = sga(attn)
        else:
            sparse_attn = self.sa(attn)
            attn = sparse_attn*1.0

        return attn, extra_loss
