from threading import Thread
import numpy as np
import torch
import torch.nn as nn
from models import RolloutEncoder, SelfAttentionEncoder

class I2AStyle(nn.Module):
    def __init__(self, config):
        super(I2AStyle, self).__init__()
        self.similar = config["similar"]
        self.rollout_encoders = nn.ModuleList(RolloutEncoder(config) for _ in range(config["similar"]))

    def encode_img_experience(self, dream, encoder, code, i):
        dream_features = torch.flip(dream["features_pred"], [0])
        dream_rewards = torch.flip(dream["reward_pred"], [0])
        code[f"forward_{i}"] = encoder(dream_features, dream_rewards)

    def forward(self, dreams, **kwargs):
        code = {}
        summ_threads = [Thread(target=self.encode_img_experience, args=(dreams[f"forward_{i}"], self.rollout_encoders[i], code, i)) for i in range(self.similar)]

        for l in range(self.similar): summ_threads[l].start()
        for m in range(self.similar): summ_threads[m].join()

        imgn_code = torch.cat(([code[f"forward_{k}"] for k in range(self.similar)]), dim=1)

        return imgn_code

class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.similar = config["similar"]
        self.encoders = nn.ModuleList(SelfAttentionEncoder(config) for _ in range(self.similar))
        self.d_k = config["sa_dk"]
        self.d_v = config["enc_out_size"]
        self.W_qs = nn.Linear(config["enc_out_size"], self.d_k, bias=False)
        self.W_ks = nn.Linear(2049, self.d_k, bias=False)
        self.W_vs = nn.Linear(2049, self.d_v, bias=False)

    def encode_imgn_experience(self, dream, encoder, code, i):
        dream_features = dream["features_pred"]
        dream_rewards = dream["reward_pred"]
        code[f"forward_{i}"] = encoder(dream_features, dream_rewards)

    def forward(self, dreams, state, **kwargs):
        code = {}
        summ_threads = [Thread(target=self.encode_imgn_experience, args=(dreams[f"forward_{i}"], self.encoders[i], code, i)) for i in range(self.similar)]

        for l in range(self.similar): summ_threads[l].start()
        for m in range(self.similar): summ_threads[m].join()

        imgn_code = torch.cat(([code[f"forward_{k}"] for k in range(self.similar)]), dim=0)
        t, b, f = imgn_code.shape
        imgn_code = imgn_code.permute(1, 0, 2)
        imgn_code = imgn_code.reshape(b*t, f)

        q = self.W_qs(state)
        k = self.W_ks(imgn_code)
        v = self.W_vs(imgn_code)

        q = q.unsqueeze(1)
        k = k.reshape(b, t, self.d_k)
        v = v.reshape(b, t, self.d_v)

        attn_weights = nn.Softmax(dim=2)(torch.bmm(q, k.transpose(1, 2)) / np.power(self.d_k, 0.5))
        attn = torch.bmm(attn_weights, v).squeeze(1)

        return attn

_summarizer_dict = {
    'i2a': I2AStyle,
    'self-attention': SelfAttention
}

def get_summarizer(name):
    return _summarizer_dict[name]

