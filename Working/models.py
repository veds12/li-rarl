import torch
import torch.nn as nn
from torchvision import transforms
from utils import positional_encoding

def VanillaMLP(
    layer_sizes,
    activation,
):

    if activation == "relu":
        _activation = nn.ReLU()
    elif activation == "tanh":
        _activation = nn.Tanh()
    else:
        raise NotImplementedError

    _layers = [_activation]

    for i in range(len(layer_sizes) - 2):
        _layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), _activation]

    _layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]

    return nn.Sequential(*_layers)


class ConvEncoder(nn.Module):
    def __init__(self, config):
        super(ConvEncoder, self).__init__()
        in_channels = config["in_channels"]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = nn.Linear(64*7*7, out_features=config["enc_out_size"])
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.preprocess = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485], std=[0.229]
                ),
            ]
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.linear(x)
        
        return x

class RolloutEncoder(nn.Module):
    def __init__(self, config):
        super(RolloutEncoder, self).__init__()
        self._input_size = (
            2048 + 1
        )  # deter_state + imag_reward; fix and use config["deter_dim"] + 1
        self._hidden_size = config["rollout_enc_size"]
        self._lstm = nn.LSTM(self._input_size, self._hidden_size, bias=True)

    def forward(self, dream_features, dream_rewards):
        input = torch.cat((dream_features, dream_rewards), dim=-1)
        encoding, (h_n, c_n) = self._lstm(input)
        code = h_n.squeeze(0)
        return code

class SelfAttentionEncoder(nn.Module):
    def __init__(self, config):
        super(SelfAttentionEncoder, self).__init__()
        self._input_size = 2049
        self._attention = nn.MultiheadAttention(embed_dim=self._input_size, num_heads=1)
        self._layer_norm = nn.LayerNorm(self._input_size)
        self._pos_enc = torch.tensor(positional_encoding(config["imgn_length"], self._input_size))
        self._pos_enc = self._pos_enc.permute(1, 0, 2)

    def forward(self, dream_features, dream_rewards):
        input = torch.cat((dream_features, dream_rewards), dim=-1)
        input = input + self._pos_enc.repeat(1, input.shape[1], 1).to(input.device).to(input.dtype)
        output, _ = self._attention(query = input, key = input, value = input)
        output = self._layer_norm(input + output)
        return output
