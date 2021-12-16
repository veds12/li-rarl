import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch import autograd


def VanillaMLP(
    layer_sizes,
    activation,
    dropout,
):

    if activation == "relu":
        _activation = nn.ReLU()
    elif activation == "tanh":
        _activation = nn.Tanh()
    else:
        raise NotImplementedError

    _layers = []

    for i in range(len(layer_sizes) - 2):
        _layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), _activation]

    _layers += [nn.Dropout(dropout), nn.Linear(layer_sizes[-2], layer_sizes[-1])]

    return nn.Sequential(*_layers)


class ConvEncoder(nn.Module):
    def __init__(self, config):
        super(ConvEncoder, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(config["enc_input_size"]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.linear = nn.Linear(1000, config["enc_out_size"])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.preprocess(x)
        x = self.model(x)
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
        input = torch.cat((dream_features, dream_rewards), dim=2)
        encoding, (h_n, c_n) = self._lstm(input)
        code = h_n.squeeze(0)
        return code
