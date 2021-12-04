import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

def VanillaMLP(
    layer_sizes,
    activation,
    dropout,
    ):
    
    if activation == 'relu':
        _activation = nn.ReLU()
    elif activation == 'tanh':
        _activation = nn.Tanh()
    else:
        raise NotImplementedError

    _layers = []

    for i in range(len(layer_sizes)-2):
        _layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), _activation]
    
    _layers += [nn.Dropout(dropout), nn.Linear(layer_sizes[-2], layer_sizes[-1])]

    return nn.Sequential(*_layers)

class ConvEncoder(nn.Module):
    def __init__(self, config):
        super(ConvEncoder, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.preprocess = transforms.Compose([
            transforms.Resize(config["enc_input_size"]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

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
        self._input_size = 2048 + 1                  # deter_state + imag_reward; fix and use config["deter_dim"] + 1
        self._hidden_size = config["rollout_enc_size"]
        self._gru = nn.GRUCell(self._input_size, self._hidden_size, bias=True)

    def forward(self, dream_buffer):
        hidden = torch.zeros(1, self._hidden_size).to(torch.device('cuda')).to(torch.float32)    # TODO: Remove hard coding for device and dtype
        #i = 0
        while len(dream_buffer) != 0:
            #print(i)
            transition = dream_buffer.pop()
            input = torch.cat((transition['features'], transition['reward'].unsqueeze(0)), dim=1)
            #print(f'Shape of input features in {input.shape}, shape of hidden state is {hidden.shape}')
            #print(f"Device of input: {input.device}, Device of hidden: {hidden.device}")
            hidden = self._gru(input, hidden)
            #print(f'Shape of output hidden is {hidden.shape}')
            #i += 1
        
        return hidden
