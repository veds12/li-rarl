import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

def VanillaMLP(
    self, 
    layer_sizes,
    activation,
    dropout,
    ):
    
    if activation == 'relu':
        _activation = nn.relu()
    elif activation == 'tanh':
        _activation = nn.tanh()
    else:
        raise NotImplementedError

    _layers = []

    for i in range(len(layer_sizes)-2):
        _layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]), act=_activation)
    
    _layers.append(nn.Dropout(dropout))
    _layers.append(nn.Linear(layer_sizes[-2], layer_sizes), act=nn.Identity())

    return nn.Sequential(*_layers)

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, input_shape, out_size):
        self.model = models.resnet18(pretrained=False)
        self.preprocess = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.linear = nn.Linear(1000, out_size)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)
        x = self.linear(x)
        return x
        

            


        