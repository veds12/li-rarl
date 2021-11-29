import torch
import torch.nn as nn

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


            


        