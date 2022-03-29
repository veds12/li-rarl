import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Conv2dModel, TransitionModel
import random
from torch.distributions.categorical import Categorical

class SPR(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(SPR, self).__init__()
        self.encoder = Conv2dModel(
            in_channels=config['in_channels'],
            channels=config['conv_channels'],
            kernel_sizes=config['kernel_sizes'],
            strides=config['strides'],
            paddings=config['paddings'],
            use_maxpool=config['use_maxpool'],
            dropout=config['dropout'],
        )

        self.dynamics_model = TransitionModel(
            channels=config['dyn_channels'],
            num_actions=config['num_actions'],
            pixels=config['pixels'],
            hidden_size=config['hidden_size'],
            limit=config['limit'],
            blocks=config['blocks'],
            norm_type=config['norm_type'],
            renormalize=config['renormalize'],
            residual=config['residual'],
        )

        self.n_actions = config['n_actions']
        
        self.traj_encoder = nn.Sequential(
            nn.Flatten(3, 5),
            nn.Linear(3136, 2048),
            nn.ReLU(),
            nn.Linear(2048, config['traj_enc_size']),
        )

    def load_chkpts(self, dyn_chkpt_path, enc_chkpt_path):
        self.dynamics_model.load_state_dict(torch.load(dyn_chkpt_path))
        self.encoder.load_state_dict(torch.load(enc_chkpt_path))

        self.dynamics_model.eval()
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.dynamics_model.parameters():
            param.requires_grad = False

    def rollout(self, x, steps=20, num_trajs=16):
        
        state_trajs = []
        action_trajs = []
        reward_trajs = []
        
        for _ in range(num_trajs):
            states = []
            rewards = []
            actions = []
            state = x
            latent = self.encoder(state)
            states.append(latent)
            for _ in range(steps):

                action = torch.randint(0, self.n_actions-1, (x.shape[0],)).long().to(x.device)
                actions.append(action)

                next_state, pred_rew = self.dynamics_model(latent, action)
                states.append(next_state)
                rewards.append(pred_rew)

                latent = next_state

            states = torch.stack(states, 0)
            #state_trajs.append(self.encoder(states))
            state_trajs.append(states)
            reward_trajs.append(torch.stack(rewards, 0))
            action_trajs.append(torch.stack(actions, 0))

        state_trajs = torch.stack(state_trajs, 0)
        action_trajs = torch.stack(action_trajs, 0).unsqueeze(-1)
        reward_trajs = torch.stack(reward_trajs, 0)

        state_trajs = self.traj_encoder(state_trajs)        
        
        return state_trajs, reward_trajs, action_trajs
        

forward_model_ = {
    "SPR": SPR,
}

def get_forward(forward_type):
    return forward_model_[forward_type]