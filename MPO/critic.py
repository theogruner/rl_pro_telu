import torch.nn.functional as F
import torch.nn as nn
import torch.tensor
import torch
import numpy as np

LAYER_1 = 200
LAYER_2 = 200


class Critic(nn.Module):

    def __init__(self, state_shape, action_shape):
        super(Critic, self).__init__()
        self.lin1 = nn.Linear(state_shape, LAYER_1, True)
        self.lin2 = nn.Linear(LAYER_1 + action_shape, LAYER_2, True)
        self.lin3 = nn.Linear(LAYER_2, 1, True)

    def forward(self, state, action):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(torch.cat((x, torch.from_numpy(action)),0)))
        x = F.relu(self.lin3(x))
        return x

critic = Critic(1,1)
a = critic.forward(0.2,np.array([0.4]))