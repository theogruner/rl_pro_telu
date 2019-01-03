import torch.nn.functional as F
import torch.nn as nn

LAYER_1 = 100
LAYER_2 = 100


class Actor(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Actor, self).__init__()
        self.lin1 = nn.Linear(state_shape, LAYER_1, True)
        self.lin2 = nn.Linear(LAYER_1, LAYER_2, True)
        self.lin3 = nn.Linear(LAYER_2, action_shape, True)

    def forward(self, state):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        x = F.tanh(self.lin3(x))
        return x