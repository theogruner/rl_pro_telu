import torch.nn.functional as F
import torch.nn as nn
import torch


class Actor(nn.Module):
    """
    Actor class for the policy neural-network
    :param state_shape: (int) shape(= dimensions) of an observation(state)
    :param action_shape: (int) shape(= dimensions) of an action
    :param layer1: (int) size of the first hidden layer (default = 400)
    :param layer2: (int) size of the second hidden layer (default = 300)
    :param norm: (bool) flag if to normalize or not
    """
    def __init__(self, state_shape, action_shape, layer1=400, layer2=300, norm=True):
        super(Actor, self).__init__()
        self.norm = norm
        if norm:
            self.state_norm = nn.BatchNorm1d(state_shape)
            self.norm1 = nn.BatchNorm1d(layer1)
            self.norm2 = nn.BatchNorm1d(layer2)
        self.lin1 = nn.Linear(state_shape, layer1, True)
        self.lin2 = nn.Linear(layer1, layer2, True)
        self.lin3 = nn.Linear(layer2, action_shape, True)

    def forward(self, state):
        """
        Forward function forwarding input through the network
        :param state: ([State]) a (batch of) state(s) of the environment
        :return: (float) output of the network(= action chosen by policy at
                  given state)
        """
        s = self.state_norm(state) if self.norm else state
        x = self.norm1(self.lin1(s)) if self.norm else self.lin1(s)
        x = F.relu(x)
        x = self.norm2(self.lin2(x)) if self.norm else self.lin2(x)
        x = F.relu(x)
        x = torch.tanh(self.lin3(x))
        return x
