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
    """
    def __init__(self, state_shape, action_shape, layer1=400, layer2=300):
        super(Actor, self).__init__()
        self.state_norm = nn.BatchNorm1d(state_shape)
        self.lin1 = nn.Linear(state_shape, layer1, True)
        self.norm1 = nn.BatchNorm1d(layer1)
        self.lin2 = nn.Linear(layer1, layer2, True)
        self.norm2 = nn.BatchNorm1d(layer2)
        self.lin3 = nn.Linear(layer2, action_shape, True)

    def forward(self, state):
        """
        Forward function for training
        :param state: ([State]) a (batch of) state(s) of the environment
        :return: (float) output of the network(= action chosen by policy at
                  given state)
        """
        #s = self.state_norm(state)
        x = F.relu(self.lin1(state))
        #x = self.norm1(x)
        x = F.relu(self.lin2(x))
        #x = self.norm2(x)
        x = torch.tanh(self.lin3(x))
        return x

    def action(self, state):
        """
        Forward function for action selection
        :param state: (State) a state of the environment
        :return: (float) output of the network(= action chosen by policy at
                  given state)
        """
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        return x
