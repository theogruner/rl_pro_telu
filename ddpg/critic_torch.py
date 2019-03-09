import torch.nn.functional as F
import torch.nn as nn
import torch


class Critic(nn.Module):
    """
    Critic class for value function network
    :param state_shape: (int) shape(= dimensions) of an observation(state)
    :param action_shape: (int) shape(= dimensions) of an action
    :param layer1: (int) size of the first hidden layer (default = 400)
    :param layer2: (int) size of the first hidden layer (default = 300)
    """
    def __init__(self, state_shape, action_shape, layer1=400, layer2=300):
        super(Critic, self).__init__()
        self.state_norm = nn.BatchNorm1d(state_shape)
        self.lin1 = nn.Linear(state_shape, layer1, True)
        self.norm1 = nn.BatchNorm1d(layer1)
        self.lin2 = nn.Linear(layer1 + action_shape, layer2, True)
        self.lin3 = nn.Linear(layer2, 1, True)

    def forward(self, state, action):
        """
        Forward function forwarding an input through the network
        :param state: (State) a state of the environment
        :param action: (Action) an action of the action-space
        :return: (float) output of the network(= Q-value for the given
                  state-action pair)
        """
        x = F.relu(self.lin1(self.state_norm(state)))
        x = self.norm1(x)
        x = F.relu(self.lin2(torch.cat((x, action), 1)))
        x = self.lin3(x)
        return x

    def eval(self, state, action):
        """
        Forward function forwarding an input through the network
        :param state: (State) a state of the environment
        :param action: (Action) an action of the action-space
        :return: (float) output of the network(= Q-value for the given
                  state-action pair)
        """
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(torch.cat((x, action), 0)))
        x = self.lin3(x)
        return x
