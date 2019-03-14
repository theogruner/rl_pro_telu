import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal


class Actor(nn.Module):
    """
    Policy network
    :param env: gym environment for state and action shapes
    :param layer1: (int) size of the first hidden layer (default = 100)
    :param layer2: (int) size of the first hidden layer (default = 100)
    """
    def __init__(self, env, layer1=100, layer2=100):
        super(Actor, self).__init__()
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)
        self.lin1 = nn.Linear(self.state_shape, layer1, True)
        self.lin2 = nn.Linear(layer1, layer2, True)
        self.mean_layer = nn.Linear(layer2, self.action_shape, True)
        self.cholesky_layer = nn.Linear(layer2, self.action_shape, True)
        # self.cholesky_layer = nn.Linear(LAYER_2, int((self.action_shape*self.action_shape + self.action_shape)/2),
        #                                 True)
        self.cholesky = torch.zeros(self.action_shape, self.action_shape)

    def forward(self, states):
        """
        Goes forward through the network
        :param states: ([State]) a (batch of) state(s) of the environment
        :return: ([float])([float]) output of the network(= mean and
                 cholesky factorization chosen by policy at given state)
        """
        x = F.relu(self.lin1(states))
        x = F.relu(self.lin2(x))
        mean = self.action_range * torch.tanh(self.mean_layer(x))
        cholesky_vector = F.softplus(self.cholesky_layer(x))

        cholesky = torch.stack([σ * torch.eye(self.action_shape) for σ in cholesky_vector])
        # cholesky = []
        # if cholesky_vector.dim() == 1 and cholesky_vector.shape[0] > 1:
        #     cholesky.append(self.to_cholesky_matrix(cholesky_vector))
        # else:
        #     for a in cholesky_vector:
        #         cholesky.append(self.to_cholesky_matrix(a))
        # cholesky = torch.stack(cholesky)
        return mean, cholesky

    def action(self, state):
        """
        Approximates an action by going forward through the network
        :param state: (State) a state of the environment
        :return: (float) an action of the action space
        """
        with torch.no_grad():
            mean, cholesky = self.forward(state)
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action = action_distribution.sample()
        return action

    def eval_step(self, state):
        """
        Approximates an action based on the mean output of the network
        :param state: (State) a state of  the environment
        :return: (float) an action of the action space
        """
        with torch.no_grad():
            action, _ = self.forward(state)
        return action

    def to_cholesky_matrix(self, cholesky_vector):
        """
        Lower triangular matrix
        :param cholesky_vector: ([float]) vector with n items
        :return: ([[float]]) Square Matrix containing the entries of the
                 vector
        """
        k = 0
        cholesky = torch.zeros(self.action_shape, self.action_shape)
        for i in range(self.action_shape):
            for j in range(self.action_shape):
                if i >= j:
                    cholesky[i][j] = cholesky_vector.item() if self.action_shape == 1 else cholesky_vector[k].item()
                    k = k + 1
        return cholesky
