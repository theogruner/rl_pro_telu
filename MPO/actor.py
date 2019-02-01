import torch.nn.functional as F
import torch.nn as nn
import torch

import gym
import quanser_robots

LAYER_1 = 100
LAYER_2 = 100


class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.state_shape = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete else env.observation_space.shape[0]
        self.action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
        self.lin1 = nn.Linear(self.state_shape, LAYER_1, True)
        self.lin2 = nn.Linear(LAYER_1, LAYER_2, True)
        # TODO output: µ(s) and cholesky factor A(s)
        self.mean_layer = nn.Linear(LAYER_2, self.action_shape, True)
        self.cholesky_layer = nn.Linear(LAYER_2, int((self.action_shape*self.action_shape + self.action_shape)/2), True)

        # self.lin3 = nn.Linear(LAYER_2, output_shape, True)
        # self.mean = torch.zeros(self.action_shape)
        self.cholesky = torch.zeros(self.action_shape,self.action_shape)

    def forward(self, state):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        mean = torch.tanh(self.mean_layer(x))
        cholesky_vector = F.softplus(self.cholesky_layer(x))
        # tuple_x = x.split(self.action_shape-1)
        # for i in range(self.action_shape):
        #     self.mean[i] = x[i]
        k = 0
        for i in range(self.action_shape):
            for j in range(self.action_shape):
                if i >= j:
                    self.cholesky[i][j] = cholesky_vector[k].item()
                    k = k + 1
        return mean, self.cholesky

# env = gym.make('Qube-v0')
env = gym.make('BallBalancerSim-v0')
observation = env.reset()
policy = Actor(env)
a, b = policy.forward(torch.tensor(observation).float())
print(b)