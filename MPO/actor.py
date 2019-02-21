import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal
import numpy as np
import gym
import quanser_robots

LAYER_1 = 100
LAYER_2 = 100


class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.state_shape = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete else env.observation_space.shape[0]
        self.action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)
        print(self.action_range)
        self.lin1 = nn.Linear(self.state_shape, LAYER_1, True)
        self.lin2 = nn.Linear(LAYER_1, LAYER_2, True)
        # TODO output: µ(s) and cholesky factor A(s)
        self.mean_layer = nn.Linear(LAYER_2, self.action_shape, True)
        self.cholesky_layer = nn.Linear(LAYER_2, int((self.action_shape*self.action_shape + self.action_shape)/2), True)

        # self.lin3 = nn.Linear(LAYER_2, output_shape, True)
        # self.mean = torch.zeros(self.action_shape)
        self.cholesky = torch.zeros(self.action_shape,self.action_shape)

    def forward(self, states):
        x = F.relu(self.lin1(states))
        x = F.relu(self.lin2(x))
        mean = torch.tanh(self.mean_layer(x))
        cholesky_vector = F.softplus(self.cholesky_layer(x))
        cholesky_list = []
        for l in range(len(states)):
            k = 0
            cholesky = torch.zeros(self.action_shape,self.action_shape)
            for i in range(self.action_shape):
                for j in range(self.action_shape):
                    if i >= j:
                        cholesky[i][j] = cholesky_vector[l][k].item()
                        k = k + 1
            cholesky_list.append(cholesky)
        return mean, cholesky_list

    def action(self, observation):
        mean, cholesky_list = self.forward(observation)
        print(mean[0])
        action_distribution = []
        for i  in range(len(cholesky_list)):
            action_distribution.append(self.action_range * MultivariateNormal(mean[i], scale_tril=cholesky_list[i]).sample())
        return action_distribution

env = gym.make('Qube-v0')
# env = gym.make('BallBalancerSim-v0')
# print(env.action_space)
# observation = env.reset()
# observation2 = env.reset()
observation = [env.reset(), env.reset()]
# print(type(observation))
policy = Actor(env)
additional_action = []
bdditional_action = []
for i in range(10):
    a = policy.action(torch.tensor(observation).float())
    additional_action.append(a)
# for i in range(100):
#     b = policy.action(torch.tensor(observation).float())
#     bdditional_action.append(b.numpy())
# cditional_action = [additional_action, bdditional_action]
# c = np.asarray(additional_action)
# a = policy.action(torch.tensor(observation).float())
print(additional_action)