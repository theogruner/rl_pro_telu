import torch.nn.functional as F
import torch.nn as nn
import torch

import gym
import numpy as np

LAYER_1 = 400
LAYER_2 = 300


class Critic(nn.Module):

    def __init__(self, state_shape, action_shape):
        super(Critic, self).__init__()
        self.lin1 = nn.Linear(state_shape, LAYER_1, True)
        self.lin2 = nn.Linear(LAYER_1 + action_shape, LAYER_2, True)
        self.lin3 = nn.Linear(LAYER_2, 1, True)

    def forward(self, state, action):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(torch.cat((x, action))))
        x = F.relu(self.lin3(x))
        return x

# env = gym.make('CartPole-v0')
# observation_space = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete else env.observation_space.shape[0]
# action_space = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
# print(observation_space, action_space)
# critic = Critic(observation_space, action_space)
# env.reset()
# state = torch.from_numpy(env.reset())
# print(state)
# action = torch.tensor(np.array([env.action_space.sample()]))
# print(action.dtype)
# action = np.array([1])
# action = torch.from_numpy(action).float()
# print(type(action))
# type(action)
# critic.forward(torch.from_numpy(env.reset()).float(),action)
# print(critic.forward(state, action))

