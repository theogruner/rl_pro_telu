import torch.nn.functional as F
import torch.nn as nn
import torch

import gym
import quanser_robots
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

#env = gym.make('Qube-v0')
#observation_space = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete else env.observation_space.shape[0]
#action_space = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
#print(observation_space, action_space)
#critic = Critic(observation_space, action_space)
#for i in range(0, 20):
#    state = torch.from_numpy(env.reset())
#    print(state)
#    action = torch.tensor(env.action_space.sample())
#    print(action)
#    print(critic.forward(state.float(), action.float()))

