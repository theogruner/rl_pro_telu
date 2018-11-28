import torch.nn.functional as F
import torch
import numpy as np
import gym

LAYER_1 = 300
LAYER_2 = 400

class Critic(object):

    def __init__(self, state_shape, action_shape):
        self.weights1 = np.random.rand(state_shape + 1, LAYER_1)
        self.weights2 = np.random.rand(action_shape + LAYER_1 + 1, LAYER_2)
        self.weightsOutput = np.random.rand(LAYER_2 + 1, 1)

    def forward(self, state, action):
        state_biased = np.append(-1, state)
        bias_tensor = torch.tensor([-1], dtype=torch.float64)
        action_tensor = torch.tensor([action], dtype=torch.float64)

        x = F.relu(torch.from_numpy(np.dot(state_biased, self.weights1)))
        #x = np.append(-1, x, action)
        x = torch.cat((bias_tensor, x, action_tensor))
        x = F.relu(torch.from_numpy(np.dot(x.numpy(), self.weights2)))
        x = torch.cat((bias_tensor, x))
        x = torch.tanh(torch.from_numpy(np.dot(x.numpy(), self.weightsOutput)))
        return x.numpy()


spasti = gym.make('CartPole-v0')
hurensohn = Critic(spasti.observation_space.shape[0], 1)
print(hurensohn.forward(spasti.reset(), 0))
#print(hurensohn.forward(spasti.reset(), 1))
