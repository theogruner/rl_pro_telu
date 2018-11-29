import torch.nn.functional as F
import torch
import numpy as np
import gym

LAYER_1 = 300
LAYER_2 = 400

class Actor_policy(object):
    def __init__(self, state_shape, action_shape):
        self.weights1 = np.random.rand(LAYER_1, state_shape + 1)
        self.weights2 = np.random.rand(LAYER_2, action_shape + LAYER_1 + 1)
        self.weightsOutput = np.random.rand(1, LAYER_2 + 1)

        self.weights1 = torch.from_numpy(self.weights1)
        self.weights2 = torch.from_numpy(self.weights2)
        self.weightsOutput = torch.from_numpy(self.weightsOutput)
        self.weights1.requires_grad = True
        self.weights2.requires_grad = True
        self.weightsOutput.requires_grad = True

    def forward(self, state, action):
        one = torch.tensor([-1], dtype=torch.float64)
        state_biased = torch.cat((one, torch.from_numpy(state),torch.from_numpy(action)))
        bias_tensor = torch.tensor([-1], dtype=torch.float64)
        action_tensor = torch.tensor([action], dtype=torch.float64)
        x = F.relu(torch.mv(self.weights1, state_biased))
        x = torch.cat((bias_tensor, x))
        x = F.relu(torch.mv(self.weights2, x))
        x = torch.cat((bias_tensor, x))
        x = torch.tanh(torch.mv(self.weightsOutput, x))
        return x

    def backprop(self, gradQ, grad):
        optimizer = torch.optim.Adam([self.weights1, self.weights2, self.weightsOutput], lr=1e-3)