import torch.nn.functional as F
import torch
import numpy as np
import gym

LAYER_1 = 400
LAYER_2 = 300


class ActorPolicy(object):
    def __init__(self, state_shape, action_shape):
        self.weights1 = np.random.rand(LAYER_1, state_shape + 1)
        self.weights2 = np.random.rand(LAYER_2, LAYER_1 + 1)
        self.weightsOutput = np.random.rand(action_shape, LAYER_2 + 1)

        self.weights1 = torch.from_numpy(self.weights1)
        self.weights2 = torch.from_numpy(self.weights2)
        self.weightsOutput = torch.from_numpy(self.weightsOutput)
        self.weights1.requires_grad = True
        self.weights2.requires_grad = True
        self.weightsOutput.requires_grad = True

    def forward(self, state):
        bias_tensor = torch.tensor([-1], dtype=torch.float64)
        state_biased = torch.cat((bias_tensor, torch.from_numpy(state)))
        x = F.relu(torch.mv(self.weights1, state_biased))
        x = torch.cat((bias_tensor, x))
        x = F.relu(torch.mv(self.weights2, x))
        x = torch.cat((bias_tensor, x))
        x = torch.tanh(torch.mv(self.weightsOutput, x))
        return x

    # def backprop(self, gradQ, grad):
    #     optimizer = torch.optim.Adam([self.weights1, self.weights2, self.weightsOutput], lr=1e-3)