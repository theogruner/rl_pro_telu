import torch.nn.functional as F
import torch
import numpy as np
import gym

LAYER_1 = 400
LAYER_2 = 300

class Critic(object):

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
        # self.weights1 = torch.normal(mean=0.5, std=torch.arange(state_shape + 1, LAYER_1))
        # self.weights2 = torch.normal(mean=0.5, std=torch.arange(action_shape + LAYER_1 + 1, LAYER_1))
        # self.weightsOutput = torch.normal(mean=0.5, std=torch.arange(LAYER_2 + 1, 1))


    def forward(self, state, action):
        # state_biased = np.append(-1, state)
        # bias_tensor = torch.tensor([-1], dtype=torch.float64)
        # action_tensor = torch.tensor([action], dtype=torch.float64)
        #
        # x = F.relu(torch.from_numpy(np.dot(state_biased, self.weights1)))
        # x = torch.cat((bias_tensor, x, action_tensor))
        # x = F.relu(torch.from_numpy(np.dot(x.numpy(), self.weights2)))
        # x = torch.cat((bias_tensor, x))
        # x = torch.tanh(torch.from_numpy(np.dot(x.numpy(), self.weightsOutput)))
        # return x.numpy()

        bias_tensor = torch.tensor([-1], dtype=torch.float64)
        state_biased = torch.cat((bias_tensor, torch.from_numpy(state)))
        action_tensor = torch.tensor([action], dtype=torch.float64)
        x = F.relu(torch.mv(self.weights1, state_biased))
        x = torch.cat((bias_tensor, x, action_tensor))
        x = F.relu(torch.mv(self.weights2, x))
        x = torch.cat((bias_tensor, x))
        x = torch.relu(torch.mv(self.weightsOutput, x))
        return x
        # return x.detach().numpy()

    def backprop(self, y, y_old):
        optimizer = torch.optim.Adam([self.weights1,self.weights2,self.weightsOutput], lr = 1e-3)
        loss_mse = torch.nn.MSELoss(reduction = 'sum')
        loss = loss_mse(y, y_old)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return self.weights1

spasti = gym.make('CartPole-v0')
hurensohn = Critic(spasti.observation_space.shape[0], 1)
# print(hurensohn.weights1.requires_grad)
print(hurensohn.forward(spasti.reset(), 0))

y = hurensohn.forward(spasti.reset(), 0)
print(y.requires_grad)
y_old = torch.tensor([0], dtype = torch.float64, requires_grad = True)


weights1 = hurensohn.backprop(y,y_old)
print(weights1.detach().numpy())

