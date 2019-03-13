import torch.nn.functional as F
import torch.nn as nn
import torch


class Actor(nn.Module):
    """
    Actor class for the policy neural-network
    :param state_shape: (int) shape(= dimensions) of an observation(state)
    :param action_shape: (int) shape(= dimensions) of an action
    :param layer1: (int) size of the first hidden layer (default = 400)
    :param layer2: (int) size of the second hidden layer (default = 300)
    """
    def __init__(self, state_shape, action_shape, layer1=400, layer2=300):
        super(Actor, self).__init__()
        #self.state_norm = nn.BatchNorm1d(state_shape)
        self.lin1 = nn.Linear(state_shape, layer1, True)
        #self.norm1 = nn.BatchNorm1d(layer1)
        self.lin2 = nn.Linear(layer1, layer2, True)
        #self.norm2 = nn.BatchNorm1d(layer2)
        self.lin3 = nn.Linear(layer2, action_shape, True)

    def forward(self, state):
        """
        Forward function for training
        :param state: ([State]) a (batch of) state(s) of the environment
        :return: (float) output of the network(= action chosen by policy at
                  given state)
        """
        #s = self.state_norm(state)
        #x = self.norm1(self.lin1(s))
        x = self.lin1(state)
        x = F.relu(x)
        #x = self.norm2(self.lin2(x))
        x = self.lin2(x)
        x = F.relu(x)
        x = torch.tanh(self.lin3(x))
        return x

    def evex(self, state):
        """
        Forward function for evaluation and exploration
        :param state: (State) a state of the environment
        :return: (float) output of the network(= action chosen by policy at
                  given state)
        """
        #s = F.batch_norm(torch.tensor([state.numpy()]),
        #                 self.state_norm.running_mean,
        #                 self.state_norm.running_var)
        #x = F.batch_norm(self.lin1(s),
        #                 self.norm1.running_mean,
        #                 self.norm1.running_var)
        x = self.lin1(state)
        x = F.relu(x)
        #x = F.batch_norm(self.lin2(x),
        #                 self.norm2.running_mean,
        #                 self.norm2.running_var)
        x=self.lin2(x)
        x = F.relu(x)
        x = torch.tanh(self.lin3(x))
        return x#[0]


#import gym
#env = gym.make('Pendulum-v0')
#obs = env.reset()
#obs1 = torch.tensor([env.reset()])
#obs2 = torch.tensor([env.reset()])

#b = torch.cat((obs1, obs2), 0)
#actor = Actor(state_shape=env.observation_space.shape[0], action_shape=env.action_space.shape[0])
#actor.eval()
#action = actor(obs)
#actor.train()
#print(action)

#norm = nn.BatchNorm1d(3)
#norm(b.float())
#print(obs)
#print(norm.running_mean)
#print(norm.running_var)
#print(F.batch_norm(torch.tensor([torch.tensor(obs)]).float(), norm.running_mean, norm.running_var))



