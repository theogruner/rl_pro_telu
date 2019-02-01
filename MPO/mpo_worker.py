import torch.nn.functional as F
import torch
import numpy as np


from critic import Critic
from actor import Actor

from buffer import ReplayBuffer

class Mpo_worker(object):
    def __init__(self, epsilon, epsilon_mu, epsilon_sigma, l_max, gamma, env):
        self.EPSILON = epsilon
        self.EPSILON_MU = epsilon_mu
        self.EPSILON_SIGMA = epsilon_sigma
        self.L_MAX = l_max #300
        self.GAMMA = gamma
        self.N = 10
        self.M = 10
        self.CAPACITY = 1e6

        #TODO arbetrary behaviour policy b(a)
        self.b = 1

        #initialize Q-network
        self.critic = Critic(env)
        self.target_critic = Critic(env)

        #initialize policy
        self.actor = Actor(env)

        #initialize replay buffer
        self.buffer = ReplayBuffer(self.CAPACITY)

        self.i = 0
        self.l_curr = 0
        self.eta = 0
        self.eta_mu = 0
        self.eta_sigma = 0
# Q-function gradient
    def gradient_critic(self, states, actions):
        q = self.critic(states, actions)
        q_ret  = self.target_critic(states, actions)
        loss_critic = F.mse_loss(q, q_ret)

    def retrace_critic (self, env, states, actions, rewards, additional_actions):
        c = self.calculate_ck(states, actions)
        q_ret = self.target_critic(states, actions)
        for j in range(0, self.M):
            mean_q = 1
            q_ret = q_ret + np.power(self.GAMMA, j)*c[j]*(rewards[j] + mean_q)
        return q_ret

    def calculate_ck(self, states, actions):
        c = np.ones(len(states))
        for i in range(len(states)):
            if self.actor(states[i]) > actions[i]:
                c[i] = c[i-1] * self.actor(states[i]) / actions[i]
        return c

# E-step gradient

# M-step gradient

    def worker(self, env):
        while self.l_curr < self.L_MAX:
# update replay buffer B
            for trajectories in range(10):
                observation = env.reset()
                for steps in range(300):
                    action = self.actor(observation)
                    new_observation, reward, done, _ = env.step(action)
                    self.buffer.push(observation, action, reward, new_observation)
                    observation = new_observation
        for k in range(0, 1000):
# sample a mini-batch of N state action pairs
            batch = self.buffer.sample(self.N)
# sample M additional action for each state
            additional_action = np.zeros(self.N*self.M)
            for samples in batch:
                for i in range(self.M):
                    mean, covariance = self.actor(samples.state)
                    covariance = torch.mm(covariance, covariance.t())
                    additional_action[i] = np.random.normal(mean = mean.detach().numpy(), cov = covariance.detach().numpy())
            batch_samples = 1
            batch_actions = 1

            self.gradient_critic(batch_samples, batch_actions)
