import torch.nn.functional as F
import torch
import numpy as np
import gym

from buffer import ReplayBuffer
from Critic import Critic
from Actor_policy import Actor_policy
from Ornstein_uhlenbeck_noise import ornstein_uhlenbeck

#Hyperparameters
x_start, theta, mu, sigma, deltat = 0, 0.15, 0, 0.2
capacity = 1e6
batch_size = 64
gamma = 0.001
tau = 0.001

class DDPG(object):
    def __init__(self):
        #initalize buffer
        self.buffer = ReplayBuffer
        #initalize actor
        self.actor = Actor_policy()
        #initilize actor optimizer
        self.actor_optimizer = torch.optim.Adam([self.actor.weights1, self.actor.weights2, self.actor.weightsOutput], lr=1e-3)
        #initalize critic
        self.critic = Critic()
        #initilize critic optimizer
        self.critic_optimizer = torch.optim.Adam([self.critic.weights1, self.critic.weights2, self.critic.weightsOutput], lr=1e-3)

    def select_action(self, action):
        self.action = action +  ornstein_uhlenbeck(x_start, theta, mu, sigma, deltat)

    def execute_action(self):
        self.observation, self.reward, self.done, _ =env.step(action)
        self.buffer.push(state, action, observation, reward)

    def update_critic(self, state_batch, action_batch, reward, gamma):
        self.critic_optimizer.zero_grad()


    def update_actor_policy(self):
        self.actor_optimizer.zero_grad()
        actor_loss = mean(self.critic(state, self.actor(state)))
        actor_loss.backward()
        self.actor_optimizer.step()
        return self.actor_loss

    def update_target_network(self, tau):
