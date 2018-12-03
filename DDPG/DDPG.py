import torch.nn.functional as F
import torch
import numpy as np
import gym

from buffer import ReplayBuffer
from Critic import Critic
from Actor_policy import ActorPolicy
from Ornstein_uhlenbeck_noise import OrnsteinUhlenbeck


class DDPG(object):
    def __init__(self, obs_space, act_space):

        self.obs_shape = 1 if type(obs_space) == gym.spaces.discrete.Discrete else obs_space.shape[0]
        self.act_shape = 1 if type(act_space) == gym.spaces.discrete.Discrete else act_space.shape[0]

        #initalize buffer
        self.buffer = ReplayBuffer(capacity)
        #initalize actor
        self.actor = ActorPolicy(self.obs_shape, self.act_shape)
        #initilize actor optimizer
        self.actor_optimizer = torch.optim.Adam([self.actor.weights1, self.actor.weights2, self.actor.weightsOutput], lr=1e-3)
        #initalize critic
        self.critic = Critic(self.obs_shape, self.act_shape)
        #initilize critic optimizer
        self.critic_optimizer = torch.optim.Adam([self.critic.weights1, self.critic.weights2, self.critic.weightsOutput], lr=1e-3)

    def select_action(self, action):
        self.action = action +  OrnsteinUhlenbeck(x_start, theta, mu, sigma, deltat)

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

# Hyperparameters
X_START, THETA, MU, SIGMA, DELTA_T = 0, 0.15, 0, 0.2, 1e-2
CAPACITY = 1e6
BATCH_SIZE = 64
GAMMA = 0.001
TAU = 0.001

#episodes
M = 1e3
#epsiode length
T = 42

def DDPG(env):

    state_shape = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete else env.observation_space.shape[0]
    action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]

    # initalize buffer
    buffer = ReplayBuffer(CAPACITY)
    # initalize actor
    actor = ActorPolicy(state_shape, action_shape)
    # initilize actor optimizer
    actor_optimizer = torch.optim.Adam([actor.weights1, actor.weights2, actor.weightsOutput], lr=1e-3)
    # initalize critic
    critic = Critic(state_shape= state_shape, action_shape=action_shape)
    # initilize critic optimizer
    critic_optimizer = torch.optim.Adam([critic.weights1, critic.weights2, critic.weightsOutput], lr=1e-3)

    target_critic = critic;
    target_actor = actor;


    observation = env.reset()
    for i in range(0, BATCH_SIZE):
        action = env.action_space.sample()
        new_observation, reward, done, _ = env.step(action)
        buffer.push(observation, action, reward, new_observation)
        observation = new_observation
        if done:
            observation = env.reset()



    for episode in range(1, M+1):
        noise = OrnsteinUhlenbeck(X_START, THETA, MU, SIGMA, DELTA_T).x
        observation = env.reset()

        for t in range(1, T+1):
            action = actor.forward(observation).numpy() + noise.x
            noise.iteration()
            #TODO action space auf output aufteilen
            action = action * 3
            new_observation, reward, done, _= env.step(action)

            buffer.push(observation, action, reward, new_observation)
            observation = new_observation

            batch = buffer.sample(BATCH_SIZE)



