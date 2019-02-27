import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import gym
import quanser_robots
import copy

from buffer import ReplayBuffer
from ornstein_uhlenbeck_noise import OrnsteinUhlenbeck
from critic_torch import Critic
from actor_torch import Actor

# Environment
# env = gym.make('BallBalancerSim-v0')
# env = gym.make('Pendulum-v0')
env = gym.make('Qube-v0')

# optimization problem
MSE = nn.MSELoss()

# Hyperparameters
X_START, THETA, MU, SIGMA, DELTA_T = 0, 0.15, 0, 0.2, 1e-2
CAPACITY = 1e6
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001

#episodes
M = int(1e4)
#epsiode length
T = 50

def ddpg_torch(env):
    state_shape = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete else env.observation_space.shape[0]
    action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
    action_range = env.action_space.high[0]
    # initialize buffer
    buffer = ReplayBuffer(CAPACITY)
    # initialize actor
    actor = Actor(state_shape=state_shape, action_shape=action_shape)
    # initialize actor optimizer
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    # initialize critic
    critic = Critic(state_shape=state_shape, action_shape=action_shape)
    # initialize critic optimizer
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    ###############################################
    # initialize target networks and copying params
    ###############################################
    target_critic = Critic(state_shape=state_shape, action_shape=action_shape)
    target_actor = Actor(state_shape=state_shape, action_shape=action_shape)

    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)
        target_param.requires_grad = False

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
        target_param.requires_grad = False

    observation = env.reset()
    for i in range(0, BATCH_SIZE):
        action = env.action_space.sample()
        new_observation, reward, done, _ = env.step(action)
        env.render()
        buffer.push(observation, action, reward, new_observation)
        observation = new_observation
        if done:
            observation = env.reset()

    for episode in range(0, M):
        noise = OrnsteinUhlenbeck(X_START, THETA, MU, SIGMA, DELTA_T,action_shape = action_shape)
        observation = env.reset()

        for t in range(1, T+1):
            action = actor.forward(torch.tensor(observation).float()).detach().numpy() + noise.x
            noise.iteration()
            #TODO action space auf output aufteilen
            action = action * action_range
            action = np.clip(action, a_min=-action_range, a_max=action_range)
            #action = action.astype(np.float32)
            #print(action)
            new_observation, reward, done, _ = env.step(action)
            env.render()

            buffer.push(observation, action, reward, new_observation)
            observation = new_observation

            sample = buffer.sample(BATCH_SIZE)
            state_batch, action_batch, reward_batch, next_state_batch = buffer.batches_from_sample(sample, BATCH_SIZE)
            state_batch, action_batch, reward_batch, next_state_batch = torch.tensor(state_batch).float(), torch.tensor(action_batch).float(), torch.tensor(reward_batch).float(), torch.tensor(next_state_batch).float()


            #y = reward_batch + GAMMA * target_critic(torch.from_numpy(next_state_batch).float(), target_actor(torch.from_numpy(next_state_batch).float()))
            y = reward_batch + GAMMA * target_critic(next_state_batch,
                                                     target_actor(next_state_batch))
            # update critic
            critic_optimizer.zero_grad()
            #target = critic(torch.from_numpy(state_batch).float(), torch.from_numpy(action_batch).float())
            target = critic(state_batch, action_batch)
            loss_critic = MSE(y, target)
            loss_critic.backward()
            critic_optimizer.step()


            #update actor
            actor_optimizer.zero_grad()
            #loss_actor = torch.zeros(1, dtype=torch.float, requires_grad = True)
            #for sample in batch:
            #    loss_actor = loss_actor + critic.forward(torch.from_numpy(sample.state).float(),actor.forward(torch.from_numpy(sample.state)).float())
            #loss_actor = critic(torch.from_numpy(state_batch).float(), actor(torch.from_numpy(state_batch).float()))
            loss_actor = critic(state_batch, actor(state_batch))
            loss_actor = -loss_actor.mean()#/len(batch)
            loss_actor.backward()
            actor_optimizer.step()

            #update parameter
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)



ddpg_torch(env)
