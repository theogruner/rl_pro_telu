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
M = int(1e3)
#epsiode length
T = 200

def ddpg_torch(env):
    state_shape = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete else env.observation_space.shape[0]
    action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]

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
            action = actor.forward(torch.tensor(observation)).detach().numpy() + noise.x
            noise.iteration()
            #TODO action space auf output aufteilen
            action = action * 5
            action = np.clip(action, a_min=-5, a_max=5)
            action = action.astype(np.float32)
            new_observation, reward, done, _ = env.step(action)
            env.render()

            #print(actor([torch.tensor(observation).float(), torch.tensor(env.reset()).float()]))
            #print(action)
            #print(critic.forward(torch.from_numpy(observation).float(), torch.tensor(env.action_space.sample()).float()))

            buffer.push(observation, action, reward, new_observation)
            observation = new_observation

            sample = buffer.sample(BATCH_SIZE)
            state_batch, action_batch, reward_batch, next_state_batch = buffer.batches_from_sample(sample, BATCH_SIZE)

            #print(critic(torch.from_numpy(state_batch).float(), torch.from_numpy(action_batch).float()))
            #print(actor(torch.from_numpy(state_batch)).float())

            y = reward_batch + GAMMA * target_critic(torch.from_numpy(next_state_batch).float(), target_actor(torch.from_numpy(next_state_batch).float()))
            # update critic
            critic_optimizer.zero_grad()
            #y = torch.zeros(BATCH_SIZE, dtype=torch.float)
            #target = torch.zeros(BATCH_SIZE, dtype=torch.float)
            #i = 0
            #for sample in batch:
            #    y[i] = sample.reward + GAMMA * target_critic.forward(torch.tensor(sample.nextState).float(), target_actor.forward(torch.tensor(sample.nextState).float()).float())
            #    target[i] = critic.forward(torch.tensor(sample.state).float(), torch.tensor(sample.action).float())
            #    i = i + 1
            target = critic(torch.from_numpy(state_batch).float(), torch.from_numpy(action_batch).float())
            loss_critic = MSE(y, target)
            #loss_critic = F.mse_loss(y, target)
            loss_critic.backward()
            critic_optimizer.step()


            #update actor
            actor_optimizer.zero_grad()
            #loss_actor = torch.zeros(1, dtype=torch.float, requires_grad = True)
            #for sample in batch:
            #    loss_actor = loss_actor + critic.forward(torch.from_numpy(sample.state).float(),actor.forward(torch.from_numpy(sample.state)).float())
            loss_actor = critic(torch.from_numpy(state_batch).float(), actor(torch.from_numpy(state_batch).float()))
            loss_actor = loss_actor.mean()#/len(batch)
            loss_actor.backward()
            actor_optimizer.step()

            #update parameter
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)



ddpg_torch(env)