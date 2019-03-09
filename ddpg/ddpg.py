import torch
import torch.nn as nn
import numpy as np
import gym
import quanser_robots

from buffer import ReplayBuffer
from noise import OrnsteinUhlenbeck
from critic_torch import Critic
from actor_torch import Actor

# Environment
# env = gym.make('BallBalancerSim-v0')
# env = gym.make('Pendulum-v0')
# env = gym.make('Qube-v0')

# optimization problem
# MSE = nn.MSELoss()

# Hyperparameters
# X_START, THETA, MU, SIGMA, DELTA_T = 0, 0.15, 0, 0.2, 1e-2
# CAPACITY = 1e6
# BATCH_SIZE = 64
# GAMMA = 0.99
# TAU = 0.001

# episodes
# M = int(1e4)
# epsiode length
# T = 50


class DDPG(object):

    def __init__(self, env, noise=None, buffer_capacity=1e6, batch_size=64,
                 gamma=0.99, tau=0.001, episodes=int(1e4), learning_rate=1e-3,
                 episode_length=60, actor_layers=None, critic_layers=None,
                 log=True, render=False, safe=True, safe_path="ddpg_model.pt"):
        # initialize env and read out shapes
        self.env = env
        self.state_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.action_range = env.action_space.high[0]
        # initialize noise/buffer/loss
        self.noise = noise if noise is not None else OrnsteinUhlenbeck(self.action_shape)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.loss = nn.MSELoss()
        # initialize some hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.episodes = episodes
        self.episode_length = episode_length
        # initialize networks and optimizer
        self.actor = Actor(self.state_shape, self.action_shape, layer1=actor_layers[0], layer2=actor_layers[1]) \
            if actor_layers is not None else Actor(self.state_shape, self.action_shape)
        self.target_actor = Actor(self.state_shape, self.action_shape, layer1=actor_layers[0], layer2=actor_layers[1]) \
            if actor_layers is not None else Actor(self.state_shape, self.action_shape)
        self.critic = Critic(self.state_shape, self.action_shape, layer1=critic_layers[0], layer2=critic_layers[1]) \
            if actor_layers is not None else Critic(self.state_shape, self.action_shape)
        self.target_critic = Critic(self.state_shape, self.action_shape, layer1=critic_layers[0], layer2=critic_layers[1]) \
            if actor_layers is not None else Critic(self.state_shape, self.action_shape)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        # fill buffer with random transitions
        self._random_trajectory(self.batch_size)
        # control/log variables
        self.episode = 0
        self.log = log
        self.render = render
        self.safe = safe
        self.safe_path = safe_path

    def _random_trajectory(self, length):
        observation = self.env.reset()
        for i in range(0, length):
            action = self.env.action_space.sample()
            new_observation, reward, done, _ = self.env.step(action)
            self.buffer.push(observation, action, reward, new_observation)
            observation = new_observation
            if done:
                observation = self.env.reset()

    def _select_action(self, observation):
        obs = torch.tensor(observation).float()
        a = self.actor.action(obs).detach().numpy() + self.noise.iteration()
        a = a * self.action_range
        a = np.clip(a, a_min=-self.action_range,
                    a_max=self.action_range)
        return a

    def _sample_batches(self, size):
        sample = self.buffer.sample(size)
        state_batch, action_batch, reward_batch, next_state_batch = \
            self.buffer.batches_from_sample(sample, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = \
            torch.tensor(state_batch).float(), torch.tensor(action_batch).float(),\
            torch.tensor(reward_batch).float(), torch.tensor(next_state_batch).float()
        return state_batch, action_batch, reward_batch, next_state_batch

    def _soft_update(self):
        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self, episodes=None, episode_length=None, render=None, safe=None, safe_path=None, log=None):
        rend = render if render is not None else self.render
        sf = safe if safe is not None else self.safe
        sf_path = safe_path if safe_path is not None else self.safe_path
        ep = episodes if episodes is not None else self.episodes
        it = episode_length if episode_length is not None else self.episode_length

        for episode in range(0, ep):
            self.noise.reset()
            observation = self.env.reset()

            for t in range(1, it + 1):
                action = self._select_action(observation)

                new_observation, reward, done, _ = self.env.step(action)
                if rend is True:
                    self.env.render()

                self.buffer.push(observation, action, reward, new_observation)
                observation = new_observation

                state_batch, action_batch, reward_batch, next_state_batch = \
                    self._sample_batches(self.batch_size)

                y = reward_batch + self.gamma * self.target_critic(next_state_batch, self.target_actor(next_state_batch))


                # update critic
                self.critic_optimizer.zero_grad()
                target = self.critic(state_batch, action_batch)
                loss_critic = self.loss(y, target)
                loss_critic.backward()
                self.critic_optimizer.step()

                # update actor
                self.actor_optimizer.zero_grad()
                loss_actor = self.critic(state_batch, self.actor(state_batch))
                loss_actor = -loss_actor.mean()
                loss_actor.backward()
                self.actor_optimizer.step()

                # update parameter
                self._soft_update()

        if sf is True:
            self.safe_model(sf_path)

    #TODO path nicht vergessen
    def eval(self, episodes, episode_length, render=True):
        reward = []
        mean_reward = []
        mean_q = []
        for episode in range(episodes):
            observation = self.env.reset()
            reward_e = []
            mean_reward_e = []
            mean_q_e = []
            for step in range(episode_length):
                state = torch.tensor(observation).float()
                action = self._select_action(state)
                # q = self.critic.eval(state, action).item()
                # mean_q_e.append(q)
                new_observation, rew, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                reward_e.append(rew.item())
                mean_reward_e.append(np.mean(reward_e).item())
                if done:
                    break
            reward.append(reward_e)
            mean_reward.append(mean_reward_e)
            mean_q.append(mean_q_e)

        return reward, mean_reward, mean_q


    def safe_model(self, path=None):
        safe_path = path if path is not None else self.safe_path
        data = {
            'epoch': self.episode,
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict()}
        torch.save(data, safe_path)

    def load_model(self, path=None):
        load_path = path if path is not None else self.safe_path
        checkpoint = torch.load(load_path)
        self.episode = checkpoint['epoch']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])


#def ddpg_torch(env):
#    state_shape = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete else env.observation_space.shape[0]
#    action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
#    action_range = env.action_space.high[0]
#    # initialize buffer
#    buffer = ReplayBuffer(CAPACITY)
#    # initialize actor
#    actor = Actor(state_shape=state_shape, action_shape=action_shape)
#    # initialize actor optimizer
#    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
#    # initialize critic
#    critic = Critic(state_shape=state_shape, action_shape=action_shape)
#    # initialize critic optimizer
#    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
#    # initialize random Process
#    noise = OrnsteinUhlenbeck(action_shape=action_shape

    #################################################
    # initialize target networks and copying params #
    #################################################
#    target_critic = Critic(state_shape=state_shape, action_shape=action_shape)
#    target_actor = Actor(state_shape=state_shape, action_shape=action_shape)#

#    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
#        target_param.data.copy_(param.data)
#        target_param.requires_grad = False

#    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
#        target_param.data.copy_(param.data)
#        target_param.requires_grad = False

#    observation = env.reset()
#    for i in range(0, BATCH_SIZE):
#        action = env.action_space.sample()
#        new_observation, reward, done, _ = env.step(action)
#        env.render()
#        buffer.push(observation, action, reward, new_observation)
#        observation = new_observation
#        if done:
#            observation = env.reset()

#    for episode in range(0, M):
#        noise.reset()
#        observation = env.reset()
#
#        for t in range(1, T+1):
#            action = actor.forward(torch.tensor(observation).float()).detach().numpy() + noise.x
#            noise.iteration()
#            #TODO action space auf output aufteilen
#            action = action * action_range
#            action = np.clip(action, a_min=-action_range, a_max=action_range)
#            #action = action.astype(np.float32)
#            #print(action)
#            new_observation, reward, done, _ = env.step(action)
#            env.render()#

#            buffer.push(observation, action, reward, new_observation)
#            observation = new_observation

#            sample = buffer.sample(BATCH_SIZE)
#            state_batch, action_batch, reward_batch, next_state_batch = buffer.batches_from_sample(sample, BATCH_SIZE)
#            state_batch, action_batch, reward_batch, next_state_batch = torch.tensor(state_batch).float(), torch.tensor(action_batch).float(), torch.tensor(reward_batch).float(), torch.tensor(next_state_batch).float()


            #y = reward_batch + GAMMA * target_critic(torch.from_numpy(next_state_batch).float(), target_actor(torch.from_numpy(next_state_batch).float()))
  #          y = reward_batch + GAMMA * target_critic(next_state_batch,
  #                                                   target_actor(next_state_batch))
            # update critic
  #          critic_optimizer.zero_grad()
            #target = critic(torch.from_numpy(state_batch).float(), torch.from_numpy(action_batch).float())
 #           target = critic(state_batch, action_batch)
 #           loss_critic = MSE(y, target)
 #           loss_critic.backward()
 #           critic_optimizer.step()


            #update actor
 #           actor_optimizer.zero_grad()
            #loss_actor = torch.zeros(1, dtype=torch.float, requires_grad = True)
            #for sample in batch:
            #    loss_actor = loss_actor + critic.forward(torch.from_numpy(sample.state).float(),actor.forward(torch.from_numpy(sample.state)).float())
            #loss_actor = critic(torch.from_numpy(state_batch).float(), actor(torch.from_numpy(state_batch).float()))
#            loss_actor = critic(state_batch, actor(state_batch))
#            loss_actor = -loss_actor.mean()#/len(batch)
#            loss_actor.backward()
#            actor_optimizer.step()

            #update parameter
#            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
#                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

#            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
#                target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)


#envv = gym.make('Pendulum-v0')
#model = DDPG(envv)
#model.load_model()
#model.train(episodes=100, episode_length=64, render=True)
#model.eval(episode_length=500, episodes=100)

