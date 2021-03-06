import torch
import torch.nn as nn
import numpy as np
import gym
import quanser_robots

from ddpg.buffer import ReplayBuffer
from ddpg.noise import OrnsteinUhlenbeck, AdaptiveParameter
from ddpg.critic_torch import Critic
from ddpg.actor_torch import Actor

from tensorboardX import SummaryWriter


class DDPG(object):
    """
    Deep Deterministic Policy Gradient (DDPG) model

    :param env: (Gym Environment) gym environment to learn from
    :param noise: (Noise) the noise to learn with
    :param noise_name: (str) id of the noise
    :param buffer_capacity: (int) capacity of the replay buffer
    :param batch_size: (int) size of the sample batches
    :param gamma: (float) discount factor
    :param tau: (float) soft update coefficient
    :param episodes: (int) number of episodes to make
    :param learning_rate: (float) learning rate of the optimization step
    :param episode_length: (int) length of an episode (= training steps per episode)
    :param actor_layers: (int, int) size of the layers of the policy network
    :param critic_layers: (int, int) size of the layers of the critic network
    :param norm: (bool) flag for using normalization in networks
    :param log: (bool) flag for logging
    :param log_name: (str) name of the log file
    :param render: (bool) flag if to render while training or not
    :param save: (bool) flag if to save the model if finished
    :param save_path: (str) path for saving and loading a model

    """
    def __init__(self, env, noise=None, noise_name=None, buffer_capacity=int(1e6), batch_size=64,
                 gamma=0.99, tau=0.001, learning_rate=1e-3,
                 episodes=int(1e4), episode_length=3000,
                 actor_layers=None, critic_layers=None, norm = True,
                 log=True, log_name=None, render=True, save=True, save_path="ddpg_model.pt"):

        # initialize env and read out shapes
        self.env = env
        self.state_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.action_range = env.action_space.high[0] if self.action_shape == 1\
            else env.action_space.high

        # initialize noise/buffer/loss
        self.noise = noise if noise is not None else OrnsteinUhlenbeck(self.action_shape)
        self.noise_name = noise_name if noise_name is not None else 'OUnoise'
        self.buffer = ReplayBuffer(buffer_capacity)
        self.loss = nn.MSELoss()

        # initialize hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.episodes = episodes
        self.episode_length = episode_length

        # initialize networks and optimizer
        self.actor = Actor(self.state_shape, self.action_shape,
                           layer1=actor_layers[0], layer2=actor_layers[1],
                           norm=norm) if actor_layers is not None \
            else Actor(self.state_shape, self.action_shape, norm=norm)
        self.target_actor = Actor(self.state_shape, self.action_shape,
                                  layer1=actor_layers[0], layer2=actor_layers[1],
                                  norm=norm) if actor_layers is not None \
            else Actor(self.state_shape, self.action_shape, norm=norm)
        self.critic = Critic(self.state_shape, self.action_shape,
                             layer1=critic_layers[0], layer2=critic_layers[1],
                             norm=norm) if actor_layers is not None\
            else Critic(self.state_shape, self.action_shape, norm=norm)
        self.target_critic = Critic(self.state_shape, self.action_shape,
                             layer1=critic_layers[0], layer2=critic_layers[1],
                             norm=norm) if actor_layers is not None \
            else Critic(self.state_shape, self.action_shape, norm=norm)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # fill buffer with random transitions
        self._random_trajectory(self.batch_size)

        # copy parameters to target networks
        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        if noise_name == 'AdaptiveParam':  # perturbed actor for ParameterNoise
            self.perturbed_actor = Actor(self.state_shape, self.action_shape,
                                         layer1=actor_layers[0], layer2=actor_layers[1],
                                         norm=norm) if actor_layers is not None \
                else Actor(self.state_shape, self.action_shape, norm=norm)

            for pert_param, target_param, param in zip(self.perturbed_actor.parameters(),
                                                       self.target_actor.parameters(),
                                                       self.actor.parameters()):
                pert_param.data.copy_(param.data)
                target_param.data.copy_(param.data)
                pert_param.requires_grad = False
                target_param.requires_grad = False
        else:
            for target_param, param in zip(self.target_actor.parameters(),
                                           self.actor.parameters()):
                target_param.data.copy_(param.data)
                target_param.requires_grad = False

        # control/log variables or flags
        self.episode = 0
        self.log = log
        self.log_name = log_name
        self.render = render
        self.save = save
        self.save_path = save_path

    def __call__(self, obs):
        return self._select_action(obs, train=False)

    def _random_trajectory(self, length):
        """
        pushes a given number of random transitions on the buffer
        :param length: (int) number of random actions to take
        """
        observation = self.env.reset()
        for i in range(0, length):
            action = self.env.action_space.sample()
            new_observation, reward, done, _ = self.env.step(action)
            self.buffer.push(observation, action, reward, new_observation)
            observation = new_observation
            if done:
                observation = self.env.reset()

    def _select_action(self, observation, train=True):
        """
        selects a action based on the policy(/target policy) and a given state for
        exploration and evaluation
        :param observation: (State) the state the decision is based on
        :param train: (bool) a flag determining wether to add noise to the action or not
        :return: (Action) the action taken following the policy plus noise
                 (target policy without noise if not training)
        """
        self.actor.eval()
        self.target_actor.eval()
        obs = torch.tensor([observation]).float()
        if self.noise_name == 'AdaptiveParam':
            self.perturbed_actor.eval()
            a = self.perturbed_actor(obs)[0].detach().numpy() if train \
                else self.target_actor(obs)[0].detach().numpy()
            self.perturbed_actor.train()
        else:
            a = self.actor(obs)[0].detach().numpy() + self.noise.get_noise() if train \
                else self.target_actor(obs)[0].detach().numpy()
        self.actor.train()
        self.target_actor.train()
        a = a * self.action_range
        a = np.clip(a, a_min=-self.action_range,
                    a_max=self.action_range)
        return a

    def _sample_batches(self, size):
        """
        samples corresponding batches of a given size for all transition elements
        :param size: (int) size of batches
        :return: ([State],[Action],[Reward],[State]) tuple of all batches
        """
        sample = self.buffer.sample(size)
        state_batch, action_batch, reward_batch, next_state_batch = \
            self.buffer.batches_from_sample(sample, size)
        state_batch, action_batch, reward_batch, next_state_batch = \
            torch.tensor(state_batch).float(), torch.tensor(action_batch).float(),\
            torch.tensor(reward_batch).float(), torch.tensor(next_state_batch).float()
        return state_batch, action_batch, reward_batch, next_state_batch

    def _soft_update(self):
        """
        soft-updates the target network with respect to the soft-update coefficent
        """
        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self, episodes=None, episode_length=None, render=None,
              save=None, save_path=None, log=None, log_name=None):
        """
        trains the model
        :param episodes: (int) number of episodes to make
        :param episode_length: (int) length of an episode (= training steps per episode)
        :param render: (bool) flag if to render while training
        :param save: (bool) flag if to save the model after training
        :param save_path: (str) path where to save the model
        :param log: (bool) flag for logging messages and recording
        :param log_name: (str) name of log file
        """
        # initialize flags and params
        rend = render if render is not None else self.render
        sf = save if save is not None else self.save
        sv_path = save_path if save_path is not None else self.save_path
        ep = episodes if episodes is not None else self.episodes
        it = episode_length if episode_length is not None else self.episode_length

        # initialize logging
        log_f = log if log is not None else self.log
        log_n = log_name if log_name is not None else self.log_name
        if log_f:
            writer = SummaryWriter("runs/" + log_n) if log_n is not None \
                else SummaryWriter()

        # start training
        for episode in range(self.episode, ep):

            # reset noise/env/logging variables
            self.noise.reset()  # TODO: maybe not for Adaptive Param noise??
            observation = self.env.reset()
            if log_f:
                reward_per_episode = 0
                q_per_ep = 0
                qloss_per_ep = 0

            # start episode
            for t in range(1, it + 1):

                # exploration
                action = self._select_action(observation)
                new_observation, reward, done, _ = self.env.step(action)
                if rend is True:
                    self.env.render()

                # logging
                if log_f:
                    reward_per_episode += reward
                    self.critic.eval()
                    q_per_ep += self.critic(torch.tensor([observation]).float(),
                                            torch.tensor([action]).float())[0].detach().item()
                    self.critic.train()

                # push transition onto the buffer
                self.buffer.push(observation, action, reward, new_observation)
                observation = new_observation

                # sample batches for training
                state_batch, action_batch, reward_batch, next_state_batch = \
                    self._sample_batches(self.batch_size)

                # update critic
                target_action = self.target_actor(next_state_batch)
                y = reward_batch + self.gamma * self.target_critic(next_state_batch, target_action)

                self.critic_optimizer.zero_grad()
                target = self.critic(state_batch, action_batch)
                loss_critic = self.loss(y, target)
                if log_f:                        # logging
                    qloss_per_ep += loss_critic
                loss_critic.backward()
                self.critic_optimizer.step()

                # update actor
                self.actor_optimizer.zero_grad()
                sample_action = self.actor(state_batch)
                loss_actor = self.critic(state_batch, sample_action)
                loss_actor = -loss_actor.mean()
                loss_actor.backward()
                self.actor_optimizer.step()

                # update target_networks
                self._soft_update()

                # update noise
                if self.noise_name == 'AdaptiveParam':
                    pert_action = self.perturbed_actor(state_batch)
                    distance = self.loss(pert_action, sample_action)
                    self.noise.set_distance(distance)
                    self.noise.iteration()
                    std = self.noise.get_noise()
                    for pert_param, param in zip(self.perturbed_actor.parameters(),
                                                   self.actor.parameters()):
                        pert_param.data.copy_(param.data + torch.normal(mean=torch.zeros(pert_param.shape),
                                                                        std=std))
                else:
                    self.noise.iteration()

                if done:
                    observation = self.env.reset()

            # logging
            if log_f:
                writer.add_scalar('data/rew_per_ep', reward_per_episode,
                                  episode+1)
                writer.add_scalar('data/mean_reward_per_ep', reward_per_episode / it,
                                  episode+1)
                writer.add_scalar('data/mean_q_per_ep', q_per_ep / it,
                                  episode+1)
                writer.add_scalar('data/mean_qloss_per_ep', qloss_per_ep / it,
                                  episode+1)
                reward_target = self.eval(10, it, render=False)
                writer.add_scalar('target/mean_rew_10_ep', reward_target,
                                  episode+1)

            # end of episode
            self.episode += 1
            if sf:
                self.save_model(sv_path)
            print("Episode " + str(episode+1) + " of " + str(ep) + " finished!")

        # finish training
        if log_f:
            writer.close()
        if sf is True:
            self.save_model(sv_path)

    def eval(self, episodes, episode_length, render=True):
        """
        method for evaluating current model (mean reward for a given number of
        episodes and episode length)
        :param episodes: (int) number of episodes for the evaluation
        :param episode_length: (int) length of a single episode
        :param render: (bool) flag if to render while evaluating
        :return: (float) meaned reward achieved in the episodes
        """

        summed_rewards = 0
        for episode in range(episodes):
            reward = 0
            observation = self.env.reset()
            for step in range(episode_length):
                action = self._select_action(observation, train=False)
                new_observation, rew, done, _ = self.env.step(action)
                reward += rew
                if render:
                    self.env.render()
                observation = new_observation if not done else self.env.reset()

            summed_rewards += reward
        return summed_rewards/episodes

    def save_model(self, path=None):
        """
        saves current model to a given path
        :param path: (str) saving path for the model
        """
        save_path = path if path is not None else self.save_path
        data = {
            'epoch': self.episode,
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict()}
        torch.save(data, save_path)

    def load_model(self, path=None):
        """
        loads a model from a given path
        :param path: (str) loading path for the model
        """
        load_path = path if path is not None else self.save_path
        checkpoint = torch.load(load_path)
        self.episode = checkpoint['epoch']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
