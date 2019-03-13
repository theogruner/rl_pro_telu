import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from scipy.optimize import minimize

import gym
import quanser_robots

from actor import Actor
from critic import Critic
from buffer import ReplayBuffer

from tensorboardX import SummaryWriter

# optimization problem
MSE = nn.MSELoss()


class MPO(object):
    """
    Maximum A Posteriori Policy Optimization (mpo)

    :param env: (Gym Environment) gym environment to learn from
    :param dual_constraint: (float) hard constraint of the dual formulation in the E-step
    :param mean_constraint: (float) hard constraint of the mean in the M-step
    :param var_constraint: (float) hard constraint of the covariance in the M-step
    :param learning_rate: (float) learning rate in the Q-function
    :param alpha: (float) scaling factor of the lagrangian multiplier in the M-step
    :param episodes: (int) number of training (evaluation) episodes
    :param episode_length: (int) step size of one episode
    :param mb_size: (int) size of the sampled mini-batch
    :param sample_episodes: (int) number of sampling episodes
    :param add_act: (int) number of additional actions
    :param actor_layers: (tuple) size of the hidden layers in the actor net
    :param critic_layers: (tuple) size of the hidden layers in the critic net
    :param log: (boolean) saves log if True
    :param log_path: (str) directory in which log is saved
    :param render: (boolean) renders the simulation if True
    :param safe: (boolean) saves the model if True
    :param safe_path: (str) path for saving and loading a model
    """
    def __init__(self, env, dual_constraint=0.1, mean_constraint=0.1, var_constraint=1e-4,
                 learning_rate=0.99, alpha=1e4, episodes=int(200),
                 episode_length=3000, mb_size=64, rerun_mb=5, sample_episodes=1, add_act=64,
                 actor_layers=None, critic_layers=None,
                 log=True, log_dir=None, render=False, safe=True, safe_path="mpo_model.pt"):
        # initialize env
        self.env = env

        # initialize some hyperparameters
        self.α = alpha  # scaling factor for the update step of η_μ
        self.ε = dual_constraint  # hard constraint for the KL
        self.ε_μ = mean_constraint  # hard constraint for the KL
        self.ε_Σ = var_constraint  # hard constraint for the KL
        self.γ = learning_rate  # learning rate
        self.episodes = episodes
        self.episode_length = episode_length
        self.mb_size = mb_size
        self.rerun_mb = rerun_mb
        self.M = add_act
        # self.CAPACITY = 1e6
        # self.BATCH_SIZE = 64
        self.action_shape = env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)

        # initialize networks and optimizer
        self.critic = Critic(env, layer1=critic_layers[0], layer2=critic_layers[1]) \
            if critic_layers is not None else Critic(env)
        self.target_critic = Critic(env, layer1=critic_layers[0], layer2=critic_layers[1]) \
            if critic_layers is not None else Critic(env)
        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.actor = Actor(env, layer1=actor_layers[0], layer2=actor_layers[1]) \
            if actor_layers is not None else Actor(env)
        self.target_actor = Actor(env, layer1=actor_layers[0], layer2=actor_layers[1]) \
            if actor_layers is not None else Actor(env)
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.mse_loss = nn.MSELoss()

        # initialize Lagrange Multiplier
        self.η = np.random.rand()
        self.η_μ = np.random.rand()
        self.η_Σ = np.random.rand()

        # initialize replay buffer
        # self.buffer = ReplayBuffer(self.CAPACITY)

        # control/log variables
        self.episode = 0
        self.sample_episodes = sample_episodes
        self.log = log
        self.log_dir = log_dir
        self.render = render
        self.safe = safe
        self.safe_path = safe_path

    # # Q-function gradient
    # def gradient_critic(self, states, actions):
    #     q = self.critic(states, actions)
    #     q_ret  = self.target_critic(states, actions)
    #     loss_critic = F.mse_loss(q, q_ret)
    #
    # def retrace_critic (self, env, states, actions, rewards, additional_actions):
    #     c = self.calculate_ck(states, actions)
    #     q_ret = self.target_critic(states, actions)
    #     for j in range(0, self.M):
    #         mean_q = 1
    #         q_ret = q_ret + np.power(self.GAMMA, j)*c[j]*(rewards[j] + mean_q)
    #     return q_ret
    #
    # def calculate_ck(self, states, actions):
    #     c = np.ones(len(states))
    #     for i in range(len(states)):
    #         if self.actor(states[i]) > actions[i]:
    #             c[i] = c[i-1] * self.actor(states[i]) / actions[i]
    #     return c
    def _sample_trajectory(self, episodes, episode_length, render):
        states = []
        rewards = []
        actions = []
        next_states = []
        mean_reward = 0
        for _ in range(episodes):
            observation = self.env.reset()
            for steps in range(episode_length):
                action = np.reshape(
                    self.target_actor.action(torch.from_numpy(observation).float()).numpy(),
                    -1)
                new_observation, reward, done, _ = self.env.step(action)
                mean_reward += reward
                if render:
                    self.env.render()
                states.append(observation)
                rewards.append(reward)
                actions.append(action)
                next_states.append(new_observation)
                if done:
                    observation = self.env.reset()
                else:
                    observation = new_observation
        states = np.array(states)
        # states = torch.tensor(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        return states, actions, rewards, next_states, mean_reward

    def train(self, episodes=None, episode_length=None, sample_episodes=None, rerun_mb=None,
              render=None, safe=None, safe_path=None, log=None, log_dir=None):
        """
        Trains a model based on MPO
        :param episodes: (int) number of training (evaluation) episodes
        :param episode_length: (int) step size of one episode
        :param sample_episodes: (int) number of sampling episodes
        :param rerun_mb: (int) number of times the episode is used for evaluation
        :param render: (boolean) renders the simulation if True
        :param safe: (boolean) saves the model if True
        :param safe_path: (str) path for saving and loading a model
        :param log: (boolean) saves log if True
        :param log_dir: (str) directory in which log is saved
        """
        rend = render if render is not None else self.render
        sf = safe if safe is not None else self.safe
        sf_path = safe_path if safe_path is not None else self.safe_path
        ep = episodes if episodes is not None else self.episodes
        it = episode_length if episode_length is not None else self.episode_length
        L = sample_episodes if sample_episodes is not None else self.sample_episodes
        rerun = rerun_mb if rerun_mb is not None else self.rerun_mb
        is_log = log if log is not None else self.log
        log_d = log_dir if log_dir is not None else self.log_dir
        if is_log:
            writer = SummaryWriter() if log_d is None else SummaryWriter(log_d)

        for episode in range(self.episode, ep):
            # Update replay buffer
            states, actions, rewards, next_states, mean_reward = self._sample_trajectory(L, it, rend)
            # observation = self.env.reset()
            # mean_reward = 0
            # for steps in range(it):
            #     action = np.reshape(self.target_actor.action(torch.from_numpy(observation).float()).detach().numpy(),
            #                         -1)
            #     new_observation, reward, done, _ = self.env.step(action)
            #     if rend is True:
            #         env.render()
            #     self.buffer.push(observation, action, reward, new_observation)
            #     observation = new_observation
            #     mean_reward += reward
            # print("Episode:     ", self.episode, "Mean reward:    ", mean_reward)
            mean_q_loss = 0
            mean_lagrange = 0
            # Find better policy by gradient descent
            for _ in range(rerun):
                for indices in BatchSampler(SubsetRandomSampler(range(it)), self.mb_size, False):
                    # for k in range(0, 1000):
                    # sample a mini-batch of N state action pairs
                    # batch = self.buffer.sample(self.N)
                    # state_batch, action_batch, reward_batch, next_state_batch = self.buffer.batches_from_sample(batch,
                    #                                                                                             self.N)
                    state_batch = states[indices]
                    action_batch = actions[indices]
                    reward_batch = rewards[indices]
                    next_state_batch = next_states[indices]

                    # sample M additional action for each state
                    target_μ, target_A = self.target_actor.forward(torch.tensor(state_batch).float())
                    target_μ.detach()
                    target_A.detach()
                    action_distribution = MultivariateNormal(target_μ, scale_tril=target_A)
                    additional_action = []
                    additional_q = []
                    for i in range(self.M):
                        action = action_distribution.sample()
                        additional_action.append(action)
                        additional_q.append(self.target_critic.forward(torch.tensor(state_batch).float(),
                                                                       action).detach().numpy())
                    # print(additional_action)
                    additional_action = torch.stack(additional_action).squeeze()
                    additional_q = np.array(additional_q).squeeze()

                    # E-step
                    # Update Q-function
                    # TODO: maybe use retrace Q-algorithm
                    y = torch.from_numpy(reward_batch).float() \
                        + self.γ * self.target_critic(
                            torch.from_numpy(next_state_batch).float(),
                            self.target_actor.action(torch.from_numpy(next_state_batch).float()))
                    self.critic_optimizer.zero_grad()
                    target = self.critic(torch.from_numpy(state_batch).float(),
                                         torch.from_numpy(action_batch).float())
                    loss_critic = MSE(y, target)
                    loss_critic.backward()
                    self.critic_optimizer.step()
                    mean_q_loss += loss_critic.item()   # TODO: can be removed

                    # Update Dual-function
                    def dual(η):
                        """
                        Dual function of the non-parametric variational
                        g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
                        """
                        max_q = np.max(additional_q, 0)
                        return η * self.ε + np.mean(max_q) \
                            + η * np.mean(np.log(np.mean(np.exp((additional_q - max_q) / η), 0)))

                    bounds = [(1e-6, None)]
                    res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                    self.η = res.x[0]
                    # print("η dual: ", self.η)

                    # M-step
                    #calculate the new q values
                    exp_Q = torch.tensor(additional_q) / self.η
                    baseline = torch.max(exp_Q, 0)[0]
                    exp_Q = torch.exp(exp_Q - baseline)
                    normalization = torch.mean(exp_Q, 0)
                    action_q = additional_action * exp_Q / normalization

                    # get Q values
                    for _ in range(5):
                        μ, A = self.actor.forward(torch.tensor(state_batch).float())
                        π = MultivariateNormal(μ, scale_tril=A)
                        # sample_π = []
                        # exp_Q = []
                        # for a in range(self.M):
                        #     new_sample_π = π.sample().detach()
                        #     new_exp_Q = self.critic.forward(
                        #         torch.from_numpy(state_batch).float(), new_sample_π).detach() / self.η
                        #     exp_Q.append(new_exp_Q)
                        #     sample_π.append(new_sample_π)
                        #
                        # exp_Q = torch.stack(exp_Q).squeeze()
                        # baseline = torch.max(exp_Q, 0)[0]
                        # exp_Q = torch.exp(exp_Q - baseline)
                        # normalization = torch.mean(exp_Q, 0)
                        # sample_π = torch.stack(sample_π).squeeze()
                        # action_q = sample_π * exp_Q / normalization

                        additional_logprob = []
                        if self.M == 1:
                            additional_logprob = π.log_prob(action_q)
                        else:
                            for column in range(self.M):
                                action_vec = action_q[column, :]
                                additional_logprob.append(π.log_prob(action_vec))
                            additional_logprob = torch.stack(additional_logprob).squeeze()
                        # print(additional_logprob.shape)
                        # print("Additional logprob: ", additional_logprob.shape)

                        inner_Σ = []
                        inner_μ = []
                        for mean, target_mean, a, target_a in zip(μ, target_μ, A, target_A):
                            Σ = a @ a.t()
                            target_Σ = target_a @ target_a.t()
                            inverse = Σ.inverse()
                            inner_Σ.append(torch.trace(inverse @ target_Σ)
                                           - Σ.size(0)
                                           + torch.log(Σ.det() / target_Σ.det()))
                            inner_μ.append((mean - target_mean) @ inverse @ (mean - target_mean))

                        inner_μ = torch.stack(inner_μ)
                        inner_Σ = torch.stack(inner_Σ)
                        # print(inner_μ.shape, inner_Σ.shape, additional_logprob.shape)
                        C_μ = 0.5 * torch.mean(inner_Σ)
                        C_Σ = 0.5 * torch.mean(inner_μ)

                        self.η_μ -= self.α * (self.ε_μ - C_μ).detach().item()
                        self.η_Σ -= self.α * (self.ε_Σ - C_Σ).detach().item()

                        if self.η_μ < 0:
                            self.η_μ = 0
                        if self.η_Σ < 0:
                            self.η_Σ = 0

                        self.actor_optimizer.zero_grad()
                        loss_policy = -(
                                torch.mean(additional_logprob)
                                + self.η_μ * (self.ε_μ - C_μ)
                                + self.η_Σ * (self.ε_Σ - C_Σ)
                        )
                        mean_lagrange += loss_policy.item()
                        # print("Lagrange: ", loss_policy.item())
                        loss_policy.backward()
                        self.actor_optimizer.step()

            # Update policy parameters
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)

            # Update critic parameters
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
            # self.writer.add_scalar('mean-reward', mean_reward/500, self.episode)

            print(
                "\n Episode:\t", episode,
                "\n Mean reward:\t", mean_reward / it,
                "\n Mean Q loss:\t", mean_q_loss / 50,
                "\n Mean Lagrange:\t", mean_lagrange / 50,
                "\n η:\t", self.η,
                "\n η_μ:\t", self.η_μ,
                "\n η_Σ:\t", self.η_Σ,
            )
            if sf is True:
                self.save_model(episode=episode, path=sf_path)
            if is_log:
                writer.add_scalar('data/mean_reward', mean_reward / it / 5, episode * it)
                writer.add_scalar('data/mean_lagrangeloss', mean_lagrange / it / 5 / 5, episode * it)
                writer.add_scalar('data/mean_qloss', mean_q_loss / it / 5, episode * it)

    def eval(self, episodes, episode_length, render=True):
        """
        Evaluates the learned model
        :param episodes: (int) number of evaluation episodes
        :param episode_length: (int) step size of one episode
        :param render: (boolean) renders the simulation if True
        """
        self.actor.eval()
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
                action = self.actor.eval_step(state).numpy()
                # obs = torch.tensor(observation).float()
                # q = self.critic.eval(state, action).item()
                # mean_q_e.append(q)
                new_observation, rew, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                reward_e.append(rew.item())
                mean_reward_e.append(np.mean(reward_e).item())
                observation = new_observation
                if done:
                    break
            reward.append(reward_e)
            mean_reward.append(mean_reward_e)
            mean_q.append(mean_q_e)

        return reward, mean_reward, mean_q

    def load_model(self, path=None):
        """
        Loading the model
        :param path: (str) file path (.pt file)
        """
        load_path = path if path is not None else self.safe_path
        checkpoint = torch.load(load_path)
        self.episode = checkpoint['epoch']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic.train()
        self.target_critic.train()
        self.actor.train()
        self.target_actor.train()

    def save_model(self, episode=0, path=None):
        """
        Saving the model
        :param episode: (int) number of learned episodes
        :param path: (str) file path (.pt file)
        """
        safe_path = path if path is not None else self.safe_path
        data = {
            'epoch': episode,
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict()
        }
        torch.save(data, safe_path)
