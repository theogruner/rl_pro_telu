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


class MpoWorker(object):
    def __init__(self, env, ε=0.1, ε_μ=5e-4, ε_Σ=1e-5, γ=0.99, α=10000,
                 episodes=int(200), episode_length=3200, mb_size=64, sample_episodes=1, add_act=64,
                 actor_layers=None, critic_layers=None,
                 log=True, render=False, safe=True, safe_path="mpo_model.pt"):
        #
        self.env = env

        # Hyperparameters
        self.α = α  # scaling factor for the update step of η_μ
        self.ε = ε  # hard constraint for the KL
        self.ε_μ = ε_μ  # hard constraint for the KL
        self.ε_Σ = ε_Σ  # hard constraint for the KL
        self.γ = γ  # learning rate
        self.episodes = episodes
        self.episode_length = episode_length
        self.mb_size = mb_size
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
        self.render =render
        self.safe = safe
        self.safe_path = safe_path
        # self.writer = SummaryWriter(log_dir="/home/theo/study/reinforcement_learning/project/Log_test")

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
                action = np.reshape(self.target_actor.action(torch.from_numpy(observation).float()).numpy(), -1)
                new_observation, reward, done, _ = self.env.step(action)
                mean_reward += reward
                if render:
                    env.render()
                states.append(observation)
                rewards.append(reward)
                actions.append(action)
                next_states.append(new_observation)
                if done:
                    observation = env.reset()
                else:
                    observation = new_observation
        states = np.array(states)
        # states = torch.tensor(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        return states, actions, rewards, next_states, mean_reward

    def train(self, episodes=None, episode_length=None, sample_episodes=None,
              render=None, safe=None, safe_path=None, log=None):
        rend = render if render is not None else self.render
        sf = safe if safe is not None else self.safe
        sf_path = safe_path if safe_path is not None else self.safe_path
        ep = episodes if episodes is not None else self.episodes
        it = episode_length if episode_length is not None else self.episode_length
        L = sample_episodes if sample_episodes is not None else self.sample_episodes

        for episode in range(ep):
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
                    + self.γ * self.target_critic(torch.from_numpy(next_state_batch).float(),
                                                  self.target_actor.action(torch.from_numpy(next_state_batch).float()))
                self.critic_optimizer.zero_grad()
                target = self.critic(torch.from_numpy(state_batch).float(), torch.from_numpy(action_batch).float())
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
                    return η * self.ε + np.mean(max_q)\
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
                    inner_Σ.append(torch.trace(inverse @ target_Σ) - Σ.size(0) + torch.log(Σ.det() / target_Σ.det()))
                    inner_μ.append((mean - target_mean) @ inverse @ (mean - target_mean))

                inner_μ = torch.stack(inner_μ)
                inner_Σ = torch.stack(inner_Σ)
                C_μ = 0.5 * torch.mean(inner_Σ)
                C_Σ = 0.5 * torch.mean(inner_μ)

                self.η_μ -= self.α * (self.ε_μ - C_μ).detach().item()
                self.η_Σ -= self.α * (self.ε_Σ - C_Σ).detach().item()

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
                print("delta μ:  ", self.η_μ*(self.ε_μ - C_μ).item(), "delta Σ", self.η_Σ*(self.ε_Σ - C_Σ).item())
                # print("#######################################################################")

                # unfinished = True
                # counter = 0
                # while(unfinished):
                #     counter += 1
                #     self.actor_optimizer.zero_grad()
                #     loss_policy = (torch.mean(additional_logprob)
                #                    + self.η_μ * (self.ε_μ - C_μ)
                #                    + self.η_Σ * (self.ε_Σ - C_Σ))
                #     loss_policy.backward()
                #     self.actor_optimizer.step()
                #     print("Loss: ", loss_policy.item())
                #
                #     μ, A = self.actor.forward(torch.tensor(state_batch).float())
                #     inner_Σ = []
                #     inner_μ = []
                #
                #     additional_logprob = []
                #     normalization = []
                #     for a in range(self.M):
                #         new_q, Q = q(torch.from_numpy(state_batch).float(), μ, A)
                #         pi = MultivariateNormal(μ, A)
                #         normalization.append(Q)
                #         additional_logprob.append(pi.log_prob(new_q))
                #     normalization = torch.mean(torch.stack(normalization).squeeze(), 0)
                #     additional_logprob = torch.stack(additional_logprob).squeeze()
                #     additional_logprob = additional_logprob/normalization
                #
                #
                #     for mean, target_mean, a, target_a in zip(μ, target_μ, A, target_A):
                #         Σ = a @ a.t()
                #         target_Σ = target_a @ target_a.t()
                #         inverse = Σ.inverse()
                #         inner_Σ.append(torch.trace(inverse @ target_Σ)
                #                        - Σ.size(0)
                #                        + torch.log(Σ.det() / target_Σ.det()))
                #         inner_μ.append((mean - target_mean) @ inverse @ (mean - target_mean))
                #
                #     inner_μ = torch.stack(inner_μ)
                #     inner_Sigma = torch.stack(inner_Σ)
                #
                #     C_μ = 0.5 * torch.mean(inner_Sigma)
                #     C_Σ = 0.5 * torch.mean(inner_μ)
                #     print(
                #         "C_μ: ", C_μ.item(),
                #         "   C_Σ: ", C_Σ.item(),
                #         "    log_prob: ", np.mean(additional_logprob.detach().numpy()),
                #         self.ε_μ - C_μ, self.ε_Σ - C_Σ)
                #
                #     self.η_μ -= self.α * (self.ε_μ - C_μ)
                #     self.η_Σ -= self.β * (self.ε_Σ - C_Σ)
                #
                #     if (self.η_μ == 0 and self.η_Σ == 0) or counter == 1:
                #         unfinished = False

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
            self.save_model(sf_path)

    def eval(self):
        state = env.reset()

    def load_model(self, path=None):
        load_path = path if path is not None else self.safe_path
        checkpoint = torch.load(load_path)
        self.episode = checkpoint['epoch']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['critic_state_dict'])

    def save_model(self, path=None):
        safe_path = path if path is not None else self.safe_path
        data = {
            'epoch': self.episode,
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict()
        }
        torch.save(data, safe_path)


# env = gym.make('Pendulum-v0')
env = gym.make('Qube-v0')
# env = gym.make('BallBalancerSim-v0')
epsilon = 0.1
epsilon_mu = 0.0005
epsilon_sigma = 0.00001
l_max = 100000
gamma = 0.99
tau = 0.001
α = 0.003
β = 0.003
τ = 0.001
path = "/home/theo/study/reinforcement_learning/project"

test = MpoWorker(env, render=True)
test.train()
