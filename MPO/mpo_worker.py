import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from scipy.optimize import minimize

import gym
import quanser_robots


from buffer import ReplayBuffer
from tensorboardX import SummaryWriter

# optimization problem
MSE = nn.MSELoss()

class MpoWorker(object):
    def __init__(self, ε, ε_μ, ε_Σ, max_episode, γ, α, β, τ, path, env):
        self.env = env

        # Hyperparameters
        self.α = α  # scaling factor for the update step of η_μ
        self.β = β  # scaling factor for the update step of η_Σ
        self.ε = ε  # hard constraint for the KL
        self.ε_μ = ε_μ  # hard constraint for the KL
        self.ε_Σ = ε_Σ  # hard constraint for the KL
        self.γ = γ  # learning rate
        self.τ = τ
        self.max_episode = max_episode  # 300
        self.episode = 0
        self.N = 64
        self.M = 100
        self.CAPACITY = 1e6
        self.BATCH_SIZE = 64
        self.action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)
        self.PATH = path

        # initialize Q-network
        self.critic = Critic(env)
        self.target_critic = Critic(env)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        # initialize critic optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # initialize policy
        self.actor = Actor(env)
        self.target_actor = Actor(env)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        # initialize policy optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # initialize replay buffer
        self.buffer = ReplayBuffer(self.CAPACITY)

        # Lagrange Multiplier
        self.η = np.random.rand()
        self.η_μ = np.random.rand()
        self.η_Σ = np.random.rand()

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
    def train(self):
        while self.episode < self.max_episode:
            self.episode += 1
            # update replay buffer B
            observation = self.env.reset()
            mean_reward = 0
            for steps in range(500):
                action = np.reshape(self.target_actor.action(torch.from_numpy(observation).float()).detach().numpy(),
                                    -1)
                new_observation, reward, done, _ = self.env.step(action)
                # env.render()
                self.buffer.push(observation, action, reward, new_observation)
                observation = new_observation
                mean_reward += reward
            print("Episode:     ", self.episode, "Mean reward:    ", mean_reward)

            # Find better policy by gradient descent
            for k in range(0, 10):
                # sample a mini-batch of N state action pairs
                batch = self.buffer.sample(self.N)
                state_batch, action_batch, reward_batch, next_state_batch = self.buffer.batches_from_sample(batch,
                                                                                                            self.N)

                # sample M additional action for each state
                target_μ, target_A = self.target_actor.forward(torch.tensor(state_batch).float())
                target_μ.detach()
                target_A.detach()
                action_distribution = MultivariateNormal(target_μ, scale_tril=target_A)
                additional_q = []
                for i in range(self.M):
                    action = action_distribution.sample()
                    additional_q.append(self.target_critic.forward(torch.tensor(state_batch).float(),
                                                                   action).detach().numpy())
                additional_q = np.array(additional_q).squeeze()

                # E-step
                # Update Q-function
                y = torch.from_numpy(reward_batch).float() \
                    + self.γ * self.target_critic(torch.from_numpy(next_state_batch).float(),
                                                  self.target_actor.action(torch.from_numpy(next_state_batch).float()))
                self.critic_optimizer.zero_grad()
                target = self.critic(torch.from_numpy(state_batch).float(), torch.from_numpy(action_batch).float())
                loss_critic = MSE(y, target)
                loss_critic.backward()
                self.critic_optimizer.step()
                print("Q-value loss: ", loss_critic.item())
                # print('bugfixing: ', self.η * self.ε
                #       + self.η * np.mean(np.log(np.mean(np.exp((additional_q - np.max(additional_q)) / self.η), 0))))
                # print(additional_q - np.max(additional_q, 0))


                def callback(xb):
                    print("exponent: ", (additional_q - np.max(additional_q)) / xb, "  exp: ",
                          np.mean(np.exp((additional_q - np.max(additional_q)) / xb)))
                    print("############################################################")

                # Update Dual-function
                def dual(η):
                    """
                    Dual function of the non-parametric variational
                    g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
                    """
                    max_add_q = np.max(additional_q, 0)
                    return η * self.ε + np.mean(max_add_q)\
                        + η * np.mean(np.log(np.mean(np.exp((additional_q - max_add_q) / η), 0)))

                bounds = [(1e-6, None)]
                res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                self.η = res.x[0]
                print("η dual: ", self.η)

                def get_q(states, action):
                    """q(a|s) = π(a|s) exp(Q(a,s)/η)"""
                    exp_Q = torch.exp(self.critic.forward(states, action).detach() / self.η)
                    q = action * exp_Q
                    # log_π = π.log_prob(sample_π * exp_Q)
                    return q, exp_Q

                # M-step
                # get Q values
                μ, A = self.actor.forward(torch.tensor(state_batch).float())
                π = MultivariateNormal(μ, scale_tril=A)
                sample_π = []
                exp_Q = []
                for a in range(self.M):
                    new_sample_π = π.sample().detach()
                    new_exp_Q = self.critic.forward(
                        torch.from_numpy(state_batch).float(), new_sample_π).detach() / self.η
                    exp_Q.append(new_exp_Q)
                    sample_π.append(new_sample_π)

                exp_Q = torch.stack(exp_Q).squeeze()
                baseline = torch.max(exp_Q, 0)[0]
                exp_Q = torch.exp(exp_Q - baseline)
                normalization = torch.mean(exp_Q, 0)
                sample_π = torch.stack(sample_π).squeeze()
                action_q = sample_π * exp_Q / normalization

                additional_logprob = []
                for column in range(self.M):
                    action_vec = action_q[column, :]
                    additional_logprob.append(π.log_prob(action_vec))
                additional_logprob = torch.stack(additional_logprob).squeeze()
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

                self.η_μ -= (self.α * (self.ε_μ - C_μ).detach().item())
                self.η_Σ -= (self.β * (self.ε_Σ - C_Σ).detach().item())

                self.actor_optimizer.zero_grad()
                loss_policy = -(
                        torch.mean(additional_logprob)
                        + self.η_μ * (self.ε_μ - C_μ)
                        + self.η_Σ * (self.ε_Σ - C_Σ)
                )
                print("Lagrange: ", loss_policy.item())
                loss_policy.backward()
                self.actor_optimizer.step()
                print("delta μ:  ", (self.ε_μ - C_μ).item(), "delta Σ", (self.ε_Σ - C_Σ).item())
                print("#######################################################################")


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
                    target_param.data.copy_(target_param.data * (1.0 - self.τ) + param.data * self.τ)

                # Update critic parameters
                for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.τ) + param.data * self.τ)
            # self.writer.add_scalar('mean-reward', mean_reward/500, self.episode)

            self.save_model()

    def eval(self):
        state = env.reset()

    def load_model(self):
        checkpoint = torch.load(os.path.join(self.PATH, 'test.pth'))
        self.episode = checkpoint['epoch']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['critic_state_dict'])

    def save_model(self):
        to_be_saved = {
            'epoch': self.episode,
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict()
        }
        torch.save(to_be_saved, os.path.join(self.PATH, 'test.pth'))

class Critic(nn.Module):

    def __init__(self, env):
        super(Critic, self).__init__()
        LAYER_1 = 200
        LAYER_2 = 200
        self.state_shape = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete \
            else env.observation_space.shape[0]
        self.action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
        self.lin1 = nn.Linear(self.state_shape, LAYER_1, True)
        self.lin2 = nn.Linear(LAYER_1 + self.action_shape, LAYER_2, True)
        self.lin3 = nn.Linear(LAYER_2, 1, True)

    def forward(self, state, action):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(torch.cat((x, action), 1)))
        x = self.lin3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        LAYER_1 = 200
        LAYER_2 = 200
        self.state_shape = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete \
            else env.observation_space.shape[0]
        self.action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)
        self.lin1 = nn.Linear(self.state_shape, LAYER_1, True)  # TODO: Normalization layer maybe?
        self.lin2 = nn.Linear(LAYER_1, LAYER_2, True)
        self.mean_layer = nn.Linear(LAYER_2, self.action_shape, True)
        self.cholesky_layer = nn.Linear(LAYER_2, self.action_shape, True)
        # self.cholesky_layer = nn.Linear(LAYER_2, int((self.action_shape*self.action_shape + self.action_shape)/2),
        #                                 True)
        self.cholesky = torch.zeros(self.action_shape,self.action_shape)

    def forward(self, states):
        x = F.relu(self.lin1(states))
        x = F.relu(self.lin2(x))
        mean = self.action_range * torch.tanh(self.mean_layer(x))
        cholesky_vector = F.softplus(self.cholesky_layer(x))

        cholesky = torch.stack([σ * torch.eye(self.action_shape) for σ in cholesky_vector])
        # cholesky = []
        # if cholesky_vector.dim() == 1 and cholesky_vector.shape[0] > 1:
        #     cholesky.append(self.to_cholesky_matrix(cholesky_vector))
        # else:
        #     for a in cholesky_vector:
        #         cholesky.append(self.to_cholesky_matrix(a))
        # cholesky = torch.stack(cholesky)
        return mean, cholesky

    def action(self, observation):
        mean, cholesky = self.forward(observation)
        # if self.action_shape == 1:
            #     action_distribution = Normal(mean, cholesky)
            # else:
        action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
        action = action_distribution.sample()
        return action

    def to_cholesky_matrix(self, cholesky_vector):
        k = 0
        cholesky = torch.zeros(self.action_shape, self.action_shape)
        for i in range(self.action_shape):
            for j in range(self.action_shape):
                if i >= j:
                    cholesky[i][j] = cholesky_vector.item() if self.action_shape == 1 else cholesky_vector[k].item()
                    k = k + 1
        return cholesky


env = gym.make('Pendulum-v0')
# env = gym.make('Qube-v0')
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

test = MpoWorker(epsilon, epsilon_mu, epsilon_sigma, l_max, gamma, α, β, τ, path, env)
test.train()
