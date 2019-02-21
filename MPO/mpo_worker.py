import torch.nn.functional as F
import torch
import torch.nn as nn
from scipy.constants import sigma
from torch.distributions import MultivariateNormal
from torch.distributions import Normal
import numpy as np
from scipy.optimize import minimize

import gym
import quanser_robots


from buffer import ReplayBuffer

# optimization problem
MSE = nn.MSELoss()

LAYER_1 = 100
LAYER_2 = 100


class Critic(nn.Module):

    def __init__(self, env):
        super(Critic, self).__init__()
        self.state_shape = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete else env.observation_space.shape[0]
        self.action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
        self.lin1 = nn.Linear(self.state_shape, LAYER_1, True)
        self.lin2 = nn.Linear(LAYER_1 + self.action_shape, LAYER_2, True)
        self.lin3 = nn.Linear(LAYER_2, 1, True)

    def forward(self, state, action):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(torch.cat((x, action),1)))
        x = self.lin3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.state_shape = 1 if type(env.observation_space) == gym.spaces.discrete.Discrete else env.observation_space.shape[0]
        self.action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)
        self.lin1 = nn.Linear(self.state_shape, LAYER_1, True)
        self.lin2 = nn.Linear(LAYER_1, LAYER_2, True)
        self.mean_layer = nn.Linear(LAYER_2, self.action_shape, True)
        self.cholesky_layer = nn.Linear(LAYER_2, int((self.action_shape*self.action_shape + self.action_shape)/2), True)
        self.cholesky = torch.zeros(self.action_shape,self.action_shape)

    def forward(self, states):
        x = F.relu(self.lin1(states))
        x = F.relu(self.lin2(x))
        mean = self.action_range * torch.tanh(self.mean_layer(x))
        cholesky_vector = F.softplus(self.cholesky_layer(x))
        # Turn into cholesky matrix
        # if self.action_shape == 1:
        #     return mean, cholesky_vector
        cholesky = []
        if cholesky_vector.dim() == 1 and cholesky_vector.shape[0] > 1:
            cholesky.append(self.to_cholesky_matrix(cholesky_vector))
        else:
            for a in cholesky_vector:
                cholesky.append(self.to_cholesky_matrix(a))
        # cholesky = torch.stack([self.to_cholesky_matrix(a for a in cholesky_vector)])
        return mean, torch.stack(cholesky)

    def action(self, observation):
        with torch.no_grad():
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

class MpoWorker(object):
    def __init__(self, ε, ε_μ, ε_Σ, l_max, γ, env):
        self.env = env
        self.ε = ε
        self.ε_μ = ε_μ
        self.ε_Σ = ε_Σ
        self.L_MAX = l_max  # 300
        self.γ = γ
        self.N = 64
        self.M = 100
        self.CAPACITY = 1e6
        self.BATCH_SIZE = 64
        self.action_shape = 1 if type(env.action_space) == gym.spaces.discrete.Discrete else env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)

        # initialize Q-network
        self.critic = Critic(env)
        self.target_critic = Critic(env)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        # initialize critic optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-4)

        # initialize policy
        self.actor = Actor(env)
        self.target_actor = Actor(env)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        # initialize policy optimizer
        self.actor_optimizer = torch.optim.Adam(self.target_actor.parameters(), lr=5e-4)

        # initialize replay buffer
        self.buffer = ReplayBuffer(self.CAPACITY)

        self.i = 0
        self.l_curr = 0
        self.η = np.random.rand()
        self.η_μ = 0
        self.η_Σ = 0

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
    def worker(self):
        while self.l_curr < self.L_MAX:
            # update replay buffer B
            observation = self.env.reset()
            for steps in range(1000):
                action = np.reshape(self.actor.action(torch.from_numpy(observation).float()).numpy(), -1)
                new_observation, reward, done, _ = self.env.step(action)
                env.render()
                self.buffer.push(observation, action, reward, new_observation)
                observation = new_observation

            # Find better policy by gradient descent
            for k in range(0, 10):
                # sample a mini-batch of N state action pairs
                batch = self.buffer.sample(self.N)
                state_batch, action_batch, reward_batch, next_state_batch = self.buffer.batches_from_sample(batch, self.N)

                # sample M additional action for each state
                μ, A = self.actor.forward(torch.tensor(state_batch).float())
                action_distribution = MultivariateNormal(μ, scale_tril=A)
                additional_actions = []
                additional_logprob = []
                additional_q = []
                for i in range(self.M):
                    action = action_distribution.sample()
                    # print(np.prod(action.numpy(), 1))
                    additional_actions.append(np.prod(action.numpy(), 1)) if self.action_shape > 1 else additional_actions.append(action.numpy())
                    additional_q.append(self.critic.forward(torch.tensor(state_batch).float(), action).detach().numpy())
                    additional_logprob.append(action_distribution.log_prob(action))
                additional_actions = np.array(additional_actions).squeeze()   # first indices: M actions for N states, second indices: N different states, third indices: all actions
                additional_q = np.array(additional_q).squeeze()
                additional_logprob = torch.stack(additional_logprob)
                # print("Additional Action: ", additional_actions)
                # print("Additional Q: ", additional_q)
                # print("log prob", additional_logprob)

                # E-step
                # Update Q-function
                y = torch.from_numpy(reward_batch).float() + self.γ * self.target_critic(torch.from_numpy(next_state_batch).float(), self.actor.action(torch.from_numpy(next_state_batch).float()))
                self.critic_optimizer.zero_grad()
                target = self.critic(torch.from_numpy(state_batch).float(), torch.from_numpy(action_batch).float())
                loss_critic = MSE(y, target)
                loss_critic.backward()
                self.critic_optimizer.step()

                # Update Dual-function
                def dual(η):
                    """Dual function of the non-parametric variational \n
                    $g(\eta) = \eta\varepsilon + \eta \sum\limits_{s\in\mathcal{S}} \log \left(\sum\limits_{a\in \mathcal{A}} \exp(\frac{Q(a, s)}{\eta})\right)$"""
                    return η * self.ε + η * np.mean(np.sum(np.exp(additional_q / η), 0))
                res = minimize(dual, np.array([self.η]))
                self.η = res.x[0]

                def q(states, μ, A):
                    """q(a|s) = π(a|s) exp(Q(a,s)/η)"""
                    π = MultivariateNormal(μ, scale_tril=A).sample()
                    return π * torch.exp(self.critic.forward(states, π) / self.η)

                # M-step

                def lagrangian(η):
                    η_μ, η_Σ = η[0], η[1]
                    return np.mean(np.mean(additional_logprob.detach().numpy(), 0) + η_μ * (self.ε_μ - C_μ.detach().numpy()) + η_Σ * (self.ε_Σ - C_Σ.detach().numpy()))

                for i in range(10):
                    target_μ, target_A = self.target_actor.forward(torch.tensor(state_batch).float())
                    inner_Σ = []
                    inner_μ = []

                    additional_logprob = []
                    for a in range(self.M):
                        additional_logprob.append(q(torch.from_numpy(state_batch).float(), target_μ, target_A))
                    additional_logprob = torch.stack(additional_logprob).squeeze()

                    for mean, target_mean, a, target_a in zip(μ, target_μ, A, target_A):
                        Σ = a @ a.t()
                        target_Σ = target_a @ target_a.t()
                        inverse = Σ.inverse()
                        inner_Σ.append(torch.trace(inverse @ target_Σ) + torch.log(Σ.det()/target_Σ.det()))
                        inner_μ.append((mean - target_mean) @ inverse @ (mean - target_mean))


                    inner_μ = torch.stack(inner_μ)
                    inner_Sigma = torch.stack(inner_Σ)


                    C_μ = 1/2 * torch.mean(inner_Sigma)
                    C_Σ = 1/2 * torch.mean(inner_μ)
                    self.actor_optimizer.zero_grad()
                    loss_policy = torch.mean(torch.mean(additional_logprob, 0) + self.η_μ * (self.ε_μ - C_μ) + self.η_Σ * (self.ε_Σ - C_Σ))
                    loss_policy.backward(retain_graph=True)
                    self.actor_optimizer.step()

                    update_η = minimize(lagrangian, np.array([self.η_μ, self.η_Σ]))
                    self.η_μ, self.η_Σ = update_η.x[0], update_η.x[1]


                # Update policy parameters
                for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    param.data.copy_(target_param.data)

                # Update critic parameters
                for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data)

env = gym.make('Pendulum-v0')
# env = gym.make('Qube-v0')
# env = gym.make('BallBalancerSim-v0')
epsilon = 0.1
epsilon_mu = 0.1
epsilon_sigma = 0.1
l_max = 100000
gamma = 0.1

test = MpoWorker(epsilon, epsilon_mu, epsilon_sigma, l_max, gamma, env)
test.worker()
