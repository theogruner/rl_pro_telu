from critic import Critic
from actor import Actor

class Mpo_worker(object):
    def __init__(self, EPSILON, EPSILON_MU, EPSILON_SIGMA, L_MAX):
        self.EPSILON = EPSILON
        self.EPSILON_MU = EPSILON_MU
        self.EPSILON_SIGMA = EPSILON_SIGMA
        self.L_MAX = L_MAX

        #initialize Q-network
        critic = Critic()

        #initialize policy
        actor = Actor

    def gradient_critic(self, states, actions):
        loss_critic =
