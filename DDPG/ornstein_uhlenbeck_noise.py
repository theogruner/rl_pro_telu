import numpy as np


class OrnsteinUhlenbeck(object):

    def __init__(self,x_start, theta, mu, sigma, delta_t,action_shape):
        self.x = x_start
        self.theta = theta
        self.mu = mu*np.ones(action_shape)
        self.sigma = sigma
        self.delta_t = delta_t

    def iteration(self):
        self.x = self.x + self.theta*(self.mu - self.x)*self.delta_t + self.sigma * self.wiener_process()

    def wiener_process(self):
        x_wiener = np.sqrt(self.delta_t)* np.random.normal(size = self.mu.shape)
        return x_wiener
