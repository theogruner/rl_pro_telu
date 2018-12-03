import numpy as np

class ornstein_uhlenbeck(object):
    def __init__(self,x_start, theta, mu, sigma, deltat):
        self.x_start = x_start
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.deltat = deltat
        self.x_old = x_start

    def iteration(self):
        self.x = self.x_old + self.theta*(self.mu - self.x_old)*self.deltat + self.sigma* self.wiener_process()
        return x

    def wiener_process(self):
        self.x_wiener = np.sqrt(self.deltat)* np.random.normal(size = self.mu.shape)
        return self.x_wiener