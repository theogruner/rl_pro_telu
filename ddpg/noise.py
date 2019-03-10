import numpy as np


class Noise(object):
    """
    Super class for noises used in this implementation
    """
    def iteration(self):
        """
        generate noise
        :return: noise (at current time step)
        """
        pass

    def reset(self):
        """
        resets the noise, hence called before episode
        """
        pass


class OrnsteinUhlenbeck(Noise):
    """
    Ornstein-Uhlenbeck noise with respect to the given parameters

    :param x_start: ([float]) initial noise (0 at default)
    :param theta: (float) mean reversion rate
    :param mu: (float) mean reversion level
    :param sigma: (float) scale of the random noise
    :param delta_t: (float) time-step size
    :param action_shape: (int) action shape (dimension of the action) of an environment
    """
    def __init__(self, action_shape, x_start=None, theta=0.15, mu=0, sigma=0.2,
                 delta_t=1e-2):
        self.x_start = x_start
        self.x = x_start
        self.theta = theta
        self.mu = mu*np.ones(action_shape)
        self.sigma = sigma
        self.delta_t = delta_t

    def iteration(self):
        """
        make a time-step in the Ornstein Uhlenbeck process
        :return: the current noise value
        """
        x = self.x
        self.x = self.x + self.theta*(self.mu - self.x)*self.delta_t \
                 + self.sigma * self._wiener_process()
        return x

    def reset(self):
        """
        resets the process to its initial position
        """
        self.x = self.x_start if self.x_start is not None else np.zeros(shape=self.mu.shape)

    def _wiener_process(self):
        """
        standard wiener process as a random generator
        :return: a random value in shape of the given action shape
        """
        return np.sqrt(self.delta_t) * np.random.normal(size=self.mu.shape)
