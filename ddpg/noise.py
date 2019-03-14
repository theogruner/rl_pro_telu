import numpy as np


class Noise(object):
    """
    Super class for noises used in this implementation
    """
    def iteration(self):
        """
        update the noise
        """
        pass

    def reset(self):
        """
        resets the noise, hence called at the start of an episode
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
        """
        self.x = self.x + self.theta*(self.mu - self.x)*self.delta_t \
            + self.sigma * self._wiener_process()

    def get_noise(self):
        """
        :return: the current noise value
        """
        return self.x

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


class AdaptiveParameter(Noise):
    """
    Adaptive Parameter noise
    :param initial_std:
    :param threshold:
    :param scaling_factor:
    :param init_distance: (float) initial distance of the policies
    """
    def __init__(self, initial_std=0.1, threshold=0.1,
                 scaling_factor=1.01, init_distance=0):
        self.initial_std = initial_std
        self.threshold = threshold
        self.alpha = scaling_factor
        self.std = initial_std
        self.distance = init_distance

    def set_distance(self, distance):
        """
        :param distance: (float) policy distance
        """
        self.distance = distance

    def iteration(self):
        """
        update noise
        """
        if self.distance <= self.threshold:
            self.std *= self.alpha
        else:
            self.std /= self.alpha

    def get_noise(self):
        """
        :return: returns current noise
        """
        return self.std

    def reset(self):
        """
        sets the noise to the initial state
        """
        self.std = self.initial_std
