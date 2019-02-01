import random
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextState'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[int(self.position)] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    @staticmethod
    def batches_from_sample(sample, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        for s in sample:
            states.append(s.state)
            actions.append(s.action)
            rewards.append(s.reward)
            next_states.append(s.nextState)
        states = np.array(states).reshape(batch_size, -1)
        actions = np.array(actions).reshape(batch_size, -1)
        rewards = np.array(rewards).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)



