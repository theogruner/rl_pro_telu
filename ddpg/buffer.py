import random
import numpy as np
from collections import namedtuple

# Transitions tuples that are stored in the buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextState'))


class ReplayBuffer(object):
    """
    Replay-buffer that stores (state, action, reward, nextState) transitions
    :param capacity: (int) capacity of the buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        """
        pushes a (state, action, reward, nextState) transition onto the buffer
        :param args: variable length of arguments, supposed to be
                     (State) state,(Action) action,(float) reward, (State)nextState
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[int(self.position)] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        samples a batch of a given size
        :param batch_size: (int) number of transitions the sample contains
        :return: ([Transition]) random sample of the buffer
        """
        return random.sample(self.buffer, batch_size)

    @staticmethod
    def batches_from_sample(sample, batch_size):
        """
        splits a batch of (state, action, reward, nextState) transitions into a
        separate batch for each transition element
        :param sample: ([Transition]) batch of Transitions
        :param batch_size: (int) the size of of the input batch
        :return: ([State], [Action], [Reward], [State]) tuple of all resulting
                 batches
        """
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



