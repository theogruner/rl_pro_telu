import argparse

import gym
import quanser_robots
from ddpg import OrnsteinUhlenbeck
from ddpg import DDPG


def _add_bool_arg(parser, name, default=False):
    """
    Adds a boolean argument to the parser
    :param parser: (ArgumentParser) the parser the arguments should be added to
    :param name: (str) name of the argument
    :param default: (bool) default value of the argument
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def _parse():
    """
    initialize the argument parser and parse the arguments
    :return: the dict version of the parsers output
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Train a ddpg model with an Ornstein Uhlenbeck noise')

    parser.add_argument('--env', type=str, default='Levitation-v0',
                        help='a gym environment ID')
    _add_bool_arg(parser, 'train', default=True)
    _add_bool_arg(parser, 'eval', default=False)
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='number of episodes for evaluation')
    parser.add_argument('--eval_ep_length', type=int, default=500,
                        help='length of an evaluation episode')
    _add_bool_arg(parser, 'eval_render', default=True)
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6),
                        help='capacity of the buffer')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size of an mini batch')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--tau', type=float, default=0.001,
                        help='soft update coefficient')
    parser.add_argument('--episodes', type=int, default=int(1e4),
                        help='number of episodes to learn')
    parser.add_argument('--episode_length', type=int, default=300,
                        help='length of an episodes (number of training steps)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate for the opimization step')
    parser.add_argument('--actor_layers', type=tuple, default=(400, 300),
                        help='size of the policy network layers')
    parser.add_argument('--critic_layers', type=tuple, default=(400, 300),
                        help='size of the critic network layers')
    _add_bool_arg(parser, 'log', default=True)
    parser.add_argument('--log_name', type=str, default=None,
                        help='name of the log file')
    _add_bool_arg(parser, 'render', default=True)
    _add_bool_arg(parser, 'save', default=True)
    parser.add_argument('--load', type=str, default=None,
                        help='loading path if given')
    parser.add_argument('--save_path', type=str, default='ddpg_model.pt',
                        help='saving path if save flag is set True')
    parser.add_argument('---theta_noise', type=float, default=0.15,
                        help='mean reversion rate of the noise (Ornstein Uhlenbeck)')
    parser.add_argument('---mu_noise', type=float, default=0.0,
                        help='mean reversion level of the noise (Ornstein Uhlenbeck)')
    parser.add_argument('---sigma_noise', type=float, default=0.2,
                        help='scale of wiener process in the noise (Ornstein Uhlenbeck)')
    parser.add_argument('---x_start_noise', type=list, default=None,
                        help='starting point of the noise (Ornstein Uhlenbeck)')

    args = parser.parse_args()
    d_args = vars(args)
    return d_args


if __name__ == '__main__':
    model_args = _parse()
    env = gym.make(model_args['env'])
    noise = OrnsteinUhlenbeck(action_shape=env.action_space.shape[0],
                              theta=model_args['theta_noise'],
                              mu=model_args['mu_noise'],
                              sigma=model_args['sigma_noise'],
                              x_start=model_args['x_start_noise'])

    model = DDPG(env, noise,
                 buffer_capacity=model_args['buffer_capacity'],
                 batch_size=model_args['batch_size'],
                 gamma=model_args['gamma'],
                 tau=model_args['tau'],
                 episodes=model_args['episodes'],
                 learning_rate=model_args['learning_rate'],
                 episode_length=model_args['episode_length'],
                 actor_layers=model_args['actor_layers'],
                 critic_layers=model_args['critic_layers'],
                 log=model_args['log'],
                 log_name=model_args['log_name'],
                 render=model_args['render'],
                 save=model_args['save'],
                 save_path=model_args['save_path'])
    if model_args['load'] is not None:
        model.load_model(model_args['load'])
    if model_args['train']:
        model.train()
    if model_args['eval']:
        model.eval(episodes=model_args['eval_episodes'],
                   episode_length=model_args['eval_ep_length'],
                   render=model_args['eval_render'])
    env.close()
