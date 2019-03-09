import argparse

import gym
import quanser_robots
from noise import OrnsteinUhlenbeck
from ddpg import DDPG


def _parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Train a ddpg model with an Ornstein Uhlenbeck noise')

    parser.add_argument('--env', type=str, default='Levitation-v0',
                        help='a gym environment ID')
    parser.add_argument('--train', type=bool, default=True,
                        help='flag wether to train or not')
    parser.add_argument('--eval', type=bool, default=False,
                        help='flag wether to evaluate or not')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='number of episodes for evaluation')
    parser.add_argument('--eval_ep_length', type=int, default=500,
                        help='length of an evaluation episode')
    parser.add_argument('--eval_render', type=bool, default=True,
                        help='wether to or not to render while evaluation')
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
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate for the optimization step')
    parser.add_argument('--episode_length', type=int, default=60,
                        help='length of an episodes (number of training steps)')
    parser.add_argument('--actor_layers', type=tuple, default=(400, 300),
                        help='size of the policy network layers')
    parser.add_argument('--critic_layers', type=tuple, default=(400, 300),
                        help='size of the critic network layers')
    parser.add_argument('--log', type=bool, default=True,
                        help='flag for log messages while learning')
    parser.add_argument('--render', type=bool, default=False,
                        help='flag if to render while learning')
    parser.add_argument('--safe', type=bool, default=True,
                        help='flag if to safe the model')
    parser.add_argument('--load', type=bool, default=False,
                        help='flag if to load a model')
    parser.add_argument('--path', type=str, default='ddpg_model.pt',
                        help='saving(loading) path if safe(load) flag is set True')
    parser.add_argument('---theta_noise', type=float, default=0.15,
                        help='mean reversion rate of the noise (Ornstein Uhlenbeck)')
    parser.add_argument('---mu_noise', type=float, default=0.0,
                        help='mean reversion level of the noise (Ornstein Uhlenbeck)')
    parser.add_argument('---sigma_noise', type=float, default=0.2,
                        help='scale of wiener process in the noise (Ornstein Uhlenbeck)')
    parser.add_argument('---x_start', type=list, default=None,
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
                              x_start=model_args['x_start'])

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
                 render=model_args['render'],
                 safe=model_args['safe'],
                 safe_path=model_args['path'])
    if model_args['load']:
        model.load_model(model_args['path'])
    if model_args['train']:
        model.train()
    if model_args['eval']:
        model.eval(episodes=model_args['eval_episodes'],
                   episode_length=model_args['eval_ep_length'],
                   render=model_args['eval_render'])
    env.close()
