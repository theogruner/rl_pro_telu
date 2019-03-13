import argparse

import gym
import quanser_robots
from mpo import MPO


def _add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def _parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Implementation of MPO on gym environments')
    parser.add_argument('--env', type=str, default='Levitation-v0',
                        help='a gym environment ID')
    _add_bool_arg(parser, 'train', default=True)
    _add_bool_arg(parser, 'eval', default=False)
    _add_bool_arg(parser, 'render', default=False)
    # parser.add_argument('--train', type=bool, default=0,
    #                     help='flag wether to train or not')
    # parser.add_argument('--eval', type=bool, default=False,
    #                     help='flag wether to evaluate or not')
    # parser.add_argument('--render', type=bool, default=False,
    #                     help='flag if to render or not')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='number of episodes for evaluation')
    parser.add_argument('--eval_ep_length', type=int, default=10000,
                        help='length of an evaluation episode')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='hard constraint of the E-step')
    parser.add_argument('--eps_mean', type=float, default=0.1,
                        help='hard constraint on C_mu')
    parser.add_argument('--eps_sigma', type=float, default=1e-4,
                        help='hard constraint on C_Sigma')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='learning rate')
    parser.add_argument('--alpha', type=float, default=1e4,
                        help='scaling factor of the '
                             'lagrangian multiplier in the M-step')
    parser.add_argument('--episodes', type=int, default=100,
                        help='number of episodes to learn')
    parser.add_argument('--episode_length', type=int, default=3000,
                        help='length of an episode (number of training steps)')
    parser.add_argument('--mb_size', type=int, default=64,
                        help='size of the mini batch')
    parser.add_argument('--rerun_mb', type=int, default=5,
                        help='number of reruns of the mini batch ')
    parser.add_argument('--add_act', type=int, default=64,
                        help='number of additional actions')
    parser.add_argument('--actor_layers', type=tuple, default=(100, 100),
                        help='size of the policy network layers')
    parser.add_argument('--critic_layers', type=tuple, default=(200, 200),
                        help='size of the critic network layers')
    # parser.add_argument('--log', type=bool, default=True,
    #                     help='flag for log messages while learning')
    _add_bool_arg(parser, 'log', default=False)
    parser.add_argument('--log_dir', type=str, default=None,
                        help='save log if safe log flag is set')
    _add_bool_arg(parser, 'save', default=True)
    parser.add_argument('--save_path', type=str, default='mpo_model.pt',
                        help='saving path if safe flag is set')
    # parser.add_argument('--safe', type=bool, default=True,
    #                     help='flag if to safe the model')
    parser.add_argument('--load', type=str, default=None,
                        help='loading path if given')

    args = parser.parse_args()
    d_args = vars(args)
    return d_args


if __name__ == '__main__':
    model_args = _parse()
    env = gym.make(model_args['env'])
    model = MPO(env,
                dual_constraint=model_args['eps'],
                mean_constraint=model_args['eps_mean'],
                var_constraint=model_args['eps_sigma'],
                learning_rate=model_args['gamma'],
                alpha=model_args['alpha'],
                episodes=model_args['episodes'],
                episode_length=model_args['episode_length'],
                mb_size=model_args['mb_size'],
                rerun_mb=model_args['rerun_mb'],
                sample_episodes=model_args['rerun_mb'],
                add_act=model_args['add_act'],
                actor_layers=model_args['actor_layers'],
                critic_layers=model_args['critic_layers'],
                log=model_args['log'],
                log_dir=model_args['log_dir'],
                render=model_args['render'],
                save=model_args['save'],
                save_path=model_args['path'])
    if model_args['load'] is not None:
        model.load_model(model_args['load'])
    if model_args['train']:
        model.train()
    if model_args['eval']:
        model.eval(episodes=model_args['eval_episodes'],
                   episode_length=model_args['eval_ep_length'],
                   render=model_args['render'])
