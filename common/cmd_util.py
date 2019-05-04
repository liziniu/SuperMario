"""
Helpers for scripts like run_atari.py.
"""


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()


def mujoco_arg_parser():
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--store_data', default=False, action='store_true')
    parser.add_argument('--aux_task', help='auxiliary task type(for acer/curiosity)', type=str, choices=["RF", "RND", "IDF"])
    parser.add_argument('--gpu', type=str, default="12,13,2")
    parser.add_argument('--mode', type=int, default=1, choices=[1, 2, 3, 4, 5])
    return parser


def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dicitonary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval


def parse_acer_mode(mode):
    if mode == 1:
        # only evaluation policy
        use_expl_collect = False
        use_eval_collect = True
        use_random_policy_expl = None
        dyna_source_list = ["eval"]
    elif mode == 2:
        use_expl_collect = True
        use_eval_collect = True
        use_random_policy_expl = False
        dyna_source_list = ["eval"]
    elif mode == 3:
        use_expl_collect = True
        use_eval_collect = True
        use_random_policy_expl = True
        dyna_source_list = ["eval"]
    elif mode == 4:
        use_expl_collect = True
        use_eval_collect = True
        use_random_policy_expl = True
        dyna_source_list = ["eval", "expl"]
    elif mode == 5:
        use_expl_collect = True
        use_eval_collect = True
        use_random_policy_expl = False
        dyna_source_list = ["eval", "expl"]
    else:
        raise ValueError("mode:{} wrong!".format(mode))
    return use_expl_collect, use_eval_collect, use_random_policy_expl, dyna_source_list