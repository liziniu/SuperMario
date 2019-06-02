

THRESHOLD = 6

ACHIEVED_GOAL_SHAPE = (16, )
DESIRED_GOAL_SHAPE = (16, )


def parse_policy_inputs(mode):
    if isinstance(mode, str):
        mode = int(mode)
    if mode == 1:
        return ['obs']
    elif mode == 2:
        return ['obs', 'desired_goal']
    elif mode == 3:
        return ['obs', 'desired_goal_state']
    elif mode == 4:
        return ['obs', 'desired_goal', 'achieved_goal']
    elif mode == 5:
        return ['obs', 'desired_goal_state', 'achieved_goal']
    elif mode == 6:
        return ['obs', 'desired_goal', 'achieved_goal', 'desired_goal_state']
    elif mode == 7:
        return ['obs', 'desired_goal_state', 'desired_goal']
    else:
        raise NotImplementedError


def atari():
    return dict(
        lrschedule='constant',
        load_path='data/goal_data_303_2503.pkl',
        nsteps=20,
        nb_train_epoch=4,
        desired_x_pos=500,
        replay_start=1000,
    )


def get_store_keys():
    return ["obs", "next_obs", "achieved_goal", "next_achieved_goal", "desired_goal", "desired_goal_state",
            "actions", "rewards", "mus", "dones", "deaths", "next_obs_infos", "desired_goal_infos"]
