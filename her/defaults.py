

THRESHOLD = 6


def atari():
    return dict(
        lrschedule='constant',
        load_path='data/goal_data_303_2503.pkl',
        nsteps=20,
        nb_train_epoch=4,
        desired_x_pos=500,
        replay_start=1000,
        policy_inputs=[
            'obs',
            'achieved_goal',
            'desired_goal',
            'desired_goal_state'
        ]
    )


def get_store_keys():
    return ["obs", "next_obs", "achieved_goal", "next_achieved_goal", "desired_goal", "desired_goal_state",
            "actions", "rewards", "mus", "dones", "deaths", "next_obs_infos", "desired_goal_infos"]
