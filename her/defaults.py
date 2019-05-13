

THRESHOLD = 6


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
    return ["obs", "next_obs", "actions", "rewards", "mus", "dones",
            "masks", "goal_obs", "goal_infos",  "next_obs_infos"]