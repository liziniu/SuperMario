def atari():
    return dict(
        lrschedule='constant',
        queue_size=1000,
        feat_dim=512,
        replay_start=200,
        goal_shape=(84, 84, 4),  #(512, )
        normalize_novelty=True,
    )


def get_store_keys():
    return ["obs", "actions", "ext_rewards", "mus", "dones",
            "masks", "goal_obs", "goal_infos",  "obs_infos"]
