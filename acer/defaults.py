def atari():
    return dict(
        lrschedule='constant',
        queue_size=1000,
        feat_dim=512,
        replay_start=200,
        goal_shape=(84, 84, 4),  #(512, )
        normalize_novelty=True,
        threshold=3,
    )


def get_store_keys():
    return ["obs", "next_obs", "actions", "int_rewards", "ext_rewards", "mus", "ext_dones", "int_dones",
            "masks", "goal_obs", "goal_infos",  "next_obs_infos"]

