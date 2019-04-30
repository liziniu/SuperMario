def atari():
    return dict(
        lrschedule='constant',
        queue_size=1000,
        feat_dim=512,
        replay_start=200,
    )
