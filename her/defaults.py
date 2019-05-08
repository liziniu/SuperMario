def atari():
    return dict(
        lrschedule='constant',
        load_path='data/goal_data_303_2503.pkl',
        nsteps=50,
        nb_train_epoch=8,
        desired_x_pos=500,
    )
