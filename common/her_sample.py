import numpy as np
from copy import deepcopy


def make_sample_her_transitions(replay_strategy, replay_k, replay_t=None):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    elif replay_strategy == 'fixed':
        future_p = 1 - (1. / (1 + replay_k))
        future_t = replay_t
    else:
        raise NotImplementedError

    def _sample_her_transitions(dones, max_length=None):
        """
        dones: (nenv, nstep)
        """
        if len(dones.shape) == 1:
            dones = np.expand_dims(dones, 0)
            flatten = True
        else:
            flatten = False
        nenv = dones.shape[0]
        if max_length is None:
            T = dones.shape[1]
        else:
            T = max_length
        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=[nenv, T]) < future_p)
        nb_her_sample = len(her_indexes[1])
        if replay_strategy == 'future':
            offset_indexes = np.random.uniform(size=nb_her_sample) * (T - her_indexes[1])
        elif replay_strategy == 'fixed':
            offset_indexes = np.minimum(her_indexes[1] + future_t, T-1)
        max_future_indexes = np.empty(shape=[nenv, T], dtype=np.int32)
        max_future_indexes.fill(T-1)
        for i in range(nenv):
            done_index = np.where(dones[i])[0]
            start = 0
            for idx in done_index:
                end = idx + 1
                max_future_indexes[i][start:end] = idx
                start = end
        max_future_indexes = max_future_indexes[her_indexes]  # downsample
        future_indexes = offset_indexes.astype(int) + her_indexes[1]
        future_indexes = np.minimum(future_indexes, max_future_indexes)
        future_indexes = (her_indexes[0], future_indexes)

        # !!! IMPORTANT !!!
        # her_indexes and future_indexes should be applied on obs_next rather than obs.
        # same with goal_info, which meaning that goal_info describes obs_next.
        assert all(her_indexes[1] <= future_indexes[1])
        if flatten:
            return her_indexes[1], future_indexes[1]
        else:
            return her_indexes, future_indexes

    return _sample_her_transitions


def test_her_sample():
    # Test her sample
    import matplotlib.pyplot as plt
    np.random.seed(3)
    nenv, nsteps, ndim = 2, 10, 1
    sample_fn = make_sample_her_transitions("future", 0.5)
    goal_obs = np.random.randint(0, 100, [nenv, nsteps], dtype=int)
    # goal_obs = np.random.randint(0, 100, [nenv, nsteps, 1], dtype=int)

    _goal_obs = np.copy(goal_obs)[..., :-1]
    _next_goal_obs = np.copy(goal_obs)[..., 1:]
    assert np.sum(_next_goal_obs[..., :-1] - _goal_obs[..., 1:]) < 1e-6

    # dones = np.empty([nenv, nsteps], dtype=bool)
    dones = np.empty([nenv, nsteps], dtype=bool)
    dones.fill(False)
    batch_size = 2
    dones_idx = [np.random.randint(0, nenv, batch_size), np.random.randint(0, nsteps, batch_size)]
    # dones_idx = np.random.randint(0, nsteps, batch_size)
    dones[dones_idx] = True
    print(dones)
    print("----------------index----------------------")
    for i in range(len(goal_obs.flatten())):
        print(i, end=",\t")
    print()
    print("----------------obs----------------------")
    for obs in goal_obs.flatten():
        print(obs, end=",\t")
    print()
    # plt.plot(goal_obs.flatten(), label="origin")
    her_index, future_idx = sample_fn(dones)
    goal_obs[her_index] = goal_obs[future_idx]
    print("----------------her_obs----------------------")
    for obs in goal_obs.flatten():
        print(obs, end=",\t")
    print()
    print("----------------her_index----------------------")
    for ind in her_index:
        # print(ind)
        print(ind, end=",\t")
    print()
    print("----------------her_future_index----------------------")
    for ind in future_idx:
        print(ind, end=",\t")

    _goal_obs = np.copy(goal_obs)[:, :-1]
    _next_goal_obs = np.copy(goal_obs)[:, 1:]
    assert np.sum(_next_goal_obs[:, :-1] - _goal_obs[:, 1:]) < 1e-6

if __name__ == "__main__":
    test_her_sample()