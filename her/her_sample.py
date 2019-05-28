import numpy as np
from copy import deepcopy


def make_sample_her_transitions(replay_strategy, replay_k, reduced_step):
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
    else:
        raise NotImplementedError

    def _sample_her_transitions(dones, deaths):
        """
        dones: (nenv, nstep)
        """
        if len(dones.shape) == 1:
            dones = np.expand_dims(dones, 0)
            deaths = np.expand_dims(deaths, 0)
            flatten = True
        else:
            flatten = False
        nenv = dones.shape[0]
        T = dones.shape[1]
        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=[nenv, T]) < future_p)

        # state that is near to deaths cannot be sampled.
        rights = np.ones_like(dones, dtype=np.bool)
        death_indexes = np.where(deaths)
        if not np.all(dones[death_indexes]):        # check done=True if deaths
            raise ValueError
        for i in range(nenv):
            death_index = np.where(deaths[i])[0]
            for idx in death_index:
                end = idx + 1
                start = max(0, end - reduced_step)
                rights[i][start:end] = False       # eliminate those approach to deaths.
        _her_indexes = ([], [])
        for k in range(len(her_indexes[0])):
            env_idx = her_indexes[0][k]
            step_idx = her_indexes[1][k]
            if rights[env_idx][step_idx]:       # choose those who has rights.
                _her_indexes[0].append(env_idx)
                _her_indexes[1].append(step_idx)
            # else:
            #     print('remove:({}, {})'.format(env_idx, step_idx))
        _her_indexes = (np.array(_her_indexes[0]), np.array(_her_indexes[1]))
        her_indexes = _her_indexes
        nb_her_sample = len(her_indexes[1])

        offset_indexes = np.random.uniform(size=nb_her_sample) * (T - her_indexes[1])

        max_future_indexes = np.empty(shape=[nenv, T], dtype=np.int32)
        max_future_indexes.fill(T-1)
        for i in range(nenv):
            done_index = np.where(dones[i])[0]
            start = 0
            for idx in done_index:
                end = idx + 1
                # state that is far to deaths cannot sample deaths state.
                if deaths[i][idx]:
                    # do not care that may be less than 0. because that will not occur in samples.
                    max_future_indexes[i][start:end] = idx - reduced_step
                else:
                    max_future_indexes[i][start:end] = idx
                start = end
        max_future_indexes = max_future_indexes[her_indexes]  # downsample
        future_indexes = offset_indexes.astype(int) + her_indexes[1]
        future_indexes = np.minimum(future_indexes, max_future_indexes)
        future_indexes = (her_indexes[0], future_indexes)

        # !!! IMPORTANT !!!
        # her_indexes and future_indexes should be applied on obs_next rather than obs.
        # same with goal_info, which meaning that goal_info describes obs_next.
        if not np.all(her_indexes[1] <= future_indexes[1]):
            import ipdb
            ipdb.set_trace()
        if flatten:
            return her_indexes[1], future_indexes[1]
        else:
            return her_indexes, future_indexes

    return _sample_her_transitions


def test_her_sample():
    np.random.seed(4)
    nenv, nsteps, reduced_step = 2, 10, 5
    dones = np.empty([nenv, nsteps], dtype=bool)
    dones.fill(False)
    batch_size = 4
    dones_idx = [np.random.randint(0, nenv, batch_size), np.random.randint(0, nsteps, batch_size)]
    # dones_idx = np.random.randint(0, nsteps, batch_size)
    dones[dones_idx] = True
    deaths = np.empty([nenv, nsteps], dtype=bool)
    deaths.fill(False)
    deaths[(dones_idx[0][0], dones_idx[1][0])] = True

    sample_fn = make_sample_her_transitions('future', 10, reduced_step)
    her_index, future_index = sample_fn(dones, deaths)
    print('-----------dones--------------')
    print(dones)
    print('-----------deaths--------------')
    print(deaths)
    print('-----------her_index----------')
    print(her_index[0])
    print(her_index[1])
    print('-----------future_index-------')
    print(future_index[0])
    print(future_index[1])

if __name__ == "__main__":
    test_her_sample()