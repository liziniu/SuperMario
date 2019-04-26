import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k):
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
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(transitions):
        """
        transitions: (nenv, nsteps, nh, nw, nc)
        """
        nenv = transitions.shape[0]
        T = transitions.shape[1]

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=[nenv, T]) < future_p)
        nb_her_sample = len(her_indexes[1])
        offset_indexes = np.random.uniform(size=nb_her_sample) * (T - her_indexes[1])
        future_indexes = np.minimum(np.maximum(1, offset_indexes.astype(int)) + her_indexes[1], T-1)
        future_indexes = [her_indexes[0], future_indexes]

        future_obs = np.copy(transitions)
        future_obs[her_indexes] = transitions[future_indexes]
        return future_obs

    return _sample_her_transitions


if __name__ == "__main__":
    # Test her sample
    nenv, nsteps, nh, nw, nc = 2, 5, 84, 84, 4
    sample_policy = make_sample_her_transitions("future", 4)
    goal_obs = np.random.randn(nenv, nsteps, nh, nw, nc)
    goal_obs = sample_policy(goal_obs)
    print(goal_obs)