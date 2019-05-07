import threading
from collections import deque
import numpy as np
import sys


class ReplayBuffer:
    def __init__(self, env, sample_goal_fn, reward_fn, nsteps, size, keys):
        """Creates a replay buffer.

        Args:
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.size = size // nsteps
        nenv = self.nenv = env.num_envs
        self.nsteps = nsteps
        self.sample_goal_fn = sample_goal_fn
        self.reward_fn = reward_fn

        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.keys = keys
        # self._trajectory_buffer = Trajectory(nenv, keys)
        self.buffers = [{key: deque(maxlen=self.size) for key in keys} for _ in range(nenv)]

        # memory management
        self.lock = threading.Lock()

    def get(self,):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        samples = {key: [] for key in self.keys + ["int_rewards"]}
        for i in range(self.nenv):
            dones = self.buffers[i]["dones"].copy()   # (1, self.current_size)
            dones = np.concatenate(dones, axis=0)

            transitions = self.buffers[i].copy()
            for key in self.keys:
                arr = np.concatenate(transitions[key], axis=0)
                if key in ["obs", "goal_obs", "next_obs"]:
                    arr.astype(self.obs_dtype)
                elif key in ["mus", "dones", "masks"]:
                    arr.astype(np.bool)
                elif key in ["actions"]:
                    arr.astype(self.ac_dtype)
                elif key in ["goal_infos", "obs_infos"]:
                    arr.astype(object)
                elif key in ["ext_rewards"]:
                    arr.astype(np.float32)
                else:
                    raise ValueError("Unknown key:{}".format(key))
                transitions[key] = arr
            her_index, future_index = self.sample_goal_fn(dones, stacked=False)
            transitions["goal_obs"][her_index] = transitions["goal_obs"][future_index]
            transitions["goal_infos"][her_index] = transitions["goal_infos"][future_index]

            index = np.random.randint(0, self.current_size)
            for key in transitions:
                if key in ["obs", "masks", "goal_obs"]:
                    start, end = index*(self.nsteps+1), (index+1)*(self.nsteps+1)
                else:
                    start, end = index*self.nsteps, (index + 1)*self.nsteps
                transitions[key] = transitions[key][start:end]

            transitions["int_rewards"] = self.reward_fn(transitions["obs_infos"], transitions["goal_infos"])

            for key in self.keys + ["int_rewards"]:
                samples[key].append(transitions[key])
        for key in self.keys + ["int_rewards"]:
            samples[key] = np.concatenate(samples[key], axis=0)
        return samples

    def put(self, episode_batch):
        """episode_batch: dict of data. (nenv, nsteps, feature_shape)

        """
        assert isinstance(episode_batch, dict)
        key = self.keys[0]
        nenv, steps = episode_batch[key].shape[:2]
        assert nenv == self.nenv
        with self.lock:
            for i in range(nenv):
                for key in self.keys:
                    self.buffers[i][key].append(episode_batch[key][i])

    @property
    def current_size(self):
        key = self.keys[0]
        return len(self.buffers[0][key])

    @property
    def memory_usage(self):
        usage = 0
        for key in self.keys:
            usage += sys.getsizeof(self.buffers[0][key])
        return (usage * 2) // (1024 ** 3)

    def has_atleast(self, frames):
        # Frames per env, so total (nenv * frames) Frames needed
        # Each buffer loc has nenv * nsteps frames
        return self.current_size * self.nsteps * self.nenv >= frames
