import threading
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
        nenv = self.nenv = env.num_envs
        self.nsteps = nsteps
        self.size = size // (nenv * self.nsteps)
        self.sample_goal_fn = sample_goal_fn
        self.reward_fn = reward_fn

        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.keys = keys
        # self._trajectory_buffer = Trajectory(nenv, keys)
        self.buffers = [{key: None for key in keys} for _ in range(nenv)]
        self._cache = [{} for _ in range(self.nenv)]

        # memory management
        self.lock = threading.Lock()
        self.current_size = 0   # num of sub-trajectories rather than transitions

    def get(self, use_cache, downsample=True):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        samples = {key: [] for key in self.keys + ["int_rewards"]}
        if not use_cache:
            cache = [{} for _ in range(self.nenv)]
            for i in range(self.nenv):
                if downsample:
                    interval = 20     # 20 sub-trajectories
                    if self.current_size < interval:
                        start, end = 0, self.current_size
                    else:
                        start = np.random.randint(0, max(self.current_size-interval, self.current_size))
                        end = min(self.current_size, start+interval)
                else:
                    start, end = 0, self.current_size
                for key in self.keys:
                    if key in ["obs", "masks"]:
                        cache[i][key] = self.buffers[i][key][start*(self.nsteps+1):end*(self.nsteps+1)].copy()
                    else:
                        cache[i][key] = self.buffers[i][key][start*self.nsteps:end*self.nsteps].copy()
            for i in range(self.nenv):
                dones = cache[i]["dones"]
                her_index, future_index = self.sample_goal_fn(dones)
                next_obs_decoded = decode_obs(cache[i]["obs"], self.nsteps)
                cache[i]["goal_obs"][her_index] = next_obs_decoded[future_index]
                cache[i]["goal_infos"][her_index] = cache[i]["obs_infos"][future_index]
            self._cache = cache.copy()
        else:
            cache = self._cache.copy()
        for i in range(self.nenv):
            transitions = cache[i]
            real_size = len(transitions["obs"]) // (self.nsteps + 1)
            index = np.random.randint(0, real_size)
            for key in self.keys:
                if key in ["obs", "masks"]:
                    start, end = index*(self.nsteps+1), (index+1)*(self.nsteps+1)
                else:
                    start, end = index*self.nsteps, (index+1)*self.nsteps
                samples[key].append(transitions[key][start:end])
            int_rewards = self.reward_fn(samples["obs_infos"][i], samples["goal_infos"][i])
            samples["int_rewards"].append(int_rewards)
        for key in self.keys + ["int_rewards"]:
            samples[key] = np.asarray(samples[key])
        return samples

    def put(self, episode_batch):
        """episode_batch: dict of data. (nenv, nsteps, feature_shape)

        """
        assert isinstance(episode_batch, dict)
        key = self.keys[0]
        nenv, steps = episode_batch[key].shape[:2]
        assert nenv == self.nenv
        for i in range(nenv):
            for key in self.keys:
                x = episode_batch[key][i]
                if self.buffers[i][key] is None:
                    if key in ["obs", "masks"]:
                        maxlen = self.size * (self.nsteps + 1)
                    else:
                        maxlen = self.size * self.nsteps
                    self.buffers[i][key] = np.empty((maxlen, ) + x.shape[1:], dtype=x.dtype)
                if key in ["obs", "masks"]:
                    start, end = self.current_size*(self.nsteps+1), (self.current_size+1)*(self.nsteps+1)
                    self.buffers[i][key][start:end] = x
                else:
                    start, end = self.current_size*self.nsteps, (self.current_size+1)*self.nsteps
                    self.buffers[i][key][start:end] = x
        self.current_size += 1
        self.current_size %= self.size

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


def decode_obs(enc_obs, nsteps):
    assert len(enc_obs) % (nsteps+1) == 0
    nb_segment = len(enc_obs) // (nsteps + 1)
    segments = np.split(enc_obs, nb_segment)
    new_arr = []
    for sub_arr in segments:
        new_arr.append(sub_arr[1:])
    new_arr = np.concatenate(new_arr, axis=0)
    return new_arr

if __name__ == "__main__":
    nsteps = 10
    enc_obs = []
    for j in range(2):
        for i in range(nsteps+1):
            enc_obs.append(j*10 + i)
    enc_obs = np.array(enc_obs)
    obs = decode_obs(enc_obs, nsteps)
    print(obs)