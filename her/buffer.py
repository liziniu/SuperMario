import threading
import numpy as np
import sys
from acer.util import goal_to_embedding

def check_reward_fn(next_obs, dones, maze_size):
    nenv, nsteps = next_obs.shape[0], next_obs.shape[1]
    rewards = np.empty([nenv, nsteps], dtype=np.float32)
    for i in range(nenv):
        for j in range(nsteps):
            if np.array_equal(next_obs[i][j], np.array([0., 0.])) and dones[i][j]:
                rewards[i][j] = 1.0
            else:
                rewards[i][j] = -0.1 / maze_size
    return rewards


class ReplayBuffer:
    def __init__(self, env, sample_goal_fn, reward_fn, nsteps, size, keys, her, goal_shape):
        """Creates a replay buffer.

        Args:
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        nenv = self.nenv = env.num_envs
        self.nsteps = nsteps
        self.size = size // self.nsteps
        self.sample_goal_fn = sample_goal_fn
        self.reward_fn = reward_fn

        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        self.goal_shape = goal_shape
        self.goal_as_image = len(goal_shape) == 3
        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.keys = keys
        # self._trajectory_buffer = Trajectory(nenv, keys)
        self.buffers = [{key: None for key in keys} for _ in range(nenv)]
        self._cache = [{} for _ in range(self.nenv)]

        self.her = her
        self.her_gain = 0.

        # memory management
        self.lock = threading.Lock()
        self.current_size = 0   # num of sub-trajectories rather than transitions

    def get(self, use_cache, downsample=True):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        samples = {key: [] for key in self.keys + ["_goal_infos", "_goal_obs"]}
        samples["her_gain"] = 0.
        if not use_cache:
            cache = [{} for _ in range(self.nenv)]
            for i in range(self.nenv):
                if downsample:
                    interval = 50     # 20 sub-trajectories
                    if self.current_size < interval:
                        start, end = 0, self.current_size
                    else:
                        start = np.random.randint(0, max(self.current_size-interval, self.current_size))
                        end = min(self.current_size, start+interval)
                else:
                    start, end = 0, self.current_size
                for key in self.keys:
                    if key in ["masks"]:
                        cache[i][key] = self.buffers[i][key][start*(self.nsteps+1):end*(self.nsteps+1)].copy()
                    else:
                        cache[i][key] = self.buffers[i][key][start*self.nsteps:end*self.nsteps].copy()
            for i in range(self.nenv):
                dones = cache[i]["dones"]
                her_index, future_index = self.sample_goal_fn(dones)
                reach_rewards = self.reward_fn(cache[i]["next_obs_infos"][None, :], cache[i]["goal_infos"][None, :])
                reach_rewards = reach_rewards.flatten()
                reach_index = np.where(reach_rewards.astype(int))
                error = np.sum(np.abs(cache[i]["rewards"][reach_index] - reach_rewards[reach_index]))
                assert error < 1e-6, "error:{}".format(error)
                cache[i]["_goal_infos"] = cache[i]["goal_infos"].copy()
                cache[i]["_goal_obs"] = cache[i]["goal_obs"].copy()
                if self.her:
                    cache[i]["_goal_infos"][her_index] = cache[i]["next_obs_infos"][future_index]
                    if self.goal_as_image:
                        cache[i]["_goal_obs"][her_index] = cache[i]["next_obs"][future_index]
                    else:
                        cache[i]["_goal_obs"][her_index] = goal_to_embedding(cache[i]["next_obs_infos"][future_index])
            self._cache = cache.copy()
        else:
            cache = self._cache.copy()

        for i in range(self.nenv):
            transitions = cache[i]
            real_size = len(transitions["obs"]) // self.nsteps
            index = np.random.randint(0, real_size)
            for key in self.keys + ["_goal_infos", "_goal_obs"]:
                if key in ["masks"]:
                    start, end = index*(self.nsteps+1), (index+1)*(self.nsteps+1)
                else:
                    start, end = index*self.nsteps, (index+1)*self.nsteps
                samples[key].append(transitions[key][start:end])

        for key in self.keys + ["_goal_infos", "_goal_obs"]:
            samples[key] = np.asarray(samples[key])
        if self.her:
            rewards = samples["rewards"]
            reach_rewards = self.reward_fn(samples["next_obs_infos"], samples["_goal_infos"])
            reach_index = np.where(reach_rewards.astype(int))
            dead_index = np.where(rewards == -1.0)

            samples["dones"][reach_index] = True        # verified by Maze experiments

            new_rewards = np.copy(rewards)
            if len(reach_index) > 0:
                new_rewards[reach_index] = 1.0
            if len(dead_index) > 0:
                new_rewards[dead_index] = -1.0

            samples["her_gain"] = np.mean(new_rewards) - np.mean(rewards)
            samples["rewards"] = new_rewards
            if samples["her_gain"] < 0.:
                import ipdb
                ipdb.set_trace()
                raise ValueError("her_gain:{} can't be less than 0.".format(samples["her_gain"]))
            samples["goal_obs"] = samples["_goal_obs"]
            samples["goal_infos"] = samples["_goal_infos"]
            samples.pop("_goal_obs")
            samples.pop("_goal_infos")
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
                    if key in ["masks"]:
                        maxlen = self.size * (self.nsteps + 1)
                    else:
                        maxlen = self.size * self.nsteps
                    self.buffers[i][key] = np.empty((maxlen, ) + x.shape[1:], dtype=x.dtype)
                if key in ["masks"]:
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
        return self.current_size * self.nsteps >= frames


def decode_obs(enc_obs, nsteps):
    assert len(enc_obs) % (nsteps+1) == 0
    nb_segment = len(enc_obs) // (nsteps + 1)
    segments = np.split(enc_obs, nb_segment)
    new_arr = []
    for sub_arr in segments:
        new_arr.append(sub_arr[1:])
    new_arr = np.concatenate(new_arr, axis=0)
    return new_arr

def decode(x, nsteps):
    assert len(x) % nsteps== 0
    nb_segment = len(x) // nsteps
    segments = np.split(x, nb_segment)
    new_arr = []
    for sub_arr in segments:
        new_arr.append(sub_arr)
    new_arr = np.asarray(new_arr)
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