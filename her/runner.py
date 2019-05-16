import numpy as np
from baselines.common.runners import AbstractEnvRunner
from common.env_util import VecFrameStack
from gym import spaces
import pickle
from baselines import logger
from queue import PriorityQueue
from common.util import DataRecorder
import os
from copy import deepcopy


class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps, load_path, reward_fn, desired_x_pos, threshold):
        super().__init__(env=env, model=model, nsteps=nsteps)
        assert isinstance(env.action_space, spaces.Discrete), 'This ACER implementation works only with discrete action spaces!'
        assert isinstance(env, VecFrameStack)

        self.nact = env.action_space.n
        nenv = self.nenv
        self.nbatch = nenv * nsteps
        self.batch_ob_shape = (nenv*(nsteps+1),) + env.observation_space.shape

        # self.obs = env.reset()
        self.obs_dtype = env.observation_space.dtype
        self.obs_shape = env.observation_space.shape
        self.ac_dtype = env.action_space.dtype
        self.nstack = self.env.nstack
        self.nc = self.batch_ob_shape[-1] // self.nstack
        
        self.recoder = DataRecorder(os.path.join(logger.get_dir(), "runner_data"))

        self.desired_x_pos = desired_x_pos
        self.load_path = load_path
        self.x_pos, self.goal_index_sorted = None, None
        self.buffer = []
        self.initialize()
        self.goals, self.goal_infos = self.get_goal(self.nenv)
        self.episode_step = np.zeros(self.nenv, dtype=np.int32)

        self.reward_fn = reward_fn
        self.threshold = threshold

    def run(self, debug=False):
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards, mb_goals = [], [], [], [], [], [],
        mb_next_obs_infos, mb_goal_infos, mb_next_obs = [], [], []
        episode_info = {}
        for _ in range(self.nsteps):
            actions, mus, states = self.model.step(self.obs, S=self.states, M=self.dones, goals=self.goals)
            if debug:
                self.env.render()
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            mb_goals.append(np.copy(self.goals))
            mb_goal_infos.append(np.copy(self.goal_infos))
            obs, rewards, dones, infos = self.env.step(actions)
            self.episode_step += 1

            next_obs = self.get_real_next_obs(obs, dones, infos)
            mb_next_obs.append(next_obs)
            mb_next_obs_infos.append(infos)
            for env_idx in range(self.nenv):
                reached = self.check_goal_reached_v2(infos[env_idx], self.goal_infos[env_idx])
                if reached:
                    final_pos = {"x_pos": infos[env_idx]["x_pos"], "y_pos": infos[env_idx]["y_pos"]}
                    mem = dict(env=env_idx, succ=True, length=self.episode_step[env_idx], final_pos=final_pos)
                    self.recoder.store(mem)
                    logger.info("env_{} succ!|goal:{}|final_pos:{}|length:{}".format(
                        env_idx, self.goal_infos[env_idx], final_pos, self.episode_step[env_idx]))
                    episode_info["succ"] = True
                    episode_info["length"] = self.episode_step[env_idx]
                    episode_info["final_pos"] = final_pos
                    self.episode_step[env_idx] = 0

                    assert self.nenv == 1
                    dones[:] = True
                    obs = self.env.reset()
                elif dones[env_idx]:
                    final_pos = {"x_pos": infos[env_idx]["x_pos"], "y_pos": infos[env_idx]["y_pos"]}
                    mem = dict(env=env_idx, succ=False, length=self.episode_step[env_idx], final_pos=final_pos)
                    self.recoder.store(mem)
                    logger.info("env_{} fail!|goal:{}|final_pos:{}|length:{}".format(
                        env_idx, self.goal_infos[env_idx], final_pos, self.episode_step[env_idx]))
                    episode_info["succ"] = False
                    episode_info["length"] = self.episode_step[env_idx]
                    episode_info["final_pos"] = {"x_pos": infos[env_idx]["x_pos"], "y_pos": infos[env_idx]["y_pos"]}
                    if infos[env_idx].get("episode"):
                        episode_info["episode"] = infos[env_idx].get("episode")
                    self.episode_step[env_idx] = 0
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.obs = obs
        mb_dones.append(self.dones)

        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_goals = np.asarray(mb_goals, dtype=self.goals.dtype).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_next_obs = np.asarray(mb_next_obs, dtype=self.obs_dtype).swapaxes(1, 0)

        mb_masks = mb_dones # Used for statefull models like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards

        mb_next_obs_infos = np.asarray(mb_next_obs_infos, dtype=object).swapaxes(1, 0)
        mb_goal_infos = np.asarray(mb_goal_infos, dtype=object).swapaxes(1, 0)
        mb_rewards = self.reward_fn(mb_next_obs_infos, mb_goal_infos)

        reach_index = np.where(mb_rewards.astype(int))
        assert np.all(mb_dones[reach_index])
        for i in reach_index[0]:
            for j in reach_index[1]:
                if abs(float(mb_next_obs_infos[i][j]["x_pos"]) - float(mb_goal_infos[i][j]["x_pos"])) > self.threshold:
                    raise ValueError("{}\n{}".format(mb_next_obs_infos[i][j], mb_goal_infos[i][j]))
                if abs(float(mb_next_obs_infos[i][j]["y_pos"]) - float(mb_goal_infos[i][j]["y_pos"])) > self.threshold:
                    raise ValueError("{}\n{}".format(mb_next_obs_infos[i][j], mb_goal_infos[i][j]))

        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.
        results = dict(
            obs=mb_obs,
            next_obs=mb_next_obs,
            actions=mb_actions,
            rewards=mb_rewards,
            mus=mb_mus,
            dones=mb_dones,
            masks=mb_masks,
            goal_obs=mb_goals,
            next_obs_infos=mb_next_obs_infos,
            goal_infos=mb_goal_infos,
            episode_info=episode_info,
        )
        return results

    def get_goal(self, nb_goal):
        goals, goal_infos = [], []
        for i in range(nb_goal):
            index = self.goal_index_sorted[i]
            data = self.buffer[index]
            goal, goal_info = data[0], data[1]
            goals.append(goal)
            goal_infos.append(goal_info)
        goals = np.asarray(goals, dtype=goal.dtype)
        goal_infos = np.asarray(goal_infos, dtype=object)
        return goals, goal_infos

    def initialize(self):
        f = open(self.load_path, "rb")
        data = []
        while True:
            try:
                data.extend(pickle.load(f))
            except Exception as e:
                logger.info(e)
                break
        obs = np.asarray([x["obs"] for x in data])
        self.x_pos = x_pos = np.asarray([x["info"]["x_pos"] for x in data])
        y_pos = np.asarray([x["info"]["y_pos"] for x in data])
        logger.info("loading {} goals".format(len(obs)))
        logger.info("goal_x_pos:{}".format(x_pos))
        for i in range(len(obs)):
            self.buffer.append([obs[i], {"x_pos": x_pos[i], "y_pos": y_pos[i]}])

        dist = list(np.abs(self.x_pos - self.desired_x_pos))
        dist_copy = deepcopy(dist)
        dist_copy.sort()
        self.goal_index_sorted = []
        for d in dist_copy:
            index = dist.index(d)
            self.goal_index_sorted.append(index)

    def check_goal_reached_v2(self, obs_info, goal_info):
        obs_x, obs_y = float(obs_info["x_pos"]), float(obs_info["y_pos"])
        goal_x, goal_y = float(goal_info["x_pos"]), float(goal_info["y_pos"])
        diff_x = abs(obs_x - goal_x)
        diff_y = abs(obs_y - goal_y)
        if diff_x <= self.threshold and diff_y <= self.threshold:
            status = True
        else:
            status = False
        return status

    def get_real_next_obs(self, next_obs, dones, infos):
        _next_obs = next_obs.copy()
        for i in range(self.nenv):
            assert self.nenv == 1
            if dones[i]:
                o = infos[i].get("next_obs", None)
                _next_obs[i] = o
        return _next_obs
