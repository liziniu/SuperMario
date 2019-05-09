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
from her.defaults import THRESHOLD


class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps, load_path, reward_fn, desired_x_pos):
        super().__init__(env=env, model=model, nsteps=nsteps)
        assert isinstance(env.action_space, spaces.Discrete), 'This ACER implementation works only with discrete action spaces!'
        assert isinstance(env, VecFrameStack)

        self.nact = env.action_space.n
        nenv = self.nenv
        self.nbatch = nenv * nsteps
        self.batch_ob_shape = (nenv*(nsteps+1),) + env.observation_space.shape

        # self.obs = env.reset()
        self.obs_dtype = env.observation_space.dtype
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

    def run(self):
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards, mb_goals = [], [], [], [], [], [],
        mb_obs_infos, mb_goal_infos = [], []
        episode_info = {}
        for _ in range(self.nsteps):
            actions, mus, states = self.model.step(self.obs, S=self.states, M=self.dones, goals=self.goals)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            mb_goals.append(np.copy(self.goals))
            mb_goal_infos.append(np.copy(self.goal_infos))
            obs, rewards, dones, infos = self.env.step(actions)
            self.episode_step += 1
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
                    # self.goals, self.goal_infos = self.step_goal()
                elif dones[env_idx]:
                    # self.goals, self.goal_infos = self.step_goal()

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
            mb_rewards.append(rewards)
            enc_obs.append(obs[..., -self.nc:])
            mb_obs_infos.append(infos)
        mb_obs.append(np.copy(self.obs))
        mb_dones.append(self.dones)
        mb_goals.append(np.copy(self.goals))

        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_goals = np.asarray(mb_goals, dtype=self.goals.dtype).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones # Used for statefull models like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards

        mb_obs_infos = np.asarray(mb_obs_infos, dtype=object).swapaxes(1, 0)
        mb_goal_infos = np.asarray(mb_goal_infos, dtype=object).swapaxes(1, 0)
        mb_int_rewards = self.reward_fn(mb_obs_infos, mb_goal_infos)
        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        results = dict(
            enc_obs=enc_obs,
            obs=mb_obs,
            actions=mb_actions,
            ext_rewards=mb_rewards,
            int_rewards=mb_int_rewards,
            mus=mb_mus,
            dones=mb_dones,
            masks=mb_masks,
            goal_obs=mb_goals,
            obs_infos=mb_obs_infos,
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

    @staticmethod
    def check_goal_reached_v2(obs_info, goal_info):
        eps = THRESHOLD
        obs_x, obs_y = float(obs_info["x_pos"]), float(obs_info["y_pos"])
        goal_x, goal_y = float(goal_info["x_pos"]), float(goal_info["y_pos"])
        dist = abs(obs_x - goal_x) + abs(obs_y - goal_y)
        if dist < eps:
            status = True
        else:
            status = False
        return status

