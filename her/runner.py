import numpy as np
from baselines.common.runners import AbstractEnvRunner
from common.env_util import VecFrameStack
from gym import spaces
import pickle
from baselines import logger
from queue import PriorityQueue


class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps, load_path, reward_fn):
        super().__init__(env=env, model=model, nsteps=nsteps)
        assert isinstance(env.action_space, spaces.Discrete), 'This ACER implementation works only with discrete action spaces!'
        assert isinstance(env, VecFrameStack)

        self.nact = env.action_space.n
        nenv = self.nenv
        self.nbatch = nenv * nsteps
        self.batch_ob_shape = (nenv*(nsteps+1),) + env.observation_space.shape

        self.obs = env.reset()
        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        self.nstack = self.env.nstack
        self.nc = self.batch_ob_shape[-1] // self.nstack

        self.load_path = load_path
        self.buffer = PriorityQueue()
        self.initialize()
        self.goals, self.goal_infos = self.get_goal(self.nenv)

        self.reward_fn = reward_fn

    def run(self):
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards, mb_goals = [], [], [], [], [], [],
        mb_obs_infos, mb_goal_infos = [], []
        for _ in range(self.nsteps):
            actions, mus, states = self.model.step(self.obs, S=self.states, M=self.dones, goals=self.goals)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            mb_goals.append(np.copy(self.goals))
            mb_goal_infos.append(np.copy(self.goal_infos))
            obs, rewards, dones, infos = self.env.step(actions)

            for env_idx in range(self.nenv):
                reached = self.check_goal_reached_v2(infos[env_idx], self.goal_infos[env_idx])
                if reached:
                    assert self.nenv == 1
                    obs = self.env.reset()
                    dones[:] = True
                    self.goals, self.goal_infos = self.step_goal()
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

        # mb_obs_infos.append(infos)
        # mb_goal_infos.append(goal_infos)

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
        # mb_int_rewards = self.reward_fn(mb_obs_infos, mb_goal_infos)
        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        results = dict(
            enc_obs=enc_obs,
            obs=mb_obs,
            actions=mb_actions,
            ext_rewards=mb_rewards,
            # int_rewards=mb_int_rewards,
            mus=mb_mus,
            dones=mb_dones,
            masks=mb_masks,
            goals=mb_goals,
            obs_infos=mb_obs_infos,
            goal_infos=mb_goal_infos
        )
        return results

    def step_goal(self, strategy="max"):
        if strategy == "max":
            goals, goal_infos = [], []
            item = self.buffer.get()
            for i in range(self.nenv):
                goal, info = item[2], [3]
                goals.append(goal)
                goal_infos.append(info)
            goals = np.asarray(goals, dtype=goal.dtype)
            goal_infos = np.asarray(goal_infos, dtype=object)
            self.buffer.put(item)
        else:
            raise NotImplementedError

        return goals, goal_infos

    def get_goal(self, nb_goal, strategy="max"):
        if strategy == "max":
            goals, goal_infos = [], []
            item = self.buffer.get()
            for i in range(nb_goal):
                goal, info = item[2], item[3]
                goals.append(goal)
                goal_infos.append(info)
            goals = np.asarray(goals, dtype=goal.dtype)
            goal_infos = np.asarray(goal_infos, dtype=object)
            self.buffer.put(item)
        else:
            raise NotImplementedError
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
        x_pos = np.asarray([x["info"]["x_pos"] for x in data])
        y_pos = np.asarray([x["info"]["y_pos"] for x in data])
        logger.info("loading {} goals".format(len(obs)))
        priority = - x_pos
        for i in range(len(obs)):
            self.buffer.put([priority[i], i, obs[i], {"x_pos": x_pos[i], "y_pos": y_pos[i]}])

    @staticmethod
    def check_goal_reached_v2(obs_info, goal_info):
        eps = 20
        obs_x, obs_y = float(obs_info["x_pos"]), float(obs_info["y_pos"])
        goal_x, goal_y = float(goal_info["x_pos"]), float(goal_info["y_pos"])
        dist = abs(obs_x - goal_x) + abs(obs_y - goal_y)
        if dist < eps:
            status = True
        else:
            status = False
        return status

