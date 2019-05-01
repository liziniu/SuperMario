import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from gym import spaces
from common.util import DataRecorder, ResultsWriter
import time
import os
from copy import deepcopy
from baselines import logger


class Runner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, save_path, store_data, reward_fn, sample_goal, dist_type):
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

        self.save_path = save_path
        self.store_data = store_data
        self.recorder = DataRecorder(save_path)
        self.max_store_length = int(3e4)

        self.dynamics = self.model.dynamics
        self.sample_goal = sample_goal
        # self.batch_goal_feat_shape = (nenv*(nsteps+1),) + env.observation_space.shape + self.dynamics.feat_shape
        self.reached_status = np.array([False for _ in range(self.nenv)], dtype=bool)
        self.goal_obs, self.goal_info = None, None
        self.reward_fn = reward_fn
        # self.results_writer = ResultsWriter(os.path.join(save_path, "evaluation.csv"))

        self.episode = np.ones(self.nenv)
        self.episode_step = np.zeros(self.nenv)
        self.episode_reached_step = np.zeros(self.nenv)
        self.goal_abs_dist = np.array([None for _ in range(self.nenv)])

        self.name = self.model.scope

        assert dist_type in ["l1", "l2"]
        self.dist_type = dist_type

    def run(self):
        if self.goal_obs is None:
            self.goal_obs, self.goal_info = self.dynamics.get_goal(nb_goal=self.nenv)
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs, mb_actions, mb_mus, mb_dones, mb_ext_rewards = [], [], [], [], []
        mb_obs_infos, mb_goal_obs, mb_goal_infos = [], [], []
        reached_step = np.zeros(self.nenv, dtype=np.int32)

        episode_infos = np.asarray([{} for _ in range(self.nenv)], dtype=object)
        for step in range(self.nsteps):
            actions, mus, states = self.model.step(self.obs, S=self.states, M=self.dones, goals=self.goal_obs)
            actions[self.reached_status] = self.simple_random_action(np.sum(self.reached_status))
            mus[self.reached_status] = self.get_mu_of_random_action()

            mb_obs.append(deepcopy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            
            obs, rewards, dones, infos = self.env.step(actions)
            # evaluation model can also generate useful novel states.
            mb_obs_infos.append([{"x_pos": info["x_pos"], "y_pos": info["y_pos"]} for info in infos])
            mb_goal_infos.append(self.goal_info)
            mb_goal_obs.append(deepcopy(self.goal_obs))
            if self.sample_goal:
                # check reached based on obs_feat and goal_feat
                for env_idx in range(self.nenv):
                    if not self.reached_status[env_idx]:
                        if self.dist_type == "l1":
                            self.reached_status[env_idx] = self.check_goal_reached_v2(infos[env_idx], self.goal_info[env_idx])
                        else:
                            raise NotImplementedError("I do not know how to compute goal_latent")
                        if self.reached_status[env_idx]:
                            reached_step[env_idx] = step
                            self.episode_reached_step[env_idx] = deepcopy(self.episode_step[env_idx])
                            self.goal_abs_dist[env_idx] = abs(float(self.goal_info[env_idx]["x_pos"])-float(infos[env_idx]["x_pos"])) + \
                                abs(float(self.goal_info[env_idx]["y_pos"])-float(infos[env_idx]["y_pos"]))
                            achieved_pos = {"x_pos": infos[env_idx]["x_pos"], "y_pos": infos[env_idx]["y_pos"]}
                            logger.info("{}_env_{}|goal_pos:{}|achieved_pos:{}|size:{}".format(
                                self.name, env_idx, self.goal_info[env_idx], achieved_pos,
                                self.dynamics.queue.qsize()))
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_ext_rewards.append(rewards)
            enc_obs.append(obs[..., -self.nc:])

            for env_idx in range(self.nenv):
                # store data for visualize
                info = infos[env_idx]
                if self.store_data:
                    data = dict(
                        x_pos=info.get("x_pos", None),
                        y_pos=info.get("y_pos", None),
                        episode=self.episode[env_idx],
                        timestep=self.episode_step[env_idx],
                        reward=rewards[env_idx],
                        env_id=env_idx,
                        act=actions[env_idx]
                    )
                    self.recorder.store(data)
                self.episode_step[env_idx] += 1
                # summary
                if self.dones[env_idx]:
                    if info.get("episode"):
                        episode_infos[env_idx]["episode"] = info.get("episode")
                    if self.store_data:
                        self.recorder.dump()
                    if self.sample_goal:
                        if self.reached_status[env_idx]:
                            reached = 1.0
                            time_ratio = self.episode_reached_step[env_idx] / self.episode_step[env_idx]
                        else:
                            reached = 0.0
                            time_ratio = 1.0
                        abs_dist = deepcopy(self.goal_abs_dist[env_idx])
                        if abs_dist is None:
                            abs_dist = abs(float(infos[env_idx]["x_pos"])-float(self.goal_info[env_idx]["x_pos"])) + \
                                       abs(float(infos[env_idx]["y_pos"])-float(self.goal_info[env_idx]["y_pos"]))
                        episode_infos[env_idx]["reached_info"] = dict(reached=reached, time_ratio=time_ratio, abs_dist=abs_dist)
                        episode_infos[env_idx]["goal_info"] = dict(x_pos=self.goal_info[env_idx]["x_pos"],
                                                                   y_pos=self.goal_info[env_idx]["y_pos"])
                        # re-plan goal
                        goal_obs, goal_info = self.dynamics.get_goal(nb_goal=1)
                        # self.goal_feat[env_idx] = goal_feat[0]
                        self.goal_obs[env_idx] = goal_obs[0]
                        self.goal_info[env_idx] = goal_info[0]
                        self.episode[env_idx] += 1
                        self.episode_step[env_idx] = 0
                        self.episode_reached_step[env_idx] = 0
                        self.reached_status[env_idx] = False
                        self.goal_abs_dist[env_idx] = None
        mb_obs.append(deepcopy(self.obs))
        mb_dones.append(self.dones)
        mb_goal_obs.append(deepcopy(self.goal_obs))  # make dimension is true. we use this to retrace q value.
        mb_goal_infos.append(deepcopy(self.goal_info))  # we assume the difference by this info can be ignored
        mb_obs_infos.append(mb_obs_infos[-1])   # we assume the difference by this info can be ignored

        mb_goal_obs = np.asarray(mb_goal_obs, dtype=np.float32).swapaxes(1, 0)
        mb_goal_infos = np.asarray(mb_goal_infos, dtype=object).swapaxes(1, 0)
        mb_obs_infos = np.asarray(mb_obs_infos, dtype=object).swapaxes(1, 0)
        for env_idx in range(self.nenv):
            if self.sample_goal:
                if self.reached_status[env_idx]:
                    start = reached_step[env_idx] + 1   # adjust states that we have reached.
                else:
                    start = len(mb_goal_obs)
            else:
                start = 0
            mb_goal_obs[env_idx][start:] = deepcopy(self.obs[env_idx])
            mb_goal_infos[env_idx][start:] = mb_obs_infos[env_idx][-1]
        if self.dist_type == "l2":
            raise NotImplementedError
        else:
            mb_int_rewards = self.reward_fn(mb_obs_infos, mb_goal_infos)[:, :-1]
        # shapes are adjusted to [nenv, nsteps, []]
        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_ext_rewards = np.asarray(mb_ext_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones  # Used for statefull models like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:]  # Used for calculating returns. The dones array is now aligned with rewards

        results = dict(
            enc_obs=enc_obs,
            obs=mb_obs,
            actions=mb_actions,
            ext_rewards=mb_ext_rewards,
            mus=mb_mus,
            dones=mb_dones,
            masks=mb_masks,
            obs_infos=mb_obs_infos,     # nenv, nsteps, two purpose: 1)put into dynamics; 2) put into buffer
            episode_infos=episode_infos,
            goal_obs=mb_goal_obs,       # nenv, nsteps+1,
            goal_infos=mb_goal_infos,
            int_rewards=mb_int_rewards
        )
        return results

    def check_goal_reached(self, obs_feat, desired_goal):
        assert obs_feat.shape == desired_goal.shape
        assert len(obs_feat.shape) == 1
        if self.dynamics.dummy:
            return False
        else:
            eps = 1e-6
            tol = 0.03
            status = (np.square(obs_feat - desired_goal).sum() / (np.square(desired_goal).sum() + eps)) < tol
            return status

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

    def simple_random_action(self, nb_action):
        return np.random.randint(0, self.env.action_space.n, nb_action)

    def get_mu_of_random_action(self):
        assert isinstance(self.env.action_space, spaces.Discrete)
        return 1/self.env.action_space.n

    def initialize(self, init_steps):
        mb_obs, mb_actions, mb_next_obs, mb_goal_infos = [], [], [], []
        for _ in range(init_steps):
            mb_obs.append(deepcopy(self.obs))
            actions = np.asarray([self.env.action_space.sample() for _ in range(self.nenv)])
            self.obs, rewards, dones, infos = self.env.step(actions)
            goal_infos = [{"x_pos": info.get("x_pos", None),
                           "y_pos": info.get("y_pos", None)} for info in infos]
            mb_goal_infos.append(goal_infos)
            mb_actions.append(actions)
            mb_next_obs.append(deepcopy(self.obs))
        mb_obs = np.asarray(mb_obs).swapaxes(1, 0)      # (nenv, nstep, obs_shape)
        mb_goal_infos = np.asarray(mb_goal_infos, dtype=object).swapaxes(1, 0)    # (nenv, nstep, dict)
        mb_actions = np.asarray(mb_actions).swapaxes(1, 0)
        mb_next_obs = np.asarray(mb_next_obs).swapaxes(1, 0)

        mb_obs = mb_obs.reshape((-1, ) + mb_obs.shape[2:])
        mb_goal_infos = mb_goal_infos.reshape(-1, )
        mb_actions = mb_actions.reshape((-1, ) + mb_actions.shape[2:])
        mb_next_obs = mb_next_obs.reshape((-1, ) + mb_next_obs.shape[2:])

        self.dynamics.put_goal(mb_obs, mb_actions, mb_next_obs, mb_goal_infos)
        self.obs = self.env.reset()

    def evaluate(self, nb_eval):
        assert self.dynamics.dummy
        goal_obs, goal_info = self.dynamics.get_goal(nb_goal=self.nenv)  # (nenv, goal_dim)
        eval_info = {"l": 0, "r": 0}
        for i in range(nb_eval):
            terminal = False
            while True:
                actions, mus, states = self.model.step(self.obs, S=self.states, M=self.dones, goals=goal_obs)
                obs, rewards, dones, infos = self.env.step(actions)
                info = infos[0]
                if info.get("episode"):
                    assert dones[0]
                    eval_info["l"] += info.get("episode")["l"]
                    eval_info["r"] += info.get("episode")["r"]
                    terminal = True
                if terminal:
                    break
                self.states = states
                self.dones = dones
                self.obs = obs
        self.obs = self.env.reset()
        eval_info["l"] /= nb_eval
        eval_info["r"] /= nb_eval
        return eval_info
