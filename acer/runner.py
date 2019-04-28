import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from gym import spaces
from common.util import DataRecorder, ResultsWriter
import time
import os
from queue import deque
from baselines import logger


class Runner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, save_path, store_data, reward_fn):
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
        # self.batch_goal_feat_shape = (nenv*(nsteps+1),) + env.observation_space.shape + self.dynamics.feat_shape
        self.reached_status = np.array([False for _ in range(self.nenv)], dtype=bool)
        self.goal_feat, self.goal_obs, self.goal_info = None, None, None
        self.reward_fn = reward_fn
        # assert self.nsteps == self.env._max_episode_steps
        self.results_writer = ResultsWriter(os.path.join(save_path, "evaluation.csv"))

        self.lenbuffer = deque(maxlen=40)  # rolling buffer for eval episode lengths
        self.rewbuffer = deque(maxlen=40)  # rolling buffer for eval episode rewards

        self.episode = np.ones(self.nenv)
        self.episode_step = np.zeros(self.nenv)
        self.episode_reached_step = np.zeros(self.nenv)

    def run(self):
        if self.goal_feat is None:
            self.goal_feat, self.goal_obs, self.goal_info = self.dynamics.get_goal(nb_goal=self.nenv)
            for env_idx in range(self.nenv):
                goal_info = self.goal_info[env_idx]
                if goal_info:
                    logger.info("env:{}|current goal pos:{} |goal queue size:{}".format(
                        env_idx, goal_info, self.dynamics.queue.qsize()))
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs, mb_actions, mb_mus, mb_dones, mb_ext_rewards = [], [], [], [], []
        mb_obs_feats, mb_goal_obs, mb_goal_infos = [], [], []
        reached_step = np.zeros(self.nenv, dtype=np.int32)

        reached_infos = np.asarray([{} for _ in range(self.nenv)], dtype=object)
        for step in range(self.nsteps):
            actions, mus, states = self.model.step(self.obs, S=self.states, M=self.dones, goals=self.goal_feat)
            actions[self.reached_status] = self.simple_random_action()
            mus[self.reached_status] = self.get_mu_of_random_action()

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            mb_goal_obs.append(np.copy(self.goal_obs))
            
            obs, rewards, dones, infos = self.env.step(actions)
            obs_feat = self.dynamics.extract_feature(obs)
            mb_obs_feats.append(obs_feat)
            goal_infos = [{"x_pos": info.get("x_pos", None),
                           "y_pos": info.get("y_pos", None)} for info in infos]
            mb_goal_infos.append(goal_infos)
            
            for env_idx in range(self.nenv):
                if not self.reached_status[env_idx]:
                    self.reached_status[env_idx] = self.check_goal_reached(obs_feat[env_idx], self.goal_feat[env_idx])
                    if self.reached_status[env_idx]:
                        reached_step[env_idx] = step
                        self.episode_reached_step[env_idx] = step
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_ext_rewards.append(rewards)
            enc_obs.append(obs[..., -self.nc:])
            
            # store data for visualize or debug
            for env_idx in range(self.nenv):
                info = infos[env_idx]
                data = dict(
                    x_pos=info.get("x_pos", None),
                    y_pos=info.get("y_pos", None),
                    goal_x_pos=self.goal_info[env_idx].get("x_pos", None),
                    goal_y_pos=self.goal_info[env_idx].get("y_pos", None),
                    episode=self.episode[env_idx],
                    timestep=self.episode_step[env_idx],
                    reached_status=self.reached_status[env_idx],
                    reward=rewards[env_idx],
                    env_id=env_idx,
                    act=actions[env_idx]
                )
                self.recorder.store(data)
                self.episode_step[env_idx] += 1
                if self.dones[env_idx]:
                    # summary
                    if self.reached_status[env_idx]:
                        reached = 1.0
                        time_ratio = self.episode_reached_step[env_idx] / self.episode_step[env_idx]
                    else:
                        reached = 0.0
                        time_ratio = 1.0
                    reached_infos[env_idx]["reached_info"] = dict(reached=reached, time_ratio=time_ratio)
                    reached_infos[env_idx]["goal_info"] = dict(x_pos=self.goal_info[env_idx]["x_pos"],
                                                               y_pos=self.goal_info[env_idx]["y_pos"])
                    if info.get("episode"):
                        reached_infos[env_idx]["episode"] = info.get("episode")

                    # re-plan goal
                    goal_feat, goal_obs, goal_info = self.dynamics.get_goal(nb_goal=1)
                    self.goal_feat[env_idx] = goal_feat[0]
                    self.goal_obs[env_idx] = goal_obs[0]
                    self.goal_info[env_idx] = goal_info[0]
                    if self.goal_info[env_idx]:
                        logger.info("env:{}|current goal pos:{}|goal queue size:{}".format(
                            env_idx, self.goal_info[env_idx], self.dynamics.queue.qsize()))
                    self.episode[env_idx] += 1
                    self.episode_step[env_idx] = 0
                    self.episode_reached_step[env_idx] = 0
                    self.reached_status[:] = False
                    self.recorder.dump()
        mb_obs.append(np.copy(self.obs))
        mb_dones.append(self.dones)
        # make dimension is true. we use this additional goal to retrace q value.
        mb_goal_obs.append(np.copy(self.goal_obs))
        obs_feat = self.dynamics.extract_feature(np.copy(self.obs))
        mb_obs_feats.append(obs_feat)

        # shapes are adjusted to [nenv, nsteps, []]
        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_ext_rewards = np.asarray(mb_ext_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones # Used for statefull models like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards
        mb_goal_obs = np.asarray(mb_goal_obs, dtype=np.float32).swapaxes(1, 0)
        mb_obs_feats = np.asarray(mb_obs_feats, dtype=np.float32).swapaxes(1, 0)
        mb_goal_infos = np.asarray(mb_goal_infos, dtype=object).swapaxes(1, 0)

        # adjust goals from the time of acting randomly
        for env_idx in range(self.nenv):
            if self.reached_status[env_idx]:
                start = reached_step[env_idx] + 1
                mb_goal_obs[env_idx][start:] = np.copy(self.obs[env_idx])

        mb_goal_obs_flatten = np.reshape(mb_goal_obs, (-1, ) + mb_goal_obs.shape[2:])   # flatten nenv and nstep
        mb_goal_feats = self.dynamics.extract_feature(mb_goal_obs_flatten)
        mb_obs_feats_flatten = np.reshape(mb_obs_feats, (-1, ) + mb_obs_feats.shape[2:])  # flatten nenv and nstep
        mb_int_rewards = self.reward_fn(mb_obs_feats_flatten, mb_goal_feats)
        mb_int_rewards = mb_int_rewards.reshape((self.nenv, self.nsteps+1))[:, :-1]     # strip the last reward

        return enc_obs, mb_obs, mb_actions, mb_ext_rewards, mb_mus, mb_dones, mb_masks,\
               mb_goal_obs, mb_goal_feats, mb_int_rewards, mb_goal_infos, reached_infos

    def check_goal_reached(self, obs_feat, desired_goal):
        assert obs_feat.shape == desired_goal.shape
        assert len(obs_feat.shape) == 1
        if self.dynamics.dummy:
            return False
        else:
            eps = 1e-6
            tol = 0.05
            status = (np.square(obs_feat - desired_goal).sum() / (np.square(desired_goal).sum() + eps)) < tol
            return status

    def simple_random_action(self):
        return self.env.action_space.sample()

    def get_mu_of_random_action(self):
        assert isinstance(self.env.action_space, spaces.Discrete)
        return 1/self.env.action_space.n

    def initialize(self, init_steps):
        mb_obs, mb_actions, mb_next_obs, mb_goal_infos = [], [], [], []
        for _ in range(init_steps):
            mb_obs.append(np.copy(self.obs))
            actions = np.asarray([self.env.action_space.sample() for _ in range(self.nenv)])
            self.obs, rewards, dones, infos = self.env.step(actions)
            goal_infos = [{"x_pos": info.get("x_pos", None),
                           "y_pos": info.get("y_pos", None)} for info in infos]
            mb_goal_infos.append(goal_infos)
            mb_actions.append(actions)
            mb_next_obs.append(np.copy(self.obs))
        self.obs = self.env.reset()
        mb_obs = np.asarray(mb_obs).swapaxes(1, 0)      # (nenv, nstep, obs_shape)
        mb_goal_infos = np.asarray(mb_goal_infos, dtype=object).swapaxes(1, 0)    # (nenv, nstep, dict)
        mb_actions = np.asarray(mb_actions).swapaxes(1, 0)
        mb_next_obs = np.asarray(mb_next_obs).swapaxes(1, 0)

        mb_obs = mb_obs.reshape((-1, ) + mb_obs.shape[2:])
        mb_goal_infos = mb_goal_infos.reshape(-1, )
        mb_actions = mb_actions.reshape((-1, ) + mb_actions.shape[2:])
        mb_next_obs = mb_next_obs.reshape((-1, ) + mb_next_obs.shape[2:])
        return mb_obs, mb_actions, mb_next_obs, mb_goal_infos

    def evaluate(self, nb_eval):
        assert self.dynamics.dummy
        assert self.nenv == 1
        self.goal_feat, goal_obs, goal_info = self.dynamics.get_goal(nb_goal=self.nenv)  # (nenv, goal_dim)
        eval_info = {"l": 0, "r": 0}
        for i in range(nb_eval):
            while True:
                obs_ = np.tile(np.copy(self.obs), [self.model.nenv, ] + [1]*len(self.obs.shape[1:]))
                if self.states:
                    states_ = np.tile(np.copy(self.states), [self.model.nenv, ] + [1]*len(self.states.shape[1:]))
                else:
                    states_ = None
                dones_ = np.tile(np.copy(self.dones), [self.model.nenv] + [1]*len(np.array(self.dones).shape[1:]))
                actions, mus, states = self.model.step(obs_, S=states_, M=dones_, goals=self.goal_feat)
                obs, rewards, dones, infos = self.env.step(actions[0])
                done = dones[0]
                info = infos[0]
                if info.get("episode"):
                    assert done
                    eval_info["l"] += info.get("episode")["l"]
                    eval_info["r"] += info.get("episode")["r"]
                    break
                self.states = states
                self.dones = dones
                self.obs = obs
        eval_info["l"] /= nb_eval
        eval_info["r"] /= nb_eval
        self.results_writer.write_row(eval_info)
        return eval_info
