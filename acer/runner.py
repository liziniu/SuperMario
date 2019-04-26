import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from gym import spaces
from common.util import DataRecorder


class Runner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, save_path, store_data):
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

        self.store_data = store_data
        self.recorder = DataRecorder(save_path)
        self.max_store_length = int(3e4)
        self.episode = np.zeros(self.nenv)
        self.timestamp = np.zeros(self.nenv)

        self.dynamics = self.model.dynamics
        self.batch_goal_feat_shape = (nenv*(nsteps+1),) + env.observation_space.shape + self.dynamics.feat_shape
        self.goal_feat = None
        self.reached_status = np.array([False for _ in range(self.nenv)], dtype=bool)

        # assert self.nsteps == self.env._max_episode_steps

    def run(self):
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards = [], [], [], [], []
        mb_goal_obs = []

        reached_time = np.zeros(self.nenv, dtype=np.int32)
        # expect: arr(nenv, goal_shape), arr(nenv, obs_shape), dict(x_pos=xx, y_pos=xx)
        # if dynamics.dummy, please set arr to an empty arr
        self.goal_feat, goal_obs, goal_info = self.dynamics.get_goal(nb_goal=self.nenv)  # (nenv, goal_dim)
        self.reached_status[:] = False
        for step in range(self.nsteps):
            actions, mus, states = self.model._step(self.obs, S=self.states, M=self.dones, goals=self.goal_feat)
            # TODO: randomly act when we achieved our goal
            actions[self.reached_status] = self.simple_random_action()
            mus[self.reached_status] = self.get_mu_of_random_action()

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            mb_goal_obs.append(goal_obs)
            obs, rewards, dones, infos = self.env.step(actions)
            obs_feat = self.dynamics.extract_feature(obs)
            for env_idx in range(self.nenv):
                if not self.reached_status[env_idx]:
                    self.reached_status[env_idx] = self.check_goal_reached(obs_feat[env_idx], self.goal_feat[env_idx])
                    if self.reached_status[env_idx]:
                        reached_time[env_idx] = step
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
            enc_obs.append(obs[..., -self.nc:])
            for env_idx in range(self.nenv):
                info = infos[env_idx]
                data = dict(
                    x_pos=info.get("x_pos", None),
                    y_pos=info.get("y_pos", None),
                    goal_x_pos=goal_info[env_idx].get("x_pos", None),
                    goal_y_pos=goal_info[env_idx].get("y_pos", None),
                    episode=self.episode[env_idx],
                    timestep=self.timestamp[env_idx],
                    reached_status=self.reached_status[env_idx],
                    reward=rewards[env_idx],
                    env_id=env_idx,
                    act=actions[env_idx]
                )
                self.recorder.store(data)
                self.timestamp[env_idx] += 1
                if info.get("episode"):
                    assert self.dones[env_idx]
                    self.recorder.dump()
                    self.episode[env_idx] += 1
                    self.timestamp[env_idx] = 0
        mb_obs.append(np.copy(self.obs))
        mb_dones.append(self.dones)
        mb_goal_obs.append(goal_obs)        # make dimension is ture

        # shapes are adjusted to [nenv, nsteps, []]
        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)

        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones # Used for statefull models like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards

        mb_goal_obs = np.asarray(mb_goal_obs, dtype=np.float32).swapaxes(1, 0)
        # adjust goals from the time of acting randomly
        for env_idx in range(self.nenv):
            if self.reached_status[env_idx]:
                start = reached_time[env_idx] + 1
                mb_goal_obs[env_idx][start:] = np.copy(self.obs[env_idx])

        mb_goal_obs_flatten = np.reshape(mb_goal_obs, (-1, ) + mb_goal_obs.shape[2:])
        goal_feats = self.dynamics.extract_feature(mb_goal_obs_flatten)
        return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks, mb_goal_obs, goal_feats

    def check_goal_reached(self, obs_feat, desired_goal):
        assert obs_feat.shape == desired_goal.shape, "current_pos'shape:{} mismatch desired_goal's shape:{}".format(current_pos.shape, desired_goal.shape)
        if self.dynamics.dummy:
            return False
        else:
            eps = 1e-4
            tol = 0.05
            status = np.square(obs_feat - desired_goal).mean() / (np.square(desired_goal).mean() + eps) < tol
            return status

    def simple_random_action(self):
        return self.env.action_space.sample()

    def get_mu_of_random_action(self):
        assert isinstance(self.env.action_space, spaces.Discrete)
        return 1/self.env.action_space.n

    def initialize(self, init_steps):
        mb_obs, mb_actions, mb_next_obs, mb_infos = [], [], [], []
        for _ in range(init_steps):
            mb_obs.append(np.copy(self.obs))
            actions = np.asarray([self.env.action_space.sample() for _ in range(self.nenv)])
            self.obs, rewards, dones, infos = self.env.step(actions)
            obs_info = [{"x_pos": info.get("x_pos", None), "y_pos": info.get("y_pos", None)} for info in infos]
            mb_infos.append(obs_info)
            mb_actions.append(actions)
            mb_next_obs.append(np.copy(self.obs))
        self.obs = self.env.reset()
        mb_obs = np.asarray(mb_obs).swapaxes(1, 0)      # (nenv, nstep, obs_shape)
        mb_infos = np.asarray(mb_infos, dtype=object).swapaxes(1, 0)    # (nenv, nstep, dict)
        mb_actions = np.asarray(mb_actions).swapaxes(1, 0)
        mb_next_obs = np.asarray(mb_next_obs).swapaxes(1, 0)

        mb_obs = mb_obs.reshape((-1, ) + mb_obs.shape[2:])
        mb_infos = mb_infos.reshape(-1, )
        mb_actions = mb_actions.reshape((-1, ) + mb_actions.shape[2:])
        mb_next_obs = mb_next_obs.reshape((-1, ) + mb_next_obs.shape[2:])
        return mb_obs, mb_actions, mb_next_obs, mb_infos


