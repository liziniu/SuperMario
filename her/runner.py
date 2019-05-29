import numpy as np
from common.env_util import VecFrameStack
from gym import spaces
from baselines import logger
from common.util import DataRecorder
import os


def goal_info_to_embedding(goal_infos, goal_dim):
    nb_tile = goal_dim // 2
    embeddings = []
    for goal_info in goal_infos:
        coordinate = np.array([goal_info["x_pos"], goal_info["y_pos"]], dtype=np.float32)
        embedding = np.tile(coordinate, nb_tile)
        embeddings.append(embedding)
    return np.asarray(embeddings)


class Runner:
    TEMPLATE = 'env_{} {}!|goal:{}|final_pos:{}|length:{}'

    def __init__(self, env, model, curriculum, nsteps, reward_fn, threshold):
        assert isinstance(env.action_space, spaces.Discrete), 'This ACER implementation works only with discrete action spaces!'
        assert isinstance(env, VecFrameStack)

        self.env = env
        self.model = model
        self.policy_inputs = self.model.policy_inputs
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.nact = env.action_space.n
        self.nbatch = nenv * nsteps
        self.obs_shape = self.model.obs_shape
        self.obs_dtype = self.model.obs_dtype
        self.ac_dtype = env.action_space.dtype
        self.achieved_goal_shape = self.model.achieved_goal_sh
        self.desired_goal_shape = self.model.desired_goal_sh
        self.desired_goal_state_shape = self.model.desired_goal_state_sh

        self.dict_obs = isinstance(self.env.observation_space, spaces.Dict)

        self.obs = np.zeros((nenv,) + self.obs_shape, dtype=self.obs_dtype)
        self.achieved_goal = np.zeros((nenv, ) + self.achieved_goal_shape, dtype=np.float32)
        self.desired_goal = np.zeros((nenv, ) + self.desired_goal_shape, dtype=np.float32)
        self.desired_goal_state = np.zeros((nenv, ) + self.desired_goal_state_shape, dtype=self.obs_dtype)
        self.desired_goal_info = np.zeros((nenv, ), dtype=object)

        self.nb_tile = self.achieved_goal.shape[-1] // 2
        if self.dict_obs:
            dict_obs = self.env.reset()
            self.obs[:] = dict_obs['observation']
            achieved_goal = dict_obs["achieved_goal"]
            self.achieved_goal[:] = np.tile(achieved_goal, [1, self.nb_tile])
        else:
            self.obs[:] = self.env.reset()

        self.nsteps = nsteps

        self.curriculum = curriculum
        self.desired_goal[:], self.desired_goal_state[:], self.desired_goal_info[:] = self.curriculum.get_current_target(nb_goal=self.nenv)

        self.recoder = DataRecorder(os.path.join(logger.get_dir(), "runner_data"))
        self.episode_step = np.zeros(self.nenv, dtype=np.int32)
        self.reward_fn = reward_fn
        self.threshold = threshold
        self.include_death = False

    def run(self, acer_steps):
        mb_obs, mb_next_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_death = [], [], [], [], [], [], []
        mb_next_obs_infos, mb_desired_goal_infos = [], []
        mb_achieved_goal, mb_next_achieved_goal, mb_desired_goal, mb_desired_goal_state = [], [], [], []
        episode_info = {}

        for step in range(self.nsteps):
            actions, mus = self.model.step({'obs': self.obs.copy(), 'achieved_goal': self.achieved_goal.copy(),
                'desired_goal': self.desired_goal.copy(), 'desired_goal_state': self.desired_goal_state.copy()})
            mb_obs.append(np.copy(self.obs))
            mb_achieved_goal.append(np.copy(self.achieved_goal))
            mb_desired_goal.append(np.copy(self.desired_goal))
            mb_desired_goal_state.append(np.copy(self.desired_goal_state))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_desired_goal_infos.append(np.copy(self.desired_goal_info))

            # step
            if self.dict_obs:
                dict_obs, _, dones, infos = self.env.step(actions)
                obs, achieved_goal = dict_obs['observation'], dict_obs['achieved_goal']
                achieved_goal = np.tile(achieved_goal, [1, self.nb_tile])   # expand from 2-d to 256-d
            else:
                obs, _, dones, infos = self.env.step(actions)
            rewards = np.zeros(self.nenv, np.float32)
            death = np.array([False for _ in range(self.nenv)], dtype=np.bool)
            self.episode_step += 1

            # get real next obs and achieved goal
            next_obs = obs.copy()
            next_achieved_goal = goal_info_to_embedding(infos, goal_dim=self.achieved_goal.shape[-1])
            for e in range(self.nenv):
                if dones[e]:
                    if self.dict_obs:
                        _dict_obs = infos[e]['next_obs']
                        _obs = _dict_obs['observation']
                    else:
                        _obs = infos[e].get('next_obs')
                    assert _obs is not None
                    next_obs[e] = _obs
            mb_next_obs.append(next_obs)
            mb_next_obs_infos.append(infos)
            mb_next_achieved_goal.append(next_achieved_goal)

            # achieved & episode done
            for e in range(self.nenv):
                reached = self.check_goal_reached_v2(infos[e], self.desired_goal_info[e])
                if reached or self.episode_step[e] > self.curriculum.allow_step or infos[e]["x_pos"] > self.desired_goal_info[e]["x_pos"] + 100:
                    # log info
                    final_pos = {"x_pos": infos[e]["x_pos"], "y_pos": infos[e]["y_pos"]}
                    if reached:
                        succ = True
                    else:
                        succ = False
                    self.recoder.store(dict(env=e, succ=succ, length=self.episode_step[e], final_pos=final_pos))
                    logger.info(self.TEMPLATE.format(e, succ, self.desired_goal_info[e], final_pos, self.episode_step[e]))

                    # episode info
                    episode_info.update({'succ': succ, 'length': self.episode_step[e], 'final_pos': final_pos})
                    self.episode_step[e] = 0

                    # reward and dones
                    if reached:
                        rewards[e] = 1.0
                    dones[e] = True

                    # reset
                    if self.dict_obs:
                        _dict_obs = self.env.reset_v2(e)
                        obs[e], achieved_goal[e] = _dict_obs['observation'][0], np.tile(_dict_obs['achieved_goal'][0], self.nb_tile)
                        assert np.array_equal(achieved_goal[e], np.tile(np.array([40., 176.]), self.nb_tile))
                    else:
                        _obs = self.env.reset_v2(e)[0]
                        obs[e] = _obs
                    # curriculum
                    self.curriculum.update(succ=succ, acer_steps=acer_steps)
                    self.desired_goal[e], self.desired_goal_state[e], self.desired_goal_info[e] = self.curriculum.get_current_target(nb_goal=1)
                elif dones[e]:
                    # log info
                    final_pos = {"x_pos": infos[e]["x_pos"], "y_pos": infos[e]["y_pos"]}
                    self.recoder.store(dict(env=e, succ=False, length=self.episode_step[e], final_pos=final_pos))
                    logger.info(self.TEMPLATE.format(e, 'fail', self.desired_goal_info[e], final_pos, self.episode_step[e]))

                    # episode info
                    episode_info.update({'succ': False, 'length': self.episode_step[e], 'final_pos': final_pos})
                    self.episode_step[e] = 0

                    # reward and death info
                    if infos[e]['is_dying'] or infos[e]['is_dead']:
                        death[e] = True
                        if self.include_death:
                            rewards[e] = -1
                    # curriculum
                    self.curriculum.update(succ=False, acer_steps=acer_steps)
                    self.desired_goal[e], self.desired_goal_state[e], self.desired_goal_info[e] = self.curriculum.get_current_target(nb_goal=1)
            # states information for statefull models like LSTM
            self.obs = obs
            if self.dict_obs:
                self.achieved_goal = achieved_goal
            mb_rewards.append(rewards)
            mb_death.append(death)
            mb_dones.append(dones)

        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_next_obs = np.asarray(mb_next_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_achieved_goal = np.asarray(mb_achieved_goal, dtype=np.float32).swapaxes(1, 0)
        mb_next_achieved_goal = np.asarray(mb_next_achieved_goal, dtype=np.float32).swapaxes(1, 0)
        mb_desired_goal = np.asarray(mb_desired_goal, dtype=np.float32).swapaxes(1, 0)
        mb_desired_goal_state = np.asarray(mb_desired_goal_state, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)

        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_death = np.asarray(mb_death, dtype=np.bool).swapaxes(1, 0)

        mb_next_obs_infos = np.asarray(mb_next_obs_infos, dtype=object).swapaxes(1, 0)
        mb_desired_goal_infos = np.asarray(mb_desired_goal_infos, dtype=object).swapaxes(1, 0)
        if not np.array_equal(mb_rewards, self.reward_fn(mb_next_obs_infos, mb_desired_goal_infos)):
            import ipdb
            ipdb.set_trace()

        results = dict(
            obs=mb_obs,
            next_obs=mb_next_obs,
            achieved_goal=mb_achieved_goal,
            next_achieved_goal=mb_next_achieved_goal,
            desired_goal=mb_desired_goal,
            desired_goal_state=mb_desired_goal_state,
            actions=mb_actions,
            rewards=mb_rewards,
            mus=mb_mus,
            dones=mb_dones,
            deaths=mb_death,
            next_obs_infos=mb_next_obs_infos,
            desired_goal_infos=mb_desired_goal_infos,
            episode_info=episode_info,
        )
        return results

    def check_goal_reached_v2(self, obs_info, goal_info):
        obs_x, obs_y = float(obs_info["x_pos"]), float(obs_info["y_pos"])
        goal_x, goal_y = float(goal_info["x_pos"]), float(goal_info["y_pos"])
        diff_x = abs(obs_x - goal_x)
        diff_y = abs(obs_y - goal_y)
        if diff_x <= self.threshold[0] and diff_y <= self.threshold[1]:
            status = True
        else:
            status = False
        return status

