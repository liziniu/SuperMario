import numpy as np
from baselines.common.runners import AbstractEnvRunner
from common.env_util import VecFrameStack
from gym import spaces
from common.util import DataRecorder, ResultsWriter
from acer.util import goal_to_embedding, check_obs, check_infos, check_goal_reached
import time
import os
from copy import deepcopy
from baselines import logger


class Runner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, store_data, reward_fn, sample_goal, threshold=None, alt_model=None,
                 use_random_policy_expl=None):
        super().__init__(env=env, model=model, nsteps=nsteps)
        assert isinstance(env.action_space,
                          spaces.Discrete), 'This ACER implementation works only with discrete action spaces!'
        assert isinstance(env, VecFrameStack)

        self.nact = env.action_space.n
        nenv = self.nenv
        self.nbatch = nenv * nsteps
        self.batch_ob_shape = (nenv * (nsteps + 1),) + env.observation_space.shape

        # self.obs = env.reset()  super method do this
        self.obs_dtype = env.observation_space.dtype
        self.obs_shape = env.observation_space.shape
        self.ac_dtype = env.action_space.dtype
        self.ac_shape = env.action_space.shape
        self.nstack = self.env.nstack
        self.nc = self.batch_ob_shape[-1] // self.nstack
        self.goal_shape = self.model.goal_shape
        self.goal_as_image = self.model.goal_as_image

        self.save_path = os.path.join(logger.get_dir(), "runner_data")
        self.store_data = store_data
        self.recorder = DataRecorder(self.save_path)

        self.dynamics = self.model.dynamics
        self.sample_goal = sample_goal
        self.threshold = threshold
        # self.batch_goal_feat_shape = (nenv*(nsteps+1),) + env.observation_space.shape + self.dynamics.feat_shape
        self.reached_status = np.array([False for _ in range(self.nenv)], dtype=bool)
        self.goals, self.goal_info = None, None
        self.reward_fn = reward_fn
        # self.results_writer = ResultsWriter(os.path.join(save_path, "evaluation.csv"))

        self.episode = np.ones(self.nenv)
        self.episode_step = np.zeros(self.nenv)
        self.episode_reached_step = np.zeros(self.nenv)
        self.episode_reward_to_go = np.zeros(self.nenv)

        self.name = self.model.scope.split("acer_")[1]

        self.alt_model = alt_model
        self.use_random_policy_expl = use_random_policy_expl
        if self.use_random_policy_expl:
            assert alt_model is not None

    def run(self, acer_step=None):
        if self.goals is None:
            self.goals, self.goal_info = self.dynamics.get_goal(nb_goal=self.nenv)
            if not self.goal_as_image:
                self.goals = goal_to_embedding(self.goal_info)
        mb_obs = np.empty((self.nenv, self.nsteps) + self.obs_shape, dtype=self.obs_dtype)
        mb_next_obs = np.empty((self.nenv, self.nsteps) + self.obs_shape, dtype=self.obs_dtype)
        mb_act = np.empty((self.nenv, self.nsteps) + self.ac_shape, dtype=self.ac_dtype)
        mb_mus = np.empty((self.nenv, self.nsteps, self.nact), dtype=np.float32)
        mb_ext_dones = np.empty((self.nenv, self.nsteps), dtype=bool)
        mb_int_dones = np.empty((self.nenv, self.nsteps), dtype=bool)
        mb_masks = np.empty((self.nenv, self.nsteps + 1), dtype=bool)
        mb_ext_rew = np.empty((self.nenv, self.nsteps), dtype=np.float32)
        mb_next_obs_infos = np.empty((self.nenv, self.nsteps), dtype=object)
        mb_goals = np.empty((self.nenv, self.nsteps) + self.goal_shape, dtype=self.obs_dtype)
        mb_goal_infos = np.empty((self.nenv, self.nsteps), dtype=object)

        reached_step, done_step = np.array([None for _ in range(self.nenv)]), np.array([None for _ in range(self.nenv)])

        episode_infos = np.asarray([{} for _ in range(self.nenv)], dtype=object)
        for step in range(self.nsteps):
            check_obs(self.obs)

            actions, mus, states = self.model.step(self.obs, S=self.states, M=self.dones, goals=self.goals)
            if self.sample_goal:
                if self.use_random_policy_expl:
                    actions[self.reached_status] = self.simple_random_action(np.sum(self.reached_status))
                    mus[self.reached_status] = self.get_mu_of_random_action()
                else:
                    if np.sum(self.reached_status) > 0:
                        alt_action, alt_mu, alt_states = self.alt_model.step(self.obs, S=self.states, M=self.dones, goals=self.goals)
                        actions[self.reached_status] = alt_action[self.reached_status]
                        mus[self.reached_status] = alt_mu[self.reached_status]

            mb_obs[:, step] = deepcopy(self.obs)
            mb_act[:, step] = actions
            mb_mus[:, step, :] = mus
            mb_masks[:, step] = deepcopy(self.dones)

            obs, rewards, dones, infos = self.env.step(actions)
            check_infos(infos, self.recorder, dones, acer_step)
            for info in infos: info.update({"source": self.name})

            mb_ext_dones[:, step] = dones
            mb_ext_rew[:, step] = rewards
            self.episode_reward_to_go[self.reached_status] += rewards[self.reached_status]
            mb_next_obs[:, step] = self.get_real_next_obs(obs, dones, infos)
            mb_next_obs_infos[:, step] = np.asarray(infos, dtype=object)
            mb_goals[:, step] = deepcopy(self.goals)
            mb_goal_infos[:, step] = deepcopy(self.goal_info)
            self.episode_step += 1
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.obs = obs

            # check reached
            reached_step = self.update_reach(reached_step, infos, step)
            # check done
            done_step[self.dones] = step

            # revise goal
            if not self.sample_goal:
                mb_goals, mb_goal_infos = self.update_goal_v1(mb_next_obs, mb_goals, infos, mb_goal_infos, done_step, step)
            else:
                mb_goals, mb_goal_infos = self.update_goal_v2(mb_next_obs, mb_goals, infos, mb_goal_infos, reached_step, step)
            # summary
            episode_infos = self.summary(episode_infos, infos, acer_step)

        mb_int_rewards = self.reward_fn(mb_next_obs_infos, mb_goal_infos)
        mb_int_dones.fill(False)
        int_dones_index = np.where(mb_int_rewards)
        mb_int_dones[int_dones_index] = True
        # shapes are adjusted to [nenv, nsteps, []]

        self.recorder.dump()
        results = dict(
            obs=mb_obs,
            next_obs=mb_next_obs,
            actions=mb_act,
            ext_rewards=mb_ext_rew,
            mus=mb_mus,
            ext_dones=mb_ext_dones,
            int_dones=mb_int_dones,
            masks=mb_masks,
            next_obs_infos=mb_next_obs_infos,  # nenv, nsteps, two purpose: 1)put into dynamics; 2) put into buffer
            episode_infos=episode_infos,
            goal_obs=mb_goals,  # nenv, nsteps+1,
            goal_infos=mb_goal_infos,
            int_rewards=mb_int_rewards
        )
        return results

    def simple_random_action(self, nb_action):
        return np.random.randint(0, self.env.action_space.n, nb_action)

    def get_mu_of_random_action(self):
        assert isinstance(self.env.action_space, spaces.Discrete)
        return np.array([1 / self.env.action_space.n for _ in range(self.env.action_space.n)])

    def initialize(self, init_steps):
        mb_obs, mb_actions, mb_next_obs, mb_goal_infos = [], [], [], []
        for _ in range(init_steps):
            mb_obs.append(deepcopy(self.obs))
            actions = np.asarray([self.env.action_space.sample() for _ in range(self.nenv)])
            self.obs, rewards, dones, infos = self.env.step(actions)
            goal_infos = np.array([{"x_pos": info.get("x_pos", None),
                                    "y_pos": info.get("y_pos", None),
                                    "source": self.name} for info in infos], dtype=object)
            next_obs = self.get_real_next_obs(np.copy(self.obs), dones, infos)
            mb_next_obs.append(next_obs)
            mb_goal_infos.append(goal_infos)
            mb_actions.append(actions)

        mb_obs = np.asarray(mb_obs).swapaxes(1, 0)  # (nenv, nstep, obs_shape)
        mb_goal_infos = np.asarray(mb_goal_infos, dtype=object).swapaxes(1, 0)  # (nenv, nstep, dict)
        mb_actions = np.asarray(mb_actions).swapaxes(1, 0)
        mb_next_obs = np.asarray(mb_next_obs).swapaxes(1, 0)

        mb_obs = mb_obs.reshape((-1,) + mb_obs.shape[2:])
        mb_goal_infos = mb_goal_infos.reshape(-1, )
        mb_actions = mb_actions.reshape((-1,) + mb_actions.shape[2:])
        mb_next_obs = mb_next_obs.reshape((-1,) + mb_next_obs.shape[2:])

        for i in range(10):
            batch_size = min(64, init_steps)
            ind = np.random.randint(0, init_steps, batch_size)
            obs, actions, next_obs = mb_obs[ind], mb_actions[ind], mb_next_obs[ind]
            nb_train_epoch = 1
            self.model.train_dynamics(obs, actions, next_obs, nb_train_epoch)
        self.dynamics.put_goal(mb_obs, mb_actions, mb_next_obs, mb_goal_infos)
        self.obs = self.env.reset()

    def evaluate(self, nb_eval):
        assert self.dynamics.dummy
        goal_obs, goal_info = self.dynamics.get_goal(nb_goal=self.nenv)  # (nenv, goal_dim)
        eval_info = {"l": 0, "r": 0, "x_pos":0, "y_pos":0}
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
                    eval_info["x_pos"] += info.get("x_pos")
                    eval_info["y_pos"] += info.get("y_pos")
                    terminal = True
                if terminal:
                    break
                self.states = states
                self.dones = dones
                self.obs = obs
            self.obs = self.env.reset()
        for key in eval_info.keys():
            eval_info[key] /= nb_eval
        return eval_info

    def log(self, mem):
        succ = "succ" if mem["is_succ"] else "fail"
        template = "env_{} {}|goal:{}|final_pos:{}|size:{}".format(
            mem["env"], succ, {"x_pos": mem["goal"]["x_pos"], "y_pos": mem["goal"]["y_pos"]},
            mem["final_pos"], self.dynamics.queue.qsize()
        )
        logger.info(template)

    def summary(self, episode_infos, infos, acer_step):
        for env_idx in range(self.nenv):
            info = infos[env_idx]
            if self.dones[env_idx]:
                assert info.get("episode")
                if info.get("episode"):
                    episode_infos[env_idx]["episode"] = info.get("episode")
                if not self.sample_goal:
                    episode_infos[env_idx]["reached_info"] = dict(source=self.name,
                                                                  x_pos=infos[env_idx]["x_pos"],
                                                                  y_pos=infos[env_idx]["y_pos"])
                else:
                    if self.reached_status[env_idx]:
                        reached = 1.0
                        time_ratio = self.episode_reached_step[env_idx] / self.episode_step[env_idx]
                        achieved_pos = {"x_pos": infos[env_idx]["x_pos"], "y_pos": infos[env_idx]["y_pos"]}
                        mem = dict(env=env_idx, is_succ=True, goal=self.goal_info[env_idx], final_pos=achieved_pos,
                                   timestep=acer_step, episode=self.episode[env_idx], step=self.episode_step[env_idx])
                        self.recorder.store(mem)
                        self.log(mem)
                        abs_dist = 10
                    else:
                        reached = 0.0
                        time_ratio = 1.0
                        achieved_pos = {"x_pos": infos[env_idx]["x_pos"], "y_pos": infos[env_idx]["y_pos"]}
                        mem = dict(env=env_idx, is_succ=False, goal=self.goal_info[env_idx], final_pos=achieved_pos,
                                   timestep=acer_step, episode=self.episode[env_idx], step=self.episode_step[env_idx])
                        self.recorder.store(mem)
                        self.log(mem)
                        abs_dist = abs(float(infos[env_idx]["x_pos"]) - float(self.goal_info[env_idx]["x_pos"])) + \
                                   abs(float(infos[env_idx]["y_pos"]) - float(self.goal_info[env_idx]["y_pos"]))
                    episode_infos[env_idx]["reached_info"] = dict(reached=reached, time_ratio=time_ratio,
                                                                  abs_dist=abs_dist, source=self.name,
                                                                  x_pos=infos[env_idx]["x_pos"],
                                                                  y_pos=infos[env_idx]["y_pos"])
                    episode_infos[env_idx]["goal_info"] = dict(x_pos=self.goal_info[env_idx]["x_pos"],
                                                               y_pos=self.goal_info[env_idx]["y_pos"],
                                                               source=self.goal_info[env_idx]["source"],
                                                               reward_to_go=self.episode_reward_to_go[env_idx])
                    # re-plan goal
                    goal_obs, goal_info = self.dynamics.get_goal(nb_goal=1)
                    if self.goal_as_image:
                        self.goals[env_idx] = goal_obs[0]
                    else:
                        self.goals[env_idx] = goal_to_embedding(goal_info[0])
                    self.goal_info[env_idx] = goal_info[0]
                    self.episode[env_idx] += 1
                    self.episode_step[env_idx] = 0
                    self.episode_reached_step[env_idx] = 0
                    self.reached_status[env_idx] = False
                    self.episode_reward_to_go[env_idx] = 0
        return episode_infos

    def get_real_next_obs(self, next_obs, dones, infos):
        _next_obs = next_obs.copy()
        for env_idx in range(self.nenv):
            if dones[env_idx]:
                o = infos[env_idx].get("next_obs", None)
                assert o is not None
                _next_obs[env_idx] = o
        return _next_obs

    def update_reach(self, reached_step, infos, step):
        if self.sample_goal:
            for env_idx in range(self.nenv):
                if not self.reached_status[env_idx]:
                    self.reached_status[env_idx] = check_goal_reached(infos[env_idx], self.goal_info[env_idx], self.threshold)
                    if self.reached_status[env_idx]:
                        reached_step[env_idx] = step
                        self.episode_reached_step[env_idx] = deepcopy(self.episode_step[env_idx])
        return reached_step

    def update_goal_v1(self, mb_next_obs, mb_goals, infos, mb_goal_infos, done_step, step):
        assert not self.sample_goal
        for env_idx in range(self.nenv):
            if self.dones[env_idx]:
                # (- - done(t)) -> (done done, done(t))
                start, end = 0, step + 1
                if self.goal_as_image:
                    mb_goals[env_idx, start:end] = mb_next_obs[env_idx, step]
                else:
                    mb_goals[env_idx, start:end] = goal_to_embedding(infos[env_idx])
                mb_goal_infos[env_idx, start:end] = infos[env_idx]
            elif step == self.nsteps - 1:
                if done_step[env_idx] is None:
                    # (- - t) -> (t, t, t)
                    start = 0
                else:
                    # (- - done - - t) -> (- - - t, t, t)
                    start = done_step[env_idx] + 1
                end = step + 1
                if end == start:
                    continue
                if self.goal_as_image:
                    mb_goals[env_idx, start:end] = mb_next_obs[env_idx, step]
                else:
                    mb_goals[env_idx, start:end] = goal_to_embedding(infos[env_idx])
                mb_goal_infos[env_idx, start:end] = infos[env_idx]
        return mb_goals, mb_goal_infos
    
    def update_goal_v2(self, mb_next_obs, mb_goals, infos, mb_goal_infos, reached_step, step):
        assert self.sample_goal
        for env_idx in range(self.nenv):
            if step != self.nsteps - 1:
                # dones is instant variable but reached_status is a transitive variable
                if self.dones[env_idx] and self.reached_status[env_idx]:
                    if reached_step[env_idx] is None:
                        # reach|[- - done] -> [done, done, done]
                        start, end = 0, step + 1
                        if self.goal_as_image:
                            mb_goals[env_idx, start:end] = mb_next_obs[env_idx, step]
                        else:
                            mb_goals[env_idx, start:end] = goal_to_embedding(infos[env_idx])
                        mb_goal_infos[env_idx, start:end] = infos[env_idx]
                    else:
                        # [- - reach(done)] -> [ - - -]  if reached_step[env_idx] == step
                        # [- - reach - - done] -> [- - - done done done]
                        start, end = reached_step[env_idx] + 1, step + 1
                        if end == start:
                            continue
                        if self.goal_as_image:
                            mb_goals[env_idx, start:end] = mb_next_obs[env_idx, step]
                        else:
                            mb_goals[env_idx, start:end] = goal_to_embedding(infos[env_idx])
                        mb_goal_infos[env_idx, start:end] = infos[env_idx]
                elif not self.dones[env_idx] and self.reached_status[env_idx]:
                    # reached|[ - - -]  if reached_step[env_idx] is None:
                    # [- - reached - -] if reached_step[env_idx] is not None
                    pass
                else:
                    # [- - - done] if self.dones[env_idx] and not self.reached_status[env_idx]
                    # [- - - - -] if not self.dones[env_idx] and not self.reached_status[env_idx]
                    pass
            else:
                if self.dones[env_idx] and self.reached_status[env_idx]:
                    if reached_step[env_idx] is None:
                        # reach|[- - done(t)] -> [done, done, done(t)]
                        start, end = 0, step + 1
                        if self.goal_as_image:
                            mb_goals[env_idx, start:end] = mb_next_obs[env_idx, step]
                        else:
                            mb_goals[env_idx, start:end] = goal_to_embedding(infos[env_idx])
                        mb_goal_infos[env_idx, start:end] = infos[env_idx]
                    else:
                        # [- - reach(done)(t)] -> [- - -]
                        # [- - reach - - done(t)] -> [- - - done done done(t)]
                        start, end = reached_step[env_idx] + 1, step + 1
                        if end == start:
                            continue
                        if self.goal_as_image:
                            mb_goals[env_idx, start:end] = mb_next_obs[env_idx, step]
                        else:
                            mb_goals[env_idx, start:end] = goal_to_embedding(infos[env_idx])
                        mb_goal_infos[env_idx, start:end] = infos[env_idx]
                elif not self.dones[env_idx] and self.reached_status[env_idx]:
                    if reached_step[env_idx] is None:
                        # reached|[ - - t]  -> reached|[t t t]
                        start, end = 0, step + 1
                    else:
                        # reached[- - r - -] -> reached|[- - - t t]
                        start, end = reached_step[env_idx] + 1, step + 1
                    if end == start:
                        continue
                    if self.goal_as_image:
                        mb_goals[env_idx, start:end] = mb_next_obs[env_idx, step]
                    else:
                        mb_goals[env_idx, start:end] = goal_to_embedding(infos[env_idx])
                else:
                    # [- - - done(t)]  if self.dones[env_idx] and not self.reached_status[env_idx]
                    # [- - - - (t)] if not self.dones[env_idx] and not self.reached_status[env_idx]
                    pass
        return mb_goals, mb_goal_infos
