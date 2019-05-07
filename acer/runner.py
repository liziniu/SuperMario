import numpy as np
from baselines.common.runners import AbstractEnvRunner
from common.env_util import VecFrameStack
from gym import spaces
from common.util import DataRecorder, ResultsWriter
import time
import os
from copy import deepcopy
from baselines import logger


def check_obs(obs):
    nenv, nc = obs.shape[0], obs.shape[-1]
    for i in range(nenv):
        for c in range(nc):
            if np.sum(obs[i][:, :, c]) == 0:
                raise ValueError


def check_infos(infos):
    stage, world = infos[0].get("stage"), infos[0].get("world")
    for info in infos[1:]:
        if info.get("stage") != stage:
            raise ValueError
        if info.get("world") != stage:
            raise ValueError


class Runner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, store_data, reward_fn, sample_goal, dist_type, alt_model=None,
                 use_random_policy_expl=None,):
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

        assert dist_type in ["l1", "l2"]
        self.dist_type = dist_type
        self.alt_model = alt_model
        self.use_random_policy_expl = use_random_policy_expl
        if self.use_random_policy_expl:
            assert alt_model is not None

    def run(self, acer_step=None):
        if self.goals is None:
            self.goals, self.goal_info = self.dynamics.get_goal(nb_goal=self.nenv)
            if not self.goal_as_image:
                self.goals = self.goal_to_embedding(self.goal_info)
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        mb_obs = np.empty((self.nenv, self.nsteps + 1) + self.obs_shape, dtype=self.obs_dtype)
        mb_act = np.empty((self.nenv, self.nsteps) + self.ac_shape, dtype=self.ac_dtype)
        mb_mus = np.empty((self.nenv, self.nsteps, self.nact), dtype=np.float32)
        mb_dones = np.empty((self.nenv, self.nsteps), dtype=bool)
        mb_masks = np.empty((self.nenv, self.nsteps + 1), dtype=bool)
        mb_ext_rew = np.empty((self.nenv, self.nsteps), dtype=np.float32)
        mb_obs_infos = np.empty((self.nenv, self.nsteps), dtype=object)
        mb_goals = np.empty((self.nenv, self.nsteps + 1) + self.goal_shape, dtype=self.obs_dtype)
        mb_goal_infos = np.empty((self.nenv, self.nsteps), dtype=object)

        # mb_obs, mb_actions, mb_mus, mb_dones, mb_ext_rewards = [], [], [], [], []
        # mb_obs_infos, mb_goals, mb_goal_infos = [], [], []
        reached_step, done_step = np.array([None for _ in range(self.nenv)]), np.array([None for _ in range(self.nenv)])

        episode_infos = np.asarray([{} for _ in range(self.nenv)], dtype=object)
        for step in range(self.nsteps):
            try:
                check_obs(self.obs)
            except ValueError:
                logger.warn("acer_step:{}, runner_step:{}, empty obs".format(acer_step, step))
                raise ValueError
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
            try:
                check_infos(infos)
            except ValueError:
                logger.warn("warning!wrong infos!program continues anyway")
                logger.info("infos:{}, dones:{}, acer_step:{}".format(infos, dones, acer_step))
                logger.info("please debug it in runner_data/data.pkl")
                self.recorder.store(infos)
                self.recorder.dump()
            for info in infos:
                info.update({"source": self.name})

            enc_obs.append(obs[..., -self.nc:])
            mb_dones[:, step] = dones
            mb_ext_rew[:, step] = rewards
            self.episode_reward_to_go[self.reached_status] += rewards[self.reached_status]
            mb_obs_infos[:, step] = np.asarray(infos, dtype=object)
            mb_goals[:, step] = deepcopy(self.goals)
            mb_goal_infos[:, step] = deepcopy(self.goal_info)
            self.episode_step += 1
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.obs = obs

            # check reached
            if self.sample_goal:
                for env_idx in range(self.nenv):
                    if not self.reached_status[env_idx]:
                        if self.dist_type == "l1":
                            self.reached_status[env_idx] = self.check_goal_reached_v2(infos[env_idx],
                                                                                      self.goal_info[env_idx])
                        else:
                            raise NotImplementedError("I do not know how to compute goal_latent")
                        if self.reached_status[env_idx]:
                            reached_step[env_idx] = step
                            self.episode_reached_step[env_idx] = deepcopy(self.episode_step[env_idx])

            # check done
            done_step[self.dones] = step

            # revise goal
            if not self.sample_goal:
                for env_idx in range(self.nenv):
                    if self.dones[env_idx]:
                        # (- - done(t)) -> (done done, done(t))
                        start, end = 0, step + 1
                        if self.goal_as_image:
                            mb_goals[env_idx, start:end] = mb_obs[env_idx, step] 
                        else:
                            mb_goals[env_idx, start:end] = self.goal_to_embedding(infos[env_idx])
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
                            mb_goals[env_idx, start:end] = mb_obs[env_idx, step]
                        else:
                            mb_goals[env_idx, start:end] = self.goal_to_embedding(infos[env_idx])
                        mb_goal_infos[env_idx, start:end] = infos[env_idx]
            else:
                for env_idx in range(self.nenv):
                    if step != self.nsteps - 1:
                        # dones is instant variable but reached_status is a transitive variable
                        if self.dones[env_idx] and self.reached_status[env_idx]:
                            if reached_step[env_idx] is None:
                                # reach|[- - done] -> [done, done, done]
                                start, end = 0, step + 1
                                if self.goal_as_image:
                                    mb_goals[env_idx, start:end] = mb_obs[env_idx, step]
                                else:
                                    mb_goals[env_idx, start:end] = self.goal_to_embedding(infos[env_idx])
                                mb_goal_infos[env_idx, start:end] = infos[env_idx]
                            else:
                                # [- - reach(done)] -> [ - - -]  if reached_step[env_idx] == step
                                # [- - reach - - done] -> [- - - done done done]
                                start, end = reached_step[env_idx] + 1, step + 1
                                if end == start:
                                    continue
                                if self.goal_as_image:
                                    mb_goals[env_idx, start:end] = mb_obs[env_idx, step]
                                else:
                                    mb_goals[env_idx, start:end] = self.goal_to_embedding(infos[env_idx])
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
                                    mb_goals[env_idx, start:end] = mb_obs[env_idx, step]
                                else:
                                    mb_goals[env_idx, start:end] = self.goal_to_embedding(infos[env_idx])
                                mb_goal_infos[env_idx, start:end] = infos[env_idx]
                            else:
                                # [- - reach(done)(t)] -> [- - -]
                                # [- - reach - - done(t)] -> [- - - done done done(t)]
                                start, end = reached_step[env_idx] + 1, step + 1
                                if end == start:
                                    continue
                                if self.goal_as_image:
                                    mb_goals[env_idx, start:end] = mb_obs[env_idx, step]
                                else:
                                    mb_goals[env_idx, start:end] = self.goal_to_embedding(infos[env_idx])
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
                                mb_goals[env_idx, start:end] = mb_obs[env_idx, step]
                            else:
                                mb_goals[env_idx, start:end] = self.goal_to_embedding(infos[env_idx])
                        else:
                            # [- - - done(t)]  if self.dones[env_idx] and not self.reached_status[env_idx]
                            # [- - - - (t)] if not self.dones[env_idx] and not self.reached_status[env_idx]
                            pass
            # summary
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
                            self.goals[env_idx] = self.goal_to_embedding(goal_info[0])
                        self.goal_info[env_idx] = goal_info[0]
                        self.episode[env_idx] += 1
                        self.episode_step[env_idx] = 0
                        self.episode_reached_step[env_idx] = 0
                        self.reached_status[env_idx] = False
                        self.episode_reward_to_go[env_idx] = 0

        # next obs and next goal
        mb_obs[:, -1] = deepcopy(self.obs)
        mb_goals[:, -1] = mb_goals[:, -2]  # we cannot use self.goal since it way be revised

        if self.dist_type == "l2":
            raise NotImplementedError
        else:
            mb_int_rewards = self.reward_fn(mb_obs_infos, mb_goal_infos)
        # shapes are adjusted to [nenv, nsteps, []]
        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)

        self.recorder.dump()
        results = dict(
            enc_obs=enc_obs,
            obs=mb_obs,
            actions=mb_act,
            ext_rewards=mb_ext_rew,
            mus=mb_mus,
            dones=mb_dones,
            masks=mb_masks,
            obs_infos=mb_obs_infos,  # nenv, nsteps, two purpose: 1)put into dynamics; 2) put into buffer
            episode_infos=episode_infos,
            goal_obs=mb_goals,  # nenv, nsteps+1,
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
        return np.array([1 / self.env.action_space.n for _ in range(self.env.action_space.n)])

    @staticmethod
    def goal_to_embedding(goal_infos):
        feat_dim = 512
        nb_tile = feat_dim // 2
        if isinstance(goal_infos, dict):
            goal_embedding = np.array([goal_infos["x_pos"], goal_infos["y_pos"]], dtype=np.float32).reshape(1, 2)
            goal_embedding = np.tile(goal_embedding, [1]*len(goal_embedding.shape[:-1])+[nb_tile])
            return goal_embedding
        
        def get_pos(x):
            return float(x["x_pos"]), float(x["y_pos"])
        vf = np.vectorize(get_pos)
        goal_pos = vf(goal_infos)
        goal_x, goal_y = np.expand_dims(goal_pos[0], -1).astype(np.float32), np.expand_dims(goal_pos[1], -1).astype(np.float32)
        goal_embedding = np.concatenate([goal_x, goal_y], axis=-1)
        goal_embedding = np.tile(goal_embedding, [1]*len(goal_embedding.shape[:-1])+[nb_tile])
        return goal_embedding

    def initialize(self, init_steps):
        mb_obs, mb_actions, mb_next_obs, mb_goal_infos = [], [], [], []
        for _ in range(init_steps):
            mb_obs.append(deepcopy(self.obs))
            actions = np.asarray([self.env.action_space.sample() for _ in range(self.nenv)])
            self.obs, rewards, dones, infos = self.env.step(actions)
            goal_infos = np.array([{"x_pos": info.get("x_pos", None),
                                    "y_pos": info.get("y_pos", None),
                                    "source": self.name} for info in infos], dtype=object)
            mb_goal_infos.append(goal_infos)
            mb_actions.append(actions)
            mb_next_obs.append(deepcopy(self.obs))
        mb_obs = np.asarray(mb_obs).swapaxes(1, 0)  # (nenv, nstep, obs_shape)
        mb_goal_infos = np.asarray(mb_goal_infos, dtype=object).swapaxes(1, 0)  # (nenv, nstep, dict)
        mb_actions = np.asarray(mb_actions).swapaxes(1, 0)
        mb_next_obs = np.asarray(mb_next_obs).swapaxes(1, 0)

        batch_size = min(128, init_steps)
        ind = np.random.randint(0, init_steps, batch_size)
        mb_obs = mb_obs.reshape((-1,) + mb_obs.shape[2:])[ind]
        mb_goal_infos = mb_goal_infos.reshape(-1, )[ind]
        mb_actions = mb_actions.reshape((-1,) + mb_actions.shape[2:])[ind]
        mb_next_obs = mb_next_obs.reshape((-1,) + mb_next_obs.shape[2:])[ind]

        for i in range(10):
            self.model.train_dynamics(mb_obs, mb_actions, mb_next_obs, 0)
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

    def log(self, mem):
        succ = "succ" if mem["is_succ"] else "fail"
        template = "env_{} {}|goal:{}|final_pos:{}|size:{}".format(
            mem["env"], succ, {"x_pos": mem["goal"]["x_pos"], "y_pos": mem["goal"]["y_pos"]},
            mem["final_pos"], self.dynamics.queue.qsize()
        )
        logger.info(template)

if __name__ == "__main__":
    # test vectorize f
    infos = []
    nenv, nstep = 1000, 2000
    for i in range(nenv * nstep):
        infos.append({"x_pos": np.random.randint(100), "y_pos": np.random.randint(100)})
    infos = np.array(infos, dtype=object).reshape([nenv, nstep])

    t = time.time()
    results1 = np.empty(shape=[nenv, nstep])
    for i in range(nenv):
        for j in range(nstep):
            results1[i][j] = infos[i][j]["x_pos"] - infos[i][j]["y_pos"]
    print("time:{}".format(time.time() -t))

    t = time.time()
    def f(x):
        return x["x_pos"] - x["y_pos"]

    vf = np.vectorize(f)
    results2 = vf(infos)
    print("time:{}".format(time.time() - t))
    assert np.sum(results1 - results2) < 1e-4