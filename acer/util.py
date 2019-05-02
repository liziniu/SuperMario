import time
import numpy as np
from baselines import logger
from common.util import EpisodeStats
from copy import deepcopy


class Acer:
    def __init__(self, runner_expl, runner_eval, model_expl, model_eval, buffer, log_interval, dyna_source_list):
        self.runner_expl = runner_expl
        self.runner_eval = runner_eval
        self.model_expl = model_expl
        self.model_eval = model_eval

        self.buffer = buffer
        self.log_interval = log_interval
        self.tstart = None
        self.steps = 0
        self.nupdates = 0
        self.dyna_source_list = dyna_source_list

        keys = []
        keys += ["expl_return", "eval_return", "expl_length", "eval_length"]
        keys += ["expl_goal_x", "expl_goal_y", "eval_goal_x", "eval_goal_y", "reward_to_go", "goal_x", "goal_y"]
        keys += ["reached_cnt", "reached_time", "goal_dist", "expl_final_x", "expl_final_y",
                 "eval_final_x", "eval_final_y"]
        keys += ["queue_max", "queue_std"]
        keys += ["int_rew_mean", "int_rew_std"]

        self.logger_keys = keys.copy()
        self.logger_keys.remove("reached_cnt")
        self.episode_stats = EpisodeStats(maxlen=50, keys=keys)

        self.goal_as_image = self.model_expl.goal_as_image

    def call(self, on_policy, model_name=None):
        logging = False
        names_ops, values_ops = [], []
        if model_name == "expl":
            runner = self.runner_expl
        else:
            runner = self.runner_eval

        if on_policy:
            # collect data
            results = runner.run()
            self.buffer.put(results["enc_obs"], results["actions"], results["ext_rewards"], results["mus"],
                            results["dones"], results["masks"], results["goal_obs"], results["goal_infos"],
                            results["obs_infos"])
            # training dynamics & put goals
            mb_obs, mb_actions, mb_next_obs, mb_obs_infos = self.adjust_dynamics_input_shape(results)
            if model_name in self.dyna_source_list:
                queue_info = self.model_expl.dynamics.put_goal(mb_obs, mb_actions, mb_next_obs, mb_obs_infos)
                if len(queue_info.keys()) > 0:
                    self.episode_stats.feed(queue_info["queue_max"], "queue_max")
                    self.episode_stats.feed(queue_info["queue_std"], "queue_std")
                names_ops_, values_ops_ = self.model_expl.train_dynamics(mb_obs, mb_actions, mb_next_obs, self.steps)
                names_ops, values_ops = names_ops + names_ops_, values_ops + values_ops_

            # store useful episode information
            self.record_episode_info(results["episode_infos"], model_name)
        else:
            results = self.buffer.get()
        obs, actions, ext_rewards, mus, dones, masks, int_rewards, goal_obs = self.adjust_policy_input_shape(results)

        self.episode_stats.feed(np.mean(int_rewards), "int_rew_mean")
        self.episode_stats.feed(np.std(int_rewards), "int_rew_std")
        # Training Policy
        assert self.model_expl.scope != self.model_eval.scope
        names_ops_, values_ops_ = self.model_eval.train_policy(
            obs, actions, ext_rewards, dones, mus, self.model_eval.initial_state, masks, self.steps, goal_obs)
        names_ops, values_ops = names_ops + names_ops_, values_ops + values_ops_
        if model_name == "expl" or not on_policy:
            names_ops_, values_ops_ = self.model_expl.train_policy(
                obs, actions, int_rewards, dones, mus, self.model_expl.initial_state, masks, self.steps, goal_obs)
            names_ops, values_ops = names_ops + names_ops_, values_ops + values_ops_
            logging = True
        self.nupdates += 1

        # Logging
        if on_policy and self.nupdates % self.log_interval == 0 and logging:
            self.log(names_ops, values_ops)

    def initialize(self):
        init_steps = int(3e3)
        self.runner_expl.initialize(init_steps)

    def evaluate(self, nb_eval):
        results = self.runner_eval.evaluate(nb_eval)
        self.episode_stats.feed(results["l"], "eval_length")
        self.episode_stats.feed(results["r"], "eval_return")

    @staticmethod
    def adjust_dynamics_input_shape(results):
        # flatten on-policy data (nenv, nstep, ...) for dynamics training and put_goal
        mb_next_obs = deepcopy(results["obs"][:, 1:])
        mb_obs = deepcopy(results["obs"][:, :-1])
        mb_obs = mb_obs.reshape((-1,) + mb_obs.shape[2:])
        mb_next_obs = mb_next_obs.reshape((-1,) + mb_next_obs.shape[2:])
        mb_actions = deepcopy(results["actions"])
        mb_actions = mb_actions.reshape((-1,) + mb_actions.shape[2:])
        mb_obs_infos = deepcopy(results["obs_infos"])
        mb_obs_infos = mb_obs_infos.reshape(-1)
        return mb_obs, mb_actions, mb_next_obs, mb_obs_infos

    def adjust_policy_input_shape(self, results):
        assert self.runner_expl.nbatch == self.runner_eval.nbatch
        runner = self.runner_expl

        obs = results["obs"].reshape(runner.batch_ob_shape)
        actions = results["actions"].reshape(runner.nbatch)
        ext_rewards = results["ext_rewards"].reshape(runner.nbatch)
        mus = results["mus"].reshape([runner.nbatch, runner.nact])
        dones = results["dones"].reshape([runner.nbatch])
        masks = results["masks"].reshape([runner.batch_ob_shape[0]])
        int_rewards = results["int_rewards"].reshape([runner.nbatch])
        if not self.goal_as_image:
            goal_obs = self.goal_to_embedding(results["goal_infos"])
            goal_obs = goal_obs.reshape([-1, goal_obs.shape[-1]])
        else:
            goal_obs = results["goal_obs"].reshape(runner.batch_ob_shape)
        return obs, actions, ext_rewards, mus, dones, masks, int_rewards, goal_obs

    @staticmethod
    def goal_to_embedding(goal_infos):
        feat_dim = 512
        nb_tile = feat_dim // 2

        def get_pos(x):
            return x["x_pos"], x["y_pos"]
        vf = np.vectorize(get_pos)
        goal_pos = vf(goal_infos)
        goal_x, goal_y = np.expand_dims(goal_pos[0], -1).astype(np.float32), np.expand_dims(goal_pos[1], -1).astype(np.float32)
        goal_embedding = np.concatenate([goal_x, goal_y], axis=-1)
        goal_embedding = np.tile(goal_embedding, [1]*len(goal_embedding.shape[:-1])+[nb_tile])
        return goal_embedding

    def record_episode_info(self, episode_infos, model_name):
        for info in episode_infos:
            reached_info = info.get("reached_info")
            if reached_info:
                source = reached_info.get("source", "")
                source = source + "_" if source != "" else ""
                if "expl" in source:
                    self.episode_stats.feed(reached_info["reached"], "reached_cnt")
                    self.episode_stats.feed(reached_info["time_ratio"], "reached_time")
                    self.episode_stats.feed(reached_info["abs_dist"], "goal_dist")
                self.episode_stats.feed(reached_info["x_pos"], source + "final_x")
                self.episode_stats.feed(reached_info["y_pos"], source + "final_y")
            goal_info = info.get("goal_info")
            if goal_info:
                source = goal_info.get("source", "")
                source = source + "_" if source != "" else ""
                self.episode_stats.feed(goal_info["x_pos"], source + "goal_x")
                self.episode_stats.feed(goal_info["y_pos"], source + "goal_y")
                self.episode_stats.feed(goal_info["x_pos"], "goal_x")
                self.episode_stats.feed(goal_info["y_pos"], "goal_y")
                if "expl" in source:
                    self.episode_stats.feed(goal_info["reward_to_go"], "reward_to_go")
            return_info = info.get("episode")
            if return_info:
                self.episode_stats.feed(return_info["l"], "{}_length".format(model_name))
                self.episode_stats.feed(return_info["r"], "{}_return".format(model_name))

    def log(self, names_ops, values_ops):
        logger.record_tabular("total_timesteps", self.steps)
        logger.record_tabular("fps", int(self.steps / (time.time() - self.tstart)))
        logger.record_tabular("time_elapse(min)", int(time.time() - self.tstart) // 60)
        logger.record_tabular("nupdates", self.nupdates)
        for key in self.logger_keys:
            logger.record_tabular(key, self.episode_stats.get_mean(key))
        logger.record_tabular("reached_ratio", self.episode_stats.get_sum("reached_cnt") / self.episode_stats.maxlen)
        for name, val in zip(names_ops, values_ops):
            logger.record_tabular(name, float(val))
        logger.dump_tabular()