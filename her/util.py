import time
import numpy as np
from baselines import logger
from common.util import EpisodeStats
from copy import deepcopy
from baselines.common.tf_util import save_variables
import functools
import os
import sys


class Acer:
    def __init__(self, runner, model, buffer, log_interval):
        self.runner = runner
        self.model = model
        self.buffer = buffer
        self.log_interval = log_interval
        self.tstart = None
        self.keys = ["episode_return", "episode_length", "succ_ratio", "final_x_pos", "final_y_pos", "rewards", "her_gain"]
        self.keys += ["put_time", "get_first", "get_second"]
        self.episode_stats = EpisodeStats(maxlen=10, keys=self.keys)
        self.steps = 0

        sess = self.model.sess
        self.save = functools.partial(save_variables, sess=sess, variables=self.model.params)

    def call(self, replay_start, nb_train_epoch):
        runner, model, buffer, steps = self.runner, self.model, self.buffer, self.steps

        results = runner.run(acer_steps=self.steps)
        if buffer is not None:
            tstart = time.time()
            buffer.put(results)
            self.episode_stats.feed(time.time()-tstart, "put_time")
        self.record_episode_info(results["episode_info"])
        policy_train_inputs = self.adjust_shape(results)
        names_ops, values_ops = model.train_policy(**policy_train_inputs)
        if buffer.has_atleast(replay_start):
            for i in range(nb_train_epoch):
                if i == 0:
                    tstart = time.time()
                    results = buffer.get(use_cache=False)
                    self.episode_stats.feed(time.time()-tstart, "get_first")
                else:
                    tstart = time.time()
                    results = buffer.get(use_cache=True)
                    self.episode_stats.feed(time.time()-tstart, "get_second")
                policy_train_inputs = self.adjust_shape(results)
                names_ops, values_ops = model.train_policy(**policy_train_inputs)
                rewards = policy_train_inputs["rewards"]
                self.episode_stats.feed(np.mean(rewards), "rewards")
                self.episode_stats.feed(results["her_gain"], "her_gain")

        if int(steps/runner.nbatch) % self.log_interval == 0:
            names_ops, values_ops = names_ops + ["memory_usage(GB)"], values_ops + [self.buffer.memory_usage]
            self.log(names_ops, values_ops)

            if int(steps/runner.nbatch) % (self.log_interval * 200) == 0:
                self.save(os.path.join(logger.get_dir(), "{}.pkl".format(self.steps)))

    def adjust_shape(self, results):
        runner = self.runner

        obs = results["obs"].reshape((runner.nbatch, ) + runner.obs_shape)
        next_obs = results["next_obs"].reshape((runner.nbatch, ) + runner.obs_shape)
        achieved_goal = results["achieved_goal"].reshape((runner.nbatch, ) + runner.achieved_goal_shape)
        next_achieved_goal = results["next_achieved_goal"].reshape((runner.nbatch, ) + runner.achieved_goal_shape)
        desired_goal = results["desired_goal"].reshape((runner.nbatch, ) + runner.desired_goal_shape)
        desired_goal_state = results["desired_goal_state"].reshape((runner.nbatch, ) + runner.desired_goal_state_shape)
        actions = results["actions"].reshape(runner.nbatch)
        rewards = results["rewards"].reshape(runner.nbatch)
        mus = results["mus"].reshape([runner.nbatch, runner.nact])
        dones = results["dones"].reshape([runner.nbatch])

        results = {
            "obs": obs,
            "next_obs": next_obs,
            "achieved_goal": achieved_goal,
            "next_achieved_goal": next_achieved_goal,
            "desired_goal": desired_goal,
            "desired_goal_state": desired_goal_state,
            "actions": actions,
            "rewards": rewards,
            "mus": mus,
            "dones": dones,
            "steps": self.steps
        }
        return results

    def record_episode_info(self, episode_info):
        returns = episode_info.get("episode", None)
        succ = episode_info.get("succ", None)
        length = episode_info.get("length", None)
        final_pos = episode_info.get("final_pos", None)
        if returns:
            self.episode_stats.feed(returns["r"], "episode_return")
        if length:
            self.episode_stats.feed(length, "episode_length")
        if succ:
            self.episode_stats.feed(succ, "succ_ratio")
            # self.episode_stats.feed(length, "succ_length")
        elif succ is not None:
            self.episode_stats.feed(succ, "succ_ratio")
        if final_pos:
            self.episode_stats.feed(final_pos["x_pos"], "final_x_pos")
            self.episode_stats.feed(final_pos["y_pos"], "final_y_pos")

    def log(self, names_ops, values_ops):
        logger.record_tabular("total_timesteps", self.steps)
        logger.record_tabular("fps", int(self.steps / (time.time() - self.tstart)))
        for name, val in zip(names_ops, values_ops):
            logger.record_tabular(name, float(val))
        for key in self.keys:
            logger.record_tabular(key, self.episode_stats.get_mean(key))
        logger.dump_tabular()
