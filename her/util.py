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
        self.keys = ["rewards", "length"]
        self.episode_stats = EpisodeStats(maxlen=10, keys=self.keys)
        self.steps = 0

        sess = self.model.sess
        self.save = functools.partial(save_variables, sess=sess, params=self.model.params)

    def call(self, replay_start, nb_train_epoch):
        runner, model, buffer, steps = self.runner, self.model, self.buffer, self.steps

        results = runner.run()
        if buffer is not None:
            buffer.put(results["enc_obs"], results["actions"], results["ext_rewards"], results["mus"], results["dones"],
                       results["masks"], results["goals"], results["goal_infos"], results["obs_infos"])

        if buffer.has_atleast(replay_start):
            names_ops, values_ops = [], []
            for i in range(nb_train_epoch):
                results = buffer.get()
                obs, actions, ext_rewards, mus, dones, masks, int_rewards, goal_obs = self.adjust_shape(results)
                names_ops, values_ops = model.train_policy(
                    obs, actions, int_rewards, dones, mus, model.initial_state, masks, steps, goal_obs)

            if int(steps/runner.nbatch) % self.log_interval == 0:
                logger.record_tabular("total_timesteps", steps)
                logger.record_tabular("fps", int(steps/(time.time() - self.tstart)))
                for name, val in zip(names_ops, values_ops):
                    logger.record_tabular(name, float(val))
                logger.dump_tabular()

                if int(steps/runner.nbatch) % (self.log_interval * 100) == 0:
                    self.save(os.path.join(logger.get_dir(), "{}.pkl".format(self.steps)))

    def adjust_shape(self, results):
        runner = self.runner

        obs = results["obs"].reshape(runner.batch_ob_shape)
        actions = results["actions"].reshape(runner.nbatch)
        ext_rewards = results["ext_rewards"].reshape(runner.nbatch)
        mus = results["mus"].reshape([runner.nbatch, runner.nact])
        dones = results["dones"].reshape([runner.nbatch])
        masks = results["masks"].reshape([runner.batch_ob_shape[0]])
        int_rewards = results["int_rewards"].reshape([runner.nbatch])
        goal_obs = results["goal_obs"].reshape(runner.batch_ob_shape)
        return obs, actions, ext_rewards, mus, dones, masks, int_rewards, goal_obs