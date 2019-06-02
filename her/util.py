import time
import numpy as np
from baselines import logger
from baselines.common.tf_util import save_variables
import functools


class Acer:
    def __init__(self, train_runner, evaluate_runner, model, buffer, log_interval):
        self.train_runner = train_runner
        self.evaluate_runner = evaluate_runner
        self.model = model
        self.buffer = buffer
        self.log_interval = log_interval
        self.tstart = None
        self.steps = 0

        sess = self.model.sess
        self.save = functools.partial(save_variables, sess=sess, variables=self.model.params)
        self.eval_interval = self.log_interval * 10
        self.evaluate_logs = [('test/final_x_pos', 0.),
                              ('test/final_y_pos', 0.),
                              ('test/success', 0.),
                              ('test/episode_length', 0.),
                              ('time/evaluate', 0.)]

    def call(self, replay_start, nb_train_epoch):
        runner, model, buffer, steps = self.train_runner, self.model, self.buffer, self.steps
        logs = []

        tstart = time.time()
        results = runner.run(acer_steps=self.steps)
        logs.extend(runner.logs())
        logs.append(('time/runner', time.time() - tstart))

        if buffer is not None:
            tstart = time.time()
            buffer.put(results)
            logs.append(('time/put', time.time() - tstart))

        policy_train_inputs = self.adjust_shape(results)
        tstart = time.time()
        names_ops, values_ops = model.train_policy(**policy_train_inputs)
        logs.extend(list(zip(names_ops, values_ops)))
        logs.append(('time/train', time.time() - tstart))

        if buffer.has_atleast(replay_start):
            for i in range(nb_train_epoch):
                if i == 0:
                    tstart = time.time()
                    results = buffer.get(use_cache=False)
                    logs.append(('time/get_first', time.time() - tstart))
                else:
                    tstart = time.time()
                    results = buffer.get(use_cache=True)
                    logs.append(('time/get_second', time.time() - tstart))
                policy_train_inputs = self.adjust_shape(results)
                names_ops, values_ops = model.train_policy(**policy_train_inputs)
                rewards = policy_train_inputs["rewards"]

                logs.extend(list(zip(names_ops, values_ops)))
                logs.append(('replay/rewards', np.mean(rewards)))
                logs.append(('replay/her_gain', results['her_gain']))
                logs.append(('replay/ag_mean', np.mean(results['achieved_goal'])))
                logs.append(('replay/g_mean', np.mean(results['desired_goal'])))
        else:
            logs.append(('time/get_first', 0.))
            logs.append(('time/get_second', 0.))
            logs.append(('replay/rewards', 0.))
            logs.append(('replay/her_gain', 0.))
            logs.append(('replay/ag_mean', 0.))
            logs.append(('replay/g_mean', 0.))
        if int(steps/runner.nbatch) % self.log_interval == 0:
            logs.append(('memory(GB)', self.buffer.memory_usage))
            if int(steps/runner.nbatch) % self.eval_interval == 0:
                self.evaluate_logs = self.evaluate_runner.evaluate()
            logs.extend(self.evaluate_logs)
            logger.record_tabular("time/total_timesteps", self.steps)
            logger.record_tabular("time/fps", int(self.steps / (time.time() - self.tstart)))
            for key, value in logs:
                logger.record_tabular(key, float(value))
            logger.dump_tabular()

    def adjust_shape(self, results):
        runner = self.train_runner

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
