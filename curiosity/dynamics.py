import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from queue import PriorityQueue
import tensorflow as tf
from gym import spaces
from common.util import fc, DataRecorder
from curiosity.auxilliary_tasks import RandomFeature, InverseDynamics, RandomNetworkDistillation
import numpy as np
import time
from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
from baselines import logger
import os
from scipy.stats import pearsonr
import seaborn as sns
from collections import deque
plt.style.use("seaborn")


def get_dist(infos):
    def f(x):
        return float(x["x_pos"])+float(x["y_pos"])

    vf = np.vectorize(f)
    dist = np.clip(vf(infos), 0, 2500)
    return dist


class DummyDynamics:
    def __init__(self, goal_shape):
        self.dummy = True
        self.feat_var = tf.zeros([])
        self.dyna_params = []
        self.aux_params = []
        self.params = self.dyna_params + self.aux_params
        self.feat_shape = []
        self.aux_loss = tf.zeros([])
        self.dyna_loss = tf.zeros([])
        self.loss = tf.zeros([])
        self.goal_shape = goal_shape

    def extract_feature(self, x):
        return x

    def put_goal(self, obs, info):
        pass

    def get_goal(self, nb_goal):
        # goal_obs, goal_info
        goal_obs = np.empty((nb_goal, ) + self.goal_shape)
        goal_obs.fill(np.nan)
        goal_info = np.array([{"x_pos": 0.0, "y_pos": 0.0} for _ in range(nb_goal)], dtype=object)
        return goal_obs, goal_info


class Dynamics:
    def __init__(self, sess, env, auxiliary_task, queue_size, feat_dim, normalize_novelty):
        self.sess = sess
        self.dummy = False
        self.make_auxiliary_task = {"RF": RandomFeature,
                                    "IDF": InverseDynamics,
                                    "RND": RandomNetworkDistillation}[auxiliary_task.upper()]
        self.auxiliary_task = self.make_auxiliary_task(env, feat_dim)
        self.obs = self.auxiliary_task.obs
        self.next_obs = self.auxiliary_task.next_obs
        self.ac = self.auxiliary_task.ac
        self.ac_space = self.auxiliary_task.ac_space

        self.feat = tf.stop_gradient(self.auxiliary_task.feature)
        self.feat_shape = tuple(self.feat.get_shape().as_list()[1:])
        self.feat_var = tf.reduce_mean(tf.nn.moments(self.feat, axes=-1)[1])
        self.out_feat = tf.stop_gradient(self.auxiliary_task.next_feature)
        self.nenv = env.num_envs if hasattr(env, 'num_envs') else 1

        with tf.variable_scope("dynamics"):
            self.novelty_tf = tf.placeholder(tf.float32, [None], "novelty_placeholder")
            if isinstance(self.auxiliary_task, RandomNetworkDistillation):
                self.dyna_loss = tf.zeros([])
                self.novelty = self.auxiliary_task.get_novelty()
            elif isinstance(self.auxiliary_task, InverseDynamics) or isinstance(self.auxiliary_task, RandomFeature):
                with tf.variable_scope("loss"):
                    self.novelty = self._get_novelty()
                    self.dyna_loss = tf.reduce_mean(self.novelty)
            else:
                raise NotImplementedError

        self.dyna_params = tf.trainable_variables("dynamics")
        self.aux_params = tf.trainable_variables(self.auxiliary_task.scope)
        self.params = self.dyna_params + self.aux_params

        self.aux_loss = self.auxiliary_task.loss
        self.loss = self.aux_loss + self.dyna_loss

        self.queue = PriorityQueue(queue_size)
        self.novelty_rms = RunningMeanStd(epsilon=1e-4)
        self.novelty_normalized = tf.clip_by_value((self.novelty_tf-self.novelty_rms.mean)/self.novelty_rms.std,
                                                   -5., 5.)

        self.normalized = normalize_novelty
        if normalize_novelty:
            logger.info("normalize novelty")
        path = logger.get_dir()
        path = os.path.join(path, "goal_data")
        self.goal_recoder = DataRecorder(path)
        self.goal_store_baseline = 1500

        self.density_estimate = deque(maxlen=int(1e4))

        self.eval_interval = 20
        self.eval_data_status = {}
        self.eval_data = []
        path = logger.get_dir()
        self.eval_path = path = os.path.join(path, "novelty_evaluation")
        self.eval_recoder = DataRecorder(path)

        path = logger.get_dir()
        path = os.path.join(path, "error_goal")
        self.error_recoder = DataRecorder(path)

    def _get_novelty(self):
        if isinstance(self.ac_space, spaces.Box):
            assert len(self.ac_space.shape) == 1
        elif isinstance(self.ac_space, spaces.Discrete):
            ac = tf.one_hot(self.ac, self.ac_space.n)
        elif isinstance(self.ac_space, spaces.MultiDiscrete):
            raise NotImplementedError
        elif isinstance(self.ac_space, spaces.MultiBinary):
            ac = tf.one_hot(self.ac, self.ac_space.n)
        else:
            raise NotImplementedError

        def add_ac(x):
            return tf.concat([x, ac], axis=-1)

        hidsize = 512
        activ = tf.nn.leaky_relu
        x = fc(add_ac(self.feat), nh=hidsize, scope="fc_1")
        if activ is not None:
            x = activ(x)

        def residual(x, scope):
            res = fc(add_ac(x), nh=hidsize, scope=scope+"_1")
            res = tf.nn.leaky_relu(res)
            res = fc(add_ac(res), nh=hidsize, scope=scope+"_2")
            return x + res

        for _ in range(4):
            x = residual(x, scope="residual_{}".format(_ + 1))
        n_out_features = self.out_feat.get_shape()[-1].value
        x = fc(add_ac(x), nh=n_out_features, scope="output")
        return tf.reduce_mean(tf.square(x - self.out_feat), axis=-1)

    def put_goal(self, obs, actions, next_obs, goal_infos):
        assert list(obs.shape)[1:] == self.obs.get_shape().as_list()[1:], "obs shape:{}.please flatten obs".format(obs.shape)
        assert list(actions.shape)[1:] == self.ac.get_shape().as_list()[1:], "action shape:{}.please flatten actions".format(actions.shape)
        assert list(next_obs.shape)[1:] == self.next_obs.get_shape().as_list()[1:], "next obs shape:{}.please flatten obs".format(next_obs.shape)
        assert len(goal_infos.shape) == 1, "info shape:{}".format(goal_infos.shape)

        # sample goal according to x_pos
        x_pos = [info["x_pos"] for info in goal_infos]
        for index, x in enumerate(x_pos):
            seg = x // self.eval_interval * self.eval_interval
            if seg not in self.eval_data_status:
                self.eval_data_status[seg] = False
            self.density_estimate.append(x)
            if not self.eval_data_status[seg]:
                self.eval_data.append({"obs": obs[index], "actions": actions[index],
                                       "next_obs": next_obs[index], "info": goal_infos[index]})
                self.eval_data_status[seg] = True
                self.eval_data = sorted(self.eval_data, key=lambda y: y["info"]["x_pos"])
        if np.max(x_pos) > self.goal_store_baseline:
            self.goal_recoder.store(self.eval_data)
            self.goal_recoder.dump()
            self.goal_store_baseline += 1000
            logger.info("store {} goal.now baseline:{}".format(len(self.eval_data), self.goal_store_baseline))
        # store goal into queue according to priority.
        novelty = self.sess.run(self.novelty, feed_dict={self.obs: obs, self.next_obs: next_obs, self.ac: actions})
        if self.normalized:
            self.novelty_rms.update(novelty)
            priority = - self.sess.run(self.novelty_normalized, feed_dict={self.novelty_tf: novelty})
        else:
            priority = - novelty
        stats = self._add_goal(obs, actions, next_obs, goal_infos, priority)
        return stats

    def get_goal(self, nb_goal, replace=True, alpha=1.0, beta=0.95):
        assert self.queue.qsize() >= nb_goal
        goal_priority, goal_feat, goal_obs, goal_act, goal_next_obs, goal_info = [], [], [], [], [], []
        while len(goal_obs) != nb_goal:
            data = self.queue.get()
            if (data[5]["x_pos"] <= 55) and (data[5]["y_pos"] <= 180):
                self.error_recoder.store(data)
                self.error_recoder.dump()
                logger.info("detecting an error goal:{} and remove it".format(data[5]))
                continue
            goal_priority.append(data[0])
            goal_obs.append(data[2])
            goal_act.append(data[3])
            goal_next_obs.append(data[4])
            goal_info.append(data[5])
        goal_priority = np.asarray(goal_priority)
        # IMPORTANT: goal is next_obs in tuple.
        goals = np.asarray(goal_next_obs)
        if replace:
            goal_act = np.asarray(goal_act)
            goal_next_obs = np.asarray(goal_next_obs)
            novelty = self.sess.run(self.novelty, feed_dict={self.obs: goal_obs, self.ac: goal_act,
                                                             self.next_obs: goal_next_obs})
            if self.normalized:
                self.novelty_rms.update(novelty)
                priority = - self.sess.run(self.novelty_normalized, feed_dict={self.novelty_tf: novelty})
            else:
                priority = - novelty

            priority = (1-alpha) * priority + alpha * goal_priority
            priority *= beta
            self._add_goal(goal_obs, goal_act, goal_next_obs, goal_info, priority)
        assert list(goals.shape)[1:] == self.obs.get_shape().as_list()[1:], "goal_obs:{}".format(goals.shape)
        return goals, goal_info

    def _add_goal(self, obs, actions, next_obs, infos, priority):
        baseline = None
        stats = dict()
        for i in range(len(priority)):
            if self.queue.qsize() < self.nenv * 5:
                data = (priority[i], time.time(), obs[i], actions[i], next_obs[i], infos[i])
                self.queue.put(data)
            else:
                if baseline is None:
                    queue_p = [-item[0] for item in self.queue.queue]
                    stats["queue_max"], stats["queue_std"] = np.max(queue_p), np.std(queue_p)
                    baseline = -0.75 * stats["queue_max"]
                if priority[i] < baseline:
                    data = (priority[i], time.time(), obs[i], actions[i], next_obs[i], infos[i])
                    if self.queue.full():
                        maxvalue_idx = np.argmax([item[0] for item in self.queue.queue])
                        self.queue.queue.pop(maxvalue_idx)
                    self.queue.put(data)
        return stats

    def evaluate(self, steps, plot=False):
        if len(self.eval_data) > 0:
            obs, act, next_obs, x_pos = [], [], [], []
            for i in range(len(self.eval_data)):
                obs.append(self.eval_data[i]["obs"])
                act.append(self.eval_data[i]["actions"])
                next_obs.append(self.eval_data[i]["next_obs"])
                x_pos.append(self.eval_data[i]["info"]["x_pos"])
            obs = np.asarray(obs, dtype=np.float32)
            act = np.asarray(act, dtype=np.float32)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            x_pos = np.asarray(x_pos, dtype=np.float32)
            novelty = self.sess.run(self.novelty, feed_dict={self.obs: obs, self.ac: act, self.next_obs: next_obs})
            p = pearsonr(x_pos, novelty)[0]
            if plot:
                plt.figure(dpi=80)
                plt.subplot(2, 1, 1)
                plt.scatter(x_pos, novelty)
                plt.title("pos & novelty")
                plt.yscale("log")
                plt.subplot(2, 1, 2)
                density = np.array(self.density_estimate)
                sns.kdeplot(density)
                plt.title("sample density")
                plt.savefig(os.path.join(self.eval_path, "{}.png".format(steps)))
                plt.close()

            self.eval_recoder.store({"x_pos": x_pos, "novelty": novelty, "p": p, "steps": steps})
            self.eval_recoder.dump()
            
            return ["pos_novelty_p"], [p]
        else:
            return ["pos_novelty_p"], [np.nan]


@U.in_session
def test():
    rms = RunningMeanStd(epsilon=1e-4)
    U.initialize()
    p = np.random.poisson(4, [100])
    rms.update(p)
    print(U.get_session().run(p - rms.mean) / rms.std)


def test_queue_replace():
    from common.env_util import make_atari
    tf.set_random_seed(1)
    np.random.seed(1)

    env = make_atari("SuperMarioBros-v0")
    sess = tf.Session()
    auxiliary_task = "RF"
    queue_size = 50
    feat_dim = 512
    nb_goal = 5
    normalize_novelty = False

    dynamics = Dynamics(sess, env, auxiliary_task, queue_size, feat_dim, normalize_novelty)
    tf.global_variables_initializer().run(session=sess)

    def f():
        obs = np.random.randn(100, 84, 84, 4,)
        act = np.random.randint(0, 6, 100)
        next_obs = np.random.randn(100, 84, 84, 4)
        info = np.array([{"x_pos": np.random.randint(1000), "y_pos":np.random.randint(100, 200)} for _ in range(100)],
                        dtype=object)
        return obs, act, next_obs, info

    data = f()
    dynamics.put_goal(data[0], data[1], data[2], data[3])
    print([-x[0] for x in dynamics.queue.queue])
    goal_obs_, goal_info_ = None, None
    for i in range(1000):
        goal_obs, goal_info = dynamics.get_goal(nb_goal, debug=True)
        if goal_obs_ is None:
            goal_obs_ = goal_obs
            goal_info_ = goal_info
        assert np.sum(goal_obs - goal_obs_) < 1e-6
        goal_obs_ = goal_obs
        goal_info_ = goal_info

if __name__ == "__main__":
    test_queue_replace()