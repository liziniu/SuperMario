from queue import PriorityQueue
import tensorflow as tf
from gym import spaces
from common.util import fc
from curiosity.auxilliary_tasks import RandomFeature, InverseDynamics, RandomNetworkDistillation
import numpy as np
import time
from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U


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
    def __init__(self, sess, env, auxiliary_task, queue_size, feat_dim):
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

    def extract_feature(self, obs):
        raise NotImplementedError("You should not call this function!")
        # assert list(obs.shape)[1:] == self.auxiliary_task.obs.get_shape().as_list()[1:], "obs's shape:{} is wrong".format(obs.shape)
        # return self.sess.run(self.feat, feed_dict={self.auxiliary_task.obs: obs})

    def put_goal(self, obs, actions, next_obs, goal_infos):
        assert list(obs.shape)[1:] == self.obs.get_shape().as_list()[1:], "obs shape:{}.please flatten obs".format(obs.shape)
        assert list(actions.shape)[1:] == self.ac.get_shape().as_list()[1:], "action shape:{}.please flatten actions".format(actions.shape)
        assert list(next_obs.shape)[1:] == self.next_obs.get_shape().as_list()[1:], "next obs shape:{}.please flatten obs".format(next_obs.shape)
        assert len(goal_infos.shape) == 1, "info shape:{}".format(goal_infos.shape)

        # if aux_task is not RF, there may should have normalize schedule to ensure proper scale.
        # self.novelty_rms.update(priority)
        novelty = self.sess.run(self.novelty, feed_dict={self.obs: obs, self.next_obs: next_obs, self.ac: actions})
        self.novelty_rms.update(novelty)
        priority = - self.sess.run(self.novelty_normalized, feed_dict={self.novelty_tf: novelty})
        stats = self._add_goal(obs, actions, next_obs, goal_infos, priority)
        return stats

    def get_goal(self, nb_goal, replace=True):
        assert self.queue.qsize() >= nb_goal
        goal_feat, goal_obs, goal_act, goal_next_obs, goal_info = [], [], [], [], []
        for i in range(nb_goal):
            data = self.queue.get()
            goal_obs.append(data[2])
            goal_act.append(data[3])
            goal_next_obs.append(data[4])
            goal_info.append(data[5])
        goal_obs = np.asarray(goal_obs)
        if replace:
            goal_act = np.asarray(goal_act)
            goal_next_obs = np.asarray(goal_next_obs)
            novelty = self.sess.run(self.novelty, feed_dict={self.obs: goal_obs, self.ac: goal_act,
                                                             self.next_obs: goal_next_obs})
            self.novelty_rms.update(novelty)
            priority = - self.sess.run(self.novelty_normalized, feed_dict={self.novelty_tf: novelty})
            self._add_goal(goal_obs, goal_act, goal_next_obs, goal_info, priority)
        assert list(goal_obs.shape)[1:] == self.obs.get_shape().as_list()[1:], "goal_obs:{}".format(goal_obs.shape)
        # goal_feat = self.sess.run(self.feat, feed_dict={self.obs: goal_obs})
        return goal_obs, goal_info

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
                    baseline = -0.85 * stats["queue_max"]
                if priority[i] < baseline:
                    data = (priority[i], time.time(), obs[i], actions[i], next_obs[i], infos[i])
                    if self.queue.full():
                        maxvalue_idx = np.argmax([item[0] for item in self.queue.queue])
                        self.queue.queue.pop(maxvalue_idx)
                    self.queue.put(data)
        return stats

if __name__ == "__main__":
    # from common.env_util import make_atari
    # env = make_atari("SuperMarioBros-v0")
    # sess = tf.Session()
    # auxiliary_task = "RF"
    # queue_size = 5000
    # feat_dim = 512
    # nb_goal = 5
    # if auxiliary_task is None:
    #     dynamics = DummyDynamics()
    # else:
    #     dynamics = Dynamics(sess, env, auxiliary_task, queue_size, feat_dim)
    # for var in dynamics.params:
    #     print(var)
    # print(dynamics.get_goal(nb_goal))
    import baselines.common.tf_util as U
    @U.in_session
    def test():
        rms = RunningMeanStd(epsilon=1e-4)
        U.initialize()
        p = np.random.poisson(4, [100])
        rms.update(p)
        print(U.get_session().run(p - rms.mean) / rms.std)

    test()