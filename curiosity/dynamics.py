from queue import PriorityQueue
import tensorflow as tf
from gym import spaces
from common.util import fc
from curiosity.auxilliary_tasks import RandomFeature, InverseDynamics, RandomNetworkDistillation
import numpy as np


class DummyDynamics:
    def __init__(self):
        self.dummy = True
        self.feat_var = tf.zeros([])
        self.dyna_params = []
        self.aux_params = []
        self.params = self.dyna_params + self.aux_params

        self.aux_loss = tf.zeros([])
        self.dyna_loss = tf.zeros([])
        self.loss = tf.zeros([])

    def extract_feature(self, x):
        return x

    def put_goal(self, obs, info):
        pass

    def get_goal(self, nb_goal):
        # goal_feat, goal_obs, goal_info
        goal_feat = np.empty([nb_goal, ])
        goal_obs = np.empty([nb_goal, ])
        goal_info = {}
        return goal_feat, goal_obs, goal_info


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

        self.goal = self.feat = tf.stop_gradient(self.auxiliary_task.feature)
        self.feat_var = tf.reduce_mean(tf.nn.moments(self.feat, [0, 1])[1])
        self.out_feat = tf.stop_gradient(self.auxiliary_task.next_feature)

        with tf.variable_scope("dynamics"):
            if isinstance(self.auxiliary_task, RandomNetworkDistillation):
                self.dyna_loss = tf.zeros([])
            elif isinstance(self.auxiliary_task, InverseDynamics) or isinstance(self.auxiliary_task, RandomFeature):
                with tf.variable_scope("loss"):
                    self.novelty = self.get_novelty()
                    self.dyna_loss = tf.reduce_mean(self.novelty)
            else:
                raise NotImplementedError

        self.dyna_params = tf.trainable_variables("dynamics")
        self.aux_params = tf.trainable_variables(self.auxiliary_task.scope)
        self.params = self.dyna_params + self.aux_params

        self.aux_loss = self.auxiliary_task.loss
        self.loss = self.aux_loss + self.dyna_loss

        self.queue = PriorityQueue(queue_size)

    def get_novelty(self):
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

    def extract_feature(self, x):
        assert list(x.shape) == self.auxiliary_task.obs.get_shape.as_list()[1:]
        return self.sess.run(self.feat, feed_dict={self.auxiliary_task.obs: x})

    def put_goal(self, obs, info):
        if len(obs.shape) > len(self.obs.get_shape().as_list()):
            obs = np.copy(obs)
            obs = obs.reshape([-1, ] + self.obs.get_shape().as_list()[1:])
        assert list(obs.shape)[1:] == self.obs.get_shape().as_list()[1:]
        priority = self.sess.run(self.dyna_loss, feed_dict={self.obs: obs})
        for i in range(len(obs)):
            data = {"obs": obs[i], "info": info[i]}
            # if aux_task is not RF, there may should have normalize schedule to ensure proper scale.
            self.queue.put(data, priority[i])

    def get_goal(self, nb_goal):
        assert self.queue.qsize() >= nb_goal
        goal_feat, goal_obs, goal_info = [], [], []
        for i in range(nb_goal):
            data = self.queue.get()
            goal_obs.append(data["obs"])
            goal_info.append(data["info"])
        goal_obs = np.asarray(goal_obs)
        assert list(goal_obs.shape)[1:] == self.obs.get_shape().as_list()[1:]
        goal_feat = self.sess.run(self.feat, feed_dict={self.obs: goal_obs})
        return goal_feat, goal_obs, goal_info

if __name__ == "__main__":
    from common.cmd_util import make_atari
    env = make_atari("SuperMarioBros-v0")
    sess = tf.Session()
    auxiliary_task = "RF"
    queue_size = 5000
    feat_dim = 512
    nb_goal = 5
    if auxiliary_task is None:
        dynamics = DummyDynamics()
    else:
        dynamics = Dynamics(sess, env, auxiliary_task, queue_size, feat_dim)
    for var in dynamics.params:
        print(var)
    print(dynamics.get_goal(nb_goal))