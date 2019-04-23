import tensorflow as tf
from gym import spaces
from util import fc
from auxilliary_tasks import RandomFeature, InverseDynamics, RandomNetworkDistillation


class Dynamics(object):
    def __init__(self, auxiliary_task, feat_dim=None, scope='dynamics'):
        self.scope = scope
        self.auxiliary_task = auxiliary_task

        self.sess = self.auxiliary_task.sess
        self.feat_dim = feat_dim
        self.obs = self.auxiliary_task.obs
        self.next_obs = self.auxiliary_task.next_obs
        self.ac = self.auxiliary_task.ac
        self.ac_space = self.auxiliary_task.ac_space

        self.features = tf.stop_gradient(self.auxiliary_task.features)

        self.out_features = tf.stop_gradient(self.auxiliary_task.next_features)

        if isinstance(self.auxiliary_task, RandomNetworkDistillation):
            self.loss = tf.zeros([])
            self.novelty = self.auxiliary_task.loss
        elif isinstance(self.auxiliary_task, InverseDynamics) or isinstance(self.auxiliary_task, RandomFeature):
            with tf.variable_scope(self.scope + "_loss"):
                self.loss = self.get_loss()
                self.novelty = self.loss
        else:
            raise NotImplementedError

    def get_loss(self):
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

        with tf.variable_scope(self.scope):
            hidsize = 64
            activ = tf.nn.leaky_relu
            x = fc(add_ac(self.features), nh=hidsize, scope="fc_1")
            if activ is not None:
                x = activ(fc)

            def residual(x, scope):
                res = fc(add_ac(x), nh=hidsize, scope=scope+"_1")
                res = tf.nn.leaky_relu(res)
                res = fc(add_ac(res), nh=hidsize, scope=scope+"_2")
                return x + res

            for _ in range(4):
                x = residual(x, scope="residual_{}".format(_ + 1))
            n_out_features = self.out_features.get_shape()[-1].value
            x = fc(add_ac(x), nh=n_out_features, scope="output")
        return tf.reduce_mean(tf.square(x - self.out_features), -1)

    def get_novelty(self, obs, next_obs=None):
        feed_dict = {self.obs: obs}
        if next_obs is not None:
            feed_dict[self.next_obs] = next_obs
        return self.sess.run(self.novelty, feed_dict=feed_dict)


