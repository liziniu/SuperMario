import tensorflow as tf
from baselines.common.distributions import make_pdtype
from util import cnn, fc


class FeatureExtractor(object):
    def __init__(self, sess, env, feat_dim=None, scope='feature_extractor'):
        self.scope = scope
        self.sess = sess
        self.feat_dim = feat_dim
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        with tf.variable_scope(scope):
            self.obs = tf.placeholder(tf.float32, [None, 84, 84, 4], name="obs")
            self.next_obs = tf.placeholder(tf.float32, [None, 84, 84, 4], name="next_obs")
            self.pdtype = make_pdtype(self.ac_space)
            self.ac = self.pdtype.sample_placeholder([None], name="ac")
            self.scope = scope

            self.feature = self.get_features(self.obs, reuse=False)
            self.next_feature = self.get_features(self.next_obs, reuse=True)
            self.loss = self.get_loss()

    def get_features(self, x, reuse):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError


class InverseDynamics(FeatureExtractor):
    def __init__(self, sess, env, feat_dim=None, scope="inverse_dynamics"):
        super(InverseDynamics, self).__init__(
            sess=sess,
            env=env,
            scope=scope,
            feat_dim=feat_dim,
        )

    def get_features(self, x, reuse):
        with tf.variable_scope(self.scope, reuse=reuse):
            feature = cnn(x, activ=tf.nn.leaky_relu, nfeat=self.feat_dim, scope="feature", reuse=reuse)
        return feature

    def get_loss(self):
        activ = tf.nn.leaky_relu
        hidsize= 64
        with tf.variable_scope(self.scope):
            x = tf.concat([self.feature, self.next_feature], -1)

            x = fc(x, "fc_1", nh=hidsize)
            x = activ(x)
            x = fc(x, "fc_2", nh=self.ac.get_shape().as_list()[-1])
            idfpd = self.pdtype.pdfromflat(x)
            return idfpd.neglogp(self.ac)


class RandomNetworkDistillation(FeatureExtractor):
    def __init__(self, sess, env, feat_dim=None, scope="random_network_distillation"):
        super(RandomNetworkDistillation, self).__init__(
            sess=sess,
            env=env,
            scope=scope,
            feat_dim=feat_dim,
        )

        with tf.variable_scope("target", reuse=False):
            self.feature_target = cnn(self.obs, activ=tf.nn.leaky_relu, nfeat=self.feat_dim, scope="target", reuse=False)

    def get_features(self, x, reuse):
        with tf.variable_scope("prediction", reuse=reuse):
            feature_predict = cnn(x, activ=tf.nn.leaky_relu, nfeat=self.feat_dim, scope="prediction", reuse=reuse)
        return feature_predict

    def get_loss(self):
        return tf.reduce_mean(tf.square(tf.stop_gradient(self.feature_target) - self.feature))


class RandomFeature(FeatureExtractor):
    def __init__(self, sess, env, feat_dim=None, scope="random_feature"):
        super(RandomFeature, self).__init__(
            sess=sess,
            env=env,
            scope=scope,
            feat_dim=feat_dim,
        )

    def get_features(self, x, reuse):
        with tf.variable_scope(self.scope, reuse=reuse):
            feature = cnn(x, scope=self.scope, activ=tf.nn.leaky_relu, nfeat=self.feat_dim, reuse=reuse)
        return feature

    def get_loss(self):
        return tf.zeros([])

