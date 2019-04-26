import tensorflow as tf
from baselines.a2c.utils import conv_to_fc, fc
import numpy as np
import os
import pickle


def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False, reuse=False):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        return tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format) + b


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


def cnn(unscaled_images, scope, activ=None, nfeat=None, reuse=False):
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = activ or tf.nn.leaky_relu
    nfeat = nfeat or 512
    h = activ(conv(scaled_images, scope+'_conv1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), reuse=reuse))
    h2 = activ(conv(h, scope+'_conv2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), reuse=reuse))
    h3 = activ(conv(h2, scope+'_conv3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), reuse=reuse))
    h3 = conv_to_fc(h3)
    return fc(h3, scope+'_conv_to_fc', nh=nfeat, init_scale=np.sqrt(2), reuse=reuse)


class DataRecorder:
    def __init__(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.file = open(os.path.join(path, "data.pkl"), "wb")
        self.file.close()
        self.memory = []

    def store(self, data):
        self.memory.append(data)

    def dump(self):
        with open(os.path.join(self.path, "data.pkl"), "ab+") as f:
            pickle.dump(self.memory, f, -1)
        self.memory = []