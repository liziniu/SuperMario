import functools
import tensorflow as tf
from baselines import logger
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.a2c.utils import batch_to_seq, seq_to_batch
from baselines.a2c.utils import cat_entropy_softmax
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.utils import get_by_index, check_shape, avg_norm, q_explained_variance
from baselines.her.normalizer import Normalizer
from common.util import gradient_add
from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np
import gym.spaces
from her.defaults import ACHIEVED_GOAL_SHAPE, DESIRED_GOAL_SHAPE


# remove last step
def strip(var, nenvs, nsteps, flat=False):
    vars = batch_to_seq(var, nenvs, nsteps, flat)
    return seq_to_batch(vars, flat)


def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
    """
    Calculates q_retrace targets;
    vs: nenv, nsteps (takes obs_{t+1} and g_t as inputs)

    :param R: Rewards
    :param D: Dones
    :param q_i: Q values for actions taken
    :param v: V values
    :param rho_i: Importance weight for each action
    :return: Q_retrace values
    """
    rho_bar = batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    rs = batch_to_seq(R, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    ds = batch_to_seq(D, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    q_is = batch_to_seq(q_i, nenvs, nsteps, True)
    vs = batch_to_seq(v, nenvs, nsteps, True)   # (by lizn, only the next state value)
    v_final = vs[-1]
    qret = v_final
    qrets = []
    for i in range(nsteps - 1, -1, -1):
        check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
        qret = rs[i] + gamma * qret * (1.0 - ds[i])
        qrets.append(qret)
        if i > 0:
            qret = (rho_bar[i] * (qret - q_is[i])) + vs[i-1]
    qrets = qrets[::-1]
    qret = seq_to_batch(qrets, flat=True)
    return qret


# For ACER with PPO clipping instead of trust region
# def clip(ratio, eps_clip):
#     # assume 0 <= eps_clip <= 1
#     return tf.minimum(1 + eps_clip, tf.maximum(1 - eps_clip, ratio))


class Model(object):
    def __init__(self, sess, policy, ob_space, ac_space, nenvs, nsteps, ent_coef, q_coef, gamma,
                 max_grad_norm, lr, rprop_alpha, rprop_epsilon, total_timesteps, lrschedule, c, trust_region,
                 alpha, delta, scope, load_path, debug, policy_inputs):
        self.sess = sess
        self.nenv = nenvs
        self.policy_inputs = policy_inputs.copy()

        nact = ac_space.n
        nbatch = nenvs * nsteps
        eps = 1e-6

        self.scope = scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.A = tf.placeholder(tf.int32, [nbatch], name="action")  # actions
            self.D = tf.placeholder(tf.float32, [nbatch], name="dones")  # dones
            self.R = tf.placeholder(tf.float32, [nbatch], name="rewards")  # rewards, not returns
            self.MU = tf.placeholder(tf.float32, [nbatch, nact], name="mus")  # mu's
            self.LR = tf.placeholder(tf.float32, [], name="lr")

            self.V_NEXT = tf.placeholder(tf.float32, [nbatch], name="value_next")  # (by lzn: we revise goal-conditioned next value)

            if isinstance(ob_space, gym.spaces.Dict):
                self.obs_shape = ob_space.spaces['observation'].shape
                self.obs_dtype = ob_space.spaces['observation'].dtype
            else:
                self.obs_shape = ob_space.shape
                self.obs_dtype = ob_space.dtype
            self.achieved_goal_sh = achieved_goal_sh = ACHIEVED_GOAL_SHAPE
            self.desired_goal_sh = desired_goal_sh = DESIRED_GOAL_SHAPE
            self.desired_goal_state_sh = desired_goal_state_sh = self.obs_shape

            self.step_obs_tf = tf.placeholder(self.obs_dtype, (nenvs,) + self.obs_shape, 'step_obs')
            self.step_achieved_goal_tf = tf.placeholder(tf.float32, (nenvs,) + achieved_goal_sh, 'step_achieved_goal')
            self.step_desired_goal_tf = tf.placeholder(tf.float32, (nenvs, ) + desired_goal_sh, 'step_desired_goal')
            self.step_desired_goal_state_tf = tf.placeholder(self.obs_dtype, (nenvs,) + desired_goal_state_sh, 'step_desired_goal_state')

            self.train_obs_tf = tf.placeholder(self.obs_dtype, (nenvs * nsteps,) + self.obs_shape, 'train_obs')
            self.train_achieved_goal_tf = tf.placeholder(tf.float32, (nenvs * nsteps,) + achieved_goal_sh, 'train_achieved_goal')
            self.train_desired_goal_tf = tf.placeholder(tf.float32, (nenvs * nsteps,) + desired_goal_sh, 'train_desired_goal')
            self.train_desired_goal_state_tf = tf.placeholder(self.obs_dtype, (nenvs * nsteps,) + desired_goal_state_sh, 'train_desired_goal_state')

            # normalize embedding
            normalizer = 2500
            step_achieved_goal_tf = self.step_achieved_goal_tf / normalizer
            step_desired_goal_tf = self.step_desired_goal_tf / normalizer
            train_achieved_goal_tf = self.train_achieved_goal_tf / normalizer
            train_desired_goal_tf = self.train_desired_goal_tf / normalizer

            step_obs_tf = self.step_obs_tf
            step_desired_goal_state_tf = self.step_desired_goal_state_tf
            train_obs_tf = self.train_obs_tf
            train_desired_goal_state_tf = self.train_desired_goal_state_tf

            assert 'obs' in policy_inputs
            logger.info('policy_inputs:{}'.format(policy_inputs))
            logger.info('achieved_goal_sh:{}'.format(self.achieved_goal_sh))
            logger.info('desired_goal_sh:{}'.format(self.desired_goal_sh))
            logger.info('normalizer:{}'.format(normalizer))
            policy_inputs.remove('obs')
            if 'desired_goal_state' in policy_inputs:
                policy_inputs.remove('desired_goal_state')
                step_state_tf = tf.concat([step_obs_tf, step_desired_goal_state_tf], axis=-1, name='step_state')
                train_state_tf = tf.concat([train_obs_tf, train_desired_goal_state_tf], axis=-1, name='train_state')
            else:
                step_state_tf = step_obs_tf
                train_state_tf = train_obs_tf

            if 'achieved_goal' in policy_inputs and 'desired_goal' not in policy_inputs:
                policy_inputs.remove('achieved_goal')
                step_goal_tf = step_achieved_goal_tf
                train_goal_tf = train_achieved_goal_tf
            elif 'achieved_goal' not in policy_inputs and 'desired_goal' in policy_inputs:
                policy_inputs.remove('desired_goal')
                step_goal_tf = step_desired_goal_tf
                train_goal_tf = train_desired_goal_tf
            elif 'achieved_goal' in policy_inputs and 'desired_goal' in policy_inputs:
                policy_inputs.remove('achieved_goal')
                policy_inputs.remove('desired_goal')
                step_goal_tf = tf.concat([step_achieved_goal_tf, step_desired_goal_tf], axis=-1, name='step_goal')
                train_goal_tf = tf.concat([train_achieved_goal_tf, train_desired_goal_tf], axis=-1, name='train_goal')
            else:
                step_goal_tf, train_goal_tf = None, None
            if len(policy_inputs) > 0:
                raise ValueError("Unused policy inputs:{}".format(policy_inputs))

            self.step_model = policy(nbatch=nenvs, nsteps=1, state_placeholder=step_state_tf, sess=self.sess,
                                     goal_placeholder=step_goal_tf)
            self.train_model = policy(nbatch=nbatch, nsteps=nsteps, state_placeholder=train_state_tf,
                                      sess=self.sess, goal_placeholder=train_goal_tf, summary_stats=True)

        variables = find_trainable_variables
        self.params = params = variables(scope)
        logger.info("========================== {} =============================".format(scope))
        for var in params:
            logger.info(var)
        logger.info("========================== {} =============================\n".format(scope))

        # create polyak averaged model
        ema = tf.train.ExponentialMovingAverage(alpha)
        ema_apply_op = ema.apply(params)

        # print("========================== Ema =============================")

        def custom_getter(getter, *args, **kwargs):
            v = ema.average(getter(*args, **kwargs))
            # print(v.name)
            return v

        # print("========================== Ema =============================")

        with tf.variable_scope(scope, custom_getter=custom_getter, reuse=True):
            self.polyak_model = policy(nbatch=nbatch, nsteps=nsteps, state_placeholder=train_state_tf,
                                       goal_placeholder=train_goal_tf, sess=self.sess,)

        # Notation: (var) = batch variable, (var)s = seqeuence variable, (var)_i = variable index by action at step i

        # action probability distributions according to self.train_model, self.polyak_model and self.step_model
        # poilcy.pi is probability distribution parameters; to obtain distribution that sums to 1 need to take softmax
        train_model_p = tf.nn.softmax(self.train_model.pi)
        polyak_model_p = tf.nn.softmax(self.polyak_model.pi)
        self.step_model_p = tf.nn.softmax(self.step_model.pi)
        # (todo by lizn, use this to calculate next value)
        v = self.v = tf.reduce_sum(train_model_p * self.train_model.q, axis=-1)  # shape is [nenvs * (nsteps)]

        # strip off last step
        # (todo by lizn, we don't need strip)
        f, f_pol, q = map(lambda var: strip(var, nenvs, nsteps), [train_model_p, polyak_model_p, self.train_model.q])
        # f, f_pol, q = map(lambda x: x, [train_model_p, polyak_model_p, self.train_model.q])
        # Get pi and q values for actions taken
        f_i = get_by_index(f, self.A)
        q_i = get_by_index(q, self.A)

        # Compute ratios for importance truncation
        rho = f / (self.MU + eps)
        rho_i = get_by_index(rho, self.A)

        # Calculate Q_retrace targets
        qret = q_retrace(self.R, self.D, q_i, self.V_NEXT, rho_i, nenvs, nsteps, gamma)  # (todo by lizn, use new next state value)

        # Calculate losses
        # Entropy
        # entropy = tf.reduce_mean(strip(self.train_model.pd.entropy(), nenvs, nsteps))
        entropy = tf.reduce_mean(cat_entropy_softmax(f))

        # Policy Graident loss, with truncated importance sampling & bias correction
        v = strip(v, nenvs, nsteps, True)  # (todo by lzn: we do not need the strip the last one)
        check_shape([qret, v, rho_i, f_i], [[nenvs * nsteps]] * 4)
        check_shape([rho, f, q], [[nenvs * nsteps, nact]] * 2)

        # Truncated importance sampling
        adv = qret - v
        logf = tf.log(f_i + eps)
        gain_f = logf * tf.stop_gradient(adv * tf.minimum(c, rho_i))  # [nenvs * nsteps]
        loss_f = -tf.reduce_mean(gain_f)

        # Bias correction for the truncation
        adv_bc = (q - tf.reshape(v, [nenvs * nsteps, 1]))  # [nenvs * nsteps, nact]
        logf_bc = tf.log(f + eps)  # / (f_old + eps)
        check_shape([adv_bc, logf_bc], [[nenvs * nsteps, nact]] * 2)
        gain_bc = tf.reduce_sum(logf_bc * tf.stop_gradient(adv_bc * tf.nn.relu(1.0 - (c / (rho + eps))) * f),
                                axis=1)  # IMP: This is sum, as expectation wrt f
        loss_bc = -tf.reduce_mean(gain_bc)

        loss_policy = loss_f + loss_bc

        # Value/Q function loss, and explained variance
        check_shape([qret, q_i], [[nenvs * nsteps]] * 2)
        ev = q_explained_variance(tf.reshape(q_i, [nenvs, nsteps]), tf.reshape(qret, [nenvs, nsteps]))
        loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i) * 0.5)

        # Net loss
        check_shape([loss_policy, loss_q, entropy], [[]] * 3)

        # Goal loss
        loss = loss_policy + q_coef * loss_q - ent_coef * entropy

        if trust_region:
            g = tf.gradients(- (loss_policy - ent_coef * entropy) * nsteps * nenvs, f)  # [nenvs * nsteps, nact]
            # k = tf.gradients(KL(f_pol || f), f)
            k = - f_pol / (f + eps)  # [nenvs * nsteps, nact] # Directly computed gradient of KL divergence wrt f
            k_dot_g = tf.reduce_sum(k * g, axis=-1)
            adj = tf.maximum(0.0, (tf.reduce_sum(k * g, axis=-1) - delta) /
                             (tf.reduce_sum(tf.square(k), axis=-1) + eps))  # [nenvs * nsteps]

            # Calculate stats (before doing adjustment) for logging.
            avg_norm_k = avg_norm(k)
            avg_norm_g = avg_norm(g)
            avg_norm_k_dot_g = tf.reduce_mean(tf.abs(k_dot_g))
            avg_norm_adj = tf.reduce_mean(tf.abs(adj))

            g = g - tf.reshape(adj, [nenvs * nsteps, 1]) * k
            grads_f = -g / (
                nenvs * nsteps)  # These are turst region adjusted gradients wrt f ie statistics of policy pi
            grads_policy = tf.gradients(f, params, grads_f)
            grads_q = tf.gradients(loss_q * q_coef, params)
            # print("=========================== gards add ==============================")
            grads = [gradient_add(g1, g2, param) for (g1, g2, param) in zip(grads_policy, grads_q, params)]
            # print("=========================== gards add ==============================\n")
            avg_norm_grads_f = avg_norm(grads_f) * (nsteps * nenvs)
            norm_grads_q = tf.global_norm(grads_q)
            norm_grads_policy = tf.global_norm(grads_policy)
        else:
            grads = tf.gradients(loss, params)

        if max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=self.LR, decay=rprop_alpha, epsilon=rprop_epsilon)
        _policy_opt_op = trainer.apply_gradients(grads)

        # so when you call _train, you first do the gradient step, then you apply ema
        with tf.control_dependencies([_policy_opt_op]):
            _train_policy = tf.group(ema_apply_op)

        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        # Ops/Summaries to run, and their names for logging
        self.run_ops_policy = [_train_policy, loss, loss_q, entropy, loss_policy, loss_f, loss_bc, ev, norm_grads]
        self.names_ops_policy = ['loss', 'loss_q', 'entropy', 'loss_policy', 'loss_f', 'loss_bc', 'explained_variance',
                                 'norm_grads']
        if trust_region:
            self.run_ops_policy = self.run_ops_policy + [
                norm_grads_q, norm_grads_policy, avg_norm_grads_f, avg_norm_k, avg_norm_g, avg_norm_k_dot_g,
                avg_norm_adj]
            self.names_ops_policy = self.names_ops_policy + [
                'norm_grads_q', 'norm_grads_policy', 'avg_norm_grads_f', 'avg_norm_k', 'avg_norm_g', 'avg_norm_k_dot_g',
                'avg_norm_adj']
        self.names_ops_policy = [scope + "_" + x for x in self.names_ops_policy]  # scope as prefix

        self.save = functools.partial(save_variables, sess=self.sess, variables=params)

        self.initial_state = self.step_model.initial_state
        with tf.variable_scope('stats'):
            with tf.variable_scope('achieved_goal'):
                self.ag_stats = Normalizer(size=self.achieved_goal_sh[0], sess=self.sess)
            with tf.variable_scope('desired_goal'):
                self.g_stats = Normalizer(size=self.desired_goal_sh[0], sess=self.sess)
        if debug:
            tf.global_variables_initializer().run(session=self.sess)
            load_variables(load_path, self.params, self.sess)
        else:
            tf.global_variables_initializer().run(session=self.sess)

    def train_policy(self, obs, next_obs, achieved_goal, next_achieved_goal, desired_goal, desired_goal_state,
                     actions, rewards, mus, dones, steps):
        verbose = False
        cur_lr = self.lr.value_steps(steps)
        # 1. calculate v_{t+1} using obs_{t+1} and g_t
        td_map = self._feed_train_policy_inputs(next_obs, next_achieved_goal, desired_goal, desired_goal_state)
        v_next = self.sess.run(self.v, feed_dict=td_map)
        # 2. use obs_t, goal_t, v_{t+1} to train policy
        td_map.update({self.train_obs_tf: obs, self.train_achieved_goal_tf: achieved_goal, self.A: actions,
                       self.R: rewards, self.D: dones, self.MU: mus, self.LR: cur_lr, self.V_NEXT: v_next})
        if verbose:
            names_ops_policy = self.names_ops_policy.copy()
            values_ops_policy = self.sess.run(self.run_ops_policy, td_map)[1:]  # strip off _train
        else:
            names_ops_policy = self.names_ops_policy.copy()[:8]  # not including trust region
            values_ops_policy = self.sess.run(self.run_ops_policy, td_map)[1:][:8]
        debug = False
        if debug:
            if np.random.uniform() < 0.25:
                if 'achieved_goal' in self.policy_inputs:
                    self.ag_stats.update(achieved_goal)
                    self.ag_stats.recompute_stats()
                    names_ops_policy += ['achieved_goal']
                    values_ops_policy += [np.mean(self.sess.run(self.ag_stats.mean))]
                if 'desired_goal' in self.policy_inputs:
                    self.g_stats.update(desired_goal)
                    self.g_stats.recompute_stats()
                    names_ops_policy += ['desired_goal']
                    values_ops_policy += [np.mean(self.sess.run(self.g_stats.mean))]

        return names_ops_policy, values_ops_policy

    def step(self, inputs):
        td_map = self._feed_step_policy_inputs(**inputs)
        return self.sess.run([self.step_model.action, self.step_model_p], feed_dict=td_map)

    def _feed_train_policy_inputs(self, obs, achieved_goal, desired_goal, desired_goal_state):
        td_map = dict()
        assert 'obs' in self.policy_inputs
        td_map[self.train_obs_tf] = obs
        if 'achieved_goal' in self.policy_inputs:
            td_map[self.train_achieved_goal_tf] = achieved_goal
        if 'desired_goal' in self.policy_inputs:
            td_map[self.train_desired_goal_tf] = desired_goal
        if 'desired_goal_state' in self.policy_inputs:
            td_map[self.train_desired_goal_state_tf] = desired_goal_state
        return td_map

    def _feed_step_policy_inputs(self, obs, achieved_goal=None, desired_goal=None, desired_goal_state=None):
        td_map = dict()
        assert 'obs' in self.policy_inputs
        td_map[self.step_obs_tf] = obs
        if 'achieved_goal' in self.policy_inputs:
            td_map[self.step_achieved_goal_tf] = achieved_goal
        if 'desired_goal' in self.policy_inputs:
            td_map[self.step_desired_goal_tf] = desired_goal
        if 'desired_goal_state' in self.policy_inputs:
            td_map[self.step_desired_goal_state_tf] = desired_goal_state
        return td_map
