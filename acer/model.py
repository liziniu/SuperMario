import functools
import tensorflow as tf
from baselines import logger
from baselines.common.tf_util import get_session, save_variables
from baselines.a2c.utils import batch_to_seq, seq_to_batch
from baselines.a2c.utils import cat_entropy_softmax
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.utils import get_by_index, check_shape, avg_norm, q_explained_variance
from common.util import gradient_add


# remove last step
def strip(var, nenvs, nsteps, flat=False):
    vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
    return seq_to_batch(vars[:-1], flat)


def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
    """
    Calculates q_retrace targets

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
    vs = batch_to_seq(v, nenvs, nsteps + 1, True)
    v_final = vs[-1]
    qret = v_final
    qrets = []
    for i in range(nsteps - 1, -1, -1):
        check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
        qret = rs[i] + gamma * qret * (1.0 - ds[i])
        qrets.append(qret)
        qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
    qrets = qrets[::-1]
    qret = seq_to_batch(qrets, flat=True)
    return qret


# For ACER with PPO clipping instead of trust region
# def clip(ratio, eps_clip):
#     # assume 0 <= eps_clip <= 1
#     return tf.minimum(1 + eps_clip, tf.maximum(1 - eps_clip, ratio))


class Model(object):
    def __init__(self, policy, dynamics, ob_space, ac_space, nenvs, nsteps, ent_coef, q_coef, gamma, max_grad_norm, lr,
                 rprop_alpha, rprop_epsilon, total_timesteps, lrschedule, c, trust_region, alpha, delta, scope):

        sess = get_session()
        nact = ac_space.n
        nbatch = nenvs * nsteps
        eps = 1e-6

        self.dynamics = dynamics
        goal_feat_shape = self.dynamics.feat_shape

        self.scope = scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            A = tf.placeholder(tf.int32, [nbatch], name="action")  # actions
            D = tf.placeholder(tf.float32, [nbatch], name="dones")  # dones
            R = tf.placeholder(tf.float32, [nbatch], name="rewards")  # rewards, not returns
            MU = tf.placeholder(tf.float32, [nbatch, nact], name="mus")  # mu's
            LR = tf.placeholder(tf.float32, [], name="lr")

            step_ob_placeholder = tf.placeholder(ob_space.dtype, (nenvs,) + ob_space.shape, "step_ob")
            if self.dynamics.dummy:
                step_goal_placeholder = None
            else:
                step_goal_placeholder = tf.placeholder(tf.float32, (nenvs,) + goal_feat_shape, "step_goal")

            train_ob_placeholder = tf.placeholder(ob_space.dtype, (nenvs * (nsteps + 1),) + ob_space.shape, "train_ob")
            if self.dynamics.dummy:
                train_goal_placeholder = None
            else:
                train_goal_placeholder = tf.placeholder(tf.float32, (nenvs * (nsteps + 1),) + goal_feat_shape, "train_goal")

            step_model = policy(nbatch=nenvs, nsteps=1, observ_placeholder=step_ob_placeholder,
                                goal_placeholder=step_goal_placeholder, sess=sess)
            train_model = policy(nbatch=nbatch, nsteps=nsteps, observ_placeholder=train_ob_placeholder,
                                 goal_placeholder=train_goal_placeholder, sess=sess)

        variables = find_trainable_variables
        params = variables(scope)
        logger.info("========================== {} =============================".format(scope))
        for var in params:
            logger.info(var)
        logger.info("========================== {} =============================\n".format(scope))

        logger.info("======================={}: Aux & Dyna =========================".format(scope))
        for var in self.dynamics.params:
            logger.info(var)
        logger.info("======================={}: Aux & Dyna =========================\n".format(scope))

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
            polyak_model = policy(nbatch=nbatch, nsteps=nsteps, observ_placeholder=train_ob_placeholder,
                                  goal_placeholder=train_goal_placeholder, sess=sess)

        # Notation: (var) = batch variable, (var)s = seqeuence variable, (var)_i = variable index by action at step i

        # action probability distributions according to train_model, polyak_model and step_model
        # poilcy.pi is probability distribution parameters; to obtain distribution that sums to 1 need to take softmax
        train_model_p = tf.nn.softmax(train_model.pi)
        polyak_model_p = tf.nn.softmax(polyak_model.pi)
        step_model_p = tf.nn.softmax(step_model.pi)
        v = tf.reduce_sum(train_model_p * train_model.q, axis=-1)  # shape is [nenvs * (nsteps + 1)]

        # strip off last step
        f, f_pol, q = map(lambda var: strip(var, nenvs, nsteps), [train_model_p, polyak_model_p, train_model.q])
        # Get pi and q values for actions taken
        f_i = get_by_index(f, A)
        q_i = get_by_index(q, A)

        # Compute ratios for importance truncation
        rho = f / (MU + eps)
        rho_i = get_by_index(rho, A)

        # Calculate Q_retrace targets
        qret = q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma)

        # Calculate losses
        # Entropy
        # entropy = tf.reduce_mean(strip(train_model.pd.entropy(), nenvs, nsteps))
        entropy = tf.reduce_mean(cat_entropy_softmax(f))

        # Policy Graident loss, with truncated importance sampling & bias correction
        v = strip(v, nenvs, nsteps, True)
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
            adj = tf.maximum(0.0, (tf.reduce_sum(k * g, axis=-1) - delta) / (
            tf.reduce_sum(tf.square(k), axis=-1) + eps))  # [nenvs * nsteps]

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
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=rprop_alpha, epsilon=rprop_epsilon)
        _policy_opt_op = trainer.apply_gradients(grads)
        if not self.dynamics.dummy:
            _train_dynamics = trainer.minimize(self.dynamics.loss)
        # so when you call _train, you first do the gradient step, then you apply ema
        with tf.control_dependencies([_policy_opt_op]):
            _train_policy = tf.group(ema_apply_op)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        # Ops/Summaries to run, and their names for logging
        run_ops_policy = [_train_policy, loss, loss_q, entropy, loss_policy, loss_f, loss_bc, ev, norm_grads]
        names_ops_policy = ['loss', 'loss_q', 'entropy', 'loss_policy', 'loss_f', 'loss_bc', 'explained_variance',
                            'norm_grads']
        if trust_region:
            run_ops_policy = run_ops_policy + [norm_grads_q, norm_grads_policy, avg_norm_grads_f, avg_norm_k,
                                               avg_norm_g, avg_norm_k_dot_g, avg_norm_adj]
            names_ops_policy = names_ops_policy + ['norm_grads_q', 'norm_grads_policy', 'avg_norm_grads_f',
                                                   'avg_norm_k', 'avg_norm_g',
                                                   'avg_norm_k_dot_g', 'avg_norm_adj']
        if not self.dynamics.dummy:
            run_ops_dynamics = [_train_dynamics, self.dynamics.aux_loss, self.dynamics.dyna_loss,
                                self.dynamics.feat_var]
            names_ops_dynamics = ["aux_loss", "dyna_loss", "goal_feat_var"]
        names_ops_policy = [scope+"_"+x for x in names_ops_policy]  # scope as prefix

        def train_policy(obs, actions, rewards, dones, mus, states, masks, steps, goal_feats):
            cur_lr = lr.value_steps(steps)
            td_map = {train_model.X: obs, polyak_model.X: obs, A: actions, R: rewards, D: dones, MU: mus, LR: cur_lr}
            if not self.dynamics.dummy:
                assert hasattr(train_model, "goals")
                td_map[train_model.goals] = goal_feats
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
                td_map[polyak_model.S] = states
                td_map[polyak_model.M] = masks
            return names_ops_policy, sess.run(run_ops_policy, td_map)[1:]  # strip off _train

        def train_dynamics(obs, actions, steps):
            cur_lr = lr.value_steps(steps)
            td_map = {self.dynamics.obs: obs[:-1], self.dynamics.next_obs: obs[1:], self.dynamics.ac: actions, LR: cur_lr}
            return names_ops_dynamics, sess.run(run_ops_dynamics, td_map)[1:]

        def _step(observation, **kwargs):
            return step_model._evaluate([step_model.action, step_model_p, step_model.state], observation, **kwargs)

        self.train_policy = train_policy
        self.train_dynamics = train_dynamics  # todo:
        self.save = functools.partial(save_variables, sess=sess, variables=params)
        self.train_model = train_model
        self.step_model = step_model
        self._step = _step
        self.step = self.step_model.step

        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)