import time
import numpy as np
from baselines import logger
from baselines.common import set_global_seeds
from acer.policies import build_policy
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from acer.buffer import Buffer
from acer.runner import Runner
from common.her_sample import make_sample_her_transitions
from acer.model import Model
from common.env_util import parser_env_id, build_env, get_env_type
from common.util import EpisodeStats


class Acer:
    def __init__(self, runner, runner_eval, model, model_eval, buffer, log_interval, eval_interval):
        self.runner = runner
        self.model = model
        self.model_eval = model_eval
        self.buffer = buffer
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.tstart = None
        self.steps = None
        self.runner_eval = runner_eval
        self.nupdates = 0

        keys = []
        keys += ["expl_return", "expl_length", "expl_x_pos", "expl_y_pos"]
        keys += ["eval_return", "eval_length", ]
        keys += ["reached_cnt", "reached_time", ]
        self.episode_stats = EpisodeStats(maxlen=40, keys=keys)

    def call(self, on_policy):
        runner, model, model_eval, buffer, steps = self.runner, self.model, self.model_eval, self.buffer, self.steps
        if on_policy:
            enc_obs, obs, actions, ext_rewards, mus, dones, masks, goal_obs, goal_feats, int_rewards, mb_goal_infos, episode_infos = runner.run()
            if buffer is not None:
                buffer.put(enc_obs, actions, ext_rewards, mus, dones, masks, goal_obs)
            
            # flatten on-policy data (nenv, nstep, ...) for dynamics training and put_goal
            mb_next_obs = np.copy(obs[:, 1:])
            mb_obs = np.copy(obs[:, :-1])
            mb_obs = mb_obs.reshape((-1, )+obs.shape[2:])
            mb_next_obs = mb_next_obs.reshape((-1, )+mb_next_obs.shape[2:])
            mb_actions = actions.copy().reshape((-1, )+actions.shape[2:])
            mb_goal_infos = mb_goal_infos.copy().reshape(-1)
            if not self.model.dynamics.dummy:
                self.model.dynamics.put_goal(mb_obs, mb_actions, mb_next_obs, mb_goal_infos)
            names_ops_dynamics, values_ops_dynamics = model.train_dynamics(mb_obs, mb_actions, mb_next_obs, steps)

            # collect episode information
            for info in episode_infos:
                reached_info = info.get("reached_info")
                if reached_info:
                    self.episode_stats.feed(reached_info["reached"], "reached_cnt")
                    self.episode_stats.feed(reached_info["time_ratio"], "reached_time")
                goal_info = info.get("goal_info")
                if goal_info:
                    self.episode_stats.feed(goal_info["x_pos"], "expl_x_pos")
                    self.episode_stats.feed(goal_info["y_pos"], "expl_y_pos")
                return_info = info.get("episode")
                if return_info:
                    self.episode_stats.feed(return_info["l"], "expl_length")
                    self.episode_stats.feed(return_info["r"], "expl_return")
        else:
            obs, actions, ext_rewards, mus, dones, masks, goal_feats, int_rewards = buffer.get()     # todo: add her
        # reshape stuff correctly except goal_feats
        # we do not reshape goal_feats since we flatten it when extract feature is called
        obs = obs.reshape(runner.batch_ob_shape)
        actions = actions.reshape([runner.nbatch])
        ext_rewards = ext_rewards.reshape([runner.nbatch])
        mus = mus.reshape([runner.nbatch, runner.nact])
        dones = dones.reshape([runner.nbatch])
        masks = masks.reshape([runner.batch_ob_shape[0]])
        int_rewards = int_rewards.reshape([runner.nbatch])

        # training policy
        if model.scope != model_eval.scope:
            names_ops_policy, values_ops_policy = model.train_policy(
                obs, actions, int_rewards, dones, mus, model.initial_state, masks, steps, goal_feats)
            # Actually we do not feed goal_feats into evaluation policy!
            names_ops_policy_, values_ops_policy_ = model_eval.train_policy(
                obs, actions, ext_rewards, dones, mus, model_eval.initial_state, masks, steps, goal_feats)
            names_ops_policy += names_ops_policy_
            values_ops_policy += values_ops_policy_
        else:
            names_ops_policy, values_ops_policy = model.train_policy(
                obs, actions, ext_rewards, dones, mus, model.initial_state, masks, steps, goal_feats
            )
        self.nupdates += 1

        # logger and evaluation
        if on_policy:
            if int(steps/runner.nbatch) % self.log_interval == 0:
                logger.record_tabular("total_timesteps", steps)
                logger.record_tabular("fps", int(steps/(time.time() - self.tstart)))
                logger.record_tabular("time_elapse(min)", int(time.time()-self.tstart)//60)
                logger.record_tabular("nupdates", self.nupdates)
                logger.record_tabular("expl_length", self.episode_stats.get_mean("expl_length"))
                logger.record_tabular("expl_return", self.episode_stats.get_mean("expl_return"))
                logger.record_tabular("eval_length", self.episode_stats.get_mean("eval_length"))
                logger.record_tabular("eval_return", self.episode_stats.get_mean("eval_return"))
                logger.record_tabular("goal_pos_x", self.episode_stats.get_mean("expl_x_pos"))
                logger.record_tabular("goal_pos_y", self.episode_stats.get_mean("expl_y_pos"))
                if not self.model.dynamics.dummy:
                    logger.record_tabular("reached_ratio", self.episode_stats.get_sum("reached_cnt")/self.episode_stats.maxlen)
                    logger.record_tabular("reached_time", self.episode_stats.get_mean("reached_time"))
                for name, val in zip(names_ops_policy+names_ops_dynamics, values_ops_policy+values_ops_dynamics):
                    logger.record_tabular(name, float(val))
                logger.dump_tabular()
            # evaluation
            if self.eval_interval and int(steps/runner.nbatch) % self.eval_interval == 0:
                eval_info = self.evaluate(nb_eval=1)
                self.episode_stats.feed(eval_info["l"], "eval_length")
                self.episode_stats.feed(eval_info["r"], "eval_return")

    def initialize(self):
        init_steps = int(1e3)
        obs, actions, next_obs, info = self.runner.initialize(init_steps)
        self.buffer.initialize(obs, actions, next_obs, info)

    def evaluate(self, nb_eval):
        return self.runner_eval.evaluate(nb_eval)


def learn(network, env, seed=None, nsteps=20, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=100, buffer_size=50000, replay_ratio=4, replay_start=10000, c=10.0, trust_region=True,
          alpha=0.99, delta=1, replay_k=4, load_path=None, save_path=None, store_data=False, dynamics=None, eval_env=None,
          eval_interval=1000, **network_kwargs):

    '''
    Main entrypoint for ACER (Actor-Critic with Experience Replay) algorithm (https://arxiv.org/pdf/1611.01224.pdf)
    Train an agent with given network architecture on a given environment using ACER.

    Parameters:
    ----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies

    env:                environment. Needs to be vectorized for parallel environment simulation.
                        The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel) (default: 20)

    nstack:             int, size of the frame stack, i.e. number of the frames passed to the step model. Frames are stacked along channel dimension
                        (last image dimension) (default: 4)

    total_timesteps:    int, number of timesteps (i.e. number of actions taken in the environment) (default: 80M)

    q_coef:             float, value function loss coefficient in the optimization objective (analog of vf_coef for other actor-critic methods)

    ent_coef:           float, policy entropy coefficient in the optimization objective (default: 0.01)

    max_grad_norm:      float, gradient norm clipping coefficient. If set to None, no clipping. (default: 10),

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    rprop_epsilon:      float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    rprop_alpha:        float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting factor (default: 0.99)

    log_interval:       int, number of updates between logging events (default: 100)

    buffer_size:        int, size of the replay buffer (default: 50k)

    replay_ratio:       int, now many (on average) batches of data to sample from the replay buffer take after batch from the environment (default: 4)

    replay_start:       int, the sampling from the replay buffer does not start until replay buffer has at least that many samples (default: 10k)

    c:                  float, importance weight clipping factor (default: 10)

    trust_region        bool, whether or not algorithms estimates the gradient KL divergence between the old and updated policy and uses it to determine step size  (default: True)

    delta:              float, max KL divergence between the old policy and updated policy (default: 1)

    alpha:              float, momentum factor in the Polyak (exponential moving average) averaging of the model parameters (default: 0.99)

    load_path:          str, path to load the model from (default: None)

    **network_kwargs:               keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                    For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''

    logger.info("Running Acer with following kwargs")
    logger.info(locals())
    logger.info("\n")

    set_global_seeds(seed)
    if not isinstance(env, VecFrameStack):
        env = VecFrameStack(env, 1)

    if eval_env is None:
        logger.warn("eval_env is None. trying to parser_env_id.")
        env_id = parser_env_id(env)
        if env_id:
            env_type, env_id = get_env_type(env_id)
            eval_env = build_env(env_id, num_env=1, alg="acer", reward_scale=1.0)
            logger.warn("sucess build eval env:{}!".format(env_id))
        else:
            raise ValueError("Rebuild Eval Env Fail!")
    else:
        if not hasattr(eval_env, "num_env"):
            logger.warn("eval env not have the attribute of num_env!")
            logger.warn("we rebuild eval_env!")
            env_id = parser_env_id(env)
            if env_id:
                env_type, env_id = get_env_type(env_id)
                eval_env = build_env(env_id, num_env=1, alg="acer", reward_scale=1.0)
                logger.warn("sucess build eval env:{}!".format(env_id))
            else:
                raise ValueError("Rebuild Eval Env Fail!")
        else:
            if eval_env.num_env != 1:
                logger.warn("eval env's num_env must 1!")
                logger.warn("we rebuild eval_env!")
                env_id = parser_env_id(env)
                if env_id:
                    env_type, env_id = get_env_type(env_id)
                    eval_env = build_env(env_id, num_env=1, alg="acer", reward_scale=1.0)
                    logger.warn("sucess build eval env:{}!".format(env_id))
                else:
                    raise ValueError("Rebuild Eval Env Fail!")

    policy = build_policy(env, network, estimate_q=True, **network_kwargs)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    nstack = env.nstack
    if dynamics.dummy:
        model_exploration = Model(
            policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef,
            q_coef=q_coef, gamma=gamma, max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha,
            rprop_epsilon=rprop_epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
            trust_region=trust_region, alpha=alpha, delta=delta, dynamics=dynamics, scope="acer")
        model_evaluation = model_exploration

        def reward_fn(current_state, desired_goal):
            return np.zeros(current_state.shape[0])

        raise NotImplementedError("Now only support evaluation nenv=1")
    else:
        from curiosity.dynamics import DummyDynamics
        dummy_dynamics = DummyDynamics()
        model_exploration = Model(
            policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef,
            q_coef=q_coef, gamma=gamma, max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha,
            rprop_epsilon=rprop_epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
            trust_region=trust_region, alpha=alpha, delta=delta, dynamics=dynamics, scope="acer_expl")
        model_evaluation = Model(
            policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef,
            q_coef=q_coef, gamma=gamma, max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha,
            rprop_epsilon=rprop_epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
            trust_region=trust_region, alpha=alpha, delta=delta, dynamics=dummy_dynamics, scope="acer_eval"
        )

        def reward_fn(current_state, desired_goal):
            eps = 1e-6
            return np.exp(-np.sum(np.square(current_state-desired_goal), -1) /
                          (eps+np.sum(np.square(desired_goal), -1)))

    runner_training = Runner(env=env, model=model_exploration, nsteps=nsteps, save_path=save_path,
                             store_data=store_data, reward_fn=reward_fn)
    runner_evaluation = Runner(env=eval_env, model=model_evaluation, nsteps=nsteps, save_path=save_path,
                               store_data=store_data, reward_fn=reward_fn)

    if replay_ratio > 0:
        sample_goal_fn = make_sample_her_transitions("future", replay_k)
        buffer = Buffer(
            env=env, nsteps=nsteps, size=buffer_size, dynamics=dynamics, reward_fn=reward_fn,
            sample_goal_fn=sample_goal_fn
        )
    else:
        buffer = None
    nbatch = nenvs*nsteps
    acer = Acer(runner_training, runner_evaluation, model_exploration, model_evaluation, buffer, log_interval, eval_interval)
    acer.tstart = time.time()

    # === init to make sure we can get goal ===
    acer.initialize()

    for acer.steps in range(0, total_timesteps, nbatch): #nbatch samples, 1 on_policy call and multiple off-policy calls
        acer.call(on_policy=True)
        if replay_ratio > 0 and buffer.has_atleast(replay_start):
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                acer.call(on_policy=False)  # no simulation steps in this

    return model_evaluation
