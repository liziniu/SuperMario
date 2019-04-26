import time
import numpy as np
from baselines import logger
from baselines.common import set_global_seeds
from acer.policies import build_policy
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.utils import EpisodeStats
from acer.buffer import Buffer
from acer.runner import Runner
from common.her_sample import make_sample_her_transitions
from acer.model import Model


class Acer:
    def __init__(self, runner, model_exploration, model_evaluation, buffer, log_interval):
        self.runner = runner
        self.model_exploration = model_exploration
        self.model_evaluation = model_evaluation
        self.buffer = buffer
        self.log_interval = log_interval
        self.tstart = None
        self.episode_stats = EpisodeStats(runner.nsteps, runner.nenv)
        self.steps = None

    def call(self, on_policy):
        runner, model, model_eval, buffer, steps = self.runner, self.model_exploration, self.model_evaluation, self.buffer, self.steps
        if on_policy:
            enc_obs, obs, actions, ext_rewards, mus, dones, masks, goal_obs, goal_feats, int_rewards = runner.run()
            self.episode_stats.feed(ext_rewards, dones)
            if buffer is not None:
                buffer.put(enc_obs, actions, ext_rewards, mus, dones, masks, goal_obs)
        else:
            obs, actions, ext_rewards, mus, dones, masks, goal_feats, int_rewards = buffer.get()     # todo: add her

        # reshape stuff correctly except goal_feats
        obs = obs.reshape(runner.batch_ob_shape)
        actions = actions.reshape([runner.nbatch])
        ext_rewards = ext_rewards.reshape([runner.nbatch])
        mus = mus.reshape([runner.nbatch, runner.nact])
        dones = dones.reshape([runner.nbatch])
        masks = masks.reshape([runner.batch_ob_shape[0]])
        # we do not reshape goal_feats, and int_rewards

        # todo: batch process
        if model.scope != model_eval.scope:
            names_ops_policy, values_ops_policy = model.train_policy(
                obs, actions, int_rewards, dones, mus, model.initial_state, masks, steps, goal_feats)

            # Actually we do not feed goal_feats into policy!
            names_ops_policy_, values_ops_policy_ = model_eval.train_policy(
                obs, actions, ext_rewards, dones, mus, model.initial_state, masks, steps, goal_feats)
            names_ops_policy += names_ops_policy_
            values_ops_policy += values_ops_policy_
        else:
            names_ops_policy, values_ops_policy = model.train_policy(
                obs, actions, ext_rewards, dones, mus, model.initial_state, masks, steps, goal_feats
            )
        if on_policy:
            names_ops_dynamics, values_ops_dynamics = model.train_dynamics(obs, actions, steps)
        if on_policy and (int(steps/runner.nbatch) % self.log_interval == 0):
            logger.record_tabular("total_timesteps", steps)
            logger.record_tabular("fps", int(steps/(time.time() - self.tstart)))
            # IMP: In EpisodicLife env, during training, we get done=True at each loss of life, not just at the terminal state.
            # Thus, this is mean until end of life, not end of episode.
            # For true episode rewards, see the monitor files in the log folder.
            logger.record_tabular("mean_episode_length", self.episode_stats.mean_length())
            logger.record_tabular("mean_episode_reward", self.episode_stats.mean_reward())
            for name, val in zip(names_ops_policy+names_ops_dynamics, values_ops_policy+values_ops_dynamics):
                logger.record_tabular(name, float(val))
            logger.dump_tabular()

    def evaluate(self):
        pass

    def initialize(self):
        init_steps = int(1e3)
        obs, actions, next_obs, info = self.runner.initialize(init_steps)
        self.buffer.initialize(obs, actions, next_obs, info)


def learn(network, env, seed=None, nsteps=20, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=100, buffer_size=50000, replay_ratio=4, replay_start=2000, c=10.0, trust_region=True,
          alpha=0.99, delta=1, replay_k=4, load_path=None, save_path=None, store_data=False, dynamics=None, **network_kwargs):

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
            return np.sum(np.square(current_state-desired_goal), -1) / (eps+np.sum(np.square(desired_goal), -1))

    runner_training = Runner(env=env, model=model_exploration, nsteps=nsteps, save_path=save_path, store_data=store_data, reward_fn=reward_fn)
    runner_evaluation = Runner(env=env, model=model_evaluation, nsteps=nsteps, save_path=save_path, store_data=store_data, reward_fn=reward_fn)

    if replay_ratio > 0:
        sample_goal_fn = make_sample_her_transitions("future", replay_k)
        buffer = Buffer(
            env=env, nsteps=nsteps, size=buffer_size, dynamics=dynamics, reward_fn=reward_fn,
            sample_goal_fn=sample_goal_fn
        )
    else:
        buffer = None
    nbatch = nenvs*nsteps
    acer = Acer(runner_training, model_exploration, model_evaluation, buffer, log_interval)
    acer.tstart = time.time()

    # === init to make sure we can get goal ===
    acer.initialize()

    for acer.steps in range(0, total_timesteps, nbatch): #nbatch samples, 1 on_policy call and multiple off-policy calls
        acer.call(on_policy=True)
        if replay_ratio > 0 and buffer.has_atleast(replay_start):
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                acer.call(on_policy=False)  # no simulation steps in this

    # === Evaluation ===
    acer.evaluate()

    return model_evaluation
