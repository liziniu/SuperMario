import time
import numpy as np
from baselines import logger
from baselines.common import set_global_seeds
from acer.policies import build_policy
from common.env_util import VecFrameStack
from her.buffer import Buffer
from her.runner import Runner
from common.her_sample import make_sample_her_transitions
from her.model import Model
from her.util import Acer, vf_dist
import sys
from baselines.common.tf_util import get_session
import os
from common.buffer import ReplayBuffer
from her.defaults import get_store_keys, THRESHOLD


def learn(network, env, seed=None, nsteps=20, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=50, buffer_size=50000, replay_ratio=4, replay_start=10000, c=10.0, trust_region=True,
          alpha=0.99, delta=1, replay_k=4, load_path=None, env_eval=None, eval_interval=300, dist_type="l1",
          save_model=False, simple_store=True, goal_shape=(84, 84, 4), nb_train_epoch=4, desired_x_pos=None,
          her=True, buffer2=True, **network_kwargs):

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
    if sys.platform == "darwin":
        log_interval = 5
        replay_start = 10

    logger.info("Running Acer with following kwargs")
    logger.info(locals())
    logger.info("\n")

    set_global_seeds(seed)
    if not isinstance(env, VecFrameStack):
        env = VecFrameStack(env, 1)

    if env_eval is None:
        raise ValueError("env_eval is required!")

    policy = build_policy(env, network, estimate_q=True, **network_kwargs)
    nenvs = env.num_envs
    nenvs_eval = env_eval.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    nstack = env.nstack
    sess = get_session()

    model = Model(
        sess=sess, policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef,
        q_coef=q_coef, gamma=gamma, max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha,
        rprop_epsilon=rprop_epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
        trust_region=trust_region, alpha=alpha, delta=delta, scope="her", goal_shape=goal_shape)

    def reward_fn_v1(current_state, desired_goal):
        eps = 1e-6
        return np.exp(-np.sum(np.square(current_state-desired_goal), -1) /
                        (eps+np.sum(np.square(desired_goal), -1)))

    def reward_fn_v2(current_pos_infos, goal_pos_infos, sparse=True):
        assert current_pos_infos.shape == goal_pos_infos.shape
        coeff = 0.03
        threshold = THRESHOLD

        dist = vf_dist(current_pos_infos, goal_pos_infos)
        if sparse:
            rewards = (dist < threshold).astype(float)
        else:
            rewards = np.exp(-coeff * dist)
        return rewards

    assert dist_type in ["l1", "l2"]
    if dist_type == "l2":
        reward_fn = reward_fn_v1
    else:
        reward_fn = reward_fn_v2

    # we still need two runner to avoid one reset others' envs.
    runner = Runner(env=env, model=model, nsteps=nsteps, reward_fn=reward_fn, load_path=load_path,
                    desired_x_pos=desired_x_pos)

    if replay_ratio > 0:
        if her:
            sample_goal_fn = make_sample_her_transitions("future", replay_k)
        else:
            def dummpy_sample():
                def sample(dones, **kwargs):
                    dummy = np.copy(dones)
                    dummy.fill(False)
                    index = np.where(dummy)
                    return index, index
                return sample
            sample_goal_fn = dummpy_sample()
        assert env.num_envs == env_eval.num_envs
        if buffer2:
            buffer = ReplayBuffer(env=env, sample_goal_fn=sample_goal_fn, nsteps=nsteps, size=buffer_size,
                                  keys=get_store_keys(), reward_fn=reward_fn)
        else:
            buffer = Buffer(env=env, nsteps=nsteps, size=buffer_size, reward_fn=reward_fn, sample_goal_fn=sample_goal_fn,
                            goal_shape=model.goal_shape)
    else:
        buffer = None
    acer = Acer(runner, model, buffer, log_interval,)
    acer.tstart = time.time()

    # === init to make sure we can get goal ===

    replay_start = replay_start * env.num_envs / (env.num_envs + env_eval.num_envs)
    onpolicy_cnt = 0

    while acer.steps < total_timesteps:
        acer.call(replay_start=replay_start, nb_train_epoch=nb_train_epoch)
        acer.steps += nenvs * nsteps
        onpolicy_cnt += 1
    acer.save(os.path.join(logger.get_dir(), "models", "{}.pkl".format(acer.steps)))

    return model