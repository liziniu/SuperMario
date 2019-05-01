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
from copy import deepcopy


class Acer:
    def __init__(self, runner_expl, runner_eval, model_expl, model_eval, buffer, log_interval):
        self.runner_expl = runner_expl
        self.runner_eval = runner_eval
        self.model_expl = model_expl
        self.model_eval = model_eval

        self.buffer = buffer
        self.log_interval = log_interval
        self.tstart = None
        self.steps = 0
        self.nupdates = 0

        keys = []
        keys += ["expl_return", "expl_length", ]
        keys += ["eval_return", "eval_length", ]
        keys += ["goal_x_pos", "goal_y_pos", ]
        keys += ["reached_cnt", "reached_time", "goal_abs_dist"]
        keys += ["queue_max", "queue_std"]

        self.logger_keys = keys.copy()
        self.logger_keys.remove("reached_cnt")
        self.episode_stats = EpisodeStats(maxlen=20, keys=keys)

    def call(self, on_policy, model_name=None, cnt=None):
        names_ops, values_ops = [], []
        if model_name == "expl":
            runner = self.runner_expl
        else:
            runner = self.runner_eval

        if on_policy:
            # collect data
            results = runner.run()
            self.buffer.put(results["enc_obs"], results["actions"], results["ext_rewards"], results["mus"],
                            results["dones"], results["masks"], results["goal_obs"], results["goal_infos"],
                            results["obs_infos"])
            # training dynamics & put goals
            mb_obs, mb_actions, mb_next_obs, mb_obs_infos = self.adjust_dynamics_input_shape(results)
            queue_info = self.model_expl.dynamics.put_goal(mb_obs, mb_actions, mb_next_obs, mb_obs_infos)
            if len(queue_info.keys()) > 0:
                self.episode_stats.feed(queue_info["queue_max"], "queue_max")
                self.episode_stats.feed(queue_info["queue_std"], "queue_std")
            names_ops_, values_ops_ = self.model_expl.train_dynamics(mb_obs, mb_actions, mb_next_obs, self.steps)
            names_ops, values_ops = names_ops + names_ops_, values_ops + values_ops_

            # store useful episode information
            self.record_episode_info(results["episode_infos"], model_name)
        else:
            results = self.buffer.get()
        obs, actions, ext_rewards, mus, dones, masks, int_rewards, goal_obs = self.adjust_policy_input_shape(results)

        # Training Policy
        assert self.model_expl.scope != self.model_eval.scope
        names_ops_, values_ops_ = self.model_eval.train_policy(
            obs, actions, ext_rewards, dones, mus, self.model_eval.initial_state, masks, self.steps, goal_obs)
        names_ops, values_ops = names_ops + names_ops_, values_ops+values_ops_
        names_ops_, values_ops_ = self.model_expl.train_policy(
            obs, actions, int_rewards, dones, mus, self.model_expl.initial_state, masks, self.steps, goal_obs)
        names_ops, values_ops = names_ops + names_ops_, values_ops + values_ops_
        self.nupdates += 1

        # Logging
        if on_policy and cnt % self.log_interval == 0:
            self.log(names_ops, values_ops)

    def initialize(self):
        init_steps = int(2e3)
        self.runner_expl.initialize(init_steps)

    def evaluate(self, nb_eval):
        results = self.runner_eval.evaluate(nb_eval)
        self.episode_stats.feed(results["l"], "eval_length")
        self.episode_stats.feed(results["r"], "eval_return")

    @staticmethod
    def adjust_dynamics_input_shape(results):
        # flatten on-policy data (nenv, nstep, ...) for dynamics training and put_goal
        mb_next_obs = deepcopy(results["obs"][:, 1:])
        mb_obs = deepcopy(results["obs"][:, :-1])
        mb_obs = mb_obs.reshape((-1,) + mb_obs.shape[2:])
        mb_next_obs = mb_next_obs.reshape((-1,) + mb_next_obs.shape[2:])
        mb_actions = deepcopy(results["actions"])
        mb_actions = mb_actions.reshape((-1,) + mb_actions.shape[2:])
        mb_obs_infos = deepcopy(results["obs_infos"])
        mb_obs_infos = mb_obs_infos.reshape(-1)
        return mb_obs, mb_actions, mb_next_obs, mb_obs_infos

    def adjust_policy_input_shape(self, results):
        assert self.runner_expl.nbatch == self.runner_eval.nbatch
        runner = self.runner_expl

        obs = results["obs"].reshape(runner.batch_ob_shape)
        actions = results["actions"].reshape(runner.nbatch)
        ext_rewards = results["ext_rewards"].reshape(runner.nbatch)
        mus = results["mus"].reshape([runner.nbatch, runner.nact])
        dones = results["dones"].reshape([runner.nbatch])
        masks = results["masks"].reshape([runner.batch_ob_shape[0]])
        int_rewards = results["int_rewards"].reshape([runner.nbatch])
        goal_obs = results["goal_obs"].reshape(runner.batch_ob_shape)
        return obs, actions, ext_rewards, mus, dones, masks, int_rewards, goal_obs
    
    def record_episode_info(self, episode_infos, model_name):
        for info in episode_infos:
            reached_info = info.get("reached_info")
            if reached_info:
                self.episode_stats.feed(reached_info["reached"], "reached_cnt")
                self.episode_stats.feed(reached_info["time_ratio"], "reached_time")
                self.episode_stats.feed(reached_info["abs_dist"], "goal_abs_dist")
            goal_info = info.get("goal_info")
            if goal_info:
                self.episode_stats.feed(goal_info["x_pos"], "goal_x_pos")
                self.episode_stats.feed(goal_info["y_pos"], "goal_y_pos")
            return_info = info.get("episode")
            if return_info:
                self.episode_stats.feed(return_info["l"], "{}_length".format(model_name))
                self.episode_stats.feed(return_info["r"], "{}_return".format(model_name))

    def log(self, names_ops, values_ops):
        logger.record_tabular("total_timesteps", self.steps)
        logger.record_tabular("fps", int(self.steps / (time.time() - self.tstart)))
        logger.record_tabular("time_elapse(min)", int(time.time() - self.tstart) // 60)
        logger.record_tabular("nupdates", self.nupdates)
        for key in self.logger_keys:
            logger.record_tabular(key, self.episode_stats.get_mean(key))
        logger.record_tabular("reached_ratio", self.episode_stats.get_sum("reached_cnt") / self.episode_stats.maxlen)
        for name, val in zip(names_ops, values_ops):
            logger.record_tabular(name, float(val))
        logger.dump_tabular()
        

def learn(network, env, seed=None, nsteps=20, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=100, buffer_size=50000, replay_ratio=4, replay_start=10000, c=10.0, trust_region=True,
          alpha=0.99, delta=1, replay_k=4, load_path=None, save_path=None, store_data=False, dynamics=None, env_eval=None,
          eval_interval=300, use_eval_model_collect=True, dist_type="l1", **network_kwargs):

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

    if env_eval is None:
        raise ValueError("env_eval is required!")
    #     logger.warn("env_eval is None. trying to parser_env_id.")
    #     env_id = parser_env_id(env)
    #     if env_id:
    #         env_type, env_id = get_env_type(env_id)
    #         num_env = env.num_envs if hasattr(env, "num_envs") else 1
    #         env_eval = build_env(env_id, num_env=num_env, alg="acer", reward_scale=1.0)
    #         logger.warn("sucess build eval env:{}!".format(env_id))
    #     else:
    #         raise ValueError("Rebuild Eval Env Fail!")
    # else:
    #     if not hasattr(env_eval, "num_env"):
    #         logger.warn("eval env not have the attribute of num_env!")
    #         logger.warn("we rebuild env_eval!")
    #         env_id = parser_env_id(env)
    #         if env_id:
    #             env_type, env_id = get_env_type(env_id)
    #             env_eval = build_env(env_id, num_env=1, alg="acer", reward_scale=1.0)
    #             logger.warn("sucess build eval env:{}!".format(env_id))
    #         else:
    #             raise ValueError("Rebuild Eval Env Fail!")
    #     else:
    #         if env_eval.num_env != 1:
    #             logger.warn("eval env's num_env must 1!")
    #             logger.warn("we rebuild env_eval!")
    #             env_id = parser_env_id(env)
    #             if env_id:
    #                 env_type, env_id = get_env_type(env_id)
    #                 env_eval = build_env(env_id, num_env=1, alg="acer", reward_scale=1.0)
    #                 logger.warn("sucess build eval env:{}!".format(env_id))
    #             else:
    #                 raise ValueError("Rebuild Eval Env Fail!")

    policy = build_policy(env, network, estimate_q=True, **network_kwargs)
    nenvs = env.num_envs
    nenvs_eval = env_eval.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    nstack = env.nstack
    if dynamics.dummy:
        raise NotImplementedError("Now only support acer with her.")
        # model_exploration = Model(
        #     policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef,
        #     q_coef=q_coef, gamma=gamma, max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha,
        #     rprop_epsilon=rprop_epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
        #     trust_region=trust_region, alpha=alpha, delta=delta, dynamics=dynamics, scope="acer")
        # model_evaluation = model_exploration
        #
        # def reward_fn(current_state, desired_goal):
        #     return np.zeros(current_state.shape[0])
        #
        # raise NotImplementedError("Now only support evaluation nenv=1")
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

        def reward_fn_v1(current_state, desired_goal):
            eps = 1e-6
            return np.exp(-np.sum(np.square(current_state-desired_goal), -1) /
                          (eps+np.sum(np.square(desired_goal), -1)))

        def reward_fn_v2(current_pos_infos, goal_pos_infos):
            assert current_pos_infos.shape == goal_pos_infos.shape
            nenv, nsteps = current_pos_infos.shape[0], current_pos_infos.shape[1]
            mb_int_rewards = np.empty([nenv, nsteps])
            eps = 1e-3
            for i in range(nenv):
                for j in range(nsteps):
                    dist = abs(float(current_pos_infos[i][j]["x_pos"]) - float(goal_pos_infos[i][j]["x_pos"])) +\
                           abs(float(current_pos_infos[i][j]["y_pos"]) - float(goal_pos_infos[i][j]["y_pos"]))
                    mb_int_rewards[i][j] = np.clip(np.exp(-dist), eps, 1)
            return mb_int_rewards

        assert dist_type in ["l1", "l2"]
        if dist_type == "l2":
            reward_fn = reward_fn_v1
        else:
            reward_fn = reward_fn_v2

    # we still need two runner to avoid one reset others' envs.
    runner_expl = Runner(env=env, model=model_exploration, nsteps=nsteps, save_path=save_path, store_data=store_data,
                         reward_fn=reward_fn, sample_goal=True, dist_type=dist_type)
    runner_eval = Runner(env=env_eval, model=model_evaluation, nsteps=nsteps, save_path=save_path, store_data=store_data,
                         reward_fn=reward_fn, sample_goal=False, dist_type=dist_type)

    if replay_ratio > 0:
        sample_goal_fn = make_sample_her_transitions("future", replay_k)
        assert env.num_envs == env_eval.num_envs
        buffer = Buffer(env=env, nsteps=nsteps, size=buffer_size, dynamics=dynamics, reward_fn=reward_fn,
                        sample_goal_fn=sample_goal_fn, dist_type=dist_type)
    else:
        buffer = None
    nbatch_expl = nenvs*nsteps
    nbatch_eval = nenvs_eval*nsteps
    acer = Acer(runner_expl, runner_eval, model_exploration, model_evaluation, buffer, log_interval)
    acer.tstart = time.time()

    # === init to make sure we can get goal ===
    acer.initialize()

    if use_eval_model_collect:
        replay_start = replay_start * env.num_envs / (env.num_envs + env_eval.num_envs)
    onpolicy_cnt = 0
    while acer.steps < total_timesteps:
        # logger.info("-------------------expl running-------------------")
        acer.call(on_policy=True, model_name="expl", cnt=onpolicy_cnt)
        acer.steps += nbatch_expl
        onpolicy_cnt += 1
        if use_eval_model_collect:
            # logger.info("-------------------eval running-------------------")
            acer.call(on_policy=True, model_name="eval", cnt=onpolicy_cnt)
            acer.steps += nbatch_eval
            onpolicy_cnt += 1
        if replay_ratio > 0:
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                if buffer.has_atleast(replay_start):
                    # logger.info("-----------using replay buffer from expl-----------")
                    acer.call(on_policy=False)
        if not use_eval_model_collect and onpolicy_cnt % eval_interval == 0:
            acer.evaluate(nb_eval=1)
    return model_evaluation
