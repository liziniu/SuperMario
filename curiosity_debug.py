import matplotlib
import numpy as np
import tensorflow as tf
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import pickle
import json
from baselines.common import set_global_seeds
from curiosity.dynamics import Dynamics
from run import build_env
from scipy.stats import pearsonr
from common.util import DataRecorder
from baselines import logger


class Model:
    def __init__(self, sess, env, aux_task, feat_dim, lr):
        self.sess = sess or tf.Session()

        self.dynamics = Dynamics(sess=self.sess, env=env, auxiliary_task=aux_task, feat_dim=feat_dim,
                                 queue_size=1000, normalize_novelty=True)

        self.obs_shape = env.observation_space.shape
        self.ac_shape = env.action_space.shape
        del env
        self.opt = tf.train.RMSPropOptimizer(lr, decay=0.99)
        self.aux_loss = self.dynamics.aux_loss
        self.dyna_loss = self.dynamics.dyna_loss
        self.loss = self.aux_loss + self.dyna_loss

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradsandvars = self.opt.compute_gradients(self.loss, params)
        self.train_op = self.opt.apply_gradients(gradsandvars)

        self.train_history = []

    def train(self, data, rollout_size, online=True, save_path=None):
        """
        :param data: list of dict. [{"obs": arr; "next_obs": arr}]
        """
        self.recoder = DataRecorder(os.path.join(save_path, "training"))
        for episode, episode_data in enumerate(data):
            episode_length = len(episode_data["obs"])
            obs, act, next_obs, x_pos = episode_data["obs"], episode_data["act"], episode_data["next_obs"], episode_data["x_pos"]
            episode_novelty = []
            if not online:
                ind = np.random.permutation(episode_length)
                obs, act, next_obs, x_pos = obs[ind], act[ind], next_obs[ind], x_pos[ind]

            for start in range(0, episode_length, rollout_size):
                end = start + rollout_size
                batch_obs, batch_act, batch_next_obs, batch_x_pos = obs[start:end], act[start:end], next_obs[start:end], x_pos[start:end]

                novelty = self.sess.run(self.dynamics.novelty, feed_dict={self.dynamics.obs: obs,
                                                                          self.dynamics.ac: act,
                                                                          self.dynamics.next_obs: next_obs})
                self.sess.run(self.train_op, feed_dict={self.dynamics.obs: batch_obs, self.dynamics.ac: batch_act,
                                                        self.dynamics.next_obs: batch_next_obs})
                p = pearsonr(x_pos, novelty)[0]
                logger.info("Episode:{}|Epoch:{}|P:{}".format(episode, start//rollout_size, p))
                episode_novelty.append(novelty)
                self.recoder.store({"x_pos": x_pos, "novelty": novelty, "episode": episode, "epoch": start//rollout_size,
                                    "p": p})
                plt.figure()
                plt.scatter(x_pos, novelty)
                # plt.yscale("log")
                plt.savefig(os.path.join(save_path, "{}_{}.png".format(episode, start//rollout_size)))
                plt.close()
            self.recoder.dump()


def preprocess_data(data):
    data_episode = []
    for episode_data in data:
        tmp = {"obs": [], "act": [], "x_pos": []}
        for t in episode_data:
            tmp["obs"].append(t["obs"][0])
            tmp["act"].append(t["act"][0])
            tmp["x_pos"].append(t["info"][0]["x_pos"])
        tmp["obs"] = np.asarray(tmp["obs"], dtype=np.float32)
        tmp["act"] = np.asarray(tmp["act"], dtype=np.float32)
        tmp["x_pos"] = np.asarray(tmp["x_pos"], dtype=np.float32)

        tmp["next_obs"] = np.copy(tmp["obs"][1:])
        tmp["obs"] = tmp["obs"][:-1]
        tmp["act"] = tmp["act"][:-1]
        tmp["x_pos"] = tmp["x_pos"][:-1]
        data_episode.append(tmp)
    return data_episode


def visualize_p(path):
    f = open(path, "rb")
    data = []
    while True:
        try:
            data.append(pickle.load(f))
        except Exception as e:
            print(e)
            break
    p = []
    episode = None
    for i in range(len(data)):
        if episode is None:
            episode = data[i]["episode"]
        ep = data[i]["episode"]
        if ep == episode:
            p.append(data[i]["p"])
        else:
            plt.figure()
            plt.plot(p)
            plt.savefig(os.path.join(path, "p_{}.png").format(episode))
            p = []
            p.append(ep)
            episode = ep
            print("Epoch:{} done".format(ep))


def main(args):
    f = open("{}/data.pkl".format(args.load_path), "rb")
    data = []
    while True:
        try:
            data.append(pickle.load(f))
        except:
            break
    print("Episode:", len(data))

    set_global_seeds(args.seed)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config, )
    env = build_env(env_id=args.env, num_env=1, alg="acer", reward_scale=1.0, env_type=args.env_type,
                    gamestate=None, seed=None, prefix="")

    model = Model(
        sess=sess,
        env=env,
        aux_task=args.aux_task,
        feat_dim=args.feat_dim,
        lr=args.lr
    )
    sess.run(tf.global_variables_initializer())

    save_path = "{}/plots-{}-{}-{}".format(
        args.load_path,
        args.memo,
        args.online,
        args.aux_task,
    )
    logger.configure(dir=save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    model.train(
        preprocess_data(data),
        rollout_size=args.rollout_size*args.num_env,
        save_path=save_path,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nb_opt", type=int, default=5)
    parser.add_argument("--lr", type=float, default=7e-4)

    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--online", action="store_true", default=False)

    parser.add_argument("--aux_task", type=str, default="RF", choices=["RF", "RND", "IDF"])
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--dyna_dim", type=int, default=512)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_path", type=str, default="logs")

    parser.add_argument('--gamestate', default=None)
    parser.add_argument("--alg", type=str, default="ppo2")
    parser.add_argument("--env_type", type=str, default="atari")
    parser.add_argument("--env", type=str)
    parser.add_argument("--num_env", type=int, default=1)
    parser.add_argument("--reward_scale", type=float, default=1.0, choices=[1.0])

    args = parser.parse_args()

    main(args)