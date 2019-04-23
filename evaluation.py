import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import seaborn as sns
import pandas as pd
import os
import argparse
import pickle
import json
from cmd_util import set_global_seeds, make_vec_env
from auxilliary_tasks import InverseDynamics, RandomFeature, RandomNetworkDistillation
from dynamics import Dynamics
from run import build_env
import datetime


class Model:
    def __init__(self, sess, env, aux_task, feat_dim, dynamics_dim, lr):
        self.make_auxiliary_task = {"RND": RandomNetworkDistillation,
                                    "IDF": InverseDynamics,
                                    "RF": RandomFeature}[aux_task.upper()]
        self.auxiliary_task = self.make_auxiliary_task(sess, env, feat_dim)
        self.dynamics = Dynamics(self.auxiliary_task, dynamics_dim)

        self.obs_shape = env.observation_space.shape
        self.ac_shape = env.action_space.shape
        del env
        self.opt = tf.train.AdamOptimizer(lr)
        self.int_rew = self.dynamics.novelty
        self.total_loss = self.auxiliary_task.loss + self.dynamics.loss
        self.loss_names = ["aux_loss", "dynamic_loss"]
        self.losses = [self.auxiliary_task.loss, self.dynamics.loss]

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradsandvars = self.opt.compute_gradients(self.total_loss, params)
        self.train_op = self.opt.apply_gradients(gradsandvars)

        self.sess = sess or tf.Session()

        self.train_history = []

    def get_reward(self, obs, next_obs, act):
        rew = []
        batch_run = 1024
        for start in range(0, len(obs), batch_run):
            end = min(len(obs), start+batch_run)
            sli = slice(start, end, 1)
            int_rew = self.sess.run(self.dynamics.novelty, feed_dict={
                self.auxiliary_task.obs: obs[sli],
                self.auxiliary_task.next_obs: next_obs[sli],
                self.auxiliary_task.ac: act[sli]
            })
            rew.extend(list(int_rew))
        rew = np.array(rew).flatten()
        return rew

    def train(self, data, rollout_size, batch_size, nb_opt, online=True, plot=False, norm=False, save_path=None):
        """
        :param data: list of dict. [{"obs": arr; "next_obs": arr}]
        """
        data = np.asarray(data, dtype=object)
        nb_sample = len(data)
        print("training data:{}".format(nb_sample))
        nb_rollout = nb_sample//rollout_size
        episode_aux_loss = []
        episode_dynamic_loss = []

        plot_data = []
        for i in range(nb_rollout):
            if online:
                samples = data[i*rollout_size:(i+1)*rollout_size]
            else:
                samples = data[:(i+1)*rollout_size]
            if plot:
                data_to_plot = data[:(i+1)*rollout_size]
                x_pos_plot, y_pos_plot, obs_plot, next_obs_plot, act_plot, value = self.dict2arr(
                    data_to_plot, keys=["x_pos", "y_pos", "obs", "next_obs", "act", "value"]
                )
                int_rew = self.get_reward(obs_plot, next_obs_plot, act_plot)
                plot_data.append(dict(
                    x_pos=x_pos_plot,
                    y_pos=y_pos_plot,
                    int_rew=int_rew,
                    value=value
                ))
                if not os.path.exists(os.path.join(save_path, "forward")):
                    os.makedirs(os.path.join(save_path, "forward"))
                if not os.path.exists(os.path.join(save_path, "inverse")):
                    os.makedirs(os.path.join(save_path, "inverse"))

                # forward
                cmap = plt.cm.get_cmap('magma')
                plt.figure(dpi=100)
                plt.subplot(1, 2, 1)
                plt.scatter(x_pos_plot, y_pos_plot, c=int_rew, cmap=cmap)
                plt.colorbar()
                plt.title("novelty")
                plt.subplot(1, 2, 2)
                plt.scatter(x_pos_plot, y_pos_plot, c=value, cmap=cmap)
                plt.title("value")
                plt.colorbar()
                plt.subplots_adjust(wspace=0.5, hspace=0)
                plt.savefig(os.path.join(save_path, "forward", "{}.png".format(i)))
                plt.close()
                # inverse
                cmap = plt.cm.get_cmap('magma_r')
                plt.figure(dpi=100)
                plt.subplot(1, 2, 1)
                plt.scatter(x_pos_plot, y_pos_plot, c=int_rew, cmap=cmap)
                plt.colorbar()
                plt.title("novelty")
                plt.subplot(1, 2, 2)
                plt.scatter(x_pos_plot, y_pos_plot, c=value, cmap=cmap)
                plt.title("value")
                plt.colorbar()
                plt.subplots_adjust(wspace=0.5, hspace=0)
                plt.savefig(os.path.join(save_path, "inverse", "{}.png".format(i)))
                plt.close()

            epoch_aux_loss = []
            epoch_dynamic_loss = []
            for j in range(nb_opt):
                np.random.shuffle(samples)
                for start in range(0, rollout_size, batch_size):
                    minibatch = samples[start: start+batch_size]
                    obs_train, next_obs_train, ac_train = self.dict2arr(minibatch, keys=["obs", "next_obs", "act"])
                    loss, _ = self.sess.run([self.losses, self.train_op],
                                            feed_dict={self.auxiliary_task.obs: obs_train,
                                                       self.auxiliary_task.next_obs: next_obs_train,
                                                       self.auxiliary_task.ac: ac_train})
                    epoch_aux_loss.append(loss[0])
                    epoch_dynamic_loss.append(loss[1])
            episode_aux_loss.append(np.mean(epoch_aux_loss))
            episode_dynamic_loss.append(np.mean(epoch_dynamic_loss))
            print("epoch:{}/{}|aux_loss:{:.4f}|dynamic_loss:{:.4f}".format(
                i, nb_rollout, np.mean(epoch_aux_loss), np.mean(epoch_dynamic_loss)))

        df = pd.DataFrame({
            "epoch": np.arange(1, len(episode_aux_loss)+1),
            "aux_loss": np.asarray(episode_aux_loss),
            "dynamic_loss": np.asarray(episode_dynamic_loss)
        })
        df.to_csv(os.path.join(save_path, "training.csv"))
        with open(os.path.join("plot_data.csv"), "wb") as f:
            pickle.dump(plot_data, f, -1)

    def dict2arr(self, data, keys):
        if isinstance(data, list):
            data = np.array(data, dtype=object)
        nb = data.shape[0]
        locals()["obs"] = np.empty((nb, *self.obs_shape), np.float32)
        locals()["next_obs"] = np.empty((nb, *self.obs_shape), np.float32)
        locals()["x_pos"] = np.empty(nb, np.float32)
        locals()["y_pos"] = np.empty(nb, np.float32)
        locals()["act"] = np.empty((nb, *self.ac_shape), np.float32)
        locals()["value"] = np.empty(nb, np.float32)
        for i in range(nb):
            for key in keys:
                locals()[key][i] = data[i][key]
        result = []
        for key in keys:
            result.append(locals()[key])
        return result


def preprocess_data(data):
    keys = ["obs", "x_pos", "y_pos"]
    data_new = {key: [] for key in keys}
    for episode_data in data:
        for key in keys:
            data_new[key].extend(episode_data[key])
    return data_new


def flatten_data(data):
    _data = []
    for episode_data in data:
        _data.extend(episode_data)
    return _data


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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    env = build_env(args)

    model = Model(
        sess=sess,
        env=env,
        aux_task=args.aux_task,
        feat_dim=args.feat_dim,
        dynamics_dim=args.dyna_dim,
        lr=args.lr
    )
    sess.run(tf.global_variables_initializer())

    save_path = "{}/{}-{}-{}".format(
        args.load_path,
        args.aux_task,
        args.env,
        args.seed,
    ).replace("logs", "plots")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    model.train(
        flatten_data(data),
        rollout_size=args.rollout_size*args.num_env,
        batch_size=args.batch_size,
        nb_opt=args.nb_opt,
        online=args.online,
        plot=args.plot,
        norm=args.norm,
        save_path=save_path,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nb_opt", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--online", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument("--memo", type=str, default=None)

    parser.add_argument("--aux_task", type=str, default="rnd", choices=["rnd", "rf", "idf"])
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--dyna_dim", type=int, default=64)

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
