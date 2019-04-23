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
    def __init__(self, sess, env, aux_task, feature_dim, dynamics_dim, lr):
        self.make_auxiliary_task = {"RND": RandomNetworkDistillation,
                                    "IDF": InverseDynamics,
                                    "RF": RandomFeature}[aux_task.upper()]
        self.obs_shape = env.observation_space.shape
        self.auxiliary_task = self.make_auxiliary_task(sess, env, feature_dim)
        self.dynamics = Dynamics(self.auxiliary_task, dynamics_dim)

        self.opt = tf.train.AdamOptimizer(lr)
        self.int_rew = self.dynamics.novelty
        self.loss = [self.auxiliary_task.loss, self.dynamics.loss]
        self.loss_name = ["aux_loss", "dynamic_loss"]

        self.to_report = {loss_name: loss for loss_name in self.loss_name for loss in self.loss}
        self.train_op = self.opt.minimize(self.loss)

        self.sess = sess or tf.Session()

        self.train_history = []

    def get_reward(self, obs, next_obs):
        return self.sess.run(self.dynamics.novelty, feed_dict={
            self.auxiliary_task.obs: obs, self.auxiliary_task.next_obs: next_obs})

    def train(self, data, rollout_size, batch_size, nb_opt, online=True, plot=False, norm=False, save_path=None):
        """
        :param data: list of dict. [{"obs": arr; "next_obs": arr}]
        """
        data = np.asarray(data, dtype=object)
        nb_sample = len(data)
        nb_rollout = nb_sample//rollout_size
        episode_aux_loss = []
        episode_dynamic_loss = []
        for i in range(nb_rollout):
            if online:
                samples = data[i*rollout_size:(i+1)*rollout_size]
            else:
                samples = data[:(i+1)*rollout_size]
            if plot:
                data_to_plot = data[:(i+1)*rollout_size]
                x_pos_plot, y_pos_plot, obs_plot, next_obs_plot = self.dict2arr(
                    data_to_plot, keys=["x_pos", "y_pos", "obs", "next_obs"]
                )
                int_rew = self.get_reward(obs_plot, next_obs_plot)
                df_tmp = pd.DataFrame({"x_pos": x_pos_plot, "y_pos": y_pos_plot, "int_rew": int_rew})
                plt.figure()
                if norm:
                    sns.scatterplot(x="x_pos", y="y_pos", hue="int_rew", hue_norm=LogNorm(), data=df_tmp)
                else:
                    sns.scatterplot(x="x_pos", y="y_pos", hue="int_rew", data=df_tmp)
                assert save_path is not None
                plt.savefig(os.path.join(save_path, "{}.png".format(i)))
                plt.close()

            epoch_aux_loss = []
            epoch_dynamic_loss = []
            for j in range(nb_opt):
                np.random.shuffle(samples)
                for start in range(0, rollout_size, batch_size):
                    minibatch = samples[start: start+batch_size]
                    obs_train, next_obs_train = self.dict2arr(minibatch, keys=["obs", "next_obs"])
                    loss, _ = self.sess.run([self.loss, self.train_op],
                                            feed_dict={self.auxiliary_task.obs: obs_train,
                                                       self.auxiliary_task.next_obs: next_obs_train})
                    epoch_aux_loss.append(loss[0])
                    epoch_dynamic_loss.append(loss[1])
            episode_aux_loss.append(np.mean(epoch_aux_loss))
            epoch_dynamic_loss.append(np.mean(epoch_dynamic_loss))
            print("epoch:{}/{}|aux_loss:{:.4f}|dynamic_loss:{}".format(
                i, nb_rollout, np.mean(epoch_aux_loss), np.mean(epoch_dynamic_loss)))

        df = pd.DataFrame({
            "epoch": np.arange(1, len(episode_aux_loss)+1),
            "aux_loss": np.asarray(episode_aux_loss),
            "dynamic_loss": np.asarray(episode_dynamic_loss)
        })
        df.to_csv(os.path.join(save_path, "training.csv"))

    def dict2arr(self, data, keys):
        if isinstance(data, list):
            data = np.array(data, dtype=object)
        nb = data.shape[0]
        locals()["obs"] = np.empty((nb, *self.obs_shape), np.float32)
        locals()["obs_next"] = np.empty((nb, *self.obs_shape), np.float32)
        locals()["x_pos"] = np.empty((nb, 1), np.float32)
        locals()["y_pos"] = np.empty((nb, 1), np.float32)
        for i in range(nb):
            for key in keys:
                locals()[key][i] = data[i][key]
        return (locals()["key"], for key in keys)


def preprocess_data(data):
    keys = ["obs", "x_pos", "y_pos"]
    data_new = {key: [] for key in keys}
    for episode_data in data:
        for key in keys:
            data_new[key].extend(episode_data[key])
    return data_new


def main(args):
    f = open("{}/data.pkl".format(args.load_path), "rb")
    raw_data = []
    while True:
        try:
            raw_data.append(pickle.load(f))
        except:
            break
    print("Episode:", len(raw_data))

    set_global_seeds(args.seed)

    # data = preprocess_data(raw_data)
    data = raw_data
    nb_samples = len(data)
    print("# of samples:", nb_samples)

    sess = tf.Session()
    env = build_env(args)

    model = Model(
        sess=sess,
        env=env,
        aux_task=args.aux_task,
        feature_dim=args.feat_dim,
        dynamics_dim=args.dyna_dim,
        lr=args.lr
    )
    sess.run(tf.global_variables_initializer())

    save_path = "{}/{}-{}-{}".format(
        args.load_path,
        args.memo,
        args.seed,
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
    )


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    model.train(
        data,
        rollout_size=args.rollout_size,
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
    parser.add_argument("--nb_opt", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--online", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument("--memo", type=str, default=None)

    parser.add_argument("--aux_task", type=str, default="rnd", choices=["rnd", "rf", "idf"])
    parser.add_argument("--aux_dim", type=int, default=512)
    parser.add_argument("--dyna_dim", type=int, default=64)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_path", type=str, default="logs")

    parser.add_argument('--gamestate', default=None)
    parser.add_argument("--alg", type=str, default="ppo2")
    parser.add_argument("--env_type", type=str, default="atari")
    parser.add_argument("--env", type=str)
    parser.add_argument("--num_env", type=int, default=1, choices=[1])

    args = parser.parse_args()

    main(args)
