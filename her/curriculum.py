import numpy as np
from collections import deque
from baselines import logger
import pickle
import os


def safe_mean(x):
    if len(x) == 0:
        return 0.
    else:
        return np.mean(np.array(x, dtype=float))


class Curriculum:
    ALLOW_STEP = {
        0: 250,
        1: 300,
        2: 350,
        3: 400,
        4: 450,
        5: 500,
        6: 550,
        7: 600,
    }

    BASELINE = 500

    def __init__(self, model, load_path, strategy, desired_x_pos=None, threshold=0.7):
        self.model = model
        self.goal_dim = self.model.achieved_goal_sh[0]
        self._load_goal(load_path)
        self.strategy = strategy
        assert strategy in ['curriculum', 'single']
        if strategy == 'single':
            assert desired_x_pos is not None
            self.desired_x_pos = desired_x_pos
            dist = list(np.abs(self.x_pos - desired_x_pos))
            dist_copy = dist.copy()
            dist_copy.sort()
            self.goal_index_sorted = []
            for d in dist_copy:
                index = dist.index(d)
                self.goal_index_sorted.append(index)
            self.segment = desired_x_pos // self.BASELINE
        else:
            self.achieve_pool = deque(maxlen=10)
            self.pointer = 0
            self.threshold = threshold
            self.maxlen = len(self.buffer)

    def get_current_target(self, nb_goal):
        if self.strategy == 'single':
            index = self.goal_index_sorted[0]
        else:
            index = self.pointer
        data = self.buffer[index]
        desired_goal_state, desired_goal_info = data[0], data[1]
        desired_goal = self._process_goal_shape(desired_goal_info)

        desired_goals = [desired_goal for _ in range(nb_goal)]
        desired_goal_states = [desired_goal_state for _ in range(nb_goal)]
        desired_goal_infos = [desired_goal_info for _ in range(nb_goal)]

        desired_goals = np.asarray(desired_goals)
        desired_goal_states = np.asarray(desired_goal_states)
        if nb_goal == 1:
            return desired_goals[0], desired_goal_states[0], desired_goal_infos[0]
        else:
            return desired_goals, desired_goal_states, desired_goal_infos

    def update(self, succ, acer_steps):
        if self.strategy == 'curriculum':
            assert isinstance(succ, bool)
            self.achieve_pool.append(succ)
            if safe_mean(self.achieve_pool) >= self.threshold and len(self.achieve_pool) == self.achieve_pool.maxlen:
                self.pointer = min(self.pointer+5, self.maxlen)
                self.achieve_pool = deque(maxlen=10)
                self.model.save(os.path.join(logger.get_dir(), "x_pos_{}_{}.pkl".format(self.desired_x_pos, acer_steps)))

    def _load_goal(self, load_path):
        self.buffer = []
        f = open(load_path, "rb")
        data = []
        while True:
            try:
                data.extend(pickle.load(f))
            except Exception as e:
                logger.info(e)
                break
        obs = np.asarray([x["obs"] for x in data])
        self.x_pos = x_pos = np.asarray([x["info"]["x_pos"] for x in data])
        y_pos = np.asarray([x["info"]["y_pos"] for x in data])
        logger.info("loading {} goals".format(len(obs)))
        logger.info("goal_x_pos:{}".format(x_pos))
        for i in range(len(obs)):
            self.buffer.append([obs[i], {"x_pos": x_pos[i], "y_pos": y_pos[i]}])
        self.buffer.sort(key=lambda x: x[-1]["x_pos"])

    def _process_goal_shape(self, goal_info):
        nb_tile = self.goal_dim // 2
        x_pos, y_pos = goal_info["x_pos"], goal_info["y_pos"]
        coordinate = np.asarray([x_pos, y_pos], dtype=np.float32)
        embedding = np.tile(coordinate, nb_tile)
        return embedding

    @property
    def allow_step(self):
        if self.strategy == 'curriculum':
            data = self.buffer[self.pointer]
            desired_goal_info = data[1]
            x_pos = desired_goal_info['x_pos']
            segment = x_pos // self.BASELINE
        else:
            segment = self.segment
        return self.ALLOW_STEP[segment]

    @property
    def current_x_pos(self):
        if self.strategy == 'curriculum':
            data = self.buffer[self.pointer]
            desired_goal_info = data[1]
            x_pos = desired_goal_info['x_pos']
            return x_pos
        else:
            return self.desired_x_pos
