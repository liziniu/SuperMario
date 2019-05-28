import numpy as np
from collections import deque


def goal_to_embedding(goal_info):
    # goal_info: dict(); desired shape: (1, 512)
    feat_dim = 512
    nb_tile = feat_dim // 2
    x_pos, y_pos = goal_info["x_pos"], goal_info["y_pos"]
    goal_embedding = np.array([x_pos, y_pos], dtype=np.float32)[None, :]
    goal_embedding = np.tile(goal_embedding, [1, nb_tile])
    return goal_embedding


def safe_mean(x):
    if len(x) == 0:
        return 0.
    else:
        return np.mean(np.array(x, dtype=float))


class Controller:
    def __init__(self, buffer, strategy, goal_as_image, desired_x_pos=None, threshold=0.7):
        self.buffer = buffer
        self.x_pos = np.array([x[-1]["x_pos"] for x in buffer])
        self.strategy = strategy
        self.goal_as_image = goal_as_image
        assert strategy in ['curriculum', 'single']
        if strategy == 'single':
            assert desired_x_pos is not None
            dist = list(np.abs(self.x_pos - desired_x_pos))
            dist_copy = dist.copy()
            dist_copy.sort()
            self.goal_index_sorted = []
            for d in dist_copy:
                index = dist.index(d)
                self.goal_index_sorted.append(index)
        else:
            self.achieve_pool = deque(maxlen=10)
            self.pointer = 0
            self.threshold = threshold
            self.maxlen = len(self.buffer)

    def get_goal(self, nb_goal):
        if self.strategy == 'single':
            goals, goal_infos = [], []
            for i in range(nb_goal):
                index = self.goal_index_sorted[0]
                data = self.buffer[index]
                goal, goal_info = data[0], data[1]
                goals.append(goal)
                goal_infos.append(goal_info)
        else:
            goals, goal_infos = [], []
            for i in range(nb_goal):
                data = self.buffer[self.pointer]
                goal, goal_info = data[0], data[1]
                goals.append(goal)
                goal_infos.append(goal_info)
        if self.goal_as_image:
            goals = np.asarray(goals, dtype=goal.dtype)
        else:
            assert nb_goal == 1
            goals = goal_to_embedding(goal_infos[0])
        goal_infos = np.asarray(goal_infos, dtype=object)
        return goals, goal_infos

    def update(self, succ):
        if self.strategy == 'curriculum':
            assert isinstance(succ, bool)
            self.achieve_pool.append(succ)
            if safe_mean(self.achieve_pool) >= self.threshold and len(self.achieve_pool) == self.achieve_pool.maxlen:
                self.pointer = min(self.pointer+5, self.maxlen)
                self.achieve_pool = deque(maxlen=10)
