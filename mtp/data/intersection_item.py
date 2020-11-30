import math
from collections import OrderedDict
from typing import List, Dict, Any

import numpy as np

from data.trajectory_item import TrajectoryItem


def _winding_number_from_arrays(a: np.array, b: np.array) -> List[float]:
    zx = a[1:, 0] - b[1:, 0]
    dzx = zx - (a[:-1, 0] - b[:-1, 0])
    zy = a[1:, 1] - b[1:, 1]
    dzy = zy - (a[:-1, 1] - b[:-1, 1])
    theta_new = (np.multiply(dzy, zx) - np.multiply(zy, dzx)) / (np.power(zx, 2) + np.power(zy, 2) + 1.0e-20)
    theta_new = theta_new / (2 * math.pi)
    integral = np.zeros_like(theta_new)
    integral[0] = theta_new[0]
    for i in range(1, integral.shape[0]):
        integral[i] = integral[i-1] + theta_new[i]
    return [0.] + [float(v) for v in integral]


def winding_numbers(t1: TrajectoryItem, t2: TrajectoryItem) -> List[float]:
    min_len = min(len(t1.positions), len(t2.positions))
    a = np.array(t1.positions[:min_len])
    b = np.array(t2.positions[:min_len])
    return _winding_number_from_arrays(a, b)


class IntersectionItem:
    def __init__(self, index: int, trajectories: List[TrajectoryItem]):
        self.index = index
        self.trajectories = trajectories
        self.winding_number_dict = OrderedDict()
        self.compute_winding_number_dict()

    def compute_winding_number_dict(self):
        for i in range(len(self.trajectories)):
            for j in range(len(self.trajectories)):
                if i == j:
                    continue
                self.winding_number_dict[(i, j)] = winding_numbers(self.trajectories[i], self.trajectories[j])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'trajectories': [t.to_dict() for t in self.trajectories],
            # 'winding_number': self.winding_number,
            # 'braid': self.braid,
        }

    def to_str(self) -> str:
        return ':'.join([str(self.index)] + [t.to_str() for t in self.trajectories])

    @staticmethod
    def parse_str(line: str):
        words = line.split(':')
        index = int(words[0])
        trajectories = []
        # extras = []
        for word in words[1:]:
            if ',' in word:
                trajectories.append(TrajectoryItem.parse_str(word))
            # else:
            #     extras.append(word)
        # winding_number = float(extras[0]) if len(extras) > 0 else None
        # braid = int(extras[1]) if len(extras) > 1 else None
        return IntersectionItem(index, trajectories)
