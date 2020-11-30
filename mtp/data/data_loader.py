from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from ..config import DatasetConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class VehicleTrajectoryDataset(Dataset):
    def __init__(self, config: DatasetConfig, test=False):
        self.config = config

        data_path = config.test_data_path if test else config.train_data_path
        data = torch.load(str(data_path))
        self.node_data = data['node']  # N, T, n, 4
        self.edge_data = data['edge']  # N, E
        self.dst_data = data['dst']  # N, n
        self.src_data = data['src']  # N, n

        self.N = self.node_data.shape[0]
        assert self.node_data.shape[0] == self.edge_data.shape[0]
        if self.edge_data.ndim == 1:
            self.edge_data.unsqueeze_(1)
        """ Order looks like [[0,0],
                              [0,1],
                              ...
                             ]
        """

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        i, j, k = idx, self.config.num_history, self.config.num_rollout

        curr_state = self.node_data[i, :j].permute(1, 0, 2).contiguous()  # Shape: [n x T x 4]
        next_state = self.node_data[i, j:j + k].permute(1, 0, 2).contiguous()  # Shape: [n x rollout_num x 4]

        dst = self.dst_data[i, :]  # n
        src = self.src_data[i, :]
        dst_zeros = torch.zeros_like(dst)
        dst_ones = torch.ones_like(dst)
        dst_onehot = torch.stack([torch.where(dst == i, dst_ones, dst_zeros) for i in range(4)], dim=-1).float()
        # src_onehot = torch.stack([torch.where(src == i, dst_ones, dst_zeros) for i in range(4)], dim=-1).float()

        target_winding = self.edge_data[i, :]  # E
        winding_zeros = torch.zeros_like(target_winding)
        winding_ones = torch.ones_like(target_winding)
        cond1 = torch.where(target_winding < 0, winding_ones, winding_zeros)
        cond2 = torch.where(target_winding >= 0, winding_ones, winding_zeros)
        winding_onehot = torch.stack([cond1, cond2], dim=-1).float()

        sample = {
            'current_state': curr_state,  # n, T, 4
            'next_state': next_state,  # n, r, 4
            'target_winding': target_winding,  # E
            'winding_index': cond2.to(dtype=torch.long),  # E
            'winding': winding_onehot,  # E, 2
            'src_index': src,  # n
            'dst_index': dst,  # n
            'dst': dst_onehot,  # n, 4
        }
        return sample

    def get_seq(self, seq_num) -> Tuple[Tensor, Tensor]:
        """ Hack to get an entire sequence
        """
        j = self.config.num_history
        curr_state = self.node_data[seq_num].permute(1, 0, 2).contiguous()  # n, T, 4
        return curr_state, self.edge_data[seq_num]


def get_trajectory_data_loader(config, test=False, batch_size=8, num_workers=4, shuffle=True):
    config = deepcopy(config)
    dataset = None #VehicleTrajectoryDataset(config, test=test)
    return None 
# DataLoader(dataset=dataset,
#                       batch_size=batch_size,
#                       shuffle=shuffle,
#                       num_workers=num_workers,
#                       worker_init_fn=worker_init_fn)
