import json
import random
from itertools import chain
from operator import itemgetter
from typing import Dict, List, Any, Union

import torch
from torch import Tensor

from config import HISTORY_WINDOW, TARGET_WINDOW, FRAME_OFFSET, TRAIN_RATIO, TEST_RATIO
from data.intersection_item import IntersectionItem
from data.trajectory_reader import TrajectoryReader
from utils.directory import PathManager

_path_manager = PathManager()


def partition_by_mode(mode: str, indices: List[int]) -> List[int]:
    train_ratio = TRAIN_RATIO
    test_ratio = TEST_RATIO
    if mode == 'train':
        index1 = 0
        index2 = int(round(len(indices) * train_ratio))
    elif mode == 'test':
        index1 = int(round(len(indices) * train_ratio))
        index2 = int(round(len(indices) * (train_ratio + test_ratio)))
    elif mode == 'valid':
        index1 = int(round(len(indices) * (train_ratio + test_ratio)))
        index2 = len(indices)
    else:
        raise ValueError('invalid mode was given: {}'.format(mode))
    return indices[index1:index2]


class DataGenerator:
    def __init__(self, num_agent: int, dim_pos: int = 2, dim_goal: int = 1):
        self.num_agent = num_agent
        self.modes = ['train', 'test']
        self.dim_pos = dim_pos
        self.dim_goal = dim_goal
        self.step = 3
        self.train_ratio = 0.9

    def fetch_data_path(self, mode: str):
        return _path_manager.fetch_data_path(num_agent=self.num_agent, mode=mode)

    def fetch_index_path(self, mode: str):
        return _path_manager.fetch_index_path(num_agent=self.num_agent, mode=mode)

    def tensor_from_intersection_item(
            self,
            item: IntersectionItem,
            offset: int = FRAME_OFFSET) -> Union[None, Dict[str, Any]]:
        """
        converts an intersection item into dictionary of str keys and tensor values
        @param item: a single IntersectionItem
        @param offset: the number of frames to ignore
        @return: dictionary of key and tensor values
        """
        try:
            N = len(item.trajectories)
            keys = sorted(list(item.winding_number_dict.keys()))
            E = len(keys)
            s = torch.tensor([t.intention[0] for t in item.trajectories]).view(1, N)  # src index
            g = torch.tensor([t.intention[1] for t in item.trajectories]).view(1, N)  # dst index
            wgl = [(key, item.winding_number_dict[key][offset:]) for key in keys]
            wtl = torch.tensor([wl[1][-1] for wl in wgl]).view(1, E)  # 1, num_windings
            l = min([len(v) for v in list(map(itemgetter(1), wgl))])  # set the minimum length among trajectories
            w = torch.stack([torch.tensor(wl[1][:l]) for wl in wgl]).transpose(0, 1)  # num_frame, num_winding
            S = w.shape[0]  # sequence length or the number of frame
            t = torch.cat([torch.tensor(item.trajectories[i].positions)[offset:S + offset, :]
                           for i in range(N)], dim=1)  # num_frame, num_agent * 2
        except:
            return None
        return {
            'trajectory': t,  # S, 2N
            'winding_number': w,  # S, E
            'winding_number_target': wtl,  # 1, E
            'src': s,  # 1, N
            'dst': g,  # 1, N
        }

    def split_into_sub_data(self, raw_data: Dict[str, Any], step: int) -> List[Dict[str, Any]]:
        trajectory = raw_data['trajectory']  # F, NT
        velocity = trajectory[1:, :] - trajectory[:-1, :]  # F-1, NT
        winding_number = raw_data['winding_number']  # F, E
        winding_number_target = raw_data['winding_number_target']  # 1, E
        src = raw_data['src']  # 1, N
        dst = raw_data['dst']  # 1, N
        num_frame = velocity.shape[0]
        window_size = HISTORY_WINDOW + TARGET_WINDOW
        sub_data_list = []
        for i in range(0, num_frame - window_size, step):
            sub_trajectory = trajectory[i + 1:i + 1 + window_size, :]  # W, 2N
            sub_velocity = velocity[i:i + window_size, :]  # W, 2N
            sub_winding_number = winding_number[i + 1:i + 1 + window_size, :]  # W, E
            sub_data_list.append((sub_trajectory, sub_velocity, sub_winding_number))
        return [{
            'trajectory': t,
            'velocity': v,
            'winding_number': w,
            'winding_number_target': winding_number_target,
            'src': src,
            'dst': dst,
        } for t, v, w in sub_data_list]

    def fetch_graph_data(self, data) -> Dict[str, Tensor]:
        data_traj = data['trajectory']  # N, S, P * A
        data_vel = data['velocity']  # N, S, P * A
        data_src = data['src']  # N, A
        data_dst = data['dst']  # N, A
        data_winding = data['winding_number_target']  # N, E

        dim_node = self.dim_pos + self.dim_pos  # traj, vel, dest
        node_data = torch.zeros((*data_traj.shape[:2], self.num_agent, dim_node), dtype=torch.float32)  # N, S, A, 2 * P
        edge_data = data_winding  # N, E
        for a in range(self.num_agent):
            traj = data_traj[..., self.dim_pos * a:self.dim_pos * (a + 1)]
            vel = data_vel[..., self.dim_pos * a:self.dim_pos * (a + 1)]
            ad = torch.cat([traj, vel], dim=-1)
            node_data[:, :, a, :] = ad
        return {
            'node': node_data,
            'edge': edge_data,
            'src': data_src,
            'dst': data_dst,
        }

    def split_data(self) -> Dict[str, Dict[str, Any]]:
        split_dict = {mode: dict() for mode in self.modes}

        reader = TrajectoryReader(self.num_agent)
        intersection_items = reader.read()

        raw_dict_list = list(map(self.tensor_from_intersection_item, intersection_items))  # List[Dict[str, Any]]
        raw_dict_list = list(filter(lambda x: x is not None, raw_dict_list))

        indices = list(range(len(raw_dict_list)))
        random.shuffle(indices)
        num_train_elem = int(round(self.train_ratio * len(indices)))
        split_dict['train']['index'] = indices[:num_train_elem]
        split_dict['test']['index'] = indices[num_train_elem:]

        for mode in self.modes:
            dgl = [self.split_into_sub_data(raw_dict_list[i], step=self.step) for i in split_dict[mode]['index']]
            dict_list = list(chain.from_iterable(dgl))
            elem = dict_list[0]
            split_dict[mode]['data'] = dict()
            for key, value in elem.items():
                if isinstance(value, torch.Tensor):
                    split_dict[mode]['data'][key] = torch.stack([d[key] for d in dict_list])
                else:
                    split_dict[mode]['data'][key] = torch.tensor([d[key] for d in dict_list])
        return split_dict

    def generate_dataset_from_split_data(self, split_data: Dict[str, Dict[str, Tensor]]):
        data_dict = dict()
        for mode in self.modes:
            raw_data = split_data[mode]['data']
            graph_data = self.fetch_graph_data(raw_data)
            data_dict[mode] = graph_data
            data_path = self.fetch_data_path(mode)
            index_path = self.fetch_index_path(mode)
            with open(str(index_path), 'w') as file:
                json.dump(split_data[mode]['index'], file)
            torch.save(graph_data, str(data_path))
        return data_dict

    def generate_data(self) -> Dict[str, Dict[str, Dict[str, Tensor]]]:
        data_exists = all(self.fetch_data_path(mode).exists() for mode in self.modes)
        if data_exists:
            data = {mode: torch.load(str(self.fetch_data_path(mode))) for mode in self.modes}
        else:
            split_dict = self.split_data()
            data = self.generate_dataset_from_split_data(split_dict)
        return data


if __name__ == '__main__':
    for i in (2, 3, 4):
        dc = DataGenerator(i)
        dc.generate_data()
