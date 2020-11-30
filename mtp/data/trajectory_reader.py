import re
from typing import List, Dict, Any, Union

import pandas as pd
from data.intersection_item import IntersectionItem
from data.trajectory_item import TrajectoryItem
from tqdm import tqdm
from utils.directory import PathManager

_path_manager = PathManager()


def _clean_up_raw_text(line: str) -> List[str]:
    line = line.replace('\n', ' ')
    line = line.replace('\"', ':')
    line = re.sub(r'(\[|\]|\(|\)|,)', '', line)
    line = re.sub(r"([' ']+)", ' ', line)
    words = line.split(':')
    words = list(filter(lambda x: x, map(lambda x: x.strip(), words)))
    return words


class TrajectoryFileHeader:
    """
    CSVFileHeader parse the first line of the csv file as a header,
    and parse lines based on the csv structure and returns an IntersectionItem instance
    """
    def __init__(self, first_line: str):
        self.first_line = first_line
        self.intention_indices = []
        self.position_indices = []
        self.data_list = self._parse_first_line()

    def __len__(self):
        return len(self.data_list)

    @property
    def num_agents(self) -> int:
        return len(self.intention_indices)

    def _parse_first_line(self) -> List[Dict[str, Any]]:
        keys = self.first_line.replace('\"', '').split(',')
        data_list = []
        for i, key in enumerate(keys):
            if key.startswith('intention'):
                vtype = tuple
                dtype = int
                self.intention_indices.append(i)
            elif key.endswith('data'):
                vtype = list
                dtype = float
                self.position_indices.append(i)
            elif key.startswith('w'):
                vtype = dtype = float
            elif 'jointpath' in key:
                vtype = dtype = str
            else:
                vtype = dtype = int
            data_list.append({'key': key, 'vtype': vtype, 'dtype': dtype})
        return data_list

    def parse_data(self, items: List[str]) -> IntersectionItem:
        assert len(self.data_list) == len(items)
        new_items = []
        for data_info, item in zip(self.data_list, items):
            if data_info['vtype'] in [tuple, list]:
                new_item = [data_info['dtype'](v) for v in item.split(' ')]
            else:
                new_item = data_info['dtype'](item)
            new_items.append(new_item)

        index = new_items[0]
        intentions = [new_items[i] for i in self.intention_indices]
        positions = [new_items[i] for i in self.position_indices]
        trajectories = [
            TrajectoryItem(
                intention=intentions[i],
                positions=list(zip(positions[i], positions[i+self.num_agents])))
            for i in range(self.num_agents)]
        return IntersectionItem(index, trajectories, *new_items[3*self.num_agents + 1:])


def _parse_float_list(src: str):
    return list(map(float, src.replace('[', '').replace(']', '').split()))


def _parse_single_item(num_agent, data_index_dict, arr, index) -> Union[IntersectionItem, None]:
    try:
        raw_item = arr[index, :]
        items = []
        for a in range(num_agent):
            intention_str, x_str, y_str = raw_item[data_index_dict[a]]
            intention = tuple(map(int, re.findall(r'([\d]+)', intention_str)))
            xl = _parse_float_list(x_str)
            yl = _parse_float_list(y_str)
            assert len(xl) == len(yl)
            items.append(TrajectoryItem(intention, list(zip(xl, yl))))
        return IntersectionItem(-1, items)
    except:
        return None


class TrajectoryReader:
    def __init__(self, num_agent: int):
        self.num_agent = num_agent
        self.raw_file_path, self.refined_file_path = _path_manager.fetch_traj_path_pair(num_agent)
        self.header = None
        self.data_index_dict = {i: [i + 1 + j * num_agent for j in range(3)] for i in range(num_agent)}

    @property
    def word_per_line(self) -> int:
        return 0 if self.header is None else len(self.header)

    def read(self) -> List[IntersectionItem]:
        return self._read_refined_file() if self.refined_file_path.exists() else self._read_raw_file()

    def _read_raw_file(self) -> List[IntersectionItem]:
        assert self.raw_file_path.exists()

        raw_data = pd.read_csv(str(self.raw_file_path))
        arr = raw_data.to_numpy()
        items = []
        for i in tqdm(range(arr.shape[0])):
            item = _parse_single_item(self.num_agent, self.data_index_dict, arr, i)
            if item is not None:
                items.append(item)
        strs = []
        for i, item in enumerate(items):
            item.index = i
            strs.append(item.to_str())
        with open(str(self.refined_file_path), 'w') as file:
            file.write('\n'.join(strs))
        return items

    def _read_refined_file(self) -> List[IntersectionItem]:
        assert self.refined_file_path.exists()
        with open(str(self.refined_file_path), 'r') as file:
            lines = file.read().splitlines()
        return [IntersectionItem.parse_str(line) for line in lines]
