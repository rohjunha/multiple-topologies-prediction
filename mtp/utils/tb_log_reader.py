"""
Modified from https://gist.github.com/tomrunia/1e1d383fb21841e8f144
"""
import math
import re
from collections import namedtuple, defaultdict
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import TARGET_WINDOW
from utils.logging import get_logger

Scalar = namedtuple('Scalar', ['timestamp', 'step', 'value'])

logger = get_logger(__name__)


def collect_data_from_tb_log(path: Path, keyword: str) -> List[Scalar]:
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'histograms': 1
    }
    event_acc = EventAccumulator(str(path), tf_size_guidance)
    event_acc.Reload()
    # print(event_acc.Tags()['scalars'])
    if keyword not in event_acc.Tags()['scalars']:
        return []
    training_accuracies = event_acc.Scalars(keyword)
    return list(map(lambda x: Scalar(*x), training_accuracies))


def plot_tb_log(values: List[Scalar], log: bool):
    steps = len(values)
    x = np.arange(steps)
    y = np.array(list(map(itemgetter(2), values)))
    if log:
        y = np.log10(y)

    label = 'training accuracy'
    if log:
        label += ' (log-scale)'
    plt.plot(x, y, label=label)

    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.title("training loss")
    plt.legend()
    plt.show()


def extract_exp_info_from_path(log_dir: Path) -> Tuple[str, int, int, str]:
    mode = log_dir.stem
    words = log_dir.parent.stem.split('_')
    if len(words) != 3:
        raise ValueError('number of words should be three: {}'.format(len(words)))

    name = '-'.join(words[0].split('-')[1:])
    num_agent = int(words[1].split('-')[1])
    seed = int(words[2].split('-')[1])
    return name, num_agent, seed, mode


def collect_log_dirs(root_dir: Path):
    log_dirs = []
    candidates = root_dir.glob('m-*')
    # candidates = root_dir.glob('gru*')
    for cand in candidates:
        if not cand.is_dir():
            continue
        train_dir = cand / 'train'
        test_dir = cand / 'test'
        if train_dir.exists():
            log_dirs.append(train_dir)
        if test_dir.exists():
            log_dirs.append(test_dir)
    return log_dirs


def collect_all_files(log_dir: Path):
    log_files = sorted(list(log_dir.glob('events.out.tfevents.*')))
    tag = 'loss/total'
    values = list(chain.from_iterable([collect_data_from_tb_log(log_file, tag) for log_file in log_files]))
    values = sorted(values, key=lambda x: x.timestamp)
    return values
