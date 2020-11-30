import random
from os import environ

import numpy as np
import torch

from .config import ModelType, fetch_model_name
from .utils.logging import get_logger

logger = get_logger(__name__)


CUDA_INDEX = 0
NUM_THREAD = 5
NUM_AGENT = 3
U_DIM = 4
W_DIM = 1
NUM_HISTORY = 15
NUM_ROLLOUT = 25
PRED_TYPE = 'cond'
USE_CONDITION = False
PREDICT_CONDITION = False
LR = 1e-3
BETA = 0.5
GRADIENT_CLIP = 1.0
BSIZE = 200
NUM_WORKERS = 6
SEED = 0
MAX_ITER = 40000
SAVE_EVERY = 1000
TEST_EVERY = 50


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Argument:
    def __init__(self):
        self.CUDA_VISIBLE_DEVICES = CUDA_INDEX
        self.OMP_NUM_THREADS = NUM_THREAD
        self.num_agent = NUM_AGENT
        self.num_history = NUM_HISTORY
        self.num_rollout = NUM_ROLLOUT
        self.pred_type = PRED_TYPE
        self.u_dim = U_DIM
        self.use_condition = USE_CONDITION
        self.predict_condition = PREDICT_CONDITION
        self.model_type = ModelType.GraphNet
        self.bsize = BSIZE
        self.num_workers = NUM_WORKERS
        self.starts_from = -1
        self.custom_index = -1
        self.custom_num_agent = -1
        self.seed = SEED
        self.lr = LR
        self.beta = BETA
        self.gradient_clip = GRADIENT_CLIP
        self.mode = 'train'
        self.max_iter = MAX_ITER
        self.save_every = SAVE_EVERY
        self.test_every = TEST_EVERY
        self.pred_w_dim = 2
        self.pred_g_dim = 4

    @property
    def device(self) -> str:
        return 'cuda:{}'.format(self.CUDA_VISIBLE_DEVICES) if self.CUDA_VISIBLE_DEVICES >= 0 else 'cpu'

    @property
    def model_name(self) -> str:
        names = [fetch_model_name(self.model_type)]
        # if not self.use_condition:
        #     names += ['uncond']
        # if self.predict_condition:
        #     names += ['pred']
        return '-'.join(names)

    @property
    def exp_name(self):
        exp_names = [
            'm-{}'.format(self.model_name),
            # 't-{}'.format(self.pred_type),
            'n-{}'.format(self.num_agent),
            's-{}'.format(self.seed),
            'b-{:4.2f}'.format(self.beta),
        ]
        return '_'.join(exp_names)


_default_args = Argument()


def fetch_argument(modifier=None) -> Argument:
    args = _default_args

    if modifier is not None:
        args = modifier(args)

    if isinstance(args.CUDA_VISIBLE_DEVICES, int) or isinstance(args.CUDA_VISIBLE_DEVICES, str):
        environ['CUDA_VISIBLE_DEVICES'] = str(args.CUDA_VISIBLE_DEVICES)
    elif isinstance(args.CUDA_VISIBLE_DEVICES, list):
        environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(v) for v in args.CUDA_VISIBLE_DEVICES])
    else:
        logger.error('invalid type of CUDA_VISIBLE_DEVICES: {}'.format(type(args.CUDA_VISIBLE_DEVICES)))
        raise TypeError()

    environ['OMP_NUM_THREADS'] = str(args.OMP_NUM_THREADS)
    return args
