import re
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from .utils.directory import PathManager, mkdir_if_not_exists
from .utils.logging import get_logger

logger = get_logger(__name__)
_path_manager = PathManager()

FRAME_OFFSET = 20  # 20
HISTORY_WINDOW = 15  # 15
TARGET_WINDOW = 25  # 25
POSE_DIMENSION = 2
WINDING_DIMENSION = 1
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VALID_RATIO = 0.1


class PrintableEnum(Enum):
    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)


class ModelType(PrintableEnum):
    GraphNet = 0
    GraphNetFullyConnected = 1
    GRU = 3


class PredictionType(PrintableEnum):
    Trajectory = 0
    WindingNumber = 1


_model_name_from_type = {
    ModelType.GraphNet: 'gn',
    ModelType.GraphNetFullyConnected: 'gn-fc',
    ModelType.GRU: 'gru'
}


_model_type_from_name = {v: k for k, v in _model_name_from_type.items()}


def fetch_model_type(model_str: str) -> ModelType:
    if model_str in _model_type_from_name:
        return _model_type_from_name[model_str]
    else:
        raise TypeError('invalid model name: {}'.format(model_str))


def fetch_model_name(model_type: ModelType) -> str:
    if model_type in _model_name_from_type:
        return _model_name_from_type[model_type]
    else:
        raise TypeError('invalid model type: {}'.format(model_type))


class DatasetDirectory:
    def __init__(self, base_dir: Path = _path_manager.data_dir):
        self.base_dir = base_dir


class DatasetConfig:
    def __init__(
            self,
            base_dir: Path,
            num_agent: int,
            num_rollout: int,
            num_history: int):
        self.base_dir = base_dir
        self.num_agent = num_agent
        self.num_rollout = num_rollout
        self.num_history = num_history

    @property
    def test_index_path(self):
        return _path_manager.fetch_index_path(self.num_agent, 'test')

    @property
    def train_index_path(self):
        return _path_manager.fetch_index_path(self.num_agent, 'train')

    @property
    def test_data_path(self):
        return _path_manager.fetch_data_path(self.num_agent, 'test')

    @property
    def train_data_path(self):
        return _path_manager.fetch_data_path(self.num_agent, 'train')


def generate_encoder_parameter(n_dim, e_dim, u_dim, w_dim, g_dim, inter_size, hidden_size):
    return {
        'n_inc': n_dim,
        'e_inc': e_dim,
        'u_inc': u_dim,
        'w_inc': w_dim,
        'g_inc': g_dim,

        'n_outc': inter_size,
        'e_outc': inter_size,
        'u_outc': inter_size,

        'edge_model_mlp1_hidden_sizes': [hidden_size, hidden_size],
        'node_model_mlp1_hidden_sizes': [hidden_size, hidden_size],
        'node_model_mlp2_hidden_sizes': [hidden_size],
        'global_model_mlp1_hidden_sizes': [hidden_size],
    }


def generate_decoder_parameter(n_dim, e_dim, u_dim, w_dim, g_dim, inter_size, hidden_size):
    return {
        'n_inc': inter_size,
        'e_inc': inter_size,
        'u_inc': inter_size,
        'w_inc': w_dim,
        'g_inc': g_dim,

        'n_outc': n_dim,
        'e_outc': e_dim,
        'u_outc': u_dim,
        'l_outc': inter_size,

        'edge_model_mlp1_hidden_sizes': [hidden_size, hidden_size],
        'node_model_mlp1_hidden_sizes': [hidden_size, hidden_size],
        'node_model_mlp2_hidden_sizes': [hidden_size],
        'global_model_mlp1_hidden_sizes': [hidden_size],

        'node_model_latent_mlp_hidden_size': hidden_size * 2,
        'edge_model_latent_mlp_hidden_size': hidden_size * 2,
    }


def generate_recurrent_parameter(w_dim, g_dim, inter_size):
    return {
        'n_inc': inter_size,
        'e_inc': inter_size,
        'u_inc': inter_size,
        'w_inc': w_dim,
        'g_inc': g_dim,

        'n_outc': inter_size,
        'e_outc': inter_size,
        'u_outc': inter_size,

        'edge_model_mlp1_hidden_sizes': [inter_size, inter_size],
        'node_model_mlp1_hidden_sizes': [inter_size, inter_size],
        'node_model_mlp2_hidden_sizes': [inter_size],
        'global_model_mlp1_hidden_sizes': [inter_size],
    }


class LayerConfig:
    def __init__(
            self,
            n_dim: int,
            e_dim: int,
            u_dim: int,
            g_dim: int,
            w_dim: int,
            pred_w_dim: int,
            pred_g_dim: int,
            winding_inter_size: int,
            winding_hidden_size: int,
            trajectory_inter_size: int,
            trajectory_hidden_size: int,
            num_agent: int,
            device: str):
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.u_dim = u_dim
        self.g_dim = g_dim
        self.w_dim = w_dim
        self.pred_w_dim = pred_w_dim
        self.pred_g_dim = pred_g_dim
        self.winding_inter = winding_inter_size
        self.winding_hidden = winding_hidden_size
        self.trajectory_inter = trajectory_inter_size
        self.trajectory_hidden = trajectory_hidden_size

        self.num_agent = num_agent
        self.device = device

        winding_args = n_dim, e_dim, u_dim, w_dim, g_dim, self.winding_inter, self.winding_hidden
        trajectory_args = n_dim, e_dim, u_dim, w_dim, g_dim, self.trajectory_inter, self.trajectory_hidden
        self.winding_encoder = generate_encoder_parameter(*winding_args)
        self.trajectory_encoder = generate_encoder_parameter(*trajectory_args)
        self.winding_decoder = generate_decoder_parameter(*winding_args)
        self.trajectory_decoder = generate_decoder_parameter(*trajectory_args)
        self.winding_decoder['n_outc'] = self.pred_g_dim
        self.winding_decoder['e_outc'] = self.pred_w_dim
        self.winding_recurrent = generate_recurrent_parameter(w_dim, g_dim, self.winding_inter)
        self.trajectory_recurrent = generate_recurrent_parameter(w_dim, g_dim, self.trajectory_inter)

    @property
    def layer_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            'winding_encoder': self.winding_encoder.copy(),
            'winding_recurrent': self.winding_recurrent.copy(),
            'winding_decoder': self.winding_decoder.copy(),
            'trajectory_encoder': self.trajectory_encoder.copy(),
            'trajectory_recurrent': self.trajectory_recurrent.copy(),
            'trajectory_decoder': self.trajectory_decoder.copy(),
        }


class TrainingConfig:
    def __init__(
            self,
            model_dir: Path,
            train_tb_dir: Path,
            test_tb_dir: Path,
            max_iter: int,
            save_every: int,
            load: bool,
            use_winding: bool,
            lr: float,
            beta: float,
            gradient_clip: float,
            custom_index: int = -1):
        self.lr = lr
        self.iter_collect = 20
        self.max_iter = max_iter
        self.save_every = save_every
        self.model_dir = mkdir_if_not_exists(model_dir)
        self.train_tb_dir = mkdir_if_not_exists(train_tb_dir)
        self.test_tb_dir = mkdir_if_not_exists(test_tb_dir)
        self.flush_secs = 10
        self.load = load
        self.beta = beta
        self.gradient_clip = gradient_clip
        self.use_winding = use_winding
        self.custom_index = custom_index
        self._model_info = None
        self._model_indices = []
        self._loaded_index = -1
        self._fetch_model_files()
        if self.optimizer_file is None or self.model_file is None:
            self.load = False
            logger.info('do not load saved files because it is empty')

    def fetch_indices(self):
        return self._model_indices

    @property
    def loaded_index(self):
        return self._loaded_index

    def _fetch_model_files(self):
        files = sorted(self.model_dir.glob('*.pt'))
        files = map(lambda f: {'path': f, 'info': re.findall(r'([\w]+)_iter([\d]+).pt', str(f))}, files)
        files = filter(lambda x: x['info'], files)
        files = map(lambda x: {'path': x['path'], 'info': x['info'][0]}, files)
        files = map(lambda x: {'path': x['path'], 'words': x['info'][0].split('_'), 'iter': int(x['info'][1])}, files)
        model_path_dict = defaultdict(dict)
        for f in list(files):
            key = 'trainer' if len(f['words']) > 1 else 'model'
            model_path_dict[f['iter']][key] = f['path']

        if not list(model_path_dict.keys()):
            return

        self._model_indices = sorted(model_path_dict.keys())
        use_custom = self.custom_index >= 0 and self.custom_index in self._model_indices
        self._loaded_index = self.custom_index if use_custom else self._model_indices[-1]
        self._model_info = model_path_dict[self._loaded_index]

    @property
    def optimizer_file(self):
        return None if self._model_info is None else self._model_info['trainer']

    @property
    def model_file(self):
        return None if self._model_info is None else self._model_info['model']


def get_config_list(args):
    dataset_dir = mkdir_if_not_exists(Path.cwd() / '.gnn/data')
    model_root_dir = mkdir_if_not_exists(Path.cwd() / '.gnn/weights')
    log_root_dir = mkdir_if_not_exists(Path.cwd() / '.gnn/logs')
    model_dir = model_root_dir / args.exp_name
    train_log_dir = log_root_dir / args.exp_name / 'train'
    test_log_dir = log_root_dir / args.exp_name / 'test'

    eff_num_agent = args.custom_num_agent if args.custom_num_agent >= 0 else args.num_agent
    dc = DatasetConfig(
        base_dir=dataset_dir,
        num_agent=eff_num_agent,
        num_rollout=args.num_rollout,
        num_history=args.num_history)

    lc = LayerConfig(
        n_dim=4,
        e_dim=2,
        u_dim=args.u_dim,
        g_dim=4,
        w_dim=2,
        pred_w_dim=args.pred_w_dim,
        pred_g_dim=args.pred_g_dim,
        winding_inter_size=20,
        winding_hidden_size=30,
        trajectory_inter_size=20,#30,
        trajectory_hidden_size=30,#50,
        num_agent=args.num_agent,
        device=args.device)

    tc = TrainingConfig(
        model_dir=model_dir,
        train_tb_dir=train_log_dir,
        test_tb_dir=test_log_dir,
        max_iter=args.max_iter,
        save_every=args.save_every,
        load=True,
        use_winding=args.use_condition,
        lr=args.lr,
        beta=args.beta,
        gradient_clip=args.gradient_clip,
        custom_index=args.custom_index)
    return dc, lc, tc, model_dir
