from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import permutations
from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from .argument import Argument
from .config import LayerConfig, ModelType
from .models.graph_net import WindingGraphNet, TrajectoryGraphNet
from .utils.logging import get_logger

dont_send_to_device = []

logger = get_logger(__name__)


class NetworkWrapper(ABC):
    def __init__(self, config: LayerConfig):
        self.config = deepcopy(config)
        self.model = None
        self.setup()

    @property
    def device(self):
        return self.config.device

    @property
    def u_dim(self):
        return self.config.u_dim

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    def train_mode(self):
        """ Put all modules into train mode
        """
        self.model.train()

    def eval_mode(self):
        """ Put all modules into eval mode
        """
        self.model.eval()

    def send_batch_to_device(self, batch):
        for key in batch.keys():
            if key in dont_send_to_device:
                continue
            if len(batch[key]) == 0:  # can happen if a modality (e.g. RGB) is not loaded
                continue
            batch[key] = batch[key].to(self.device)

    def save(self, filename):
        """ Save the model as a checkpoint
        """
        checkpoint = {'model': self.model.state_dict()}
        torch.save(checkpoint, filename)

    def load(self, filename):
        """ Load the model checkpoint
        """
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model'])

    def construct_graph(self, x):
        raise NotImplementedError

    def construct_hidden_graph(self, bsize: int, num_agent: int, hidden_size: int):
        raise NotImplementedError

    # @abstractmethod
    # def forward(self, curr_state, next_state, winding_onehot, goal_onehot, tar_winding, tar_goal, criterion_dict):
    #     pass


class GraphNetWrapper(NetworkWrapper):
    def setup(self):
        """ Setup model, losses, optimizers, misc
        """
        self.trajectory_model = TrajectoryGraphNet(self.config)
        self.winding_model = WindingGraphNet(self.config)
        self.trajectory_model.to(self.device)
        self.winding_model.to(self.device)

    def train_mode(self):
        self.trajectory_model.train()
        self.winding_model.train()

    def eval_mode(self):
        self.trajectory_model.eval()
        self.winding_model.eval()

    def construct_graph(self, x):
        """ Construct Graph from Billiards data

            Node features are simply states
            Edge connections are fully connected
            Edge features are center offsets
            Global attribute u starts at 0

            @param x: a [B x n x d] torch tensor.
                      Batch size B, sequence length T, n objects, each row is [x,y,dx,dy]

            @return: a Data instance with Data.x.shape = [Bn x Td]
        """
        if x.dim() != 3:
            if x.dim() == 4:
                x = x.squeeze()
            elif x.dim() == 2:
                x = x.unsqueeze(0)
        assert x.dim() == 3

        B, n, d = x.shape
        # logger.info('x.shape: {}'.format(x.shape))

        # Compute edge connections
        edge_index = torch.tensor(list(permutations(range(n), 2)), dtype=torch.long)
        edge_index = edge_index.t().contiguous()  # Shape: [2 x E], E = n * (n - 1)

        # Compute edge features for all graphs
        src, dest = edge_index
        edge_attrs = x[:, dest, :2] - x[:, src, :2]  # Shape: [B x E x T x 2]
        e = edge_attrs.shape[1]

        # U vector. |U|-dimensional 0-vector
        u = torch.zeros((1, self.u_dim), dtype=torch.float, device=x.device)

        # Create list of Data objects, then call Batch.from_data_list()
        data_objs = [Data(x=x[b].view(n, -1),
                          edge_index=edge_index,
                          edge_attr=edge_attrs[b].view(e, -1),
                          u=u.clone())
                     for b in range(B)]
        batch = Batch.from_data_list(data_objs).to(x.device)

        return batch

    def construct_hidden_graph(self, bsize: int, num_agent: int, hidden_size: int):
        # Compute edge connections
        edge_index = torch.tensor(list(permutations(range(num_agent), 2)), dtype=torch.long)
        edge_index = edge_index.t().contiguous()  # Shape: [2 x E], E = n^2
        e = edge_index.shape[1]

        # U vector. |U|-dimensional 0-vector
        x = torch.zeros((num_agent, hidden_size), dtype=torch.float32, device=self.device)
        u = torch.zeros((1, hidden_size), dtype=torch.float32, device=self.device)
        edge_attr = torch.zeros((e, hidden_size), dtype=torch.float32, device=self.device)

        # Create list of Data objects, then call Batch.from_data_list()
        data_objs = [Data(x=x.clone(),
                          edge_index=edge_index,
                          edge_attr=edge_attr.clone(),
                          u=u.clone())
                     for _ in range(bsize)]
        batch = Batch.from_data_list(data_objs).to(x.device)
        return batch

    def train_trajectory(self, curr_state, next_state, winding_onehot, goal_onehot, criterion_dict):
        T = curr_state.shape[2]
        tI = self.config.trajectory_inter
        B, n, rn, d = next_state.shape

        prd_traj = torch.zeros_like(next_state)
        node_loss = torch.tensor(0., dtype=torch.float, device=self.device)
        goal_loss = torch.tensor(0., dtype=torch.float, device=self.device)
        winding_loss = torch.tensor(0., dtype=torch.float, device=self.device)
        hidden_trajectory = self.construct_hidden_graph(B, n, tI)
        for i in range(T):
            curr_graph = self.construct_graph(curr_state[:, :, i, :])
            _, hidden_trajectory = \
                self.trajectory_model(curr_graph, hidden_trajectory, winding_onehot, goal_onehot)

        for i in range(rn):
            curr_graph, hidden_trajectory = \
                self.trajectory_model(curr_graph, hidden_trajectory, winding_onehot, goal_onehot)

            prd_node = curr_graph.x  # B, n, d
            tar_node = next_state[:, :, i, :].view(-1, d)
            prd_traj[:, :, i, :] = curr_graph.x.view(B, n, d)

            node_loss += criterion_dict['node'](prd_node, tar_node)

        loss = node_loss + goal_loss + winding_loss
        loss = loss / rn
        loss_dict = {
            'total_loss': loss,
            'node_loss': node_loss,
            'goal_loss': goal_loss,
            'winding_loss': winding_loss,
        }
        return B, loss_dict, prd_traj

    def train_winding(self, curr_state, next_state, tar_winding, tar_goal, criterion_dict):
        T = curr_state.shape[2]
        wI = self.config.winding_inter
        B, n, rn, d = next_state.shape

        node_loss = torch.tensor(0., dtype=torch.float, device=self.device)
        goal_loss = torch.tensor(0., dtype=torch.float, device=self.device)
        winding_loss = torch.tensor(0., dtype=torch.float, device=self.device)
        hidden_winding = self.construct_hidden_graph(B, n, wI)
        for i in range(T):
            curr_graph = self.construct_graph(curr_state[:, :, i, :])
            hidden_winding, winding, goal, ze_mu, ze_var, zn_mu, zn_var = self.winding_model(curr_graph, hidden_winding)

        for i in range(rn):
            curr_graph = self.construct_graph(next_state[:, :, i, :])
            hidden_winding, prd_winding, prd_goal, ze_mu, ze_var, zn_mu, zn_var = \
                self.winding_model(curr_graph, hidden_winding)

            winding_loss += criterion_dict['winding'](prd_winding, tar_winding.view(-1, ), ze_mu, ze_var)
            goal_loss += criterion_dict['goal'](prd_goal, tar_goal.view(-1, ), zn_mu, zn_var)

        loss = node_loss + goal_loss + winding_loss
        loss = loss / rn
        loss_dict = {
            'total_loss': loss,
            'node_loss': node_loss,
            'goal_loss': goal_loss,
            'winding_loss': winding_loss,
        }
        return B, loss_dict, prd_winding, prd_goal

    def eval_trajectory(self, curr_state, num_rollout, winding_onehot, goal_onehot):
        B, n, T, d = curr_state.shape
        tI = self.config.trajectory_inter

        prd_traj = torch.zeros((B, n, num_rollout, d), dtype=curr_state.dtype)
        hidden_trajectory = self.construct_hidden_graph(B, n, tI)
        for i in range(T):
            curr_graph = self.construct_graph(curr_state[:, :, i, :])
            _, hidden_trajectory = \
                self.trajectory_model(curr_graph, hidden_trajectory, winding_onehot, goal_onehot)
        for i in range(num_rollout):
            curr_graph, hidden_trajectory = \
                self.trajectory_model(curr_graph, hidden_trajectory, winding_onehot, goal_onehot)
            prd_traj[:, :, i, :] = curr_graph.x.view(B, n, d)
        return B, prd_traj

    def eval_winding(self, curr_state, w_dim, g_dim):
        B, n, T, d = curr_state.shape
        wI = self.config.winding_inter

        hidden_winding = self.construct_hidden_graph(B, n, wI)
        for i in range(T):
            curr_graph = self.construct_graph(curr_state[:, :, i, :])
            hidden_winding, prd_winding, prd_goal, ze_mu, ze_var, zn_mu, zn_var = \
                self.winding_model(curr_graph, hidden_winding)
        latent_dict = {
            'node_mu': zn_mu,
            'node_var': zn_var,
            'edge_mu': ze_mu,
            'edge_var': ze_var,
        }

        pl = []
        for i in range(100):
            w_, g_ = self.fetch_node_and_edge_from_latent_dict(latent_dict)
            w_ = torch.argmax(w_, dim=1).view(w_dim, -1)
            g_ = torch.argmax(g_, dim=1).view(g_dim, -1)
            pl.append(torch.cat((g_, w_), dim=-1))
        pl = torch.stack(pl, dim=0)  # N, num_item, dim

        output = []
        for i in range(pl.shape[1]):
            rows = pl[:, i, :]
            unique_rows = torch.unique(rows, dim=0)
            freq = [len(np.where((rows == unique_rows[r]).all(axis=1))[0]) for r in range(unique_rows.shape[0])]
            freq_rows = sorted(zip(freq, unique_rows), key=lambda x: x[0], reverse=True)
            output.append(freq_rows)
        return output

    def train(self, curr_state, next_state, winding_onehot, goal_onehot, tar_winding, tar_goal, criterion_dict):
        T = curr_state.shape[2]
        wI = self.config.winding_inter
        tI = self.config.trajectory_inter
        B, n, rn, d = next_state.shape

        prd_traj = torch.zeros_like(next_state)
        node_loss = torch.tensor(0., dtype=torch.float, device=self.device)
        goal_loss = torch.tensor(0., dtype=torch.float, device=self.device)
        winding_loss = torch.tensor(0., dtype=torch.float, device=self.device)
        hidden_winding = self.construct_hidden_graph(B, n, wI)
        hidden_trajectory = self.construct_hidden_graph(B, n, tI)
        for i in range(T):
            curr_graph = self.construct_graph(curr_state[:, :, i, :])
            hidden_winding, prd_winding, prd_goal, ze_mu, ze_var, zn_mu, zn_var = \
                self.winding_model(curr_graph, hidden_winding)
            _, hidden_trajectory = \
                self.trajectory_model(curr_graph, hidden_trajectory, winding_onehot, goal_onehot)

        for i in range(rn):
            # todo: transform into onehot from prd
            prd_winding_onehot = prd_winding
            prd_goal_onehot = prd_goal

            curr_graph, hidden_trajectory = \
                self.trajectory_model(curr_graph, hidden_trajectory, winding_onehot, goal_onehot)
            hidden_winding, prd_winding, prd_goal, ze_mu, ze_var, zn_mu, zn_var = \
                self.winding_model(curr_graph, hidden_winding)

            prd_node = curr_graph.x  # B, n, d
            tar_node = next_state[:, :, i, :].view(-1, d)
            prd_traj[:, :, i, :] = curr_graph.x.view(B, n, d)

            winding_loss += criterion_dict['winding'](prd_winding, tar_winding.view(-1, ), ze_mu, ze_var)
            goal_loss += criterion_dict['goal'](prd_goal, tar_goal.view(-1, ), zn_mu, zn_var)

        loss = node_loss + goal_loss + winding_loss
        loss = loss / rn
        loss_dict = {
            'total_loss': loss,
            'node_loss': node_loss,
            'goal_loss': goal_loss,
            'winding_loss': winding_loss,
        }
        latent_dict = {
            'node_mu': zn_mu,
            'node_var': zn_var,
            'edge_mu': ze_mu,
            'edge_var': ze_var,
        }
        return B, loss_dict, prd_traj, prd_winding, prd_goal, latent_dict

    def fetch_node_and_edge_from_latent_dict(self, latent_dict: Dict[str, torch.Tensor]):
        return self.winding_model.fetch_node_and_edge_from_latent_dict(latent_dict)


# class GraphNetNoRecurrencyWrapper(NetworkWrapper):
#     def setup(self):
#         """ Setup model, losses, optimizers, misc
#         """
#         # Whole model, for nn.DataParallel
#         self.model = GraphNetNoRecurrency(self.config)
#         self.model.to(self.device)
#
#     def run_on_batch(self, batch, rn: int):
#         """ Run algorithm on batch of images in eval mode
#
#             @param batch: A Python dictionary with keys:
#                             'current_state' : a [B x n x d] torch tensor
#                             'next_state' : a [B x n x d] torch tensor
#         """
#         self.eval_mode()
#         self.send_batch_to_device(batch)
#         curr_state = batch['current_state']
#         assert curr_state.ndim == 3
#         n, T, d = curr_state.shape
#         curr_state = curr_state.unsqueeze(0)  # B, n, T, d
#
#         out = torch.zeros((rn, n, d), dtype=torch.float32, device=curr_state.device)
#         with torch.no_grad():
#             for i in range(T):
#                 curr_data = curr_state[:, :, i, :].squeeze().unsqueeze(0)  # 1, n, d
#                 curr_graph = self.construct_graph(curr_data)
#                 curr_graph, _ = self.model(curr_graph)
#
#             for i in range(rn):
#                 curr_graph, _ = self.model(curr_graph)
#                 out[i] = curr_graph.x.view(n, d)
#         return out
#
#     def rollout(self, s0, T):
#         """ Run sequence starting from state 0 (s0) for T steps
#
#             @param x: s0 [n x pT x d] torch tensor.
#             @param T: Number of time steps to rollout
#         """
#         return self.run_on_batch({'current_state': s0}, T)
#
#     def construct_graph(self, x):
#         """ Construct Graph from Billiards data
#
#             Node features are simply states
#             Edge connections are fully connected
#             Edge features are center offsets
#             Global attribute u starts at 0
#
#             @param x: a [B, n, T, d] torch tensor.
#                       Batch size B, sequence length T, n objects, each row is [x,y,dx,dy]
#
#             @return: a Data instance with Data.x.shape = [Bn x Td]
#         """
#         if x.dim() != 4:
#             if x.dim() == 5:
#                 x = x.squeeze()
#             elif x.dim() == 3:
#                 x = x.unsqueeze(0)
#         assert x.dim() == 4
#
#         B, n, T, d = x.shape
#
#         # Compute edge connections
#         edge_index = torch.tensor(list(permutations(range(n), 2)), dtype=torch.long)
#         edge_index = edge_index.t().contiguous()  # Shape: [2 x E], E = n * (n - 1)
#
#         # Compute edge features for all graphs
#         src, dest = edge_index
#         edge_attrs = x[:, dest, :, :2] - x[:, src, :, :2]  # Shape: [B x E x T x 2]
#         e = edge_attrs.shape[1]
#
#         # U vector. |U|-dimensional 0-vector
#         u = torch.zeros((1, self.u_dim * T), dtype=torch.float, device=x.device)
#
#         # Create list of Data objects, then call Batch.from_data_list()
#         data_objs = [Data(x=x[b].view(n, -1),
#                           edge_index=edge_index,
#                           edge_attr=edge_attrs[b].view(e, -1),
#                           u=u.clone())
#                      for b in range(B)]
#         batch = Batch.from_data_list(data_objs).to(x.device)
#
#         return batch
#
#     def forward(self, curr_state, next_state, winding, dst_tensor, criterion_dict):
#         T = curr_state.shape[2]
#         B, n, rn, d = next_state.shape
#
#         prd_traj = torch.zeros_like(next_state)
#         node_loss = torch.tensor(0., dtype=torch.float, device=self.device)
#         for i in range(T):
#             curr_graph = self.construct_graph(curr_state)
#             curr_graph, _ = self.model(curr_graph, winding, dst_tensor)
#
#         for i in range(rn):
#             curr_graph, prd_winding = self.model(curr_graph, winding, dst_tensor)
#             prd_node = curr_graph.x[:, -d:]  # B, n, d
#             tar_node = next_state[:, :, i, :].view(-1, d)
#             prd_traj[:, :, i, :] = curr_graph.x[:, -d:].view(B, n, d)
#             node_loss += criterion_dict['node'](prd_node, tar_node)
#
#         loss = node_loss
#         loss = loss / rn
#         loss2 = torch.sum((prd_traj - next_state) ** 2) / prd_traj.numel()
#         loss_dict = {
#             'total_loss': loss,
#             'total_loss2': loss2,
#             'node_loss': node_loss
#         }
#         return B, loss_dict, prd_traj
#
#
# class GRUWrapper(NetworkWrapper):
#     def setup(self):
#         """ Setup model, losses, optimizers, misc
#         """
#         # Whole model, for nn.DataParallel
#         self.model = GRUModel(self.config)
#         self.model.to(self.device)
#
#     def run_on_batch(self, batch, rn: int):
#         """ Run algorithm on batch of images in eval mode
#
#             @param batch: A Python dictionary with keys:
#                             'current_state' : a [B x n x d] torch tensor
#                             'next_state' : a [B x n x d] torch tensor
#         """
#         self.eval_mode()
#         self.send_batch_to_device(batch)
#         curr_state = batch['current_state']
#         assert curr_state.ndim == 3
#         n, T, d = curr_state.shape
#         curr_state = curr_state.unsqueeze(0)  # B, n, T, d
#
#         out = torch.zeros((rn, n, d), dtype=torch.float32, device=curr_state.device)
#         with torch.no_grad():
#             for i in range(T):
#                 curr_data = curr_state[:, :, i, :].squeeze().unsqueeze(0)  # 1, n, d
#                 curr_graph = self.construct_graph(curr_data)
#                 curr_graph, _ = self.model(curr_graph)
#
#             for i in range(rn):
#                 curr_graph, _ = self.model(curr_graph)
#                 out[i] = curr_graph.x.view(n, d)
#         return out
#
#     def rollout(self, s0, T):
#         """ Run sequence starting from state 0 (s0) for T steps
#
#             @param x: s0 [n x pT x d] torch tensor.
#             @param T: Number of time steps to rollout
#         """
#         return self.run_on_batch({'current_state': s0}, T)
#
#     def forward(self, curr_state, next_state, winding, dst_tensor, criterion_dict):
#         T = curr_state.shape[2]  # B, n, T, d
#         B, n, rn, d = next_state.shape  # B, s, in
#         tar_traj = next_state
#
#         x = curr_state.permute(0, 2, 1, 3).contiguous().view(B, T, -1)  # B, T, n * d
#         winding = winding.view(B, -1)
#         dst_tensor = dst_tensor.view(B, -1)
#         prd_traj = self.model(x, winding, dst_tensor).view(B, rn, n, d)  # B, rn, n * d
#         prd_traj = prd_traj.permute(0, 2, 1, 3).contiguous()  # B, n, rn, d
#         node_loss = criterion_dict['node'](prd_traj, tar_traj)
#         loss2 = torch.sum((prd_traj - tar_traj) ** 2) / prd_traj.numel()
#
#         loss = node_loss
#         loss_dict = {
#             'total_loss': loss,
#             'total_loss2': loss2,
#             'node_loss': node_loss
#         }
#         return B, loss_dict, prd_traj


def fetch_model_iterator(config: LayerConfig, args: Argument) -> NetworkWrapper:
    if args.model_type == ModelType.GraphNet:
        return GraphNetWrapper(config)
    # elif args.model_type == ModelType.GraphNetFullyConnected:
    #     return GraphNetNoRecurrencyWrapper(config)
    # elif args.model_type == ModelType.GRU:
    #     return GRUWrapper(config)
    else:
        raise TypeError('invalid model type: {}'.format(args.model_type))
