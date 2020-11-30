from typing import Dict

import torch
from torch.nn import Module

from ..config import LayerConfig
from ..models.edge_model import ProbEdgeModel, UnconditionedRecurrentEdgeModel, \
    ConditionedRecurrentEdgeModel, UnconditionedEdgeModel, ConditionedEdgeModel
from ..models.global_model import GlobalModel, RecurrentGlobalModel
from ..models.meta_layer import RecurrentMetaLayer, ProbBinaryMetaLayer, BinaryMetaLayer
from ..models.node_model import ProbNodeModel, UnconditionedRecurrentNodeModel, \
    ConditionedNodeModel, ConditionedRecurrentNodeModel, UnconditionedNodeModel


class GraphNetBase(Module):
    def __init__(self, config: LayerConfig):
        Module.__init__(self)
        self.config = config
        self.layer_config = self.config.layer_configs

    @property
    def device(self):
        return self.config.device

    @property
    def n_dim(self):
        return self.config.n_dim

    @property
    def e_dim(self):
        return self.config.e_dim

    @property
    def w_dim(self):
        return self.config.w_dim

    @property
    def g_dim(self):
        return self.config.g_dim

    @property
    def pred_w_dim(self):
        return self.config.pred_w_dim

    @property
    def pred_g_dim(self):
        return self.config.pred_g_dim

    @property
    def u_dim(self):
        return self.config.u_dim


class WindingGraphNet(GraphNetBase):
    def __init__(self, config: LayerConfig):
        GraphNetBase.__init__(self, config)
        ec = self.layer_config['winding_encoder']
        rc = self.layer_config['winding_recurrent']
        dc = self.layer_config['winding_decoder']

        self.encoder = BinaryMetaLayer(
            UnconditionedEdgeModel(ec),
            UnconditionedNodeModel(ec),
            GlobalModel(ec)).to(self.device)
        self.recurrent = RecurrentMetaLayer(
            UnconditionedRecurrentEdgeModel(rc),
            UnconditionedRecurrentNodeModel(rc),
            RecurrentGlobalModel(rc)).to(self.device)
        self.decoder = ProbBinaryMetaLayer(
            ProbEdgeModel(dc),
            ProbNodeModel(dc),
            GlobalModel(dc)).to(self.device)

    def fetch_node_and_edge_from_latent_dict(self, latent_dict: Dict[str, torch.Tensor]):
        w = self.decoder.edge_model.fetch_edge_from_latent_variable(latent_dict['edge_mu'], latent_dict['edge_var'])
        g = self.decoder.node_model.fetch_node_from_latent_variable(latent_dict['node_mu'], latent_dict['node_var'])
        return w, g

    def forward(self, g, h):
        """ Forward pass of GraphNet

            @param g: a graph (torch_geometric.Data) instance. (batched)
            @param h: a hidden graph
            @param w: binary winding numbers

            @return: a graph (torch_geometric.Data) instance. (batched)
        """
        v, e, u = g.x, g.edge_attr, g.u
        h_v, h_e, h_u = h.x, h.edge_attr, h.u

        v, e, u = self.encoder(v, g.edge_index, e, u, g.batch)
        _, _, _, h_v, h_e, h_u = self.recurrent(v, g.edge_index, e, u, h_v, h_e, h_u, g.batch)
        g_, w_, _, zn_mu, zn_var, ze_mu, ze_var = self.decoder(h_v, g.edge_index, h_e, h_u, g.batch)

        new_hc = h.clone()
        new_hc.x = h_v
        new_hc.edge_attr = h_e
        new_hc.u = h_u

        return new_hc, w_, g_, ze_mu, ze_var, zn_mu, zn_var


class TrajectoryGraphNet(GraphNetBase):
    def __init__(self, config: LayerConfig):
        GraphNetBase.__init__(self, config)
        ec = self.layer_config['trajectory_encoder']
        rc = self.layer_config['trajectory_recurrent']
        dc = self.layer_config['trajectory_decoder']

        self.encoder = BinaryMetaLayer(
            ConditionedEdgeModel(ec),
            ConditionedNodeModel(ec),
            GlobalModel(ec)).to(self.device)
        self.recurrent = RecurrentMetaLayer(
            ConditionedRecurrentEdgeModel(rc),
            ConditionedRecurrentNodeModel(rc),
            RecurrentGlobalModel(rc)).to(self.device)
        self.decoder = BinaryMetaLayer(
            ConditionedEdgeModel(dc),
            ConditionedNodeModel(dc),
            GlobalModel(dc)).to(self.device)

    def forward(self, g, h, w, goal):
        """ Forward pass of GraphNet

            @param g: a graph (torch_geometric.Data) instance. (batched)
            @param h: a hidden graph
            @param w: binary winding numbers

            @return: a graph (torch_geometric.Data) instance. (batched)
        """
        v, e, u = g.x, g.edge_attr, g.u
        h_v, h_e, h_u = h.x, h.edge_attr, h.u

        v, e, u = self.encoder(v, g.edge_index, e, u, g.batch, w, goal)
        _, _, _, h_v, h_e, h_u = self.recurrent(v, g.edge_index, e, u, h_v, h_e, h_u, g.batch, w, goal)
        v, e, u = self.decoder(h_v, g.edge_index, h_e, h_u, g.batch, w, goal)

        new_g = g.clone()
        new_g.x = g.x + v
        new_g.edge_attr = g.edge_attr + e
        new_g.u = g.u + u

        new_h = h.clone()
        new_h.x = h_v
        new_h.edge_attr = h_e
        new_h.u = h_u

        return new_g, new_h


# class GraphNetNoRecurrency(GraphNetBase):
#     def __init__(self, config: LayerConfig):
#         GraphNetBase.__init__(self, config)
#         ec = self.layer_config['encoder']
#         dc = self.layer_config['decoder']
#         ec['n_inc'] *= HISTORY_WINDOW
#         ec['e_inc'] *= HISTORY_WINDOW
#         ec['u_inc'] *= HISTORY_WINDOW
#         self.encoder = BinaryMetaLayer(EdgeModel(ec, pred_type), NodeModel(ec, pred_type), GlobalModel(ec)).to(self.device)
#         self.decoder = BinaryMetaLayer(EdgeModel(dc, pred_type), NodeModel(dc, pred_type), GlobalModel(dc)).to(self.device)
#
#     def forward(self, g, w, goal):
#         """ Forward pass of GraphNet
#
#             @param g: a graph (torch_geometric.Data) instance. (batched)
#             @param w: binary winding numbers
#
#             @return: a graph (torch_geometric.Data) instance. (batched)
#         """
#         v, e, u = g.x, g.edge_attr, g.u
#         B = u.shape[0]
#         n = v.shape[0] // B
#         v, e, u = self.encoder(v, g.edge_index, e, u, g.batch, w, goal)
#         v, e, u = self.decoder(v, g.edge_index, e, u, g.batch, w, goal)
#
#         new_g = g.clone()
#
#         nv = g.x[..., -self.n_dim:] + v
#         if self.pred_w_dim > 0:
#             ne = g.edge_attr[..., -self.e_dim:] + e[..., -self.e_dim-self.pred_w_dim:-self.pred_w_dim]
#             winding_number = e[..., -self.pred_w_dim:].view(B, n * (n - 1) * self.pred_w_dim)
#         else:
#             ne = g.edge_attr[..., -self.e_dim:] + e[..., -self.e_dim:]
#             winding_number = None
#         nu = g.u[..., -self.u_dim:] + u
#
#         new_g.x = torch.cat([g.x[..., :-self.n_dim], nv], dim=-1)
#         new_g.edge_attr = torch.cat([g.edge_attr[..., :-self.e_dim], ne], dim=-1)
#         new_g.u = torch.cat([g.u[..., :-self.u_dim], nu], dim=-1)
#
#         return new_g, winding_number