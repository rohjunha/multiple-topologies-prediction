import torch
from torch.nn import Sequential as Seq, Linear, LayerNorm, ReLU, GRU, Module
from torch_scatter import scatter_mean

from ..models.model_base import reparameterize


class NodeModelBase(Module):
    def __init__(self, config):
        Module.__init__(self)
        self.mlp1_inc = config['n_inc'] + config['e_outc']
        self.mlp1_hs1 = config['node_model_mlp1_hidden_sizes'][0]
        self.mlp1_hs2 = config['node_model_mlp1_hidden_sizes'][1]
        self.mlp2_hs1 = config['node_model_mlp2_hidden_sizes'][0]

        self.dim_out = config['n_outc']
        self.g_inc = config['g_inc']
        self.node_mlp_1 = Seq(Linear(self.mlp1_inc, self.mlp1_hs1),
                              LayerNorm(self.mlp1_hs1),
                              ReLU(),
                              Linear(self.mlp1_hs1, self.mlp1_hs2))

        self.mlp2_inc_uncond = config['n_inc'] + self.mlp1_hs2 + config['u_inc']
        self.mlp2_inc_cond = self.mlp2_inc_uncond + self.mlp1_hs2

    @property
    def mlp2_inc(self):
        raise NotImplementedError

    def _agg(self, x, edge_index, edge_attr):
        row, col = edge_index
        # Concat source node features, edge features. Shape: [E x (n_inc + e_outc)]
        srcnode_edge = torch.cat([x[row], edge_attr], dim=1)
        # Run this through an MLP... Shape: [E x hs2]
        srcnode_edge = self.node_mlp_1(srcnode_edge)
        # Mean-aggregation for every node (dest node for an edge). Shape: [N x hs2]
        per_node_aggs = scatter_mean(srcnode_edge, col, dim=0, dim_size=x.size(0))
        return per_node_aggs

    def forward(self, *args):
        """ Node Model of Graph Net layer

            @param x: [N x n_inc], where N is the number of nodes.
            @param edge_index: [2 x E] with max entry N - 1.
            @param edge_attr: [E x e_inc]
            @param u: [B x u_inc]
            @param batch: [N] with max entry B - 1.
            @param goal: [N x g_inc]

            @return: a [N x n_outc] torch tensor
        """
        raise NotImplementedError


class ProbNodeModel(NodeModelBase):
    def __init__(self, config):
        NodeModelBase.__init__(self, config)
        self.dim_lh = config['node_model_latent_mlp_hidden_size']
        self.dim_l = config['l_outc']
        self.node_mlp_2 = Seq(Linear(self.mlp2_inc, self.mlp2_hs1), LayerNorm(self.mlp2_hs1), ReLU(),
                              Linear(self.mlp2_hs1, self.dim_lh), LayerNorm(self.dim_lh), ReLU())
        self.mlp_m = Linear(self.dim_lh, self.dim_l)
        self.mlp_v = Linear(self.dim_lh, self.dim_l)
        self.mlp_x = Linear(self.dim_l, self.dim_out)

    @property
    def mlp2_inc(self):
        return self.mlp2_inc_uncond

    def forward(self, x, edge_index, edge_attr, u, batch, goal):
        per_node_aggs = self._agg(x, edge_index, edge_attr)
        node_agg_u = torch.cat([x, per_node_aggs, u[batch]], dim=1)
        x = self.node_mlp_2(node_agg_u)
        z_mu = self.mlp_m(x)
        z_var = self.mlp_v(x)
        z = reparameterize(z_mu, z_var)
        x = self.mlp_x(z)
        return x, z_mu, z_var

    def fetch_node_from_latent_variable(self, z_mu, z_var):
        z = reparameterize(z_mu, z_var)
        return self.mlp_x(z)


class UnconditionedNodeModelBase(NodeModelBase):
    def __init__(self, config):
        NodeModelBase.__init__(self, config)

    @property
    def mlp2_inc(self):
        return self.mlp2_inc_uncond


class ConditionedNodeModelBase(NodeModelBase):
    def __init__(self, config):
        NodeModelBase.__init__(self, config)

    @property
    def mlp2_inc(self):
        return self.mlp2_inc_cond


class UnconditionedNodeModel(UnconditionedNodeModelBase):
    def __init__(self, config):
        UnconditionedNodeModelBase.__init__(self, config)
        self.node_mlp_2 = Seq(Linear(self.mlp2_inc, self.mlp2_hs1), LayerNorm(self.mlp2_hs1), ReLU(),
                              Linear(self.mlp2_hs1, self.dim_out))

    def forward(self, x, edge_index, edge_attr, u, batch, goal):
        per_node_aggs = self._agg(x, edge_index, edge_attr)
        node_agg_u = torch.cat([x, per_node_aggs, u[batch]], dim=1)
        x = self.node_mlp_2(node_agg_u)
        return x


class ConditionedNodeModel(ConditionedNodeModelBase):
    def __init__(self, config):
        ConditionedNodeModelBase.__init__(self, config)
        self.fg = Seq(Linear(self.g_inc, self.mlp1_hs1), LayerNorm(self.mlp1_hs1), ReLU(),
                      Linear(self.mlp1_hs1, self.mlp1_hs2), LayerNorm(self.mlp1_hs2))
        self.node_mlp_2 = Seq(Linear(self.mlp2_inc, self.mlp2_hs1), LayerNorm(self.mlp2_hs1), ReLU(),
                              Linear(self.mlp2_hs1, self.dim_out))

    @property
    def mlp2_inc(self):
        return self.mlp2_inc_cond

    def forward(self, x, edge_index, edge_attr, u, batch, goal):
        per_node_aggs = self._agg(x, edge_index, edge_attr)
        goal = self.fg(goal.view(x.shape[0], -1))
        node_agg_u = torch.cat([x, per_node_aggs, goal, u[batch]], dim=1)
        x = self.node_mlp_2(node_agg_u)
        return x


class RecurrentNodeModelBase(NodeModelBase):
    def __init__(self, config):
        NodeModelBase.__init__(self, config)
        self.gru = GRU(self.mlp2_hs1, self.mlp2_hs1, 1, batch_first=True)
        self.fc_out = Linear(self.mlp2_hs1, config['n_outc'])

    def _agg(self, x, edge_index, edge_attr):
        row, col = edge_index
        # Concat source node features, edge features. Shape: [E x (n_inc + e_outc)]
        srcnode_edge = torch.cat([x[row], edge_attr], dim=1)
        # Run this through an MLP... Shape: [E x hs2]
        srcnode_edge = self.node_mlp_1(srcnode_edge)
        # Mean-aggregation for every node (dest node for an edge). Shape: [N x hs2]
        per_node_aggs = scatter_mean(srcnode_edge, col, dim=0, dim_size=x.size(0))
        return per_node_aggs

    def _forward(self, node_agg_u, h):
        # Run through MLP. Shape: [N x n_outc]
        x = self.node_mlp_2(node_agg_u)
        x = x.unsqueeze(1)
        h = h.unsqueeze(0)
        x, h = self.gru(x, h)
        x = x.squeeze()
        h = x.squeeze()
        x = self.fc_out(x)
        return x, h

    def forward(self, *args):
        """ Node Model of Graph Net layer

            @param x: [N x n_inc], where N is the number of nodes.
            @param edge_index: [2 x E] with max entry N - 1.
            @param edge_attr: [E x e_inc]
            @param h: [N x hidden]
            @param u: [B x u_inc]
            @param batch: [N] with max entry B - 1.

            @return: a [N x n_outc] torch tensor
            @return: h [N x hidden]
        """
        raise NotImplementedError


class UnconditionedRecurrentNodeModel(UnconditionedNodeModelBase, RecurrentNodeModelBase):
    def __init__(self, config):
        UnconditionedNodeModelBase.__init__(self, config)
        RecurrentNodeModelBase.__init__(self, config)
        self.node_mlp_2 = Seq(Linear(self.mlp2_inc, self.mlp2_hs1), LayerNorm(self.mlp2_hs1), ReLU())

    def forward(self, x, edge_index, edge_attr, h, u, batch, goal):
        per_node_aggs = self._agg(x, edge_index, edge_attr)
        node_agg_u = torch.cat([x, per_node_aggs, u[batch]], dim=1)
        return self._forward(node_agg_u, h)


class ConditionedRecurrentNodeModel(ConditionedNodeModelBase, RecurrentNodeModelBase):
    def __init__(self, config):
        ConditionedNodeModelBase.__init__(self, config)
        RecurrentNodeModelBase.__init__(self, config)
        self.fg = Seq(Linear(self.g_inc, self.mlp1_hs1), LayerNorm(self.mlp1_hs1), ReLU(),
                      Linear(self.mlp1_hs1, self.mlp1_hs2), LayerNorm(self.mlp1_hs2))
        self.node_mlp_2 = Seq(Linear(self.mlp2_inc, self.mlp2_hs1), LayerNorm(self.mlp2_hs1), ReLU())

    def forward(self, x, edge_index, edge_attr, h, u, batch, goal):
        per_node_aggs = self._agg(x, edge_index, edge_attr)
        goal = self.fg(goal)
        goal = goal.view(x.shape[0], -1)
        node_agg_u = torch.cat([x, per_node_aggs, goal, u[batch]], dim=1)
        return self._forward(node_agg_u, h)
