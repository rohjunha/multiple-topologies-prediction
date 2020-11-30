import torch
from torch.nn import Sequential as Seq, Linear, LayerNorm, ReLU, GRU, Module

from ..models.model_base import reparameterize
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EdgeModelBase(Module):
    def __init__(self, config):
        Module.__init__(self)

        self.dim_out = config['e_outc']
        self.dim_in = config['e_inc'] + 2 * config['n_inc'] + config['u_inc']
        self.dim_w1, self.dim_w2 = config['w_inc'], self.dim_in // 2
        self.dim_h1 = config['edge_model_mlp1_hidden_sizes'][0]
        self.dim_h2 = config['edge_model_mlp1_hidden_sizes'][1]

    def forward(self, *args):
        """ Edge Model of Graph Net layer

            @param src: [E x n_inc], where E is the number of edges. Treat E as batch size
            @param dest: [E x n_inc], where E is the number of edges.
            @param edge_attr: [E x e_inc]
            @param u: [B x u_inc], where B is the number of graphs.
            @param batch: [E] with max entry B - 1.
            @param winding: [E x 2] with max entry 1.

            @return a [E x e_outc] torch tensor
        """
        raise NotImplementedError


class ProbEdgeModel(EdgeModelBase):
    def __init__(self, config):
        EdgeModelBase.__init__(self, config)
        self.dim_x1 = self.dim_in
        self.dim_lh = config['edge_model_latent_mlp_hidden_size']
        self.dim_l = config['l_outc']
        self.mlp_h = Seq(Linear(self.dim_x1, self.dim_h1), LayerNorm(self.dim_h1), ReLU(),
                         Linear(self.dim_h1, self.dim_lh), LayerNorm(self.dim_lh), ReLU())
        self.mlp_m = Linear(self.dim_lh, self.dim_l)
        self.mlp_v = Linear(self.dim_lh, self.dim_l)
        self.mlp_w = Linear(self.dim_l, self.dim_out)

    def forward(self, src, dest, edge_attr, u, batch, winding):
        if u.ndim == 1:
            u.unsqueeze_(0)
        x = torch.cat([src, dest, edge_attr, u[batch]], 1)
        x = self.mlp_h(x)
        z_mu = self.mlp_m(x)
        z_var = self.mlp_v(x)
        z = reparameterize(z_mu, z_var)
        x = self.mlp_w(z)
        return x, z_mu, z_var

    def fetch_edge_from_latent_variable(self, z_mu, z_var):
        z = reparameterize(z_mu, z_var)
        return self.mlp_w(z)


class ConditionedEdgeModel(EdgeModelBase):
    def __init__(self, config):
        EdgeModelBase.__init__(self, config)
        self.dim_x1 = self.dim_in + self.dim_w2
        self.mlp_h = Seq(Linear(self.dim_x1, self.dim_h1), LayerNorm(self.dim_h1), ReLU(),
                         Linear(self.dim_h1, self.dim_h2), LayerNorm(self.dim_h2), ReLU(),
                         Linear(self.dim_h2, self.dim_out))
        self.winding_mlp = Seq(Linear(self.dim_w1, self.dim_w2), LayerNorm(self.dim_w2), ReLU())

    def forward(self, src, dest, edge_attr, u, batch, winding):
        if u.ndim == 1:
            u.unsqueeze_(0)
        w = self.winding_mlp(winding)
        x = torch.cat([src, dest, edge_attr, u[batch], w], 1)  # Shape: [E x (2*n_inc + e_inc + u_inc)]
        x = self.mlp_h(x)
        return x


class UnconditionedEdgeModel(EdgeModelBase):
    def __init__(self, config):
        EdgeModelBase.__init__(self, config)
        self.dim_x1 = self.dim_in
        self.mlp_h = Seq(Linear(self.dim_x1, self.dim_h1), LayerNorm(self.dim_h1), ReLU(),
                         Linear(self.dim_h1, self.dim_h2), LayerNorm(self.dim_h2), ReLU(),
                         Linear(self.dim_h2, self.dim_out))

    def forward(self, src, dest, edge_attr, u, batch, winding):
        if u.ndim == 1:
            u.unsqueeze_(0)
        x = torch.cat([src, dest, edge_attr, u[batch]], 1)  # Shape: [E x (2*n_inc + e_inc + u_inc)]
        x = self.mlp_h(x)
        return x


class RecurrentEdgeModelBase(EdgeModelBase):
    def __init__(self, config):
        EdgeModelBase.__init__(self, config)
        self.gru = GRU(self.dim_h1, self.dim_h2, 1, batch_first=True)
        self.mlp_x = Linear(self.dim_h2, self.dim_out)

    def _forward(self, x, h):
        x = self.mlp1(x)
        x = x.unsqueeze(1)
        h = h.unsqueeze(0)
        x, h = self.gru(x, h)
        x = x.squeeze()
        h = h.squeeze()
        x = self.mlp_x(x)
        return x, h

    def forward(self, *args):
        """ Edge Model of Graph Net layer

            @param src: [E x n_inc], where E is the number of edges. Treat E as batch size
            @param dest: [E x n_inc], where E is the number of edges.
            @param edge_attr: [E x e_inc]
            @param h: [E x hidden]
            @param u: [B x u_inc], where B is the number of graphs.
            @param batch: [E] with max entry B - 1.
            @param winding: [E] with max entry 1.

            @return a [E x e_outc] torch tensor
            @return h [E x hidden]
        """
        raise NotImplementedError


class UnconditionedRecurrentEdgeModel(RecurrentEdgeModelBase):
    def __init__(self, config):
        RecurrentEdgeModelBase.__init__(self, config)
        self.dim_x1 = self.dim_in
        self.mlp1 = Seq(Linear(self.dim_x1, self.dim_h1), LayerNorm(self.dim_h1), ReLU())

    def forward(self, src, dest, edge_attr, h, u, batch, winding):
        x = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self._forward(x, h)


class ConditionedRecurrentEdgeModel(RecurrentEdgeModelBase):
    def __init__(self, config):
        RecurrentEdgeModelBase.__init__(self, config)
        self.dim_x1 = self.dim_in + self.dim_w2
        self.mlp1 = Seq(Linear(self.dim_x1, self.dim_h1), LayerNorm(self.dim_h1), ReLU())
        self.winding_mlp = Seq(Linear(self.dim_w1, self.dim_w2), LayerNorm(self.dim_w2), ReLU())

    def forward(self, src, dest, edge_attr, h, u, batch, winding):
        w = self.winding_mlp(winding)
        x = torch.cat([src, dest, edge_attr, u[batch], w], 1)  # Shape: [E x (2*n_inc + e_inc + u_inc)]
        return self._forward(x, h)
