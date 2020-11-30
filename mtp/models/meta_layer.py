from torch.nn import Module

from ..utils.logging import get_logger

logger = get_logger(__name__)


class MetaLayerBase(Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        Module.__init__(self)
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)

    def forward(self, *args):
        """
        update all variables in a graph (node, edge, global)
        @param x: vectorized raw (node) data (B * n, x_dim)
        @param edge_index: tuple of edge indices and each index has a size of (B * n, )
        @param edge_attr: edge attributes (B * E, e_dim)
        @param u: global variables (B, u_dim)
        @param batch: global index (N, ) mapping from the edge index to the graph index
        @param winding: onehot winding number (B, E, 2)
        @param goal: onehot goal indicator (B, 4)
        @return: updated values x, edge_attr, u
        """
        raise NotImplementedError


class ProbBinaryMetaLayer(MetaLayerBase):
    def __init__(self, edge_model, node_model, global_model):
        MetaLayerBase.__init__(self, edge_model, node_model, global_model)

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, winding=None, goal=None):
        row, col = edge_index
        BE = edge_attr.shape[0]
        edge_attr, ze_mu, ze_var = self.edge_model(x[row], x[col], edge_attr, u,
                                                   batch if batch is None else batch[row],
                                                   winding if winding is None else winding.view(BE, -1))
        x, zn_mu, zn_var = self.node_model(x, edge_index, edge_attr, u, batch, goal)
        u = self.global_model(x, edge_index, edge_attr, u, batch)
        return x, edge_attr, u, zn_mu, zn_var, ze_mu, ze_var


class BinaryMetaLayer(MetaLayerBase):
    def __init__(self, edge_model, node_model, global_model):
        MetaLayerBase.__init__(self, edge_model, node_model, global_model)

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None, winding=None, goal=None):
        row, col = edge_index
        BE = edge_attr.shape[0]
        edge_attr = self.edge_model(x[row], x[col], edge_attr, u,
                                    batch if batch is None else batch[row],
                                    winding if winding is None else winding.view(BE, -1))
        x = self.node_model(x, edge_index, edge_attr, u, batch, goal)
        u = self.global_model(x, edge_index, edge_attr, u, batch)
        return x, edge_attr, u


class RecurrentMetaLayer(MetaLayerBase):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        MetaLayerBase.__init__(self, edge_model, node_model, global_model)

    def forward(self, x, edge_index, edge_attr=None, u=None,
                h_x=None, h_e=None, h_u=None,
                batch=None, winding=None, goal=None):
        row, col = edge_index
        BE = edge_attr.shape[0]
        edge_attr, h_e = self.edge_model(x[row], x[col], edge_attr, h_e, u,
                                         batch if batch is None else batch[row],
                                         winding if winding is None else winding.view(BE, -1))
        x, h_x = self.node_model(x, edge_index, edge_attr, h_x, u, batch, goal)
        u, h_u = self.global_model(x, edge_index, edge_attr, h_u, u, batch)
        return x, edge_attr, u, h_x, h_e, h_u
