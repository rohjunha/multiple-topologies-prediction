import torch
from torch.nn import Module, Sequential as Seq, Linear, LayerNorm, ReLU, GRU, functional as F
from torch_scatter import scatter_mean


class GlobalModel(Module):
    def __init__(self, config):
        Module.__init__(self)

        inc = config['u_inc'] + config['n_outc'] + config['e_outc']
        hs1 = config['global_model_mlp1_hidden_sizes'][0]

        self.global_mlp = Seq(Linear(inc, hs1),
                              LayerNorm(hs1),
                              ReLU(),
                              Linear(hs1, config['u_outc']))

    def forward(self, x, edge_index, edge_attr, u, batch):
        """ Global Update of Graph Net Layer

            @param x: [N x n_outc], where N is the number of nodes.
            @param edge_index: [2 x E] with max entry N - 1.
            @param edge_attr: [E x e_outc]
            @param u: [B x u_inc]
            @param batch: [N] with max entry B - 1.

            @return: a [B x u_outc] torch tensor
        """

        row, col = edge_index
        edge_batch = batch[row]  # edge_batch is same as batch in EdgeModel.forward(). Shape: [E]

        per_batch_edge_aggregations = scatter_mean(edge_attr, edge_batch, dim=0)  # Shape: [B x e_outc]
        per_batch_node_aggregations = scatter_mean(x, batch, dim=0)  # Shape: [B x n_outc]

        out = torch.cat([u, per_batch_node_aggregations, per_batch_edge_aggregations],
                        dim=1)  # Shape: [B x (u_inc + n_outc + e_outc)]
        return self.global_mlp(out)


class RecurrentGlobalModel(Module):
    def __init__(self, config):
        Module.__init__(self)

        inc = config['u_inc'] + config['n_outc'] + config['e_outc']
        hs1 = config['global_model_mlp1_hidden_sizes'][0]

        self.fc1 = Linear(inc, hs1)
        self.in1 = LayerNorm(hs1)
        self.gru = GRU(hs1, hs1, 1, batch_first=True)
        self.fc2 = Linear(hs1, config['u_outc'])

    def forward(self, x, edge_index, edge_attr, h, u, batch):
        """ Global Update of Graph Net Layer

            @param x: [N x n_outc], where N is the number of nodes.
            @param edge_index: [2 x E] with max entry N - 1.
            @param edge_attr: [E x e_outc]
            @param h: [B x hidden]
            @param u: [B x u_inc]
            @param batch: [N] with max entry B - 1.

            @return: a [B x u_outc] torch tensor
        """

        row, col = edge_index
        edge_batch = batch[row]  # edge_batch is same as batch in EdgeModel.forward(). Shape: [E]

        per_batch_edge_aggregations = scatter_mean(edge_attr, edge_batch, dim=0)  # Shape: [B x e_outc]
        per_batch_node_aggregations = scatter_mean(x, batch, dim=0)  # Shape: [B x n_outc]

        x = torch.cat([u, per_batch_node_aggregations, per_batch_edge_aggregations], dim=1)  # Shape: [B x (u_inc + n_outc + e_outc)]
        x = F.relu(self.in1(self.fc1(x)))
        x = x.unsqueeze(1)
        h = h.unsqueeze(0)
        x, h = self.gru(x, h)
        x = x.squeeze()
        h = x.squeeze()
        x = self.fc2(x)
        return x, h
