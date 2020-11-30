from typing import Union

import torch
from torch.nn import Module, ReLU

from config import LayerConfig, TARGET_WINDOW
from utils.logging import get_logger

logger = get_logger(__name__)


class GRUModel(Module):
    def __init__(self, config: LayerConfig):
        torch.nn.Module.__init__(self)
        self.config = config
        self.layer_config = config.layer_configs

        self.num_agent = config.num_agent
        self.n_dim = self.layer_config['encoder']['n_inc']
        self.input_size = self.n_dim * self.num_agent
        self.hidden_size = 16
        self.num_layers = 1
        self.output_size = self.n_dim * self.num_agent
        self.output_len = TARGET_WINDOW
        self.use_condition = config.use_condition
        self.g_dim = config.g_dim
        self.w_dim = config.w_dim

        if self.use_condition:
            self.input_size = (self.n_dim + self.g_dim) * self.num_agent \
                              + self.w_dim * self.num_agent * (self.num_agent - 1)
        else:
            self.input_size = self.n_dim * self.num_agent

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.nl = ReLU()

    def forward(self, x: torch.Tensor, w, g, h: Union[torch.Tensor, None] = None):
        # x: B x S x Dx
        # w: B x Dw
        # goal: B x Dg
        if self.use_condition:
            B, S = x.shape[:2]
            Dw, Dg = w.shape[1], g.shape[1]
            w = w.view(B, 1, Dw).expand(B, S, Dw)  # B, S, Dw
            g = g.view(B, 1, Dg).expand(B, S, Dg)  # B, S, Dg
            x = torch.cat([x, w, g], dim=-1)
        x = self.fc2(self.nl(self.fc1(x)))  # B x S x H

        output = []
        for _ in range(self.output_len):
            _, h = self.gru(x, h)  # _, B x 1 x H
            y = self.fc4(self.nl(self.fc3(h.squeeze())))  # B x OUT
            output.append(y)  # B x OUT
        y = torch.stack(output, dim=1)  # B x S x OUT
        return y
