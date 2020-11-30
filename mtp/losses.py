import torch
import torch.nn as nn


class NormalizedMSELoss(torch.nn.Module):
    """ Compute Normalized MSE loss
    """

    def __init__(self, weighted=False):
        """ Compute state-wise normalized MSE Loss.
            Divide normal MSE loss by standard deviation
        """
        torch.nn.Module.__init__(self)
        self.MSELoss = nn.MSELoss(reduction='none')

    def forward(self, x, target):
        """ Compute state-wise normalized MSE loss

            Note: d = 4. State is: [x, y, dx, dy]

            @param x: a [N x d] torch.FloatTensor of values
            @param target: a [N x d] torch.LongTensor of values
        """
        N, d = x.shape
        temp = self.MSELoss(x, target) # Shape: [N x d]

        with torch.no_grad():
            weights = 1 / x.std(dim=0, keepdim=True)  # Shape: [1 x d]
            weight_mask = weights.repeat(N, 1)

        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 

        return loss
