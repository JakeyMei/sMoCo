import torch
import torch.nn as nn


class SimSiam(nn.Module):

    def __init__(self, encoder, dim, pred_dim=512):
        super(SimSiam, self).__init__()
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

        self.encoder = encoder

    def forward(self, anchor, pos):
        loss, _, _, z1 = self.encoder(anchor[0], anchor[1], mlp=True)
        _, _, _, z2 = self.encoder(pos[0], pos[1], mlp=True)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return loss, p1, p2, z1.detach(), z2.detach()