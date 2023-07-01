import torch
import torch.nn as nn
import math
import numpy as np

class OneClassLoss(nn.Module):
    def __init__(self, device):
        super(OneClassLoss, self).__init__()
        self.device = device
        self.mean = torch.zeros((1))
        self.similarity_f = nn.CosineSimilarity(dim=1)
        
    def forward(self, h):
        if h.shape[0]==0:
            return torch.zeros((1)).to(self.device)
        
        # save mean of the old  cluster
        old_mean = self.mean.detach().to(self.device)

        # update mean of the cluster with new data points
        self.mean = old_mean + torch.sum(h, dim=0)/h.shape[0]

        # L2 loss 
        # loss = torch.sum((h-self.mean)**2)/self.mean.shape[0]

        # L1 loss
        # loss = torch.sum((h-self.mean))/self.mean.shape[0]

        # cosine similarity loss
        sim = self.similarity_f(self.mean.unsqueeze(0), h)
        loss = -sim.mean()

        return loss