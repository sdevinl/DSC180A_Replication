# Aggregators
import torch.nn.functional as F
import torch.nn as nn
from src.data import *


class Mean_Agg(torch.nn.Module):
    '''
    GraphSAGE Mean Aggregator
    '''

    def __init__(self, in_feats, out_feats):
        super(Mean_Agg, self).__init__()

        self.fc_x = nn.Linear(in_feats, out_feats)
        self.fc_neigh = nn.Linear(in_feats, out_feats)
        self.out_feats = out_feats

    def forward(self, x, neigh):
        # X: batch of nodes
        agg_neigh = neigh.view(x.size(0), -1, neigh.size(1))
        agg_neigh = agg_neigh.mean(dim=1)

        output = torch.cat([self.fc_x(x), self.fc_neigh(agg_neigh)], dim=1)

        # Average of Neighborhood feature matrix for each node in batch
        return F.relu(output)


class MaxPool_Agg(torch.nn.Module):
    '''
    GraphSAGE Pooling Aggregator
    '''

    def __init__(self, in_feasts, out_feats):
        "..."

    def forward(self, x, neigh):
        '...'


# GraphSAGE Models and Layers

class GS_Layer(torch.nn.Module):
    '''
    GraphSAGE Layer
    '''

    def __init__(self):
        super(GS_Layer, self).__init__()
        '...'

    def forward(self, X, steps, A):
        # X: batch of nodes
        # steps: steps from node for neighborhood
        # A: adjacency matrix to find nodes in neighborhood
        '...'


# params:
# A is the adj matrix
# X is the feature matrix
class GCN_Layer(torch.nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_feats, out_feats):
        super(GCN_Layer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = np.random.randn(in_feats, out_feats)
        self.weight = nn.Parameter(torch.Tensor(self.weight))

    def forward(self, prev_output, A, prep_A=None):
        '''
        Propogation Rule:
        params: pred_A - specify how to prepare A, with or without normalization
        '''
        prev_output = torch.Tensor(prev_output)
        A = torch.Tensor(A)

        right_term = torch.mm(prev_output, self.weight)

        # Unnormalized
        if prep_A == None:
            output = torch.mm(A, right_term)
        # Normalized with Kipf & Welling
        elif prep_A == "norm":
            I = torch.eye(A.shape[0])
            A_hat = A + I
            D_hat = torch.Tensor(np.diag(A_hat.sum(axis=1) ** (-1 / 2)))
            output = torch.mm(D_hat, A_hat)
            output = torch.mm(output, D_hat)
            output = torch.mm(output, right_term)

        return output


# LPA-GCN

class LPA_GCN_Layer(torch.nn.Module):
    def __init__(self, in_feats, out_feats, A):
        super(LPA_GCN_Layer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = np.random.randn(in_feats, out_feats)
        self.weight = nn.Parameter(torch.Tensor(self.weight))
        A = torch.Tensor(A)
        self.mask_A = A.clone()
        self.mask_A = nn.Parameter(self.mask_A)

    def forward(self, X, A, Y):
        X = torch.Tensor(X)
        A = torch.Tensor(A)
        Y = torch.Tensor(Y)

        right_term = torch.mm(X, self.weight)
        # Hadamard A'
        A = A * self.mask_A
        # Normalize D^-1 * A'
        A = F.normalize(A, p=1, dim=1)

        output = torch.mm(A, right_term)
        Y_hat = torch.mm(A, Y)
        return output, Y_hat

