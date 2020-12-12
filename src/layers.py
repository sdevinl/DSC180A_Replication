# Aggregators
import torch.nn.functional as F
import torch.nn as nn
from src.data import *


# Aggregators
class Mean_Agg(torch.nn.Module):
    '''
    GraphSAGE Mean Aggregator
    '''

    def __init__(self):
        super(Mean_Agg, self).__init__()

    def forward(self, h, A, W, activation='relu'):
        A = torch.tensor(A)
        A = A.float()
        h = h.float()

        # X: batch of nodes
        h1 = h
        h = torch.sparse.mm(A, h) / torch.sparse.sum(A)
        print(h1.shape, h.shape)
        h = torch.cat((h1, h), 1)
        h = torch.spmm(W, h.T.float())

        if activation == 'relu':
            h = F.relu(h.T)

        return h


class MaxPool_Agg(torch.nn.Module):
    '''
    GraphSAGE Pooling Aggregator
    '''

    def __init__(self, in_feasts, out_feats):
        ...

    def forward(self, x, neigh):
        ...

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
"""class GCN_Layer(torch.nn.Module):
    '''
    #Simple GCN layer
    '''

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

        return output"""

class GCN_Layer(nn.Module):
    def __init__(self, in_features, out_features, norm = True):
        super(GCN_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        self.norm = norm

    def reset_parameters(self):
        stdv = 1 / (self.weight.size(1))**0.5
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, layer_in, adj, norm=True):
        if norm:
            adj_hat = adj
            D_hat = torch.diag((torch.sparse.sum(adj_hat, 1).values())**(-1/2))
            adj_hat = torch.sparse.mm(adj_hat, D_hat)
            adj_hat = torch.spmm(adj_hat, D_hat)
            adj = adj_hat

        out = torch.spmm(adj, torch.mm(layer_in, self.weight))
        return out + self.bias

# LPA-GCN
class GCN_LPA_Layer(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super(GCN_LPA_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.mask = nn.Parameter(adj.clone()).to_dense()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / (self.weight.size(1)) ** 0.5
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, y, norm=True):
        adj = adj.to_dense()
        if norm:
            adj = adj * self.mask
            adj = F.normalize(adj, p=1, dim=1)
        output = torch.mm(adj, torch.mm(x, self.weight))
        y_hat = torch.mm(adj, y)
        return output + self.bias, y_hat


