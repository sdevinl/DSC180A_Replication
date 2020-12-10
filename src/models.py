from src.layers import *


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        """
        Simple GCN Model
        """
        super(GCN, self).__init__()

        # GCN Layers
        self.gc1 = GCN_Layer(nfeat, nhid)
        self.gc2 = GCN_Layer(nhid, nclass)
        # self.gc3 = GCN_Layer(nhid-300, nclass)

    def forward(self, X, A, prep_A):
        """
        """

        X = F.relu(self.gc1(X, A, prep_A))
        # x = F.relu(self.gc2(x, adj))
        X = self.gc2(X, A, prep_A)
        return F.log_softmax(X, dim=1)


class GCN_LPA(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, A):
        super(GCN_LPA, self).__init__()

        self.gcn_lpa1 = LPA_GCN_Layer(nfeat, nhid, A)
        self.gcn_lpa2 = LPA_GCN_Layer(nhid, nclass, A)

    def forward(self, X, A, Y):
        X, Y_hat = self.gcn_lpa1(X, A, Y)
        X = F.relu(X)
        X, Y_hat = self.gcn_lpa2(X, A, Y_hat)
        return F.relu(X), F.relu(Y_hat)


# GraphSAGE
class GS(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, agg='mean', num_samples=25, dropout=.5):
        """
        GraphSAGE Model
        """
        super(GS, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.agg = agg
        self.num_samples = num_samples
        # self.num_layers = len(nhid) + 1

        self.W = torch.randn(nfeat, 2 * nfeat)
        self.W = nn.Parameter(self.W)

        if self.agg == 'mean':
            self.agg = Mean_Agg()
        elif self.agg == 'maxpool':
            self.agg = MaxPool_Agg()

        self.gc1 = GCN_Layer(nfeat, nhid)
        self.gc2 = GCN_Layer(nhid, nclass)

    def forward(self, X, A, K=1, activation='relu', prep_A='norm'):

        # shape of H = number of nodes x number of features
        h = torch.tensor(X)

        for k in np.arange(K):
            h = self.agg(h, A, self.W, activation)

        h = F.relu(self.gc1(h.float(), A, prep_A))
        X = self.gc2(h, A, prep_A)

        return F.log_softmax(h, dim=1)
