from src.layers import *

'''"
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
        return F.log_softmax(X, dim=1)"'''

class GCN(nn.Module):
    def __init__(self, feature_size, hidden_size, class_size):
        super(GCN, self).__init__()
        self.gc1 = GCN_Layer(feature_size, hidden_size)
        self.gc2 = GCN_Layer(hidden_size, class_size)

    def forward(self, x, adj, norm):
        x = F.relu(self.gc1(x, adj, norm))
        x = self.gc2(x, adj, norm)
        return F.log_softmax(x, dim=1)



'''class GCN_LPA(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, A):
        super(GCN_LPA, self).__init__()

        self.gcn_lpa1 = LPA_GCN_Layer(nfeat, nhid, A)
        self.gcn_lpa2 = LPA_GCN_Layer(nhid, nclass, A)

    def forward(self, X, A, Y):
        X, Y_hat = self.gcn_lpa1(X, A, Y)
        X = F.relu(X)
        X, Y_hat = self.gcn_lpa2(X, A, Y_hat)
        return F.relu(X), F.relu(Y_hat)'''

class GCN_LPA(nn.Module):
    def __init__(self, feature_size, hidden_size, class_size, adj):
        super(GCN_LPA, self).__init__()
        self.gc1 = GCN_LPA_Layer(feature_size, hidden_size, adj)
        self.gc2 = GCN_LPA_Layer(hidden_size, class_size, adj)

    def forward(self, x, adj, y, norm=True):
        x, y_hat = self.gc1(x, adj, y, norm)
        x = F.relu(x)
        x, y_hat = self.gc2(x, adj, y_hat, norm)
        return F.log_softmax(x, dim=1), F.log_softmax(y_hat,dim=1)


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

    def forward(self, X, A, K=1, activation='relu', norm=True):

        # shape of H = number of nodes x number of features
        h = X

        for k in np.arange(K):
            h = self.agg(h, A, self.W, activation)

        h = F.relu(self.gc1(h, A))
        X = self.gc2(h, A)

        return F.log_softmax(h, dim=1)
