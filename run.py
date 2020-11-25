#!/usr/bin/env python
# coding: utf-8

# In[25]:
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing as prep
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('target', type=str, default=None)
args = parser.parse_args()


# In[26]:

try:
        d1 = (pd.read_csv('teams/DSC180A_FA20_A00/b01graphdataanalysis/cora.content', sep ='\t', header=None))
        d2 = pd.read_csv('teams/DSC180A_FA20_A00/b01graphdataanalysis/cora.cites', sep ='\t', header=None)
except:

    d1 = pd.read_csv('data/cora.content', sep ='\t', header=None)
    d2 = pd.read_csv('data/cora.cites', sep ='\t', header=None)

# In[27]:

def main():

    print('Args', args.target)
    # -- My Implementation ---
    d1 = (pd.read_csv('data/cora.content', sep='\t', header=None))
    d2 = pd.read_csv('data/cora.cites', sep='\t', header=None)
    if args.target == 'test':
        print('Test')
        d1 = pd.DataFrame(data=np.arange(0, 10))
        d1 = pd.concat([d1, pd.DataFrame(np.random.randint(2, size=(10, 1433)))], axis=1)
        d1 = pd.concat([d1, pd.DataFrame(np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']))], axis=1)
        d1.columns = np.arange(0, 1435)

        d2 = pd.DataFrame(np.random.randint(10, size=[50, 2]))

    else:
        d1 = (pd.read_csv('data/cora.content', sep='\t', header=None))
        d2 = pd.read_csv('data/cora.cites', sep='\t', header=None)

    # Label Encoder
    le = prep.LabelEncoder()
    le.fit(d1[1434])
    d1[1434] = le.transform(d1[1434])

    # Feature Matrix and Labels
    d1 = d1.set_index(0)
    d1 = d1.sort_index()
    d1 = d1.reset_index()
    labels = d1[1434]
    labels = torch.Tensor(labels).long()
    d1 = d1.drop(columns=[0, 1434])

    X = np.array(d1)

    # Create label distibution for LPA
    labels_distr = np.zeros([len(labels), len(le.classes_)])
    for row in range(len(labels)):
        labels_distr[row][labels[row]] = 1

    # Adjacency matrix
    d2 = pd.crosstab(d2[0], d2[1])
    idx = d2.columns.union(d2.index)
    d2 = d2.reindex(index = idx, columns = idx, fill_value=0)
    d2 = d2.reset_index().drop(columns=['index'])
    d2.columns = np.arange(len(d2.columns))

    A = np.array(d2)

    # Make Train/Test Sets
    train_idx = list(d2.sample(frac=.9).index)
    test_idx = list(set(d2.index) - set(train_idx))

    train_A = d2.loc[train_idx, train_idx]
    train_X = d1.loc[train_idx]
    train_Y = labels[train_idx]

    test_A = d2.loc[test_idx, test_idx]
    test_X = d1.loc[test_idx]
    test_Y = labels[test_idx]

    # Accuracy Function
    def accuracy(output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    # Aggregators
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
            ...

        def forward(self, x, neigh):
            ...

    # GraphSAGE Models and Layers

    class GS_Layer(torch.nn.Module):
        '''
        GraphSAGE Layer
        '''

        def __init__(self):
            super(GS_Layer,self).__init__()
            ...

        def forward(self, X, steps, A):
            # X: batch of nodes
            # steps: steps from node for neighborhood
            # A: adjacency matrix to find nodes in neighborhood
            ...

    class GS(torch.nn.Module):
        def __init__(self, nfeat, nhid, nclass):
            """
            GraphSAGE Model
            """
            super(GCN, self).__init__()

            '''
            gc1 = Aggregator(batch X, neighborhood of X)
            gc1 = Layer
            gc2 = Aggregator(batch X, neighborhood of X)
            gc2 = Layer
            '''

        def forward(self, X, A, prep_A):
            X = F.relu(self.gc1(X, A, prep_A))
            #x = F.relu(self.gc2(x, adj))
            X = self.gc2(X, A, prep_A)
            return F.log_softmax(X, dim=1)

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
                D_hat = torch.Tensor(np.diag(A_hat.sum(axis=1) ** (-1/2)))
                output = torch.mm(D_hat, A_hat)
                output = torch.mm(output, D_hat)
                output = torch.mm(output, right_term)


            return output

    class GCN(torch.nn.Module):
        def __init__(self, nfeat, nhid, nclass):
            """
            Simple GCN Model
            """
            super(GCN, self).__init__()

            # GCN Layers
            self.gc1 = GCN_Layer(nfeat, nhid)
            self.gc2 = GCN_Layer(nhid, nclass)
            #self.gc3 = GCN_Layer(nhid-300, nclass)

        def forward(self, X, A, prep_A):
            """
            """

            X = F.relu(self.gc1(X, A, prep_A))
            #x = F.relu(self.gc2(x, adj))
            X = self.gc2(X, A, prep_A)
            return F.log_softmax(X, dim=1)

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

    print('LPA-GCN')
    GCN_LPA_model = GCN_LPA(X.shape[1], 300, len(le.classes_), A )
    optimizer = torch.optim.SGD(GCN_LPA_model.parameters(), lr=.1)
    criterion = torch.nn.CrossEntropyLoss()

    Lambda = .4
    epochs = 10
    for epoch in np.arange(epochs):
        t = time.time()
        GCN_LPA_model.train()
        optimizer.zero_grad()
        output, Y_hat = GCN_LPA_model(X, A, labels_distr)

        loss_gcn = criterion(output[train_idx], labels[train_idx])
        loss_lpa = criterion(Y_hat[train_idx], labels[train_idx])

        acc = accuracy(output[train_idx], labels[train_idx])
        loss_train = loss_gcn + Lambda * loss_lpa

        loss_train.backward()
        optimizer.step()

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc.item()),
              'time: {:.4f}s'.format(time.time() - t))

    # Model and Optimizer

    # GCN takes in number of papers, number hidden layers, and number of classes
    model = GCN(X.shape[1], 300, len(le.classes_))

    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    criterion = torch.nn.CrossEntropyLoss()

    # Train and Test functions
    def train(epoch, prep_A = None):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(X, A, prep_A)
        loss = criterion(output[train_idx], labels[train_idx])
        acc = accuracy(output[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(acc.item()),
              'time: {:.4f}s'.format(time.time() - t))

    def test(prep_A = None):
        model.eval()
        output = model(X, A, prep_A)
        loss_test = criterion(output[test_idx], labels[test_idx])
        acc_test = accuracy(output[test_idx], labels[test_idx])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    # Train model
    def run(epochs, prep_A):
        t_total = time.time()
        for epoch in range(epochs):
            train(epoch, prep_A)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        test(prep_A)

    print('GCN')
    run(30, 'norm')


main()






