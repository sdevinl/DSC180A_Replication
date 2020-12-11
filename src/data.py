import pandas as pd
import numpy as np
import torch
import networkx as nx
from sklearn import preprocessing as prep


def make_dataset(targets):
    try:
        if targets[0] == 'cora':
            pass
    except IndexError:
        print('No data target found, using test')
        targets[0] ='test'

    if targets[0] == 'cora':
        try:
            d1 = pd.read_csv('teams/DSC180A_FA20_A00/b01graphdataanalysis/cora.content', sep='\t', header=None)
            d2 = pd.read_csv('teams/DSC180A_FA20_A00/b01graphdataanalysis/cora.cites', sep='\t', header=None)

        except:
            d1 = pd.read_csv('data/cora.content', sep='\t', header=None)
            d2 = pd.read_csv('data/cora.cites', sep='\t', header=None)

        testfile = open('test/coradata.txt', 'w')
        testfile.write('')
        testfile.close()
        # New Output
        testfile = open('test/coradata.txt', 'a')

    elif targets[0] == 'test':
        d1 = pd.DataFrame(data=np.arange(0, 10))
        d1 = pd.concat([d1, pd.DataFrame(np.random.randint(2, size=(10, 1433)))], axis=1)
        d1 = pd.concat([d1, pd.DataFrame(np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']))], axis=1)
        d1.columns = np.arange(0, 1435)
        d2 = pd.DataFrame(np.random.randint(10, size=[50, 2]))

        # Clear Text File
        testfile = open('test/testdata.txt', 'w')
        testfile.write('')
        testfile.close()
        # New Output
        testfile = open('test/testdata.txt', 'a')

    elif targets[0] == 'ogb':
        from ogb.nodeproppred import NodePropPredDataset

        d = NodePropPredDataset('ogbn-arxiv', root='/datasets/ogb/ogbn-arxiv')
        d2 = pd.DataFrame(d[0][0]['edge_index'].T)
        d1 = pd.DataFrame(d[0][0]['node_feat'])
        labels = d[0][1]

        # Clear Text File
        testfile = open('test/ogbdata.txt', 'w')
        testfile.write('')
        testfile.close()
        # New Output
        testfile = open('test/ogbdata.txt', 'a')

    elif targets[0] == 'ogbsample':
        from ogb.nodeproppred import NodePropPredDataset

        d = NodePropPredDataset('ogbn-arxiv', root='/datasets/ogb/ogbn-arxiv')
        d2 = pd.DataFrame(d[0][0]['edge_index'].T)
        d1 = pd.DataFrame(d[0][0]['node_feat'])
        labels = d[0][1]

        # Sample of OGB dataset
        d2 = d2.sort_values(0).iloc[:2000]
        partial_idx = list(set(d2[0].unique()) | set(d2[1].unique()))
        d1 = d1.iloc[partial_idx]

        # Clear Text File
        testfile = open('test/ogbdata.txt', 'w')
        testfile.write('')
        testfile.close()
        # New Output
        testfile = open('test/ogbdata.txt', 'a')


    le = prep.LabelEncoder()
    le.fit(d1.iloc[:, -1])
    d1.iloc[:, -1] = le.transform(d1.iloc[:, -1])

    if targets[0] == 'test' or targets[0] == 'cora':
        d1 = d1.set_index(0)
        d1 = d1.sort_index()
        d1 = d1.reset_index()
        labels = d1.iloc[:, -1]

    elif targets[0] == 'ogbsample':
        labels = d[0][1].flatten()[partial_idx]

    else:
        labels = d[0][1].flatten()

    labels = torch.Tensor(labels).long()

    labels_distr = np.zeros([len(labels), len(le.classes_)])
    for row in range(len(labels)):
        labels_distr[row][labels[row]] = 1

    return testfile, d1, d2, labels, targets, labels_distr, le


def preprocess_data(d1, d2, labels, targets):
    # Adjacency matrix
    G = nx.Graph()
    G.add_edges_from(d2.values)
    A = nx.adjacency_matrix(G).toarray()
    d2 = pd.DataFrame(A)

    # Feature Matrix and Labels

    # labels = labels[:2137].flatten()
    # labels = torch.Tensor(labels).long()

    if (targets[0] == 'cora') or (targets[0] == 'test'):
        columns_to_drop = [0, d1.iloc[:, -1].name]

    else:
        columns_to_drop = [d1.iloc[:, -1].name]

    d1 = d1.drop(columns=columns_to_drop)

    X = np.array(d1)

    # Create label distibution for LPA

    # Make Train/Test Sets
    train_idx = list(d2.sample(frac=.9).index)
    test_idx = list(set(d2.index) - set(train_idx))

    if targets[0] == 'ogb':
        X = X.astype('int64')

    return X, A, train_idx, test_idx, labels
