import pandas as pd
import numpy as np
from sklearn import preprocessing as prep
import torch


def make_dataset(targets):
    if len(targets) == 0 or targets[0] != 'test':
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

    return testfile, d1, d2

def preprocess_data(d1, d2):
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
    d2 = d2.reindex(index=idx, columns=idx, fill_value=0)
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

    return X, A, train_idx, test_idx, labels, labels_distr, le


