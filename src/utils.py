from scipy import sparse
import torch
import numpy as np

# Accuracy Function
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def to_adj_list(edges, labels):
    adj = sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sparse.eye(adj.shape[0])
    return matrix_to_tensor(adj)

def cat_encode(label_column):
    d = {}
    count = 0
    for i in label_column.value_counts().index:
        d[i] = count
        count += 1
    return np.array(label_column.apply(lambda x: d[x]))

def matrix_to_tensor(mtx):
    coo = mtx.tocoo().astype(np.float32)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def accuracy(output, labels):
    return ((output.max(1)[1].type_as(labels)).eq(labels).double()).sum() / len(labels)
