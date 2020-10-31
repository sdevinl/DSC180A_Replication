#!/usr/bin/env python
# coding: utf-8

# In[25]:


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
#import scipy.sparse as sp


# In[26]:

try:
        feature_list = (pd.read_csv('/teams/DSC180A_FA20_A00/b01graphdataanalysis/cora.content', sep ='\t', header=None))
        edge_list = pd.read_csv('/teams/DSC180A_FA20_A00/b01graphdataanalysis/cora.cites', sep ='\t', header=None)
        keys = pd.read_csv('/teams/DSC180A_FA20_A00/b01graphdataanalysis/cora.content', sep ='\t', header=None)[[0,1434]]
except:
        feature_list = pd.read_csv('/data/cora.content', sep ='\t', header=None)
        edge_list = pd.read_csv('/data/cora.cites', sep ='\t', header=None)

# In[27]:


##Data Ingestion
labels = feature_list[1434]

index_key = {
        'Case_Based': 0,
        'Genetic_Algorithms': 1,
        'Neural_Networks': 2,
        'Probabilistic_Methods': 3,
        'Reinforcement_Learning': 4,
        'Rule_Learning': 5,
        'Theory': 6
}

##One Hot Labels
one_hot_labels = []
for i in labels:
    to_append = np.zeros(7)
    label_index = index_key[i]
    to_append[label_index] = 1
    one_hot_labels.append(to_append)
one_hot_labels = np.array(one_hot_labels)


##Label Generator
label_list = []
for i in labels:
    to_append = np.zeros(7)
    label_index = index_key[i]
    label_list.append(label_index) 
    
##Labels & Features to Tensor format    
labels = torch.LongTensor(label_list) 
features = torch.FloatTensor(np.array(feature_list.iloc[:, 1:1434]))


# In[29]:


class BorisGraphNet(nn.Module):
    def __init__(self):
        super(BorisGraphNet, self).__init__()
        self.fc1 = nn.Linear(1433, 500, bias=False)
        self.fc2 = nn.Linear(500, 7, bias=False)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# In[30]:


gcn_model = BorisGraphNet()
gcn_optimizer = optim.SGD(gcn_model.parameters(), lr=.03)#, weight_decay=1e-1)


# In[45]:


def accuracy(output, labels):
    prediction = output.max(1)[1].type_as(labels)
    correct = (prediction.eq(labels).double()).sum()
    return correct / len(labels)


# In[46]:


def train_gcn(epoch):
    gcn_model.train()
    gcn_optimizer.zero_grad()
    output = gcn_model(features)
    loss = F.cross_entropy(output, labels)
    acc = accuracy(output, labels)
    loss.backward()
    gcn_optimizer.step()
    if (epoch + 1) % 50 == 0:
        print('Epoch: {}'.format(epoch+1),
              'loss: {:.4f}'.format(loss.item()),
             'accuracy: {:.4f}'.format(acc.item()))


# In[47]:


def main():
    epochs = 300
    for epoch in range(epochs):
        train_gcn(epoch)


# In[48]:


main()


# In[ ]:




