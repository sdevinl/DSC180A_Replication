## DSC180A PyTorch implementation and benchmark of GCN variants

**About:**  
  This repository contains the implementation and the benchmarks results of the classical GCN, GraphSAGE, and LPA-GCN on the citation networks Cora and OGB-arXiv.

**Basic Parameters:**  
  data:  The input citation data such as cora, ogb, ogbsample, and test (default: test)
  epochs:  Number of epochs (default: 10)  
  predA: Adjacency matrix normalization such as none and norm (default: norm)  

**Example Usage:**    
  python run.py cora 100 norm  
