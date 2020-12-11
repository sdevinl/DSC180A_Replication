# DSC180A A PyTorch Benchmark of GCN, GraphSAGE, and LPA-GCN on Citation graph networks.

**Basic Parameters:**

  data:  The input citation data such as cora, ogb, ogbsample, and test (default: test)
  epochs:  Number of epochs (default: 10)  
  predA: Adjacency matrix normalization such as none and norm (default: norm)  

**Example Usage:**
  python run.py cora 100 norm  
