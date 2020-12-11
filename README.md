## DSC180A PyTorch implementation and benchmark of GCN variants

**About:**  
  This repository contains the implementation and the benchmarks results of the classical GCN, GraphSAGE, and LPA-GCN on the citation networks Cora and OGB-arXiv.  
  
**Setting Up Docker Image**  
  A docker image has been created to run the required environment of this project. The docker repository is:      https://hub.docker.com/repository/docker/sdevinl/dsc180a
  
**Example run.py usage:**    
  python run.py cora 100 norm  

**Basic Parameters:**  
  data:  The input citation data such as cora, ogb, ogbsample, and test (default: test)  
  epochs:  Number of epochs (default: 10)  
  predA: Adjacency matrix normalization such as none and norm (default: norm)  
  
 **Output:**  
  The output can be found in test directory. It will contain the outputs of each of the different models for each data set.


