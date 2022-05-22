'''
====================================
        Hyperparameters
====================================
'''

from Hyperparameters import Learning_rate, Training_epoch, Out_num_features, Negative_sample_ratio
import time
dscom_time = time.time()

'''
=====================================
        Dataset Loading
=====================================
'''

import numpy as np

node_feature = np.loadtxt(open("./dataset_tbu/node_features.txt"),dtype=int,delimiter=" ",skiprows=0)
adjacency = np.loadtxt(open("./dataset_tbu/edges.txt"),dtype=int,delimiter=" ",skiprows=0)
    
num_nodes = len(node_feature)



'''
===================================================
        Convert Dataset to Torch.Data Type
===================================================
'''

import torch
from torch_geometric.data import Data

# append "degree" and "1/degree" into the feature
import networkx as nx
adj = nx.to_dict_of_lists(nx.from_edgelist(adjacency))

for i in range(num_nodes):
    if not (i in adj):
        adj.update({i:[]})

deg = np.ndarray((num_nodes,2))
for node in range(num_nodes):
    if (len(adj[node])==0):
        deg[node][0] = 0
        deg[node][1] = 0
    else:
        deg[node][0] = len(adj[node])
        deg[node][1] = 1./len(adj[node])
node_feature = np.concatenate((node_feature,deg),axis=1)
    
num_features = len(node_feature[0])
feature_max = np.max(node_feature,axis=0)
feature_min = np.min(node_feature,axis=0)
node_feature = node_feature-feature_min
std = feature_max-feature_min
for i in range(len(std)):
    if (std[i]==0):
        std[i] = 1.
node_feature = node_feature/std
print(node_feature[0])

node_feature = torch.tensor(node_feature).float()    
edge = torch.tensor(adjacency, dtype=torch.long)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Data(x=node_feature, edge_index = np.transpose(edge)).to(device)


'''
=============================================
          Package Use for Model.
=============================================
'''

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


'''
==============================================
       Definition of DSCOM (GAT part).
==============================================
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# GAT Definition
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hidden1 = 8
        self.in_head = 8
        self.out_head = 1
        self.hidden2 = 10
        
        self.conv1 = GATConv(num_features, self.hidden1, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hidden1*self.in_head, self.hidden2, concat=False, heads=self.out_head, dropout=0.6)
        self.mlp = nn.Linear(self.hidden2*self.out_head, Out_num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x, alpha = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.mlp(x)
        x = F.elu(x)
        
        return x, alpha  #F.softmax(x, dim=1), alpha



def seed_select(model, data, Num_Seeds):
    
    out_emb, (new_edge, alpha) = model(data)
    clstr_time  = time.perf_counter()
    
    '''
    =============================================================
            Spectral Clustering (Community Division).
    =============================================================
    '''

    from sklearn.cluster import spectral_clustering

    new_adjacency = np.zeros((num_nodes, num_nodes))
    new_edge = new_edge.long()

    for i in range(len(new_edge[0])):
        new_adjacency[int(new_edge[0][i]),int(new_edge[1][i])] = alpha[i].mean()

    # print(new_adjacency)

    new_adjacency = 0.5*(new_adjacency+new_adjacency.transpose())        # to make sure it is symmetric.


    Num_Clusters = Num_Seeds
    clusters = spectral_clustering(new_adjacency, n_clusters = Num_Clusters)


    # Get the "average" feature of each cluster.     
    centroids = []
    comm_nodes = []

    for i in range(Num_Clusters):
        comm_nodes = comm_nodes + [[x for x in range(len(out_emb)) if clusters[x]==i]]
        clu_feature = torch.mean(out_emb[comm_nodes[-1]], dim=0)
        centroids = centroids + [clu_feature]
    
    # print(centroids)


    # Separate each community.
    communities = []
    for i in range(Num_Clusters):
        sub_nodes = comm_nodes[i]
        x = [x for x in range(len(adjacency)) if (adjacency[x][0] in sub_nodes and adjacency[x][1] in sub_nodes)]
        sub_edges = adjacency[x]
        sub_alpha = alpha[x]
        communities = communities + [[sub_nodes, sub_edges, sub_alpha]]

    """
    ==========================
            Time Record.
    ==========================
    """

    clstr_time = time.perf_counter()-clstr_time
    subtime = time.perf_counter()
    
    """
    ========================================================
            Seed Selection on Different Community
    ========================================================
    """

    def Core_k_truss(subgraph,k_min,k_max):
        if (k_min+1 >= k_max):
            truss = nx.k_truss(subgraph,k_min)
            return truss.nodes
        k = int((k_min+k_max)/2)
        truss = nx.k_truss(subgraph,k)
        if (len(truss)==0):
            return Core_k_truss(subgraph,k_min,k-1)
        if (len(truss)>0):
            return Core_k_truss(subgraph,k,k_max)
        
        
    import networkx as nx
    
    Seeds_mxdg = []      # choose the node with maximal degeree
    Seeds_mxct = []      # choose the node with maximal betweenness-centrality
    Seeds_core = []      # k-core
    Seeds_pgRk = []      # PageRank
    
    time_mxdg = time_mxct = time_core = time_pgRk = time.perf_counter()-subtime

    for i in range(Num_Clusters):
    
        pre_time = time.perf_counter()
    
        subgraph = nx.Graph()
        subgraph.add_nodes_from(communities[i][0])
        sub_edges = communities[i][1]
        sub_alpha = communities[i][2]
        for x in range(len(sub_edges)):
            subgraph.add_edge(int(sub_edges[x][0]),int(sub_edges[x][1]),weight=int(sub_alpha[x].mean()))
        
        # Time record.
        pre_time = time.perf_counter()-pre_time
        subtime = time.perf_counter()
        
        # by max-degree
        core_num = []
        indices = []
        for j in subgraph.nodes:
            core_num.append(subgraph.degree[j])
            indices.append(j)
        core = indices[core_num.index(max(core_num))]
        Seeds_mxdg.append(core)
            
        # Time record.
        time_mxdg += (time.perf_counter()-subtime+pre_time)
        subtime = time.perf_counter()
            
        # by max-closeness-centrality
        core_num = nx.algorithms.centrality.closeness_centrality(subgraph)
        core = max(core_num, key=core_num.get)
        Seeds_mxct.append(core)
    
        # Time record.
        time_mxct += (time.perf_counter()-subtime+pre_time)
        subtime = time.perf_counter()

        
        subgraph = nx.Graph()
        subgraph.add_nodes_from(communities[i][0])
        sub_edges = communities[i][1]
        sub_alpha = communities[i][2]
        for x in range(len(sub_edges)):
            subgraph.add_edge(int(sub_edges[x][0]),int(sub_edges[x][1])) #,weight=int(sub_alpha[x].mean()))
        
    
        # by k-core
        core_num = nx.core_number(subgraph)
        core = max(core_num, key=core_num.get)
        Seeds_core.append(core)
    
        # Time record.
        time_core += (time.perf_counter()-subtime+pre_time)
        subtime = time.perf_counter()
        
        #pageRank
        core_num = nx.pagerank(subgraph, alpha=0.5)
        core = max(core_num, key=core_num.get)
        Seeds_pgRk.append(core)
    
        # TIme record.
        time_pgRk += (time.perf_counter()-subtime+pre_time)
        

    '''
    =========================================
             Time Record & Results.
    =========================================
    '''

    # max-deg-DSCOM
    time_mxdg = clstr_time + time_mxdg
    # Seed selected: Seeds_mxdg

    # max-centrality-DSCOM
    time_mxct = clstr_time + time_mxct
    # Seed selected: Seeds_mxct

    # k-core-DSCOM
    time_core = clstr_time + time_core
    # Seed selected: Seeds_core

    # pageRank-DSCOM
    time_pgRk = clstr_time + time_pgRk
    # Seed selected: Seeds_pgRk


    # print("max-deg:\n",Seeds_mxdg,"\t time=",time_mxdg,end="\n\n")
    # print("max-cent:\n",Seeds_mxct,"\t time=",time_mxct,end="\n\n")
    # print("k-core:\n",Seeds_core,"\t time=",time_core,end="\n\n")
    # print("pageRank:\n",Seeds_pgRk,"\t time=",time_pgRk,end="\n\n")


    return Seeds_mxdg, time_mxdg, Seeds_mxct, time_mxct, Seeds_core, time_core, Seeds_pgRk, time_pgRk

'''
==========================================
==========================================
==========================================

            END of DSCOM
            
==========================================
==========================================
==========================================
'''



'''
==========================================
==========================================
==========================================

        Ablationexperiment
        
==========================================
==========================================
==========================================
'''            


def ablationexperiment(model, data, Num_Seeds, adjacency):
    
    out_emb, (new_edge, alpha) = model(data)
    subtime = time.perf_counter()
    
    
    '''
    ____________________________________
    
    Out_embeddings -> k-Means
    ____________________________________
    
    '''
    
    from sklearn.cluster import k_means
    
    feat = out_emb.detach().to('cpu').numpy()
    centroid, label, _ = k_means(feat, Num_Seeds, init='k-means++')
    
    Seeds_woCom = []
    for i in range(Num_Seeds):
        cluster = [x for x in range(len(label)) if label[x]==i]
        dist = feat[cluster] - centroid[i]
        dist = [np.linalg.norm(x) for x in dist]
        core = cluster[dist.index(min(dist))]
        Seeds_woCom.append(core)


    '''
    ___________________________________________
    
    Time Record & Results.
    ___________________________________________
    '''

    time_woCom = dscom_time + time.perf_counter()-subtime
    # Seed selection is Seeds_woCom
    # print("w.o. Community, k-means:\n",Seeds_woCom,end="\n\n")


    '''
    _____________________________________________
    
    GAT -> IMM
    _____________________________________________
    '''
    
    subtime = time.perf_counter()
    
    inter_file = open("./tmp/GATtoIMM.txt","w")
    for i in range(len(new_edge[0])):
        print("{} {} {:.8f}".format(int(new_edge[0][i]),int(new_edge[1][i]),
                                    alpha[i].mean()),file=inter_file)
    
    from comp_algos.IMM.IMM import GreedyCoverage, Sampling, compute
    import comp_algos.IMM.IMM as tools
    import math
    # Read in graph
    graph = tools.readGraph_direct("./tmp/GATtoIMM.txt")
        
    # Change l to achieve the 1-n^l probability
    l = 3 * (1 + math.log(2) / math.log(num_nodes))
    EPSILON = 0.5
        
    # Sampling RR-sets
    R = Sampling(graph, Num_Seeds, EPSILON, l)
        
    # Greedy algorithm
    Seeds_onGAT = GreedyCoverage(R, Num_Seeds)
    time_onGAT = dscom_time + time.perf_counter()-subtime
        
    
    '''
    ____________________________________
    
    pageRank on original dataset
    ____________________________________
    
    '''
    
    cls_adj = np.zeros((num_nodes,num_nodes))
    for x in adjacency:
        cls_adj[x[0],x[1]] = 1
       
    cls_adj = 0.5*(cls_adj+np.transpose(cls_adj))
    from sklearn.cluster import spectral_clustering
    clusters = spectral_clustering(cls_adj, n_clusters = Num_Seeds)
    
    communities = []
    for i in range(Num_Seeds):
        sub_nodes = [x for x in range(num_nodes) if clusters[x]==i]
        sub_edges = [x for x in adjacency if (x[0] in sub_nodes and x[1] in sub_nodes)]
        communities = communities + [[sub_nodes, sub_edges]]
        
        
    Seeds_clstr_pgRk = []
    import networkx as nx
    for i in range(Num_Seeds):
        subgraph = nx.Graph()
        subgraph.add_nodes_from(communities[i][0])
        for x in communities[i][1]:
            subgraph.add_edge(x[0],x[1])
            
        core_num = nx.pagerank(subgraph, alpha=0.5)
        core = max(core_num, key=core_num.get)
        Seeds_clstr_pgRk.append(core)
        
        
    return Seeds_woCom, Seeds_onGAT, Seeds_clstr_pgRk