'''
=====================================
        Hyperparameters
=====================================
'''

import sys
sys.path.append('..')
from Hyperparameters import Random_Seed

'''
=====================================
        Dataset Loading
=====================================
'''

import numpy as np
import networkx as nx

node_feature = np.loadtxt(open("../dataset_tbu/node_features.txt"),dtype=int,delimiter=" ",skiprows=0)
adjacency = np.loadtxt(open("../dataset_tbu/edges.txt"),dtype=int,delimiter=" ",skiprows=0)

num_nodes = len(node_feature)
num_features = len(node_feature[0])
    
adjacency = nx.to_dict_of_lists(nx.from_edgelist(adjacency))

'''
===========================================
        Diffusion Model Loading
===========================================
'''

from Hyperparameters import Num_Seeds
from Hyperparameters import Scalar_factor, Offset

import pandas as pd
df = pd.read_csv('../diffusion_model_v.csv')
v = np.array(df.values.tolist()).reshape(-1)
df = pd.read_csv('../diffusion_model_W.csv')
W = np.array(df.values.tolist())

'''
===========================================
            Generalized IC
===========================================
'''

def sigmoid(x):
    return 1./ ( 1. + np.exp(-x) )

probabilities = []
for u in range(num_nodes):
    h_u = node_feature[u]
    x = np.empty(len(adjacency[u]))
        
    for i in range(len(adjacency[u])):
        h_w = node_feature[adjacency[u][i]]
        x[i] = v.transpose() @ np.tanh(W @ np.append(h_u,h_w))
            
    x = sigmoid(Scalar_factor * x + Offset).tolist()
    probabilities = probabilities + [x]


import queue
import random
random.seed(Random_Seed)

def General_IC(g,seeds):
    random.seed(random.randint(-2147483648,2147483648))
    
    [node_feature, adjacency] = g
    active = queue.Queue()
    visited = np.zeros(num_nodes,dtype=bool)
    
    for seed in seeds:
        active.put_nowait(seed)
        visited[seed] = True
    
    influence = len(seeds)
    while (not active.empty()):
        u = active.get_nowait()
        x = probabilities[u]
        
        for i in range(len(adjacency[u])):
            r = random.uniform(0,1)
            if ((not visited[adjacency[u][i]]) and r<x[i]):
                node = adjacency[u][i]
                active.put_nowait(node)
                visited[node] = True            # Activate node.
                influence = influence+1
    
    return influence


'''
======================================
         Definition of CELF
======================================
'''

def CELF(g, k):
    """
    CELF算法计算最大影响力的k个节点
    :param g: 图
    :param k: 节点数量
    :return: (最大影响力节点集合（以set存储）, 最大影响力)
    """
    nodes = range(len(g[0]))
    
    V = set(nodes)
    max_increment = {i:General_IC(g, {i}) for i in V}
    t_max_increment = max_increment.copy()
    t = sorted(max_increment.items(), key=lambda x:x[1], reverse=True)
    A = set([t[0][0]])
    max_influence = t[0][1]
    
    del t_max_increment[t[0][0]]
    
    for _ in range(k - 1):
        tmp = 0
        t = sorted(t_max_increment.items(), key=lambda x:x[1], reverse=True)
        for v, _ in t:
            if (tmp==0):
                increment_A_and_v = General_IC(g, A | {v}) - max_influence
                max_increment_current = increment_A_and_v
                max_increment[v] = t_max_increment[v] = increment_A_and_v
                max_increment_node = v
                tmp = 1
            if max_increment[v] > max_increment_current:
                increment_A_and_v = General_IC(g, A | {v}) - max_influence
                if max_increment_current < increment_A_and_v:
                    max_increment_current = increment_A_and_v
                    max_increment[v] = t_max_increment[v] = increment_A_and_v
                    max_increment_node = v
        A.add(max_increment_node)
        print(A)
        max_influence = max_influence + max_increment_current
        del t_max_increment[max_increment_node]
        
    return A, max_influence


'''
==================================================
        Seed Selection Based on CELF-Greedy
==================================================
'''

S, influence = CELF([node_feature, adjacency], Num_Seeds)
S = list(S)
print(S, influence)

import pandas as pd
df = pd.DataFrame(S)
df.to_csv('../Seeds_Selection/CELF_diffusion_known.csv',index=False)