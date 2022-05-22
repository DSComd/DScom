
'''
================================================
           Hyperparameters
================================================
'''
from Hyperparameters import Num_Chains, Window_length_half, v, Feature_concat, W, Negative_sample_ratio, Scalar_factor, Offset, Random_Seed, Diff_Model
file = open("ID.txt","r")
test_id = file.readline()[:-1]
print("test_id:",test_id,"[END]")


'''
================================================
           Dataset Loading
================================================
'''

import numpy as np

node_feature = np.loadtxt(open("./dataset_tbu/node_features.txt"),dtype=int,delimiter=" ",skiprows=0)
adjacency = np.loadtxt(open("./dataset_tbu/edges.txt"),dtype=int,delimiter=" ",skiprows=0)

'''
==================================================
   Parameters in Diffusion Chain Generation
==================================================
'''

import networkx as nx
adjacency = nx.to_dict_of_lists(nx.from_edgelist(adjacency))
        
np.random.seed(Random_Seed)

num_nodes = len(node_feature)
num_features = len(node_feature[0])

for i in range(num_nodes):
    if not (i in adjacency):
        adjacency.update({i:[]})

if not (v is None):
    v = np.array(v)
    Feature_concat = len(v)

if W is None:
    W = np.random.random((Feature_concat, 2*len(node_feature[0])))
    
if v is None:
    v = np.random.random(Feature_concat)

'''
============================================
        Diffusion Chain Generation
============================================
'''

'''
def softmax(x):
    x -= np.max(x, keepdims = True) 
    x = np.exp(x) / np.sum(np.exp(x), keepdims = True)
    return x
'''

def sigmoid(x):
    return 1./ ( 1. + np.exp(-x) )


# General Diffusion Model
def Gen_IC(adjacency):
    probabilities = []
    for u in range(num_nodes):
        h_u = node_feature[u]
        x = np.empty(len(adjacency[u]))
        
        for i in range(len(adjacency[u])):
            h_w = node_feature[adjacency[u][i]]
            x[i] = v.transpose() @ np.tanh(W @ np.append(h_u,h_w))
            
        x = sigmoid(Scalar_factor * x + Offset).tolist()
    
        probabilities = probabilities + [x]
        
    return probabilities

# Naive IC Model
def IC(adjacency):
    probabilities = []
    for u in range(num_nodes):
        x = np.ones(len(adjacency[u])) * 1./len(adjacency[u])
        probabilities = probabilities + [x]
    return probabilities


if (Diff_Model=="General_IC"):
    probabilities = Gen_IC(adjacency)
else:
    probabilities = IC(adjacency)


print(Diff_Model)


import queue
import random
random.seed(Random_Seed)

thr = np.random.uniform(0,1,size=num_nodes)

def BFS_diffusion(seed):
    q = queue.Queue()
    endpoints = queue.Queue()
    prev = np.zeros(num_nodes,dtype=int)
    weight = np.ones(num_nodes,dtype=float)
    visited = np.zeros(num_nodes,dtype=bool)
    
    samples = None
    q.put_nowait(seed)
    prev[seed] = -1
    visited[seed] = True
    while (not q.empty()):
        u = q.get_nowait()
        x = probabilities[u]
        tmp = 0
        tr = random.uniform(0.9,1)
        
        if (Diff_Model=="LT"):
            for i in range(len(adjacency[u])):
                weight[adjacency[u][i]] += x[i]*tr
                if (not visited[adjacency[u][i]]) and (weight[adjacency[u][i]]>thr[adjacency[u][i]]):
                    node = adjacency[u][i]
                    q.put_nowait(node)
                    prev[node] = u
                    visited[node] = True            # Activate node.
                    tmp = tmp+1
                
        
        else:
            for i in range(len(adjacency[u])):
                r = random.uniform(0,1)
                if ((not visited[adjacency[u][i]]) and r<x[i]):
                    node = adjacency[u][i]
                    q.put_nowait(node)
                    prev[node] = u
                    visited[node] = 1.            # Activate node.
                    tmp = tmp+1
                
                    # Single-Way Pairs
                    """
                    count = 0
                    nghbr = prev[node]
                    while ((nghbr>-1) and (count<Window_length)):
                        if (samples is None):
                            samples = [[nghbr,node]]
                        else:
                            samples = np.append(samples,[[nghbr,node]],axis=0)
                        nghbr = prev[nghbr]
                        count = count + 1
                    
                    # Generate similar pairs
                    """
        
               
        if (tmp==0):
            endpoints.put_nowait(u)
    
    # Double-Way Pairs
    visited = np.zeros(num_nodes,dtype=bool)
    while (not endpoints.empty()):
        u = endpoints.get_nowait()
        chain = [u]
        u = prev[u]
        while (u>-1):
            chain = chain + [u]
            u = prev[u]
        for i in range(len(chain)):
            if (not visited[chain[i]]):
                pos_ = chain[i-Window_length_half:i] + chain[i+1:i+Window_length_half+1]
                visited[chain[i]] = True
                if (len(pos_)==0):
                    continue
                if (samples is None):
                    samples = [[chain[i]] + pos_]
                else:
                    samples = samples + [[chain[i]] + pos_ ]
                    
    return samples

pairs = None

while (pairs is None):
    node_ = random.randint(0,num_nodes-1)
    pairs = BFS_diffusion(node_)

num_ = len(pairs)
while (num_<Num_Chains):
    node_ = random.randint(0,num_nodes-1)
    new_pairs = BFS_diffusion(node_)
    if (new_pairs is None): continue
    pairs = pairs + new_pairs
    num_ = num_ + len(new_pairs)
    
#pairs = pairs[:Num_Chains]
#print(pairs)

pairs = pairs[:Num_Chains]

'''
=================================================
          Diffusion Chains Save
=================================================
'''

import pandas as pd
df = pd.DataFrame(pairs)
df.to_csv('./Pairs.csv',index=False)
df.to_csv('./example/'+test_id+'/Pairs.csv',index=False)

df = pd.DataFrame(W)
df.to_csv('./diffusion_model_W.csv',index=False)
df.to_csv('./example/'+test_id+'/diffusion_model_W.csv',index=False)

df = pd.DataFrame(v)
df.to_csv('./diffusion_model_v.csv',index=False)
df.to_csv('./example/'+test_id+'/diffusion_model_v.csv',index=False)


'''
=============================================
    Explanation of the Output Files
=============================================

[diffusion_model_W.csv]
    the W parameter in our generalized diffusion model.
    
[diffusion_model_v.csv]
    the v parameter in out generalized diffusion model.

[Pairs.csv]
    For each column, rows with positive indices contains
        several nodes close to the node in row 0 on the 
        diffusion chains.
    Let pairs(i)(0) = u. Then for any j s.t. pairs(i)(j)=v
        is not None, pair(u,v) is a positive sample.
'''