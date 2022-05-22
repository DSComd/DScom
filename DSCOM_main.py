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

node_feature = torch.tensor(node_feature).float()    
edge = torch.tensor(adjacency, dtype=torch.long)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Data(x=node_feature, edge_index = np.transpose(edge)).to(device)


'''
==================================================
      Generated Diffusion Chain Extraction
==================================================
'''

import pandas as pd
df = pd.read_csv('./Pairs.csv')
df = df.values.tolist()
pairs = None

for i in df:
    new_pairs = None
    for j in range(1,len(i)):
        if np.isnan(i[j]):
            break
        if new_pairs is None:
            new_pairs = [int(i[j])]
        else:
            new_pairs = new_pairs + [int(i[j])]
    new_pairs = [int(i[0]), new_pairs]
    if pairs is None:
        pairs = [new_pairs]
    else:
        pairs = pairs + [new_pairs]


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

    
'''
============================================
            Negative Sampling.
============================================
'''
    
num_samples = len(pairs)
import random

def negative_sampling(pairs):
    samples = None
    for i in range(num_samples):
        new_batch = pairs[i]
        batch_size = len(new_batch[1])
        w = new_batch[0]
        negs_ = None
        for j in range(batch_size*Negative_sample_ratio):
            u = w
            while (u==w):
                u = random.randint(0,num_nodes-1)
            if (negs_ is None):
                negs_ = [u]
            else:
                negs_ = negs_ + [u]
            
        new_batch = new_batch + [negs_]
        
        if (samples is None):
            samples = [new_batch]
        else:
            samples = samples + [new_batch]
        
    return samples




'''
=========================================
                Train.
=========================================
'''

from math import isnan, isinf

# Training Loop
def Train():
    
    global min_loss, model_backup, model
    
    for epoch in range(Training_epoch):
        model.train()
        optimizer.zero_grad()
        out_emb, (new_edge, alpha) = model(data)
        
        #print(out_emb[120])
        
        train_samples = negative_sampling(pairs)
        
        #acc = 0
        #count = 0
        
        loss = torch.tensor(0, dtype=torch.float)
        for batch in train_samples:
            curr = torch.tensor(batch[0], dtype=torch.long)
            pos = torch.tensor(batch[1], dtype=torch.long)
            neg = torch.tensor(batch[2], dtype=torch.long)
            loss = torch.add(loss, torch.sum( - torch.log( torch.sigmoid( torch.clamp(out_emb[pos]@out_emb[curr],max=5,min=-5) ) ) ) )         
            if isnan(loss) or isinf(loss):
                print("========\n\tNan/Inf Loss Encountered in Positive Samples.")
                print(pos)
                print(out_emb[pos])
                print(torch.clamp(out_emb[pos]@out_emb[curr],min=-5,max=5))
                print(torch.sigmoid( torch.clamp(out_emb[pos]@out_emb[curr],min=-5,max=5) ) )
                return True, -1
            
            loss = torch.add(loss, torch.sum( - torch.log( 1. - torch.sigmoid( torch.clamp(out_emb[neg]@out_emb[curr],max=5,min=-5) ) ) ) )
            if isnan(loss) or isinf(loss):
                print("========\n\tNan/Inf Loss Encountered in Negative Sampling.")
                print(torch.clamp(out_emb[neg]@out_emb[curr],min=-5,max=5))
                print( 1. - torch.sigmoid( torch.clamp(out_emb[neg]@out_emb[curr],min=-5,max=5) ) )
                return True, -1
            
        if (epoch%100 == 0):
            print(epoch)
            print(loss)
            print(out_emb)
            #print(acc/count)
        
        if (min_loss>loss):
            min_loss = loss
            model_backup = model
        
        loss.backward()
        
        optimizer.step()
        
        
    return False, loss

# End of "Train"
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
model = GAT().to(device)
data = dataset.to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=5e-4)

min_loss = float("inf")
model_backup = None
p = True
while (p):
    p, loss = Train()

model = model_backup
out_emb, (new_edge, alpha) = model(data)

torch.save(model,"./model.pth")

print(min_loss)
print(out_emb)


'''
=====================================
            Time Record.
=====================================
'''

dscom_time = time.time() - dscom_time
