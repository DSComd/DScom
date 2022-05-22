import time
import math

from random import random
from copy import deepcopy

import networkx as nx

# for later use
from degreeDiscount import degreeDiscountIC2
from randomHeuristic import randomHeuristic

from func import CELF


class Env:
    ''' The General ENV, namely the graph
    Input: file -- address of the graph data
    Input: budget -- size of budget
    Comment: the structure in which we run our model
    '''
    def __init__(
    self,
    file,
    budget = 10 # 50
    ):
        self.netInput = []
        self.graph ={}
        self.seed = set()

        self.budget = budget
        self.graph = read_from_txt(file)
        self.graph_ = deepcopy(self.graph)
        # Virtual graph
        # Node first tested on it
        # If accepted, replace graph with graph_

        self.list = strength_list(self.graph,self.budget)
        self.maxGain = self.list[0][1]

        #print(self.graph)
    def update_graph(self):
        '''
        Replace graph by graph_
        '''
        self.graph = self.graph_

    def backup_graph(self):
        '''
        Update the graph_
        '''
        self.graph_ = deepcopy(self.graph)

    def list(self):
        '''
        Output: list of nodes ordered by degree with marginal gain of each
        '''
        return self.list

    def steps(self, step, A, T): # accept or reject node
        '''
        Make action, steps to next stage
        Input: step -- index of node stepped
        Input: A -- bool, if this node is really added into seed set
        Input: T -- bool, if we are trainning
        Output: R -- Reward, marginal gain of a node, later used as machine Reward
        '''
        node = {self.list[step][0]}

        # No need to cal gain if not training
        if T == 1:
            R =  IC(self.graph, self.seed | node) - IC(self.graph, self.seed)
        else:
            R = 0

        # If this action is really made
        # 1. Add the node into seed set
        # 2. Update the graph (Collided Nodes)
        if A == 1:
            self.seed = self.seed | node
            self.update_graph()

        return R

    def reward(self,seed):
        '''
        Run IC and gain of whole seed set on graph
        '''
        return IC(self.graph,seed)

    def node2feat(self,step):
        '''
        Covert a node index to its attributes: individual influence and collision
        backup graph in this stage, since it's the first step to work
        Input: step -- index of current node in self.list
        Output (private): self.netInput
        1. influence ratio
        2. collision ratio (self not included)
        Output: influence -- influence of individual node
        Output: coll -- collision of the node with seed set
        '''
        # Preparation
        # Update the graph_
        print("Feature gen")
        node = {self.list[step][0]}
        self.backup_graph()

        influence,coll,inf,touch = collision(self.graph,self.graph_,node, 0) # inf and influence?
        self.netInput = [touch/(influence - 1),inf/(influence-1),coll/(influence-1)] # ?netInput
        print("Feature generated")
        return influence,coll

def read_from_txt(filename, p = 0.01):
    """
    From TXT file read node pairs and assign active probability to edge by random
    Input: filename -- TXT file address
    Input: p -- weight of each edge
    Output: g, a graph in dict
    """
    print("start text read")
    g = {}
    with open(filename) as f:
        lines = f.readlines()[1:]
        for line in lines:
            #e = [int(s) for s in line.replace('\n', '').split(' ')]
            line = line.replace('\t',' ')
            e = [int(s) for s in line.replace('\n', '').split(' ')]
            p = random() * 0.1 # average edge weight
            r = 1 - p # r?

            if e[0] in g.keys():
                if e[1] in g[e[0]].keys():
                    x = g[e[0]][e[1]]
                    g[e[0]][e[1]] = 1 + (x-1)*r
                else:
                    g[e[0]][e[1]] = p
            else:
                g[e[0]] = {"On": 0}
                g[e[0]][e[1]] = p

            if e[1] in g.keys():
                if e[0] in g[e[1]].keys():

                    x = g[e[1]][e[0]]
                    g[e[1]][e[0]] = 1 + (x-1)*r
                else:
                    g[e[1]][e[0]] = p
            else:
                g[e[1]] = {"On": 0}
                g[e[1]][e[0]] = p
    print("Text read")
    return g
# not read

def collision(graph,graph_,Node, R): # graph_: duplication of graph
    """
    calculate a new node's collison with seed set
    :param g: the graph
    :param A: new seed
    :param R: if this collison should be recorded
    :return: influence
    """
    print("start coll")
    res = 0.0
    coll = 0.0
    inf = 0.0
    touch = 0.0
    IC_ITERATION = 100

    # if R ==0, record everything on graph_, the backup graph
    if R == 0:
        g = graph_
    else:
        g = graph

    # recorded influence and collision of individual node
    for _ in range(IC_ITERATION):
        total_activated_nodes = set()
        for u in Node:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    for w, weight in g[v].items():
                        r = random() # decide whether is activated
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            g[w]["On"] += -1
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)
            res += len(total_activated_nodes)

    # record "On" nodes, the potentially activated nodes
    # 2nd step of collision algorithm, calculate the three ratios after H simulations
    for _ in range(10):
        total_activated_nodes = set()
        for u in Node:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    for w, weight in g[v].items():
                        r = random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            if g[w]["On"] <= -IC_ITERATION * 1.2:
                                coll+= 1
                            if g[w]["On"] <= -IC_ITERATION * 0.5 and g[w]["On"] >= -IC_ITERATION:
                                inf += 1
                            if g[w]["On"] >= -IC_ITERATION * 0.5 and g[w]["On"] <= 0:
                                touch += 1
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)

    # end = time.time()
    print("end coll")
    return res / IC_ITERATION, (coll) /10, inf/10, touch/10 #to include itself


def IC(g, A): # res: total activated nodes; but this algorithm returns an average of res
    """
    calculate seed set's influence in IC（Independent Cascade）model
    :param g: a graph in dict
    :param A: seed nodes in set
    :return: influence
    """
    res = 0.0
    IC_ITERATION = 1000
    for _ in range(IC_ITERATION):
        total_activated_nodes = set()
        for u in A:
            l1 = {u}
            l2 = set()
            failed_nodes = set()
            activated_nodes = {u}
            while len(l1):
                for v in l1:
                    #print(g[v].items(),flush=True)
                    for w, weight in g[v].items():
                        #print(w,weight,flush=True)
                        r = random()
                        # each node has only one chance to be activated and should only be actived once
                        if w not in failed_nodes and w not in activated_nodes and r < weight:
                            l2.add(w)
                            activated_nodes.add(w)
                        else:
                            failed_nodes.add(w)
                l1 = l2
                l2 = set()
            total_activated_nodes.update(activated_nodes)
        res += len(total_activated_nodes)
    # print("IC", res/IC_ITERATION)
    return res / IC_ITERATION


def strength_list(g,budget): # what does A stands for?
    '''
    Input: budget -- size of seed set
    Return: A -- budget*50 nodes, ordered by degree and individual influence attached
    '''
    G = nx.Graph()
    # -----changed file path-------
    with open(r'/content/drive/MyDrive/data/NetPHY.txt') as f:
        f.readline()
        for line in f:
            u, v = list(map(int, line.split()))
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)

    S =degreeDiscountIC2(G, budget*20)
    T =randomHeuristic(G, budget*20, p=.01)
    print(IC(g,S[0:budget]))
    print(IC(g,T[0:budget]))

    max_increment = {i:IC(g, {i}) for i in g.keys()}
    t = sorted(max_increment.items(), key=lambda x:x[1], reverse=True)
    A = t[0:budget*20]
    return A;

def strength_list(g,budget):
    '''
    Input: budget -- size of seed set
    Return: A -- budget*50 nodes, ordered by degree and individual influence attached
    '''
    print("started strength list")
    G = nx.Graph()
    # -----changed file path-------
    with open(r'/content/drive/MyDrive/data/NetPHY.txt') as f:
        f.readline()
        for line in f:
            u, v = list(map(int, line.split()))
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)

    S =degreeDiscountIC2(G, budget*20)
    T =randomHeuristic(G, budget*20, p=.01)
    print("1")
    print(IC(g,S[0:budget]))
    print("2")
    print(IC(g,T[0:budget]))

    time1 = time.time()
    max_increment = {i:IC(g, {i}) for i in g.keys()}
    t = sorted(max_increment.items(), key=lambda x:x[1], reverse=True)
    A = t[0:budget*20]
    print('time cost is:',time.time() - time1)
    return A;

def OtherTime(g,budget):
    print("started log")
    G = nx.Graph()
    # -----changed file path-------
    with open(r'/content/drive/MyDrive/data/NetPHY.txt') as f:
        f.readline()
        for line in f:
            u, v = list(map(int, line.split()))
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u,v, weight=1)
    # -----changed file path-------
    log = open(r"/content/drive/MyDrive/V&A/ExpRes/0.1_phy_10_70_Q&A.txt","w")
    print("Degree",file = log, flush=True)
    for i in range(10,30,10): # 10, 80, 10
        start = time.time()
        S =degreeDiscountIC2(G, i)
        Time = time.time() - start
        print(i,' ',IC(g,S),' ',Time,' ', file = log,flush=True)
    print("Degree finished")


    print("Random",file = log, flush=True)
    for i in range(10,30,10):
        start = time.time()
        S =randomHeuristic(G, i)
        Time = time.time() - start
        print(i,' ',IC(g,S),' ',Time,' ', file = log,flush=True)
    print("Random finished")

    # print("CELF",file = log, flush=True)
    # for i in range(10,30,10):
    #    start = time.time()
    #    A,B=CELF(g, i,log)
    #    Time = time.time() - start
    #    print(i,' ',B,' ',Time,' ', file = log,flush=True)
    # print("CELF finished")

    print("V&A",file = log, flush=True) # !this step, no print
    print("VA finished")
    log.close()

#test = Env(r"/content/drive/MyDrive/data/NetPHY.txt",60)
#print(test.list)
#print(test.graph)
#print(CELF.CELF_plus_plus(test.graph, 300))
#test = Env(r"C:\Users\siwei\OneDrive\CNA\大创\complex_network_course-master\Homework_4\DBLP.txt")
#print(CELF.CELF_plus_plus(test.graph, 300))
