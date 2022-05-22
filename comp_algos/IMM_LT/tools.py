import random
from Graph import Graph

def readGraph_direct(path):
    # Initialization
    
    parentss = {}
    children = {}
    edges = {}
    nodes = set()
    
    # Line by line read in edges
    f = open(path, 'r')
    for line in f.readlines():
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue
        row = line.split()
        src = int(row[0])
        dst = int(row[1])
        nodes.add(src)
        nodes.add(dst)
        if children.get(src) is None:
            children[src] = set()
        if parentss.get(dst) is None:
            parentss[dst] = set()
        edges[(src, dst)] = float(row[2])
        children[src].add(dst)
        parentss[dst].add(src)
     
    # # Set naive IC edge weight
    # for edge in edges:
    #     dst = edge[1]
    #     edges[edge] = 1 / len(parentss[dst])
    return Graph(nodes, edges, children, parentss)

def readGraph_undirect(path):
    parentss = {}
    children = {}
    edges = {}
    nodes = set()
    f = open(path, 'r')
    for line in f.readlines():
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue
        row = line.split()
        src = int(row[0])
        dst = int(row[1])
        nodes.add(src)
        nodes.add(dst)
        if children.get(src) is None:
            children[src] = set()
        if children.get(dst) is None:
            children[dst] = set()
        if parentss.get(src) is None:
            parentss[src] = set()
        if parentss.get(dst) is None:
            parentss[dst] = set()
        edges[(src, dst)] = 0
        edges[(dst, src)] = 0
        children[src].add(dst)
        children[dst].add(src)
        parentss[src].add(dst)
        parentss[dst].add(src)
    for edge in edges:
        dst = edge[1]
        edges[edge] = 1 / len(parentss[dst])
    return Graph(nodes, edges, children, parentss)

def chunkIt(list, n):
    avg = len(list) / float(n)
    out = []
    last = 0.0
    while last < len(list):
        out.append(list[int(last):int(last + avg)])
        last += avg
    return out

