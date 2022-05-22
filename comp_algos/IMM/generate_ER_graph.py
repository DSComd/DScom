import networkx as nx
import random
  
G = nx.erdos_renyi_graph(500, 1/200)

f = open("ER_graph.txt", "w")
for edge in G.edges:
    print(edge[0], edge[1], random.random(), file=f)
f.close()
