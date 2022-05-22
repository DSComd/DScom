import networkx as nx
import random
  
G = nx.erdos_renyi_graph(500, 1/100)

f = open("ER_graph.txt", "w")
for edge in G.edges:
    print(edge[0], edge[1], random.random()/10, file=f)
f.close()
