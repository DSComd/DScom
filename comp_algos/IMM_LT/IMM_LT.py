import random
import copy
import math
from scipy.special import comb

class Graph:
    # Set Class Attributes
    nodes = None
    edges = None
    children = None
    parentss = None
    
    def __init__(self, nodes, edges, children, parentss):
        self.n = len(nodes)
        self.nodes = nodes
        self.edges = edges
        self.children = children
        self.parentss = parentss
        self.threshold = {}
        self.accumulate_weight = {}
        
    def get_children(self, node):
        # Get the children set of the node.
        return self.children.get(node, set())
    def get_parentss(self, node):
        # Get the parents set of the node.
        return self.parentss.get(node, set())

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


def reverseSearch(graph, root):
    """ Generate a RR-set for node *root* with BFS realized with queue structure. """
    
    # Initialize the threshold
    for node in graph.nodes:
        graph.threshold[node] = random.random()
        graph.accumulate_weight[node] = 0
    
    # Initialization for new_activate and activated
    current_node = root
    activated = set()
    
    # Sample (See IMM LT part)
    while (current_node not in activated) and (current_node != -1):
        activated.add(current_node)
        parentss = graph.get_parentss(current_node)
        candidates = list(parentss) + [-1]
        weight_list = [graph.edges[(parent, current_node)] for parent in parentss]
        assert 1-sum(weight_list)>=0
        weight_list.append(1-sum(weight_list))
        current_node = random.choices(candidates, weights=weight_list)[0]  
    return activated

def generateRRset(graph):
    """ Generate one random RR-set. """
    # Uniformly sample one node from the node set
    selectedNode = random.randint(1, len(graph.nodes)) # TODO: reset the range according to the dataset node indexes
    
    # Sample a RR-set for selected node
    RRset = reverseSearch(graph, selectedNode)
    return RRset

def RRCollection(graph, theta):
    """ Sample theta times random RR-sets. """
    R = []
    for i in range(theta):
        RR = generateRRset(graph)
        R.append(RR)
    return R

def estimate(graph, seeds, R):
    """ 
    Estimate the influence of a seed set with generated RR-sets, R.
    Return fraction of RR-sets covered by node set seeds.
    """
    counter = 0
    for RR in R:
        if len(seeds & RR) != 0:
            counter += 1
    influence = counter / len(R)
    return influence

def compute(graph, seeds, N=2000):
    """ Estimate the influence spread with Monte-Carlo approach, repetition time N. """
    influence = 0
    for i in range(N):
        for node in graph.nodes:
            graph.threshold[node] = random.random()
            graph.accumulate_weight[node] = 0
        newly_activate = []
        newly_activate.extend(seeds)
        activated = copy.deepcopy(seeds)
        while len(newly_activate) != 0:
            current_node = newly_activate.pop(0)
            children = graph.get_children(current_node)
            for child in children:
                if child not in activated:
                    graph.accumulate_weight[child] += graph.edges[(current_node, child)]
                    if graph.accumulate_weight[child] > graph.threshold[child]:
                        activated.add(child)
                        newly_activate.append(child)
        influence += len(activated)
    influence = influence/N
    return influence

def GreedyCoverage(R, k):
    # Initialization
    seeds = set()
    count_dict = {}  # Record the number of uncovered RR-sets will be covered by each node 
    covered = set()  # Record the RR-sets already covered by seed set
    for RR in R:
        for node in RR:
            if count_dict.get(node) is None:
                count_dict[node] = 1
            else:
                count_dict[node] += 1
    
    # Select k nodes with greedy approach
    for i in range(k):
        seed = max(count_dict, key=count_dict.get) # Select the node covers most RR-sets
        if (count_dict[seed] == 0):  # If all RR-sets have already been covered
            seeds = seeds | set(random.sample(list(set(count_dict.keys()) - seeds), k=k-i))
            break
        seeds.add(seed)
        for j in range(len(R)):  # Update the count_dict and covered after selecting the node
            if (j not in covered) and (seed in R[j]):
                covered.add(j)
                for node in R[j]:
                    count_dict[node] -= 1
                    
    return seeds

def Sampling(g, k, e, l):
    R = []
    LB = 1
    e_ = e * math.sqrt(2)
    lambda_1 = (2+2*e_/3) * (math.log(comb(g.n, k)) + l * math.log(g.n) + math.log(math.log(g.n, 2))) * g.n
    theta_old = 0  # Keep the old samples
    
    for i in range(int(math.log(g.n, 2))):
        x = g.n / (2**i)
        theta = lambda_1 / x
        R.extend(RRCollection(g, math.ceil(theta - theta_old)))
        theta_old = theta
        seeds = GreedyCoverage(R, k)
        fraction = estimate(g, seeds, R)
        if (g.n * fraction >= (1+e_) * x):
            LB = g.n * fraction / (1 + e_)
            break
    
    alpha = math.sqrt(l * math.log(g.n) + math.log(2))
    beta = math.sqrt((1-1/math.e) * (math.log(comb(g.n, k)) + l * math.log(g.n) + math.log(2)))
    lambda_2 = 2 * g.n * ((1 - 1/math.e) * alpha + beta)**2 / (e**2)
    theta = lambda_2 / LB
    R = RRCollection(g, math.ceil(theta))
    
    return R

if __name__ == '__main__':
    # Read in graph
    graph = readGraph_direct(GRAPH_PATH)
    
    # Change l to achieve the 1-n^l probability
    l = l * (1 + math.log(2) / math.log(graph.n))
    
    # Sampling RR-sets
    R = Sampling(graph, k, EPSILON, l)
    
    # Greedy algorithm
    seeds = GreedyCoverage(R, k)
    print(seeds)
    
    for i in range(5):
        influence = compute(graph, seeds)
        print(influence)
    

    # R = RRCollection(graph, theta=20000)
    # influence = estimate(graph, seeds, R)

