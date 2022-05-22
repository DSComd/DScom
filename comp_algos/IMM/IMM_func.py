import tools as tools
import random
import copy
import math
from scipy.special import comb

def reverseSearch(graph, root):
    """ Generate a RR-set for node *root* with BFS realized with queue structure. """
    
    # Initialization
    searchSet = set() # Set containing the nodes already searched once to avoid revisit the same node multiple times
    queue = []
    queue.append(root)
    searchSet.add(root)
    
    # BFS
    while len(queue) != 0:
        current_node = queue.pop(0) # Pop out the first element in the list
        parentss = graph.get_parentss(current_node)
        for parent in parentss:
            if parent not in searchSet:
                rate = graph.edges[(parent, current_node)] # Propagation probability in IC model
                if tools.isHappened(rate):
                    searchSet.add(parent)
                    queue.append(parent)
    return searchSet

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
    influence = len(graph.nodes) * counter / len(R)
    return influence

def compute(graph, seeds, N=2000):
    """ Estimate the influence spread with Monte-Carlo approach, repetition time N. """
    influence = 0
    for i in range(N):
        queue = []
        queue.extend(seeds)
        checked = copy.deepcopy(seeds)
        while len(queue) != 0:
            current_node = queue.pop(0)
            children = graph.get_children(current_node)
            for child in children:
                if child not in checked:
                    rate = graph.edges[(current_node, child)]
                    if tools.isHappened(rate):
                        checked.add(child)
                        queue.append(child)
        influence += len(checked)
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