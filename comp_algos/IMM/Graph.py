class Graph:
    # Set Default
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
        
    def get_children(self, node):
        # Get the children set of the node.
        return self.children.get(node, set())
    def get_parentss(self, node):
        # Get the parents set of the node.
        return self.parentss.get(node, set())