import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy

from collections.abc import Iterable
from typing import Union
from collections import deque
from itertools import groupby
import random

class Node:
    
    def __init__(self, index: int, neighbors: Union[set[int],list[int],int], phases = None):
        self.index = index
        if type(neighbors) == int:
            self.neighbors = set([neighbors])
        else:     
            self.neighbors = set(neighbors)
        self.degree = len(self.neighbors)
        #fluxes should be a dictionary that takes an index of a neighbor as a key
        #as gives back the phase associated with traveling to that neighbor as an item
        if phases is None or phases is set():
            self.phases = {neighbor: 1j for neighbor in self.neighbors}
        else:
            self.phases = phases

    def add_neighbor(self, new_neighbors: Union[int, Iterable[int]], phase = 1j) -> None:
        """Adds a neighbor or list of neighbors to a Node by their index """
        
        if isinstance(new_neighbors, int) or isinstance(new_neighbors, np.int64):
            self.neighbors.add(new_neighbors)
            self.phases[new_neighbors] = phase
        else:
            for neighbor in new_neighbors:
                if not isinstance(neighbor, int):
                    raise TypeError("All neighbors must be integers.")
                self.neighbors.add(neighbor)
                self.phases[neighbor] = phase

        self.degree = len(self.neighbors)

    def remove_neighbor(self, old_neighbors: Union[int, Iterable[int]]) -> None:
        """Removes a neighbor or list of neighbors to a Node by their index;
           if a given index is not a neighbor, it is ignored"""

        if isinstance(old_neighbors, int) or isinstance(old_neighbors, np.int64):
            self.neighbors.discard(old_neighbors)
        else:
            for neighbor in old_neighbors:
                self.neighbors.discard(neighbor)    
        self.degree = len(self.neighbors)

                
class ConnectedGraph:
    
    def __init__(self, nodes: Union[Node, set[Node],list[Node]]):
        self.nodes = set(nodes)
        self.node_map = {node.index: node for node in self.nodes}
        self.adj = None
        self.fluxed = None #the adjacency matrix with the fluxes

    def construct_adj(self) -> None:
        """
        Constructs adjacency matrix of a graph
        """
        # Get all node indices in a sorted list (so we have a stable ordering).
        sorted_indices = sorted(self.node_map.keys())
        
        # Map each node's index to the row/column in the adjacency matrix.
        index_to_row = {node_index: i for i, node_index in enumerate(sorted_indices)}
        
        n = len(sorted_indices)
        # Initialize an n x n matrix filled with zeros.
        adj_matrix = np.zeros((n,n))
        
        # Fill the adjacency matrix.
        for node_index, node_obj in self.node_map.items():
            i = index_to_row[node_index]
            for neighbor_index in node_obj.neighbors:
                if neighbor_index not in self.node_map:
                    continue
                
                j = index_to_row[neighbor_index]
                adj_matrix[i,j] = 1

        self.adj = adj_matrix

    #TODO
    def construct_fluxed(self) -> None:
        """
        Constructs the adjacency matrix weighted by the phase change associated
        with each edge
        """
        # Get all node indices in a sorted list (so we have a stable ordering).
        sorted_indices = sorted(self.node_map.keys())

        # Map each node's index to the row/column in the adjacency matrix.
        index_to_row = {node_index: i for i, node_index in enumerate(sorted_indices)}
        
        n = len(sorted_indices)
        fluxed = np.zeros((n,n),dtype=complex)
        
        for node_idx, node in self.node_map.items():
            i = index_to_row[node_idx]
            for neighbor_idx in node.neighbors:
                if neighbor_idx not in self.node_map:
                    continue
                j = index_to_row[neighbor_idx]
                fluxed[i,j] = node.phases[neighbor_idx]
                #fluxed[j,i] = self.node_map[neighbor_idx].phases[node_idx]

        self.fluxed = fluxed

    def weighted_adj(self, flux) -> np.array:
        """
        cascaded : output of cascade(numbers)
        flux     : flux  

        produces the fluxed up adjacency matrix of a 
        glued tree. you only need to run cascade 
        once, and you can evaluate different flux 
        points with a very simple manipulation
        """
        a = np.real(self.fluxed)
        b = np.imag(self.fluxed)
        return b*np.exp(a*flux*1j)

        
    def distance(self, nodeA: Node, nodeB: Node) -> int:
        """
        BFS on graph between two nodes, returns -1 if there is no path
        (these are connected graphs so this shouldn't happen) or the distance between 
        the nodes
        """
        if nodeA.index == nodeB.index:
            return 0

        distances =  {node: float('inf') for node in self.node_map}
        distances[nodeA.index] = 0
        queue = deque([nodeA])

        while queue:
            current = queue.popleft()
            for n_idx in current.neighbors: 
                if distances[n_idx] == float('inf'):  # Not visited
                    distances[n_idx] = distances[current.index] + 1
                    #quit early if the distance between the nodes is calculated properly
                    if n_idx == nodeB.index:
                        return distances[n_idx]
                    queue.append(self.node_map[n_idx])
        return -1

    def all_distances(self, vertex):
        """Returns a dictionary of distances from a vertex to all other vertics."""
        distances = {node: float('inf') for node in self.node_map}
        distances[vertex.index] = 0
        queue = deque([vertex])
        
        while queue:
            current = queue.popleft()
            for n_idx in current.neighbors: 
                if distances[n_idx] == float('inf'):  # Not visited
                    distances[n_idx] = distances[current.index] + 1
                    queue.append(self.node_map[n_idx])
        return distances         
        
    def is_tree(self) -> bool:
        """Uses BFS traversal to check if a graph is a tree (acyclic; connectivity is assumed)"""
        visited = set()  
        first_vertex = self.node_map[min(self.node_map)]

        queue = deque([(first_vertex, None)])  # Store (current_node, parent)
    
        while queue:
            current, parent = queue.popleft()
    
            if current in visited:
                continue  
    
            visited.add(current)
    
            for n_idx in current.neighbors:   
                if self.node_map[n_idx] == parent:
                    continue 
                if self.node_map[n_idx] in visited:
                    return False  # loop detected
                queue.append((self.node_map[n_idx], current))
    
        return True 

    
    def is_original(self, vertex: Node) -> bool:
        """Checks if a vertex is original 
        (all other vertices of the same distance from the vertex are of the same)"""       
        distances = self.all_distances(vertex)

        #group all the nodes that are of distance d away from the vertex
        of_dist_d = {}
        for node in distances.keys():
            if distances[node] in of_dist_d:
                of_dist_d[distances[node]] = of_dist_d[distances[node]] + [node] 
            else:
                of_dist_d[distances[node]] = [node]    

        #check all the nodes of distance d and confirm they are all of the same degree
        for d in of_dist_d.keys():
            degrees = []
            for node_idx in of_dist_d[d]:
                degrees.append(self.node_map[node_idx].degree)
            if len(set(degrees)) > 1:
                return False

        return True

    def add_node(self, nodes: Union[Node,list[Node],set[Node]]) -> None:
        """Adds a node to a graph """
        if isinstance(nodes, Node):
            #adds node itself
            self.nodes.add(nodes)
            self.node_map[nodes.index] = nodes
    
            #adds references of the node to its neighbors
            for neighbor_idx in nodes.neighbors:
                if neighbor_idx in self.node_map:
                    self.node_map[neighbor_idx].add_neighbor(nodes.index, -1*np.conjugate(nodes.phases[neighbor_idx]))
        else:
            for node in nodes:
                self.nodes.add(node)
                self.node_map[node.index] = node
        
                #adds references of the node to its neighbors
                for neighbor_idx in node.neighbors:
                    if neighbor_idx in self.node_map:
                        self.node_map[neighbor_idx].add_neighbor(node.index,-1*np.conjugate(node.phases[neighbor_idx]))
            

    def remove_node(self, nodes: Union[Node,list[Node],set[Node]]) -> None:
        """Removes a node from a graph """
        if isinstance(nodes,Node):
            #removes references of the nodes from its neighbors    
            for neighbor_idx in nodes.neighbors:
                if neighbor_idx in self.node_map:
                    self.node_map[neighbor_idx].remove_neighbor(nodes.index)

            #removes node itself
            self.nodes.discard(nodes)
            if nodes.index in self.node_map:
                self.node_map.pop(nodes.index, None)
    
            
        else:
            for node in nodes:
                #removes references of the nodes from its neighbors    
                for neighbor_idx in node.neighbors:
                    if neighbor_idx in self.node_map:
                        self.node_map[neighbor_idx].remove_neighbor(node.index)    

                #removes node itself
                self.nodes.discard(node)
                if node.index in self.node_map:
                    self.node_map.pop(node.index, None)
        
                
    def deep_copy(self):
        # Return a deep copy of the instance
        return copy.deepcopy(self) 

    def edges(self):
        for node in self.nodes:
            print("node index: ", node.index)
            print("neighbors: ", node.neighbors)

class Tree(ConnectedGraph):

    def __init__(self, nodes: Union[Node, set[Node],list[Node]]):
        super().__init__(nodes) 

        #need to grab roots and check properties
        self.roots = self.find_roots()
        self.rooted = True if len(self.roots) > 0 else False
        self.good = self.check_good()
        self.depth = self.roots[0][1] if len(self.roots) == 1 else None
        self.pnary = self.check_pnary()
    
    def find_roots(self):
        """This function searches the graph for roots and returns a list of tuples 
        containing the root and the depth of tree for the chosen root"""
        roots = []
        for pos_root in self.nodes:
            distances = self.all_distances(pos_root)
            leaf_distances = [dist for node_idx, dist in distances.items() if self.node_map[node_idx].degree == 1]

            # Check if all leaves are at the same depth
            if len(set(leaf_distances)) == 1:  # All leaves have the same depth
                depth = leaf_distances[0]
                roots.append((pos_root, depth))
        return roots       

    def check_good(self):
        """Check if a tree is good (has an original root and one vertex with degree greater than 1)"""
        if not self.rooted:
            return False
        else:
            for root in self.roots:
                if not self.is_original(root[0]):
                    continue
                else:
                    if any(node.degree > 1  for node in self.nodes):
                        return True
        return False

    def check_pnary(self):
        if not self.good:
            return False
        for root in self.roots:
            if root[0].degree > 1:
                special_root = root[0]
                p = special_root.degree
        for node in self.nodes:
            if node.degree not in [1,p,p+1]:
                return False
        return True

##### FUNCTIONS #####
def graph_from_adj(adj: np.array) -> ConnectedGraph:
    """Creates a Connected Graph Object From an Adjacency Matrix"""
    num_nodes = np.shape(adj)[0]
    node_map = {}
    nodes = []
    
    for i in range(num_nodes):
        if i not in node_map:
            curr_node = Node(i,[])
        else:
            curr_node = node_map[i]
        
        if num_nodes > 1:
            row = adj[i,:]

            #for j in range(i,num_nodes):
            for j in range(0,num_nodes):
                if row[j] != 0:
                    if j not in node_map:
                        new_node = Node(j,[i])

                        nodes.append(new_node)
                        node_map[j] = new_node
                    else:
                        new_node = node_map[j]

                    #flux = adj[i,j]
                    #curr_node.add_neighbor(j, flux)
                    curr_node.add_neighbor(j)
                    #for adjify mat, a_ij = -(a_ji)^*
                    #new_node.add_neighbor(i)

        if i not in node_map:
            nodes.append(curr_node)
            node_map[i] = curr_node
                
    return ConnectedGraph(nodes)

def graph_from_fluxed(adj: np.array) -> ConnectedGraph:

    """Creates a Connected Graph Object From an fluxed Matrix"""
    num_nodes = np.shape(adj)[0]
    node_map = {}
    nodes = []
    
    for i in range(num_nodes):
        if i not in node_map:
            curr_node = Node(i,[])
        else:
            curr_node = node_map[i]
        
        if num_nodes > 1:
            row = adj[i,:]

            for j in range(i,num_nodes):
                if row[j] != 0:
                    if j not in node_map:
                        new_node = Node(j,[i])

                        nodes.append(new_node)
                        node_map[j] = new_node
                    else:
                        new_node = node_map[j]

                    flux = adj[i,j]
                    curr_node.add_neighbor(j, flux)
                    #for adjify mat, a_ij = -(a_ji)^*
                    new_node.add_neighbor(i, -1*np.conjugate(flux))

        if i not in node_map:
            nodes.append(curr_node)
            node_map[i] = curr_node
                
    return ConnectedGraph(nodes)

    
def generate_good_tree(X: list[int]):
    """
    Generates a good tree from a sequence X
    """
    depth = len(X)
    NodeDict =  {1: Node(index=1, neighbors=np.arange(2,2+X[-1],1))}
    for child in NodeDict[1].neighbors:
        NodeDict[child] = Node(index=child,neighbors=1)

    for layer in range(depth,1,-1):
        parent_layer_mag = np.cumprod(X[depth-layer+1:])[-1]
        #upper
        #parents = [Node(index=i,neighbors =[]) for i in range (tree_mag(X[depth-layer+2:])+1, tree_mag(X[depth-layer+1:])+1) ]
        parent_indices = [i for i in range (tree_mag(X[depth-layer+2:])+1, tree_mag(X[depth-layer+1:])+1) ]
        #i represents parend index
        for i, parent in enumerate(parent_indices):
            if parent not in NodeDict:
                NodeDict[parent_indices] = Node(index = parent, neighbors=[]) 
                
            childIndices = [k for k in range(parent_indices[0] + parent_layer_mag + X[depth-layer]*i , parent_indices[0] + parent_layer_mag + X[depth-layer] * (i+1))  ]
            for child in childIndices:
                if child not in NodeDict:
                    NodeDict[child] = Node(index = child, neighbors=parent)
                else:
                    NodeDict[child].add_neighbor(parent)
                NodeDict[parent].add_neighbor(child)
            
    
    Nodes = [v for k,v in NodeDict.items()]

    return Tree(nodes=Nodes)

def pl_adj(adj_matrix,title=""):
    """
    Plots a graph based on its adjacency matrix.

    Parameters:
        adj_matrix (np.ndarray): The adjacency matrix of the graph.
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    pos = nx.planar_layout(G)

    # Draw the graph
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", font_weight="bold", node_size=500, font_size=10)
    plt.title(title)
    plt.show()

def pl_graph(graph: ConnectedGraph, positions=None, title=""):
    """Visualizes a Graph"""
    
    G = nx.Graph()
    
    for node in graph.nodes:
        for neighbor_index in node.neighbors:
            if not G.has_edge(node.index, neighbor_index):
                G.add_edge(node.index, neighbor_index)
    
    if positions is None:
        positions = nx.spring_layout(G)
        
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos=positions, with_labels=True, node_color="lightblue", edge_color="gray", font_weight="bold", node_size=500, font_size=10)
    plt.title(title)
    plt.show()
    
"""Make sure the fluxes are transferred correctly """
def get_edges(A):
    u, v = np.where(np.triu(A, k=1) == 1)

    #for the special case of the square lattice
    if A.shape[0] == 1:
        u, v = np.where(np.triu(A, k=0) != 0)

    num_edges = len(u)

    #adding this to test Andrew's special lattice
    if num_edges == 0 and A.shape[0] != 1:
        u, v = np.where(np.tril(A, k=0) != 0)
        num_edges = len(u)



    # Sort edges by their start node (and secondarily by the end node)
    order = np.lexsort((v, u))
    u = u[order]
    v = v[order]
    # Group edges by their starting node
    grouped_edges = [( list(group)) for key, group in groupby(zip(u, v), key=lambda x: x[0])]

    return grouped_edges, num_edges

def shift_graph(graph: ConnectedGraph, shift: int):
    """Changes the index of every element of a graph by a shift"""
    Nodes = []
    for node in graph.nodes:
        Nodes.append(Node(index = node.index+ shift, neighbors=[i+ shift for i in node.neighbors]))
    return ConnectedGraph(Nodes)

def generate_random_cycle_graph(tree: Tree):
    """ 
    Generates a random cycle graph by copying a tree and then adding edges along the bottom layer of the two trees
    
    """
    tree_copy = tree.deep_copy()
    tree_size =  len(tree_copy.nodes)
    bot_layer = [int(node.index) for node in tree.nodes if node.degree == 1]
    copy_bot_layer = [int(leaf) + tree_size for leaf in bot_layer]

    RandomCycleGraph = ConnectedGraph(nodes = tree_copy.nodes | shift_graph(tree, tree_size ).nodes )
    #### Add random edges between the leaves at the deepest level
    for leaf in bot_layer:
        nn_indices = random.sample((0,len(copy_bot_layer)-1),2)
        new_neighbors = (copy_bot_layer[nn_indices[0]], copy_bot_layer[nn_indices[1]])
                
        RandomCycleGraph.node_map[leaf].add_neighbor(new_neighbors[0])
        RandomCycleGraph.node_map[new_neighbors[0]].add_neighbor(leaf)
        
        RandomCycleGraph.node_map[leaf].add_neighbor(new_neighbors[1])
        RandomCycleGraph.node_map[new_neighbors[1]].add_neighbor(leaf)
        
        for neighbor_index in new_neighbors:
            if RandomCycleGraph.node_map[neighbor_index].degree > 2:
                copy_bot_layer.remove(neighbor_index)
    
    return RandomCycleGraph


###### Helping with Counting ######
def tree_mag(X:list[int]):
    d = len(X)
    if type(X) == np.ndarray:
        X = X.tolist()
    X = [1]+X
    total = 1
    for i in range(0, d):
        total += np.prod(X[-(i+1):])
    return total
def gt_mag(X:list[int]):
    return tree_mag(X) + tree_mag(X[1:])


########## Tree Coordinates ############
def tree_coords(X: list[int], xd=1, yd=1):
    depth = len(X)
    coord_dict = {1: (0,yd)}
    #start from middle layer:
    midrange = list(range(tree_mag(X[1:])+1, tree_mag(X)+1))
    line = lambda x: yd/xd*x+yd
    print(line(0))
    #give coordinates to the middle level
    for i,vertex in  enumerate(midrange):
        y = 0#yd*(depth)
        x_coords = np.linspace(-xd, xd, len(midrange))
        coord_dict[vertex] = (x_coords[i] ,y)
    for layer in range(depth,1,-1):
        x_coords = []
        parent_layer_mag = np.cumprod(X[depth-layer+1:])[-1]
        #upper
        parents = list(
            range(
                tree_mag(X[depth - layer + 2 :]) + 1,
                tree_mag(X[depth - layer + 1 :]) + 1,)
        )
        #print(upper_parents)
        for i, node in enumerate(parents):
            children = list(
                range(
                    parents[0] + parent_layer_mag + X[depth - layer] * i,
                    parents[0] + parent_layer_mag + X[depth - layer] * (i + 1),
                )
            )
            x_coords.append( np.mean([coord_dict[j][0] for j in children] ))
        y_coord = line(x_coords[0])
        for i, node in enumerate(parents):
            coord_dict[node] = (x_coords[i] , y_coord)
    return coord_dict

def rand_cycle_coords(X: list[int], xd=1, yd=1,spacing=1):
    left_coords = tree_coords(X,xd,yd)
    coords = left_coords.copy()
    tree_size = tree_mag(X)
    for coord in left_coords.keys():
        coords[coord+tree_size]  = (left_coords[coord][0], -1*left_coords[coord][1] - spacing)
    return coords