import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy

from collections.abc import Iterable
from typing import Union
from collections import deque
from itertools import groupby
import random

from Andrew.gt import cascade
from _counttools import gen_necklace

edge_color = '#264653'
vertex_color ='#E76F51'

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
        #adjacency matries to be constructed later
        self.adj = None
        self.fluxed = None #the adjacency matrix with the fluxes
        self.period = None #construct the periodified matrix later

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

    def construct_period(self, loop_flux) -> None:
        """Generates the periodified fluxed adjacency matrix at a given loop flux with a k depedence """
        if type(self.adj) == None:
            self.construct_adj()
        if type(self.fluxed) != np.ndarray:
            self.construct_fluxed()

        def period(phase):
        # copy so we never clobber `base`
            R = self.weighted_adj(loop_flux).copy()
            R[0, :]      += np.exp(1j*phase)  * R[-1, :]
            R[:, 0]      += np.exp(-1j*phase) * R[:, -1]
            return R[:-1, :-1]

        self.period = period

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

def tree_from_fluxed(adj: np.array) -> Tree:

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
                
    return Tree(nodes)

    
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
                NodeDict[parent] = Node(index = parent, neighbors=[]) 
                
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

def pl_graph(ax, graph, positions=None, title="" ,vertex_size=500,outline_weight = 5,edge_weight=10, margin = 10):
    """Visualizes a Graph"""
    
    G = nx.Graph()
    for node in graph.nodes:
        for neighbor_index in node.neighbors:
            if not G.has_edge(node.index, neighbor_index):
                G.add_edge(node.index, neighbor_index)
    for node in graph.nodes:
        G.add_node(node.index)
        G.nodes[node.index]['label'] = node.index
    
    if positions is None:
        positions = nx.spring_layout(G)
            
    nx.draw_networkx_nodes(G, positions, node_color=vertex_color, edgecolors=edge_color, linewidths=outline_weight, node_size=vertex_size, ax=ax)
    nx.draw_networkx_labels(G, positions, labels=nx.get_node_attributes(G, 'label'),font_size=15,font_color="black")

    for edge in G.edges(data='weight'):
        nx.draw_networkx_edges(G, positions, edgelist=[edge], width=edge_weight,edge_color=edge_color,ax=ax)
    
    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal', adjustable='box')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    
    return ax

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
def tree_diamond_coords(X: list[int], xd=1, yd=1):
    depth = len(X)
    coord_dict = {1: (0,yd)}
    #start from middle layer:
    midrange = list(range(tree_mag(X[1:])+1, tree_mag(X)+1))
    line = lambda x: yd/xd*x+yd
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

def tree_coords(X: list[int], xd=1, yd=1):
    depth = len(X)
    coord_dict = {1: (0, yd)}
    #start from middle layer:
    midrange = list(range(tree_mag(X[1:])+1, tree_mag(X)+1))

    #give coordinates to the middle level
    for i,vertex in  enumerate(midrange):
        y = 0
        x_coords = np.linspace(-xd, xd, len(midrange))
        coord_dict[vertex] = (x_coords[i] , y)

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
        y_coord = yd* (depth-layer+1)/depth
        for i, node in enumerate(parents):
            coord_dict[node] = (x_coords[i] , y_coord)
    return coord_dict



def rand_cycle_diamond_coords(X: list[int], xd=1, yd=1,spacing=1):
    left_coords = tree_diamond_coords(X,xd,yd)
    coords = left_coords.copy()
    tree_size = tree_mag(X)
    for coord in left_coords.keys():
        coords[coord+tree_size]  = (left_coords[coord][0], -1*left_coords[coord][1] - spacing)
    return coords

def rand_cycle_coords(X: list[int], xd=1, yd=1,spacing=1):
    left_coords = tree_coords(X,xd,yd)
    coords = left_coords.copy()
    tree_size = tree_mag(X)
    for coord in left_coords.keys():
        coords[coord+tree_size]  = (left_coords[coord][0], -1*left_coords[coord][1] - spacing)
    return coords

def glued_tree_coords(X: list[int], xd=1, yd=1,spacing=1):
    left_coords = tree_coords(X,xd,yd)
    coords = left_coords.copy() #tree_coords(X,xd,yd)
    tree_size = tree_mag(X)
    for coord in left_coords.keys():
        coords[coord+tree_size]  = (left_coords[coord][0], -1*left_coords[coord][1] - spacing)
    return coords

#### Getting the Phases ####
def shift_graph(graph: ConnectedGraph, shift: int):
    """Changes the index of every element of a graph by a shift"""
    Nodes = []
    graph_copy = graph.deep_copy()
    for node in graph_copy.nodes:
        Nodes.append(Node(index = node.index+ shift, neighbors=[i+ shift for i in node.neighbors],phases = {key+shift:value for key,value in node.phases.items()}))
    return ConnectedGraph(Nodes)

def flip_graph(graph: ConnectedGraph):
    """Rverses the ordering of a graph"""
    Nodes = []
    N = len(graph.node_map)
    flipped_dict = {i: N+1-i for i in list(graph.node_map.keys())}
    for node in graph.nodes:
        Nodes.append(Node(index = flipped_dict[node.index], neighbors=[flipped_dict[i] for i in node.neighbors],phases = {flipped_dict[key]:value for key,value in node.phases.items()}))
    return ConnectedGraph(Nodes)

def remove_phases(graph: ConnectedGraph):
    """Returns a copy of a graph with all phases removed"""
    Nodes = []
    graph_copy = graph.deep_copy()
    for node in graph_copy.nodes:
        Nodes.append(Node(index = node.index, neighbors=[i for i in node.neighbors],phases = {key:1j for key,value in node.phases.items()}))
    return ConnectedGraph(Nodes)


def change_basis(adj_matrix, new_order):
    """
    Changes the basis of an adjacency matrix by permuting its rows and columns.
    
    Parameters:
        adj_matrix (np.ndarray): The original adjacency matrix.
        new_order (list): The new order of site labels, where the indices correspond to the new labels and the values are the old labels (1-based).
    
    Returns:
        np.ndarray: The adjacency matrix in the new basis.
    """
    
    # Permute rows and columns
    new_adj_matrix = adj_matrix[np.ix_(new_order, new_order)]
    
    return new_adj_matrix

def bfs_order(adj_matrix, start=0):
    """
    Perform a breadth-first traversal on a graph represented by its adjacency matrix.
    
    Parameters:
        adj_matrix (list of lists or numpy.ndarray): The adjacency matrix of the graph.
            It is assumed that a nonzero entry at adj_matrix[i][j] indicates an edge from vertex i to j.
        start (int): The starting vertex for the BFS.
        
    Returns:
        list: A list of vertices in the order they are visited by the BFS.
    """
    n = len(adj_matrix)            # Number of vertices
    visited = [False] * n          # Keep track of visited vertices
    order = []                     # List to store the BFS order
    queue = deque([start])         # Initialize the queue with the start vertex
    
    visited[start] = True          # Mark the starting vertex as visited
    
    while queue:
        current = queue.popleft()  # Get the next vertex from the queue
        order.append(current)      # Add it to the order
        
        # Check all possible neighbors of the current vertex
        for neighbor, edge in enumerate(adj_matrix[current]):
            # If there is an edge and the neighbor has not been visited
            if edge and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                
    return order

def fluxedTree(X: list[int]):
    
    adj_mat = cascade(X)
    order = bfs_order(adj_mat)
    gt_mat = change_basis(adj_mat, order)
    
    cut = tree_mag(X)
    tree_mat = gt_mat[:cut,:cut]
    
    tree = shift_graph(tree_from_fluxed(tree_mat),1)
    
    return tree
    
def createGT(X: list[int]):
    gt = graph_from_fluxed(cascade(X))
    gt.construct_fluxed()
    return gt

def generate_random_cycle_graph(tree: Tree, necklace = None):
    """ 
    Generates a random cycle graph by copying a tree and then adding edges along the bottom layer of the two trees
    
    """
    #making some changes to work with a flipped graph
    tree_copy = tree.deep_copy()
    #tree_copy = flip_graph(tree)
    tree_size =  len(tree_copy.nodes)
    bot_layer = [int(node.index) for node in tree_copy.nodes if node.degree == 1]
    bot_layer.sort()
    copy_bot_layer = [int(leaf) + tree_size for leaf in bot_layer]

    RandomCycleGraph = ConnectedGraph(nodes = tree_copy.nodes | shift_graph(tree_copy, tree_size).nodes )
    if not necklace or len(necklace) != 2*len(bot_layer):
        necklace = gen_necklace(len(bot_layer))

    odd_leaf_index = bot_layer[int(necklace[0]/2)]
    left_neighbor = copy_bot_layer[int(necklace[-1]/2)-1]
    right_neighbor =  copy_bot_layer[int(necklace[1]/2)-1]
    RandomCycleGraph.node_map[odd_leaf_index].add_neighbor(left_neighbor)
    RandomCycleGraph.node_map[left_neighbor].add_neighbor(odd_leaf_index)
    RandomCycleGraph.node_map[odd_leaf_index].add_neighbor(right_neighbor)
    RandomCycleGraph.node_map[right_neighbor].add_neighbor(odd_leaf_index)

    #connect odd beads to their neighbors
    for bead_index in range(2,len(necklace),2):
        odd_leaf_index = bot_layer[int(necklace[bead_index]/2)]
        left_neighbor = copy_bot_layer[int(necklace[bead_index-1]/2)-1]
        right_neighbor =  copy_bot_layer[int(necklace[bead_index+1]/2)-1]

        RandomCycleGraph.node_map[odd_leaf_index].add_neighbor(left_neighbor)
        RandomCycleGraph.node_map[left_neighbor].add_neighbor(odd_leaf_index)
        RandomCycleGraph.node_map[odd_leaf_index].add_neighbor(right_neighbor)
        RandomCycleGraph.node_map[right_neighbor].add_neighbor(odd_leaf_index)

    return RandomCycleGraph

def generate_half_rgc(tree: Tree, necklace = None):
    tree_copy = tree.deep_copy()
    tree_size =  len(tree_copy.nodes)
    bot_layer = [int(node.index) for node in tree_copy.nodes if node.degree == 1]
    bot_layer.sort()
    copy_bot_layer = [int(leaf) + tree_size for leaf in bot_layer]
    bot_half_graph = remove_phases(shift_graph(tree_copy, tree_size))
    RandomCycleGraph = ConnectedGraph(nodes = tree_copy.nodes | bot_half_graph.nodes )
    if not necklace or len(necklace) != 2*len(bot_layer):
        necklace = gen_necklace(len(bot_layer))

    odd_leaf_index = bot_layer[int(necklace[0]/2)]
    left_neighbor = copy_bot_layer[int(necklace[-1]/2)-1]
    right_neighbor =  copy_bot_layer[int(necklace[1]/2)-1]
    RandomCycleGraph.node_map[odd_leaf_index].add_neighbor(left_neighbor)
    RandomCycleGraph.node_map[left_neighbor].add_neighbor(odd_leaf_index)
    RandomCycleGraph.node_map[odd_leaf_index].add_neighbor(right_neighbor)
    RandomCycleGraph.node_map[right_neighbor].add_neighbor(odd_leaf_index)

    #connect odd beads to their neighbors
    for bead_index in range(2,len(necklace),2):
        odd_leaf_index = bot_layer[int(necklace[bead_index]/2)]
        left_neighbor = copy_bot_layer[int(necklace[bead_index-1]/2)-1]
        right_neighbor =  copy_bot_layer[int(necklace[bead_index+1]/2)-1]

        RandomCycleGraph.node_map[odd_leaf_index].add_neighbor(left_neighbor)
        RandomCycleGraph.node_map[left_neighbor].add_neighbor(odd_leaf_index)
        RandomCycleGraph.node_map[odd_leaf_index].add_neighbor(right_neighbor)
        RandomCycleGraph.node_map[right_neighbor].add_neighbor(odd_leaf_index)

    return RandomCycleGraph