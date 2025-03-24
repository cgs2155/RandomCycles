import numpy as np
from itertools import groupby

def pt_mag(p,d):
    """
    Returns:
        int: number of leaves in a p-nary tree
    """
    return int((p**(d+1)-1)/(p-1))

def gt_mag(p,d):
    """
    Returns:
        int: number of leaves in a glued p-nary tree
    """
    return int(pt_mag(p,d) + pt_mag(p,d-1))

def change_basis(adj_matrix, new_order):
    """
    Changes the basis of an adjacency matrix by permuting its rows and columns.
    
    Parameters:
        adj_matrix (np.ndarray): The original adjacency matrix.
        new_order (list): The new order of site labels, where the indices correspond
                          to the new labels and the values are the old labels (1-based).
    
    Returns:
        np.ndarray: The adjacency matrix in the new basis.
    """
    
    # Permute rows and columns
    new_adj_matrix = adj_matrix[np.ix_(new_order, new_order)]
    
    return new_adj_matrix

def extend(mat, dim_plus):
    """
    Extends the dimensions of a square matrix by adding zeros.
    
    Parameters:
        mat (np.ndarray): The original square matrix.
        dim_plus (int): The number of additional rows and columns to add.
    
    Returns:
        np.ndarray: The extended matrix with added rows and columns filled with zeros.
    """
    dim = mat.shape[0]
    edim = dim + dim_plus
    
    # Create a new zero matrix with the extended dimensions
    e_mat = np.zeros((edim, edim), dtype=mat.dtype)
    
    # Copy the original matrix into the top-left corner
    e_mat[:dim, :dim] = mat
    
    return e_mat

##### INSERTING TREES AT ALL EDGES

def insert_gtree_all(A,p,d):
    #number of edges in a undirected graph without loops
    num_edges = np.sum(A)/2
    #original dimension of A
    dim = np.shape(A)[0]
    #extend the graph to account for the new sites
    tmag = gt_mag(p,d)
    Ap_mag = num_edges*tmag
    A_gtree, old_edges = adjust_basis_all(A,p,d)
    A_gtree = A_gtree*0
    
    roots = np.array(list((set(np.array(old_edges).flatten()))))
    all_sites = set(list(np.arange(len(A_gtree))))
    leaves = [leaf for leaf in all_sites if leaf not in roots]
    #loop through the edges and insert a glued tree
    for tree_num, edge in enumerate(old_edges):
        #connect the roots to its children
        tf_leaf = tree_num*(tmag-2)#first leaf in the top tree
        bf_leaf = (tree_num+1)*(tmag-2) + -1#first leaf in the bottom tree
        for l in range(p):
            #top root
            A_gtree[edge[0], leaves[tf_leaf + l]] = A_gtree[leaves[tf_leaf + l],edge[0]] = 1
            #bottom root
            A_gtree[edge[-1], leaves[bf_leaf - l]] = A_gtree[leaves[bf_leaf - l],edge[-1]] = 1
        #generate the rest of tree
        for layer in range(1,d):
            for leaf in range(0, p**layer):
                #connect each leaf to its children
                tp_idx = tf_leaf + p**layer - p + leaf #top parent index
                bp_idx = bf_leaf - (p**layer - p + leaf) #bottom parent index
                for child in range(0,p):
                    tc_idx = tp_idx + p**layer + child + (p-1)*leaf
                    bc_idx = bp_idx - (p**layer + child + (p-1)*leaf)

                    A_gtree[leaves[tp_idx], leaves[tc_idx]] = A_gtree[leaves[tc_idx],leaves[tp_idx]] = 1
                    A_gtree[leaves[bp_idx], leaves[bc_idx]] = A_gtree[leaves[bc_idx],leaves[bp_idx]] = 1
    
    return A_gtree

def adjust_basis_all(A,p,d):
    """
    Takes the adjacency matrix of a graph and extends it, adjusting the basis prior to
    the insertion of glued trees at every edge
    
    Parameters:
        A (np.ndarray): The original adjacency matrix.
        p (int): The number of children a leaf may have
        d (int): The glued tree depth

    Returns:
        np.ndarry: The adjacency matrix for a glued tree without new edges
        and basis adjusted to enfore breadth-first traversal
    
    """
    # Compute how many new nodes to add per edge
    extension = gt_mag(p, d) - 2
    dim = A.shape[0]

    # Extract all edges (upper triangle to avoid duplicates)
    u, v = np.where(np.triu(A, k=1) == 1)
    num_edges = len(u)

    # Extend the adjacency matrix to accommodate new nodes
    Ap_mag = num_edges * extension
    A_gtree = extend(A, Ap_mag)

    # Sort edges by their start node (and secondarily by the end node)
    order = np.lexsort((v, u))
    u = u[order]
    v = v[order]

    # Group edges by their starting node
    grouped_edges = [(key, list(group)) for key, group in groupby(zip(u, v), key=lambda x: x[0])]

    site_dict = {}    # Maps original node to its position in the new basis
    new_basis = []     # The new ordering of nodes
    edges_out = []     # Records edges in terms of new indices
    curr_edge = 0

    # Process each group of edges originating from the same start node
    for (node_u, group_edges) in grouped_edges:
        # If this node hasn't been placed in the new basis, place it now
        if node_u not in site_dict:
            site_dict[node_u] = len(new_basis)
            new_basis.append(node_u)

        # For each edge from node_u to node_v
        for (_, node_v) in group_edges:
            # Append extension nodes for the glued tree on this edge
            new_sites = np.arange(dim + curr_edge*extension, dim + (curr_edge+1)*extension)
            for ns in new_sites:
                new_basis.append(ns)

            site_in_new_basis = site_dict[node_u]

            # If the target node hasn't appeared yet, add it now
            if node_v not in site_dict:
                site_dict[node_v] = len(new_basis)
                new_basis.append(node_v)

            edge_in_new_basis = site_dict[node_v]
            edges_out.append([site_in_new_basis, edge_in_new_basis])

            curr_edge += 1

    new_basis = np.array(new_basis, dtype=int)
    return change_basis(A_gtree, new_basis), edges_out

###### inserting a single p_tree at an edge

def insert_gtree(A, p, d, edge):
    """
    Inserts a glued tree at a single specified edge in the graph.

    Parameters:
        A (np.ndarray): The original adjacency matrix.
        p (int): The number of children a leaf may have.
        d (int): The glued tree depth.
        edge (tuple of int): The two sites connected by an edge.

    Returns:
        np.ndarray: The updated adjacency matrix with the glued tree inserted.
    """
    # Adjust the basis and prepare the adjacency matrix for the insertion
    A_gtree, new_edge = adjust_basis(A, p, d, edge)
    # Number of new sites to be inserted
    tmag = gt_mag(p, d)
    extension = tmag - 2

    # Identify roots and leaves
    roots = np.array([new_edge[0], new_edge[1]])
    child_sites = np.arange(new_edge[0]+1, new_edge[0]+extension+1)
    leaves = child_sites
    # Indices for the first and last leaf of the tree
    tf_leaf = 0  # First leaf in the top tree
    bf_leaf = extension-1 # Last leaf in the bottom tree

    for l in range(p):
        #top root
        A_gtree[roots[0], leaves[tf_leaf + l]] = A_gtree[leaves[tf_leaf + l],roots[0]] = 1
        #bottom root
        A_gtree[roots[-1], leaves[bf_leaf - l]] = A_gtree[leaves[bf_leaf - l],roots[-1]] = 1
    #generate the rest of tree
    for layer in range(1,d):
        for leaf in range(0, p**layer):
            #connect each leaf to its children
            tp_idx = tf_leaf + p**layer - p + leaf #top parent index
            bp_idx = bf_leaf - (p**layer - p + leaf) #bottom parent index
            for child in range(0,p):
                tc_idx = tp_idx + p**layer + child + (p-1)*leaf
                bc_idx = bp_idx - (p**layer + child + (p-1)*leaf)

                A_gtree[leaves[tp_idx], leaves[tc_idx]] = A_gtree[leaves[tc_idx],leaves[tp_idx]] = 1
                A_gtree[leaves[bp_idx], leaves[bc_idx]] = A_gtree[leaves[bc_idx],leaves[bp_idx]] = 1

    return A_gtree



def adjust_basis(A, p, d, edge):
    """
    Adjusts the adjacency matrix for a graph to account for the insertion of new vertices 
    and reorders the basis for breadth-first traversal.

    Parameters:
        A (np.ndarray): The original adjacency matrix.
        p (int): The number of children a leaf may have.
        d (int): The glued tree depth.
        edge (tuple of int): The two sites connected by an edge.

    Returns:
        tuple: The updated adjacency matrix and the adjusted edge.
    """
    # Calculate the number of new vertices to insert
    extension = gt_mag(p, d) - 2
    dim = A.shape[0]  # Original number of vertices

    # Extend the adjacency matrix to accommodate new vertices
    A_gtree = extend(A, extension)

    # Remove the existing edge between the specified vertices
    A_gtree[edge[0], edge[1]] = A_gtree[edge[1], edge[0]] = 0 

    # Create the new basis order
    new_basis = np.concatenate((
        np.arange(edge[0]),               # Vertices before the first vertex of the edge
        [edge[0]],                        # The first vertex of the edge
        np.arange(dim, dim + extension),  # Newly added vertices
        np.arange(edge[0] + 1, dim)       # Remaining original vertices
    ))

    # Update the new edge to account for basis adjustment
    new_edge = (edge[0], edge[-1]+extension)

    # Change the basis of the adjacency matrix
    return change_basis(A_gtree, new_basis), new_edge

##### HAMILTONIAN GENERATION HAPPENS DOWN HERE


def get_edges(A):
    u, v = np.where(np.triu(A, k=1) == 1)

    #for the special case of the square lattice
    if A.shape[0] == 1:
        u, v = np.where(np.triu(A, k=0) == 1)
    #print(u)
    num_edges = len(u)

    # Sort edges by their start node (and secondarily by the end node)
    order = np.lexsort((v, u))
    u = u[order]
    v = v[order]
    # Group edges by their starting node
    grouped_edges = [(key, list(group)) for key, group in groupby(zip(u, v), key=lambda x: x[0])]

    return grouped_edges, num_edges

def gtree_ham(T0, p, d, afb= False, *args):
    """
    Inserts glued trees at multiple edges in the graph, applying a series of 
    Fourier-transformed matrices to generate a set of modified adjacency matrices.
    
    Parameters:
        T0 (np.ndarray): The unit cell adjacency matrix/Hamiltonian
        p (int): The number of children a leaf may have.
        d (int): The glued tree depth.
        *args (tuple of np.ndarray, optional): Positive translation operators 
        (T^+_i) in the frequency basis. Must be of same dimension as T0.

    Returns:
        np.ndarray: The k-basis Hamiltonian of the graph with glued trees inserted 
        at every edge
    """
    #list that carries all the hopping operators
    #T_matrices = np.append(T0, args)

    #number of old nodes
    dim = T0.shape[0]
    #size of the inserted glued tree at each end
    gt_size = gt_mag(p, d) 

    # Group edges by their starting node
    T0_grouped_edges, num_edges = get_edges(T0)

    # Extend the adjacency matrix to accommodate new nodes
    num_new_sites = [num_edges * gt_size]
    
    #collect the number of edges from the other matrices by which to expand the matrix by

    #list of edges involved in inter-cell hopping
    hopping_edges = []
    
    for i, T_mat in enumerate(args):
        # Group edges by their starting node
        T_mat_grouped_edges, T_mat_num_edges = get_edges(T_mat)
    
        # Extend the adjacency matrix to accommodate new nodes
        num_new_sites.append(T_mat_num_edges * (gt_size-2))
        hopping_edges.append(T_mat_grouped_edges)
    
    #new adjacency matrix
    T0_gtree = extend(T0, sum(num_new_sites))*0
    
    #create the inserted glued tree intra-cell hopping matrix
    T0_edges = [edge for site in T0_grouped_edges for edge in site[1]]
    roots = np.array(list((set(np.array(T0_edges).flatten()))))    
    leaves = np.arange(dim, dim+num_new_sites[0],1)

    tree_num = 0
    #loop through the edges and insert a glued tree
    for tree_num, edge in enumerate(T0_edges):
        #connect the roots to its children
        tf_leaf = tree_num*(gt_size-2)#first leaf in the top tree
        bf_leaf = (tree_num+1)*(gt_size-2) + -1#first leaf in the bottom tree
        for l in range(p):
            #top root
            T0_gtree[edge[0], leaves[tf_leaf + l]] = T0_gtree[leaves[tf_leaf + l],edge[0]] = 1
            #bottom root
            T0_gtree[edge[-1], leaves[bf_leaf - l]] = T0_gtree[leaves[bf_leaf - l],edge[-1]] = 1
        #generate the rest of tree
        for layer in range(1,d):
            for leaf in range(0, p**layer):
                #connect each leaf to its children
                tp_idx = tf_leaf + p**layer - p + leaf #top parent index
                bp_idx = bf_leaf - (p**layer - p + leaf) #bottom parent index
                for child in range(0,p):
                    tc_idx = tp_idx + p**layer + child + (p-1)*leaf
                    bc_idx = bp_idx - (p**layer + child + (p-1)*leaf)

                    T0_gtree[leaves[tp_idx], leaves[tc_idx]] = T0_gtree[leaves[tc_idx],leaves[tp_idx]] = 1
                    T0_gtree[leaves[bp_idx], leaves[bc_idx]] = T0_gtree[leaves[bc_idx],leaves[bp_idx]] = 1        

    curr_tree_num = tree_num
    #create the glued tree inter-cell hopping matrices
    Tplus_g = []
    
    for i, Ti in enumerate(args):
        Ti_curr = T0_gtree*0
        Ti_edges = [edge for site in hopping_edges[i] for edge in site[1]]
        roots = np.array(list((set(np.array(Ti_edges).flatten()))))    
        leaves = np.arange(dim+sum(num_new_sites[:i+1]), dim+sum(num_new_sites[:i+1])+num_new_sites[i+1],1)                
        for tree_num, edge in enumerate(Ti_edges):
            #connect the roots to its children
            tf_leaf = (curr_tree_num+tree_num)*(gt_size-2)#first leaf in the top tree
            bf_leaf = (curr_tree_num+tree_num+1)*(gt_size-2) + -1#first leaf in the bottom tree
        
            for l in range(p):
                #top root
                T0_gtree[edge[0], leaves[tf_leaf + l]] = T0_gtree[leaves[tf_leaf + l],edge[0]] = 1
                #bottom root which adds k-dependent phase
                Ti_curr[edge[-1], leaves[bf_leaf - l]] = 1
            #generate the rest of tree
            for layer in range(1,d):
                for leaf in range(0, p**layer):
                    #connect each leaf to its children
                    tp_idx = tf_leaf + p**layer - p + leaf #top parent index
                    bp_idx = bf_leaf - (p**layer - p + leaf) #bottom parent index
                    for child in range(0,p):
                        tc_idx = tp_idx + p**layer + child + (p-1)*leaf
                        bc_idx = bp_idx - (p**layer + child + (p-1)*leaf)
    
                        T0_gtree[leaves[tp_idx], leaves[tc_idx]] = T0_gtree[leaves[tc_idx],leaves[tp_idx]] = 1
                        T0_gtree[leaves[bp_idx], leaves[bc_idx]] = T0_gtree[leaves[bc_idx],leaves[bp_idx]] = 1

        curr_tree_num += tree_num
        Tplus_g.append(Ti_curr)


    return T0_gtree, Tplus_g



##### Tiling
def c_r(p: int,n: int)-> float:
    """
        p : a branching factor
        n : integer index ranging from one to p 


        see definition of child ratios in the paper.
    """
    assert 0 < n <= p

    
    return 1 - 2*(n-1)/(p-1)


