import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
np.set_printoptions(linewidth=10000)

def child_ratios(j: int,p: int)-> float:
    """
        j : integer index ranging from one to p 
        p : a branching factor

        see definition of child ratios in the paper.
    """
    assert 0 < j <= p

    return 1 - 2*(j-1)/(p-1)

def first_ham(p: int):
    """ 
        p    : branching factor.

        returns the (p + 2 by p +2 ) unfluxed adjacency matrix of the pnary 
        glued tree at depth one. The placquette flux is the variable flux this 
        should realistically only be used as a helper function to cascade below.
    """
    prefactor = 2 *child_ratios(1,p) - 2*child_ratios(2,p) 
    fluxrow = [0 if i == 0 or i == p+1 else -child_ratios(i,p)/prefactor + 1j for i in range(p+2)]
    fluxrow = np.array(fluxrow)

    first_elem = np.zeros(p + 2)
    first_elem[0] = 1
    out = np.outer(first_elem, fluxrow) - np.outer(np.conjugate(fluxrow),first_elem) 
    return out + np.flip(out)

def iterate_matrix(input_matrix, branching_factor: int):
    """
        input_matrix     : square hermitian matrix 
        branching_factor : number of copies of input_matrix along the diagonal of the output

        See paper. This function calculates T_d by using T_{d-1} and some trickery. 
        You have to build matrices like this or you will get unexpected behavior
        related to the fact that complex exponentials are periodic. 
        You could avoid this trickery if you solved the flux tiling problem in general. 
    """
    # take a tensor product of input matrix, and then pad the resulting matrix with zeros on all sides
    start = np.pad(np.kron(np.identity(branching_factor), input_matrix),1)

    # extract phases from previous iteration
    ### trickery start
    off_diag = np.diagonal(start,1) 
    phases = np.real(off_diag.copy())
    for x in range(2,len(phases)):
        if phases[x-1] == 0:
            phases[x] = 0
    phase_apply = (1 - 4 * np.sum(phases))/(2 - 2*child_ratios(2,branching_factor)) 
    ### trickery end

    # see App A of paper
    phase_vector = np.array([-child_ratios(i + 1, branching_factor)*phase_apply + 1j for i in range(0,branching_factor)])


    old_first = np.zeros(len(input_matrix))
    old_first[0] = 1
    padded_row = np.pad(np.kron(phase_vector,old_first),1)

    # new_first is first vector of T_d
    new_first = np.zeros(len(start))
    new_first[0] = 1

    start = start + 2*(np.outer(new_first, padded_row) - np.outer(np.conjugate(padded_row),new_first))
    start = (start + np.flip(start))/2
    return start

def cascade(numbers):
    """ 
        numbers : list of branching factors

        Calculates the antisymmetric flux function and 
        unfluxed adjacency matrix of a glued tree with 
        numbers branching factors
    """
    branching_factor = numbers[0]
    assert branching_factor > 1

    result = first_ham(branching_factor)
    for x in range(1,len(numbers)): 
        result = iterate_matrix(result,numbers[x])
    return result 

def adjify(cascaded,flux):
    """
        cascaded : output of cascade(numbers)
        flux     : flux  

        produces the fluxed up adjacency matrix of a 
        glued tree. you only need to run cascade 
        once, and you can evaluate different flux 
        points with a very simple manipulation
    """
    a = np.real(cascaded)
    b = np.imag(cascaded)
    return b*np.exp(a*flux*1j)


def periodify(adj,phase: float, unitcells: int = 1):
    """
        adj       : adjacency matrix to be periodified
        phase     : phase associated with an eigenvalue
            of a translation operator
        unitcells : number of unit cells to be included

        This function produces the hamiltonian of a single 
        particle hopping on a circular chain made of links 
        with adjacency matrix adj 
    """
    length = len(adj)
    result = np.zeros((length*unitcells -unitcells + 1,length*unitcells - unitcells + 1))*1j

    x = 0
    while length + x - 1 < len(result):
        result[x:length+x,x:length+x] = adj
        x = x + length - 1
    # print(np.real(result))
    result[0,:] = result[0,:] + np.exp(1j*phase)*result[-1,:]
    result[:,0] = result[:,0] + np.exp(-1j*phase)*result[:,-1]
    return result[0:-1,0:-1]

def test():
    # TODO: write some tests lol
    return 0 

if __name__ == "__main__":
    test()

