import numpy as np
import math
from scipy.linalg import expm
from _graphtools import *
from _vectools import *
from tqdm import tqdm 

p = 2
n_max = 6
AFB = 2 * np.pi / p

# Loop over different system sizes (or iterations)
for i in range(1, n_max+1):
    X = [p] * i
    ftree = fluxedTree(X)
    ftree.construct_fluxed()

    rand = generate_random_cycle_graph(ftree)
    rand.construct_fluxed()
    rand.construct_adj()

    # Obtain the Hamiltonians for both cases
    bare_hamiltonian = rand.adj
    fluxed_hamiltonian = rand.weighted_adj(AFB)
    # Define initial state and target state
    N = len(rand.node_map)
    psi_i = e_n(0, N)

    end_idx= tree_mag(X)
    end = e_n(end_idx,N)#e_n(N - 1, N)

    # Define the time array over which evolution is evaluated
    t = np.linspace(0, 2*i*p, 500)
    delta_t = t[1]
    U_bare = expm(-1j * bare_hamiltonian * delta_t)
    U_fluxed = expm(-1j * fluxed_hamiltonian * delta_t)


    prob_flux = [0]
    prob_bare = [0]

    psi_curr_bare   = psi_i.copy()
    psi_curr_fluxed = psi_i.copy()    
    # Use tqdm to show progress for each time step
    for time in tqdm(t[1:], desc=f"Iteration {i}: processing time steps", unit="time step"):
        psi_curr_bare = U_bare@psi_curr_bare
        psi_curr_fluxed = U_fluxed@psi_curr_fluxed
        prob_flux.append(np.abs(end.conj() @ psi_curr_fluxed)**2)
        prob_bare.append(np.abs(end.conj() @ psi_curr_bare)**2)

    # Save the time and probability arrays to a file for later plotting
    filename = f"walkresults/{p}rc_{i}.npz"
    np.savez(filename, t=t, prob_flux=prob_flux, prob_bare=prob_bare, fluxed_adj = rand.fluxed)
    print(f"Results for iteration {i} saved to {filename}")
