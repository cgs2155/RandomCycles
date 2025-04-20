import numpy as np
import math
from scipy.linalg import expm
from _graphtools import *
from _vectools import *
from _counttools import *
from tqdm import tqdm  # import tqdm for progress bars

def limit(eigvecs, a,b):
    total = 0
    for i in range(0,len(a)):
        vec = eigvecs[:,i]
        total += np.abs(a.conj() @ vec * b.conj() @ vec)**2
    return total

p = 3
n_max = 1

# Loop over different system sizes (or iterations)
for i in range(1, n_max+1):
    #generate the unfluxed tree2
    X = [p] * i
    ftree = fluxedTree(X)
    ftree.construct_fluxed()

    rand = generate_random_cycle_graph(ftree)
    rand.construct_adj()
    bare_hamiltonian = rand.adj
    bare_eigvals, bare_eigvecs = np.linalg.eigh(bare_hamiltonian)

    N = len(rand.node_map)
    psi_i = e_n(0, N)
    end = e_n(N - 1, N)

    # Define the time array over which evolution is evaluated
    num_flux=2000
    #fluxes = np.linspace(2*np.pi/num_flux,2*np.pi, num_flux-1)
    fluxes = np.linspace(0,4*np.pi, num_flux)

    #now generate the averaged fluxed random cycle graph
    necklace_seeds = list(enumerate_necklaces(int(np.prod(X))))
    flux_limit_dict={}#{0:limit(bare_eigvecs, psi_i, end)}

    for seed in tqdm(necklace_seeds, desc=f"Iteration{i}: Sweeping Necklace", unit="Necklace"):
        #generate seed
        rand = generate_random_cycle_graph(ftree,seed)
        rand.construct_fluxed()

        #loop over possible fluxes
        for flux in fluxes:
            #create Hamiltonian for this 
            fluxed_hamiltonian = rand.weighted_adj(flux) 
            fluxed_eigvals, fluxed_eigvecs = np.linalg.eigh(fluxed_hamiltonian)   
    
            flux_lim_curr  = limit(fluxed_eigvecs, psi_i,end)            

            if flux in flux_limit_dict:
                flux_limit_dict[flux].append(flux_lim_curr)
            else:
                flux_limit_dict[flux] =  [flux_lim_curr]

    #average the time-series in the prob_flux_dict in order to get the average evolution over all necklaces
    for key, value in flux_limit_dict.items():
        flux_limit_dict[key] = np.mean(value)   
    
    # Save the time and probability arrays to a file for later plotting
    filename = f"limitresults/{p}_{i}_limit.npz"
    np.savez(filename, flux_limit_dict=flux_limit_dict)
    print(f"Results for iteration {i} saved to {filename}")