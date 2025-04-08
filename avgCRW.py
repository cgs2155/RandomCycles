import numpy as np
import math
from scipy.linalg import expm
from _graphtools import *
from _vectools import *
from _counttools import *
from tqdm import tqdm  # import tqdm for progress bars

def evolve(H, psi0, t):
    U = expm(-1j * H * t)  # time evolution operator U(t)
    psi_t = U @ psi0
    return psi_t

def U_evo(eigvals, eigvecs, t):
    phase_factors = np.exp(-1j * eigvals * t)
    return eigvecs @ (phase_factors * (eigvecs.conj().T))

def roundup(x):
    return math.ceil(x / 10.0) * 10

p = 2
n_max = 2
AFB = 2 * np.pi / p

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
    num_time=400
    num_flux=200
    t = np.linspace(0, 2*i*p, num_time)
    fluxes = np.linspace(4*np.pi/(num_flux),4*np.pi, num_flux-1)
    delta_t = t[1]
    U_bare = expm(-1j * bare_hamiltonian * delta_t)

    prob_bare = [0]
    psi_curr_bare = psi_i

    # Use tqdm to show progress for each time step
    for time in t[1:]:
        psi_curr_bare = U_bare@psi_curr_bare
        prob_bare.append(np.abs(end.conj() @ psi_curr_bare)**2)

    #now generate the averaged fluxed random cycle graph
    necklace_seeds = list(enumerate_necklaces(int(np.prod(X))))
    prob_flux_dict={0:prob_bare}

    for seed in tqdm(necklace_seeds, desc=f"Iteration{i}: Sweeping Necklace", unit="Necklace"):
        #generate seed
        rand = generate_random_cycle_graph(ftree,seed)
        rand.construct_fluxed()

        #loop over possible fluxes
        for flux in fluxes:
            #create Hamiltonian for this 
            fluxed_hamiltonian = rand.weighted_adj(flux) 
            fluxed_eigvals, fluxed_eigvecs = np.linalg.eigh(fluxed_hamiltonian)   
            U_fluxed = expm(-1j * fluxed_hamiltonian * delta_t)
        

            prob_flux_curr = [0]
            psi_curr_fluxed = psi_i
    
            for time in t[1:]:
                psi_curr_fluxed = U_fluxed@psi_curr_fluxed
                prob_flux_curr.append(np.abs(end.conj() @ psi_curr_fluxed)**2)
            
            if flux in prob_flux_dict:
                prob_flux_dict[flux].append(prob_flux_curr)
            else:
                prob_flux_dict[flux] =  [prob_flux_curr]

    #average the time-series in the prob_flux_dict in order to get the average evolution over all necklaces
    for key, value in prob_flux_dict.items():
        if key != 0:
            prob_flux_dict[key] = [sum(group) / len(group) for group in zip(*value)]    
    
    # Save the time and probability arrays to a file for later plotting
    filename = f"walkresults/{p}avg_rc_{i}.npz"
    np.savez(filename, t=t, prob_flux_dict=prob_flux_dict, prob_bare=prob_bare)
    print(f"Results for iteration {i} saved to {filename}")