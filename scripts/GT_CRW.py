import random as random

from _graphtools import *
from _vectools import *
from _counttools import *
from numba import njit

import numpy as np
import warnings
import argparse

#### Functions

#### command line arguments 

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(prog = 'Continous Random Walks on Glued Trees', description = 'Calculates Time Evolution on Glued Random Cycles Trees')

parser.add_argument('-X', help = 'Growth Sequence')
parser.add_argument('-t', type = int, help = 'Number of Time Steps')
parser.add_argument('-tmax', type = float, help = 'Maximum Time')
parser.add_argument('-f', type = int, help = 'Number of Flux Steps')

parser.add_argument('-outfile', help = 'Output file holding neccessary information for plotting and generation seeds')
parser.add_argument('--AFB', action='store_true', help='Include AFB points')


args = parser.parse_args()


#### Runtime
if __name__ == '__main__':
    #If beginning a calculation
    X = [int(i) for i in args.X.split(",")]
    gt = createGT(X)
    gt.construct_adj()
    bare_hamiltonian = gt.adj
    bare_eigvals, bare_eigvecs = np.linalg.eigh(bare_hamiltonian)

    N = len(gt.node_map)
    psi_i = e_n(0, N)
    end = e_n(N - 1, N)

    t = np.linspace(0,args.tmax, args.t)
    delta_t = t[1]
    fluxes = np.linspace(0,2*np.pi, args.f)

    if args.AFB:
        Z = np.prod(X)
        FBP = np.array([2*np.pi/Z * i for i in range(1, Z+1)])
        fluxes = np.sort(np.append(fluxes, FBP))

    U_bare = expm(-1j * bare_hamiltonian * delta_t)

    prob_bare = [0]
    psi_curr_bare = psi_i

    # Use tqdm to show progress for each time step
    for time in t[1:]:
        psi_curr_bare = U_bare@psi_curr_bare
        prob_bare.append(np.abs(end.conj() @ psi_curr_bare)**2)


    prob_flux_dict={0:prob_bare}

    #loop over possible fluxes
    for flux in fluxes[1:]:
        #create Hamiltonian for this 
        fluxed_hamiltonian = gt.weighted_adj(flux) 
        U_fluxed = expm(-1j * fluxed_hamiltonian * delta_t)

        prob_flux_curr = [0]
        psi_curr_fluxed = psi_i

        for time in t[1:]:
            psi_curr_fluxed = U_fluxed@psi_curr_fluxed
            prob_flux_curr.append(np.abs(end.conj() @ psi_curr_fluxed)**2)
        
        prob_flux_dict[flux] = prob_flux_curr

# Save the time and probability arrays to a file for later plotting
    if args.outfile is None:
        outfile = f"walkresults/{X}_gt_CRW.npz"
    else:
        outfile = args.outfile
    np.savez(outfile, t=t, prob_flux_dict=prob_flux_dict, prob_bare=prob_bare, sequence=X)
    print(f"Results saved to {outfile}")
