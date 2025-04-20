import random as random

from _graphtools import *
from _vectools import *
from _counttools import *
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import warnings
import argparse


#### Functions
@njit
def limit(eigvecs, a, b):
    total = 0.0
    n = a.shape[0]  # dimension of the vectors; same as number of rows in eigvecs
    # Loop over each eigenvector, which is assumed to be stored as a column
    for i in range(n):
        dot_a = 0.0 + 0.0j
        dot_b = 0.0 + 0.0j
        # Calculate the dot products for a and b with the i-th eigenvector
        dot_a = np.conjugate(a) @ eigvecs[:,i]
        dot_b = np.conjugate(b) @ eigvecs[:,i]

        #for j in range(n):
            #dot_a += np.conjugate(a[j]) * eigvecs[j, i]
            #dot_b += np.conjugate(b[j]) * eigvecs[j, i]

        prod = dot_a * dot_b
        total += prod.real * prod.real + prod.imag * prod.imag  # equivalent to np.abs(prod)**2
    return total

def compute_flux_limit(gt, flux, psi_i, end):
    # Build fluxed Hamiltonian
    fluxed_hamiltonian = gt.weighted_adj(flux)
    # Diagonalize
    _, fluxed_eigvecs = np.linalg.eigh(fluxed_hamiltonian)
    # Compute limit
    return limit(fluxed_eigvecs, psi_i, end)


#### command line arguments 

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(prog = 'MC Simulation of Continous Random Walks', description = 'Calculates Time Evolution on Glued Random Cycles Trees')

parser.add_argument('-X', help = 'Growth Sequence')
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

    fluxes = np.linspace(0,2*np.pi, args.f)

    if args.AFB:
        Z = np.prod(X)
        FBP = np.array([2*np.pi/Z * i for i in range(1, Z+1)])
        fluxes = np.sort(np.append(fluxes, FBP))
    #now generate the averaged fluxed random cycle graph
    flux_limits = []
    flux_limits = Parallel(n_jobs=8)(
        delayed(compute_flux_limit)(gt, flux, psi_i, end)
        for flux in tqdm(fluxes)
    )

# Save the time and probability arrays to a file for later plotting
    if args.outfile is None:
        outfile = f"limitresults/{X}_gt_lim.npz"
    else:
        outfile = args.outfile
    np.savez(outfile, fluxes=fluxes, flux_limits=flux_limits, sequence=X)
    print(f"Results saved to {outfile}")

    