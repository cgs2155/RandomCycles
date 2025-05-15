import random as random

from _graphtools import *
from _vectools import *
from _counttools import *
from numba import njit

import numpy as np
import warnings
import argparse
import pickle
import tqdm as tqdm

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
        for j in range(n):
            dot_a += np.conjugate(a[j]) * eigvecs[j, i]
            dot_b += np.conjugate(b[j]) * eigvecs[j, i]
        prod = dot_a * dot_b
        total += prod.real * prod.real + prod.imag * prod.imag  # equivalent to np.abs(prod)**2
    return total

#### command line arguments 

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(prog = 'MC of Limiting Distributions', description = 'Calculates Time Evolution on Glued Random Cycles Trees')

parser.add_argument('-X', help = 'Growth Sequence')
parser.add_argument('-f', type = int, help = 'Number of Flux Steps')
parser.add_argument('-s', type = int, help = 'Seed')

parser.add_argument('-N', type=int, help="Number of Necklaces Sampled")

parser.add_argument('-c', help = 'The file to use if continuing old calculations')

parser.add_argument('-outfile', help = 'Output file holding neccessary information for plotting and generation seeds')
parser.add_argument('--AFB', action='store_true', help='Include AFB points')


args = parser.parse_args()


#### Runtime
if __name__ == '__main__':
    #If beginning a calculation
    if args.c is None:
        seed = args.s
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**32 - 1)
        #set random seed
        necklace_rng = random.Random(seed)

        X = [int(i) for i in args.X.split(",")]
        neck_len = np.prod(X)

        ftree = fluxedTree(X)
        ftree.construct_fluxed()

        N = 2*tree_mag(X)
        psi_i = e_n(0, N)
        end_idx= tree_mag(X)
        end = e_n(end_idx,N)

        fluxes = np.linspace(0,4*np.pi, args.f)
        if args.AFB:
            Z = np.prod(X)
            FBP = np.array([2*np.pi/Z * i for i in range(1, Z+1)])
            fluxes = np.sort(np.append(fluxes, FBP))

        #now generate the averaged fluxed random cycle graph
        neck_strings=set()

        with tqdm.tqdm(total=args.N,desc="Generating Necklaces") as pbar:
            while len(neck_strings) < args.N:
                curr = len(neck_strings)
                neck = gen_necklace(neck_len, necklace_rng)
                if neck not in neck_strings:
                    neck_strings.add(neck)
                if curr < len(neck_strings):
                    pbar.update(1)

        flux_limit_dict={}

        for neck in tqdm.tqdm(neck_strings,desc=f"Getting Necklace Limit", unit="Necklace"):       
            #generate seed
            rgc = generate_random_cycle_graph(ftree,neck)
            rgc.construct_fluxed()

            #loop over possible fluxes
            for flux in fluxes:
                #create Hamiltonian for this 
                fluxed_hamiltonian = rgc.weighted_adj(flux) 
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
        if args.outfile is None:
            outfile = f"limitresults/{seed}_{args.N}_MClim.npz"
        else:
            outfile = args.outfile

        np.savez(outfile, flux_limit_dict=flux_limit_dict, seed=seed, N=args.N, sequence=X)
        print(f"Results saved to {outfile}")

    #If continuing a calculation
    else:
        old_file = args.c
        old_data = np.load(old_file,allow_pickle=True)

        X = old_data["sequence"]
        seed = int(old_data["seed"])
        oldN = old_data["N"]
        neck_rng = random.Random(seed)

        neck_len = np.prod(X)
        old_flux_dict = old_data["flux_limit_dict"].item()
        fluxes = list(old_flux_dict.keys())
        ftree = fluxedTree(X)
        ftree.construct_fluxed()

        neck_strings = set()
        trash_neck = set()
        with tqdm.tqdm(total=oldN,desc="Generating Old Necklaces") as pbar:      
            while len(trash_neck) < oldN:
                curr = len(trash_neck)
                neck = gen_necklace(neck_len, neck_rng)
                if neck not in trash_neck:
                    trash_neck.add(neck)
                if curr < len(trash_neck):
                    pbar.update(1)

        with tqdm.tqdm(total=args.N,desc="Generating Necklaces") as pbar:      
            while len(neck_strings) < args.N:
                curr = len(neck_strings)
                neck = gen_necklace(neck_len, neck_rng)
                if neck not in neck_strings and neck not in trash_neck:
                    neck_strings.add(neck)
                if curr < len(neck_strings):
                    pbar.update(1)

        flux_limit_dict={}
        N = 2*tree_mag(X)
        psi_i = e_n(0, N)
        end_idx= tree_mag(X)
        end = e_n(end_idx,N)

        for neck in tqdm.tqdm(neck_strings,desc=f"Getting Necklace Limit", unit="Necklace"):       
            #generate seed
            rgc = generate_random_cycle_graph(ftree,neck)
            rgc.construct_fluxed()

            #loop over possible fluxes
            for flux in fluxes:
                #create Hamiltonian for this 
                fluxed_hamiltonian = rgc.weighted_adj(flux) 
                fluxed_eigvals, fluxed_eigvecs = np.linalg.eigh(fluxed_hamiltonian)   

                flux_lim_curr  = limit(fluxed_eigvecs, psi_i,end)            
                if flux in flux_limit_dict:
                    flux_limit_dict[flux].append(flux_lim_curr)
                else:
                    flux_limit_dict[flux] =  [flux_lim_curr]

        #average the time-series in the prob_flux_dict in order to get the average evolution over all necklaces
        for key, value in flux_limit_dict.items():
            flux_limit_dict[key] = np.mean(value)   


        #average the new necklaces with the old necklaces
        M = oldN
        N = len(neck_strings)
        norm = M+N
        new_prob_flux_dict = {key: (N/norm* flux_limit_dict[key] + M/norm* old_flux_dict[key]) for key in flux_limit_dict}     

        #Save the time and probability arrays to a file for later plotting
        if args.outfile is None:
            outfile = f"limitresults/{seed}_{norm}_MClim.npz"
        else:
            outfile = args.outfile

        rng_bytes = pickle.dumps(neck_rng.getstate())
        rng_arr   = np.frombuffer(rng_bytes, dtype=np.uint8)

        np.savez(outfile, flux_limit_dict=flux_limit_dict, seed=seed, sequence=X, N = norm)
        print(f"Results saved to {outfile}")