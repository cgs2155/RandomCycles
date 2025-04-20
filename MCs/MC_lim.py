import random as random

from _graphtools import *
from _vectools import *
from _counttools import *
from numba import njit

import numpy as np
import warnings
import argparse
import pickle

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
parser = argparse.ArgumentParser(prog = 'MC Simulation of Continous Random Walks', description = 'Calculates Time Evolution on Glued Random Cycles Trees')

parser.add_argument('-X', help = 'Growth Sequence')
parser.add_argument('-f', type = int, help = 'Number of Flux Steps')
parser.add_argument('-s', type = int, help = 'Seed')

parser.add_argument('-N', type=int, help="Number of Necklaces Sampled")

parser.add_argument('-c', help = 'The file to use if continuing old calculations')

parser.add_argument('-outfile', help = 'Output file holding neccessary information for plotting and generation seeds')


args = parser.parse_args()


#### Runtime
if __name__ == '__main__':
    #If beginning a calculation
    if args.c is None:
        seed = args.s
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**32 - 1)
        #set random seed
        #random.seed(seed)
        necklace_rng = random.Random(seed)

        X = [int(i) for i in args.X.split(",")]
        neck_len = np.prod(X)

        ftree = fluxedTree(X)
        ftree.construct_fluxed()

        rand = generate_random_cycle_graph(ftree)
        rand.construct_adj()
        bare_hamiltonian = rand.adj
        bare_eigvals, bare_eigvecs = np.linalg.eigh(bare_hamiltonian)

        N = len(rand.node_map)
        psi_i = e_n(0, N)
        end = e_n(N - 1, N)

        fluxes = np.linspace(0,4*np.pi, args.f)

        #now generate the averaged fluxed random cycle graph
        neck_strings=set()
        with tqdm(total=args.N,desc="Generating Necklaces") as pbar:
            while len(neck_strings) < args.N:
                curr = len(neck_strings)
                neck_strings.add(gen_necklace(neck_len, necklace_rng))
                if curr < len(neck_strings):
                    pbar.update(1)
        flux_limit_dict={}

        for neck in tqdm(neck_strings,desc=f"Getting Necklace Limit", unit="Necklace"):       
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

        #save state
        rng_bytes = pickle.dumps(necklace_rng.getstate())
        rng_arr   = np.frombuffer(rng_bytes, dtype=np.uint8)

        np.savez(outfile, flux_limit_dict=flux_limit_dict, seed=seed, necklaces=neck_strings, sequence=X,rngstate=rng_arr)
        print(f"Results saved to {outfile}")

    #If continuing a calculation
    else:
        old_file = args.c
        old_data = np.load(old_file,allow_pickle=True)

        X = old_data["sequence"]
        seed = int(old_data["seed"])
        old_necklaces = old_data["necklaces"].item()

        rng_arr = old_data["rngstate"]              # a uint8 array
        rng_bytes = rng_arr.tobytes()
        saved_state = pickle.loads(rng_bytes)
        neck_rng = random.Random()
        neck_rng.setstate(saved_state)

        neck_len = np.prod(X)


        old_flux_dict = old_data["flux_limit_dict"].item()
        fluxes = list(old_flux_dict.keys())

        ftree = fluxedTree(X)
        ftree.construct_fluxed()

        rand = generate_random_cycle_graph(ftree)
        rand.construct_adj()

        neck_strings=set()
        with tqdm(total=args.N,desc="Generating Necklaces") as pbar:
            while len(neck_strings) < args.N:
                curr = len(neck_strings)
                new_neck = gen_necklace(neck_len, necklace_rng)
                if new_neck not in old_necklaces:
                    neck_strings.add(new_neck)                    
                if curr < len(neck_strings):
                    pbar.update(1)

        flux_limit_dict={}
        N = len(rand.node_map)
        psi_i = e_n(0, N)
        end = e_n(N - 1, N)

        for neck in tqdm(neck_strings,desc=f"Getting Necklace Limit", unit="Necklace"):       
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
        norm = len(old_necklaces) + len(neck_strings)
        new_prob_flux_dict = {key: (len(neck_strings)/norm* flux_limit_dict[key] + len(old_necklaces)/norm* old_flux_dict[key]) for key in flux_limit_dict}     

        #Save the time and probability arrays to a file for later plotting
        neck_strings = old_necklaces | neck_strings

        if args.outfile is None:
            outfile = f"limitresults/{seed}_{len(neck_strings)}_MClim.npz"
        else:
            outfile = args.outfile

        rng_bytes = pickle.dumps(neck_rng.getstate())
        rng_arr   = np.frombuffer(rng_bytes, dtype=np.uint8)

        np.savez(outfile, flux_limit_dict=flux_limit_dict, seed=seed, necklaces=neck_strings, sequence=X, rngstate=rng_arr)
        print(f"Results saved to {outfile}")