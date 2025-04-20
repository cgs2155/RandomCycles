import random as random

from _graphtools import *
from _vectools import *
from _counttools import *
from numba import njit

import numpy as np
import warnings
import argparse
import time

#import ray
##### Functions
@ray.remote
def evolve_necklace(neck, ftree, fluxes, delta_t, psi_i, t, end):
    """
    Given a necklace seed string, builds the fluxed graph, loops over flux values,
    and computes the evolution time-series for each flux.
    Returns a dictionary mapping flux->evolution probability array.
    """
    # generate the random cycle graph using the given necklace seed
    rgc = generate_random_cycle_graph(ftree, neck)
    rgc.construct_fluxed()

    results = {}  # to store the simulation for each flux
    for flux in fluxes:
        # generate the Hamiltonian for this flux and create the time evolution operator
        fluxed_hamiltonian = rgc.weighted_adj(flux)
        # For safety, if you need eigen-decomposition you can add it here:
        # fluxed_eigvals, fluxed_eigvecs = np.linalg.eigh(fluxed_hamiltonian)
        U_fluxed = expm(-1j * fluxed_hamiltonian * delta_t)
        
        # Initialize the probability array and the state vector for this flux simulation
        prob_flux_curr = np.zeros(len(t), dtype=np.float64)
        psi_curr_flux = psi_i.copy()

        # Evolve the state in time
        for index in range(1, len(t)):
            psi_curr_flux = U_fluxed @ psi_curr_flux
            prob_flux_curr[index] = np.abs(end.conj() @ psi_curr_flux) ** 2

        results[flux] = prob_flux_curr

    return results



#### command line arguments 

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(prog = 'MC Simulation of Continous Random Walks', description = 'Calculates Time Evolution on Glued Random Cycles Trees')

parser.add_argument('-X', help = 'Growth Sequence')
parser.add_argument('-t', type = int, help = 'Number of Time Steps')
parser.add_argument('-tmax', type = float, help = 'Maximum Time')
parser.add_argument('-f', type = int, help = 'Number of Flux Steps')
parser.add_argument('-N', type=int, help="Number of Necklaces Sampled")

parser.add_argument('-s', type = int, help = 'Seed')
parser.add_argument('-c', help = 'The file to use if continuing old calculations')
parser.add_argument('-outfile', help = 'Output file holding neccessary information for plotting and generation seeds')
parser.add_argument('--AFB', action='store_true', help='Include AFB points')


args = parser.parse_args()

#### Runtime
if __name__ == '__main__':
    start = time.time()
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

        t = np.linspace(0,args.tmax, args.t)
        fluxes = np.linspace(4*np.pi/(args.f),4*np.pi, args.f-1)

        if args.AFB:
            Z = np.prod(X)
            FBP = np.array([2*np.pi/Z * i for i in range(1, Z+1)])
            fluxes = np.sort(np.append(fluxes, FBP))

        delta_t = t[1]

        U_bare = expm(-1j * bare_hamiltonian * delta_t)

        prob_bare = np.zeros(len(t))
        psi_curr_bare = psi_i

        for index in range(1,len(t)):
            psi_curr_bare = U_bare@psi_curr_bare
            prob_bare[index] = (np.abs(end.conj() @ psi_curr_bare)**2)

        #now generate the averaged fluxed random cycle graph
        neck_strings=set()
        with tqdm(total=args.N,desc="Generating Necklaces") as pbar:
            while len(neck_strings) < args.N:
                curr = len(neck_strings)
                neck_strings.add(gen_necklace(neck_len, necklace_rng))
                if curr < len(neck_strings):
                    pbar.update(1)
        prob_flux_dict={0:prob_bare}

        for neck in tqdm(neck_strings,desc=f"Evolving Necklaces", unit="Necklace"):       
            #generate seed
            rgc = generate_random_cycle_graph(ftree,neck)
            rgc.construct_fluxed()

            #loop over possible fluxes
            for flux in fluxes:
                #create Hamiltonian for this 
                fluxed_hamiltonian = rgc.weighted_adj(flux) 
                fluxed_eigvals, fluxed_eigvecs = np.linalg.eigh(fluxed_hamiltonian)   
                U_fluxed = expm(-1j * fluxed_hamiltonian * delta_t)
            
                prob_flux_curr = np.zeros(len(t))
                psi_curr_fluxed = psi_i
        
                for index in range(1, len(t)):
                    psi_curr_fluxed = U_bare@psi_curr_fluxed
                    prob_flux_curr[index] = (np.abs(end.conj() @ psi_curr_bare)**2)

                if flux in prob_flux_dict:
                    prob_flux_dict[flux].append(prob_flux_curr)
                else:
                    prob_flux_dict[flux] =  [prob_flux_curr]

        #average the time-series in the prob_flux_dict in order to get the average evolution over all necklaces
        for key, value in prob_flux_dict.items():
            if key != 0:
                prob_flux_dict[key] = [sum(group) / len(group) for group in zip(*value)]    
    
    # Save the time and probability arrays to a file for later plotting
        if args.outfile is None:
            outfile = f"walkresults/{seed}_{args.N}_MC.npz"
        else:
            outfile = args.outfile
        np.savez(outfile, t=t, prob_flux_dict=prob_flux_dict, prob_bare=prob_bare, seed=seed, necklaces=neck_strings, sequence=X)
        print(f"Results saved to {outfile}")

    #If continuing a calculation
    else:
        old_file = args.c
        old_data = np.load(old_file,allow_pickle=True)

        X = old_data["sequence"]
        seed = int(old_data["seed"])
        old_necklaces = old_data["necklaces"].item()
        necklace_rng = random.Random(seed)
        neck_len = np.prod(X)

        t = old_data["t"]
        delta_t = t[1]
        prob_bare = old_data["prob_bare"]

        old_flux_dict = old_data["prob_flux_dict"].item()
        fluxes = list(old_flux_dict.keys())[1:]

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

        #evolve the new necklaces
        prob_flux_dict={0:prob_bare}

        N = len(rand.node_map)
        psi_i = e_n(0, N)
        end = e_n(N - 1, N)

        for neck in tqdm(neck_strings,desc=f"Evolving Necklaces", unit="Necklace"):       
            #generate seed
            rgc = generate_random_cycle_graph(ftree,neck)
            rgc.construct_fluxed()

            #loop over possible fluxes
            for flux in fluxes:
                #create Hamiltonian for this 
                fluxed_hamiltonian = rgc.weighted_adj(flux) 
                U_fluxed = expm(-1j * fluxed_hamiltonian * delta_t)
                
                prob_flux_curr = np.zeros(len(t))
                psi_curr_fluxed = psi_i
        
                for index in range(1,len(t)):
                    psi_curr_fluxed = U_fluxed@psi_curr_fluxed
                    prob_flux_curr[index] = (np.abs(end.conj() @ psi_curr_fluxed)**2)

        #average the time-series in the prob_flux_dict in order to get the average evolution over all necklaces
        for key, value in prob_flux_dict.items():
            if key != 0:
                prob_flux_dict[key] = [sum(group) / len(group) for group in zip(*value)]    


        #average the new necklaces with the old necklaces
        norm = len(old_necklaces) + len(neck_strings)
        new_prob_flux_dict = {key: (len(neck_strings)/norm* np.array(prob_flux_dict[key])+ len(old_necklaces)/norm* np.array(old_flux_dict[key])) for key in prob_flux_dict}     

        #Save the time and probability arrays to a file for later plotting
        neck_strings = old_necklaces | neck_strings

        if args.outfile is None:
            outfile = f"walkresults/{seed}_{len(neck_strings)}_MC.npz"
        else:
            outfile = args.outfile
        np.savez(outfile, t=t, prob_flux_dict=new_prob_flux_dict, prob_bare=prob_bare, seed=seed, necklaces=neck_strings, sequence=X)
        print(f"Results saved to {outfile}")

        end = time.time()
        print(f"Elapsed Time: {end-start} seconds")