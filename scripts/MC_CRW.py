import random as random

from _graphtools import *
from _vectools import *
from _counttools import *
from numba import njit

import numpy as np
import warnings
import argparse
import time
import pickle
import tqdm as tqdm

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
        #set seed
        necklace_rng = random.Random(seed)

        X = [int(i) for i in args.X.split(",")]
        neck_len = np.prod(X)

        ftree = fluxedTree(X)
        ftree.construct_fluxed()

        rand = generate_random_cycle_graph(ftree)
        rand.construct_adj()
        bare_hamiltonian = rand.adj

        N = 2*tree_mag(X)
        psi_i = e_n(0, N)
        end_idx= tree_mag(X)
        end = e_n(end_idx,N)

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
        neck_list = []

        with tqdm.tqdm(total=args.N,desc="Generating Necklaces") as pbar:
            while len(neck_strings) < args.N:
                curr = len(neck_strings)
                neck = gen_necklace(neck_len, necklace_rng)
                if neck not in neck_strings:
                    neck_strings.add(neck)
                    neck_list.append(neck)

                if curr < len(neck_strings):
                    pbar.update(1)

        prob_flux_dict={0:prob_bare}

        for neck in tqdm.tqdm(neck_list,desc=f"Evolving Necklaces", unit="Necklace"):       
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
        
                for index in range(1, len(t)):
                    psi_curr_fluxed = U_fluxed@psi_curr_fluxed
                    prob_flux_curr[index] = (np.abs(end.conj() @ psi_curr_fluxed)**2)

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

        #save state
        rng_bytes = pickle.dumps(necklace_rng.getstate())
        rng_arr   = np.frombuffer(rng_bytes, dtype=np.uint8)

        #np.savez(outfile, t=t, prob_flux_dict=prob_flux_dict, prob_bare=prob_bare, seed=seed, necklaces=neck_strings, sequence=X, rngstate = rng_arr, N=args.N)
        np.savez(outfile, t=t, prob_flux_dict=prob_flux_dict, prob_bare=prob_bare, seed=seed,sequence=X, rngstate = rng_arr, N=args.N)

        print(f"Results saved to {outfile}")

    #If continuing a calculation
    else:
        old_file = args.c
        old_data = np.load(old_file,allow_pickle=True)

        X = old_data["sequence"]
        seed = int(old_data["seed"])
        #old_necklaces = old_data["necklaces"].item()
        oldN = old_data["N"]

        #rng_arr = old_data["rngstate"]              # a uint8 array
        #rng_bytes = rng_arr.tobytes()
        #saved_state = pickle.loads(rng_bytes)
        #neck_rng = random.Random()
        #neck_rng.setstate(saved_state)
        neck_rng = random.Random(seed)

        neck_len = np.prod(X)

        t = old_data["t"]
        delta_t = t[1]
        prob_bare = old_data["prob_bare"]

        old_flux_dict = old_data["prob_flux_dict"].item()
        fluxes = list(old_flux_dict.keys())[1:]

        ftree = fluxedTree(X)

        #neck_strings = set(old_necklaces)
        #neck_list = []

        """with tqdm.tqdm(total=args.N,desc="Generating Necklaces") as pbar:
            while len(neck_list) < args.N:
                curr = len(neck_strings)
                neck = gen_necklace(neck_len, neck_rng)
                if neck not in neck_strings:
                    neck_strings.add(neck)
                    neck_list.append(neck)
                if curr < len(neck_strings):
                    pbar.update(1)
        """
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
        #evolve the new necklaces
        prob_flux_dict={0:prob_bare}

        N = 2*tree_mag(X)
        psi_i = e_n(0, N)
        end_idx= tree_mag(X)
        end = e_n(end_idx,N)

        #for neck in tqdm.tqdm(neck_list,desc=f"Evolving Necklaces", unit="Necklace"):  
        for neck in tqdm.tqdm(neck_strings,desc=f"Evolving Necklaces", unit="Necklace"):       
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
        
                for index in range(1, len(t)):
                    psi_curr_fluxed = U_fluxed@psi_curr_fluxed
                    prob_flux_curr[index] = (np.abs(end.conj() @ psi_curr_fluxed)**2)

                if flux in prob_flux_dict:
                    prob_flux_dict[flux].append(prob_flux_curr)
                else:
                    prob_flux_dict[flux] =  [prob_flux_curr]

        #average the time-series in the prob_flux_dict in order to get the average evolution over all necklaces
        for key, value in prob_flux_dict.items():
            if key != 0:
                prob_flux_dict[key] = [sum(group) / len(group) for group in zip(*value)]    

        #average the new necklaces with the old necklaces
        #M = len(old_necklaces)
        #N = len(neck_list)
        M = oldN
        N = len(neck_strings)
        norm = M+N
        new_prob_flux_dict = {key: (N/norm* np.array(prob_flux_dict[key])+ M/norm* np.array(old_flux_dict[key])) for key in prob_flux_dict}     

        #Save the time and probability arrays to a file for later plotting

        if args.outfile is None:
            outfile = f"walkresults/{seed}_{norm}_MC.npz"
        else:
            outfile = args.outfile

        #save state
        rng_bytes = pickle.dumps(neck_rng.getstate())
        rng_arr   = np.frombuffer(rng_bytes, dtype=np.uint8)
        
        #np.savez(outfile, t=t, prob_flux_dict=new_prob_flux_dict, prob_bare=prob_bare, seed=seed, necklaces=neck_strings, sequence=X,rngstate = rng_arr)
        np.savez(outfile, t=t, prob_flux_dict=new_prob_flux_dict, prob_bare=prob_bare, seed=seed, sequence=X,rngstate = rng_arr, N = norm)
        print(f"Results saved to {outfile}")

        stop = time.time()
        print(f"Elapsed Time: {stop-start} seconds")