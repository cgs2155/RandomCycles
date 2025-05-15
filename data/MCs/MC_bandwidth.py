from _walktools import tbw
from _graphtools import *
from _vectools import e_n
from _counttools import gen_necklace

import warnings 
import argparse
import time
import tqdm as tqdm
import pickle

#### FUNCTIONS #####
def bandwidths(graph, face_fluxes, kvalues):
    bws = np.zeros(len(face_fluxes))
    #number of eigenvalues (subtract one for identifying the first and last node)
    N = len(graph.nodes) - 1
    for i, flux in enumerate(face_fluxes):
        graph.construct_period(flux)
        bands = np.zeros((len(kvalues),N)) 
        #print(np.shape(bands))
        for j, k in enumerate(kvalues):
            #print(j)
            #print(type(j))
            bands[j] = np.linalg.eigvalsh(graph.period(k))
        bws[i] = tbw(np.transpose(bands))
    return bws

#### command line arguments 

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(prog = 'MC Simulation of Continous Random Walks', description = 'Calculates Time Evolution on Glued Random Cycles Trees')

parser.add_argument('-X', help = 'Growth Sequence')
parser.add_argument('-f', type = int, help = 'Number of Flux Steps')
parser.add_argument('-k', type = int, help = 'Number of k Steps')
parser.add_argument('-N', type=int, help="Number of Necklaces Sampled")

parser.add_argument('-s', type = int, help = 'Seed')
parser.add_argument('-c', help = 'The file to use if continuing old calculations')
parser.add_argument('-outfile', help = 'Output file holding neccessary information for plotting and generation seeds')
parser.add_argument('--AFB', action='store_true', help='Include AFB points')

args = parser.parse_args()

#### RUNTIME
if __name__ == '__main__':
    start = time.time()
     #If beginning a calculation
    if args.c is None:
        seed = args.s
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**32 - 1)

        necklace_rng = random.Random(seed)

        X = [int(i) for i in args.X.split(",")]
        neck_len = np.prod(X)

        ftree = fluxedTree(X)       
        fluxes = np.linspace(0,4*np.pi, args.f-1)
        kvalues = np.linspace(0, 2*np.pi, args.k)

        if args.AFB:
            Z = np.prod(X)
            FBP = np.array([2*np.pi/Z * i for i in range(1, Z+1)])
            fluxes = np.sort(np.append(fluxes, FBP))


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

        new_sum = None
        for neck in tqdm.tqdm(neck_list, desc="Evolving Necklaces"):
            #generate seed
            rgc = generate_random_cycle_graph(ftree,neck)
            rgc.construct_fluxed()

            bw = bandwidths(rgc, fluxes, kvalues)
            if new_sum is None:
                new_sum = bw.copy()
            else:
                new_sum += bw
        neck_widths = new_sum / len(neck_list)
    
    # Save the time and probability arrays to a file for later plotting
        if args.outfile is None:
            outfile = f"bw/{seed}_{args.N}_MCbw.npz"
        else:
            outfile = args.outfile
        
        #save state
        rng_bytes = pickle.dumps(necklace_rng.getstate())
        rng_arr   = np.frombuffer(rng_bytes, dtype=np.uint8)

        np.savez(outfile, fluxes=fluxes, k=args.k, avgbw=neck_widths, seed=seed, necklaces=neck_strings, sequence=X, rngstate=rng_arr)
        print(f"Results saved to {outfile}")

    #If continuing a calculation
    else:
        old_file = args.c
        old_data = np.load(old_file,allow_pickle=True)

        X = old_data["sequence"]
        seed = int(old_data["seed"])
        old_necklaces = old_data["necklaces"].item()
        old_neckwidths = old_data['avgbw']

        rng_arr = old_data["rngstate"]              # a uint8 array
        rng_bytes = rng_arr.tobytes()
        saved_state = pickle.loads(rng_bytes)
        neck_rng = random.Random()
        neck_rng.setstate(saved_state)

        neck_len = np.prod(X)

        fluxes = old_data['fluxes']
        kvalues = np.linspace(0, 2*np.pi, old_data['k'])

        ftree = fluxedTree(X)
        ftree.construct_fluxed()

        rand = generate_random_cycle_graph(ftree)
        rand.construct_adj()


        neck_strings = set(old_necklaces)
        neck_list = []

        with tqdm.tqdm(total=args.N,desc="Generating Necklaces") as pbar:
            while len(neck_list) < args.N:
                curr = len(neck_strings)
                neck = gen_necklace(neck_len, neck_rng)
                if neck not in neck_strings:
                    neck_strings.add(neck)
                    neck_list.append(neck)
                if curr < len(neck_strings):
                    pbar.update(1)

        new_sum = None
        for neck in tqdm.tqdm(neck_list, desc="Evolving Necklaces"):
            #generate seed
            rgc = generate_random_cycle_graph(ftree,neck)
            rgc.construct_fluxed()

            bw = bandwidths(rgc, fluxes, kvalues)
            if new_sum is None:
                new_sum = bw.copy()
            else:
                new_sum += bw
        new_avg = new_sum / len(neck_list)

        #average the new necklaces with the old necklaces
        M = len(old_necklaces)
        N = len(neck_list)
        neck_widths = (M*old_neckwidths + N*new_avg) / (M + N)

        #Save the time and probability arrays to a file for later plotting
        neck_strings = old_necklaces | neck_strings

        if args.outfile is None:
            outfile = f"bw/{seed}_{len(neck_strings)}_MCbw.npz"
        else:
            outfile = args.outfile

        rng_bytes = pickle.dumps(neck_rng.getstate())
        rng_arr   = np.frombuffer(rng_bytes, dtype=np.uint8)

        np.savez(outfile, fluxes=fluxes, k=old_data['k'], avgbw=neck_widths, seed=seed, necklaces=neck_strings, sequence=X, rngstate=rng_arr)
        print(f"Results saved to {outfile}")

        end = time.time()
        print(f"Elapsed Time: {end-start} seconds")