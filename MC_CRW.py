import random as random

from _graphtools import *
from _vectools import *
from _counttools import *

import numpy as np
import math
import warnings
import argparse

#### Functions
def evolve(H, psi0, t):
    U = expm(-1j * H * t)  # time evolution operator U(t)
    psi_t = U @ psi0
    return psi_t


#### command line arguments

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(prog = 'MC Simulation of Continous Random Walks', description = 'Calculates Time Evolution on Glued Random Cycles Trees')

parser.add_argument('X', type = list(int), help = 'Growth Sequence')
parser.add_argument('t', type = int, help = 'Number of Time Steps')
parser.add_argument('tmax', type = float, help = 'Maximum Time')
parser.add_argument('f', type = int, help = 'Number of Flux Steps')
parser.add_argument('N', type=int, help="Number of Necklaces Sampled")

parser.add_argument('-s', type = int, help = 'Seed')
parser.add_argument('-outfile', help = 'Output file holding neccessary information for plotting and generation seeds')


args = parser.parse_args()


#### Runtime
if __name__ == '__main__':

    seed = args.s
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
    random.seed(seed)

    X = args.X

    ftree = fluxedTree(X)
    ftree.construct_fluxed()

    rand = generate_random_cycle_graph(ftree)
    rand.construct_adj()
    bare_hamiltonian = rand.adj
    bare_eigvals, bare_eigvecs = np.linalg.eigh(bare_hamiltonian)

    N = len(rand.node_map)
    psi_i = e_n(0, N)
    end = e_n(N - 1, N)

    t = np.linsapce(0,args.tmax, args.t)
    fluxes = np.linspace(2*np.pi/(args.f),2*np.pi, args.f-1)
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
