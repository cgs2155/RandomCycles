#!/usr/bin/env python3
import random
import argparse
import numpy as np
from tqdm import tqdm

# Import the gen_necklace function. Adjust the import path as needed.
from _counttools import gen_necklace

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce necklaces from a given seed."
    )
    parser.add_argument('seed', type=int, help="Seed value to reproduce the necklaces")
    parser.add_argument('N', type=int, help="Number of necklaces to generate")
    # Assuming your necklace generation requires a 'node count' or similar parameter,
    # adjust the default value as required by your application.
    parser.add_argument('--nodes', type=int, default=5,
                        help="Number of nodes (if applicable) for gen_necklace")

    args = parser.parse_args()
    
    # Set the seed for reproducibility
    #random.seed(args.seed)
    necklace_rng = random.Random(args.seed)

    necklaces = set()
    with tqdm(total=args.N,desc="Recovering Necklaces") as pbar:
        while len(necklaces) < args.N:
            curr = len(necklaces)
            necklaces.add(gen_necklace(args.nodes,necklace_rng))
            if curr < len(necklaces):
                pbar.update(1)
        
    # Optionally, output in a reproducible order (e.g., sorted order)
    outfile=f"{args.seed}_{args.nodes}_{args.N}.npz"
    np.savez(outfile, necklaces=necklaces)
    print(f"Results saved to {outfile}")

if __name__ == "__main__":
    main()
