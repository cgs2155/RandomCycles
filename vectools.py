import numpy as np

def e_n(n, d):
    # Creates a d-dimensional vector with 1 at the n-th position and 0 elsewhere
    vec = np.zeros(d)
    vec[n] = 1
    return vec