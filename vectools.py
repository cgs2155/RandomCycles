import numpy as np
from scipy.linalg import expm

H_gate = 1/np.sqrt(2) * np.array([[ 1, 1],
                                  [ 1,-1]])

def e_n(n, d):
    # Creates a d-dimensional vector with 1 at the n-th position and 0 elsewhere
    vec = np.zeros(d)
    vec[n] = 1
    return vec

def evolve(H, psi0, t):
    U = expm(-1j * H * t)  # time evolution operator U(t)
    psi_t = U @ psi0
    return psi_t
