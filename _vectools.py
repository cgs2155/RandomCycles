import numpy as np
from scipy.linalg import expm

H_gate = 1/np.sqrt(2) * np.array([[ 1, 1],
                                  [ 1,-1]])

def e_n(n, d):
    # Creates a d-dimensional vector with 1 at the n-th position and 0 elsewhere
    vec = np.zeros(d)
    vec[n] = 1
    return vec

def direct_sum(A, B):
  """
  Computes the direct sum of two matrices. (Thank you Google)

  Args:
    A: The first matrix (NumPy array).
    B: The second matrix (NumPy array).

  Returns:
    The direct sum of A and B as a NumPy array.
  """
  m, n = A.shape
  p, q = B.shape
  result = np.zeros((m + p, n + q))
  result[:m, :n] = A
  result[m:, n:] = B
  return result


""" def C_n(X: list[int]):
    C_n_list = [np.array([[0]])]  # maybe H_0 is a 1x1 zero matrix?

    for n in range(1, len(X)+1):
        d = 2*n+1  # dimension of the matrix
        F_num = np.zeros((d, d))

        for m in range(d-1):
            prefactor= np.sqrt(X[m])
            outer_product = np.outer(e_n(m, d), e_n(m+1, d)) + np.outer(e_n(m+1, d), e_n(m, d))
            F_num += prefactor * outer_product

        F_n_list.append(F_num)

    return F_n_list"""

def temp_C_n(n):
    d = 2*n+2  # dimension of the matrix
    C_n = np.zeros((d, d))
    for i in range(0,d-1):
        C_n[i+1,i] = C_n[i,i+1] = np.sqrt(2)
    C_n[n+1,n] = C_n[n,n+1] = 2
    return C_n

def evolve(H, psi0, t):
    U = expm(-1j * H * t)  # time evolution operator U(t)
    psi_t = U @ psi0
    return psi_t

def limit(eigvecs, a, b):
    """The limiting distribution of the transition between states a and b for a system with given eigenvectors"""
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

