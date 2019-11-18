import numpy as np

def LU(A):
    LU, P = pivot(A) # Retourne la matrice PA pivotee ainsi que le vecteur de pivotage
    m,n = LU.shape
    for k in range(m):
        LU[k+1:,k]    /= LU[k,k] # Division par le pivot
        LU[k+1:,k+1:] -= np.einsum('i,j->ij' ,LU[k+1:,k], LU[k,k+1:]) # Outer product
    return LU, P


def cholesky(A):
    n = len(A)
    L = np.zeros_like(A, shape=(n,n))
    for i in range(n):
        for k in range(i+1):
            tmp_sum = np.einsum('i,j->', L[i,:k], L[k,:k])
            if (i == k):
                L[i][k] = np.sqrt(A[i][i] - tmp_sum)
            else:
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L
