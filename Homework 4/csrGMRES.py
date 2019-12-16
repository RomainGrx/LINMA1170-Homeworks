#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : Wednesday, 11 December 2019
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

def mysolve(A,b, **kwargs):
    sA, iA, jA = CSRformat(A)
    return True, csrGMRES(sA, iA, jA, b, 10e-10)[0]


def csr_arnoldi(sA, iA, jA, v0, k:int, *M):
    n = len(v0)
    H = np.zeros((k+1,k))
    Q = np.zeros((n,k+1))
    q = v0/np.linalg.norm(v0)
    Q[:,0] = q
    for j in range(k):
        v = csr_dot(sA, iA, jA, Q[:,j])
        # v = csr_dot(sA, iA, jA, q)
        # print('v ', Q[:,j])
        for i in range(j+1):
            H[i,j] = np.dot(Q[:,i].conj(), v)
            v -= H[i,j]*Q[:,i]
        H[j+1,j] = np.linalg.norm(v)
        if H[j+1, j] != 0 and j!=n-1:
            q = v / H[j+1, j]
            Q[:, j+1] = q
        else:
            return Q, H
    return Q,  H


def csrGMRES(sA,iA,jA,b,rtol,prec=False,x0=None):
    n = len(b)
    res_history = []
    x = []
    v = np.zeros_like(sA, shape=(n,n))
    h = np.zeros((n+1, n))
    if x0 is not None:
        v0 = np.array(b - csr_dot(sA, iA, jA, x0))
    else:
        x0 = np.zeros(n)
        v0 = np.array(b)

    if not prec:
        sM, iM, jM = np.ones(n), np.arange(n+1), np.arange(n)
    else:
        sM, iM, jM = csrILU0(sA, iA, jA)
        v0 = csr_dot(sM, iM, jM, v0)


    beta = np.linalg.norm(v0, ord=2)
    e1 = np.zeros(n+1)
    e1[0] = 1
    H = np.zeros((n+1,n))
    Q = np.zeros((n,n))
    Q[:,0] = v0/np.linalg.norm(v0)
    for j in range(n):
        v = csr_dot(sA, iA, jA, Q[:,j])
        if prec :
            v = csr_dot(sM, iM, jM, v)
        for i in range(j+1):
            H[i,j] = np.dot(Q[:,i].conj(), v)
            v -= H[i,j]*Q[:,i]
        H[j+1,j] = np.linalg.norm(v)
        if H[j+1, j] != 0 and j!=n-1:
            q = v / H[j+1, j]
            Q[:, j+1] = q
        y = np.linalg.lstsq(H, beta*e1, rcond=None)[0]
        x.append(x0+np.dot(Q,y))
        res_history.append(np.linalg.norm(csr_dot(sA, iA, jA, x[-1])-b, ord=2))
        if res_history[-1] < rtol :
            print('Solution trouvée en ',len(res_history),' itérations')
            break



    return x[-1], res_history

def csr_dot(sA, iA, jA, p):
    n = len(iA) - 1
    assert n == len(p)
    dot = np.zeros(n)
    for i in range(n):
        dot[i] = np.dot(sA[iA[i]:iA[i+1]], p[jA[iA[i]:iA[i+1]]])
    return dot


def csrILU0(sA,iA,jA):
    sM = np.array(sA)
    tol = 1e-7
    n = len(iA) - 1

    # Make vectorial getter
    def get(i,j):
        return sM[iA[i]:iA[i+1]][jA[iA[i]:iA[i+1]]==j]

    for i in range(1,n):
        icol     = jA[iA[i]: iA[i+1]]
        i_line   = sM[iA[i]: iA[i+1]]
        k_ranger = icol[icol<i]
        for k in k_ranger:
            kcol   = jA[iA[k]:iA[k+1]]
            k_line = sM[iA[k]:iA[k+1]]
            kpivot = get(k,k)
            if len(kpivot) == 0:
                print('ATTENTION : Division par zero pour pivot (%d,%d)'%(k,k), file=sys.stderr)
            else:
                sM[iA[i]:iA[i+1]][icol==k] /= kpivot
            j_line = icol[icol>k]
            for j in j_line:
                Akj = get(k,j)
                Aij = get(i,j)
                Aik = get(i,k)
                if len(Aik) != 0 and len(Akj) != 0 and len(Aij) != 0:
                    sM[iA[i]:iA[i+1]][icol==j] -= Aik*Akj

    return sM, iA, jA


def CSRformat(A):
    """
    Args:
        A (numpy.array) : Coefficient matrix of size n to be factorized as CSR format
    Returns:
        sA (numpy.array) : Vector that contains all non zero elements of 'A'
        iA (numpy.array) : Vector of size n that contains the first non zero element in 'sA' for each row of 'A'
        jA (numpy.array) : Vector that contains the column index for each element contained in 'sA'
    """
    m, n  = A.shape
    (row, col)  = np.nonzero(A) # Liste toutes les valeures non-nulles
    iA = np.zeros(m+1, dtype=int) # Contiendra l'index de la colonne du premier élément non-nul d'une ligne
    jA = col # Simplement tous les indexs des éléments non-nuls
    bool = (A!=0) # Produit une matrice binaire avec 1 pour les valeurs non-nulles
    sumline = np.sum(bool, axis=1, dtype=int) # Somme le nombre d'éléments non-nulles par ligne

    iA[1:] = np.cumsum(sumline) # Renvoie le vecteur avec à chaque index la somme de tous les élements non-nuls précedents

    sA = np.zeros(len(row))
    sA[:] = A[row,col] # Renvoie un vecteur avec tous la valeur de tous les éléments non-nuls
    return sA, iA, jA

def invCSRformat(sA, iA, jA):
    """
    Args:
        sA (numpy.array) : Vector that contains all elements
        iA (numpy.array) : Vector of size n that contains the index column for the first non zero element for each row of the dense matrix
        jA (numpy.array) : Vector that contains the column index for each element contained in 'sA'
    Returns:
        A (numpy.array) : Coefficient matrix of size n to be densified
    """
    m = len(iA)-1
    A = np.zeros_like(sA, shape=(m,m))
    for i in range(m):
        firstIndex = iA[i]
        length     = iA[i+1] - firstIndex
        for j in range(length):
            curIndex = firstIndex+j
            A[i, jA[curIndex]] = sA[curIndex]
    return A


if __name__=='__main__':
    n = 10
    # A = np.random.rand(n,n)
    # A = np.array([[1,0,0],[1,2,0],[0,3,2]], dtype=float)
    # A = np.arange(9, dtype=float).reshape((3,3))+1

    # A = np.random.rand(n,n)
    # b = np.random.rand(n)

    # A = np.eye(n)
    # b = np.ones(n)

    A = np.tril(np.ones((n,n)))
    b = np.arange(n)+1

    # A = np.array([[1,0,0],[1,1,0],[1,1,1]])
    # b = np.arange(n)+1
    print(np.linalg.solve(A,b))
    # print(GMRes(A, b, np.zeros(len(b)))[-1])
    # Arnoldi = l.eigs(A)
    # ARN = arnoldi_iteration(A,b, 3)
    # print(ARN[1])
    # print(Arnoldi[0])

    sA, iA, jA = CSRformat(A)
    arn = csrGMRES(sA, iA, jA, b, 10e-12)
    print(arn[0])
    res = arn[1]
    plt.Figure()
    plt.plot(np.arange(len(res)), res)
    plt.show()




    # A = np.matrix('1 1; 3 -4')
    # b = np.array([3, 2])
    # x0 = np.array([0, 0])
    # print(GMRes(A,b,x0)[-1])

    # A = np.ones((n,n))
    # sA, iA, jA = CSRformat(A)
    # P = np.ones(n)
    # L = csr_dot(sA, iA, jA, P)
    # print(invCSRformat(*L))
    # print(L)
