#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : Wednesday, 23 December 2019
"""

import numpy as np
from numpy import linalg as lin
import sys
import matplotlib.pyplot as plt
import ndt

SOLVER = 'csrGMRES'


def mysolve(A,b, **kwargs):
    if SOLVER == 'csrGMRES':
        sA, iA, jA = CSRformat(np.array(A))
        return True, csrGMRES(sA, iA, jA, np.array(b), prec=True, rtol=1e-10)[0]
    else:
        return False, None


def csrGMRES(sA,iA,jA,b,rtol,prec=False, x0=None, res_history=[], max_iterations=1000):
    """
    Args:
        sA (numpy.array) : Vector that contains all elements of the coefficient matrix
        iA (numpy.array) : Vector of size n+1 that contains the index column for the first non zero element for each row of the dense matrix
        jA (numpy.array) : Vector that contains the column index for each element contained in 'sA'
        b  (numpy.array) : Dependent variable values
        rtol (float)     : The residual tolerance
        prec (boolean)   : Preconditionning with ILU
    Returns:
        x  (numpy.array) : The solution of the system
        res_history (numpy.array) : An array with residual at each iteration
    """
    if(len(res_history)!=0):print('COUCOU')
    n = len(b)
    x = []
    H = np.zeros_like(sA, shape=(n+1,n)) # Hessenberg matrix
    Q = np.zeros_like(sA, shape=(n,n))
    e1beta = np.zeros_like(sA, shape=(n+1,)) # [1, 0, ..., 0]
    sA, iA, jA = np.asarray(sA), np.asarray(iA), np.asarray(jA)
    # [1.] Définir le premier résidu
    if x0 is not None:
        r0 = np.array(b - csr_dot(sA, iA, jA, x0))
    else:
        x0 = np.zeros(n)
        r0 = np.array(b)

    # Preconditionnement
    if prec:
        sM, iM, jM = csrILU0(sA, iA, jA)
        r0 = csr_tridot(sM, iM, jM, r0)

    # [2.] Définir la norme du résidu
    beta = np.linalg.norm(r0, ord=2)
    e1beta[0] = beta

    # [3.] Premier V0 = r0 / beta
    Q[:,0] = np.divide(r0, beta)

    for j in range(n):
        # [4.] Définition de w = A@Vj
        w = csr_dot(sA, iA, jA, Q[:,j])

        if prec :
            w = csr_tridot(sM, iM, jM, w)

        for i in range(j+1):
            # [5.] H[i,j] = Vi @ w
            H[i,j] = np.dot(Q[:,i], w)

            # [6.] w -= H[i,j] @ Vi
            w = w - np.dot(H[i,j],Q[:,i])

        # [7.] H[j+1,j] = ||w||2
        H[j+1,j] = np.linalg.norm(w, ord=2)

        # [8.]
        if H[j+1, j] != 0 and j < n-1:
            Q[:, j+1] = np.divide(w, H[j+1, j])

        # [9.]
        y = QRSolve(H, e1beta) # Resous aux moindres carrees

        x.append(x0 + np.dot(Q, y)) # Solution plus précise

        valres = b - csr_dot(sA, iA, jA, x[-1]) # Valeur du résidu
        if prec:
            valres = csr_tridot(sM, iM, jM, valres)

        res_history.append(np.linalg.norm(valres, ord=2)) # Append la norme du nouveau residu
        if res_history[-1] < rtol or len(res_history)==max_iterations: # Condition d'arret atteinte
            break
    if res_history[-1] >= rtol and len(res_history)<max_iterations: # Condition d'arret non satisfaite
        return csrGMRES(sA, iA, jA, b, rtol= rtol, prec=prec, x0=x[-1], res_history=res_history)
    return x[-1], np.asarray(res_history)



def csrILU0(sA,iA,jA):
    """
    Args:
        sA (numpy.array) : Vector that contains all elements of the coefficient matrix
        iA (numpy.array) : Vector of size n+1 that contains the index column for the first non zero element for each row of the dense matrix
        jA (numpy.array) : Vector that contains the column index for each element contained in 'sA'
    Returns:
        sM (numpy.array) : Vector that contains all elements of the incomplete LU factorisation squeezed
        iA (numpy.array) : Vector of size n+1 that contains the index column for the first non zero element for each row of sM
        jA (numpy.array) : Vector that contains the column index for each element contained in 'sM'
    """
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
            Aik = get(i,k)
            for j in j_line:
                Akj = get(k,j)
                Aij = get(i,j)
                if len(Aik) != 0 and len(Akj) != 0 and len(Aij) != 0:
                    sM[iA[i]:iA[i+1]][icol==j] -= Aik*Akj

    return sM, iA, jA

def QRSolve(A,b):
    """
    Args:
        A (numpy.array) : Coefficient matrix
        b (numpy.array) : Dependent variable values
    Returns:
        x (numpy.array) : The solution of the linear system 'Ax=b'
    """
    m,n = A.shape
    Q,R = lin.qr(A)
    y = np.dot(np.conj(Q).T,b)
    x = np.zeros_like(A, shape=(n,))

    for i in range(n-1,-1,-1):
        if(R[i,i]!=0):
            x[i] = np.divide((y[i] - np.dot(R[i,i:n],x[i:n])), R[i,i])
    return x

def csr_dot(sA, iA, jA, b):
    """
    Args:
        sA (numpy.array) : Vector that contains all elements of the coefficient matrix
        iA (numpy.array) : Vector of size n+1 that contains the index column for the first non zero element for each row of the dense matrix
        jA (numpy.array) : Vector that contains the column index for each element contained in 'sA'
        b  (numpy.array) : Vector
    Returns:
        dot (numpy.array): The dot product (A@b)
    """
    n = len(iA) - 1
    assert n == len(b)
    dot = np.zeros_like(sA, shape=(n,))
    for i in range(n):
        dot[i] = np.dot(sA[iA[i]:iA[i+1]], (b[jA[iA[i]:iA[i+1]]]))
    return dot

def csr_tridot(sA, iA, jA, b):
    """
    Args:
        sA (numpy.array) : Vector that contains all elements of the coefficient matrix
        iA (numpy.array) : Vector of size n+1 that contains the index column for the first non zero element for each row of the dense matrix
        jA (numpy.array) : Vector that contains the column index for each element contained in 'sA'
        b  (numpy.array) : Vector
    Returns:
        dot (numpy.array): The dot product (LU@b)
    """
    n = len(b)
    assert n == len(b)
    dot = np.zeros_like(sA, shape=(n,))

    # Forward (Ly=b)
    for i in range(n):
        allcols  = jA[iA[i]:iA[i+1]]
        allelems = sA[iA[i]:iA[i+1]]
        fitbool  = np.argwhere(allcols<i).reshape((-1,))
        fitelems = allelems[fitbool]
        fitcols  = allcols[fitbool]
        dot[i]    = b[i] - np.dot(fitelems,dot[fitcols])

    # Backward (Ux=y)
    for i in range(n-1, -1, -1):
        allcols  = jA[iA[i]:iA[i+1]]
        allelems = sA[iA[i]:iA[i+1]]
        pivot    = allelems[np.argwhere(allcols==i).squeeze()]
        fitbool  = np.argwhere(allcols>i).reshape((-1,))
        fitelems = allelems[fitbool]
        fitcols  = allcols[fitbool]
        dot[i]    = np.divide(dot[i] - np.dot(fitelems,dot[fitcols]), pivot)

    return dot

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

    sA = np.zeros_like(A, shape=(len(row),))
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

def csr_arnoldi(sA, iA, jA, v0, k:int, *M):
    n = len(v0)
    H = np.zeros((k+1,k))
    Q = np.zeros((n,k+1))
    q = v0/np.linalg.norm(v0)
    Q[:,0] = q
    for j in range(k):
        v = csr_dot(sA, iA, jA, Q[:,j])
        for i in range(j+1):
            H[i,j] = np.dot(Q[:,i], v)
            v -= H[i,j]*Q[:,i]
        H[j+1,j] = np.linalg.norm(v)
        if H[j+1, j] != 0 and j!=n-1:
            q = v / H[j+1, j]
            Q[:, j+1] = q
        else:
            return Q, H
    return Q,  H

if __name__=='__main__':
    ndt.main(show=True, solver=True)
