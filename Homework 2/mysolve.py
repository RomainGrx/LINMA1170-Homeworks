#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date   : Monday, November 18th 2019
"""
# ======================= Import ==============================
# In[0]

import numpy as np
import time
SOLVER = None

# ======================= SOLVER ==============================

# ------------------------- QR ----------------------------
# In[1]

def QRsolve(A,b):
    """
    Args:
        A (numpy.array) : Coefficient matrix
        b (numpy.array) : Dependent variable values
    Returns:
        x (numpy.array) : The solution of the linear system 'Ax=b'
    """
    A = np.asarray(A)
    V, R = QR(A)
    Q = np.divide(V, np.outer(np.ones(len(A)), np.diagonal(R)))
    y = (Q.T)@b
    x = backward(R, y)
    return x

def QRfactorize(A):
    return QR(A)

def QR(A):
    """
    Args:
        A (numpy.array) : Matrix to be factorized
    Returns:
        V (numpy.array) : Contains wk vectors to construct Q matrix
        R (numpy.array) : triangular matrix
    """
    A = np.asarray(A)
    m,n = A.shape
    R = np.zeros_like(A, shape=(n,n))
    V = np.array(A)

    for i in range(n):
        R[i,i] = np.linalg.norm(V[:,i], axis=0)
        vr = np.divide(V[:,i], R[i,i])
        R[i,i+1:n] = vr @ V[:,i+1:n]
        V[:,i+1:n] -= np.einsum('i,j->ij', vr, R[i,i+1:n])
    return (V, R)


# ------------------------- LU ----------------------------
# In[2]

def LU(A):
    """
    Args:
        A (numpy.array)  : Matrix to be factorized as LU
    Returns:
        LU (numpy.array) : The lower and the upper matrix squeezed
        P (numpy.array)  : Swivel vector with the number of permutations in the last element
    """
    LU, P = pivot(A) # Retourne la matrice PA pivotee ainsi que le vecteur de pivotage
    m,n = LU.shape
    for k in range(m):
        LU[k+1:,k]    /= LU[k,k] # Division par le pivot
        LU[k+1:,k+1:] -= np.einsum('i,j->ij' ,LU[k+1:,k], LU[k,k+1:]) # Outer product
    return LU, P

def LUsolve(A,b):
    """
    Args:
        A (numpy.array) : Coefficient matrix
        b (numpy.array) : Dependent variable values
    Returns:
        x (numpy.array) : The solution of the linear system 'Ax=b'
    """
    PLU, P = LU(A)
    b  = b[P[:-1]]
    L = np.tril(PLU, -1)+np.eye(len(A))
    U = np.triu(PLU, 0)
    y  = forward(L,b)
    return backward(U, y)

def ILU0(A):
    """
    Args:
        A (numpy.array)  : Matrix to be factorized as incomplete LU
    Returns:
        LU (numpy.array) : The lower and the upper matrix squeezed
        P (numpy.array)  : Swivel vector with the number of permutations in the last element
    """
    tol = 1e-7
    ILU, P = pivot(A)
    m,n = ILU.shape
    for i in range(1,m):
        index = np.nonzero(ILU[i,:i]>tol)[0]
        for k in index:
            subindex = np.nonzero(ILU[i,k+1:n]>tol)[0] + (k+1)
            ILU[i,k] /= ILU[k,k]
            ILU[i,subindex] -= ILU[i,k]*ILU[k,subindex]
    return ILU, P

def ILUsolve(A,b):
    """
    Args:
        A (numpy.array) : Coefficient matrix
        b (numpy.array) : Dependent variable values
    Returns:
        x (numpy.array) : The solution of the linear system 'Ax=b'
    """
    PILU, P = ILU0(A)
    b  = b[P[:-1]]
    L = np.tril(PILU, -1)+np.eye(len(A))
    U = np.triu(PILU, 0)
    y  = forward(L,b)
    return backward(U, y)

def pivot(A):
    """
    Args:
        A (numpy.array) : Matrix to be pivoted
    Returns:
        A (numpy.array) : Permuted matrix such that all diagonal elements are the sub-maximum
        P (numpy.array) : Swivel vector with the number of permutations in the last element
    """
    A = np.array(A)
    m,n = A.shape
    P = np.arange(m+1)
    P[m] = 0
    for j in range(m):
        row = np.argmax(abs(A[j:m,j]))+j
        if j != row:
            P[m] += 1
            P[[j, row]] = P[[row, j]]
            A[[j,row]] = A[[row,j]]
    return A, P

def forward(L,b):
    """
    Args:
        L (numpy.array) : Lower coefficient matrix
        b (numpy.array) : Dependent variable values
    Returns:
        y (numpy.array) : The solution of the forward substitution 'Ly=b'
    """
    n = len(b)
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = np.divide(b[i] - np.einsum('j,j->', L[i,:i], y[:i]), L[i,i])
    return y

def backward(U, y):
    """
    Args:
        U (numpy.array) : Upper coefficient matrix
        y (numpy.array) : Dependent variable values
    Returns:
        x (numpy.array) : The solution of the backward substitution 'Ux=y'
    """
    n = len(y)
    x = np.zeros_like(y)
    for i in range(n-1, -1, -1):
      x[i] = np.divide((y[i] - np.einsum('i,i->', U[i, i:], x[i:])), U[i, i])
    return x

# ------------------------- Cholesky ----------------------------
# In[3]

def cholesky(A):
    """
    Args:
        A (numpy.array) : Symmmetric, positive definite matrix to be factorized as L
    Returns:
        L (numpy.array) : The lower matrix
    """
    n = len(A)
    L = np.zeros_like(A, shape=(n,n))

    for i in range(n):
        for k in range(i+1):
            tmp_sum = np.dot(L[i,:k], L[k,:k])
            if (i == k):
                L[i,k] = np.sqrt(A[i, i] - tmp_sum)
            else:
                L[i,k] = (1.0 / L[k,k] * (A[i,k] - tmp_sum))
    return L

def Choleskysolve(A,b):
    """
    Args:
        A (numpy.array) : Symmmetric, positive definite coefficient matrix
        b (numpy.array) : Dependent variable values
    Returns:
        x (numpy.array) : The solution of the linear system 'Ax=b'
    """
    A = np.array(A)
    L = cholesky(A)
    y = forward(L,b)
    x = backward(L.T,y)
    return x

# ------------------------- mysolve ----------------------------
# In[4]

def mysolve(A, b):
    global SOLVER
    if SOLVER == "LU":
        return True, LUsolve(A, b)
    elif SOLVER == "ILU":
        return True, ILUsolve(A, b)
    elif SOLVER == "QR":
        return True, QRsolve(A, b)
    elif SOLVER == "Cholesky":
        return True, Choleskysolve(A, b)
    elif SOLVER == "Numpy":
        return True, np.linalg.solve(A, b)
    else:
        return False, None

def setSolver(solver):
    global SOLVER
    SOLVER = solver
