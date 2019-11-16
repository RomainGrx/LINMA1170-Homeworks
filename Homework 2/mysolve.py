#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date   : Thursday, 07 November 2019
"""

# In[0]

import numpy as np
import scipy.linalg as l
import QR
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
    V, R = QRfactorize(A)
    Q = np.divide(V, np.outer(np.ones(len(A)), np.diagonal(R)))
    y = (Q.T)@b
    x = backward(R, y)
    return x

def QRfactorize(A):
    """
    Args:
        A (numpy.array) : Matrix to be factorized
    Returns:
        V (numpy.array) : Contains wk vectors to construct Q matrix
        R (numpy.array) : triangular matrix
    """
    m,n = A.shape
    R = np.zeros_like(A, shape=(n,n))
    V = np.array(A)

    # np.fill_diagonal(R, np.linalg.norm(A, axis=0))
    # Q = np.divide(A, R.diagonal())

    for i in range(n):
        R[i,i] = np.linalg.norm(V[:,i], axis=0)
        vr = np.divide(V[:,i], R[i,i])
        R[i,i+1:n] = vr @ V[:,i+1:n]
        V[:,i+1:n] -= np.einsum('i,j->ij', vr, R[i,i+1:n])
    return (V, R)


# ------------------------- LU ----------------------------
# In[2]

def forward(L,b):
    n = len(b)
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = b[i] - np.einsum('j,j->', L[i,:i], y[:i])
    return y

def backward(U, y):
    x = np.zeros_like(y)
    for i in range(len(x)-1, -1, -1):
      x[i] = np.divide((y[i] - np.einsum('i,i->', U[i, i:], x[i:])), U[i, i])
    return x

def LUsolve(A,b):
    A = np.asarray(A)
    A, P = pivot(A)
    b  = np.dot(P, b)
    MY = LU(A)
    U = np.triu(MY, 0)
    y  = forward(MY,b)
    return backward(U, y)

def ILUsolve(A,b):
    A = np.asarray(A)
    A, P = pivot(A)
    b  = np.dot(P, b)
    MY = ILU0(A)
    U = np.triu(MY, 0)
    y  = forward(MY,b)
    return backward(U, y)


def pivot(A):
    """
    Args:
        A (numpy.array) : Matrix to be pivoted
    Returns:
        A (numpy.array) : Matrix pivoted such that all diagonal elements are the sub-maximum
    """
    m = len(A)
    copy = A.copy()
    P = np.eye(m)
    for j in range(m):
        row = np.argmax(abs(copy[j:m,j]))+j
        if j != row:
            P[[j, row]] = P[[row, j]]
            copy[[j,row]] = copy[[row,j]]

    return copy, P

def LU(A):
    """
    Args:
        A (numpy.array) : Matrix to be factorized as LU
    Returns:
        LU (numpy.array) : The lower and the upper matrix squeezed
    """
    LU = np.array(A)
    n = LU.shape[0]
    for k in range(n):
        LU[k+1:,k] = np.divide(LU[k+1:,k], LU[k,k])
        LU[k+1:,k+1:] -= np.einsum('i,j->ij' ,LU[k+1:,k], LU[k,k+1:] )
    return LU

def ILU0(A):
    A = np.array(A)
    n = len(A)
    for i in range(1,n):
        index = np.nonzero(A[i,:i])[0]
        for k in index:
            subindex = np.nonzero(A[i,k+1:n])[0] + (k+1)
            A[i,k] /= A[k,k]
            A[i,subindex] -= A[i,k]*A[k,subindex]
    return A

# ======================= PLOTER ==============================
# In[3]

def setSolver(solver):
    global SOLVER
    SOLVER = solver

def mysolve(A, b):
    global SOLVER
    if SOLVER == "LU":
        return True, LUsolve(A, b)
    elif SOLVER == "QR":
        return True, QRsolve(A, b)
    else:
        return False, None

# ======================= OLD ==============================
# ----------------------------------------------------------
# ------------------ Test sur la matrice A -----------------
# ----------------------------------------------------------

def all_test(A, name):
    print("\n\n--------------------------------------------------------")
    print("                     %s                                      "%(name))
    print("--------------------------------------------------------\n")
    if(is_complex(A)):
        print("> Complex")
    else:
        print("> Real")
    if(is_symetric(A)):
        print("> Symetric")
    else:
        print("> Not Symetric")
    print("> %s" %(definite_values[definite(A)]))
    if(is_hermitian(A)):
        print("> Hermitian")
    else:
        print("> Not hermitian")
    if(is_invertible(A)):
        print("> Invertible")
    else:
        print("> Not invertible")
    if(is_unitary(A)):
        print("> Unitary")
    else:
        print("> Not unitary")

    print("\n--------------------------------------------------------\n\n")


definite_values = {-2 : "Définie négative", -1 : "Semi-définie négative", 0 : "Non-définie", 1 : "Semi-définie positive", 2: "Définie positive"}

# @return : {-2 : negative, -1 : semi-négative, 0 : non-définie, 1 : semi-positive, 2: positive}
def definite(A):
    ret = 0
    if(is_symetric(A)):
        eigenvalues = l.eig(A)[0]
        if (eigenvalues >= 0).all():
            if len(np.nonzero(eigenvalues)[0]) == len(eigenvalues):
                ret = 2
            else:
                ret = 1
        elif (eigenvalues <= 0).all():
            if len(np.nonzero(eigenvalues)[0]) == len(eigenvalues):
                ret = -2
            else:
                ret = -1
    return ret

def is_symetric(A):
    # if(is_complex(A)) : return False
    return (A==A.T).all()

def is_complex(A):
    return np.iscomplex(A).any()

def is_invertible(A):
    return (l.det(A)!=0)

def is_hermitian(A):
    return (A==np.matrix.getH(A)).all()

def is_unitary(A):
    return ((A@np.matrix.getH(A))==np.eye(len(A))).all()

# ----------------------------------------------------------
# ---------------- Affichage de la matrice -----------------
# ----------------------------------------------------------

def plot_matrix(A, name):
    plt.title(name)
    plt.imshow(np.clip(A*A, 0, 1), cmap='Greys')
    plt.savefig('matrix/%s' %(name))

def save_matrix(A, filename):
    matrix = np.matrix(A[:20,:20])
    with open("matrix/%s"%filename,'wb') as fd:
        for line in matrix:
            np.savetxt(fd, line, fmt='%.2f')
        fd.close()

# ----------------------------------------------------------
# -------------- Information sur le système  ---------------
# ----------------------------------------------------------

def get_singular_values(A):
    return np.linalg.svd(A)[1]

def get_min_max_singular_values(A):
    S = np.linalg.svd(A)[1]
    return [min(S), max(S)]

# ----------------------------------------------------------
n = 3
A = np.random.rand(n+1,n)
b = np.random.rand(n)
print(A,'\n')
Q,R = np.linalg.qr(A)
print(Q@R,'\n')
V,R = QRfactorize(A)
Q = np.divide(V, np.outer(np.ones(len(A)), np.diagonal(R)))
print(Q@R,'\n')
