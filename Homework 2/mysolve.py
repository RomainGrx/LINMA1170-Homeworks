#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date   : Thursday, 07 November 2019
"""

# In[0]

import numpy as np
import scipy.linalg as l
SOLVER = None

# ======================= SOLVER ==============================

# ------------------------- QR ----------------------------
# In[1]

def QRsolve(A,b):
    return None

def QRfactorize(A):
    """
    Args:
        A (numpy.array) : Matrix to be factorized
    Returns:
        numpy.array : factorized matrix A
    """
    return None

# ------------------------- LU ----------------------------
# In[2]

def forward(L,b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def backward(U, y):
    x = np.zeros(len(y))
    for i in range(len(x)-1, -1, -1):
      x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]
    return x

def LUsolve(A,b,P):
    A  = np.dot(P, A)
    b  = np.dot(P, b)
    MY = LU(A)
    y  = forward(MY,b)
    return backward(MY, y)


def pivot(A):
    m = len(A)
    P = np.eye(m)
    for j in range(m):
        row = np.argmax(abs(A[j:m,j]))+j
        if j != row:
            P[j], P[row] = P[row], P[j]

    return P

def LU(PA):
    """
    Args:
        A (numpy.array) : Matrix to be factorized as LU
    Returns:
        LU (numpy.array) : The lower and the upper matrix squeezed
    """
    LU = PA.copy()
    n = LU.shape[0]

    for k in range(n):
        LU[k+1:,k] = LU[k+1:,k] / LU[k,k]
        LU[k+1:,k+1:] = LU[k+1:,k+1:] - np.outer( LU[k+1:,k], LU[k,k+1:] )

    return LU

def LU_OLD(A):
    """
    Args:
        A (numpy.array) : Matrix to be factorized as LU
    Returns:
        P (numpy.array) : The pivot matrix
        L (numpy.array) : The lower factorized matrix
        U (numpy.array) : The upper factorized matrix
    """
    n = len(A)

    L = np.zeros((n,n))
    U = np.zeros((n,n))

    P = pivot(A)
    PA = np.dot(P, A)

    for j in range(n):
        L[j, j] = 1.0

        for i in range(j+1):
            s1 = sum(U[k, j] * L[i, k] for k in range(i))
            U[i, j] = PA[i, j] - s1

        for i in range(j, n):
            s2 = sum(U[k, j] * L[i, k] for k in range(j))
            L[i, j] = (PA[i, j] - s2) / U[j, j]

    return (P, L, U)

def ILU0(A):
    return None

# ======================= PLOTER ==============================
# In[3]

def setSolver(solver):
    global SOLVER
    SOLVER = solver

def mysolve(A, b):
    if SOLVER == "LU":
        return True, LUsolve(A, b, 0)
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
    return svd(A)[1]

def get_min_max_singular_values(A):
    S = svd(A)[1]
    return [min(S), max(S)]

# ----------------------------------------------------------


# A = np.array([[1, 1, 1 ],
#               [2, 3, -1],
#               [1, -1, 1]])

# A = np.array([[2, -1, 0, 0, 0],
#      [-1, 2, -1, 0, 0],
#      [0, -1, 2, -1, 0],
#      [0, 0, -1, 2, -1],
#      [0, 0, 0, -1, 2]])

A = np.array([[1,1,-1],[1,-2,3],[2,3,1]])

b = np.array([6, 5, -1])

# P, L, U = LU(A)

# print(L)
# print(L_inv(L))

B = np.array([[1,0,0],[1.5, 1, 0],[3, 14, 1]])

MYL = np.array([[1, 0, 0],
                [1, 1, 0],
                [1, 1, 1]])
MYB = np.array([6, 3, 1])

MYU = np.array([[1, 1, 1],
                [0, 1, 1],
                [0, 0, 1]])

B = np.dot(pivot(A), A)
# print(LU_vectorized(B))
print(LUsolve(MYU, MYB, pivot(MYU)))
