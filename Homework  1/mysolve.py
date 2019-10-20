
# The function mysolve(A, b) is invoked by ndt.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ndt.py

import numpy as np
import numpy.linalg as l
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, gmres
from scipy.linalg import qr, lu, svd

SolverType = 'scipy'


def mysolve(A, b, SolverType = SolverType):
    if SolverType == 'scipy':
        return True, spsolve(A, b)

    elif SolverType == 'QR':
        Q, R = qr(A)
        return True, spsolve(R, (Q.T)@b)

    elif SolverType == 'LU':
        P, L, U = lu(A)
        Y = spsolve(L, b)
        return True, spsolve(U, Y)

    elif SolverType == 'GMRES':
        return True, gmres(A, b)[0]
    else:
        return False, 0


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

