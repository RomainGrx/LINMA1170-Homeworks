import numpy as np

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

def LUsolve(A,b, **kwargs):
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
    x = backward(U, y)
    return x

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
