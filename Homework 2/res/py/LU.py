def LU(A):
    """
    Args:
        A (numpy.array) : Matrix to be factorized as LU
    Returns:
        LU (numpy.array) : The lower and the upper matrix squeezed
    """
    LU = A
    n = LU.shape[0]
    for k in range(n):
        LU[k+1:,k] = np.divide(LU[k+1:,k], LU[k,k])
        LU[k+1:,k+1:] -= np.einsum('i,j->ij' ,LU[k+1:,k], LU[k,k+1:] )
    return LU
