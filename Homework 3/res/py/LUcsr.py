def CSRformat(A):
    m, n  = A.shape
    iA = np.zeros(m+1, dtype=int) # Contiendra l'index de la colonne du premier element non-nul d'une ligne
    (row, col)  = np.nonzero(A) # Liste toutes les valeures non-nulles
    jA = col # Simplement tous les indexs des colonnes des elements non-nuls
    ax, count = np.unique(row, return_counts=True) # Somme le nombre d'elements non-nulles par ligne
    iA[ax+1] = count
    iA = np.cumsum(iA) # Renvoie le vecteur de la somme cumulative par ligne
    sA = A[row,col] # Renvoie un vecteur avec toutes les valeurs des elements non-nuls
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

# ======================= RMCK ==============================
# In[2]

def RCMK(iA, jA):
    """
    Args:
        iA (numpy.array) : Vector of size n that contains the index column for the first element for each row
        jA (numpy.array) : Vector that contains the column index for each element
    Returns:
        r (numpy.array) : Swivel vector
    """
    def pop(A):
        elem = A[0]
        A = np.delete(np.roll(A, -1), -1)
        return A, elem

    def push(A, elem):
        return np.append(A, elem)


    n = len(iA)
    degree = np.diff(iA)
    degree2 = np.copy(degree)
    in_R = np.zeros(n-1, dtype=bool)
    still = np.ones(n-1,dtype=bool)
    sort = np.argsort(degree)

    R = np.array([], dtype=int)
    Q = np.array([],dtype=int)

    while len(R)<n-1:
        preC = np.argmin(degree[still])
        C = 2*preC - sum(still[:preC])
        for k in range(len(Q)):
            Q, C = pop(Q)
            if C not in R:
                break
        R = push(R,C)

        q = np.array([],dtype=int)
        for i in range(degree[C]):
            next = jA[i+sum(degree2[:C])]
            if next not in R:
                q = push(q,next)

        q = q[np.argsort(degree[q])]
        Q = push(Q, q).astype(int)
        still[C] = False
    return R[::-1]



# ======================= LUcsr ==============================
# In[3]

def LUcsr(sA, iA, jA, left=None, *band):
    n = len(iA) - 1
    if(band==()):
        left, right, width, maxsize = bandwidth(iA, jA)
    else:
        left, right, width, maxsize = band[0]
    sLU = np.zeros_like(sA, shape=(maxsize))
    iLU = np.zeros_like(iA)
    jLU = np.zeros_like(jA, shape=(maxsize))

    def get(i,j):
        if type(i) is int:
            i = [i]
            j = [j]
        i = np.array(i)
        j = np.array(j)
        length = len(i)
        z = np.zeros(length)
        maxright = i + right
        maxleft  = i - left
        clipper = np.logical_and(j>maxleft, j<maxright)
        ind = i*(width-1) + j
        upper_bool = np.maximum(z, maxright-n)
        upper_area = 0.5*upper_bool*(upper_bool-1)
        lower_bool = np.maximum(z,left-2-i)
        lower_area = 0.5*((left-2)*(left-1) - (lower_bool)*(lower_bool+1))
        return ((ind - upper_area - lower_area)*clipper).astype(int).squeeze()

    # Mapping des éléments contenus dans 'sA' dans 'sLU'
    parser  = 0
    for i in range(n):
        ranger   = np.arange(iA[i], iA[i+1])
        ind      = get([i]*len(ranger), jA[ranger])
        sLU[ind] = sA[ranger]
        iLU[i]   = parser
        max_left  = min(left, i+1)
        max_right = min(right, n-i)
        first = i - max_left  + 1
        last  = i + max_right - 1
        length= last - first + 1
        jLU[parser:parser+length] = np.arange(first, last+1)
        parser += length
    iLU[-1] = parser

    # LU factorisation
    for k in range(n):
        k_ranger = np.arange(k+1, min(k+left, n)) # Range de la colonne sous le pivot max
        ind = get(k_ranger, [k]*len(k_ranger)) # Les indices dans sLU de cette colonne
        sLU[ind] /= sLU[get(k,k)] # Division par le pivot
        max = min(k+left, k+right, n) # L'élément maximum le moins loin sur le rang et la colonne k
        outer_ranger = np.arange(k+1,max) # Range du futur outer product
        len_outer = len(outer_ranger)
        outer = np.outer(sLU[get(outer_ranger, [k]*len_outer)], sLU[get([k]*len_outer, outer_ranger)]) # Outer product du carré de longueur 'len_outer' centré sur la diagonale
        sLU[get(np.repeat(outer_ranger, len_outer), np.tile(outer_ranger, len_outer))] -= outer.flatten() # Réduction du carré d'éléments concerné

    return sLU, iLU, jLU

def LUcsrsolve(A, b, **kwargs):
    """
    Args:
        A (numpy.array) : Coefficient matrix
        b (numpy.array) : Dependent variable values
    Returns:
        x (numpy.array) : The solution of the linear system 'Ax=b' done with LU factorization and sparse matrix
    """
    RMCK = False
    for key, value in kwargs.items():
        if key == 'RMCK':
            RMCK = value

    L = np.array(A)
    m, n = A.shape
    sA, iA, jA = CSRformat(L)

    if RMCK:
        r = RCMK(iA, jA)
        L = (L[:,r])[r,:]
        b = b[r]
        sA, iA, jA = CSRformat(L)
    band = left, right, width, maxsize = bandwidth(iA, jA)
    if width <= 0.75*n:
        sLU, iLU, jLU = LUcsr(sA, iA, jA, band)
        def get(i,j):
            if type(i) is int:
                i = [i]
                j = [j]
            i = np.array(i)
            j = np.array(j)
            length = len(i)
            z = np.zeros(length)
            maxright = i + right
            maxleft  = i - left
            clipper = np.logical_and(j>maxleft, j<maxright) # Inverse de j<=maxleft V j>=maxright pour mettre à True(1) les valeurs dans la bande
            ind = i*(width-1) + j
            upper_bool = np.maximum(z, maxright-n)
            upper_area = 0.5*upper_bool*(upper_bool-1)
            lower_bool = np.maximum(z,left-2-i)
            lower_area = 0.5*((left-2)*(left-1) - (lower_bool)*(lower_bool+1))
            return ((ind - upper_area - lower_area)*clipper).astype(int).squeeze()

        x = np.zeros_like(sLU, shape=(n,))
        y = np.zeros_like(sLU, shape=(n,))

        # Forward (Ly=b)
        for i in range(n):
            max_left  = max(0, i-left+1)
            ranger = np.arange(max_left, i)
            ind = get([i]*len(ranger), ranger)
            y[i] = b[i] - np.dot(sLU[ind], y[ranger])

        # Backward (Ux=y)
        for i in range(n-1, -1, -1):
            max_right  = min(i+right, n)
            ranger = np.arange(i+1,max_right)
            ind = get([i]*len(ranger), ranger)
            x[i] = np.divide(y[i] - np.dot(sLU[ind], x[ranger]), sLU[get(i,i)])
        if RMCK : x = x[np.argsort(r)]

    else:
        x = LUsolve(A,b)
    return x

def bandwidth(iA, jA):
    """
    Args:
        iA (numpy.array) : Vector of size n that contains the index column for the first element for each row
        jA (numpy.array) : Vector that contains the column index for each element
    Returns:
        left     (int) : The left  bound compared to the diagonal element
        right    (int) : The right bound compared to the diagonal element
        width    (int) : The band width
        bandsize (int) : The number of elements that fill in the band matrix
    """
    n  = len(iA) - 1
    ii = np.arange(n)
    first_ind  = jA[iA[:-1]]
    last_ind   = jA[iA[1:]-1]
    left  = max(max(ii - first_ind) +1, 1)
    right = max(max(last_ind - ii)  +1, 1)

    upper_zeros = n - right
    lower_zeros = n - left
    upper_area  = int(upper_zeros*(upper_zeros+1)) # The number of all upper zeros
    lower_area  = int(lower_zeros*(lower_zeros+1)) # The number of all lower zeros
    zeros_area  = (upper_area + lower_area)/2      # All matrix zeros outside the band matrix
    bandsize    = int(n*n - zeros_area)            # The number of elements in the band matrix
    width       = left + right - 1                 # The band width
    return left, right, width, bandsize
