for k in range(n):
    k_ranger = np.arange(k+1, min(k+left, n)) # Range de la colonne sous le pivot max
    ind = get(k_ranger, [k]*len(k_ranger)) # Les indices dans sLU de cette colonne
    sLU[ind] /= sLU[get(k,k)] # Division par le pivot
    max = min(k+left, k+right, n) # L'indice le plus petit contenant un elem non nul
    outer_ranger = np.arange(k+1,max) # Range du futur outer product
    len_outer = len(outer_ranger)
    outer = np.outer(sLU[get(outer_ranger, [k]*len_outer)], sLU[get([k]*len_outer, outer_ranger)]) # Outer product du carre de longueur 'len_outer' centre sur la diagonale
    sLU[get(np.repeat(outer_ranger, len_outer), np.tile(outer_ranger, len_outer))] -= outer.flatten() # Reduction du carre d'elements concerne
