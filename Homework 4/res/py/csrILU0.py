def csrILU0(sA,iA,jA):
    sM = np.array(sA)
    n = len(iA) - 1
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
