def csrGMRES(sA,iA,jA,b,rtol, prec=False, x0=None, res_history=[], max_iterations=1e10):
    n = len(b)
    H = np.zeros_like(sA, shape=(n+1,n)) # Hessenberg matrix
    Q = np.zeros_like(sA, shape=(n,n))   # Espace de Krylov
    e1beta = np.zeros_like(sA, shape=(n+1,)) # (beta, 0, ..., 0)
    if x0 is not None: # [1.] Definir le premier residu
        r0 = np.array(b - csr_dot(sA, iA, jA, x0))
    else:
        x0 = np.zeros(n)
        r0 = np.array(b)
    if prec: # Preconditionnement
        sM, iM, jM = csrILU0(sA, iA, jA) # ILU factorisation
        r0 = csr_tridot(sM, iM, jM, r0)
    beta = np.linalg.norm(r0, ord=2) # [2.] Definir la norme du residu
    e1beta[0] = beta
    Q[:,0] = np.divide(r0, beta) # [3.] Premier V0 = r0 / beta
    for j in range(n):
        w = csr_dot(sA, iA, jA, Q[:,j]) # [4.] Definition de w = A@Vj
        if prec : w = csr_tridot(sM, iM, jM, w)
        for i in range(j+1):
            H[i,j] = np.dot(Q[:,i], w) # [5.] H[i,j] = Vi @ w
            w = w - np.dot(H[i,j],Q[:,i]) # [6.] w -= H[i,j] @ Vi
        H[j+1,j] = np.linalg.norm(w, ord=2) # [7.] H[j+1,j] = ||w||2
        if H[j+1, j] != 0 and j < n-1:
            Q[:, j+1] = np.divide(w, H[j+1, j]) # [8.]
        y = QRSolve(H, e1beta) # [9.] Resous aux moindres carrees
        x = x0 + np.dot(Q, y) # Solution plus precise (x_n)
        valres = b - csr_dot(sA, iA, jA, x) # Valeur du residu
        if prec: valres = csr_tridot(sM, iM, jM, valres)
        res_history.append(np.linalg.norm(valres, ord=2)) # Append la norme du nouveau residu
        if res_history[-1] < rtol or len(res_history)==max_iterations: # Condition d'arret atteinte
            return x, np.asarray(res_history)
    return csrGMRES(sA, iA, jA, b, rtol=rtol, prec=prec, x0=x, res_history=res_history, max_iterations=max_iterations)
