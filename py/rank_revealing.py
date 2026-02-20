import numpy as np
from typing import Tuple, List

# QR with column pivoting
def PivotedQR(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Implements the QR Decomposition with Column Pivoting algorithm.
    Args:
        X: Input matrix
    Returns:
        Tuple containing (Q, R, P, rank)
    """
    # Array initialization
    Xc = np.copy(X)
    n, p = Xc.shape 
    #t = min(n,p) 
    R = np.zeros([p,p]) # Upper triangular R
    Q = np.zeros([n,p]) # Orthogonal Q
    P = np.arange(p)    # Permutation P
    
    # v_j = ||X[:,j]||^2, j=1,...,n
    v = np.zeros(p)
    for j in range(p):
        v_j = Xc[:,j].T @ Xc[:,j]
        v[j] = v_j
    pk = np.argmax(v)   # Determine an index p1 such that v_p1 is maximal
    maxV = v[pk]
    # Gram-Schmidt process
    rank = 0
    for k in range(p):
        #if k == t:
        #    break
        
        # SWAP X, v, P, R
        Xc[:, [pk, k]] = Xc[:, [k, pk]]
        v[[pk, k]] = v[[k, pk]]
        P[[pk, k]] = P[[k, pk]]
        if k > 0:
            R[0:k,[pk,k]] = R[0:k,[k,pk]]        
        
        # Orthogonalization and R update
        Q[:,k] = Xc[:,k] - Q[:,0:k] @ R[0:k, k]
        R[k,k] = np.sqrt(Q[:,k].T @ Q[:,k])
        Q[:,k] = Q[:,k] / R[k,k]
        # Re-orthogonalization is needed? ...
        R[k,k+1:p] = Q[:,k].T @ Xc[:,k+1:p]
        
        rank += 1 # Rank increment
                
        # Update v_j
        for j in range(k+1, p):
            v[j] = v[j] - R[k,j] * R[k,j]
        # Determine an index p_k+1 >= k+1 such that v_p_k+1 is maximal
        if k < p-1:
            pk = k+1 + np.argmax(v[k+1:])
            pass
        # If v_pk+1 is sufficiently small, leave k
        if v[pk] < 10:
            break
            
    return Q, R, P, rank

# Partial rank-revealing LU
def prrldu(M_: np.ndarray, cutoff: float = 0.0, 
           maxdim: int = np.iinfo(np.int32).max, mindim: int = 1
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int], List[int], List[int], float]:
    """
    Implements the Partial Rank-Revealing LU Decomposition algorithm.
    Args:
        M_: Input matrix
        cutoff: Tolerance for considering values as zero
        maxdim: Maximum dimension for the decomposition
        mindim: Minimum dimension for the decomposition
    Returns:
        Tuple containing (L, d, U, row_perm_inv, col_perm_inv, inf_error)
    """
    assert maxdim > 0, "maxdim must be positive"
    assert mindim > 0, "mindim must be positive"
    mindim = min(maxdim, mindim)
    
    M = M_.copy()
    Nr, Nc = M.shape
    k = min(Nr, Nc)
    
    # Initialize permutations
    rps = list(range(Nr))
    cps = list(range(Nc))
    
    # Find pivots
    inf_error = 0.0
    s = 0
    while s < k:
        Mabs = np.abs(M[s:, s:])
        if Mabs.size == 0:
            break        
        Mabs_max = np.max(Mabs)
        #print(f"Iter {s}: Mabs_max = {Mabs_max}")
        if Mabs_max < cutoff:
            inf_error = Mabs_max
            break
            
        piv = np.unravel_index(np.argmax(Mabs), Mabs.shape)
        piv = (piv[0] + s, piv[1] + s)
        
        # Swap rows and columns
        M[[s, piv[0]], :] = M[[piv[0], s], :]
        M[:, [s, piv[1]]] = M[:, [piv[1], s]]
        
        if s < k - 1:
            M[(s+1):, (s+1):] = M[(s+1):, (s+1):] - np.outer(M[(s+1):, s], M[s, (s+1):]) / M[s, s]
        
        rps[s], rps[piv[0]] = rps[piv[0]], rps[s]
        cps[s], cps[piv[1]] = cps[piv[1]], cps[s]
        s += 1
        #utils.MatrixSparseStat(M)
   
    # Commented on Dec.23/2024
    #M = M_[rps, :][:, cps]
    #utils.MatrixSparseStat(M)
    
    # Initialize L, d, U
    L = np.eye(Nr, k)
    d = np.zeros(k)
    U = np.eye(k, Nc)
    rank = 0
    
    for s in range(min(k, maxdim)):
        P = M[s, s]
        d[s] = P

        if rank < mindim:
            pass
        elif P == 0 or (abs(P) < cutoff and rank + 1 > mindim):
            break
            
        if P == 0:
            P = 1
        rank += 1
        
        if s < Nr - 1:
            piv_col = M[(s+1):, s]
            L[(s+1):, s] = piv_col / P
        if s < Nc - 1:
            piv_row = M[s, (s+1):]
            U[s, (s+1):] = piv_row / P
        
        # Commented on Dec.23/2024
        #if s < k - 1:
        #    M[(s+1):, (s+1):] = M[(s+1):, (s+1):] - np.outer(piv_col, piv_row) / P
        
    L = L[:, :rank]
    d = d[:rank]
    U = U[:rank, :]
    
    # Create inverse permutations
    row_perm_inv = [0] * len(rps)
    for i, p in enumerate(rps):
        row_perm_inv[p] = i
    col_perm_inv = [0] * len(cps)
    for i, p in enumerate(cps):
        col_perm_inv[p] = i
    
    return L, d, U, row_perm_inv, col_perm_inv, rps, cps, inf_error