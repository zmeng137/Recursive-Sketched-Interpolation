import numpy as np
import time as tm
from scipy.linalg import solve, qr

from typing import Tuple, Union, List
from rank_revealing import prrldu, PivotedQR

# Interpolative decomposition based on PRRLU
def interpolative_prrldu(M: np.ndarray, cutoff: float = 0.0, maxdim: int = np.iinfo(np.int32).max, mindim: int = 1) -> Tuple[np.ndarray, np.ndarray, List[int], float]:
    """
    Compute interpolative decomposition (ID) from PRRLDU.
    Args:
        M: Input matrix
        **kwargs: Additional keyword arguments passed to prrldu
    Returns:
        Tuple containing (C, Z, pivot_columns, inf_error)
        - C: Matrix containing selected columns
        - Z: Interpolation matrix
        - pivot_columns: List of pivot column indices
        - inf_error: Error measure from PRRLDU
    """
    L, d, U, ipr, ipc, pr, pc, inf_error = prrldu(M, cutoff, maxdim, mindim)  # Compute PRRLDU decomposition
    k = len(d)
    U11 = U[:, :k]         # Extract relevant submatrices
    iU11 = np.linalg.solve(U11, np.eye(U.shape[0])) # Compute inverse of U11 through backsolving
    ZjJ = iU11 @ U         # Compute interpolation matrix
    CIj = L @ np.diag(d) @ U11   # Compute selected columns
    C = CIj[ipr, :]   # Apply inverse row permutation to get C
    Z = ZjJ[:, ipc]   # Apply inverse column permutation to get Z
    pivot_cols = [ipc.index(i) for i in range(k)]  # Get pivot columns (convert from inverse permutation)
    return C, Z, pivot_cols, inf_error

# Interpolative decomposition based on QRCP
def interpolative_qr(M, maxdim):
    """
    Compute interpolative decomposition (ID) from QRCP.
    Args:
        M: Input matrix
        maxdim: Maximum dimension cutoff
    Returns:
        Tuple containing (C, Z, cols)
        - C: Matrix containing selected columns
        - Z: Interpolation matrix
        - cols: skeleton columns
    """
    k = maxdim
    #Q , R , P = qr(M, pivoting =True, mode ='economic', check_finite = False)
    Mc = np.copy(M)
    Q , R , P, rank = PivotedQR(Mc)
    if rank < k:
        k = rank
    R_k = R[:k, :k]
    cols = P[:k]
    C = M[:, cols]
    Z = solve(R_k.T @ R_k, C.T @ M, overwrite_a=True, overwrite_b=True, assume_a ='pos')
    return C , Z, cols
    
# Interpolative decomposition by nuclear score
def interpolative_nuclear(M, cutoff=0.0, maxdim=None):
    '''
    Interpolative Decomposition (Nuclear)
    M = C * X
    cutoff - truncation threshold
    maxdim - maximum rank
    '''
    if maxdim is None:
        maxdim = min(M.shape)
    
    maxdim = min(maxdim, M.shape[0], M.shape[1])
    cols = []
    K = M.T @ M
    m = K.shape[0]
    Kt = K
    error = 0.0

    for t in range(maxdim):
        Kt2 = Kt @ Kt
        # Select the column with the maximum score
        l = max((p for p in range(m) if p not in cols), key=lambda p: Kt2[p, p] / Kt[p, p])
        max_err2 = Kt2[l, l] / Kt[l, l]
        cols.append(l)
        error = np.sqrt(np.abs(max_err2))
        if max_err2 < cutoff**2:
            break
        # Schur complement step
        Kt = K - K[:, cols] @ solve(K[np.ix_(cols, cols)], K[cols, :])
    
    # C selection    
    C = M[:, cols]
    
    # X = C \ M
    X = np.zeros([len(cols), M.shape[1]])
    qc, rc = qr(C,mode='economic')
    for i in range(M.shape[1]):
        m = M[:, i]
        z = qc.T @ m
        X[:,i] = solve(rc, z)
     
    # Enforce interpolation structure
    for w in range(len(cols)):
        for r in range(len(cols)):
            X[r, cols[w]] = 1.0 if r == w else 0.0

    return C, X, cols, error

# CUR decomposition (cross with inverse)
def cur_prrldu(M: np.ndarray, cutoff: float = 0.0, maxdim: int = np.iinfo(np.int32).max, mindim: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    L, d, U, ipr, ipc, pr, pc, inf_error = prrldu(M, cutoff, maxdim, mindim)  # Compute PRRLDU decomposition
    rank = len(d)   
    r_subset = M[pr[0:rank], :]   # subset of rows
    c_subset = M[:, pc[0:rank]]   # subset of columns 
    cross = M[pr[0:rank], :]     
    cross = cross[:, pc[0:rank]]  # cross matrix
    return r_subset, c_subset, cross, rank, pr, pc

# CUR decomposition (cross without inverse)
def cur_prrldu_ninv(M: np.ndarray, cutoff: float = 0.0, maxdim: int = np.iinfo(np.int32).max, mindim: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    L, d, U, ipr, ipc, pr, pc, inf_error = prrldu(M, cutoff, maxdim, mindim)  # Compute PRRLDU decomposition
    rank = len(d)   
    r_subset = M[pr[0:rank], :]   # subset of rows
    c_subset = M[:, pc[0:rank]]   # subset of columns 
    cross = M[pr[0:rank], :]     
    cross = cross[:, pc[0:rank]]  # cross matrix
    return r_subset, c_subset, cross, rank, pr, pc