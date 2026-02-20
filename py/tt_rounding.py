import os
import sys
import numpy as np
import time as tm

from interpolative import interpolative_prrldu

def TT_rounding(tt, eps, max_rank):
    # Dimension and copy tensor objects
    start_t = tm.time()
    ndim = len(tt)
    tt_cores = [tt[i].copy() for i in range(ndim)]

    # Left-to-right QR (L2R orthogonalization)
    for k in range(len(tt_cores)-1):
        core = tt_cores[k]
        r1, n, r2 = core.shape
        core = core.reshape(r1*n, r2)
        Q, R = np.linalg.qr(core)
        new_r2 = Q.shape[1]
        tt_cores[k] = Q.reshape(r1, n, new_r2)
        tt_cores[k+1] = np.tensordot(R, tt_cores[k+1], axes=[1,0])

    # Right-to-left Truncated SVD (rounding)
    for k in reversed(range(1, len(tt_cores))):
        core = tt_cores[k]
        r1, n, r2 = core.shape
        core = core.reshape(r1, n*r2)
        U, S, Vt = np.linalg.svd(core, full_matrices=False)
        
        # Truncate by tolerance and max_rank, .... todo?
        r = min(np.sum(S > eps), max_rank)
        
        U = U[:, :r]
        S = S[:r]
        Vt = Vt[:r, :]
        tt_cores[k] = Vt.reshape(r, n, r2)
        tt_cores[k-1] = np.tensordot(tt_cores[k-1], U @ np.diag(S), axes=[2,0])
    
    end_t = tm.time()
    print(f"Runtime of the naive TT rounding algorithm: {(end_t - start_t) * 1000:.2f} ms.")
    return tt_cores

def TT_rounding_ID(tt, max_rank):
    # Dimension and copy tensor objects
    start_t = tm.time()
    ndim = len(tt)
    tt_cores = [tt[i].copy() for i in range(ndim)]

    # Left-to-right QR (L2R orthogonalization)
    for k in range(len(tt_cores)-1):
        core = tt_cores[k]
        r1, n, r2 = core.shape
        core = core.reshape(r1*n, r2)
        Q, R = np.linalg.qr(core)
        new_r2 = Q.shape[1]
        tt_cores[k] = Q.reshape(r1, n, new_r2)
        tt_cores[k+1] = np.tensordot(R, tt_cores[k+1], axes=[1,0])

    # Right-to-left Truncated SVD (rounding)
    for k in reversed(range(1, len(tt_cores))):
        core = tt_cores[k]
        r1, n, r2 = core.shape
        core = core.reshape(r1, n*r2)
        C, Z, cols, _ = interpolative_prrldu(core, cutoff=0.0, maxdim=max_rank)
        r = C.shape[1]
        tt_cores[k] = Z.reshape(r, n, r2)
        tt_cores[k-1] = np.tensordot(tt_cores[k-1], C, axes=[2,0])
    
    end_t = tm.time()
    print(f"Runtime of the naive TT rounding algorithm: {(end_t - start_t) * 1000:.2f} ms.")
    return tt_cores