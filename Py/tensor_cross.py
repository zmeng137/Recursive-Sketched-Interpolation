import numpy as np
import tensorly as tl

from interpolation import interpolative_prrldu, interpolative_qr, cur_prrldu
# Rank-revealing based exact TT decomposition
# Evaluation based TT Cross format...


# Rank-revealing-based exact TT CUR Decomposition
def TT_CUR(tensorX: tl.tensor, r_max: int, eps: float, verbose: int = 0):
    shape = tensorX.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)       # Get the number of dimension
     
    W = tensorX        # Copy tensor X -> W
    nbar = W.size      # Total size of W
    r = 1              # Rank r
    ttList = []        # list storing TT-factors
    ttList_cc = []     # Tensor-train including intermediate cross cores
    
    iterlist = list(range(1, dim))  # Create iteration list: 1, 2, ..., d-1
    iterlist.reverse()              # Reverse the iteration list: d-1, ..., 1 
    for i in iterlist:
        W = tl.reshape(W, [int(nbar / r / shape[i]), int(r * shape[i])])  # Reshape W       
        
        #C, X, cols, error = interpolative_prrldu(W, cutoff=delta, maxdim=r_max)
        r_subset, c_subset, cross_inv, cross, rank = cur_prrldu(W, eps, r_max)
        ri = rank    

        # Append new TT-factor
        Ti = tl.reshape(cross_inv @ r_subset, [ri, shape[i], r])
        ttList.append(Ti)                                          
        ttList_cc.append(tl.reshape(r_subset, [ri, shape[i], r]))  
        ttList_cc.append(cross)
         
        nbar = int(nbar * ri / shape[i] / r)  # New total size of W
        r = ri             # Renewal r
        W = c_subset[:, 0:ri]     # W = U[..] * S[..]
        
    T1 = tl.reshape(W, [1, shape[0], r])
    ttList.append(T1)    
    ttList_cc.append(T1)
    ttList.reverse()
    ttList_cc.reverse()
    return ttList, ttList_cc