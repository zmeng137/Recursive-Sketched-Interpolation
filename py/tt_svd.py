import numpy as np
import tensorly as tl

def TT_SVD(tensorX: tl.tensor, r_max: int, eps: float, verbose: int = 0) -> list[tl.tensor]:
    shape = tensorX.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)       # Get the number of dimension
    delta = (eps / np.sqrt(dim - 1)) * tl.norm(tensorX, 2)  # Truncation parameter
    
    W = tensorX        # Copy tensor X -> W
    nbar = W.size      # Total size of W
    r = 1              # Rank r
    ttList = []        # list storing tt factors
    iterlist = list(range(1, dim))  # Create iteration list: 1, 2, ..., d-1
    iterlist.reverse()              # Reverse the iteration list: d-1, ..., 1 
    
    for i in iterlist:
        W = tl.reshape(W, [int(nbar / r / shape[i]), int(r * shape[i])])  # Reshape W
        U, S, Vh = np.linalg.svd(W)  # SVD of W matrix
        # Compute rank r
        s = 0
        j = S.size 
    
        while s <= delta * delta:  # r_delta_i = min(j:sigma_j+1^2 + sigma_j+2^2 + ... <= delta^2)
            j -= 1
            s += S[j] * S[j]
            if j == 0:
                break
        j += 1
        ri = min(j, r_max)  # r_i-1 = min(r_max, r_delta_i)
    
        if verbose == 1:
            approxLR = U[:, 0:ri] @ np.diag(S[0:ri]) @ Vh[0:ri, :]
            rerror = tl.norm(approxLR - W, 2) / tl.norm(W, 2)
            print(f"Iteration {i} -- low rank approximation error = {rerror}")
    
        Ti = tl.reshape(Vh[0:ri, :], [ri, shape[i], r])
        nbar = int(nbar * ri / shape[i] / r)  # New total size of W
        r = ri  # Renewal r
        W = U[:, 0:ri] @ np.diag(S[0:ri])  # W = U[..] * S[..]
        ttList.append(Ti)  # Append new factor
    
    T1 = tl.reshape(W, [1, shape[0], r])
    ttList.append(T1)    
    ttList.reverse()
    return ttList