import numpy as np
import tensorly as tl
from interpolation import interpolative_prrldu, cur_prrldu, cur_prrldu_ninv

# PRRLU-based Tensor-Train Interpolative Decomposition
def TT_IDPRRLDU(tensorX: tl.tensor, r_max: int, eps: float, verbose: int = 0) -> list[tl.tensor]:
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
        C, X, cols, error = interpolative_prrldu(W, cutoff=delta, maxdim=r_max)
        ri = C.shape[1]  # r_i-1 = min(r_max, r_delta_i)
    
        if verbose == 1:
            rerror = tl.norm(C @ X - W, 2) / tl.norm(W, 2)
            print(f"Iteration {i} -- low rank id approximation error = {rerror}")
    
        Ti = tl.reshape(X[0:ri, :], [ri, shape[i], r])
        nbar = int(nbar * ri / shape[i] / r)  # New total size of W
        r = ri             # Renewal r
        W = C[:, 0:ri]     # W = U[..] * S[..]
        ttList.append(Ti)  # Append new factor
    
    T1 = tl.reshape(W, [1, shape[0], r])
    ttList.append(T1)    
    ttList.reverse()
    return ttList

# PRRLU-based (Exact) Tensor-Train CUR Decomposition (Sweep from Left to Right)
def TT_CUR_L2R(tensor: tl.tensor, r_max: int, eps: float, verbose = 1):
    shape = tensor.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)      # Get the number of dimension
     
    W = tensor      # Copy tensor X -> W
    nbar = W.size   # Total size of W
    r = 1           # Initial TT-Rank r=1
    TTCore = []     # list storing TT-factors
    TTCore_cc = []  # Tensor-train including intermediate cross cores
    TTRank = [1]    # TT-Rank list
    InterpSet = {}  # One-sided nested set

    for i in range(dim-1):
        curr_dim = shape[i]  # Current dimension
        W = tl.reshape(W, [int(r * curr_dim), int(nbar / r / curr_dim)])  # Reshape W       
        
        # CUR decomposition based on PRRLDU
        r_subset, c_subset, cross_inv, cross, rank, pr, pc = cur_prrldu(W, eps, r_max)
        pr = pr[0:rank]  # Row skeleton 
        pc = pc[0:rank]  # Col skeleton

        # Mapping between r/c selection and tensor index pivots
        if i == 0:
            InterpSet[i] = np.array(pr).reshape(-1, 1)
        else:
            I = np.empty([rank, i+1])
            prev_I = InterpSet[i-1]
            for j in range(rank):
                p_I_idx = pr[j] // curr_dim
                c_i_idx = pr[j] % curr_dim
                I[j,0:i] = prev_I[p_I_idx]
                I[j,i] = c_i_idx
            InterpSet[i] = I        

        # Append new TT-factor
        Ti = tl.reshape(c_subset @ cross_inv, [r, shape[i], rank])
        TTCore.append(Ti)                                          
        TTCore_cc.append(tl.reshape(c_subset, [r, shape[i], rank]))  
        TTCore_cc.append(cross)
        TTRank.append(rank)

        nbar = int(nbar * rank / shape[i] / r)  # New total size of W
        r = rank  # Renewal r
        W = r_subset[0:rank,:]
        
        # Check the nested condition
        if (verbose):
            for ele in np.nditer(W):
                match_idx = np.argwhere(tensor == ele)
                nested_idx = match_idx[0][0:i+1]
                is_present = np.any(np.all(InterpSet[i] == nested_idx, axis=1))
                if (is_present == False):
                    print("Nested Interpolation error!")
        
    T_last = tl.reshape(W, [r, shape[-1], 1])
    TTCore.append(T_last)    
    TTCore_cc.append(T_last)
    TTRank.append(1)
    return TTCore, TTCore_cc, TTRank, InterpSet

# PRRLU-based (Exact) Tensor-Train CUR Decomposition (Sweep from Right to Left)
def TT_CUR_R2L(tensor: tl.tensor, r_max: int, eps: float, verbose = 1):
    shape = tensor.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)      # Get the number of dimension
     
    W = tensor      # Copy tensor X -> W
    nbar = W.size   # Total size of W
    r = 1           # Initial TT-Rank r=1
    TTCore = []     # list storing TT-factors
    TTCore_cc = []  # Tensor-train including intermediate cross cores
    TTRank = [1]    # TT-Rank list
    InterpSet = {}  # One-sided nested set

    # Mapping between r/c selection and tensor index pivots
    iterlist = list(range(1, dim))  # Create iteration list: 1, 2, ..., d-1
    iterlist.reverse()              # Reverse the iteration list: d-1, ..., 1 
    for i in iterlist:
        curr_dim = shape[i]  # Current dimension
        W = tl.reshape(W, [int(nbar / r / shape[i]), int(r * shape[i])], order='F')  # Reshape W       
        
        r_subset, c_subset, cross_inv, cross, rank, pr, pc = cur_prrldu(W, eps, r_max)
        pr = pr[0:rank]  # Row skeleton 
        pc = pc[0:rank]  # Col skeleton

        # Mapping between r/c selection and tensor index pivots
        pc = np.array(pc)
        if i == dim-1:
            InterpSet[i] = np.array(pc).reshape(-1,1)
        else:
            I = np.empty([rank, dim-i])
            prev_I = InterpSet[i+1]
            for j in range(rank):
                p_I_idx = pc[j] // curr_dim
                c_i_idx = pc[j] % curr_dim
                I[j,1:] = prev_I[p_I_idx]
                I[j,0] = c_i_idx
            InterpSet[i] = I        

        # Append new TT-factor
        Ti = tl.reshape(cross_inv @ r_subset, [rank, shape[i], r], order='F')
        TTCore.append(Ti)                                          
        TTCore_cc.append(tl.reshape(r_subset, [rank, shape[i], r], order='F'))  
        TTCore_cc.append(cross)
        TTRank.append(rank)

        nbar = int(nbar * rank / shape[i] / r)  # New total size of W
        r = rank  # Renewal r
        W = c_subset[:, 0:rank]

       # Check the nested condition
        if (verbose):
            for ele in np.nditer(W):
                match_idx = np.argwhere(tensor == ele)
                nested_idx = match_idx[0][i:]
                is_present = np.any(np.all(InterpSet[i] == nested_idx, axis=1))
                if (is_present == False):
                    print("Nested Interpolation error!")     
        
    T_last = tl.reshape(W, [1, shape[0], r], order='F')
    TTCore.append(T_last)    
    TTCore_cc.append(T_last)
    TTRank.append(1)
    TTCore.reverse()
    TTCore_cc.reverse()
    TTRank.reverse()
    return TTCore, TTCore_cc, TTRank, InterpSet

# Slice tensor: T[I,:]
def slice_first_modes(arr, indices):
    # Slice the first len(indices) modes with given indices
    slicing = tuple(indices) + tuple(slice(None) for _ in range(arr.ndim - len(indices)))
    return arr[slicing]  # Use square brackets, not parentheses

# Slice tensor: T[:,J]
def slice_last_modes(arr, indices):
    # Slice the last len(indices) modes with given indices
    slicing = tuple(slice(None) for _ in range(arr.ndim - len(indices))) + tuple(indices)
    return arr[slicing] 

# Assemble TT-Cores by (fully nested) interpolation pivots 
def cross_core_interp_assemble(tensor: tl.tensor, I_interpSet: dict, J_interpSet: dict, TTRank: np.array):
    shape = tensor.shape  # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)      # Get the number of dimension 
    assert len(TTRank) == dim + 1, "Number of TT-Ranks should equal tensor order + 1, i.e., with rank = 1 at boundaries!"
    TTCore_cross = []

    # Assmebly of TT-Cores via interpolation sets
    for d in range(dim):
        # Initialize TT-cores and cross matrices
        core = np.empty([TTRank[d], shape[d], TTRank[d+1]])
        cross_mat = np.empty([TTRank[d+1], TTRank[d+1]])
        
        # Construct TT-cores
        if d == 0:
            assert TTRank[d+1] == len(J_interpSet[1]), "Interpolation set size != Given rank"
            for j in range(TTRank[d+1]):
                J_slice = J_interpSet[1][j].astype(int).tolist()
                core[0,:,j] = slice_last_modes(tensor, J_slice)
            TTCore_cross.append(core) 
        elif d == dim-1:
            assert TTRank[d] == len(I_interpSet[d-1]), "Interpolation set size != Given rank"
            for i in range(TTRank[d]):
                I_slice = I_interpSet[d-1][i].astype(int).tolist()
                core[i,:,0] = slice_first_modes(tensor, I_slice)
            TTCore_cross.append(core)
        else:
            for i in range(TTRank[d]):
                I_slice = I_interpSet[d-1][i].astype(int).tolist()
                for j in range(TTRank[d+1]):
                    J_slice = J_interpSet[d+1][j].astype(int).tolist()
                    temp = slice_first_modes(tensor, I_slice)
                    core[i,:,j] = slice_last_modes(temp, J_slice)
            TTCore_cross.append(core)

        # Construct cross matrices
        if d != dim-1:
            for i in range(TTRank[d+1]):
                I_slice = I_interpSet[d][i].astype(int).tolist()
                for j in range(TTRank[d+1]):
                    J_slice = J_interpSet[d+1][j].astype(int).tolist()
                    temp = slice_first_modes(tensor, I_slice)
                    cross_mat[i,j] = slice_last_modes(temp, J_slice)
            TTCore_cross.append(cross_mat)
    return TTCore_cross

# Compute inverse of cross matrices and merge them into TT-cores
def cross_inv_merge(TTCore_cross, dimension):
    TTCores = [TTCore_cross[0]]
    for i in range(dimension-1):
        core = TTCore_cross[2*i+2]
        cross = TTCore_cross[2*i+1]
        core_shape0 = core.shape[0]
        core_shape1 = core.shape[1]
        core_shape2 = core.shape[2]    
        cross_inv = np.linalg.inv(cross)
        core_reshape = core.reshape(core_shape0,-1)
        merge = cross_inv @ core_reshape
        new_core = merge.reshape(core_shape0, core_shape1, core_shape2)
        TTCores.append(new_core)
    return TTCores